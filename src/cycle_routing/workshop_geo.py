from contextlib import contextmanager

from functools import partial

import hashlib

import json

from pathlib import Path

from urllib.parse import urlparse

import geopandas as gpd

import numpy as np

import pandas as pd

from shapely.geometry import LineString

from tqdm.auto import tqdm

class OSMUnavailableError(RuntimeError):
    pass

def load_points(path):
    path = Path(path)
    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path)
        if not {"lon", "lat"} <= set(table):
            raise ValueError("В CSV нужны столбцы lon и lat — долгота и широта")
        xy = table[["lon", "lat"]].apply(pd.to_numeric, errors="coerce")
        if not np.isfinite(xy).all().all() or not xy.lon.between(-180, 180).all() or not xy.lat.between(-90, 90).all():
            raise ValueError("В CSV некорректные координаты lon/lat")
        result = gpd.GeoDataFrame(table, geometry=gpd.points_from_xy(xy.lon, xy.lat), crs=4326)
    else:
        result = gpd.read_file(path)
    if result.crs is None:
        raise ValueError("У файла с точками не указана система координат")
    if not result.geometry.dropna().geom_type.isin(["Point", "MultiPoint"]).all():
        raise ValueError("Файл должен содержать точки")
    return result

def within_radius(geometries, points, radius_m):
    if not np.isfinite(radius_m) or radius_m < 0:
        raise ValueError("Радиус должен быть конечным и неотрицательным")
    lines = geometries.geometry if isinstance(geometries, gpd.GeoDataFrame) else geometries
    points = points.geometry if isinstance(points, gpd.GeoDataFrame) else points
    if lines.crs is None or points.crs is None:
        raise ValueError("У геометрий и точек должен быть указан CRS")
    found = np.zeros(len(lines), dtype=bool)
    if not len(lines) or not len(points):
        return pd.Series(found, index=lines.index)
    crs = lines.crs
    if not crs.is_projected or crs.to_epsg() == 3857 or any(a.unit_name != "metre" for a in crs.axis_info):
        crs = lines.estimate_utm_crs()
    metric = lines.to_crs(crs)
    valid_points = points[points.notna() & ~points.is_empty].to_crs(crs)
    if len(valid_points):
        hits = valid_points.sindex.query(metric, predicate="dwithin", distance=radius_m)
        found[hits[0]] = True
    return pd.Series(found, index=lines.index)

@contextmanager
def _osm_settings(folder):
    import osmnx as ox
    from .preparation import OSM_WAY_TAGS

    values = {"use_cache": True, "cache_folder": Path(folder) / "http_cache",
              "http_user_agent": "cycle-routing-workshop/0.1",
              "requests_timeout": min(30, ox.settings.requests_timeout),
              "useful_tags_way": sorted(set(ox.settings.useful_tags_way) | set(OSM_WAY_TAGS))}
    previous = {name: getattr(ox.settings, name) for name in values}
    try:
        for name, value in values.items():
            setattr(ox.settings, name, value)
        yield ox
    finally:
        for name, value in previous.items():
            setattr(ox.settings, name, value)

def _osm_request(ox, request, *, progress=True, urls=None):
    from requests.exceptions import RequestException
    from osmnx._errors import ResponseStatusCodeError
    from osmnx import _overpass
    from .config import GraphConfig

    original = ox.settings.overpass_url
    original_kwargs = ox.settings.requests_kwargs
    original_pause = _overpass._get_overpass_pause
    hooks = dict(original_kwargs.get("hooks", {}))
    response_hooks = hooks.get("response", [])
    response_hooks = [response_hooks] if callable(response_hooks) else list(response_hooks)
    def check_response(response, *args, **kwargs):
        # Fail over on HTTP errors, including /status failures that OSMnx otherwise
        # interprets as a reason to sleep for a minute. HTTP cache hits skip this hook.
        response.raise_for_status()
        return response
    ox.settings.requests_kwargs = {**original_kwargs, "hooks": {**hooks, "response": [*response_hooks, check_response]}}
    # Preserve server slot limits, but skip OSMnx's fallback minute-long sleep
    # when /status is unreachable or malformed. Interpreter errors trigger failover.
    _overpass._get_overpass_pause = partial(original_pause, default_pause=0)
    last_error = None
    try:
        endpoints = list(dict.fromkeys(urls if urls is not None else [original, *GraphConfig().overpass_urls]))
        with tqdm(endpoints, desc="Серверы OSM", unit="сервер", disable=not progress, leave=False) as bar:
            for url in bar:
                host = urlparse(url).netloc
                bar.set_postfix_str(host)
                ox.settings.overpass_url = url
                try:
                    # OSMnx reads its HTTP cache before any network requests.
                    # A separate /status probe would prevent offline cache hits.
                    result = request()
                    bar.update()
                    return result
                except (RequestException, ResponseStatusCodeError) as error:
                    last_error = error
                    if progress:
                        tqdm.write(f"{host}: {type(error).__name__}; пробуем следующий сервер")
    finally:
        ox.settings.overpass_url = original
        ox.settings.requests_kwargs = original_kwargs
        _overpass._get_overpass_pause = original_pause
    raise OSMUnavailableError("Не удалось загрузить OSM: серверы недоступны. Повторите позже.") from last_error

def city_from_addresses(start, end, *, folder, margin_m=2000, progress=True):
    from .osm_area import prepare_osm_area
    from .workshop import load_city
    from .workshop_data import write_city_pack
    from requests.exceptions import RequestException
    from osmnx._errors import ResponseStatusCodeError

    if not all(isinstance(s, str) and s.strip() for s in (start, end)):
        raise ValueError("Укажите два полных адреса с городом")
    if not np.isfinite(margin_m) or margin_m <= 0:
        raise ValueError("Запас вокруг маршрута должен быть положительным")
    params = {"start": start.strip(), "end": end.strip(), "margin_m": float(margin_m), "version": 2}
    key = hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()[:16]
    base = Path(folder)
    target = base / key
    if (target / "manifest.json").exists():
        with tqdm(total=1, desc="Город: кеш", disable=not progress) as bar:
            result = load_city(target)
            prepare_osm_area(folder=target, roads=result.osm, nodes=result.nodes, crs=result.osm.crs,
                             progress=progress)
            bar.update()
        return result
    with tqdm(total=5, desc="Город", unit="этап", disable=not progress) as bar:
        with _osm_settings(base) as ox:
            coords = []
            for label in ("start", "end"):
                bar.set_postfix_str("адрес A" if label == "start" else "адрес B")
                try:
                    coords.append(ox.geocode(params[label]))
                except (RequestException, ResponseStatusCodeError) as error:
                    raise OSMUnavailableError("Не удалось определить адрес: сервис геокодирования недоступен.") from error
                bar.update()
            points = gpd.GeoSeries(gpd.points_from_xy([p[1] for p in coords], [p[0] for p in coords]), crs=4326)
            crs = points.estimate_utm_crs()
            metric = points.to_crs(crs)
            if metric.iloc[0].distance(metric.iloc[1]) > 100_000:
                raise ValueError("Укажите две точки одного города")
            area = LineString(metric.to_list()).buffer(margin_m).envelope
            polygon = gpd.GeoSeries([area], crs=crs).to_crs(4326).iloc[0]
            bar.set_postfix_str("сеть OSM")
            roads, nodes = prepare_osm_area(folder=target, polygon=polygon, crs=crs, progress=progress)
            bar.update()
        bar.set_postfix_str("геометрия")
        available = nodes[nodes.node.isin(set(roads.u) | set(roads.v))]
        if available.empty:
            raise ValueError("В загруженной области нет дорожной сети")
        snapped = [available.loc[available.geometry.distance(p).idxmin()] for p in metric]
        if any(row.geometry.distance(p) > margin_m for row, p in zip(snapped, metric)):
            raise ValueError("Адрес далеко от сети: увеличьте margin_m")
        if snapped[0].node == snapped[1].node:
            raise ValueError("Обе точки привязались к одному узлу: выберите более удалённые адреса")
        pairs = pd.DataFrame([{"od_id": "AB01", "origin": int(snapped[0].node), "destination": int(snapped[1].node),
                               "zone": "open", "start_lat": coords[0][0], "start_lon": coords[0][1],
                               "end_lat": coords[1][0], "end_lon": coords[1][1]}])
        bar.update()
        bar.set_postfix_str("сохранение")
        write_city_pack(roads, nodes, pairs, pd.Series(np.nan, index=roads.index),
                        pd.Series("open", index=roads.index), folder=target, city="свой маршрут",
                        split_params={"method": "addresses", "margin_m": float(margin_m)})
        result = load_city(target)
        bar.update()
    return result
