import hashlib

import json

from pathlib import Path

import geopandas as gpd

import requests

from shapely.geometry import Point

from tqdm.auto import tqdm

from .config import GraphConfig

from .point_data import POINT_TYPES, point_files, point_query_key, save_points_snapshot

def _point_query(bounds):
    west, south, east, north = bounds
    box = f"({south},{west},{north},{east})"
    selectors = set()
    for item in POINT_TYPES.values():
        for key, values in item["tags"].items():
            for value in values:
                selectors.add(f'nwr[{json.dumps(key)}={json.dumps(value)}]{box};')
    return '[out:json][timeout:25];(' + ''.join(sorted(selectors)) + ');out center;'

def _download_fixed_points(bounds, *, progress=True):
    from .workshop_geo import OSMUnavailableError
    last_error = None
    urls = GraphConfig().overpass_urls
    with tqdm(urls, desc="Точки OSM", unit="сервер", leave=False, disable=not progress) as bar:
        for url in bar:
            try:
                response = requests.get(url.rstrip('/') + '/interpreter', params={"data": _point_query(bounds)},
                                        headers={"User-Agent": "cycle-routing-workshop/0.1"}, timeout=(5, 35))
                response.raise_for_status()
                payload = response.json()
                if "elements" not in payload or "remark" in payload:
                    raise ValueError("Неполный ответ OSM")
                return payload, response.url.split('?')[0]
            except (requests.RequestException, ValueError) as error:
                last_error = error
                if progress:
                    tqdm.write(f"{url}: {type(error).__name__}; следующий сервер")
    raise OSMUnavailableError("Не удалось подготовить точки: серверы OSM недоступны.") from last_error

def _point_table(payload):
    records = []
    for element in payload["elements"]:
        tags = element.get("tags", {})
        position = element if element["type"] == "node" else element.get("center", {})
        if not {"lat", "lon"} <= set(position):
            raise ValueError("У объекта OSM нет координат")
        for name, item in POINT_TYPES.items():
            if any(tags.get(key) in values for key, values in item["tags"].items()):
                records.append({**tags, "point_id": f'{element["type"]}/{element["id"]}', "тип": name,
                                "name": tags.get("name"), "geometry": Point(position["lon"], position["lat"])})
    if not records:
        return gpd.GeoDataFrame(columns=["point_id", "тип", "name", "geometry"], geometry="geometry", crs=4326)
    return gpd.GeoDataFrame(records, geometry="geometry", crs=4326).drop_duplicates(["тип", "point_id"])

def prepare_osm_area(*, folder, crs, polygon=None, roads=None, nodes=None, progress=True, refresh_points=False, point_buffer_m=500):
    from .preparation import _download_graph, edges_table, nodes_table, GraphDownloadError
    from .workshop_geo import OSMUnavailableError, _osm_settings
    folder = Path(folder)
    if roads is None:
        if polygon is None:
            raise ValueError("Нужна область или готовая дорожная сеть")
        with _osm_settings(folder):
            try:
                graph = _download_graph(bbox=None, center=None, dist=None, polygon=polygon,
                                        config=GraphConfig(retain_all=True), progress=progress)
            except GraphDownloadError as error:
                raise OSMUnavailableError(str(error)) from error
        roads, _ = edges_table(graph, crs)
        nodes = nodes_table(graph, crs)
    if nodes is None:
        raise ValueError("Нужны узлы дорожной сети")
    if not 0 < point_buffer_m < float("inf"):
        raise ValueError("Запас вокруг сети должен быть положительным и конечным")
    metric_roads = roads.to_crs(crs)
    if not metric_roads.crs.is_projected or any(a.unit_name != "metre" for a in metric_roads.crs.axis_info):
        raise ValueError("Для подготовки точек нужны координаты сети в метрах")
    crs_key = json.dumps(metric_roads.crs.to_authority()) if metric_roads.crs.to_authority() else metric_roads.crs.to_wkt()
    network_key = hashlib.sha256(b"".join(metric_roads.geometry.to_wkb()) + crs_key.encode()).hexdigest()
    area = metric_roads.geometry.union_all().envelope.buffer(point_buffer_m).envelope
    bounds = gpd.GeoSeries([area], crs=crs).to_crs(4326).total_bounds.tolist()
    files = point_files(folder)
    if files and not refresh_points:
        meta = json.loads(files[0].read_text(encoding="utf-8"))
        if (meta.get("bounds") == bounds and meta.get("query_key") == point_query_key()
                and meta.get("network_key") == network_key and meta.get("point_buffer_m") == point_buffer_m):
            return roads, nodes
    payload, endpoint = _download_fixed_points(bounds, progress=progress)
    table = _point_table(payload)
    downloaded = len(table)
    if downloaded:
        nearby = metric_roads.sindex.nearest(table.to_crs(crs).geometry, return_all=False,
                                            max_distance=point_buffer_m)[0]
        table = table.iloc[sorted(set(nearby))].copy()
    save_points_snapshot(folder, table, categories=POINT_TYPES,
                         source="OpenStreetMap contributors", license="ODbL-1.0",
                         source_url="https://www.openstreetmap.org/copyright", endpoint=endpoint,
                         bounds=bounds, query_key=point_query_key(), network_key=network_key,
                         point_buffer_m=float(point_buffer_m), downloaded=downloaded,
                         scope="Points within the buffer of the prepared road network",
                         representation="OSM node coordinates; centers for ways and relations",
                         osm_timestamp=payload.get("osm3s", {}).get("timestamp_osm_base"))
    return roads, nodes
