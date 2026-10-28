"""Maps for the hackathon notebook: three light static maps and one offline Leaflet route explorer."""

from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer

from .graph import edge_costs
from .routing import build_routes, orient_coordinates
from .tags import has_any
from .workshop import features, gpx_overlap


ASSETS = Path(__file__).with_name("assets")


def _axes(title, size=(9, 8)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax


def plot_popularity(lab):
    """Measured roads coloured by popularity 0–100; the test area and unmeasured roads are told apart."""
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    fig, ax = _axes("Популярность улиц, 0–100")
    roads, known = lab.osm, lab.popularity.notna()
    hidden = lab.test_area if lab.test_area is not None else pd.Series(False, index=roads.index)
    roads[hidden].plot(ax=ax, color="#fdba74", linewidth=0.4)
    roads[~hidden & ~known].plot(ax=ax, color="#cbd5e1", linewidth=0.3)
    order = lab.popularity[known].sort_values().index
    roads.loc[order].plot(ax=ax, column=lab.popularity[order], cmap="viridis", norm=Normalize(0, 100),
                          linewidth=0.7, legend=True, legend_kwds={"shrink": 0.5, "label": "популярность"})
    handles = [Line2D([], [], color="#cbd5e1", linewidth=3, label="нет измерения"),
               Line2D([], [], color="#fdba74", linewidth=3, label="тестовая зона: измерения скрыты")]
    ax.legend(handles=handles[: 1 + bool(hidden.any())], loc="lower left", fontsize=9)
    return fig


def plot_network(network):
    """What the filter left: the routed network, removed roads and pieces cut off from it."""
    roads = network.lab.osm
    fig, ax = _axes("Сеть после фильтра")
    layers = [(network.removed, "#ef4444", 0.5, "убрано фильтром"),
              (network.cut_off, "#f59e0b", 0.9, "оторвано от основной сети, не используется"),
              (roads.index.isin(network.roads.index), "#1f2937", 0.35, "основная сеть: маршруты строятся по ней")]
    for mask, color, width, label in layers:
        part = roads[mask]
        if len(part):
            part.plot(ax=ax, color=color, linewidth=width, label=f"{label}: {len(part):,}")
    ax.legend(loc="lower left", fontsize=9)
    return fig


def plot_pairs(lab):
    """Test pairs: the shortest path of every A–B, numbered."""
    import matplotlib.pyplot as plt
    routes = build_routes(lab.osm[lab.reference], lab.nodes, lab.pairs, {}, cache_dir=lab.folder / "route_cache")
    fig, ax = _axes(f"Тестовые пары A–B ({len(lab.pairs)}) и их кратчайшие пути")
    lab.osm.plot(ax=ax, color="#e5e7eb", linewidth=0.3)
    colors = plt.get_cmap("tab20")
    for n, route in enumerate(routes.itertuples()):
        for part in getattr(route.geometry, "geoms", [route.geometry]):
            ax.plot(*part.xy, color=colors(n % 20), linewidth=1.8)
        middle = route.geometry.interpolate(0.5, normalized=True)
        ax.annotate(route.od_id[-2:], (middle.x, middle.y), fontsize=7, ha="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    return fig


def _number(value):
    return float(value) if value is not None and pd.notna(value) else None


def _lines(geometries, crs, attrs=None):
    """Leaflet [lat, lon] lines; with ``attrs`` also one attribute row per produced line."""
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lines, rows = [], []
    for n, geometry in enumerate(geometries):
        for part in getattr(geometry, "geoms", [geometry]):
            lon, lat = to_wgs.transform(*part.xy)
            lines.append([[round(a, 5), round(b, 5)] for a, b in zip(lat, lon)])
            rows.append(attrs[n] if attrs is not None else None)
    return (lines, rows) if attrs is not None else lines


def _codes(lab):
    """Per road, the colour classes the map shows: [popularity or -1, bike, road type, surface, lit, speed]."""
    r, f = lab.osm, features(lab.osm)
    road = np.select([f["велодорожка"].eq(1), f["крупная_дорога"].eq(1), f["жилая_улица"].eq(1), f["пешеходная"].eq(1),
                      has_any(r.highway, {"service", "track"})], [0, 1, 2, 3, 4], 5)
    surface = np.select([r.surface.isna(), has_any(r.surface, {"asphalt", "concrete", "concrete:plates"}),
                         has_any(r.surface, {"gravel", "fine_gravel", "ground", "dirt", "unpaved", "compacted", "sand",
                                             "grass", "earth", "mud", "pebblestone"})], [3, 0, 2], 1)
    lit = np.select([r.lit.isna(), has_any(r.lit, "no")], [2, 0], 1)
    speed = pd.to_numeric(r.maxspeed, errors="coerce")
    speed = np.select([speed.isna(), speed.le(30), speed.le(50)], [3, 0, 1], 2)
    popularity = lab.popularity.round().fillna(-1).astype(int)
    codes = np.column_stack([popularity, f["велодорожка"].astype(int), road, surface, lit, speed]).tolist()
    return dict(zip(r.index, codes))


def route_explorer(lab, result, *, od_id=None, filename=None, tracks=None):
    """One selected A–B and solution on an offline map. ``tracks`` adds GPX lines (e.g. ``lab.tracks``)."""
    from IPython.display import IFrame

    roads = lab.osm.set_index(["u", "v", "key"], drop=False)
    xy = lab.nodes.set_index("node")[["x", "y"]]
    transform = Transformer.from_crs(lab.osm.crs, "EPSG:4326", always_xy=True)
    keys = {tuple(int(x) for x in key) for path in result.routes.edge_route for key in path}
    index = pd.MultiIndex.from_frame(lab.osm[["u", "v", "key"]])
    trips, popularity = dict(zip(index, lab.trips)), dict(zip(index, lab.popularity))
    codes = _codes(lab)
    row_of = dict(zip(index, lab.osm.index))
    segments = {}
    for key in sorted(keys):
        row = roads.loc[key]
        lon, lat = transform.transform(*zip(*orient_coordinates(row.geometry, tuple(xy.loc[key[0]]))))
        segments[":".join(map(str, key))] = {
            "line": [[round(a, 6), round(b, 6)] for a, b in zip(lat, lon)],
            "highway": str(row.highway) if pd.notna(row.highway) else "нет данных",
            "surface": str(row.surface) if pd.notna(row.surface) else "нет данных",
            "length": float(row.length_m), "trips": _number(trips[key]), "pop": _number(popularity[key]),
            "a": codes[row_of[key]],
        }
    models = list(dict.fromkeys(result.routes.algorithm))
    costs = {}
    for name in models:
        values = edge_costs(lab.osm, "length_m" if name == "shortest" else result.solutions[name]["cost"]) / lab.osm.length_m
        mapping = dict(zip(index, values))
        costs[name] = {":".join(map(str, key)): round(float(mapping[key]), 4) for key in keys}
    routes = {}
    for row in result.routes.itertuples():
        routes.setdefault(row.od_id, {})[row.algorithm] = {
            "edges": [":".join(map(str, map(int, key))) for key in row.edge_route],
            "length": float(row.route_length_m), "turns": int(row.turns),
        }
    gpx = {}
    if tracks is not None:
        overlap = gpx_overlap(result.routes, tracks)
        gpx = dict(zip(zip(overlap.od_id, overlap.algorithm), overlap.nearest_pct))
    detail = {}
    for row in result.per_od.itertuples():
        detail.setdefault(row.od_id, {})[row.algorithm] = {
            "popularity": _number(row.popularity), "counted": _number(row.counted), "note": row.note,
            **({"gpx": _number(gpx.get((row.od_id, row.algorithm)))} if tracks is not None else {}),
        }
    context = lab.osm
    if len(context) > 100_000:  # a whole external city is much larger than the route corridor
        bounds = result.routes.total_bounds
        context = context.cx[bounds[0] - 1500:bounds[2] + 1500, bounds[1] - 1500:bounds[3] + 1500]
    context = context.assign(pair=[tuple(sorted(p)) for p in zip(context.u, context.v)]).drop_duplicates("pair")
    lines, attrs = _lines(context.geometry.simplify(6, preserve_topology=False), lab.osm.crs,
                          [codes[i] for i in context.index])
    payload = {
        "segments": segments, "routes": routes, "costs": costs, "detail": detail, "models": models,
        "context": {"lines": lines, "attrs": attrs},
        "tracks": _lines(tracks.to_crs(lab.osm.crs).geometry.simplify(6), lab.osm.crs) if tracks is not None else [],
        "labels": {name: ("кратчайший" if name == "shortest" else result.solutions[name].get("label", name)) for name in models},
        "ods": [od for od in lab.pairs.od_id if od in routes], "first": od_id or next(iter(routes)), "zone": lab.name,
    }
    template = (ASSETS / "route_explorer.html").read_text(encoding="utf-8")
    document = (template.replace("__LEAFLET_CSS__", (ASSETS / "leaflet/leaflet.css").read_text(encoding="utf-8"))
                .replace("__LEAFLET_JS__", (ASSETS / "leaflet/leaflet.js").read_text(encoding="utf-8"))
                .replace("__DATA__", json.dumps(payload, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
                         .replace("<", "\\u003c")))
    if filename:
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(document, encoding="utf-8")
    return IFrame("about:blank", width="100%", height=660, extras=[
        'title="Сравнение велосипедных маршрутов"',
        f'srcdoc="{html.escape(document, quote=True)}"',
        'style="border:1px solid #cbd5e1;border-radius:8px"', 'loading="lazy"',
    ])
