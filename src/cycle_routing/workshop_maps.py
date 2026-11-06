from __future__ import annotations

import html

import json

from pathlib import Path

import geopandas as gpd

import numpy as np

import pandas as pd

from pyproj import Transformer

from .graph import edge_costs

from .routing import build_routes, orient_coordinates

from .tags import has_any

from .workshop import gpx_overlap

ASSETS = Path(__file__).with_name("assets")

def _turn_examples(network):
    from .graph import build_graph
    from .routing import edge_bearings

    graph = build_graph(network.roads, network.city.nodes, "length_m")
    bearings = {edge: edge_bearings(graph, edge) for edge in graph.edges(keys=True)}
    street = has_any(network.roads.highway, {
        "residential", "living_street", "tertiary", "secondary", "primary", "unclassified", "cycleway",
    })
    preferred = dict(zip(zip(network.roads.u, network.roads.v, network.roads.key), street))
    best = {}

    def quality(edges):
        return sum(1000 * (not preferred[edge]) + max(0, 45 - graph.edges[edge]["length_m"]) for edge in edges)

    def offer(kind, incoming, outgoing, deviation):
        score = quality([incoming, *outgoing]) + deviation
        if kind not in best or score < best[kind][0]:
            best[kind] = (score, incoming, outgoing)

    for node in graph:
        outgoing = list(graph.out_edges(node, keys=True))
        for incoming in graph.in_edges(node, keys=True):
            choices = [(edge, (bearings[edge][0] - bearings[incoming][1] + 180) % 360 - 180) for edge in outgoing]
            for edge, angle in choices:
                if abs(angle) <= 25 and edge[1] != incoming[0]:
                    offer("straight", incoming, [edge], abs(angle))
                elif 25 < abs(angle) <= 55:
                    offer("slight", incoming, [edge], abs(abs(angle) - 40))
                elif abs(angle) >= 150 and edge[1] == incoming[0]:
                    offer("u_turn", incoming, [edge], abs(abs(angle) - 180))
            right = [(edge, angle) for edge, angle in choices if 55 < angle < 150]
            left = [(edge, angle) for edge, angle in choices if -150 < angle < -55]
            if right and left:
                r, ra = min(right, key=lambda item: quality([item[0]]) + abs(item[1] - 90))
                l, la = min(left, key=lambda item: quality([item[0]]) + abs(item[1] + 90))
                offer("sides", incoming, [r, l], abs(ra - 90) + abs(la + 90))

    kinds = ["straight", "slight", "u_turn", "sides"]
    return graph, bearings, [(kind, best[kind][1], best[kind][2]) if kind in best else (kind, None, []) for kind in kinds]

def plot_turn_examples(network, penalty):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from shapely.geometry import LineString

    graph, bearings, examples = _turn_examples(network)
    blue, green, orange = "#2563eb", "#059669", "#d97706"
    titles = {"straight": "Прямо", "slight": "Плавный изгиб", "u_turn": "Разворот", "sides": "Направо / налево"}

    def point(bearing, radius=1):
        radians = np.deg2rad(bearing)
        return np.array([np.sin(radians), np.cos(radians)]) * radius

    def line(edge):
        start = (graph.nodes[edge[0]]["x"], graph.nodes[edge[0]]["y"])
        return LineString(orient_coordinates(graph.edges[edge]["geometry"], start))

    def arrow(ax, geometry, start, end, color):
        ax.annotate("", xy=geometry.interpolate(end).coords[0], xytext=geometry.interpolate(start).coords[0],
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.8, "mutation_scale": 19}, zorder=9)

    fig, axes = plt.subplots(2, 2, figsize=(9, 8.3))
    fig.subplots_adjust(left=0.04, right=0.97, top=0.87, bottom=0.15, hspace=0.35, wspace=0.12)
    fig.suptitle("Повороты на реальных участках дорожной сети", fontsize=19, fontweight="bold", y=0.985)
    fig.legend(handles=[
        Line2D([], [], color="#cbd5e1", linewidth=3, label="дороги вокруг"),
        Line2D([], [], color=blue, linewidth=3, label="въезд"),
        Line2D([], [], color=green, linewidth=3, label="выезд"),
        Line2D([], [], color=orange, linewidth=3, label="вариант налево"),
    ], loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=4, frameon=False, fontsize=10)

    for ax, (kind, incoming, outgoing) in zip(axes.flat, examples):
        if incoming is None:
            ax.axis("off")
            ax.set_title(titles[kind], fontsize=14, fontweight="bold")
            ax.text(0.5, 0.5, "В выбранной сети такого манёвра нет", ha="center", transform=ax.transAxes)
            continue
        junction = graph.nodes[incoming[1]]
        x, y, radius = junction["x"], junction["y"], 65
        nearby = network.city.osm.cx[x - radius:x + radius, y - radius:y + radius]
        nearby.plot(ax=ax, color="#cbd5e1", linewidth=1.5, zorder=1)
        ax.set(xlim=(x - radius, x + radius), ylim=(y - radius, y + radius))
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(titles[kind], fontsize=14, fontweight="bold", pad=10)
        before = bearings[incoming][1]
        incoming_line = line(incoming)
        ax.plot(*incoming_line.xy, color=blue, linewidth=5, solid_capstyle="round", zorder=4)
        span = min(radius * 0.9, incoming_line.length)
        arrow(ax, incoming_line, incoming_line.length - 0.65 * span, incoming_line.length - 0.2 * span, blue)
        continuation = np.array([x, y]) + point(before, 24)
        ax.plot([x, continuation[0]], [y, continuation[1]], color=blue, linestyle="--", linewidth=1.4, zorder=6)

        # Put the compass in the corner furthest from the highlighted travel directions.
        occupied = [(before + 180) % 360, *[bearings[edge][0] for edge in outgoing]]
        corners = [(45, [0.65, 0.65, 0.34, 0.34]), (315, [0.01, 0.65, 0.34, 0.34]),
                   (135, [0.65, 0.01, 0.34, 0.34]), (225, [0.01, 0.01, 0.34, 0.34])]
        _, position = max(corners, key=lambda corner: min(abs((angle - corner[0] + 180) % 360 - 180) for angle in occupied))
        compass = ax.inset_axes(position)
        compass.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), xticks=[], yticks=[])
        compass.set_aspect("equal")
        compass.patch.set_facecolor("white")
        compass.patch.set_alpha(0.96)
        for spine in compass.spines.values():
            spine.set_visible(False)
        circle = np.deg2rad(np.linspace(0, 360, 181))
        compass.plot(np.sin(circle), np.cos(circle), color="#94a3b8", linewidth=1)
        for bearing, label in [(0, "0°/360°"), (90, "90°"), (180, "180°"), (270, "270°")]:
            compass.text(*point(bearing, 1.24), label, ha="center", va="center", fontsize=7, color="#64748b")
        old = point(before, 0.9)
        compass.plot([0, old[0]], [0, old[1]], color=blue, linestyle="--", linewidth=1.8)

        captions = []
        for index, edge in enumerate(outgoing):
            color = green if index == 0 else orange
            after = bearings[edge][0]
            angle = (after - before + 180) % 360 - 180
            angle_label = "0°" if abs(angle) < 0.5 else f"{angle:+.0f}°"
            geometry = line(edge)
            ax.plot(*geometry.xy, color=color, linewidth=3.2,
                    linestyle="--" if kind == "u_turn" else "-", solid_capstyle="round", zorder=5)
            span = min(radius * 0.9, geometry.length)
            arrow(ax, geometry, 0.6 * span if kind == "u_turn" else 0.35 * span, 0.93 * span, color)

            arc = np.array([point(bearing, 16) for bearing in np.linspace(before, before + angle, 70)])
            ax.plot(x + arc[:, 0], y + arc[:, 1], color=color, linewidth=1.8, zorder=7)
            position = np.array([x, y]) + point(before + angle / 2, 27)
            ax.text(*position, angle_label, color=color, fontsize=11, fontweight="bold", ha="center", va="center",
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1}, zorder=10)
            end = point(after, 0.95)
            compass.annotate("", xy=end, xytext=(0, 0),
                             arrowprops={"arrowstyle": "-|>", "color": color, "lw": 1.8, "mutation_scale": 11})
            metres = penalty(graph, incoming, edge)
            direction = ("направо: " if index == 0 else "налево: ") if kind == "sides" else ""
            captions.append(f"{direction}{before:.0f}° → {after:.0f}°; угол {angle_label}; штраф {metres:g} м")

        ax.scatter([x], [y], s=30, color="#0f172a", edgecolors="white", zorder=11)
        # A scale bar makes the real street geometry distinct from a schematic intersection.
        ax.plot([x - 55, x - 35], [y - 53, y - 53], color="#475569", linewidth=2, zorder=12)
        ax.text(x - 45, y - 48, "20 м", ha="center", va="bottom", fontsize=8, color="#475569", zorder=12)
        ax.text(0.5, -0.05, "\n".join(captions), transform=ax.transAxes, ha="center", va="top",
                fontsize=9.5, color="#334155", linespacing=1.6)

    fig.text(0.5, 0.073, "Угол = (направление после − направление до + 180°) % 360° − 180°",
             ha="center", fontsize=12, fontweight="bold", color="#0f172a")
    fig.text(0.5, 0.041, "Пунктир у точки — прежнее направление. Штраф добавляется к стоимости, в условных метрах.",
             ha="center", fontsize=10, color="#475569")
    return fig

def _axes(title, size=(9, 8)):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=size, layout="constrained")
    ax.set_title(title, fontsize=11)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig, ax

def plot_popularity(city):
    from matplotlib.colors import Normalize
    from matplotlib.lines import Line2D
    fig, ax = _axes("Популярность улиц, 0–100")
    roads, known = city.osm, city.popularity.notna()
    hidden = city.test_area if city.test_area is not None else pd.Series(False, index=roads.index)
    roads[hidden].plot(ax=ax, color="#fdba74", linewidth=0.4)
    roads[~hidden & ~known].plot(ax=ax, color="#cbd5e1", linewidth=0.3)
    order = city.popularity[known].sort_values().index
    roads.loc[order].plot(ax=ax, column=city.popularity[order], cmap="viridis", norm=Normalize(0, 100),
                          linewidth=0.7, legend=True, legend_kwds={"shrink": 0.5, "label": "популярность"})
    handles = [Line2D([], [], color="#cbd5e1", linewidth=3, label="нет измерения"),
               Line2D([], [], color="#fdba74", linewidth=3, label="закрытая зона: измерения скрыты")]
    ax.legend(handles=handles[: 1 + bool(hidden.any())], loc="lower left", fontsize=9)
    return fig

def plot_pairs(city):
    import matplotlib.pyplot as plt
    routes = build_routes(city.osm, city.nodes, city.pairs, {}, cache_dir=city.folder / "route_cache")
    fig, ax = _axes(f"Пары A–B ({len(city.pairs)}) и их кратчайшие пути")
    city.osm.plot(ax=ax, color="#e5e7eb", linewidth=0.3)
    colors = plt.get_cmap("tab20")
    for n, route in enumerate(routes.itertuples()):
        for part in getattr(route.geometry, "geoms", [route.geometry]):
            ax.plot(*part.xy, color=colors(n % 20), linewidth=1.8)
        middle = route.geometry.interpolate(0.5, normalized=True)
        ax.annotate(route.od_id[-2:], (middle.x, middle.y), fontsize=7, ha="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))
    return fig

def plot_nearby(city, points, radius_m=50):
    from matplotlib.lines import Line2D
    from .workshop_geo import within_radius

    if not np.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("Радиус должен быть положительным")
    roads = city.osm
    crs = roads.crs
    if not crs.is_projected or crs.to_epsg() == 3857:
        crs = roads.estimate_utm_crs()
    roads, points = roads.to_crs(crs), points.to_crs(crs)
    fig, ax = _axes(f"Точки и участки в радиусе {radius_m:g} м", size=(8, 5))
    points = points[points.geometry.notna() & ~points.geometry.is_empty]
    if points.empty or roads.empty:
        ax.text(.5, .5, "Нет точек в выбранной области", ha="center", transform=ax.transAxes)
        return fig
    hits, distances = roads.sindex.nearest(points.geometry, return_distance=True)
    candidates = np.unique(hits[0][distances <= radius_m])
    neighbors = points.sindex.query(points.geometry, predicate="dwithin", distance=2*radius_m)
    counts = np.bincount(neighbors[0], minlength=len(points))
    focus = candidates[np.argmin(counts[candidates])] if len(candidates) else 0
    point = points.geometry.iloc[int(focus)]
    point = point.representative_point()
    x, y, span = point.x, point.y, max(150, radius_m * 3)
    local = roads.cx[x-span:x+span, y-span:y+span]
    nearby = points.cx[x-span-radius_m:x+span+radius_m, y-span-radius_m:y+span+radius_m]
    nearby.geometry.buffer(radius_m).plot(ax=ax, color="#dbeafe", alpha=.6, edgecolor="#93c5fd", linewidth=.7)
    if len(local):
        local.plot(ax=ax, color="#cbd5e1", linewidth=1.4)
        selected = local[within_radius(local, nearby, radius_m)]
        if len(selected):
            selected.plot(ax=ax, color="#ea580c", linewidth=3)
    nearby.plot(ax=ax, color="#2563eb", markersize=28, edgecolor="white", zorder=5)
    ax.set(xlim=(x-span, x+span), ylim=(y-span, y+span))
    ax.legend(handles=[Line2D([], [], color="#cbd5e1", label="дороги"),
                       Line2D([], [], color="#ea580c", label=f"в радиусе {radius_m:g} м"),
                       Line2D([], [], marker="o", color="#2563eb", linestyle="", label="точки")],
              loc="lower left", fontsize=9)
    return fig

def _number(value):
    return float(value) if value is not None and pd.notna(value) else None

def _lines(geometries, crs, attrs=None):
    to_wgs = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lines, rows = [], []
    for n, geometry in enumerate(geometries):
        for part in getattr(geometry, "geoms", [geometry]):
            lon, lat = to_wgs.transform(*part.xy)
            lines.append([[round(a, 5), round(b, 5)] for a, b in zip(lat, lon)])
            rows.append(attrs[n] if attrs is not None else None)
    return (lines, rows) if attrs is not None else lines

def _codes(city):
    r = city.osm
    bike_values = r.cycleway.fillna("").str.split(";")
    bike = (has_any(r.highway, "cycleway")
            | bike_values.map(lambda values: bool(set(values) - {"", "no", "none", "separate"})))
    road = np.select([bike, has_any(r.highway, {"primary", "secondary", "trunk", "primary_link", "secondary_link"}),
                      has_any(r.highway, {"residential", "living_street"}),
                      has_any(r.highway, {"footway", "pedestrian", "steps", "path"}),
                      has_any(r.highway, {"service", "track"})], [0, 1, 2, 3, 4], 5)
    surface = np.select([r.surface.isna(), has_any(r.surface, {"asphalt", "concrete", "concrete:plates"}),
                         has_any(r.surface, {"gravel", "fine_gravel", "ground", "dirt", "unpaved", "compacted", "sand",
                                             "grass", "earth", "mud", "pebblestone"})], [3, 0, 2], 1)
    lit = np.select([r.lit.isna(), has_any(r.lit, "no")], [2, 0], 1)
    speed = pd.to_numeric(r.maxspeed, errors="coerce")
    speed = np.select([speed.isna(), speed.le(30), speed.le(50)], [3, 0, 1], 2)
    popularity = city.popularity.round().fillna(-1).astype(int)
    codes = np.column_stack([popularity, bike.astype(int), road, surface, lit, speed]).tolist()
    return dict(zip(r.index, codes))

def _feature_data(city, features):
    roads = city.roads
    base = pd.DataFrame(roads.drop(columns=roads.geometry.name))
    if features is None:
        features = base
    table = features(roads) if callable(features) else features
    if isinstance(table, pd.Series):
        table = table.to_frame(name=table.name or "Признак")
    if not isinstance(table, pd.DataFrame) or table.index.has_duplicates or table.columns.has_duplicates:
        raise ValueError("Признаки должны быть таблицей с уникальными строками и столбцами")
    if not city.osm.index.isin(table.index).all():
        raise ValueError("Признаки должны покрывать все участки city.roads")
    table = table.reindex(city.osm.index)
    table = pd.concat([base.drop(columns=base.columns.intersection(table.columns)), table], axis=1)
    definitions, columns = [], []
    for name in table:
        values = table[name]
        definition = {"label": str(name), "index": 6 + len(columns)}
        if pd.api.types.is_numeric_dtype(values) or pd.api.types.is_bool_dtype(values):
            values = pd.to_numeric(values).astype(float)
            if np.isinf(values).any():
                raise ValueError("Признаки не должны содержать бесконечность")
            known = values.dropna()
            definition.update(kind="number", minimum=float(known.min()) if len(known) else 0.,
                              maximum=float(known.max()) if len(known) else 1.)
            columns.append([_number(v) for v in values])
        else:
            values = values.astype("string")
            categories = list(values.dropna().unique())
            definition.update(kind="category", categories=categories)
            columns.append([None if pd.isna(v) else categories.index(v) for v in values])
        definitions.append(definition)
    return definitions, {idx: [col[j] for col in columns] for j, idx in enumerate(table.index)}

def _map_layers(city, layers):
    payload = []
    palette = ["#16a34a", "#a855f7", "#ea580c", "#0891b2", "#dc2626"]
    combined = {name: {"data": data, "radius": 4, "visible": False} for name, data in getattr(city, "points", {}).items()}
    combined.update({name: {"data": data, "radius": 4} for name, data in getattr(city, "map_layers", {}).items()})
    combined.update(layers or {})
    for n, (name, layer) in enumerate(combined.items()):
        options = layer if isinstance(layer, dict) else {}
        if options:
            layer = options["data"]
        color = options.get("color", palette[n % len(palette)])
        radius = float(options.get("radius", 6))
        visible = options.get("visible", True)
        if not isinstance(color, str) or not np.isfinite(radius) or radius <= 0 or not isinstance(visible, bool):
            raise ValueError("У слоя нужны color, положительный radius и visible=True/False")
        if isinstance(layer, pd.Series) and not isinstance(layer, gpd.GeoSeries):
            if not pd.api.types.is_bool_dtype(layer) or not city.roads.index.isin(layer.index).all():
                raise ValueError("Маска слоя должна содержать True/False для всех дорог")
            layer = city.roads.loc[layer.reindex(city.roads.index).fillna(False)]
        if isinstance(layer, gpd.GeoSeries):
            layer = gpd.GeoDataFrame(geometry=layer)
        if not isinstance(layer, gpd.GeoDataFrame) or layer.crs is None:
            raise ValueError("Слой должен быть GeoDataFrame/GeoSeries с CRS или маской дорог")
        geo = layer[[layer.geometry.name]].copy()
        if "name" in layer and layer.geometry.name != "name":
            geo["name"] = layer["name"].astype("string")
        if geo.geometry.name != "geometry":
            geo = geo.rename_geometry("geometry")
        geo = geo[geo.geometry.notna() & ~geo.geometry.is_empty].to_crs(4326)
        payload.append({"name": str(name), "count": len(geo), "color": color, "radius": radius, "visible": visible,
                        "data": json.loads(geo.to_json(drop_id=True))})
    return payload

def route_explorer(city, result, *, od_id=None, filename=None, tracks=None, features=None, layers=None):
    from IPython.display import IFrame

    if result.routes.empty:
        raise ValueError("Нет построенных маршрутов для карты")
    roads = city.osm.set_index(["u", "v", "key"], drop=False)
    xy = city.nodes.set_index("node")[["x", "y"]]
    transform = Transformer.from_crs(city.osm.crs, "EPSG:4326", always_xy=True)
    keys = {tuple(int(x) for x in key) for path in result.routes.edge_route for key in path}
    index = pd.MultiIndex.from_frame(city.osm[["u", "v", "key"]])
    trips, popularity = dict(zip(index, city.trips)), dict(zip(index, city.popularity))
    codes = _codes(city)
    definitions, extra = _feature_data(city, features)
    for i in codes:
        codes[i].extend(extra[i])
    row_of = dict(zip(index, city.osm.index))
    segments = {}
    for key in sorted(keys):
        row = roads.loc[key]
        student_row = city.roads.loc[row_of[key]]
        lon, lat = transform.transform(*zip(*orient_coordinates(row.geometry, tuple(xy.loc[key[0]]))))
        segments[":".join(map(str, key))] = {
            "line": [[round(a, 6), round(b, 6)] for a, b in zip(lat, lon)],
            "highway": str(student_row["тип_дороги"]) if pd.notna(student_row["тип_дороги"]) else "нет данных",
            "surface": str(student_row["покрытие"]) if pd.notna(student_row["покрытие"]) else "нет данных",
            "length": float(row.length_m), "trips": _number(trips[key]), "pop": _number(popularity[key]),
            "a": codes[row_of[key]],
        }
    models = list(dict.fromkeys(result.routes.algorithm))
    costs = {}
    for name in models:
        cost = "length_m" if name == "shortest" else result.solutions[name].get("cost")
        if cost is None:
            costs[name] = {":".join(map(str, key)): None for key in keys}
        else:
            values = edge_costs(city.osm, cost) / city.osm.length_m
            mapping = dict(zip(index, values))
            costs[name] = {":".join(map(str, key)): round(float(mapping[key]), 4) for key in keys}
    routes = {}
    for row in result.routes.itertuples():
        routes.setdefault(row.od_id, {})[row.algorithm] = {
            "edges": [":".join(map(str, map(int, key))) for key in row.edge_route],
            "length": float(row.route_length_m), "turns": int(row.turns),
            "search_ms": _number(row.search_ms),
        }
    gpx = {}
    if tracks is not None:
        overlap = gpx_overlap(result.routes, tracks)
        gpx = dict(zip(zip(overlap.od_id, overlap.algorithm), overlap.nearest_pct))
    detail = {}
    for row in result.per_od.itertuples():
        detail.setdefault(row.od_id, {})[row.algorithm] = {
            "popularity": _number(row.popularity) if city.trips.notna().any() else None,
            "counted": _number(row.counted), "note": row.note,
            **({"gpx": _number(gpx.get((row.od_id, row.algorithm)))} if tracks is not None else {}),
        }
    context = city.osm
    context = context.assign(pair=[tuple(sorted(p)) for p in zip(context.u, context.v)]).drop_duplicates("pair")
    lines, attrs = _lines(context.geometry.simplify(6, preserve_topology=False), city.osm.crs,
                          [codes[i] for i in context.index])
    payload = {
        "segments": segments, "routes": routes, "costs": costs, "detail": detail, "models": models,
        "context": {"lines": lines, "attrs": attrs},
        "features": definitions, "layers": _map_layers(city, layers),
        "tracks": _lines(tracks.to_crs(city.osm.crs).geometry.simplify(6), city.osm.crs) if tracks is not None else [],
        "labels": {name: ("кратчайший" if name == "shortest" else result.solutions[name].get("label", name)) for name in models},
        "ods": [od for od in city.pairs.od_id if od in routes], "first": od_id or next(iter(routes)), "zone": city.name,
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
