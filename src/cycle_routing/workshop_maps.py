"""Static road maps, real turn examples and an offline Leaflet route explorer for the notebook."""

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
from .workshop import gpx_overlap


ASSETS = Path(__file__).with_name("assets")


def _turn_examples(network):
    """Pick actual connected street edges by geometry, without popularity labels or ML predictions."""
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
    """Four real locations in the student's network, with travel arrows and compass-angle insets."""
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 11))
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
    """Measured roads coloured by popularity 0–100; the test area and unmeasured roads are told apart."""
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
               Line2D([], [], color="#fdba74", linewidth=3, label="тестовая зона: измерения скрыты")]
    ax.legend(handles=handles[: 1 + bool(hidden.any())], loc="lower left", fontsize=9)
    return fig


def plot_network(network):
    """What the filter left: the routed network, removed roads and pieces cut off from it."""
    roads = network.city.osm
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


def plot_pairs(city):
    """Test pairs: the shortest path of every A–B, numbered."""
    import matplotlib.pyplot as plt
    routes = build_routes(city.osm[city.reference], city.nodes, city.pairs, {}, cache_dir=city.folder / "route_cache")
    fig, ax = _axes(f"Тестовые пары A–B ({len(city.pairs)}) и их кратчайшие пути")
    city.osm.plot(ax=ax, color="#e5e7eb", linewidth=0.3)
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


def _codes(city):
    """Per road, the colour classes the map shows: [popularity or -1, bike, road type, surface, lit, speed]."""
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


def route_explorer(city, result, *, od_id=None, filename=None, tracks=None):
    """One selected A–B and solution on an offline map. ``tracks`` adds GPX lines (e.g. ``city.tracks``)."""
    from IPython.display import IFrame

    roads = city.osm.set_index(["u", "v", "key"], drop=False)
    xy = city.nodes.set_index("node")[["x", "y"]]
    transform = Transformer.from_crs(city.osm.crs, "EPSG:4326", always_xy=True)
    keys = {tuple(int(x) for x in key) for path in result.routes.edge_route for key in path}
    index = pd.MultiIndex.from_frame(city.osm[["u", "v", "key"]])
    trips, popularity = dict(zip(index, city.trips)), dict(zip(index, city.popularity))
    codes = _codes(city)
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
        values = edge_costs(city.osm, "length_m" if name == "shortest" else result.solutions[name]["cost"]) / city.osm.length_m
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
    context = city.osm
    if len(context) > 100_000:  # a whole external city is much larger than the route corridor
        bounds = result.routes.total_bounds
        context = context.cx[bounds[0] - 1500:bounds[2] + 1500, bounds[1] - 1500:bounds[3] + 1500]
    context = context.assign(pair=[tuple(sorted(p)) for p in zip(context.u, context.v)]).drop_duplicates("pair")
    lines, attrs = _lines(context.geometry.simplify(6, preserve_topology=False), city.osm.crs,
                          [codes[i] for i in context.index])
    payload = {
        "segments": segments, "routes": routes, "costs": costs, "detail": detail, "models": models,
        "context": {"lines": lines, "attrs": attrs},
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
