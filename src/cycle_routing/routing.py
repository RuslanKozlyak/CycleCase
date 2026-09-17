"""Stage 4 - routes: turn models, one route, and a batch run over presets.

A search algorithm is a NetworkX path function used as is:

    search(graph, source, target, weight="cost") -> [node, ...]

``nx.dijkstra_path`` (default), ``nx.bellman_ford_path``, ``nx.astar_path``, or your own with the same
signature; a name from ``SEARCH`` works too. ``turns`` is the turn model: a name from ``TURNS``,
``TurnConfig`` fields as a dict (``{"left_penalty_m": 60}``), a ``TurnConfig``, a function
``penalty(graph, incoming_edge, outgoing_edge) -> metres``, or None. With a turn model the search runs
on ``turn_graph``: its nodes are the directed street edges, so a turn is an ordinary weighted edge.
"""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import functools
import hashlib
import inspect
import json
import math
from pathlib import Path

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
import shapely
from shapely.geometry import LineString
from shapely.ops import linemerge
from tqdm.auto import tqdm

from .config import OSM_WAY_TAGS, TurnConfig
from .graph import WEIGHT, build_graph, edge_costs
from .preparation import SCHEMA_VERSION


SEARCH = {"dijkstra": nx.dijkstra_path, "bellman_ford": nx.bellman_ford_path, "astar": nx.astar_path}
TURNS = {
    "none": None,
    "default": TurnConfig(),
    "min_turns": TurnConfig(slight_penalty_m=80.0, right_penalty_m=200.0, left_penalty_m=300.0, u_turn_penalty_m=900.0),
}

EdgeKey = tuple[int, int, int]
ROUTE_COLUMNS = ["od_id", "algorithm", "edge_route", "route_length_m", "turns", "geometry"]


def orient_coordinates(geometry, start_xy: tuple[float, float]) -> list[tuple[float, float]]:
    """Coordinates of an edge line in travel direction: the end nearest to ``start_xy`` comes first."""

    coordinates = list(geometry.coords)
    ux, uy = start_xy
    direct = (coordinates[0][0] - ux) ** 2 + (coordinates[0][1] - uy) ** 2
    reverse = (coordinates[-1][0] - ux) ** 2 + (coordinates[-1][1] - uy) ** 2
    return coordinates if direct <= reverse else list(reversed(coordinates))


def _oriented_coordinates(graph, edge: EdgeKey) -> list[tuple[float, float]]:
    u, v, key = edge
    geometry = graph.edges[u, v, key].get("geometry")
    start = (graph.nodes[u]["x"], graph.nodes[u]["y"])
    if geometry is None or geometry.geom_type != "LineString":
        return [start, (graph.nodes[v]["x"], graph.nodes[v]["y"])]
    return orient_coordinates(geometry, start)


def _bearing(start: tuple[float, float], end: tuple[float, float]) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    return math.degrees(math.atan2(dx, dy)) % 360


def edge_bearings(graph, edge: EdgeKey) -> tuple[float, float]:
    """Travel bearing when entering and when leaving the edge."""

    coordinates = _oriented_coordinates(graph, edge)
    return _bearing(coordinates[0], coordinates[1]), _bearing(coordinates[-2], coordinates[-1])


def turn_kind(incoming_bearing: float, outgoing_bearing: float, config: TurnConfig) -> str:
    """"straight", "slight", "right", "left" or "u_turn" from the two travel bearings."""

    difference = (outgoing_bearing - incoming_bearing + 180) % 360 - 180
    absolute = abs(difference)
    if absolute <= config.straight_angle_max_deg:
        return "straight"
    if absolute <= config.slight_angle_max_deg:
        return "slight"
    if absolute >= config.u_turn_angle_min_deg:
        return "u_turn"
    # Compass bearings increase clockwise: positive difference is a right turn.
    return "right" if difference > 0 else "left"


def turn_penalty(graph, incoming: EdgeKey, outgoing: EdgeKey, config: TurnConfig = TurnConfig()) -> float:
    """Penalty in metres of the turn from ``incoming`` onto ``outgoing`` by their geometry."""

    return config.penalty(turn_kind(edge_bearings(graph, incoming)[1], edge_bearings(graph, outgoing)[0], config))


def search_function(search):
    """A ``SEARCH`` name or a path function, as a path function."""

    return SEARCH[search] if isinstance(search, str) else search


def turn_model(turns):
    """A ``TURNS`` name, ``TurnConfig`` fields as a dict, a ``TurnConfig``, a function or None."""

    if isinstance(turns, str):
        return TURNS[turns]
    return TurnConfig(**turns) if isinstance(turns, dict) else turns


def turn_graph(graph, turns) -> nx.DiGraph:
    """Graph for searching with turns: nodes are ``("start", node)``, street edges ``(u, v, key)`` and
    ``("end", node)``; driving from one street edge onto the next costs its ``cost`` plus the turn penalty.
    """

    penalty = turns if callable(turns) else functools.partial(turn_penalty, config=turns)
    result = nx.DiGraph()
    for u, v, key, cost in graph.edges(keys=True, data=WEIGHT):
        edge = (u, v, key)
        result.add_edge(("start", u), edge, **{WEIGHT: cost})
        result.add_edge(edge, ("end", v), **{WEIGHT: 0.0})
        for _, after, after_key, after_cost in graph.out_edges(v, keys=True, data=WEIGHT):
            following = (v, after, after_key)
            result.add_edge(edge, following, **{WEIGHT: after_cost + penalty(graph, edge, following)})
    return result


def find_route(graph, origin: int, destination: int, *, search=nx.dijkstra_path, turns=TurnConfig()) -> dict | None:
    """Cheapest route between two nodes by the graph weight ``cost``; None when there is none.

    Returns ``edge_route`` (list of ``(u, v, key)``), ``route_length_m``, ``turns`` (left, right and
    U-turns, counted the same way for every turn model) and ``geometry``.
    """

    search, turns = search_function(search), turn_model(turns)
    try:
        if turns is None:
            nodes = search(graph, origin, destination, weight=WEIGHT)
            edge_route = [
                (u, v, min(graph[u][v], key=lambda key: graph[u][v][key][WEIGHT])) for u, v in zip(nodes, nodes[1:])
            ]
        else:
            # ponytail: cached on the graph per turn model; rebuild the graph if you change its edges.
            cache = graph.graph.setdefault("turn_graphs", {})
            if turns not in cache:
                cache[turns] = turn_graph(graph, turns)
            edge_route = search(cache[turns], ("start", origin), ("end", destination), weight=WEIGHT)[1:-1]
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if not edge_route:
        return None
    lines = [LineString(_oriented_coordinates(graph, edge)) for edge in edge_route]
    geometry = shapely.union_all(lines)
    if geometry.geom_type == "MultiLineString":
        merged = linemerge(geometry)
        if not merged.is_empty:
            geometry = merged
    return {
        "edge_route": edge_route,
        "route_length_m": float(sum(graph.edges[edge]["length_m"] for edge in edge_route)),
        "turns": count_turns(graph, edge_route),
        "geometry": geometry,
    }


def count_turns(graph, edge_route: list[EdgeKey], config: TurnConfig = TurnConfig()) -> int:
    """Left, right and U-turns along the route; slight bends do not count."""

    bearings = [edge_bearings(graph, tuple(edge)) for edge in edge_route]
    return sum(
        turn_kind(incoming[1], outgoing[0], config) in {"left", "right", "u_turn"}
        for incoming, outgoing in zip(bearings, bearings[1:])
    )


# --- batch run over presets with a disk cache ------------------------------------------------


# Physical-length baseline every preset is compared with.
SHORTEST = {"label": "кратчайший", "cost": "length_m", "turns": None, "search": nx.dijkstra_path}


def _digest(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()[:16]


def _fingerprint(model):
    """Cache identity of a turn model or a search function: its values or its code."""

    if model is None:
        return None
    if is_dataclass(model):
        return {type(model).__name__: asdict(model)}
    if isinstance(model, functools.partial):
        return {"partial": _fingerprint(model.func), "args": repr(model.args), "kwargs": repr(sorted(model.keywords.items()))}
    try:
        code = inspect.getsource(model).encode()
    except (OSError, TypeError):
        code = getattr(getattr(model, "__code__", None), "co_code", repr(model).encode())
    name = f"{getattr(model, '__module__', '')}.{getattr(model, '__qualname__', type(model).__name__)}"
    return {name: hashlib.sha256(code).hexdigest()[:12]}


def _pairs(pairs) -> list[tuple[str, int, int]]:
    if isinstance(pairs, pd.DataFrame):
        pairs = pairs[["od_id", "origin", "destination"]].itertuples(index=False, name=None)
    return [(str(od_id), int(origin), int(destination)) for od_id, origin, destination in pairs]


def _preset_routes(edges, nodes, pairs, algorithm: str, preset: dict, cache_dir: Path | None) -> pd.DataFrame:
    cost = edge_costs(edges, preset["cost"])
    turns = turn_model(preset.get("turns", TurnConfig()))
    search = search_function(preset.get("search", nx.dijkstra_path))
    path = None
    if cache_dir is not None:
        key = json.dumps({
            "schema": SCHEMA_VERSION,
            "tags": list(OSM_WAY_TAGS),
            "edges": _digest(edges["u"], edges["v"], edges["key"], edges["length_m"]),
            "cost": _digest(cost),
            "pairs": [list(pair) for pair in pairs],
            "turns": _fingerprint(turns),
            "search": _fingerprint(search),
        }, sort_keys=True, default=str)
        path = Path(cache_dir) / f"{algorithm}_{hashlib.sha256(key.encode()).hexdigest()[:12]}.pkl"
        if path.exists():
            return pd.read_pickle(path)
    graph = build_graph(edges, nodes, cost)
    records = []
    for od_id, origin, destination in pairs:
        route = find_route(graph, origin, destination, search=search, turns=turns)
        if route is not None:
            records.append({"od_id": od_id, "algorithm": algorithm, **route})
    frame = pd.DataFrame(records, columns=ROUTE_COLUMNS)
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(path)
    return frame


def build_routes(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    pairs,
    presets: dict[str, dict],
    *,
    cache_dir: Path | None = None,
) -> gpd.GeoDataFrame:
    """Shortest route plus one route per preset for every start/finish pair.

    ``pairs`` is ``load_od_pairs(...)`` or a list of ``(od_id, origin, destination)``. A preset is a dict
    with ``cost``, ``turns`` and ``search`` in any form ``edge_costs`` and ``find_route`` accept; each goes through ``build_graph`` and
    ``find_route``. With ``cache_dir`` routes are stored on disk under a key made of the edge set, the
    cost of every edge, the pairs, the turn model and the search algorithm.
    """

    pairs = _pairs(pairs)
    if "shortest" in presets:
        raise ValueError("'shortest' is reserved for the physical-length baseline")
    runs = {"shortest": SHORTEST, **presets}
    frames = [
        _preset_routes(edges, nodes, pairs, algorithm, preset, cache_dir)
        for algorithm, preset in tqdm(runs.items(), desc="Routes by preset")
    ]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=edges.crs)
