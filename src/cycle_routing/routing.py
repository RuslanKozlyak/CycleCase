from __future__ import annotations

from dataclasses import asdict, is_dataclass

import functools

import hashlib

import inspect

import json

import math

from pathlib import Path

import re

import sys

import time

import geopandas as gpd

import networkx as nx

import numpy as np

import pandas as pd

import shapely

from shapely.geometry import LineString

from shapely.ops import linemerge

from tqdm import tqdm

from .config import OSM_WAY_TAGS, TurnConfig

from .graph import WEIGHT, build_graph, edge_costs

SEARCH = {"dijkstra": nx.dijkstra_path, "bellman_ford": nx.bellman_ford_path, "astar": nx.astar_path}

TURNS = {'none': None, 'default': TurnConfig()}

EdgeKey = tuple[int, int, int]

ROUTE_COLUMNS = ["od_id", "algorithm", "edge_route", "route_length_m", "turns", "search_ms", "geometry"]

ROUTE_CACHE_VERSION = 3

def orient_coordinates(geometry, start_xy: tuple[float, float]) -> list[tuple[float, float]]:

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

    coordinates = _oriented_coordinates(graph, edge)
    return _bearing(coordinates[0], coordinates[1]), _bearing(coordinates[-2], coordinates[-1])

def turn_kind(incoming_bearing: float, outgoing_bearing: float, config: TurnConfig) -> str:

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

    return config.penalty(turn_kind(edge_bearings(graph, incoming)[1], edge_bearings(graph, outgoing)[0], config))

def search_function(search):

    return SEARCH[search] if isinstance(search, str) else search

def turn_model(turns):

    if isinstance(turns, str):
        return TURNS[turns]
    return TurnConfig(**turns) if isinstance(turns, dict) else turns

def turn_graph(graph, turns) -> nx.DiGraph:

    penalty = turns if callable(turns) else functools.partial(turn_penalty, config=turns)
    result = nx.DiGraph()
    for u, v, key, cost in graph.edges(keys=True, data=WEIGHT):
        edge = (u, v, key)
        # Search heuristics can use coordinates in both the street and the turn graph.
        for node, street_node in [(("start", u), u), (edge, v), (("end", v), v)]:
            result.add_node(node, x=graph.nodes[street_node]["x"], y=graph.nodes[street_node]["y"])
        result.add_edge(("start", u), edge, **{WEIGHT: cost})
        result.add_edge(edge, ("end", v), **{WEIGHT: 0.0})
        for _, after, after_key, after_cost in graph.out_edges(v, keys=True, data=WEIGHT):
            following = (v, after, after_key)
            result.add_edge(edge, following, **{WEIGHT: after_cost + penalty(graph, edge, following)})
    return result

def find_route(graph, origin: int, destination: int, *, search=nx.dijkstra_path, turns=TurnConfig()) -> dict | None:

    search, turns = search_function(search), turn_model(turns)
    try:
        if turns is None:
            started = time.perf_counter()
            nodes = search(graph, origin, destination, weight=WEIGHT)
            elapsed = time.perf_counter() - started
            edge_route = [
                (u, v, min(graph[u][v], key=lambda key: graph[u][v][key][WEIGHT])) for u, v in zip(nodes, nodes[1:])
            ]
        else:
            # Cache the turn graph for each penalty function.
            cache = graph.graph.setdefault("turn_graphs", {})
            if turns not in cache:
                cache[turns] = turn_graph(graph, turns)
            started = time.perf_counter()
            edge_route = search(cache[turns], ("start", origin), ("end", destination), weight=WEIGHT)[1:-1]
            elapsed = time.perf_counter() - started
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None
    if not edge_route:
        return None
    return route_from_edges(graph, edge_route, search_ms=1000 * elapsed)

def route_from_edges(graph, edge_route, *, search_ms=float("nan")) -> dict:
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
        "search_ms": search_ms,
        "geometry": geometry,
    }

def count_turns(graph, edge_route: list[EdgeKey], config: TurnConfig = TurnConfig()) -> int:

    bearings = [edge_bearings(graph, tuple(edge)) for edge in edge_route]
    return sum(
        turn_kind(incoming[1], outgoing[0], config) in {"left", "right", "u_turn"}
        for incoming, outgoing in zip(bearings, bearings[1:])
    )

SHORTEST = {"label": "кратчайший", "cost": "length_m", "turns": None, "search": nx.dijkstra_path}

def _digest(*arrays) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()[:16]

def _fingerprint(model):

    if model is None:
        return None
    if callable(getattr(model, "cache_key", None)):
        return {"cache_key": model.cache_key()}
    if is_dataclass(model):
        # JSON's default=str truncates large arrays and can give distinct custom
        # searches the same key. Hash their complete state, keeping simple configs readable.
        state = asdict(model)
        try:
            json.dumps(state)
        except TypeError:
            import cloudpickle
            return {type(model).__name__: hashlib.sha256(cloudpickle.dumps(model)).hexdigest()}
        return {type(model).__name__: state}
    if isinstance(model, functools.partial):
        import cloudpickle
        return {"partial": hashlib.sha256(cloudpickle.dumps(model)).hexdigest()}
    if inspect.isfunction(model) and (model.__closure__ or model.__defaults__ or model.__kwdefaults__
                                      or inspect.getclosurevars(model).globals):
        import cloudpickle
        state = (model, inspect.getclosurevars(model).globals)
        return {"function": hashlib.sha256(cloudpickle.dumps(state)).hexdigest()}
    if not inspect.isfunction(model) and not inspect.isbuiltin(model):
        import cloudpickle
        return {type(model).__name__: hashlib.sha256(cloudpickle.dumps(model)).hexdigest()}
    try:
        code = inspect.getsource(model).encode()
    except Exception:  # no readable source: built-ins, or a function unpickled from a notebook
        code = getattr(getattr(model, "__code__", None), "co_code", repr(model).encode())
    name = f"{getattr(model, '__module__', '')}.{getattr(model, '__qualname__', type(model).__name__)}"
    return {name: hashlib.sha256(code).hexdigest()[:12]}

def _pairs(pairs) -> list[tuple[str, int, int]]:
    if isinstance(pairs, pd.DataFrame):
        pairs = pairs[["od_id", "origin", "destination"]].itertuples(index=False, name=None)
    return [(str(od_id), int(origin), int(destination)) for od_id, origin, destination in pairs]

def _solution_routes(edges, nodes, pairs, algorithm: str, chosen: dict, cache_dir: Path | None) -> pd.DataFrame:
    cost = edge_costs(edges, chosen["cost"])
    turns = turn_model(chosen.get("turns", TurnConfig()))
    search = search_function(chosen.get("search", nx.dijkstra_path))
    path = None
    if cache_dir is not None:
        key = json.dumps({
            "schema": ROUTE_CACHE_VERSION,
            "tags": list(OSM_WAY_TAGS),
            "edges": _digest(edges["u"], edges["v"], edges["key"], edges["length_m"]),
            "geometry": hashlib.sha256(b"".join(edges.geometry.to_wkb())).hexdigest(),
            "coordinates": _digest(nodes["node"], nodes["x"], nodes["y"]),
            "crs": str(edges.crs),
            "cost": _digest(cost),
            "pairs": [list(pair) for pair in pairs],
            "turns": _fingerprint(turns),
            "search": _fingerprint(search),
        }, sort_keys=True, default=str)
        name = re.sub(r'[<>:"/\\|?*\s]+', "_", algorithm)  # any solution name must make a valid file name
        path = Path(cache_dir) / f"{name}_{hashlib.sha256(key.encode()).hexdigest()[:12]}.pkl"
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
    solutions: dict[str, dict],
    *,
    cache_dir: Path | None = None,
    progress: bool = True,
) -> gpd.GeoDataFrame:

    pairs = _pairs(pairs)
    if "shortest" in solutions:
        raise ValueError("'shortest' is reserved for the physical-length baseline")
    runs = {"shortest": SHORTEST, **solutions}
    frames = [
        _solution_routes(edges, nodes, pairs, algorithm, chosen, cache_dir)
        for algorithm, chosen in tqdm(runs.items(), desc="Маршруты", file=sys.stdout,
                                      disable=not progress or len(runs) == 1)
    ]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=edges.crs)
