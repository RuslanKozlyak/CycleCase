"""Stage 4 - routing algorithm: shortest and turn-aware comfort routes for every preset."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from itertools import count
import json
from pathlib import Path
import heapq
import math

import networkx as nx
import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
import shapely
from shapely.geometry import LineString
from shapely.ops import linemerge
from tqdm.auto import tqdm

from .comfort import add_comfort_cost
from .config import TurnConfig


EdgeKey = tuple[int, int, int]


def _oriented_coordinates(graph, edge: EdgeKey) -> list[tuple[float, float]]:
    u, v, key = edge
    data = graph.edges[u, v, key]
    geometry = data.get("geometry")
    coordinates = (
        list(geometry.coords)
        if geometry is not None and geometry.geom_type == "LineString"
        else [(graph.nodes[u]["x"], graph.nodes[u]["y"]), (graph.nodes[v]["x"], graph.nodes[v]["y"])]
    )
    ux, uy = graph.nodes[u]["x"], graph.nodes[u]["y"]
    direct = (coordinates[0][0] - ux) ** 2 + (coordinates[0][1] - uy) ** 2
    reverse = (coordinates[-1][0] - ux) ** 2 + (coordinates[-1][1] - uy) ** 2
    return coordinates if direct <= reverse else list(reversed(coordinates))


def _bearing(start: tuple[float, float], end: tuple[float, float]) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    return math.degrees(math.atan2(dx, dy)) % 360


def _edge_bearings(graph, edge: EdgeKey) -> tuple[float, float]:
    """Travel bearing when entering and when leaving the edge."""

    coordinates = _oriented_coordinates(graph, edge)
    return _bearing(coordinates[0], coordinates[1]), _bearing(coordinates[-2], coordinates[-1])


def _turn_kind(incoming_bearing: float, outgoing_bearing: float, config: TurnConfig) -> str:
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


# A search algorithm is any function
#     search(graph, origin, destination, weight, turns) -> list[EdgeKey] | None
# returning the directed edges of the cheapest route by edge attribute ``weight`` ("length" or
# "comfort_cost"); ``turns`` is the preset's TurnConfig or None. Give a preset {"search": fn} to swap it.


def dijkstra(graph, origin: int, destination: int, weight: str, turns: TurnConfig | None = None) -> list[EdgeKey] | None:
    """NetworkX Dijkstra over nodes; ignores turn penalties, takes the cheapest parallel edge."""

    try:
        nodes = nx.shortest_path(graph, origin, destination, weight=weight)
    except nx.NetworkXNoPath:
        return None
    return [
        (u, v, min(graph[u][v], key=lambda key: graph[u][v][key].get(weight, math.inf)))
        for u, v in zip(nodes, nodes[1:])
    ] or None


def dijkstra_turns(graph, origin: int, destination: int, weight: str, turns: TurnConfig) -> list[EdgeKey] | None:
    """Dijkstra whose state is the incoming directed edge, so every turn adds its penalty."""

    start = (origin, None)
    distances = {start: 0.0}
    previous = {}
    sequence = count()
    queue = [(0.0, next(sequence), start)]
    target = None
    bearings: dict[EdgeKey, tuple[float, float]] = {}

    def edge_bearings(edge: EdgeKey) -> tuple[float, float]:
        if edge not in bearings:
            bearings[edge] = _edge_bearings(graph, edge)
        return bearings[edge]

    while queue:
        cost, _, state = heapq.heappop(queue)
        node, incoming = state
        if cost > distances.get(state, math.inf):
            continue
        if node == destination:
            target = state
            break
        for _, neighbour, key, data in graph.out_edges(node, keys=True, data=True):
            outgoing = (node, neighbour, key)
            new_cost = cost + float(data.get(weight, data.get("length", 1.0)))
            if incoming is not None:
                new_cost += turns.penalty(_turn_kind(edge_bearings(incoming)[1], edge_bearings(outgoing)[0], turns))
            new_state = (neighbour, outgoing)
            if new_cost < distances.get(new_state, math.inf):
                distances[new_state] = new_cost
                previous[new_state] = (state, outgoing)
                heapq.heappush(queue, (new_cost, next(sequence), new_state))

    if target is None or target == start:
        return None
    edges = []
    state = target
    while state != start:
        state, edge = previous[state]
        edges.append(edge)
    return edges[::-1]


def route_record(
    graph,
    origin: int,
    destination: int,
    algorithm: str,
    od_id: str,
    *,
    turn_config: TurnConfig | None = None,
    search=None,
):
    """One route as a dict; ``search`` defaults to ``dijkstra_turns`` with turn penalties, else ``dijkstra``."""

    search = search or (dijkstra_turns if turn_config is not None else dijkstra)
    weight = "length" if algorithm == "shortest" else "comfort_cost"
    edge_route = search(graph, origin, destination, weight, turn_config)
    if not edge_route:
        return None
    lines = [LineString(_oriented_coordinates(graph, edge)) for edge in edge_route]
    geometry = shapely.union_all(lines)
    if geometry.geom_type == "MultiLineString":
        merged = linemerge(geometry)
        if not merged.is_empty:
            geometry = merged
    return {
        "od_id": od_id,
        "algorithm": algorithm,
        "edge_route": edge_route,
        "route_length_m": float(sum(graph.edges[edge]["length"] for edge in edge_route)),
        "turns": count_turns(graph, edge_route),
        "geometry": geometry,
    }


def count_turns(graph, edge_route: list[EdgeKey], config: TurnConfig = TurnConfig()) -> int:
    """Left, right and U-turns along the route; slight bends do not count."""

    bearings = [_edge_bearings(graph, tuple(edge)) for edge in edge_route]
    return sum(
        _turn_kind(incoming[1], outgoing[0], config) in {"left", "right", "u_turn"}
        for incoming, outgoing in zip(bearings, bearings[1:])
    )


def sample_od_pairs(
    graph,
    candidate_nodes,
    n_pairs: int,
    seed: int,
    *,
    min_length_m: float = 1_500,
    max_length_m: float = 10_000,
    prefix: str = "DE",
):
    rng = np.random.default_rng(seed)
    nodes = np.asarray(candidate_nodes)
    pairs, seen = [], set()
    for _ in range(max(2_000, 200 * n_pairs)):
        origin, destination = rng.choice(nodes, size=2, replace=False).tolist()
        if (origin, destination) in seen:
            continue
        seen.add((origin, destination))
        try:
            length_m = nx.shortest_path_length(graph, origin, destination, weight="length")
        except nx.NetworkXNoPath:
            continue
        if min_length_m <= length_m <= max_length_m:
            pairs.append((f"{prefix}-{len(pairs) + 1:02d}", origin, destination))
        if len(pairs) == n_pairs:
            return pairs
    raise RuntimeError(f"Only {len(pairs)} OD pairs found")


def _preset_routes(graph, pairs, algorithm, config, turns, search, cache_dir: Path | None) -> pd.DataFrame:
    path = None
    if cache_dir is not None:
        # ponytail: the graph is identified by its size only; clear cache_dir after re-downloading OSM.
        key = json.dumps({
            "graph": [graph.number_of_nodes(), graph.number_of_edges(), str(graph.graph["crs"])],
            "pairs": [[od_id, int(origin), int(destination)] for od_id, origin, destination in pairs],
            "config": config,
            "turns": asdict(turns) if turns is not None else None,
            "search": getattr(search, "__name__", None),
        }, sort_keys=True)
        path = Path(cache_dir) / f"{algorithm}_{hashlib.sha256(key.encode()).hexdigest()[:12]}.pkl"
        if path.exists():
            return pd.read_pickle(path)
    if config is not None:
        add_comfort_cost(graph, config)
    records = [
        route_record(graph, origin, destination, algorithm, od_id, turn_config=turns, search=search)
        for od_id, origin, destination in pairs
    ]
    frame = pd.DataFrame([record for record in records if record is not None])
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_pickle(path)
    return frame


def build_routes(graph, pairs, presets: dict[str, dict], *, cache_dir: Path | None = None) -> gpd.GeoDataFrame:
    """Shortest route plus one route per preset (``{"config": ..., "turns": TurnConfig | None, "search": fn}``).

    With ``cache_dir`` each preset's routes are stored on disk and reused while the graph size,
    OD pairs, weights and turn penalties stay the same. A cache hit does not reweight ``graph``.
    """

    pairs = list(pairs)
    if "shortest" in presets:
        raise ValueError("'shortest' is reserved for the physical-length baseline")
    frames = [_preset_routes(graph, pairs, "shortest", None, None, None, cache_dir)]
    frames += [
        _preset_routes(graph, pairs, algorithm, preset["config"], preset.get("turns"), preset.get("search"), cache_dir)
        for algorithm, preset in tqdm(presets.items(), desc="Routes by preset")
    ]
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=graph.graph["crs"])


def nodes_near_observations(graph, observed, max_distance_m: float = 25.0):
    """Return graph-node IDs inside the spatial support of observed edges."""

    nodes = ox.graph_to_gdfs(graph, edges=False)
    matches = gpd.sjoin_nearest(
        nodes[["geometry"]],
        observed[["geometry"]],
        how="left",
        max_distance=max_distance_m,
        distance_col="coverage_distance_m",
    )
    return matches.loc[matches["index_right"].notna()].index.unique().to_numpy()
