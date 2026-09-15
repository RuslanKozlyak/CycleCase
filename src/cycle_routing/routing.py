"""Shortest/comfort route construction helpers."""

from __future__ import annotations

from itertools import count
import heapq
import math

import networkx as nx
import geopandas as gpd
import numpy as np
import osmnx as ox
from shapely.geometry import LineString
from shapely.ops import linemerge

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


def _turn_kind(graph, incoming: EdgeKey, outgoing: EdgeKey, config: TurnConfig) -> str:
    incoming_coordinates = _oriented_coordinates(graph, incoming)
    outgoing_coordinates = _oriented_coordinates(graph, outgoing)
    incoming_bearing = _bearing(incoming_coordinates[-2], incoming_coordinates[-1])
    outgoing_bearing = _bearing(outgoing_coordinates[0], outgoing_coordinates[1])
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


def shortest_edge_path_with_turns(
    graph,
    origin: int,
    destination: int,
    *,
    weight: str = "comfort_cost",
    turn_config: TurnConfig,
) -> tuple[list[EdgeKey], float, float] | None:
    """Dijkstra whose state includes the exact incoming directed edge."""

    start = (origin, None)
    distances = {start: 0.0}
    previous = {}
    sequence = count()
    queue = [(0.0, next(sequence), start)]
    target = None

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
            edge_cost = float(data.get(weight, data.get("length", 1.0)))
            turn_penalty = (
                turn_config.penalty(
                    _turn_kind(graph, incoming, outgoing, turn_config)
                )
                if incoming is not None
                else 0.0
            )
            new_cost = cost + edge_cost + turn_penalty
            new_state = (neighbour, outgoing)
            if new_cost < distances.get(new_state, math.inf):
                distances[new_state] = new_cost
                previous[new_state] = (state, outgoing, turn_penalty)
                heapq.heappush(queue, (new_cost, next(sequence), new_state))

    if target is None:
        return None

    edges = []
    total_turn_penalty = 0.0
    state = target
    while state != start:
        state, edge, penalty = previous[state]
        edges.append(edge)
        total_turn_penalty += penalty
    edges.reverse()
    return edges, distances[target], total_turn_penalty


def _edges_to_gdf(graph, edges: list[EdgeKey]) -> gpd.GeoDataFrame:
    rows = []
    for u, v, key in edges:
        data = dict(graph.edges[u, v, key])
        data.update({"u": u, "v": v, "key": key})
        if data.get("geometry") is None:
            data["geometry"] = LineString(
                [(graph.nodes[u]["x"], graph.nodes[u]["y"]), (graph.nodes[v]["x"], graph.nodes[v]["y"])]
            )
        rows.append(data)
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=graph.graph["crs"])


def route_record(
    graph,
    origin: int,
    destination: int,
    algorithm: str,
    od_id: str,
    *,
    turn_config: TurnConfig | None = None,
):
    weight = "length" if algorithm == "shortest" else "comfort_cost"
    use_turns = algorithm != "shortest" and turn_config is not None and turn_config.enabled
    if use_turns:
        result = shortest_edge_path_with_turns(
            graph,
            origin,
            destination,
            weight=weight,
            turn_config=turn_config,
        )
        if result is None:
            return None
        edge_route, objective_cost, turn_penalty_total = result
        route = [origin, *(edge[1] for edge in edge_route)]
        edges = _edges_to_gdf(graph, edge_route)
    else:
        route = ox.routing.shortest_path(graph, origin, destination, weight=weight)
        if route is None or len(route) < 2:
            return None
        edges = ox.routing.route_to_gdf(graph, route, weight=weight)
        edge_route = list(edges.index)
        objective_cost = float(edges[weight].sum())
        turn_penalty_total = 0.0

    geometry = edges.geometry.union_all()
    if geometry.geom_type == "MultiLineString":
        merged = linemerge(geometry)
        if not merged.is_empty:
            geometry = merged
    return {
        "od_id": od_id,
        "algorithm": algorithm,
        "route": route,
        "edge_route": edge_route,
        "route_length_m": float(edges["length"].sum()),
        "objective_cost": objective_cost,
        "turn_penalty_total_m": turn_penalty_total,
        "route_stad_trips_mean": float(edges["stad_trips"].mean()) if "stad_trips" in edges else np.nan,
        "geometry": geometry,
    }


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


def build_routes(
    graph,
    pairs,
    *,
    turn_config: TurnConfig | None = None,
    comfort_configs=None,
) -> gpd.GeoDataFrame:
    """Build a shortest route and one or more named comfort candidates."""

    pairs = list(pairs)
    records = [
        route_record(graph, origin, destination, "shortest", od_id)
        for od_id, origin, destination in pairs
    ]
    if comfort_configs is None:
        comfort_configs = {"comfort": None}
    for algorithm, config in comfort_configs.items():
        if algorithm == "shortest":
            raise ValueError("'shortest' is reserved for the physical-length baseline")
        if config is not None:
            add_comfort_cost(graph, config)
        records.extend(
            route_record(
                graph,
                origin,
                destination,
                algorithm,
                od_id,
                turn_config=turn_config,
            )
            for od_id, origin, destination in pairs
        )
    return gpd.GeoDataFrame(
        [record for record in records if record is not None],
        geometry="geometry",
        crs=graph.graph["crs"],
    )


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
