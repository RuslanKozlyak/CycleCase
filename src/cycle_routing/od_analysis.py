"""Describe OD coverage and construct a deterministic diagnostic set without popularity labels."""

from __future__ import annotations

from collections import Counter

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from pyproj import Transformer

from .evaluation import route_popularity_metrics
from .graph import build_graph


LENGTH_BANDS = ("1.5–3 km", "3–6 km", "6–10 km")
ORIENTATIONS = ("E–W", "NE–SW", "N–S", "NW–SE")


def _bands(length_m, dx, dy):
    angle = np.degrees(np.arctan2(dy, dx)) % 180
    return LENGTH_BANDS[min(int(np.searchsorted([3_000, 6_000], length_m, side="right")), 2)], ORIENTATIONS[int((angle + 22.5) % 180 // 45)]


def _street_lengths(edges):
    table = pd.DataFrame({
        "street": [tuple(sorted((u, v))) for u, v in zip(edges.u, edges.v)],
        "length": edges.length_m.to_numpy(),
    })
    return table.groupby("street").length.mean().to_dict()


def _streets(path):
    return {tuple(sorted((int(u), int(v)))) for u, v, _ in path}


def route_overlap(routes: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Length-weighted Jaccard of physical streets; reverse directions count together."""

    lengths = _street_lengths(edges)
    sets = {row.od_id: _streets(row.edge_route) for row in routes.itertuples()}
    ids = list(sets)
    result = pd.DataFrame(0.0, index=ids, columns=ids)
    for i, a in enumerate(ids):
        for b in ids[i:]:
            union = sum(lengths[e] for e in sets[a] | sets[b])
            value = sum(lengths[e] for e in sets[a] & sets[b]) / union if union else 0.0
            result.loc[a, b] = result.loc[b, a] = value
    return result


def describe_od_pairs(edges, nodes, pairs, shortest_routes, *, surveyed, partition=None, known=None):
    """Shortest-route distance, direction, geography, road types and survey coverage per OD."""

    coordinates = nodes.set_index("node")[["x", "y"]]
    physical = nx.Graph()
    physical.add_edges_from(zip(edges.u, edges.v))
    degree = dict(physical.degree())
    bridges = {tuple(sorted(edge)) for edge in nx.bridges(physical)}
    tags = edges[["u", "v", "key", "length_m", "highway"]].assign(surveyed=surveyed)
    if partition is not None:
        tags = tags.assign(partition=partition)
    if known is not None:
        tags = tags.assign(has_observation=known)
    tags = tags.set_index(["u", "v", "key"])
    endpoints = pairs.set_index("od_id")
    rows = []
    for route in shortest_routes.itertuples():
        pair = endpoints.loc[route.od_id]
        a = coordinates.loc[int(pair.origin)].to_numpy()
        b = coordinates.loc[int(pair.destination)].to_numpy()
        used = tags.loc[[tuple(edge) for edge in route.edge_route]]
        length = used.length_m.sum()
        direct = np.linalg.norm(b - a)
        band, direction = _bands(length, *(b - a))
        highway = used.highway.fillna("").str.split(";")
        path_nodes = {int(n) for u, v, _ in route.edge_route for n in (u, v)}
        bridge_rows = [tuple(sorted((int(u), int(v)))) in bridges for u, v, _ in route.edge_route]
        rows.append({
            "od_id": route.od_id, "origin": int(pair.origin), "destination": int(pair.destination),
            "start_x": a[0], "start_y": a[1], "end_x": b[0], "end_y": b[1],
            "length_km": length / 1000, "direct_km": direct / 1000,
            "network_to_direct": length / direct if direct else np.nan,
            "length_band": band, "orientation": direction,
            "junctions_per_km": sum(degree[n] >= 3 for n in path_nodes) / (length / 1000),
            "unavoidable_street_pct": 100 * used.loc[bridge_rows, "length_m"].sum() / length,
            "survey_coverage_pct": 100 * used.loc[used.surveyed, "length_m"].sum() / length,
            **({"measurement_coverage_pct": 100 * used.loc[used.has_observation, "length_m"].sum() / length}
               if known is not None else {}),
            "cycleway_pct": 100 * used.loc[highway.map(lambda v: "cycleway" in v), "length_m"].sum() / length,
            "major_road_pct": 100 * used.loc[highway.map(lambda v: bool(set(v) & {"primary", "secondary"})), "length_m"].sum() / length,
            **({name + "_coverage_pct": 100 * used.loc[used.partition.eq(name), "length_m"].sum() / length
                for name in ("train", "validation", "test")} if partition is not None else {}),
            **({name + "_measurement_coverage_pct": 100 * used.loc[used.partition.eq(name) & used.has_observation, "length_m"].sum() / length
                for name in ("train", "validation", "test")} if partition is not None and known is not None else {}),
        })
    return pd.DataFrame(rows).set_index("od_id")


def od_set_summary(description, shortest_routes, edges):
    points = np.vstack([description[["start_x", "start_y"]], description[["end_x", "end_y"]]])
    cells = {tuple(row) for row in np.floor(points / 1000).astype(int)}
    streets = [_streets(path) for path in shortest_routes.edge_route]
    lengths = _street_lengths(edges)
    union = set().union(*streets)
    overlap = route_overlap(shortest_routes, edges).to_numpy()
    counts = Counter(e for path in streets for e in path)
    return pd.Series({
        "OD": len(description), "unique_endpoints": len(set(description.origin) | set(description.destination)),
        "endpoint_cells_1km": len(cells), "median_length_km": description.length_km.median(),
        "median_network_to_direct": description.network_to_direct.median(),
        "median_junctions_per_km": description.junctions_per_km.median(),
        "median_major_road_pct": description.major_road_pct.median(),
        "median_survey_coverage_pct": description.survey_coverage_pct.median(),
        "unique_shortest_streets_km": sum(lengths[e] for e in union) / 1000,
        "mean_pairwise_overlap_pct": 100 * overlap[np.triu_indices(len(overlap), 1)].mean() if len(overlap) > 1 else 0,
        "max_routes_on_same_street": max(counts.values(), default=0),
        **({"shortest_without_test": int(description.test_coverage_pct.eq(0).sum())}
           if "test_coverage_pct" in description else {}),
        **({"median_measurement_coverage_pct": description.measurement_coverage_pct.median(),
            "shortest_without_test_measurement": int(description.test_measurement_coverage_pct.eq(0).sum())}
           if "test_measurement_coverage_pct" in description else {}),
    })


def select_topology_od(
    edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, candidate_nodes, *,
    surveyed: pd.Series, n_pairs=20, n_anchors=28, min_length_m=1_500,
    max_length_m=10_000, min_survey_coverage=0.75,
):
    """Deterministic spatial/distance/direction/corridor coverage, without trips or preset results.

    Farthest-point junction anchors avoid sampling in proportion to node density. Shortest
    paths constrain length and survey support. Greedy selection cycles length/direction strata,
    preferring unused endpoints and physical corridors. It is a diagnostic set, not OD demand.
    """

    if n_pairs < 1 or n_anchors < 2 or not 0 <= min_survey_coverage <= 1:
        raise ValueError("Need positive pair count, at least two anchors and coverage in [0, 1]")
    graph = build_graph(edges, nodes, "length_m")
    strong = max(nx.strongly_connected_components(graph), key=len)
    candidates = sorted({int(node) for node in candidate_nodes if node in strong})
    junctions = [n for n in candidates if len(set(graph.predecessors(n)) | set(graph.successors(n))) >= 3]
    if len(junctions) >= n_anchors:
        candidates = junctions
    if len(candidates) < 2:
        raise ValueError("Not enough candidate nodes in one strongly connected component")
    xy = np.array([(graph.nodes[n]["x"], graph.nodes[n]["y"]) for n in candidates])
    chosen = [int(np.argmin(((xy - xy.mean(axis=0)) ** 2).sum(axis=1)))]
    distance = ((xy - xy[chosen[0]]) ** 2).sum(axis=1)
    while len(chosen) < min(n_anchors, len(candidates)):
        index = int(np.argmax(distance))
        if distance[index] == 0:
            break
        chosen.append(index)
        distance = np.minimum(distance, ((xy - xy[index]) ** 2).sum(axis=1))
    anchors = [candidates[i] for i in chosen]
    lengths = _street_lengths(edges)
    lookup = edges[["u", "v", "key", "length_m"]].assign(surveyed=surveyed).set_index(["u", "v", "key"])
    pool = []
    for i, origin in enumerate(anchors):
        distances, paths = nx.single_source_dijkstra(graph, origin, cutoff=max_length_m, weight="length_m")
        for destination in anchors[i + 1:]:
            length = distances.get(destination, 0)
            if not min_length_m <= length <= max_length_m:
                continue
            node_path = paths[destination]
            path = [(u, v, min(graph[u][v], key=lambda k: (graph[u][v][k]["length_m"], k)))
                    for u, v in zip(node_path, node_path[1:])]
            used = lookup.loc[path]
            coverage = float(used.loc[used.surveyed, "length_m"].sum() / length)
            if coverage < min_survey_coverage:
                continue
            a = np.array([graph.nodes[origin]["x"], graph.nodes[origin]["y"]])
            b = np.array([graph.nodes[destination]["x"], graph.nodes[destination]["y"]])
            band, direction = _bands(length, *(b - a))
            pool.append({"origin": origin, "destination": destination, "length_band": band,
                         "orientation": direction, "length_km": length / 1000,
                         "survey_coverage_pct": 100 * coverage, "streets": _streets(path)})
    if len(pool) < n_pairs:
        raise ValueError(f"Only {len(pool)} eligible topological OD candidates; need {n_pairs}")
    selected, endpoint_use, union, pair_use = [], Counter(), set(), set()
    for step in range(n_pairs):
        available = [r for r in pool if (r["origin"], r["destination"]) not in pair_use]
        target_length = LENGTH_BANDS[step % len(LENGTH_BANDS)]
        target_direction = ORIENTATIONS[step % len(ORIENTATIONS)]
        strata = [r for r in available if r["length_band"] == target_length and r["orientation"] == target_direction]
        if not strata:
            strata = [r for r in available if r["length_band"] == target_length] or available
        def score(row):
            street_set = row["streets"]
            total = sum(lengths[e] for e in street_set)
            novel = sum(lengths[e] for e in street_set - union) / total
            unused = sum(endpoint_use[n] == 0 for n in (row["origin"], row["destination"]))
            return 2 * novel + unused - 0.25 * (endpoint_use[row["origin"]] + endpoint_use[row["destination"]])
        winner = max(strata, key=score)
        selected.append(winner)
        pair_use.add((winner["origin"], winner["destination"]))
        endpoint_use.update((winner["origin"], winner["destination"]))
        union.update(winner["streets"])
    to_wgs84 = Transformer.from_crs(edges.crs, "EPSG:4326", always_xy=True)
    records = []
    for i, row in enumerate(selected, 1):
        a, b = graph.nodes[row["origin"]], graph.nodes[row["destination"]]
        lon, lat = to_wgs84.transform([a["x"], b["x"]], [a["y"], b["y"]])
        records.append({"od_id": f"TOPO-{i:02d}", "origin": row["origin"], "destination": row["destination"],
                        "start_lat": lat[0], "start_lon": lon[0], "end_lat": lat[1], "end_lon": lon[1],
                        "start_label": "start", "end_label": "finish"})
    return pd.DataFrame(records), pd.DataFrame([{k: v for k, v in r.items() if k != "streets"} for r in pool])


def evaluate_od_models(routes, edges, trips, *, mask, min_coverage_pct=10):
    """Fair model comparison: same covered OD subset and lengths for every metric in a scope."""

    details = route_popularity_metrics(routes, edges, trips, mask=mask)
    baseline = details[details.algorithm.eq("shortest")].set_index("od_id")
    details = details.join(baseline[["route_length_m", "popularity_log_mean"]].add_prefix("baseline_"), on="od_id")
    details["gain"] = details.popularity_log_mean - details.baseline_popularity_log_mean
    details["detour_pct"] = 100 * (details.route_length_m / details.baseline_route_length_m - 1)
    counts = details.assign(covered=details.coverage_pct.ge(min_coverage_pct) & details.gain.notna()).groupby("od_id").agg(
        all_covered=("covered", "all"), models=("algorithm", "nunique"),
    )
    common = counts.index[counts.all_covered & counts.models.eq(routes.algorithm.nunique())]
    details["included"] = details.od_id.isin(common)
    path_data = routes.copy()
    path_data["signature"] = path_data.edge_route.map(lambda p: tuple(tuple(e) for e in p))
    signatures = path_data.set_index(["algorithm", "od_id"]).signature
    base_signatures = signatures.loc["shortest"]
    rows = []
    for algorithm, full in details.groupby("algorithm", sort=False):
        frame = full[full.included]
        selected = routes[routes.algorithm.eq(algorithm) & routes.od_id.isin(common)]
        changed = sum(path != base_signatures.loc[od] for od, path in signatures.loc[algorithm].items())
        rows.append({"algorithm": algorithm, "n_routes": len(full), "n_evaluated": len(frame),
                     "changed_vs_shortest": changed, "popularity_ratio": np.exp(frame.gain.median()),
                     "median_log_gain": frame.gain.median(), "median_detour_pct": frame.detour_pct.median(),
                     "all_od_median_detour_pct": full.detour_pct.median(),
                     "p90_detour_pct": frame.detour_pct.quantile(0.9),
                     "median_turns": selected.turns.median(), "median_coverage_pct": frame.coverage_pct.median(),
                     "wins_vs_shortest": int((frame.gain > 0).sum())})
    return pd.DataFrame(rows).set_index("algorithm"), details
