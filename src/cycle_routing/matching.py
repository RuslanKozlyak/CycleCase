"""Stage 3 - mapping observations onto graph edges: STADTRADELN volumes and GPX tracks."""

from __future__ import annotations

from collections import defaultdict

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from .config import MatchConfig
from .tags import edge_osmid_set


EdgeKey = tuple[int, int, int]
TRIPS = "number_of_matched_trips"


def sample_geometry(geometry, step_m: float):
    lines = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
    sampled = []
    for line in lines:
        distances = np.arange(0, line.length + step_m, step_m).clip(max=line.length)
        sampled.extend(line.interpolate(float(distance)) for distance in np.unique(distances))
    return sampled


def _nearest_indices(observed_geometry, candidates: gpd.GeoDataFrame, max_distance: float, tie_tolerance: float):
    # Line-to-line distance is zero even when two segments only touch at one endpoint. The midpoint
    # identifies the actual road piece while still selecting both directed copies of the same edge.
    distances = candidates.geometry.distance(observed_geometry.interpolate(0.5, normalized=True))
    if distances.empty or distances.min() > max_distance:
        return [], None
    minimum = float(distances.min())
    return distances.index[distances <= minimum + tie_tolerance].tolist(), minimum


def match_stadtradeln_to_graph(
    graph: nx.MultiDiGraph,
    observed: gpd.GeoDataFrame,
    config: MatchConfig | None = None,
) -> tuple[nx.MultiDiGraph, dict[str, float]]:
    """Put STADTRADELN trips on graph edges as ``stad_trips``.

    A segment is matched by ``osm_way_id`` when a same-ID edge lies within ``osmid_max_distance_m``,
    otherwise to the nearest edge within ``geometry_max_distance_m`` (plus its reverse twin, because
    volumes have no direction). Several segments on one edge are averaged weighted by segment length.
    """

    cfg = config or MatchConfig()
    if observed.crs is None or graph.graph.get("crs") is None:
        raise ValueError("Both graph and observed data must have a CRS")
    observed = observed.to_crs(graph.graph["crs"])

    edges = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True).reset_index()
    edges["edge_key"] = list(zip(edges["u"], edges["v"], edges["key"]))
    edges["osmids"] = [edge_osmid_set(row) for row in edges.to_dict("records")]
    osmid_to_edges: dict[int, list[int]] = defaultdict(list)
    for edge_index, osmids in edges["osmids"].items():
        for osmid in osmids:
            osmid_to_edges[osmid].append(edge_index)

    assignments: dict[EdgeKey, list[tuple[float, float]]] = defaultdict(list)  # edge -> [(trips, segment length)]
    distances: dict[object, float] = {}  # matched segment -> match distance

    def assign(observed_index, edge_indices, distance):
        row = observed.loc[observed_index]
        distances[observed_index] = distance
        for edge_index in edge_indices:
            assignments[edges.at[edge_index, "edge_key"]].append((float(row[TRIPS]), float(row.geometry.length)))

    unmatched = []
    for observed_index, row in observed.iterrows():
        osmid = row.get("osm_way_id")
        candidates = osmid_to_edges.get(int(osmid), []) if pd.notna(osmid) else []
        selected, distance = (
            _nearest_indices(row.geometry, edges.loc[candidates], cfg.osmid_max_distance_m, cfg.direction_tie_tolerance_m)
            if candidates else ([], None)
        )
        if selected:
            assign(observed_index, selected, distance)
        else:
            unmatched.append(observed_index)

    if unmatched:
        midpoints = observed.loc[unmatched, ["geometry"]].copy()
        midpoints.geometry = midpoints.geometry.interpolate(0.5, normalized=True)
        nearest = (
            gpd.sjoin_nearest(midpoints, edges[["geometry"]], max_distance=cfg.geometry_max_distance_m, distance_col="d")
            .reset_index(names="observed_index")
            .sort_values(["observed_index", "d", "index_right"])
            .drop_duplicates("observed_index")
        )
        for match in nearest.itertuples():
            edge_index = int(match.index_right)
            u, v, _ = edges.at[edge_index, "edge_key"]
            selected = [edge_index]
            # Mirror the directionless volume to the reverse twin of the same OSM way, never to a one-way edge.
            reverse = edges[(edges["u"] == v) & (edges["v"] == u)] if u != v else edges.iloc[0:0]
            base_osmids = edges.at[edge_index, "osmids"]
            if base_osmids and not reverse.empty:
                reverse = reverse[reverse["osmids"].map(lambda values: bool(values & base_osmids))]
            if not reverse.empty:
                midpoint = observed.at[match.observed_index, "geometry"].interpolate(0.5, normalized=True)
                close = reverse.geometry.distance(midpoint) <= match.d + cfg.direction_tie_tolerance_m
                selected += [index for index in reverse.index[close] if index != edge_index]
            assign(match.observed_index, selected, float(match.d))

    for _, _, _, data in graph.edges(keys=True, data=True):
        data["stad_trips"] = 0.0
    for edge_key, values in assignments.items():
        trips, lengths = zip(*values)
        graph.edges[edge_key]["stad_trips"] = float(np.average(trips, weights=np.maximum(lengths, 0.01)))

    matched = pd.Series(distances, dtype=float)
    metrics = {
        "stad_match_rate": len(matched) / len(observed) if len(observed) else 0.0,
        "p95_match_distance_m": float(matched.quantile(0.95)) if len(matched) else np.nan,
        "osm_edge_match_rate": len(assignments) / graph.number_of_edges() if graph.number_of_edges() else 0.0,
    }
    return graph, metrics


def snap_tracks_to_edges(
    graph,
    tracks: gpd.GeoDataFrame,
    *,
    step_m: float = 25.0,
    max_distance_m: float = 25.0,
) -> pd.DataFrame:
    """Sample every track each ``step_m`` metres and attach samples to the nearest edge."""

    rows = []
    for track in tracks.itertuples():
        points = sample_geometry(track.geometry, step_m)
        edges, distances = ox.distance.nearest_edges(
            graph, X=[point.x for point in points], Y=[point.y for point in points], return_dist=True
        )
        rows.extend(
            {"od_id": track.activity_id, "u": int(u), "v": int(v), "key": int(key), "weight": step_m}
            for (u, v, key), distance in zip(edges, distances, strict=True)
            if distance <= max_distance_m
        )
    return pd.DataFrame(rows, columns=["od_id", "u", "v", "key", "weight"])


def track_edge_usage(snapped: pd.DataFrame, features: pd.DataFrame) -> pd.Series:
    """Number of distinct tracks passing each edge, in either direction."""

    pairs = snapped.assign(pair=[frozenset(pair) for pair in zip(snapped["u"], snapped["v"])])
    counts = pairs.drop_duplicates(["od_id", "pair"])["pair"].value_counts()
    feature_pairs = [frozenset(pair) for pair in zip(features.index.get_level_values("u"), features.index.get_level_values("v"))]
    return pd.Series(counts.reindex(feature_pairs).fillna(0).to_numpy(), index=features.index, name="tracks")
