"""Auditable STADTRADELN-to-OSM edge matching."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from .config import MatchConfig
from .tags import edge_osmid_set


EdgeKey = tuple[int, int, int]


def _normalise_osmid(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _aggregate(assignments: list[tuple[float, float]], how: str) -> float:
    values = np.asarray([item[0] for item in assignments], dtype=float)
    if how == "sum":
        return float(values.sum())
    if how == "max":
        return float(values.max())
    if how == "mean":
        return float(values.mean())
    weights = np.asarray([max(item[1], 0.01) for item in assignments], dtype=float)
    return float(np.average(values, weights=weights))


def _edge_table(graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    edges = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True).reset_index()
    edges["edge_key"] = list(zip(edges["u"], edges["v"], edges["key"]))
    edges["osmids"] = [edge_osmid_set(row) for row in edges.to_dict("records")]
    return edges


def _nearest_candidate_indices(
    observed_geometry,
    candidates: gpd.GeoDataFrame,
    *,
    max_distance: float,
    tie_tolerance: float,
) -> tuple[list[int], float | None]:
    # Line-to-line distance is zero even when two segments only touch at one
    # endpoint. The midpoint identifies the actual road piece while still
    # selecting both directed copies of the same physical edge.
    midpoint = observed_geometry.interpolate(0.5, normalized=True)
    distances = candidates.geometry.distance(midpoint)
    if distances.empty or not np.isfinite(distances.min()):
        return [], None
    minimum = float(distances.min())
    if minimum > max_distance:
        return [], minimum
    selected = distances.index[distances <= minimum + tie_tolerance].tolist()
    return selected, minimum


def match_stadtradeln_to_graph(
    graph: nx.MultiDiGraph,
    observed: gpd.GeoDataFrame,
    config: MatchConfig | None = None,
) -> tuple[nx.MultiDiGraph, dict[str, Any], gpd.GeoDataFrame]:
    """Match each observed segment once, then mirror equal directed OSM edges.

    Exact ``osm_way_id`` candidates are validated spatially. A geometry-only
    fallback is used only when the ID pass fails. Multiple STAD subdivisions
    assigned to one OSM edge are length-weighted by default, never blindly
    summed. The returned audit table exposes every decision and distance.
    """

    cfg = config or MatchConfig()
    if observed.crs is None or graph.graph.get("crs") is None:
        raise ValueError("Both graph and observed data must have a CRS")
    graph_crs = str(graph.graph["crs"])
    if observed.crs != gpd.GeoSeries(crs=graph_crs).crs:
        observed = observed.to_crs(graph_crs)
    if cfg.flow_field not in observed.columns:
        raise KeyError(f"Missing flow column {cfg.flow_field!r}")

    edges = _edge_table(graph)
    osmid_to_edges: dict[int, list[int]] = defaultdict(list)
    for edge_index, osmids in edges["osmids"].items():
        for osmid in osmids:
            osmid_to_edges[osmid].append(edge_index)

    assignments: dict[EdgeKey, list[tuple[float, float]]] = defaultdict(list)
    methods: dict[EdgeKey, set[str]] = defaultdict(set)
    audit_rows: list[dict[str, Any]] = []
    unmatched_indices: list[Any] = []

    has_osmid = cfg.osmid_field in observed.columns
    for observed_index, row in observed.iterrows():
        osmid = _normalise_osmid(row.get(cfg.osmid_field)) if has_osmid else None
        candidate_ids = osmid_to_edges.get(osmid, []) if osmid is not None else []
        selected: list[int] = []
        distance = None
        if candidate_ids:
            selected, distance = _nearest_candidate_indices(
                row.geometry,
                edges.loc[candidate_ids],
                max_distance=cfg.osmid_max_distance_m,
                tie_tolerance=cfg.direction_tie_tolerance_m,
            )
        if selected:
            method = "osmid+geometry"
        else:
            method = "unmatched"
            unmatched_indices.append(observed_index)

        edge_keys = [edges.at[index, "edge_key"] for index in selected]
        flow = float(row[cfg.flow_field])
        observed_length = float(row.geometry.length)
        for edge_key in edge_keys:
            assignments[edge_key].append((flow, observed_length))
            methods[edge_key].add(method)
        audit_rows.append(
            {
                "observed_index": observed_index,
                "osmid": osmid,
                "match_method": method,
                "match_distance_m": distance,
                "matched_edge_count": len(edge_keys),
                "matched_edges": edge_keys,
            }
        )

    # Geometry fallback is vectorised, then reduced to one nearest physical edge.
    if unmatched_indices:
        fallback = observed.loc[unmatched_indices, [cfg.flow_field, "geometry"]].copy()
        fallback["source_geometry"] = fallback.geometry
        fallback.geometry = fallback.geometry.interpolate(0.5, normalized=True)
        fallback["observed_index"] = fallback.index
        joined = gpd.sjoin_nearest(
            fallback,
            edges[["edge_key", "geometry"]],
            how="left",
            max_distance=cfg.geometry_max_distance_m,
            distance_col="match_distance_m",
        )
        joined = joined.sort_values(["observed_index", "match_distance_m", "index_right"])
        best = joined.drop_duplicates("observed_index")
        audit_position = {row["observed_index"]: position for position, row in enumerate(audit_rows)}
        for _, match in best.iterrows():
            if pd.isna(match.get("index_right")):
                continue
            observed_index = match["observed_index"]
            edge_index = int(match["index_right"])
            observed_row = observed.loc[observed_index]
            edge_key = edges.at[edge_index, "edge_key"]
            selected_indices = [edge_index]

            # STAD traffic volume has no direction column. If the nearest OSM
            # edge has an equivalent reverse edge, transfer the same observed
            # flow to it as well. Do not mirror a genuinely one-way edge.
            u, v, _ = edge_key
            if u != v:
                reverse = edges[(edges["u"] == v) & (edges["v"] == u)]
                base_osmids = edges.at[edge_index, "osmids"]
                if not reverse.empty:
                    if base_osmids:
                        same_osmid = reverse["osmids"].map(
                            lambda values: bool(values & base_osmids)
                        ).astype(bool)
                        reverse = reverse.loc[same_osmid]
                    if not reverse.empty:
                        midpoint = observed_row.geometry.interpolate(0.5, normalized=True)
                        reverse_distances = gpd.GeoSeries(
                            reverse["geometry"], index=reverse.index, crs=edges.crs
                        ).distance(midpoint)
                        reverse = reverse.loc[
                            reverse_distances
                            <= float(match["match_distance_m"])
                            + cfg.direction_tie_tolerance_m
                        ]
                        selected_indices.extend(
                            index for index in reverse.index if index != edge_index
                        )

            edge_keys = [edges.at[index, "edge_key"] for index in selected_indices]
            for selected_key in edge_keys:
                assignments[selected_key].append(
                    (
                        float(observed_row[cfg.flow_field]),
                        float(observed_row.geometry.length),
                    )
                )
                methods[selected_key].add("geometry")
            audit = audit_rows[audit_position[observed_index]]
            audit.update(
                {
                    "match_method": "geometry",
                    "match_distance_m": float(match["match_distance_m"]),
                    "matched_edge_count": len(edge_keys),
                    "matched_edges": edge_keys,
                }
            )

    for _, _, _, data in graph.edges(keys=True, data=True):
        data["stad_trips"] = 0.0
        data["stad_match_method"] = "none"
    for edge_key, values in assignments.items():
        data = graph.edges[edge_key]
        data["stad_trips"] = _aggregate(values, cfg.aggregate_trips)
        data["stad_match_method"] = "+".join(sorted(methods[edge_key]))
        data["stad_segment_count"] = len(values)

    audit = gpd.GeoDataFrame(audit_rows).merge(
        observed[["geometry"]], left_on="observed_index", right_index=True, how="left"
    )
    audit = gpd.GeoDataFrame(audit, geometry="geometry", crs=observed.crs)
    method_counts = audit["match_method"].value_counts()
    matched = audit["match_method"] != "unmatched"
    unique_ids = audit["osmid"].dropna().nunique()
    matched_unique_ids = audit.loc[matched, "osmid"].dropna().nunique()
    edge_method_counts = pd.Series(
        [data.get("stad_match_method", "none") for _, _, _, data in graph.edges(keys=True, data=True)]
    ).value_counts()
    metrics = {
        "stad_segments": len(audit),
        "stad_unique_osmids": int(unique_ids),
        "stad_matched_via_osmid": int(method_counts.get("osmid+geometry", 0)),
        "stad_matched_via_geometry_only": int(method_counts.get("geometry", 0)),
        "stad_unmatched": int(method_counts.get("unmatched", 0)),
        "stad_match_rate": float(matched.mean()) if len(audit) else 0.0,
        "stad_unique_osmid_match_rate": float(matched_unique_ids / unique_ids) if unique_ids else 0.0,
        "median_match_distance_m": float(audit.loc[matched, "match_distance_m"].median()) if matched.any() else np.nan,
        "p95_match_distance_m": float(audit.loc[matched, "match_distance_m"].quantile(0.95)) if matched.any() else np.nan,
        "osm_edges": graph.number_of_edges(),
        "osm_edges_matched": int(graph.number_of_edges() - edge_method_counts.get("none", 0)),
        "osm_edge_match_rate": float(1 - edge_method_counts.get("none", 0) / graph.number_of_edges()) if graph.number_of_edges() else 0.0,
        "aggregate_trips": cfg.aggregate_trips,
    }
    return graph, metrics, audit
