"""Popularity and observed-GPX evaluation metrics."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import osmnx as ox
import pandas as pd
from shapely.geometry import Point
from shapely.ops import linemerge
from tqdm.auto import tqdm

from .comfort import add_comfort_cost
from .routing import route_record
from .config import TurnConfig


def sample_geometry(geometry, step_m: float):
    lines = list(geometry.geoms) if geometry.geom_type == "MultiLineString" else [geometry]
    sampled = []
    for line in lines:
        distances = np.arange(0, line.length + step_m, step_m).clip(max=line.length)
        sampled.extend(line.interpolate(float(distance)) for distance in np.unique(distances))
    return sampled


def sample_routes(routes: gpd.GeoDataFrame, step_m: float) -> gpd.GeoDataFrame:
    records = []
    for row in routes.itertuples():
        line = linemerge(row.geometry) if row.geometry.geom_type == "MultiLineString" else row.geometry
        if line.geom_type != "LineString":
            continue
        for sequence, point in enumerate(sample_geometry(line, step_m)):
            records.append({"od_id": row.od_id, "algorithm": row.algorithm, "sequence": sequence, "geometry": point})
    return gpd.GeoDataFrame(records, geometry="geometry", crs=routes.crs)


def evaluate_corridor_popularity(
    routes: gpd.GeoDataFrame,
    observed_edges: gpd.GeoDataFrame,
    *,
    sample_step_m: float = 25.0,
    match_distance_m: float = 25.0,
    popular_threshold: float | None = None,
) -> pd.DataFrame:
    threshold = (
        float(popular_threshold)
        if popular_threshold is not None
        else float(observed_edges["number_of_matched_trips"].quantile(0.75))
    )
    rows = []
    observed_index = observed_edges[["number_of_matched_trips", "geometry"]]
    for route in tqdm(routes.itertuples(), total=len(routes), desc="STAD matching"):
        points = sample_geometry(route.geometry, sample_step_m)
        samples = gpd.GeoDataFrame({"sample_id": range(len(points))}, geometry=points, crs=routes.crs)
        matched = (
            gpd.sjoin_nearest(
                samples,
                observed_index,
                how="left",
                max_distance=match_distance_m,
                distance_col="match_distance_m",
            )
            .sort_values("match_distance_m")
            .drop_duplicates("sample_id")
        )
        covered = matched["number_of_matched_trips"].notna()
        flows = matched.loc[covered, "number_of_matched_trips"]
        rows.append(
            {
                "od_id": route.od_id,
                "algorithm": route.algorithm,
                "route_length_m": route.route_length_m,
                "objective_cost": getattr(route, "objective_cost", np.nan),
                "turn_penalty_total_m": getattr(route, "turn_penalty_total_m", 0.0),
                "match_rate": float(covered.mean()),
                "popularity_log_mean": float(np.log1p(flows).mean()) if len(flows) else np.nan,
                "flow_median": float(flows.median()) if len(flows) else np.nan,
                "popular_edge_share": float((flows >= threshold).mean()) if len(flows) else np.nan,
                "route_stad_trips_mean": getattr(route, "route_stad_trips_mean", np.nan),
            }
        )
    return pd.DataFrame(rows)


def fraction_near(source_line, target_line, step_m: float, tolerance_m: float) -> float:
    samples = sample_geometry(source_line, step_m)
    return float(np.mean([point.distance(target_line) <= tolerance_m for point in samples]))


def route_overlap(predicted, observed_line, *, step_m: float = 25.0, tolerance_m: float = 35.0):
    precision = fraction_near(predicted, observed_line, step_m, tolerance_m)
    recall = fraction_near(observed_line, predicted, step_m, tolerance_m)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_personal_routes(
    graph,
    personal_tracks: gpd.GeoDataFrame,
    *,
    step_m: float = 25.0,
    tolerance_m: float = 35.0,
    turn_config: TurnConfig | None = None,
    comfort_configs=None,
) -> pd.DataFrame:
    """Compare shortest and named comfort models with observed GPX tracks."""

    if comfort_configs is None:
        comfort_configs = {"comfort": None}
    rows = []
    activities = list(personal_tracks.itertuples())
    locations = []
    for activity in activities:
        start = Point(activity.geometry.coords[0])
        end = Point(activity.geometry.coords[-1])
        locations.append(
            (
                activity,
                int(ox.distance.nearest_nodes(graph, X=start.x, Y=start.y)),
                int(ox.distance.nearest_nodes(graph, X=end.x, Y=end.y)),
            )
        )

    algorithms = [("shortest", None), *comfort_configs.items()]
    for algorithm, config in algorithms:
        if config is not None:
            add_comfort_cost(graph, config)
        for activity, origin, destination in tqdm(
            locations,
            total=len(locations),
            desc=f"Personal GPX: {algorithm}",
        ):
            predicted = route_record(
                graph,
                origin,
                destination,
                algorithm,
                activity.activity_id,
                turn_config=turn_config,
            )
            if predicted is None:
                continue
            precision, recall, f1 = route_overlap(
                predicted["geometry"], activity.geometry, step_m=step_m, tolerance_m=tolerance_m
            )
            rows.append(
                {
                    "activity_id": activity.activity_id,
                    "algorithm": algorithm,
                    "route_length_m": predicted["route_length_m"],
                    "objective_cost": predicted["objective_cost"],
                    "turn_penalty_total_m": predicted["turn_penalty_total_m"],
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "edge_route": predicted["edge_route"],
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    shortest = (
        result.loc[result["algorithm"] == "shortest", ["activity_id", "route_length_m"]]
        .rename(columns={"route_length_m": "shortest_length_m"})
    )
    result = result.merge(shortest, on="activity_id", how="left", validate="many_to_one")
    result["detour_pct"] = 100 * (
        result["route_length_m"] / result["shortest_length_m"] - 1
    )
    return result


def compare_route_models(
    metrics: pd.DataFrame,
    *,
    baseline: str = "shortest",
) -> pd.DataFrame:
    """Return one long-form comparison row per OD and non-baseline model."""

    baseline_rows = metrics.loc[metrics["algorithm"] == baseline].copy()
    baseline_rows = baseline_rows.rename(
        columns={
            column: f"baseline_{column}"
            for column in baseline_rows.columns
            if column not in {"od_id", "algorithm"}
        }
    ).drop(columns="algorithm")
    candidates = metrics.loc[metrics["algorithm"] != baseline].copy()
    result = candidates.merge(baseline_rows, on="od_id", how="inner", validate="many_to_one")
    result = result.rename(columns={"algorithm": "model"})
    result["baseline_algorithm"] = baseline
    result["detour_pct"] = 100 * (
        result["route_length_m"] / result["baseline_route_length_m"] - 1
    )
    if "popularity_log_mean" in result:
        result["popularity_gain"] = (
            result["popularity_log_mean"] - result["baseline_popularity_log_mean"]
        )
    if "match_rate" in result:
        result["min_match_rate"] = result[["match_rate", "baseline_match_rate"]].min(axis=1)
    return result


def compare_algorithms(metrics: pd.DataFrame) -> pd.DataFrame:
    """Put shortest and comfort metrics side by side for each OD pair."""

    shortest = metrics.query("algorithm == 'shortest'").add_suffix("_shortest")
    comfort = metrics.query("algorithm == 'comfort'").add_suffix("_comfort")
    comparison = comfort.merge(
        shortest,
        left_on="od_id_comfort",
        right_on="od_id_shortest",
        validate="one_to_one",
    )
    comparison["detour_pct"] = 100 * (
        comparison["route_length_m_comfort"] / comparison["route_length_m_shortest"] - 1
    )
    comparison["popularity_gain"] = (
        comparison["popularity_log_mean_comfort"]
        - comparison["popularity_log_mean_shortest"]
    )
    return comparison


def summarize_personal_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    return (
        metrics.groupby("algorithm")
        .agg(
            activities=("activity_id", "nunique"),
            median_precision=("precision", "median"),
            median_recall=("recall", "median"),
            median_f1=("f1", "median"),
            median_detour_pct=("detour_pct", "median"),
        )
        .reset_index()
    )


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


def route_edges_long(routes: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Explode ``edge_route`` lists into one row per traversed edge weighted by its length."""

    rows = [
        {"group": row.algorithm, "od_id": row.od_id, "u": u, "v": v, "key": key}
        for row in routes.itertuples()
        for u, v, key in row.edge_route
    ]
    result = pd.DataFrame(rows)
    result["weight"] = features["length_m"].reindex(pd.MultiIndex.from_frame(result[["u", "v", "key"]])).to_numpy()
    return result


def composition_table(edge_rows: pd.DataFrame, features: pd.DataFrame, column: str) -> pd.DataFrame:
    """Share of travelled length (%) in every category of ``features[column]`` per group."""

    values = features[column].reindex(pd.MultiIndex.from_frame(edge_rows[["u", "v", "key"]])).to_numpy()
    totals = edge_rows.assign(category=values).groupby(["group", "category"], sort=False)["weight"].sum()
    return (100 * totals / totals.groupby(level="group").transform("sum")).unstack(fill_value=0.0)


def route_edge_association(
    routes: pd.DataFrame,
    features: gpd.GeoDataFrame,
    value_column: str,
    *,
    popular_threshold: float,
    buffer_m: float = 300.0,
) -> pd.DataFrame:
    """How strongly route edge choice follows edge popularity inside each OD corridor.

    The corridor is every edge within ``buffer_m`` of any candidate route for that OD.
    Two directions of a street count as one edge.
    """

    streets = features[[value_column, "length_m", "geometry"]].copy()
    streets["pair"] = [frozenset(pair) for pair in zip(streets.index.get_level_values("u"), streets.index.get_level_values("v"))]
    streets = streets.sort_values(value_column, ascending=False).drop_duplicates("pair")
    streets = streets.set_index("pair")
    rows = []
    for od_id, od_routes in routes.groupby("od_id", sort=False):
        selected = {
            row.algorithm: {frozenset((u, v)) for u, v, _ in row.edge_route}
            for row in od_routes.itertuples()
        }
        chosen = streets.loc[streets.index.isin(set().union(*selected.values()))]
        area = chosen.geometry.union_all().buffer(buffer_m)
        corridor = streets.iloc[streets.sindex.query(area, predicate="intersects")]
        for algorithm, pairs in selected.items():
            rows.append(pd.DataFrame({
                "od_id": od_id,
                "algorithm": algorithm,
                "selected": corridor.index.isin(pairs),
                "length_m": corridor["length_m"].to_numpy(),
                "value": corridor[value_column].to_numpy(),
            }))
    pooled = pd.concat(rows, ignore_index=True)
    pooled["log_value"] = np.log1p(pooled["value"])
    pooled["popular"] = pooled["value"] >= popular_threshold

    def weighted_share(frame, mask):
        return float(frame.loc[mask, "length_m"].sum() / frame["length_m"].sum())

    summary = []
    corridor_rows = pooled.loc[pooled["algorithm"] == pooled["algorithm"].iloc[0]]
    for algorithm, frame in pooled.groupby("algorithm", sort=False):
        route = frame.loc[frame["selected"]]
        weights = frame["length_m"].to_numpy()
        cov = np.cov(frame["selected"].astype(float), frame["log_value"], aweights=weights)
        corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        summary.append({
            "algorithm": algorithm,
            "corr_selected_vs_log_popularity": float(corr),
            "popular_length_share": weighted_share(route, route["popular"]),
            "popularity_lift": float(
                np.average(route["value"], weights=route["length_m"])
                / np.average(frame["value"], weights=weights)
            ),
        })
    summary.append({
        "algorithm": "corridor (all edges)",
        "corr_selected_vs_log_popularity": np.nan,
        "popular_length_share": weighted_share(corridor_rows, corridor_rows["popular"]),
        "popularity_lift": 1.0,
    })
    return pd.DataFrame(summary).set_index("algorithm")
