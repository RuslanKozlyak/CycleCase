"""Stage 5 - metrics for already built routes: popularity and road composition."""

from __future__ import annotations

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .matching import TRIPS, sample_geometry


def evaluate_corridor_popularity(
    routes: gpd.GeoDataFrame,
    observed: gpd.GeoDataFrame,
    *,
    popular_threshold: float,
    sample_step_m: float = 25.0,
    match_distance_m: float = 25.0,
) -> pd.DataFrame:
    """STADTRADELN trips seen from points every ``sample_step_m`` along each route."""

    observed = observed[[TRIPS, "geometry"]]
    rows = []
    for route in tqdm(routes.itertuples(), total=len(routes), desc="STAD matching"):
        points = sample_geometry(route.geometry, sample_step_m)
        samples = gpd.GeoDataFrame({"sample_id": range(len(points))}, geometry=points, crs=routes.crs)
        flows = (
            gpd.sjoin_nearest(samples, observed, max_distance=match_distance_m, distance_col="d")
            .sort_values("d")
            .drop_duplicates("sample_id")[TRIPS]
        )
        rows.append({
            "od_id": route.od_id,
            "algorithm": route.algorithm,
            "route_length_m": route.route_length_m,
            "turns": route.turns,
            "popularity_log_mean": float(np.log1p(flows).mean()) if len(flows) else np.nan,
            "popular_edge_share": float((flows >= popular_threshold).mean()) if len(flows) else np.nan,
        })
    return pd.DataFrame(rows)


def compare_route_models(metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per OD and preset with detour and popularity gain against the shortest route."""

    shortest = metrics.loc[metrics["algorithm"] == "shortest"].drop(columns="algorithm").add_prefix("baseline_")
    result = (
        metrics.loc[metrics["algorithm"] != "shortest"]
        .merge(shortest, left_on="od_id", right_on="baseline_od_id", validate="many_to_one")
        .rename(columns={"algorithm": "model"})
    )
    result["detour_pct"] = 100 * (result["route_length_m"] / result["baseline_route_length_m"] - 1)
    result["popularity_gain"] = result["popularity_log_mean"] - result["baseline_popularity_log_mean"]
    return result


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
    buffer_m: float = 300.0,
) -> pd.Series:
    """Per preset: length-weighted correlation of "edge chosen" with ``log(1 + value)``.

    Pooled over OD corridors: every street within ``buffer_m`` of any route of that OD. Two directions
    of a street count as one edge.
    """

    streets = features[[value_column, "length_m", "geometry"]].copy()
    streets["pair"] = [frozenset(pair) for pair in zip(streets.index.get_level_values("u"), streets.index.get_level_values("v"))]
    streets = streets.sort_values(value_column, ascending=False).drop_duplicates("pair").set_index("pair")
    rows = []
    for _, od_routes in routes.groupby("od_id", sort=False):
        selected = {row.algorithm: {frozenset((u, v)) for u, v, _ in row.edge_route} for row in od_routes.itertuples()}
        chosen = streets.loc[streets.index.isin(set().union(*selected.values()))]
        # Index query per route edge: a union + buffer of long routes can exhaust GEOS memory.
        _, positions = streets.sindex.query(chosen.geometry, predicate="dwithin", distance=buffer_m)
        corridor = streets.iloc[np.unique(positions)]
        for algorithm, pairs in selected.items():
            rows.append(pd.DataFrame({
                "algorithm": algorithm,
                "selected": corridor.index.isin(pairs).astype(float),
                "length_m": corridor["length_m"].to_numpy(),
                "log_value": np.log1p(corridor[value_column].to_numpy()),
            }))

    def weighted_corr(frame):
        cov = np.cov(frame["selected"], frame["log_value"], aweights=frame["length_m"])
        return float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))

    pooled = pd.concat(rows)
    return pd.Series(
        {algorithm: weighted_corr(frame) for algorithm, frame in pooled.groupby("algorithm", sort=False)},
        name="corr_selected_vs_log_popularity",
    )
