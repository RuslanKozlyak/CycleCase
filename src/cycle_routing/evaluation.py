"""Stage 5 - score: metrics of ready routes. A black box for students.

``score(routes, city, cache_dir=...)`` reads the city's cached observations itself; the functions below
are its insides. Only this module and the maps look at popularity.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .comfort import comfort_feature_table
from .graph import RIDEABILITY_REASON_LABELS, rideability_issues
from .matching import sample_geometry
from .preparation import (
    CITIES,
    StadtradelnSource,
    load_observations,
    load_popularity,
    prepare_city,
    reference_mask,
)


def evaluate_corridor_popularity(
    routes: gpd.GeoDataFrame,
    observed: gpd.GeoDataFrame,
    *,
    popular_threshold: float,
    sample_step_m: float = 25.0,
    match_distance_m: float = 25.0,
) -> pd.DataFrame:
    """Observed ``trips`` seen from points every ``sample_step_m`` along each route."""

    observed = observed[["trips", "geometry"]]
    rows = []
    for route in tqdm(routes.itertuples(), total=len(routes), desc="STAD matching"):
        points = sample_geometry(route.geometry, sample_step_m)
        samples = gpd.GeoDataFrame({"sample_id": range(len(points))}, geometry=points, crs=routes.crs)
        flows = (
            gpd.sjoin_nearest(samples, observed, max_distance=match_distance_m, distance_col="d")
            .sort_values("d")
            .drop_duplicates("sample_id")["trips"]
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
    if "popularity_log_mean" in result:
        result["popularity_gain"] = result["popularity_log_mean"] - result["baseline_popularity_log_mean"]
    return result


def route_edges_long(routes: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """Explode ``edge_route`` lists into one row per traversed edge weighted by its length."""

    rows = [
        {"group": row.algorithm, "od_id": row.od_id, "u": u, "v": v, "key": key}
        for row in routes.itertuples()
        for u, v, key in row.edge_route
    ]
    result = pd.DataFrame(rows, columns=["group", "od_id", "u", "v", "key"])
    result["weight"] = features["length_m"].reindex(pd.MultiIndex.from_frame(result[["u", "v", "key"]])).to_numpy()
    return result


def route_edge_value_sums(edge_rows: pd.DataFrame, features: pd.DataFrame, column: str) -> pd.Series:
    """Sum an edge value over every traversed edge, including repeated uses across routes."""

    index = pd.MultiIndex.from_frame(edge_rows[["u", "v", "key"]])
    values = features[column].reindex(index).fillna(0.0).to_numpy()
    return pd.Series(values, index=edge_rows.index).groupby(edge_rows["group"], sort=False).sum()


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


def edge_kind_counts(edge_rows: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Number of unsafe route edges per group, summed over all routes."""

    used = edges.set_index(["u", "v", "key"]).reindex(pd.MultiIndex.from_frame(edge_rows[["u", "v", "key"]]))
    issues = rideability_issues(used)
    kinds = issues.rename(columns={reason: f"{label}, рёбер" for reason, label in RIDEABILITY_REASON_LABELS.items()})
    return kinds.groupby(edge_rows["group"].to_numpy()).sum()


def _stadtradeln_scores(routes, city, cache_dir, streets, models) -> pd.DataFrame:
    source = CITIES[city].observations
    observed = load_observations(city, cache_dir=cache_dir)
    metrics = evaluate_corridor_popularity(
        routes,
        observed,
        popular_threshold=float(observed["trips"].quantile(source.popular_quantile)),
        sample_step_m=source.sample_step_m,
        match_distance_m=source.match_distance_m,
    )
    comparison = compare_route_models(metrics)
    gain = comparison.groupby("model")["popularity_gain"]
    by_model = metrics.groupby("algorithm")
    return pd.DataFrame({
        "объезд, %": comparison.groupby("model")["detour_pct"].median(),
        "поворотов": by_model["turns"].median(),
        "поездок × кратчайший": np.exp(gain.median()),
        "популярные улицы, %": 100 * by_model["popular_edge_share"].median(),
        "популярнее кратчайшего": gain.apply(lambda g: f"{(g > 0).sum()}/{len(g)}"),
        "связь с популярностью": route_edge_association(routes, streets, "trips"),
    }).reindex(models).fillna({"объезд, %": 0.0, "поездок × кратчайший": 1.0})


def _track_scores(routes, city, features, edge_rows, models) -> pd.DataFrame:
    source = CITIES[city].observations
    edge_rows = edge_rows.assign(
        trips=features["trips"].reindex(pd.MultiIndex.from_frame(edge_rows[["u", "v", "key"]])).to_numpy()
    )
    used = features.loc[pd.MultiIndex.from_frame(edge_rows[["u", "v", "key"]].drop_duplicates())]
    explained = comfort_feature_table(gpd.GeoDataFrame(used.reset_index(), geometry="geometry", crs=features.crs))
    major = composition_table(edge_rows, explained, "road_class").reindex(
        columns=["крупная дорога", "крупная + велополоса"], fill_value=0.0
    )
    detour = compare_route_models(routes[["od_id", "algorithm", "route_length_m"]])
    by_model = routes.groupby("algorithm")
    return pd.DataFrame({
        "длина, км": by_model["route_length_m"].median() / 1000,
        "объезд, %": detour.groupby("model")["detour_pct"].median(),
        "поворотов": by_model["turns"].median(),
        "по вашим улицам, %": edge_rows.groupby("group").apply(
            lambda g: 100 * g["weight"][g["trips"] >= source.popular_min_trips].sum() / g["weight"].sum()
        ),
        "крупные улицы, %": major.sum(axis=1),
    }).reindex(models).fillna({"объезд, %": 0.0})


def score(routes: gpd.GeoDataFrame, city: str, *, cache_dir: Path) -> pd.DataFrame:
    """Metrics per model (rows in the order of ``routes``), from the city's cached observations.

    ``routes`` comes from ``build_routes`` and must contain the ``shortest`` baseline. Konstanz: detour,
    turns, trips along the route against the shortest, share of popular streets, association with
    popularity. Saint Petersburg: length, detour, turns, share of your streets, share of major streets.
    Both: the sum of observed trips over all traversed edges and how many edges of a few special kinds
    the routes use. Repeated use of an edge by different OD routes is counted repeatedly.
    """

    if "shortest" not in set(routes["algorithm"]):
        raise ValueError("routes must contain the 'shortest' baseline, as build_routes returns")
    edges, _ = prepare_city(city, cache_dir=cache_dir)
    popularity = load_popularity(city, cache_dir=cache_dir)
    if not popularity[["u", "v", "key"]].equals(edges[["u", "v", "key"]]):
        raise ValueError(f"Popularity cache of {city!r} does not match its edge table; rebuild with refresh=True")
    base = edges.drop(columns=[column for column in popularity.columns if column in edges and column not in ("u", "v", "key")])
    features = base.assign(trips=popularity["trips"].to_numpy()).set_index(["u", "v", "key"])
    streets = features[reference_mask(base, city).to_numpy()]
    models = list(dict.fromkeys(routes["algorithm"]))
    edge_rows = route_edges_long(routes, features)

    if isinstance(CITIES[city].observations, StadtradelnSource):
        table = _stadtradeln_scores(routes, city, cache_dir, streets, models)
    else:
        table = _track_scores(routes, city, features, edge_rows, models)
    trip_sums = route_edge_value_sums(edge_rows, features, "trips").rename("сумма поездок по рёбрам")
    return (
        table
        .join(trip_sums.reindex(models, fill_value=0.0))
        .join(edge_kind_counts(edge_rows, base).reindex(models, fill_value=0))
    )
