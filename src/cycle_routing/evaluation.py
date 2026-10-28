"""Stage 5 - score: metrics of ready routes. A black box for students.

``score(routes, city, cache_dir=...)`` reads the city's cached observations itself; the functions below
are its insides. ML experiments may additionally use training labels, kept outside the routing graph.
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from .comfort import comfort_feature_table
from .graph import RIDEABILITY_REASON_LABELS, edge_costs, is_rideable, rideability_issues
from .matching import sample_geometry
from .preparation import (
    CITIES,
    StadtradelnSource,
    load_observations,
    load_popularity,
    observed_area,
    prepare_city,
    reference_mask,
)


def weighted_corr(x, y, weights) -> float:
    """Pearson correlation of two edge values, weighted by edge length; NaN when either is constant."""

    x, y, weights = (np.asarray(values, dtype=float) for values in (x, y, weights))
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if keep.sum() < 2 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return float("nan")
    cov = np.cov(x[keep], y[keep], aweights=weights[keep])
    spread = cov[0, 0] * cov[1, 1]
    return float(cov[0, 1] / np.sqrt(spread)) if spread > 0 else float("nan")


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
    """Sum known values over traversed edges; NaN when a model has no measured values."""

    index = pd.MultiIndex.from_frame(edge_rows[["u", "v", "key"]])
    values = features[column].reindex(index).to_numpy()
    return pd.Series(values, index=edge_rows.index).groupby(edge_rows["group"], sort=False).sum(min_count=1)


def route_popularity_metrics(
    routes: pd.DataFrame,
    edges: pd.DataFrame,
    trips: pd.Series,
    *,
    mask: pd.Series,
    popular_threshold: float | None = None,
) -> pd.DataFrame:
    """Per-route length-weighted mean log(1 + trips) on a chosen evaluation partition.

    Missing labels are always excluded, even if ``mask`` includes them. Unsurveyed/training
    streets are not treated as zero-popularity test streets. Report
    coverage explicitly: routes without evaluated length get NaN, never a fabricated score.
    Segmentation and repeated edges cannot inflate a raw trip-count sum here.
    """

    if not trips.index.equals(edges.index) or not mask.index.equals(edges.index) or mask.isna().any():
        raise ValueError("Trips and evaluation mask must be aligned with edges")
    mask = mask & trips.notna()
    selected = trips.loc[mask]
    if not np.isfinite(selected).all() or (selected < 0).any():
        raise ValueError("Evaluation trips must be finite and nonnegative")
    features = edges[["u", "v", "key", "length_m"]].assign(
        log_trips=np.log1p(trips.where(mask)), evaluate=mask,
    ).set_index(["u", "v", "key"])
    if not features.index.is_unique:
        raise ValueError("Edge identifiers must be unique")
    rows = []
    for route in routes.itertuples():
        path = pd.MultiIndex.from_tuples([tuple(edge) for edge in route.edge_route], names=["u", "v", "key"])
        used = features.reindex(path)
        if used["length_m"].isna().any():
            raise ValueError("A route contains edges missing from the evaluation table")
        evaluated = used.loc[used["evaluate"].astype(bool)]
        length = float(used["length_m"].sum())
        covered = float(evaluated["length_m"].sum())
        value = float(np.average(evaluated["log_trips"], weights=evaluated["length_m"])) if covered else np.nan
        rows.append({
            "od_id": route.od_id, "algorithm": route.algorithm,
            "route_length_m": length, "popularity_log_mean": value,
            "coverage_pct": 100 * covered / length if length else 0.0,
            "popular_edge_share": (float(evaluated.loc[evaluated.log_trips.ge(np.log1p(popular_threshold)), "length_m"].sum() / covered)
                                   if covered and popular_threshold is not None else np.nan),
        })
    return pd.DataFrame(rows)


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

    streets = features.loc[features[value_column].notna(), [value_column, "length_m", "geometry"]].copy()
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

    pooled = pd.concat(rows)
    return pd.Series(
        {
            algorithm: weighted_corr(frame["selected"], frame["log_value"], frame["length_m"])
            for algorithm, frame in pooled.groupby("algorithm", sort=False)
        },
        name="corr_selected_vs_log_popularity",
    )


def edge_trips(edges: pd.DataFrame, city: str, *, cache_dir: Path) -> pd.Series:
    """Observed trips aligned with ``edges``; works on a filtered subset of the city's table."""

    popularity = load_popularity(city, cache_dir=cache_dir)
    values = pd.Series(
        popularity["trips"].to_numpy(), index=pd.MultiIndex.from_frame(popularity[["u", "v", "key"]])
    )
    aligned = values.reindex(pd.MultiIndex.from_frame(edges[["u", "v", "key"]]))
    return pd.Series(aligned.to_numpy(), index=edges.index, name="trips")


def comfort_correlation(
    edges: gpd.GeoDataFrame,
    cost,
    city: str,
    *,
    cache_dir: Path,
    mask: pd.Series | None = None,
) -> float:
    """How well a comfort function explains observed popularity, edge by edge.

    The comfort of an edge is ``-log(cost / length_m)``: its price per metre, with length divided out
    because length is not a preference, and on a log scale because comfort weights multiply. The
    result is that comfort correlated with ``log1p(trips)``, weighted by edge length, over ``mask`` -
    by default the rideable edges of :func:`observed_area`. Unlike :func:`score` this looks at the
    whole network instead of a handful of routes, so two profiles that happen to pick the same route
    still get different numbers. ``cost`` is anything :func:`edge_costs` accepts.

    A cost proportional to length is the same everywhere and explains nothing: the result is NaN.
    """

    if mask is None:
        mask = observed_area(edges, city, cache_dir=cache_dir) & is_rideable(edges)
    length = edges["length_m"].to_numpy()
    comfort = -np.log(edge_costs(edges, cost).to_numpy() / length)
    trips = edge_trips(edges, city, cache_dir=cache_dir).to_numpy()
    keep = np.asarray(mask, dtype=bool) & np.isfinite(comfort) & np.isfinite(trips)
    if keep.sum() < 2:
        return float("nan")
    return weighted_corr(comfort[keep], np.log1p(trips[keep]), length[keep])


def comfort_scores(
    edges: gpd.GeoDataFrame,
    presets: dict[str, dict],
    city: str,
    *,
    cache_dir: Path,
) -> pd.Series:
    """:func:`comfort_correlation` for every preset, labelled as in :func:`score`.

    The edge mask is built once and reused, so adding a profile costs one pass over the table.
    """

    mask = observed_area(edges, city, cache_dir=cache_dir) & is_rideable(edges)
    return pd.Series(
        {
            preset.get("label", name): comfort_correlation(
                edges, preset["cost"], city, cache_dir=cache_dir, mask=mask
            )
            for name, preset in presets.items()
        },
        name="связь комфорта с популярностью",
    )


def edge_kind_counts(edge_rows: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Number of unsafe route edges per group, summed over all routes."""

    used = edges.set_index(["u", "v", "key"]).reindex(pd.MultiIndex.from_frame(edge_rows[["u", "v", "key"]]))
    issues = rideability_issues(used)
    kinds = issues.rename(columns={reason: f"{label}, рёбер" for reason, label in RIDEABILITY_REASON_LABELS.items()})
    return kinds.groupby(edge_rows["group"].to_numpy()).sum()


def _stadtradeln_scores(routes, city, features, streets, models) -> pd.DataFrame:
    source = CITIES[city].observations
    flat = features.reset_index()
    metrics = route_popularity_metrics(routes, flat, flat["trips"], mask=flat["trips"].notna(),
                                       popular_threshold=float(flat["trips"].quantile(source.popular_quantile)))
    metrics = metrics.merge(routes[["od_id", "algorithm", "turns"]], on=["od_id", "algorithm"], validate="one_to_one")
    comparison = compare_route_models(metrics)
    gain = comparison.groupby("model")["popularity_gain"]
    by_model = metrics.groupby("algorithm")
    ratios = np.exp(gain.median()).reindex(models)
    if metrics.loc[metrics.algorithm.eq("shortest"), "popularity_log_mean"].notna().any():
        ratios.loc["shortest"] = 1.0
    return pd.DataFrame({
        "объезд, %": comparison.groupby("model")["detour_pct"].median(),
        "поворотов": by_model["turns"].median(),
        "поездок × кратчайший": ratios,
        "популярные улицы, %": 100 * by_model["popular_edge_share"].median(),
        "популярнее кратчайшего": gain.apply(lambda g: f"{(g > 0).sum()}/{g.notna().sum()}"),
        "связь с популярностью": route_edge_association(routes, streets, "trips"),
        "покрытие измерениями, %": by_model["coverage_pct"].median(),
    }).reindex(models).fillna({"объезд, %": 0.0})


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
        table = _stadtradeln_scores(routes, city, features, streets, models)
    else:
        table = _track_scores(routes, city, features, edge_rows, models)
    trip_sums = route_edge_value_sums(edge_rows, features, "trips").rename("сумма поездок по рёбрам")
    return (
        table
        .join(trip_sums.reindex(models))
        .join(edge_kind_counts(edge_rows, base).reindex(models, fill_value=0))
    )
