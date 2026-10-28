"""Learn street popularity from OSM features, then use predictions as routing weights.

Observed trips are labels, never input features. Split the surveyed network before fitting:
both directions and parallel edges of a street belong to the same spatial partition.
"""

from __future__ import annotations

from dataclasses import dataclass

import geopandas as gpd
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from threadpoolctl import threadpool_limits


CATEGORICAL_FEATURES = ("highway", "surface", "smoothness", "cycleway", "bicycle", "lit", "service", "access")
NUMERIC_FEATURES = ("maxspeed", "lanes", "width", "oneway")
FEATURE_COLUMNS = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)


def popularity_features(edges: pd.DataFrame) -> pd.DataFrame:
    """Only road attributes: no trips, IDs, geometry, coordinates or segment length."""

    features = edges.loc[:, FEATURE_COLUMNS].copy()
    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].fillna("__missing__").astype(str)
    for column in NUMERIC_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="coerce").astype(float)
    return features.replace([np.inf, -np.inf], np.nan)


def spatial_partition(
    edges: gpd.GeoDataFrame,
    eligible: pd.Series,
    *,
    cell_m: float = 1_000.0,
    random_state: int = 42,
) -> pd.Series:
    """Assign surveyed streets to train/validation/test (60/20/20% of spatial blocks).

    ``eligible`` defines geographic scope, not label availability: callers must separately
    exclude missing targets. Fractions are by blocks, not rows. Canonical street midpoints
    keep reverse/parallel edges in one partition.
    Coordinates are used only for this split, never as model features.
    """

    if edges.crs is None or edges.crs.is_geographic:
        raise ValueError("Spatial partition needs a projected metric CRS")
    if not np.isfinite(cell_m) or cell_m <= 0:
        raise ValueError("cell_m must be positive and finite")
    if not eligible.index.equals(edges.index) or eligible.isna().any():
        raise ValueError("eligible must be a complete mask aligned with edges")
    midpoints = edges.geometry.interpolate(0.5, normalized=True)
    streets = pd.DataFrame({
        "a": np.minimum(edges["u"], edges["v"]),
        "b": np.maximum(edges["u"], edges["v"]),
        "x": midpoints.x,
        "y": midpoints.y,
    }, index=edges.index)
    centers = streets.groupby(["a", "b"])[["x", "y"]].transform("mean")
    cells = np.floor(centers / cell_m).astype("int64").astype(str)
    groups = (cells["x"] + ":" + cells["y"]).loc[eligible]
    if groups.nunique() < 5:
        raise ValueError("Need at least five surveyed blocks; reduce cell_m")
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=random_state)
    rest, test = next(splitter.split(groups, groups=groups))
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=random_state + 1)
    train, validation = next(splitter.split(groups.iloc[rest], groups=groups.iloc[rest]))
    partition = pd.Series("outside", index=edges.index, name="partition")
    partition.loc[groups.index[rest[train]]] = "train"
    partition.loc[groups.index[rest[validation]]] = "validation"
    partition.loc[groups.index[test]] = "test"
    return partition


def popularity_cost(
    edges: pd.DataFrame,
    predicted_log_trips: pd.Series,
    *,
    strength: float,
    reference_log: float,
) -> pd.Series:
    """Length times exp(-strength * predicted attractiveness); strength=0 is shortest.

    Predictions are log(1 + trips). Centering uses only the training mean. Clipping
    limits detours and keeps all costs strictly positive, even for extreme settings.
    """

    if not np.isfinite(strength) or strength < 0 or not np.isfinite(reference_log):
        raise ValueError("strength must be nonnegative and finite; reference_log must be finite")
    if not predicted_log_trips.index.equals(edges.index) or not np.isfinite(predicted_log_trips).all():
        raise ValueError("Predictions must be finite and aligned with edges")
    attraction = np.clip(predicted_log_trips.to_numpy() - reference_log, -3.0, 3.0)
    exponent = np.clip(-strength * attraction, -6.0, 6.0)
    return (edges["length_m"] * np.exp(exponent)).rename("cost")


@dataclass
class PopularityModel:
    estimator: Pipeline
    reference_log: float

    def predict_log_trips(self, edges: pd.DataFrame) -> pd.Series:
        with threadpool_limits(limits=2):
            values = self.estimator.predict(popularity_features(edges))
        return pd.Series(np.maximum(values, 0.0), index=edges.index, name="predicted_log_trips")


def fit_popularity_model(
    train_edges: pd.DataFrame,
    train_trips: pd.Series,
    *,
    max_iter: int = 120,
    max_leaf_nodes: int = 15,
    min_samples_leaf: int = 20,
    random_state: int = 42,
) -> PopularityModel:
    """Fit gradient boosting on log(1 + trips), weighted by street length.

    Pass training rows only. Validation/test targets are not accepted by this API.
    Unknown categories and absent numeric tags are supported at prediction time.
    """

    if not train_trips.index.equals(train_edges.index):
        raise ValueError("Training labels must be aligned with training edges")
    if len(train_edges) < 2 or not np.isfinite(train_trips).all() or (train_trips < 0).any():
        raise ValueError("Need at least two training rows with finite nonnegative trips")
    weights = train_edges["length_m"].to_numpy(dtype=float)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Training lengths must be finite and positive")
    transform = ColumnTransformer([
        ("categories", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan,
                                      max_categories=255), list(CATEGORICAL_FEATURES)),
        # A missing numeric tag is -1, outside physical widths/lanes/speeds. This also
        # supports a training partition where a numeric feature is entirely absent.
        ("numbers", SimpleImputer(strategy="constant", fill_value=-1, keep_empty_features=True), list(NUMERIC_FEATURES)),
    ])
    regressor = HistGradientBoostingRegressor(
        max_iter=max_iter, max_leaf_nodes=max_leaf_nodes, min_samples_leaf=min_samples_leaf,
        learning_rate=0.08, l2_regularization=1.0, early_stopping=False,
        categorical_features=[True] * len(CATEGORICAL_FEATURES) + [False] * len(NUMERIC_FEATURES),
        random_state=random_state,
    )
    estimator = Pipeline([("features", transform), ("regressor", regressor)])
    target = np.log1p(train_trips.to_numpy(dtype=float))
    with threadpool_limits(limits=2):
        estimator.fit(popularity_features(train_edges), target, regressor__sample_weight=weights / weights.mean())
    return PopularityModel(estimator, reference_log=float(np.average(target, weights=weights)))
