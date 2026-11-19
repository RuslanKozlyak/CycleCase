from __future__ import annotations

import geopandas as gpd

import numpy as np

import pandas as pd

from sklearn.model_selection import GroupShuffleSplit

def spatial_partition(
    edges: gpd.GeoDataFrame,
    eligible: pd.Series,
    *,
    cell_m: float = 1_000.0,
    random_state: int = 42,
) -> pd.Series:

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
