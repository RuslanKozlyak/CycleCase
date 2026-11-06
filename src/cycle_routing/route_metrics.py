from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import pandas as pd

@dataclass(frozen=True)
class PopularityScale:
    counts: np.ndarray
    mass: np.ndarray

    @classmethod
    def fit(cls, trips: pd.Series, lengths: pd.Series):
        if not trips.index.equals(lengths.index):
            raise ValueError("Popularity labels and lengths must be aligned")
        known = trips.notna()
        if not known.any() or not np.isfinite(trips[known]).all() or trips[known].lt(0).any():
            raise ValueError("Need known nonnegative popularity labels")
        if not np.isfinite(lengths[known]).all() or lengths[known].le(0).any():
            raise ValueError("Popularity lengths must be positive and finite")
        mass = lengths[known].groupby(trips[known]).sum().sort_index()
        return cls(mass.index.to_numpy(float), (mass / mass.sum()).to_numpy(float))

    def transform(self, trips: pd.Series) -> pd.Series:
        x = trips.to_numpy(float)
        cumulative = np.r_[0., np.cumsum(self.mass)]
        left = np.searchsorted(self.counts, x, side="left")
        right = np.searchsorted(self.counts, x, side="right")
        result = 50 * (cumulative[left] + cumulative[right])
        result[~np.isfinite(x)] = np.nan
        return pd.Series(result, index=trips.index, name="popularity")
