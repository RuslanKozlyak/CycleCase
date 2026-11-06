import numpy as np

def weighted_corr(x, y, weights) -> float:

    x, y, weights = (np.asarray(values, dtype=float) for values in (x, y, weights))
    keep = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    if keep.sum() < 2 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return float("nan")
    cov = np.cov(x[keep], y[keep], aweights=weights[keep])
    spread = cov[0, 0] * cov[1, 1]
    return float(cov[0, 1] / np.sqrt(spread)) if spread > 0 else float("nan")
