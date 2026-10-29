"""Compare route popularity (maximize) with mean physical distance (minimize)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_front(popularity: pd.Series, distance: pd.Series) -> pd.Series:
    """Keep nondominated finite points; equal points belong to the front together."""
    values = pd.concat([popularity, distance.reindex(popularity.index)], axis=1).to_numpy(dtype=float, na_value=np.nan)
    finite = np.isfinite(values).all(axis=1)
    result = np.zeros(len(values), dtype=bool)
    points = values[finite]
    # Row i dominates column j: at least as popular, at most as long, strictly better in one.
    dominates = ((points[:, None, 0] >= points[None, :, 0])
                 & (points[:, None, 1] <= points[None, :, 1])
                 & ((points[:, None, 0] > points[None, :, 0])
                    | (points[:, None, 1] < points[None, :, 1])))
    result[finite] = ~dominates.any(axis=0)
    return pd.Series(result, index=popularity.index, name="Парето")


def pareto_table(per_od: pd.DataFrame, labels: dict | None = None) -> pd.DataFrame:
    """Average over a fixed OD bank. Incomplete solutions have no comparable distance.

    Uses the actual popularity, including for long routes. Missing routes contribute zero
    popularity and exclude the solution from the front; they never make its distance shorter.
    """
    rows = []
    pairs = pd.Index(per_od.od_id.unique())
    for name, group in per_od.groupby("algorithm", sort=False):
        group = group.set_index("od_id").reindex(pairs)
        valid = (np.isfinite(group.length_m) & group.length_m.gt(0) & np.isfinite(group.popularity))
        complete = len(pairs) > 0 and bool(valid.all())
        rows.append({"algorithm": name, "популярность": group.popularity.where(valid, 0).mean(),
                     "длина, км": group.length_m.mean() / 1000 if complete else np.nan,
                     "маршрутов": int(valid.sum()), "всего пар": len(pairs)})
    columns = ["algorithm", "популярность", "длина, км", "маршрутов", "всего пар"]
    result = pd.DataFrame(rows, columns=columns).set_index("algorithm")
    result["Парето"] = pareto_front(result["популярность"], result["длина, км"])
    if labels:
        result = result.rename(index=labels)
    return result


def plot_pareto(table: pd.DataFrame, *, title="Популярность и расстояние"):
    """Plot unrounded objectives; annotate distinct frontier points, grouping ties."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    available = table[table["длина, км"].notna()]
    rest = available[~available["Парето"]]
    front = available[available["Парето"]].sort_values("длина, км")
    ax.scatter(rest["длина, км"], rest["популярность"], c="#94a3b8", s=38, alpha=.7,
               label="Остальные полные решения", zorder=2)
    ax.plot(front["длина, км"], front["популярность"], ".--", color="#0d9488", markersize=12,
            linewidth=1.3, label="Парето-фронт", zorder=3)
    incomplete = len(table) - len(available)
    subtitle = f"\nНеполных решений вне фронта: {incomplete}" if incomplete else ""
    ax.set(title=title + subtitle, xlabel="Средняя длина маршрута, км — меньше лучше",
           ylabel="Средняя популярность, 0–100 — больше лучше")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=.15)
    ax.margins(x=.15, y=.15)
    ax.legend(loc="lower right", fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    occupied = []
    offsets = [(6, 9), (6, -17), (-6, 9), (-6, -17), (8, 25), (8, -32),
               (-8, 25), (-8, -32), (30, 15), (30, -25), (-30, 15), (-30, -25), (8, 42), (8, -49)]
    centers = ax.transData.transform(front[["длина, км", "популярность"]].drop_duplicates().to_numpy())
    for (distance, popularity), group in front.groupby(["длина, км", "популярность"], sort=False):
        names = ["кратчайший" if str(name) == "shortest" else str(name) for name in group.index]
        label = ", ".join(names) if len(names) <= 3 else f"{names[0]} (+{len(names) - 1})"
        annotation = ax.annotate(label, (distance, popularity), xytext=offsets[0], textcoords="offset points",
                                 fontsize=8, annotation_clip=False,
                                 bbox={"facecolor": "white", "edgecolor": "none", "alpha": .8, "pad": .6})
        choices = []
        for dx, dy in offsets:
            annotation.set_position((dx, dy))
            annotation.set_ha("left" if dx > 0 else "right")
            box = annotation.get_window_extent(renderer).expanded(1.08, 1.15)
            collision = sum(box.overlaps(other) for other in occupied)
            markers = sum(box.contains(x, y) for x, y in centers)
            outside = not (ax.bbox.contains(box.x0, box.y0) and ax.bbox.contains(box.x1, box.y1))
            choices.append((10000 * (collision + outside) + 100 * markers + abs(dx) + abs(dy), dx, dy, box))
        _, dx, dy, box = min(choices, key=lambda item: item[0])
        annotation.set_position((dx, dy))
        annotation.set_ha("left" if dx > 0 else "right")
        occupied.append(box)
        if abs(dx) + abs(dy) > 30:
            ax.annotate("", (distance, popularity), xytext=(dx, dy), textcoords="offset points",
                        arrowprops={"arrowstyle": "-", "color": "#64748b", "lw": .6})
    return fig
