import numpy as np
import pandas as pd

from cycle_routing.pareto import pareto_front, pareto_table, plot_pareto


def test_front_keeps_tradeoffs_and_equal_points_but_not_dominated_or_nonfinite():
    points = pd.DataFrame({
        "popularity": [60, 80, 70, 80, 80, np.nan, 90, 50, 79],
        "distance": [1, 2, 3, 2, 2.1, 0, np.inf, .5, 2],
    }, index=list("abcdefghi"))
    front = pareto_front(points.popularity, points.distance)
    assert list(front[front].index) == ["a", "b", "d", "h"]


def test_partial_solution_cannot_win_by_omitting_long_od():
    rows = [
        ("baseline", "short", 1000, 50), ("baseline", "long", 9000, 50),
        ("popular", "short", 1200, 80), ("popular", "long", 10000, 80),
        ("partial", "short", 100, 100),
    ]
    result = pareto_table(pd.DataFrame(rows, columns=["algorithm", "od_id", "length_m", "popularity"]))
    assert result.loc["baseline", "длина, км"] == 5
    assert result.loc["popular", "длина, км"] == 5.6
    assert result.loc["partial", "популярность"] == 50
    assert result.loc["partial", "маршрутов"] == 1 and result.loc["partial", "всего пар"] == 2
    assert pd.isna(result.loc["partial", "длина, км"]) and not result.loc["partial", "Парето"]
    assert result.loc[["baseline", "popular"], "Парето"].all()
    figure = plot_pareto(result)
    assert "км" in figure.axes[0].get_xlabel()


def test_front_uses_unrounded_values_and_global_competitors():
    points = pd.DataFrame({"pop": [70.00001, 70.00002, 70], "km": [1, 1.00001, 1.00001]})
    assert pareto_front(points["pop"], points.km).tolist() == [True, True, False]
    empty = pareto_table(pd.DataFrame(columns=["algorithm", "od_id", "length_m", "popularity"]))
    assert empty.empty and empty["Парето"].dtype == bool
