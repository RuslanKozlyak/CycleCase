import numpy as np
import pandas as pd
import pytest

from cycle_routing import build_graph, find_route, is_rideable, make_edges, make_nodes, prepare_city
from cycle_routing.evaluation import edge_trips, route_popularity_metrics
from cycle_routing.learning import fit_popularity_model, popularity_cost, spatial_partition


def test_spatial_partitions_keep_street_directions_together_and_exclude_unsurveyed():
    coordinates = {node: (500 * (node // 2), 10 * (node % 2)) for node in range(24)}
    nodes = make_nodes(coordinates, crs="EPSG:25832")
    edges = make_edges(
        [{"u": u, "v": v} for u in range(0, 24, 2) for u, v in ((u, u + 1), (u + 1, u))],
        crs="EPSG:25832", nodes=nodes,
    )
    eligible = pd.Series(True, index=edges.index)
    eligible.iloc[-2:] = False
    partition = spatial_partition(edges, eligible, cell_m=200, random_state=42)

    assert set(partition[eligible]) == {"train", "validation", "test"}
    assert (partition[~eligible] == "outside").all()
    for u, v in zip(range(0, 24, 2), range(1, 24, 2)):
        assert partition[(edges.u.isin([u, v])) & (edges.v.isin([u, v]))].nunique() == 1
    pd.testing.assert_series_equal(partition, spatial_partition(edges, eligible, cell_m=200))
    shuffled = edges.sample(frac=1, random_state=7)
    pd.testing.assert_series_equal(
        partition, spatial_partition(shuffled, eligible.loc[shuffled.index], cell_m=200).loc[edges.index],
    )


def test_learning_uses_road_features_and_changes_the_route_towards_popularity(toy_city):
    edges, nodes = prepare_city("toytown", cache_dir=toy_city["cache_dir"])
    trips = edge_trips(edges, "toytown", cache_dir=toy_city["cache_dir"])
    allowed = is_rideable(edges)
    train = allowed & trips.notna()
    model = fit_popularity_model(edges[train], trips[train], min_samples_leaf=1, max_iter=100)
    prediction = model.predict_log_trips(edges)
    leaked = edges.assign(trips=99999.0, popularity=-999.0, u=edges.u + 100, x=1.0)
    pd.testing.assert_series_equal(prediction, model.predict_log_trips(leaked))
    unknown = edges.copy()
    unknown["surface"] = "unseen_surface"
    assert np.isfinite(model.predict_log_trips(unknown)).all()

    network = edges[allowed]
    zero = popularity_cost(network, prediction[allowed], strength=0, reference_log=model.reference_log)
    pd.testing.assert_series_equal(zero, network.length_m.astype(float).rename("cost"))
    strong = popularity_cost(network, prediction[allowed], strength=1, reference_log=model.reference_log)
    shortest = find_route(build_graph(network, nodes, zero), 1, 4, turns=None)
    popular = find_route(build_graph(network, nodes, strong), 1, 4, turns=None)
    assert shortest["edge_route"] != popular["edge_route"]
    routes = pd.DataFrame([
        {"od_id": "A", "algorithm": "shortest", **shortest},
        {"od_id": "A", "algorithm": "ml", **popular},
    ])
    scored = route_popularity_metrics(routes, edges, trips, mask=allowed).set_index("algorithm")
    assert scored.loc["ml", "popularity_log_mean"] > scored.loc["shortest", "popularity_log_mean"]
    assert np.isfinite(popularity_cost(network, prediction[allowed], strength=1e6,
                                      reference_log=model.reference_log)).all()
    assert (strong > 0).all()


def test_route_popularity_uses_only_evaluation_labels_and_is_length_weighted():
    edges = pd.DataFrame({"u": [1, 2, 3], "v": [2, 3, 4], "key": [0, 0, 0], "length_m": [10, 30, 60]})
    trips = pd.Series([1.0, 15.0, 99999.0])
    routes = pd.DataFrame([
        {"od_id": "A", "algorithm": "model", "edge_route": [(1, 2, 0), (2, 3, 0), (3, 4, 0)]},
        {"od_id": "B", "algorithm": "model", "edge_route": [(3, 4, 0)]},
    ])
    mask = pd.Series([True, True, False])
    result = route_popularity_metrics(routes, edges, trips, mask=mask).set_index("od_id")
    assert result.loc["A", "popularity_log_mean"] == pytest.approx((10 * np.log(2) + 30 * np.log(16)) / 40)
    assert result.loc["A", "coverage_pct"] == 40.0
    assert pd.isna(result.loc["B", "popularity_log_mean"])
    assert result.loc["B", "coverage_pct"] == 0.0
    changed = trips.copy()
    changed.iloc[-1] = 0.0
    pd.testing.assert_frame_equal(
        result, route_popularity_metrics(routes, edges, changed, mask=mask).set_index("od_id"),
    )


def test_learning_rejects_invalid_or_misaligned_training_labels(toy_city):
    edges, _ = prepare_city("toytown", cache_dir=toy_city["cache_dir"])
    with pytest.raises(ValueError, match="aligned"):
        fit_popularity_model(edges, pd.Series(1.0, index=edges.index[::-1]))
    with pytest.raises(ValueError, match="nonnegative"):
        fit_popularity_model(edges, pd.Series(-1.0, index=edges.index))
    with pytest.raises(ValueError, match="finite"):
        fit_popularity_model(edges, pd.Series(np.nan, index=edges.index))


def test_metrics_exclude_unknowns_but_keep_measured_zero_in_coverage():
    edges = pd.DataFrame({"u": [1, 2, 3], "v": [2, 3, 4], "key": [0, 0, 0], "length_m": [10, 30, 60]})
    routes = pd.DataFrame([
        {"od_id": "A", "algorithm": "model", "edge_route": [(1, 2, 0), (2, 3, 0), (3, 4, 0)]},
        {"od_id": "B", "algorithm": "model", "edge_route": [(2, 3, 0)]},
        {"od_id": "C", "algorithm": "model", "edge_route": [(1, 2, 0)]},
    ])
    scored = route_popularity_metrics(routes, edges, pd.Series([0., np.nan, 9.]),
                                      mask=pd.Series(True, index=edges.index)).set_index("od_id")
    assert scored.loc["A", "coverage_pct"] == 70
    assert scored.loc["A", "popularity_log_mean"] == pytest.approx(60 * np.log(10) / 70)
    assert pd.isna(scored.loc["B", "popularity_log_mean"]) and scored.loc["B", "coverage_pct"] == 0
    assert scored.loc["C", "popularity_log_mean"] == 0 and scored.loc["C", "coverage_pct"] == 100
