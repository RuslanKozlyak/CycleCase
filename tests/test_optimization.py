from copy import deepcopy

import networkx as nx

from cycle_routing import DEFAULT_COMFORT_CONFIG
from cycle_routing.optimization import (
    ComfortOptimizationConfig,
    ComfortParameter,
    comfort_config_with_parameters,
    optimize_comfort_weights,
    score_comfort_config,
    spatial_od_split,
)
from cycle_routing.routing import build_routes


def _choice_graph():
    graph = nx.MultiDiGraph(crs="EPSG:25832")
    for node, coordinates in {
        1: (0, 0),
        2: (100, 0),
        3: (100, 50),
        4: (200, 0),
    }.items():
        graph.add_node(node, x=coordinates[0], y=coordinates[1])
    for u, v in ((1, 2), (2, 4)):
        graph.add_edge(
            u,
            v,
            length=100,
            highway="residential",
            surface="asphalt",
            stad_trips=0,
            stad_match_method="none",
        )
    for u, v in ((1, 3), (3, 4)):
        graph.add_edge(
            u,
            v,
            length=120,
            highway="path",
            surface="asphalt",
            stad_trips=100,
            stad_match_method="geometry",
        )
    return graph


def test_candidate_config_is_a_deep_copy():
    base = deepcopy(DEFAULT_COMFORT_CONFIG)
    parameter = ComfortParameter("highway", "path", 0.3, 1.5)
    candidate = comfort_config_with_parameters(base, [0.4], [parameter])
    assert candidate["highway"]["path"] == 0.4
    assert base["highway"]["path"] == DEFAULT_COMFORT_CONFIG["highway"]["path"]


def test_popularity_score_rewards_config_that_selects_observed_path():
    graph = _choice_graph()
    pairs = [("OD-1", 1, 4)]
    default = deepcopy(DEFAULT_COMFORT_CONFIG)
    unpopular, _ = score_comfort_config(
        graph,
        pairs,
        default,
        optimization_config=ComfortOptimizationConfig(free_detour_ratio=0.25),
    )
    preferred = deepcopy(default)
    preferred["highway"]["path"] = 0.4
    popular, _ = score_comfort_config(
        graph,
        pairs,
        preferred,
        optimization_config=ComfortOptimizationConfig(free_detour_ratio=0.25),
    )
    assert popular.iloc[0]["popularity_log_mean"] > unpopular.iloc[0]["popularity_log_mean"]
    assert popular.iloc[0]["score"] > unpopular.iloc[0]["score"]


def test_build_routes_supports_two_named_comfort_models():
    graph = _choice_graph()
    default = deepcopy(DEFAULT_COMFORT_CONFIG)
    optimized = deepcopy(default)
    optimized["highway"]["path"] = 0.4
    routes = build_routes(
        graph,
        [("OD-1", 1, 4)],
        comfort_configs={
            "comfort_default": default,
            "comfort_optimized": optimized,
        },
    )
    assert set(routes["algorithm"]) == {
        "shortest",
        "comfort_default",
        "comfort_optimized",
    }


def test_spatial_split_has_no_overlap():
    graph = nx.MultiDiGraph()
    pairs = []
    for index in range(10):
        graph.add_node(index, x=index, y=0)
        graph.add_node(index + 100, x=index + 0.5, y=0)
        pairs.append((f"OD-{index}", index, index + 100))
    splits = spatial_od_split(graph, pairs)
    ids = [[pair[0] for pair in values] for values in splits.values()]
    assert [len(values) for values in ids] == [6, 2, 2]
    assert len(set().union(*(set(values) for values in ids))) == 10


def test_optimizer_returns_validation_selected_config_and_all_split_metrics():
    graph = _choice_graph()
    pairs = [(f"OD-{index}", 1, 4) for index in range(4)]
    result = optimize_comfort_weights(
        graph,
        {
            "train": pairs[:2],
            "validation": pairs[2:3],
            "test": pairs[3:],
        },
        deepcopy(DEFAULT_COMFORT_CONFIG),
        parameters=[ComfortParameter("highway", "path", 0.3, 1.5)],
        optimization_config=ComfortOptimizationConfig(
            maxiter=1,
            popsize=5,
            polish=False,
            validation_candidates=3,
            show_progress=False,
            free_detour_ratio=0.25,
        ),
    )
    assert result.nfev > 0
    assert len(result.parameters) == 1
    assert set(result.comparison["split"]) == {"train", "validation", "test"}
    assert set(result.comparison["model"]) == {
        "comfort_default",
        "comfort_optimized",
    }
    assert result.history["validation_selection_score"].notna().sum() == 3
