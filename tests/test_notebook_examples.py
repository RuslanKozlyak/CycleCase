"""Exercise the editable notebook functions with small graphs and actual routing adapters."""

import ast
import json
from pathlib import Path

import networkx as nx
import pandas as pd
import pytest

from cycle_routing import build_graph, find_route, make_edges, make_nodes
from cycle_routing.config import TurnConfig
from cycle_routing.routing import edge_bearings, turn_graph, turn_penalty
from cycle_routing.tags import has_any
from cycle_routing.workshop import comfort_cost, features, rideable, russian, train_custom_model


@pytest.fixture
def examples():
    path = Path(__file__).resolve().parents[1] / "notebooks/hackathon_konstanz.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    names = {"features", "rideable", "my_turn_penalty", "my_search", "greedy_search", "train_model"}
    namespace = dict(pd=pd, nx=nx, has_any=has_any, edge_bearings=edge_bearings)
    from sklearn.tree import DecisionTreeRegressor
    namespace["DecisionTreeRegressor"] = DecisionTreeRegressor
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            tree = ast.parse("".join(cell["source"]))
            tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
            exec(compile(tree, str(path), "exec"), namespace)
    return namespace


def example_roads():
    nodes = make_nodes({1: (0, 0), 2: (0, 100), 3: (100, 100)}, crs="EPSG:32636")
    roads = make_edges([
        {"u": 1, "v": 2, "length_m": 100, "highway": "residential;primary_link", "cycleway": "no;lane",
         "surface": "asphalt", "maxspeed": 50},
        {"u": 2, "v": 3, "length_m": 100, "highway": "path", "surface": "ground", "lit": "no"},
        {"u": 1, "v": 3, "length_m": 250, "highway": "cycleway"},
    ], nodes=nodes, crs=nodes.crs)
    return roads, nodes


def test_notebook_features_match_defaults_and_edits_reach_cost_and_model(examples):
    from types import SimpleNamespace
    from sklearn.dummy import DummyRegressor

    roads, _ = example_roads()
    notebook_features = examples["features"]
    pd.testing.assert_frame_equal(notebook_features(russian(roads)), features(roads))
    assert list(notebook_features(russian(roads))) == ["длина_м", "велодорожка", "крупная_дорога", "асфальт"]

    def edited(roads):
        return notebook_features(roads).assign(мой_признак=2.0)

    cost = comfort_cost({"мой_признак": 0.5}, features=edited)
    pd.testing.assert_series_equal(cost(roads), (roads.length_m * 2).rename("cost"))
    city = SimpleNamespace(osm=roads, trips=pd.Series([0.0, 9.0, float("nan")], index=roads.index))
    seen = []

    def train(x, y):
        seen.append(x.copy())
        return DummyRegressor().fit(x, y)

    trained = train_custom_model(city, train, features=edited)
    pd.testing.assert_series_equal(seen[-1]["длина_м"], roads.loc[city.trips.notna(), "length_m"], check_names=False)
    assert seen[-1]["мой_признак"].eq(2).all() and len(seen[-1]) == 2
    assert trained.features(roads)["мой_признак"].eq(2).all()
    assert trained.predict_log_trips(roads).notna().all()


def test_feature_length_is_the_single_cost_multiplier(examples):
    roads, _ = example_roads()

    def doubled_length(roads):
        table = examples["features"](roads)
        table["длина_м"] *= 2
        return table

    cost = comfort_cost({}, features=doubled_length)
    pd.testing.assert_series_equal(cost(roads), (2 * roads.length_m).rename("cost"))


def test_visible_rideable_matches_the_library_on_restrictions_and_overrides(examples):
    from itertools import product
    from cycle_routing.graph import FORBIDDEN_HIGHWAY

    rows = list(product(
        ["residential", "footway", *sorted(FORBIDDEN_HIGHWAY)],
        [None, "private", "customers;delivery"],
        [None, "yes", "designated", "permissive", "no", "dismount", "private", "yes;no"],
        [None, "private"],
    ))
    roads = pd.DataFrame(rows, columns=["highway", "access", "bicycle", "service"])
    roads.loc[len(roads)] = ["residential;steps", None, "yes", None]
    translated = russian(roads)
    pd.testing.assert_series_equal(examples["rideable"](translated), rideable(roads))


@pytest.mark.parametrize("graph_type", [nx.DiGraph, nx.MultiDiGraph])
def test_greedy_explores_dead_ends_and_cycles_without_comparing_node_types(examples, graph_type):
    graph = graph_type()
    for node, point in [(0, (0, 0)), (1, (1, 1)), (("dead",), (1, -1)), ("goal", (2, 0)), ("outside", (5, 0))]:
        graph.add_node(node, x=point[0], y=point[1])
    graph.add_edges_from([(0, ("dead",)), (0, 1), (("dead",), 0), (1, "goal")], cost=1)
    greedy = examples["greedy_search"]
    assert greedy(graph, 0, "goal") == [0, 1, "goal"]
    assert greedy(graph, 0, 0) == [0]
    with pytest.raises(nx.NetworkXNoPath):
        greedy(graph, 0, "outside")
    with pytest.raises(nx.NodeNotFound):
        greedy(graph, 0, "missing")


def test_greedy_uses_distance_while_dijkstra_minimizes_cost(examples):
    graph = nx.DiGraph()
    for node, point in [(0, (0, 0)), (1, (9, 0)), (2, (0, 1)), (3, (10, 0))]:
        graph.add_node(node, x=point[0], y=point[1])
    graph.add_weighted_edges_from([(0, 1, 100), (1, 3, 100), (0, 2, 1), (2, 3, 1)], weight="cost")
    assert examples["greedy_search"](graph, 0, 3) == [0, 1, 3]
    assert examples["my_search"](graph, 0, 3) == [0, 2, 3]


def test_notebook_turn_penalties_and_greedy_work_in_turn_graph(examples):
    roads, nodes = example_roads()
    graph = build_graph(roads, nodes, "length_m")
    penalty = examples["my_turn_penalty"]
    incoming, outgoing = (1, 2, 0), (2, 3, 0)
    assert penalty(graph, incoming, outgoing) == turn_penalty(
        graph, incoming, outgoing, TurnConfig(right_penalty_m=10)
    ) == 10
    expanded = turn_graph(graph, penalty)
    assert expanded.nodes[incoming] == {"x": graph.nodes[2]["x"], "y": graph.nodes[2]["y"]}
    for turns in [None, penalty]:
        route = find_route(graph, 1, 3, search=examples["greedy_search"], turns=turns)
        assert route is not None and route["edge_route"][-1][1] == 3


def test_turn_diagram_uses_the_notebook_function_and_signed_angles(examples):
    import matplotlib.pyplot as plt
    from types import SimpleNamespace
    from cycle_routing.workshop_maps import plot_turn_examples

    nodes = make_nodes({0: (0, 0), 1: (0, -100), 2: (0, 100), 3: (64, 76), 4: (100, 0), 5: (-100, 0)}, crs="EPSG:32636")
    roads = make_edges([
        {"u": u, "v": v, "highway": "residential", "length_m": 100}
        for u, v in [(1, 0), (0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
    ], nodes=nodes, crs=nodes.crs)
    network = SimpleNamespace(roads=roads, city=SimpleNamespace(osm=roads, nodes=nodes))
    measured = []

    def penalty(graph, incoming, outgoing):
        before = edge_bearings(graph, incoming)[1]
        after = edge_bearings(graph, outgoing)[0]
        metres = examples["my_turn_penalty"](graph, incoming, outgoing)
        assert incoming in graph.edges and outgoing in graph.edges and incoming[1] == outgoing[0]
        measured.append(((after - before + 180) % 360 - 180, metres))
        return metres

    figure = plot_turn_examples(network, penalty)
    assert [value for _, value in measured] == [0, 8, 90, 10, 30]
    angles = [angle for angle, _ in measured]
    assert abs(angles[0]) <= 25 and 25 < abs(angles[1]) <= 55 and abs(angles[2]) >= 150
    assert 55 < angles[3] < 150 and -150 < angles[4] < -55
    assert len(figure.axes) == 4
    plt.close(figure)


def test_notebook_train_model_fits_and_predicts(examples):
    x = pd.DataFrame({"велодорожка": [0, 0, 1, 1]})
    y = pd.Series([0.0, 0.0, 3.0, 3.0])
    model = examples["train_model"](x, y)
    assert list(model.predict(x)) == list(y)
