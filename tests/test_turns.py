import networkx as nx
from shapely.geometry import LineString

from cycle_routing.config import TurnConfig
from cycle_routing.routing import route_record, shortest_edge_path_with_turns


def test_right_turn_penalty_is_added_to_objective():
    graph = nx.MultiDiGraph(crs="EPSG:32636")
    graph.add_node(1, x=0, y=-1)
    graph.add_node(2, x=0, y=0)
    graph.add_node(3, x=1, y=0)
    graph.add_edge(
        1,
        2,
        comfort_cost=1,
        length=1,
        geometry=LineString([(0, -1), (0, 0)]),
    )
    graph.add_edge(
        2,
        3,
        comfort_cost=1,
        length=1,
        geometry=LineString([(0, 0), (1, 0)]),
    )
    result = shortest_edge_path_with_turns(
        graph,
        1,
        3,
        turn_config=TurnConfig(enabled=True, right_penalty_m=20),
    )
    edge_path, objective, turn_penalty = result
    assert edge_path == [(1, 2, 0), (2, 3, 0)]
    assert objective == 22
    assert turn_penalty == 20


def test_named_comfort_model_uses_turn_penalties():
    graph = nx.MultiDiGraph(crs="EPSG:32636")
    graph.add_node(1, x=0, y=-1)
    graph.add_node(2, x=0, y=0)
    graph.add_node(3, x=1, y=0)
    graph.add_edge(
        1,
        2,
        comfort_cost=1,
        length=1,
        geometry=LineString([(0, -1), (0, 0)]),
    )
    graph.add_edge(
        2,
        3,
        comfort_cost=1,
        length=1,
        geometry=LineString([(0, 0), (1, 0)]),
    )
    record = route_record(
        graph,
        1,
        3,
        "comfort_optimized",
        "OD-1",
        turn_config=TurnConfig(enabled=True, right_penalty_m=20),
    )
    assert record["objective_cost"] == 22
    assert record["turn_penalty_total_m"] == 20
