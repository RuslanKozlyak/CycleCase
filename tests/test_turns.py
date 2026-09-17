import networkx as nx
import pytest

from cycle_routing import COMFORT_PRESETS, build_graph, build_routes, find_route, make_edges, make_nodes
from cycle_routing.config import TurnConfig

CRS = "EPSG:32636"


def _corner_or_straight():
    """1 -> 3: north then right at 2 (200 m), one straight diagonal edge (250 m), or a detour via 4 and 5."""

    nodes = make_nodes({1: (0, 0), 2: (0, 100), 3: (100, 100), 4: (0, 205), 5: (100, 205)}, crs=CRS)
    edges = make_edges(
        [{"u": u, "v": v, "length_m": length}
         for u, v, length in ((1, 2, 100), (2, 3, 100), (1, 3, 250), (2, 4, 105), (4, 5, 100), (5, 3, 0.1))],
        crs=CRS, nodes=nodes,
    )
    return build_graph(edges, nodes, "length_m"), edges, nodes


def _route(graph, **kwargs):
    return find_route(graph, 1, 3, **kwargs)["edge_route"]


def test_turn_penalty_changes_the_chosen_route():
    graph, _, _ = _corner_or_straight()
    assert _route(graph, turns=None) == [(1, 2, 0), (2, 3, 0)]
    assert _route(graph, turns=TurnConfig(right_penalty_m=0)) == [(1, 2, 0), (2, 3, 0)]
    assert _route(graph, turns=TurnConfig(right_penalty_m=60)) == [(1, 3, 0)]


@pytest.mark.parametrize("search", [nx.dijkstra_path, nx.bellman_ford_path, nx.astar_path, nx.shortest_path])
def test_networkx_path_functions_plug_in_as_is(search):
    graph, _, _ = _corner_or_straight()
    assert _route(graph, search=search, turns=None) == [(1, 2, 0), (2, 3, 0)]
    assert _route(graph, search=search, turns=TurnConfig(right_penalty_m=60)) == [(1, 3, 0)]


def test_no_route_gives_none():
    graph, _, _ = _corner_or_straight()
    assert find_route(graph, 3, 1) is None
    assert find_route(graph, 3, 1, turns=None) is None
    assert find_route(graph, 1, 999) is None


def test_turn_model_can_be_replaced_by_a_function():
    """Own turn model: free where the road just bends, expensive at a real junction."""

    graph, _, _ = _corner_or_straight()

    def junction_only(graph, incoming, outgoing):
        return 60.0 if graph.out_degree(outgoing[0]) > 1 else 0.0

    assert graph.out_degree(2) > 1  # node 2 is a junction: 2->3 and 2->4
    assert _route(graph, turns=junction_only) == [(1, 3, 0)]
    assert _route(graph, turns=lambda graph, incoming, outgoing: 0.0) == [(1, 2, 0), (2, 3, 0)]


def test_min_turns_preset_prefers_longer_straight_route():
    # 1 -> 4 either as a zig-zag of 3 edges (2 turns, 300 m) or around a straight line of 2 edges (1 turn, 340 m)
    nodes = make_nodes({1: (0, 0), 2: (100, 0), 3: (100, 100), 4: (200, 100), 5: (170, 0), 6: (200, 0)}, crs=CRS)
    edges = make_edges(
        [{"u": u, "v": v, "length_m": length, "highway": "residential", "surface": "asphalt"}
         for u, v, length in ((1, 2, 100), (2, 3, 100), (3, 4, 100), (1, 5, 170), (5, 6, 30), (6, 4, 140))],
        crs=CRS, nodes=nodes,
    )
    routes = build_routes(edges, nodes, [("OD", 1, 4)], {"min_turns": COMFORT_PRESETS["min_turns"]})
    routes = routes.set_index("algorithm")
    assert routes.loc["shortest", "turns"] == 2
    assert routes.loc["min_turns", "turns"] == 1
    assert routes.loc["min_turns", "route_length_m"] > routes.loc["shortest", "route_length_m"]


def test_build_routes_reuses_disk_cache(tmp_path, monkeypatch):
    import cycle_routing.routing as routing

    _, edges, nodes = _corner_or_straight()
    presets = {"balanced": COMFORT_PRESETS["balanced"]}
    first = routing.build_routes(edges, nodes, [("OD", 1, 3)], presets, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.pkl"))) == 2

    monkeypatch.setattr(routing, "find_route", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recomputed")))
    second = routing.build_routes(edges, nodes, [("OD", 1, 3)], presets, cache_dir=tmp_path)
    assert second.drop(columns="geometry").equals(first.drop(columns="geometry"))

    monkeypatch.undo()
    changed = {"balanced": {**COMFORT_PRESETS["balanced"], "search": nx.bellman_ford_path}}
    routing.build_routes(edges, nodes, [("OD", 1, 3)], changed, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.pkl"))) == 3
