import networkx as nx
from shapely.geometry import LineString

from cycle_routing.config import TurnConfig
from cycle_routing.routing import build_routes, dijkstra, dijkstra_turns, route_record


def _corner_or_straight():
    """1 -> 3: north then right at 2 (200 m), one straight diagonal edge (250 m), or a detour via 4 and 5."""

    graph = nx.MultiDiGraph(crs="EPSG:32636")
    points = {1: (0, 0), 2: (0, 100), 3: (100, 100), 4: (0, 205)}
    for node, (x, y) in points.items():
        graph.add_node(node, x=x, y=y)
    graph.add_node(5, x=100, y=205)
    for u, v, length in ((1, 2, 100), (2, 3, 100), (1, 3, 250), (2, 4, 105), (4, 5, 100), (5, 3, 0.1)):
        graph.add_edge(u, v, length=length, comfort_cost=length,
                       geometry=LineString([(graph.nodes[u]["x"], graph.nodes[u]["y"]), (graph.nodes[v]["x"], graph.nodes[v]["y"])]))
    return graph


def test_turn_penalty_changes_the_chosen_route():
    graph = _corner_or_straight()
    assert dijkstra(graph, 1, 3, "comfort_cost") == [(1, 2, 0), (2, 3, 0)]
    assert dijkstra_turns(graph, 1, 3, "comfort_cost", TurnConfig(right_penalty_m=0)) == [(1, 2, 0), (2, 3, 0)]
    assert dijkstra_turns(graph, 1, 3, "comfort_cost", TurnConfig(right_penalty_m=60)) == [(1, 3, 0)]


def test_route_record_accepts_any_search_function():
    graph = _corner_or_straight()
    calls = []

    def fixed_detour(graph, origin, destination, weight, turns):
        calls.append((origin, destination, weight, turns))
        return [(1, 2, 0), (2, 4, 0), (4, 5, 0), (5, 3, 0)]

    record = route_record(graph, 1, 3, "balanced", "OD", search=fixed_detour)
    assert calls == [(1, 3, "comfort_cost", None)]
    assert record["edge_route"][-1] == (5, 3, 0)
    assert abs(record["route_length_m"] - 305.1) < 1e-9
    assert record["turns"] == 2  # right at 4, right at 5

    routes = build_routes(graph, [("OD", 1, 3)], {"custom": {"config": {}, "search": fixed_detour}})
    assert routes.set_index("algorithm").loc["custom", "route_length_m"] > routes.set_index("algorithm").loc["shortest", "route_length_m"]


def test_min_turns_preset_prefers_longer_straight_route():
    from cycle_routing.config import COMFORT_PRESETS
    from cycle_routing.routing import build_routes

    # 1 -> 4 either as a zig-zag of 3 edges (2 turns, 300 m) or around a straight line of 2 edges (0 turns, 340 m)
    graph = nx.MultiDiGraph(crs="EPSG:32636")
    points = {1: (0, 0), 2: (100, 0), 3: (100, 100), 4: (200, 100), 5: (170, 0), 6: (200, 0)}
    for node, (x, y) in points.items():
        graph.add_node(node, x=x, y=y)
    for u, v, length in ((1, 2, 100), (2, 3, 100), (3, 4, 100), (1, 5, 170), (5, 6, 30), (6, 4, 140)):
        graph.add_edge(u, v, length=length, highway="residential", surface="asphalt",
                       geometry=LineString([points[u], points[v]]))
    routes = build_routes(graph, [("OD", 1, 4)], {"min_turns": COMFORT_PRESETS["min_turns"]}).set_index("algorithm")
    assert routes.loc["shortest", "turns"] == 2
    assert routes.loc["min_turns", "turns"] == 1
    assert routes.loc["min_turns", "route_length_m"] > routes.loc["shortest", "route_length_m"]


def test_build_routes_reuses_disk_cache(tmp_path, monkeypatch):
    import cycle_routing.routing as routing
    from cycle_routing.config import COMFORT_PRESETS

    graph = nx.MultiDiGraph(crs="EPSG:32636")
    for node, (x, y) in {1: (0, 0), 2: (100, 0), 3: (200, 0)}.items():
        graph.add_node(node, x=x, y=y)
    graph.add_edge(1, 2, length=100, highway="residential")
    graph.add_edge(2, 3, length=100, highway="cycleway")
    presets = {"balanced": COMFORT_PRESETS["balanced"]}
    first = routing.build_routes(graph, [("OD", 1, 3)], presets, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.pkl"))) == 2

    monkeypatch.setattr(routing, "route_record", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recomputed")))
    second = routing.build_routes(graph, [("OD", 1, 3)], presets, cache_dir=tmp_path)
    assert second.drop(columns="geometry").equals(first.drop(columns="geometry"))

    changed = {"balanced": {**COMFORT_PRESETS["balanced"], "turns": None}}
    monkeypatch.undo()
    routing.build_routes(graph, [("OD", 1, 3)], changed, cache_dir=tmp_path)
    assert len(list(tmp_path.glob("*.pkl"))) == 3
