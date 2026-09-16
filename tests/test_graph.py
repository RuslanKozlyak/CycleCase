from cycle_routing.config import GraphConfig
from cycle_routing.graph import graph_cache_path


def test_graph_cache_changes_with_simplification_and_network_type(tmp_path):
    everything = graph_cache_path(tmp_path, "Konstanz", GraphConfig())
    raw = graph_cache_path(tmp_path, "Konstanz", GraphConfig(simplify=False))
    bike = graph_cache_path(tmp_path, "Konstanz", GraphConfig(network_type="bike"))
    assert len({everything.name, raw.name, bike.name}) == 3


def test_default_filter_keeps_sidewalks_and_drops_what_bikes_cannot_use():
    from cycle_routing.graph import is_rideable

    assert is_rideable({"highway": "footway"})  # uncomfortable, not forbidden: weights decide
    assert is_rideable({"highway": "track", "surface": "gravel"})
    assert is_rideable({"highway": "service", "access": "private", "bicycle": "yes"})  # explicitly allowed
    assert not is_rideable({"highway": "motorway"})
    assert not is_rideable({"highway": "steps"})
    assert not is_rideable({"highway": "residential", "bicycle": "no"})
    assert not is_rideable({"highway": "service", "access": "private"})
    assert not is_rideable({"highway": "service", "service": "private"})


def test_filter_edges_drops_edges_and_orphan_nodes():
    import networkx as nx

    from cycle_routing.graph import filter_edges

    graph = nx.MultiDiGraph(crs="EPSG:32636")
    for node in (1, 2, 3):
        graph.add_node(node, x=node, y=0)
    graph.add_edge(1, 2, highway="residential", length=10)
    graph.add_edge(2, 3, highway="steps", length=10)
    kept = filter_edges(graph, lambda tags: tags["highway"] != "steps")
    assert list(kept.edges(keys=True)) == [(1, 2, 0)]
    assert set(kept.nodes) == {1, 2}
