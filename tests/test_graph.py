from cycle_routing.config import GraphConfig
from cycle_routing.graph import graph_cache_path


def test_graph_cache_changes_with_simplification_and_network_type(tmp_path):
    bike = graph_cache_path(tmp_path, "Konstanz", GraphConfig())
    raw = graph_cache_path(tmp_path, "Konstanz", GraphConfig(simplify=False))
    all_roads = graph_cache_path(tmp_path, "Konstanz", GraphConfig(network_type="all"))
    assert len({bike.name, raw.name, all_roads.name}) == 3
