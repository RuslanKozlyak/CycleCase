import pytest

from cycle_routing.config import GraphConfig
from cycle_routing.graph import graph_cache_path


def test_graph_cache_changes_with_simplification_and_network_type(tmp_path):
    everything = graph_cache_path(tmp_path, "Konstanz", GraphConfig())
    raw = graph_cache_path(tmp_path, "Konstanz", GraphConfig(simplify=False))
    bike = graph_cache_path(tmp_path, "Konstanz", GraphConfig(network_type="bike"))
    assert len({everything.name, raw.name, bike.name}) == 3


def test_default_filter_keeps_sidewalks_and_drops_what_bikes_cannot_use():
    from cycle_routing import is_rideable, make_edges, make_nodes

    tags = [
        ({"highway": "footway"}, True),  # uncomfortable, not forbidden: weights decide
        ({"highway": "track", "surface": "gravel"}, True),
        ({"highway": "service", "access": "private", "bicycle": "yes"}, True),  # explicitly allowed
        ({"highway": "motorway"}, False),
        ({"highway": "steps"}, False),
        ({"highway": "residential", "bicycle": "no"}, False),
        ({"highway": "service", "access": "private"}, False),
        ({"highway": "service", "service": "private"}, False),
    ]
    nodes = make_nodes({1: (0, 0), 2: (10, 0)}, crs="EPSG:3857")
    edges = make_edges([{"u": 1, "v": 2, "key": key, **row} for key, (row, _) in enumerate(tags)], crs="EPSG:3857", nodes=nodes)
    assert is_rideable(edges).tolist() == [expected for _, expected in tags]
    assert is_rideable(edges, allow={"steps"})[4]
    with pytest.raises(KeyError):
        is_rideable(edges, allow={"teleport"})
