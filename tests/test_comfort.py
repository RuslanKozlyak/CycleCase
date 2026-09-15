import networkx as nx

from cycle_routing.comfort import add_comfort_cost
from cycle_routing.tags import edge_osmid_set, tag_values


def test_aggregated_tags_survive_simplification_shape():
    assert tag_values(["residential", "cycleway"]) == {"residential", "cycleway"}
    assert edge_osmid_set({"osmid": [10, 20]}) == {10, 20}


def test_comfort_cost_is_additive_edge_weight():
    graph = nx.MultiDiGraph(crs="EPSG:25832")
    graph.add_edge(1, 2, length=100, highway="cycleway", surface="asphalt")
    add_comfort_cost(graph)
    edge = graph.edges[1, 2, 0]
    assert edge["comfort_factor"] == 0.55
    assert abs(edge["comfort_cost"] - 55) < 1e-9


def test_missing_surface_uses_configured_default():
    graph = nx.MultiDiGraph(crs="EPSG:25832")
    graph.add_edge(1, 2, length=100, highway="residential")
    add_comfort_cost(graph)
    assert graph.edges[1, 2, 0]["comfort_factor"] == 1.2


def test_feature_table_multipliers_reproduce_comfort_factor():
    from shapely.geometry import LineString

    from cycle_routing.comfort import comfort_feature_table

    graph = nx.MultiDiGraph(crs="EPSG:25832")
    graph.add_node(1, x=0, y=0)
    graph.add_node(2, x=100, y=0)
    graph.add_edge(
        1, 2, length=100, highway="primary", surface="sett", cycleway="lane",
        maxspeed="70", lanes="4", lit="no", geometry=LineString([(0, 0), (100, 0)]),
    )
    add_comfort_cost(graph)
    row = comfort_feature_table(graph).iloc[0]
    multipliers = ["highway_factor", "surface_factor", "cycleway_mult", "footway_mult",
                   "maxspeed_mult", "lanes_mult", "lit_mult"]
    assert row["road_class"] == "крупная + велополоса"
    assert row["surface_class"] == "брусчатка"
    assert abs(row[multipliers].prod() - graph.edges[1, 2, 0]["comfort_factor"]) < 1e-12
    assert row["comfort_factor"] == graph.edges[1, 2, 0]["comfort_factor"]
