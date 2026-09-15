import networkx as nx
import pandas as pd
from shapely.geometry import LineString

from cycle_routing.comfort import comfort_feature_table
from cycle_routing.evaluation import composition_table, route_edge_association, route_edges_long


def _ladder():
    """Two parallel streets from node 1 to node 4: busy north (1-2-4), quiet south (1-3-4)."""

    graph = nx.MultiDiGraph(crs="EPSG:25832")
    for node, (x, y) in {1: (0, 0), 2: (50, 50), 3: (50, -50), 4: (100, 0)}.items():
        graph.add_node(node, x=x, y=y)
    for (u, v), trips, highway in (((1, 2), 100, "cycleway"), ((2, 4), 100, "cycleway"),
                                   ((1, 3), 0, "primary"), ((3, 4), 0, "primary")):
        for a, b in ((u, v), (v, u)):
            geometry = LineString([(graph.nodes[a]["x"], graph.nodes[a]["y"]), (graph.nodes[b]["x"], graph.nodes[b]["y"])])
            graph.add_edge(a, b, length=geometry.length, highway=highway, stad_trips=trips, geometry=geometry)
    return graph


def test_association_rewards_route_on_popular_edges():
    graph = _ladder()
    features = comfort_feature_table(graph)
    routes = pd.DataFrame({
        "od_id": ["A", "A"],
        "algorithm": ["busy", "quiet"],
        "edge_route": [[(1, 2, 0), (2, 4, 0)], [(1, 3, 0), (3, 4, 0)]],
    })
    result = route_edge_association(routes, features, "stad_trips")
    assert result["busy"] > 0.99
    assert result["quiet"] < -0.99

    composition = composition_table(route_edges_long(routes, features), features, "road_class")
    assert composition.loc["busy", "велодорожка"] == 100
    assert composition.loc["quiet", "крупная дорога"] == 100
