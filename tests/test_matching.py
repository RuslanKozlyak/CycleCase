import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString

from cycle_routing.config import MatchConfig
from cycle_routing.matching import match_stadtradeln_to_graph


def _graph():
    graph = nx.MultiDiGraph(crs="EPSG:25832")
    graph.add_node(1, x=0, y=0)
    graph.add_node(2, x=100, y=0)
    geometry = LineString([(0, 0), (100, 0)])
    graph.add_edge(1, 2, osmid=[10, 11], length=100, geometry=geometry)
    graph.add_edge(2, 1, osmid=[10, 11], length=100, geometry=geometry)
    return graph


def test_osmid_match_selects_both_directions_without_summing_subsegments():
    observed = gpd.GeoDataFrame(
        {
            "osm_way_id": [10, 10],
            "number_of_matched_trips": [20, 40],
        },
        geometry=[LineString([(0, 0), (25, 0)]), LineString([(25, 0), (100, 0)])],
        crs="EPSG:25832",
    )
    graph, metrics, audit = match_stadtradeln_to_graph(_graph(), observed)
    expected = (20 * 25 + 40 * 75) / 100
    assert graph.edges[1, 2, 0]["stad_trips"] == expected
    assert graph.edges[2, 1, 0]["stad_trips"] == expected
    assert metrics["stad_match_rate"] == 1
    assert set(audit["match_method"]) == {"osmid+geometry"}


def test_id_candidate_is_rejected_when_geometry_is_far_away():
    observed = gpd.GeoDataFrame(
        {"osm_way_id": [10], "number_of_matched_trips": [20]},
        geometry=[LineString([(0, 100), (100, 100)])],
        crs="EPSG:25832",
    )
    graph, metrics, _ = match_stadtradeln_to_graph(
        _graph(),
        observed,
        MatchConfig(osmid_max_distance_m=10, geometry_max_distance_m=10),
    )
    assert metrics["stad_match_rate"] == 0
    assert graph.edges[1, 2, 0]["stad_trips"] == 0


def test_geometry_fallback_mirrors_directionless_flow_to_reverse_edge():
    observed = gpd.GeoDataFrame(
        {"osm_way_id": [999], "number_of_matched_trips": [20]},
        geometry=[LineString([(10, 1), (90, 1)])],
        crs="EPSG:25832",
    )
    graph, metrics, audit = match_stadtradeln_to_graph(_graph(), observed)
    assert metrics["stad_matched_via_geometry_only"] == 1
    assert audit.iloc[0]["matched_edge_count"] == 2
    assert graph.edges[1, 2, 0]["stad_trips"] == 20
    assert graph.edges[2, 1, 0]["stad_trips"] == 20


def test_geometry_fallback_keeps_one_way_edge_when_reverse_is_absent():
    graph = _graph()
    graph.remove_edge(2, 1, 0)
    observed = gpd.GeoDataFrame(
        {"osm_way_id": [999], "number_of_matched_trips": [20]},
        geometry=[LineString([(10, 1), (90, 1)])],
        crs="EPSG:25832",
    )
    graph, metrics, audit = match_stadtradeln_to_graph(graph, observed)
    assert metrics["stad_matched_via_geometry_only"] == 1
    assert audit.iloc[0]["matched_edge_count"] == 1
    assert graph.edges[1, 2, 0]["stad_trips"] == 20
