import networkx as nx
import pandas as pd
import pytest
from shapely.geometry import LineString

from cycle_routing import observed_area, prepare_city
from cycle_routing.comfort import comfort_feature_table
from cycle_routing.evaluation import (
    comfort_correlation,
    composition_table,
    edge_trips,
    route_edge_association,
    route_edge_value_sums,
    route_edges_long,
    weighted_corr,
)


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


def test_route_edge_value_sums_all_trips_and_counts_repeated_edges():
    features = pd.DataFrame(
        {"trips": [10.0, 20.0]},
        index=pd.MultiIndex.from_tuples([(1, 2, 0), (2, 3, 0)], names=["u", "v", "key"]),
    )
    edge_rows = pd.DataFrame({
        "group": ["mine", "mine", "mine", "other"],
        "u": [1, 2, 1, 2],
        "v": [2, 3, 2, 3],
        "key": [0, 0, 0, 0],
    })

    result = route_edge_value_sums(edge_rows, features, "trips")
    assert result.to_dict() == {"mine": 40.0, "other": 20.0}


def test_observed_area_keeps_every_observed_edge_and_drops_the_unsurveyed(toy_city):
    edges, _ = prepare_city("toytown", cache_dir=toy_city["cache_dir"])
    trips = edge_trips(edges, "toytown", cache_dir=toy_city["cache_dir"])

    assert observed_area(edges, "toytown", cache_dir=toy_city["cache_dir"]).all()

    # A grid far finer than the town: only the surveyed roads stay inside.
    tight = observed_area(edges, "toytown", cache_dir=toy_city["cache_dir"], cell_m=5.0, dilate=1)
    assert tight[trips > 0].all(), "an edge with observed trips must never fall outside the area"
    assert not tight.all(), "the steps and the private driveway carry no observations"


def test_comfort_correlation_follows_the_observed_streets(toy_city):
    """In the toy town the cycleway carries the trips and the primary does not."""

    cache_dir = toy_city["cache_dir"]
    edges, _ = prepare_city("toytown", cache_dir=cache_dir)

    def weights(cycleway: float, primary: float):
        # "cycleway;path" is one value of a joined tag, so match on the first token.
        def cost(frame):
            classes = frame["highway"].str.split(";").str[0]
            return frame["length_m"] * classes.map({"cycleway": cycleway, "primary": primary}).fillna(1.0)

        return cost

    likes_cycleways = comfort_correlation(edges, weights(0.25, 4.0), "toytown", cache_dir=cache_dir)
    likes_primaries = comfort_correlation(edges, weights(4.0, 0.25), "toytown", cache_dir=cache_dir)

    assert likes_cycleways > 0.5
    # Reciprocal weights negate every comfort, so the correlation flips sign and keeps its size.
    assert likes_primaries == pytest.approx(-likes_cycleways)
    # Cost proportional to length says nothing about any street.
    assert pd.isna(comfort_correlation(edges, "length_m", "toytown", cache_dir=cache_dir))


def test_unknown_trip_sum_remains_unknown_and_explicit_zero_remains_zero():
    features = pd.DataFrame({"trips": [float("nan"), 0.]},
                            index=pd.MultiIndex.from_tuples([(1, 2, 0), (2, 3, 0)], names=["u", "v", "key"]))
    rows = pd.DataFrame({"group": ["unknown", "zero"], "u": [1, 2], "v": [2, 3], "key": [0, 0]})
    result = route_edge_value_sums(rows, features, "trips")
    assert pd.isna(result["unknown"])
    assert result["zero"] == 0


def test_correlation_excludes_unknown_target_instead_of_zero_filling():
    assert weighted_corr([1., 2., 1000.], [1., 2., float("nan")], [1., 1., 100000.]) == pytest.approx(1)
