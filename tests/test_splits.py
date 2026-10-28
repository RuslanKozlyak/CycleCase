import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point

from cycle_routing.splits import by_routes, by_sectors

X0, Y0 = 513_000, 5_280_000


def grid_city(n, step=250):
    """n x n two-way residential grid in metres (EPSG:25832)."""
    nodes = pd.DataFrame([(i * n + j, X0 + j * step, Y0 + i * step) for i in range(n) for j in range(n)],
                         columns=["node", "x", "y"])
    nodes = gpd.GeoDataFrame(nodes, geometry=[Point(x, y) for x, y in zip(nodes.x, nodes.y)], crs="EPSG:25832")
    rows = []
    for i in range(n):
        for j in range(n):
            for di, dj in ((0, 1), (1, 0)):
                if i + di < n and j + dj < n:
                    a, b = i * n + j, (i + di) * n + j + dj
                    for u, v in ((a, b), (b, a)):
                        rows.append({"u": u, "v": v, "key": 0, "length_m": float(step), "highway": "residential",
                                     **dict.fromkeys(["surface", "smoothness", "cycleway", "bicycle", "lit", "service", "access"]),
                                     "maxspeed": np.nan, "lanes": np.nan, "width": np.nan, "oneway": False})
    edges = pd.DataFrame(rows)
    xy = nodes.set_index("node")[["x", "y"]]
    geometry = [LineString([xy.loc[u], xy.loc[v]]) for u, v in zip(edges.u, edges.v)]
    return gpd.GeoDataFrame(edges, geometry=geometry, crs="EPSG:25832"), nodes


def shortest_edges(edges, origin, destination):
    graph = nx.DiGraph()
    graph.add_weighted_edges_from(zip(edges.u, edges.v, edges.length_m))
    path = nx.dijkstra_path(graph, origin, destination)
    index = dict(zip(zip(edges.u, edges.v), edges.index))
    return [index[pair] for pair in zip(path, path[1:])]


def test_route_zones_keep_every_shortest_path_inside_its_zone():
    edges, nodes = grid_city(12)
    everywhere = pd.Series(True, index=edges.index)
    split = by_routes(edges, nodes, everywhere, rideable=everywhere, per_zone_stratum=2, separation_m=400,
                      reach_m=600, buffer_m=100, min_endpoint_m=200, max_overlap=0.5, min_variety=1, order_seeds=range(3),
                      per_band=80, seed=1)
    counts = split.pairs.zone.value_counts()
    assert counts.get("open", 0) == counts.get("closed", 0) > 0
    assert set(split.zone) <= {"open", "closed", "buffer"} and split.params["min_gap_m"] >= 400
    for pair in split.pairs.itertuples():
        assert split.zone.loc[shortest_edges(edges, pair.origin, pair.destination)].eq(pair.zone).all()


def test_sector_split_uses_pairs_inside_one_colour():
    edges, nodes = grid_city(14)
    everywhere = pd.Series(True, index=edges.index)
    split = by_sectors(edges, nodes, everywhere, rideable=everywhere, sector_m=1750, per_zone_stratum=2,
                       buffer_m=100, min_endpoint_m=200, max_overlap=0.5, min_variety=1, order_seeds=range(3), per_band=120,
                       seed=1)
    counts = split.pairs.zone.value_counts()
    assert counts.get("open", 0) == counts.get("closed", 0) > 0
    for pair in split.pairs[split.pairs.zone.eq("closed")].itertuples():
        inside = split.zone.loc[shortest_edges(edges, pair.origin, pair.destination)].eq("closed")
        assert inside.mean() >= 0.5


def test_route_variety_counts_distinct_paths_within_the_detour_limit(toy_city):
    from cycle_routing import is_rideable, prepare_city
    from cycle_routing.splits import route_variety
    edges, nodes = prepare_city("toytown", cache_dir=toy_city["cache_dir"])
    edges = edges[is_rideable(edges)]
    # 1 -> 4: the short primary road or the slightly longer cycleway; 1 -> 3: the only reasonable path,
    # the way round over 2 and 4 is three times longer than the 20% limit allows.
    variety = route_variety(edges, nodes, pd.DataFrame({"origin": [1, 1], "destination": [4, 3]}))
    assert variety.tolist() == [2, 1]


def test_candidates_below_the_variety_threshold_are_dropped(tmp_path):
    from cycle_routing.splits import candidate_corridors
    cache = tmp_path / "candidates.pkl"
    pd.DataFrame({"origin": [1, 2, 3], "destination": [4, 5, 6], "variety": [2, 5, 7]}).to_pickle(cache)
    kept = candidate_corridors(None, None, None, min_variety=5, cache=cache)
    assert kept.variety.tolist() == [5, 7]
