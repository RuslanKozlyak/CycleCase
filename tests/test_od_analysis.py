import numpy as np
import pandas as pd

from cycle_routing import build_routes, make_edges, make_nodes
from cycle_routing.od_analysis import describe_od_pairs, evaluate_od_models, route_overlap, select_topology_od


def grid():
    nodes = make_nodes({5*y+x: (1000*x, 1000*y) for y in range(5) for x in range(5)}, crs="EPSG:25832")
    rows=[]
    for y in range(5):
        for x in range(5):
            u=5*y+x
            for v in ([u+1] if x<4 else []) + ([u+5] if y<4 else []):
                rows.extend([{"u":u,"v":v,"highway":"residential"},{"u":v,"v":u,"highway":"residential"}])
    return make_edges(rows, crs="EPSG:25832", nodes=nodes), nodes


def test_selection_is_deterministic_and_does_not_use_popularity_or_node_density_order():
    edges,nodes=grid()
    surveyed=pd.Series(True,index=edges.index)
    pairs,pool=select_topology_od(edges,nodes,nodes.node,surveyed=surveyed,n_pairs=8,n_anchors=15)
    repeated,_=select_topology_od(edges.assign(trips=np.arange(len(edges))*1000),nodes,nodes.node[::-1],
                                  surveyed=surveyed,n_pairs=8,n_anchors=15)
    pd.testing.assert_frame_equal(pairs,repeated)
    assert len(pairs)==8 and not pairs[["origin","destination"]].duplicated().any()
    assert pool.length_km.between(1.5,10).all()
    assert pool.survey_coverage_pct.eq(100).all()
    assert pool.length_band.nunique()>=2 and pool.orientation.nunique()==4
    routes=build_routes(edges,nodes,pairs,{})
    info=describe_od_pairs(edges,nodes,pairs,routes,surveyed=surveyed)
    assert len(info)==8 and info.survey_coverage_pct.eq(100).all()
    assert info.unavoidable_street_pct.eq(0).all(), "The grid offers physical alternatives to every edge"
    assert info.junctions_per_km.gt(0).all()


def test_description_identifies_a_corridor_without_alternatives():
    nodes=make_nodes({1:(0,0),2:(1000,0),3:(2000,0)},crs="EPSG:25832")
    edges=make_edges([{"u":u,"v":v,"highway":"residential"} for u,v in [(1,2),(2,1),(2,3),(3,2)]],
                     crs="EPSG:25832",nodes=nodes)
    pairs=pd.DataFrame({"od_id":["A"],"origin":[1],"destination":[3]})
    routes=build_routes(edges,nodes,pairs,{})
    info=describe_od_pairs(edges,nodes,pairs,routes,surveyed=pd.Series(True,index=edges.index))
    assert info.loc["A","unavoidable_street_pct"]==100
    assert info.loc["A","junctions_per_km"]==0


def test_overlap_compares_physical_streets_and_lengths_in_both_directions():
    edges=pd.DataFrame({"u":[1,2,2,3],"v":[2,1,3,2],"length_m":[100,100,300,300]})
    routes=pd.DataFrame({"od_id":["A","B","C"],"edge_route":[[(1,2,0)],[(2,1,0)],[(1,2,0),(2,3,0)]]})
    overlap=route_overlap(routes,edges)
    assert overlap.loc["A","B"]==1
    assert overlap.loc["A","C"]==0.25
    assert np.allclose(overlap,overlap.T)


def test_model_comparison_uses_the_same_od_subset_for_popularity_detour_and_turns():
    edges=pd.DataFrame({"u":[1,2,1],"v":[2,3,3],"key":[0,0,0],"length_m":[100,100,250]})
    routes=pd.DataFrame([
        {"od_id":"A","algorithm":"shortest","edge_route":[(1,2,0),(2,3,0)],"turns":1},
        {"od_id":"A","algorithm":"model","edge_route":[(1,3,0)],"turns":0},
        {"od_id":"B","algorithm":"shortest","edge_route":[(1,2,0)],"turns":0},
        {"od_id":"B","algorithm":"model","edge_route":[(1,3,0)],"turns":5},
    ])
    summary,details=evaluate_od_models(routes,edges,pd.Series([0.,10.,100.]),mask=pd.Series([False,True,True]))
    assert summary.n_evaluated.eq(1).all()
    assert summary.loc["model","median_detour_pct"]==25
    assert summary.loc["model","all_od_median_detour_pct"]==87.5
    assert summary.loc["model","median_turns"]==0
    assert set(details.loc[details.included,"od_id"])=={"A"}
