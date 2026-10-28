import geopandas as gpd
import numpy as np
import pandas as pd
import pytest

from cycle_routing import prepare_city, is_rideable
from cycle_routing.evaluation import edge_trips
from cycle_routing.workshop import (
    EvaluationRules, compare, evaluate_routes, export_solution, features, load_lab,
    preset, priority_cost, route_diagnostics, run_routes,
)
from cycle_routing.workshop_data import write_lab_pack
from cycle_routing.workshop_maps import route_explorer


@pytest.fixture
def lab(toy_city,tmp_path):
    edges,nodes=prepare_city("toytown",cache_dir=toy_city["cache_dir"])
    edges=edges[is_rideable(edges)]
    pairs=pd.DataFrame({"od_id":["A"],"origin":[1],"destination":[4],
                        "start_lat":[47.66],"start_lon":[9.17],"end_lat":[47.661],"end_lon":[9.171]})
    trips=edge_trips(edges,"toytown",cache_dir=toy_city["cache_dir"])
    # The north road is held out; south is public for validation, with explicit low counts.
    partition=pd.Series("train",index=edges.index)
    partition.loc[edges.highway.str.contains("primary")]= "validation"
    partition.loc[edges.highway.str.contains("cycleway")]= "test"
    public=tmp_path/"public"; private=tmp_path/"private"
    write_lab_pack(edges,nodes,pairs,trips,partition,folder=public,private_folder=private,city="toy")
    return load_lab(public),private


def test_features_have_no_targets_and_numeric_defaults_are_explicit(lab):
    data,_=lab
    f=features(data.edges.assign(trips=1e9,has_observation=True,popularity=42))
    assert not {"trips","has_observation","popularity","u","v","key","geometry"}&set(f)
    assert f.width_m_missing.equals(data.edges.width.isna())
    assert f.loc[f.width_m_missing, "width_m"].eq(3).all()
    assert f.loc[~f.width_m_missing, "width_m"].equals(data.edges.loc[data.edges.width.notna(), "width"].rename("width_m"))
    cost=priority_cost(traffic=2,cycling=1)(data.edges)
    assert cost.index.equals(data.edges.index) and np.isfinite(cost).all() and cost.gt(0).all()


def test_public_pack_does_not_expose_test_labels_and_load_never_downloads(lab):
    data,private=lab
    assert not (data.folder/"labels_test.parquet").exists()
    assert (private/"test_labels.parquet").exists()
    with pytest.raises(ValueError,match="организатор"):
        data.labels_for("test")
    with pytest.raises(FileNotFoundError,match="подготовленного"):
        load_lab(data.folder/"missing")


def test_baseline_has_50_points_and_over_budget_does_not_disappear(lab):
    data,private=lab
    solutions={"bike":preset("bike",cost=priority_cost(traffic=10,cycling=2),turns=None)}
    routes=run_routes(data,solutions)
    labels=pd.read_parquet(private/"test_labels.parquet")
    # Test labels only cover the north path; evaluate instead all explicitly known streets here.
    public=data.labels_for("validation")
    private_y=labels.set_index(["u","v","key"]).trips.reindex(pd.MultiIndex.from_frame(data.edges[["u","v","key"]]))
    y=public.combine_first(pd.Series(private_y.to_numpy(),index=data.edges.index))
    full=data.edges[["u","v","key"]].assign(trips=y).dropna(subset="trips")
    result=evaluate_routes(data,routes,labels=full,rules=EvaluationRules(max_detour_pct=0),solutions=solutions)
    assert result.table.loc["shortest","балл / 100"]==50
    assert result.table.loc["bike","балл / 100"]==0
    assert result.table.loc["bike","OD для оценки"]==1
    missing=evaluate_routes(data,routes[routes.algorithm.eq("shortest")],labels=full,solutions=solutions)
    assert missing.table.loc["bike","балл / 100"]==0
    assert missing.table.loc["bike","OD для оценки"]==1


def test_invalid_paths_and_forged_lengths_are_not_accepted(lab):
    data,_=lab
    routes=run_routes(data,{"mine":preset("mine",cost="length_m",turns=None)})
    forged=routes.copy(); forged["route_length_m"]=1
    result=evaluate_routes(data,forged,solutions={"mine":{}})
    assert result.table.loc["mine","объезд, %"]==0
    forged=routes.copy(); index=forged.index[forged.algorithm.eq("mine")][0]
    forged.at[index,"edge_route"]=[(1,4,999)]
    with pytest.raises(ValueError,match="Некорректный"):
        evaluate_routes(data,forged)


def test_submission_roundtrip_and_offline_map(lab,tmp_path):
    data,_=lab
    result=compare(data,{"mine":preset("mine",cost="length_m",turns=None)})
    diagnostics=route_diagnostics(data,result)
    assert diagnostics.loc["mine","OD изменено"]==0
    folder=export_solution(data,result,algorithm="mine",folder=tmp_path/"submission")
    restored=gpd.read_parquet(folder/"routes.parquet")
    assert len(restored)==1
    evaluate_routes(data,restored,solutions={"mine":{}})
    path=tmp_path/"map.html"
    view=route_explorer(data,result,filename=path)
    text=path.read_text(encoding="utf-8")
    assert "__DATA__" not in text and "__LEAFLET_JS__" not in text
    assert "L.map('map'" in text and 'src="http' not in text
    assert "нет данных" in text and 'title="Сравнение' in view._repr_html_()
