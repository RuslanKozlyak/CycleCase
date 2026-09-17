import json

import pandas as pd
import pytest

from cycle_routing import EDGE_COLUMNS, load_od_pairs, prepare_city, preparation_report
from cycle_routing.config import GraphConfig
from cycle_routing.preparation import (
    POPULARITY_COLUMNS,
    edges_table,
    graph_cache_path,
    load_observations,
    load_popularity,
    make_edges,
    validate_edges,
)

from conftest import CRS, raw_graph


def test_graph_cache_changes_with_simplification_and_network_type(tmp_path):
    everything = graph_cache_path(tmp_path, "Konstanz", GraphConfig())
    raw = graph_cache_path(tmp_path, "Konstanz", GraphConfig(simplify=False))
    bike = graph_cache_path(tmp_path, "Konstanz", GraphConfig(network_type="bike"))
    assert len({everything.name, raw.name, bike.name}) == 3


def test_edge_table_is_flat_and_has_one_row_per_graph_edge():
    graph = raw_graph()
    graph.remove_edge(2, 4, 0)
    graph.add_edge(2, 4, osmid=[11, 12], length=110.0, oneway=True, highway=["cycleway", "Path"],
                   lit=["yes", "no"], maxspeed=["30", "50"], width="2,5 m")  # no geometry: straight line
    edges, osmids = edges_table(graph, CRS)

    validate_edges(edges)
    assert list(edges.columns) == list(EDGE_COLUMNS)
    assert len(edges) == graph.number_of_edges()
    assert list(zip(edges["u"], edges["v"], edges["key"])) == list(graph.edges(keys=True))
    cells = edges.drop(columns="geometry").to_numpy().ravel()
    assert not any(isinstance(cell, (list, tuple, set, dict)) for cell in cells)

    row = edges[(edges["u"] == 2) & (edges["v"] == 4)].iloc[0]
    assert row["highway"] == "cycleway;path" and row["lit"] == "no;yes"
    assert row["maxspeed"] == 30.0 and row["width"] == 2.5 and bool(row["oneway"])
    assert row.geometry.length == pytest.approx(116.6, abs=0.1)  # straight line between the projected nodes
    assert osmids[row.name] == [11, 12]
    primary = edges[(edges["u"] == 1) & (edges["v"] == 3)].iloc[0]
    assert primary["cycleway"] == "no" and pd.isna(primary["bicycle"]) and primary["lanes"] == 4.0


def test_make_edges_follows_the_schema():
    edges = make_edges(
        [{"u": 1, "v": 2, "highway": "Primary", "maxspeed": 50}], crs=CRS,
        nodes=pd.DataFrame({"node": [1, 2], "x": [0.0, 30.0], "y": [0.0, 40.0]}),
    )
    validate_edges(edges)
    assert edges.loc[0, "highway"] == "primary" and edges.loc[0, "length_m"] == 50.0
    assert pd.isna(edges.loc[0, "surface"])


def test_prepare_city_caches_tables_and_keeps_popularity_apart(toy_city):
    cache_dir = toy_city["cache_dir"]
    edges, nodes = prepare_city("toytown", cache_dir=cache_dir)

    folder = cache_dir / "toytown"
    for name in ("edges", "nodes", "popularity", "observations", "od_pairs"):
        assert (folder / f"{name}.parquet").exists()
    assert list(folder.glob("*.graphml"))
    assert list(edges.columns) == list(EDGE_COLUMNS)
    assert list(nodes.columns) == ["node", "x", "y", "geometry"]
    assert len(edges) == raw_graph().number_of_edges()  # every edge, rideable or not
    assert {"steps", "service"} <= set(edges["highway"])

    popularity = load_popularity("toytown", cache_dir=cache_dir)
    assert list(popularity.columns) == list(POPULARITY_COLUMNS)
    assert popularity[["u", "v", "key"]].equals(edges[["u", "v", "key"]])
    trips = popularity.set_index(["u", "v", "key"])["trips"]
    assert trips[(1, 2, 0)] == trips[(2, 1, 0)] == 120  # directionless volume on both directions
    assert trips[(1, 5, 0)] == 0  # steps are not in the reference network

    stored = pd.read_parquet(folder / "edges.parquet")
    assert not {"trips", "stad_trips", "gpx_tracks", "popularity"} & set(stored.columns)
    assert set(stored.columns) == set(EDGE_COLUMNS)

    pairs = load_od_pairs("toytown", cache_dir=cache_dir)
    assert pairs[["od_id", "origin", "destination"]].values.tolist() == [["A", 1, 4]]
    assert len(load_observations("toytown", cache_dir=cache_dir)) == 4
    assert preparation_report("toytown", cache_dir=cache_dir)["stad_match_rate"] == 1


def test_prepare_city_reuses_the_cache_until_its_key_changes(toy_city):
    from cycle_routing import preparation

    cache_dir = toy_city["cache_dir"]
    first, _ = prepare_city("toytown", cache_dir=cache_dir)
    second, _ = prepare_city("toytown", cache_dir=cache_dir)
    assert len(toy_city["downloads"]) == 1
    assert second.drop(columns="geometry").equals(first.drop(columns="geometry"))
    meta = json.loads((cache_dir / "toytown" / "meta.json").read_text(encoding="utf-8"))
    assert meta["key"]["schema"] == preparation.SCHEMA_VERSION and "access" in meta["key"]["tags"]

    # A new schema version rebuilds the tables from the GraphML without downloading again.
    def rebuilt(*args, **kwargs):
        raise RuntimeError("rebuilt")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(preparation, "SCHEMA_VERSION", preparation.SCHEMA_VERSION + 1)
        patch.setattr(preparation, "edges_table", rebuilt)
        with pytest.raises(RuntimeError, match="rebuilt"):
            prepare_city("toytown", cache_dir=cache_dir)
    assert len(toy_city["downloads"]) == 1

    prepare_city("toytown", cache_dir=cache_dir, refresh=True)
    assert len(toy_city["downloads"]) == 2


def test_gpx_city_counts_tracks_per_edge(toy_city):
    cache_dir = toy_city["cache_dir"]
    prepare_city("toytracks", cache_dir=cache_dir)
    trips = load_popularity("toytracks", cache_dir=cache_dir).set_index(["u", "v", "key"])["trips"]
    assert trips[(1, 2, 0)] == trips[(2, 1, 0)] == 2
    assert trips[(1, 3, 0)] == 0
