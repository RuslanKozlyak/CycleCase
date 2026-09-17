from pathlib import Path

import folium
import pandas as pd

from cycle_routing.preparation import make_edges, make_nodes
from cycle_routing import visualization


def test_mapped_popularity_map_colours_observed_edges_and_adds_feature_popup(monkeypatch):
    nodes = make_nodes({1: (0, 0), 2: (100, 0), 3: (200, 0)}, crs="EPSG:3857")
    edges = make_edges(
        [
            {"u": 1, "v": 2, "highway": "cycleway", "surface": "asphalt"},
            {"u": 2, "v": 3, "highway": "primary", "maxspeed": 60},
        ],
        crs=nodes.crs,
        nodes=nodes,
    )
    trips = pd.DataFrame({"u": [1, 2], "v": [2, 3], "key": [0, 0], "trips": [120.0, 0.0]})
    monkeypatch.setattr(visualization, "load_popularity", lambda *args, **kwargs: trips)

    fmap = visualization.mapped_popularity_map(edges, "test", cache_dir=Path("."))

    layers = [child for child in fmap._children.values() if isinstance(child, folium.GeoJson)]
    assert len(layers) == 1
    features = layers[0].data["features"]
    assert len(features) == 1
    assert features[0]["properties"]["trips"] == "120"
    assert features[0]["properties"]["highway"] == "cycleway"
    assert features[0]["properties"]["surface"] == "asphalt"


def test_mapped_popularity_map_rejects_missing_edge_values(monkeypatch):
    nodes = make_nodes({1: (0, 0), 2: (100, 0)}, crs="EPSG:3857")
    edges = make_edges([{"u": 1, "v": 2}], crs=nodes.crs, nodes=nodes)
    trips = pd.DataFrame({"u": [], "v": [], "key": [], "trips": []})
    monkeypatch.setattr(visualization, "load_popularity", lambda *args, **kwargs: trips)

    try:
        visualization.mapped_popularity_map(edges, "test", cache_dir=Path("."))
    except ValueError as error:
        assert "no data for 1 edges" in str(error)
    else:
        raise AssertionError("missing popularity keys must be rejected")
