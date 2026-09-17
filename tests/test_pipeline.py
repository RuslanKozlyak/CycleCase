import numpy as np
import pandas as pd
import pytest

from cycle_routing import (
    COMFORT_PRESETS,
    ConnectivityWarning,
    build_graph,
    check_connectivity,
    default_comfort,
    filter_edges,
    make_edges,
    make_nodes,
)
from cycle_routing.comfort import comfort_components


CRS = "EPSG:3857"


def _tables():
    nodes = make_nodes({1: (0, 0), 2: (10, 0), 3: (20, 0), 4: (100, 0), 5: (110, 0)}, crs=CRS)
    edges = make_edges(
        [
            {"u": 1, "v": 2, "highway": "residential", "surface": "asphalt"},
            {"u": 2, "v": 3, "highway": "primary", "surface": "sett"},
            {"u": 4, "v": 5, "highway": "cycleway", "surface": "asphalt"},
        ],
        crs=CRS,
        nodes=nodes,
    )
    return edges, nodes


def test_build_graph_uses_public_cost_and_keeps_no_tags_or_popularity():
    edges, nodes = _tables()
    leaked = edges.assign(stad_trips=[10, 20, 30])

    def cost(table):
        assert "highway" in table and "stad_trips" in table
        return table["length_m"] * pd.Series([1.0, 2.0, 3.0], index=table.index)

    with pytest.warns(ConnectivityWarning):
        graph = build_graph(leaked, nodes, cost)
    assert graph.edges[2, 3, 0]["cost"] == 20.0
    assert set(graph.edges[2, 3, 0]) == {"cost", "length_m", "geometry"}


@pytest.mark.parametrize("bad", [[1.0, 0.0, 1.0], [1.0, -1.0, 1.0], [1.0, np.nan, 1.0]])
def test_build_graph_rejects_invalid_cost(bad):
    edges, nodes = _tables()
    with pytest.raises(ValueError, match="Стоимость"):
        build_graph(edges, nodes, pd.Series(bad, index=edges.index))


def test_vector_comfort_matches_scalar_reference():
    edges, _ = _tables()
    expected = pd.Series(
        [comfort_components(row)["comfort_cost"] for row in edges.drop(columns="geometry").to_dict("records")],
        index=edges.index,
    )
    pd.testing.assert_series_equal(default_comfort(edges), expected.rename("cost"))


def test_connectivity_warns_and_can_keep_the_largest_component():
    edges, nodes = _tables()
    with pytest.warns(ConnectivityWarning):
        checked = check_connectivity(edges, nodes, report=False)
    assert len(checked) == 3
    with pytest.warns(ConnectivityWarning):
        largest = check_connectivity(edges, nodes, keep_largest=True, report=False)
    assert list(zip(largest["u"], largest["v"])) == [(1, 2), (2, 3)]


def test_filter_keeps_the_masked_rows_and_optionally_the_largest_piece():
    edges, _ = _tables()
    kept = filter_edges(edges, edges["highway"] != "primary", report=False)
    assert list(zip(kept["u"], kept["v"])) == [(1, 2), (4, 5)]
    kept = filter_edges(edges, lambda table: table["highway"] != "primary", keep_largest=True, report=True)
    assert len(kept) == 1


def test_filter_can_restore_real_connectors_without_restoring_rejected_branches():
    nodes = make_nodes({
        1: (0, 0), 2: (10, 0), 3: (20, 0), 4: (30, 0), 5: (40, 0), 6: (20, 10),
    }, crs=CRS)
    edges = make_edges(
        [
            {"u": 1, "v": 2, "highway": "residential"},
            {"u": 2, "v": 3, "highway": "steps"},
            {"u": 3, "v": 4, "highway": "steps"},
            {"u": 4, "v": 5, "highway": "residential"},
            {"u": 3, "v": 6, "highway": "steps"},
        ],
        crs=CRS,
        nodes=nodes,
    )

    kept = filter_edges(
        edges,
        edges["highway"] != "steps",
        connect_components=True,
        report=False,
    )
    assert list(zip(kept["u"], kept["v"])) == [(1, 2), (2, 3), (3, 4), (4, 5)]
    assert check_connectivity(kept, nodes, report=False).equals(kept)


def test_all_presets_use_the_same_public_contract():
    assert len(COMFORT_PRESETS) == 8
    assert all(
        set(preset) == {"label", "description", "cost", "turns", "search"}
        for preset in COMFORT_PRESETS.values()
    )
    assert all(callable(preset["cost"]) and callable(preset["search"]) for preset in COMFORT_PRESETS.values())


def test_plain_data_and_code_are_the_same_interface():
    from cycle_routing import build_routes, default_comfort, drop_values, edge_costs

    edges, nodes = _tables()
    by_tool = filter_edges(edges, drop_values(edges, highway=["primary"]), report=False)
    by_code = filter_edges(edges, lambda table: table["highway"] != "primary", report=False)
    assert by_tool.equals(by_code)

    weights = {"highway": {"primary": 9.0}}
    assert edge_costs(edges, weights)[1] > edge_costs(edges, "balanced")[1]
    pd.testing.assert_series_equal(edge_costs(edges, "balanced"), default_comfort(edges))

    light = {"label": "json", "cost": weights, "turns": {"left_penalty_m": 60}, "search": "bellman_ford"}
    with pytest.warns(ConnectivityWarning):
        routes = build_routes(edges, nodes, [("OD", 1, 3)], {"json": light})
    assert set(routes["algorithm"]) == {"shortest", "json"}
