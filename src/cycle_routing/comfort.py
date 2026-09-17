"""Stage 3 - comfort: the edge table turned into a routing cost in metres.

A comfort function takes the whole edge table and returns one cost per edge:

    def my_comfort(edges: gpd.GeoDataFrame) -> pd.Series: ...

``default_comfort`` is the reference one: 10 multipliers from OSM tags times the length.
``comfort_components`` computes the same model for a single edge; it only feeds
``comfort_feature_table``, the table that explains every multiplier.
"""

from __future__ import annotations

from collections.abc import Mapping
import math

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd

from .config import DEFAULT_COMFORT_CONFIG
from .tags import first_number, has_any, tag_values


LANES_PENALTY_FROM = 3
NARROW_WIDTH_M = 2.0
MIN_LENGTH_M = 0.1
NO_CYCLEWAY = {"no", "none", "separate"}
MAJOR_HIGHWAYS = {"motorway", "trunk", "primary", "secondary"}
FOOT_HIGHWAYS = {"footway", "pedestrian", "steps", "corridor"}
OFFROAD_HIGHWAYS = {"path", "track", "bridleway"}
SURFACE_CLASSES = {
    "асфальт/бетон": {"asphalt", "paved", "concrete", "concrete:plates", "concrete:lanes"},
    "плитка": {"paving_stones", "paving_stones:30"},
    "брусчатка": {"sett", "cobblestone", "unhewn_cobblestone"},
    "грунт/гравий": {
        "compacted", "fine_gravel", "gravel", "pebblestone", "ground", "dirt",
        "earth", "mud", "sand", "grass", "unpaved", "woodchips",
    },
}
MULTIPLIERS = (
    "highway_factor", "surface_factor", "cycleway_mult", "bicycle_factor", "smoothness_factor",
    "service_factor", "maxspeed_mult", "lanes_mult", "width_mult", "lit_mult",
)


def road_class(highways: set[str], has_cycleway: bool, bicycle: set[str]) -> str:
    """Human-readable road category used in tables and maps."""

    base = {value.removesuffix("_link") for value in highways}
    if "cycleway" in base or (base & (FOOT_HIGHWAYS | OFFROAD_HIGHWAYS) and "designated" in bicycle):
        return "велодорожка"
    if base & MAJOR_HIGHWAYS:
        return "крупная + велополоса" if has_cycleway else "крупная дорога"
    if base & FOOT_HIGHWAYS:
        return "тротуар/пешеходная"
    if base & OFFROAD_HIGHWAYS:
        return "тропа/грунтовка"
    return "улица"


def surface_class(surfaces: set[str]) -> str:
    if not surfaces:
        return "не указано"
    for label, values in SURFACE_CLASSES.items():
        if surfaces & values:
            return label
    return "другое"


# --- vectorised model: the whole table at once -----------------------------------------------


def _max_weight(column: pd.Series, weights: Mapping[str, float], unknown: float, missing: float) -> np.ndarray:
    """Largest weight among the ``;``-separated values of each cell; ``missing`` for empty cells."""

    lookup = {
        text: max(weights.get(value, unknown) for value in text.split(";"))
        for text in column.dropna().unique()
    }
    return column.map(lookup).fillna(missing).to_numpy(dtype=float)


def comfort_factors(edges: pd.DataFrame, config: Mapping[str, object] = DEFAULT_COMFORT_CONFIG) -> pd.DataFrame:
    """The 10 multipliers of every edge and their product ``comfort_factor`` (at least ``min_factor``)."""

    cfg = config
    mult, speed, defaults = cfg["multipliers"], cfg["maxspeed"], cfg["defaults"]
    highway_is_cycleway = has_any(edges["highway"], "cycleway").to_numpy()
    painted = edges["cycleway"].map(
        {text: bool(set(text.split(";")) - NO_CYCLEWAY) for text in edges["cycleway"].dropna().unique()}
    ).fillna(False).to_numpy(dtype=bool)
    maxspeed = edges["maxspeed"].to_numpy(dtype=float)
    lanes = edges["lanes"].to_numpy(dtype=float)
    width = edges["width"].to_numpy(dtype=float)
    extra = np.minimum((maxspeed - speed["threshold_kmh"]) / speed["scale_kmh"], speed["max_extra"])
    factors = {
        # No highway tag counts as "unclassified", like in the scalar model.
        "highway_factor": _max_weight(
            edges["highway"], cfg["highway"], defaults["highway"],
            cfg["highway"].get("unclassified", defaults["highway"]),
        ),
        "surface_factor": _max_weight(edges["surface"], cfg["surface"], defaults["surface"], defaults["surface"]),
        # highway=cycleway already has its own factor: do not reward it twice.
        "cycleway_mult": np.where(painted & ~highway_is_cycleway, mult["dedicated_cycleway"], 1.0),
        "bicycle_factor": _max_weight(edges["bicycle"], cfg["bicycle"], 1.0, 1.0),
        "smoothness_factor": _max_weight(edges["smoothness"], cfg["smoothness"], 1.0, 1.0),
        "service_factor": _max_weight(edges["service"], cfg["service"], 1.0, 1.0),
        "maxspeed_mult": np.where((maxspeed > 0) & (maxspeed > speed["threshold_kmh"]), 1.0 + extra, 1.0),
        "lanes_mult": np.where(lanes >= LANES_PENALTY_FROM, mult["many_lanes"], 1.0),
        "width_mult": np.where((width > 0) & (width < NARROW_WIDTH_M), mult["narrow"], 1.0),
        "lit_mult": np.where(has_any(edges["lit"], "no").to_numpy(), mult["unlit"], 1.0),
    }
    product = factors["highway_factor"]
    for name in MULTIPLIERS[1:]:
        product = product * factors[name]
    factors["comfort_factor"] = np.maximum(product, cfg["min_factor"])
    return pd.DataFrame(factors, index=edges.index)


def default_comfort(edges: pd.DataFrame, config: Mapping[str, object] = DEFAULT_COMFORT_CONFIG) -> pd.Series:
    """Reference comfort function: ``length_m × comfort_factor`` for every edge, in metres.

    ``config`` holds the weights (see ``DEFAULT_COMFORT_CONFIG``); presets are this function with
    other weights. Copy it and change what you like.
    """

    factor = comfort_factors(edges, config)["comfort_factor"].to_numpy()
    length = np.maximum(edges["length_m"].to_numpy(dtype=float), MIN_LENGTH_M)
    return pd.Series(length * factor, index=edges.index, name="cost")


def add_comfort_cost(
    graph: nx.MultiDiGraph,
    config: Mapping[str, object] | None = None,
    cost=None,
) -> nx.MultiDiGraph:
    """Legacy graph adapter retained for old examples and regression tests.

    The tabular pipeline uses :func:`default_comfort` and :func:`build_graph`.  This adapter keeps the
    original public API usable without giving it access to prepared popularity data.
    """

    cfg = config or DEFAULT_COMFORT_CONFIG
    for _, _, _, data in graph.edges(keys=True, data=True):
        row = {**data, "length_m": data.get("length_m", data.get("length", 0.0))}
        if cost is None:
            components = comfort_components(row, cfg)
            data["comfort_factor"] = components["comfort_factor"]
            data["comfort_cost"] = components["comfort_cost"]
        else:
            length = max(float(row["length_m"] or 0.0), MIN_LENGTH_M)
            data["comfort_cost"] = float(cost(data))
            data["comfort_factor"] = data["comfort_cost"] / length
    return graph


# --- scalar model: one edge, for the explanation table ---------------------------------------


def _edge_tags(row: Mapping[str, object]) -> dict:
    highways = tag_values(row.get("highway")) or {"unclassified"}
    bicycle = tag_values(row.get("bicycle"))
    return {
        "highways": highways,
        "surfaces": tag_values(row.get("surface")),
        "bicycle": bicycle,
        "has_cycleway": bool(tag_values(row.get("cycleway")) - NO_CYCLEWAY) or "cycleway" in highways,
        "smoothness": tag_values(row.get("smoothness")),
        "service": tag_values(row.get("service")),
        "maxspeed_kmh": first_number(row.get("maxspeed")),
        "lanes": first_number(row.get("lanes")),
        "width_m": first_number(row.get("width")),
        "unlit": "no" in tag_values(row.get("lit")),
    }


def _comfort_factors(tags: dict, cfg: Mapping[str, object]) -> dict[str, float]:
    mult = cfg["multipliers"]
    ms_cfg = cfg["maxspeed"]
    defaults = cfg["defaults"]
    maxspeed = tags["maxspeed_kmh"]
    speed_extra = 0.0
    if maxspeed and maxspeed > ms_cfg["threshold_kmh"]:
        speed_extra = min((maxspeed - ms_cfg["threshold_kmh"]) / ms_cfg["scale_kmh"], ms_cfg["max_extra"])
    return {
        "highway_factor": max(cfg["highway"].get(value, defaults["highway"]) for value in tags["highways"]),
        "surface_factor": (
            max(cfg["surface"].get(value, defaults["surface"]) for value in tags["surfaces"])
            if tags["surfaces"]
            else defaults["surface"]
        ),
        # highway=cycleway already has its own factor: do not reward it twice.
        "cycleway_mult": (
            mult["dedicated_cycleway"]
            if tags["has_cycleway"] and "cycleway" not in tags["highways"]
            else 1.0
        ),
        "bicycle_factor": max((cfg["bicycle"].get(value, 1.0) for value in tags["bicycle"]), default=1.0),
        "smoothness_factor": max((cfg["smoothness"].get(value, 1.0) for value in tags["smoothness"]), default=1.0),
        "service_factor": max((cfg["service"].get(value, 1.0) for value in tags["service"]), default=1.0),
        "maxspeed_mult": 1.0 + speed_extra,
        "lanes_mult": mult["many_lanes"] if tags["lanes"] and tags["lanes"] >= LANES_PENALTY_FROM else 1.0,
        "width_mult": mult["narrow"] if tags["width_m"] and tags["width_m"] < NARROW_WIDTH_M else 1.0,
        "lit_mult": mult["unlit"] if tags["unlit"] else 1.0,
    }


def comfort_components(row: Mapping[str, object], config: Mapping[str, object] | None = None) -> dict:
    """OSM features of one edge (a row of the edge table) and every multiplier of its comfort factor."""

    cfg = config or DEFAULT_COMFORT_CONFIG
    tags = _edge_tags(row)
    factors = _comfort_factors(tags, cfg)
    length = float(row.get("length_m", 0.0))
    comfort_factor = max(float(math.prod(factors.values())), cfg["min_factor"])
    return {
        "highway": ";".join(sorted(tags["highways"])),
        "road_class": road_class(tags["highways"], tags["has_cycleway"], tags["bicycle"]),
        "surface": ";".join(sorted(tags["surfaces"])) or None,
        "surface_class": surface_class(tags["surfaces"]),
        "has_cycleway": tags["has_cycleway"],
        "bicycle": ";".join(sorted(tags["bicycle"])) or None,
        "smoothness": ";".join(sorted(tags["smoothness"])) or None,
        "service": ";".join(sorted(tags["service"])) or None,
        "maxspeed_kmh": tags["maxspeed_kmh"],
        "lanes": tags["lanes"],
        "width_m": tags["width_m"],
        "unlit": tags["unlit"],
        "length_m": length,
        **factors,
        "comfort_factor": comfort_factor,
        "comfort_cost": max(length, MIN_LENGTH_M) * comfort_factor,
    }


def comfort_feature_table(
    edges: gpd.GeoDataFrame | nx.MultiDiGraph,
    config: Mapping[str, object] | None = None,
) -> gpd.GeoDataFrame:
    """Explanation table, one row per edge: features, road and surface class, all 10 multipliers.

    Indexed by ``(u, v, key)``. Row by row, so pass a sample or the edges of a route, not a whole city.
    """

    if isinstance(edges, nx.MultiDiGraph):
        graph = edges
        rows = {
            (u, v, key): {
                **comfort_components(
                    {**data, "length_m": data.get("length_m", data.get("length", 0.0))}, config
                ),
                **({"stad_trips": float(data["stad_trips"])} if "stad_trips" in data else {}),
            }
            for u, v, key, data in graph.edges(keys=True, data=True)
        }
        table = pd.DataFrame.from_dict(rows, orient="index")
        table.index = pd.MultiIndex.from_tuples(table.index, names=["u", "v", "key"])
        geometry = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True).geometry
        return gpd.GeoDataFrame(table, geometry=geometry.reindex(table.index), crs=graph.graph["crs"])

    rows = [comfort_components(row, config) for row in edges.drop(columns="geometry").to_dict("records")]
    index = pd.MultiIndex.from_frame(edges[["u", "v", "key"]])
    table = pd.DataFrame(rows, index=index, columns=None if rows else ["road_class", "surface_class"])
    return gpd.GeoDataFrame(table, geometry=edges.geometry.to_numpy(), crs=edges.crs)
