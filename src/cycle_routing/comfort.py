"""Edge-level bicycle comfort cost model."""

from __future__ import annotations

from collections.abc import Mapping
import math

import geopandas as gpd
import networkx as nx
import osmnx as ox
import pandas as pd

from .config import DEFAULT_COMFORT_CONFIG
from .tags import first_number, tag_values


CYCLEWAY_KEYS = ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both")
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


def _edge_tags(data: Mapping[str, object]) -> dict:
    highways = tag_values(data.get("highway")) or {"unclassified"}
    cycle_tags = set().union(*(tag_values(data.get(key)) for key in CYCLEWAY_KEYS))
    bicycle = tag_values(data.get("bicycle"))
    return {
        "highways": highways,
        "surfaces": tag_values(data.get("surface")),
        "bicycle": bicycle,
        "has_cycleway": bool(cycle_tags - {"no", "none", "separate"}) or "cycleway" in highways,
        "footway_without_bicycle": "footway" in highways and not bicycle & {"yes", "designated", "permissive"},
        "maxspeed_kmh": first_number(data.get("maxspeed")),
        "lanes": first_number(data.get("lanes")),
        "unlit": "no" in tag_values(data.get("lit")),
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
        "footway_mult": mult["footway_without_bicycle"] if tags["footway_without_bicycle"] else 1.0,
        "maxspeed_mult": 1.0 + speed_extra,
        "lanes_mult": (
            mult["many_lanes"] if tags["lanes"] and tags["lanes"] >= cfg["lanes_penalty_from"] else 1.0
        ),
        "lit_mult": mult["unlit"] if tags["unlit"] else 1.0,
    }


def _factor_and_cost(data: Mapping[str, object], factors: dict[str, float], cfg) -> tuple[float, float]:
    comfort_factor = max(float(math.prod(factors.values())), cfg["min_factor"])
    length = max(float(data.get("length", 1.0)), cfg["min_length_m"])
    return comfort_factor, length * comfort_factor


def comfort_components(data: Mapping[str, object], config: Mapping[str, object] | None = None) -> dict:
    """Return the OSM features of one edge and every multiplier of its comfort factor."""

    cfg = config or DEFAULT_COMFORT_CONFIG
    tags = _edge_tags(data)
    factors = _comfort_factors(tags, cfg)
    comfort_factor, comfort_cost = _factor_and_cost(data, factors, cfg)
    return {
        "highway": ";".join(sorted(tags["highways"])),
        "road_class": road_class(tags["highways"], tags["has_cycleway"], tags["bicycle"]),
        "surface": ";".join(sorted(tags["surfaces"])) or None,
        "surface_class": surface_class(tags["surfaces"]),
        "has_cycleway": tags["has_cycleway"],
        "footway_without_bicycle": bool(tags["footway_without_bicycle"]),
        "maxspeed_kmh": tags["maxspeed_kmh"],
        "lanes": tags["lanes"],
        "unlit": tags["unlit"],
        "length_m": float(data.get("length", 0.0)),
        **factors,
        "comfort_factor": comfort_factor,
        "comfort_cost": comfort_cost,
    }


def add_comfort_cost(
    graph: nx.MultiDiGraph,
    config: Mapping[str, object] | None = None,
) -> nx.MultiDiGraph:
    """Add ``comfort_factor`` and additive ``comfort_cost`` to every edge."""

    cfg = config or DEFAULT_COMFORT_CONFIG
    for _, _, _, data in graph.edges(keys=True, data=True):
        factors = _comfort_factors(_edge_tags(data), cfg)
        data["comfort_factor"], data["comfort_cost"] = _factor_and_cost(data, factors, cfg)
    return graph


def comfort_feature_table(
    graph: nx.MultiDiGraph,
    config: Mapping[str, object] | None = None,
) -> gpd.GeoDataFrame:
    """One row per directed edge: OSM features, comfort multipliers, and observed trips."""

    rows = {
        (u, v, key): {
            **comfort_components(data, config),
            **({"stad_trips": float(data["stad_trips"])} if "stad_trips" in data else {}),
        }
        for u, v, key, data in graph.edges(keys=True, data=True)
    }
    table = pd.DataFrame.from_dict(rows, orient="index")
    table.index = pd.MultiIndex.from_tuples(table.index, names=["u", "v", "key"])
    geometry = ox.graph_to_gdfs(graph, nodes=False, fill_edge_geometry=True).geometry
    return gpd.GeoDataFrame(table, geometry=geometry.reindex(table.index), crs=graph.graph["crs"])
