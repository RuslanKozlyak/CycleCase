"""Configuration objects shared by notebooks and library modules."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass


# OSM way tags downloaded with the graph: 10 features picked by association with STADTRADELN trips
# in Konstanz (cycleway:* counts as one feature), oneway/junction that OSMnx needs to orient edges,
# and access for the rideability filter. Changing this list changes every cache key.
OSM_FEATURE_TAGS = (
    "highway",
    "surface",
    "smoothness",
    "lit",
    "maxspeed",
    "cycleway",
    "cycleway:left",
    "cycleway:right",
    "cycleway:both",
    "bicycle",
    "lanes",
    "width",
    "service",
)
OSM_WAY_TAGS = ("oneway", "junction", "access", *OSM_FEATURE_TAGS)


@dataclass(frozen=True, slots=True)
class GraphConfig:
    """Parameters that determine graph semantics and its cache identity."""

    network_type: str = "all"  # "all" keeps sidewalks and forbidden roads; keep_edge / weights deal with them
    simplify: bool = True
    retain_all: bool = False
    truncate_by_edge: bool = True
    requests_timeout: int = 300
    overpass_urls: tuple[str, ...] = (
        "https://overpass-api.de/api",
        "https://maps.mail.ru/osm/tools/overpass/api",
        "https://overpass.private.coffee/api",
    )

    def graph_kwargs(self) -> dict[str, object]:
        return {
            "network_type": self.network_type,
            "simplify": self.simplify,
            "retain_all": self.retain_all,
            "truncate_by_edge": self.truncate_by_edge,
        }


@dataclass(frozen=True, slots=True)
class MatchConfig:
    """Rules for assigning STADTRADELN segments to routable OSM edges."""

    osmid_max_distance_m: float = 20.0
    geometry_max_distance_m: float = 15.0
    direction_tie_tolerance_m: float = 0.25


@dataclass(frozen=True, slots=True)
class TurnConfig:
    """Geometry-based turn penalties expressed in equivalent metres; pass ``None`` for no penalties."""

    straight_penalty_m: float = 0.0
    slight_penalty_m: float = 8.0
    right_penalty_m: float = 20.0
    left_penalty_m: float = 30.0
    u_turn_penalty_m: float = 90.0
    straight_angle_max_deg: float = 25.0
    slight_angle_max_deg: float = 55.0
    u_turn_angle_min_deg: float = 150.0

    def penalty(self, turn_kind: str) -> float:
        return float(getattr(self, f"{turn_kind}_penalty_m"))


DEFAULT_COMFORT_CONFIG = {
    "highway": {
        "cycleway": 0.55,
        "living_street": 0.80,
        "path": 0.85,
        "residential": 1.00,
        "pedestrian": 1.10,
        "service": 1.10,
        "track": 1.20,
        "unclassified": 1.15,
        "footway": 1.30,
        "tertiary": 1.25,
        "secondary": 1.55,
        "primary": 2.10,
        "trunk": 3.50,
        "motorway": 8.00,
        "steps": 6.00,
    },
    "surface": {
        "asphalt": 1.00,
        "paved": 1.02,
        "concrete": 1.05,
        "paving_stones": 1.12,
        "sett": 1.45,
        "cobblestone": 1.55,
        "compacted": 1.15,
        "fine_gravel": 1.20,
        "gravel": 1.35,
        "ground": 1.50,
        "dirt": 1.65,
        "sand": 2.10,
        "grass": 1.80,
        "unpaved": 1.40,
    },
    "smoothness": {
        "excellent": 0.95,
        "good": 1.00,
        "intermediate": 1.10,
        "bad": 1.35,
        "very_bad": 1.70,
        "horrible": 2.50,
        "very_horrible": 3.50,
        "impassable": 8.00,
    },
    # Driveways and parking aisles carry almost no bicycle trips.
    "service": {"driveway": 1.40, "parking_aisle": 1.50, "drive-through": 1.50, "alley": 1.15},
    # use_sidepath: cyclists must ride the separate path next to this road.
    "bicycle": {"designated": 0.85, "use_sidepath": 1.60, "discouraged": 1.50, "dismount": 3.00, "private": 3.00, "no": 5.00},
    "multipliers": {
        "dedicated_cycleway": 0.65,
        "many_lanes": 1.20,
        "narrow": 1.20,
        "unlit": 1.12,
    },
    "maxspeed": {"threshold_kmh": 50.0, "scale_kmh": 100.0, "max_extra": 0.50},
    "defaults": {"highway": 1.25, "surface": 1.20},
    "min_factor": 0.35,
}


def with_overrides(base: dict, overrides: dict) -> dict:
    """Deep copy of ``base`` with nested ``overrides`` applied."""

    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict):
            result[key] = with_overrides(result.get(key, {}), value)
        else:
            result[key] = value
    return result


def __getattr__(name: str):
    """Keep ``cycle_routing.config.COMFORT_PRESETS`` as a lazy compatibility alias."""

    if name == "COMFORT_PRESETS":
        from .presets import COMFORT_PRESETS

        return COMFORT_PRESETS
    raise AttributeError(name)
