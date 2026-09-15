"""Configuration objects shared by notebooks and library modules."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal


EXTRA_OSM_TAGS = (
    "surface",
    "smoothness",
    "cycleway",
    "cycleway:left",
    "cycleway:right",
    "cycleway:both",
    "cycleway:left:lane",
    "cycleway:right:lane",
    "bicycle",
    "bicycle_road",
    "lit",
    "maxspeed",
    "lanes",
    "sidewalk",
    "incline",
    "width",
)


@dataclass(frozen=True, slots=True)
class GraphConfig:
    """Parameters that determine graph semantics and its cache identity."""

    network_type: str = "bike"
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

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MatchConfig:
    """Rules for assigning STADTRADELN segments to routable OSM edges."""

    osmid_field: str = "osm_way_id"
    flow_field: str = "number_of_matched_trips"
    osmid_max_distance_m: float = 20.0
    geometry_max_distance_m: float = 15.0
    direction_tie_tolerance_m: float = 0.25
    aggregate_trips: Literal["length_weighted_mean", "mean", "max", "sum"] = (
        "length_weighted_mean"
    )


@dataclass(frozen=True, slots=True)
class TurnConfig:
    """Geometry-based turn penalties expressed in equivalent metres."""

    enabled: bool = False
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
    "multipliers": {
        "dedicated_cycleway": 0.65,
        "footway_without_bicycle": 1.35,
        "many_lanes": 1.20,
        "unlit": 1.12,
    },
    "maxspeed": {"threshold_kmh": 50.0, "scale_kmh": 100.0, "max_extra": 0.50},
    "lanes_penalty_from": 3.0,
    "defaults": {"highway": 1.25, "surface": 1.20},
    "min_factor": 0.35,
    "min_length_m": 0.1,
}
