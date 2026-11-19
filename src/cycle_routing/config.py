from dataclasses import dataclass

OSM_CONNECT_TIMEOUT = 15
OVERPASS_QUERY_TIMEOUT = 180

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

    network_type: str = "all"  # OSM network type
    simplify: bool = True
    retain_all: bool = False
    truncate_by_edge: bool = True
    requests_timeout: int = 300
    overpass_urls: tuple[str, ...] = (
        "https://ovp.dresel.systems/api",
        "https://overpass-api.de/api",
        "https://maps.mail.ru/osm/tools/overpass/api",
        "https://overpass.private.coffee/api",
    )

    def servers_for(self, bounds=None):
        # The dresel mirror contains European data only.
        europe = (bounds is not None and -25 <= bounds[0] <= bounds[2] <= 45
                  and 35 <= bounds[1] <= bounds[3] <= 72)
        return tuple(url for url in self.overpass_urls if "ovp.dresel.systems" not in url or europe)

    def graph_kwargs(self) -> dict[str, object]:
        return {
            "network_type": self.network_type,
            "simplify": self.simplify,
            "retain_all": self.retain_all,
            "truncate_by_edge": self.truncate_by_edge,
        }

@dataclass(frozen=True, slots=True)
class TurnConfig:
    straight_penalty_m: float = 0.0
    slight_penalty_m: float = 0.0
    right_penalty_m: float = 0.0
    left_penalty_m: float = 0.0
    u_turn_penalty_m: float = 0.0
    straight_angle_max_deg: float = 25.0
    slight_angle_max_deg: float = 55.0
    u_turn_angle_min_deg: float = 150.0

    def penalty(self, turn_kind: str) -> float:
        return float(getattr(self, f'{turn_kind}_penalty_m'))
