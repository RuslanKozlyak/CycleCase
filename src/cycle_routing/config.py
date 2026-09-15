"""Configuration objects shared by notebooks and library modules."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass


# OSM way tags kept on graph edges: 10 features picked by association with STADTRADELN trips
# in Konstanz (cycleway:* counts as one feature), plus oneway/junction that OSMnx needs to orient edges.
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
OSM_WAY_TAGS = ("oneway", "junction", *OSM_FEATURE_TAGS)


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


def _with_overrides(base: dict, overrides: dict) -> dict:
    """Deep copy of ``base`` with nested ``overrides`` applied."""

    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict):
            result[key] = _with_overrides(result.get(key, {}), value)
        else:
            result[key] = value
    return result


_FAST_CONFIG = _with_overrides(DEFAULT_COMFORT_CONFIG, {
    "highway": {
        "cycleway": 0.85, "living_street": 0.95, "path": 1.05, "service": 1.05,
        "unclassified": 1.0, "pedestrian": 1.15, "footway": 1.2, "track": 1.3,
        "tertiary": 1.0, "secondary": 1.05, "primary": 1.15, "trunk": 2.0,
    },
    "surface": {"sett": 1.2, "cobblestone": 1.25, "gravel": 1.2, "ground": 1.3, "dirt": 1.4, "unpaved": 1.25},
    "multipliers": {"dedicated_cycleway": 0.9, "many_lanes": 1.0, "narrow": 1.0, "unlit": 1.0},
    "smoothness": {"bad": 1.15, "very_bad": 1.3},
    "service": {"driveway": 1.1, "parking_aisle": 1.1},
    "maxspeed": {"max_extra": 0.1},
    "defaults": {"highway": 1.1, "surface": 1.05},
})


# Rider profiles: comfort weights plus turn penalties. Each preset exaggerates one preference
# so that routes visibly differ.
COMFORT_PRESETS = {
    "balanced": {
        "label": "сбалансированный",
        "description": "Исходные веса: умеренный штраф крупных дорог и плохого покрытия.",
        "config": deepcopy(DEFAULT_COMFORT_CONFIG),
    },
    "fast": {
        "label": "быстрый",
        "description": "Почти кратчайший путь: тип дороги и покрытие почти не штрафуются.",
        "config": _FAST_CONFIG,
    },
    "min_turns": {
        "label": "минимум поворотов",
        "description": "Длинные прямые участки: веса как у быстрого, поворот стоит в 10 раз дороже обычного.",
        "config": deepcopy(_FAST_CONFIG),
        "turns": TurnConfig(slight_penalty_m=80.0, right_penalty_m=200.0, left_penalty_m=300.0, u_turn_penalty_m=900.0),
    },
    "avenues": {
        "label": "по проспектам",
        "description": (
            "Как в ваших треках по СПб: крупные улицы без штрафа, заметный бонус за автобусные "
            "и велополосы на них (лучший вариант перебора весов по GPX)."
        ),
        "config": _with_overrides(DEFAULT_COMFORT_CONFIG, {
            "highway": {"secondary": 1.0, "primary": 1.0},
            "multipliers": {"dedicated_cycleway": 0.5},
        }),
    },
    "quiet": {
        "label": "спокойный",
        "description": "Подальше от машин: штраф улиц с движением и скоростью выше 30 км/ч, бонус жилых зон и парков.",
        "config": _with_overrides(DEFAULT_COMFORT_CONFIG, {
            "highway": {
                "cycleway": 0.5, "living_street": 0.6, "pedestrian": 0.8, "path": 0.75, "footway": 1.1,
                "service": 1.3, "unclassified": 1.6, "tertiary": 2.2, "secondary": 3.2, "primary": 4.5, "trunk": 7.0,
            },
            # A painted lane does not make a busy road calm.
            "multipliers": {"dedicated_cycleway": 1.0, "many_lanes": 1.8},
            "maxspeed": {"threshold_kmh": 30.0, "scale_kmh": 40.0, "max_extra": 1.0},
        }),
    },
    "bike_infra": {
        "label": "велоинфраструктура",
        "description": "Только велодорожки и велополосы, остальные улицы и тротуары заметно дороже.",
        "config": _with_overrides(DEFAULT_COMFORT_CONFIG, {
            "highway": {
                "cycleway": 0.3, "living_street": 1.1, "service": 1.5, "unclassified": 1.4, "tertiary": 1.6,
                "path": 1.2, "track": 1.6, "footway": 2.0, "pedestrian": 1.8,
            },
            "multipliers": {"dedicated_cycleway": 0.35},
            "bicycle": {"designated": 0.6, "use_sidepath": 2.5},
            "min_factor": 0.25,
        }),
    },
    "smooth": {
        "label": "гладкий асфальт",
        "description": "Шоссейный велосипед: сильный штраф брусчатки, плитки, грунта и неизвестного покрытия.",
        "config": _with_overrides(DEFAULT_COMFORT_CONFIG, {
            "highway": {"path": 1.5, "track": 2.2, "footway": 1.6, "pedestrian": 1.4, "secondary": 1.3, "primary": 1.6},
            "surface": {
                "paved": 1.05, "concrete": 1.1, "paving_stones": 1.6, "sett": 2.8, "cobblestone": 3.2,
                "compacted": 2.0, "fine_gravel": 2.2, "gravel": 3.0, "ground": 3.5, "dirt": 3.5,
                "sand": 5.0, "grass": 5.0, "unpaved": 3.0,
            },
            "smoothness": {"intermediate": 1.3, "bad": 2.0, "very_bad": 3.0, "horrible": 5.0},
            "multipliers": {"narrow": 1.5},
            "defaults": {"surface": 1.6},
        }),
    },
    "nature": {
        "label": "парки и грунт",
        "description": "Гревел и прогулка: тропы, парки и грунт в плюс, крупные дороги в минус.",
        "config": _with_overrides(DEFAULT_COMFORT_CONFIG, {
            "highway": {
                "path": 0.55, "track": 0.65, "cycleway": 0.6, "footway": 0.9, "pedestrian": 0.9,
                "tertiary": 1.5, "secondary": 2.2, "primary": 3.0,
            },
            "surface": {
                "compacted": 0.9, "fine_gravel": 0.9, "gravel": 1.0, "ground": 1.0, "dirt": 1.05,
                "unpaved": 1.0, "grass": 1.3, "paving_stones": 1.0,
            },
            "multipliers": {"unlit": 1.0},
            "defaults": {"surface": 1.0},
        }),
    },
}
# Presets without their own "turns" use the ordinary penalties.
for _preset in COMFORT_PRESETS.values():
    _preset.setdefault("turns", TurnConfig())
