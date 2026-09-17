"""Ready-made rider profiles: a comfort function, a turn model and a search algorithm each.

A preset is exactly what a student solution is; every field is either plain data or code:

    {"label": ..., "description": ...,
     "cost": {"highway": {"primary": 3.0}} | "quiet" | cost(edges) -> Series,
     "turns": "default" | {"left_penalty_m": 60} | TurnConfig | penalty(graph, incoming, outgoing) | None,
     "search": "dijkstra" | nx.astar_path | own path function}

and goes through the same public calls: ``build_graph(edges, nodes, cost=preset["cost"])`` and
``find_route(graph, ..., search=preset["search"], turns=preset["turns"])``.
"""

from __future__ import annotations

import networkx as nx
import pandas as pd
from shapely.geometry import LineString

from .comfort import default_comfort
from .config import DEFAULT_COMFORT_CONFIG, TurnConfig, with_overrides
from .graph import edge_costs
from .preparation import make_edges
from .routing import TURNS, turn_model


class _Preset(dict):
    """Preset with the documented keys plus a legacy ``config`` lookup.

    Iteration exposes only ``label``, ``description``, ``cost``, ``turns`` and ``search``.  Older
    notebooks can still read ``preset["config"]`` while the tabular pipeline uses only ``cost``.
    """

    def __init__(self, values, config):
        super().__init__(values)
        self._config = config

    def __getitem__(self, key):
        return self._config if key == "config" else super().__getitem__(key)

    def get(self, key, default=None):
        return self._config if key == "config" else super().get(key, default)


FAST_CONFIG = with_overrides(DEFAULT_COMFORT_CONFIG, {
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
AVENUES_CONFIG = with_overrides(DEFAULT_COMFORT_CONFIG, {
    "highway": {"secondary": 1.0, "primary": 1.0},
    "multipliers": {"dedicated_cycleway": 0.5},
})
QUIET_CONFIG = with_overrides(DEFAULT_COMFORT_CONFIG, {
    "highway": {
        "cycleway": 0.5, "living_street": 0.6, "pedestrian": 0.8, "path": 0.75, "footway": 1.1,
        "service": 1.3, "unclassified": 1.6, "tertiary": 2.2, "secondary": 3.2, "primary": 4.5, "trunk": 7.0,
    },
    # A painted lane does not make a busy road calm.
    "multipliers": {"dedicated_cycleway": 1.0, "many_lanes": 1.8},
    "maxspeed": {"threshold_kmh": 30.0, "scale_kmh": 40.0, "max_extra": 1.0},
})
BIKE_INFRA_CONFIG = with_overrides(DEFAULT_COMFORT_CONFIG, {
    "highway": {
        "cycleway": 0.3, "living_street": 1.1, "service": 1.5, "unclassified": 1.4, "tertiary": 1.6,
        "path": 1.2, "track": 1.6, "footway": 2.0, "pedestrian": 1.8,
    },
    "multipliers": {"dedicated_cycleway": 0.35},
    "bicycle": {"designated": 0.6, "use_sidepath": 2.5},
    "min_factor": 0.25,
})
SMOOTH_CONFIG = with_overrides(DEFAULT_COMFORT_CONFIG, {
    "highway": {"path": 1.5, "track": 2.2, "footway": 1.6, "pedestrian": 1.4, "secondary": 1.3, "primary": 1.6},
    "surface": {
        "paved": 1.05, "concrete": 1.1, "paving_stones": 1.6, "sett": 2.8, "cobblestone": 3.2,
        "compacted": 2.0, "fine_gravel": 2.2, "gravel": 3.0, "ground": 3.5, "dirt": 3.5,
        "sand": 5.0, "grass": 5.0, "unpaved": 3.0,
    },
    "smoothness": {"intermediate": 1.3, "bad": 2.0, "very_bad": 3.0, "horrible": 5.0},
    "multipliers": {"narrow": 1.5},
    "defaults": {"surface": 1.6},
})
NATURE_CONFIG = with_overrides(DEFAULT_COMFORT_CONFIG, {
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
})


def balanced_comfort(edges):
    return default_comfort(edges, DEFAULT_COMFORT_CONFIG)


def fast_comfort(edges):
    return default_comfort(edges, FAST_CONFIG)


def avenues_comfort(edges):
    return default_comfort(edges, AVENUES_CONFIG)


def quiet_comfort(edges):
    return default_comfort(edges, QUIET_CONFIG)


def bike_infra_comfort(edges):
    return default_comfort(edges, BIKE_INFRA_CONFIG)


def smooth_comfort(edges):
    return default_comfort(edges, SMOOTH_CONFIG)


def nature_comfort(edges):
    return default_comfort(edges, NATURE_CONFIG)


# Each preset exaggerates one preference so that routes visibly differ.
COMFORT_PRESETS = {
    "balanced": {
        "label": "сбалансированный",
        "description": "Исходные веса: умеренный штраф крупных дорог и плохого покрытия.",
        "cost": balanced_comfort,
        "config": DEFAULT_COMFORT_CONFIG,
        "turns": TurnConfig(),
        "search": nx.dijkstra_path,
    },
    "fast": {
        "label": "быстрый",
        "description": "Почти кратчайший путь: тип дороги и покрытие почти не штрафуются.",
        "cost": fast_comfort,
        "config": FAST_CONFIG,
        "turns": TurnConfig(),
        "search": nx.dijkstra_path,
    },
    "min_turns": {
        "label": "минимум поворотов",
        "description": "Длинные прямые участки: веса как у быстрого, поворот стоит в 10 раз дороже обычного.",
        "cost": fast_comfort,
        "config": FAST_CONFIG,
        "turns": TURNS["min_turns"],
        "search": nx.dijkstra_path,
    },
    "avenues": {
        "label": "по проспектам",
        "description": (
            "Как в ваших треках по СПб: крупные улицы без штрафа, заметный бонус за автобусные "
            "и велополосы на них (лучший вариант перебора весов по GPX)."
        ),
        "cost": avenues_comfort,
        "config": AVENUES_CONFIG,
        "turns": TurnConfig(),
        "search": nx.dijkstra_path,
    },
    "quiet": {
        "label": "спокойный",
        "description": "Подальше от машин: штраф улиц с движением и скоростью выше 30 км/ч, бонус жилых зон и парков.",
        "cost": quiet_comfort,
        "config": QUIET_CONFIG,
        "turns": TurnConfig(),
        "search": nx.dijkstra_path,
    },
    "bike_infra": {
        "label": "велоинфраструктура",
        "description": "Только велодорожки и велополосы, остальные улицы и тротуары заметно дороже.",
        "cost": bike_infra_comfort,
        "config": BIKE_INFRA_CONFIG,
        "turns": TurnConfig(),
        "search": nx.dijkstra_path,
    },
    "smooth": {
        "label": "гладкий асфальт",
        "description": "Шоссейный велосипед: сильный штраф брусчатки, плитки, грунта и неизвестного покрытия.",
        "cost": smooth_comfort,
        "config": SMOOTH_CONFIG,
        "turns": TurnConfig(),
        "search": nx.dijkstra_path,
    },
    "nature": {
        "label": "парки и грунт",
        "description": "Гревел и прогулка: тропы, парки и грунт в плюс, крупные дороги в минус.",
        "cost": nature_comfort,
        "config": NATURE_CONFIG,
        "turns": TurnConfig(),
        "search": nx.dijkstra_path,
    },
}

COMFORT_PRESETS = {
    name: _Preset({key: value for key, value in preset.items() if key != "config"}, preset["config"])
    for name, preset in COMFORT_PRESETS.items()
}
for preset in COMFORT_PRESETS.values():
    preset["cost"].config = preset["config"]


# Typical edges for comparing comfort functions; 100 m each.
TYPICAL_EDGES = {
    "primary": {"highway": "primary", "surface": "asphalt", "maxspeed": 60.0, "lanes": 4.0},
    "primary + велополоса": {"highway": "primary", "surface": "asphalt", "cycleway": "lane", "maxspeed": 60.0},
    "secondary": {"highway": "secondary", "surface": "asphalt"},
    "жилая улица": {"highway": "residential", "surface": "asphalt"},
    "велодорожка": {"highway": "cycleway", "surface": "asphalt"},
    "тротуар": {"highway": "footway", "surface": "paving_stones"},
    "брусчатка": {"highway": "residential", "surface": "sett"},
    "грунтовка": {"highway": "track", "surface": "gravel"},
}


def cost_overview(presets: dict[str, dict], examples: dict[str, dict] = TYPICAL_EDGES) -> pd.DataFrame:
    """Cost of one metre of typical edges under each preset, plus its left-turn penalty.

    Works for any comfort function: it only calls ``preset["cost"]`` on a small edge table.
    """

    edges = make_edges(
        [
            {"u": index, "v": index + 1, "geometry": LineString([(0, 0), (100, 0)]), **tags}
            for index, tags in enumerate(examples.values())
        ],
        crs="EPSG:3857",
    )
    rows = {}
    for name, preset in presets.items():
        per_metre = edge_costs(edges, preset["cost"]).to_numpy() / edges["length_m"].to_numpy()
        turns = turn_model(preset.get("turns"))
        rows[preset.get("label", name)] = {
            **dict(zip(examples, per_metre.round(3))),
            "поворот налево, м": turns.left_penalty_m if isinstance(turns, TurnConfig) else None,
        }
    return pd.DataFrame(rows).T
