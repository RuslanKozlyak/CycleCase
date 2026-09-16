"""Bicycle comfort routing, split by pipeline stage.

data -> graph -> matching -> comfort + routing -> evaluation -> visualization
"""

from .comfort import add_comfort_cost, comfort_feature_table
from .config import COMFORT_PRESETS, DEFAULT_COMFORT_CONFIG, OSM_FEATURE_TAGS, GraphConfig, MatchConfig, TurnConfig
from .data import load_clean_personal_gpx, load_stadtradeln
from .evaluation import (
    compare_route_models,
    composition_table,
    evaluate_corridor_popularity,
    route_edge_association,
    route_edges_long,
)
from .graph import filter_edges, is_rideable, load_graph
from .matching import match_stadtradeln_to_graph, snap_tracks_to_edges, track_edge_usage
from .routing import (
    build_routes,
    count_turns,
    dijkstra,
    dijkstra_turns,
    edge_bearings,
    nodes_near_observations,
    sample_od_pairs,
    turn_kind,
)
