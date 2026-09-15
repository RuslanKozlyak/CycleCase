"""Tools for bicycle comfort routing and observed-route evaluation."""

from .comfort import add_comfort_cost, comfort_components, comfort_feature_table
from .config import (
    DEFAULT_COMFORT_CONFIG,
    EXTRA_OSM_TAGS,
    GraphConfig,
    MatchConfig,
    TurnConfig,
)
from .datasets import download_file, load_stadtradeln
from .evaluation import (
    compare_algorithms,
    compare_route_models,
    composition_table,
    evaluate_corridor_popularity,
    evaluate_personal_routes,
    route_edge_association,
    route_edges_long,
    route_overlap,
    sample_geometry,
    snap_tracks_to_edges,
    summarize_personal_metrics,
    track_edge_usage,
)
from .gpx import geodesic_length_m, load_clean_personal_gpx, load_gpx_tracks
from .matching import match_stadtradeln_to_graph
from .optimization import (
    DEFAULT_COMFORT_PARAMETERS,
    ComfortOptimizationConfig,
    ComfortOptimizationResult,
    ComfortParameter,
    comfort_config_with_parameters,
    optimize_comfort_weights,
    score_comfort_config,
    spatial_od_split,
)
from .osm import (
    GraphDownloadError,
    configure_osmnx,
    graph_build_summary,
    graph_cache_path,
    load_or_download_graph,
)
from .routing import build_routes, nodes_near_observations, route_record, sample_od_pairs

__all__ = [
    "DEFAULT_COMFORT_CONFIG",
    "EXTRA_OSM_TAGS",
    "GraphConfig",
    "GraphDownloadError",
    "MatchConfig",
    "TurnConfig",
    "ComfortOptimizationConfig",
    "ComfortOptimizationResult",
    "ComfortParameter",
    "DEFAULT_COMFORT_PARAMETERS",
    "add_comfort_cost",
    "build_routes",
    "compare_algorithms",
    "compare_route_models",
    "comfort_config_with_parameters",
    "configure_osmnx",
    "download_file",
    "comfort_components",
    "comfort_feature_table",
    "composition_table",
    "route_edge_association",
    "route_edges_long",
    "snap_tracks_to_edges",
    "track_edge_usage",
    "evaluate_corridor_popularity",
    "evaluate_personal_routes",
    "geodesic_length_m",
    "graph_build_summary",
    "graph_cache_path",
    "load_clean_personal_gpx",
    "load_gpx_tracks",
    "load_or_download_graph",
    "load_stadtradeln",
    "match_stadtradeln_to_graph",
    "nodes_near_observations",
    "optimize_comfort_weights",
    "route_overlap",
    "route_record",
    "sample_geometry",
    "sample_od_pairs",
    "score_comfort_config",
    "spatial_od_split",
    "summarize_personal_metrics",
]
