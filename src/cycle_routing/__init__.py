"""Bicycle comfort routing on a flat edge table.

    prepare_city -> edges, nodes -> filter_edges -> build_graph(edges, nodes, cost)
                 -> find_route / build_routes -> score

Students write a filter mask, a comfort function, and optionally a turn model and a search algorithm
(any NetworkX path function such as ``nx.dijkstra_path`` or ``nx.astar_path``).
"""

from .comfort import comfort_components, comfort_factors, comfort_feature_table, default_comfort
from .config import DEFAULT_COMFORT_CONFIG, TurnConfig, with_overrides
from .evaluation import score
from .graph import (
    ConnectivityWarning,
    build_graph,
    check_connectivity,
    drop_values,
    edge_costs,
    filter_edges,
    is_rideable,
    largest_component,
    restore_connectors,
    rideability_issues,
)
from .preparation import CITIES, EDGE_COLUMNS, load_od_pairs, make_edges, make_nodes, prepare_city, preparation_report
from .presets import COMFORT_PRESETS, cost_overview
from .routing import SEARCH, TURNS, build_routes, count_turns, edge_bearings, find_route, turn_graph, turn_kind, turn_penalty
from .tags import has_any
