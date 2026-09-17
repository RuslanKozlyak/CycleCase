"""Stage 2 - from the edge table to a routing graph.

    edges --filter_edges(edges, keep)--> edges --build_graph(edges, nodes, cost)--> nx.MultiDiGraph

``keep`` is a pandas mask or a function ``keep(edges) -> mask``; ``drop_values`` builds a simple one.
``check_connectivity`` and ``build_graph`` warn when the network falls apart into pieces that no route can cross.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Iterable
import functools
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import pandas as pd

from .comfort import default_comfort
from .config import DEFAULT_COMFORT_CONFIG, with_overrides
from .tags import has_any


WEIGHT = "cost"  # routing weight of a graph edge, metres

MOTORWAY_HIGHWAY = {"motorway", "motorway_link", "trunk", "trunk_link"}
OTHER_FORBIDDEN_HIGHWAY = {
    "construction", "proposed", "planned", "razed", "abandoned", "platform", "elevator", "escalator",
    "corridor", "bus_guideway", "raceway", "busway",
}
FORBIDDEN_HIGHWAY = MOTORWAY_HIGHWAY | OTHER_FORBIDDEN_HIGHWAY | {"steps"}
CLOSED_ACCESS = {"private", "no", "customers", "permit", "delivery"}
BICYCLE_FORBIDDEN = {"no", "dismount", "private"}
BICYCLE_ALLOWED = {"yes", "designated", "permissive"}
RIDEABILITY_REASON_LABELS = {
    "steps": "лестницы",
    "motorway": "автомагистрали",
    "closed_access": "закрытый доступ",
    "bicycle_forbidden": "bicycle=no/dismount",
    "other_forbidden": "стройки, платформы, лифты",
}


class ConnectivityWarning(UserWarning):
    """The edges fall apart into pieces that no route can cross."""


def rideability_issues(edges: pd.DataFrame) -> pd.DataFrame:
    """Why a bicycle may not ride each edge: one True/False column per reason (reasons may overlap).

    An explicit ``bicycle=yes/designated/permissive`` overrides only a closed ``access``/``service``.
    """

    return pd.DataFrame({
        "steps": has_any(edges["highway"], "steps"),
        "motorway": has_any(edges["highway"], MOTORWAY_HIGHWAY),
        "closed_access": (
            (has_any(edges["access"], CLOSED_ACCESS) | has_any(edges["service"], "private"))
            & ~has_any(edges["bicycle"], BICYCLE_ALLOWED)
        ),
        "bicycle_forbidden": has_any(edges["bicycle"], BICYCLE_FORBIDDEN),
        "other_forbidden": has_any(edges["highway"], OTHER_FORBIDDEN_HIGHWAY),
    }, index=edges.index)


def is_rideable(edges: pd.DataFrame, *, allow: Iterable[str] = ()) -> pd.Series:
    """Example filter mask: True where a bicycle may ride at all; ``allow`` lifts some reasons.

    Drops roads closed to bicycles (motorway, trunk, ``bicycle=no``), steps, private and service-only
    access. Everything merely uncomfortable (sidewalks, tracks, cobblestones) stays: comfort weights
    deal with it.
    """

    return ~rideability_issues(edges).drop(columns=list(allow)).any(axis=1)


def drop_values(edges: pd.DataFrame, **columns: Iterable[str] | str) -> pd.Series:
    """Mask that is False on edges having any of the given values in a column, True elsewhere.

    ``drop_values(edges, surface=["sand"])``; masks combine with ``&`` and ``|``.
    """

    dropped = pd.Series(False, index=edges.index)
    for column, values in columns.items():
        dropped |= has_any(edges[column], values)
    return ~dropped


# --- connectivity ----------------------------------------------------------------------------


def _pieces(edges: pd.DataFrame, *, strongly: bool = False) -> list[set]:
    graph = nx.from_pandas_edgelist(edges, "u", "v", create_using=nx.DiGraph)
    pieces = nx.strongly_connected_components(graph) if strongly else nx.weakly_connected_components(graph)
    return sorted(pieces, key=len, reverse=True)


def largest_component(edges: pd.DataFrame, *, strongly: bool = False) -> pd.Series:
    """Mask of the edges inside the largest connected piece (by nodes); strong pieces respect one-ways."""

    largest = next(iter(_pieces(edges, strongly=strongly)), set())
    return edges["u"].isin(largest) & edges["v"].isin(largest)


def graph_order(edges: pd.DataFrame) -> pd.DataFrame:
    """Rows in the order ``build_graph(edges, ...).edges`` iterates them.

    A graph lists edges node by node, nodes in order of first appearance. Matching observations in this
    order keeps tie-breaking identical to the graph-based pipeline this project started from.
    """

    ends = np.column_stack([edges["u"].to_numpy(), edges["v"].to_numpy()]).ravel()
    node_rank = pd.factorize(ends)[0][0::2]
    pair_rank = pd.factorize(pd.MultiIndex.from_arrays([edges["u"], edges["v"]]))[0]
    return edges.iloc[np.lexsort((pair_rank, node_rank))]


def restore_connectors(edges: pd.DataFrame, mask: pd.Series) -> pd.Series:
    """Mask of rejected real OSM edges needed to reconnect the kept weak components.

    Kept edges have zero cost and rejected edges cost their physical length in a minimum spanning
    forest. Non-terminal leaves are pruned, so rejected cul-de-sacs are not restored. All available
    directions of a selected street segment are returned. Components already disconnected in the raw
    OSM table cannot be joined without inventing geometry and therefore remain separate.
    """

    connectors = pd.Series(False, index=edges.index)
    if mask.all() or not mask.any():
        return connectors

    topology = nx.Graph()
    pair_positions: dict[frozenset, list[int]] = {}
    pair_kept: dict[frozenset, bool] = {}
    pair_length: dict[frozenset, float] = {}
    for position, (u, v, kept, length) in enumerate(zip(edges["u"], edges["v"], mask, edges["length_m"])):
        if u == v:
            continue
        pair = frozenset((u, v))
        pair_positions.setdefault(pair, []).append(position)
        pair_kept[pair] = pair_kept.get(pair, False) or bool(kept)
        value = float(length) if pd.notna(length) and float(length) > 0 else 1.0
        pair_length[pair] = min(pair_length.get(pair, value), value)

    for pair in pair_positions:
        u, v = tuple(pair)
        kept = pair_kept[pair]
        topology.add_edge(u, v, weight=0.0 if kept else pair_length[pair], kept=kept, pair=pair)

    terminals = set(edges.loc[mask, "u"]) | set(edges.loc[mask, "v"])
    selected: set[frozenset] = set()
    for component in nx.connected_components(topology):
        component_terminals = terminals.intersection(component)
        if len(component_terminals) < 2:
            continue
        tree = nx.minimum_spanning_tree(topology.subgraph(component), weight="weight")
        leaves = deque(node for node in tree if tree.degree(node) <= 1 and node not in component_terminals)
        while leaves:
            node = leaves.popleft()
            if node not in tree or tree.degree(node) > 1 or node in component_terminals:
                continue
            neighbours = list(tree.neighbors(node))
            tree.remove_node(node)
            for neighbour in neighbours:
                if tree.degree(neighbour) <= 1 and neighbour not in component_terminals:
                    leaves.append(neighbour)
        selected.update(data["pair"] for _, _, data in tree.edges(data=True) if not data["kept"])

    restored_positions = [position for pair in selected for position in pair_positions[pair]]
    if restored_positions:
        connectors.iloc[restored_positions] = True
    return connectors


def filter_edges(
    edges: gpd.GeoDataFrame,
    keep,
    *,
    connect_components: bool = False,
    keep_largest: bool = False,
    report: bool = True,
) -> gpd.GeoDataFrame:
    """``edges[keep]``, where ``keep`` is a True/False mask or a function ``keep(edges) -> mask``.

    ``connect_components=True`` brings back the rejected OSM edges that reconnect the kept pieces;
    ``keep_largest=True`` drops everything outside the largest piece.
    """

    mask = keep(edges) if callable(keep) else keep
    filtered = int((~mask).sum())
    if connect_components:
        mask = mask | restore_connectors(edges, mask)
    kept = edges[mask]
    if keep_largest:
        kept = kept[largest_component(kept)]
    if report:
        km = (edges["length_m"].sum() - kept["length_m"].sum()) / 1000
        print(
            f"Маска убрала {_number(filtered)} рёбер; после связности осталось {_number(len(kept))} "
            f"из {_number(len(edges))} (убрано {_number(km, 1)} км)."
        )
    return kept.copy()


def check_connectivity(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame | None = None,
    *,
    keep_largest: bool = False,
    report: bool = True,
) -> gpd.GeoDataFrame:
    """Warn when the edges fall apart into pieces; ``keep_largest=True`` keeps only the largest one."""

    pieces = len(_pieces(edges))
    largest = largest_component(edges)
    if report:
        outside = edges[~largest]
        print(
            f"Связных частей: {_number(pieces)}. Вне крупнейшей {_number(len(outside))} рёбер "
            f"({_number(outside['length_m'].sum() / 1000, 1)} км)."
        )
    if pieces > 1:
        warnings.warn(
            f"Сеть распалась на {pieces} частей; между ними маршрут не построится. "
            "keep_largest=True оставит крупнейшую.",
            ConnectivityWarning,
            stacklevel=2,
        )
    return (edges[largest] if keep_largest else edges).copy()


def graph_cache_path(directory, city, config):
    """Compatibility import for the raw GraphML cache path now owned by preparation."""

    from .preparation import graph_cache_path as _graph_cache_path

    return _graph_cache_path(directory, city, config)


def _number(value: float, digits: int = 0) -> str:
    return f"{value:,.{digits}f}".replace(",", " ")


# --- graph -----------------------------------------------------------------------------------


def edge_costs(
    edges: gpd.GeoDataFrame,
    cost,
) -> pd.Series:
    """Routing cost of every edge, checked.

    ``cost`` is a function ``cost(edges) -> Series``, weights overriding ``DEFAULT_COMFORT_CONFIG``
    (``{"highway": {"primary": 3.0}}``), a preset name (``"quiet"``), a column name or a ready Series.
    """

    if isinstance(cost, dict):
        cost = functools.partial(default_comfort, config=with_overrides(DEFAULT_COMFORT_CONFIG, cost))
    elif isinstance(cost, str) and cost not in edges:
        from .presets import COMFORT_PRESETS  # presets import routing, which imports this module

        cost = COMFORT_PRESETS[cost]["cost"]
    values = edges[cost] if isinstance(cost, str) else cost(edges) if callable(cost) else cost
    if not isinstance(values, pd.Series):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(edges),):
            raise ValueError(f"Стоимость: ожидалось {len(edges)} значений, получено {values.shape}")
        values = pd.Series(values, index=edges.index)
    if len(values) != len(edges):
        raise ValueError(f"Стоимость: ожидалось {len(edges)} значений, получено {len(values)}")
    if not values.index.equals(edges.index):
        raise ValueError("Стоимость: индекс Series должен совпадать с индексом edges")
    values = values.astype(float)
    problems = {
        "NaN": values.isna(),
        "бесконечная": np.isinf(values),
        "не положительная": values <= 0,
    }
    for problem, bad in problems.items():
        if bad.any():
            example = edges.loc[bad, ["u", "v", "key"]].iloc[0].tolist()
            raise ValueError(f"Стоимость {problem} у {int(bad.sum())} рёбер, например у {tuple(example)}")
    return values.rename(WEIGHT)


def build_graph(
    edges: gpd.GeoDataFrame,
    nodes: gpd.GeoDataFrame,
    cost,
) -> nx.MultiDiGraph:
    """Routing graph from the edge table; the edge weight is ``"cost"``, in metres.

    ``cost`` is anything ``edge_costs`` accepts: a comfort function, weights, a preset name, a column
    name (``"length_m"`` for the shortest path) or a ready Series. Graph edges get only ``cost``, ``length_m`` and
    ``geometry``, never the tags; nodes get ``x`` and ``y``. Warns when the graph has several pieces.
    """

    weights = edge_costs(edges, cost)
    graph = nx.MultiDiGraph(crs=edges.crs)
    graph.add_edges_from(zip(
        edges["u"].tolist(),
        edges["v"].tolist(),
        edges["key"].tolist(),
        (
            {WEIGHT: weight, "length_m": length, "geometry": geometry}
            for weight, length, geometry in zip(weights.tolist(), edges["length_m"].tolist(), edges.geometry.array)
        ),
    ))
    coordinates = nodes.set_index("node")[["x", "y"]]
    order = pd.Index(list(graph.nodes))
    missing = order.difference(coordinates.index)
    if len(missing):
        raise ValueError(f"В nodes нет {len(missing)} узлов из edges, например {missing[0]}")
    coordinates = coordinates.loc[order]
    for node, x, y in zip(order.tolist(), coordinates["x"].tolist(), coordinates["y"].tolist()):
        attributes = graph.nodes[node]
        attributes["x"], attributes["y"] = x, y

    pieces = nx.number_weakly_connected_components(graph)
    if pieces > 1:
        warnings.warn(
            f"Граф состоит из {pieces} несвязанных частей: между ними маршрута нет. "
            "filter_edges(..., keep_largest=True) оставит крупнейшую.",
            ConnectivityWarning,
            stacklevel=2,
        )
    return graph
