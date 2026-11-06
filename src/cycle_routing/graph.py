from __future__ import annotations

import warnings

import geopandas as gpd

import networkx as nx

import numpy as np

import pandas as pd

WEIGHT = "cost"  # routing weight of a graph edge, metres

class ConnectivityWarning(UserWarning):
    pass

def build_graph(edges: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, cost) -> nx.MultiDiGraph:
    weights = edge_costs(edges, cost)
    graph = nx.MultiDiGraph(crs=edges.crs)
    graph.add_edges_from(zip(edges['u'].tolist(), edges['v'].tolist(), edges['key'].tolist(), ({WEIGHT: weight, 'length_m': length, 'geometry': geometry} for weight, length, geometry in zip(weights.tolist(), edges['length_m'].tolist(), edges.geometry.array))))
    coordinates = nodes.set_index('node')[['x', 'y']]
    order = pd.Index(list(graph.nodes))
    missing = order.difference(coordinates.index)
    if len(missing):
        raise ValueError(f'В nodes нет {len(missing)} узлов из edges, например {missing[0]}')
    coordinates = coordinates.loc[order]
    for node, x, y in zip(order.tolist(), coordinates['x'].tolist(), coordinates['y'].tolist()):
        attributes = graph.nodes[node]
        attributes['x'], attributes['y'] = (x, y)
    pieces = nx.number_weakly_connected_components(graph)
    if pieces > 1:
        warnings.warn(f'Граф состоит из {pieces} несвязанных частей: между ними маршрута нет. Маршруты возможны только внутри связных частей.', ConnectivityWarning, stacklevel=2)
    return graph

def edge_costs(edges, cost):
    values = edges[cost] if isinstance(cost, str) else cost(edges) if callable(cost) else cost
    if not isinstance(values, pd.Series):
        values = np.asarray(values, dtype=float)
        if values.shape != (len(edges),):
            raise ValueError("Стоимость должна содержать одно значение на дорогу")
        values = pd.Series(values, index=edges.index)
    if not values.index.equals(edges.index):
        raise ValueError("Индекс стоимости должен совпадать с индексом дорог")
    values = values.astype(float)
    if not np.isfinite(values).all() or values.le(0).any():
        raise ValueError("Стоимость должна быть положительной и конечной")
    return values.rename(WEIGHT)
