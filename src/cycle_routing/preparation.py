from __future__ import annotations

from collections.abc import Iterable, Mapping

import geopandas as gpd

import networkx as nx

import numpy as np

import osmnx as ox

import pandas as pd

import shapely

from .config import GraphConfig, OSM_WAY_TAGS

from .tags import first_number, join_values, tag_values

TAG_COLUMNS = {
    "highway": ("highway",),
    "surface": ("surface",),
    "smoothness": ("smoothness",),
    "cycleway": ("cycleway", "cycleway:left", "cycleway:right", "cycleway:both"),
    "bicycle": ("bicycle",),
    "lit": ("lit",),
    "service": ("service",),
    "access": ("access",),
}

NUMBER_COLUMNS = ("maxspeed", "lanes", "width")

EDGE_COLUMNS = ("u", "v", "key", "length_m", *TAG_COLUMNS, *NUMBER_COLUMNS, "oneway", "geometry")

NODE_COLUMNS = ("node", "x", "y", "geometry")

class GraphDownloadError(RuntimeError):
    pass

def _download_graph(
    *,
    bbox: tuple[float, float, float, float] | None,
    center: tuple[float, float] | None,
    dist: float | None,
    config: GraphConfig,
    polygon=None,
    progress=True,
) -> nx.MultiDiGraph:
    from .workshop_geo import OSMUnavailableError, _osm_request
    kwargs = config.graph_kwargs()
    def request():
        if polygon is not None:
            return ox.graph_from_polygon(polygon, **kwargs)
        if bbox is not None:
            return ox.graph.graph_from_bbox(bbox, **kwargs)
        if center is None or dist is None:
            raise ValueError("Pass either bbox or both center and dist")
        return ox.graph.graph_from_point(center, dist=dist, **kwargs)
    try:
        return _osm_request(ox, request, urls=config.overpass_urls, progress=progress)
    except OSMUnavailableError as error:
        raise GraphDownloadError(str(error)) from error

def _joined_tags(data: Iterable[Mapping], sources: tuple[str, ...]) -> list[str | None]:
    cache: dict[tuple, str | None] = {}
    cells = []
    for attributes in data:
        raw = tuple(attributes.get(source) for source in sources)
        key = tuple(tuple(value) if isinstance(value, list) else value for value in raw)
        if key not in cache:
            cache[key] = join_values(set().union(*(tag_values(value) for value in raw)))
        cells.append(cache[key])
    return cells

def _flag(value) -> bool:
    return any(value) if isinstance(value, list) else bool(value)

def edges_table(graph: nx.MultiDiGraph, crs) -> tuple[gpd.GeoDataFrame, pd.Series]:

    u, v, key, data = zip(*graph.edges(keys=True, data=True))
    columns: dict[str, object] = {
        "u": np.array(u, dtype="int64"),
        "v": np.array(v, dtype="int64"),
        "key": np.array(key, dtype="int64"),
        "length_m": np.array([attributes["length"] for attributes in data], dtype="float64"),
    }
    for column, sources in TAG_COLUMNS.items():
        columns[column] = pd.Series(_joined_tags(data, sources), dtype=str)
    for column in NUMBER_COLUMNS:
        values = [first_number(attributes.get(column)) for attributes in data]
        columns[column] = np.array([np.nan if value is None else value for value in values], dtype="float64")
    columns["oneway"] = np.array([_flag(attributes.get("oneway")) for attributes in data], dtype=bool)

    # Same straight-line fill as ox.graph_to_gdfs for edges that OSMnx left without geometry.
    xy = {node: (attributes["x"], attributes["y"]) for node, attributes in graph.nodes(data=True)}
    geometry = [
        attributes["geometry"] if "geometry" in attributes else shapely.LineString([xy[start], xy[end]])
        for start, end, attributes in zip(u, v, data)
    ]
    edges = gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries(geometry, crs=graph.graph["crs"])).to_crs(crs)
    osmids = pd.Series([attributes.get("osmid") for attributes in data], index=edges.index, name="osmid")
    return edges[list(EDGE_COLUMNS)], osmids

def nodes_table(graph: nx.MultiDiGraph, crs) -> gpd.GeoDataFrame:

    ids, data = zip(*graph.nodes(data=True))
    points = gpd.GeoSeries(
        gpd.points_from_xy([attributes["x"] for attributes in data], [attributes["y"] for attributes in data]),
        crs=graph.graph["crs"],
    ).to_crs(crs)
    return gpd.GeoDataFrame(
        {"node": np.array(ids, dtype="int64"), "x": points.x.to_numpy(), "y": points.y.to_numpy()},
        geometry=points,
    )

def validate_edges(edges: gpd.GeoDataFrame) -> None:

    if list(edges.columns) != list(EDGE_COLUMNS):
        raise ValueError(f"edge columns {list(edges.columns)} != {list(EDGE_COLUMNS)}")
    expected = {"u": "int64", "v": "int64", "key": "int64", "length_m": "float64", "oneway": "bool"}
    expected |= dict.fromkeys(NUMBER_COLUMNS, "float64")
    for column, dtype in expected.items():
        if edges[column].dtype != dtype:
            raise ValueError(f"{column}: dtype {edges[column].dtype}, expected {dtype}")
    for column in TAG_COLUMNS:
        if not pd.api.types.is_string_dtype(edges[column]) or edges[column].dtype == object:
            raise ValueError(f"{column}: dtype {edges[column].dtype}, expected str")
    if edges.geometry.name != "geometry" or edges.crs is None or not edges.crs.is_projected:
        raise ValueError("geometry must be the 'geometry' column in a metric CRS")

def make_nodes(coordinates: Mapping[int, tuple[float, float]], *, crs) -> gpd.GeoDataFrame:

    ids = list(coordinates)
    x = [float(coordinates[node][0]) for node in ids]
    y = [float(coordinates[node][1]) for node in ids]
    return gpd.GeoDataFrame(
        {"node": np.array(ids, dtype="int64"), "x": x, "y": y}, geometry=gpd.points_from_xy(x, y), crs=crs
    )

def make_edges(records: Iterable[Mapping], *, crs, nodes: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:

    xy = {} if nodes is None else dict(zip(nodes["node"], zip(nodes["x"], nodes["y"])))
    rows = []
    for record in records:
        row = {"key": 0, **record}
        if row.get("geometry") is None:
            row["geometry"] = shapely.LineString([xy[row["u"]], xy[row["v"]]])
        row.setdefault("length_m", row["geometry"].length)
        rows.append(row)
    frame = pd.DataFrame(rows)

    def column_or(name, default):
        return frame[name] if name in frame else pd.Series(default, index=frame.index)

    columns = {
        "u": frame["u"].astype("int64"),
        "v": frame["v"].astype("int64"),
        "key": frame["key"].astype("int64"),
        "length_m": frame["length_m"].astype("float64"),
    }
    for column in TAG_COLUMNS:
        cells = [join_values(tag_values(value)) for value in column_or(column, None)]
        columns[column] = pd.Series(cells, index=frame.index, dtype=str)
    for column in NUMBER_COLUMNS:
        columns[column] = column_or(column, np.nan).astype("float64")
    columns["oneway"] = column_or("oneway", False).fillna(False).astype(bool)
    return gpd.GeoDataFrame(columns, geometry=gpd.GeoSeries(frame["geometry"], crs=crs))[list(EDGE_COLUMNS)]
