"""A toy city that goes through the real prepare_city: OSM download and observations are faked."""

import geopandas as gpd
import networkx as nx
from pyproj import Transformer
import pytest
from shapely.geometry import LineString

from cycle_routing import preparation
from cycle_routing.config import GraphConfig
from cycle_routing.preparation import AddressPair, City, GpxSource, StadtradelnSource

CRS = "EPSG:25832"
LON, LAT = 9.17, 47.66
# Metres east/north of (LON, LAT). North road 1-2-4 is a cycleway, south road 1-3-4 a slightly shorter
# primary, 1-5-4 goes down steps; 6 hangs on a private driveway.
POINTS = {1: (0, 0), 2: (100, 60), 3: (100, -55), 4: (200, 0), 5: (100, 0), 6: (0, -80)}
WAYS = [  # (u, v, osmid, tags)
    (1, 2, 10, {"highway": "cycleway", "surface": "asphalt"}),
    (2, 4, 11, {"highway": ["cycleway", "path"], "surface": "asphalt", "lit": "yes"}),
    (1, 3, 20, {"highway": "primary", "surface": "asphalt", "maxspeed": "60", "lanes": "4", "cycleway:right": "no"}),
    (3, 4, 21, {"highway": "primary", "surface": ["asphalt", "sett"], "maxspeed": ["50", "70"]}),
    (1, 5, 30, {"highway": "steps"}),
    (5, 4, 31, {"highway": "footway", "width": "1.5"}),
    (1, 6, 40, {"highway": "service", "service": "driveway", "access": "private"}),
]
TRIPS = {10: 120, 11: 100, 20: 5, 21: 8}


def _to_lonlat():
    to_utm = Transformer.from_crs("EPSG:4326", CRS, always_xy=True)
    x0, y0 = to_utm.transform(LON, LAT)
    back = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
    return x0, y0, {node: back.transform(x0 + dx, y0 + dy) for node, (dx, dy) in POINTS.items()}


def raw_graph() -> nx.MultiDiGraph:
    """OSMnx-like unprojected graph: two-way streets, list-valued tags on simplified edges."""

    x0, y0, lonlat = _to_lonlat()
    graph = nx.MultiDiGraph(crs="epsg:4326", simplified=True)
    for node, (lon, lat) in lonlat.items():
        graph.add_node(node, x=lon, y=lat, street_count=2)
    for u, v, osmid, tags in WAYS:
        length = ((POINTS[u][0] - POINTS[v][0]) ** 2 + (POINTS[u][1] - POINTS[v][1]) ** 2) ** 0.5
        for a, b, reverse in ((u, v, False), (v, u, True)):
            geometry = LineString([lonlat[a], lonlat[b]])
            graph.add_edge(a, b, osmid=osmid, length=length, oneway=False, reversed=reverse, geometry=geometry, **tags)
    return graph


def observed_segments() -> gpd.GeoDataFrame:
    x0, y0, _ = _to_lonlat()
    rows = [(osmid, TRIPS[osmid], u, v) for u, v, osmid, _ in WAYS if osmid in TRIPS]
    return gpd.GeoDataFrame(
        {"osm_way_id": [row[0] for row in rows], "number_of_matched_trips": [row[1] for row in rows]},
        geometry=[
            LineString([(x0 + POINTS[u][0], y0 + POINTS[u][1]), (x0 + POINTS[v][0], y0 + POINTS[v][1])])
            for _, _, u, v in rows
        ],
        crs=CRS,
    )


def gpx_tracks() -> gpd.GeoDataFrame:
    """Two rides along the north road; they start and end off the junctions, so no ties at nodes."""

    x0, y0, _ = _to_lonlat()
    north = LineString([(x0 + dx, y0 + dy) for dx, dy in ((10, 6), POINTS[2], (190, 6))])
    return gpd.GeoDataFrame(
        {"activity_id": ["t1", "t2"], "file": ["t1.gpx", "t2.gpx"]}, geometry=[north, north], crs=CRS
    ).to_crs("EPSG:4326")


def _register(monkeypatch, name: str, observations) -> None:
    _, _, lonlat = _to_lonlat()
    city = City(
        crs=CRS,
        observations=observations,
        od=(AddressPair("A", start=lonlat[1][::-1], end=lonlat[4][::-1]),),
        bbox=(LON - 0.01, LAT - 0.01, LON + 0.01, LAT + 0.01),
        graph=GraphConfig(retain_all=True),
    )
    monkeypatch.setitem(preparation.CITIES, name, city)


@pytest.fixture
def toy_city(tmp_path, monkeypatch):
    """(name, cache_dir) of a STADTRADELN-like city; downloads are counted in ``downloads``."""

    downloads = []

    def download(**kwargs):
        downloads.append(kwargs)
        return raw_graph()

    monkeypatch.setattr(preparation, "_download_graph", download)
    monkeypatch.setattr(preparation, "load_stadtradeln", lambda *args, **kwargs: observed_segments())
    monkeypatch.setattr(preparation, "load_clean_personal_gpx", lambda *args, **kwargs: gpx_tracks())
    _register(monkeypatch, "toytown", StadtradelnSource(url="-", archive_name="-", popular_quantile=0.5))
    _register(monkeypatch, "toytracks", GpxSource(directory="-", bounds=(0, 0, 90, 90), popular_min_trips=1))
    return {"cache_dir": tmp_path / "cities", "downloads": downloads}
