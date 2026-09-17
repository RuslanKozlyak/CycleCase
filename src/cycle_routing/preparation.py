"""Stage 1 - data preparation. Students do not need to look inside.

``prepare_city`` downloads OSM, lays its tags out into a flat edge table, attaches observed trips to a
reference set of edges and caches everything in ``<cache_dir>/<city>/``:

    edges.parquet         the table students work with (EDGE_COLUMNS); nothing derived from trips
    nodes.parquet         node coordinates (NODE_COLUMNS)
    popularity.parquet    u, v, key, trips   - read only by score() and the maps
    observations.parquet  observed segments or GPS tracks - read only by score() and the maps
    od_pairs.parquet      fixed start/finish pairs of the city
    meta.json             cache key and preparation report; written last
    <city>_*.graphml      the raw OSM download

Popularity is kept out of the edge table on purpose: it is what routes are judged by, so a cost
function must not be able to see it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import warnings

import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError
import pandas as pd
from pyproj import Transformer
from requests import RequestException
import shapely

from .config import OSM_WAY_TAGS, GraphConfig, MatchConfig
from .data import load_clean_personal_gpx, load_stadtradeln
from .graph import ConnectivityWarning, build_graph, is_rideable, largest_component
from .matching import match_stadtradeln, snap_tracks_to_edges, track_edge_usage
from .tags import first_number, join_values, tag_values


PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Bump when the tables are built differently: every city cache is rebuilt (the GraphML is kept).
SCHEMA_VERSION = 1

# Edge table column -> OSM tags merged into it (cycleway:* counts as one feature).
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
POPULARITY_COLUMNS = ("u", "v", "key", "trips")
CACHE_FILES = ("edges", "nodes", "popularity", "observations", "od_pairs")
_GEO_FILES = {"edges", "nodes", "observations"}


# --- city descriptions -----------------------------------------------------------------------


@dataclass(frozen=True)
class StadtradelnSource:
    """Aggregated STADTRADELN trips per street segment; score() samples routes against them."""

    url: str
    archive_name: str
    popular_quantile: float = 0.75  # a segment is popular from this quantile of trips
    sample_step_m: float = 25.0  # route points for the corridor metric
    match_distance_m: float = 25.0  # route point -> segment
    match: MatchConfig = MatchConfig()


@dataclass(frozen=True)
class GpxSource:
    """Personal GPX tracks; the popularity of an edge is the number of tracks along it."""

    directory: str  # relative to the project root
    bounds: tuple[float, float, float, float]  # lon/lat box; points outside are dropped
    step_m: float = 25.0  # track points attached to edges
    max_distance_m: float = 25.0
    popular_min_trips: int = 2  # "your street": at least this many tracks


@dataclass(frozen=True)
class RandomPairs:
    """Random start/finish nodes inside the observed area, 1.5-10 km apart by the shortest path."""

    n_pairs: int = 20
    seed: int = 42
    near_observations_m: float = 25.0
    min_length_m: float = 1_500.0
    max_length_m: float = 10_000.0
    prefix: str = "OD"


@dataclass(frozen=True)
class AddressPair:
    """A start/finish given by coordinates (lat, lon); routes start at the nearest node."""

    od_id: str
    start: tuple[float, float]
    end: tuple[float, float]
    start_label: str = "старт"
    end_label: str = "финиш"


@dataclass(frozen=True)
class City:
    """Everything that defines a city's data. Pass ``bbox`` or ``center`` + ``radius_m``."""

    crs: str  # metric CRS of all tables
    observations: StadtradelnSource | GpxSource
    od: RandomPairs | tuple[AddressPair, ...]
    bbox: tuple[float, float, float, float] | None = None  # left, bottom, right, top (degrees)
    center: tuple[float, float] | None = None  # lat, lon
    radius_m: float | None = None
    graph: GraphConfig = field(default_factory=GraphConfig)
    # The reference network keeps only its largest strongly connected part (OD pairs, observations).
    largest_component: bool = False


CITIES = {
    "konstanz": City(
        crs="EPSG:25832",
        # STADTRADELN coverage plus a margin; the top reaches Ludwigshafen/Bodman, the only land link
        # between the Bodanrueck peninsula (Konstanz) and the north shore of the Ueberlinger See.
        bbox=(9.016051299999999, 47.6378767, 9.468499900000001, 47.86),
        # retain_all: the bbox cuts the network into pieces that still carry observed trips.
        graph=GraphConfig(retain_all=True),
        observations=StadtradelnSource(
            url=(
                "https://offenedaten-konstanz.de/sites/default/files/"
                "Verkehrsmengen%202.0_SR%202024_Konstanz_UTM32_je_Wochentag_gesamt.zip"
            ),
            archive_name="stadtradeln_konstanz_2024_traffic_volumes.zip",
        ),
        od=RandomPairs(n_pairs=20, seed=42, prefix="DE"),
    ),
    "saint_petersburg": City(
        crs="EPSG:32636",
        # Covers the personal tracks and both addresses plus 1.2 km.
        center=(59.9756026128307, 30.332670746354687),
        radius_m=6143.895531549872,
        largest_component=True,
        observations=GpxSource(directory="cicle_gpx", bounds=(29.40, 59.55, 30.90, 60.35)),
        od=(
            AddressPair(
                "address",
                start=(60.0063227, 30.3942986),
                end=(59.9438142, 30.2959418),
                start_label="Гражданский проспект, 27 к2",
                end_label="Биржевая линия, 14",
            ),
        ),
    ),
}


# --- raw OSM graph ---------------------------------------------------------------------------


class GraphDownloadError(RuntimeError):
    """Raised when every configured Overpass endpoint rejects a graph query."""


def _graph_fingerprint(config: GraphConfig, area=None) -> str:
    payload = {
        **asdict(config),
        **({"area": list(area)} if area is not None else {}),
        "overpass_urls": None,
        "useful_tags_way": sorted(OSM_WAY_TAGS),
        "schema": 2,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:10]


def graph_cache_path(directory: Path, city: str, config: GraphConfig, area=None) -> Path:
    """Return a cache name that changes when graph semantics or saved tags change."""

    safe_city = "".join(char.lower() if char.isalnum() else "_" for char in city).strip("_")
    return directory / (
        f"{safe_city}_{config.network_type}_simp{int(config.simplify)}_"
        f"{_graph_fingerprint(config, area)}.graphml"
    )


def _download_graph(
    *,
    bbox: tuple[float, float, float, float] | None,
    center: tuple[float, float] | None,
    dist: float | None,
    config: GraphConfig,
) -> nx.MultiDiGraph:
    kwargs = config.graph_kwargs()
    failures: list[str] = []
    original_url = ox.settings.overpass_url
    try:
        for endpoint in config.overpass_urls:
            ox.settings.overpass_url = endpoint.rstrip("/")
            try:
                if bbox is not None:
                    return ox.graph.graph_from_bbox(bbox, **kwargs)
                if center is None or dist is None:
                    raise ValueError("Pass either bbox or both center and dist")
                return ox.graph.graph_from_point(center, dist=dist, **kwargs)
            except (RequestException, InsufficientResponseError, ResponseStatusCodeError) as exc:
                failures.append(f"{endpoint}: {type(exc).__name__}: {exc}")
    finally:
        ox.settings.overpass_url = original_url

    detail = "\n".join(f"- {failure}" for failure in failures)
    raise GraphDownloadError(
        "Не удалось скачать OSM-граф ни с одного Overpass endpoint. "
        "Это сетевая ошибка до этапа simplify, а не ошибка геометрии.\n"
        f"{detail}"
    )


def load_raw_graph(folder: Path, city: str, *, refresh: bool = False) -> nx.MultiDiGraph:
    """OSM graph with only ``OSM_WAY_TAGS``, unprojected; the GraphML in ``folder`` is the cache."""

    spec = CITIES[city]
    path = graph_cache_path(folder, city, spec.graph, area=(spec.bbox, spec.center, spec.radius_m))
    if refresh:
        path.unlink(missing_ok=True)
    if path.exists():
        return ox.io.load_graphml(path)
    folder.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = folder / "http_cache"
    ox.settings.requests_timeout = spec.graph.requests_timeout
    # Only the selected tags: OSMnx defaults (name, ref, ...) are dropped on purpose.
    ox.settings.useful_tags_way = list(OSM_WAY_TAGS)
    graph = _download_graph(bbox=spec.bbox, center=spec.center, dist=spec.radius_m, config=spec.graph)
    ox.io.save_graphml(graph, path)
    return graph


# --- flat tables -----------------------------------------------------------------------------


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
    """Flat edge table in ``crs`` (EDGE_COLUMNS, graph edge order) and the OSM way ids of each edge.

    Several values of one tag (OSMnx merges ways when simplifying) are joined with ``;``; a missing tag
    is an empty cell; numeric tags keep their first number.
    """

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
    """Node table in ``crs`` (NODE_COLUMNS, graph node order)."""

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
    """Raise ``ValueError`` unless ``edges`` follows the flat schema."""

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
    """Node table from ``{node: (x, y)}``: toy networks for tests and examples."""

    ids = list(coordinates)
    x = [float(coordinates[node][0]) for node in ids]
    y = [float(coordinates[node][1]) for node in ids]
    return gpd.GeoDataFrame(
        {"node": np.array(ids, dtype="int64"), "x": x, "y": y}, geometry=gpd.points_from_xy(x, y), crs=crs
    )


def make_edges(records: Iterable[Mapping], *, crs, nodes: gpd.GeoDataFrame | None = None) -> gpd.GeoDataFrame:
    """Edge table in the full schema from a few dicts: toy networks for tests and examples.

    Each record needs ``u`` and ``v``; ``key`` defaults to 0, ``length_m`` to the geometry length,
    ``geometry`` to a straight line between the ``nodes`` coordinates, tags to empty cells.
    """

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


# --- reference network, observations, OD pairs -----------------------------------------------


def reference_mask(edges: pd.DataFrame, city: str) -> pd.Series:
    """Edges that observations are attached to and OD pairs are drawn from.

    ``is_rideable``, plus the largest strongly connected part for cities that ask for it. score() uses
    the same set as the pool of streets a rider could have taken.
    """

    mask = is_rideable(edges)
    if CITIES[city].largest_component:
        mask &= largest_component(edges[mask], strongly=True).reindex(edges.index, fill_value=False)
    return mask


def nodes_near_observations(graph, observed: gpd.GeoDataFrame, max_distance_m: float = 25.0):
    """Return graph-node IDs inside the spatial support of observed edges."""

    nodes = ox.graph_to_gdfs(graph, edges=False)
    matches = gpd.sjoin_nearest(
        nodes[["geometry"]],
        observed[["geometry"]],
        how="left",
        max_distance=max_distance_m,
        distance_col="coverage_distance_m",
    )
    return matches.loc[matches["index_right"].notna()].index.unique().to_numpy()


def sample_od_pairs(
    graph,
    candidate_nodes,
    n_pairs: int,
    seed: int,
    *,
    min_length_m: float = 1_500,
    max_length_m: float = 10_000,
    prefix: str = "DE",
):
    rng = np.random.default_rng(seed)
    nodes = np.asarray(candidate_nodes)
    pairs, seen = [], set()
    for _ in range(max(2_000, 200 * n_pairs)):
        origin, destination = rng.choice(nodes, size=2, replace=False).tolist()
        if (origin, destination) in seen:
            continue
        seen.add((origin, destination))
        try:
            length_m = nx.shortest_path_length(graph, origin, destination, weight="length_m")
        except nx.NetworkXNoPath:
            continue
        if min_length_m <= length_m <= max_length_m:
            pairs.append((f"{prefix}-{len(pairs) + 1:02d}", origin, destination))
        if len(pairs) == n_pairs:
            return pairs
    raise RuntimeError(f"Only {len(pairs)} OD pairs found")


def _od_pairs(city: str, reference: gpd.GeoDataFrame, nodes: gpd.GeoDataFrame, observations) -> pd.DataFrame:
    spec = CITIES[city]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConnectivityWarning)
        graph = build_graph(reference, nodes, "length_m")
    to_wgs84 = Transformer.from_crs(spec.crs, "EPSG:4326", always_xy=True)
    rows = []
    if isinstance(spec.od, RandomPairs):
        od = spec.od
        candidates = nodes_near_observations(graph, observations, od.near_observations_m)
        for od_id, origin, destination in sample_od_pairs(
            graph, candidates, od.n_pairs, od.seed,
            min_length_m=od.min_length_m, max_length_m=od.max_length_m, prefix=od.prefix,
        ):
            (start_lon, end_lon), (start_lat, end_lat) = to_wgs84.transform(
                [graph.nodes[origin]["x"], graph.nodes[destination]["x"]],
                [graph.nodes[origin]["y"], graph.nodes[destination]["y"]],
            )
            rows.append((od_id, origin, destination, start_lat, start_lon, end_lat, end_lon, "старт", "финиш"))
    else:
        for pair in spec.od:
            points = gpd.GeoSeries(
                gpd.points_from_xy([pair.start[1], pair.end[1]], [pair.start[0], pair.end[0]]), crs="EPSG:4326"
            ).to_crs(spec.crs)
            origin, destination = ox.distance.nearest_nodes(graph, X=points.x.to_numpy(), Y=points.y.to_numpy())
            rows.append((
                pair.od_id, int(origin), int(destination), *pair.start, *pair.end, pair.start_label, pair.end_label,
            ))
    return pd.DataFrame(rows, columns=[
        "od_id", "origin", "destination", "start_lat", "start_lon", "end_lat", "end_lon", "start_label", "end_label",
    ]).astype({"origin": "int64", "destination": "int64"})


def _observe(city: str, folder: Path, reference: gpd.GeoDataFrame, osmids: pd.Series):
    """Trips per reference edge, the observations themselves, and a short report."""

    spec = CITIES[city]
    source = spec.observations
    if isinstance(source, StadtradelnSource):
        observed = load_stadtradeln(folder / "stadtradeln", url=source.url, archive_name=source.archive_name)
        trips, report = match_stadtradeln(reference.assign(osmid=osmids), observed, source.match)
        observations = gpd.GeoDataFrame(
            {"trips": observed["number_of_matched_trips"]}, geometry=observed.geometry, crs=observed.crs
        ).to_crs(spec.crs)
        return trips, observations, report
    tracks = load_clean_personal_gpx(PROJECT_ROOT / source.directory, bounds=source.bounds).to_crs(spec.crs)
    snapped = snap_tracks_to_edges(reference, tracks, step_m=source.step_m, max_distance_m=source.max_distance_m)
    trips = track_edge_usage(snapped, reference)
    return trips, tracks, {"tracks": len(tracks), "edges_with_tracks": int((trips > 0).sum())}


# --- cache -----------------------------------------------------------------------------------


def cache_key(city: str) -> dict:
    """What the cached tables depend on; a different key rebuilds them."""

    payload = {
        "schema": SCHEMA_VERSION,
        "tags": list(OSM_WAY_TAGS),
        "columns": list(EDGE_COLUMNS),
        "city": {"name": city, **asdict(CITIES[city])},
    }
    return json.loads(json.dumps(payload, default=str))


def _is_cached(folder: Path, city: str) -> bool:
    meta = folder / "meta.json"
    if not meta.exists() or not all((folder / f"{name}.parquet").exists() for name in CACHE_FILES):
        return False
    return json.loads(meta.read_text(encoding="utf-8")).get("key") == cache_key(city)


def _read(folder: Path, name: str):
    path = folder / f"{name}.parquet"
    return gpd.read_parquet(path) if name in _GEO_FILES else pd.read_parquet(path)


def prepare_city(city: str, *, cache_dir: Path, refresh: bool = False) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    """Download OSM, attach observations, return ``(edges, nodes)``. Cached as GeoParquet.

    ``edges`` holds every downloaded edge, rideable or not: filtering is the student's job
    (``is_rideable`` is the example). ``refresh=True`` downloads OSM again and rebuilds everything,
    so the numbers may change; without it a cache is rebuilt only when its key changes.
    """

    if city not in CITIES:
        raise KeyError(f"Unknown city {city!r}; known: {', '.join(CITIES)}")
    spec = CITIES[city]
    folder = Path(cache_dir) / city
    if not refresh and _is_cached(folder, city):
        return _read(folder, "edges"), _read(folder, "nodes")

    raw = load_raw_graph(folder, city, refresh=refresh)
    edges, osmids = edges_table(raw, spec.crs)
    nodes = nodes_table(raw, spec.crs)
    del raw
    validate_edges(edges)

    reference_rows = reference_mask(edges, city)
    reference = edges[reference_rows]
    trips, observations, report = _observe(city, folder, reference, osmids[reference_rows])
    popularity = pd.DataFrame({"u": edges["u"], "v": edges["v"], "key": edges["key"], "trips": 0.0})
    popularity.loc[trips.index, "trips"] = trips.to_numpy(dtype=float)
    pairs = _od_pairs(city, reference, nodes, observations)

    (folder / "meta.json").unlink(missing_ok=True)
    edges.to_parquet(folder / "edges.parquet")
    nodes.to_parquet(folder / "nodes.parquet")
    popularity.to_parquet(folder / "popularity.parquet")
    observations.to_parquet(folder / "observations.parquet")
    pairs.to_parquet(folder / "od_pairs.parquet")
    report = {
        "edges": len(edges),
        "nodes": len(nodes),
        "reference_edges": int(reference_rows.sum()),
        "od_pairs": len(pairs),
        **report,
    }
    meta = {"key": cache_key(city), "report": report}
    (folder / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return edges, nodes


def _cached_table(city: str, cache_dir: Path, name: str):
    folder = Path(cache_dir) / city
    if not _is_cached(folder, city):
        prepare_city(city, cache_dir=cache_dir)
    return _read(folder, name)


def load_od_pairs(city: str, *, cache_dir: Path) -> pd.DataFrame:
    """Fixed start/finish pairs of the city: ``od_id``, ``origin``, ``destination`` and map markers."""

    return _cached_table(city, cache_dir, "od_pairs")


def load_popularity(city: str, *, cache_dir: Path) -> pd.DataFrame:
    """Observed trips per edge (u, v, key, trips), in the order of ``edges``. For metrics and maps only."""

    return _cached_table(city, cache_dir, "popularity")


def load_observations(city: str, *, cache_dir: Path) -> gpd.GeoDataFrame:
    """Observed STADTRADELN segments or GPS tracks in the city CRS. For metrics and maps only."""

    return _cached_table(city, cache_dir, "observations")


def preparation_report(city: str, *, cache_dir: Path) -> dict:
    """Numbers from the last preparation: table sizes and how well observations matched."""

    _cached_table(city, cache_dir, "od_pairs")
    return json.loads((Path(cache_dir) / city / "meta.json").read_text(encoding="utf-8"))["report"]
