"""Stage 2 - routing graph: download OSM with failover, cache it, project it."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import networkx as nx
import osmnx as ox
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError
from requests import RequestException

from .config import OSM_WAY_TAGS, GraphConfig


class GraphDownloadError(RuntimeError):
    """Raised when every configured Overpass endpoint rejects a graph query."""


def _cache_fingerprint(config: GraphConfig) -> str:
    payload = {
        **asdict(config),
        "overpass_urls": None,
        "useful_tags_way": sorted(OSM_WAY_TAGS),
        "schema": 2,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return digest[:10]


def graph_cache_path(directory: Path, city: str, config: GraphConfig) -> Path:
    """Return a cache name that changes when graph semantics or saved tags change."""

    safe_city = "".join(char.lower() if char.isalnum() else "_" for char in city).strip("_")
    return directory / (
        f"{safe_city}_{config.network_type}_simp{int(config.simplify)}_"
        f"{_cache_fingerprint(config)}.graphml"
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


def load_graph(
    cache_dir: Path,
    city: str,
    config: GraphConfig,
    crs,
    *,
    bbox: tuple[float, float, float, float] | None = None,
    center: tuple[float, float] | None = None,
    dist: float | None = None,
    largest_component: bool = False,
) -> nx.MultiDiGraph:
    """OSM bike graph with only ``OSM_WAY_TAGS``, cached as GraphML and projected to ``crs``.

    Pass ``bbox`` or ``center`` + ``dist``. Changing the config or the tag list changes the cache file.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_dir / "http_cache"
    ox.settings.requests_timeout = config.requests_timeout
    # Only the selected tags: OSMnx defaults (name, ref, access, ...) are dropped on purpose.
    ox.settings.useful_tags_way = list(OSM_WAY_TAGS)

    path = graph_cache_path(cache_dir, city, config)
    if path.exists():
        graph = ox.io.load_graphml(path)
    else:
        graph = _download_graph(bbox=bbox, center=center, dist=dist, config=config)
        ox.io.save_graphml(graph, path)
    if largest_component:
        graph = ox.truncate.largest_component(graph, strongly=True)
    return ox.projection.project_graph(graph, to_crs=crs)
