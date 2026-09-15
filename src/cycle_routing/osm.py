"""OSMnx setup, resilient Overpass downloads, and cache metadata."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import networkx as nx
import osmnx as ox
from osmnx._errors import InsufficientResponseError, ResponseStatusCodeError
import pandas as pd
from requests import RequestException

from .config import EXTRA_OSM_TAGS, GraphConfig


class GraphDownloadError(RuntimeError):
    """Raised when every configured Overpass endpoint rejects a graph query."""


def configure_osmnx(cache_folder: Path, config: GraphConfig) -> None:
    """Configure cache, timeout, and tags *before* graph creation or loading."""

    cache_folder.mkdir(parents=True, exist_ok=True)
    ox.settings.use_cache = True
    ox.settings.cache_folder = cache_folder
    ox.settings.requests_timeout = config.requests_timeout
    ox.settings.useful_tags_way = list(
        dict.fromkeys([*ox.settings.useful_tags_way, *EXTRA_OSM_TAGS])
    )


def _cache_fingerprint(config: GraphConfig) -> str:
    payload = {
        **asdict(config),
        "overpass_urls": None,
        "useful_tags_way": sorted(EXTRA_OSM_TAGS),
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


def load_or_download_graph(
    path: Path,
    *,
    config: GraphConfig,
    bbox: tuple[float, float, float, float] | None = None,
    center: tuple[float, float] | None = None,
    dist: float | None = None,
    force_download: bool = False,
) -> nx.MultiDiGraph:
    """Load a semantic cache or download sequentially with endpoint failover."""

    path.parent.mkdir(parents=True, exist_ok=True)
    configure_osmnx(path.parent / "http_cache", config)
    if path.exists() and not force_download:
        return ox.io.load_graphml(path)

    graph = _download_graph(bbox=bbox, center=center, dist=dist, config=config)
    graph.graph["cycle_routing_graph_config"] = json.dumps(asdict(config), ensure_ascii=False)
    graph.graph["cycle_routing_extra_tags"] = json.dumps(EXTRA_OSM_TAGS)
    ox.io.save_graphml(graph, path)
    return graph


def graph_build_summary(
    city: str,
    downloaded_graph: nx.MultiDiGraph,
    routing_graph: nx.MultiDiGraph,
    config: GraphConfig,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "city": city,
                **{key: value for key, value in config.as_dict().items() if key != "overpass_urls"},
                "downloaded_nodes": downloaded_graph.number_of_nodes(),
                "downloaded_edges": downloaded_graph.number_of_edges(),
                "routing_nodes": routing_graph.number_of_nodes(),
                "routing_edges": routing_graph.number_of_edges(),
                "downloaded_crs": str(downloaded_graph.graph.get("crs")),
                "routing_crs": str(routing_graph.graph.get("crs")),
                "graph_simplified_flag": downloaded_graph.graph.get("simplified", False),
            }
        ]
    )
