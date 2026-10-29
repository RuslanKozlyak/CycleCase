"""Organizer-only preparation of hackathon packs. Public files never include closed-zone OD or labels.

Public pack (``notebooks/data/hackathon/<name>``): roads, nodes, open-zone pairs and labels, manifest.
Private folder (``outputs/hackathon_private/<name>``): closed-zone pairs and labels, zone of every road.
The split is pluggable: any function from ``splits.SPLITS`` (or your own returning ``splits.Split``).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .evaluation import edge_trips
from .graph import is_rideable, largest_component
from .preparation import CITIES, observed_area, prepare_city
from .splits import SPLITS
from .workshop import FORMAT_VERSION, TARGETS, reference_roads


def _digest(folder, names):
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in names}


def write_city_pack(roads, nodes, pairs, trips, zone, *, folder, private_folder=None, city, split_params, tracks=None):
    """``pairs`` carries a ``zone`` column; only ``open`` pairs and open-zone labels become public."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not trips.index.equals(roads.index) or not zone.index.equals(roads.index):
        raise ValueError("Labels and zones must align with roads")
    if TARGETS & set(roads):
        raise ValueError("Roads must not contain popularity labels")
    roads = roads.reset_index(drop=True)
    roads.insert(0, "edge_id", np.arange(len(roads)))
    trips, zone = trips.reset_index(drop=True), zone.reset_index(drop=True)
    labels = roads[["edge_id"]].assign(trips=trips)
    roads.to_parquet(folder / "roads.parquet")
    nodes.to_parquet(folder / "nodes.parquet")
    pairs[pairs.zone.eq("open")].drop(columns="zone").reset_index(drop=True).to_parquet(folder / "pairs.parquet")
    labels[zone.eq("open") & trips.notna()].to_parquet(folder / "labels.parquet")
    names = ["roads.parquet", "nodes.parquet", "pairs.parquet", "labels.parquet"]
    if zone.ne("open").any():  # where labels are hidden, so the map can say "test area", not "no data"
        roads[["edge_id"]].assign(test_area=zone.ne("open")).to_parquet(folder / "test_area.parquet")
        names.append("test_area.parquet")
    if tracks is not None:
        tracks.to_parquet(folder / "tracks.parquet")
        names.append("tracks.parquet")
    hashes = _digest(folder, names)
    dataset_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    manifest = {
        "format_version": FORMAT_VERSION, "dataset_id": dataset_id, "city": city, "files": hashes,
        "roads": len(roads), "nodes": len(nodes), "open_pairs": int(pairs.zone.eq("open").sum()),
        "open_labels": int((zone.eq("open") & trips.notna()).sum()), "split": split_params,
        "label_policy": "NaN = no measurement; explicit zero kept; closed-zone labels withheld",
    }
    (folder / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    for stale in folder.glob("route_cache"):
        for path in stale.glob("*.pkl"):
            path.unlink()
    if zone.eq("closed").any():
        if private_folder is None:
            raise ValueError("A closed zone needs an organizer folder")
        private = Path(private_folder)
        private.mkdir(parents=True, exist_ok=True)
        pairs[pairs.zone.eq("closed")].drop(columns="zone").reset_index(drop=True).to_parquet(private / "closed_pairs.parquet")
        labels[zone.eq("closed") & trips.notna()].to_parquet(private / "closed_labels.parquet")
        roads[["edge_id"]].assign(zone=zone).to_parquet(private / "zones.parquet")
        (private / "dataset.json").write_text(json.dumps({"dataset_id": dataset_id, "city": city, "split": split_params},
                                                         ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def konstanz_roads(source_cache):
    """All roads of the largest strongly connected piece inside the observation footprint, with labels."""
    edges, nodes = prepare_city("konstanz", cache_dir=Path(source_cache))
    footprint = observed_area(edges, "konstanz", cache_dir=Path(source_cache))
    roads = edges[footprint]
    roads = roads[largest_component(roads, strongly=True)].copy()
    nodes = nodes[nodes.node.isin(set(roads.u) | set(roads.v))].copy()
    return roads, nodes, edge_trips(roads, "konstanz", cache_dir=Path(source_cache))


def prepare_konstanz(*, source_cache, folder, private_folder, split="routes", candidates_cache=None, **params):
    roads, nodes, trips = konstanz_roads(source_cache)
    make = SPLITS[split] if isinstance(split, str) else split
    result = make(roads, nodes, trips.notna(), rideable=reference_roads(roads), cache=candidates_cache, **params)
    return write_city_pack(roads, nodes, result.pairs, trips, result.zone, folder=folder, private_folder=private_folder,
                          city="konstanz", split_params=result.params)


def prepare_spb_from_graph(*, source_graph, gpx_directory, folder, margin_m=2000):
    """Personal GPX check: rideable roads within ``margin_m`` of the straight line between the fixed
    address pair, GPX visits as labels, GPX lines for the map."""
    import osmnx as ox
    from shapely.geometry import LineString
    from shapely.ops import unary_union
    from .data import load_clean_personal_gpx
    from .matching import snap_tracks_to_edges, track_edge_usage
    from .preparation import _od_pairs, edges_table, nodes_table

    spec = CITIES["saint_petersburg"]
    raw = ox.load_graphml(source_graph)
    edges, _ = edges_table(raw, spec.crs)
    nodes = nodes_table(raw, spec.crs)
    del raw
    edges = edges[is_rideable(edges)]
    edges = edges[largest_component(edges, strongly=True)].copy()
    tracks = load_clean_personal_gpx(Path(gpx_directory), bounds=spec.observations.bounds).to_crs(edges.crs)
    pairs = _od_pairs("saint_petersburg", edges, nodes, tracks)
    xy = nodes.set_index("node")[["x", "y"]]
    area = unary_union([LineString([xy.loc[o], xy.loc[d]]).buffer(margin_m) for o, d in zip(pairs.origin, pairs.destination)])
    edges = edges[edges.intersects(area)]
    edges = edges[largest_component(edges, strongly=True)].copy()
    nodes = nodes[nodes.node.isin(set(edges.u) | set(edges.v))].copy()
    tracks = tracks.clip(area)
    tracks = tracks[~tracks.geometry.is_empty][["activity_id", "geometry"]].reset_index(drop=True)
    snapped = snap_tracks_to_edges(edges, tracks, step_m=spec.observations.step_m, max_distance_m=spec.observations.max_distance_m)
    trips = track_edge_usage(snapped, edges)
    zone = pd.Series("open", index=edges.index)
    manifest = write_city_pack(edges, nodes, pairs.assign(zone="open"), trips, zone, folder=folder, city="saint_petersburg",
                              split_params={"method": "external", "source_graph": Path(source_graph).name}, tracks=tracks)
    return manifest
