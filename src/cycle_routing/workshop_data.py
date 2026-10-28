"""Organizer-only preparation. Public files never include test labels or original observations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .evaluation import edge_trips
from .graph import build_graph, is_rideable, largest_component
from .learning import spatial_partition
from .od_analysis import describe_od_pairs, select_topology_od
from .preparation import CITIES, load_observations, nodes_near_observations, observed_area, prepare_city
from .routing import build_routes


def write_lab_pack(edges, nodes, pairs, trips, partition, *, folder, private_folder=None, city="konstanz", external=False):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    if not trips.index.equals(edges.index) or not partition.index.equals(edges.index):
        raise ValueError("Labels and partition must align with streets")
    if not is_rideable(edges).all(): raise ValueError("Student network must contain only rideable streets")
    edges.to_parquet(folder/"edges.parquet"); nodes.to_parquet(folder/"nodes.parquet")
    pairs.to_parquet(folder/"pairs.parquet")
    pd.DataFrame({"split":partition}).to_parquet(folder/"partition.parquet")
    public_splits=["external"] if external else ["train","validation"]
    files=["edges.parquet","nodes.parquet","pairs.parquet","partition.parquet"]
    targets=edges[["u","v","key"]].assign(trips=trips)
    for split in public_splits:
        mask=trips.notna() & (pd.Series(True,index=edges.index) if external else partition.eq(split))
        name=f"labels_{split}.parquet"
        targets.loc[mask].to_parquet(folder/name); files.append(name)
    hashes={name:hashlib.sha256((folder/name).read_bytes()).hexdigest() for name in files}
    dataset_id=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()[:16]
    manifest={"format_version":1,"dataset_id":dataset_id,"city":city,"files":hashes,
              "public_splits":public_splits,"edges":len(edges),"nodes":len(nodes),"od_pairs":len(pairs),
              "label_policy":"NaN=unknown; explicit zero retained; test labels withheld",
              "region":"largest strongly connected component inside the observation footprint" if not external else "prepared external GPX network",
              "od_policy":"deterministic geography/distance/orientation/corridor coverage" if not external else "fixed address pair"}
    (folder/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    if not external:
        if private_folder is None: raise ValueError("Organizer test folder is required")
        private_folder=Path(private_folder); private_folder.mkdir(parents=True,exist_ok=True)
        targets.loc[trips.notna() & partition.eq("test")].to_parquet(private_folder/"test_labels.parquet")
        (private_folder/"dataset.json").write_text(json.dumps({"dataset_id":dataset_id,"city":city}),encoding="utf-8")
    return manifest


def prepare_konstanz(*, source_cache, folder, private_folder, seed=42):
    edges,nodes=prepare_city("konstanz",cache_dir=Path(source_cache))
    footprint=observed_area(edges,"konstanz",cache_dir=Path(source_cache))
    core=edges[is_rideable(edges)&footprint].copy()
    core=core[largest_component(core,strongly=True)].copy()
    nodes=nodes[nodes.node.isin(set(core.u)|set(core.v))].copy()
    trips=edge_trips(core,"konstanz",cache_dir=Path(source_cache))
    geographic=pd.Series(True,index=core.index)
    partition=spatial_partition(core,geographic,random_state=seed)
    graph=build_graph(core,nodes,"length_m")
    candidates=nodes_near_observations(graph,load_observations("konstanz",cache_dir=Path(source_cache)))
    del graph
    pairs,_=select_topology_od(core,nodes,candidates,surveyed=geographic,n_pairs=20,n_anchors=28)
    reference=build_routes(core,nodes,pairs,{},cache_dir=Path(folder)/"route_cache")
    description=describe_od_pairs(core,nodes,pairs,reference,surveyed=geographic,partition=partition,known=trips.notna())
    pairs=pairs.merge(description[["length_km","length_band","orientation","junctions_per_km"]],left_on="od_id",right_index=True)
    return write_lab_pack(core,nodes,pairs,trips,partition,folder=folder,private_folder=private_folder,city="konstanz")


def prepare_spb_from_graph(*, source_graph, gpx_directory, folder):
    """Use an explicitly selected local graph as-is; never rename it into an unrelated OSM cache key."""
    import osmnx as ox
    from .data import load_clean_personal_gpx
    from .matching import snap_tracks_to_edges,track_edge_usage
    from .preparation import edges_table,nodes_table,_od_pairs

    raw=ox.load_graphml(source_graph)
    edges,_=edges_table(raw,CITIES["saint_petersburg"].crs)
    nodes=nodes_table(raw,CITIES["saint_petersburg"].crs)
    del raw
    edges=edges[is_rideable(edges)].copy()
    edges=edges[largest_component(edges,strongly=True)].copy()
    nodes=nodes[nodes.node.isin(set(edges.u)|set(edges.v))].copy()
    source=CITIES["saint_petersburg"].observations
    tracks=load_clean_personal_gpx(Path(gpx_directory),bounds=source.bounds).to_crs(edges.crs)
    snapped=snap_tracks_to_edges(edges,tracks,step_m=source.step_m,max_distance_m=source.max_distance_m)
    trips=track_edge_usage(snapped,edges)
    pairs=_od_pairs("saint_petersburg",edges,nodes,tracks)
    partition=pd.Series("external",index=edges.index)
    manifest=write_lab_pack(edges,nodes,pairs,trips,partition,folder=folder,city="saint_petersburg",external=True)
    manifest["source_graph"]=Path(source_graph).name
    (Path(folder)/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8")
    return manifest
