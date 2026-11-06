from __future__ import annotations

import hashlib

import json

from pathlib import Path

import numpy as np

from .workshop import FORMAT_VERSION, TARGETS

def _digest(folder, names):
    return {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in names}

def write_city_pack(roads, nodes, pairs, trips, zone, *, folder, city, split_params, tracks=None):
    if not zone.eq('open').all():
        raise ValueError('Можно сохранять только открытые данные')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if not trips.index.equals(roads.index) or not zone.index.equals(roads.index):
        raise ValueError('Labels and zones must align with roads')
    if TARGETS & set(roads):
        raise ValueError('Roads must not contain popularity labels')
    roads = roads.reset_index(drop=True)
    roads.insert(0, 'edge_id', np.arange(len(roads)))
    trips, zone = (trips.reset_index(drop=True), zone.reset_index(drop=True))
    labels = roads[['edge_id']].assign(trips=trips)
    roads.to_parquet(folder / 'roads.parquet')
    nodes.to_parquet(folder / 'nodes.parquet')
    pairs[pairs.zone.eq('open')].drop(columns='zone').reset_index(drop=True).to_parquet(folder / 'pairs.parquet')
    labels[zone.eq('open') & trips.notna()].to_parquet(folder / 'labels.parquet')
    names = ['roads.parquet', 'nodes.parquet', 'pairs.parquet', 'labels.parquet']
    if tracks is not None:
        tracks.to_parquet(folder / 'tracks.parquet')
        names.append('tracks.parquet')
    hashes = _digest(folder, names)
    dataset_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()[:16]
    manifest = {'format_version': FORMAT_VERSION, 'dataset_id': dataset_id, 'city': city, 'files': hashes, 'roads': len(roads), 'nodes': len(nodes), 'open_pairs': int(pairs.zone.eq('open').sum()), 'open_labels': int((zone.eq('open') & trips.notna()).sum()), 'split': split_params, 'label_policy': 'NaN = no measurement; explicit zero kept; closed-zone labels withheld'}
    (folder / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    for stale in folder.glob('route_cache'):
        for path in stale.glob('*.pkl'):
            path.unlink()
    return manifest
