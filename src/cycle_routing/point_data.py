import hashlib

import json

from pathlib import Path

from tempfile import NamedTemporaryFile

import geopandas as gpd

import pandas as pd

POINT_TYPES = {
    "ворота": {"description": "Ворота и шлагбаумы", "tags": {"barrier": ["gate", "lift_gate", "swing_gate", "chain"]}},
    "столбики и ограждения": {"description": "Столбики, ограждения и узкие проходы", "tags": {"barrier": ["cycle_barrier", "bollard", "stile", "kissing_gate", "turnstile"]}},
    "светофоры": {"description": "Светофоры", "tags": {"highway": ["traffic_signals"]}},
    "переходы": {"description": "Пешеходные переходы", "tags": {"highway": ["crossing"]}},
    "переезды": {"description": "Железнодорожные переезды и переходы", "tags": {"railway": ["level_crossing", "crossing"]}},
    "остановки": {"description": "Остановки общественного транспорта", "tags": {"highway": ["bus_stop"], "public_transport": ["platform"]}},
    "питьевая вода": {"description": "Питьевая вода", "tags": {"amenity": ["drinking_water"]}},
    "ремонт": {"description": "Велоремонт и насосы", "tags": {"amenity": ["bicycle_repair_station"], "service:bicycle:pump": ["yes"]}},
    "велопарковки": {"description": "Велосипедные парковки", "tags": {"amenity": ["bicycle_parking"]}},
}

def point_query_key():
    tags = {name: item["tags"] for name, item in POINT_TYPES.items()}
    return hashlib.sha256(json.dumps(tags, sort_keys=True).encode()).hexdigest()

def point_files(folder):
    folder = Path(folder)
    manifest = folder / "points" / "manifest.json"
    if not manifest.exists():
        return []
    meta = json.loads(manifest.read_text(encoding="utf-8"))
    data = manifest.with_name("points.parquet")
    if meta.get("format_version") != 1 or hashlib.sha256(data.read_bytes()).hexdigest() != meta["sha256"]:
        raise ValueError("Изменён набор точек; восстановите исходные файлы")
    return [manifest, data]

def load_points_catalog(folder, crs):
    files = point_files(folder)
    if not files:
        return {}
    meta = json.loads(files[0].read_text(encoding="utf-8"))
    table = gpd.read_parquet(files[1]).to_crs(crs)
    if not table.geometry.geom_type.eq("Point").all():
        raise ValueError("В наборе точек ожидаются геометрии Point")
    return {name: table.loc[table["тип"].eq(name)].copy() for name in meta["categories"]}

def points_dictionary(city):
    rows = [{"тип": name, "описание": POINT_TYPES.get(name, {}).get("description", name), "объектов": len(data)}
            for name, data in city.points.items()]
    table = pd.DataFrame(rows, columns=["тип", "описание", "объектов"]).set_index("тип").style
    manifest = Path(city.folder) / "points" / "manifest.json"
    if manifest.exists() and not json.loads(manifest.read_text(encoding="utf-8")).get("complete", False):
        table = table.set_caption("Локальный снимок: набор точек неполный")
    return table

def save_points_snapshot(folder, table, *, categories, source, complete=True, **metadata):
    target = Path(folder) / "points"
    target.mkdir(parents=True, exist_ok=True)
    table = table.to_crs(4326).reset_index(drop=True)
    if not table.geometry.geom_type.eq("Point").all() or table["тип"].isin(categories).eq(False).any():
        raise ValueError("Некорректные точки или категории")
    if table.duplicated(["тип", "point_id"]).any():
        raise ValueError("Повторяющиеся точки")
    with NamedTemporaryFile(dir=target, suffix=".parquet", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        table.to_parquet(temporary)
        temporary.replace(target / "points.parquet")
    finally:
        temporary.unlink(missing_ok=True)
    meta = {"format_version": 1, "categories": list(categories), "source": source,
            "complete": complete, "sha256": hashlib.sha256((target / "points.parquet").read_bytes()).hexdigest(),
            **metadata}
    (target / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta
