"""Stage 1 - raw data: download and load STADTRADELN volumes and personal GPX tracks."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import gpxpy
from pyproj import Geod
import requests
from shapely.geometry import LineString
from tqdm.auto import tqdm


GEOD = Geod(ellps="WGS84")


def download_file(url: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        return destination
    temporary = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0)) or None
        with temporary.open("wb") as output, tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(1024 * 1024):
                if chunk:
                    output.write(chunk)
                    bar.update(len(chunk))
    temporary.replace(destination)
    return destination


def load_stadtradeln(
    data_dir: Path,
    *,
    url: str,
    archive_name: str = "stadtradeln_traffic_volumes.zip",
) -> gpd.GeoDataFrame:
    """Download when necessary and load the aggregate TrafficEdge layer."""

    paths = sorted(data_dir.rglob("*_gesamt/trafficvolumes.gpkg"))
    if not paths:
        archive_path = download_file(url, data_dir / archive_name)
        with ZipFile(archive_path) as archive:
            archive.extractall(data_dir)
        paths = sorted(data_dir.rglob("*_gesamt/trafficvolumes.gpkg"))
    if not paths:
        raise FileNotFoundError("The archive contains no *_gesamt/trafficvolumes.gpkg")
    observed = gpd.read_file(paths[0], layer="TrafficEdge")
    if observed.crs is None or observed.crs.to_epsg() != 25832:
        raise ValueError(f"Expected STADTRADELN EPSG:25832, got {observed.crs}")
    return observed


def load_clean_personal_gpx(
    directory: Path,
    *,
    bounds: tuple[float, float, float, float] = (29.40, 59.55, 30.90, 60.35),
    max_speed_kmh: float = 80.0,
    max_gap_seconds: float = 600.0,
) -> gpd.GeoDataFrame:
    """Drop out-of-city, duplicate, non-monotonic, and implausible GPX points."""

    records = []
    min_lon, min_lat, max_lon, max_lat = bounds
    for path in sorted(directory.glob("*.gpx")):
        with path.open(encoding="utf-8-sig") as stream:
            gpx = gpxpy.parse(stream)
        for track_index, track in enumerate(gpx.tracks, 1):
            for segment_index, segment in enumerate(track.segments, 1):
                kept = []
                for point in segment.points:
                    if not (min_lon <= point.longitude <= max_lon and min_lat <= point.latitude <= max_lat):
                        continue
                    if kept:
                        previous = kept[-1]
                        _, _, distance_m = GEOD.inv(
                            previous.longitude,
                            previous.latitude,
                            point.longitude,
                            point.latitude,
                        )
                        if abs(distance_m) < 1:
                            continue
                        if previous.time and point.time:
                            seconds = (point.time - previous.time).total_seconds()
                            if seconds <= 0 or (
                                seconds <= max_gap_seconds
                                and 3.6 * abs(distance_m) / seconds > max_speed_kmh
                            ):
                                continue
                    kept.append(point)
                if len(kept) >= 2:
                    records.append(
                        {
                            "activity_id": f"{path.stem}:T{track_index}:S{segment_index}",
                            "file": path.name,
                            "geometry": LineString([(p.longitude, p.latitude) for p in kept]),
                        }
                    )
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
