from __future__ import annotations

from pathlib import Path

import geopandas as gpd

import gpxpy

from pyproj import Geod

from shapely.geometry import LineString

GEOD = Geod(ellps="WGS84")

def load_clean_personal_gpx(
    directory: Path,
    *,
    bounds: tuple[float, float, float, float] = (29.40, 59.55, 30.90, 60.35),
    max_speed_kmh: float = 80.0,
    max_gap_seconds: float = 600.0,
    max_jump_m: float = 150.0,
) -> gpd.GeoDataFrame:

    records = []
    min_lon, min_lat, max_lon, max_lat = bounds
    for path in sorted(directory.glob("*.gpx")):
        with path.open(encoding="utf-8-sig") as stream:
            gpx = gpxpy.parse(stream)
        for track_index, track in enumerate(gpx.tracks, 1):
            for segment_index, segment in enumerate(track.segments, 1):
                parts, kept = [], []
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
                        if abs(distance_m) > max_jump_m:
                            parts.append(kept)
                            kept = []
                    kept.append(point)
                parts.append(kept)
                for part in parts:
                    if len(part) >= 2:
                        records.append(
                            {
                                "activity_id": f"{path.stem}:T{track_index}:S{segment_index}",
                                "file": path.name,
                                "geometry": LineString([(p.longitude, p.latitude) for p in part]),
                            }
                        )
    return gpd.GeoDataFrame(records, geometry="geometry", crs="EPSG:4326")
