"""GPX parsing and conservative cleaning for Saint Petersburg tracks."""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import gpxpy
import pandas as pd
from pyproj import Geod
from shapely.geometry import LineString


GEOD = Geod(ellps="WGS84")


def geodesic_length_m(coordinates: list[tuple[float, float]]) -> float:
    if len(coordinates) < 2:
        return 0.0
    longitudes, latitudes = zip(*coordinates)
    return abs(float(GEOD.line_length(longitudes, latitudes)))


def load_gpx_tracks(directory: Path) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
    """Parse GPX files into track geometries and point-level observations."""

    track_rows: list[dict[str, object]] = []
    point_rows: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.gpx")):
        with path.open(encoding="utf-8-sig") as stream:
            gpx = gpxpy.parse(stream)
        for track_index, track in enumerate(gpx.tracks, 1):
            for segment_index, segment in enumerate(track.segments, 1):
                if len(segment.points) < 2:
                    continue
                coordinates = [(point.longitude, point.latitude) for point in segment.points]
                route_id = f"{path.stem}:T{track_index}:S{segment_index}"
                times = [point.time for point in segment.points if point.time]
                elevations = [point.elevation for point in segment.points if point.elevation is not None]
                track_rows.append(
                    {
                        "route_id": route_id,
                        "file": path.name,
                        "name": track.name or path.stem,
                        "track_type": track.type,
                        "points": len(segment.points),
                        "start_time": min(times) if times else pd.NaT,
                        "end_time": max(times) if times else pd.NaT,
                        "min_elevation_m": min(elevations) if elevations else None,
                        "max_elevation_m": max(elevations) if elevations else None,
                        "distance_km": geodesic_length_m(coordinates) / 1000,
                        "geometry": LineString(coordinates),
                    }
                )
                cumulative_m = 0.0
                previous = None
                for point_index, point in enumerate(segment.points):
                    if previous is not None:
                        _, _, step_m = GEOD.inv(
                            previous.longitude,
                            previous.latitude,
                            point.longitude,
                            point.latitude,
                        )
                        cumulative_m += max(step_m, 0.0)
                    point_rows.append(
                        {
                            "route_id": route_id,
                            "point_index": point_index,
                            "longitude": point.longitude,
                            "latitude": point.latitude,
                            "elevation_m": point.elevation,
                            "time": point.time,
                            "distance_km": cumulative_m / 1000,
                        }
                    )
                    previous = point
    tracks = gpd.GeoDataFrame(track_rows, geometry="geometry", crs="EPSG:4326")
    if tracks.empty:
        raise ValueError("В GPX-файлах не найдено сегментов минимум с двумя точками")
    tracks["duration_h"] = (
        pd.to_datetime(tracks["end_time"], utc=True) - pd.to_datetime(tracks["start_time"], utc=True)
    ).dt.total_seconds() / 3600
    tracks["avg_speed_kmh"] = tracks["distance_km"].div(tracks["duration_h"]).where(tracks["duration_h"] > 0)
    return tracks, pd.DataFrame(point_rows)


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
