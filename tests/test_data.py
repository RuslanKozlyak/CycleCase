from datetime import datetime, timedelta

import numpy as np

from cycle_routing.data import load_clean_personal_gpx

START = datetime(2024, 5, 1, 10, 0, 0)


def write_gpx(path, points):
    """One track segment of (lat, lon, time) points."""
    body = "".join(f'<trkpt lat="{lat}" lon="{lon}"><time>{time:%Y-%m-%dT%H:%M:%SZ}</time></trkpt>'
                   for lat, lon, time in points)
    path.write_text('<?xml version="1.0"?><gpx version="1.1" creator="test" xmlns="http://www.topografix.com/GPX/1/1">'
                    f"<trk><trkseg>{body}</trkseg></trk></gpx>", encoding="utf-8")


def ride(lat, start, n=5):
    """About 11 m between points, every 5 seconds."""
    return [(lat + 0.0001 * i, 30.30, start + timedelta(seconds=5 * i)) for i in range(n)]


def test_long_jump_after_a_pause_starts_a_new_line_of_the_same_ride(tmp_path):
    # 20 minutes without signal and ~510 m further: not a road to draw, but the same ride.
    write_gpx(tmp_path / "ride.gpx", ride(59.95, START) + ride(59.955, START + timedelta(minutes=20)))
    tracks = load_clean_personal_gpx(tmp_path).to_crs("EPSG:32636")
    assert len(tracks) == 2 and tracks.activity_id.nunique() == 1
    steps = np.concatenate([np.hypot(*np.diff(np.asarray(g.coords), axis=0).T) for g in tracks.geometry])
    assert steps.max() < 150


def test_continuous_ride_stays_one_line(tmp_path):
    write_gpx(tmp_path / "ride.gpx", ride(59.95, START, n=20))
    tracks = load_clean_personal_gpx(tmp_path)
    assert len(tracks) == 1 and len(tracks.geometry.iloc[0].coords) == 20
