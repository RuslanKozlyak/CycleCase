"""External dataset download and loading helpers."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import geopandas as gpd
import requests
from tqdm.auto import tqdm


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
