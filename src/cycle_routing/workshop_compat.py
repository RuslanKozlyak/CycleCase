"""Compatibility for solutions exported before Russian tag values and a single feature function."""

from dataclasses import dataclass

import geopandas as gpd
import pandas as pd

from .tags import has_any


def legacy_roads(roads: pd.DataFrame) -> pd.DataFrame:
    """Student view: Russian column names, no graph keys. Accepts OSM or already Russian names."""
    from .workshop import RU, osm

    roads = osm(roads)
    view = roads[[c for c in [*RU, "geometry"] if c in roads]].rename(columns=RU)
    return view.rename_geometry("геометрия") if isinstance(view, gpd.GeoDataFrame) else view


def legacy_features(roads: pd.DataFrame) -> pd.DataFrame:
    """Ready features for weights, from OSM tags only (never from trips)."""
    from .workshop import osm

    r = osm(roads)
    f = pd.DataFrame(index=r.index)
    f["длина_м"] = r.length_m
    bike_values = r.cycleway.fillna("").str.split(";")
    f["велодорожка"] = (has_any(r.highway, "cycleway")
                        | bike_values.map(lambda values: bool(set(values) - {"", "no", "none", "separate"}))).astype(float)
    f["крупная_дорога"] = has_any(r.highway, {"primary", "secondary", "trunk", "primary_link", "secondary_link"}).astype(float)
    f["жилая_улица"] = has_any(r.highway, {"residential", "living_street"}).astype(float)
    f["пешеходная"] = has_any(r.highway, {"footway", "pedestrian", "steps", "path"}).astype(float)
    f["плохое_покрытие"] = (has_any(r.surface, {"sett", "cobblestone", "unpaved", "gravel", "ground", "dirt", "sand"})
                            | has_any(r.smoothness, {"bad", "very_bad", "horrible", "impassable"})).astype(float)
    f["без_освещения"] = has_any(r.lit, "no").astype(float)
    f["узкая"] = (r.width.notna() & r.width.lt(2)).astype(float)
    for source, name, default, missing in [("maxspeed", "скорость", 30., "скорость_неизвестна"),
                                           ("lanes", "полосы", 1., "полосы_неизвестны"),
                                           ("width", "ширина", 3., "ширина_неизвестна")]:
        values = pd.to_numeric(r[source], errors="coerce")
        f[missing] = values.isna().astype(float)
        f[name] = values.fillna(default)
    f["покрытие_неизвестно"] = r.surface.isna().astype(float)
    return f


@dataclass
class LegacyFeatures:
    make: object
    additions: tuple = ()

    def __call__(self, roads):
        view = legacy_roads(roads)
        table = self.make(view).copy()
        for make in self.additions:
            more = make(view)
            table[more.columns] = more
        return table


@dataclass
class LegacyFilter:
    keep: object

    def __call__(self, roads):
        return self.keep(legacy_roads(roads))


def restore_features(state, current_default):
    make = state.get("base_features")
    if make is None or make is current_default:
        make = legacy_features
    return LegacyFeatures(make, state.get("extra", ()))
