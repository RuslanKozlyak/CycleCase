"""Student API for the hackathon notebook: prepared data, solution blocks, one scoring table, export.

The notebook sees labels of the open zone only. The organizer keeps closed-zone OD and labels and
scores an exported solution with exactly the same ``score``.

Students see roads with Russian column names (``lab.roads``); files and the routing library keep the
OSM names (``lab.osm``). Their own functions always receive the Russian table.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .evaluation import weighted_corr
from .graph import is_rideable, largest_component
from .learning import PopularityModel, fit_popularity_model, spatial_partition
from .presets import COMFORT_PRESETS
from .routing import build_routes
from .tags import has_any

KEY = ["u", "v", "key"]
FORMAT_VERSION = 2
MAX_DETOUR_PCT = 20
UNKNOWN_POPULARITY = 50.0  # a road without measurement counts as an average measured road
TARGETS = {"trips", "has_observation", "popularity"}
RU = {"length_m": "длина_м", "highway": "тип_дороги", "surface": "покрытие", "smoothness": "ровность",
      "cycleway": "велополоса", "bicycle": "велосипед", "lit": "освещение", "service": "служебный_проезд",
      "access": "доступ", "maxspeed": "макс_скорость", "lanes": "полос", "width": "ширина_м", "oneway": "одностороннее"}
OSM = {ru: name for name, ru in RU.items()}


def russian(roads: pd.DataFrame) -> pd.DataFrame:
    """Student view: Russian column names, no graph keys. Accepts OSM or already Russian names."""
    roads = osm(roads)
    view = roads[[c for c in [*RU, "geometry"] if c in roads]].rename(columns=RU)
    return view.rename_geometry("геометрия") if isinstance(view, gpd.GeoDataFrame) else view


def osm(roads: pd.DataFrame) -> pd.DataFrame:
    table = roads.rename(columns=OSM)
    if isinstance(table, gpd.GeoDataFrame) and table.geometry.name == "геометрия":
        table = table.rename_geometry("geometry")
    return table


# --- data ---------------------------------------------------------------------------------------


def percentile(trips: pd.Series, length: pd.Series) -> pd.Series:
    """Popularity 0–100: share of measured street length with fewer trips; NaN = no measurement."""
    known = trips.dropna().sort_values(kind="stable")
    share = length.loc[known.index] / length.loc[known.index].sum()
    middle = share.cumsum() - share / 2
    return (100 * middle.groupby(known.to_numpy()).transform("mean")).reindex(trips.index).rename("popularity")


def reference_roads(roads: pd.DataFrame) -> pd.Series:
    """Roads of the fixed shortest-path baseline: rideable and in the largest strongly connected piece."""
    rideable_part = roads[is_rideable(roads)]
    mask = pd.Series(False, index=roads.index)
    mask.loc[rideable_part.index] = largest_component(rideable_part, strongly=True).to_numpy()
    return mask


def rideable(roads: pd.DataFrame) -> pd.Series:
    """Ready filter: no stairs, no bicycle bans, no private or construction roads."""
    return is_rideable(osm(roads))


@dataclass
class Lab:
    folder: Path
    osm: gpd.GeoDataFrame                  # OSM column names: files and routing
    nodes: gpd.GeoDataFrame
    pairs: pd.DataFrame
    trips: pd.Series                       # NaN = no measurement, 0 = measured zero
    manifest: dict
    tracks: gpd.GeoDataFrame | None = None
    test_area: pd.Series | None = None     # roads whose labels are hidden for grading
    name: str = "открытая зона"

    @cached_property
    def roads(self) -> gpd.GeoDataFrame:
        return russian(self.osm)

    @cached_property
    def reference(self) -> pd.Series:
        return reference_roads(self.osm)

    @cached_property
    def popularity(self) -> pd.Series:
        return percentile(self.trips, self.osm.length_m)


def _align(roads, labels, column="trips"):
    if column == "trips" and (labels.edge_id.duplicated().any() or not np.isfinite(labels.trips).all()
                              or labels.trips.lt(0).any()):
        raise ValueError("Метки должны быть уникальными, известными и неотрицательными")
    return labels.set_index("edge_id")[column].reindex(roads.edge_id).set_axis(roads.index).rename(column)


def load_city(folder: str | Path) -> Lab:
    """Read a prepared pack only. Never downloads or rebuilds the city in a student session."""
    folder = Path(folder)
    path = folder / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Нет подготовленного набора {folder}. Его выдаёт организатор; инструкция в HACKATHON.md.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != FORMAT_VERSION:
        raise ValueError("Набор старого формата: подготовьте его заново (tools/prepare_hackathon_data.py)")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((folder / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Изменён файл данных {name}; распакуйте исходный набор заново")
    roads, nodes = gpd.read_parquet(folder / "roads.parquet"), gpd.read_parquet(folder / "nodes.parquet")
    if TARGETS & set(roads):
        raise ValueError("Метки популярности не должны входить в таблицу дорог")
    optional = lambda name: folder / name if name in manifest["files"] else None
    tracks = gpd.read_parquet(optional("tracks.parquet")) if optional("tracks.parquet") else None
    test_area = (_align(roads, pd.read_parquet(optional("test_area.parquet")), "test_area").fillna(False).astype(bool)
                 if optional("test_area.parquet") else pd.Series(False, index=roads.index, name="test_area"))
    return Lab(folder, roads, nodes, pd.read_parquet(folder / "pairs.parquet"),
               _align(roads, pd.read_parquet(folder / "labels.parquet")), manifest, tracks, test_area)


def closed_city(lab: Lab, private_folder: str | Path) -> Lab:
    """Organizer only: the same roads with the closed-zone OD and labels."""
    private = Path(private_folder)
    meta = json.loads((private / "dataset.json").read_text(encoding="utf-8"))
    if meta["dataset_id"] != lab.manifest["dataset_id"]:
        raise ValueError("Закрытые данные относятся к другому набору")
    return Lab(lab.folder, lab.osm, lab.nodes, pd.read_parquet(private / "closed_pairs.parquet"),
               _align(lab.osm, pd.read_parquet(private / "closed_labels.parquet")), lab.manifest,
               name="закрытая зона")


KINDS = {"длина_м": "число, м", "макс_скорость": "число, км/ч", "полос": "число", "ширина_м": "число, м",
         "одностороннее": "да/нет", "геометрия": "линия в метрах"}


def data_dictionary(lab: Lab):
    """One row per column of ``city.roads``, like a competition data page, with every value of each category
    (most frequent first); the target is listed last. Shown wrapped, nothing cut."""
    rows = {}
    for column in lab.roads.columns:
        kind, values = KINDS.get(column, "категория"), lab.roads[column]
        if column == "геометрия":
            text = "EPSG:25832"
        elif kind == "категория":
            text = ", ".join(values.dropna().str.split(";").explode().value_counts().index)
        elif kind == "да/нет":
            text = "True / False"
        else:
            text = f"{values.min():g} … {values.max():g}"
        rows[column] = {"тип": kind, "значения": text + ("; пусто — нет данных" if values.isna().any() else "")}
    rows["поездки (city.trips)"] = {"тип": "число, цель", "значения": f"{lab.trips.min():g} … {lab.trips.max():g}; "
                                    "пусто — не измерено; в city.roads не входит"}
    return (pd.DataFrame(rows).T.style
            .set_properties(**{"text-align": "left", "white-space": "normal"})
            .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]))


def example_rows(lab: Lab) -> pd.DataFrame:
    """The fewest rows in which every column has at least one filled value."""
    table = lab.roads.drop(columns="геометрия")
    filled = table.notna()
    left, chosen = set(filled.columns[filled.any()]), []
    while left:
        best = (100 * filled[list(left)].sum(axis=1) + filled.sum(axis=1)).idxmax()
        chosen.append(best)
        left -= set(filled.columns[filled.loc[best]])
    return table.loc[chosen]


def _typical_rows(table, n):
    """The most completely filled road of each of the ``n`` most common road types."""
    filled = table.notna().sum(axis=1)
    kinds = table["тип_дороги"].value_counts().index[:n]
    return [filled[table["тип_дороги"].eq(kind)].idxmax() for kind in kinds]


# --- filter -------------------------------------------------------------------------------------


@dataclass
class Network:
    lab: Lab
    roads: gpd.GeoDataFrame       # OSM roads routing uses: kept and connected
    removed: pd.Series            # removed by the filter
    cut_off: pd.Series            # kept, but not connected to the main piece: not used by routes

    def summary(self) -> str:
        km = lambda mask: self.lab.osm.length_m[mask].sum() / 1000
        nodes = set(self.roads.u) | set(self.roads.v)
        lost = int((~self.lab.pairs.origin.isin(nodes) | ~self.lab.pairs.destination.isin(nodes)).sum())
        text = (f"Осталось {len(self.roads):,} участков ({self.roads.length_m.sum()/1000:.0f} км) из {len(self.lab.osm):,}. "
                f"Фильтр убрал {int(self.removed.sum()):,} ({km(self.removed):.0f} км), "
                f"оторвано от основной сети и не используется {int(self.cut_off.sum()):,} ({km(self.cut_off):.0f} км).")
        return text + (f" Без маршрута останутся пары: {lost}." if lost else "")


def build_network(lab: Lab, keep=None) -> Network:
    """``keep(roads) -> Series[bool]``; None keeps the baseline network. Only the main connected piece is
    routed: from a cut-off piece you cannot reach the rest of the city (or come back, on one-way streets)."""
    mask = lab.reference if keep is None else keep(lab.roads) if callable(keep) else keep
    if not isinstance(mask, pd.Series) or not mask.index.equals(lab.osm.index) or mask.dtype != bool:
        raise ValueError("Фильтр должен вернуть pd.Series из True/False с тем же индексом, что у roads")
    kept = lab.osm[mask]
    connected = pd.Series(False, index=lab.osm.index)
    connected.loc[kept.index] = largest_component(kept, strongly=True).to_numpy()
    return Network(lab, lab.osm[connected], ~mask, mask & ~connected)


# --- solution blocks ----------------------------------------------------------------------------


def features(roads: pd.DataFrame) -> pd.DataFrame:
    """Ready features for weights, from OSM tags only (never from trips)."""
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


def feature_catalog() -> pd.DataFrame:
    return pd.DataFrame({
        "велодорожка": ("0/1", "тип_дороги = cycleway или велополоса на дороге"),
        "крупная_дорога": ("0/1", "тип_дороги = primary, secondary, trunk"),
        "жилая_улица": ("0/1", "тип_дороги = residential, living_street"),
        "пешеходная": ("0/1", "тип_дороги = footway, pedestrian, steps, path"),
        "плохое_покрытие": ("0/1", "брусчатка, грунт, гравий или плохая ровность"),
        "без_освещения": ("0/1", "освещение = no"),
        "узкая": ("0/1", "ширина_м < 2"),
        "скорость": ("км/ч", "макс_скорость; нет данных → 30"),
        "полосы": ("число", "полос; нет данных → 1"),
        "ширина": ("м", "ширина_м; нет данных → 3"),
        "скорость_неизвестна, полосы_неизвестны, ширина_неизвестна, покрытие_неизвестно": ("0/1", "в OSM нет тега"),
    }, index=["значения", "из чего"]).T


@dataclass
class ComfortCost:
    """``длина × max(0.1, 1 + Σ вес × признак)``: positive weights avoid a feature, negative prefer it."""
    weights: tuple
    extra: tuple = ()

    def table(self, roads):
        table = features(roads)
        view = russian(roads) if self.extra else None
        for make in self.extra:
            more = make(view)
            if not isinstance(more, pd.DataFrame) or not more.index.equals(table.index):
                raise ValueError("Свои признаки: функция должна вернуть pd.DataFrame с тем же индексом, что у roads")
            table[more.columns] = more
        return table

    def __call__(self, roads):
        table = self.table(roads)
        unknown = [name for name, _ in self.weights if name not in table]
        if unknown:
            raise KeyError(f"Нет признака {unknown}. Доступны: {list(table.columns.drop('длина_м'))}")
        factor = pd.Series(1.0, index=table.index)
        for name, weight in self.weights:
            factor += weight * table[name].astype(float)
        return (table["длина_м"] * factor.clip(lower=0.1)).rename("cost")


def comfort_cost(weights: dict, extra=()) -> ComfortCost:
    return ComfortCost(tuple(weights.items()), (extra,) if callable(extra) else tuple(extra))


@dataclass
class OwnCost:
    """A team's own cost function; it receives the Russian roads table."""
    function: object

    def __call__(self, roads):
        return pd.Series(self.function(russian(roads)), index=roads.index, name="cost")


@dataclass
class MLFeature:
    """Predicted popularity as a feature ``ml_популярность`` in 0..1 (share of roads predicted less popular)."""
    model: object  # anything with predict_log_trips(roads): the ready model or a team's TrainedModel

    def __call__(self, roads):
        predicted = self.model.predict_log_trips(osm(roads))
        return pd.DataFrame({"ml_популярность": predicted.rank(pct=True)}, index=roads.index)


def ml_feature(model) -> MLFeature:
    return MLFeature(model)


def _report(corr):
    text = "нет (прогноз одинаковый)" if not np.isfinite(corr) else f"{corr:.2f}"
    print(f"Связь прогноза с реальной популярностью на отложенных улицах: {text} (1 — идеально, 0 — никакой)")


def _held_out(lab, known, seed):
    """Measured roads and the ~20% of ~1 km blocks held out for the check; None when the area is too small."""
    roads = lab.osm[known]
    try:
        return roads, spatial_partition(roads, pd.Series(True, index=roads.index), random_state=seed).eq("validation")
    except ValueError:
        print("Проверка на отложенных улицах пропущена: слишком мало кварталов с измерениями.")
        return roads, None


def train_popularity(lab: Lab, *, max_iter=120, max_leaf_nodes=15, seed=42) -> PopularityModel:
    """The ready model: gradient boosting on OSM tags. Fit on measured open-zone roads; first report the fit on
    held-out ~1 km blocks."""
    known = lab.trips.notna()
    roads, held = _held_out(lab, known, seed)
    trips = lab.trips[known]
    if held is not None:
        check = fit_popularity_model(roads[~held], trips[~held], max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
                                     random_state=seed)
        _report(weighted_corr(check.predict_log_trips(roads[held]), np.log1p(trips[held]), roads.length_m[held]))
    return fit_popularity_model(roads, trips, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes, random_state=seed)


@dataclass
class TrainedModel:
    """A team's sklearn model fitted on the ready features plus their own ones."""
    model: object
    extra: tuple
    columns: list

    def features(self, roads):
        return ComfortCost((), self.extra).table(roads)[self.columns]

    def predict_log_trips(self, roads):
        return pd.Series(self.model.predict(self.features(roads)), index=roads.index, name="predicted_log_trips")


def train_model(lab: Lab, model, *, extra=(), seed=42) -> TrainedModel:
    """Your own model: any sklearn object with fit/predict (a regressor or a Pipeline). It learns log(1 + trips) of
    measured open-zone roads from the ready features and your own (``extra=my_features``); first the fit is checked
    on held-out ~1 km blocks. ``ml_feature(...)`` turns its prediction into the feature ``ml_популярность``."""
    from sklearn.base import clone
    extra = (extra,) if callable(extra) else tuple(extra)
    known = lab.trips.notna()
    table = ComfortCost((), extra).table(lab.osm)
    columns = [c for c in table.columns if c != "длина_м"]
    x, y = table.loc[known, columns], np.log1p(lab.trips[known])
    roads, held = _held_out(lab, known, seed)
    print("Признаки модели:", ", ".join(columns))
    if held is not None and held.any() and not held.all():
        check = clone(model).fit(x[~held.to_numpy()], y[~held.to_numpy()])
        _report(weighted_corr(check.predict(x[held.to_numpy()]), y[held.to_numpy()], roads.length_m[held]))
    return TrainedModel(clone(model).fit(x, y), extra, columns)


def solution(cost, *, turns="default", search="dijkstra", label=None) -> dict:
    if callable(cost) and not isinstance(cost, (ComfortCost, OwnCost)):
        cost = OwnCost(cost)
    return {"label": label or "моё решение", "cost": cost, "turns": turns, "search": search}


def presets(*names) -> dict:
    return {name: COMFORT_PRESETS[name] for name in (names or tuple(COMFORT_PRESETS))}


def _example_roads(lab, table, names, n):
    """A plain typical road, then a road where each named feature is present, padded with typical roads."""
    typical = _typical_rows(lab.roads, n)
    filled, chosen = lab.roads.notna().sum(axis=1), []
    for name in names:
        present = filled[table[name].ne(0) & ~filled.index.isin(chosen)]
        if len(present):
            chosen.append(present.idxmax())
    plain = [row for row in typical if row not in chosen and table.loc[row, names].eq(0).all()][:1]
    return plain + chosen + [row for row in typical if row not in chosen + plain][: max(0, n - len(chosen) - len(plain))]


def preview_features(lab: Lab, extra, n: int = 5) -> pd.DataFrame:
    """Roads where your own features are present: check them before giving them a weight."""
    table = comfort_cost({}, extra=extra).table(lab.osm)
    names = [c for c in table.columns if c not in features(lab.osm.iloc[:0])]
    return lab.roads.loc[_example_roads(lab, table, names, n), ["тип_дороги"]].join(table[names])


def preview_cost(lab: Lab, cost, n: int = 5, extra=()) -> pd.DataFrame:
    """Quick check without routing: a road for each feature that counts, the features and the cost in the last
    column. For your own formula pass ``extra=my_features`` to see your features next to the cost."""
    if isinstance(cost, ComfortCost):
        values = cost(lab.osm)
        table, names = cost.table(lab.osm), [name for name, _ in cost.weights]
    else:
        values = (cost if isinstance(cost, OwnCost) else OwnCost(cost))(lab.osm)
        table = comfort_cost({}, extra=extra).table(lab.osm)
        names = [c for c in table.columns if c not in features(lab.osm.iloc[:0])]
    rows = _example_roads(lab, table, names, n)
    result = lab.roads.loc[rows, ["тип_дороги"]].join(table.loc[rows, names + ["длина_м"]])
    result["цена 1 м"] = (values.loc[rows] / result["длина_м"]).round(2)
    result["стоимость"] = values.loc[rows].round(1)
    return result.round({"длина_м": 1})


# --- scoring ------------------------------------------------------------------------------------


@dataclass
class Result:
    table: pd.DataFrame
    per_od: pd.DataFrame
    routes: gpd.GeoDataFrame
    solutions: dict
    lab: Lab
    network: Network

    def plot(self):
        import matplotlib.pyplot as plt
        values = self.table["популярность"].sort_values()
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(values) + 1), layout="constrained")
        colors = ["#9ca3af" if name == "кратчайший" else "#2563eb" for name in values.index]
        ax.barh(values.index, values.to_numpy(), color=colors)
        ax.axvline(values["кратчайший"], color="#6b7280", linestyle=":")
        ax.set_xlim(max(0, values.min() - 5), min(100, values.max() + 3))
        ax.set_xlabel("популярность улиц маршрута, 0–100")
        ax.spines[["top", "right"]].set_visible(False)
        return fig


def _per_od(lab, routes, names, max_detour_pct):
    lookup = pd.DataFrame({"length": lab.osm.length_m.to_numpy(), "pop": lab.popularity.to_numpy(),
                           "bad": (~is_rideable(lab.osm)).to_numpy()}, index=pd.MultiIndex.from_frame(lab.osm[KEY]))
    rows = []
    for route in routes.itertuples():
        used = lookup.loc[[tuple(int(x) for x in edge) for edge in route.edge_route]]
        known = used["pop"].notna()
        rows.append({"algorithm": route.algorithm, "od_id": route.od_id, "length_m": route.route_length_m,
                     "turns": route.turns, "search_ms": route.search_ms, "bad_m": used.length[used.bad].sum(),
                     "coverage_pct": 100 * used.length[known].sum() / used.length.sum(),
                     "popularity": np.average(used["pop"].fillna(UNKNOWN_POPULARITY), weights=used.length)})
    grid = pd.MultiIndex.from_product([names, lab.pairs.od_id], names=["algorithm", "od_id"])
    d = pd.DataFrame(rows).set_index(["algorithm", "od_id"]).reindex(grid).reset_index()
    base = d[d.algorithm.eq("shortest")].set_index("od_id")
    d = d.join(base[["length_m", "popularity"]].add_prefix("shortest_"), on="od_id")
    d["detour_pct"] = 100 * (d.length_m / d.shortest_length_m - 1)
    missing, long = d.length_m.isna(), d.detour_pct.gt(max_detour_pct + 1e-9)
    d["counted"] = np.select([missing, long], [0.0, d.shortest_popularity], d.popularity)
    d["note"] = np.select([missing, long], ["нет маршрута", f"объезд больше {max_detour_pct}%: в оценку идёт кратчайший"], "")
    return d


def _table(per_od, solutions):
    base = per_od[per_od.algorithm.eq("shortest")].set_index("od_id").counted
    rows = {}
    for name in ["shortest", *solutions]:
        d = per_od[per_od.algorithm.eq(name)].set_index("od_id")
        better = int((d.counted > base.reindex(d.index) + 0.5).sum())
        rows["кратчайший" if name == "shortest" else name] = {
            "популярность": d.counted.mean(),
            "популярнее кратчайшего": "—" if name == "shortest" else f"{better} из {len(d)}",
            "объезд, %": d.detour_pct.median(),
            "поворотов": d.turns.mean(),
            "неудобные участки, м": d.bad_m.sum(),
            "время поиска, мс": d.search_ms.median(),
        }
    table = pd.DataFrame(rows).T.sort_values("популярность", ascending=False)
    digits = {"популярность": 1, "объезд, %": 1, "поворотов": 1, "неудобные участки, м": 0, "время поиска, мс": 2}
    return table.astype({c: float for c in digits}).round(digits)


def score(lab: Lab, solutions: dict, *, network: Network | None = None, max_detour_pct=MAX_DETOUR_PCT,
          progress: bool = True) -> Result:
    """Route every solution on every OD and compare with the shortest path of the baseline network.

    популярность — average over OD of the route popularity 0–100 weighted by length; a road without
    measurement counts as an average one (50). A route longer than the shortest by more than
    ``max_detour_pct`` counts as the shortest; no route counts 0.
    """
    if "shortest" in solutions:
        raise ValueError("Имя shortest зарезервировано для кратчайшего пути")
    network = network or build_network(lab)
    cache = lab.folder / "route_cache"
    frames = [build_routes(lab.osm[lab.reference], lab.nodes, lab.pairs, {}, cache_dir=cache)]
    if solutions:
        own = build_routes(network.roads, lab.nodes, lab.pairs, solutions, cache_dir=cache, progress=progress)
        frames.append(own[own.algorithm.ne("shortest")])
    routes = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=lab.osm.crs)
    per_od = _per_od(lab, routes, ["shortest", *solutions], max_detour_pct)
    return Result(_table(per_od, solutions), per_od, routes, solutions, lab, network)


def gpx_overlap(routes: gpd.GeoDataFrame, tracks: gpd.GeoDataFrame, buffer_m: float = 30) -> pd.DataFrame:
    """Share of each route's length within ``buffer_m`` of the nearest ride (best single GPX activity)
    and of any ride."""
    rides = tracks.to_crs(routes.crs).dissolve("activity_id").buffer(buffer_m)
    everywhere = rides.union_all()
    rows = []
    for route in routes.itertuples():
        shares = rides.map(lambda ride: route.geometry.intersection(ride).length) / route.geometry.length
        rows.append({"algorithm": route.algorithm, "od_id": route.od_id, "nearest_pct": 100 * shares.max(),
                     "ride": shares.idxmax(), "any_pct": 100 * route.geometry.intersection(everywhere).length / route.geometry.length})
    return pd.DataFrame(rows)


def gpx_table(result: Result, tracks: gpd.GeoDataFrame, buffer_m: float = 30) -> pd.DataFrame:
    """Per solution: how much of the route follows the nearest real ride and any real ride."""
    overlap = gpx_overlap(result.routes, tracks, buffer_m)
    names = {"shortest": "кратчайший", **{name: name for name in result.solutions}}
    table = overlap.groupby("algorithm")[["nearest_pct", "any_pct"]].mean().rename(index=names)
    table.columns = [f"по ближайшему GPX-треку, %", "по любым GPX-трекам, %"]
    return table.sort_values(table.columns[0], ascending=False).round(0)


# --- export -------------------------------------------------------------------------------------


def export_solution(lab: Lab, chosen: dict, *, keep=None, folder) -> Path:
    """Save the solution (functions included) and the filter; the organizer runs it on the closed zone."""
    import cloudpickle
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload = cloudpickle.dumps({"solution": chosen, "keep": keep})
    cloudpickle.loads(payload)
    (folder / "solution.pkl").write_bytes(payload)
    (folder / "submission.json").write_text(json.dumps({
        "dataset_id": lab.manifest["dataset_id"], "label": chosen.get("label"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder


def load_submission(folder):
    import cloudpickle
    payload = cloudpickle.loads((Path(folder) / "solution.pkl").read_bytes())
    return payload["solution"], payload["keep"]


def grade(lab: Lab, private_folder, submission) -> Result:
    """Organizer: run an exported solution on the closed zone with the same rules."""
    meta = json.loads((Path(submission) / "submission.json").read_text(encoding="utf-8"))
    if meta["dataset_id"] != lab.manifest["dataset_id"]:
        raise ValueError("Решение подготовлено для другого набора данных")
    closed = closed_city(lab, private_folder)
    chosen, keep = load_submission(submission)
    return score(closed, {"решение": chosen}, network=build_network(closed, keep))
