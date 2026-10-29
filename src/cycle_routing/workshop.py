"""Student API for the hackathon notebook: prepared data, solution blocks, one scoring table, export.

The notebook sees labels of the open zone only. The organizer keeps closed-zone OD and labels and
scores an exported solution with exactly the same ``score``.

Students see roads with Russian column names (``city.roads``); files and the routing library keep the
OSM names (``city.osm``). Their own functions always receive the Russian table.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import hashlib
import json
from pathlib import Path
import re

import geopandas as gpd
import numpy as np
import pandas as pd

from .evaluation import weighted_corr
from .graph import is_rideable, largest_component
from .learning import PopularityModel, fit_popularity_model, spatial_partition
from .pareto import pareto_table, plot_pareto
from .presets import COMFORT_PRESETS
from .routing import build_routes
from .tags import has_any

KEY = ["u", "v", "key"]
FORMAT_VERSION = 2
UNKNOWN_POPULARITY = 50.0  # a road without measurement counts as an average measured road
TARGETS = {"trips", "has_observation", "popularity"}
RU = {"length_m": "длина_м", "highway": "тип_дороги", "surface": "покрытие", "smoothness": "ровность",
      "cycleway": "велополоса", "bicycle": "велосипед", "lit": "освещение", "service": "служебный_проезд",
      "access": "доступ", "maxspeed": "макс_скорость", "lanes": "полос", "width": "ширина_м", "oneway": "одностороннее"}
OSM = {ru: name for name, ru in RU.items()}


def russian(roads: pd.DataFrame) -> pd.DataFrame:
    """Student view: Russian column names and tag values, with underscores between words."""
    roads = osm(roads)
    view = roads[[c for c in [*RU, "geometry"] if c in roads]].rename(columns=RU)
    for column, mapping in VALUE_NAMES.items():
        if column in view:
            view[column] = _translate_values(view[column], mapping)
    return view.rename_geometry("геометрия") if isinstance(view, gpd.GeoDataFrame) else view


def osm(roads: pd.DataFrame) -> pd.DataFrame:
    """Convert the student view back to OSM codes for routing and ready models."""
    table = roads.rename(columns=OSM)
    for column, mapping in OSM_VALUES.items():
        if column in roads:
            table[OSM[column]] = _translate_values(table[OSM[column]], mapping)
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
class CityData:
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


def load_city(folder: str | Path) -> CityData:
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
    return CityData(folder, roads, nodes, pd.read_parquet(folder / "pairs.parquet"),
               _align(roads, pd.read_parquet(folder / "labels.parquet")), manifest, tracks, test_area)


def closed_city(city: CityData, private_folder: str | Path) -> CityData:
    """Organizer only: the same roads with the closed-zone OD and labels."""
    private = Path(private_folder)
    meta = json.loads((private / "dataset.json").read_text(encoding="utf-8"))
    if meta["dataset_id"] != city.manifest["dataset_id"]:
        raise ValueError("Закрытые данные относятся к другому набору")
    return CityData(city.folder, city.osm, city.nodes, pd.read_parquet(private / "closed_pairs.parquet"),
               _align(city.osm, pd.read_parquet(private / "closed_labels.parquet")), city.manifest,
               name="закрытая зона")


KINDS = {"длина_м": "число, м", "макс_скорость": "число, км/ч", "полос": "число", "ширина_м": "число, м",
         "одностороннее": "да/нет", "геометрия": "линия"}

# One reversible vocabulary for the student data and its dictionary.
VALUE_LABELS = {
    "тип_дороги": {
        "cycleway": "велодорожка", "footway": "пешеходная дорожка", "path": "тропа",
        "service": "служебный проезд", "residential": "жилая улица", "living_street": "жилая зона",
        "pedestrian": "пешеходная улица", "steps": "лестница", "track": "хозяйственная дорога",
        "primary": "главная дорога", "secondary": "дорога второго значения", "tertiary": "местная дорога",
        "trunk": "скоростная дорога", "motorway": "автомагистраль", "unclassified": "прочая местная дорога",
        "road": "дорога неизвестного типа", "bridleway": "конная дорожка", "bus_stop": "автобусная остановка",
        "corridor": "коридор", "primary_link": "съезд с главной дороги",
        "secondary_link": "съезд с дороги второго значения", "tertiary_link": "съезд с местной дороги",
        "trunk_link": "съезд со скоростной дороги", "motorway_link": "съезд с автомагистрали",
        "construction": "строящаяся дорога", "proposed": "предлагаемая дорога", "planned": "планируемая дорога",
        "razed": "снесённая дорога", "abandoned": "заброшенная дорога", "platform": "платформа",
        "elevator": "лифт", "escalator": "эскалатор", "bus_guideway": "направляемая автобусная дорога",
        "raceway": "гоночная трасса", "busway": "автобусная дорога",
    },
    "покрытие": {
        "asphalt": "асфальт", "paved": "твёрдое покрытие", "unpaved": "нет твёрдого покрытия",
        "concrete": "бетон", "concrete:lanes": "бетонные колеи", "concrete:lines": "бетонные полосы",
        "concrete:plates": "бетонные плиты", "paving_stones": "тротуарная плитка", "sett": "брусчатка",
        "cobblestone": "булыжник", "unhewn_cobblestone": "необработанный булыжник", "compacted": "уплотнённая смесь",
        "fine_gravel": "мелкий гравий", "gravel": "гравий", "ground": "грунт", "dirt": "грунт без покрытия",
        "earth": "земля", "sand": "песок", "grass": "трава", "grass_paver": "газонная решётка",
        "metal": "металл", "metal_grid": "металлическая решётка", "mud": "грязь", "pebblestone": "галька",
        "rock": "скала", "stone": "камень", "rubber": "резина", "tartan": "спортивное покрытие",
        "wood": "дерево", "woodchips": "щепа", "artificial_turf": "искусственный газон", "plastic": "пластик",
        "building_passage": "проезд через здание (нестандартный тег)", "service": "служебный проезд (нестандартный тег)",
        "3": "неопределённое покрытие (нестандартный тег)",
    },
    "ровность": {
        "excellent": "отличная", "very_good": "очень хорошая", "good": "хорошая", "intermediate": "средняя",
        "bad": "плохая", "very_bad": "очень плохая", "horrible": "ужасная", "very_horrible": "крайне плохая",
        "impassable": "непроезжая", "thin_rollers": "для маленьких колёс", "thin_wheels": "для узких колёс",
        "wheels": "для обычных колёс", "robust_wheels": "для прочных колёс", "high_clearance": "нужен высокий клиренс",
        "off_road_wheels": "для внедорожных колёс", "specialized_off_road_wheels": "для специальных внедорожных колёс",
    },
    "велополоса": {
        "yes": "есть", "no": "нет", "none": "отсутствует", "lane": "велополоса", "track": "отделённая велодорожка",
        "separate": "велодорожка отдельной линией OSM", "share_busway": "совместно с автобусами",
        "shared_lane": "совместная полоса с машинами", "crossing": "велопереезд", "designated": "предназначена для велосипедов",
    },
    "велосипед": {
        "yes": "разрешён", "no": "запрещён", "designated": "предназначено для велосипедов",
        "destination": "только к месту назначения", "permissive": "с разрешения владельца", "dismount": "спешиться",
        "discouraged": "не рекомендуется", "use_sidepath": "использовать соседнюю велодорожку",
        "agricultural": "только для сельхозработ", "forestry": "только для лесных работ", "unknown": "неизвестно",
        "private": "частный доступ",
    },
    "освещение": {"yes": "есть", "no": "нет", "24/7": "круглосуточное", "automatic": "автоматическое", "limited": "ограниченное"},
    "служебный_проезд": {
        "driveway": "подъезд к зданию", "parking_aisle": "проезд по парковке", "alley": "узкий проезд",
        "drive-through": "проезд через обслуживание", "emergency_access": "проезд экстренных служб",
        "slipway": "спуск к воде", "tiefgarageneinfahrt": "въезд в подземный гараж", "yes": "служебный проезд",
        "living_street": "проезд в жилой зоне",
        "private": "частный проезд",
    },
    "доступ": {
        "yes": "разрешён", "no": "запрещён", "private": "частный", "permissive": "с разрешения владельца",
        "destination": "только к месту назначения", "customers": "только для клиентов", "delivery": "только для доставки",
        "permit": "по пропуску", "designated": "для указанного транспорта", "agricultural": "только для сельхозработ",
        "forestry": "только для лесных работ",
    },
}


VALUE_NAMES = {
    column: {value: re.sub(r"[^\w]+", "_", label.lower()).strip("_") for value, label in mapping.items()}
    for column, mapping in VALUE_LABELS.items()
}
OSM_VALUES = {column: {name: value for value, name in mapping.items()} for column, mapping in VALUE_NAMES.items()}


def _translate_values(values, mapping):
    # Translate each token independently: aggregated tags and missing values keep their meaning.
    lookup = {value: ";".join(mapping.get(part, part) for part in value.split(";"))
              for value in values.dropna().unique()}
    translated = values.map(lookup, na_action="ignore")
    if values.dtype == object or isinstance(values.dtype, pd.StringDtype):
        return translated.astype(values.dtype)
    return translated


def _crs_description(crs):
    """Explain the geometry using familiar words; keep technical CRS metadata in the data files."""
    if crs is None:
        return "линия дороги; единицы координат не указаны"
    units = {axis.unit_name for axis in crs.axis_info}
    if units == {"metre"}:
        return "линия дороги; координаты в метрах"
    if units == {"degree"}:
        return "линия дороги; координаты в градусах"
    return "линия дороги"


def data_dictionary(city: CityData):
    """One row per column of ``city.roads``, like a competition data page, with every value of each category
    (most frequent first); the target is listed last. Shown wrapped, nothing cut."""
    rows = {}
    for column in city.roads.columns:
        kind, values = KINDS.get(column, "категория"), city.roads[column]
        if column == "геометрия":
            text = _crs_description(city.osm.crs)
        elif kind == "категория":
            text = ", ".join(values.dropna().astype("string").str.split(";").explode().value_counts().index)
        elif kind == "да/нет":
            text = "да (True) / нет (False)"
        else:
            text = f"{values.min():g} … {values.max():g}"
        rows[column] = {"тип": kind, "значения": text + ("; пусто — нет данных" if values.isna().any() else "")}
    rows["поездки (city.trips)"] = {"тип": "число, цель", "значения": f"{city.trips.min():g} … {city.trips.max():g}; "
                                    "пусто — не измерено; в city.roads не входит"}
    return (pd.DataFrame(rows).T.style
            .set_properties(**{"text-align": "left", "white-space": "normal"})
            .set_table_styles([{"selector": "th", "props": [("text-align", "left")]}]))


def example_rows(city: CityData) -> pd.DataFrame:
    """The fewest rows in which every column has at least one filled value."""
    table = city.roads.drop(columns="геометрия")
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
    city: CityData
    roads: gpd.GeoDataFrame       # OSM roads routing uses: kept and connected
    removed: pd.Series            # removed by the filter
    cut_off: pd.Series            # kept, but not connected to the main piece: not used by routes

    def summary(self) -> str:
        km = lambda mask: self.city.osm.length_m[mask].sum() / 1000
        nodes = set(self.roads.u) | set(self.roads.v)
        lost = int((~self.city.pairs.origin.isin(nodes) | ~self.city.pairs.destination.isin(nodes)).sum())
        text = (f"Осталось {len(self.roads):,} участков ({self.roads.length_m.sum()/1000:.0f} км) из {len(self.city.osm):,}. "
                f"Фильтр убрал {int(self.removed.sum()):,} ({km(self.removed):.0f} км), "
                f"оторвано от основной сети и не используется {int(self.cut_off.sum()):,} ({km(self.cut_off):.0f} км).")
        return text + (f" Без маршрута останутся пары: {lost}." if lost else "")


def build_network(city: CityData, keep=None) -> Network:
    """``keep(roads) -> Series[bool]``; None keeps the baseline network. Only the main connected piece is
    routed: from a cut-off piece you cannot reach the rest of the city (or come back, on one-way streets)."""
    mask = city.reference if keep is None else keep(city.roads) if callable(keep) else keep
    if not isinstance(mask, pd.Series) or not mask.index.equals(city.osm.index) or mask.dtype != bool:
        raise ValueError("Фильтр должен вернуть pd.Series из True/False с тем же индексом, что у roads")
    kept = city.osm[mask]
    connected = pd.Series(False, index=city.osm.index)
    connected.loc[kept.index] = largest_component(kept, strongly=True).to_numpy()
    return Network(city, city.osm[connected], ~mask, mask & ~connected)


# --- solution blocks ----------------------------------------------------------------------------


def features(roads: pd.DataFrame) -> pd.DataFrame:
    """Length and three example indicators; the notebook supplies its editable function."""
    roads = russian(roads)
    f = pd.DataFrame(index=roads.index)
    f["длина_м"] = roads["длина_м"]
    f["велодорожка"] = has_any(roads["тип_дороги"], "велодорожка").astype(float)
    f["крупная_дорога"] = has_any(
        roads["тип_дороги"], {"главная_дорога", "дорога_второго_значения", "скоростная_дорога"}
    ).astype(float)
    f["асфальт"] = has_any(roads["покрытие"], "асфальт").astype(float)
    return f


def _feature_table(roads, make_features):
    table = make_features(russian(roads))
    if not isinstance(table, pd.DataFrame) or not table.index.equals(roads.index):
        raise ValueError("Признаки: функция должна вернуть pd.DataFrame с тем же индексом, что у roads")
    return table.copy()


@dataclass
class ComfortCost:
    """``длина × max(0.1, 1 + Σ вес × признак)``: positive weights avoid a feature, negative prefer it."""
    weights: tuple
    make_features: object = features

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "make_features" not in state:
            from .workshop_compat import restore_features
            self.make_features = restore_features(state, features)

    def table(self, roads):
        return _feature_table(roads, self.make_features)

    def __call__(self, roads):
        table = self.table(roads)
        unknown = [name for name, _ in self.weights if name not in table]
        if unknown:
            raise KeyError(f"Нет признака {unknown}. Доступны: {list(table.columns)}")
        factor = pd.Series(1.0, index=table.index)
        for name, weight in self.weights:
            factor += weight * table[name].astype(float)
        length = table["длина_м"] if "длина_м" in table else (
            roads["длина_м"] if "длина_м" in roads else roads["length_m"]
        )
        return (length * factor.clip(lower=0.1)).rename("cost")


def comfort_cost(weights: dict, *, features=features) -> ComfortCost:
    return ComfortCost(tuple(weights.items()), features)


@dataclass
class OwnCost:
    """A team's own cost function; it receives the Russian roads table."""
    function: object
    _russian_values: bool = True

    def __call__(self, roads):
        view = russian(roads)
        if not vars(self).get("_russian_values", False):
            from .workshop_compat import legacy_roads
            view = legacy_roads(roads)
        return pd.Series(self.function(view), index=roads.index, name="cost")


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


def _held_out(city, known, seed):
    """Measured roads and the ~20% of ~1 km blocks held out for the check; None when the area is too small."""
    roads = city.osm[known]
    try:
        return roads, spatial_partition(roads, pd.Series(True, index=roads.index), random_state=seed).eq("validation")
    except ValueError:
        print("Проверка на отложенных улицах пропущена: слишком мало кварталов с измерениями.")
        return roads, None


def train_popularity(city: CityData, *, max_iter=120, max_leaf_nodes=15, seed=42) -> PopularityModel:
    """The ready model: gradient boosting on OSM tags. Fit on measured open-zone roads; first report the fit on
    held-out ~1 km blocks."""
    known = city.trips.notna()
    roads, held = _held_out(city, known, seed)
    trips = city.trips[known]
    if held is not None:
        check = fit_popularity_model(roads[~held], trips[~held], max_iter=max_iter, max_leaf_nodes=max_leaf_nodes,
                                     random_state=seed)
        _report(weighted_corr(check.predict_log_trips(roads[held]), np.log1p(trips[held]), roads.length_m[held]))
    return fit_popularity_model(roads, trips, max_iter=max_iter, max_leaf_nodes=max_leaf_nodes, random_state=seed)


@dataclass
class TrainedModel:
    """A team's fitted model and the one feature function shared by training and prediction."""
    model: object
    columns: list
    make_features: object = features

    def __setstate__(self, state):
        self.__dict__.update(state)
        if "make_features" not in state:
            from .workshop_compat import restore_features
            self.make_features = restore_features(state, features)

    def features(self, roads):
        return _feature_table(roads, self.make_features)[self.columns]

    def predict_log_trips(self, roads):
        return pd.Series(self.model.predict(self.features(roads)), index=roads.index, name="predicted_log_trips")


def train_custom_model(city: CityData, train, *, features=features, seed=42) -> TrainedModel:
    """Prepare OSM features and log(1 + trips), then call ``train(x, y) -> fitted model``.

    The notebook owns fitting. This service handles missing labels, spatial validation and reuse of
    the same features for prediction. ``train`` creates a fresh model on each call.
    """
    known = city.trips.notna()
    table = _feature_table(city.osm, features)
    columns = list(table.columns)
    x, y = table.loc[known, columns], np.log1p(city.trips[known])
    roads, held = _held_out(city, known, seed)
    print("Признаки модели:", ", ".join(columns))
    if held is not None and held.any() and not held.all():
        check = train(x.loc[~held], y.loc[~held])
        _report(weighted_corr(check.predict(x[held.to_numpy()]), y[held.to_numpy()], roads.length_m[held]))
    return TrainedModel(train(x, y), columns, features)


def solution(cost, *, turns="default", search="dijkstra", label=None) -> dict:
    if callable(cost) and not isinstance(cost, (ComfortCost, OwnCost)):
        cost = OwnCost(cost)
    return {"label": label or "моё решение", "cost": cost, "turns": turns, "search": search}


def presets(*names) -> dict:
    return {name: COMFORT_PRESETS[name] for name in (names or tuple(COMFORT_PRESETS))}


def _example_roads(city, table, names, n):
    """A plain typical road, then a road where each named feature is present, padded with typical roads."""
    typical = _typical_rows(city.roads, n)
    filled, chosen = city.roads.notna().sum(axis=1), []
    for name in names:
        present = filled[table[name].ne(0) & ~filled.index.isin(chosen)]
        if len(present):
            chosen.append(present.idxmax())
    plain = [row for row in typical if row not in chosen and table.loc[row, names].eq(0).all()][:1]
    rows = plain + chosen + [row for row in typical if row not in chosen + plain]
    return rows[:n]


def preview_features(city: CityData, features=features, n: int = 5) -> pd.DataFrame:
    """Show the supplied features on representative roads, before choosing their weights."""
    table = _feature_table(city.osm, features)
    names = [c for c in table.columns if c != "длина_м"]
    return city.roads.loc[_example_roads(city, table, names, n), ["тип_дороги"]].join(table)


def preview_cost(city: CityData, cost, n: int = 5, *, features=features) -> pd.DataFrame:
    """Quick check without routing: a road for each feature that counts, the features and the cost in the last
    column. ``features`` is the same single feature function used by the solution."""
    if isinstance(cost, ComfortCost):
        values = cost(city.osm)
        table, names = cost.table(city.osm), [name for name, _ in cost.weights]
    else:
        values = (cost if isinstance(cost, OwnCost) else OwnCost(cost))(city.osm)
        table = _feature_table(city.osm, features)
        names = [c for c in table.columns if c != "длина_м"]
    rows = _example_roads(city, table, names, n)
    result = city.roads.loc[rows, ["тип_дороги"]].join(table.loc[rows, names])
    result["длина_м"] = city.roads.loc[rows, "длина_м"]
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
    city: CityData
    network: Network

    @property
    def pareto(self):
        """Unrounded popularity/distance comparison on every original OD."""
        return pareto_table(self.per_od, {"shortest": "кратчайший"})

    def plot_pareto(self):
        return plot_pareto(self.pareto)

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


def _per_od(city, routes, names):
    lookup = pd.DataFrame({"length": city.osm.length_m.to_numpy(), "pop": city.popularity.to_numpy(),
                           "bad": (~is_rideable(city.osm)).to_numpy()}, index=pd.MultiIndex.from_frame(city.osm[KEY]))
    rows = []
    for route in routes.itertuples():
        used = lookup.loc[[tuple(int(x) for x in edge) for edge in route.edge_route]]
        known = used["pop"].notna()
        rows.append({"algorithm": route.algorithm, "od_id": route.od_id, "length_m": route.route_length_m,
                     "turns": route.turns, "search_ms": route.search_ms, "bad_m": used.length[used.bad].sum(),
                     "coverage_pct": 100 * used.length[known].sum() / used.length.sum(),
                     "popularity": np.average(used["pop"].fillna(UNKNOWN_POPULARITY), weights=used.length)})
    grid = pd.MultiIndex.from_product([names, city.pairs.od_id], names=["algorithm", "od_id"])
    d = pd.DataFrame(rows).set_index(["algorithm", "od_id"]).reindex(grid).reset_index()
    base = d[d.algorithm.eq("shortest")].set_index("od_id")
    d = d.join(base[["length_m", "popularity"]].add_prefix("shortest_"), on="od_id")
    d["detour_pct"] = 100 * (d.length_m / d.shortest_length_m - 1)
    missing = d.length_m.isna()
    d["counted"] = d.popularity.where(~missing, 0.0)
    d["note"] = np.where(missing, "нет маршрута", "")
    return d


def _table(per_od, solutions):
    base = per_od[per_od.algorithm.eq("shortest")].set_index("od_id").counted
    comparison = pareto_table(per_od)
    rows = {}
    for name in ["shortest", *solutions]:
        d = per_od[per_od.algorithm.eq(name)].set_index("od_id")
        better = int((d.counted > base.reindex(d.index) + 0.5).sum())
        rows["кратчайший" if name == "shortest" else name] = {
            "Парето": bool(comparison.loc[name, "Парето"]),
            "популярность": d.counted.mean(),
            "длина, км": comparison.loc[name, "длина, км"],
            "маршрутов": f"{comparison.loc[name, 'маршрутов']} из {comparison.loc[name, 'всего пар']}",
            "популярнее кратчайшего": "—" if name == "shortest" else f"{better} из {len(d)}",
            "объезд, %": d.detour_pct.median(),
            "поворотов": d.turns.mean(),
            "неудобные участки, м": d.bad_m.sum(),
            "время поиска, мс": d.search_ms.median(),
        }
    table = pd.DataFrame(rows).T
    digits = {"популярность": 1, "длина, км": 3, "объезд, %": 1, "поворотов": 1, "неудобные участки, м": 0, "время поиска, мс": 2}
    table = table.astype({"Парето": bool, **{c: float for c in digits}})
    return table.sort_values(["Парето", "длина, км"], ascending=[False, True]).round(digits)


def score(city: CityData, solutions: dict, *, network: Network | None = None, progress: bool = True) -> Result:
    """Route every solution on every OD and compare with the shortest path of the baseline network.

    популярность — average over OD of the route popularity 0–100 weighted by length; a road without
    measurement counts as an average one (50). Actual popularity is counted regardless of detour;
    no route counts 0. The Pareto front maximizes popularity and minimizes mean physical distance,
    comparing only solutions that found a route for every original OD.
    """
    if "shortest" in solutions:
        raise ValueError("Имя shortest зарезервировано для кратчайшего пути")
    network = network or build_network(city)
    cache = city.folder / "route_cache"
    frames = [build_routes(city.osm[city.reference], city.nodes, city.pairs, {}, cache_dir=cache)]
    if solutions:
        own = build_routes(network.roads, city.nodes, city.pairs, solutions, cache_dir=cache, progress=progress)
        frames.append(own[own.algorithm.ne("shortest")])
    routes = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), geometry="geometry", crs=city.osm.crs)
    per_od = _per_od(city, routes, ["shortest", *solutions])
    return Result(_table(per_od, solutions), per_od, routes, solutions, city, network)


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


def export_solution(city: CityData, chosen: dict, *, keep=None, folder) -> Path:
    """Save the solution (functions included) and the filter; the organizer runs it on the closed zone."""
    import cloudpickle
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    payload = cloudpickle.dumps({"solution": chosen, "keep": keep, "data_language": "ru_values_v1"})
    cloudpickle.loads(payload)
    (folder / "solution.pkl").write_bytes(payload)
    (folder / "submission.json").write_text(json.dumps({
        "dataset_id": city.manifest["dataset_id"], "label": chosen.get("label"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return folder


def load_submission(folder):
    import cloudpickle
    payload = cloudpickle.loads((Path(folder) / "solution.pkl").read_bytes())
    keep = payload["keep"]
    if payload.get("data_language") != "ru_values_v1" and keep is not None:
        from .workshop_compat import LegacyFilter
        keep = LegacyFilter(keep)
    return payload["solution"], keep


def grade(city: CityData, private_folder, submission) -> Result:
    """Organizer: run an exported solution on the closed zone with the same rules."""
    meta = json.loads((Path(submission) / "submission.json").read_text(encoding="utf-8"))
    if meta["dataset_id"] != city.manifest["dataset_id"]:
        raise ValueError("Решение подготовлено для другого набора данных")
    closed = closed_city(city, private_folder)
    chosen, keep = load_submission(submission)
    return score(closed, {"решение": chosen}, network=build_network(closed, keep))
