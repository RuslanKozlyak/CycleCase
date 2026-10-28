"""Small student API: prepared data, comfort features, solutions and a fixed evaluation protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from .evaluation import route_popularity_metrics, weighted_corr
from .graph import build_graph, edge_costs, is_rideable
from .learning import PopularityModel, popularity_cost
from .presets import COMFORT_PRESETS
from .routing import build_routes, count_turns
from .tags import has_any


def features(roads: pd.DataFrame) -> pd.DataFrame:
    """OSM-only features for formulas. Numeric defaults are explicit; missing flags are retained."""
    result = roads.drop(columns=[c for c in roads if c not in {
        "length_m", "highway", "surface", "smoothness", "cycleway", "bicycle", "lit", "service",
        "access", "maxspeed", "lanes", "width", "oneway",
    }]).copy()
    bike_values = roads.cycleway.fillna("").str.split(";")
    result["bike_infra"] = (has_any(roads.highway, "cycleway")
                            | bike_values.map(lambda values: bool(set(values) - {"", "no", "none", "separate"}))).astype(float)
    result["major_road"] = has_any(roads.highway, {"primary", "secondary", "trunk", "primary_link", "secondary_link"}).astype(float)
    result["rough"] = (has_any(roads.surface, {"sett", "cobblestone", "unpaved", "gravel", "ground", "dirt", "sand"})
                       | has_any(roads.smoothness, {"bad", "very_bad", "horrible", "impassable"})).astype(float)
    result["unlit"] = has_any(roads.lit, "no").astype(float)
    result["narrow"] = (roads.width.notna() & roads.width.lt(2)).astype(float)
    for source, name, default in [("maxspeed", "speed_kmh", 30.), ("lanes", "lanes", 1.), ("width", "width_m", 3.)]:
        values = pd.to_numeric(roads[source], errors="coerce")
        result[name + "_missing"] = values.isna()
        result[name] = values.fillna(default)
    result["surface_missing"] = roads.surface.isna()
    return result


def feature_catalog() -> pd.DataFrame:
    return pd.DataFrame([
        ("length_m", "Длина участка, м", "из OSM"),
        ("bike_infra", "Велодорожка или велополоса: 0/1", "неизвестное = 0"),
        ("major_road", "Крупная дорога: 0/1", "по highway"),
        ("rough", "Грубое покрытие или плохая ровность: 0/1", "неизвестное = 0"),
        ("unlit", "Явно неосвещённая улица: 0/1", "неизвестное = 0"),
        ("narrow", "Известная ширина менее 2 м: 0/1", "неизвестное = 0"),
        ("speed_kmh", "Ограничение скорости автомобилей, км/ч", "30; есть speed_kmh_missing"),
        ("lanes", "Количество полос", "1; есть lanes_missing"),
        ("width_m", "Ширина, м", "3; есть width_m_missing"),
        ("highway / surface / …", "Исходные OSM-теги", "NaN сохранён; has_any учитывает теги с ;"),
    ], columns=["признак", "смысл", "если данных нет"]).set_index("признак")


@dataclass(frozen=True)
class PriorityCost:
    traffic: float = 0.6
    rough: float = 0.4
    cycling: float = 0.7
    unlit: float = 0.2
    narrow: float = 0.2

    def __post_init__(self):
        if not all(np.isfinite(value) and value >= 0 for value in asdict(self).values()):
            raise ValueError("Приоритеты должны быть конечными и неотрицательными")

    def __call__(self, roads):
        f = features(roads)
        penalty = 1 + self.traffic*f.major_road + self.rough*f.rough + self.unlit*f.unlit + self.narrow*f.narrow
        return (f.length_m * penalty / (1 + self.cycling*f.bike_infra)).rename("cost")


def priority_cost(**priorities) -> PriorityCost:
    return PriorityCost(**priorities)


def preset(label: str, *, cost, turns="default", search="dijkstra") -> dict:
    return {"label": label, "cost": cost, "turns": turns, "search": search}


def presets(*names) -> dict:
    return {name: COMFORT_PRESETS[name] for name in (names or tuple(COMFORT_PRESETS))}


def preset_catalog() -> pd.DataFrame:
    return pd.DataFrame({name: {"название": p["label"], "идея": p["description"]}
                         for name, p in COMFORT_PRESETS.items()}).T


@dataclass(frozen=True)
class LearnedCost:
    model: PopularityModel
    strength: float = 0.5

    def __call__(self, roads):
        return popularity_cost(roads, self.model.predict_log_trips(roads), strength=self.strength,
                               reference_log=self.model.reference_log)


def learned_cost(model, *, strength=0.5):
    return LearnedCost(model, strength)


@dataclass(frozen=True)
class EvaluationRules:
    max_detour_pct: float = 20.
    min_coverage_pct: float = 10.

    def __post_init__(self):
        if not np.isfinite(self.max_detour_pct) or self.max_detour_pct < 0:
            raise ValueError("Лимит объезда должен быть конечным и неотрицательным")
        if not np.isfinite(self.min_coverage_pct) or not 0 < self.min_coverage_pct <= 100:
            raise ValueError("Порог покрытия должен быть в (0, 100]")


@dataclass
class CityLab:
    folder: Path
    edges: gpd.GeoDataFrame
    nodes: gpd.GeoDataFrame
    pairs: pd.DataFrame
    partition: pd.Series
    labels: dict[str, pd.Series]
    manifest: dict

    def labels_for(self, split="validation"):
        if split not in self.labels:
            raise ValueError(f"Метки {split!r} не выданы. Итоговый test запускает организатор.")
        return self.labels[split].copy()

    def training_data(self):
        y = self.labels_for("train")
        return self.edges[y.notna()].copy(), y[y.notna()]

    def summary(self):
        return pd.Series({"город": self.manifest["city"], "рёбер": len(self.edges), "узлов": len(self.nodes),
                          "пар A–B": len(self.pairs), **{f"меток {name}": int(y.notna().sum()) for name,y in self.labels.items()}},
                         name="подготовлено")


def load_lab(folder: str | Path) -> CityLab:
    """Read a prepared pack only. Never downloads or reconstructs the city in a student session."""
    folder = Path(folder)
    path = folder / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"Нет подготовленного набора {folder}. Его выдаёт организатор; инструкция в HACKATHON.md.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format_version") != 1:
        raise ValueError("Неподдерживаемая версия учебного набора")
    for name, digest in manifest["files"].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Изменён файл данных {name}; распакуйте исходный набор заново")
    edges, nodes = gpd.read_parquet(folder/"edges.parquet"), gpd.read_parquet(folder/"nodes.parquet")
    if {"trips", "has_observation", "popularity"} & set(edges):
        raise ValueError("Метки популярности не должны входить в таблицу признаков")
    partition = pd.read_parquet(folder/"partition.parquet")["split"].set_axis(edges.index)
    labels = {}
    for split in manifest["public_splits"]:
        values = pd.read_parquet(folder/f"labels_{split}.parquet")
        labels[split] = _align_labels(edges, values)
    return CityLab(folder, edges, nodes, pd.read_parquet(folder/"pairs.parquet"), partition, labels, manifest)


def _align_labels(edges, labels):
    keys = ["u", "v", "key"]
    if labels.duplicated(keys).any() or not np.isfinite(labels.trips).all() or labels.trips.lt(0).any():
        raise ValueError("Метки должны быть уникальными, известными и неотрицательными")
    values = labels.set_index(keys).trips.reindex(pd.MultiIndex.from_frame(edges[keys]))
    return pd.Series(values.to_numpy(), index=edges.index, name="trips")


def run_routes(lab, solutions, *, od_ids=None):
    if "shortest" in solutions:
        raise ValueError("Имя shortest зарезервировано для общей базы сравнения")
    pairs = lab.pairs if od_ids is None else lab.pairs[lab.pairs.od_id.isin(od_ids)]
    if pairs.empty: raise ValueError("Не найдены выбранные OD")
    return build_routes(lab.edges, lab.nodes, pairs, solutions, cache_dir=lab.folder/"route_cache")


def validate_routes(lab, routes):
    """Recompute lengths/turns and reject discontinuous paths or foreign roads/OD IDs."""
    if routes.duplicated(["algorithm", "od_id"]).any():
        raise ValueError("Для одного решения должна быть одна строка на OD")
    pairs = lab.pairs.set_index("od_id")
    graph = build_graph(lab.edges, lab.nodes, "length_m")
    result = routes.copy()
    result["route_length_m"] = result.route_length_m.astype(float)
    for index, route in result.iterrows():
        if route.od_id not in pairs.index: raise ValueError(f"Неизвестная OD: {route.od_id}")
        path = [tuple(int(n) for n in edge) for edge in route.edge_route]
        pair = pairs.loc[route.od_id]
        if (not path or path[0][0] != pair.origin or path[-1][1] != pair.destination
                or any(a[1] != b[0] for a,b in zip(path,path[1:]))
                or any(not graph.has_edge(*key) for key in path)):
            raise ValueError(f"Некорректный путь {route.algorithm}/{route.od_id}")
        result.at[index, "route_length_m"] = sum(graph.edges[key]["length_m"] for key in path)
        result.at[index, "turns"] = count_turns(graph, path)
    return result


@dataclass
class Comparison:
    routes: gpd.GeoDataFrame
    table: pd.DataFrame
    per_od: pd.DataFrame
    rules: EvaluationRules
    split: str
    solutions: dict

    def plot(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
        for name,row in self.table.iterrows():
            ax.scatter(row["объезд, %"], row["популярность × shortest"], s=80)
            ax.annotate(name, (row["объезд, %"], row["популярность × shortest"]), xytext=(5,5), textcoords="offset points")
        ax.axvline(self.rules.max_detour_pct, color="#dc2626", linestyle=":", label="лимит на каждую OD")
        ax.axhline(1, color="#9ca3af", linestyle=":")
        ax.set(xlabel="Медианный объезд по всем OD, %", ylabel="Медианная популярность относительно shortest",
               title=f"{self.split}: популярность и длина")
        ax.grid(alpha=.2); ax.legend()
        return fig


def evaluate_routes(lab, routes, *, split="validation", rules=None, labels=None, solutions=None):
    """Fixed baseline-eligible OD: candidates cannot remove cases from their denominator.

    Per-case points = 100*sigmoid(log popularity gain), with 0 points for a missing route,
    inadequate measurement coverage or detour over the cap. Baseline is exactly 50.
    """
    rules = rules or EvaluationRules()
    y = lab.labels_for(split) if labels is None else _align_labels(lab.edges, labels)
    routes = validate_routes(lab, routes)
    reference = run_routes(lab, {})
    candidates = routes[~routes.algorithm.eq("shortest")]
    routes = gpd.GeoDataFrame(pd.concat([reference, candidates], ignore_index=True), crs=lab.edges.crs)
    detail = route_popularity_metrics(routes, lab.edges, y, mask=y.notna())
    base = detail[detail.algorithm.eq("shortest")].set_index("od_id")
    fixed_ids = base.index[base.coverage_pct.ge(rules.min_coverage_pct) & base.popularity_log_mean.notna()]
    if not len(fixed_ids): raise ValueError(f"В {split} нет OD с достаточным покрытием; организатору нужно проверить набор")
    models = ["shortest", *dict.fromkeys([*(solutions or {}), *candidates.algorithm])]
    detail = detail.set_index(["algorithm", "od_id"]).reindex(pd.MultiIndex.from_product([models,lab.pairs.od_id],names=["algorithm","od_id"])).reset_index()
    detail = detail.join(base[["route_length_m", "popularity_log_mean"]].add_prefix("base_"), on="od_id")
    detail["detour_pct"] = 100*(detail.route_length_m/detail.base_route_length_m-1)
    detail["gain"] = detail.popularity_log_mean-detail.base_popularity_log_mean
    detail["in_evaluation"] = detail.od_id.isin(fixed_ids)
    detail["passed"] = detail.route_length_m.notna() & detail.coverage_pct.ge(rules.min_coverage_pct) & detail.detour_pct.le(rules.max_detour_pct+1e-7) & detail.gain.notna()
    detail["points"] = np.where(detail.passed,100/(1+np.exp(-detail.gain.clip(-50,50))),0.)
    detail["reason"] = np.select([detail.route_length_m.isna(),detail.coverage_pct.lt(rules.min_coverage_pct)|detail.gain.isna(),detail.detour_pct.gt(rules.max_detour_pct+1e-7)],
                                  ["нет маршрута","мало измерений","слишком большой объезд"],default="зачтено")
    rows=[]
    for model in models:
        full=detail[detail.algorithm.eq(model)]; counted=full[full.in_evaluation]
        rows.append({"решение":model,"балл / 100":counted.points.mean(),"зачтено OD":int(counted.passed.sum()),
                     "OD для оценки":len(fixed_ids),"маршрутов":int(full.route_length_m.notna().sum()),
                     "популярность × shortest":float(np.exp(counted.gain.median())),
                     "объезд, %":full.detour_pct.median(),"объезд p90, %":full.detour_pct.quantile(.9),
                     "покрытие, %":counted.coverage_pct.median()})
    table=pd.DataFrame(rows).set_index("решение").sort_values("балл / 100",ascending=False)
    return Comparison(routes,table,detail,rules,split,solutions or {})


def compare(lab, solutions, *, split="validation", rules=None):
    return evaluate_routes(lab,run_routes(lab,solutions),split=split,rules=rules,solutions=solutions)


def route_diagnostics(lab, result):
    """Explain changed paths using length shares of OSM attributes, independent of popularity labels."""
    f = features(lab.edges).set_axis(pd.MultiIndex.from_frame(lab.edges[["u", "v", "key"]]))
    base = result.routes[result.routes.algorithm.eq("shortest")].set_index("od_id").edge_route
    rows = []
    for route in result.routes.itertuples():
        path = [tuple(key) for key in route.edge_route]
        used = f.loc[path]
        reference = [tuple(key) for key in base.loc[route.od_id]]
        rows.append({"решение": route.algorithm, "OD изменено": int(path != reference),
                     "велоинфраструктура, %": np.average(used.bike_infra, weights=used.length_m)*100,
                     "крупные дороги, %": np.average(used.major_road, weights=used.length_m)*100,
                     "грубое покрытие, %": np.average(used.rough, weights=used.length_m)*100,
                     "поворотов / км": route.turns / (route.route_length_m/1000)})
    data = pd.DataFrame(rows)
    columns = ["велоинфраструктура, %", "крупные дороги, %", "грубое покрытие, %", "поворотов / км"]
    return data.groupby("решение")[columns].mean().join(data.groupby("решение")["OD изменено"].sum())


def prediction_quality(lab, model, *, split="validation"):
    y=lab.labels_for(split); known=y.notna(); target=np.log1p(y[known]); weights=lab.edges.loc[known,"length_m"]
    predicted=model.predict_log_trips(lab.edges[known])
    return pd.DataFrame({"MAE log(1+trips)":{
        "среднее train":np.average(abs(target-model.reference_log),weights=weights),
        "ML":np.average(abs(target-predicted),weights=weights)},
        "корреляция":{"среднее train":np.nan,"ML":weighted_corr(target,predicted,weights)}})


def export_solution(lab, result, *, algorithm, folder):
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    routes=result.routes[result.routes.algorithm.eq(algorithm)]
    if routes.empty: raise ValueError(f"Нет решения {algorithm}")
    routes.to_parquet(folder/"routes.parquet")
    result.table.to_csv(folder/"validation.csv")
    result.per_od[result.per_od.algorithm.eq(algorithm)].to_csv(folder/"per_od.csv",index=False)
    (folder/"submission.json").write_text(json.dumps({"dataset_id":lab.manifest["dataset_id"],"algorithm":algorithm,
                                                    "rules":asdict(result.rules)},ensure_ascii=False,indent=2),encoding="utf-8")
    return folder
