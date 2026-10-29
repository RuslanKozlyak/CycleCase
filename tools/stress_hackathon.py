"""Stress the existing workshop API without changing it or executing notebooks.

Run: .venv/Scripts/python.exe tools/stress_hackathon.py --jobs 2
Or run --phase filters first, then --phase candidates to reuse that completed audit.
Use --phase grader to check grade_all against the completed isolated runs.
Use --phase report to regenerate tables/plots without repeating searches.
Repeat a case with --phase rerun --case H069 --reason "explanation".
Reports, isolated submissions and caches go to outputs/stress_hackathon/<pack>/.
The 100 honest candidates are frozen using public data before closed evaluation.
Deliberately invalid/adversarial probes are never eligible for the leaderboard.
Each candidate runs in a separate process, with a timeout and independent route checks.
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import hashlib
from importlib.metadata import version
from heapq import heappop, heappush
from itertools import count
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import cloudpickle
import networkx as nx
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from cycle_routing.graph import build_graph, is_rideable
from cycle_routing.grading import tiers
from cycle_routing.learning import spatial_partition
from cycle_routing.routing import edge_bearings
from cycle_routing.tags import has_any
from cycle_routing.workshop import (
    CityData, build_network, closed_city, comfort_cost, export_solution, features,
    load_city, load_submission, percentile, presets, rideable, score, solution,
    train_custom_model, train_popularity,
)

BASE = {"крупная_дорога": 0.6, "велодорожка": -0.4, "асфальт": -0.2}
KEY = ["u", "v", "key"]


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA:
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def frame_city():
    """Adversarial probes inspect only their own isolated evaluator process."""
    import inspect
    frame = inspect.currentframe().f_back
    while frame:
        candidate = frame.f_locals.get("city")
        if isinstance(candidate, CityData):
            return candidate
        frame = frame.f_back
    raise RuntimeError("No evaluator city in the local stack")


@dataclass
class Filter:
    kind: str
    value: object = None

    def __call__(self, roads):
        ok = rideable(roads)
        road, surface = roads["тип_дороги"], roads["покрытие"]
        k, v = self.kind, self.value
        if k == "reference": return ok
        if k == "all": return pd.Series(True, index=roads.index)
        if k == "empty": return pd.Series(False, index=roads.index)
        if k == "drop_types": return ok & ~has_any(road, v)
        if k == "only_types": return ok & has_any(road, v)
        if k == "asphalt": return ok & has_any(surface, "асфальт")
        if k == "known_surface": return ok & surface.notna()
        if k == "no_rough": return ok & ~has_any(surface, {"грунт", "гравий", "песок", "брусчатка", "грязь"})
        if k == "lit": return ok & has_any(roads["освещение"], {"есть", "круглосуточное"})
        if k == "speed": return ok & roads["макс_скорость"].fillna(v[1]).le(v[0])
        if k == "width": return ok & roads["ширина_м"].fillna(v[1]).ge(v[0])
        if k == "bicycle": return ok & has_any(roads["велосипед"], {"разрешён", "предназначено_для_велосипедов"})
        if k == "length_max": return ok & roads["длина_м"].le(v)
        if k == "length_min": return ok & roads["длина_м"].ge(v)
        if k == "random": return ok & pd.Series(np.random.default_rng(71).random(len(roads)) >= v, index=roads.index)
        if k == "west": return ok & roads.geometry.interpolate(0.5, normalized=True).x.le(v)
        if k == "drop_indices": return ok & ~roads.index.isin(v)
        if k == "one_edge": return pd.Series(roads.index == roads.index[0], index=roads.index)
        if k == "list": return ok.tolist()
        if k == "float": return ok.astype(float)
        if k == "nullable": return ok.astype("boolean")
        if k == "short": return ok.iloc[:-1]
        if k == "reversed_index": return ok.iloc[::-1]
        if k == "na": return ok.astype("boolean").mask(roads.index == roads.index[0])
        if k == "raise": raise RuntimeError("Deliberate team filter exception")
        if k == "mutate_length":
            roads["длина_м"] = 1e-9
            return ok
        if k == "mutate_types":
            roads["тип_дороги"] = "велодорожка"
            return ok
        if k == "drop_inplace":
            roads.drop(roads.index[:10], inplace=True)
            return rideable(roads)
        if k == "stack_pairs":
            city = frame_city()
            city.pairs = city.pairs.iloc[:1].copy()
            return ok
        if k == "stack_labels":
            city = frame_city()
            # Do not read private files: demonstrate the mutable CityData exposed in the callback stack.
            graph = build_graph(city.osm[is_rideable(city.osm)], city.nodes, "length_m")
            used = set()
            for pair in city.pairs.itertuples():
                path = nx.dijkstra_path(graph, int(pair.origin), int(pair.destination), weight="cost")
                used.update(zip(path, path[1:]))
            city.trips = pd.Series([1e9 if (u, w) in used else 0.0 for u, w in zip(city.osm.u, city.osm.v)], index=city.osm.index)
            return ok
        raise KeyError(k)


@dataclass
class Recipe:
    mode: str = "basic"
    model: object = None

    def __call__(self, roads):
        f = features(roads)
        if self.mode != "basic":
            f["жилая_улица"] = has_any(roads["тип_дороги"], {"жилая_улица", "жилая_зона"}).astype(float)
            f["пешеходная"] = has_any(roads["тип_дороги"], {"пешеходная_дорожка", "пешеходная_улица", "тропа", "лестница"}).astype(float)
            f["плохое_покрытие"] = has_any(roads["покрытие"], {"брусчатка", "булыжник", "грунт", "гравий", "песок", "грязь"}).astype(float)
            f["нет_света"] = has_any(roads["освещение"], "нет").astype(float)
            f["покрытие_неизвестно"] = roads["покрытие"].isna().astype(float)
            f["велополоса"] = (roads["велополоса"].notna() & ~has_any(roads["велополоса"], {"нет", "отсутствует", "велодорожка_отдельной_линией_osm"})).astype(float)
            f["велосипед_разрешён"] = has_any(roads["велосипед"], {"разрешён", "предназначено_для_велосипедов"}).astype(float)
        if self.mode in {"numeric", "interactions", "geo"}:
            for column, name, default in [("макс_скорость", "скорость", 30), ("ширина_м", "ширина", 3), ("полос", "полосы", 1)]:
                values = pd.to_numeric(roads[column], errors="coerce")
                f[name] = values.fillna(default)
                f[name + "_нет"] = values.isna().astype(float)
            f["односторонняя"] = roads["одностороннее"].astype(float)
        if self.mode in {"interactions", "geo"}:
            f["магистраль_без_велополосы"] = f["крупная_дорога"] * (1 - f["велополоса"])
            f["скорость_без_света"] = f["скорость"] * f["нет_света"]
            f["пешеходная_с_велосипедом"] = f["пешеходная"] * f["велосипед_разрешён"]
        if self.mode == "geo":
            mid = roads.geometry.interpolate(0.5, normalized=True)
            f["x_км"], f["y_км"] = mid.x / 1000, mid.y / 1000
        if self.model is not None:
            from cycle_routing.workshop import osm
            predicted = self.model.predict_log_trips(osm(roads))
            f["ml_популярность"] = predicted.rank(pct=True)
        return f


@dataclass
class Formula:
    mode: str

    def __call__(self, roads):
        f = Recipe("numeric")(roads)
        length = roads["длина_м"]
        if self.mode == "types":
            prices = {"велодорожка": .45, "жилая_зона": .7, "жилая_улица": .8, "главная_дорога": 1.8, "пешеходная_дорожка": 1.5}
            return length * roads["тип_дороги"].str.split(";").str[0].map(prices).fillna(1)
        if self.mode == "exponential": return length * np.exp(.8 * f["крупная_дорога"] - .7 * f["велодорожка"] + .5 * f["плохое_покрытие"])
        if self.mode == "product": return length * (1 + f["крупная_дорога"]) * (1 + .8 * f["плохое_покрытие"]) * (1 - .5 * f["велодорожка"])
        if self.mode == "maximum": return length * (1 + f[["крупная_дорога", "плохое_покрытие", "нет_света"]].max(axis=1))
        if self.mode == "hops": return pd.Series(1.0, index=roads.index)
        if self.mode == "sqrt_length": return length.clip(lower=.01) ** .5
        if self.mode == "long_edges": return length.clip(lower=.01) ** 1.3
        if self.mode == "speed": return length / (12 + 8 * f["велодорожка"] - 4 * f["плохое_покрытие"])
        if self.mode == "zero": return pd.Series(0.0, index=roads.index)
        if self.mode == "negative": return -length
        if self.mode == "nan": return length * np.nan
        if self.mode == "infinity": return length * np.inf
        raise KeyError(self.mode)


def hop(graph, u, v, weight):
    data = graph[u][v]
    return min(a[weight] for a in data.values()) if graph.is_multigraph() else data[weight]


def heuristic_scale(graph, weight):
    key = "stress_scale_" + weight
    if key not in graph.graph:
        ratios = []
        for u, v, attrs in graph.edges(data=True):
            a, b = graph.nodes[u], graph.nodes[v]
            distance = math.hypot(a["x"] - b["x"], a["y"] - b["y"])
            if distance > 1e-9:
                ratios.append(attrs[weight] / distance)
        graph.graph[key] = .99 * min(ratios, default=0.0)
    return graph.graph[key]


@dataclass
class Search:
    mode: str
    fixed: object = None

    def __call__(self, graph, source, target, weight="cost"):
        mode = self.mode
        if mode == "hang":
            time.sleep(3600)
        if mode == "restore_network":
            graph.add_nodes_from(self.fixed.nodes(data=True))
            for u, v, key, attrs in self.fixed.edges(keys=True, data=True):
                graph.add_edge(u, v, key=key, **attrs)
            return nx.dijkstra_path(graph, source, target, weight=weight)
        if source not in graph or target not in graph:
            raise nx.NodeNotFound()
        if mode == "none": raise nx.NetworkXNoPath()
        if mode == "fixed": return list(self.fixed)
        if mode == "prefix": return nx.dijkstra_path(graph, source, target, weight=weight)[:2]
        if mode == "suffix": return nx.dijkstra_path(graph, source, target, weight=weight)[-2:]
        if mode == "reverse": return nx.dijkstra_path(graph, target, source, weight=weight)
        if mode == "disconnected_turn_path":
            first = next(iter(graph.successors(source)))
            last = next(iter(graph.predecessors(target)))
            return [source, first, last, target]
        if mode == "forge_lengths":
            nx.set_edge_attributes(graph, .001, "length_m")
            return nx.dijkstra_path(graph, source, target, weight=weight)
        goal = graph.nodes[target]
        def distance(node):
            point = graph.nodes[node]
            return math.hypot(point["x"] - goal["x"], point["y"] - goal["y"])
        if mode == "astar":
            scale = heuristic_scale(graph, weight)
            return nx.astar_path(graph, source, target, weight=weight, heuristic=lambda a, b: scale * distance(a))
        if mode == "bfs":
            queue, seen = deque([[source]]), {source}
            while queue:
                path = queue.popleft()
                if path[-1] == target: return path
                for v in graph.successors(path[-1]):
                    if v not in seen:
                        seen.add(v)
                        queue.append(path + [v])
            raise nx.NetworkXNoPath()
        if mode == "greedy_list":
            paths, seen = [[source]], {source}
            while paths:
                path = min(paths, key=lambda p: distance(p[-1]))
                paths.remove(path)
                if path[-1] == target: return path
                for v in graph.successors(path[-1]):
                    if v not in seen:
                        seen.add(v)
                        paths.append(path + [v])
            raise nx.NetworkXNoPath()
        order = count()
        queue, costs, previous = [(0.0, next(order), source)], {source: 0.0}, {source: None}
        while queue:
            _, _, current = heappop(queue)
            if current == target:
                path = [current]
                while current != source:
                    current = previous[current]
                    path.append(current)
                return path[::-1]
            for v in graph.successors(current):
                value = costs[current] + hop(graph, current, v, weight)
                if mode in {"greedy_heap", "beam"}:
                    if v in previous: continue
                    priority = distance(v)
                else:
                    if value >= costs.get(v, math.inf): continue
                    priority = value
                previous[v], costs[v] = current, value
                heappush(queue, (priority, next(order), v))
            if mode == "beam" and len(queue) > 32:
                queue = sorted(queue)[:32]
        raise nx.NetworkXNoPath()


@dataclass(frozen=True)
class Turn:
    mode: str
    strength: float = 1

    def __call__(self, graph, incoming, outgoing):
        angle = (edge_bearings(graph, outgoing)[0] - edge_bearings(graph, incoming)[1] + 180) % 360 - 180
        if self.mode == "junction": return self.strength if graph.out_degree(incoming[1]) > 2 else 0.0
        if self.mode == "absolute": return self.strength * abs(angle)
        if self.mode == "left": return self.strength if -150 < angle < -55 else 0.0
        if self.mode == "uturn": return self.strength if abs(angle) >= 150 else 0.0
        if self.mode == "negative": return -self.strength
        raise KeyError(self.mode)


@dataclass
class Case:
    id: str
    label: str
    family: str
    chosen: dict
    keep: object = None
    kind: str = "honest"
    settings: object = None
    timeout: int = 180
    route_key: str | None = None


def filter_specs(city):
    midpoint_x = city.osm.geometry.interpolate(.5, normalized=True).x
    endpoints = set(city.pairs.iloc[:3].origin) | set(city.pairs.iloc[:3].destination)
    removed = city.osm.index[city.osm.u.isin(endpoints) | city.osm.v.isin(endpoints)].tolist()
    regular = [
        ("базовый", Filter("reference")), ("все дороги", Filter("all")),
        ("без пешеходных дорожек", Filter("drop_types", ["пешеходная_дорожка", "пешеходная_улица"])),
        ("только пешеходные дорожки", Filter("only_types", ["пешеходная_дорожка", "пешеходная_улица"])),
        ("без троп и хозяйственных", Filter("drop_types", ["тропа", "хозяйственная_дорога"])),
        ("без главных дорог", Filter("drop_types", ["главная_дорога", "дорога_второго_значения"])),
        ("только жилые и вело", Filter("only_types", ["жилая_улица", "жилая_зона", "велодорожка"])),
        ("только асфальт", Filter("asphalt")), ("известное покрытие", Filter("known_surface")),
        ("без плохого покрытия", Filter("no_rough")), ("только освещённые", Filter("lit")),
        ("скорость до 30, пропуски разрешены", Filter("speed", (30, 30))),
        ("скорость до 50, пропуски исключены", Filter("speed", (50, 999))),
        ("ширина от 2, пропуски разрешены", Filter("width", (2, 3))),
        ("велосипед явно разрешён", Filter("bicycle")), ("короче 100 м", Filter("length_max", 100)),
        ("длиннее 20 м", Filter("length_min", 20)), ("удалить случайные 5%", Filter("random", .05)),
        ("удалить случайные 20%", Filter("random", .2)), ("западная половина", Filter("west", float(midpoint_x.median()))),
    ]
    extra = [
        ("удалить всё", Filter("empty")), ("одно ребро", Filter("one_edge")),
        ("удалить концы трёх открытых OD", Filter("drop_indices", removed)),
        ("только велодорожки", Filter("only_types", ["велодорожка"])),
        ("светлая улица, узкая ширина", Filter("width", (4, 0))),
        ("удалить случайные 70%", Filter("random", .7)),
        ("мутация длины во входной таблице", Filter("mutate_length")),
        ("мутация типа во входной таблице", Filter("mutate_types")),
    ]
    invalid = [(name, Filter(name)) for name in ["list", "float", "nullable", "short", "reversed_index", "na", "raise", "drop_inplace"]]
    return regular, extra, invalid


def fit_models(city, out):
    from sklearn.base import clone
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.tree import DecisionTreeRegressor
    models, rows = {}, []
    started = time.perf_counter()
    models["ready_hgb"] = train_popularity(city)
    rows.append({"model": "ready_hgb", "features": "OSM tags", "seconds": time.perf_counter() - started})
    specs = [
        ("dummy", DummyRegressor(), "numeric"), ("stump", DecisionTreeRegressor(max_depth=1, random_state=7), "basic"),
        ("tree8", DecisionTreeRegressor(max_depth=8, min_samples_leaf=20, random_state=7), "numeric"),
        ("forest", RandomForestRegressor(n_estimators=80, max_depth=14, min_samples_leaf=10, random_state=7, n_jobs=1), "numeric"),
        ("extra_trees", ExtraTreesRegressor(n_estimators=100, max_depth=16, min_samples_leaf=8, random_state=7, n_jobs=1), "numeric"),
        ("hgb", HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, random_state=7), "interactions"),
        ("hgb_geo", HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, random_state=7), "geo"),
        ("ridge", make_pipeline(StandardScaler(), Ridge(alpha=10)), "numeric"),
        ("knn", make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=15, weights="distance", n_jobs=1)), "numeric"),
        ("gradient_boosting", GradientBoostingRegressor(n_estimators=100, max_depth=3, learning_rate=.05, random_state=7), "numeric"),
    ]
    for name, estimator, mode in specs:
        started = time.perf_counter()
        def train(x, y, template=estimator):
            return clone(template).fit(x, y)
        models[name] = train_custom_model(city, train, features=Recipe(mode), seed=7)
        rows.append({"model": name, "features": mode, "columns": list(models[name].columns), "seconds": time.perf_counter() - started})
        print("Trained", name, f"{rows[-1]['seconds']:.1f}s", flush=True)
    write_json(out / "models.json", rows)
    return models


def make_candidates(city, models):
    cases = []
    def add(label, family, chosen, keep=None, settings=None):
        ident = f"H{len(cases):03d}"
        cases.append(Case(ident, label, family, {**chosen, "label": label}, keep, settings=settings))
    for name, chosen in presets().items(): add(name, "preset", chosen)
    rng = np.random.default_rng(71)
    manual = [BASE, {"велодорожка": -.8}, {"крупная_дорога": 2}, {"асфальт": -.6},
              {"жилая_улица": -.5, "крупная_дорога": .8}, {"пешеходная": 1.5},
              {"плохое_покрытие": 2}, {"велополоса": -.6, "крупная_дорога": .7}]
    for _ in range(8):
        manual.append({"велодорожка": float(rng.uniform(-.9, -.1)), "крупная_дорога": float(rng.uniform(0, 2.5)),
                       "плохое_покрытие": float(rng.uniform(0, 2)), "жилая_улица": float(rng.uniform(-.6, .2)),
                       "пешеходная": float(rng.uniform(0, 1.5)), "нет_света": float(rng.uniform(0, .7))})
    for i, weights in enumerate(manual):
        add(f"веса {i + 1}", "weights", solution(comfort_cost(weights, features=Recipe("binary"))), settings=weights)
    for mode in ["types", "exponential", "product", "maximum", "hops", "sqrt_length", "long_edges", "speed"]:
        add("формула " + mode, "formula", solution(Formula(mode)))
    for mode, weights in [("numeric", {"скорость": .015, "ширина": -.03}),
                          ("interactions", {"магистраль_без_велополосы": 1.3, "пешеходная_с_велосипедом": -.6}),
                          ("numeric", {"скорость_нет": .8, "покрытие_неизвестно": .5}),
                          ("interactions", {"скорость_без_света": .025, "велополоса": -.5})]:
        add("признаки " + str(len(cases)), "features", solution(comfort_cost(weights, features=Recipe(mode))), settings=weights)
    regular, _, _ = filter_specs(city)
    for label, keep in regular:
        add("фильтр: " + label, "filter", solution(comfort_cost(BASE), turns=None), keep, vars(keep))
    turns = [None, "default", "min_turns", {"left_penalty_m": 80, "right_penalty_m": 10},
             {"u_turn_penalty_m": 1000}, Turn("junction", 40), Turn("absolute", .3),
             Turn("left", 120), Turn("uturn", 300), {"slight_penalty_m": 30, "left_penalty_m": 50, "right_penalty_m": 50}]
    for i, turn in enumerate(turns):
        add(f"повороты {i + 1}", "turns", solution(comfort_cost(BASE), turns=turn), settings=str(turn))
    searches = [("nx_dijkstra", nx.dijkstra_path, None), ("nx_dijkstra_turns", nx.dijkstra_path, "default"),
                ("nx_bellman", nx.bellman_ford_path, None), ("nx_bellman_turns", nx.bellman_ford_path, "default"),
                ("astar", Search("astar"), None), ("astar_turns", Search("astar"), "default"),
                ("own_dijkstra", Search("dijkstra"), None), ("own_dijkstra_turns", Search("dijkstra"), "default"),
                ("greedy_list", Search("greedy_list"), None), ("greedy_heap_turns", Search("greedy_heap"), "default"),
                ("bfs", Search("bfs"), None), ("beam32", Search("beam"), None)]
    for label, search, turn in searches:
        add(label, "search", solution(comfort_cost(BASE), turns=turn, search=search))
    for name, model in models.items():
        for strength in [.4, .9]:
            weights = {**BASE, "ml_популярность": -strength}
            add(f"ML {name} α={strength}", "ml", solution(comfort_cost(weights, features=Recipe("basic", model))), settings={"model": name, "strength": strength})
    assert len(cases) == 100, len(cases)
    return cases


def make_audits(city, model):
    regular, extras, invalid = filter_specs(city)
    cases = []
    for label, keep in regular + extras + invalid:
        kind = "invalid" if (label, keep) in invalid else "filter_audit"
        cases.append(Case(f"F{len(cases):03d}", label, "filter_audit", solution(comfort_cost(BASE), turns=None), keep, kind, vars(keep)))
    probes = []
    def add(label, chosen, keep=None, timeout=180, key=None):
        probes.append(Case(f"A{len(probes):03d}", label, "adversarial", chosen, keep, "adversarial", timeout=timeout, route_key=key))
    for mode in ["zero", "negative", "nan", "infinity"]:
        add("invalid cost: " + mode, solution(Formula(mode), turns=None))
    for mode in ["none", "prefix", "suffix", "reverse"]:
        add("search: " + mode, solution(comfort_cost(BASE), turns=None, search=Search(mode)))
    add("search hangs", solution(comfort_cost(BASE), turns=None, search=Search("hang")), timeout=15)
    add("reserved English name", solution(comfort_cost(BASE), turns=None), key="shortest")
    add("Russian baseline name collision", solution(comfort_cost(BASE), turns=None), key="кратчайший")
    add("filter shrinks OD bank via callback stack", solution(comfort_cost(BASE), turns=None), Filter("stack_pairs"))
    add("filter rewrites labels via callback stack", solution("length_m", turns=None), Filter("stack_labels"))
    predicted = model.predict_log_trips(city.osm)
    eligible = city.reference & city.test_area
    favorite = predicted[eligible].idxmax()
    edge = city.osm.loc[favorite]
    add("fixed ML-selected segment, wrong OD", solution(comfort_cost(BASE), turns=None, search=Search("fixed", (int(edge.u), int(edge.v)))))
    cost = comfort_cost({**BASE, "ml_популярность": -2.0}, features=Recipe("basic", model))
    add("forge physical lengths inside search", solution(cost, turns=None, search=Search("forge_lengths")))
    add("negative turn penalties", solution(comfort_cost(BASE), turns=Turn("negative", 10000), search=nx.bellman_ford_path), timeout=35)
    restored = build_graph(city.osm[city.reference], city.nodes, "length_m")
    add("search restores roads removed by empty filter", solution(comfort_cost(BASE), turns=None, search=Search("restore_network", restored)), Filter("empty"))
    add("disconnected path with correct OD endpoints", solution(comfort_cost(BASE), search=Search("disconnected_turn_path")))
    return cases, probes


def snapshot(city):
    return {"roads": city.osm.copy(deep=True), "pairs": city.pairs.copy(deep=True), "trips": city.trips.copy(deep=True),
            "popularity": percentile(city.trips, city.osm.length_m)}


def audit_result(result, frozen, network, key, baseline):
    pairs = frozen["pairs"].set_index("od_id")
    roads = frozen["roads"]
    lookup = pd.DataFrame({"length": roads.length_m.to_numpy(), "pop": frozen["popularity"].to_numpy(),
                           "bad": (~is_rideable(roads)).to_numpy()}, index=pd.MultiIndex.from_frame(roads[KEY]))
    allowed = set(map(tuple, network.roads[KEY].to_numpy()))
    own = result.routes[result.routes.algorithm.eq(key)]
    actual = result.per_od[result.per_od.algorithm.eq(key)].set_index("od_id")
    base = baseline.set_index("od_id")
    records, issues = [], []
    if len(result.table) != 2:
        issues.append("baseline_name_collision_in_table")
    if len(actual) != len(pairs): issues.append("OD_bank_changed")
    if not result.city.osm.equals(roads): issues.append("road_data_mutated")
    if not result.city.trips.equals(frozen["trips"]): issues.append("labels_mutated")
    if not result.city.pairs.equals(frozen["pairs"]): issues.append("pairs_mutated")
    rows = {row.od_id: row for row in own.itertuples()}
    for od, pair in pairs.iterrows():
        row = rows.get(od)
        flags, length, popularity = [], np.nan, np.nan
        if row is not None:
            path = [tuple(map(int, edge)) for edge in row.edge_route]
            if not path or path[0][0] != pair.origin or path[-1][1] != pair.destination: flags.append("wrong_endpoints")
            if any(a[1] != b[0] for a, b in zip(path, path[1:])): flags.append("disconnected_path")
            if not all(edge in allowed for edge in path): flags.append("outside_filter")
            try:
                used = lookup.loc[path]
                length = float(used.length.sum())
                popularity = float(np.average(used["pop"].fillna(50), weights=used.length))
                if not np.isclose(length, row.route_length_m, rtol=1e-8, atol=1e-6): flags.append("forged_length")
            except (KeyError, ZeroDivisionError):
                flags.append("unknown_edges")
        true_detour = 100 * (length / base.loc[od, "length_m"] - 1)
        safe_score = 0.0 if row is None or any(flag != "forged_length" for flag in flags) else popularity
        official = float(actual.loc[od, "counted"]) if od in actual.index else np.nan
        records.append({"od_id": od, "official": official, "audited": safe_score, "has_route": row is not None,
                        "true_length_m": length, "true_detour_pct": true_detour, "flags": ";".join(flags)})
    detail = pd.DataFrame(records)
    issues.extend(sorted({flag for values in detail["flags"] if values for flag in values.split(";")}))
    own_mean = float(actual.counted.mean()) if len(actual) else np.nan
    if np.isfinite(own_mean) and not np.isclose(own_mean, detail.audited.mean(), atol=1e-7): issues.append("score_disagrees_with_independent_audit")
    now_base = result.per_od[result.per_od.algorithm.eq("shortest")].set_index("od_id")
    if set(now_base.index) != set(base.index) or not np.allclose(now_base.counted.reindex(base.index), base.counted, equal_nan=True):
        issues.append("baseline_changed")
    return detail, sorted(set(issues))


def worker(bundle_path, case_id, out, fresh_cache=False):
    bundle = cloudpickle.loads(Path(bundle_path).read_bytes())
    case = next(case for case in bundle["cases"] if case.id == case_id)
    out = Path(out)
    cache_dir = out / "fresh_cache" / case.id / str(time.time_ns()) if fresh_cache else out / "cache"
    records = []
    for zone in ["open", "closed"]:
        started = time.perf_counter()
        row = {"id": case.id, "label": case.label, "family": case.family, "kind": case.kind, "zone": zone}
        try:
            city = load_city(bundle["public"])
            if zone == "closed": city = closed_city(city, bundle["private"])
            city = replace(city, folder=cache_dir)
            frozen = snapshot(city)
            network = build_network(city, case.keep)
            graph = nx.from_pandas_edgelist(network.roads, "u", "v", create_using=nx.DiGraph)
            reachable = [p.origin in graph and p.destination in graph and nx.has_path(graph, p.origin, p.destination) for p in frozen["pairs"].itertuples()]
            row.update(input_edges=len(frozen["roads"]), kept_edges=len(network.roads), removed=int(network.removed.sum()),
                       cut_off=int(network.cut_off.sum()), expected_pairs=len(frozen["pairs"]), reachable=sum(reachable))
            key = case.route_key or case.id
            chosen = {**case.chosen, "label": case.label}
            # Exercise the same export/load path teams use; keep each candidate's files isolated.
            submission = out / (case.id + "_submission_" + zone)
            export_solution(city, chosen, keep=case.keep, folder=submission)
            chosen, _ = load_submission(submission)
            result = score(city, {key: chosen}, network=network, progress=False)
            baseline = pd.read_pickle(out / ("baseline_" + zone + ".pkl"))
            detail, issues = audit_result(result, frozen, network, key, baseline)
            actual = result.per_od[result.per_od.algorithm.eq(key)]
            own = result.routes[result.routes.algorithm.eq(key)]
            row.update(status="ok", score=float(actual.counted.mean()), audited_score=float(detail.audited.mean()),
                       routes=len(own), failed=len(detail) - len(own), counted_pairs=len(actual),
                       search_ms=float(actual.search_ms.median()), detour_pct=float(actual.detour_pct.median()),
                       unsafe_m=float(actual.bad_m.sum()), coverage_pct=float(actual.coverage_pct.mean()),
                       flags=issues, valid_paths=int(detail["flags"].eq("").sum()),
                       detours_over_20_pct=int(detail.true_detour_pct.gt(20 + 1e-9).sum()))
            detail.to_csv(out / f"{case.id}_{zone}_audit.csv", index=False)
            result.per_od.to_pickle(out / f"{case.id}_{zone}_per_od.pkl")
            own.to_pickle(out / f"{case.id}_{zone}_routes.pkl")
        except Exception as error:
            row.update(status="error", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        row["wall_s"] = time.perf_counter() - started
        records.append(row)
        write_json(out / (case.id + ".json"), records)
        print(case.id, zone, row["status"], row.get("score"), flush=True)


def run_cases(cases, bundle, out, jobs, fresh_cache=False):
    def launch(case):
        target = out / (case.id + ".json")
        command = [sys.executable, "-u", "-X", "utf8", str(Path(__file__).resolve()), "--worker", case.id, "--bundle", str(bundle), "--output", str(out)]
        if fresh_cache:
            command.append("--fresh-cache")
        env = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        started = time.perf_counter()
        with (out / (case.id + ".log")).open("w", encoding="utf-8") as log:
            try:
                process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=case.timeout,
                                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                state = "worker_exit" if process.returncode else "missing_result"
            except subprocess.TimeoutExpired:
                state = "timeout"
        rows = json.loads(target.read_text(encoding="utf-8")) if target.exists() else []
        incomplete = False
        for zone in ["open", "closed"]:
            if not any(row["zone"] == zone for row in rows):
                rows.append({"id": case.id, "label": case.label, "family": case.family, "kind": case.kind,
                             "zone": zone, "status": "not_run_after_" + state if incomplete else state,
                             "wall_s": time.perf_counter() - started})
                incomplete = True
        write_json(target, rows)
        return rows
    records = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(launch, case): case for case in cases}
        for done, future in enumerate(as_completed(futures), 1):
            case = futures[future]
            rows = future.result()
            records.extend(rows)
            print(f"[{done}/{len(cases)}] {case.id} {case.label}: " + ", ".join(f"{r['zone']}={r['status']}:{r.get('score', '')}" for r in rows), flush=True)
    return records


def protected_hashes():
    files = [*Path(ROOT / "src").rglob("*.py"), *Path(ROOT / "notebooks").rglob("*.ipynb")]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files if ".ipynb_checkpoints" not in path.parts}


def input_hashes(public, private):
    files = [public / "manifest.json", private / "dataset.json", private / "closed_pairs.parquet", private / "closed_labels.parquet"]
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}


def prepare(city, private, out):
    rows = []
    for zone, data in [("open", city), ("closed", closed_city(city, private))]:
        data = replace(data, folder=out / "cache")
        result = score(data, {}, progress=False)
        result.per_od.to_pickle(out / ("baseline_" + zone + ".pkl"))
        rows.append({"zone": zone, "pairs": len(data.pairs), "roads": len(data.osm), "reference_roads": int(data.reference.sum()),
                     "known_labels": int(data.trips.notna().sum()), "zero_labels": int(data.trips.eq(0).sum()),
                     "baseline_score": float(result.per_od.counted.mean()),
                     "baseline_length_median": float(result.per_od.length_m.median()), "measured_length_pct": float(result.per_od.coverage_pct.mean())})
        grouped = data.osm.assign(trips=data.trips, known=data.trips.notna()).groupby("highway", dropna=False).agg(
            edges=("length_m", "size"), length_m=("length_m", "sum"), measured=("known", "sum"), median_trips=("trips", "median"))
        grouped.to_csv(out / ("road_types_" + zone + ".csv"))
    write_json(out / "data_summary.json", rows)


def data_diagnostics(city, private, out):
    """Read-only explanations of missing tags, labels and filter fragmentation."""
    reference = city.reference
    rows = []
    for name, values in city.roads.items():
        if name == city.roads.geometry.name:
            continue
        known = values.notna() & values.astype(str).ne("")
        rows.append({"field": name, "known_edges_pct": 100 * known[reference].mean(),
                     "known_length_pct": 100 * city.osm.loc[reference & known, "length_m"].sum() / city.osm.loc[reference, "length_m"].sum()})
    pd.DataFrame(rows).to_csv(out / "tag_completeness.csv", index=False)
    closed = closed_city(city, private)
    meta = {"public_known": int(city.trips.notna().sum()), "closed_known": int(closed.trips.notna().sum()),
            "labels_known_in_both": int((city.trips.notna() & closed.trips.notna()).sum()),
            "hidden_area_edges": int(city.test_area.sum()),
            "public_labels_in_hidden_area": int(city.trips[city.test_area].notna().sum()),
            "identical_directed_OD": len(set(zip(city.pairs.origin, city.pairs.destination)) & set(zip(closed.pairs.origin, closed.pairs.destination))),
            "public_unique_endpoints": len(set(city.pairs.origin) | set(city.pairs.destination)),
            "closed_unique_endpoints": len(set(closed.pairs.origin) | set(closed.pairs.destination))}
    write_json(out / "data_diagnostics.json", meta)
    regular, extras, _ = filter_specs(city)
    fragmentation = []
    for i, (label, keep) in enumerate((regular + extras)[:26]):
        mask = keep(city.roads)
        graph = nx.from_pandas_edgelist(city.osm[mask], "u", "v", create_using=nx.DiGraph)
        components = list(nx.strongly_connected_components(graph))
        for zone, data in [("open", city), ("closed", closed)]:
            reachable = sum(p.origin in graph and p.destination in graph and nx.has_path(graph, p.origin, p.destination)
                            for p in data.pairs.itertuples())
            fragmentation.append({"id": f"F{i:03d}", "label": label, "zone": zone,
                                  "mask_edges": int(mask.sum()), "strong_components": len(components),
                                  "largest_component_nodes": max(map(len, components), default=0),
                                  "reachable_before_component_cut": reachable})
    pd.DataFrame(fragmentation).to_csv(out / "filter_fragmentation.csv", index=False)
    sidewalk_diagnostics(city, private, out)


def sidewalk_diagnostics(city, private, out):
    """Audit pedestrian/road topology on fixed OD; never invent connecting edges."""
    import shapely
    roads = city.osm
    foot = has_any(roads.highway, {"footway", "pedestrian"})
    street = has_any(roads.highway, {"residential", "living_street", "service", "primary", "secondary", "tertiary", "unclassified", "primary_link", "secondary_link", "tertiary_link"}) & ~foot
    allowed, reference = is_rideable(roads), city.reference
    def graph(mask):
        return nx.from_pandas_edgelist(roads[mask], "u", "v", edge_attr="length_m", edge_key="key", create_using=nx.MultiDiGraph)
    layers = {"all": pd.Series(True, index=roads.index), "rideable_before_SCC": allowed,
              "baseline": reference, "without_footways": reference & ~foot, "streets_only": reference & street,
              "footways_only": reference & foot,
              "foot_path_cycleway": reference & (foot | has_any(roads.highway, {"path", "cycleway"}))}
    graphs = {name: graph(mask) for name, mask in layers.items()}
    stats, leaves, components = [], [], []
    for name, current in graphs.items():
        simple = nx.Graph(current)
        simple.remove_edges_from(list(nx.selfloop_edges(simple)))
        core = set(nx.k_core(simple, k=2))
        dead = [node for node in simple if simple.degree(node) <= 1]
        foot_nodes = set(roads.loc[layers[name] & foot, "u"]) | set(roads.loc[layers[name] & foot, "v"])
        pieces = list(nx.strongly_connected_components(current))
        largest = max(pieces, key=len, default=set())
        pivot = next(iter(largest), None)
        enters = nx.descendants(current, pivot) | {pivot} if pivot is not None else set()
        returns = nx.ancestors(current, pivot) | {pivot} if pivot is not None else set()
        traps = enters - returns
        stats.append({"layer": name, "edges": current.number_of_edges(), "nodes": len(current),
                      "weak_components": nx.number_weakly_connected_components(current), "strong_components": len(pieces),
                      "leaf_nodes": len(dead), "footway_leaf_nodes": len(set(dead) & foot_nodes),
                      "directed_sinks": sum(current.out_degree(n) == 0 for n in current),
                      "nodes_outside_undirected_2core": len(simple) - len(core),
                      "enterable_without_return_to_main_SCC": len(traps)})
        leaves.extend({"layer": name, "node": int(n), "kind": "directed_trap" if n in traps else "leaf",
                       "incident_footway": n in foot_nodes} for n in set(dead) | traps)
    pd.DataFrame(stats).to_csv(out / "sidewalk_topology.csv", index=False)
    leaf_table = pd.DataFrame(leaves)
    coordinates = city.nodes.set_index("node")
    leaf_table = leaf_table.join(coordinates[["x", "y"]], on="node")
    leaf_table.to_csv(out / "sidewalk_dead_ends.csv", index=False)
    for layer in ["rideable_before_SCC", "baseline"]:
        pedestrian = graph(layers[layer] & foot)
        other = graph(layers[layer] & ~foot)
        for i, piece in enumerate(nx.weakly_connected_components(pedestrian)):
            components.append({"layer": layer, "component": i, "nodes": len(piece),
                               "entry_nodes_from_other_types": sum(n in other and other.in_degree(n) > 0 for n in piece),
                               "exit_nodes_to_other_types": sum(n in other and other.out_degree(n) > 0 for n in piece)})
    pd.DataFrame(components).to_csv(out / "sidewalk_components.csv", index=False)
    lookup = roads.set_index(KEY)
    foot_keys = set(map(tuple, roads.loc[foot, KEY].to_numpy()))
    od_rows, examples = [], {}
    for zone, data in [("open", city), ("closed", closed_city(city, private))]:
        for layer, current in graphs.items():
            for pair in data.pairs.itertuples():
                row = {"zone": zone, "layer": layer, "od_id": pair.od_id, "origin": int(pair.origin), "destination": int(pair.destination),
                       "origin_present": pair.origin in current, "destination_present": pair.destination in current, "has_path": False}
                try:
                    nodes = nx.dijkstra_path(current, pair.origin, pair.destination, weight="length_m")
                    edges = [(u, v, min(current[u][v], key=lambda k: current[u][v][k]["length_m"])) for u, v in zip(nodes, nodes[1:])]
                    lengths = lookup.loc[edges, "length_m"]
                    row.update(has_path=True, length_m=float(lengths.sum()), footway_length_m=float(sum(lookup.loc[e, "length_m"] for e in edges if e in foot_keys)))
                    if zone == "open" and layer in {"baseline", "without_footways"}:
                        examples[(pair.od_id, layer)] = edges
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    pass
                od_rows.append(row)
    od = pd.DataFrame(od_rows)
    od.to_csv(out / "sidewalk_OD.csv", index=False)
    od.groupby(["zone", "layer"]).agg(pairs=("has_path", "size"), reachable=("has_path", "sum"),
                                      origins_present=("origin_present", "sum"), destinations_present=("destination_present", "sum")).to_csv(out / "sidewalk_OD_summary.csv")
    # Nearby geometry does not establish access: keep these as review candidates, never repair the graph.
    street_nodes = set(roads.loc[reference & street, "u"]) | set(roads.loc[reference & street, "v"])
    raw_street_nodes = set(roads.loc[street, "u"]) | set(roads.loc[street, "v"])
    dangling = leaf_table[leaf_table.layer.eq("baseline") & leaf_table.incident_footway & ~leaf_table.node.isin(street_nodes)].copy()
    candidates = []
    street_rows = roads[reference & street]
    if len(dangling) and len(street_rows):
        tree = shapely.STRtree(street_rows.geometry.to_numpy())
        points = coordinates.geometry.reindex(dangling.node).to_numpy()
        indices, distances = tree.query_nearest(points, max_distance=15, return_distance=True, all_matches=False)
        for (point_index, road_index), distance in zip(indices.T, distances):
            sample, segment = dangling.iloc[point_index], street_rows.iloc[road_index]
            candidates.append({"node": int(sample.node), "distance_to_street_m": float(distance),
                               "street_u": int(segment.u), "street_v": int(segment.v), "street_key": int(segment.key),
                               "street_connection_in_raw_graph": int(sample.node) in raw_street_nodes,
                               "blocked_incident_edges": int((~allowed & (roads.u.eq(sample.node) | roads.v.eq(sample.node))).sum())})
    near = pd.DataFrame(candidates, columns=["node", "distance_to_street_m", "street_u", "street_v", "street_key", "street_connection_in_raw_graph", "blocked_incident_edges"])
    near.to_csv(out / "sidewalk_near_street_candidates.csv", index=False)
    missing = (set(roads.u) | set(roads.v)) - set(coordinates.index)
    geom_start, geom_end = shapely.get_point(roads.geometry.array, 0), shapely.get_point(roads.geometry.array, -1)
    u_points, v_points = coordinates.geometry.reindex(roads.u).to_numpy(), coordinates.geometry.reindex(roads.v).to_numpy()
    mismatch = np.maximum(np.minimum(shapely.distance(geom_start, u_points), shapely.distance(geom_end, u_points)),
                          np.minimum(shapely.distance(geom_start, v_points), shapely.distance(geom_end, v_points)))
    meta = {"sidewalk_specific_tags_available": [name for name in ["footway", "sidewalk", "crossing", "bridge", "tunnel", "layer"] if name in roads],
            "footway_proxy_definition": "highway contains footway or pedestrian; not all of these are sidewalks",
            "edges_with_missing_node": int((~roads.u.isin(coordinates.index) | ~roads.v.isin(coordinates.index)).sum()),
            "missing_nodes": len(missing), "empty_or_invalid_geometry": int((roads.geometry.is_empty | ~roads.geometry.is_valid).sum()),
            "geometry_endpoint_mismatch_over_1m": int((mismatch > 1).sum()),
            "footway_leaf_near_street_within_15m": len(near),
            "warning": "Spatial proximity alone cannot prove a missing connection or allow cycling; no graph edges or OD endpoints were changed."}
    bundle_path = out / "bundle.pkl"
    if bundle_path.exists():
        base_leaves = set(leaf_table[leaf_table.layer.eq("baseline") & leaf_table.kind.eq("leaf")].node)
        foot_leaves = set(leaf_table[leaf_table.layer.eq("baseline") & leaf_table.kind.eq("leaf") & leaf_table.incident_footway].node)
        traversals = []
        leaf_sets = {}
        saved = cloudpickle.loads(bundle_path.read_bytes())
        for case in saved["cases"]:
            if case.kind != "honest":
                continue
            signature = repr(case.keep)
            if signature not in leaf_sets:
                selected = build_network(city, case.keep).roads
                simple = nx.from_pandas_edgelist(selected, "u", "v", create_using=nx.Graph)
                simple.remove_edges_from(list(nx.selfloop_edges(simple)))
                selected_leaves = {n for n in simple if simple.degree(n) <= 1}
                selected_foot = selected[has_any(selected.highway, {"footway", "pedestrian"})]
                selected_foot_nodes = set(selected_foot.u) | set(selected_foot.v)
                leaf_sets[signature] = selected_leaves, selected_leaves & selected_foot_nodes
            network_leaves, network_foot_leaves = leaf_sets[signature]
            for zone in ["open", "closed"]:
                path = out / f"{case.id}_{zone}_routes.pkl"
                if not path.exists():
                    continue
                for route in pd.read_pickle(path).itertuples():
                    nodes = [route.edge_route[0][0]] + [edge[1] for edge in route.edge_route]
                    internal = set(nodes[1:-1]) - {nodes[0], nodes[-1]}
                    traversals.append({"id": case.id, "zone": zone, "od_id": route.od_id,
                                       "internal_baseline_leaf_nodes": len(internal & base_leaves),
                                       "internal_footway_leaf_nodes": len(internal & foot_leaves),
                                       "internal_network_leaf_nodes": len(internal & network_leaves),
                                       "internal_network_footway_leaf_nodes": len(internal & network_foot_leaves),
                                       "repeated_nodes": len(nodes) - len(set(nodes))})
        traversal = pd.DataFrame(traversals)
        if len(traversal):
            traversal.to_csv(out / "sidewalk_route_dead_ends.csv", index=False)
            meta.update(honest_routes_checked=len(traversal),
                        routes_with_internal_baseline_leaf=int(traversal.internal_baseline_leaf_nodes.gt(0).sum()),
                        routes_with_internal_footway_leaf=int(traversal.internal_footway_leaf_nodes.gt(0).sum()),
                        routes_with_internal_network_leaf=int(traversal.internal_network_leaf_nodes.gt(0).sum()),
                        routes_with_internal_network_footway_leaf=int(traversal.internal_network_footway_leaf_nodes.gt(0).sum()),
                        routes_with_repeated_nodes=int(traversal.repeated_nodes.gt(0).sum()))
    write_json(out / "sidewalk_diagnostics.json", meta)
    plot_sidewalk_examples(city, lookup, leaf_table, near, examples, out)


def plot_sidewalk_examples(city, lookup, leaves, near, examples, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from shapely.geometry import box
    valid = [(od, edges) for (od, layer), edges in examples.items() if layer == "baseline" and (od, "without_footways") in examples]
    if not valid or not (leaves.layer.eq("baseline") & leaves.incident_footway).any():
        return
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
    roads = city.osm
    nodes = city.nodes.set_index("node")
    foot = has_any(roads.highway, {"footway", "pedestrian"})
    candidate = near.sort_values("distance_to_street_m").iloc[0] if len(near) else None
    leaf = int(candidate.node) if candidate is not None else int(leaves[leaves.layer.eq("baseline") & leaves.incident_footway].iloc[0].node)
    point = nodes.loc[leaf]
    window = box(point.x - 100, point.y - 100, point.x + 100, point.y + 100)
    local = roads[roads.geometry.intersects(window)]
    removed = ~city.reference.loc[local.index]
    if removed.any():
        local[removed].plot(ax=axes[0], color="#fca5a5", linewidth=2, linestyle=":")
    local[~removed & ~foot.loc[local.index]].plot(ax=axes[0], color="#9ca3af", linewidth=2)
    local[~removed & foot.loc[local.index]].plot(ax=axes[0], color="#2563eb", linewidth=3)
    axes[0].scatter([point.x], [point.y], c="#dc2626", s=65, zorder=5)
    axes[0].set_title(f"Конец пешеходной ветки: {leaf}\nсиний — пешеходные; серый — остальные; розовый — отсечено")
    axes[0].set(xlim=(point.x - 100, point.x + 100), ylim=(point.y - 100, point.y + 100))
    chosen, edges = max(valid, key=lambda item: sum(lookup.loc[e, "length_m"] for e in item[1] if "footway" in str(lookup.loc[e, "highway"])))
    base = lookup.loc[edges]
    alternative = lookup.loc[examples[(chosen, "without_footways")]]
    bounds = np.vstack([base.total_bounds, alternative.total_bounds])
    xmin, ymin, xmax, ymax = bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max()
    window = box(xmin - 100, ymin - 100, xmax + 100, ymax + 100)
    roads[roads.geometry.intersects(window)].plot(ax=axes[1], color="#d1d5db", linewidth=.5)
    alternative.plot(ax=axes[1], color="#f97316", linewidth=3, label="Без footway/pedestrian")
    base.plot(ax=axes[1], color="#2563eb", linewidth=2, label="Базовый кратчайший")
    pair = city.pairs[city.pairs.od_id.eq(chosen)].iloc[0]
    xy = nodes.loc[[pair.origin, pair.destination]]
    axes[1].scatter(xy.x, xy.y, c=["#059669", "#dc2626"], s=45, zorder=5)
    axes[1].set_title(f"Один и тот же открытый OD: {chosen}\nсиний — базовый, оранжевый — без пешеходных")
    axes[1].set(xlim=(xmin - 100, xmax + 100), ylim=(ymin - 100, ymax + 100))
    for ax in axes:
        ax.set_aspect("equal")
        ax.set_axis_off()
    fig.savefig(out / "sidewalk_examples.png", dpi=150)
    plt.close(fig)


def run_official_grader(bundle, out):
    """Exercise the actual multi-team grader too, using the frozen route cache."""
    from cycle_routing.grading import Submission, grade_all
    city = replace(load_city(bundle["public"]), folder=out / "cache")
    cases = [c for c in bundle["cases"] if c.kind == "honest" or c.id in {"A000", "F020"}]
    cases.sort(key=lambda c: (c.id not in {"A000", "F020"}, c.id))
    submissions = []
    for case in cases:
        folder = out / (case.id + "_submission_closed")
        chosen, keep = load_submission(folder)
        submissions.append(Submission(case.id, folder, chosen, keep))
    graded = grade_all(city, bundle["private"], submissions, with_open=True, seed=71)
    graded.table.to_csv(out / "official_grader.csv")
    graded.errors.to_csv(out / "official_grader_errors.csv")
    graded.intervals.to_csv(out / "official_grader_intervals.csv")
    recorded = pd.DataFrame(json.loads((out / "results.json").read_text(encoding="utf-8")))
    deltas = []
    for case in cases:
        if case.id not in graded.table.index:
            continue
        expected = recorded[recorded.id.eq(case.id) & recorded.zone.eq("closed") & recorded.status.eq("ok")]
        if len(expected):
            deltas.append(abs(float(graded.table.loc[case.id, "популярность, закрытая зона"]) - float(expected.score.iloc[0])))
    result = {"submissions": len(cases), "ranked": len(graded.table) - 1, "errors": len(graded.errors),
              "maximum_score_difference_due_to_rounding": max(deltas, default=0),
              "empty_filter_score": float(graded.table.loc["F020", "популярность, закрытая зона"])}
    write_json(out / "official_grader_check.json", result)
    assert set(graded.errors.index) == {"A000"}, graded.errors
    assert {c.id for c in cases if c.kind == "honest"} <= set(graded.table.index)
    assert result["empty_filter_score"] == 0 and max(deltas, default=0) <= .051, result
    print("Official grader:", result, flush=True)


def plot_report(leaderboard, filters, frame, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), layout="constrained")
    for family, group in leaderboard.groupby("family"):
        axes[0, 0].scatter(group.open, group.closed, s=30, alpha=.7, label=family)
    axes[0, 0].plot([0, 80], [0, 80], color="gray", ls=":", lw=1)
    axes[0, 0].set(xlabel="Открытая зона", ylabel="Закрытая зона", title="100 пресетов: перенос результата")
    axes[0, 0].legend(fontsize=8, ncol=2)
    top = leaderboard.sort_values("closed").tail(10)
    y = np.arange(len(top))
    axes[0, 1].errorbar(top.gain_vs_baseline, y,
                      xerr=[top.gain_vs_baseline - top.gain_ci90_low, top.gain_ci90_high - top.gain_vs_baseline],
                      fmt="o", color="#2563eb", capsize=3)
    axes[0, 1].axvline(0, color="gray", ls=":")
    axes[0, 1].set(yticks=y, yticklabels=[f"{i}: {r.label}" for i, r in top.iterrows()],
                   xlabel="Баллы относительно кратчайшего; парный 90% интервал", title="Лучшие честные варианты")
    selected = ["F000", "F001", "F007", "F010", "F012", "F014", "F017", "F018", "F020", "F022", "F023"]
    filt = filters[filters.zone.eq("closed") & filters.id.isin(selected)].sort_values("id")
    y = np.arange(len(filt))
    axes[1, 0].barh(y, filt.reachable, color="#0d9488")
    axes[1, 0].set(yticks=y, yticklabels=[f"{r.id}: {r.label}" for r in filt.itertuples()],
                   xlim=(0, 38), xlabel="Связных OD-пар из 36", title="Цена жёстких фильтров")
    attacks = frame[frame.zone.eq("closed") & frame.id.isin(["A007", "A011", "A012", "A013", "A014"])]
    y = np.arange(len(attacks))
    axes[1, 1].barh(y + .18, attacks.score, height=.35, label="Текущий score", color="#f97316")
    axes[1, 1].barh(y - .18, attacks.audited_score, height=.35, label="Независимый пересчёт", color="#2563eb")
    labels = {"A007": "Обратное направление OD", "A011": "Сокращение банка OD", "A012": "Подмена меток", "A013": "Одно ребро вместо любого OD", "A014": "Подмена физической длины"}
    axes[1, 1].set(yticks=y, yticklabels=[labels[r.id] for r in attacks.itertuples()], xlabel="Баллы", title="Проверки обхода оценивания")
    axes[1, 1].legend(fontsize=9)
    for ax in axes.flat:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=.15)
    fig.savefig(out / "overview.png", dpi=150)
    plt.close(fig)


def report(cases, records, out):
    data_summary = json.loads((out / "data_summary.json").read_text(encoding="utf-8"))
    for row in data_summary:
        row["baseline_score"] = float(pd.read_pickle(out / ("baseline_" + row["zone"] + ".pkl")).counted.mean())
    write_json(out / "data_summary.json", data_summary)
    frame = pd.DataFrame(records)
    frame.to_csv(out / "all_results.csv", index=False)
    summary = frame.groupby(["kind", "zone", "status"]).size().rename("count").reset_index()
    summary.to_csv(out / "statuses.csv", index=False)
    honest = frame[frame.kind.eq("honest") & frame.status.eq("ok")].copy()
    scores = honest.pivot(index="id", columns="zone", values="score").dropna()
    catalogue = {case.id: case for case in cases}
    rng = np.random.default_rng(71)
    evaluated = []
    columns, values = [], []
    for ident in scores.index:
        path = out / f"{ident}_closed_audit.csv"
        detail = pd.read_csv(path).sort_values("od_id")
        zone_rows = honest[honest.id.eq(ident)]
        issues = sorted({flag for flags in zone_rows["flags"] for flag in (flags if isinstance(flags, list) else [])})
        evaluated.append({"id": ident, "label": catalogue[ident].label, "family": catalogue[ident].family,
                          "open": scores.loc[ident, "open"], "closed": scores.loc[ident, "closed"],
                          "audit_flags": ";".join(issues), "closed_audited": detail.audited.mean(),
                          "failed_closed": int((~detail.has_route).sum()),
                          "detours_over_20_pct_closed": int(detail.true_detour_pct.gt(20 + 1e-9).sum()),
                          "unsafe_m": float(zone_rows[zone_rows.zone.eq("closed")].unsafe_m.iloc[0]),
                          "search_ms": float(zone_rows[zone_rows.zone.eq("closed")].search_ms.iloc[0])})
        if not issues:
            columns.append(ident)
            values.append(detail.official.to_numpy())
    leaderboard = pd.DataFrame(evaluated).set_index("id") if evaluated else pd.DataFrame()
    uncertainty = {}
    if columns:
        matrix = np.array(values).T
        draws = rng.integers(0, len(matrix), size=(4000, len(matrix)))
        means = pd.Series(matrix.mean(axis=0), index=columns)
        boots = matrix[draws].mean(axis=1)
        tier = tiers(means, boots)
        leaderboard["tier"] = tier
        intervals = np.percentile(boots, [5, 95], axis=0)
        for j, ident in enumerate(columns):
            leaderboard.loc[ident, "ci90_low"] = intervals[0, j]
            leaderboard.loc[ident, "ci90_high"] = intervals[1, j]
        top = means.idxmax()
        pos = columns.index(top)
        wins = np.isclose(boots, boots.max(axis=1, keepdims=True), atol=1e-9)
        uncertainty = {"top": top, "top_label": catalogue[top].label, "top_tier_size": int(tier.eq(1).sum()),
                       "top_bootstrap_win_share": float((wins[:, pos] / wins.sum(axis=1)).mean()),
                       "top_bootstrap_among_best_pct": 100 * float(wins[:, pos].mean()),
                       "rank_agreement_open_closed": float(scores.open.corr(scores.closed, method="spearman"))}
        baseline = pd.read_pickle(out / "baseline_closed.pkl").sort_values("od_id").counted.to_numpy()
        difference = boots - baseline[draws].mean(axis=1)[:, None]
        for j, ident in enumerate(columns):
            leaderboard.loc[ident, "gain_vs_baseline"] = means[ident] - baseline.mean()
            leaderboard.loc[ident, "gain_ci90_low"], leaderboard.loc[ident, "gain_ci90_high"] = np.percentile(difference[:, j], [5, 95])
        signatures = {}
        for ident in columns:
            paths = []
            for zone in ["open", "closed"]:
                routes = pd.read_pickle(out / f"{ident}_{zone}_routes.pkl")
                paths.extend((zone, row.od_id, tuple(tuple(map(int, edge)) for edge in row.edge_route))
                             for row in routes.sort_values("od_id").itertuples())
            signature = hashlib.sha256(repr(paths).encode()).hexdigest()[:16]
            signatures.setdefault(signature, []).append(ident)
        write_json(out / "route_duplicates.json", signatures)
        uncertainty["distinct_route_sets"] = len(signatures)
        selection = scores.open.idxmax()
        uncertainty["selected_on_open"] = selection
        uncertainty["selected_on_open_closed_rank"] = float(scores.closed.rank(ascending=False, method="min").loc[selection])
        upper = np.triu_indices(len(columns), 1)
        halfwidths, separated = [], []
        for i, j in zip(*upper):
            low, high = np.percentile(boots[:, i] - boots[:, j], [5, 95])
            halfwidths.append((high - low) / 2)
            separated.append(low > 0 or high < 0)
        uncertainty.update(median_pairwise_half_width=float(np.median(halfwidths)), distinguishable_pairs_pct=100 * float(np.mean(separated)))
    if len(leaderboard):
        leaderboard.sort_values("closed", ascending=False).to_csv(out / "leaderboard.csv")
        leaderboard.groupby("family").agg(candidates=("closed", "size"), best_closed=("closed", "max"),
                                          median_closed=("closed", "median"), worst_closed=("closed", "min")).to_csv(out / "families.csv")
    filters = frame[frame.family.eq("filter_audit")].copy()
    filter_columns = [c for c in ["id", "label", "zone", "status", "input_edges", "kept_edges", "removed", "cut_off", "expected_pairs", "reachable", "routes", "counted_pairs", "score", "unsafe_m", "error_type", "error"] if c in filters]
    filter_table = filters[filter_columns]
    fragmentation_path = out / "filter_fragmentation.csv"
    if fragmentation_path.exists():
        filter_table = filter_table.merge(pd.read_csv(fragmentation_path).drop(columns="label"), on=["id", "zone"], how="left")
    filter_table.to_csv(out / "filters.csv", index=False)
    flagged = frame[frame.status.eq("ok") & frame["flags"].map(lambda value: isinstance(value, list) and bool(value))]
    write_json(out / "findings.json", flagged.drop(columns="traceback", errors="ignore").to_dict("records"))
    write_json(out / "uncertainty.json", uncertainty)
    honest_count = sum(case.kind == "honest" for case in cases)
    lines = ["# Стресс-тест командных решений", "", "Проверяется существующая система; библиотека и ноутбуки не исправлялись.", "",
             "## Набор и протокол", "", f"Честных вариантов в этом прогоне: {honest_count}. Они заморожены по открытым данным до оценки закрытой зоны. "
             "Каждый оценивается на всех исходных парах каждой зоны в отдельном процессе. Намеренные злоупотребления не участвуют в рейтинге.",
             "Проверки: начало/конец и связность пути, принадлежность рёбер фильтру, физическая длина из исходных данных, "
             "все исходные OD, независимый пересчёт балла, сохранение меток и базового сравнения.", "", "```", summary.to_string(index=False), "```", "",
             "## Данные", "", "```", pd.read_json(out / "data_summary.json").to_string(index=False), "```", "",
             "## Лучшие честные варианты по текущей метрике", ""]
    if len(leaderboard):
        top = leaderboard.sort_values("closed", ascending=False).head(15)
        lines += ["```", top[["label", "family", "open", "closed", "tier", "failed_closed", "unsafe_m", "search_ms", "audit_flags"]].round(2).to_string(), "```", ""]
        lines += ["## Различимость и выбор победителя", "", "```json", json.dumps(clean(uncertainty), ensure_ascii=False, indent=2), "```", "",
                  "90% интервалы и ярусы рассчитаны парным bootstrap по одним и тем же закрытым OD, как в текущем проверяющем коде. "
                  "Это исследовательское сравнение 100 вариантов: интервалы не корректируются на множественный выбор. "
                  "Поиск максимума по закрытой зоне нельзя превращать в подбор настроек студентами.", "",
                  "Для победителя сначала нужно принять правила допустимости фильтров и маршрутов; ошибки целостности исключают решение. "
                  "Потом сравнивать популярность на полном фиксированном банке закрытых OD. "
                  "Близкие баллы рассматривать по парным разницам/ярусам. Скорость — только дополнительный критерий среди корректных полных решений.", ""]
    lines += ["## Независимо обнаруженные нарушения", "", "```", flagged[[c for c in ["id", "label", "zone", "score", "audited_score", "flags"] if c in flagged]].to_string(index=False), "```", "",
              "Полные результаты: all_results.csv; фильтры: F*.json; независимые проверки путей: *_audit.csv; "
              "ошибки и тайм-ауты: отдельные журналы *.log. Список всех настроек: catalogue.json.", "",
              "Время поиска измерено существующей библиотекой, включая сохранённые измерения из кеша. "
              "Процессы выполнялись параллельно, поэтому эти времена не являются строгим серийным бенчмарком CPU.", ""]
    lines += ["## Фильтры отдельно", "", "```", filter_table[filter_table.zone.eq("closed")][[c for c in ["id", "label", "status", "kept_edges", "cut_off", "reachable_before_component_cut", "reachable", "counted_pairs", "score"] if c in filter_table]].round(2).to_string(index=False), "```", "",
              "reachable_before_component_cut — достижимость после маски, до выбора крупнейшей сильносвязной компоненты; reachable — после. "
              "Это разные потери: фильтр может разорвать граф, затем удаление остальных компонент дополнительно убирает пригодные локальные пути. "
              "Отсутствующие маршруты дают 0 на исходном наборе OD. Медиана времени на найденных маршрутах не измеряет способность решить все OD.", ""]
    completeness = out / "tag_completeness.csv"
    if completeness.exists():
        lines += ["## Заполненность исходных признаков", "", "Проценты среди рёбер базовой доступной сети. Пропуск метки не равен измеренному нулю; неизвестная популярность в маршруте считается как 50.", "", "```", pd.read_csv(completeness).round(2).to_string(index=False), "```", "",
                  "```json", (out / "data_diagnostics.json").read_text(encoding="utf-8"), "```", ""]
    sidewalk = out / "sidewalk_diagnostics.json"
    if sidewalk.exists():
        info = json.loads(sidewalk.read_text(encoding="utf-8"))
        lines += ["## Пешеходные дорожки, проезжая сеть и тупики", "",
                  "Тротуары здесь нельзя выделить точно: в выгрузке отсутствуют footway=sidewalk, sidewalk и crossing, а также bridge/tunnel/layer. "
                  "Для этой диагностики пешеходные рёбра — highway с footway или pedestrian; туда попадают и дорожки в парках. "
                  "streets_only включает service, residential и автомобильные классы, но исключает path/track/cycleway. without_footways сохраняет остальные типы. "
                  "OD не перемещались к другой сети, соединительные рёбра не добавлялись.", "",
                  "```", pd.read_csv(out / "sidewalk_topology.csv").to_string(index=False), "```", "",
                  "```", pd.read_csv(out / "sidewalk_OD_summary.csv").to_string(index=False), "```", "",
                  "Ноль маршрутов только по пешеходным рёбрам не доказывает отсутствия пешеходного пути в реальности: часть OD стоит на рёбрах другого типа, "
                  "а отдельно нарисованные дорожки могут не соединяться без проезда по другим типам. Числа origins_present/destinations_present отделяют это от разрывов внутри слоя.", "",
                  "Сильная связность базовой сети исключает заезд в участок без направленного возврата в основную сеть. "
                  "Двусторонняя тупиковая ветка при этом остаётся допустимой: из неё можно вернуться. Обычный Dijkstra с положительной стоимостью не станет делать бесполезный заезд и возврат; "
                  "необычные штрафы поворотов могут сделать такой манёвр выгодным. Кастомный поиск может ошибаться — это проверяется по возвращённому маршруту.", "",
                  "```json", json.dumps(info, ensure_ascii=False, indent=2), "```", "",
                  f"Найдено {info['footway_leaf_near_street_within_15m']} концов пешеходных веток не на узлах проезжей сети, но не далее 15 м от её геометрии. "
                  "Это кандидаты на просмотр, а не доказанные дефекты OSM: возможны ограждение, разные уровни или отсутствие разрешённого перехода. "
                  "Список: sidewalk_near_street_candidates.csv. Детали всех OD: sidewalk_OD.csv; проверки заездов: sidewalk_route_dead_ends.csv.", "",
                  "internal_baseline_* считает тупики исходной сети: при расширении сети решения они могут перестать быть тупиками. "
                  "Фактический заезд проверяется по internal_network_* в сети конкретного решения. У H063 штраф левого поворота равен 120, а разворот бесплатен; "
                  "такое сочетание может поощрять заезд в ветку и возврат вместо левого поворота.", "",
                  "![Реальные участки сети и альтернативы одного OD](sidewalk_examples.png)", ""]
        traversal_path = out / "sidewalk_route_dead_ends.csv"
        if traversal_path.exists():
            traversal = pd.read_csv(traversal_path)
            entered = traversal[traversal.internal_network_leaf_nodes.gt(0)]
            if len(entered):
                lines += ["Фактические заезды в нефинишные тупики в собственной сети решения (повторные узлы означают возврат/петлю, сами по себе не нарушение):", "", "```", entered.to_string(index=False), "```", ""]
    if len(leaderboard):
        lines += ["## Семейства решений", "", "```", pd.read_csv(out / "families.csv").round(2).to_string(index=False), "```", "",
                  "## Разница с кратчайшим", "", "Число объездов >20% приведено справочно, штрафа или подмены популярности нет.", "", "```", leaderboard.sort_values("closed", ascending=False).head(15)[["label", "gain_vs_baseline", "gain_ci90_low", "gain_ci90_high", "detours_over_20_pct_closed"]].round(2).to_string(), "```", "",
                  "Даже при парном bootstrap OD могут зависеть друг от друга через общие дороги. Ярусы текущего grader — эвристика сравнения, а не доказательство единственного победителя. "
                  "Допуск небезопасных дорог сейчас не является условием исключения: unsafe_m только показывается. Это правило нужно определить до соревнования.", ""]
        if "H080" in leaderboard.index:
            lines += ["Постоянная DummyRegressor тоже может улучшить маршрутный балл: её постоянный признак меняет свободный член в формуле стоимости и относительную силу остальных предпочтений. "
                      "Это не свидетельство качества предсказаний. Поэтому полезно сравнивать ML с простой перенастройкой весов, а не только с одной исходной формулой.", ""]
    grader = out / "official_grader_check.json"
    if grader.exists():
        lines += ["## Проверка настоящего grade_all", "", "```json", grader.read_text(encoding="utf-8"), "```", "",
                  "100 честных решений, один некорректный cost и пустой фильтр прошли через grade_all. Некорректный cost должен остаться в errors, пустой фильтр — получить 0.", ""]
    history = out / "attempt_history.json"
    if history.exists():
        attempts = json.loads(history.read_text(encoding="utf-8"))
        lines += ["## Повторные запуски", "", "Исходные статусы и причины повторов сохранены в attempt_history.json; они не стираются окончательными результатами.", ""]
        lines += [f"- {attempt['id']}: {attempt['reason']}; было: " + ", ".join(row["zone"] + "=" + row["status"] for row in attempt["previous"]) for attempt in attempts]
        lines += [""]
    if len(leaderboard) and set(["A007", "A011", "A012", "A013", "A014"]) <= set(frame.id):
        plot_report(leaderboard, filters, frame, out)
        closed_rows = frame[frame.zone.eq("closed")].set_index("id")
        winner = leaderboard.closed.idxmax()
        selected = uncertainty["selected_on_open"]
        highlights = ["## Что показал этот прогон", "",
                      f"Лучший наблюдаемый честный результат: **{winner}, {leaderboard.loc[winner, 'label']} — {leaderboard.loc[winner, 'closed']:.2f}**. "
                      f"Лучший по открытой зоне: {selected}; его место по закрытой: {uncertainty['selected_on_open_closed_rank']:.0f}. "
                      f"В первом ярусе текущего алгоритма сравнения: {uncertainty['top_tier_size']}. Это не основание объявлять единственного победителя по сотым балла.", "",
                      f"Пустая маска даёт 0, все 36 OD остаются в знаменателе. В то же время пример A013 возвращает одно и то же ребро вместо требуемых OD "
                      f"и получает **{closed_rows.loc['A013', 'score']:.2f}** вместо **{closed_rows.loc['A013', 'audited_score']:.2f}** после проверки концов пути. "
                      "Ребро выбрано моделью по открытым меткам внутри публично известной скрытой области, без чтения закрытых меток.", "",
                      f"Подмена меток через callback (A012): {closed_rows.loc['A012', 'score']:.2f} против {closed_rows.loc['A012', 'audited_score']:.2f}. "
                      f"Подмена физической длины в search (A014): {closed_rows.loc['A014', 'score']:.2f} против {closed_rows.loc['A014', 'audited_score']:.2f}. "
                      "Возврат обычной булевой маски не даёт такого доступа сам по себе: эти примеры намеренно исполняют произвольный Python-код и исследуют стек вызовов либо изменяют переданный граф.", "",
                      "Перед выбором победителя нужны проверки начала/конца и непрерывности пути, пересчёт длины по исходным рёбрам, фиксированный банк OD и неизменяемое базовое сравнение. "
                      "Пользовательский код должен исполняться с ограничением времени отдельно от доверенного оценивания и закрытых меток. Одного отдельного процесса, в котором ему по-прежнему доступны эти метки, недостаточно.", "",
                      "Нужно заранее решить, допустимы ли рёбра вне rideable и как учитывать отсутствие маршрутов при дополнительном сравнении скорости. "
                      "Сейчас основной балл штрафует потери OD, но unsafe_m не исключает решение, а время считается по найденным маршрутам. "
                      "Увеличение независимого закрытого набора OD поможет различать близкие решения; повторный подбор на нынешней закрытой зоне обесценит её как финальную проверку.", "",
                      "Этот скрипт обнаруживает перечисленные нарушения и ограничивает длительность своих запусков. Он не является песочницей для произвольного вредоносного кода и не доказывает отсутствие иных обходов.", "",
                      "![Результаты стресc-теста](overview.png)", ""]
        if "A016" in closed_rows.index and "A017" in closed_rows.index:
            highlights += [f"Дополнительная проверка A016: фильтр удалил все рёбра, но search восстановил их в своём графе; текущая оценка **{closed_rows.loc['A016', 'score']:.2f}**, "
                           "при независимой проверке принадлежности фильтру — **0**. "
                           f"A017 сохраняет правильные OD-концы, но возвращает разорванную цепочку: {closed_rows.loc['A017', 'score']:.2f} вместо 0. "
                           "Поэтому одной проверки концов пути недостаточно.", ""]
        lines[4:4] = highlights
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default="konstanz_more")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--phase", choices=["all", "filters", "candidates", "probes", "rerun", "append-probes", "report", "diagnostics", "sidewalks", "grader"], default="all")
    parser.add_argument("--case", action="append", default=[], help="Case ID to repeat with --phase rerun (repeatable)")
    parser.add_argument("--reason", default="manual repetition", help="Recorded explanation for --phase rerun")
    parser.add_argument("--fresh-cache", action="store_true", help="Ignore prior route caches in a repeated diagnostic")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker")
    parser.add_argument("--bundle", type=Path)
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
    with threadpool_limits(limits=1):
        if args.worker:
            worker(args.bundle, args.worker, args.output, args.fresh_cache)
            return
        out = (args.output or ROOT / "outputs/stress_hackathon" / args.pack).resolve()
        out.mkdir(parents=True, exist_ok=True)
        before = protected_hashes()
        public = ROOT / "notebooks/data/hackathon" / args.pack
        private = ROOT / "outputs/hackathon_private" / args.pack
        inputs = input_hashes(public, private)
        if args.phase in {"rerun", "append-probes", "report", "diagnostics", "sidewalks", "grader"}:
            previous = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
            assert previous["protected_hashes"] == before, "Protected source files changed since the run"
            assert previous.get("input_hashes", inputs) == inputs, "Dataset changed since the run"
            if args.phase in {"diagnostics", "sidewalks"}:
                if args.phase == "sidewalks":
                    sidewalk_diagnostics(load_city(public), private, out)
                else:
                    data_diagnostics(load_city(public), private, out)
                assert before == protected_hashes()
                return
            bundle_path = out / "bundle.pkl"
            saved = cloudpickle.loads(bundle_path.read_bytes())
            cases = saved["cases"]
            if args.phase == "append-probes":
                # Extend only adversarial diagnostics; the original 100 candidates stay frozen.
                model = next(c for c in cases if c.id == "H078").chosen["cost"].make_features.model
                _, proposed = make_audits(load_city(public), model)
                extra = [c for c in proposed if c.id not in {old.id for old in cases}]
                cases.extend(extra)
                bundle_path.write_bytes(cloudpickle.dumps(saved))
                write_json(out / "catalogue.json", [{"id": c.id, "label": c.label, "kind": c.kind, "family": c.family,
                                                     "settings": c.settings, "cost": str(c.chosen.get("cost")), "filter": str(c.keep), "turns": str(c.chosen.get("turns")),
                                                     "search": str(c.chosen.get("search"))} for c in cases])
                run_cases(extra, bundle_path, out, args.jobs)
            if args.phase == "rerun":
                assert args.case and set(args.case) <= {c.id for c in cases}, "Supply existing --case IDs"
                history_path = out / "attempt_history.json"
                history = json.loads(history_path.read_text(encoding="utf-8")) if history_path.exists() else []
                for case in cases:
                    if case.id in args.case:
                        history.append({"id": case.id, "reason": args.reason, "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                        "previous": json.loads((out / (case.id + ".json")).read_text(encoding="utf-8"))})
                write_json(history_path, history)
                # Reconstruct this script's rules after a harness-only correction; do not retrain or retune candidates.
                for case in cases:
                    turn = case.chosen.get("turns")
                    if type(turn).__name__ == "Turn":
                        case.chosen["turns"] = Turn(**vars(turn))
                    search = case.chosen.get("search")
                    if case.id in args.case and type(search).__name__ == "Search":
                        case.chosen["search"] = Search(**vars(search))
                bundle_path.write_bytes(cloudpickle.dumps(saved))
                run_cases([c for c in cases if c.id in args.case], bundle_path, out, args.jobs, args.fresh_cache)
            if args.phase == "grader":
                run_official_grader(saved, out)
            records = [row for case in cases for row in json.loads((out / (case.id + ".json")).read_text(encoding="utf-8"))]
            # An open-zone hang prevents the closed evaluation; never describe it as a second measured timeout.
            for case in cases:
                rows = [row for row in records if row["id"] == case.id]
                if len(rows) == 2 and all(row["status"] == "timeout" for row in rows):
                    next(row for row in rows if row["zone"] == "closed")["status"] = "not_run_after_timeout"
                    write_json(out / (case.id + ".json"), rows)
            write_json(out / "results.json", records)
            report(cases, records, out)
            previous.update(honest_candidates=sum(c.kind == "honest" for c in cases),
                            filters=sum(c.family == "filter_audit" for c in cases),
                            adversarial=sum(c.kind == "adversarial" for c in cases),
                            harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), input_hashes=inputs, source_changes=[])
            previous["versions"].update({name: version(name) for name in ["scikit-learn", "geopandas", "cloudpickle"]})
            previous.update(seed=71, bootstrap_draws=4000, updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            write_json(out / "run_manifest.json", previous)
            assert before == protected_hashes()
            print("Updated report:", out / "report.md", flush=True)
            return
        city = load_city(public)
        prepare(city, private, out)
        data_diagnostics(city, private, out)
        regular, extras, invalid = filter_specs(city)
        filters = [Case(f"F{i:03d}", label, "filter_audit", solution(comfort_cost(BASE), turns=None), keep,
                        "invalid" if i >= len(regular) + len(extras) else "filter_audit", vars(keep))
                   for i, (label, keep) in enumerate(regular + extras + invalid)]
        bundle = out / "bundle.pkl"
        common = {"public": public, "private": private, "output": out}
        cases, records = [], []
        if args.phase == "candidates":
            previous = json.loads((out / "run_manifest.json").read_text(encoding="utf-8"))
            assert previous["pack"] == args.pack and previous["protected_hashes"] == before, "The source/data context changed; rerun the filter phase"
            assert previous.get("input_hashes", inputs) == inputs, "Dataset changed since the filter phase"
            for case in filters:
                saved = json.loads((out / (case.id + ".json")).read_text(encoding="utf-8"))
                assert len(saved) == 2 and all(row["id"] == case.id and row["label"] == case.label for row in saved)
                records.extend(saved)
            cases = filters[:]
            print("Reused the completed independent filter audit:", len(filters), flush=True)
        elif args.phase != "probes":
            bundle.write_bytes(cloudpickle.dumps({**common, "cases": filters}))
            print("Independent filter audit:", len(filters), "filters", flush=True)
            records = run_cases(filters, bundle, out, args.jobs)
            cases = filters[:]
        if args.phase in {"all", "candidates", "probes"}:
            print("Training models using open labels only...", flush=True)
            models = {"ready_hgb": train_popularity(city)} if args.phase == "probes" else fit_models(city, out)
            candidates = [] if args.phase == "probes" else make_candidates(city, models)
            _, adversarial = make_audits(city, models["ready_hgb"])
            cases += candidates + adversarial
            bundle.write_bytes(cloudpickle.dumps({**common, "cases": cases}))
            write_json(out / "catalogue.json", [{"id": c.id, "label": c.label, "kind": c.kind, "family": c.family,
                                                "settings": c.settings, "cost": str(c.chosen.get("cost")), "filter": str(c.keep), "turns": str(c.chosen.get("turns")),
                                                "search": str(c.chosen.get("search"))} for c in cases])
            print("Frozen candidates:", len(candidates), "; separate adversarial probes:", len(adversarial), flush=True)
            records += run_cases(adversarial + candidates, bundle, out, args.jobs)
            if candidates:
                sidewalk_diagnostics(city, private, out)
        write_json(out / "results.json", records)
        report(cases, records, out)
        if args.phase == "all":
            run_official_grader(cloudpickle.loads(bundle.read_bytes()), out)
            report(cases, records, out)
        after = protected_hashes()
        changed = [path for path in before if before[path] != after.get(path)]
        manifest = {"pack": args.pack, "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    "phase": args.phase, "jobs": args.jobs, "honest_candidates": sum(c.kind == "honest" for c in cases),
                    "filters": sum(c.family == "filter_audit" for c in cases), "adversarial": sum(c.kind == "adversarial" for c in cases), "source_changes": changed,
                    "protected_hashes": before, "input_hashes": inputs, "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    "versions": {"python": sys.version, "numpy": np.__version__, "pandas": pd.__version__, "networkx": nx.__version__}}
        manifest["versions"].update({name: version(name) for name in ["scikit-learn", "geopandas", "cloudpickle"]})
        manifest.update(seed=71, bootstrap_draws=4000, updated_at_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        write_json(out / "run_manifest.json", manifest)
        assert not changed, changed
        print("Finished. Report:", out / "report.md", flush=True)


if __name__ == "__main__":
    main()
