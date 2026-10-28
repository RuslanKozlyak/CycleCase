"""Organizer: try ideas teams could write themselves in the notebook blocks; score them on both zones.

    python tools/try_solutions.py --pack konstanz

Every idea is written the way a team would write it in the notebook: weights, own features, own cost,
turns, search, ML. A deliberately forbidden "cheat" (cost from lab.trips) shows why the closed zone exists.
Writes outputs/solution_ideas/<pack>/ideas.csv and ideas.md.
"""
from pathlib import Path
import argparse
import math
import sys

import networkx as nx
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from cycle_routing.workshop import (
    build_network, closed_city, comfort_cost, features, load_city, ml_feature, presets, rideable, score, solution,
    train_popularity,
)

BASE = {"крупная_дорога": 0.6, "плохое_покрытие": 0.4, "велодорожка": -0.4, "без_освещения": 0.2}


# --- own features: what a team could add in my_features(roads) -----------------------------------------

def asphalt(roads):
    return pd.DataFrame({"асфальт": roads["покрытие"].eq("asphalt").astype(float)}, index=roads.index)


def bikes_welcome(roads):
    return pd.DataFrame({"велосипед_разрешён": roads["велосипед"].isin(["yes", "designated"]).astype(float)}, index=roads.index)


def calm(roads):
    return pd.DataFrame({"тихая": roads["макс_скорость"].le(30).astype(float)}, index=roads.index)


def main_with_lane(roads):
    lane = roads["велополоса"].notna() & ~roads["велополоса"].isin(["no", "separate"])
    main = roads["тип_дороги"].isin(["primary", "secondary", "tertiary"])
    return pd.DataFrame({"магистраль_с_полосой": (main & lane).astype(float)}, index=roads.index)


def long_segment(roads):
    return pd.DataFrame({"длинный_участок": roads["длина_м"].gt(150).astype(float)}, index=roads.index)


# --- own cost: a whole formula --------------------------------------------------------------------------

TYPE_PRICE = {"cycleway": 0.5, "living_street": 0.7, "residential": 0.8, "path": 0.9, "tertiary": 1.0, "unclassified": 1.0,
              "service": 1.1, "track": 1.2, "secondary": 1.3, "footway": 1.4, "pedestrian": 1.3, "primary": 1.6}


def by_type(roads):
    first = roads["тип_дороги"].str.split(";").str[0]
    return roads["длина_м"] * first.map(TYPE_PRICE).fillna(1.0)


def exponential(roads):
    f = features(roads)
    return roads["длина_м"] * np.exp(0.7 * f["крупная_дорога"] - 0.6 * f["велодорожка"] + 0.4 * f["плохое_покрытие"])


# --- turns ----------------------------------------------------------------------------------------------

def junction_turns(graph, incoming, outgoing):
    """A turn costs only at a real junction; a bend of the same street is free."""
    return 40.0 if graph.out_degree(outgoing[0]) > 2 else 0.0


# --- search: graph is a MultiDiGraph with "cost" and "length_m" on every edge (turns=None) ---------------

def hop(graph, u, v, attr):
    return min(data[attr] for data in graph[u][v].values())


def bidirectional(graph, source, target, weight):
    return nx.bidirectional_dijkstra(graph, source, target, weight=weight)[1]


def astar(graph, source, target, weight):
    xy = graph.nodes
    # cost is at least 0.1 x length, so 0.1 x straight distance never overestimates
    return nx.astar_path(graph, source, target, weight=weight,
                         heuristic=lambda a, b: 0.1 * math.dist((xy[a]["x"], xy[a]["y"]), (xy[b]["x"], xy[b]["y"])))


def candidates(graph, source, target, weight, shares=(1.0, 0.7, 0.4, 0.0)):
    """Paths for a mix of comfort and length: share 1 is pure comfort, 0 is the shortest path."""
    for share in shares:
        mix = lambda u, v, d, s=share: min(s * a[weight] + (1 - s) * a["length_m"] for a in d.values())
        path = nx.dijkstra_path(graph, source, target, weight=mix)
        yield path, sum(hop(graph, u, v, "length_m") for u, v in zip(path, path[1:]))


def within_detour(graph, source, target, weight, limit=1.15):
    """The most comfortable path that is at most 15% longer than the shortest one."""
    found = list(candidates(graph, source, target, weight))
    shortest = found[-1][1]
    return next(path for path, length in found if length <= limit * shortest)


def best_by_ml(prediction, limit=1.15):
    """Among comfort/length mixes within the detour limit, take the path with the highest predicted popularity."""
    def search(graph, source, target, weight):
        found = list(candidates(graph, source, target, weight))
        shortest = found[-1][1]
        def predicted(path):
            hops = list(zip(path, path[1:]))
            return np.average([prediction.get(h, 0.0) for h in hops], weights=[hop(graph, *h, "length_m") for h in hops])
        return max((path for path, length in found if length <= limit * shortest), key=predicted)
    return search


# --- own ML: a different model on the same open labels --------------------------------------------------

def forest_feature(lab):
    from sklearn.ensemble import RandomForestRegressor
    known = lab.trips.notna()
    x = features(lab.roads).drop(columns="длина_м")
    model = RandomForestRegressor(n_estimators=150, max_depth=12, min_samples_leaf=20, n_jobs=-1, random_state=0)
    model.fit(x[known], np.log1p(lab.trips[known]), sample_weight=lab.roads["длина_м"][known])

    def rf_popularity(roads):
        predicted = pd.Series(model.predict(features(roads).drop(columns="длина_м")), index=roads.index)
        return pd.DataFrame({"rf_популярность": predicted.rank(pct=True)}, index=roads.index)
    return rf_popularity


def popular_classifier(lab):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    known = lab.trips.notna()
    x = features(lab.roads).drop(columns="длина_м")
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(x[known], lab.popularity[known].ge(75))

    def chance_popular(roads):
        return pd.DataFrame({"шанс_популярной": model.predict_proba(features(roads).drop(columns="длина_м"))[:, 1]},
                            index=roads.index)
    return chance_popular


def ideas(lab):
    model = train_popularity(lab)
    ml = ml_feature(model)
    known_popularity = lab.popularity  # open zone only: closed roads are NaN here

    def cheat(roads):  # forbidden: the cost reads the target
        return roads["длина_м"] * (1.6 - known_popularity.reindex(roads.index).fillna(50) / 100)

    prediction = model.predict_log_trips(lab.osm)
    by_hop = {}
    for u, v, p in zip(lab.osm.u, lab.osm.v, prediction):
        by_hop[(u, v)] = max(p, by_hop.get((u, v), -np.inf))
    w = lambda weights, extra=(): comfort_cost(weights, extra=extra)
    return {
        # block, idea, solution
        "веса: пример из ноутбука": ("веса", solution(w(BASE), turns="default")),
        "веса: любитель велодорожек": ("веса", solution(w({"велодорожка": -0.8}), turns=None)),
        "веса: прочь от машин": ("веса", solution(w({"крупная_дорога": 1.5, "скорость": 0.01}), turns=None)),
        "веса: тихие жилые улицы": ("веса", solution(w({"жилая_улица": -0.4, "крупная_дорога": 0.8}), turns=None)),
        "веса: без грунта": ("веса", solution(w({"плохое_покрытие": 1.0, "покрытие_неизвестно": 0.2}), turns=None)),
        "веса: без тротуаров": ("веса", solution(w({**BASE, "пешеходная": 0.8}), turns=None)),
        "веса: освещённые": ("веса", solution(w({"без_освещения": 0.6}), turns=None)),
        "признак: асфальт": ("свои признаки", solution(w({**BASE, "асфальт": -0.3}, asphalt), turns=None)),
        "признак: велосипед разрешён": ("свои признаки", solution(w({**BASE, "велосипед_разрешён": -0.4}, bikes_welcome), turns=None)),
        "признак: тихие 30 км/ч": ("свои признаки", solution(w({**BASE, "тихая": -0.3}, calm), turns=None)),
        "признак: магистраль с полосой": ("свои признаки", solution(w({**BASE, "магистраль_с_полосой": -0.5}, main_with_lane), turns=None)),
        "признак: длинные участки": ("свои признаки", solution(w({**BASE, "длинный_участок": -0.3}, long_segment), turns=None)),
        "стоимость: цена по типу дороги": ("своя стоимость", solution(by_type, turns=None)),
        "стоимость: экспонента": ("своя стоимость", solution(exponential, turns=None)),
        "повороты: без штрафов": ("повороты", solution(w(BASE), turns=None)),
        "повороты: min_turns": ("повороты", solution(w(BASE), turns="min_turns")),
        "повороты: налево 60 м": ("повороты", solution(w(BASE), turns={"left_penalty_m": 60, "right_penalty_m": 15})),
        "повороты: только на перекрёстках": ("повороты", solution(w(BASE), turns=junction_turns)),
        "поиск: двусторонний Дейкстра": ("поиск", solution(w(BASE), turns=None, search=bidirectional)),
        "поиск: A*": ("поиск", solution(w(BASE), turns=None, search=astar)),
        "поиск: объезд не больше 15%": ("поиск", solution(w({"велодорожка": -0.8, "крупная_дорога": 1.0}), turns=None, search=within_detour)),
        "поиск: лучший из кандидатов по ML": ("поиск + ML", solution(w(BASE), turns=None, search=best_by_ml(by_hop))),
        "ML: сила 0.4": ("ML", solution(w({**BASE, "ml_популярность": -0.4}, ml), turns=None)),
        "ML: сила 0.8": ("ML", solution(w({**BASE, "ml_популярность": -0.8}, ml), turns=None)),
        "ML: сила 1.5": ("ML", solution(w({**BASE, "ml_популярность": -1.5}, ml), turns=None)),
        "ML: только прогноз": ("ML", solution(w({"ml_популярность": -0.8}, ml), turns=None)),
        "ML: прогноз + min_turns": ("ML", solution(w({**BASE, "ml_популярность": -0.8}, ml), turns="min_turns")),
        "ML: свой случайный лес": ("свой ML", solution(w({**BASE, "rf_популярность": -0.8}, forest_feature(lab)), turns=None)),
        "ML: классификатор «популярная улица»": ("свой ML", solution(w({**BASE, "шанс_популярной": -0.8}, popular_classifier(lab)), turns=None)),
        "ЗАПРЕЩЕНО: стоимость из lab.trips": ("нарушение правил", solution(cheat, turns=None)),
        **{f"пресет: {name}": ("пресет", preset) for name, preset in presets().items()},
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", default="konstanz")
    args = parser.parse_args()
    out = ROOT/"outputs/solution_ideas"/args.pack
    out.mkdir(parents=True, exist_ok=True)
    lab = load_city(ROOT/"notebooks/data/hackathon"/args.pack)
    closed = closed_city(lab, ROOT/"outputs/hackathon_private"/args.pack)
    catalogue = ideas(lab)
    tried = {name: entry[1] for name, entry in catalogue.items()}
    results = {zone: score(z, tried, network=build_network(z, rideable)) for zone, z in (("open", lab), ("closed", closed))}
    rows = {}
    for name, (block, _) in catalogue.items():
        o, c = results["open"].table.loc[name], results["closed"].table.loc[name]
        same = results["open"].routes.pipe(lambda r: r[r.algorithm.isin([name, "shortest"])])
        paths = same.assign(p=same.edge_route.map(lambda p: tuple(map(tuple, p)))).pivot(index="od_id", columns="algorithm", values="p")
        rows[name] = {"блок": block, "открытая": o["популярность"], "закрытая": c["популярность"],
                      "популярнее кратчайшего (закр.)": c["популярнее кратчайшего"], "объезд, %": c["объезд, %"],
                      "поворотов": c["поворотов"], "время поиска, мс": o["время поиска, мс"],
                      "путь = кратчайший, пар": int((paths[name] == paths["shortest"]).sum())}
    table = pd.DataFrame(rows).T.sort_values("закрытая", ascending=False)
    base = {zone: results[zone].table.loc["кратчайший", "популярность"] for zone in results}
    table.to_csv(out/"ideas.csv")
    (out/"ideas.md").write_text(f"Кратчайший: открытая {base['open']}, закрытая {base['closed']}\n\n```\n"
                                + table.to_string() + "\n```\n", encoding="utf-8")
    print(f"кратчайший: открытая {base['open']}, закрытая {base['closed']}; пар {len(lab.pairs)} / {len(closed.pairs)}")
    print(table.to_string())


if __name__ == "__main__":
    main()
