"""Organizer: are the tried solutions really different, and can 25 pairs tell their scores apart?

    python tools/check_distinctness.py --pack konstanz

Diversity: for every two solutions, the shared share of route length (|A∩B| / |A∪B| by length) averaged
over the pairs of both zones; groups of practically identical solutions.
Distinguishability: paired bootstrap over pairs of the same zone. Two solutions are told apart when the
90% interval of their score difference excludes zero; tiers group solutions not told apart from the
tier's best. Writes outputs/solution_ideas/<pack>/: diversity and tier tables, similarity map, CI plot.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"tools"))
from cycle_routing.grading import tiers
from cycle_routing.workshop import build_network, closed_city, load_city, rideable, score
from try_solutions import ideas

SAME = 0.95      # mean shared length above which two solutions count as practically the same
BOOTS = 2000


def edge_lengths(result):
    length = dict(zip(zip(result.city.osm.u, result.city.osm.v, result.city.osm.key), result.city.osm.length_m))
    return {(r.algorithm, r.od_id): {tuple(map(int, e)): length[tuple(map(int, e))] for e in r.edge_route}
            for r in result.routes.itertuples()}


def similarity(results, names):
    """Mean over pairs of shared route length |A∩B| / |A∪B| for every two solutions."""
    sims = np.zeros((len(names), len(names)))
    count = 0
    for result in results.values():
        paths = edge_lengths(result)
        for od in result.city.pairs.od_id:
            sets = [paths.get((name, od), {}) for name in names]
            for i, a in enumerate(sets):
                for j in range(i, len(sets)):
                    b = sets[j]
                    union = sum({**a, **b}.values())
                    shared = sum(v for k, v in a.items() if k in b)
                    sims[i, j] += shared / union if union else 1.0
            count += 1
    sims /= count
    sims = np.triu(sims) + np.triu(sims, 1).T
    return pd.DataFrame(sims, index=names, columns=names)


def groups(sim):
    """Connected groups of solutions whose routes share at least SAME of their length."""
    left, found = list(sim.index), []
    while left:
        group, frontier = {left[0]}, [left[0]]
        while frontier:
            current = frontier.pop()
            for other in sim.index[sim.loc[current].ge(SAME)]:
                if other not in group:
                    group.add(other)
                    frontier.append(other)
        found.append(sorted(group, key=list(sim.index).index))
        left = [name for name in left if name not in group]
    return [g for g in found if len(g) > 1]


def bootstrap(result, names, rng):
    """Per-pair counted popularity, resampled over pairs: (boots x solutions) of means."""
    table = result.per_od.pivot(index="od_id", columns="algorithm", values="counted")[["shortest", *names]]
    draws = rng.integers(0, len(table), size=(BOOTS, len(table)))
    return table, table.to_numpy()[draws].mean(axis=1)


def zone_report(result, names, rng):
    table, boots = bootstrap(result, names, rng)
    columns = list(table.columns)
    means = pd.Series(table.mean().to_numpy(), index=columns)
    base = boots[:, columns.index("shortest")]
    rows = {}
    for i, name in enumerate(columns):
        if name == "shortest":
            continue
        diff = boots[:, i] - base
        rows[name] = {"популярность": means[name], "минус кратчайший": means[name] - means["shortest"],
                      "90% ДИ от": np.percentile(diff, 5), "90% ДИ до": np.percentile(diff, 95)}
    report = pd.DataFrame(rows).T
    report["лучше кратчайшего"] = report["90% ДИ от"] > 0
    report = report.join(tiers(means.drop("shortest"), boots[:, 1:]))
    pairs = boots[:, 1:, None] - boots[:, 1:, None].transpose(0, 2, 1)
    low, high = np.percentile(pairs, 5, axis=0), np.percentile(pairs, 95, axis=0)
    upper = np.triu_indices(len(names), 1)
    told_apart = ((low > 0) | (high < 0))[upper]
    width = (high - low)[upper] / 2
    summary = {"pairs": len(table), "solution_pairs_told_apart_pct": round(100 * told_apart.mean(), 1),
               "median_ci_half_width": round(float(np.median(width)), 2),
               "better_than_shortest": int(report["лучше кратчайшего"].sum()), "tiers": int(report["ярус"].max())}
    return report.sort_values("популярность", ascending=False), summary


def draw(sim, reports, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    order = sim.index[leaves_list(linkage(squareform(1 - sim.to_numpy(), checks=False), "average"))]
    fig, ax = plt.subplots(figsize=(13, 11), layout="constrained")
    image = ax.imshow(sim.loc[order, order], vmin=0.3, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.6, label="общая доля длины маршрутов, среднее по парам")
    ax.set_title("Насколько маршруты решений совпадают (обе зоны)")
    fig.savefig(out/"similarity.png", dpi=110)
    fig, axes = plt.subplots(1, 2, figsize=(14, 10), layout="constrained", sharey=True)
    order = reports["closed"].index[::-1]
    for ax, (zone, report) in zip(axes, reports.items()):
        report = report.loc[order]
        y = np.arange(len(report))
        ax.errorbar(report["минус кратчайший"], y, xerr=[report["минус кратчайший"] - report["90% ДИ от"],
                    report["90% ДИ до"] - report["минус кратчайший"]], fmt="o", color="#2563eb", ecolor="#93c5fd")
        ax.axvline(0, color="#6b7280", linestyle=":")
        ax.set_yticks(y, report.index, fontsize=7)
        ax.set_title(f"{zone}: популярность минус кратчайший, 90% ДИ")
    fig.savefig(out/"distinguishability.png", dpi=110)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", default="konstanz")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = ROOT/"outputs/solution_ideas"/args.pack
    out.mkdir(parents=True, exist_ok=True)
    city = load_city(ROOT/"notebooks/data/hackathon"/args.pack)
    closed = closed_city(city, ROOT/"outputs/hackathon_private"/args.pack)
    tried = {name: entry[1] for name, entry in ideas(city).items()}
    names = list(tried)
    results = {zone: score(z, tried, network=build_network(z, rideable)) for zone, z in (("open", city), ("closed", closed))}

    sim = similarity(results, ["shortest", *names])
    same_groups = groups(sim.drop(index="shortest", columns="shortest"))
    per_od_distinct = pd.concat([
        r.routes.assign(p=r.routes.edge_route.map(lambda p: tuple(map(tuple, p)))).groupby("od_id").p.nunique()
        for r in results.values()])
    diversity = pd.DataFrame({
        "сходство с кратчайшим": sim["shortest"].drop("shortest"),
        "самое похожее решение": sim.drop(index="shortest", columns="shortest").apply(
            lambda row: row.drop(row.name).idxmax(), axis=1),
        "сходство с ним": sim.drop(index="shortest", columns="shortest").apply(lambda row: row.drop(row.name).max(), axis=1),
    }).round(3)
    rng = np.random.default_rng(args.seed)
    reports, summaries = {}, {}
    for zone, result in results.items():
        reports[zone], summaries[zone] = zone_report(result, names, rng)
        reports[zone].round(2).to_csv(out/f"tiers_{zone}.csv")
    diversity.to_csv(out/"diversity.csv")
    sim.round(3).to_csv(out/"similarity.csv")
    draw(sim, reports, out)
    from scipy.stats import spearmanr
    fair = [name for name in names if not name.startswith("ЗАПРЕЩЕНО")]
    closed_per_od = results["closed"].per_od
    summary = {
        "pairs_per_zone": {zone: len(r.city.pairs) for zone, r in results.items()},
        "split": city.manifest["split"],
        "ideas_rank_agreement_open_closed": round(float(spearmanr(reports["open"].loc[fair, "популярность"],
                                                                  reports["closed"].loc[fair, "популярность"]).statistic), 3),
        "closed_routes_measured_in_zone_median_pct": round(float(closed_per_od[closed_per_od.algorithm.ne("shortest")]
                                                                 .coverage_pct.median()), 1),
        "distinct_routes_per_pair": {"median": float(per_od_distinct.median()), "min": int(per_od_distinct.min()),
                                     "max": int(per_od_distinct.max()), "solutions": len(names) + 1},
        "mean_similarity_between_solutions": round(float(sim.drop(index="shortest", columns="shortest")
                                                         .to_numpy()[np.triu_indices(len(names), 1)].mean()), 3),
        "practically_identical_groups": same_groups, **{f"scores_{z}": s for z, s in summaries.items()},
    }
    (out/"distinctness.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nразнообразие:\n" + diversity.sort_values("сходство с ним", ascending=False).to_string())
    for zone, report in reports.items():
        print(f"\n{zone}:\n" + report.round(2).to_string())


if __name__ == "__main__":
    main()
