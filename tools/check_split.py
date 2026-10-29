"""Organizer check of a split: do ready presets and weight variants rank the same in the open and closed zone?

    python tools/check_split.py --pack konstanz

Writes outputs/zone_planning/<pack>/: zone map, both score tables and the agreement report.
Agreement is compared with two random halves of the same zone: OD noise caps what any split can reach.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from cycle_routing.workshop import closed_city, comfort_cost, load_city, presets, score, solution
from cycle_routing.workshop_compat import legacy_features


def candidates():
    base = {"крупная_дорога": 0.6, "плохое_покрытие": 0.4, "велодорожка": -0.4, "без_освещения": 0.2}
    variants = {"w_bike0": {**base, "велодорожка": 0}, "w_bike_strong": {**base, "велодорожка": -0.8},
                "w_major0": {**base, "крупная_дорога": 0}, "w_major2": {**base, "крупная_дорога": 2},
                "w_rough2": {**base, "плохое_покрытие": 2}}
    return {**presets(), **{name: solution(comfort_cost(w, features=legacy_features), turns=None, label=name) for name, w in variants.items()}}


def agreement(per_open, per_closed, rng, boots=1000):
    piv = {z: d.pivot(index="od_id", columns="algorithm", values="counted") for z, d in (("open", per_open), ("closed", per_closed))}
    piv["closed"] = piv["closed"][piv["open"].columns]
    rho = spearmanr(piv["open"].mean(), piv["closed"].mean()).statistic
    resample = lambda t: t.iloc[rng.integers(0, len(t), len(t))].mean()
    boot = [spearmanr(resample(piv["open"]), resample(piv["closed"])).statistic for _ in range(boots)]
    halves = []
    for table in piv.values():
        for _ in range(boots // 2):
            order = rng.permutation(len(table))
            halves.append(spearmanr(table.iloc[order[: len(table)//2]].mean(), table.iloc[order[len(table)//2:]].mean()).statistic)
    return {"spearman_open_vs_closed": round(float(rho), 3), "bootstrap_5_95": np.round(np.percentile(boot, [5, 95]), 3).tolist(),
            "same_zone_random_halves_median": round(float(np.median(halves)), 3)}


def draw(city, zones, closed, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from cycle_routing.routing import build_routes
    fig, ax = plt.subplots(figsize=(12, 12), layout="constrained")
    for name, color in {"open": "#93c5fd", "closed": "#fdba74", "buffer": "#d1d5db"}.items():
        city.osm[zones.eq(name).to_numpy()].plot(ax=ax, color=color, linewidth=0.4)
    for zone_lab, color in ((city, "#1d4ed8"), (closed, "#c2410c")):
        build_routes(city.osm[city.reference], city.nodes, zone_lab.pairs, {}, cache_dir=city.folder/"route_cache").plot(
            ax=ax, color=color, linewidth=1.6)
    counts = zones.value_counts()
    ax.set_title(f"Открытая зона (синяя) {counts.get('open', 0):,} участков, закрытая (оранж.) {counts.get('closed', 0):,},"
                 f" буфер {counts.get('buffer', 0):,}. Линии — кратчайшие пути пар", fontsize=11)
    ax.set_aspect("equal"); ax.set_axis_off()
    fig.savefig(out, dpi=100)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", default="konstanz")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    out = ROOT/"outputs/zone_planning"/args.pack
    out.mkdir(parents=True, exist_ok=True)
    city = load_city(ROOT/"notebooks/data/hackathon"/args.pack)
    private = ROOT/"outputs/hackathon_private"/args.pack
    closed = closed_city(city, private)
    zones = pd.read_parquet(private/"zones.parquet").set_index("edge_id").zone.reindex(city.osm.edge_id)
    tried = candidates()
    results = {"open": score(city, tried), "closed": score(closed, tried)}
    for zone, result in results.items():
        result.table.to_csv(out/f"{zone}_table.csv")
        print(f"\n{zone}: {len(result.city.pairs)} OD\n{result.table.to_string()}")
    city_km = {z: round(city.osm.length_m[zones.eq(z).to_numpy()].sum()/1000, 1) for z in ("open", "closed", "buffer")}
    report = {"pack": args.pack, "split": city.manifest["split"], "pairs": {z: len(r.city.pairs) for z, r in results.items()},
              "zone_km": city_km, **agreement(results["open"].per_od, results["closed"].per_od, np.random.default_rng(args.seed))}
    (out/"report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    draw(city, zones, closed, out/"zones_map.png")
    print("\n" + json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
