"""Organizer: load the solutions teams send, score them on the closed zone, rank them fairly.

A submission is the folder written by ``export_solution`` (``submission.json`` + ``solution.pkl``) or a ZIP of it.
Loading a submission runs the team's code, so open only solutions from participants.
Close scores are not ranked apart: tiers group teams whose difference from the tier's best is within the noise of
the closed pairs (paired bootstrap over pairs, 90%).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
from tqdm import tqdm

from .workshop import CityData, Result, build_network, closed_city, load_submission, score

BOOTS = 2000
COLUMNS = ["популярность", "популярнее кратчайшего", "объезд, %", "поворотов", "неудобные участки, м",
           "время поиска, мс", "открытая зона"]


@dataclass
class Submission:
    team: str
    folder: Path | None = None
    solution: dict | None = None
    keep: object = None
    status: str = "ок"


def _unzip(path: Path, target: Path) -> Path:
    shutil.rmtree(target, ignore_errors=True)
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if target.resolve() not in (target / name).resolve().parents and (target / name).resolve() != target.resolve():
                raise ValueError(f"в архиве путь вне папки: {name}")
        archive.extractall(target)
    return target


def load_submissions(folder, city: CityData) -> list[Submission]:
    """Every subfolder or ZIP in ``folder`` is one team, named after it. Broken ones are kept with a status."""
    folder = Path(folder)
    found = []
    for path in sorted(folder.iterdir()):
        if path.name.startswith(".") or not (path.is_dir() or path.suffix.lower() == ".zip"):
            continue
        team = path.stem
        try:
            if path.suffix.lower() == ".zip":
                path = _unzip(path, folder / ".unpacked" / team)
            meta_path = next(path.rglob("submission.json"), None)
            if meta_path is None:
                raise ValueError("нет submission.json: пришлите папку из export_solution")
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("dataset_id") != city.manifest["dataset_id"]:
                raise ValueError("решение сделано для другого набора данных")
            chosen, keep = load_submission(meta_path.parent)
            found.append(Submission(team, meta_path.parent, chosen, keep))
        except Exception as error:  # a broken submission must not stop the others
            found.append(Submission(team, path, status=f"не загружено: {error}"))
    return found


def submissions_table(submissions: list[Submission]) -> pd.DataFrame:
    return pd.DataFrame([{"команда": s.team, "решение": (s.solution or {}).get("label", ""), "статус": s.status}
                         for s in submissions]).set_index("команда")


def bootstrap(per_od: pd.DataFrame, names, rng, boots=BOOTS):
    """Means of the counted popularity over resampled pairs: (boots x names)."""
    table = per_od.pivot(index="od_id", columns="algorithm", values="counted")[list(names)]
    draws = rng.integers(0, len(table), size=(boots, len(table)))
    return table.to_numpy()[draws].mean(axis=1)


def tiers(means: pd.Series, boots: np.ndarray) -> pd.Series:
    """Greedy tiers: a solution stays in the current tier while the tier's best is not surely better (90%)."""
    order = means.sort_values(ascending=False).index
    position = {name: i for i, name in enumerate(means.index)}
    result, tier, top = {}, 1, order[0]
    for name in order:
        if np.percentile(boots[:, position[top]] - boots[:, position[name]], 5) > 0:
            tier, top = tier + 1, name
        result[name] = tier
    return pd.Series(result, name="ярус")


@dataclass
class Leaderboard:
    table: pd.DataFrame
    errors: pd.DataFrame
    result: Result
    intervals: pd.DataFrame = field(default_factory=pd.DataFrame)

    def plot(self):
        """Each team's popularity minus the shortest path, with its 90% interval."""
        import matplotlib.pyplot as plt
        data = self.intervals.sort_values("разница")
        fig, ax = plt.subplots(figsize=(7, 0.35 * len(data) + 1.2), layout="constrained")
        y = np.arange(len(data))
        ax.errorbar(data["разница"], y, xerr=[data["разница"] - data["от"], data["до"] - data["разница"]],
                    fmt="o", color="#2563eb", ecolor="#93c5fd")
        ax.axvline(0, color="#6b7280", linestyle=":")
        ax.set_yticks(y, data.index)
        ax.set_xlabel("популярность минус кратчайший, закрытая зона, 90% интервал")
        ax.spines[["top", "right"]].set_visible(False)
        return fig


def grade_all(city: CityData, private_folder, submissions: list[Submission], *, with_open=True, seed=0) -> Leaderboard:
    """Score every loaded submission on the closed zone (and, for reference, on the open one)."""
    closed = closed_city(city, private_folder)
    per_od, routes, rows, solutions, errors, base, open_shortest = [], [], {}, {}, {}, None, np.nan
    for s in tqdm(submissions, desc="Команды"):
        if s.solution is None:
            errors[s.team] = s.status
            continue
        try:
            chosen = {**s.solution, "label": s.team}
            result = score(closed, {s.team: chosen}, network=build_network(closed, s.keep), progress=False)
            base = base or result
            per_od.append(result.per_od[result.per_od.algorithm.eq(s.team)])
            routes.append(result.routes[result.routes.algorithm.eq(s.team)])
            rows[s.team] = result.table.loc[s.team]
            solutions[s.team] = chosen
            if with_open:
                seen = score(city, {s.team: chosen}, network=build_network(city, s.keep), progress=False).table
                rows[s.team]["открытая зона"], open_shortest = seen.loc[s.team, "популярность"], seen.loc["кратчайший", "популярность"]
        except Exception as error:
            errors[s.team] = f"ошибка при оценке: {error}"
    errors = pd.DataFrame({"статус": errors}).rename_axis("команда")
    if base is None:
        return Leaderboard(pd.DataFrame(), errors, None)
    per_od = pd.concat([base.per_od[base.per_od.algorithm.eq("shortest")], *per_od], ignore_index=True)
    routes = gpd.GeoDataFrame(pd.concat([base.routes[base.routes.algorithm.eq("shortest")], *routes], ignore_index=True),
                              geometry="geometry", crs=city.osm.crs)
    rows["кратчайший"] = base.table.loc["кратчайший"].copy()
    rows["кратчайший"]["открытая зона"] = open_shortest
    names = ["shortest", *solutions]
    boots = bootstrap(per_od, names, np.random.default_rng(seed))
    labels = ["кратчайший", *solutions]
    table = pd.DataFrame(rows).T.loc[labels, [c for c in COLUMNS if c != "открытая зона" or with_open]]
    for column in table.columns.drop("популярнее кратчайшего"):
        table[column] = pd.to_numeric(table[column])
    table.insert(0, "ярус", tiers(pd.Series(per_od.groupby("algorithm").counted.mean()[names].to_numpy(), index=labels), boots))
    spread = pd.Series((np.percentile(boots, 95, axis=0) - np.percentile(boots, 5, axis=0)) / 2, index=labels)
    table.insert(2, "±", spread.round(1))
    if with_open:
        table.insert(3, "популярность, открытая зона", table.pop("открытая зона").round(1))
    difference = boots - boots[:, [0]]
    intervals = pd.DataFrame({"разница": difference.mean(axis=0), "от": np.percentile(difference, 5, axis=0),
                              "до": np.percentile(difference, 95, axis=0)}, index=labels).drop(index="кратчайший")
    table = (table.sort_values(["ярус", "популярность"], ascending=[True, False])
             .rename(columns={"популярность": "популярность, закрытая зона"}))
    return Leaderboard(table, errors, Result(table, per_od, routes, solutions, closed, None), intervals)
