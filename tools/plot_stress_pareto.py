"""Re-evaluate saved stress-test routes without a detour cap; plot both Pareto fronts.

    .venv/Scripts/python.exe tools/plot_stress_pareto.py --pack konstanz_more

No notebook, route search, model training, or changes to the original stress report are needed.
Only honest cases with successful independent audits are compared. Incomplete cases remain in
the CSV, but cannot reach the front by dropping OD. Distance is the mean physical length in km.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from cycle_routing.pareto import pareto_table, plot_pareto


def render(source: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    catalogue = json.loads((source / "catalogue.json").read_text(encoding="utf-8"))
    metadata = {case["id"]: {"label": case["label"], "family": case["family"]}
                for case in catalogue if case["kind"] == "honest"}
    metadata["shortest"] = {"label": "кратчайший", "family": "baseline"}
    excluded, tables = [], {}
    for zone, title in [("open", "Открытая зона"), ("closed", "Закрытая зона")]:
        baseline = pd.read_pickle(source / f"baseline_{zone}.pkl")
        pairs = pd.Index(baseline.od_id)
        frames = [baseline]
        for ident in metadata:
            if ident == "shortest":
                continue
            rows = json.loads((source / (ident + ".json")).read_text(encoding="utf-8"))
            record = next(row for row in rows if row["zone"] == zone)
            if record["status"] != "ok" or record.get("flags"):
                excluded.append({"id": ident, "zone": zone, "reason": str(record.get("flags") or record["status"])})
                continue
            audit = pd.read_csv(source / f"{ident}_{zone}_audit.csv").set_index("od_id")
            if audit["flags"].fillna("").ne("").any() or set(audit.index) != set(pairs):
                raise ValueError(f"Invalid fixed-OD audit for {ident}/{zone}")
            data = pd.read_pickle(source / f"{ident}_{zone}_per_od.pkl")
            data = data[data.algorithm.eq(ident)].set_index("od_id").reindex(pairs)
            # Independent raw lengths, not graph attributes a custom search could have overwritten.
            data["length_m"] = audit.true_length_m.reindex(pairs)
            data["algorithm"] = ident
            frames.append(data.reset_index())
        combined = pd.concat(frames, ignore_index=True)
        combined["counted"] = combined.popularity.where(combined.length_m.notna(), 0)
        table = pareto_table(combined)
        table = table.join(pd.DataFrame(metadata).T).sort_values(["Парето", "длина, км"], ascending=[False, True])
        bad = combined.groupby("algorithm").bad_m.sum()
        table["вне rideable, м"] = bad.reindex(table.index)
        table.to_csv(output / f"{zone}.csv")
        table[table["Парето"]].to_csv(output / f"{zone}_front.csv")
        figure = plot_pareto(table, title=title + ": без ограничения объезда")
        figure.savefig(output / f"{zone}.png", dpi=170)
        plt.close(figure)
        tables[zone] = table
        print(f"{zone}: {len(table) - 1} candidates + baseline; {int(table['Парето'].sum())} points on front; "
              f"{int(table['длина, км'].isna().sum())} incomplete", flush=True)
        print(table[table["Парето"]][["label", "популярность", "длина, км"]].round(3).to_string(), flush=True)
    pd.DataFrame(excluded, columns=["id", "zone", "reason"]).to_csv(output / "excluded.csv", index=False)
    lines = ["# Парето-фронт решений без ограничения объезда", "",
             "Переоценены сохранённые маршруты 100 обычных решений и кратчайшего. Поиск и обучение не повторялись. "
             "Старый отчёт стресс-теста остаётся историческим снимком прежнего правила.", "",
             "Максимизируем среднюю популярность по исходным OD и минимизируем среднюю физическую длину маршрута в км. "
             "Участки взвешиваются по длине внутри маршрута, OD имеют одинаковый вес. Лимита 20% и подстановки балла кратчайшего нет. "
             "Неизмеренные участки по-прежнему имеют оценку 50; способ заполнения пропусков здесь не меняется.", "",
             "Неполные решения остаются в CSV с числом найденных пар и пустым сравнимым расстоянием; во фронте они не участвуют. "
             "Проверки концов, цепочки, фильтра, исходной длины и неизменности данных взяты из независимого аудита. "
             "Наличие участков вне rideable показано отдельным столбцом и не вводит нового правила допуска.", "",
             "Доминирование считается до округления: другая точка не длиннее и не менее популярна, хотя бы одно неравенство строгое. "
             "Равные точки остаются вместе. Членство во фронте описывает наблюдаемые средние; оно не является статистическим доказательством различий.", ""]
    for zone, title in [("open", "Открытая зона"), ("closed", "Закрытая зона")]:
        table = tables[zone]
        lines += ["## " + title, "", f"![{title}]({zone}.png)", "", "```",
                  table[table["Парето"]][["label", "family", "популярность", "длина, км", "маршрутов", "всего пар", "вне rideable, м"]].round(3).to_string(), "```", "",
                  f"Полная таблица: {zone}.csv. Только фронт: {zone}_front.csv.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", default="konstanz_more")
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    source = args.source or ROOT / "outputs/stress_hackathon" / args.pack
    render(source, args.output or source / "pareto")


if __name__ == "__main__":
    main()
