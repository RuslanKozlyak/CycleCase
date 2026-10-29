"""Build the reviewers' notebook: drop team submissions into a folder, get the closed-zone leaderboard.

    python tools/build_grading_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/hackathon_grading.ipynb

Organizer only: it reads the closed zone and is never packed into the student ZIP.
"""
from pathlib import Path
from textwrap import dedent

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(dedent(text).strip()))


def code(text):
    cells.append(nbf.v4.new_code_cell(dedent(text).strip()))


md("""
    # Проверка решений команд

    Положите присланные папки или ZIP-архивы (из `export_solution`) в папку `SUBMISSIONS`, по одной на команду,
    и выполните все ячейки. Оценка — на закрытой зоне по тем же правилам, что в ноутбуке команд.
    Загрузка решения выполняет код команды: кладите сюда только решения участников.
""")
code("""
    from pathlib import Path
    import sys

    from IPython.display import display

    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
    sys.path.insert(0, str(ROOT / "src"))
    from cycle_routing.grading import grade_all, load_submissions, submissions_table
    from cycle_routing.workshop import load_city
    from cycle_routing.workshop_maps import route_explorer

    PACK = "konstanz_more"  # тот же набор, что у команд
    SUBMISSIONS = ROOT / "outputs/hackathon/submissions"  # сюда — папки или ZIP команд
    city = load_city(ROOT / "notebooks/data/hackathon" / PACK)
    PRIVATE = ROOT / "outputs/hackathon_private" / PACK
    OUTPUT = ROOT / "outputs/hackathon/grading" / PACK
    OUTPUT.mkdir(parents=True, exist_ok=True)
""")

md("## 1. Решения")
code("""
    teams = load_submissions(SUBMISSIONS, city)
    submissions_table(teams)
""")

md("""
    ## 2. Итог на закрытой зоне

    **ярус** — команды одного яруса не различаются: разница в пределах шума закрытых пар.
    **популярность, закрытая зона** — итоговая оценка, **±** — её 90% интервал.
    **популярность, открытая зона** — для сравнения: так решение выглядело у команды.
""")
code("""
    board = grade_all(city, PRIVATE, teams)
    board.table.to_csv(OUTPUT / "leaderboard.csv")
    display(board.table)
    board.plot();
""")
code('board.errors if len(board.errors) else "Все решения оценены."')

md("## 3. Маршруты на закрытой зоне")
code('route_explorer(board.result.city, board.result, filename=OUTPUT / "closed_routes.html")')

md("## 4. Баллы по парам")
code("""
    by_pair = board.result.per_od.pivot(index="od_id", columns="algorithm", values="counted").rename(columns={"shortest": "кратчайший"})
    by_pair.to_csv(OUTPUT / "by_pair.csv")
    by_pair.round(0)
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
target = ROOT / "notebooks/hackathon_grading.ipynb"
nbf.write(notebook, target)
print(f"Created {target.name}: {len(cells)} cells, {sum(c.cell_type == 'code' for c in cells)} code cells")
