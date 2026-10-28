"""Build the student notebook from code. Execute it separately before distribution:

    python tools/build_hackathon_notebook.py
    jupyter nbconvert --to notebook --execute --inplace notebooks/hackathon_konstanz.ipynb
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
    # Веломаршрут по популярным улицам

    Постройте маршруты A–B по улицам, где действительно ездят велосипедисты, не сильно длиннее кратчайшего пути.
    Популярность улиц — поездки STADTRADELN 2024 в Констанце. Итог считается на закрытой части города.
    Меняйте ячейки с ✏️.
""")
code("""
    from pathlib import Path
    import sys

    import networkx as nx
    import pandas as pd
    from IPython.display import display

    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
    sys.path.insert(0, str(ROOT / "src"))
    from cycle_routing.graph import drop_values
    from cycle_routing.workshop import (
        build_network, comfort_cost, data_dictionary, example_rows, export_solution, feature_catalog, gpx_table,
        load_city, ml_feature, presets, preview_cost, preview_features, rideable, score, solution, train_model,
        train_popularity,
    )
    from cycle_routing.workshop_maps import plot_network, plot_pairs, plot_popularity, route_explorer

    PACK = "konstanz_more"  # набор данных: другое разбиение города — другое имя
    city = load_city(ROOT / "notebooks/data/hackathon" / PACK)
    OUTPUT = ROOT / "outputs/hackathon" / PACK
""")

md("""
    ## 1. Данные

    `city.roads` — участки дорог, `city.trips` — поездки по участку (известны только в открытой части города),
    `city.pairs` — тестовые пары. Ниже — все значения каждого столбца и несколько строк, где заполнен каждый столбец.
""")
code("""
    display(data_dictionary(city))
    display(example_rows(city))
""")
code("plot_popularity(city);")

md("""
    ## 2. Фильтр ✏️

    Какие дороги можно использовать. Участки, оторванные от основной сети, в маршрутах не используются.
""")
code("""
    def keep(roads):
        ok = rideable(roads)  # готовый фильтр: без лестниц, запретов для велосипеда и частных проездов
        # ok &= drop_values(roads, тип_дороги=["primary"])  # пример: ещё убрать крупные дороги
        return ok

    network = build_network(city, keep)
    print(network.summary())
    plot_network(network);
""")

md("## 3. Тестовые пары")
code("plot_pairs(city);")

md("""
    ## 4. Решение ✏️

    Стоимость участка = длина × (1 + Σ вес × признак). Плюс — избегать, минус — предпочитать.
    Готовые признаки ниже уже посчитаны из тегов для каждого участка. Свои добавляйте в `my_features`:
    их можно взвешивать так же, как готовые.
""")
code("feature_catalog()")
md("""
    ### Свои признаки ✏️

    Пример: два новых признака. Таблица под кодом показывает участки, где они есть.
""")
code("""
    def my_features(roads):
        f = pd.DataFrame(index=roads.index)
        f["быстрые_машины"] = roads["макс_скорость"].ge(50).astype(float)  # 1, если машины едут 50 км/ч и быстрее
        f["асфальт"] = roads["покрытие"].eq("asphalt").astype(float)
        return f

    preview_features(city, my_features)
""")
md("""
    ### Веса ✏️

    Свои признаки стоят в весах рядом с готовыми. Быстрая проверка без поиска маршрутов: признаки с весом
    на примерах дорог и стоимость участка в последнем столбце.
""")
code("""
    WEIGHTS = {"крупная_дорога": 0.6, "плохое_покрытие": 0.4, "велодорожка": -0.4, "без_освещения": 0.2,
               "быстрые_машины": 0.3, "асфальт": -0.2}
    my_cost = comfort_cost(WEIGHTS, extra=my_features)
    preview_cost(city, my_cost)
""")
md("""
    ### Своя формула ✏️

    Необязательно: формула целиком — функция `roads → положительная стоимость каждого участка`.
    Здесь она использует те же свои признаки; в сравнении ниже это решение «моя формула».
""")
code("""
    def my_formula(roads):
        f = my_features(roads)
        return roads["длина_м"] * (1 + 0.8 * f["быстрые_машины"] - 0.3 * f["асфальт"])

    preview_cost(city, my_formula, extra=my_features)
""")
md("### Повороты и поиск ✏️")
code("""
    TURNS = {"left_penalty_m": 30, "right_penalty_m": 10}  # None — без штрафов; "min_turns" — готовый режим

    def my_search(graph, source, target, weight):
        # двусторонний Дейкстра: тот же путь, что у обычного, но быстрее
        return nx.bidirectional_dijkstra(graph, source, target, weight=weight)[1]

    mine = solution(my_cost, turns=TURNS, search=my_search, label="моё решение")
    formula = solution(my_formula, turns=TURNS, search=my_search, label="моя формула")
""")

md("""
    ## 5. ML ✏️

    Необязательно: модель предсказывает популярность улицы, а прогноз становится признаком `ml_популярность` (0–1).
    Готовая модель — градиентный бустинг по тегам OSM.
""")
code("""
    model = train_popularity(city)
    ml_cost = comfort_cost({**WEIGHTS, "ml_популярность": -0.8}, extra=[my_features, ml_feature(model)])
    with_ml = solution(ml_cost, turns=TURNS, search=my_search, label="моё + ML")
""")
md("""
    ### Своя модель ✏️

    Любая модель sklearn с `fit` и `predict`. Она учится на готовых признаках и ваших `my_features`
    предсказывать `log(1 + поездки)`. Пример специально слабый — дерево глубины 1; попробуйте лучше.
""")
code("""
    from sklearn.tree import DecisionTreeRegressor

    my_model = train_model(city, DecisionTreeRegressor(max_depth=1), extra=my_features)
    my_ml_cost = comfort_cost({**WEIGHTS, "ml_популярность": -0.8}, extra=[my_features, ml_feature(my_model)])
    with_my_ml = solution(my_ml_cost, turns=TURNS, search=my_search, label="моё + своя ML")
""")

md("""
    ## 6. Сравнение

    **популярность** — главный критерий, 0–100. Улица без измерения считается средней (50).
    Если маршрут длиннее кратчайшего больше чем на 20%, в оценку этой пары идёт популярность кратчайшего:
    большой объезд не даёт выигрыша.
""")
code("""
    solutions = {"моё": mine, "моя формула": formula, "моё + ML": with_ml, "моё + своя ML": with_my_ml,
                 **presets("balanced", "fast", "min_turns")}
    result = score(city, solutions, network=network)
    display(result.table)
    result.plot();
""")

md("""
    ## 7. Разбор маршрутов

    Выберите пару и решение; щелчок по участку покажет его теги.
""")
code('route_explorer(city, result, filename=OUTPUT / "routes.html")')

md("""
    ## 8. Петербург

    Решение переносится без изменений; красные линии — реальные поездки (GPX). Совпадение — доля длины маршрута
    в пределах 30 м от ближайшей поездки. Блок работает, если есть набор Петербурга.
""")
code("""
    SPB = ROOT / "notebooks/data/hackathon/saint_petersburg"
    if (SPB / "manifest.json").exists():
        spb = load_city(SPB)
        spb_result = score(spb, {"моё": mine, "моё + ML": with_ml}, network=build_network(spb, keep))
        display(gpx_table(spb_result, spb.tracks))
        display(route_explorer(spb, spb_result, tracks=spb.tracks, filename=OUTPUT / "spb_routes.html"))
    else:
        print("Набора Петербурга нет — блок пропущен.")
""")

md("## 9. Сдача ✏️")
code("""
    CHOSEN = mine  # или formula, with_ml, with_my_ml
    folder = export_solution(city, CHOSEN, keep=keep, folder=OUTPUT / "submission")
    print("Сдайте папку", folder, "и этот ноутбук")
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
target = ROOT / "notebooks/hackathon_konstanz.ipynb"
nbf.write(notebook, target)
print(f"Created {target.name}: {len(cells)} cells, {sum(c.cell_type == 'code' for c in cells)} code cells")
