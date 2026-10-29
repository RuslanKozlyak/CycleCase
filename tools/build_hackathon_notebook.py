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

    Постройте маршруты A–B по улицам, где действительно ездят велосипедисты. Сравните популярность и длину маршрутов.
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
        build_network, comfort_cost, data_dictionary, example_rows, export_solution, gpx_table,
        load_city, ml_feature, presets, preview_cost, preview_features, score, solution, train_custom_model,
        train_popularity,
    )
    from cycle_routing.tags import has_any
    from cycle_routing.routing import edge_bearings
    from cycle_routing.workshop_maps import plot_network, plot_pairs, plot_popularity, plot_turn_examples, route_explorer

    PACK = "konstanz_more"  # набор данных: другое разбиение города — другое имя
    city = load_city(ROOT / "notebooks/data/hackathon" / PACK)
    OUTPUT = ROOT / "outputs/hackathon" / PACK
""")

md("""
    ## 1. Данные

    `city.roads` — участки дорог, `city.trips` — поездки по участку (известны только в открытой части города),
    `city.pairs` — тестовые пары. `data_dictionary` описывает исходные столбцы `city.roads`.
    Имена столбцов и значения категорий — по-русски с подчёркиваниями: например, `пешеходная_дорожка`, `асфальт`.
    Эти же значения используйте в фильтрах и признаках. Ниже — словарь и примеры заполненных строк.
""")
code("""
    display(data_dictionary(city))
    display(example_rows(city))
""")
code("plot_popularity(city);")

md("""
    ## 2. Фильтр ✏️

    Какие дороги можно использовать. Участки, оторванные от основной сети, в маршрутах не используются.
    Ниже весь код `rideable`: проверяем тип дороги, доступ и разрешение велосипеда.
    Явное разрешение велосипеда отменяет ограничение доступа, но не запрет типа дороги.
    Пешеходные дорожки сохраняются, если нет явного запрета велосипеда.
""")
code("""
    def rideable(roads):
        # Типы дорог, исключаемые из сети независимо от остальных тегов.
        forbidden_road = has_any(roads["тип_дороги"], {
            "лестница", "автомагистраль", "съезд_с_автомагистрали", "скоростная_дорога",
            "съезд_со_скоростной_дороги", "строящаяся_дорога", "предлагаемая_дорога",
            "планируемая_дорога", "снесённая_дорога", "заброшенная_дорога", "платформа",
            "лифт", "эскалатор", "коридор", "направляемая_автобусная_дорога",
            "гоночная_трасса", "автобусная_дорога",
        })

        # Закрытый доступ или частный служебный проезд.
        closed_access = has_any(roads["доступ"], {
            "частный", "запрещён", "только_для_клиентов", "по_пропуску", "только_для_доставки",
        }) | has_any(roads["служебный_проезд"], "частный_проезд")

        # Велосипед может быть разрешён даже при ограничении общего доступа.
        bicycle_allowed = has_any(roads["велосипед"], {
            "разрешён", "предназначено_для_велосипедов", "с_разрешения_владельца",
        })
        bicycle_forbidden = has_any(roads["велосипед"], {"запрещён", "спешиться", "частный_доступ"})

        return ~(forbidden_road | bicycle_forbidden | (closed_access & ~bicycle_allowed))
""")
code("""
    def keep(roads):
        ok = rideable(roads)  # готовый фильтр: без лестниц, запретов для велосипеда и частных проездов
        # ok &= drop_values(roads, тип_дороги=["главная_дорога"])  # пример: ещё убрать крупные дороги
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
    `features(roads)` превращает исходные характеристики дорог в числовые признаки.
    В примере есть `длина_м` и три признака: велодорожка, крупная дорога, асфальт.
    `длина_м` — базовый множитель стоимости; остальные признаки получают веса из `WEIGHTS`.
    Меняйте и добавляйте столбцы здесь.
    `has_any` проверяет значение среди разделённых `;` тегов участка.
    Одна и та же функция передаётся в стоимость и свою ML-модель.
""")
code('''
    def features(roads):
        f = pd.DataFrame(index=roads.index)
        f["длина_м"] = roads["длина_м"]  # Длина участка в метрах.
        f["велодорожка"] = has_any(roads["тип_дороги"], "велодорожка").astype(float)
        f["крупная_дорога"] = has_any(
            roads["тип_дороги"], {"главная_дорога", "дорога_второго_значения", "скоростная_дорога"}
        ).astype(float)
        f["асфальт"] = has_any(roads["покрытие"], "асфальт").astype(float)
        return f

    preview_features(city, features)
''')
md("""
    ### Веса ✏️

    Ключи весов — признаки из `features`; `длина_м` уже используется как множитель, отдельный вес ей не нужен.
    Быстрая проверка без поиска маршрутов: признаки с весом
    на примерах дорог и стоимость участка в последнем столбце.
""")
code("""
    WEIGHTS = {
        "крупная_дорога": 0.6,
        "велодорожка": -0.4,
        "асфальт": -0.2,
    }
    my_cost = comfort_cost(WEIGHTS, features=features)
    preview_cost(city, my_cost)
""")
md("""
    ### Своя формула ✏️

    Необязательно: формула целиком — функция `roads → положительная стоимость каждого участка`.
    Здесь она использует те же свои признаки; в сравнении ниже это решение «моя формула».
""")
code("""
    def my_formula(roads):
        f = features(roads)
        return f["длина_м"] * (1 + 0.8 * f["крупная_дорога"] - 0.3 * f["асфальт"])

    preview_cost(city, my_formula, features=features)
""")
md("""
    ### Повороты ✏️

    `my_turn_penalty` получает два направленных участка `(откуда, куда, ключ)` и возвращает штраф в метрах.
    `edge_bearings` берёт направления по геометрии: конец входящего участка и начало следующего.
    Направление задаётся на круге: 0° / 360° — вверх, 90° — вправо, 180° — вниз, 270° — влево.
    Входящее направление — куда мы двигались до точки поворота.
    Разность направлений от −180° до 180°: плюс — направо, минус — налево.
    Формула в коде учитывает переход через 360°: из 350° в 10° — это +20°, а не −340°.
    `%` — остаток от деления; прибавление и вычитание 180° дают угол поворота со знаком.
    Стоимость перехода = стоимость следующего участка + штраф. Начало маршрута — без штрафа.
    Под кодом — четыре реальных ситуации из выбранной сети: синим показан въезд, зелёным — выезд.
    Маленький круг помогает сопоставить направления на карте с углом 0–360°.
    В таблице оценки считаются левые, правые повороты и развороты; плавные изгибы до 55° не считаются.
""")
code("""
    def my_turn_penalty(graph, incoming, outgoing):
        incoming_direction = edge_bearings(graph, incoming)[1]
        outgoing_direction = edge_bearings(graph, outgoing)[0]
        angle = (outgoing_direction - incoming_direction + 180) % 360 - 180
        if abs(angle) <= 25:
            return 0.0  # прямо
        if abs(angle) <= 55:
            return 8.0  # плавный изгиб
        if abs(angle) >= 150:
            return 90.0  # разворот
        return 10.0 if angle > 0 else 30.0  # направо / налево
""")
code("plot_turn_examples(network, my_turn_penalty);")
md("""
    ### Поиск ✏️

    Обычный Дейкстра из NetworkX ищет минимальную сумму стоимостей.
    Жадный поиск ниже написан с нуля: храним пути в списке и выбираем тот, чей конец ближе к цели по прямой.
    Длина ребра описывает один участок дороги, а `distance` оценивает оставшееся расстояние до цели.
    Квадрат расстояния даёт тот же порядок близости, поэтому извлекать корень не нужно.
    Поиск не учитывает стоимости рёбер
    и не гарантирует лучший маршрут. Оба алгоритма попадают в сравнение.
    С поворотами вершины графа — участки дорог, их координаты — конец участка.
""")
code("""
    def my_search(graph, source, target, weight="cost"):
        return nx.dijkstra_path(graph, source, target, weight=weight)

    def greedy_search(graph, source, target, weight="cost"):
        if source not in graph or target not in graph:
            raise nx.NodeNotFound("Начало или конец отсутствует в графе")

        # Квадрат расстояния от вершины до цели по прямой, а не длина соседнего ребра.
        def distance(node):
            point, goal = graph.nodes[node], graph.nodes[target]
            return (point["x"] - goal["x"])**2 + (point["y"] - goal["y"])**2

        paths = [[source]]  # Начинаем с пути из одной вершины.
        visited = {source}  # Каждую вершину добавляем только один раз, чтобы избежать циклов.

        while paths:
            # Выбираем путь, последняя вершина которого ближе всего к цели.
            path = paths[0]
            for candidate in paths:
                if distance(candidate[-1]) < distance(path[-1]):
                    path = candidate
            paths.remove(path)
            current = path[-1]

            if current == target:
                return path  # Путь уже хранится целиком.

            # Продлеваем выбранный путь до каждого ещё не посещённого соседа.
            for neighbor in graph.successors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    paths.append(path + [neighbor])

        # Все доступные вершины проверены, но цель не достигнута.
        raise nx.NetworkXNoPath("Путь не найден")

    mine = solution(my_cost, turns=my_turn_penalty, search=my_search, label="моё решение")
    formula = solution(my_formula, turns=my_turn_penalty, search=my_search, label="моя формула")
    greedy = solution(my_cost, turns=my_turn_penalty, search=greedy_search, label="жадный поиск")
""")

md("""
    ## 5. ML ✏️

    Необязательно: модель предсказывает популярность улицы, а прогноз становится признаком `ml_популярность` (0–1).
    Готовая модель — градиентный бустинг по тегам OSM.
""")
code("""
    model = train_popularity(city)

    def features_with_ml(roads):
        f = features(roads)
        f["ml_популярность"] = ml_feature(model)(roads)["ml_популярность"]
        return f

    ml_cost = comfort_cost(
        {**WEIGHTS, "ml_популярность": -0.8}, features=features_with_ml
    )
    with_ml = solution(ml_cost, turns=my_turn_penalty, search=my_search, label="моё + ML")
""")
md("""
    ### Своя модель ✏️

    `train_model(x, y)` получает таблицу признаков и цель `log(1 + поездки)`, вызывает `fit`
    и возвращает обученную модель. Пример специально слабый — дерево глубины 1; попробуйте лучше.
    `train_custom_model` вычисляет `features`, исключает неизвестные поездки,
    проверяет модель на отложенных кварталах и затем обучает новую модель на всех открытых метках.
""")
code("""
    from sklearn.tree import DecisionTreeRegressor

    def train_model(x, y):
        model = DecisionTreeRegressor(max_depth=1, random_state=42)
        model.fit(x, y)
        return model

    my_model = train_custom_model(city, train_model, features=features)

    def features_with_my_model(roads):
        f = features(roads)
        f["ml_популярность"] = ml_feature(my_model)(roads)["ml_популярность"]
        return f

    my_ml_cost = comfort_cost(
        {**WEIGHTS, "ml_популярность": -0.8}, features=features_with_my_model
    )
    with_my_ml = solution(my_ml_cost, turns=my_turn_penalty, search=my_search, label="моё + своя ML")
""")

md("""
    ## 6. Сравнение

    Сравниваем **популярность** (больше лучше) и **среднюю длину маршрута, км** (меньше лучше).
    Лимита объезда нет: учитывается популярность самого маршрута при любой длине.
    Улица без измерения пока считается средней (50).
    **Парето-фронт** — решения, для которых нет другого одновременно не длиннее и не менее популярного,
    причём хотя бы по одному показателю строго лучше. Равные точки остаются на фронте вместе.
    Средние считаются по одним и тем же OD. Решения с пропущенными маршрутами видны в таблице,
    но не участвуют во фронте: удаление трудных пар фильтром не делает решение короче.
""")
code("""
    solutions = {
        item["label"]: item
        for item in [mine, formula, greedy, with_ml, with_my_ml, *presets("balanced", "fast", "min_turns").values()]
    }
    result = score(city, solutions, network=network)
    display(result.table)
    result.plot_pareto();
""")

md("""
    ## 7. Разбор маршрутов

    Выберите пару и решение; щелчок по участку покажет его теги.
""")
code('route_explorer(city, result, filename=OUTPUT / "routes.html")')

md("""
    ## 8. Петербург

    Решение переносится без изменений; красные линии — реальные поездки (GPX). Совпадение — доля длины маршрута
    в пределах 30 м от ближайшей поездки. Подготовленный набор Петербурга должен быть на диске.
""")
code("""
    SPB = ROOT / "notebooks/data/hackathon/saint_petersburg"
    spb = load_city(SPB)
    spb_solutions = {item["label"]: item for item in [mine, with_ml]}
    spb_result = score(spb, spb_solutions, network=build_network(spb, keep))
    display(gpx_table(spb_result, spb.tracks))
    display(route_explorer(spb, spb_result, tracks=spb.tracks, filename=OUTPUT / "spb_routes.html"))
""")
for cell in cells[-2:]:
    cell.metadata["tags"] = ["local-spb"]

md("## 9. Сдача ✏️")
code("""
    CHOSEN = mine  # или formula, greedy, with_ml, with_my_ml
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
