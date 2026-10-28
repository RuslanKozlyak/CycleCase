"""Build the short, editable student notebook. Execute it separately before distribution."""
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
    # Веломаршрут: лаборатория хакатона

    **Задача:** по дорожной сети Констанца построить маршруты A–B, проходящие по более популярным
    у велосипедистов улицам, при ограниченном объезде относительно кратчайшего пути.
    Соберите собственный пресет, объясните его поведение и сравните с готовыми решениями.

    В Констанце используются **агрегированные потоки STADTRADELN**, сопоставленные с OSM.
    Индивидуальных GPX-треков и реальных поездок между нашими A–B в этом наборе нет.
    Поэтому мы измеряем популярность улиц маршрута, а не точность восстановления чьей-то поездки.
    Популярность также не является прямым измерением безопасности или комфорта.

    Порядок работы: **данные → стоимость дороги → поиск → оценка → ML → сдача**.
    Меняйте ячейки с пометкой **«Ваша настройка»**. Загрузка и сопоставление данных уже выполнены организатором.
""")
code("""
    from pathlib import Path
    import sys
    import numpy as np
    import pandas as pd
    from IPython.display import display

    ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
    sys.path.insert(0, str(ROOT / "src"))
    from cycle_routing.workshop import (
        load_lab, features, feature_catalog, preset_catalog, preset, presets, priority_cost,
        EvaluationRules, compare, route_diagnostics, learned_cost, prediction_quality, export_solution,
    )
    from cycle_routing.workshop_maps import plot_od_bank, route_explorer
    from cycle_routing.learning import fit_popularity_model

    OUTPUT = ROOT / "outputs/hackathon_demo"
    OUTPUT.mkdir(parents=True, exist_ok=True)
""")
md("""
    ## 1. Готовые данные

    Выдан один компактный сильно связный район в зоне наблюдений: из любой вершины можно доехать в любую другую.
    Остальные компоненты исходного региона не участвуют. Дороги без измерений сохранены как возможные соединения.

    `edges` — направленные участки дорог: ключ `(u, v, key)`, длина в метрах, геометрия и OSM-теги.
    `nodes` — узлы с координатами. `pairs` — общий банк из 20 OD.
    `labels` — отдельно хранящиеся количества поездок на известных участках.
    **NaN = нет измерения; 0 = явно известный ноль.** Нельзя заменять NaN нулём или удалять такие дороги из графа.
""")
code("""
    lab = load_lab(ROOT / "notebooks/data/hackathon/konstanz")
    display(lab.summary().to_frame())
    display(lab.edges.drop(columns="geometry").head(3))
""")
md("""
    OD выбраны заранее, без оптимизации под какой-либо пресет: распределённые по географии перекрёстки,
    разные расстояния и направления, штраф за повторяющиеся коридоры. У всех решений один банк A–B.
    Это диагностические сценарии, а не частоты реальных поездок. Сильная связность устраняет разрывы,
    но не гарантирует, что у каждой OD есть альтернативный путь или достаточные измерения.
""")
code("""
    display(lab.pairs[["od_id", "length_km", "length_band", "orientation", "junctions_per_km"]].round(2))
    plot_od_bank(lab);
""")
md("""
    ## 2. Из чего собирается решение

    Поиск минимизирует сумму **стоимостей рёбер + штрафов за повороты**.
    Комфорт задаётся положительной стоимостью: чем привлекательнее участок, тем дешевле его метр.
    Кратчайший путь `shortest` использует только длину и служит общей базой.

    Пресет содержит `cost`, `turns`, `search`. В `cost` можно передать готовую настройку или свою функцию.
    `turns=None` отключает штрафы; словарь задаёт их в эквивалентных метрах стоимости.
    `search` принимает `dijkstra`, `astar`, `bellman_ford` или совместимую функцию поиска.
    При одинаковой стоимости точные алгоритмы обычно дают одинаковый оптимальный путь;
    простая замена названия поиска сама по себе не добавляет комфорта. OSM-ограничения поворотов здесь не моделируются.
""")
code("""
    display(preset_catalog())
    display(feature_catalog())
""")
md(r"""
    ### Ваша настройка A: приоритеты без сложного кода

    `traffic`, `rough`, `unlit`, `narrow` увеличивают цену нежелательных дорог;
    `cycling` уменьшает цену велоинфраструктуры. Все коэффициенты неотрицательны.

    $$cost = length \cdot \frac{1 + traffic\cdot major + rough\cdot roughness + unlit\cdot dark + narrow\cdot narrowness}
    {1 + cycling\cdot bike}.$$

    Пропущенный OSM-тег не доказывает отсутствие свойства. Значения по умолчанию и флаги пропусков
    перечислены выше; их можно поменять в своей функции.
""")
code("""
    # ВАША НАСТРОЙКА: числа, штрафы за повороты и алгоритм поиска.
    my_solution = preset(
        "Мой пресет",
        cost=priority_cost(traffic=0.6, rough=0.4, cycling=0.7, unlit=0.2, narrow=0.2),
        turns={"left_penalty_m": 30, "right_penalty_m": 10},
        search="dijkstra",
    )
""")
md("""
    ### Ваша настройка B: собственная формула

    Функция получает всю таблицу дорог и возвращает `pd.Series` с тем же индексом.
    `features(roads)` предоставляет готовые признаки и исходные теги; здесь нет количеств поездок.
    Возвращайте конечную **положительную** стоимость, например длину × цену метра.
    Для тегов с несколькими значениями используйте `cycle_routing.tags.has_any`.
""")
code("""
    def my_formula(roads):
        f = features(roads)
        fast_traffic = (f.speed_kmh > 40) & ~f.speed_kmh_missing
        price_per_m = (1 + 0.5*f.major_road + 0.8*f.rough + 0.3*fast_traffic) / (1 + 0.8*f.bike_infra)
        return f.length_m * price_per_m

    formula_solution = preset("Своя формула", cost=my_formula, turns="default")
""")
md(r"""
    ## 3. Как оценивать пресет

    Для измеренных участков маршрута считаем среднее $L=\sum l_e\log(1+trips_e)/\sum l_e$.
    Длины служат весами: разбиение улицы на мелкие рёбра не создаёт бонуса.
    Для той же OD сравниваем $\Delta=L_{solution}-L_{shortest}$;
    $\exp(\Delta)$ показывает отношение геометрических средних **(1 + число поездок)**.
    Покрытие — доля всей длины маршрута с измерением из выбранной части данных.

    **Предлагаемое правило хакатона:** маршрут даёт $100/(1+\exp(-\Delta))$ баллов;
    кратчайший всегда даёт 50. За отсутствие пути, объезд более 20% или покрытие менее 10% — **0 баллов**.
    Это ноль за невыполнение условия, а не нулевое количество поездок на неизвестных дорогах.
    Итог — среднее по фиксированным OD, где у кратчайшего достаточно измерений.
    Неудачные решения не могут уменьшить этот знаменатель. Для остальных OD показываем диагностические метрики.

    `train` используется для обучения; `validation` — для выбора коэффициентов;
    закрытый `test` — только у организатора. Разделение пространственными блоками около 1 км,
    обратные направления одной улицы в одной части. На границах остаётся пространственная зависимость.
    Это проверка переноса на другие улицы, а не независимые новые GPX или новая выборка OD.
""")
code("""
    # Правила общие для всех команд. Для официального сравнения их фиксирует организатор.
    RULES = EvaluationRules(max_detour_pct=20, min_coverage_pct=10)
    solutions = {**presets(), "my": my_solution, "formula": formula_solution}
    result = compare(lab, solutions, rules=RULES)
    display(result.table.round(2))
""")
md("""
    ## 4. Видно ли влияние настроек?

    Посмотрите, сколько OD изменилось относительно кратчайшего, какие дороги стали выбираться,
    выросла ли популярность и сколько маршрутов прошло ограничение объезда.
    «OD изменено» — точное отличие последовательности рёбер; доли дорожных свойств усреднены по OD.
    На участке без альтернатив даже сильный приоритет может не изменить путь.
""")
code("""
    display(route_diagnostics(lab, result).round(2))
    result.plot();
""")
md("""
    ### Одна карта вместо наложения всех маршрутов

    Выберите A–B, решение и способ окраски. Кратчайший показан серым пунктиром,
    отличающиеся участки — оранжевым. Щёлкните по участку: видны теги, цена метра и доступное измерение.
    Режим измерений использует **только публичные train + validation**;
    баллы в карточке вычисляются только по validation. Неизвестные количества показаны серым.
    Карта работает без сети; OSM-подложка подключается по отдельной галочке.
""")
code("""
    route_explorer(lab, result, filename=OUTPUT / "routes.html")
""")
code("""
    # ВАША НАСТРОЙКА: разбираем одну пару подробно.
    OD = lab.pairs.od_id.iloc[0]
    columns = ["algorithm", "detour_pct", "coverage_pct", "gain", "in_evaluation", "points", "reason"]
    display(result.per_od.loc[result.per_od.od_id.eq(OD), columns].round(3))
""")
md("""
    ## 5. Проверяем чувствительность одного приоритета

    Меняем только `cycling`, оставляя остальные цены и повороты одинаковыми.
    Изменение состава пути связываем с изменением популярности; улучшение не обязано быть монотонным.
    Аналогично можно проверить трафик, покрытие, освещение и штрафы за повороты.
""")
code("""
    sweep = {
        f"cycling={value:g}": preset(
            f"cycling={value:g}", cost=priority_cost(traffic=0.6, rough=0.4, cycling=value), turns=None
        )
        for value in (0, 0.7, 2)
    }
    sensitivity = compare(lab, sweep, rules=RULES)
    display(sensitivity.table.join(route_diagnostics(lab, sensitivity)[["OD изменено", "велоинфраструктура, %"]]).round(2))
""")
md("""
    ## 6. Дополнение: ML-популярность

    Пример обучает градиентный бустинг предсказывать `log(1+trips)` только по OSM-тегам.
    Ни количество поездок, ни ID, координаты или геометрия не подаются модели как признаки.
    Поменять модель можно в `learning.py`, сохранив метод `predict_log_trips(roads)`.
    Для собственной модели/признаков сохраните правило: обучаться только на train.
""")
code("""
    train_roads, train_trips = lab.training_data()
    model = fit_popularity_model(train_roads, train_trips, max_iter=120, max_leaf_nodes=15)
    display(prediction_quality(lab, model).round(3))
""")
md(r"""
    Превращаем прогноз в стоимость: $cost = length\cdot\exp(-\alpha\cdot attractiveness)$,
    где привлекательность — прогноз относительно среднего train с ограничением экстремальных значений.
    При `strength=0` и отключённых поворотах получаем кратчайший путь.
    Проверяем несколько сил влияния **по validation**, а затем сравниваем ML с теми же готовыми пресетами.
    Хороший MAE прогноза сам по себе не гарантирует лучший маршрут.
""")
code("""
    # ВАША НАСТРОЙКА: сила ML, модель или сочетание ML с собственной формулой.
    ml_solutions = {
        f"ml_{strength:g}": preset(f"ML, сила {strength:g}", cost=learned_cost(model, strength=strength), turns=None)
        for strength in (0, 0.25, 0.5, 1)
    }
    final_solutions = {**solutions, **ml_solutions}
    final = compare(lab, final_solutions, rules=RULES)
    display(final.table.join(route_diagnostics(lab, final)[["OD изменено"]]).round(2))
""")
code("""
    best_ml = final.table.loc[list(ml_solutions), "балл / 100"].idxmax()
    print("Лучшая проверенная сила ML на validation:", best_ml)
    route_explorer(lab, final, filename=OUTPUT / "routes_with_ml.html")
""")
md("""
    ## 7. Необязательный перенос на Петербург

    Организатор отдельно готовит локальный граф и личные GPX. Для этого блока переключите флаг,
    если набор `saint_petersburg` уже есть рядом с Констанцем.
    Пресет и обученная на Констанце модель переносятся **без дообучения и подбора по Петербургу**.
    Количество посещений GPX здесь получено пространственным сопоставлением с дорогами.
    Это дополнительная проверка на одном личном наборе; она не заменяет основной рейтинг.
    Непосещённые рёбра остаются неизвестными. Холодный сезон отдельно не тестируется.
""")
code("""
    RUN_SPB = False  # ВАША НАСТРОЙКА: True при наличии отдельно выданного набора.
    if RUN_SPB:
        spb = load_lab(ROOT / "notebooks/data/hackathon/saint_petersburg")
        transfer = {"my": my_solution, "formula": formula_solution, best_ml: ml_solutions[best_ml]}
        spb_result = compare(spb, transfer, split="external", rules=RULES)
        display(spb_result.table.round(2))
        display(route_explorer(spb, spb_result, filename=OUTPUT / "spb_routes.html"))
    else:
        print("Петербург отключён. Основной эксперимент — Констанц.")
""")
md("""
    ## 8. Что сдаёт команда

    Выберите одно решение, сохраните ноутбук с формулой/моделью и объясните: что изменили,
    почему ждёте эффекта, на каких OD он появился и какие ограничения обнаружили.
    Экспорт содержит пути на всех 20 OD, публичный отчёт и идентификатор набора.
    Организатор пересчитает длины и баллы по закрытым меткам test с общими правилами.
    Test нельзя использовать для выбора формулы, силы ML или OD.
""")
code("""
    CHOSEN = "my"  # ВАША НАСТРОЙКА: например, "formula" или best_ml.
    submission = export_solution(lab, final, algorithm=CHOSEN, folder=OUTPUT / "submission")
    final.table.to_csv(OUTPUT / "validation_all.csv")
    sensitivity.table.to_csv(OUTPUT / "sensitivity.csv")
    print("Результат для сдачи:", submission)
""")
md("""
    **Идеи продолжения:** другая формула стоимости; подбор коэффициентов только по validation;
    сочетание ML и дорожных приоритетов; собственные признаки OSM; своя модель поворотов;
    поиск нескольких кандидатов с ограничением длины. Каждый вариант проходит один и тот же протокол оценки.

    Подробности запуска и команды организатора — в `HACKATHON.md`.
""")

notebook = nbf.v4.new_notebook(cells=cells)
notebook.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
}
target = ROOT / "notebooks/hackathon_konstanz.ipynb"
nbf.write(notebook, target)
print(f"Created {target.name}: {len(cells)} cells, {sum(c.cell_type == 'code' for c in cells)} code cells")
