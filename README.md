# CicleGpx

Учебный проект по маршрутизации велосипеда. Данные представлены плоской таблицей рёбер, а части,
которые участники хакатона могут менять, оформлены обычными функциями.

```text
prepare_city() -> edges, nodes -> pandas-фильтр -> check_connectivity()
                                                -> build_graph(edges, nodes, cost)
                                                -> find_route() / build_routes()
                                                -> score()
```

Популярность улиц не входит в `edges` и не попадает в маршрутный граф. Она хранится отдельно в
`popularity.parquet` и доступна только оценке и визуализации.

## Быстрый старт

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\jupyter-lab.exe
```

Откройте `notebooks/comfort_routing.ipynb`. При первом запуске `prepare_city` скачает OSM и
наблюдения; следующие запуски читают GeoParquet-кеш из `notebooks/data/cities/`.

## Публичный пайплайн

```python
from pathlib import Path

from cycle_routing import (
    COMFORT_PRESETS,
    build_graph,
    build_routes,
    filter_edges,
    is_rideable,
    load_od_pairs,
    prepare_city,
    score,
)

cache = Path("notebooks/data/cities")
edges, nodes = prepare_city("konstanz", cache_dir=cache)
edges = filter_edges(edges, is_rideable, keep_largest=True)

preset = COMFORT_PRESETS["balanced"]
graph = build_graph(edges, nodes, cost=preset["cost"])
pairs = load_od_pairs("konstanz", cache_dir=cache)
routes = build_routes(edges, nodes, pairs, COMFORT_PRESETS, cache_dir=Path("outputs/routes_cache"))
metrics = score(routes, "konstanz", cache_dir=cache)
```

На рёбрах готового графа остаются только `cost`, `length_m` и `geometry`; OSM-теги и наблюдения
остаются в таблицах.

## Точки расширения

Каждое поле решения задаётся данными (без кода) или функцией (кодом) — интерфейс один, формы смешиваются.

| Поле | Без кода | Кодом |
|---|---|---|
| фильтр `filter_edges(edges, keep)` | `drop_values(edges, колонка=[значения])` — маска без рёбер с этими значениями; маски складываются через `&` | любая pandas-маска или `keep(edges) -> Series[bool]` |
| стоимость `cost` | `{"highway": {"primary": 3.0}}` — изменения весов `DEFAULT_COMFORT_CONFIG`, или имя пресета `"quiet"` | `cost(edges) -> Series[float]` в эквивалентных метрах |
| повороты `turns` | `"none"`, `"default"`, `"min_turns"` (`TURNS`) или `{"left_penalty_m": 60}` | `penalty(graph, incoming, outgoing) -> float` |
| поиск `search` | `"dijkstra"`, `"bellman_ford"`, `"astar"` (`SEARCH`) | функция networkx как есть или своя `search(graph, source, target, weight="cost") -> [узлы]` |

Профиль — словарь `{"label", "description", "cost", "turns", "search"}`; профиль без кода — обычный JSON.
Восемь готовых профилей `COMFORT_PRESETS` устроены так же. С поворотами поиск идёт по `turn_graph`: его
узлы — рёбра улиц, а поворот — обычное ребро со штрафом.

В таблице `score` метрика `сумма поездок по рёбрам` складывает наблюдаемое число поездок по всем
пройденным рёбрам и повторно учитывает ребро, если оно используется в нескольких OD-маршрутах.

## Структура проекта

- `preparation.py` — города, загрузка OSM/наблюдений, плоские таблицы и кеш;
- `graph.py` — фильтрация, связность и сборка минимального маршрутного графа;
- `comfort.py` — векторная модель комфорта и объяснение её множителей;
- `presets.py` — восемь готовых функций стоимости;
- `routing.py` — поиск, повороты, пакетный прогон и кеш маршрутов;
- `evaluation.py` — чёрный ящик `score`;
- `visualization.py` — карты, принимающие таблицу рёбер;
- `TASK.md` — задание участникам.

Старые графовые функции оставлены как адаптеры для прежних примеров и регрессионных тестов, но новый
ноутбук использует только табличный API.

## Проверка

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\jupyter.exe nbconvert --to notebook --execute notebooks/comfort_routing.ipynb `
  --output comfort_routing.executed.ipynb
```

Эталонные таблицы находятся в `outputs/konstanz_presets.csv` и `outputs/spb_presets.csv`.
