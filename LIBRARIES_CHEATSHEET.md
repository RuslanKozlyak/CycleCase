# Шпаргалка: pandas, NetworkX и scikit-learn

К ноутбуку [hackathon_konstanz.ipynb](notebooks/hackathon_konstanz.ipynb).
Примеры используют его переменные. Строки с `f[...]` добавляйте внутрь `features`, перед `return f`.

## pandas

### Таблица, столбец и индекс

`DataFrame` — таблица, `Series` — столбец с индексом. Индекс — метка строки, не обязательно её порядковый номер.

```python
import pandas as pd

roads = city.roads
display(roads.head())                         # первые 5 строк
print(roads.shape)                            # число строк и столбцов
print(roads.columns)                          # названия столбцов
print(roads.dtypes)                           # типы данных

display(roads["длина_м"])                      # Series
display(roads[["длина_м", "тип_дороги"]])       # DataFrame
display(roads.iloc[:3])                       # строки по позиции
```

`pd.DataFrame(index=roads.index)` создаёт таблицу признаков с теми же метками строк, что у дорог.
При присваивании столбцов pandas совмещает значения **по индексу**. Сохраняйте индекс в `features` и функции стоимости.

### Сравнения

Сравнение столбца с числом возвращает `True` или `False` для каждой строки — булеву маску.

| Метод | Оператор | Смысл |
|---|---|---|
| `s.ge(100)` | `s >= 100` | Больше или равно |
| `s.gt(100)` | `s > 100` | Больше |
| `s.le(100)` | `s <= 100` | Меньше или равно |
| `s.lt(100)` | `s < 100` | Меньше |
| `s.eq(100)` | `s == 100` | Равно |
| `s.ne(100)` | `s != 100` | Не равно |

```python
s = pd.Series([50, 100, 150])
print(s.ge(100).tolist())                     # [False, True, True]
display(roads.loc[roads["длина_м"] >= 100])     # участки от 100 м
```

### Маски: и, или, отрицание

| Запись | Смысл |
|---|---|
| `a & b` | Оба условия выполнены |
| `a \| b` | Выполнено хотя бы одно условие |
| `~a` | Обратить True и False |
| `mask &= extra` | То же, что `mask = mask & extra` |
| `roads.loc[mask]` | Выбрать строки с True |

```python
mask = (roads["длина_м"] >= 100) & (roads["длина_м"] <= 500)
display(roads.loc[mask, ["длина_м", "тип_дороги"]])
```

В выражениях со столбцами нужны `&`, `|`, `~` и скобки вокруг сравнений.
Python-операторы `and`, `or`, `not` предназначены для отдельных условий.
Выбор строк в примере меняет только показанную таблицу; дорожная сеть остаётся целиком.

### any и all

`any()` — есть хотя бы одно True, `all()` — все значения True.

```python
mask = pd.Series([False, True, False])
print(mask.any())                            # True
print(mask.all())                            # False
print(mask.sum())                            # 1: число True

checks = pd.DataFrame({"a": [True, False], "b": [False, False]})
print(checks.any(axis=1).tolist())            # [True, False]: по строкам
print(checks.any(axis=0).tolist())            # [True, False]: по столбцам
print(checks.any(axis=None))                 # True: по всей таблице
```

Без `axis` таблица проверяется по столбцам. Поэтому `if checks.any():` не даёт одного условия.
У пустой булевой Series `any()` вернёт False, а `all()` — True.

### isin и составные дорожные теги

`isin` сравнивает ячейку целиком. Для тегов, соединённых `;`, в проекте есть функция `has_any`.

```python
from cycle_routing.tags import has_any

s = pd.Series(["асфальт", "асфальт;гравий", None])
print(s.isin(["асфальт"]).tolist())           # [True, False, False]
print(has_any(s, "асфальт").tolist())         # [True, True, False]
print(has_any(s, {"асфальт", "гравий"}).tolist())  # [True, True, False]
```

`has_any` — функция проекта, не метод pandas. Названия значений берите из `data_dictionary(city)`.
Она даёт **маску по дорогам**, а `mask.any()` — **одно условие** для всего столбца.

Пример своего признака:

```python
bike = has_any(roads["тип_дороги"], "велодорожка")
asphalt = has_any(roads["покрытие"], "асфальт")
f["велодорожка_с_асфальтом"] = (bike & asphalt).astype(int)
```

### Пропуски и числа

| Приём | Для чего |
|---|---|
| `s.isna()` / `s.notna()` | Найти отсутствующие / известные значения |
| `s.fillna(0)` | Заменить пропуски, если это соответствует смыслу данных |
| `s.astype(int)` | Преобразовать True/False в 1/0 |
| `s.astype(float)` | Преобразовать в числа с дробной частью |
| `pd.to_numeric(s, errors="coerce")` | Превратить строки в числа; непонятные значения → NaN |
| `s.clip(lower=0.1)` | Ограничить значения снизу |
| `s.value_counts()` | Посчитать частоты значений |
| `s.mean()` / `s.min()` / `s.max()` / `s.sum()` | Среднее, минимум, максимум, сумма |
| `s.to_numpy()` | Получить массив без индекса pandas |

В `city.trips` пропуск означает «не измерено», а ноль — «измерено ноль поездок».
Это разные значения; заменять неизвестные поездки нулями нельзя.

### Точки и геометрия

`city.roads` и наборы точек — таблицы GeoPandas: к операциям pandas в них добавлена геометрия.

```python
points = city.points
display(points_dictionary(city))             # типы точек в текущем наборе
display(points["светофоры"].head())           # один тип
display(points["велопарковки"].columns)       # доступные поля
```

Внутри `features` можно проверить близость разных типов точек:

```python
f["светофор_50м"] = within_radius(roads, points["светофоры"], 50).astype(int)
f["парковка_100м"] = within_radius(roads, points["велопарковки"], 100).astype(int)
```

`within_radius` — функция проекта. Радиус задаётся в метрах, расстояние считается до всей линии дороги.
Результат — маска с индексом дорог. Точки доступны локально и включаются в слоях карты.

### groupby и agg: свои метрики

`result.route_table` содержит маршруты с русскими названиями столбцов:

| Столбец | Что хранит |
|---|---|
| `пара` | Идентификатор пары A–B |
| `решение` | Название решения |
| `длина_м` | Физическая длина маршрута |
| `поворотов` | Число поворотов |
| `время_поиска_мс` | Время поиска, если оно измерено |
| `рёбра` | Последовательность участков `(u, v, key)` |
| `геометрия` | Линия маршрута |

```python
routes = result.route_table
routes["поворотов_на_км"] = routes["поворотов"] / (routes["длина_м"] / 1000)

comparison = result.comparison
comparison["поворотов на км"] = routes.groupby("решение")["поворотов_на_км"].mean().round(1)
display(comparison)
```

`groupby("решение")` собирает маршруты одного решения в группу, `mean()` усредняет значения внутри неё.
В примере каждый найденный маршрут имеет одинаковый вес. Число найденных маршрутов показано отдельно.

Несколько операций для каждой группы:

```python
summary = routes.groupby("решение").agg(
    средняя_длина_м=("длина_м", "mean"),
    максимальная_длина_м=("длина_м", "max"),
    среднее_поворотов=("поворотов", "mean"),
)
display(summary)
```

В записи `новое_имя=("столбец", "операция")` слева задаётся имя результата, справа — что и как считать.

### Частые ошибки

| Симптом | Что проверить |
|---|---|
| `The truth value of a Series is ambiguous` | Для столбцов нужны `&` и `|`; для одного условия — `any()` или `all()` |
| Составной тег не найден | Используется ли `has_any` вместо точного равенства? |
| В признаке или стоимости появился NaN | Совпадают ли индексы? Есть ли пропуски в исходном столбце? |
| Не найден столбец маршрута | Используется ли `result.route_table` и название из таблицы выше? |

Документация: [выбор строк](https://pandas.pydata.org/docs/user_guide/indexing.html),
[groupby](https://pandas.pydata.org/docs/user_guide/groupby.html),
[any](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.any.html).

## NetworkX (nx)

### Граф дорожной сети

Вершины — узлы сети, рёбра — направленные участки дорог. Используется `MultiDiGraph`:
между двумя вершинами может быть несколько рёбер, поэтому участок задаётся тройкой `(u, v, key)`.
Наличие пути A → B не гарантирует наличие пути B → A.

| Выражение | Результат |
|---|---|
| `node in graph` | Есть ли вершина |
| `graph.nodes[node]` | Атрибуты вершины, в том числе `x`, `y` |
| `graph.successors(node)` | Куда можно выйти из вершины |
| `graph.predecessors(node)` | Откуда можно прийти |
| `graph.edges[u, v, key]` | Атрибуты конкретного ребра |
| `graph.number_of_nodes()` | Число вершин |
| `graph.number_of_edges()` | Число рёбер |
| `nx.has_path(graph, source, target)` | Есть ли направленный путь |

`weight="cost"` означает: складывать атрибут `cost` у рёбер. При стоимости, равной длине,
получается кратчайший путь. При штрафах минимальная стоимость может соответствовать более длинному пути.
Дорожные признаки вычисляются по таблице дорог, а поиск получает уже готовые стоимости.

### Алгоритмы поиска

| Алгоритм | Вызов | Что учитывает |
|---|---|---|
| Дейкстра | `nx.dijkstra_path(G, s, t, weight="cost")` | Сумму неотрицательных стоимостей |
| A* | `nx.astar_path(G, s, t, heuristic=h, weight="cost")` | Стоимость пути и оценку расстояния до цели |
| Беллман — Форд | `nx.bellman_ford_path(G, s, t, weight="cost")` | Допускает отрицательные веса без достижимых отрицательных циклов; обычно медленнее |
| Двунаправленный Дейкстра | `nx.bidirectional_dijkstra(G, s, t, weight="cost")` | Возвращает `(стоимость, путь)` |
| Поиск в ширину, BFS | `nx.shortest_path(G, s, t)` | Число рёбер: длины и штрафы не учитываются |

В проекте стоимости должны быть положительными, даже если выбранный алгоритм допускает отрицательные веса.
Явно передавайте `weight="cost"`: значение по умолчанию у функций NetworkX может быть другим.

### Как подключить поиск

Функция принимает граф, начало, конец и имя веса. Возвращает список вершин, включая оба конца.

```python
def my_search(graph, source, target, weight="cost"):
    return nx.dijkstra_path(graph, source, target, weight=weight)

mine = solution(my_cost, turns=my_turn_penalty, search=my_search, label="моё решение")
```

Для двунаправленного Дейкстры извлеките путь из пары:

```python
def bidirectional_search(graph, source, target, weight="cost"):
    distance, path = nx.bidirectional_dijkstra(graph, source, target, weight=weight)
    return path
```

### A* и жадный поиск

В ноутбуке `greedy_search` — собственная функция. Она ориентируется на близость к цели,
не учитывая накопленную стоимость, и не гарантирует лучший путь.

A* учитывает и пройденную стоимость `g`, и оценку остатка `h`. Для гарантии лучшего пути
`h` не должна завышать остаток. Без `heuristic` используется ноль, и A* работает как Дейкстра.
Обычное расстояние по прямой может завышать стоимость, если предпочтительные дороги стоят дешевле своей длины.
Квадрат расстояния из `greedy_search` также не является готовой допустимой оценкой для A*.

### Поиск с поворотами

`turns=my_turn_penalty` включает учёт поворотов. Тогда вершины переданного поиску графа —
участки `(u, v, key)` и служебные начало/конец, а переходы соответствуют поворотам.
Стоимость включает следующий участок и штраф. Возвращать нужно вершины **этого** графа;
они не обязательно являются целыми числами.

### Ещё несколько функций

| Задача | Функция | Особенность |
|---|---|---|
| Обход в ширину / глубину | `nx.bfs_tree`, `nx.dfs_tree` | Исследуют достижимость |
| Слабая связность | `nx.weakly_connected_components` | Игнорирует направления |
| Сильная связность | `nx.strongly_connected_components` | Учитывает направления |
| Пути от одной вершины | `nx.single_source_dijkstra` | Возвращает расстояния и пути |
| Альтернативные пути без повторов | `nx.shortest_simple_paths` | Генератор; MultiDiGraph не поддерживается |

Генератор выдаёт пути постепенно. `list(...)` запрашивает все пути, что может потребовать много времени.

### Частые ошибки

| Симптом | Что проверить |
|---|---|
| `nx.NodeNotFound` | Есть ли начало и конец в графе? |
| `nx.NetworkXNoPath` | Есть ли направленный путь? |
| Поиск игнорирует стоимость | Передаётся ли `weight="cost"`? |
| Возвращается неправильный результат | Нужен список вершин, а не пара `(стоимость, путь)` |

Документация: [поиск путей](https://networkx.org/documentation/stable/reference/algorithms/shortest_paths.html),
[A*](https://networkx.org/documentation/stable/reference/algorithms/generated/networkx.algorithms.shortest_paths.astar.astar_path.html).

## scikit-learn (sklearn)

### Что получает модель

В проекте решается задача **регрессии** — предсказания числа, а не класса.

| Переменная | Содержимое |
|---|---|
| `x` | Таблица: строки — участки дорог, столбцы — числовые признаки из `features` |
| `y` | Цель для каждой строки: `log(1 + число поездок)` |
| `model` | Модель scikit-learn, которую создаёт и обучает `train_model` |
| `my_model` | Обёртка проекта: модель вместе с названиями признаков и функцией `features` |

`train_custom_model` подготавливает `x` и `y`, проверяет модель на отложенных кварталах,
затем обучает итоговую модель на всех известных открытых измерениях. Строки без измерений в обучение не попадают.

### fit и predict

| Вызов | Действие |
|---|---|
| `model.fit(x, y)` | Обучить модель на признаках и ответах |
| `model.predict(x_new)` | Получить прогноз для новых строк с теми же признаками |
| `model.get_params()` | Посмотреть параметры модели |
| `model.set_params(max_depth=3)` | Изменить глубину дерева; затем обучить его заново |

В ноутбуке заменяйте функцию `train_model`. Например, можно попробовать небольшое дерево:

```python
from sklearn.tree import DecisionTreeRegressor

def train_model(x, y):
    model = DecisionTreeRegressor(max_depth=2, random_state=42)
    model.fit(x, y)
    return model

my_model = train_custom_model(city, train_model, features=features)
display(my_model.predict_log_trips(city.roads).head())
```

Прогноз остаётся в той же шкале, что `y`: `log(1 + поездки)`.
После изменения модели выполните ячейки от обучения до сравнения и карты заново.

### Какие модели можно попробовать

| Модель | Импорт | Идея |
|---|---|---|
| `DummyRegressor` | `from sklearn.dummy import DummyRegressor` | Начальный вариант: всегда предсказывает среднее |
| `Ridge` | `from sklearn.linear_model import Ridge` | Складывает признаки с весами; ограничивает слишком большие веса |
| `DecisionTreeRegressor` | `from sklearn.tree import DecisionTreeRegressor` | Делит дороги по условиям вида «признак больше порога» |
| `RandomForestRegressor` | `from sklearn.ensemble import RandomForestRegressor` | Усредняет прогнозы нескольких деревьев |

Это альтернативы для экспериментов. Начальная модель в ноутбуке остаётся `DummyRegressor`.

### Основные параметры

| Параметр | Где используется | Что меняет |
|---|---|---|
| `strategy="mean"` | DummyRegressor | Предсказывать среднее значение цели |
| `max_depth` | Дерево, лес | Максимальная глубина дерева; большая глубина позволяет выучить более сложные зависимости |
| `min_samples_leaf` | Дерево, лес | Минимальное число обучающих строк в конечной группе |
| `n_estimators` | Лес | Число деревьев; больше деревьев требует больше времени |
| `alpha` | Ridge | Насколько сильно ограничивать веса |
| `random_state` | Дерево, лес | Фиксирует случайность для повторяемого эксперимента |
| `n_jobs` | Лес | Число параллельных задач; `1` — одна, `-1` — все доступные процессоры |

Сложная модель может хорошо запомнить обучающие дороги и хуже работать на новых.
Сравнивайте результаты на отложенных кварталах, а не только на обучении.

### Pipeline: подготовка признаков вместе с моделью

Для Ridge важно учитывать масштаб признаков: длина в сотнях метров и признак 0/1 имеют разный разброс.
`Pipeline` объединяет подготовку и модель в одну цепочку. Её обучение тоже выполняется через `fit`.

Другой вариант `train_model`:

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline

def train_model(x, y):
    model = make_pipeline(
        SimpleImputer(strategy="median", keep_empty_features=True),
        StandardScaler(),
        Ridge(alpha=1.0),
    )
    model.fit(x, y)
    return model
```

- `SimpleImputer` заполняет пропуски в признаках медианами обучающих значений; полностью пустой столбец сохраняет и заполняет нулём.
- `StandardScaler` вычитает среднее и делит на стандартное отклонение обучающего признака.
- `Ridge` учится на подготовленной таблице.

Для новых дорог цепочка использует параметры, найденные при обучении. Заполняются пропуски **признаков**, не неизвестные поездки.
Деревьям и лесу выравнивание масштаба обычно не требуется.

### Как прогноз влияет на маршрут

`ml_feature(my_model)(roads)` превращает прогноз в признак `ml_популярность` от 0 до 1 по рангу прогноза среди переданных дорог.
Высокое значение означает более высокий прогноз относительно других дорог.
В ноутбуке этот столбец добавляет `features_with_my_model`.

```python
ML_WEIGHTS = WEIGHTS.copy()
ML_WEIGHTS["ml_популярность"] = -0.25
my_ml_cost = comfort_cost(ML_WEIGHTS, features=features_with_my_model)
```

Отрицательный вес делает участки с высоким прогнозом дешевле. Вес 0 отключает влияние прогноза.
Проверка прогноза на отложенных улицах и популярность построенных маршрутов — разные показатели.

### Частые ошибки

| Симптом | Что проверить |
|---|---|
| `NotFittedError` | Был ли вызван `fit` до `predict`? |
| Модель не принимает строки или пропуски | Возвращает ли `features` числовые столбцы? Нужна ли обработка пропусков? |
| Не совпадают названия или число признаков | Одинаковы ли признаки при обучении и прогнозе? После изменения `features` модель нужно переобучить |
| Начальный прогноз одинаковый | Для DummyRegressor это ожидаемо |
| Изменения модели не видны на маршрутах | Выполнены ли обучение, создание стоимости, сборка и сравнение заново? Не равен ли вес прогноза нулю? |

Документация: [начало работы](https://scikit-learn.org/stable/getting_started.html),
[DummyRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html),
[DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html),
[RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html),
[Ridge](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html),
[SimpleImputer](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html),
[StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html),
[Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.make_pipeline.html).
