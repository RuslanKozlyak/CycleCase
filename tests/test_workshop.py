from dataclasses import replace

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import LineString

from cycle_routing import prepare_city
from cycle_routing.evaluation import edge_trips
from cycle_routing.tags import has_any
from cycle_routing.workshop import (
    build_network, closed_city, comfort_cost, data_dictionary, example_rows, export_solution, features, grade,
    gpx_table, load_city, ml_feature, osm, preview_cost, preview_features, rideable, russian, score, solution, train_custom_model,
)
from cycle_routing.graph import drop_values
from cycle_routing.workshop_data import write_city_pack
from cycle_routing.workshop_maps import route_explorer


@pytest.fixture
def city(toy_city, tmp_path):
    """North cycleway (popular) and south primary (quiet) are open; the footway is the closed zone."""
    roads, nodes = prepare_city("toytown", cache_dir=toy_city["cache_dir"])
    trips = edge_trips(roads, "toytown", cache_dir=toy_city["cache_dir"])
    footway = has_any(roads.highway, "footway")
    trips[footway] = 7.0
    zone = pd.Series(np.where(footway | has_any(roads.highway, "steps"), "closed", "open"), index=roads.index)
    pairs = pd.DataFrame({"od_id": ["OD01", "CL01"], "origin": [1, 1], "destination": [4, 4], "zone": ["open", "closed"],
                          "start_lat": [47.66] * 2, "start_lon": [9.17] * 2, "end_lat": [47.661] * 2, "end_lon": [9.171] * 2})
    public, private = tmp_path / "public", tmp_path / "private"
    write_city_pack(roads, nodes, pairs, trips, zone, folder=public, private_folder=private, city="toy",
                   split_params={"method": "test"})
    return load_city(public), private


def bike():
    return solution(comfort_cost({"велодорожка": -0.9, "крупная_дорога": 2}), turns=None, label="bike")


def test_student_table_is_russian_and_features_have_no_targets(city):
    data, _ = city
    assert {"тип_дороги", "покрытие", "длина_м", "геометрия"} <= set(data.roads) and "highway" not in data.roads
    assert data.roads.index.equals(data.osm.index)
    f = features(data.roads.assign(trips=1e9, popularity=42))
    assert list(f) == ["длина_м", "велодорожка", "крупная_дорога", "асфальт"]
    assert "пешеходная_дорожка" in set(data.roads["тип_дороги"])
    assert "footway" in set(data.osm.highway)
    assert rideable(data.roads).equals(rideable(data.osm))


def test_russian_values_preserve_compound_tags_missing_values_and_geometry(city):
    from geopandas.testing import assert_geodataframe_equal
    from cycle_routing.workshop import RU, VALUE_NAMES, OSM_VALUES

    data, _ = city
    source = data.osm.copy()
    source.loc[source.index[0], "highway"] = "footway;service;residential;path;track;steps;secondary;living_street"
    original = source.copy()
    translated = russian(source)
    assert translated.iloc[0]["тип_дороги"] == (
        "пешеходная_дорожка;служебный_проезд;жилая_улица;тропа;хозяйственная_дорога;лестница;дорога_второго_значения;жилая_зона"
    )
    for column, mapping in VALUE_NAMES.items():
        assert len(mapping) == len(OSM_VALUES[column])  # every translated name can be reversed unambiguously
        assert not any(" " in value for value in translated[column].dropna())
        assert translated[column].isna().equals(source[next(key for key, value in RU.items() if value == column)].isna())
    assert_geodataframe_equal(osm(translated), source[[*RU, "geometry"]])
    assert_geodataframe_equal(russian(translated), translated)
    assert_geodataframe_equal(source, original)
    pd.testing.assert_series_equal(
        drop_values(translated, тип_дороги=["пешеходная_дорожка"]),
        ~has_any(source.highway, "footway"), check_names=False,
    )


def test_dictionary_accepts_an_entirely_unknown_category(city):
    data, _ = city
    data = replace(data, osm=data.osm.assign(surface=float("nan")))
    values = data_dictionary(data).data.loc["покрытие", "значения"]
    assert "нет данных" in values


def test_dictionary_explains_geometry_without_technical_crs_names(city):
    from pyproj import CRS

    data, _ = city
    crs = CRS.from_user_input(CRS.from_epsg(25832).to_json_dict())
    data = replace(data, osm=data.osm.set_crs(crs, allow_override=True))
    description = data_dictionary(data).data.loc["геометрия", "значения"]
    assert description == "линия дороги; координаты в метрах"


def test_public_pack_hides_closed_zone_but_marks_it(city):
    data, private = city
    assert {p.name for p in data.folder.iterdir()} >= {"roads.parquet", "labels.parquet", "pairs.parquet", "test_area.parquet"}
    assert not any("closed" in p.name or "zones" in p.name for p in data.folder.iterdir())
    assert list(data.pairs.od_id) == ["OD01"]
    footway = has_any(data.osm.highway, "footway")
    assert data.trips[footway].isna().all() and data.test_area[footway].all() and not data.test_area[~footway & ~has_any(data.osm.highway, "steps")].any()
    assert data.trips.notna().sum() == 8  # north and south street, two ways each, both directions
    closed = closed_city(data, private)
    assert list(closed.pairs.od_id) == ["CL01"] and closed.trips.dropna().eq(7).all()
    with pytest.raises(FileNotFoundError, match="подготовленного"):
        load_city(data.folder / "missing")
    (data.folder / "labels.parquet").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="Изменён"):
        load_city(data.folder)


def test_score_counts_actual_popularity_even_for_long_detours(city):
    data, _ = city
    result = score(data, {"bike": bike()})
    table, per_od = result.table, result.per_od.set_index("algorithm")
    assert table.loc["bike", "популярность"] > table.loc["кратчайший", "популярность"]
    assert table.loc["bike", "популярнее кратчайшего"] == "1 из 1"
    assert table.loc["кратчайший", "объезд, %"] == 0
    assert per_od.loc["bike", "search_ms"] >= 0
    north = has_any(data.osm.highway, "cycleway")
    longer = replace(data, osm=data.osm.assign(length_m=data.osm.length_m.where(~north, data.osm.length_m * 4)))
    actual = score(longer, {"bike": bike()})
    long_route = actual.per_od.set_index("algorithm").loc["bike"]
    assert long_route.detour_pct > 20
    assert long_route.counted == long_route.popularity and long_route.note == ""
    assert long_route.counted > actual.per_od.set_index("algorithm").loc["shortest", "counted"]
    assert actual.pareto.loc["bike", "Парето"] and actual.pareto.loc["кратчайший", "Парето"]
    actual.plot_pareto()


def test_unmeasured_roads_count_as_average(city):
    data, _ = city
    north = has_any(data.osm.highway, "cycleway")
    unmeasured = replace(data, trips=data.trips.where(~north))
    per_od = score(unmeasured, {"bike": bike()}).per_od.set_index("algorithm")
    assert per_od.loc["bike", "coverage_pct"] == 0 and per_od.loc["bike", "counted"] == 50


def test_own_cost_function_gets_russian_table(city):
    data, _ = city
    seen = []

    def my_cost(roads):
        seen.append(set(roads.columns))
        return roads["длина_м"] * (1 + 5 * roads["тип_дороги"].eq("главная_дорога"))

    per_od = score(data, {"own": solution(my_cost, turns=None)}).per_od.set_index("algorithm")
    assert "тип_дороги" in seen[0] and "highway" not in seen[0]
    assert per_od.loc["own", "popularity"] > per_od.loc["shortest", "popularity"]
    preview = preview_cost(data, my_cost, n=2)
    assert list(preview.columns)[-1] == "стоимость" and len(preview) == 2


def test_preview_shows_weighted_features_and_cost_last(city):
    data, _ = city
    preview = preview_cost(data, comfort_cost({"велодорожка": -0.5}), n=3)
    assert list(preview.columns) == ["тип_дороги", "велодорожка", "длина_м", "цена 1 м", "стоимость"]
    assert (preview["цена 1 м"].where(preview["велодорожка"].eq(1)).dropna() == 0.5).all()
    with pytest.raises(KeyError, match="велодорожка"):
        preview_cost(data, comfort_cost({"no_such_feature": 1}))


def test_filter_without_route_counts_zero(city):
    data, _ = city
    network = build_network(data, lambda roads: rideable(roads) & roads["тип_дороги"].eq("велодорожка"))
    assert "Без маршрута останутся пары: 1" in network.summary()
    result = score(data, {"bike": bike()}, network=network)
    per_od = result.per_od.set_index("algorithm")
    assert per_od.loc["bike", "counted"] == 0 and per_od.loc["bike", "note"] == "нет маршрута"
    assert not result.pareto.loc["bike", "Парето"] and pd.isna(result.pareto.loc["bike", "длина, км"])
    with pytest.raises(ValueError, match="True/False"):
        build_network(data, lambda roads: roads["длина_м"])


def test_data_dictionary_lists_every_value_and_examples_cover_every_column(city):
    data, _ = city
    dictionary = data_dictionary(data).data
    assert len(dictionary) == len(data.roads.columns) + 1 and list(dictionary.columns) == ["тип", "значения"]
    assert "поездки (city.trips)" in dictionary.index
    kinds = data.roads["тип_дороги"].dropna().str.split(";").explode().unique()
    assert all(kind in dictionary.loc["тип_дороги", "значения"] for kind in kinds) and "…" not in dictionary.loc["тип_дороги", "значения"]
    assert "пешеходная_дорожка" in dictionary.loc["тип_дороги", "значения"]
    assert "footway" not in dictionary.loc["тип_дороги", "значения"]
    assert "асфальт" in dictionary.loc["покрытие", "значения"]
    rows = example_rows(data)
    filled_anywhere = data.roads.drop(columns="геометрия").notna().any()
    assert rows.notna().any()[filled_anywhere].all() and len(rows) < len(data.roads)


def test_previews_show_own_features_where_they_are_present(city):
    data, _ = city

    def feature_function(roads):
        return pd.DataFrame({"лестница": roads["тип_дороги"].eq("лестница").astype(float)}, index=roads.index)

    shown = preview_features(data, feature_function)
    assert list(shown.columns) == ["тип_дороги", "лестница"] and shown["лестница"].eq(1).any()
    formula = lambda roads: roads["длина_м"] * (1 + feature_function(roads)["лестница"])
    preview = preview_cost(data, formula, features=feature_function)
    assert list(preview.columns) == ["тип_дороги", "лестница", "длина_м", "цена 1 м", "стоимость"]
    assert (preview["цена 1 м"].where(preview["лестница"].eq(1)).dropna() == 2).all()


def test_export_and_grade_on_closed_zone(city, tmp_path):
    data, private = city
    keep = lambda roads: rideable(roads) | has_any(roads["тип_дороги"], "пешеходная_дорожка")
    folder = export_solution(data, bike(), keep=keep, folder=tmp_path / "submission")
    result = grade(data, private, folder)
    assert set(result.table.index) == {"решение", "кратчайший"}
    assert result.city.name == "закрытая зона"


def test_previously_exported_cost_still_works(city):
    import cloudpickle
    from cycle_routing.workshop_compat import legacy_features

    data, _ = city
    cost = comfort_cost({"плохое_покрытие": 0.5}, features=legacy_features)
    expected = cost(data.osm)
    del cost.make_features
    cost.__dict__.update(base_features=features, extra=())  # simulate the earlier serialized format
    restored = cloudpickle.loads(cloudpickle.dumps(cost))
    pd.testing.assert_series_equal(restored(data.osm), expected)


def test_route_explorer_is_offline_and_shows_tracks(city, tmp_path):
    data, _ = city
    result = score(data, {"bike": bike()})
    tracks = gpd.GeoDataFrame({"activity_id": ["ride"]}, geometry=[LineString(data.osm.geometry.iloc[0].coords)], crs=data.osm.crs)
    path = tmp_path / "map.html"
    view = route_explorer(data, result, filename=path, tracks=tracks)
    text = path.read_text(encoding="utf-8")
    assert "__DATA__" not in text and "__LEAFLET_JS__" not in text and 'src="http' not in text
    assert "GPX-треки" in text and '"tracks":[[' in text and "популярность улицы" in text and '"gpx":' in text
    assert 'title="Сравнение' in view._repr_html_()


def test_gpx_table_measures_overlap_with_rides(city):
    data, _ = city
    result = score(data, {"bike": bike()})
    north = data.osm[has_any(data.osm.highway, "cycleway")]
    tracks = gpd.GeoDataFrame({"activity_id": ["north"] * len(north)}, geometry=list(north.geometry), crs=data.osm.crs)
    table = gpx_table(result, tracks)
    assert table.loc["bike"].iloc[0] == 100 and table.loc["кратчайший"].iloc[0] < 50


def test_any_solution_name_works_with_the_route_cache(city):
    data, _ = city
    table = score(data, {'веса: "моё" / v2?': bike()}).table
    assert 'веса: "моё" / v2?' in table.index
    assert all(":" not in p.name and "?" not in p.name for p in (data.folder / "route_cache").iterdir())


def test_own_sklearn_model_learns_from_the_supplied_features(city):
    from sklearn.tree import DecisionTreeRegressor
    data, _ = city

    def feature_function(roads):
        table = features(roads)
        table["велоулица"] = has_any(roads["тип_дороги"], "велодорожка").astype(float)
        return table

    seen = []

    def train_model(x, y):
        seen.append((x.copy(), y.copy()))
        model = DecisionTreeRegressor(max_depth=1, random_state=42)
        model.fit(x, y)
        return model

    trained = train_custom_model(data, train_model, features=feature_function)
    x, y = seen[-1]
    assert x.index.equals(data.trips.dropna().index)
    pd.testing.assert_series_equal(y, np.log1p(data.trips.dropna()))
    assert "велоулица" in trained.columns and "велодорожка" in trained.columns and "длина_м" in trained.columns
    predicted = trained.predict_log_trips(data.osm)
    assert predicted.index.equals(data.osm.index) and predicted.notna().all()
    cost = comfort_cost({"ml_популярность": -0.8}, features=ml_feature(trained))
    per_od = score(data, {"ml": solution(cost, turns=None)}).per_od.set_index("algorithm")
    assert per_od.loc["ml", "popularity"] > per_od.loc["shortest", "popularity"]  # the popular north road is learned
