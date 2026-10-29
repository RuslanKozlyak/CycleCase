import json
import shutil
import zipfile

import matplotlib
import pytest

from cycle_routing.grading import grade_all, load_submissions, submissions_table
from cycle_routing.workshop import comfort_cost, export_solution, solution

from test_workshop import bike, city  # noqa: F401  (fixture)

matplotlib.use("Agg")


@pytest.fixture
def submissions(city, tmp_path):
    data, _ = city
    folder = tmp_path / "submissions"
    export_solution(data, bike(), folder=folder / "велосипедисты")
    export_solution(data, solution(comfort_cost({"крупная_дорога": 0.1}), turns=None, label="почти кратчайший"),
                    folder=tmp_path / "zipped" / "inside")
    shutil.make_archive(str(folder / "архив"), "zip", tmp_path / "zipped")
    (folder / "пустая").mkdir()
    other = export_solution(data, bike(), folder=folder / "чужой набор")
    meta = json.loads((other / "submission.json").read_text(encoding="utf-8"))
    (other / "submission.json").write_text(json.dumps({**meta, "dataset_id": "other"}), encoding="utf-8")
    with zipfile.ZipFile(folder / "взлом.zip", "w") as archive:
        archive.writestr("../evil.txt", "x")
    return folder


def test_submissions_load_with_a_status_for_each_team(city, submissions):
    data, _ = city
    status = submissions_table(load_submissions(submissions, data))["статус"]
    assert status["велосипедисты"] == "ок" and status["архив"] == "ок"
    assert "submission.json" in status["пустая"] and "другого набора" in status["чужой набор"]
    assert "вне папки" in status["взлом"]
    assert not (submissions.parent / "evil.txt").exists()


def test_leaderboard_ranks_teams_on_the_closed_zone(city, submissions):
    data, private = city
    board = grade_all(data, private, load_submissions(submissions, data))
    assert set(board.table.index) == {"велосипедисты", "архив", "кратчайший"}
    assert list(board.table.columns[:4]) == ["Парето", "популярность, закрытая зона", "±", "ярус популярности"]
    assert board.table["ярус популярности"].min() == 1 and board.table["популярность, открытая зона"].notna().all()
    assert board.table["Парето"].dtype == bool and "длина, км" in board.table
    assert board.table["Парето"].equals(board.pareto["Парето"].reindex(board.table.index))
    assert set(board.errors.index) == {"пустая", "чужой набор", "взлом"}
    assert board.result.city.name == "закрытая зона" and set(board.result.routes.algorithm) == {"shortest", "велосипедисты", "архив"}
    board.plot()
    board.plot_pareto()
