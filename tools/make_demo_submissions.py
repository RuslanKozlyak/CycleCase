"""Organizer: write a few demo team submissions to try notebooks/hackathon_grading.ipynb before real ones arrive.

    python tools/make_demo_submissions.py --pack konstanz_more

Teams solve the task in different ways, one is sent as a ZIP; two are broken on purpose (no submission.json,
another dataset) to show how the reviewers' notebook reports them.
"""
from pathlib import Path
import argparse
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from cycle_routing.workshop import comfort_cost, export_solution, features, load_city, ml_feature, rideable, solution, train_popularity
from cycle_routing.tags import has_any

WEIGHTS = {
    "крупная_дорога": 0.6,
    "велодорожка": -0.4,
    "асфальт": -0.2,
}
TYPE_PRICE = {"велодорожка": 0.5, "жилая_зона": 0.7, "жилая_улица": 0.8, "местная_дорога": 1.0,
              "дорога_второго_значения": 1.3, "главная_дорога": 1.6}


def by_type(roads):
    return roads["длина_м"] * roads["тип_дороги"].str.split(";").str[0].map(TYPE_PRICE).fillna(1.0)


def no_sidewalks(roads):
    return rideable(roads) & ~has_any(roads["тип_дороги"], {"пешеходная_дорожка", "пешеходная_улица"})


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", default="konstanz_more")
    parser.add_argument("--other-pack", default="konstanz", help="a different pack, for the 'wrong dataset' example")
    parser.add_argument("--out", type=Path, default=ROOT/"outputs/hackathon/submissions")
    args = parser.parse_args()
    shutil.rmtree(args.out, ignore_errors=True)
    city = load_city(ROOT/"notebooks/data/hackathon"/args.pack)
    model = train_popularity(city)
    def features_with_ml(roads):
        table = features(roads)
        table["ml_популярность"] = ml_feature(model)(roads)["ml_популярность"]
        return table

    teams = {
        "Велосипедисты": (solution(comfort_cost({**WEIGHTS, "ml_популярность": -0.8}, features=features_with_ml), turns=None,
                                   label="веса + ML"), None),
        "Без поворотов": (solution(comfort_cost(WEIGHTS), turns="min_turns", label="веса + минимум поворотов"), None),
        "Табличка": (solution(by_type, turns=None, label="цена по типу дороги"), None),
        "Без тротуаров": (solution(comfort_cost(WEIGHTS), turns="default", label="веса без тротуаров"), no_sidewalks),
    }
    for team, (chosen, keep) in teams.items():
        export_solution(city, chosen, keep=keep, folder=args.out/team)
    shutil.make_archive(str(args.out/"Архивом"), "zip", args.out/"Табличка")
    (args.out/"Пустая папка").mkdir()
    other = load_city(ROOT/"notebooks/data/hackathon"/args.other_pack)
    export_solution(other, solution(comfort_cost(WEIGHTS), turns=None), folder=args.out/"Старый набор")
    print("Demo submissions:", ", ".join(sorted(p.name for p in args.out.iterdir())))


if __name__ == "__main__":
    main()
