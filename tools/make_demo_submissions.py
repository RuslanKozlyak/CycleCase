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
from cycle_routing.workshop import comfort_cost, export_solution, load_city, ml_feature, rideable, solution, train_popularity

WEIGHTS = {"крупная_дорога": 0.6, "плохое_покрытие": 0.4, "велодорожка": -0.4, "без_освещения": 0.2}
TYPE_PRICE = {"cycleway": 0.5, "living_street": 0.7, "residential": 0.8, "tertiary": 1.0, "secondary": 1.3, "primary": 1.6}


def by_type(roads):
    return roads["длина_м"] * roads["тип_дороги"].str.split(";").str[0].map(TYPE_PRICE).fillna(1.0)


def no_sidewalks(roads):
    return rideable(roads) & ~roads["тип_дороги"].isin(["footway", "pedestrian"])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", default="konstanz_more")
    parser.add_argument("--other-pack", default="konstanz", help="a different pack, for the 'wrong dataset' example")
    parser.add_argument("--out", type=Path, default=ROOT/"outputs/hackathon/submissions")
    args = parser.parse_args()
    shutil.rmtree(args.out, ignore_errors=True)
    lab = load_city(ROOT/"notebooks/data/hackathon"/args.pack)
    model = train_popularity(lab)
    teams = {
        "Велосипедисты": (solution(comfort_cost({**WEIGHTS, "ml_популярность": -0.8}, extra=ml_feature(model)), turns=None,
                                   label="веса + ML"), None),
        "Без поворотов": (solution(comfort_cost(WEIGHTS), turns="min_turns", label="веса + минимум поворотов"), None),
        "Табличка": (solution(by_type, turns=None, label="цена по типу дороги"), None),
        "Без тротуаров": (solution(comfort_cost(WEIGHTS), turns="default", label="веса без тротуаров"), no_sidewalks),
    }
    for team, (chosen, keep) in teams.items():
        export_solution(lab, chosen, keep=keep, folder=args.out/team)
    shutil.make_archive(str(args.out/"Архивом"), "zip", args.out/"Табличка")
    (args.out/"Пустая папка").mkdir()
    other = load_city(ROOT/"notebooks/data/hackathon"/args.other_pack)
    export_solution(other, solution(comfort_cost(WEIGHTS), turns=None), folder=args.out/"Старый набор")
    print("Demo submissions:", ", ".join(sorted(p.name for p in args.out.iterdir())))


if __name__ == "__main__":
    main()
