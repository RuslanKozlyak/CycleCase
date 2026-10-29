"""Organizer: score all team submissions in a folder on the closed zone (same as notebooks/hackathon_grading.ipynb).

    python tools/grade_hackathon.py outputs/hackathon/submissions --pack konstanz_more

Every subfolder or ZIP in the folder is one team (the folder written by export_solution).
"""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from cycle_routing.grading import grade_all, load_submissions, submissions_table
from cycle_routing.workshop import load_city


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("submissions", type=Path)
    parser.add_argument("--pack", default="konstanz_more")
    args = parser.parse_args()
    city = load_city(ROOT/"notebooks/data/hackathon"/args.pack)
    teams = load_submissions(args.submissions, city)
    print(submissions_table(teams).to_string())
    board = grade_all(city, ROOT/"outputs/hackathon_private"/args.pack, teams)
    board.table.to_csv(args.submissions/"leaderboard.csv")
    board.pareto.to_csv(args.submissions/"pareto.csv")
    if board.result is not None:
        board.plot_pareto().savefig(args.submissions/"pareto.png", dpi=160)
    print("\n" + board.table.to_string())
    if len(board.errors):
        print("\n" + board.errors.to_string())


if __name__ == "__main__":
    main()
