"""Organizer grades submitted routes using withheld test labels and an independent shortest path."""
from pathlib import Path
import argparse
import json
import sys
import geopandas as gpd
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cycle_routing.workshop import load_lab,evaluate_routes,EvaluationRules


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission",type=Path)
    parser.add_argument("--data",type=Path,default=ROOT/"notebooks/data/hackathon/konstanz")
    parser.add_argument("--private",type=Path,default=ROOT/"outputs/hackathon_private/konstanz")
    args=parser.parse_args()
    lab=load_lab(args.data)
    metadata=json.loads((args.submission/"submission.json").read_text(encoding="utf-8"))
    private=json.loads((args.private/"dataset.json").read_text(encoding="utf-8"))
    if metadata["dataset_id"]!=lab.manifest["dataset_id"] or private["dataset_id"]!=lab.manifest["dataset_id"]:
        raise ValueError("Submission and test labels refer to another dataset")
    routes=gpd.read_parquet(args.submission/"routes.parquet")
    if set(routes.algorithm)!={metadata["algorithm"]}:raise ValueError("Unexpected algorithms in submission")
    result=evaluate_routes(lab,routes,split="test",labels=pd.read_parquet(args.private/"test_labels.parquet"),
                           rules=EvaluationRules(),solutions={metadata["algorithm"]:{}})
    result.table.to_csv(args.submission/"test_score.csv")
    result.per_od.to_csv(args.submission/"test_per_od.csv",index=False)
    print(result.table.round(3).to_string())


if __name__=="__main__":main()
