"""Prepare a Konstanz pack (open zone public, closed zone private) with a chosen split; optional SPb GPX pack.

    python tools/prepare_hackathon_data.py                      # split by OD corridors -> pack "konstanz"
    python tools/prepare_hackathon_data.py --split sectors      # checkerboard -> pack "konstanz_sectors"
    python tools/prepare_hackathon_data.py --spb-graph PATH     # also the personal GPX check

The notebook picks a pack with PACK = "<name>". Candidate corridors are cached and shared by all splits.
"""
from pathlib import Path
import argparse
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from cycle_routing.splits import SPLITS
from cycle_routing.workshop_data import prepare_konstanz, prepare_spb_from_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=sorted(SPLITS), default="routes")
    parser.add_argument("--name", help="pack folder name; default konstanz or konstanz_<split>")
    parser.add_argument("--params", default="{}", help='JSON with split parameters, e.g. {"per_zone_stratum": 6}')
    parser.add_argument("--data-dir", type=Path, default=ROOT/"notebooks/data/hackathon")
    parser.add_argument("--private-dir", type=Path, default=ROOT/"outputs/hackathon_private")
    parser.add_argument("--source-cache", type=Path, default=ROOT/"notebooks/data/cities")
    parser.add_argument("--spb-graph", type=Path, help="local GraphML of SPb; the pack stays out of the student ZIP")
    args = parser.parse_args()
    name = args.name or ("konstanz" if args.split == "routes" else f"konstanz_{args.split}")
    manifest = prepare_konstanz(source_cache=args.source_cache, folder=args.data_dir/name, private_folder=args.private_dir/name,
                                split=args.split, candidates_cache=args.private_dir/"candidates_konstanz.pkl",
                                **json.loads(args.params))
    print(f"{name}: {manifest['roads']} roads, {manifest['open_pairs']} open OD, {manifest['open_labels']} open labels;"
          f" split {json.dumps(manifest['split'], ensure_ascii=False)}", flush=True)
    if args.spb_graph:
        manifest = prepare_spb_from_graph(source_graph=args.spb_graph, gpx_directory=ROOT/"cicle_gpx",
                                          folder=args.data_dir/"saint_petersburg")
        print("saint_petersburg:", manifest["roads"], "roads,", manifest["open_pairs"], "OD", flush=True)


if __name__ == "__main__":
    main()
