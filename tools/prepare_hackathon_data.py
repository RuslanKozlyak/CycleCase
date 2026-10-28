"""Prepare public Konstanz and private test labels before class; optional local SPb transfer pack."""
from pathlib import Path
import argparse
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from cycle_routing.workshop import load_lab,presets,run_routes
from cycle_routing.workshop_data import prepare_konstanz,prepare_spb_from_graph


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"notebooks/data/hackathon")
    parser.add_argument("--private-dir",type=Path,default=ROOT/"outputs/hackathon_private/konstanz")
    parser.add_argument("--source-cache",type=Path,default=ROOT/"notebooks/data/cities")
    parser.add_argument("--warm-presets",action="store_true")
    parser.add_argument("--spb-graph",type=Path,help="Explicit local GraphML; prepares a separate personal GPX pack, never added to the student ZIP")
    args=parser.parse_args()
    manifest=prepare_konstanz(source_cache=args.source_cache,folder=args.data_dir/"konstanz",private_folder=args.private_dir)
    print("Konstanz prepared:",manifest["edges"],"edges,",manifest["od_pairs"],"OD",flush=True)
    if args.warm_presets:
        run_routes(load_lab(args.data_dir/"konstanz"),presets())
        print("Standard preset route cache prepared",flush=True)
    if args.spb_graph:
        manifest=prepare_spb_from_graph(source_graph=args.spb_graph,gpx_directory=ROOT/"cicle_gpx",folder=args.data_dir/"saint_petersburg")
        print("SPb prepared:",manifest["edges"],"edges",flush=True)


if __name__=="__main__":main()
