"""Organizer smoke run: freeze Konstanz validation choice, then evaluate the optional SPb pack."""
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from cycle_routing.learning import fit_popularity_model
from cycle_routing.workshop import load_lab, compare, preset, priority_cost, learned_cost, prediction_quality
from cycle_routing.workshop_maps import route_explorer


def main():
    lab = load_lab(ROOT/"notebooks/data/hackathon/konstanz")
    model = fit_popularity_model(*lab.training_data())
    ml = {f"ml_{s:g}": preset(f"ML {s:g}", cost=learned_cost(model, strength=s), turns=None) for s in (0, .25, .5, 1)}
    validation = compare(lab, ml)
    chosen = validation.table.loc[list(ml), "балл / 100"].idxmax()
    print("Konstanz validation selected:", chosen, flush=True)
    spb = load_lab(ROOT/"notebooks/data/hackathon/saint_petersburg")
    solutions = {"my": preset("Мой пресет", cost=priority_cost(), turns={"left_penalty_m":30, "right_penalty_m":10}), chosen:ml[chosen]}
    result = compare(spb, solutions, split="external")
    output = ROOT/"outputs/hackathon_demo"
    result.table.to_csv(output/"spb_transfer.csv")
    route_explorer(spb, result, filename=output/"spb_routes.html")
    (output/"transfer_check.json").write_text(json.dumps({"selected_on_konstanz":chosen,
        "spb_edges":len(spb.edges),"spb_known_labels":int(spb.labels_for("external").notna().sum()),
        "spb_routes":len(result.routes),"spb_evaluated_od":int(result.per_od.loc[result.per_od.algorithm.eq("shortest"),"in_evaluation"].sum())},indent=2), encoding="utf-8")
    print("SPb transfer checked:", len(result.routes), "routes", flush=True)


if __name__ == "__main__":
    main()
