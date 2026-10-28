"""Run the distributed notebook from an isolated copy of the student ZIP, without master/test data."""
from pathlib import Path
import json
import zipfile

import nbformat
from nbclient import NotebookClient
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main():
    archive = ROOT/"outputs/hackathon_student.zip"
    stage = ROOT/".tmp/hackathon_student_check"
    stage.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        assert not any(part in name for name in names for part in (
            "test_labels", "labels_test", "hackathon_private", "saint_petersburg", "cicle_gpx", "data/cities",
        ))
        assert all(stage.resolve() in (stage/name).resolve().parents for name in names)
        source.extractall(stage)
    notebook = nbformat.read(stage/"notebooks/hackathon_konstanz.ipynb", as_version=4)
    assert not any(ord(char)<32 and char not in "\n\r\t" for cell in notebook.cells for char in cell.source)
    print("Running isolated student notebook without route caches...", flush=True)
    NotebookClient(notebook, timeout=1200, kernel_name="python3", resources={"metadata":{"path":str(stage/"notebooks")}}).execute()
    nbformat.write(notebook, stage/"notebooks/hackathon_konstanz.ipynb")
    output = stage/"outputs/hackathon_demo"
    actual = pd.read_csv(output/"validation_all.csv", index_col=0)
    expected = pd.read_csv(ROOT/"outputs/hackathon_demo/validation_all.csv", index_col=0)
    pd.testing.assert_frame_equal(actual, expected, atol=1e-10, rtol=1e-10)
    assert not any(o.output_type=="error" for c in notebook.cells for o in c.get("outputs", []))
    verification = {"isolated_student_run":"passed", "archive_files":len(names),
                    "private_or_personal_data_in_archive":False,
                    "code_cells_executed":sum(c.cell_type=="code" for c in notebook.cells),
                    "solutions_compared":len(actual), "scores_match_main_run":True,
                    "browser_visual_check":"unavailable in this session"}
    (ROOT/"outputs/hackathon_demo/verification.json").write_text(json.dumps(verification,indent=2),encoding="utf-8")
    print("Isolated student run passed:", len(actual), "solutions, identical scores", flush=True)


if __name__ == "__main__":
    main()
