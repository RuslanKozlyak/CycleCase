"""Run the distributed notebook from an isolated copy of the student ZIP, without closed-zone or personal data."""
from pathlib import Path
import json
import shutil
import zipfile

import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = ("closed", "zones", "hackathon_private", "saint_petersburg", "cicle_gpx", "data/cities", "tracks", "old_packs")


def main():
    archive = ROOT/"outputs/hackathon_student.zip"
    stage = ROOT/".tmp/hackathon_student_check"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        assert not any(word in name for name in names for word in FORBIDDEN)
        assert all(stage.resolve() in (stage/name).resolve().parents for name in names)
        source.extractall(stage)
    notebook = nbformat.read(stage/"notebooks/hackathon_konstanz.ipynb", as_version=4)
    print("Running the isolated student notebook...", flush=True)
    NotebookClient(notebook, timeout=1800, kernel_name="python3", resources={"metadata": {"path": str(stage/"notebooks")}}).execute()
    errors = [o for c in notebook.cells for o in c.get("outputs", []) if o.output_type == "error"]
    assert not errors, errors
    pack = next((stage/"outputs/hackathon").iterdir())
    assert (pack/"submission/solution.pkl").exists()
    verification = {"isolated_student_run": "passed", "archive_files": len(names),
                    "closed_or_personal_data_in_archive": False,
                    "code_cells_executed": sum(c.cell_type == "code" for c in notebook.cells)}
    (ROOT/"outputs/hackathon").mkdir(parents=True, exist_ok=True)
    (ROOT/"outputs/hackathon/verification.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print("Isolated student run passed:", verification, flush=True)


if __name__ == "__main__":
    main()
