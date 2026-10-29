"""Package the public pack for students; assert that closed-zone and personal data stay outside.

    python tools/package_hackathon.py            # the pack the student notebook uses
"""
from pathlib import Path
import argparse
import json
import re
import zipfile

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT/"notebooks/hackathon_konstanz.ipynb"
FORBIDDEN = ("closed", "zones", "hackathon_private", "saint_petersburg", "cicle_gpx", "data/cities", "tracks", "old_packs")


def notebook_pack():
    """The pack the student notebook loads (its PACK = "..." line), so notebook and data always match."""
    return re.search(r'PACK = \\"([^\\"]+)\\"', NOTEBOOK.read_text(encoding="utf-8")).group(1)


def package(destination, pack=None):
    pack = pack or notebook_pack()
    data = ROOT/"notebooks/data/hackathon"/pack
    manifest = json.loads((data/"manifest.json").read_text(encoding="utf-8"))
    paths = [ROOT/"pyproject.toml", ROOT/"HACKATHON.md", ROOT/"notebooks/hackathon_konstanz.ipynb", data/"manifest.json"]
    paths += [p for p in (ROOT/"src/cycle_routing").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths += [data/name for name in manifest["files"]]  # only what the manifest lists: no caches, no stale files
    names = [p.relative_to(ROOT).as_posix() for p in paths]
    if any(word in name for name in names for word in FORBIDDEN):
        raise ValueError("Unexpected private data in student archive")
    # The organizer's executed copy may show the personal SPb block; students get a notebook without outputs.
    notebook = nbformat.read(ROOT/"notebooks/hackathon_konstanz.ipynb", as_version=4)
    # The local SPb example requires personal data; the student pack contains Konstanz only.
    notebook.cells = [cell for cell in notebook.cells if "local-spb" not in cell.metadata.get("tags", [])]
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs, cell.execution_count = [], None
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, name in zip(paths, names):
            if name.endswith(".ipynb"):
                archive.writestr(name, nbformat.writes(notebook))
            else:
                archive.write(path, name)
        archive.writestr("README.md", "# Веломаршрут: студенческий набор\n\nИнструкция: [HACKATHON.md](HACKATHON.md).\n"
                         "Начните с notebooks/hackathon_konstanz.ipynb.\n"
                         "Данные Констанца подготовлены; закрытая часть города в этот набор не входит.\n")
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Student archive integrity check failed")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pack", help="default: the PACK of the student notebook")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/hackathon_student.zip")
    args = parser.parse_args()
    result = package(args.output, args.pack)
    print(f"Student archive: {result} ({result.stat().st_size/1024**2:.1f} MB)")
