"""Package only the public Konstanz lab; assert that private and personal data stay outside."""
from pathlib import Path
import argparse
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def package(destination):
    data = ROOT / "notebooks/data/hackathon/konstanz"
    if not (data / "manifest.json").exists():
        raise FileNotFoundError("Run tools/prepare_hackathon_data.py first")
    paths = [ROOT/"pyproject.toml", ROOT/"HACKATHON.md", ROOT/"notebooks/hackathon_konstanz.ipynb"]
    paths += [p for p in (ROOT/"src/cycle_routing").rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    paths += [p for p in data.iterdir() if p.is_file()]
    names = [p.relative_to(ROOT).as_posix() for p in paths]
    forbidden = ("test_labels", "labels_test", "hackathon_private", "saint_petersburg", "cicle_gpx", "data/cities")
    if any(word in name for name in names for word in forbidden):
        raise ValueError("Unexpected private data in student archive")
    # Route caches are deliberately omitted: participants can recompute paths from readable source.
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path, name in zip(paths, names):
            archive.write(path, name)
        archive.writestr("README.md", "# Веломаршрут: студенческий набор\n\nИнструкция: [HACKATHON.md](HACKATHON.md).\n"
                         "Начните с notebooks/hackathon_konstanz.ipynb.\n"
                         "Данные Констанца подготовлены, закрытые test-метки в этот набор не входят.\n")
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("Student archive integrity check failed")
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/hackathon_student.zip")
    args = parser.parse_args()
    result = package(args.output)
    print(f"Student archive: {result} ({result.stat().st_size/1024**2:.1f} MB)")
