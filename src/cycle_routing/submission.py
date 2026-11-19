"""Export a solution together with its runtime and environment description."""
from importlib import metadata
from pathlib import Path
import json
import platform
import re
import shutil
import sys
import zipfile

import cloudpickle


def _requirements(extra):
    # Include imported libraries (also model classes), plus the routing runtime.
    names = {"cloudpickle", "geopandas", "gpxpy", "matplotlib", "networkx", "numpy",
             "osmnx", "pandas", "pyarrow", "pyproj", "requests", "scikit-learn",
             "scipy", "shapely", "tqdm", "folium", "ipython"}
    providers = metadata.packages_distributions()
    for module in list(sys.modules):
        names.update(providers.get(module.split(".")[0], []))
    names.update(extra)
    lines = set()
    for name in names:
        if re.sub(r"[-_.]+", "-", name).lower() in {"cicle-gpx", "pip", "setuptools", "wheel"}:
            continue
        lines.add(f"{name}=={metadata.version(name)}")
    return "\n".join(sorted(lines, key=str.lower)) + "\n"


def export_bundle(city, chosen, *, folder, files=(), extra_requirements=()):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    package = Path(__file__).resolve().parent
    project = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists()), package.parent.parent)
    sources = {}
    python_paths = {".", "src"}
    for name, module in list(sys.modules.items()):
        file = getattr(module, "__file__", None)
        if not file or name == "__main__" or name.startswith(("cycle_routing", "jury_routing")):
            continue
        path = Path(file).resolve()
        excluded = {"outputs", "_grading", ".agents", ".codex", "tests", "tools", "__pycache__"}
        if (path.suffix == ".py" and path.is_relative_to(project)
                and not excluded.intersection(path.relative_to(project).parts)
                and not path.is_relative_to(Path(sys.prefix).resolve())
                and path.is_file()):
            sources[path.relative_to(project)] = path
            import_root = path.parent
            for _ in range(len(name.split(".")) - (path.name != "__init__.py")):
                import_root = import_root.parent
            if import_root.is_relative_to(project):
                python_paths.add(import_root.relative_to(project).as_posix())
    # Import local modules from code/ on the receiving computer. Serializing them
    # by value captures globals such as __file__ from the student's machine and
    # breaks access to bundled resources after moving the archive. Notebook
    # functions remain serialized by value automatically by cloudpickle.
    payload = cloudpickle.dumps({"solution": chosen, "data_language": "ru_values_v1"})
    cloudpickle.loads(payload)
    (folder / "solution.pkl").write_bytes(payload)
    manifest = {"format_version": 2, "dataset_id": city.manifest["dataset_id"],
                "label": chosen.get("label"), "python": platform.python_version(),
                "python_paths": sorted(python_paths)}
    (folder / "submission.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (folder / "requirements.txt").write_text(_requirements(extra_requirements), encoding="utf-8")
    # Keep an exported folder usable by the same runner as its ZIP.
    for name in ["runtime", "code"]:
        target = (folder / name).resolve()
        if target.parent != folder.resolve():
            raise ValueError("Некорректная папка экспорта")
        if target.exists():
            shutil.rmtree(target)
    for path in package.rglob("*"):
        if path.is_file() and "__pycache__" not in path.parts and (path.suffix == ".py" or "assets" in path.parts):
            target = folder / "runtime/cycle_routing" / path.relative_to(package)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
    for item in files:
        path = Path(item).resolve()
        if not path.is_relative_to(project) or not path.is_file():
            raise ValueError(f"Дополнительный файл должен находиться в проекте: {item}")
        sources[path.relative_to(project)] = path
    for relative, path in sources.items():
        target = folder / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    (folder / "code").mkdir(exist_ok=True)
    archive = folder.with_name(folder.name + ".zip")
    members = [folder / name for name in ["solution.pkl", "submission.json", "requirements.txt"]]
    members.extend(p for name in ["runtime", "code"] for p in (folder / name).rglob("*") if p.is_file())
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(members):
            zipped.write(path, path.relative_to(folder))
    return folder
