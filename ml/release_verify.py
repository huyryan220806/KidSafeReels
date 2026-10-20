"""Verify pinned local model artifacts; optionally install the team's checked ZIP."""
import argparse
import json
from pathlib import Path
import zipfile

from ml.pilot_data import ROOT, read_json, sha


def verify_artifacts(directory=None, manifest=None):
    directory = Path(directory or ROOT / "models/checkpoints")
    manifest = manifest or read_json(ROOT / "models/artifacts.json")
    problems = []
    if (directory / "inference-config.json").exists() and not any(e["path"] == "inference-config.json" for e in manifest["files"]):
        problems.append("unverified inference-config.json not included in this bundle")
    for entry in manifest["files"]:
        path = directory / entry["path"]
        if not path.resolve().is_relative_to(directory.resolve()):
            raise ValueError("Artifact path outside model directory")
        if not path.is_file():
            problems.append(f"missing: {entry['path']}")
        elif path.stat().st_size != entry["bytes"] or sha(path) != entry["sha256"]:
            problems.append(f"checksum mismatch: {entry['path']}")
    if problems:
        raise ValueError("; ".join(problems))
    return {"version": manifest["version"], "verified_files": len(manifest["files"])}


def install_bundle(archive, directory=None, manifest=None):
    import hashlib
    directory = Path(directory or ROOT / "models/checkpoints")
    manifest = manifest or read_json(ROOT / "models/artifacts.json")
    expected = {e["path"]: e for e in manifest["files"]}
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        if len(names) != len(set(names)) or set(names) != set(expected) | {"bundle-manifest.json"}:
            raise ValueError("Unexpected/missing/duplicate bundle entries")
        # Check against the trusted repository manifest, never trust a manifest shipped in the ZIP.
        for name, entry in expected.items():
            if not (directory / name).resolve().is_relative_to(directory.resolve()):
                raise ValueError("Invalid artifact path")
            if bundle.getinfo(name).file_size != entry["bytes"]:
                raise ValueError("Bundle size mismatch")
            with bundle.open(name) as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != entry["sha256"]:
                raise ValueError("Bundle checksum mismatch")
        for name in expected:
            path = directory / name
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(path.suffix + ".installing")
            with bundle.open(name) as source, partial.open("wb") as target:
                import shutil
                shutil.copyfileobj(source, target)
            partial.replace(path)
        # Restoring a v1 bundle must also restore its default inference choices.
        config = directory / "inference-config.json"
        if "inference-config.json" not in expected and config.is_file():
            config.unlink()
    return verify_artifacts(directory, manifest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--install-bundle", type=Path)
    parser.add_argument("--manifest", type=Path, help="Trusted repository manifest; defaults to the active release")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args()
    manifest = read_json(args.manifest) if args.manifest else None
    print(json.dumps(install_bundle(args.install_bundle, args.model_dir, manifest) if args.install_bundle
                     else verify_artifacts(args.model_dir, manifest)))


if __name__ == "__main__":
    main()
