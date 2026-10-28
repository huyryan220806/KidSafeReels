"""Stage measured classifiers as CPU v3, preserving v2 for rollback."""
import argparse
import json
import shutil
import zipfile

from ml.experiment_metrics import write_json
from ml.package_improved import test_pair
from ml.release_verify import verify_artifacts
from ml.toxic_baseline import ROOT, sha256


def passes_metrics(task, baseline, candidate, external=None, eligible=True):
    keys = ("f1", "recall", "precision") if task == "toxic" else ("f1", "recall")
    accepted = (eligible and candidate["macro_f1"] > baseline["macro_f1"]
                and all(candidate[k] >= baseline[k] for k in keys))
    if task == "violence":
        accepted = accepted and external is not None and all(
            external["selected"][k] >= external["baseline"][k] for k in ("macro_f1", "recall"))
    return bool(accepted)


def stage(run_id):
    public = ROOT / "reports" / run_id
    source = ROOT / "models/checkpoints"
    destination = source / "release-v3"
    if (destination / "bundle-manifest.json").exists():
        raise ValueError("Release already staged; verify rather than overwrite")
    old = json.loads((ROOT / "models/artifacts-v2.json").read_text(encoding="utf-8"))
    verify_artifacts(source, old)
    config = json.loads((source / "inference-config.json").read_text(encoding="utf-8"))
    files = {e["path"]: source / e["path"] for e in old["files"] if e["path"] != "inference-config.json"}
    decisions = {}
    external_report = json.loads((public / "airtlab_evaluation.json").read_text(encoding="utf-8"))
    external = {r["variant"]: r for r in external_report["metrics"]}
    for task in ("toxic", "violence"):
        selection_path = public / task / "selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        evaluation = json.loads((public / task / "evaluation.json").read_text(encoding="utf-8"))
        if evaluation["selection_sha256"] != sha256(selection_path):
            raise ValueError("Evaluation does not match the frozen selection")
        if task == "violence" and external_report["selection_sha256"] != sha256(selection_path):
            raise ValueError("External evaluation selection mismatch")
        pair = test_pair(evaluation)
        eligible = selection["selected"].get("eligible", True) if task == "toxic" else selection["configuration"]["features"] != "baseline"
        accepted = passes_metrics(task, pair["baseline"], pair["selected"], external, eligible)
        decisions[task] = dict(accepted=accepted, baseline_test=pair["baseline"], candidate_test=pair["selected"])
        if not accepted:
            continue
        files.pop(config[task])
        private = ROOT / "runs" / run_id / task
        if task == "toxic":
            for entry in selection["files"]:
                original = private / "selected" / entry["path"]
                if sha256(original) != entry["sha256"]:
                    raise ValueError("Frozen PhoBERT checksum mismatch")
                files[f"toxic-phobert-v3/{entry['path']}"] = original
            config[task] = "toxic-phobert-v3/kidsafe-config.json"
        else:
            original = private / "selected.joblib"
            if sha256(original) != selection["artifact_sha256"]:
                raise ValueError("Frozen Violence checksum mismatch")
            backbone = source / "pretrained/r3d18-kinetics400.pth"
            if sha256(backbone) != selection["backbone_provenance"]["sha256"]:
                raise ValueError("Frozen R3D checksum mismatch")
            config[task] = "violence-r3d18-v3.joblib"
            files[config[task]] = original
            files[backbone.name] = backbone
    decision = dict(decisions=decisions, external=external, release_gate_sha256=sha256(public / "release_gate.json"))
    if not any(d["accepted"] for d in decisions.values()):
        write_json(public / "release_decision.json", dict(decision, config=config, status="retain_cpu_v2"))
        print("No classifier passes the release gate; retaining v2.")
        return
    config["version"] = "kidsafe-cpu-v3"
    destination.mkdir(parents=True, exist_ok=True)
    for name, original in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    write_json(destination / "inference-config.json", config)
    names = sorted([*files, "inference-config.json"])
    manifest = dict(version=config["version"], run_id=run_id,
        files=[dict(path=n, bytes=(destination/n).stat().st_size, sha256=sha256(destination/n)) for n in names],
        asr=old["asr"], classifiers=decisions)
    write_json(destination / "bundle-manifest.json", manifest)
    write_json(ROOT / "models/artifacts-v3.json", manifest)
    verify_artifacts(destination, manifest)
    write_json(public / "release_decision.json", dict(decision, config=config, status="staged_pending_integration_checks"))
    archive = source / "bundle/kidsafe-cpu-v3.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
        for name in [*names, "bundle-manifest.json"]:
            zipped.write(destination / name, name)
    write_json(public / "bundle.json", dict(filename=archive.name, bytes=archive.stat().st_size, sha256=sha256(archive)))
    (archive.parent / "kidsafe-cpu-v3.sha256.txt").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    print(json.dumps(dict(directory=str(destination), archive=str(archive), config=config), indent=2))


def activate(run_id):
    source = ROOT / "models/checkpoints/release-v3"
    destination = ROOT / "models/checkpoints"
    manifest_path = ROOT / "models/artifacts-v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validation = json.loads((ROOT / "reports" / run_id / "integration_checks.json").read_text(encoding="utf-8"))
    if manifest["run_id"] != run_id or not validation.get("passed") or validation.get("manifest_sha256") != sha256(manifest_path):
        raise ValueError("Require passing integration for this exact bundle")
    verify_artifacts(source, manifest)
    for entry in sorted(manifest["files"], key=lambda e: e["path"] == "inference-config.json"):
        target = destination / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".installing")
        shutil.copy2(source / entry["path"], temporary)
        temporary.replace(target)
    write_json(ROOT / "models/artifacts.json", manifest)
    verify_artifacts(destination, manifest)
    report = ROOT / "reports" / run_id / "release_decision.json"
    decision = json.loads(report.read_text(encoding="utf-8"))
    write_json(report, dict(decision, status="active_after_integration_checks"))
    print("Activated kidsafe-cpu-v3. Restart API to load it.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stage", "activate"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    {"stage": stage, "activate": activate}[args.command](args.run_id)
