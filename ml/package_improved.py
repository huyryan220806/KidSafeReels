"""Stage a measured CPU v2 bundle; activate only after integration checks pass."""
import argparse
import json
from pathlib import Path
import shutil
import zipfile

from ml.experiment_metrics import write_json
from ml.inference_config import DEFAULT
from ml.release_verify import verify_artifacts
from ml.toxic_baseline import ROOT, sha256


def test_pair(report):
    return {r["variant"]: r for r in report["metrics"] if r["split"] == "test"}


def stage(run_id):
    public = ROOT / "reports" / run_id
    source = ROOT / "models/checkpoints"
    destination = source / "release-v2"
    if (destination / "bundle-manifest.json").exists():
        raise ValueError("Staged release already exists; verify it rather than overwrite")
    selection = {task: json.loads((public / task / "selection.json").read_text(encoding="utf-8"))
                 for task in ("toxic", "violence", "asr")}
    evaluations = {task: json.loads((public / task / "evaluation.json").read_text(encoding="utf-8"))
                   for task in ("toxic", "violence", "asr")}
    config, decisions = dict(DEFAULT), {}
    config["version"] = "kidsafe-cpu-v2"
    files = {"resnet18-imagenet1k-v1.pth": source / "resnet18-imagenet1k-v1.pth"}
    for task in ("toxic", "violence"):
        pair = test_pair(evaluations[task])
        baseline, candidate = pair["baseline"], pair["selected"]
        accepted = candidate["recall"] >= baseline["recall"] and candidate["macro_f1"] > baseline["macro_f1"]
        if task == "toxic":
            accepted &= candidate["precision"] >= baseline["precision"]
        artifact = ROOT / "runs" / run_id / task / "selected.joblib"
        if sha256(artifact) != selection[task]["artifact_sha256"]:
            raise ValueError("Frozen candidate checksum mismatch")
        if accepted:
            config[task] = f"{task}-cpu-v2.joblib"
            files[config[task]] = artifact
        else:
            files[config[task]] = source / config[task]
        decisions[task] = dict(accepted=bool(accepted), active_file=config[task],
                              baseline_test=baseline, candidate_test=candidate,
                              rule="require no recall regression and improved macro-F1; Toxic also requires no precision regression")
    asr = evaluations["asr"]
    base = next(r for r in asr["metrics"] if r["config"] == "base-greedy" and r["split"] == "test")
    candidate = next(r for r in asr["metrics"] if r["config"] == asr["selected"] and r["split"] == "test")
    asr_accepted = candidate["wer"] < base["wer"] and candidate["cer"] < base["cer"] and candidate["real_time_factor"] <= 2.5
    if asr_accepted:
        if asr["selected"] not in ("base-beam5", "small-beam5"):
            raise ValueError("Unsupported ASR choice")
        config["asr_directory"] = "faster-whisper-small" if asr["selected"] == "small-beam5" else "faster-whisper-base"
        config["asr_version"] = f"whisper-{asr['selected']}"
        config["asr_options"] = {"beam_size": 5, "temperature": 0}
    decisions["asr"] = dict(accepted=asr_accepted, baseline_test=base, candidate_test=candidate)
    for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt"):
        relative = f"{config['asr_directory']}/{name}"
        files[relative] = source / relative
    asr_source = (source / config["asr_directory"] / "provenance.json"
                  if config["asr_directory"] == "faster-whisper-small" else source / "asr_provenance.json")
    provenance_name = (f"{config['asr_directory']}/provenance.json"
                       if config["asr_directory"] == "faster-whisper-small" else "asr_provenance.json")
    files[provenance_name] = asr_source
    destination.mkdir(parents=True, exist_ok=True)
    for name, original in files.items():
        target = destination / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
    write_json(destination / "inference-config.json", config)
    names = sorted([*files, "inference-config.json"])
    manifest = dict(version=config["version"], run_id=run_id,
        files=[dict(path=n, bytes=(destination/n).stat().st_size, sha256=sha256(destination/n)) for n in names],
        asr=json.loads(asr_source.read_text(encoding="utf-8")),
        toxic=dict(report=f"reports/{run_id}/toxic/evaluation.json", threshold=decisions['toxic']['baseline_test']['threshold'] if not decisions['toxic']['accepted'] else selection['toxic']['selected']['threshold']),
        violence=dict(report=f"reports/{run_id}/violence/evaluation.json", threshold=selection['violence']['selected']['threshold'] if decisions['violence']['accepted'] else decisions['violence']['baseline_test']['threshold']))
    for task in ("toxic", "violence"):
        manifest[task]["active_variant"] = "selected" if decisions[task]["accepted"] else "baseline"
    write_json(destination / "bundle-manifest.json", manifest)
    write_json(ROOT / "models/artifacts-v2.json", manifest)
    write_json(public / "release_decision.json", dict(config=config, decisions=decisions, status="staged_pending_integration_checks"))
    verify_artifacts(destination, manifest)
    archive = source / "bundle/kidsafe-cpu-v2.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=1) as zipped:
        for name in [*names, "bundle-manifest.json"]:
            zipped.write(destination / name, name)
    (archive.parent / "kidsafe-cpu-v2.sha256.txt").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="utf-8")
    print(json.dumps(dict(directory=str(destination), archive=str(archive), config=config), indent=2))


def activate(run_id):
    source = ROOT / "models/checkpoints/release-v2"
    destination = ROOT / "models/checkpoints"
    manifest = json.loads((source / "bundle-manifest.json").read_text(encoding="utf-8"))
    if manifest["run_id"] != run_id:
        raise ValueError("Staged bundle belongs to another run")
    validation = ROOT / "reports" / run_id / "integration_checks.json"
    if not validation.is_file() or not json.loads(validation.read_text(encoding="utf-8")).get("passed"):
        raise ValueError("Record passing integration checks before activation")
    verify_artifacts(source, manifest)
    old_manifest = ROOT / "models/artifacts-v1.json"
    if not old_manifest.exists():
        shutil.copy2(ROOT / "models/artifacts.json", old_manifest)
    # Copy configuration last; existing model weights have separate versioned names.
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
    decision["status"] = "active_after_integration_checks"
    write_json(report, decision)
    print(f"Activated {manifest['version']}; restart API to load the new bundle.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stage", "activate"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    {"stage": stage, "activate": activate}[args.command](args.run_id)
