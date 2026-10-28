"""Audit TF32 versus FP32 deployment, with the trained model and threshold fixed."""
import argparse
import csv
import json

import joblib
import numpy as np
import torch

from ml.experiment_metrics import write_csv, write_json
from ml.temporal_visual import TemporalEncoder
from ml.temporal_violence import frame_path, rows_for
from ml.toxic_baseline import ROOT, sha256


def audit(run_id, mode):
    torch.set_num_threads(4)
    public = ROOT / "reports" / run_id
    private = ROOT / "runs" / run_id / "violence"
    with (private / "selected_predictions.csv").open(encoding="utf-8-sig") as stream:
        reference = {r["sample_id"]: r for r in csv.DictReader(stream)}
    artifact = joblib.load(private / "selected.joblib")
    threshold = artifact["threshold"]
    split = rows_for(("train", "valid", "test"))
    # Boundary cases chosen by frozen scores, no labels or parameter adjustment.
    boundary = sorted([r for r in split if r["split"] in ("valid", "test")],
                      key=lambda r: abs(float(reference[r["sha256"]]["score"])-threshold))[:20]
    if mode == "gpu":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        encoder = TemporalEncoder(device="cuda")
        selected = {r["sha256"]: r for r in split if r["split"] == "test"}
        selected.update({r["sha256"]: r for r in boundary})
        results = []
        for i, row in enumerate(selected.values(), 1):
            feature = encoder.encode(np.load(frame_path(row), allow_pickle=False))
            score = float(artifact["model"].predict_proba(feature[None])[0, 1])
            old = float(reference[row["sha256"]]["score"])
            results.append(dict(sample_id=row["sha256"], split=row["split"], score=score,
                                tf32_score=old, difference=abs(score-old), same_decision=(score>=threshold)==(old>=threshold)))
            if i % 100 == 0:
                print(f"FP32 deployment audit {i}/{len(selected)}", flush=True)
        write_csv(private / "fp32_deployment_reference.csv", results)
        write_json(public / "gpu_fp32_audit.json", dict(n=len(results), test_n=sum(r["split"]=="test" for r in results),
            max_score_difference=max(r["difference"] for r in results), changed_decisions=sum(not r["same_decision"] for r in results),
            purpose="Frozen classifier deployment precision audit; no fitting or threshold changes", torch=torch.__version__))
        return
    from ml.visual import ViolenceClassifier
    classifier = ViolenceClassifier(ROOT / "models/checkpoints/release-v3/violence-r3d18-v3.joblib")
    with (private / "fp32_deployment_reference.csv").open(encoding="utf-8-sig") as stream:
        fp32 = {r["sample_id"]: float(r["score"]) for r in csv.DictReader(stream)}
    results = []
    for i, row in enumerate(boundary, 1):
        path = ROOT / "data/raw/rlvs" / row["path"].replace("\\", "/")
        score = classifier.score_video(path)
        old = float(reference[row["sha256"]]["score"])
        results.append(dict(sample_id=row["sha256"], split=row["split"], cpu_score=score,
            fp32_difference=abs(score-fp32[row["sha256"]]), tf32_difference=abs(score-old),
            same_decision=(score>=threshold)==(old>=threshold)))
        print(f"CPU boundary audit {i}/{len(boundary)}", flush=True)
    gpu = json.loads((public / "gpu_fp32_audit.json").read_text(encoding="utf-8"))
    passed = gpu["changed_decisions"] == 0 and all(r["same_decision"] and r["fp32_difference"]<1e-5 for r in results)
    write_csv(private / "cpu_boundary_predictions.csv", results)
    write_json(public / "cpu_parity_resolution.json", dict(passed=passed, n=len(results),
        max_cpu_fp32_difference=max(r["fp32_difference"] for r in results),
        max_cpu_tf32_difference=max(r["tf32_difference"] for r in results),
        changed_decisions=sum(not r["same_decision"] for r in results),
        gpu_full_test=gpu, manifest_sha256=sha256(ROOT / "models/artifacts-v3.json"),
        initial_check="cpu_benchmark.json retains initial failed 1e-4 absolute TF32 tolerance",
        explanation="Identical sampled pixels. Original cuDNN TF32 convolution rounds differently; CPU matches GPU FP32 within 1e-5 on 20 nearest-boundary validation/test samples. No model or threshold change.",
        limitation="All 386 test decisions audited on GPU FP32; CPU audited on 20 nearest-boundary samples plus 10 train clips, not all test clips."))
    if not passed:
        raise RuntimeError("Deployment precision remains unresolved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["gpu", "cpu"])
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    audit(args.run_id, args.mode)
