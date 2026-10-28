"""Compare CPU deployment scores to frozen train predictions, never retune thresholds."""
import argparse
import csv
import json
import platform
import time

import numpy as np
import torch

from ml.deep_inference import load_text_classifier
from ml.experiment_metrics import write_csv, write_json
from ml.inference_config import load_config
from ml.model_experiments import load_toxic
from ml.toxic_baseline import ROOT, sha256
from ml.visual import ViolenceClassifier


def benchmark(run_id):
    directory = ROOT / "models/checkpoints/release-v3"
    public = ROOT / "reports" / run_id
    config = load_config(directory)
    decision = json.loads((public / "release_decision.json").read_text(encoding="utf-8"))
    rows, summaries = [], {}
    for task in ("toxic", "violence"):
        variant = "selected" if decision["decisions"][task]["accepted"] else "baseline"
        with (ROOT / "runs" / run_id / task / f"{variant}_predictions.csv").open(encoding="utf-8-sig") as stream:
            reference = {r["sample_id"]: r for r in csv.DictReader(stream) if r["split"] == "train"}
        start = time.perf_counter()
        model = load_text_classifier(directory / config[task]) if task == "toxic" else ViolenceClassifier(directory / config[task])
        load_seconds = time.perf_counter() - start
        if task == "toxic":
            data, _ = load_toxic()
            samples = list(zip(data["train"]["ids"], data["train"]["x"]))[:32]
        else:
            split = json.loads((ROOT / "data/splits/rlvs-v1.json").read_text(encoding="utf-8"))["rows"]
            samples = [(r["sha256"], ROOT / "data/raw/rlvs" / r["path"].replace("\\", "/")) for r in split if r["split"] == "train"][:10]
        measured = []
        for sid, value in samples:
            start = time.perf_counter()
            if task == "toxic":
                score = model.predict(value)["score"]
            elif getattr(model, "is_temporal", False):
                score = model.score_video(value)
            else:
                from ml.media import sample_frames
                score = model.score(sample_frames(value))
            elapsed = time.perf_counter() - start
            expected = float(reference[sid]["score"])
            item = dict(task=task, sample_id=sid, seconds=elapsed, score=score, reference_score=expected,
                        absolute_difference=abs(score-expected),
                        same_decision=int(score >= model.threshold) == int(reference[sid]["prediction"]))
            measured.append(item)
            print(f"{task} CPU {len(measured)}/{len(samples)}: {elapsed:.3f}s, score delta={abs(score-expected):.8f}", flush=True)
        rows.extend(measured)
        summaries[task] = dict(n=len(measured), load_seconds=load_seconds,
            first_seconds=measured[0]["seconds"], warm_median_seconds=float(np.median([r["seconds"] for r in measured[1:]])),
            warm_p95_seconds=float(np.percentile([r["seconds"] for r in measured[1:]], 95)),
            max_score_difference=max(r["absolute_difference"] for r in measured),
            all_decisions_match=all(r["same_decision"] for r in measured))
        del model
    passed = all(r["max_score_difference"] < 1e-4 and r["all_decisions_match"] for r in summaries.values())
    write_csv(ROOT / "runs" / run_id / "cpu_parity.csv", rows)
    write_json(public / "cpu_benchmark.json", dict(passed=passed, summaries=summaries,
        torch=torch.__version__, device="cpu", threads=4, processor=platform.processor(),
        manifest_sha256=sha256(ROOT / "models/artifacts-v3.json"),
        limit="Small deterministic train subset, one pass; classifier preprocessing+inference only, excludes ASR/API. Not target-video quality metrics."))
    if not passed:
        raise RuntimeError("CPU deployment differs from the evaluated checkpoint")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    benchmark(parser.parse_args().run_id)
