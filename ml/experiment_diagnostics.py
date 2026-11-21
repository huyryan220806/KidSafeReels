"""Export existing prediction errors for review; no inference or label changes."""
import argparse
import csv
import hashlib
import json

from ml.experiment_metrics import write_csv, write_json
from ml.model_experiments import locations, load_toxic
from ml.toxic_baseline import ROOT


def export(run_id, task):
    private, public = locations(run_id, task)
    if not (public / "evaluation.json").exists():
        raise ValueError("Complete the frozen evaluation before error review")
    if task == "toxic":
        data, _ = load_toxic()
        source = {hashlib.sha256(text.encode()).hexdigest(): {"text": text}
                  for d in data.values() for text in d["x"]}
    else:
        manifest = json.loads((ROOT / "data/processed/rlvs/manifest.json").read_text(encoding="utf-8"))
        source = {r["sha256"]: {"video_path": str(ROOT / "data/raw/rlvs" / r["path"].replace("\\", "/")),
                  "duration_seconds": r["duration"]} for r in manifest["rows"]}
    errors, grouped = [], []
    for variant in ("baseline", "selected"):
        with (private / f"{variant}_predictions.csv").open(encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            if row["split"] == "train" or row["outcome"] not in ("FP", "FN"):
                continue
            details = source[row["sample_id"]].copy()
            if task == "toxic" and details["text"].lstrip().startswith(("=", "+", "-", "@")):
                details["text"] = "'" + details["text"]
            errors.append(dict(variant=variant, **row, **details, reviewer_note=""))
        for split in ("valid", "test"):
            part = [r for r in rows if r["split"] == split]
            for outcome in ("TN", "FP", "FN", "TP"):
                selected = [r for r in part if r["outcome"] == outcome]
                grouped.append(dict(variant=variant, split=split, outcome=outcome, count=len(selected)))
    if errors:
        write_csv(private / "error_review.csv", errors)
    write_csv(public / "error_counts.csv", grouped)
    write_json(public / "error_review.json", dict(private_file=f"runs/{run_id}/{task}/error_review.csv",
                rows=len(errors), labels="existing benchmark annotations; no user-video or pilot labels",
                warning="Review is descriptive after frozen evaluation. Do not tune thresholds against test errors."))
    print(f"{task}: {len(errors)} validation/test error rows exported privately")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--task", required=True, choices=["toxic", "violence"])
    args = parser.parse_args()
    export(args.run_id, args.task)
