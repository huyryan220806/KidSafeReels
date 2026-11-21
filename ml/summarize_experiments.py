"""Combine recorded classifier results without fitting or re-evaluating test data."""
import argparse
import json

from ml.experiment_metrics import write_csv
from ml.model_experiments import locations


def summarize(run_id):
    rows, confusion = [], []
    for task in ("toxic", "violence"):
        _, public = locations(run_id, task)
        report = json.loads((public / "evaluation.json").read_text(encoding="utf-8"))
        rows.extend(report["metrics"])
        for row in report["metrics"]:
            identity = {key: row[key] for key in ("run_id", "task", "dataset", "variant", "split", "threshold", "n")}
            confusion.extend(dict(**identity, cell=cell, count=row[cell]) for cell in ("tn", "fp", "fn", "tp"))
    destination = public.parent
    write_csv(destination / "metrics_by_split.csv", rows)
    write_csv(destination / "comparison.csv", [row for row in rows if row["split"] == "test"])
    write_csv(destination / "confusion_matrix.csv", confusion)
    print(f"Combined {len(rows)} recorded metric rows in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    summarize(parser.parse_args().run_id)
