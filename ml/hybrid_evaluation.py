"""Descriptive ablation on the previously reported ViCTSD benchmark, no tuning.

Reuse frozen PhoBERT scores. Written-text rules do not evaluate ASR quality.
Keep row-level results private; publish only aggregate metrics and a plot.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                             log_loss, precision_recall_fscore_support)

from ml.experiment_metrics import clean_svg, write_csv, write_json
from ml.model_experiments import ROOT, load_toxic
from ml.toxic_rules import LEXICON, ToxicRules


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def decision_metrics(labels, decisions):
    y, pred = np.asarray(labels), np.asarray(decisions)
    if y.ndim != 1 or y.shape != pred.shape or not len(y):
        raise ValueError("Expected equally sized nonempty vectors")
    if not set(y).issubset({0, 1}) or not set(pred).issubset({0, 1}):
        raise ValueError("Expected binary labels and decisions")
    p, r, f, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return dict(n=len(y), negative=int(support[0]), positive=int(support[1]),
                accuracy=float(accuracy_score(y, pred)),
                balanced_accuracy=float(balanced_accuracy_score(y, pred)) if len(set(y)) == 2 else None,
                precision=float(p[1]) if tp + fp else None,
                recall=float(r[1]) if tp + fn else None, f1=float(f[1]) if tp + fp + fn else None,
                macro_f1=float(np.mean(f)), false_positive_rate=float(fp / (fp + tn)) if fp + tn else None,
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def align_cache(rows, split, values, threshold):
    selected = [row for row in rows if row["split"] == split]
    by_id = {row["sample_id"]: row for row in selected}
    if len(by_id) != len(selected) or set(by_id) != set(values["ids"]):
        raise ValueError("Cached IDs missing, extra or duplicated")
    aligned = [by_id[sid] for sid in values["ids"]]
    scores = np.asarray([float(row["score"]) for row in aligned])
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Invalid cached scores")
    for i, row in enumerate(aligned):
        if (int(row["label"]) != values["y"][i] or row["group_id"] != values["groups"][i]
                or int(row["prediction"]) != int(scores[i] >= threshold)):
            raise ValueError("Cached label, group or operating threshold mismatch")
    return scores


def evaluate(run_id):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_id):
        raise ValueError("Use a lowercase run ID")
    out, public = ROOT / "runs" / run_id, ROOT / "reports" / run_id
    if (out / "protocol.json").exists() or public.exists():
        raise ValueError("Existing evaluation: use a new run ID; do not overwrite")
    cache = ROOT / "runs/model-improvement-2026-10-28/toxic/selected_predictions.csv"
    config_path = ROOT / "models/checkpoints/toxic-phobert-v3/kidsafe-config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    rules = ToxicRules()
    threshold = config["threshold"]
    # Save the frozen protocol BEFORE opening labels/text or computing metrics.
    protocol = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                    evaluation="descriptive_reused_benchmark_not_new_independent_test",
                    dataset="ViCTSD", splits=["valid", "test"], training_performed=False,
                    model_version=config["version"], model_threshold=threshold,
                    lexicon_version=rules.version, lexicon_sha256=rules.sha256,
                    cached_predictions_sha256=sha(cache), model_config_sha256=sha(config_path),
                    model_manifest_sha256=sha(ROOT / "models/artifacts.json"),
                    evaluator_sha256=sha(__file__), rules_code_sha256=sha(Path(__file__).with_name("toxic_rules.py")),
                    methods=["model_only", "rule_only", "hybrid"],
                    rule_policy="Any exact flag rule; review abbreviations do not flag",
                    hybrid_policy="model_prediction OR rule_prediction; no fused probability",
                    audio_scope="Written text only: does not evaluate ASR or the audio-quality gate",
                    tuning="None. Freeze seed lexicon from project policy and known development clips; do not revise from this test.",
                    test_exposure="Benchmark already reported for previous models; not an unseen target-video evaluation",
                    losses="Only model_only has real log loss. Rules/hybrid have no probabilistic output.")
    write_json(out / "protocol.json", protocol)
    (out / "frozen-lexicon.json").write_bytes(LEXICON.read_bytes())
    data, audit = load_toxic()
    write_json(out / "dataset-audit.json", audit)
    with cache.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    metrics, private = [], []
    for split in protocol["splits"]:
        values = data[split]
        scores = align_cache(rows, split, values, threshold)
        matches = [rules.match(text) for text in values["x"]]
        lexical = np.array([any(m["action"] == "flag" for m in found) for found in matches])
        predictions = dict(model_only=scores >= threshold, rule_only=lexical,
                           hybrid=(scores >= threshold) | lexical)
        for method, pred in predictions.items():
            metric = dict(split=split, method=method, **decision_metrics(values["y"], pred))
            metric["log_loss"] = float(log_loss(values["y"], scores, labels=[0, 1])) if method == "model_only" else None
            metrics.append(metric)
        for i, sid in enumerate(values["ids"]):
            private.append(dict(sample_id=sid, split=split, label=int(values["y"][i]),
                                model_score=float(scores[i]),
                                **{k: int(v[i]) for k, v in predictions.items()},
                                rule_ids=";".join(m["id"] for m in matches[i])))
    write_csv(out / "predictions.csv", private)
    write_csv(public / "metrics.csv", metrics)
    write_json(public / "protocol.json", protocol)
    write_json(public / "metrics.json", metrics)
    plot(public, [row for row in metrics if row["split"] == "test"])
    return metrics


def plot(directory, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["method"].replace("_", " ") for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    x, width = np.arange(len(rows)), .23
    for offset, key, label in ((-1, "precision", "Precision"), (0, "recall", "Recall"), (1, "f1", "F1")):
        bars = axes[0].bar(x + offset * width, [r[key] for r in rows], width, label=label)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=8)
    axes[0].set(xticks=x, xticklabels=names, ylim=(0, 1.12), title="Toxic class (109 / 988)")
    axes[0].legend(loc="upper left", ncols=3, fontsize=8)
    for offset, key, label in ((-.18, "fp", "False positives"), (.18, "fn", "False negatives")):
        bars = axes[1].bar(x + offset, [r[key] for r in rows], .36, label=label)
        axes[1].bar_label(bars)
    axes[1].set(xticks=x, xticklabels=names, ylim=(0, max(r["fn"] for r in rows) * 1.25), title="Error counts")
    axes[1].legend(fontsize=8)
    fig.suptitle("ViCTSD test — reused benchmark, frozen rules, written text only")
    for suffix in ("png", "svg"):
        path = directory / f"ablation.{suffix}"
        fig.savefig(path, dpi=180)
        if suffix == "svg":
            clean_svg(path)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run_id), indent=2))
