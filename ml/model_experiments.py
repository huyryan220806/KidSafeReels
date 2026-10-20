"""CPU experiments: select with validation, then explicitly evaluate frozen selection."""
import argparse
from datetime import datetime
import hashlib
import json
import platform
from pathlib import Path
import time

import joblib
import numpy as np
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit, GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.experiment_metrics import (binary_metrics, choose_operating_point, clean_svg, plot_results,
                                   threshold_rows, write_csv, write_json)
from ml.toxic_baseline import ROOT, DATA, clean_splits, read_split, sha256

SPLITS = ("train", "valid", "test")
FRAME_COUNT = 24


def locations(run_id, task):
    if not run_id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in run_id):
        raise ValueError("Use a lowercase run ID")
    return ROOT / "runs" / run_id / task, ROOT / "reports" / run_id / task


def load_toxic():
    provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))
    for name, info in provenance["files"].items():
        if sha256(DATA / name) != info["sha256"]:
            raise ValueError("ViCTSD checksum mismatch")
    cleaned, audit = clean_splits({s: read_split(DATA / f"ViCTSD_{s}.csv") for s in SPLITS})
    data = {}
    for split, values in cleaned.items():
        x, y = zip(*values)
        ids = [hashlib.sha256(text.encode()).hexdigest() for text in x]
        data[split] = dict(x=list(x), y=np.array(y), ids=ids, groups=ids)
    return data, {"dataset": "ViCTSD", "provenance": provenance, "audit": audit}


def load_visual(features="baseline", needed=SPLITS):
    from ml.visual import BACKBONE, SAMPLING_VERSION, VisualEncoder, aggregate_features
    from ml.media import sample_frames
    path = ROOT / "data/splits/rlvs-v1.json"
    rows = json.loads(path.read_text(encoding="utf-8"))["rows"]
    groups = {s: {r["group_id"] for r in rows if r["split"] == s} for s in SPLITS}
    if any(groups[a] & groups[b] for a, b in (("train", "valid"), ("train", "test"), ("valid", "test"))):
        raise ValueError("Source group leakage")
    backbone_hash = sha256(BACKBONE)
    encoder = None
    data = {}
    for split in needed:
        selected = [r for r in rows if r["split"] == split]
        matrix = []
        started = time.perf_counter()
        for i, row in enumerate(selected):
            if features == "baseline":
                key = hashlib.sha256((row["sha256"] + backbone_hash + SAMPLING_VERSION).encode()).hexdigest()
                feature = np.load(ROOT / "data/processed/rlvs" / f"{key}.npy", allow_pickle=False)
            else:
                key = hashlib.sha256((row["sha256"] + backbone_hash + f"uniform{FRAME_COUNT}-frame-embeddings-v1").encode()).hexdigest()
                cache = ROOT / "data/processed/rlvs-dense" / f"{key}.npy"
                if cache.exists():
                    embeddings = np.load(cache, allow_pickle=False)
                else:
                    video = ROOT / "data/raw/rlvs" / row["path"].replace("\\", "/")
                    if sha256(video) != row["sha256"]:
                        raise ValueError(f"Video checksum mismatch: {row['path']}")
                    if encoder is None:
                        encoder = VisualEncoder()
                    embeddings = encoder.encode_frames(sample_frames(video, FRAME_COUNT))
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    np.save(cache, embeddings)
                feature = aggregate_features(embeddings, features)
            if not np.isfinite(feature).all():
                raise ValueError("Nonfinite feature")
            matrix.append(feature)
            if (i + 1) % 100 == 0:
                print(f"{features} {split}: {i + 1}/{len(selected)} ({time.perf_counter()-started:.1f}s)", flush=True)
        data[split] = dict(x=np.array(matrix), y=np.array([r["label"] for r in selected]),
                           ids=[r["sha256"] for r in selected], groups=[r["group_id"] for r in selected])
    canonical = hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return data, {"dataset": "RLVS", "manifest_sha256": sha256(path), "rows_canonical_sha256": canonical,
                  "backbone_sha256": backbone_hash,
                  "features": features, "frame_count": 8 if features == "baseline" else FRAME_COUNT}


def toxic_model(kind, c, balanced):
    char = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                           max_features=100000, sublinear_tf=True)
    vectorizer = char if kind == "char" else FeatureUnion([
        ("char", char),
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=60000,
                                 sublinear_tf=True, token_pattern=r"(?u)\b\w+\b")),
    ])
    return make_pipeline(vectorizer, LogisticRegression(C=c, class_weight="balanced" if balanced else None,
                                                        solver="liblinear", max_iter=1000, random_state=42))


def baseline_path(task):
    return ROOT / "models/checkpoints" / ("toxic-tfidf-v1.joblib" if task == "toxic" else "violence-resnet18-v1.joblib")


def select(run_id, task):
    private, public = locations(run_id, task)
    if (public / "selection.json").exists():
        raise ValueError("Selection already frozen; use evaluate or a new run ID")
    private.mkdir(parents=True, exist_ok=True)
    baseline = joblib.load(baseline_path(task))
    policy = {
        "run_id": run_id, "task": task, "seed": 42, "created_at": datetime.now().astimezone().isoformat(),
        "baseline_sha256": sha256(baseline_path(task)),
        "selection_rule": ("maximize validation macro-F1 subject to positive recall and precision >= baseline"
                           if task == "toxic" else "maximize validation recall subject to FP <= baseline and macro-F1 >= baseline - 0.005"),
        "test_policy": "No test predictions during selection. Existing benchmark test was previously reported; no new independent-test claim.",
        "environment": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                        "platform": platform.platform(), "processor": platform.processor(), "device": "cpu"},
    }
    configurations = ([{"kind": kind, "C": c, "balanced": balanced}
                       for kind, c, balanced in (("char", 4, False), ("char", 1, False),
                         ("word-char", .5, True), ("word-char", 2, True), ("word-char", 8, True), ("word-char", 4, False))]
                      if task == "toxic" else
                      [{"features": f, "C": c} for f in ("baseline", "mean-max", "mean-max-std-diff") for c in (.01, .1, 1)])
    policy["configurations"] = configurations
    write_json(public / "protocol.json", policy)
    if task == "toxic":
        datasets, audit = load_toxic()
    else:
        datasets, audit = load_visual(needed=("train", "valid"))
    base_scores = baseline["model"].predict_proba(datasets["valid"]["x"])[:, 1]
    base_metrics = binary_metrics(datasets["valid"]["y"], base_scores, baseline["threshold"])
    all_rows = [dict(candidate="baseline", **base_metrics)]
    models = {"baseline": (baseline["model"], {"features": "baseline"} if task == "violence" else {"kind": "baseline"}, 0.)}
    for i, config in enumerate(configurations):
        name = f"candidate-{i+1:02d}"
        if task == "violence":
            if audit["features"] != config["features"]:
                datasets, audit = load_visual(config["features"], needed=("train", "valid"))
            model = make_pipeline(StandardScaler(), LogisticRegression(C=config["C"], class_weight="balanced",
                                    solver="liblinear", max_iter=1000, random_state=42))
        else:
            model = toxic_model(config["kind"], config["C"], config["balanced"])
        started = time.perf_counter()
        model.fit(datasets["train"]["x"], datasets["train"]["y"])
        elapsed = time.perf_counter() - started
        scores = model.predict_proba(datasets["valid"]["x"])[:, 1]
        models[name] = (model, config, elapsed)
        rows = [dict(candidate=name, **r) for r in threshold_rows(datasets["valid"]["y"], scores)]
        all_rows.extend(rows)
        joblib.dump(model, private / f"{name}.joblib")
        write_csv(public / "validation_search.csv", all_rows)
        best = max(rows, key=lambda r: r["macro_f1"] or 0)
        print(f"{task} {name} {config}: fit={elapsed:.2f}s best validation macro-F1={best['macro_f1']:.4f}", flush=True)
    selected = choose_operating_point(task, all_rows, base_metrics)
    model, config, fit_seconds = models[selected["candidate"]]
    artifact = dict(baseline)
    artifact.update(model=model, threshold=selected["threshold"], version=f"{task}-cpu-v2-{run_id}",
                    configuration=config, sklearn_version=sklearn.__version__, run_id=run_id)
    if task == "violence" and config["features"] != "baseline":
        artifact.update(sampling=f"uniform{FRAME_COUNT}-{config['features']}-v2", frame_count=FRAME_COUNT,
                        aggregation=config["features"])
    joblib.dump(artifact, private / "selected.joblib")
    if task == "violence":
        audit.update(features=config["features"], frame_count=8 if config["features"] == "baseline" else FRAME_COUNT)
    selection = {**policy, "selected": selected, "configuration": config, "fit_seconds": fit_seconds,
                 "selected_at": datetime.now().astimezone().isoformat(),
                 "data_audit": audit, "baseline_validation": base_metrics,
                 "artifact_sha256": sha256(private / "selected.joblib"), "test_evaluated": False}
    write_json(public / "selection.json", selection)
    print(json.dumps({"selected": selected, "configuration": config}, indent=2), flush=True)


def learning_curve(model, dataset, valid, threshold):
    history = []
    x, y = dataset["x"], dataset["y"]
    for fraction in (.2, .4, .6, .8, 1.):
        if fraction == 1:
            ids = np.arange(len(y)); current = model
        else:
            if isinstance(x, list):
                ids, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=fraction, random_state=42).split(np.zeros(len(y)), y))
            else:
                ids, _ = next(GroupShuffleSplit(n_splits=1, train_size=fraction, random_state=42).split(x, y, dataset["groups"]))
            current = clone(model)
            current.fit([x[i] for i in ids] if isinstance(x, list) else x[ids], y[ids])
        for split, inputs, labels in (("train", [x[i] for i in ids] if isinstance(x, list) else x[ids], y[ids]),
                                      ("valid", valid["x"], valid["y"])):
            measured = binary_metrics(labels, current.predict_proba(inputs)[:, 1], threshold)
            history.append(dict(train_samples=len(ids), train_fraction=fraction, split=split, **measured))
    return history


def evaluate(run_id, task):
    private, public = locations(run_id, task)
    if (public / "evaluation.json").exists():
        raise ValueError("Test already evaluated for this frozen selection")
    selection = json.loads((public / "selection.json").read_text(encoding="utf-8"))
    if sha256(private / "selected.joblib") != selection["artifact_sha256"]:
        raise ValueError("Selected artifact changed after freezing")
    if sha256(baseline_path(task)) != selection["baseline_sha256"]:
        raise ValueError("Baseline artifact changed")
    selected = joblib.load(private / "selected.joblib")
    baseline = joblib.load(baseline_path(task))
    if task == "toxic":
        base_data, _ = load_toxic(); new_data = base_data
    else:
        base_data, _ = load_visual(); new_data, _ = load_visual(selection["configuration"]["features"])
    rows, predictions, confusion = [], {}, []
    for variant, artifact, datasets in (("baseline", baseline, base_data), ("selected", selected, new_data)):
        predictions[variant] = {}
        individual = []
        for split in SPLITS:
            d = datasets[split]
            start = time.perf_counter()
            scores = artifact["model"].predict_proba(d["x"])[:, 1]
            seconds = time.perf_counter() - start
            measured = binary_metrics(d["y"], scores, artifact["threshold"])
            rows.append(dict(run_id=run_id, task=task, dataset="ViCTSD" if task == "toxic" else "RLVS",
                             variant=variant, model_version=artifact["version"], split=split,
                             predict_batch_seconds=seconds, **measured))
            predictions[variant][split] = (d["y"], scores)
            confusion.extend(dict(variant=variant, split=split, cell=cell, count=measured[cell]) for cell in ("tn", "fp", "fn", "tp"))
            for sample_id, group, label, score in zip(d["ids"], d["groups"], d["y"], scores):
                pred = int(score >= artifact["threshold"])
                outcome = ("TP" if pred else "FN") if label else ("FP" if pred else "TN")
                individual.append(dict(sample_id=sample_id, group_id=group, split=split, label=int(label),
                                       score=float(score), prediction=pred, outcome=outcome))
        write_csv(private / f"{variant}_predictions.csv", individual)
    history = learning_curve(selected["model"], new_data["train"], new_data["valid"], selected["threshold"])
    write_csv(public / "metrics_by_split.csv", rows)
    write_csv(public / "comparison.csv", [r for r in rows if r["split"] == "test"])
    write_csv(public / "confusion_matrix.csv", confusion)
    write_csv(public / "learning_curve.csv", history)
    plot_results(public, rows, predictions, history)
    plot_thresholds(public, selection)
    write_json(public / "evaluation.json", {"run_id": run_id, "task": task,
               "selection_sha256": sha256(public / "selection.json"),
               "evaluated_at": datetime.now().astimezone().isoformat(), "metrics": rows,
               "loss_definition": "unweighted binary cross-entropy on predicted probabilities; not regularized solver objective",
               "learning_curve_axis": "independent fitted training subsets, not epochs",
               "test_status": "previously reported benchmark; no unseen target-video accuracy claim"})
    print(json.dumps([r for r in rows if r["split"] == "test"], indent=2), flush=True)


def plot_thresholds(public, selection):
    import csv
    import matplotlib.pyplot as plt
    with (public / "validation_search.csv").open(encoding="utf-8-sig") as stream:
        rows = [r for r in csv.DictReader(stream) if r["candidate"] == selection["selected"]["candidate"]]
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for metric in ("precision", "recall", "macro_f1"):
        valid = [r for r in rows if r[metric]]
        ax.plot([float(r["threshold"]) for r in valid], [float(r[metric]) for r in valid], label=metric)
    ax.axvline(selection["selected"]["threshold"], linestyle="--", color="black", label="Selected on validation")
    ax.set(xlabel="Threshold", ylabel="Metric", ylim=(0, 1.05), title="Validation operating point")
    ax.legend(); ax.grid(alpha=.2)
    for suffix in ("png", "svg"):
        fig.savefig(public / f"validation_threshold.{suffix}", dpi=160)
    clean_svg(public / "validation_threshold.svg")
    plt.close(fig)


def render(run_id, task):
    """Redraw existing measurements without fitting or evaluating test again."""
    import csv
    private, public = locations(run_id, task)
    evaluation = json.loads((public / "evaluation.json").read_text(encoding="utf-8"))
    predictions = {}
    for variant in ("baseline", "selected"):
        with (private / f"{variant}_predictions.csv").open(encoding="utf-8-sig") as stream:
            data = list(csv.DictReader(stream))
        predictions[variant] = {split: (np.array([int(r['label']) for r in data if r['split'] == split]),
                                        np.array([float(r['score']) for r in data if r['split'] == split])) for split in SPLITS}
    with (public / "learning_curve.csv").open(encoding="utf-8-sig") as stream:
        history = [dict(r, train_samples=int(r['train_samples']), log_loss=float(r['log_loss']),
                        accuracy=float(r['accuracy'])) for r in csv.DictReader(stream)]
    plot_results(public, evaluation["metrics"], predictions, history)
    plot_thresholds(public, json.loads((public / "selection.json").read_text(encoding="utf-8")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["select", "evaluate", "render"])
    parser.add_argument("--task", choices=["toxic", "violence"], required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    {"select": select, "evaluate": evaluate, "render": render}[args.command](args.run_id, args.task)


if __name__ == "__main__":
    main()
