"""Measured binary metrics and export helpers; no text/media in public reports."""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    confusion_matrix, log_loss, precision_recall_curve, precision_recall_fscore_support,
    roc_auc_score, roc_curve,
)


def binary_metrics(labels, scores, threshold):
    y, p = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or p.shape != y.shape or not len(y):
        raise ValueError("Expected equally sized nonempty label/score vectors")
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be a finite probability")
    if not set(y).issubset({0, 1}) or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("Invalid binary labels or probability scores")
    pred = p >= threshold
    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=[0, 1], zero_division=np.nan)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    both = len(set(y)) == 2
    result = dict(n=len(y), negative=int(support[0]), positive=int(support[1]),
                  threshold=float(threshold), accuracy=float(accuracy_score(y, pred)),
                  balanced_accuracy=float(balanced_accuracy_score(y, pred)) if both else None,
                  log_loss=float(log_loss(y, np.column_stack([1 - p, p]), labels=[0, 1])),
                  precision_negative=float(precision[0]), recall_negative=float(recall[0]),
                  f1_negative=float(f1[0]), precision=float(precision[1]), recall=float(recall[1]),
                  f1=float(f1[1]), macro_f1=float(np.mean(f1)),
                  roc_auc=float(roc_auc_score(y, p)) if both else None,
                  average_precision=float(average_precision_score(y, p)) if both else None,
                  tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))
    missing = []
    for key, value in result.items():
        if isinstance(value, float) and not np.isfinite(value):
            result[key] = None
        if result[key] is None:
            missing.append(key)
    result["undefined_metrics"] = ";".join(missing)
    return result


def write_json(path, payload):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8", newline="\n")


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot export an empty table")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def clean_svg(path):
    """Remove renderer-only trailing spaces without changing vector content."""
    path = Path(path)
    path.write_text("\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines()) + "\n",
                    encoding="utf-8", newline="\n")


def threshold_rows(labels, scores):
    return [binary_metrics(labels, scores, float(t)) for t in np.linspace(.05, .95, 91)]


def choose_operating_point(task, rows, baseline):
    """Fixed validation-only constraints; caller includes baseline among candidates."""
    if task == "toxic":
        eligible = [r for r in rows if r["recall"] >= baseline["recall"]
                    and r["precision"] is not None and r["precision"] >= baseline["precision"]]
        key = lambda r: (r["macro_f1"] or 0, r["precision"], -r["fp"], -abs(r["threshold"] - .5))
    elif task == "violence":
        eligible = [r for r in rows if r["fp"] <= baseline["fp"]
                    and (r["macro_f1"] or 0) >= baseline["macro_f1"] - .005]
        key = lambda r: (r["recall"], r["macro_f1"], -r["fp"], -abs(r["threshold"] - .5))
    else:
        raise ValueError(task)
    if not eligible:
        raise ValueError("No eligible validation configuration; retain baseline")
    return max(eligible, key=key)


def plot_results(directory, rows, predictions, history, *, label_description="Nhãn thật", scope_note=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(directory)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})

    def save(fig, name):
        fig.savefig(directory / f"{name}.png", dpi=160, bbox_inches="tight")
        fig.savefig(directory / f"{name}.svg", bbox_inches="tight")
        clean_svg(directory / f"{name}.svg")
        plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout="constrained")
    for ax, metric in zip(axes.flat, ("accuracy", "precision", "recall", "f1", "macro_f1", "log_loss")):
        for i, variant in enumerate(("baseline", "selected")):
            data = [next(r for r in rows if r["variant"] == variant and r["split"] == s)
                    for s in ("train", "valid", "test")]
            values = [r[metric] for r in data]
            bars = ax.bar(np.arange(3) + (i - .5) * .35, values, .35,
                          label="Baseline" if variant == "baseline" else "Ứng viên chọn trên validation")
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_xticks(range(3), ["Train", "Validation", "Test"])
        ax.set_title("Loss: binary cross-entropy (thấp hơn tốt hơn)" if metric == "log_loss" else metric)
        if metric != "log_loss":
            ax.set_ylim(0, 1.13)
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend()
    fig.suptitle(scope_note or f"{rows[0]['task'].upper()} — so sánh trên benchmark; chưa phải độ chính xác của app video")
    save(fig, "metrics_by_split")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout="constrained")
    for ax, variant in zip(axes, ("baseline", "selected")):
        r = next(r for r in rows if r["variant"] == variant and r["split"] == "test")
        matrix = np.array([[r['tn'], r['fp']], [r['fn'], r['tp']]])
        ax.imshow(matrix, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix):
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > matrix.max()/2 else "black")
        ax.set(xticks=[0, 1], yticks=[0, 1], xlabel="Nhãn dự đoán", ylabel=label_description,
               title=f"{'Baseline' if variant == 'baseline' else 'Ứng viên'} | test n={r['n']}")
    if scope_note:
        fig.suptitle(scope_note, fontsize=10)
    save(fig, "confusion_matrix")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for variant in ("baseline", "selected"):
        y, scores = predictions[variant]["test"]
        if len(set(y)) != 2:
            continue
        precision, recall, _ = precision_recall_curve(y, scores)
        fpr, tpr, _ = roc_curve(y, scores)
        axes[0].plot(recall, precision, label=f"{variant}: AP={average_precision_score(y, scores):.3f}")
        axes[1].plot(fpr, tpr, label=f"{variant}: AUC={roc_auc_score(y, scores):.3f}")
    axes[0].set(xlabel="Recall", ylabel="Precision", title="Test precision–recall")
    axes[1].set(xlabel="False-positive rate", ylabel="True-positive rate", title="Test ROC")
    for ax in axes:
        ax.legend(); ax.grid(alpha=.2)
    if scope_note:
        fig.suptitle(scope_note, fontsize=10)
    save(fig, "test_curves")

    if history:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
        for split in ("train", "valid"):
            subset = [r for r in history if r["split"] == split]
            for ax, metric in zip(axes, ("log_loss", "accuracy")):
                ax.plot([r["train_samples"] for r in subset], [r[metric] for r in subset], "o-", label=split)
                ax.set(xlabel="Số mẫu huấn luyện (không phải epoch)", ylabel=metric)
                ax.legend(); ax.grid(alpha=.2)
        fig.suptitle("Đường học: fit các tập train con độc lập, giữ cấu hình đã chọn")
        save(fig, "learning_curve")
