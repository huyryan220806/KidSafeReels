"""RLVS exploratory visual baseline: grouping/split before feature fitting."""
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ml.download_rlvs import DEST
from ml.media import fingerprint, inspect_video, sample_frames
from ml.toxic_baseline import choose_threshold, metrics, sha256
from ml.visual import BACKBONE, CLASSIFIER, ROOT, SAMPLING_VERSION, VisualEncoder

INSPECTION_VERSION = "uniform8-dhash3-max60-4k-v1"


def similar_groups(fingerprints):
    parents = list(range(len(fingerprints)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i, left in enumerate(fingerprints):
        for j in range(i):
            # Conservative grouping: all three sampled scenes must be similar.
            if all((a ^ b).bit_count() <= 10 for a, b in zip(left, fingerprints[j])):
                parents[root(i)] = root(j)
    return np.array([root(i) for i in range(len(parents))])


def make_split(labels, groups):
    indices = np.arange(len(labels))
    outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    remaining, test = next(outer.split(indices, labels, groups))
    inner = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
    train_sub, valid_sub = next(inner.split(remaining, labels[remaining], groups[remaining]))
    splits = {"train": remaining[train_sub], "valid": remaining[valid_sub], "test": test}
    for name, ids in splits.items():
        if set(labels[ids]) != {0, 1}:
            raise ValueError(f"Both labels required in {name}")
    assert not (set(groups[splits["train"]]) & set(groups[splits["test"]]))
    assert not (set(groups[splits["valid"]]) & set(groups[splits["test"]]))
    assert not (set(groups[splits["train"]]) & set(groups[splits["valid"]]))
    return splits


def publish_split(manifest):
    destination = ROOT / "data/splits/rlvs-v1.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = {"dataset": "RLVS Kaggle version 1", "seed": 42,
               "grouping": "exact SHA256 + heuristic three-frame dHash; original source IDs unavailable",
               "rows": [{k: row[k] for k in ("path", "sha256", "label", "split", "group_id")}
                        for row in manifest["rows"]]}
    destination.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding="utf-8")


def scan_videos(videos):
    entries, labels_by_hash = [], {}
    for path in videos:
        label = {"Violence": 1, "NonViolence": 0}.get(path.parent.name)
        if label is None:
            continue
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        entries.append((path, label, digest))
        labels_by_hash.setdefault(digest, set()).add(label)
    conflicting = {digest for digest, labels in labels_by_hash.items() if len(labels) > 1}
    return entries, conflicting


def main():
    cache_dir = ROOT / "data/processed/rlvs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    videos = sorted(DEST.rglob("*.mp4"))
    if not videos:
        raise FileNotFoundError("Run python -m ml.download_rlvs first")
    rows, failures, seen = [], [], set()
    # Resolve label conflicts before selecting one copy of each video.
    entries, conflicting = scan_videos(videos)
    cached_rows = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("inspection_version") == INSPECTION_VERSION:
            cached_rows = {r["sha256"]: r for r in previous["rows"]}
    # Inspect content and group before feature extraction and before any model fit.
    for i, (path, label, digest) in enumerate(entries):
        if digest in conflicting:
            failures.append({"path": str(path.relative_to(DEST)), "reason": "conflicting_exact_duplicate_labels"})
            continue
        if digest in seen:
            failures.append({"path": str(path.relative_to(DEST)), "reason": "exact_duplicate"})
            continue
        seen.add(digest)
        try:
            cached = cached_rows.get(digest)
            if cached:
                duration, hashes = cached["duration"], cached["fingerprint"]
            else:
                duration = inspect_video(path).duration
                hashes = fingerprint(sample_frames(path))
            rows.append({"path": str(path.relative_to(DEST)), "label": label, "sha256": digest,
                         "duration": duration, "fingerprint": hashes})
        except ValueError as exc:
            failures.append({"path": str(path.relative_to(DEST)), "reason": str(exc)})
        if (i + 1) % 100 == 0:
            print(f"Inspected {i + 1}/{len(videos)}", flush=True)
    labels = np.array([row["label"] for row in rows])
    groups = similar_groups([r["fingerprint"] for r in rows])
    splits = make_split(labels, groups)
    for split, indices in splits.items():
        for i in indices:
            rows[i]["split"] = split
            rows[i]["group_id"] = int(groups[i])
    manifest_path.write_text(json.dumps({"inspection_version": INSPECTION_VERSION, "rows": rows,
                                         "excluded": failures}, ensure_ascii=False, indent=2), encoding="utf-8")
    publish_split({"rows": rows})
    encoder = VisualEncoder()
    backbone_sha = sha256(BACKBONE)
    features = []
    start = time.perf_counter()
    for i, row in enumerate(rows):
        key = hashlib.sha256((row["sha256"] + backbone_sha + SAMPLING_VERSION).encode()).hexdigest()
        cached = cache_dir / f"{key}.npy"
        if cached.exists():
            feature = np.load(cached, allow_pickle=False)
        else:
            feature = encoder.encode(sample_frames(DEST / row["path"]))
            np.save(cached, feature)
        features.append(feature)
        if (i + 1) % 100 == 0:
            print(f"Features {i + 1}/{len(rows)}", flush=True)
    feature_seconds = time.perf_counter() - start
    matrix = np.array(features)
    model = make_pipeline(StandardScaler(), LogisticRegression(
        C=0.1, class_weight="balanced", solver="liblinear", random_state=42, max_iter=1000))
    start = time.perf_counter()
    model.fit(matrix[splits["train"]], labels[splits["train"]])
    fit_seconds = time.perf_counter() - start
    val_scores = model.predict_proba(matrix[splits["valid"]])[:, 1]
    threshold = choose_threshold(labels[splits["valid"]], val_scores)
    artifact = {"model": model, "version": "violence-resnet18-v1", "threshold": threshold,
                "sklearn_version": sklearn.__version__, "sampling": SAMPLING_VERSION,
                "backbone_sha256": backbone_sha}
    joblib.dump(artifact, CLASSIFIER)
    test_scores = model.predict_proba(matrix[splits["test"]])[:, 1]
    report = {
        "version": artifact["version"], "source": json.loads((DEST / "provenance.json").read_text()),
        "split": {s: {"count": len(ids), "violence": int(labels[ids].sum()), "groups": len(set(groups[ids]))}
                  for s, ids in splits.items()},
        "grouping": "SHA256 exact dedup + three-frame dHash all distances <=10; source IDs unavailable",
        "excluded_count": len(failures), "unique_groups": len(set(groups)),
        "conflicting_unique_videos": len(conflicting),
        "threshold": threshold, "threshold_objective": "validation macro-F1",
        "validation": metrics(labels[splits["valid"]], val_scores >= threshold),
        "test": metrics(labels[splits["test"]], test_scores >= threshold),
        "metric_label_mapping": {"non_toxic": "non_violence", "toxic": "violence"},
        "seed": 42, "sampling": SAMPLING_VERSION, "classifier_C": 0.1,
        "feature_seconds": feature_seconds, "fit_seconds": fit_seconds,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "sklearn": sklearn.__version__, "device": "cpu", "torch_threads": 4},
        "manifest_sha256": sha256(manifest_path), "artifact_sha256": sha256(CLASSIFIER),
        "backbone_sha256": backbone_sha,
        "limitations": ["Exploratory RLVS result, not short-video child-safety accuracy.",
                        "Heuristic groups do not guarantee original-source independence.",
                        "No temporal ground truth: clip score cannot localize violent moments.",
                        "Uniform 8-frame sampling may miss brief events; no motion modeling."],
    }
    # Use domain-correct names while reusing the common binary metric utility.
    for key in ("validation", "test"):
        cr = report[key]["classification_report"]
        cr["non_violence"] = cr.pop("non_toxic")
        cr["violence"] = cr.pop("toxic")
    del report["metric_label_mapping"]
    (ROOT / "reports/visual_baseline_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
