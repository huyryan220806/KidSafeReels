"""Reproducible CPU baseline: python -m ml.toxic_baseline --help."""

import argparse
import csv
import hashlib
import json
import platform
import time
import unicodedata
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline

REVISION = "e1ffb854ef7b7b7ae45a56d132c1f125e02e55e2"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/victsd"
ARTIFACT = ROOT / "models/checkpoints/toxic-tfidf-v1.joblib"


def normalize(text):
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def download():
    DATA.mkdir(parents=True, exist_ok=True)
    manifest = {"source": "https://github.com/tarudesu/ViCTSD", "revision": REVISION, "files": {}}
    for split in ("train", "valid", "test"):
        name = f"ViCTSD_{split}.csv"
        url = f"https://raw.githubusercontent.com/tarudesu/ViCTSD/{REVISION}/{name}"
        with urllib.request.urlopen(url, timeout=60) as response:
            payload = response.read()
        path = DATA / name
        path.write_bytes(payload)
        manifest["files"][name] = {"url": url, "sha256": sha256(path), "bytes": len(payload)}
    (DATA / "provenance.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Downloaded pinned source to {DATA}")


def read_split(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"Comment", "Toxicity"}.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing Comment/Toxicity columns: {path}")
        rows = []
        for row in reader:
            text = normalize(row["Comment"])
            label = row["Toxicity"].strip()
            if not text or label not in {"0", "1"}:
                raise ValueError(f"Empty text or invalid label at CSV line {reader.line_num}")
            rows.append((text, int(label)))
        return rows


def clean_splits(splits):
    """Keep first text per split; drop conflicts and overlaps from later splits.

    No training decision depends on test labels: conflicts are resolved only
    inside each split, then train > validation > test determines ownership.
    """
    seen = set()
    cleaned, audit = {}, {}
    for split in ("train", "valid", "test"):
        labels = {}
        for text, label in splits[split]:
            labels.setdefault(text, set()).add(label)
        keep = [(text, next(iter(values))) for text, values in labels.items()
                if len(values) == 1 and text not in seen]
        audit[split] = {
            "original": len(splits[split]), "retained": len(keep),
            "duplicate_rows": len(splits[split]) - len(labels),
            "conflicting_texts": sum(len(v) > 1 for v in labels.values()),
            "overlap_with_earlier_split": sum(text in seen for text in labels),
            "toxic": sum(label for _, label in keep),
        }
        seen.update(labels)
        if set(label for _, label in keep) != {0, 1}:
            raise ValueError(f"Both classes required after cleaning: {split}")
        cleaned[split] = keep
    return cleaned, audit


def choose_threshold(labels, scores):
    candidates = np.linspace(0.05, 0.95, 91)
    # Maximize macro F1 on validation only; deterministic tie-breaking.
    return float(max(candidates, key=lambda t: (
        f1_score(labels, scores >= t, average="macro", zero_division=0), -abs(t - 0.5))))


def metrics(labels, predicted):
    return {
        "classification_report": classification_report(labels, predicted, labels=[0, 1],
            target_names=["non_toxic", "toxic"], output_dict=True, zero_division=0),
        "confusion_matrix_true_rows_predicted_columns": confusion_matrix(labels, predicted, labels=[0, 1]).tolist(),
    }


def train():
    provenance = json.loads((DATA / "provenance.json").read_text(encoding="utf-8"))
    for name, meta in provenance["files"].items():
        if sha256(DATA / name) != meta["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {name}")
    splits, audit = clean_splits({s: read_split(DATA / f"ViCTSD_{s}.csv") for s in ("train", "valid", "test")})
    x_train, y_train = zip(*splits["train"])
    x_valid, y_valid = zip(*splits["valid"])
    x_test, y_test = zip(*splits["test"])
    model = Pipeline([
        ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                  max_features=100000, sublinear_tf=True)),
        ("classifier", LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear",
                                          max_iter=1000, random_state=42)),
    ])
    start = time.perf_counter()
    model.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - start
    validation_scores = model.predict_proba(x_valid)[:, 1]
    threshold = choose_threshold(y_valid, validation_scores)
    artifact = {"model": model, "threshold": threshold, "version": "toxic-tfidf-v1",
                "sklearn_version": sklearn.__version__, "normalization": "NFC-casefold-whitespace-v1"}
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, ARTIFACT)
    # Test used only after preprocessing, estimator and threshold are frozen.
    start = time.perf_counter()
    test_scores = model.predict_proba(x_test)[:, 1]
    inference_seconds = time.perf_counter() - start
    report = {
        "model_version": artifact["version"], "seed": 42, "threshold": threshold,
        "threshold_objective": "validation macro-F1", "data": provenance, "audit": audit,
        "validation": metrics(y_valid, validation_scores >= threshold),
        "test": metrics(y_test, test_scores >= threshold),
        "test_always_non_toxic": metrics(y_test, np.zeros(len(y_test), dtype=int)),
        "fit_seconds": fit_seconds, "test_batch_seconds": inference_seconds,
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "sklearn": sklearn.__version__, "numpy": np.__version__},
        "artifact_sha256": sha256(ARTIFACT),
        "limitations": ["Comment classification only; not evaluated on ASR or video.",
                        "Exact normalized text deduplication only; near duplicates/topic leakage not ruled out.",
                        "Scores are not calibrated safety probabilities.",
                        "No public redistribution license found in upstream repository at inspection."],
    }
    target = ROOT / "reports/toxic_baseline_metrics.json"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"report": str(target), "threshold": threshold, "test": report["test"]}, indent=2))


class ToxicClassifier:
    def __init__(self, path=ARTIFACT):
        # joblib/pickle is executable: load only locally trained/trusted artifacts.
        artifact = joblib.load(path)
        if artifact["sklearn_version"] != sklearn.__version__:
            raise ValueError("Artifact sklearn version differs from current environment")
        self.model, self.threshold, self.version = artifact["model"], artifact["threshold"], artifact["version"]

    def predict(self, text):
        if not isinstance(text, str) or not normalize(text):
            raise ValueError("Text must be a non-empty string")
        score = float(self.model.predict_proba([normalize(text)])[0, 1])
        return {"label": "toxic_language", "score": score, "flagged": score >= self.threshold,
                "threshold": self.threshold, "model_version": self.version, "scope": "text_only"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "train", "predict"])
    parser.add_argument("--text")
    args = parser.parse_args()
    if args.command == "download":
        download()
    elif args.command == "train":
        train()
    else:
        if not args.text or not args.text.strip():
            parser.error("predict requires non-empty --text")
        print(json.dumps(ToxicClassifier().predict(args.text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
