"""Compare human annotations, build auditable gold, evaluate frozen pilot once.

No default label and no tuning on test. All annotation/evaluation outputs stay private.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import math
from pathlib import Path
import subprocess
from time import perf_counter

from ml.pilot_annotation import LABELS, annotation_rows, validate_annotations
from ml.pilot_data import MANIFEST, ROOT, load_manifest, read_json, sha, write_json


def agreement(left, right):
    pairs = [(a, b) for a, b in zip(left, right, strict=True) if type(a) is int and type(b) is int]
    if not pairs:
        return {"support": 0, "agreement": None, "cohen_kappa": None}
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    expected = sum(sum(a == k for a, _ in pairs) * sum(b == k for _, b in pairs) for k in (0, 1)) / n**2
    return {"support": n, "agreement": observed,
            "cohen_kappa": (observed - expected) / (1 - expected) if expected < 1 else None}


def compare(data, huy, app):
    for name, payload in (("huy", huy), ("app", app)):
        count = validate_annotations(payload, data, name)
        if count != len(annotation_rows(data, name)):
            raise ValueError(f"{name}: labels incomplete ({count}); no gold or metrics created")
    left = {r["video_id"]: r for r in huy["rows"]}
    right = {r["video_id"]: r for r in app["rows"]}
    conflicts = []
    for video_id, row in left.items():
        for label in LABELS:
            a, b = row[label], right.get(video_id, {}).get(label, row[label])
            if a != b or a == "uncertain":
                conflicts.append({"video_id": video_id, "label": label, "huy": a,
                    "app": b if video_id in right else None, "value": None,
                    "reviewer": "", "reason": "", "reviewed": False})
    stats = {label: agreement([left[k][label] for k in right], [right[k][label] for k in right]) for label in LABELS}
    return {"manifest_sha256": sha(MANIFEST), "status": "needs_adjudication" if conflicts else "agreed",
            "double_annotated_clips": len(right), "agreement": stats, "decisions": conflicts}


def make_gold(data, huy, app, adjudication):
    review = compare(data, huy, app)
    if adjudication.get("manifest_sha256") != sha(MANIFEST):
        raise ValueError("Adjudication manifest mismatch")
    decisions = adjudication.get("decisions", [])
    supplied = {(r["video_id"], r["label"]): r for r in decisions}
    required = {(r["video_id"], r["label"]) for r in review["decisions"]}
    if len(supplied) != len(decisions) or set(supplied) != required:
        raise ValueError("Missing/duplicate/unexpected adjudication decisions")
    for row in supplied.values():
        value = row["value"]
        if not (value is None or type(value) is int and value in (0, 1)):
            raise ValueError("Final label must be 0/1 or null with documented exclusion")
        if row.get("reviewed") is not True or not row.get("reviewer", "").strip() or not row.get("reason", "").strip():
            raise ValueError("Every conflict/uncertain label needs human review and a reason")
    left = {r["video_id"]: r for r in huy["rows"]}
    records = []
    for clip in data["clips"]:
        video_id = clip["video_id"]
        labels = {key: supplied[(video_id, key)]["value"] if (video_id, key) in supplied else left[video_id][key]
                  for key in LABELS}
        records.append({"video_id": video_id, "split": clip["split"], "group_id": clip["group_id"],
                        "language": left[video_id]["language"], "labels": labels})
    return {"status": "ready", "manifest_sha256": sha(MANIFEST), "created_at": datetime.now(timezone.utc).isoformat(),
            "agreement": review["agreement"], "adjudication": decisions, "records": records}


def binary_metrics(truth, predicted):
    """Report abstention separately. No positives -> positive recall/F1 unestimable."""
    if len(truth) != len(predicted) or any(type(v) is not int or v not in (0, 1) for v in truth):
        raise ValueError("Invalid binary truth")
    if any(not (p is None or type(p) is int and p in (0, 1)) for p in predicted):
        raise ValueError("Invalid binary prediction")
    pairs = [(y, p) for y, p in zip(truth, predicted) if p is not None]
    tn = sum(y == 0 and p == 0 for y, p in pairs)
    fp = sum(y == 0 and p == 1 for y, p in pairs)
    fn = sum(y == 1 and p == 0 for y, p in pairs)
    tp = sum(y == 1 and p == 1 for y, p in pairs)
    return {"support": len(truth), "positive_support": sum(truth), "scored_support": len(pairs),
        "scored_positive_support": tp + fn, "abstained": len(truth) - len(pairs),
        "abstained_positive": sum(y == 1 and p is None for y, p in zip(truth, predicted)),
        "coverage": len(pairs) / len(truth) if truth else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall_conditional_on_scored": tp / (tp + fn) if tp + fn else None,
        "f1_conditional_on_scored": 2 * tp / (2 * tp + fp + fn) if tp + fn else None,
        "confusion_matrix_scored": [[tn, fp], [fn, tp]]}


def risk_union(values):
    if 1 in values:
        return 1
    return None if None in values else 0


def evaluate_predictions(records, predictions, thresholds):
    ids = [r["video_id"] for r in records]
    if len(set(ids)) != len(ids) or set(ids) != set(predictions):
        raise ValueError("Prediction/gold cohort mismatch")
    decisions = {}
    for video_id, scores in predictions.items():
        decisions[video_id] = {}
        for label in LABELS:
            value = scores[label]
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("Invalid score")
            decisions[video_id][label] = None if value is None else int(value >= thresholds[label])
    per_label = {}
    for label in LABELS:
        cohort = [r for r in records if r["labels"][label] is not None]
        per_label[label] = binary_metrics([r["labels"][label] for r in cohort],
                                         [decisions[r["video_id"]][label] for r in cohort])
        per_label[label]["unknown_gold_excluded"] = len(records) - len(cohort)
    # A fair removal experiment needs the same video IDs and same binary risk target.
    complete = [r for r in records if all(r["labels"][k] is not None and decisions[r["video_id"]][k] is not None for k in LABELS)]
    y = [risk_union(list(r["labels"].values())) for r in complete]
    ablation = {}
    for mode in ("visual_only", "text_only", "fusion"):
        pred = []
        for row in complete:
            d = decisions[row["video_id"]]
            pred.append(d["violence"] if mode == "visual_only" else d["toxic_language"] if mode == "text_only" else risk_union(list(d.values())))
        ablation[mode] = binary_metrics(y, pred)
    return {"per_label": per_label, "ablation_target": "any of the two P0 risks; shared fully labeled and fully scored cohort only",
            "ablation_video_ids": [r["video_id"] for r in complete], "ablation": ablation,
            "macro_f1_two_labels": sum(per_label[k]["f1_conditional_on_scored"] for k in LABELS) / 2
                if all(per_label[k]["f1_conditional_on_scored"] is not None for k in LABELS) else None,
            "warning": "Ablation measures detection of either risk after removing a modality; branches predict different labels. No claim of learned fusion improvement."}


def run_evaluation(data, gold_path, split, output):
    gold = read_json(gold_path)
    if gold.get("status") != "ready" or gold.get("manifest_sha256") != sha(MANIFEST):
        raise ValueError("Ready human gold for this manifest is required before predictions")
    records = gold["records"]
    by_id = {r["video_id"]: r for r in records}
    if len(by_id) != len(records) or set(by_id) != {c["video_id"] for c in data["clips"]}:
        raise ValueError("Gold does not cover the frozen manifest")
    for c in data["clips"]:
        r = by_id[c["video_id"]]
        if r["split"] != c["split"] or r["group_id"] != c["group_id"]:
            raise ValueError("Gold split/group changed")
        if set(r["labels"]) != set(LABELS) or any(not (v is None or type(v) is int and v in (0, 1)) for v in r["labels"].values()):
            raise ValueError("Gold contains non-final labels")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit code and documentation before evaluation to freeze the exact revision")
    audit = read_json(ROOT / "reports/pilot_data_audit.json")
    if audit["manifest_sha256"] != sha(MANIFEST):
        raise ValueError("Pilot audit manifest mismatch")
    media_hashes = {row["video_id"]: row["sha256"] for row in audit["records"]}
    for clip in data["clips"]:
        if clip["split"] == split and sha(ROOT / clip["video_path"]) != media_hashes.get(clip["video_id"]):
            raise ValueError("Pilot media changed since the published audit")
    from ml.release_verify import verify_artifacts
    verify_artifacts()
    artifacts = read_json(ROOT / "models/artifacts.json")
    thresholds = {"violence": artifacts["violence"]["threshold"], "toxic_language": artifacts["toxic"]["threshold"]}
    protocol = {"version": "pilot-eval-v1", "manifest_sha256": sha(MANIFEST), "gold_sha256": sha(gold_path),
                "artifacts_sha256": sha(ROOT / "models/artifacts.json"), "thresholds": thresholds, "split": split,
                "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "media_sha256": {c["video_id"]: media_hashes[c["video_id"]] for c in data["clips"] if c["split"] == split}}
    output = Path(output)
    if not output.resolve().is_relative_to((ROOT / "data/private").resolve()):
        raise ValueError("Evaluation output must stay under data/private to keep raw evidence private")
    if output.exists():
        raise FileExistsError("Never overwrite an evaluation run")
    marker = ROOT / "data/private/pilot-v1/test-evaluation-lock.json"
    if split == "test":
        if marker.exists():
            prior = read_json(marker)
            if prior["protocol"] != protocol or prior["status"] == "complete":
                raise ValueError("Test already used or protocol changed; retain audit instead of rerunning/tuning")
        else:
            write_json(marker, {"protocol": protocol, "status": "started"})
    from ml.inference import VideoService
    service = VideoService(ROOT / "models/checkpoints")
    if service.visual.threshold != thresholds["violence"] or service.text.threshold != thresholds["toxic_language"]:
        raise ValueError("Loaded thresholds differ from frozen artifact metadata")
    chosen = [r for r in records if r["split"] == split]
    predictions, latency = {}, []
    cache_dir = output.parent / (output.stem + "-predictions")
    for row in chosen:
        clip = next(c for c in data["clips"] if c["video_id"] == row["video_id"])
        path = ROOT / clip["video_path"]
        cache = cache_dir / (row["video_id"] + ".json")
        expected = {"protocol": protocol, "media_sha256": sha(path)}
        if cache.exists():
            saved = read_json(cache)
            if saved["identity"] != expected:
                raise ValueError("Prediction cache identity mismatch")
        else:
            started = perf_counter()
            result = service.analyze(path)
            saved = {"identity": expected, "result": result.model_dump(), "seconds": perf_counter() - started}
            write_json(cache, saved)
        predictions[row["video_id"]] = {k: saved["result"]["scores"][k] for k in LABELS}
        latency.append(saved["seconds"])
    report = {"protocol": protocol, "metrics": evaluate_predictions(chosen, predictions, thresholds),
              "pipeline_seconds_excluding_model_load": latency, "created_at": datetime.now(timezone.utc).isoformat()}
    write_json(output, report)
    if split == "test":
        write_json(marker, {"protocol": protocol, "status": "complete", "report_sha256": sha(output)})
    print(f"Evaluation saved: {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["compare", "gold", "evaluate"])
    parser.add_argument("--huy", type=Path)
    parser.add_argument("--app", type=Path)
    parser.add_argument("--adjudication", type=Path)
    parser.add_argument("--gold", type=Path)
    parser.add_argument("--split", choices=["train", "validation", "test"], default="validation")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = load_manifest()
    if args.command == "evaluate":
        if not args.gold:
            parser.error("--gold is required")
        run_evaluation(data, args.gold, args.split, args.output)
        return
    if not args.huy or not args.app:
        parser.error("--huy and --app are required")
    if args.output.exists():
        raise FileExistsError("Do not overwrite independent annotations or adjudication")
    huy, app = read_json(args.huy), read_json(args.app)
    result = compare(data, huy, app)
    if args.command == "gold":
        if not args.adjudication:
            parser.error("--adjudication is required")
        result = make_gold(data, huy, app, read_json(args.adjudication))
        result["annotation_sha256"] = {"huy": sha(args.huy), "app": sha(args.app), "adjudication": sha(args.adjudication)}
    if not args.output.resolve().is_relative_to((ROOT / "data/private").resolve()):
        raise ValueError("Human annotations and gold must stay under data/private")
    write_json(args.output, result)
    print(f"Saved {args.command}: {args.output}")


if __name__ == "__main__":
    main()
