"""Build a private, portable Toxic annotation pack without loading any model."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import shutil
import zipfile

from ml.speech_intake import ROOT, blank_row, checked_media_path, read, sha, validate, write

RUN = "app-toxic-dev30-2026-11-02"
PRIVATE = ROOT / "data/private" / RUN
PACK = PRIVATE / "app30"
SOURCE = ROOT / "runs/toxic-precision-2026-10-29"
REVIEWED = ROOT / "runs/toxic-reviewed-2026-10-29"


def select_rows(manifest, overlap_ids, predictions, per_label=10):
    """Retain independent overlap; add only source-disjoint TRAIN disagreements."""
    by_id = {r["video_id"]: r for r in manifest["speech"]}
    if len(overlap_ids) != len(set(overlap_ids)):
        raise ValueError("Duplicate overlap IDs")
    chosen, groups, sources = [], set(), set()

    def add(row):
        if row["split"] not in ("train", "valid"):
            raise ValueError("Test clips are excluded from this development review")
        if row["training_group"] in groups or row["source_group_id"] in sources:
            raise ValueError("Repeated source in annotation queue")
        chosen.append(row)
        groups.add(row["training_group"])
        sources.add(row["source_group_id"])

    for sid in overlap_ids:
        add(by_id[sid])
    for label in (0, 1):
        candidates = []
        for row in manifest["speech"]:
            if row["split"] != "train" or row["training_label"] != label:
                continue
            pred = predictions.get(row["video_id"])
            if pred is not None and pred["flagged"] != bool(label):
                candidates.append(row)
        # Strong false alarms / low-scoring misses first; source labels are only hypotheses.
        def rank(row):
            score = predictions[row["video_id"]]["score"]
            return (score is not None, (score if label else -score) if score is not None else 0,
                    row["video_id"])
        count = 0
        for row in sorted(candidates, key=rank):
            if row["training_group"] in groups or row["source_group_id"] in sources:
                continue
            add(row)
            count += 1
            if count == per_label:
                break
        if count != per_label:
            raise ValueError("Not enough distinct training disagreement sources")
    random.Random(42).shuffle(chosen)
    return chosen


def load_predictions(manifest):
    by_id = {r["video_id"]: r for r in manifest["speech"]}
    predictions = {}
    with (SOURCE / "toxic/predictions.csv").open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["variant"] != "baseline" or row["split"] not in ("train", "valid"):
                continue
            source = by_id[row["video_id"]]
            if row["split"] != source["split"] or int(row["label"]) != source["training_label"]:
                raise ValueError("Prediction metadata disagrees with frozen split")
            predictions[row["video_id"]] = dict(flagged=row["hybrid_flag"] == "True",
                score=float(row["score"]) if row["score"] else None)
    # Account for the recovery already deployed; do not reselect its fixed misses as errors.
    recovery_paths = sorted((REVIEWED / "recovery").glob("review-*.json"))
    for path in recovery_paths:
        row = read(path)
        source = by_id[row["video_id"]]
        if row["sha256"] != source["sha256"] or source["split"] not in ("train", "valid"):
            raise ValueError("Recovery cache does not match development media")
        predictions[row["video_id"]] = dict(flagged=row["recovered"]["flagged"],
                                            score=row["recovered"]["score"])
    return predictions, recovery_paths


def build_pack(root, destination, rows, dataset):
    """Whitelist portable fields; no previous annotations, labels, ASR or source paths."""
    if destination.exists():
        raise ValueError("Pack already exists; preserve drafts and use the existing pack")
    for row in rows:
        if sha(checked_media_path(root, row["path"])) != row["sha256"]:
            raise ValueError("Source media changed")
    destination.mkdir(parents=True)
    portable = dict(dataset=dataset, rows=[dict(video_id=r["video_id"],
        duration_seconds=r["duration_seconds"]) for r in rows],
        assignments={"app": [r["video_id"] for r in rows]})
    write(destination / "review-manifest.json", portable)
    payload = dict(manifest_sha256=sha(destination / "review-manifest.json"), annotator="app",
                   rows=[blank_row(r["video_id"], "app") for r in rows])
    config = dict(dataset=dataset, initial=payload, clips={})
    (destination / "media").mkdir()
    for row in rows:
        name = f"media/{row['video_id']}.mp4"
        shutil.copy2(checked_media_path(root, row["path"]), destination / name)
        if sha(destination / name) != row["sha256"]:
            raise ValueError("Media copy checksum mismatch")
        config["clips"][row["video_id"]] = dict(src=name, duration=row["duration_seconds"])
    template = (root / "ml/templates/speech_review.html").read_text(encoding="utf-8")
    page = template.replace("__REVIEW_CONFIG__", json.dumps(config, ensure_ascii=False).replace("<", "\\u003c"))
    (destination / "review-app.html").write_text(page, encoding="utf-8")
    write(destination / "annotations-app.blank.json", payload)
    shutil.copy2(root / "docs/APP_TOXIC_ANNOTATION.md", destination / "HUONG_DAN.md")
    # Relative path works after extraction anywhere; no Python/model installation required.
    (destination / "Mo-Phieu.cmd").write_text(
        '@echo off\nstart "" "%~dp0review-app.html"\n', encoding="ascii")
    checksums = {p.relative_to(destination).as_posix(): sha(p)
                 for p in sorted(destination.rglob("*")) if p.is_file()}
    write(destination / "CHECKSUMS.json", checksums)
    archive_path = destination.parent / f"{dataset}.zip"
    if archive_path.exists():
        raise ValueError("Existing archive will not be overwritten")
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(destination.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(destination).as_posix())
    with zipfile.ZipFile(archive_path) as archive:
        if archive.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    return archive_path


def check_submission(pack, submission, require_complete=False):
    manifest_path = pack / "review-manifest.json"
    manifest, payload = read(manifest_path), read(submission)
    result = validate(payload, manifest, "app", sha(manifest_path))
    for row in payload["rows"]:
        if row["toxic_language"] == "uncertain" and not row["notes"].strip():
            raise ValueError(f"{row['video_id']}: explain uncertainty in notes")
    result["full_verified_transcripts"] = sum(r["transcript_verified"] for r in payload["rows"])
    result["transcripts_entered"] = sum(bool(r["corrected_transcript"].strip()) for r in payload["rows"])
    result["binary_labels"] = result["positive"] + result["negative"]
    if require_complete and result["missing"]:
        raise ValueError(f"Still missing {result['missing']} labels; partial drafts remain valid")
    return result


def prepare():
    if PRIVATE.exists():
        raise ValueError("This handoff is already prepared; do not overwrite existing drafts")
    manifest_path = SOURCE / "manifest.json"
    overlap_path = REVIEWED / "manifest.json"
    manifest = read(manifest_path)
    overlap = read(overlap_path)["rows"]
    if len(overlap) != 10:
        raise ValueError("Expected ten previously reviewed development clips")
    by_id = {r["video_id"]: r for r in manifest["speech"]}
    for row in overlap:
        source = by_id[row["video_id"]]
        if any(source[key] != row[key] for key in ("sha256", "split", "training_group")):
            raise ValueError("Reviewed overlap differs from frozen source manifest")
    predictions, recovery_paths = load_predictions(manifest)
    selected = select_rows(manifest, [r["video_id"] for r in overlap], predictions)
    archive = build_pack(ROOT, PACK, selected, RUN)
    # The private mapping stays outside the ZIP sent to the reviewer.
    write(PRIVATE / "selection-private.json", dict(rows=selected,
        overlap_ids=[r["video_id"] for r in overlap],
        inputs={str(p.relative_to(ROOT)): sha(p) for p in
                [manifest_path, overlap_path, SOURCE / "toxic/predictions.csv", *recovery_paths]},
        selection="Ten overlap + ten suspected FP and ten suspected FN from TRAIN, unique source groups; seed42; recovery overlays applied"))
    report = dict(dataset=RUN, prepared_utc=datetime.now(timezone.utc).isoformat(),
        clips=len(selected), overlap_clips=len(overlap), additional_clips=len(selected)-len(overlap),
        split_counts=dict(Counter(r["split"] for r in selected)),
        source_groups=len({r["training_group"] for r in selected}),
        total_seconds=round(sum(r["duration_seconds"] for r in selected), 3),
        archive_name=archive.name, archive_bytes=archive.stat().st_size, archive_sha256=sha(archive),
        manifest_sha256=sha(PACK / "review-manifest.json"),
        blank_status=check_submission(PACK, PACK / "annotations-app.blank.json"),
        scope="Development annotation and independent overlap review; no test clips, no model run, no new accuracy claim",
        distribution="Code/instructions in Git; ZIP/media/submissions private; no message sent to teammate")
    write(ROOT / "reports/APP_TOXIC_ANNOTATION_2026-11-02.json", report)
    (PRIVATE / "CONTINUE.md").write_text(
        "# Bàn giao gán nhãn Toxic — 02/11/2026\n\n"
        "Gói app30 đã tạo; phiếu 30 nhãn trống. Không chạy prepare lại. "
        "Huy gửi ZIP riêng cho thành viên app và link docs/APP_TOXIC_ANNOTATION.md.\n\n"
        "Khi nhận JSON: giữ nguyên bản trong submissions/, chạy check --require-complete. "
        "Đối chiếu 10 clip chéo với nhãn Huy, lưu bất đồng và phân xử riêng; không tự tạo gold. "
        "24 train / 6 validation giữ nguyên. Không có test, chưa train mới. Violence v3 giữ nguyên.\n",
        encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check"))
    parser.add_argument("--pack", type=Path, default=PACK)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        if args.input is None:
            parser.error("check requires --input")
        print(json.dumps(check_submission(args.pack, args.input, args.require_complete), ensure_ascii=False, indent=2))
