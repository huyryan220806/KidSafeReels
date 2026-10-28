"""Audit local speech candidates and build blind human-review pages; no model calls."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/Video_Train"
PRIVATE = ROOT / "data/private/speech200-2026-10-28"
REPORT = ROOT / "reports/SPEECH200_INTAKE_2026-10-28.json"


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def checked_media_path(base, value):
    if not isinstance(value, str) or Path(value).is_absolute():
        raise ValueError("Expected a relative media path")
    path = (base / value).resolve()
    if not path.is_relative_to(base.resolve()) or path.suffix.lower() != ".mp4":
        raise ValueError("Media path outside source pack or wrong format")
    return path


def decode_audit(path):
    import av
    from PIL import ImageStat
    with av.open(str(path)) as container:
        if len(container.streams.video) != 1 or len(container.streams.audio) != 1:
            raise ValueError("Expected one video and one audio stream")
        video = container.streams.video[0]
        if video.width * video.height > 3840 * 2160:
            raise ValueError("Frame dimensions exceed app limit")
        duration = float(container.duration / av.time_base) if container.duration else 0
        if not 0 < duration <= 60.001:
            raise ValueError("Duration outside app limit")
        for stream in container.streams:
            stream.codec_context.thread_count = 1
        frames, samples, audio_seconds = 0, 0, 0.
        fingerprints, next_second, last_timestamp = [], 0, -1.
        for frame in container.decode(video=0, audio=0):
            if getattr(frame, "is_corrupt", False):
                raise ValueError("Corrupt decoded frame")
            if isinstance(frame, av.VideoFrame):
                frames += 1
                if frame.pts is None or frames > 18000:
                    raise ValueError("Missing video PTS or too many frames")
                timestamp = float(frame.pts * frame.time_base)
                if timestamp < last_timestamp or timestamp > 60.001:
                    raise ValueError("Invalid decoded video timestamp")
                last_timestamp = timestamp
                if timestamp + 1e-6 >= next_second:
                    gray = frame.to_image().convert("L").resize((9, 8))
                    pixels = list(gray.getdata())
                    bits = [pixels[y*9+x+1] > pixels[y*9+x] for y in range(8) for x in range(8)]
                    fingerprints.append(dict(seconds=timestamp, dhash=f"{sum(int(b)<<i for i,b in enumerate(bits)):016x}",
                                             gray_stddev=ImageStat.Stat(gray).stddev[0]))
                    next_second = math.floor(timestamp) + 1
            else:
                samples += frame.samples
                audio_seconds += frame.samples / frame.sample_rate
        if frames == 0 or samples == 0 or (video.frames and frames != video.frames):
            raise ValueError("Incomplete media decode")
        return dict(duration_seconds=duration, decoded_video_frames=frames,
                    decoded_audio_seconds=audio_seconds, has_audio=True, width=video.width, height=video.height,
                    fingerprints=fingerprints)


def reviewer_ids(rows, annotator):
    if annotator == "huy":
        return [r["video_id"] for r in rows]
    if annotator != "app":
        raise ValueError("Unknown annotator")
    selected, used = set(), set()
    rng = random.Random(42)
    for label in (0, 1):
        groups = defaultdict(list)
        for row in rows:
            if row["user_reported_clip_label"] == label and row["source_group_id"] not in used:
                groups[row["source_group_id"]].append(row["video_id"])
        chosen = rng.sample(sorted(groups), min(20, len(groups)))
        selected.update(min(groups[g]) for g in chosen)
        used.update(chosen)
    return [r["video_id"] for r in rows if r["video_id"] in selected]


def near_duplicate_candidates(rows, fingerprints):
    """A conservative review queue, never an automatic duplicate verdict."""
    hashes = {r["video_id"]: {int(f["dhash"], 16) for f in fingerprints[r["video_id"]]
                             if f["gray_stddev"] > 12} for r in rows}
    pairs = []
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            matches = [(x, y) for x in hashes[a["video_id"]] for y in hashes[b["video_id"]]
                       if (x ^ y).bit_count() <= 2]
            if len({x for x, _ in matches}) >= 3 and len({y for _, y in matches}) >= 3:
                pairs.append(dict(left=a["video_id"], right=b["video_id"], matching_hash_pairs=len(matches),
                                  same_source=a["source_group_id"] == b["source_group_id"],
                                  cross_reported_label=a["user_reported_clip_label"] != b["user_reported_clip_label"],
                                  human_duplicate_verdict=None))
    return pairs


def audit():
    base = SOURCE / "Toxic/KidSafeReels-100-Clip-LoiNoi"
    sources = [(base, 1, "vi-speech"), (base / "Non_Toxic", 0, "vi-normal")]
    clips, source_digests = [], {}
    for directory, label, prefix in sources:
        source_manifest = directory / "Thong-tin-nguon/manifest.json"
        metadata = read(source_manifest)
        cohort = metadata["clips"]
        expected = {f"{prefix}-{i:03d}" for i in range(1, 101)}
        if len(cohort) != 100 or {r["video_id"] for r in cohort} != expected:
            raise ValueError(f"Expected 100 unique IDs in {prefix}")
        actual_files = {p.resolve() for p in directory.glob("*.mp4")}
        if actual_files != {checked_media_path(directory, r["video_path"]) for r in cohort}:
            raise ValueError("Media inventory differs from source manifest")
        source_digests[prefix] = sha(source_manifest)
        clips.extend((directory, label, row) for row in cohort)
    known = {}
    for path, key, dataset in ((ROOT / "data/private/violence-drive-2026-10-28/verified-manifest.json", "rows", "AIRTLab-external-used"),
                               (ROOT / "data/splits/rlvs-v1.json", "rows", "RLVS")):
        if path.exists():
            known.update({r["sha256"]: dataset for r in read(path)[key]})

    def verify(item):
        directory, label, row = item
        path = checked_media_path(directory, row["video_path"])
        if sha(path) != row["sha256"] or path.stat().st_size != row["size_bytes"]:
            raise ValueError(f"Source checksum mismatch: {row['video_id']}")
        if path.stat().st_size > 50 * 1024**2:
            raise ValueError("Video exceeds app upload limit")
        media = decode_audit(path)
        if abs(media["duration_seconds"] - row["duration_seconds"]) > .1:
            raise ValueError("Manifest duration differs from decoded media")
        source_id = row.get("source_youtube_id")
        if not isinstance(source_id, str) or not re.fullmatch(r"[\w-]{11}", source_id):
            raise ValueError("Missing original video group")
        start, end = row["source_start_seconds"], row["source_end_seconds"]
        if not all(type(v) in (int, float) and math.isfinite(v) for v in (start, end)) or not 0 <= start < end:
            raise ValueError("Invalid source interval")
        if abs((end-start)-media["duration_seconds"]) > .15:
            raise ValueError("Source interval duration mismatch")
        return dict(original_video_id=row["video_id"], path=path.relative_to(ROOT).as_posix(), sha256=row["sha256"],
                    source_group_id="youtube-" + source_id, source_start=start, source_end=end,
                    source_channel_id=row.get("source_channel_id"), query_speaker_group=row.get("query_speaker_group"),
                    prior_exact_match=known.get(row["sha256"]), source_rights_status=row.get("rights_status", "unverified"),
                    user_reported_clip_label=label, label_origin="User description of supplied cohort; per-clip review pending",
                    human_label=None, split=None, **media)

    rows = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, row in enumerate(pool.map(verify, clips), 1):
            rows.append(row)
            if i % 10 == 0:
                print(f"Speech hash + full video/audio decode {i}/200", flush=True)
    if len({r["sha256"] for r in rows}) != len(rows):
        raise ValueError("Duplicate speech bytes require review before annotation assignment")
    random.Random(20261028).shuffle(rows)
    fingerprints = {}
    for i, row in enumerate(rows, 1):
        row["video_id"] = f"review-{i:03d}"
        fingerprints[row["video_id"]] = row.pop("fingerprints")
    near_pairs = near_duplicate_candidates(rows, fingerprints)
    violence = [dict(filename=p.name, sha256=sha(p)) for p in sorted((SOURCE / "Violence").glob("*.mp4"))]
    for row in violence:
        row["prior_exact_match"] = known.get(row["sha256"])
    overlap_pairs = sum(a["source_group_id"] == b["source_group_id"] and
                        max(a["source_start"], b["source_start"]) < min(a["source_end"], b["source_end"])
                        for i, a in enumerate(rows) for b in rows[i+1:])
    manifest = dict(version="speech200-intake-v2", source_manifest_sha256=source_digests, rows=rows,
                    human_labels_completed=0, split_status="not_assigned_pending_original_stream_and_reupload_audit",
                    user_reported_positive_clips=100, user_reported_negative_clips=100,
                    training_ready=False, assignments={a: reviewer_ids(rows, a) for a in ("huy", "app")})
    frozen = PRIVATE / "manifest.json"
    if frozen.exists() and read(frozen) != manifest:
        raise ValueError("Intake differs from existing annotation manifest; preserve current submissions")
    if not frozen.exists():
        write(frozen, manifest)
    write(PRIVATE / "duplicate_review.json", dict(method="1 fps dHash, distance <=2, >=3 distinct nonflat hashes in both clips; heuristic only",
                                                  pairs=near_pairs, fingerprints=fingerprints))
    write(PRIVATE / "violence_duplicates.json", violence)
    report = dict(recorded_at=datetime.now().astimezone().isoformat(), speech_clips=len(rows),
        source_videos=len({r["source_group_id"] for r in rows}), source_overlapping_trim_pairs=overlap_pairs,
        duration_seconds=sum(r["duration_seconds"] for r in rows), min_duration=min(r["duration_seconds"] for r in rows),
        max_duration=max(r["duration_seconds"] for r in rows), full_video_audio_decode_passed=len(rows),
        sha256_verified=len(rows), exact_duplicates_within_speech=0,
        speech_prior_exact_matches=sum(r["prior_exact_match"] is not None for r in rows),
        source_prescreen_counts=dict(Counter(str(r.get("proposal", {}).get("toxic_language")) for _, _, r in clips)),
        source_human_label_counts=dict(Counter(str(r.get("labels", {}).get("toxic_language")) for _, _, r in clips)),
        user_reported_positive_clips=100, user_reported_negative_clips=100, reviewed_negative_videos=0,
        cohorts={str(label):dict(clips=sum(r["user_reported_clip_label"] == label for r in rows),
                  source_videos=len({r["source_group_id"] for r in rows if r["user_reported_clip_label"] == label}),
                  duration_seconds=sum(r["duration_seconds"] for r in rows if r["user_reported_clip_label"] == label)) for label in (0,1)},
        near_duplicate_candidates=len(near_pairs),
        cross_source_near_duplicate_candidates=sum(not p["same_source"] for p in near_pairs),
        cross_label_near_duplicate_candidates=sum(p["cross_reported_label"] for p in near_pairs),
        source_ids_shared_between_cohorts=len({r["source_group_id"] for r in rows if r["user_reported_clip_label"] == 0} &
                                              {r["source_group_id"] for r in rows if r["user_reported_clip_label"] == 1}),
        source_rights_status_counts=dict(Counter(r["source_rights_status"] for r in rows)),
        violence_files=len(violence), violence_prior_matches=dict(Counter(r["prior_exact_match"] or "unmatched" for r in violence)),
        assignments={a:len(ids) for a,ids in manifest["assignments"].items()},
        annotation_manifest_sha256=sha(frozen), trained=False, moderation_metrics=None,
        limitations=["Both cohorts selected by ASR/lexical prescreen; user-reported labels are separate from per-clip reviewed labels",
                     "Original video IDs are provisional groups; same-stream reuploads remain unaudited",
                     "Rights unverified in source metadata; keep raw media private and preserve provenance",
                     "dHash candidates require review; low match counts do not prove reuploads absent",
                     "No model features, ASR, moderation predictions, train/test split or model changes; dHash is only a leakage audit"])
    write(REPORT, report)
    return manifest


def blank_row(video_id, annotator):
    return dict(video_id=video_id, annotator=annotator, watched_full=False, language=None,
                toxic_language=None, evidence=[], corrected_transcript="", transcript_verified=False, notes="")


def validate(payload, manifest, annotator, digest):
    if not isinstance(payload, dict) or payload.get("manifest_sha256") != digest or payload.get("annotator") != annotator:
        raise ValueError("Wrong manifest or annotator")
    expected = manifest["assignments"][annotator]
    rows = payload.get("rows", [])
    if (not isinstance(rows, list) or len(rows) != len(expected)
            or any(not isinstance(r, dict) or not isinstance(r.get("video_id"), str) for r in rows)
            or {r["video_id"] for r in rows} != set(expected)):
        raise ValueError("Missing, duplicate or unassigned IDs")
    durations = {r["video_id"]: r["duration_seconds"] for r in manifest["rows"]}
    complete, uncertain, positive, negative = 0, 0, 0, 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Review rows must be objects")
        if row.get("annotator") != annotator or type(row.get("watched_full")) is not bool:
            raise ValueError("Invalid reviewer or watched status")
        value = row.get("toxic_language")
        if not (value is None or value == "uncertain" or type(value) is int and value in (0, 1)):
            raise ValueError("Expected null, uncertain, 0 or 1")
        language = row.get("language")
        if language not in (None, "vi", "other", "no_speech", "uncertain"):
            raise ValueError("Invalid language")
        if value is not None and (not row["watched_full"] or language is None):
            raise ValueError("Confirm full listening and language before labeling")
        if value == 1 and (not row.get("evidence") or language in ("no_speech", "uncertain")):
            raise ValueError("Positive needs audible speech and timestamped evidence")
        if value == 0 and language == "uncertain":
            raise ValueError("Unclear speech cannot be a confirmed negative")
        if value == 0 and row.get("evidence"):
            raise ValueError("Negative label conflicts with Toxic evidence")
        if not isinstance(row.get("evidence"), list):
            raise ValueError("Evidence must be a list")
        for interval in row["evidence"]:
            if not isinstance(interval, dict):
                raise ValueError("Evidence intervals must be objects")
            start, end = interval.get("start_seconds"), interval.get("end_seconds")
            if (type(start) not in (int,float) or type(end) not in (int,float)
                    or not math.isfinite(start) or not math.isfinite(end)
                    or not 0 <= start < end <= durations[row["video_id"]]
                    or not isinstance(interval.get("reason"), str) or not interval["reason"].strip()):
                raise ValueError("Invalid evidence interval or reason")
        if type(row.get("transcript_verified")) is not bool or not isinstance(row.get("corrected_transcript"), str):
            raise ValueError("Invalid transcript fields")
        if not isinstance(row.get("notes"), str):
            raise ValueError("Notes must be text")
        if row["transcript_verified"] and (not row["watched_full"] or not row["corrected_transcript"].strip()):
            raise ValueError("Transcript confirmation needs text and full listening")
        complete += value is not None
        uncertain += value == "uncertain"
        positive += type(value) is int and value == 1
        negative += type(value) is int and value == 0
    return dict(assigned=len(rows), completed=complete, uncertain=uncertain, positive=positive, negative=negative,
                missing=len(rows)-complete, note="Structure only, not a quality audit or adjudicated gold")


def pack():
    manifest = read(PRIVATE / "manifest.json")
    digest = sha(PRIVATE / "manifest.json")
    (PRIVATE / "media").mkdir(exist_ok=True)
    for row in manifest["rows"]:
        original = ROOT / row["path"]
        if sha(original) != row["sha256"]:
            raise ValueError("Media changed after audit")
        target = PRIVATE / "media" / (row["video_id"] + ".mp4")
        if not target.exists():
            shutil.copy2(original, target)
        if sha(target) != row["sha256"]:
            raise ValueError("Review copy checksum mismatch")
    template = (ROOT / "ml/templates/speech_review.html").read_text(encoding="utf-8")
    for annotator in ("huy", "app"):
        payload = dict(manifest_sha256=digest, annotator=annotator,
                       rows=[blank_row(sid, annotator) for sid in manifest["assignments"][annotator]])
        config = dict(dataset="speech200", initial=payload, clips={r["video_id"]:dict(duration=r["duration_seconds"], src=f"media/{r['video_id']}.mp4")
                                             for r in manifest["rows"] if r["video_id"] in manifest["assignments"][annotator]})
        page = template.replace("__REVIEW_CONFIG__", json.dumps(config, ensure_ascii=False).replace("<", "\\u003c"))
        (PRIVATE / f"review-{annotator}.html").write_text(page, encoding="utf-8")
        blank = PRIVATE / f"annotations-{annotator}.blank.json"
        if not blank.exists():
            write(blank, payload)
    shutil.copy2(ROOT / "docs/SPEECH200_ANNOTATION.md", PRIVATE / "HUONG_DAN.md")
    shutil.copy2(ROOT / "docs/ANNOTATION_GUIDE.md", PRIVATE / "ANNOTATION_GUIDE.md")
    with zipfile.ZipFile(PRIVATE / "speech200-app-review.zip", "w", compression=zipfile.ZIP_STORED) as archive:
        for name in ("review-app.html", "HUONG_DAN.md", "ANNOTATION_GUIDE.md"):
            archive.write(PRIVATE / name, name)
        archive.writestr("annotations-app.blank.json", json.dumps(dict(manifest_sha256=digest, annotator="app",
                        rows=[blank_row(sid, "app") for sid in manifest["assignments"]["app"]]), ensure_ascii=False, indent=2))
        for sid in manifest["assignments"]["app"]:
            archive.write(PRIVATE / "media" / f"{sid}.mp4", f"media/{sid}.mp4")
    print(f"Blind review ready: {PRIVATE}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "pack", "check"])
    parser.add_argument("--annotator", choices=["huy", "app"])
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        audit(); pack()
    elif args.command == "pack":
        pack()
    else:
        if not args.annotator or not args.input:
            parser.error("check requires --annotator and --input")
        print(json.dumps(validate(read(args.input), read(PRIVATE / "manifest.json"), args.annotator, sha(PRIVATE / "manifest.json")), ensure_ascii=False))
