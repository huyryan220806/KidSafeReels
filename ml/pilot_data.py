"""Reproducible public pilot media, timestamped preprocessing and leakage audit.

No classifier is invoked. Frozen labels remain null; human annotation is separate.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import subprocess
from time import perf_counter, sleep

from ml.pilot_sources import HEADERS, ROOT

MANIFEST = ROOT / "data/pilot/manifest.json"
OUTPUT = ROOT / "data/processed/pilot-v1"
MODELS = ROOT / "models/checkpoints/faster-whisper-base"
CONFIG = {"version": 2, "frames_per_second": 1, "beam_size": 1, "vad_filter": True,
          "condition_on_previous_text": False, "device": "cpu", "compute_type": "int8", "cpu_threads": 4}


def sha(path, algorithm="sha256"):
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_manifest(data):
    """Fail before extraction if groups, source IDs, paths or excerpt ranges leak."""
    sources = {s["source_id"]: s for s in data["sources"]}
    if len(sources) != len(data["sources"]) or not sources:
        raise ValueError("Empty/duplicate source IDs")
    groups, hashes, ids, ranges = {}, {}, set(), defaultdict(list)
    for source in sources.values():
        if source["split"] not in {"train", "validation", "test"}:
            raise ValueError("Unknown split")
        for mapping, key in ((groups, source["group_id"]), (hashes, source["sha1"])):
            previous = mapping.setdefault(key, source["split"])
            if previous != source["split"]:
                raise ValueError("Group or identical source crosses splits")
        if source["license"] != "CC-BY-3.0" or not source["rights_evidence_url"]:
            raise ValueError("Unreviewed license")
    for clip in data["clips"]:
        video_id = clip["video_id"]
        if not video_id.startswith("vi-pilot-") or not video_id[9:].isdigit() or video_id in ids:
            raise ValueError("Invalid/duplicate video ID")
        ids.add(video_id)
        if clip["video_path"] != f"data/processed/pilot-v1/media/{video_id}.mp4":
            raise ValueError("Unexpected output path")
        source = sources[clip["source_id"]]
        if any(clip[key] != source[key] for key in ("group_id", "split")):
            raise ValueError("Clip/source split mismatch")
        start, end = clip["start_seconds"], clip["end_seconds"]
        if not all(math.isfinite(v) for v in (start, end)) or not 0 <= start < end <= source["duration_seconds"] or end - start > 60:
            raise ValueError("Invalid excerpt interval")
        if any(label is not None for label in clip["labels"].values()):
            raise ValueError("Frozen manifest must not contain inferred gold labels")
        for other_start, other_end in ranges[clip["source_id"]]:
            if max(start, other_start) < min(end, other_end):
                raise ValueError("Overlapping excerpts")
        ranges[clip["source_id"]].append((start, end))
    if not ids:
        raise ValueError("No clips")


def load_manifest(path=MANIFEST):
    path = Path(path)
    expected = path.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
    if sha(path) != expected:
        raise ValueError("Frozen manifest checksum mismatch")
    data = read_json(path)
    validate_manifest(data)
    return data


def download(source):
    import requests
    from urllib.parse import urlparse
    if urlparse(source["download_url"]).hostname != "upload.wikimedia.org":
        raise ValueError("Only the reviewed Commons originals can be downloaded")
    suffix = Path(urlparse(source["download_url"]).path).suffix
    path = ROOT / "data/raw/pilot" / (source["source_id"] + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == source["size_bytes"] and sha(path, "sha1") == source["sha1"]:
        return path
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(3):
        try:
            with requests.get(source["download_url"], headers=HEADERS, stream=True, timeout=(20, 90)) as response:
                response.raise_for_status()
                with partial.open("wb") as handle:
                    for block in response.iter_content(1024 * 1024):
                        handle.write(block)
                        if handle.tell() > source["size_bytes"]:
                            raise ValueError("Download size differs from frozen source")
            break
        except requests.RequestException:
            if attempt == 2:
                raise
            sleep(2 ** attempt)
    if partial.stat().st_size != source["size_bytes"] or sha(partial, "sha1") != source["sha1"]:
        raise ValueError("Original checksum mismatch; do not silently accept a new file version")
    partial.replace(path)
    return path


def prepare(data):
    import imageio_ffmpeg
    from ml.media import inspect_video
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    state_path = OUTPUT / "media-state.json"
    state = read_json(state_path) if state_path.exists() else {"manifest_sha256": sha(MANIFEST), "clips": {}}
    if state["manifest_sha256"] != sha(MANIFEST):
        raise ValueError("Media cache belongs to a different manifest")
    for source in data["sources"]:
        print(f"download {source['source_id']}", flush=True)
        original = download(source)
        for clip in (c for c in data["clips"] if c["source_id"] == source["source_id"]):
            path = ROOT / clip["video_path"]
            saved = state["clips"].get(clip["video_id"])
            if path.exists() and saved and sha(path) == saved["sha256"]:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".partial.mp4")
            command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-ss", str(clip["start_seconds"]),
                       "-i", str(original), "-t", str(clip["end_seconds"] - clip["start_seconds"]),
                       "-map", "0:v:0", "-map", "0:a:0?", "-vf",
                       "scale=640:480:force_original_aspect_ratio=decrease:force_divisible_by=2,setsar=1",
                       "-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", "-b:a", "96k", "-ar", "16000", "-ac", "1",
                       "-movflags", "+faststart", "-threads", "2", str(partial)]
            subprocess.run(command, check=True, capture_output=True, timeout=180)
            info = inspect_video(partial)
            if abs(info.duration - (clip["end_seconds"] - clip["start_seconds"])) > 0.3:
                raise ValueError("Transcoded duration mismatch")
            partial.replace(path)
            state["clips"][clip["video_id"]] = {"sha256": sha(path), "size_bytes": path.stat().st_size,
                "duration_seconds": info.duration, "has_audio": info.has_audio, "source_sha1": source["sha1"],
                "ffmpeg_version": imageio_ffmpeg.get_ffmpeg_version(), "created_at": datetime.now(timezone.utc).isoformat()}
            write_json(state_path, state)
            print(f"prepared {clip['video_id']} {info.duration:.3f}s", flush=True)


def extract_frames(path, directory, source_start):
    import av
    from PIL import ImageStat
    directory.mkdir(parents=True, exist_ok=True)
    rows, next_second, previous_time = [], 0, -1.0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        for frame in container.decode(video=0):
            if frame.pts is None:
                raise ValueError("Decoded frame has no presentation timestamp")
            timestamp = float(frame.pts * frame.time_base)
            if timestamp < previous_time:
                raise ValueError("Nonmonotonic frame timestamps")
            previous_time = timestamp
            if timestamp + 1e-6 < next_second:
                continue
            # Save actual decoded PTS, not the requested sampling time or frame_index/fps.
            filename = f"frame-{len(rows):03d}.jpg"
            picture = frame.to_image()
            picture.save(directory / filename, quality=85)
            gray = picture.convert("L").resize((9, 8))
            pixels = [gray.getpixel((x, y)) for y in range(8) for x in range(9)]
            bits = [pixels[y * 9 + x + 1] > pixels[y * 9 + x] for y in range(8) for x in range(8)]
            dhash = sum(int(bit) << i for i, bit in enumerate(bits))
            rows.append({"file": filename, "clip_seconds": timestamp, "source_seconds": source_start + timestamp,
                         "pts": frame.pts, "time_base": str(frame.time_base), "sha256": sha(directory / filename),
                         "dhash": f"{dhash:016x}", "gray_stddev": ImageStat.Stat(gray).stddev[0]})
            next_second = math.floor(timestamp) + 1
    if not rows:
        raise ValueError("No decoded frames")
    return rows


def cache_key(manifest_hash, media_hash, model_hash, config=CONFIG):
    value = {"manifest": manifest_hash, "media": media_hash, "asr_model": model_hash, "config": config}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def bound_asr_interval(start, end, duration):
    """Whisper timing can overshoot; retain raw times and explicitly mark clipping."""
    if not all(math.isfinite(v) for v in (start, end, duration)) or end <= start or duration <= 0:
        raise ValueError("Invalid ASR interval")
    bounded_start, bounded_end = max(0, start), min(duration, end)
    if bounded_start >= bounded_end:
        return None
    return {"start_seconds": bounded_start, "end_seconds": bounded_end,
            "raw_start_seconds": start, "raw_end_seconds": end,
            "timing_clipped": bounded_start != start or bounded_end != end}


def preprocess(data):
    import av
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    from ml.media import inspect_video
    model_files = ["model.bin", "config.json", "tokenizer.json", "vocabulary.txt"]
    model_hash = {name: sha(MODELS / name) for name in model_files}
    model = None
    state = read_json(OUTPUT / "media-state.json")
    for clip in data["clips"]:
        path = ROOT / clip["video_path"]
        actual_hash = sha(path)
        if actual_hash != state["clips"][clip["video_id"]]["sha256"]:
            raise ValueError("Prepared media changed; run prepare before preprocessing")
        key = cache_key(sha(MANIFEST), actual_hash, model_hash)
        target = OUTPUT / "features" / clip["video_id"] / "preprocess.json"
        if target.exists():
            old = read_json(target)
            if old["cache_key"] == key and all((target.parent / "frames" / f["file"]).exists() for f in old["frames"]):
                print(f"cached {clip['video_id']}", flush=True)
                continue
        started = perf_counter()
        info = inspect_video(path)
        frames = extract_frames(path, target.parent / "frames", clip["start_seconds"])
        transcript = {"status": "missing_audio", "segments": [], "language": None, "language_probability": None}
        if info.has_audio:
            if model is None:
                model = WhisperModel(str(MODELS), device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True)
            audio = decode_audio(str(path), sampling_rate=16000)
            chunks, metadata = model.transcribe(audio, beam_size=1, vad_filter=True, condition_on_previous_text=False)
            segments, rejected = [], []
            for segment in chunks:
                interval = bound_asr_interval(segment.start, segment.end, info.duration)
                if interval is None:
                    rejected.append({"raw_start_seconds": segment.start, "raw_end_seconds": segment.end,
                                     "reason": "ASR interval entirely outside media duration"})
                    continue
                segments.append({**interval,
                    "source_start_seconds": clip["start_seconds"] + interval["start_seconds"],
                    "source_end_seconds": clip["start_seconds"] + interval["end_seconds"],
                    "text": segment.text, "no_speech_probability": segment.no_speech_prob,
                    "average_log_probability": segment.avg_logprob})
            transcript = {"status": "transcribed" if segments else "no_speech_detected", "segments": segments,
                          "rejected_intervals": rejected,
                          "language": metadata.language, "language_probability": metadata.language_probability,
                          "vi_gate_passed": metadata.language == "vi" and metadata.language_probability >= 0.5,
                          "warning": "Automatic transcript and approximate word timing, NOT human gold labels"}
        result = {"video_id": clip["video_id"], "cache_key": key, "manifest_sha256": sha(MANIFEST),
                  "media_sha256": actual_hash, "asr_model_sha256": model_hash, "config": CONFIG,
                  "pyav_version": av.__version__, "duration_seconds": info.duration, "frames": frames,
                  "transcript": transcript, "elapsed_seconds": round(perf_counter() - started, 3),
                  "processed_at": datetime.now(timezone.utc).isoformat()}
        write_json(target, result)
        print(f"processed {clip['video_id']} frames={len(frames)} asr={transcript['status']} {result['elapsed_seconds']}s", flush=True)


def audit(data):
    validate_manifest(data)
    state = read_json(OUTPUT / "media-state.json")
    if state["manifest_sha256"] != sha(MANIFEST):
        raise ValueError("Stale media state")
    summaries, hashes, near_pairs, errors = [], {}, [], []
    for clip in data["clips"]:
        video_id = clip["video_id"]
        path = ROOT / clip["video_path"]
        digest = sha(path)
        if digest != state["clips"][video_id]["sha256"]:
            errors.append(f"{video_id}: media checksum mismatch")
        if digest in hashes:
            errors.append(f"{video_id}: identical clip {hashes[digest]}")
        hashes[digest] = video_id
        record = read_json(OUTPUT / "features" / video_id / "preprocess.json")
        if record["media_sha256"] != digest or record["manifest_sha256"] != sha(MANIFEST):
            errors.append(f"{video_id}: stale preprocessing")
        for frame in record["frames"]:
            if sha(OUTPUT / "features" / video_id / "frames" / frame["file"]) != frame["sha256"]:
                errors.append(f"{video_id}: changed frame")
            if not 0 <= frame["clip_seconds"] < record["duration_seconds"]:
                errors.append(f"{video_id}: frame timestamp outside clip")
            if abs(frame["source_seconds"] - clip["start_seconds"] - frame["clip_seconds"]) > 1e-6:
                errors.append(f"{video_id}: source timestamp mismatch")
        summaries.append({"video_id": video_id, "split": clip["split"], "group_id": clip["group_id"],
                          "source_id": clip["source_id"], "frame_count": len(record["frames"]),
                          "asr_status": record["transcript"]["status"],
                          "language": record["transcript"]["language"],
                          "language_probability": record["transcript"]["language_probability"],
                          "asr_segment_count": len(record["transcript"]["segments"]),
                          "asr_timing_clipped_count": sum(s.get("timing_clipped", False) for s in record["transcript"]["segments"]),
                          "asr_rejected_interval_count": len(record["transcript"].get("rejected_intervals", [])),
                          "duration_seconds": record["duration_seconds"], "size_bytes": path.stat().st_size,
                          "sha256": digest, "processing_seconds": record["elapsed_seconds"]})
    for a, b in combinations(data["clips"], 2):
        if a["split"] == b["split"]:
            continue
        fa = read_json(OUTPUT / "features" / a["video_id"] / "preprocess.json")["frames"]
        fb = read_json(OUTPUT / "features" / b["video_id"] / "preprocess.json")["frames"]
        matches = {(x["dhash"], y["dhash"]) for x in fa for y in fb if x["gray_stddev"] > 12 and y["gray_stddev"] > 12
                   and (int(x["dhash"], 16) ^ int(y["dhash"], 16)).bit_count() <= 2}
        if len({x for x, _ in matches}) >= 2 and len({y for _, y in matches}) >= 2:
            near_pairs.append({"a": a["video_id"], "b": b["video_id"], "matching_hash_pairs": len(matches)})
    result = {"dataset_version": data["dataset_version"], "manifest_sha256": sha(MANIFEST),
              "audited_at": datetime.now(timezone.utc).isoformat(), "clips": len(summaries),
              "parent_sources": len(data["sources"]), "groups": len({c["group_id"] for c in data["clips"]}),
              "split_counts": dict(Counter(c["split"] for c in data["clips"])),
              "total_frames": sum(c["frame_count"] for c in summaries),
              "total_asr_segments": sum(c["asr_segment_count"] for c in summaries),
              "total_processing_seconds": round(sum(c["processing_seconds"] for c in summaries), 3),
              "total_media_bytes": sum(c["size_bytes"] for c in summaries),
              "errors": errors, "cross_split_near_duplicate_candidates": near_pairs,
              "near_duplicate_method": "1fps dHash; >=2 distinct nonflat hashes per clip at Hamming distance <=2. Heuristic, not proof of absence.",
              "human_labels_completed": 0, "predictions_run": False, "records": summaries}
    write_json(ROOT / "reports/pilot_data_audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "records"}, ensure_ascii=True))
    if errors or near_pairs:
        raise ValueError("Pilot audit needs review; see reports/pilot_data_audit.json")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "preprocess", "audit"])
    args = parser.parse_args()
    data = load_manifest()
    {"prepare": prepare, "preprocess": preprocess, "audit": audit}[args.command](data)


if __name__ == "__main__":
    main()
