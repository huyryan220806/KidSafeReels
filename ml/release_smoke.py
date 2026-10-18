"""Exercise real HTTP, spawned app worker and real CPU models without pilot labels.

Only train clips are sent. Public output contains technical checks, no predictions.
"""
import json
import os
from pathlib import Path
import socket
import threading
from time import perf_counter, sleep

from ml.pilot_data import ROOT, read_json, write_json
from ml.release_verify import verify_artifacts


def make_fixtures(directory):
    import subprocess
    import imageio_ffmpeg
    directory.mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    for name, seconds, audio in (("silent.mp4", 2, False), ("tone.mp4", 2, True), ("long.mp4", 61, False)):
        command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-f", "lavfi", "-i",
                   f"color=c=gray:s=160x120:r=10:d={seconds}"]
        if audio:
            command += ["-f", "lavfi", "-i", "sine=frequency=440:sample_rate=16000:duration=2", "-c:a", "aac"]
        command += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-threads", "2", str(directory / name)]
        subprocess.run(command, check=True, capture_output=True, timeout=60)
    (directory / "broken.mp4").write_bytes(b"invalid media fixture")


def main():
    import numpy as np
    import requests
    import uvicorn
    from kidsafe.schemas import AnalysisResult
    verify_artifacts()
    os.environ["KIDSAFE_BACKEND"] = "model"
    os.environ["KIDSAFE_MODEL_DIR"] = str(ROOT / "models/checkpoints")
    private = ROOT / "data/private/release-smoke"
    make_fixtures(private)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config("apps.api.main:app", host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        for _ in range(150):
            if server.started:
                break
            sleep(.1)
        if not server.started:
            raise RuntimeError("HTTP server did not start")
        base = f"http://127.0.0.1:{port}"
        health = requests.get(base + "/health", timeout=5).json()
        assert health["backend"] == "model"
        rows = []
        cases = [(f"pilot-train-{n:03d}", ROOT / f"data/processed/pilot-v1/media/vi-pilot-{n:03d}.mp4", 200, None) for n in range(5, 10)]
        cases += [("synthetic-silent", private / "silent.mp4", 200, None),
                  ("synthetic-tone", private / "tone.mp4", 200, None),
                  ("too-long", private / "long.mp4", 422, None),
                  ("corrupt-media", private / "broken.mp4", 422, None),
                  ("unsupported-extension", private / "silent.mp4", 415, "clip.exe"),
                  ("container-extension-mismatch", private / "silent.mp4", 415, "clip.webm")]
        for name, path, expected_status, filename in cases:
            started = perf_counter()
            with path.open("rb") as handle:
                response = requests.post(base + "/v1/analyze", files={"file": (filename or path.name, handle)}, timeout=230)
            elapsed = perf_counter() - started
            payload = response.json()
            write_json(private / (name + ".json"), payload)
            valid = response.status_code == expected_status
            if response.status_code == 200:
                result = AnalysisResult.model_validate(payload)
                valid &= result.mode == "model" and result.scores.violence is not None
                valid &= result.scores.sexual is None and result.scores.dangerous_acts is None
                if name == "synthetic-silent":
                    valid &= result.scores.toxic_language is None and not any(e.modality == "audio_text" for e in result.evidence)
                if name.startswith("pilot"):
                    valid &= result.scores.toxic_language is not None
            rows.append({"case": name, "expected_http": expected_status, "actual_http": response.status_code,
                         "passed": bool(valid), "request_seconds": round(elapsed, 4),
                         "cold_worker": len(rows) == 0})
            print(json.dumps(rows[-1]), flush=True)
        response = requests.post(base + "/v1/analyze", files={"file": ("empty.mp4", b"")}, timeout=10)
        rows.append({"case": "empty", "expected_http": 400, "actual_http": response.status_code, "passed": response.status_code == 400})
        response = requests.post(base + "/v1/analyze", timeout=10)
        rows.append({"case": "missing-file-field", "expected_http": 422, "actual_http": response.status_code, "passed": response.status_code == 422})
        warm = [r["request_seconds"] for r in rows if r.get("case", "").startswith("pilot") and not r["cold_worker"]]
        report = {"date": "2026-10-18", "transport": "HTTP loopback + actual spawned worker + CPU model artifacts",
                  "app_commit_integrated": "d82798d", "artifact_verification": verify_artifacts(),
                  "pilot_scope": "train only; no gold or moderation quality claims; raw outputs private",
                  "warm_25_second_clip_count": len(warm), "warm_median_seconds": float(np.median(warm)),
                  "warm_p95_seconds": float(np.percentile(warm, 95)),
                  "latency_limit": "Four warm clips from one group, one run each; descriptive only, not representative performance",
                  "passed": sum(r["passed"] for r in rows), "total": len(rows), "cases": rows}
        write_json(ROOT / "reports/release_smoke_results.json", report)
        assert all(r["passed"] for r in rows), "See reports/release_smoke_results.json"
    finally:
        server.should_exit = True
        thread.join(timeout=15)
        if thread.is_alive():
            raise RuntimeError("HTTP server did not stop cleanly")


if __name__ == "__main__":
    main()
