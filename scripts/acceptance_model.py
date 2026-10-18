"""Second-machine HTTP acceptance, using only synthetic and FLEURS fixtures.

Never reads pilot clips or annotation submissions, never overwrites ML reports.
Run against an already running model API. Detailed results remain private.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import requests

from apps.validation import validate_result
from scripts.make_demo_videos import make_video

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    private = ROOT / "data/private/app-acceptance-tien"
    private.mkdir(parents=True, exist_ok=True)
    health = requests.get(args.url + "/health", timeout=5)
    health.raise_for_status()
    if health.json().get("backend") != "model":
        raise SystemExit("Acceptance requires model backend; mock is not accepted.")
    fixtures = {
        "silent.mp4": {}, "tone.mp4": {"audio": True}, "video.mov": {}, "video.webm": {},
        "boundary.mp4": {"seconds": 60, "fps": 2}, "long.mp4": {"seconds": 61, "fps": 2},
    }
    for name, options in fixtures.items():
        make_video(private / name, **options)
    (private / "broken.mp4").write_bytes(b"APP ACCEPTANCE INVALID VIDEO")
    speech = ROOT / "data/processed/smoke/vietnamese-speech.mp4"
    if not speech.is_file():
        raise SystemExit("Prepare the licensed FLEURS speech fixture first: python -m ml.smoke_video --prepare-only")
    rows = []
    cases = [(name, private / name, 200) for name in ("silent.mp4", "tone.mp4", "video.mov", "video.webm", "boundary.mp4")]
    cases += [("vietnamese-speech", speech, 200), ("speech-warm-repeat", speech, 200),
              ("too-long", private / "long.mp4", 422), ("corrupt", private / "broken.mp4", 422)]
    for name, path, expected in cases:
        started = perf_counter()
        with path.open("rb") as handle:
            response = requests.post(args.url + "/v1/analyze", files={"file": (path.name, handle)}, timeout=230)
        elapsed = perf_counter() - started
        payload = response.json()
        (private / f"{name}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        valid = response.status_code == expected
        row = {"case": name, "expected_http": expected, "actual_http": response.status_code,
               "request_seconds": round(elapsed, 4), "cold_worker": not rows}
        if response.status_code == 200:
            result = validate_result(payload, backend="model")
            valid &= result.scores.violence is not None
            valid &= result.scores.sexual is None and result.scores.dangerous_acts is None
            if name in {"silent.mp4", "video.mov", "video.webm", "boundary.mp4"}:
                valid &= result.scores.toxic_language is None
            if "speech" in name:
                valid &= result.scores.toxic_language is not None
            row.update(mode=result.mode, decision=result.decision, evaluated_labels=result.evaluated_labels,
                       evidence_count=len(result.evidence), server_seconds=result.processing_seconds)
        row["passed"] = bool(valid)
        rows.append(row)
        print(json.dumps(row), flush=True)
    for name, files, expected in [
        ("empty", {"file": ("empty.mp4", b"")}, 400),
        ("missing-field", None, 422),
        ("wrong-extension", {"file": ("clip.txt", b"fixture")}, 415),
        ("container-mismatch", {"file": ("clip.webm", (private / "silent.mp4").read_bytes())}, 415),
    ]:
        response = requests.post(args.url + "/v1/analyze", files=files, timeout=10)
        rows.append({"case": name, "expected_http": expected, "actual_http": response.status_code,
                     "passed": response.status_code == expected})
    report = {
        "recorded_at": datetime.now().astimezone().isoformat(),
        "machine": "Tien, Intel Core i5-12450H, separate from Huy's Ryzen 5 7535HS",
        "python": platform.python_version(), "platform": platform.platform(),
        "scope": "Actual HTTP + spawned worker + CPU models. Synthetic visuals and FLEURS Vietnamese speech only. No pilot inference or quality metrics.",
        "speech_sha256": hashlib.sha256(speech.read_bytes()).hexdigest(),
        "passed": sum(row["passed"] for row in rows), "total": len(rows), "cases": rows,
    }
    output = ROOT / "reports/app_acceptance_tien_results.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['passed']}/{report['total']} acceptance cases passed. {output}", flush=True)
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
