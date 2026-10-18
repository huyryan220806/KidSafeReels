"""Exercise the running mock API using generated video fixtures and record real latency."""

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import platform
from time import perf_counter

import requests

from apps.validation import validate_result


def main():
    parser = argparse.ArgumentParser(description="Smoke test API mock đang chạy bằng video tổng hợp.")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--videos", type=Path, default=Path("data/private/app-demo"))
    parser.add_argument("--output", type=Path, default=Path("data/private/app-demo/smoke.json"))
    args = parser.parse_args()
    health = requests.get(f"{args.url}/health", timeout=5)
    health.raise_for_status()
    if health.json().get("backend") != "mock":
        raise SystemExit("Smoke test nay chi danh cho backend mock; khong chay thay danh gia model.")
    rows = [{"case": "health", "expected": 200, "actual": health.status_code}]
    for name, expected in [
        ("01-khong-am-thanh.mp4", 200), ("02-co-am-thanh.mp4", 200),
        ("03-video.mov", 200), ("04-video.webm", 200),
        ("05-dung-60-giay.mp4", 200), ("06-qua-61-giay.mp4", 422),
        ("07-file-hong.mp4", 422), ("08-file-rong.mp4", 400),
    ]:
        with (args.videos / name).open("rb") as video:
            start = perf_counter()
            response = requests.post(f"{args.url}/v1/analyze", files={"file": (name, video)}, timeout=225)
            elapsed = perf_counter() - start
        row = {"case": name, "expected": expected, "actual": response.status_code, "http_seconds": round(elapsed, 4)}
        if response.status_code == 200:
            result = validate_result(response.json(), backend="mock")
            row["server_seconds"] = result.processing_seconds
        rows.append(row)
    for name, payload, expected in [("missing-file", None, 422), ("wrong-extension", ("sample.txt", b"test"), 415)]:
        response = requests.post(f"{args.url}/v1/analyze", files={"file": payload} if payload else None, timeout=10)
        rows.append({"case": name, "expected": expected, "actual": response.status_code})
    report = {
        "recorded_at": datetime.now(timezone(timedelta(hours=7))).isoformat(),
        "python": platform.python_version(), "platform": platform.platform(),
        "scope": "Real media decoding + mock API only. Not model inference or ML quality evaluation.",
        "passed": sum(row["expected"] == row["actual"] for row in rows), "total": len(rows), "cases": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{report['passed']}/{report['total']} passed. Report: {args.output.resolve()}")
    if report["passed"] != report["total"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
