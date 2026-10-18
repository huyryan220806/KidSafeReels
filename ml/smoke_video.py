"""Exercise real local models through HTTP; no accuracy claims from smoke cases."""
import argparse
from fractions import Fraction
import io
import json
import math
import os
from pathlib import Path
import platform
import tarfile
import time

import av
import numpy as np
import requests
from fastapi.testclient import TestClient
from faster_whisper.audio import decode_audio

from ml.toxic_baseline import ROOT, sha256

FIXTURES = ROOT / "data/processed/smoke"
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"


def write_video(path, seconds, audio=None):
    with av.open(str(path), "w") as out:
        video = out.add_stream("libx264", rate=10)
        video.width, video.height, video.pix_fmt = 256, 144, "yuv420p"
        sound = out.add_stream("aac", rate=16000) if audio is not None else None
        if sound:
            sound.layout = "mono"
        for i in range(math.ceil(seconds * 10)):
            pixels = np.full((144, 256, 3), (50, 90, 120), dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts, frame.time_base = i, Fraction(1, 10)
            for packet in video.encode(frame):
                out.mux(packet)
        for packet in video.encode():
            out.mux(packet)
        if sound:
            for i in range(0, len(audio), 1024):
                frame = av.AudioFrame.from_ndarray(audio[i:i + 1024][None, :], format="fltp", layout="mono")
                frame.sample_rate, frame.pts, frame.time_base = 16000, i, Fraction(1, 16000)
                for packet in sound.encode(frame):
                    out.mux(packet)
            for packet in sound.encode():
                out.mux(packet)


def prepare():
    FIXTURES.mkdir(parents=True, exist_ok=True)
    raw = ROOT / "data/raw/fleurs"
    raw.mkdir(parents=True, exist_ok=True)
    base = f"https://huggingface.co/datasets/google/fleurs/resolve/{FLEURS_REVISION}/data/vi_vn"
    # Stream only the first WAV, without downloading or extracting the full archive.
    with requests.get(f"{base}/audio/dev.tar.gz", stream=True, timeout=(15, 90)) as response:
        response.raise_for_status()
        with tarfile.open(fileobj=response.raw, mode="r|gz") as archive:
            for member in archive:
                if member.isfile() and member.name.endswith(".wav"):
                    if member.size > 5 * 1024 * 1024:
                        raise ValueError("Unexpected speech fixture size")
                    wav = raw / "first-dev.wav"
                    wav.write_bytes(archive.extractfile(member).read())
                    name = Path(member.name).name
                    break
            else:
                raise ValueError("No WAV in FLEURS archive")
    response = requests.get(f"{base}/dev.tsv", timeout=(15, 60))
    response.raise_for_status()
    matching = [line.split("\t") for line in response.text.splitlines() if name in line.split("\t")]
    provenance = {"dataset": "google/fleurs", "subset": "vi_vn", "split": "dev", "filename": name,
                  "revision": FLEURS_REVISION, "license": "CC-BY-4.0", "sha256": sha256(wav),
                  "source": "https://huggingface.co/datasets/google/fleurs", "tsv_row": matching}
    (raw / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    audio = decode_audio(str(wav), sampling_rate=16000)
    if not 0 < len(audio) <= 60 * 16000:
        raise ValueError("Speech fixture exceeds the short-video limit")
    write_video(FIXTURES / "vietnamese-speech.mp4", len(audio) / 16000, audio)
    write_video(FIXTURES / "silent.mp4", 2)
    write_video(FIXTURES / "too-long.mp4", 61)
    print(f"Prepared fixtures; Vietnamese audio: {name}, {len(audio) / 16000:.2f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Download one FLEURS WAV and create local fixtures")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.prepare or args.prepare_only:
        prepare()
    if args.prepare_only:
        return
    from apps.api.main import app
    os.environ["KIDSAFE_BACKEND"] = "model"
    manifest = json.loads((ROOT / "data/processed/rlvs/manifest.json").read_text(encoding="utf-8"))
    rows = manifest["rows"]
    selected = [r for r in rows if r["split"] == "test" and r["label"] == 0][:3]
    selected += [r for r in rows if r["split"] == "test" and r["label"] == 1][:2]
    if len(selected) != 5:
        raise ValueError("Need five held-out RLVS clips for the smoke test")
    cases = [(f"rlvs-{i}", ROOT / "data/raw/rlvs" / r["path"], 200) for i, r in enumerate(selected)]
    cases += [("silent", FIXTURES / "silent.mp4", 200),
              ("vietnamese-speech", FIXTURES / "vietnamese-speech.mp4", 200),
              ("too-long", FIXTURES / "too-long.mp4", 422)]
    results = []
    with TestClient(app) as client:
        for name, path, expected in cases:
            with path.open("rb") as stream:
                started = time.perf_counter()
                response = client.post("/v1/analyze", files={"file": (path.name, stream, "video/mp4")})
                seconds = time.perf_counter() - started
            payload = response.json()
            assert response.status_code == expected, (name, payload)
            if expected == 200:
                assert payload["mode"] == "model"
                assert payload["scores"]["violence"] is not None
                assert payload["scores"]["sexual"] is None
                if name == "silent":
                    assert payload["scores"]["toxic_language"] is None
                if name == "vietnamese-speech":
                    assert payload["scores"]["toxic_language"] is not None, payload
            results.append({"case": name, "input_sha256": sha256(path), "status": response.status_code,
                            "request_seconds": seconds, "result": payload})
            print(f"{name}: HTTP {response.status_code}, {seconds:.2f}s", flush=True)
        response = client.post("/v1/analyze", files={"file": ("corrupt.mp4", io.BytesIO(b"corrupt"), "video/mp4")})
        assert response.status_code == 422, response.text
        results.append({"case": "corrupt", "status": response.status_code, "result": response.json()})
    report = {"scope": "FastAPI TestClient in-process integration; not browser/network testing, accuracy benchmark or representative latency estimate",
              "environment": {"python": platform.python_version(), "platform": platform.platform(), "device": "cpu"},
              "sources": {"rlvs": "First 3 non-violence and 2 violence held-out manifest rows; no selection by prediction",
                          "speech": json.loads((ROOT / "data/raw/fleurs/provenance.json").read_text(encoding="utf-8"))},
              "cases": results}
    (ROOT / "reports/video_smoke_results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
