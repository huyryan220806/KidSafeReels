"""Generate synthetic codec fixtures, never ML training/evaluation data."""

import argparse
from pathlib import Path
import sys

import av
import numpy as np


def make_video(path, *, seconds=2, fps=10, audio=False, container_format=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    webm = path.suffix == ".webm"
    with av.open(str(path), "w", format=container_format) as output:
        video = output.add_stream("libvpx" if webm else "libx264", rate=fps)
        video.width, video.height = 160, 96
        video.pix_fmt = "yuv420p"
        video.options = {"deadline": "realtime"} if webm else {"preset": "ultrafast"}
        audio_stream = output.add_stream("libopus" if webm else "aac", rate=48000) if audio else None
        if audio_stream is not None:
            audio_stream.layout = "mono"
        for index in range(round(seconds * fps)):
            pixels = np.zeros((96, 160, 3), dtype=np.uint8)
            pixels[:, :, 1] = 70 + index % 120
            pixels[:, (index * 3) % 140: (index * 3) % 140 + 20, :] = (230, 190, 80)
            frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
            frame.pts = index
            for packet in video.encode(frame):
                output.mux(packet)
        for packet in video.encode():
            output.mux(packet)
        if audio_stream is not None:
            # A quiet sine wave tests the presence of an audio track, not speech recognition.
            samples = round(seconds * 48000)
            waveform = (0.03 * np.sin(2 * np.pi * 440 * np.arange(samples) / 48000)).astype(np.float32)
            for offset in range(0, samples, 1024):
                frame = av.AudioFrame.from_ndarray(waveform[None, offset:offset + 1024], format="fltp", layout="mono")
                frame.sample_rate = 48000
                frame.pts = offset
                for packet in audio_stream.encode(frame):
                    output.mux(packet)
            for packet in audio_stream.encode():
                output.mux(packet)
    return path


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Tạo video tổng hợp để kiểm thử app; không phải dữ liệu đánh giá ML.")
    parser.add_argument("--output", type=Path, default=Path("data/private/app-demo"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for name, options in {
        "01-khong-am-thanh.mp4": {},
        "02-co-am-thanh.mp4": {"audio": True},
        "03-video.mov": {},
        "04-video.webm": {},
        "05-dung-60-giay.mp4": {"seconds": 60, "fps": 2},
        "06-qua-61-giay.mp4": {"seconds": 61, "fps": 2},
    }.items():
        make_video(args.output / name, **options)
    (args.output / "07-file-hong.mp4").write_bytes(b"SYNTHETIC INVALID VIDEO")
    (args.output / "08-file-rong.mp4").write_bytes(b"")
    print(f"Đã tạo 8 file kiểm thử tổng hợp tại {args.output.resolve()}. Không dùng để báo cáo chất lượng model.")


if __name__ == "__main__":
    main()
