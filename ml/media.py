"""Local short-video validation and deterministic sampling."""
from dataclasses import dataclass
from pathlib import Path
import math

import av
import cv2
import numpy as np
from PIL import Image

from kidsafe.errors import InvalidMediaError


@dataclass
class VideoInfo:
    duration: float
    has_audio: bool


def inspect_video(path: Path, max_seconds=60.0):
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise InvalidMediaError("File không có luồng hình ảnh.")
            stream = container.streams.video[0]
            duration = (float(container.duration / av.time_base) if container.duration is not None
                        else float(stream.duration * stream.time_base) if stream.duration is not None else 0)
            if not math.isfinite(duration) or duration <= 0:
                raise InvalidMediaError("Không xác định được thời lượng video.")
            if duration > max_seconds:
                raise InvalidMediaError(f"Video dài quá {max_seconds:g} giây.")
            if stream.width * stream.height > 3840 * 2160:
                raise InvalidMediaError("Độ phân giải vượt giới hạn 4K.")
            return VideoInfo(duration, bool(container.streams.audio))
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        if isinstance(exc, InvalidMediaError):
            raise
        raise InvalidMediaError("Không mở được video hoặc media bị hỏng.") from exc


def sample_frames(path: Path, count=8):
    cap = cv2.VideoCapture(str(path))
    try:
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if length <= 0:
            raise InvalidMediaError("Không đọc được frame video.")
        indices = np.unique(np.linspace(0, length - 1, count).astype(int))
        frames = []
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = cap.read()
            if not ok:
                raise InvalidMediaError("Không giải mã được frame lấy mẫu.")
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        return frames
    finally:
        cap.release()


def fingerprint(frames):
    """Three 64-bit difference hashes, only a heuristic for similar clips."""
    hashes = []
    for index in (0, len(frames) // 2, len(frames) - 1):
        pixels = np.asarray(frames[index].convert("L").resize((9, 8)))
        bits = (pixels[:, 1:] > pixels[:, :-1]).ravel()
        hashes.append(sum(int(bit) << i for i, bit in enumerate(bits)))
    return hashes
