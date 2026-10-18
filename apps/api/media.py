"""Bounded media inspection. Run only inside the supervised worker process."""

from dataclasses import dataclass
import math
from pathlib import Path

import av

MAX_DURATION = 60.0
MAX_PIXELS = 3840 * 2160
MAX_FRAMES = 18000


class MediaError(ValueError):
    def __init__(self, message, status_code=422):
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class MediaInfo:
    duration: float
    has_audio: bool
    frames: int


def inspect_video(path: Path) -> MediaInfo:
    try:
        # No network protocols or playlists: uploads must be self-contained local media.
        with av.open(str(path), options={"protocol_whitelist": "file", "err_detect": "explode"}) as container:
            formats = set(container.format.name.split(","))
            expected = {"webm", "matroska"} if path.suffix.lower() == ".webm" else {"mov", "mp4"}
            if not formats.intersection(expected):
                raise MediaError("Nội dung file không khớp định dạng MP4, MOV hoặc WebM.", 415)
            if not container.streams.video:
                raise MediaError("File không có luồng hình ảnh video.")
            if len(container.streams.video) != 1:
                raise MediaError("Chỉ hỗ trợ video có một luồng hình ảnh.")
            stream = container.streams.video[0]
            if stream.width * stream.height > MAX_PIXELS:
                raise MediaError("Độ phân giải vượt giới hạn 3840 × 2160 điểm ảnh. Hãy giảm kích thước video.")
            declared = []
            if container.duration is not None:
                declared.append(float(container.duration / av.time_base))
            if stream.duration is not None and stream.time_base:
                declared.append(float(stream.duration * stream.time_base))
            if any(not math.isfinite(value) or value > MAX_DURATION + 0.001 for value in declared):
                raise MediaError("Video vượt giới hạn 60 giây.")
            stream.thread_type = "SLICE"
            stream.codec_context.thread_count = 2
            rate = float(stream.average_rate) if stream.average_rate else 0
            fallback_step = 1 / rate if rate > 0 else 0
            first_time = None
            previous_time = None
            end_time = 0.0
            frames = 0
            for frame in container.decode(stream):
                frames += 1
                if frames > MAX_FRAMES:
                    raise MediaError("Video có quá nhiều khung hình để xử lý. Hãy giảm tốc độ khung hình.")
                if frame.width * frame.height > MAX_PIXELS:
                    raise MediaError("Khung hình vượt giới hạn độ phân giải.")
                if frame.is_corrupt:
                    raise MediaError("Video có khung hình hỏng. Hãy xuất lại file.")
                timestamp = frame.time
                if timestamp is None or not math.isfinite(timestamp):
                    raise MediaError("Không đọc được mốc thời gian của video.")
                if first_time is None:
                    first_time = timestamp
                relative = timestamp - first_time
                if previous_time is not None and relative < previous_time:
                    raise MediaError("Mốc thời gian video không hợp lệ.")
                step = float(frame.duration * frame.time_base) if frame.duration and frame.time_base else fallback_step
                if step <= 0 and previous_time is not None:
                    step = relative - previous_time
                end_time = max(end_time, relative + step)
                previous_time = relative
                if end_time > MAX_DURATION + 0.001:
                    raise MediaError("Video vượt giới hạn 60 giây.")
            if not frames or end_time <= 0:
                raise MediaError("Không giải mã được khung hình video.")
            if stream.frames and frames < stream.frames:
                raise MediaError("Video bị thiếu khung hình hoặc bị cắt hỏng.")
            duration = max([end_time, *declared])
            if duration > MAX_DURATION + 0.001:
                raise MediaError("Video vượt giới hạn 60 giây.")
            return MediaInfo(duration, bool(container.streams.audio), frames)
    except MediaError:
        raise
    except (av.FFmpegError, OSError, ValueError, OverflowError) as exc:
        raise MediaError("File video bị hỏng hoặc codec chưa được hỗ trợ. Hãy xuất lại MP4/H.264.") from exc
