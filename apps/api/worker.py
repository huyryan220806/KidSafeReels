"""One reusable process per API instance; timeouts terminate work before file cleanup."""

import logging
import multiprocessing
import os
from pathlib import Path
import signal
import threading

from apps.validation import validate_result

logger = logging.getLogger(__name__)


class AnalysisError(RuntimeError):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code


def worker_loop(connection):
    # Parent owns shutdown; Ctrl+C must not interrupt a worker mid-write.
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    from apps.api.media import MediaError, inspect_video

    try:
        while True:
            request = connection.recv()
            if request is None:
                return
            path, backend = request
            try:
                media = inspect_video(Path(path))
            except MediaError as exc:
                connection.send(("error", exc.status_code, str(exc)))
                continue
            connection.send(("media",))
            from kidsafe.pipeline import ModelNotReadyError, analyze_video
            from kidsafe.errors import InvalidMediaError

            try:
                result = analyze_video(Path(path), backend=backend)
            except InvalidMediaError as exc:
                connection.send(("error", 422, str(exc)))
                continue
            except ModelNotReadyError:
                connection.send(("error", 503, "Model chưa sẵn sàng. Kiểm tra checkpoint và hướng dẫn bàn giao của Huy."))
                continue
            except Exception:
                logger.exception("Pipeline failed; no mock fallback was used")
                connection.send(("error", 500, "Phân tích thất bại. Kiểm tra log API rồi thử lại."))
                continue
            try:
                result = validate_result(result, backend=backend, duration=media.duration)
                if backend == "mock":
                    result.warnings = [
                        "MÔ PHỎNG: video đã được kiểm tra kỹ thuật; chưa phân tích rủi ro nội dung."
                        if warning == "MÔ PHỎNG: chưa giải mã hoặc phân tích nội dung video." else warning
                        for warning in result.warnings
                    ]
                if not media.has_audio:
                    if any(item.modality == "audio_text" for item in result.evidence):
                        raise ValueError("Audio evidence without an audio track")
                    result.warnings.append("Video không có luồng âm thanh; không thể đánh giá lời nói từ audio.")
                connection.send(("result", result.model_dump()))
            except (ValueError, TypeError):
                connection.send(("error", 502, "Pipeline trả kết quả không đúng hợp đồng API. Cần kiểm tra tích hợp model."))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


def timeout_setting(name, default):
    try:
        value = float(os.getenv(name, default))
        if not 0 < value <= 600:
            raise ValueError
        return value
    except ValueError as exc:
        raise AnalysisError(503, f"Cấu hình {name} phải lớn hơn 0 và không quá 600 giây.") from exc


class AnalyzerWorker:
    def __init__(self, target=worker_loop):
        self._target = target
        self._lock = threading.Lock()
        self._process = None
        self._connection = None

    def _start(self):
        if self._process is not None and self._process.is_alive():
            return
        self.close()
        context = multiprocessing.get_context("spawn")
        parent, child = context.Pipe()
        process = context.Process(target=self._target, args=(child,), daemon=True)
        try:
            process.start()
        except Exception:
            parent.close()
            child.close()
            raise
        child.close()
        self._process, self._connection = process, parent

    def close(self):
        if self._process is not None:
            if self._process.is_alive():
                self._process.terminate()
            self._process.join(timeout=5)
            if self._process.is_alive():
                self._process.kill()
                self._process.join()
            self._process.close()
        if self._connection is not None:
            self._connection.close()
        self._process = self._connection = None

    def _receive(self, seconds, message):
        if not self._connection.poll(seconds):
            self.close()
            raise AnalysisError(504, message)
        response = self._connection.recv()
        if response[0] == "error":
            raise AnalysisError(response[1], response[2])
        return response

    def analyze(self, path, backend):
        if backend not in {"mock", "model"}:
            raise AnalysisError(503, "KIDSAFE_BACKEND chỉ nhận mock hoặc model.")
        media_timeout = timeout_setting("KIDSAFE_MEDIA_TIMEOUT", 30)
        model_timeout = timeout_setting("KIDSAFE_MODEL_TIMEOUT", 180)
        if not self._lock.acquire(blocking=False):
            raise AnalysisError(429, "Máy đang phân tích video khác. Hãy chờ hoàn tất rồi thử lại.")
        try:
            self._start()
            self._connection.send((str(path), backend))
            response = self._receive(media_timeout, "Kiểm tra video quá thời gian cho phép. Hãy giảm độ phân giải hoặc xuất lại video.")
            if response[0] != "media":
                raise RuntimeError("Unexpected worker response")
            response = self._receive(model_timeout, "Model xử lý quá thời gian cho phép. Hãy dùng video ngắn hơn hoặc kiểm tra cấu hình model.")
            if response[0] != "result":
                raise RuntimeError("Unexpected worker response")
            return validate_result(response[1], backend=backend)
        except AnalysisError:
            raise
        except Exception as exc:
            self.close()
            logger.exception("Analysis worker failed")
            raise AnalysisError(500, "Tiến trình phân tích bị lỗi. Bạn có thể thử lại.") from exc
        finally:
            self._lock.release()
