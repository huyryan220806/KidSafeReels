"""CPU video inference with explicit scope and no implicit model downloads."""
import argparse
from functools import lru_cache
import os
import pickle
from pathlib import Path
import sys
import threading
from time import perf_counter
from uuid import uuid4

from kidsafe.errors import InvalidMediaError, ModelNotReadyError
from kidsafe.schemas import AnalysisResult, Evidence, RiskScores

LOCK = threading.Lock()
DEFAULT_MODELS = Path(__file__).resolve().parents[1] / "models/checkpoints"


def combine_result(visual_score, visual_threshold, audio_segments, text_model, duration, versions, warnings, visual_frame_count=8):
    evidence = []
    flagged = visual_score >= visual_threshold
    if flagged:
        evidence.append(Evidence(
            modality="visual", start_seconds=0, end_seconds=duration, label="violence", score=visual_score,
            description=f"Điểm Violence ở cấp toàn video từ tối đa {visual_frame_count} frame lấy mẫu; chưa định vị cảnh bạo lực."))
    text_scores = []
    for start, end, text in audio_segments:
        if not text.strip():
            continue
        prediction = text_model.predict(text)
        text_scores.append(prediction["score"])
        if prediction["flagged"]:
            flagged = True
            evidence.append(Evidence(
                modality="audio_text", start_seconds=max(0, min(duration, start)),
                end_seconds=max(0, min(duration, max(start, end))), label="toxic_language",
                score=prediction["score"], description=f"Đoạn ASR được model đánh dấu: {text.strip()}"))
    scores = RiskScores(violence=visual_score, toxic_language=max(text_scores) if text_scores else None)
    return AnalysisResult(
        analysis_id=str(uuid4()), mode="model", model_version=versions,
        decision="flagged" if flagged else "review", scores=scores,
        evaluated_labels=[key for key, score in scores.model_dump().items() if score is not None],
        evidence=evidence,
        warnings=warnings + [
            "Bản thử nghiệm: chưa đánh giá toàn pipeline trên video ngắn tiếng Việt; dưới ngưỡng vẫn cần kiểm tra.",
            "Sexual và Dangerous Acts chưa được hỗ trợ.",
            "Toxic lấy điểm cao nhất theo đoạn ASR; ngưỡng mới được chọn trên bình luận, chưa hiệu chỉnh cho video.",
        ], processing_seconds=0,
    )


class VideoService:
    def __init__(self, directory):
        from ml.inference_config import load_config
        directory = Path(directory)
        config = load_config(directory)
        for name in ("resnet18-imagenet1k-v1.pth", config["violence"], config["toxic"],
                     f"{config['asr_directory']}/model.bin", f"{config['asr_directory']}/tokenizer.json"):
            if not (directory / name).is_file():
                raise ModelNotReadyError(f"Thiếu artifact local: {name}. Xem docs/VIDEO_PIPELINE.md.")
        from faster_whisper import WhisperModel
        from ml.toxic_baseline import ToxicClassifier
        from ml.visual import ViolenceClassifier
        try:
            self.visual = ViolenceClassifier(directory / config["violence"])
            self.text = ToxicClassifier(directory / config["toxic"])
            self.asr_options = config["asr_options"]
            self.asr_version = config["asr_version"]
            self.asr = WhisperModel(str(directory / config["asr_directory"]), device="cpu", compute_type="int8",
                                    cpu_threads=4, local_files_only=True)
        except (OSError, ValueError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
            raise ModelNotReadyError("Không nạp được artifact model local; kiểm tra phiên bản và checksum.") from exc

    def analyze(self, path):
        from ml.media import inspect_video, sample_frames
        info = inspect_video(path)
        frame_count = getattr(self.visual, "frame_count", 8)
        frames = sample_frames(path) if frame_count == 8 else sample_frames(path, count=frame_count)
        visual_score = self.visual.score(frames)
        warnings, segments = [], []
        if not info.has_audio:
            warnings.append("Video không có audio; Toxic chưa được đánh giá.")
        else:
            from faster_whisper.audio import decode_audio
            import av
            try:
                audio = decode_audio(str(path), sampling_rate=16000)
                if len(audio) > 60 * 16000 + 16000:
                    raise InvalidMediaError("Luồng audio dài quá giới hạn.")
                if not len(audio):
                    raise ValueError("Empty decoded audio")
                chunks, metadata = self.asr.transcribe(audio, task="transcribe", vad_filter=True,
                    condition_on_previous_text=False, **getattr(self, "asr_options", {"beam_size": 1}))
                if metadata.language != "vi" or metadata.language_probability < 0.5:
                    warnings.append("Không xác nhận được lời nói tiếng Việt; Toxic chưa được đánh giá.")
                else:
                    segments = [(s.start, s.end, s.text) for s in chunks if s.no_speech_prob < 0.6]
                    if not segments:
                        warnings.append("Không nhận được lời nói đủ rõ; Toxic chưa được đánh giá.")
            except InvalidMediaError:
                raise
            except (av.error.FFmpegError, RuntimeError, ValueError) as exc:
                warnings.append(f"Không hoàn thành ASR ({type(exc).__name__}); Toxic chưa được đánh giá.")
        return combine_result(visual_score, self.visual.threshold, segments, self.text, info.duration,
                              f"{self.visual.version}+{self.text.version}+{getattr(self, 'asr_version', 'whisper-base')}",
                              warnings, visual_frame_count=frame_count)


@lru_cache(maxsize=1)
def get_service(directory):
    return VideoService(directory)


def analyze(path):
    started = perf_counter()
    directory = os.getenv("KIDSAFE_MODEL_DIR", str(DEFAULT_MODELS))
    # Serialize local inference to bound memory; app keeps using synchronous API.
    with LOCK:
        result = get_service(directory).analyze(path)
    result.processing_seconds = round(perf_counter() - started, 4)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze(args.video).model_dump_json(indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result, encoding="utf-8")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        print(result)


if __name__ == "__main__":
    main()
