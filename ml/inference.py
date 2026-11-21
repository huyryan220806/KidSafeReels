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
from ml.toxic_rules import MODES, lexical_decision, load_rules
from ml.asr_recovery import VERSION as ASR_RECOVERY_VERSION, recover_repetition

LOCK = threading.Lock()
DEFAULT_MODELS = Path(__file__).resolve().parents[1] / "models/checkpoints"
ASR_FILTER_VERSION = "asr-native-filter-v2"


def combine_result(visual_score, visual_threshold, audio_segments, text_model, duration, versions, warnings, visual_frame_count=8, *, toxic_rules=None):
    evidence = []
    flagged = visual_score >= visual_threshold
    if flagged:
        evidence.append(Evidence(
            modality="visual", start_seconds=0, end_seconds=duration, label="violence", score=visual_score,
            description=f"Điểm Violence ở cấp toàn video từ tối đa {visual_frame_count} frame lấy mẫu; chưa định vị cảnh bạo lực."))
    text_scores = []
    for segment in audio_segments:
        start, end, text = segment[:3]
        quality = segment[3] if len(segment) > 3 else None
        if not text.strip():
            continue
        prediction = text_model.predict(text)
        text_scores.append(prediction["score"])
        lexical = lexical_decision(toxic_rules.match(text), quality) if toxic_rules else None
        descriptions = []
        if prediction["flagged"]:
            descriptions.append("Model Toxic vượt ngưỡng.")
        if lexical and lexical["matches"]:
            terms = ", ".join(f'“{m["term"]}”' for m in lexical["matches"])
            if lexical["flagged"]:
                descriptions.append(f"Bộ luật từ/cụm từ phát hiện: {terms}.")
            elif not lexical["audio_eligible"]:
                descriptions.append(f"Từ/cụm cần nghe lại: {terms}; tín hiệu ASR chưa đủ để bộ luật tự cảnh báo.")
            else:
                descriptions.append(f"Viết tắt cần xem ngữ cảnh: {terms}; bộ luật không tự cảnh báo.")
        if descriptions:
            flagged |= prediction["flagged"] or bool(lexical and lexical["flagged"])
            descriptions.append(f"Transcript ASR (có thể sai): {text.strip()}")
            descriptions.append("Mốc bao phủ đoạn ASR, không phải thời điểm chính xác của từng từ.")
            evidence.append(Evidence(
                modality="audio_text", start_seconds=max(0, min(duration, start)),
                end_seconds=max(0, min(duration, max(start, end))), label="toxic_language",
                score=prediction["score"], description=" ".join(descriptions)))
    scores = RiskScores(violence=visual_score, toxic_language=max(text_scores) if text_scores else None)
    return AnalysisResult(
        analysis_id=str(uuid4()), mode="model", model_version=versions,
        decision="flagged" if flagged else "review", scores=scores,
        evaluated_labels=[key for key, score in scores.model_dump().items() if score is not None],
        evidence=evidence,
        warnings=warnings + ([
            "Cảnh báo Toxic kết hợp model và bộ luật từ/cụm từ. Chỉ số Toxic là điểm model, không phải điểm bộ luật; bộ luật có thể cảnh báo khi điểm model thấp.",
            "Bộ luật đối chiếu transcript tự động; hãy nghe lại đoạn được chỉ ra vì ASR có thể chép sai. Gợi ý xem lại không tự tạo cảnh báo.",
        ] if toxic_rules else []) + [
            "Bản thử nghiệm: chưa đánh giá toàn pipeline trên video ngắn tiếng Việt; dưới ngưỡng vẫn cần kiểm tra.",
            "Sexual và Dangerous Acts chưa được hỗ trợ.",
            "Toxic lấy điểm cao nhất theo đoạn ASR; ngưỡng mới được chọn trên bình luận, chưa hiệu chỉnh cho video.",
        ], processing_seconds=0,
    )


class VideoService:
    def __init__(self, directory, *, toxic_mode=None):
        from ml.inference_config import load_config
        directory = Path(directory)
        self.toxic_mode = toxic_mode if toxic_mode is not None else os.getenv("KIDSAFE_TOXIC_MODE", "hybrid")
        if self.toxic_mode not in MODES:
            raise ModelNotReadyError("KIDSAFE_TOXIC_MODE phải là hybrid hoặc model_only.")
        try:
            self.toxic_rules = load_rules() if self.toxic_mode == "hybrid" else None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise ModelNotReadyError("Không nạp được bộ luật Toxic; kiểm tra code và lexicon.") from exc
        config = load_config(directory)
        for name in ("resnet18-imagenet1k-v1.pth", config["violence"], config["toxic"],
                     f"{config['asr_directory']}/model.bin", f"{config['asr_directory']}/tokenizer.json"):
            if not (directory / name).is_file():
                raise ModelNotReadyError(f"Thiếu artifact local: {name}. Xem docs/VIDEO_PIPELINE.md.")
        from faster_whisper import WhisperModel
        from ml.deep_inference import load_text_classifier
        from ml.visual import ViolenceClassifier
        try:
            self.visual = ViolenceClassifier(directory / config["violence"])
            self.text = load_text_classifier(directory / config["toxic"])
            self.asr_options = config["asr_options"]
            self.asr_version = config["asr_version"]
            self.asr_recovery_enabled = (config["asr_directory"] == "faster-whisper-small"
                                         and self.asr_options == {"beam_size": 5, "temperature": 0})
            self.asr = WhisperModel(str(directory / config["asr_directory"]), device="cpu", compute_type="int8",
                                    cpu_threads=4, local_files_only=True)
        except (OSError, ValueError, RuntimeError, KeyError, EOFError, pickle.UnpicklingError) as exc:
            raise ModelNotReadyError("Không nạp được artifact model local; kiểm tra phiên bản và checksum.") from exc

    def analyze(self, path):
        from ml.media import inspect_video, sample_frames
        info = inspect_video(path)
        frame_count = getattr(self.visual, "frame_count", 8)
        if getattr(self.visual, "is_temporal", False):
            visual_score = self.visual.score_video(path)
        else:
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
                    # faster-whisper already filters silence jointly with avg_logprob.
                    # A second no_speech_prob-only gate discards accepted speech, e.g. rap.
                    decoded = [dict(start=s.start, end=s.end, text=s.text, avg_logprob=s.avg_logprob,
                                    compression_ratio=getattr(s, "compression_ratio", None),
                                    no_speech_prob=getattr(s, "no_speech_prob", None))
                               for s in chunks if s.text.strip()]
                    if getattr(self, "asr_recovery_enabled", False):
                        decoded, recovery_warning = recover_repetition(self.asr, audio, decoded,
                            language=metadata.language, language_probability=metadata.language_probability)
                        if recovery_warning:
                            warnings.append(recovery_warning)
                    segments = [(s['start'], s['end'], s['text'],
                                 {"avg_logprob": s['avg_logprob'], "compression_ratio": s['compression_ratio']})
                                for s in decoded]
                    if not segments:
                        warnings.append("Không nhận được lời nói đủ rõ; Toxic chưa được đánh giá.")
            except InvalidMediaError:
                raise
            except (av.error.FFmpegError, RuntimeError, ValueError) as exc:
                warnings.append(f"Không hoàn thành ASR ({type(exc).__name__}); Toxic chưa được đánh giá.")
        rules = getattr(self, "toxic_rules", None)
        version = f"{self.visual.version}+{self.text.version}+{getattr(self, 'asr_version', 'whisper-base')}+{ASR_FILTER_VERSION}"
        if getattr(self, "asr_recovery_enabled", False):
            version += "+" + ASR_RECOVERY_VERSION
        if rules:
            version += "+" + rules.version
        return combine_result(visual_score, self.visual.threshold, segments, self.text, info.duration,
                              version, warnings, visual_frame_count=frame_count, toxic_rules=rules)


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
