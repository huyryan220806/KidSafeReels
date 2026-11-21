"""ML lead replaces the model branch without changing the API contract."""

from pathlib import Path
from uuid import uuid4

from kidsafe.schemas import AnalysisResult, RiskScores
from kidsafe.errors import ModelNotReadyError


def analyze_video(path: Path, *, backend: str = "mock") -> AnalysisResult:
    if backend == "model":
        try:
            from ml.inference import analyze
            return analyze(path)
        except ImportError as exc:
            raise ModelNotReadyError("Thiếu thư viện video ML. Xem docs/VIDEO_PIPELINE.md.") from exc
    if backend != "mock":
        raise ValueError(f"Unsupported backend: {backend}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return AnalysisResult(
        analysis_id=str(uuid4()),
        mode="mock",
        model_version="mock-v1",
        decision="review",
        scores=RiskScores(),
        warnings=[
            "MÔ PHỎNG: chưa giải mã hoặc phân tích nội dung video.",
            "Chưa có model; mọi điểm số đều chưa xác định, không phải bằng 0.",
        ],
        processing_seconds=0,
    )
