"""ML lead replaces the model branch without changing the API contract."""

from pathlib import Path
from uuid import uuid4

from kidsafe.schemas import AnalysisResult, RiskScores


class ModelNotReadyError(RuntimeError):
    pass


def analyze_video(path: Path, *, backend: str = "mock") -> AnalysisResult:
    if backend == "model":
        raise ModelNotReadyError("Model chưa được tích hợp. Dùng mock để phát triển app.")
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
