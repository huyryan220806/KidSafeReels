from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

RiskLabel = Literal["violence", "sexual", "toxic_language", "dangerous_acts"]
Score = Annotated[float, Field(ge=0, le=1)]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RiskScores(ContractModel):
    violence: Score | None = None
    sexual: Score | None = None
    toxic_language: Score | None = None
    dangerous_acts: Score | None = None


class Evidence(ContractModel):
    modality: Literal["visual", "audio_text", "ocr"]
    start_seconds: float = Field(ge=0)
    end_seconds: float = Field(ge=0)
    label: RiskLabel
    score: Score
    description: str

    @model_validator(mode="after")
    def valid_interval(self):
        if self.end_seconds < self.start_seconds:
            raise ValueError("Evidence end must be >= start")
        return self


class AnalysisResult(ContractModel):
    schema_version: Literal["1.0"] = "1.0"
    analysis_id: str
    mode: Literal["mock", "model"]
    model_version: str
    decision: Literal["no_risk_detected", "review", "flagged"]
    scores: RiskScores
    evaluated_labels: list[RiskLabel] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    processing_seconds: float = Field(ge=0)
