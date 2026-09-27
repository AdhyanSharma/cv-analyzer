"""Pydantic schemas for V14 evaluation and audit features."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class EvaluationLabelItem(BaseModel):
    candidate_id: str = Field(..., min_length=1, max_length=36)
    label: str = Field(...)

    @field_validator("label")
    @classmethod
    def validate_label(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"qualified", "not_qualified"}:
            raise ValueError("label must be 'qualified' or 'not_qualified'.")
        return value


class EvaluationRunRequest(BaseModel):
    threshold: float = Field(default=50.0, ge=0.0, le=100.0)
    labels: List[EvaluationLabelItem] = Field(..., min_length=1, max_length=500)


class EvaluationMetrics(BaseModel):
    sample_size: int
    threshold: float
    accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1: Optional[float] = None
    true_positive: int
    true_negative: int
    false_positive: int
    false_negative: int
    predicted_qualified: int
    actual_qualified: int
    actual_not_qualified: int


class EvaluationRunResponse(BaseModel):
    id: str
    owner_id: str
    job_id: str
    created_at: str
    metrics: EvaluationMetrics


class AuditLogResponse(BaseModel):
    id: str
    owner_id: str
    action: str
    entity_type: str
    entity_id: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: str
