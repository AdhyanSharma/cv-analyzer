"""Pydantic request/response schemas for the CV Analyzer FastAPI backend (V10)."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ResumeTextInput(BaseModel):
    filename: str = Field(..., min_length=1, max_length=255)
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def clean_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Resume text cannot be empty.")
        return value


class AnalyzeTextRequest(BaseModel):
    job_description: str = Field(..., min_length=1)
    resumes: List[ResumeTextInput] = Field(..., min_length=1, max_length=50)
    top_n_keywords: int = Field(default=20, ge=5, le=100)
    include_raw_text: bool = False

    @field_validator("job_description")
    @classmethod
    def clean_jd(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Job description cannot be empty.")
        return value


class AnalyzeFileResponse(BaseModel):
    filename: str
    extracted_characters: int


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class APIError(BaseModel):
    detail: str


class GenericDictResponse(BaseModel):
    data: Dict[str, Any]


class AnalysisResponse(BaseModel):
    job_description: Dict[str, Any]
    candidates: List[Dict[str, Any]]
    meta: Dict[str, Any]
