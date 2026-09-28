"""Pydantic schemas for V11 authentication and job management."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, EmailStr, Field, field_validator


class RegisterRequest(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=2, max_length=120)
    password: str = Field(..., min_length=8, max_length=128)


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)


class UserPublic(BaseModel):
    id: str
    email: str
    full_name: str
    created_at: str
    active: bool


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserPublic


class JobCreateRequest(BaseModel):
    title: str = Field(..., min_length=2, max_length=160)
    company: str = Field(default="", max_length=160)
    description: str = Field(..., min_length=20)

    @field_validator("title", "description")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty.")
        return value


class JobUpdateRequest(BaseModel):
    title: Optional[str] = Field(default=None, min_length=2, max_length=160)
    company: Optional[str] = Field(default=None, max_length=160)
    description: Optional[str] = Field(default=None, min_length=20)
    status: Optional[str] = Field(default=None)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip().lower()
        if value not in {"open", "closed"}:
            raise ValueError("status must be 'open' or 'closed'.")
        return value


class JobResponse(BaseModel):
    id: str
    owner_id: str
    title: str
    company: str
    description: str
    status: str
    created_at: str
    updated_at: str


class ApplicationStatusRequest(BaseModel):
    status: str = Field(...)

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        value = value.strip().lower()
        allowed = {"new", "reviewing", "shortlisted", "hold", "rejected"}
        if value not in allowed:
            raise ValueError(f"status must be one of: {', '.join(sorted(allowed))}.")
        return value


class CandidateApplicationResponse(BaseModel):
    application_id: str
    job_id: str
    candidate_id: str
    resume_version_id: str | None = None
    resume_version_number: int | None = None
    resume_version_filename: str | None = None
    resume_version_reused: bool = False
    status: str
    name: str
    email: str
    phone: str
    linkedin: str
    github: str
    resume_filename: str
    analysis: Dict[str, Any]
    created_at: str
    updated_at: str


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class AuthenticatedUserResponse(BaseModel):
    user: UserPublic
