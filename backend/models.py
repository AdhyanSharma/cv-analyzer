"""SQLAlchemy ORM models for the V12 recruitment platform."""
from __future__ import annotations

from sqlalchemy import (
    Boolean,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    email: Mapped[str] = mapped_column(
        String(320),
        nullable=False,
        unique=True,
        index=True,
    )
    full_name: Mapped[str] = mapped_column(String(120), nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="1",
    )

    jobs: Mapped[list["Job"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
    )
    candidates: Mapped[list["Candidate"]] = relationship(
        back_populates="owner",
        cascade="all, delete-orphan",
    )


class Job(Base):
    __tablename__ = "jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    title: Mapped[str] = mapped_column(String(160), nullable=False)
    company: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
        default="",
        server_default="",
    )
    description: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
        default="open",
        server_default="open",
    )
    created_at: Mapped[str] = mapped_column(String(40), nullable=False)
    updated_at: Mapped[str] = mapped_column(String(40), nullable=False)

    owner: Mapped[User] = relationship()
    applications: Mapped[list["Application"]] = relationship(
        back_populates="job",
        cascade="all, delete-orphan",
    )


class Candidate(Base):
    __tablename__ = "candidates"
    __table_args__ = (
        UniqueConstraint(
            "owner_id",
            "resume_hash",
            name="uq_candidates_owner_resume_hash",
        ),
        Index("idx_candidates_resume_hash", "resume_hash"),
    )

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    owner_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(
        String(160),
        nullable=False,
        default="",
        server_default="",
    )
    email: Mapped[str] = mapped_column(
        String(320),
        nullable=False,
        default="",
        server_default="",
    )
    phone: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
        default="",
        server_default="",
    )
    linkedin: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        server_default="",
    )
    github: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="",
        server_default="",
    )
    resume_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )
    resume_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    resume_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )
    created_at: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )
    updated_at: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    owner: Mapped[User] = relationship(
        back_populates="candidates",
    )
    applications: Mapped[list["Application"]] = relationship(
        back_populates="candidate",
        cascade="all, delete-orphan",
    )
    resume_versions: Mapped[list["ResumeVersion"]] = relationship(
        back_populates="candidate",
        cascade="all, delete-orphan",
        order_by="ResumeVersion.version_number",
    )


class ResumeVersion(Base):
    __tablename__ = "resume_versions"
    __table_args__ = (
        UniqueConstraint(
            "candidate_id",
            "version_number",
            name="uq_resume_versions_candidate_version",
        ),
        UniqueConstraint(
            "candidate_id",
            "resume_hash",
            name="uq_resume_versions_candidate_hash",
        ),
        Index(
            "ix_resume_versions_candidate",
            "candidate_id",
        ),
        Index(
            "ix_resume_versions_hash",
            "resume_hash",
        ),
        Index(
            "uq_resume_versions_current_candidate",
            "candidate_id",
            unique=True,
            postgresql_where=text("is_current = true"),
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )

    candidate_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
    )

    version_number: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    resume_filename: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
    )

    resume_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    resume_hash: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
    )

    is_current: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=True,
        server_default="true",
    )

    created_at: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    updated_at: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    candidate: Mapped["Candidate"] = relationship(
        back_populates="resume_versions",
    )

    analyses: Mapped[list["ResumeVersionAnalysis"]] = relationship(
        back_populates="resume_version",
        cascade="all, delete-orphan",
    )


class ResumeVersionAnalysis(Base):
    __tablename__ = "resume_version_analyses"
    __table_args__ = (
        UniqueConstraint(
            "resume_version_id",
            "job_id",
            name="uq_resume_version_analysis_version_job",
        ),
        Index(
            "ix_resume_version_analyses_version",
            "resume_version_id",
        ),
        Index(
            "ix_resume_version_analyses_job",
            "job_id",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )

    resume_version_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("resume_versions.id", ondelete="CASCADE"),
        nullable=False,
    )

    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
    )

    analysis_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )

    created_at: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    updated_at: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    resume_version: Mapped["ResumeVersion"] = relationship(
        back_populates="analyses",
    )

    job: Mapped["Job"] = relationship()


class Application(Base):
    __tablename__ = "applications"
    __table_args__ = (
        UniqueConstraint(
            "job_id",
            "candidate_id",
            name="uq_applications_job_candidate",
        ),
        Index(
            "idx_applications_job",
            "job_id",
        ),
        Index(
            "idx_applications_candidate",
            "candidate_id",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    candidate_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("candidates.id", ondelete="CASCADE"),
        nullable=False,
    )
    resume_version_id: Mapped[str | None] = mapped_column(
        String(36),
        ForeignKey("resume_versions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    status: Mapped[str] = mapped_column(
        String(30),
        nullable=False,
        default="new",
        server_default="new",
    )
    analysis_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )
    updated_at: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    job: Mapped[Job] = relationship(
        back_populates="applications",
    )
    candidate: Mapped[Candidate] = relationship(
        back_populates="applications",
    )
    resume_version: Mapped["ResumeVersion | None"] = relationship()


class AuditLog(Base):
    __tablename__ = "audit_logs"
    __table_args__ = (
        Index(
            "ix_audit_logs_owner_created",
            "owner_id",
            "created_at",
        ),
        Index(
            "ix_audit_logs_entity",
            "entity_type",
            "entity_id",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )
    owner_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    action: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
    )
    entity_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
    )
    entity_id: Mapped[str] = mapped_column(
        String(36),
        nullable=False,
        default="",
        server_default="",
    )
    metadata_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    owner: Mapped[User] = relationship()


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"
    __table_args__ = (
        Index(
            "ix_evaluation_runs_owner_job",
            "owner_id",
            "job_id",
        ),
        Index(
            "ix_evaluation_runs_created",
            "created_at",
        ),
    )

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
    )
    owner_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    metrics_json: Mapped[dict] = mapped_column(
        JSON,
        nullable=False,
        default=dict,
    )
    created_at: Mapped[str] = mapped_column(
        String(40),
        nullable=False,
    )

    owner: Mapped[User] = relationship()
    job: Mapped[Job] = relationship()