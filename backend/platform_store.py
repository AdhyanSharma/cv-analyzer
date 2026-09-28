"""V12 persistence layer.

Public method names intentionally match V11 so the FastAPI API contract does
not need to change when moving from SQLite to PostgreSQL.
"""
from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from database import SessionLocal, create_db_engine, get_database_url
from models import (
    Application,
    AuditLog,
    Base,
    Candidate,
    EvaluationRun,
    Job,
    ResumeVersion,
    ResumeVersionAnalysis,
    User,
)


def utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def _user(row: User) -> Dict[str, Any]:
    return {
        "id": row.id,
        "email": row.email,
        "full_name": row.full_name,
        "password_hash": row.password_hash,
        "created_at": row.created_at,
        "active": bool(row.active),
    }


def _user_public(row: User) -> Dict[str, Any]:
    item = _user(row)
    item.pop("password_hash", None)
    return item


def _job(row: Job) -> Dict[str, Any]:
    return {
        "id": row.id,
        "owner_id": row.owner_id,
        "title": row.title,
        "company": row.company,
        "description": row.description,
        "status": row.status,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def _candidate(row: Candidate) -> Dict[str, Any]:
    return {
        "id": row.id,
        "owner_id": row.owner_id,
        "name": row.name,
        "email": row.email,
        "phone": row.phone,
        "linkedin": row.linkedin,
        "github": row.github,
        "resume_filename": row.resume_filename,
        "resume_text": row.resume_text,
        "resume_hash": row.resume_hash,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def _application(row: Application, candidate: Candidate, *, include_resume_text: bool = False) -> Dict[str, Any]:
    analysis = row.analysis_json
    if isinstance(analysis, str):
        import json

        try:
            analysis = json.loads(analysis)
        except json.JSONDecodeError:
            analysis = {}
    item: Dict[str, Any] = {
        "application_id": row.id,
        "job_id": row.job_id,
        "candidate_id": row.candidate_id,
        "resume_version_id": row.resume_version_id,
        "resume_version_number": (
            row.resume_version.version_number
            if row.resume_version
            else None
        ),
        "resume_version_filename": (
            row.resume_version.resume_filename
            if row.resume_version
            else None
        ),
        "status": row.status,
        "analysis": analysis if isinstance(analysis, dict) else {},
        "created_at": row.created_at,
        "updated_at": row.updated_at,
        "name": candidate.name,
        "email": candidate.email,
        "phone": candidate.phone,
        "linkedin": candidate.linkedin,
        "github": candidate.github,
        "resume_filename": candidate.resume_filename,
    }
    if include_resume_text:
        item["resume_text"] = candidate.resume_text
    return item


class PlatformStore:
    """Repository used by V11/V12 API handlers.

    Parameters are optional so tests can create an isolated SQLite database.
    """

    def __init__(
        self,
        db_path: str | None = None,
        *,
        database_url: str | None = None,
        auto_create: bool | None = None,
    ) -> None:
        if database_url:
            url = database_url
        elif db_path:
            url = f"sqlite:///{db_path.replace(chr(92), '/') }"
        else:
            url = get_database_url()

        self.database_url = url
        self.engine: Engine = create_db_engine(url)
        self.SessionLocal: sessionmaker[Session] = sessionmaker(
            bind=self.engine,
            autoflush=False,
            expire_on_commit=False,
            class_=Session,
        )

        if auto_create is None:
            auto_create = os.getenv("AUTO_CREATE_DB", "true").strip().lower() in {"1", "true", "yes", "on"}
            if not url.startswith("sqlite") and "AUTO_CREATE_DB" not in os.environ:
                auto_create = False
        if auto_create:
            Base.metadata.create_all(self.engine)

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def dispose(self) -> None:
        self.engine.dispose()

    def _session(self) -> Session:
        return self.SessionLocal()

    def _upsert_resume_version_analysis(
        self,
        session,
        *,
        resume_version_id: str,
        job_id: str,
        analysis: Dict[str, Any],
        now: str,
    ) -> ResumeVersionAnalysis:
        row = session.scalar(
            select(ResumeVersionAnalysis).where(
                ResumeVersionAnalysis.resume_version_id == resume_version_id,
                ResumeVersionAnalysis.job_id == job_id,
            )
        )

        if row:
            row.analysis_json = analysis
            row.updated_at = now
            return row

        row = ResumeVersionAnalysis(
            id=str(uuid.uuid4()),
            resume_version_id=resume_version_id,
            job_id=job_id,
            analysis_json=analysis,
            created_at=now,
            updated_at=now,
        )

        session.add(row)
        return row

    def create_user(self, email: str, full_name: str, password_hash: str) -> Dict[str, Any]:
        row = User(
            id=str(uuid.uuid4()),
            email=email.lower().strip(),
            full_name=full_name.strip(),
            password_hash=password_hash,
            created_at=utc_now(),
            active=True,
        )
        session = self._session()
        try:
            session.add(row)
            session.commit()
            session.refresh(row)
            return _user_public(row)
        except IntegrityError as exc:
            session.rollback()
            raise ValueError("An account with this email already exists.") from exc
        finally:
            session.close()

    def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        session = self._session()
        try:
            row = session.scalar(select(User).where(User.email == email.lower().strip()))
            return _user(row) if row else None
        finally:
            session.close()

    def get_user_public(self, user_id: str) -> Optional[Dict[str, Any]]:
        session = self._session()
        try:
            row = session.get(User, user_id)
            return _user_public(row) if row else None
        finally:
            session.close()

    def create_job(self, owner_id: str, title: str, company: str, description: str) -> Dict[str, Any]:
        row = Job(
            id=str(uuid.uuid4()),
            owner_id=owner_id,
            title=title.strip(),
            company=company.strip(),
            description=description.strip(),
            status="open",
            created_at=utc_now(),
            updated_at=utc_now(),
        )
        session = self._session()
        try:
            session.add(row)
            session.commit()
            session.refresh(row)
            return _job(row)
        finally:
            session.close()

    def list_jobs(self, owner_id: str) -> List[Dict[str, Any]]:
        session = self._session()
        try:
            rows = session.scalars(
                select(Job).where(Job.owner_id == owner_id).order_by(Job.created_at.desc())
            ).all()
            return [_job(row) for row in rows]
        finally:
            session.close()

    def get_job(self, owner_id: str, job_id: str) -> Optional[Dict[str, Any]]:
        session = self._session()
        try:
            row = session.scalar(select(Job).where(Job.id == job_id, Job.owner_id == owner_id))
            return _job(row) if row else None
        finally:
            session.close()

    def update_job(self, owner_id: str, job_id: str, **fields: Any) -> Optional[Dict[str, Any]]:
        allowed = {"title", "company", "description", "status"}
        session = self._session()
        try:
            row = session.scalar(select(Job).where(Job.id == job_id, Job.owner_id == owner_id))
            if not row:
                return None
            for key, value in fields.items():
                if key in allowed and value is not None:
                    setattr(row, key, value.strip() if isinstance(value, str) else value)
            row.updated_at = utc_now()
            session.commit()
            session.refresh(row)
            return _job(row)
        finally:
            session.close()

    def delete_job(self, owner_id: str, job_id: str) -> bool:
        session = self._session()
        try:
            row = session.scalar(select(Job).where(Job.id == job_id, Job.owner_id == owner_id))
            if not row:
                return False
            session.delete(row)
            session.commit()
            return True
        finally:
            session.close()

    def get_candidate_by_resume_hash(
        self,
        owner_id: str,
        resume_hash: str,
    ) -> Optional[Dict[str, Any]]:
        """Return the owner's candidate matching any uploaded resume version hash."""
        clean_hash = (resume_hash or "").strip().lower()
        if not clean_hash:
            return None

        session = self._session()
        try:
            row = session.scalar(
                select(Candidate)
                .join(
                    ResumeVersion,
                    ResumeVersion.candidate_id == Candidate.id,
                )
                .where(
                    Candidate.owner_id == owner_id,
                    ResumeVersion.resume_hash == clean_hash,
                )
                .order_by(ResumeVersion.created_at.desc())
                .limit(1)
            )

            return _candidate(row) if row else None
        finally:
            session.close()

    def upsert_candidate(
        self,
        owner_id: str,
        *,
        name: str,
        email: str,
        phone: str,
        linkedin: str,
        github: str,
        resume_filename: str,
        resume_text: str,
        resume_hash: str | None = None,
    ) -> Dict[str, Any]:
        session = self._session()

        try:
            clean_email = (email or "").strip().lower()
            clean_hash = (resume_hash or "").strip().lower() or None

            row = None

            # ---------------------------------------------------------
            # 1. Exact resume hash identity
            # ---------------------------------------------------------
            if clean_hash:
                row = session.scalar(
                    select(Candidate)
                    .join(
                        ResumeVersion,
                        ResumeVersion.candidate_id == Candidate.id,
                    )
                    .where(
                        Candidate.owner_id == owner_id,
                        ResumeVersion.resume_hash == clean_hash,
                    )
                    .order_by(ResumeVersion.created_at.desc())
                    .limit(1)
                )

            # ---------------------------------------------------------
            # 2. Fall back to candidate email identity
            # ---------------------------------------------------------
            if not row and clean_email:
                row = session.scalar(
                    select(Candidate)
                    .where(
                        Candidate.owner_id == owner_id,
                        Candidate.email == clean_email,
                    )
                    .order_by(Candidate.created_at.desc())
                )

            now = utc_now()

            # ---------------------------------------------------------
            # 3. Existing candidate
            # ---------------------------------------------------------
            if row:
                row.name = (name or "").strip()
                row.email = clean_email or row.email
                row.phone = (phone or "").strip()
                row.linkedin = (linkedin or "").strip()
                row.github = (github or "").strip()

                existing_version = None

                if clean_hash:
                    existing_version = session.scalar(
                        select(ResumeVersion)
                        .where(
                            ResumeVersion.candidate_id == row.id,
                            ResumeVersion.resume_hash == clean_hash,
                        )
                        .order_by(
                            ResumeVersion.version_number.desc()
                        )
                        .limit(1)
                    )

                # -----------------------------------------------------
                # Same resume already exists in this candidate's history
                # -----------------------------------------------------
                if existing_version:
                    # Do NOT replace the current resume with an older one.
                    row.updated_at = now

                    session.commit()
                    session.refresh(row)

                    result = _candidate(row)
                    result["resume_version"] = {
                        "id": existing_version.id,
                        "version_number": existing_version.version_number,
                        "resume_filename": existing_version.resume_filename,
                        "resume_hash": existing_version.resume_hash,
                        "is_current": existing_version.is_current,
                    }
                    result["resume_version_existing"] = True

                    return result

                # -----------------------------------------------------
                # New resume version for existing candidate
                # -----------------------------------------------------
                latest_version = session.scalar(
                    select(func.max(ResumeVersion.version_number))
                    .where(
                        ResumeVersion.candidate_id == row.id
                    )
                )

                next_version = (latest_version or 0) + 1

                # Mark previous current version as historical.
                current_versions = session.scalars(
                    select(ResumeVersion)
                    .where(
                        ResumeVersion.candidate_id == row.id,
                        ResumeVersion.is_current.is_(True),
                    )
                ).all()

                for version in current_versions:
                    version.is_current = False
                    version.updated_at = now

                new_version = ResumeVersion(
                    id=str(uuid.uuid4()),
                    candidate_id=row.id,
                    version_number=next_version,
                    resume_filename=resume_filename,
                    resume_text=resume_text,
                    resume_hash=clean_hash,
                    is_current=True,
                    created_at=now,
                    updated_at=now,
                )

                session.add(new_version)

                # Candidate keeps the latest/current resume.
                row.resume_filename = resume_filename
                row.resume_text = resume_text
                row.resume_hash = clean_hash
                row.updated_at = now

                session.commit()
                session.refresh(row)
                session.refresh(new_version)

                result = _candidate(row)
                result["resume_version"] = {
                    "id": new_version.id,
                    "version_number": new_version.version_number,
                    "resume_filename": new_version.resume_filename,
                    "resume_hash": new_version.resume_hash,
                    "is_current": True,
                }
                result["resume_version_existing"] = False

                return result

            # ---------------------------------------------------------
            # 4. Brand-new candidate
            # ---------------------------------------------------------
            row = Candidate(
                id=str(uuid.uuid4()),
                owner_id=owner_id,
                name=(name or "").strip(),
                email=clean_email,
                phone=(phone or "").strip(),
                linkedin=(linkedin or "").strip(),
                github=(github or "").strip(),
                resume_filename=resume_filename,
                resume_text=resume_text,
                resume_hash=clean_hash,
                created_at=now,
                updated_at=now,
            )

            session.add(row)
            session.flush()

            first_version = ResumeVersion(
                id=str(uuid.uuid4()),
                candidate_id=row.id,
                version_number=1,
                resume_filename=resume_filename,
                resume_text=resume_text,
                resume_hash=clean_hash,
                is_current=True,
                created_at=now,
                updated_at=now,
            )

            session.add(first_version)

            session.commit()
            session.refresh(row)
            session.refresh(first_version)

            result = _candidate(row)
            result["resume_version"] = {
                "id": first_version.id,
                "version_number": 1,
                "resume_filename": first_version.resume_filename,
                "resume_hash": first_version.resume_hash,
                "is_current": True,
            }
            result["resume_version_existing"] = False

            return result

        finally:
            session.close()

    def list_resume_versions(
        self,
        owner_id: str,
        candidate_id: str,
    ) -> List[Dict[str, Any]]:
        """Return all resume versions for an owned candidate."""
        session = self._session()

        try:
            candidate = session.scalar(
                select(Candidate).where(
                    Candidate.id == candidate_id,
                    Candidate.owner_id == owner_id,
                )
            )

            if not candidate:
                return []

            rows = session.scalars(
                select(ResumeVersion)
                .where(
                    ResumeVersion.candidate_id == candidate_id,
                )
                .order_by(
                    ResumeVersion.version_number.asc()
                )
            ).all()

            return [
                {
                    "id": row.id,
                    "candidate_id": row.candidate_id,
                    "version_number": row.version_number,
                    "resume_filename": row.resume_filename,
                    "resume_hash": row.resume_hash,
                    "is_current": row.is_current,
                    "created_at": row.created_at,
                    "updated_at": row.updated_at,
                }
                for row in rows
            ]

        finally:
            session.close()

    def add_application(
        self,
        job_id: str,
        candidate_id: str,
        analysis: Dict[str, Any],
        resume_version_id: str | None = None,
    ) -> Dict[str, Any]:
        session = self._session()

        try:
            row = session.scalar(
                select(Application).where(
                    Application.job_id == job_id,
                    Application.candidate_id == candidate_id,
                )
            )

            now = utc_now()

            if row:
                row.analysis_json = analysis

                if resume_version_id:
                    row.resume_version_id = resume_version_id
                    self._upsert_resume_version_analysis(
                        session,
                        resume_version_id=resume_version_id,
                        job_id=job_id,
                        analysis=analysis,
                        now=now,
                    )

                row.updated_at = now

            else:
                row = Application(
                    id=str(uuid.uuid4()),
                    job_id=job_id,
                    candidate_id=candidate_id,
                    resume_version_id=resume_version_id,
                    status="new",
                    analysis_json=analysis,
                    created_at=now,
                    updated_at=now,
                )
                session.add(row)

                if resume_version_id:
                    self._upsert_resume_version_analysis(
                        session,
                        resume_version_id=resume_version_id,
                        job_id=job_id,
                        analysis=analysis,
                        now=now,
                    )

            session.commit()
            session.refresh(row)

            candidate = session.get(Candidate, candidate_id)

            if not candidate:
                raise ValueError("Candidate not found after application creation.")

            return _application(
                row,
                candidate,
                include_resume_text=False,
            )

        finally:
            session.close()

    def list_applications(self, owner_id: str, job_id: str) -> List[Dict[str, Any]]:
        session = self._session()
        try:
            stmt = (
                select(Application, Candidate)
                .join(Candidate, Candidate.id == Application.candidate_id)
                .join(Job, Job.id == Application.job_id)
                .where(Job.owner_id == owner_id, Application.job_id == job_id)
                .order_by(Application.created_at.desc())
            )
            rows = session.execute(stmt).all()
            return [_application(app, candidate) for app, candidate in rows]
        finally:
            session.close()

    def get_application(self, owner_id: str, job_id: str, candidate_id: str) -> Optional[Dict[str, Any]]:
        session = self._session()
        try:
            stmt = (
                select(Application, Candidate)
                .join(Candidate, Candidate.id == Application.candidate_id)
                .join(Job, Job.id == Application.job_id)
                .where(
                    Job.owner_id == owner_id,
                    Application.job_id == job_id,
                    Application.candidate_id == candidate_id,
                )
            )
            result = session.execute(stmt).first()
            if not result:
                return None
            app, candidate = result
            return _application(app, candidate, include_resume_text=True)
        finally:
            session.close()

    def update_application_status(self, owner_id: str, job_id: str, candidate_id: str, status: str) -> Optional[Dict[str, Any]]:
        session = self._session()
        try:
            stmt = (
                select(Application)
                .join(Job, Job.id == Application.job_id)
                .where(
                    Job.owner_id == owner_id,
                    Application.job_id == job_id,
                    Application.candidate_id == candidate_id,
                )
            )
            row = session.scalar(stmt)
            if not row:
                return None
            row.status = status
            row.updated_at = utc_now()
            session.commit()
        finally:
            session.close()
        return self.get_application(owner_id, job_id, candidate_id)

    def get_resume_version_analysis(
        self,
        owner_id: str,
        resume_version_id: str,
        job_id: str,
    ) -> Optional[Dict[str, Any]]:
        session = self._session()

        try:
            row = session.scalar(
                select(ResumeVersionAnalysis)
                .join(
                    ResumeVersion,
                    ResumeVersion.id == ResumeVersionAnalysis.resume_version_id,
                )
                .join(
                    Candidate,
                    Candidate.id == ResumeVersion.candidate_id,
                )
                .where(
                    Candidate.owner_id == owner_id,
                    ResumeVersionAnalysis.resume_version_id == resume_version_id,
                    ResumeVersionAnalysis.job_id == job_id,
                )
            )

            if not row:
                return None

            analysis = row.analysis_json

            if isinstance(analysis, str):
                import json

                try:
                    analysis = json.loads(analysis)
                except json.JSONDecodeError:
                    analysis = {}

            return {
                "id": row.id,
                "resume_version_id": row.resume_version_id,
                "job_id": row.job_id,
                "analysis": analysis if isinstance(analysis, dict) else {},
                "created_at": row.created_at,
                "updated_at": row.updated_at,
            }

        finally:
            session.close()

    def compare_resume_versions(
        self,
        owner_id: str,
        candidate_id: str,
        version1: int,
        version2: int,
        job_id: str | None = None,
    ) -> Optional[Dict[str, Any]]:
        """Compare two resume versions, optionally for a specific job."""

        if version1 == version2:
            raise ValueError("version1 and version2 must be different.")

        session = self._session()

        try:
            candidate = session.scalar(
                select(Candidate).where(
                    Candidate.id == candidate_id,
                    Candidate.owner_id == owner_id,
                )
            )

            if not candidate:
                return None

            versions = session.scalars(
                select(ResumeVersion)
                .where(
                    ResumeVersion.candidate_id == candidate_id,
                    ResumeVersion.version_number.in_([version1, version2]),
                )
                .order_by(ResumeVersion.version_number.asc())
            ).all()

            version_map = {row.version_number: row for row in versions}

            first = version_map.get(version1)
            second = version_map.get(version2)

            if not first or not second:
                return None

            def extract_analysis_value(
                analysis: dict,
                *keys: str,
            ) -> Any:
                for key in keys:
                    value = analysis.get(key)

                    if value is not None:
                        return value

                nested_scores = analysis.get("scores")

                if isinstance(nested_scores, dict):
                    for key in keys:
                        value = nested_scores.get(key)

                        if value is not None:
                            return value

                return None

            def extract_list(
                analysis: dict,
                *keys: str,
            ) -> list:
                for key in keys:
                    value = analysis.get(key)

                    if isinstance(value, list):
                        return value

                requirements = analysis.get("requirements")

                if isinstance(requirements, dict):
                    for key in keys:
                        value = requirements.get(key)

                        if isinstance(value, list):
                            return value

                return []

            first_analysis = {}
            second_analysis = {}

            if job_id:
                first_analysis_row = session.scalar(
                    select(ResumeVersionAnalysis)
                    .where(
                        ResumeVersionAnalysis.resume_version_id == first.id,
                        ResumeVersionAnalysis.job_id == job_id,
                    )
                )

                second_analysis_row = session.scalar(
                    select(ResumeVersionAnalysis)
                    .where(
                        ResumeVersionAnalysis.resume_version_id == second.id,
                        ResumeVersionAnalysis.job_id == job_id,
                    )
                )

                if first_analysis_row:
                    first_analysis = (
                        first_analysis_row.analysis_json
                        if isinstance(first_analysis_row.analysis_json, dict)
                        else {}
                    )

                if second_analysis_row:
                    second_analysis = (
                        second_analysis_row.analysis_json
                        if isinstance(second_analysis_row.analysis_json, dict)
                        else {}
                    )

            first_ats = extract_analysis_value(
                first_analysis,
                "ats_score",
                "final_score",
            )

            second_ats = extract_analysis_value(
                second_analysis,
                "ats_score",
                "final_score",
            )

            first_matched = {
                str(x).strip()
                for x in extract_list(
                    first_analysis,
                    "matched_skills",
                    "required_matched",
                )
                if str(x).strip()
            }

            second_matched = {
                str(x).strip()
                for x in extract_list(
                    second_analysis,
                    "matched_skills",
                    "required_matched",
                )
                if str(x).strip()
            }

            first_missing = {
                str(x).strip()
                for x in extract_list(
                    first_analysis,
                    "missing_skills",
                    "required_missing",
                )
                if str(x).strip()
            }

            second_missing = {
                str(x).strip()
                for x in extract_list(
                    second_analysis,
                    "missing_skills",
                    "required_missing",
                )
                if str(x).strip()
            }

            added_skills = sorted(second_matched - first_matched, key=str.lower)
            removed_skills = sorted(first_matched - second_matched, key=str.lower)
            newly_missing = sorted(second_missing - first_missing, key=str.lower)
            resolved_missing = sorted(first_missing - second_missing, key=str.lower)

            ats_change = None

            if isinstance(first_ats, (int, float)) and isinstance(second_ats, (int, float)):
                ats_change = float(second_ats) - float(first_ats)

            return {
                "candidate_id": candidate_id,
                "job_id": job_id,
                "version1": {
                    "id": first.id,
                    "version_number": first.version_number,
                    "resume_filename": first.resume_filename,
                    "resume_hash": first.resume_hash,
                    "is_current": first.is_current,
                    "created_at": first.created_at,
                },
                "version2": {
                    "id": second.id,
                    "version_number": second.version_number,
                    "resume_filename": second.resume_filename,
                    "resume_hash": second.resume_hash,
                    "is_current": second.is_current,
                    "created_at": second.created_at,
                },
                "comparison": {
                    "ats_score": {
                        "version1": first_ats,
                        "version2": second_ats,
                        "change": ats_change,
                    },
                    "skills": {
                        "added": added_skills,
                        "removed": removed_skills,
                    },
                    "missing_skills": {
                        "resolved": resolved_missing,
                        "newly_missing": newly_missing,
                    },
                },
            }

        finally:
            session.close()

    def create_audit_log(
        self,
        owner_id: str,
        *,
        action: str,
        entity_type: str,
        entity_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        row = AuditLog(
            id=str(uuid.uuid4()),
            owner_id=owner_id,
            action=action.strip(),
            entity_type=entity_type.strip(),
            entity_id=(entity_id or "").strip(),
            metadata_json=metadata or {},
            created_at=utc_now(),
        )
        session = self._session()
        try:
            session.add(row)
            session.commit()
            session.refresh(row)
            return {
                "id": row.id,
                "owner_id": row.owner_id,
                "action": row.action,
                "entity_type": row.entity_type,
                "entity_id": row.entity_id,
                "metadata": row.metadata_json if isinstance(row.metadata_json, dict) else {},
                "created_at": row.created_at,
            }
        finally:
            session.close()

    def list_audit_logs(self, owner_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 500))
        session = self._session()
        try:
            rows = session.scalars(
                select(AuditLog)
                .where(AuditLog.owner_id == owner_id)
                .order_by(AuditLog.created_at.desc())
                .limit(limit)
            ).all()
            return [
                {
                    "id": row.id,
                    "owner_id": row.owner_id,
                    "action": row.action,
                    "entity_type": row.entity_type,
                    "entity_id": row.entity_id,
                    "metadata": row.metadata_json if isinstance(row.metadata_json, dict) else {},
                    "created_at": row.created_at,
                }
                for row in rows
            ]
        finally:
            session.close()

    def create_evaluation_run(
        self,
        owner_id: str,
        job_id: str,
        metrics: Dict[str, Any],
    ) -> Dict[str, Any]:
        row = EvaluationRun(
            id=str(uuid.uuid4()),
            owner_id=owner_id,
            job_id=job_id,
            metrics_json=metrics,
            created_at=utc_now(),
        )
        session = self._session()
        try:
            session.add(row)
            session.commit()
            session.refresh(row)
            return {
                "id": row.id,
                "owner_id": row.owner_id,
                "job_id": row.job_id,
                "created_at": row.created_at,
                "metrics": row.metrics_json if isinstance(row.metrics_json, dict) else {},
            }
        finally:
            session.close()

    def list_evaluation_runs(self, owner_id: str, job_id: str, limit: int = 20) -> List[Dict[str, Any]]:
        limit = max(1, min(int(limit), 100))
        session = self._session()
        try:
            rows = session.scalars(
                select(EvaluationRun)
                .where(EvaluationRun.owner_id == owner_id, EvaluationRun.job_id == job_id)
                .order_by(EvaluationRun.created_at.desc())
                .limit(limit)
            ).all()
            return [
                {
                    "id": row.id,
                    "owner_id": row.owner_id,
                    "job_id": row.job_id,
                    "created_at": row.created_at,
                    "metrics": row.metrics_json if isinstance(row.metrics_json, dict) else {},
                }
                for row in rows
            ]
        finally:
            session.close()


# Keep the V11 import contract: api.py does `from platform_store import store`.
store = PlatformStore()


