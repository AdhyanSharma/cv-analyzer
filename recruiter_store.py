"""
Persistent recruiter data store for the AI Resume Screening Platform.

Uses SQLite from Python's standard library. No extra dependency is required.

Stores:
- Candidate workflow state
- Candidate scores/contact metadata
- Status change history
- Recruiter note history

The database is created next to this file as:
    recruiter_data.db

For local development this is persistent across Streamlit restarts.
For cloud deployment, use an external database for durable storage.
"""

from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "recruiter_data.db")

STATUS_OPTIONS = [
    "New",
    "Reviewing",
    "Shortlisted",
    "Hold",
    "Rejected",
]

DEFAULT_STATUS = "New"


# ============================================================
# CONNECTION / SCHEMA
# ============================================================


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open a SQLite connection with sensible settings."""
    connection = sqlite3.connect(db_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection


def initialize_database(db_path: str = DB_PATH) -> None:
    """Create database tables and indexes when missing."""
    with get_connection(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                candidate_key TEXT PRIMARY KEY,
                job_fingerprint TEXT NOT NULL,
                resume_name TEXT NOT NULL,
                candidate_name TEXT NOT NULL,
                email TEXT DEFAULT '',
                phone TEXT DEFAULT '',
                linkedin TEXT DEFAULT '',
                github TEXT DEFAULT '',
                ats_score REAL DEFAULT 0,
                requirement_score REAL,
                quality_score REAL DEFAULT 0,
                semantic_score REAL DEFAULT 0,
                skill_score REAL DEFAULT 0,
                keyword_score REAL DEFAULT 0,
                experience_years REAL DEFAULT 0,
                status TEXT NOT NULL DEFAULT 'New',
                shortlisted INTEGER NOT NULL DEFAULT 0,
                notes TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_candidates_job
                ON candidates(job_fingerprint);

            CREATE INDEX IF NOT EXISTS idx_candidates_resume
                ON candidates(job_fingerprint, resume_name);

            CREATE TABLE IF NOT EXISTS status_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_key TEXT NOT NULL,
                old_status TEXT NOT NULL,
                new_status TEXT NOT NULL,
                changed_at TEXT NOT NULL,
                FOREIGN KEY(candidate_key) REFERENCES candidates(candidate_key)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_status_history_candidate
                ON status_history(candidate_key, changed_at);

            CREATE TABLE IF NOT EXISTS note_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_key TEXT NOT NULL,
                notes TEXT NOT NULL,
                changed_at TEXT NOT NULL,
                FOREIGN KEY(candidate_key) REFERENCES candidates(candidate_key)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_note_history_candidate
                ON note_history(candidate_key, changed_at);
            """
        )


# ============================================================
# IDENTIFICATION
# ============================================================


def make_candidate_key(job_fingerprint: str, resume_name: str) -> str:
    """Create a stable candidate key for one JD + resume filename pair."""
    import hashlib

    raw = f"{job_fingerprint}::{resume_name}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ============================================================
# CANDIDATE UPSERT
# ============================================================


def upsert_candidate(
    *,
    job_fingerprint: str,
    resume_name: str,
    candidate_name: str,
    email: str = "",
    phone: str = "",
    linkedin: str = "",
    github: str = "",
    ats_score: float = 0.0,
    requirement_score: Optional[float] = None,
    quality_score: float = 0.0,
    semantic_score: float = 0.0,
    skill_score: float = 0.0,
    keyword_score: float = 0.0,
    experience_years: float = 0.0,
    preserve_workflow: bool = True,
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Insert/update candidate metadata without losing workflow state."""
    initialize_database(db_path)

    key = make_candidate_key(job_fingerprint, resume_name)
    now = _utc_now()

    with get_connection(db_path) as connection:
        existing = connection.execute(
            "SELECT status, shortlisted, notes, created_at FROM candidates WHERE candidate_key = ?",
            (key,),
        ).fetchone()

        if existing and preserve_workflow:
            status = existing["status"]
            shortlisted = int(existing["shortlisted"])
            notes = existing["notes"]
            created_at = existing["created_at"]
        else:
            status = DEFAULT_STATUS
            shortlisted = 0
            notes = ""
            created_at = now

        connection.execute(
            """
            INSERT INTO candidates (
                candidate_key,
                job_fingerprint,
                resume_name,
                candidate_name,
                email,
                phone,
                linkedin,
                github,
                ats_score,
                requirement_score,
                quality_score,
                semantic_score,
                skill_score,
                keyword_score,
                experience_years,
                status,
                shortlisted,
                notes,
                created_at,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(candidate_key) DO UPDATE SET
                candidate_name = excluded.candidate_name,
                email = excluded.email,
                phone = excluded.phone,
                linkedin = excluded.linkedin,
                github = excluded.github,
                ats_score = excluded.ats_score,
                requirement_score = excluded.requirement_score,
                quality_score = excluded.quality_score,
                semantic_score = excluded.semantic_score,
                skill_score = excluded.skill_score,
                keyword_score = excluded.keyword_score,
                experience_years = excluded.experience_years,
                status = excluded.status,
                shortlisted = excluded.shortlisted,
                notes = excluded.notes,
                created_at = excluded.created_at,
                updated_at = excluded.updated_at
            """,
            (
                key,
                job_fingerprint,
                resume_name,
                str(candidate_name or "Candidate"),
                str(email or ""),
                str(phone or ""),
                str(linkedin or ""),
                str(github or ""),
                float(ats_score or 0),
                None if requirement_score is None else float(requirement_score),
                float(quality_score or 0),
                float(semantic_score or 0),
                float(skill_score or 0),
                float(keyword_score or 0),
                float(experience_years or 0),
                status,
                shortlisted,
                notes,
                created_at,
                now,
            ),
        )

    return get_candidate_state(job_fingerprint, resume_name, db_path=db_path)


# ============================================================
# WORKFLOW STATE
# ============================================================


def get_candidate_state(
    job_fingerprint: str,
    resume_name: str,
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Read workflow state for one candidate."""
    initialize_database(db_path)
    key = make_candidate_key(job_fingerprint, resume_name)

    with get_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT status, shortlisted, notes, updated_at
            FROM candidates
            WHERE candidate_key = ?
            """,
            (key,),
        ).fetchone()

    if row is None:
        return {
            "status": DEFAULT_STATUS,
            "shortlisted": False,
            "notes": "",
            "updated_at": None,
        }

    return {
        "status": row["status"],
        "shortlisted": bool(row["shortlisted"]),
        "notes": row["notes"] or "",
        "updated_at": row["updated_at"],
    }


def load_workflow(
    job_fingerprint: str,
    resume_names: Iterable[str],
    db_path: str = DB_PATH,
) -> Dict[str, Dict[str, Any]]:
    """Load workflow states keyed by resume filename."""
    initialize_database(db_path)

    names = [str(name) for name in resume_names if str(name).strip()]
    if not names:
        return {}

    placeholders = ",".join("?" for _ in names)
    params: List[Any] = [job_fingerprint, *names]

    workflow: Dict[str, Dict[str, Any]] = {
        name: {
            "status": DEFAULT_STATUS,
            "shortlisted": False,
            "notes": "",
            "updated_at": None,
        }
        for name in names
    }

    with get_connection(db_path) as connection:
        rows = connection.execute(
            f"""
            SELECT resume_name, status, shortlisted, notes, updated_at
            FROM candidates
            WHERE job_fingerprint = ?
              AND resume_name IN ({placeholders})
            """,
            params,
        ).fetchall()

    for row in rows:
        workflow[row["resume_name"]] = {
            "status": row["status"],
            "shortlisted": bool(row["shortlisted"]),
            "notes": row["notes"] or "",
            "updated_at": row["updated_at"],
        }

    return workflow


def save_candidate_state(
    *,
    job_fingerprint: str,
    resume_name: str,
    status: str,
    shortlisted: bool,
    notes: str,
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Persist workflow state and append status/note history when changed."""
    initialize_database(db_path)

    if status not in STATUS_OPTIONS:
        raise ValueError(
            f"Invalid status '{status}'. Allowed statuses: {STATUS_OPTIONS}"
        )

    key = make_candidate_key(job_fingerprint, resume_name)
    now = _utc_now()
    notes = str(notes or "").strip()

    with get_connection(db_path) as connection:
        row = connection.execute(
            "SELECT status, shortlisted, notes FROM candidates WHERE candidate_key = ?",
            (key,),
        ).fetchone()

        if row is None:
            connection.execute(
                """
                INSERT INTO candidates (
                    candidate_key,
                    job_fingerprint,
                    resume_name,
                    candidate_name,
                    status,
                    shortlisted,
                    notes,
                    created_at,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    key,
                    job_fingerprint,
                    resume_name,
                    os.path.splitext(os.path.basename(resume_name))[0] or "Candidate",
                    status,
                    int(bool(shortlisted)),
                    notes,
                    now,
                    now,
                ),
            )
            old_status = DEFAULT_STATUS
            old_shortlisted = False
            old_notes = ""
        else:
            old_status = row["status"]
            old_shortlisted = bool(row["shortlisted"])
            old_notes = row["notes"] or ""

            connection.execute(
                """
                UPDATE candidates
                SET status = ?, shortlisted = ?, notes = ?, updated_at = ?
                WHERE candidate_key = ?
                """,
                (
                    status,
                    int(bool(shortlisted)),
                    notes,
                    now,
                    key,
                ),
            )

        if old_status != status:
            connection.execute(
                """
                INSERT INTO status_history (
                    candidate_key,
                    old_status,
                    new_status,
                    changed_at
                ) VALUES (?, ?, ?, ?)
                """,
                (key, old_status, status, now),
            )

        if old_notes != notes:
            connection.execute(
                """
                INSERT INTO note_history (
                    candidate_key,
                    notes,
                    changed_at
                ) VALUES (?, ?, ?)
                """,
                (key, notes, now),
            )

        # If a user moves a candidate out of Shortlisted status explicitly,
        # keep the stored shortlist toggle aligned with that action.
        if status == "Shortlisted" and not shortlisted:
            connection.execute(
                "UPDATE candidates SET shortlisted = 1 WHERE candidate_key = ?",
                (key,),
            )
            shortlisted = True

    return {
        "status": status,
        "shortlisted": bool(shortlisted),
        "notes": notes,
        "updated_at": now,
    }


# ============================================================
# HISTORY
# ============================================================


def get_status_history(
    job_fingerprint: str,
    resume_name: str,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    """Return candidate status history, newest first."""
    initialize_database(db_path)
    key = make_candidate_key(job_fingerprint, resume_name)

    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT old_status, new_status, changed_at
            FROM status_history
            WHERE candidate_key = ?
            ORDER BY id DESC
            """,
            (key,),
        ).fetchall()

    return [
        {
            "old_status": row["old_status"],
            "new_status": row["new_status"],
            "changed_at": row["changed_at"],
        }
        for row in rows
    ]


def get_note_history(
    job_fingerprint: str,
    resume_name: str,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    """Return recruiter note history, newest first."""
    initialize_database(db_path)
    key = make_candidate_key(job_fingerprint, resume_name)

    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT notes, changed_at
            FROM note_history
            WHERE candidate_key = ?
            ORDER BY id DESC
            """,
            (key,),
        ).fetchall()

    return [
        {
            "notes": row["notes"],
            "changed_at": row["changed_at"],
        }
        for row in rows
    ]


def get_pipeline_counts(
    job_fingerprint: str,
    db_path: str = DB_PATH,
) -> Dict[str, int]:
    """Return status/shortlist counts for a job fingerprint."""
    initialize_database(db_path)

    counts = {
        "total": 0,
        "new": 0,
        "reviewing": 0,
        "shortlisted": 0,
        "hold": 0,
        "rejected": 0,
    }

    with get_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT status, shortlisted, COUNT(*) AS count
            FROM candidates
            WHERE job_fingerprint = ?
            GROUP BY status, shortlisted
            """,
            (job_fingerprint,),
        ).fetchall()

    for row in rows:
        counts["total"] += int(row["count"])

        status_key = {
            "New": "new",
            "Reviewing": "reviewing",
            "Hold": "hold",
            "Rejected": "rejected",
        }.get(row["status"])

        if status_key:
            counts[status_key] += int(row["count"])

        if bool(row["shortlisted"]):
            counts["shortlisted"] += int(row["count"])

    return counts


def clear_job_data(
    job_fingerprint: str,
    db_path: str = DB_PATH,
) -> None:
    """Delete persistent workflow/history for one job fingerprint."""
    initialize_database(db_path)

    with get_connection(db_path) as connection:
        rows = connection.execute(
            "SELECT candidate_key FROM candidates WHERE job_fingerprint = ?",
            (job_fingerprint,),
        ).fetchall()

        keys = [row["candidate_key"] for row in rows]

        for key in keys:
            connection.execute(
                "DELETE FROM status_history WHERE candidate_key = ?",
                (key,),
            )
            connection.execute(
                "DELETE FROM note_history WHERE candidate_key = ?",
                (key,),
            )

        connection.execute(
            "DELETE FROM candidates WHERE job_fingerprint = ?",
            (job_fingerprint,),
        )


# ============================================================
# TEST
# ============================================================


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        test_db = os.path.join(tmp, "test_recruiter.db")
        job = "demo-job-001"

        initialize_database(test_db)

        state = upsert_candidate(
            job_fingerprint=job,
            resume_name="Adhyan_Sharma.pdf",
            candidate_name="Adhyan Sharma",
            email="adhyan@example.com",
            ats_score=88.5,
            requirement_score=84.0,
            quality_score=90.0,
            db_path=test_db,
        )

        assert state["status"] == "New"
        assert state["shortlisted"] is False

        save_candidate_state(
            job_fingerprint=job,
            resume_name="Adhyan_Sharma.pdf",
            status="Reviewing",
            shortlisted=False,
            notes="Check project depth.",
            db_path=test_db,
        )

        save_candidate_state(
            job_fingerprint=job,
            resume_name="Adhyan_Sharma.pdf",
            status="Shortlisted",
            shortlisted=True,
            notes="Proceed to technical screen.",
            db_path=test_db,
        )

        loaded = get_candidate_state(
            job,
            "Adhyan_Sharma.pdf",
            db_path=test_db,
        )

        assert loaded["status"] == "Shortlisted"
        assert loaded["shortlisted"] is True

        history = get_status_history(
            job,
            "Adhyan_Sharma.pdf",
            db_path=test_db,
        )
        assert len(history) == 2

        notes = get_note_history(
            job,
            "Adhyan_Sharma.pdf",
            db_path=test_db,
        )
        assert len(notes) == 2

        print("Persistent recruiter store is working correctly.")
