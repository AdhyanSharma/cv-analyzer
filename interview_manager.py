"""
Interview Management Engine
---------------------------

Persistent interview scheduling, scorecards, feedback, history,
dashboard statistics, and calendar (.ics) export.

Uses the same SQLite database as recruiter_store.py.
No external dependencies beyond the Python standard library.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from recruiter_store import (
    DB_PATH,
    make_candidate_key,
    initialize_database,
)

# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

INTERVIEW_ROUNDS = [
    "Screening Call",
    "Technical Round",
    "Machine Learning / AI Round",
    "System Design Round",
    "HR Round",
    "Final Round",
]

INTERVIEW_TYPES = [
    "Video",
    "Phone",
    "In-person",
    "Online Test",
]

INTERVIEW_STATUSES = [
    "Scheduled",
    "Completed",
    "Cancelled",
    "Rescheduled",
]

RECOMMENDATION_OPTIONS = [
    "Pending",
    "Proceed",
    "Hold",
    "Reject",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_interview_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    initialize_database(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def initialize_interview_database(db_path: str = DB_PATH) -> None:
    """Create interview tables and indexes in the shared recruiter DB."""
    initialize_database(db_path)

    with get_interview_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS interviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_key TEXT NOT NULL,
                job_fingerprint TEXT NOT NULL,
                resume_name TEXT NOT NULL,
                candidate_name TEXT NOT NULL,

                interview_round TEXT NOT NULL,
                interview_type TEXT NOT NULL,
                scheduled_date TEXT NOT NULL,
                scheduled_time TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL DEFAULT 60,

                interviewer TEXT DEFAULT '',
                meeting_link TEXT DEFAULT '',

                status TEXT NOT NULL DEFAULT 'Scheduled',

                technical_score REAL,
                communication_score REAL,
                problem_solving_score REAL,
                overall_score REAL,

                recommendation TEXT NOT NULL DEFAULT 'Pending',
                feedback TEXT DEFAULT '',
                internal_notes TEXT DEFAULT '',

                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_interviews_job
                ON interviews(job_fingerprint);

            CREATE INDEX IF NOT EXISTS idx_interviews_candidate
                ON interviews(candidate_key);

            CREATE INDEX IF NOT EXISTS idx_interviews_schedule
                ON interviews(job_fingerprint, scheduled_date, scheduled_time);

            CREATE TABLE IF NOT EXISTS interview_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                interview_id INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                event_detail TEXT DEFAULT '',
                created_at TEXT NOT NULL,
                FOREIGN KEY(interview_id) REFERENCES interviews(id)
                    ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_interview_history_interview
                ON interview_history(interview_id, created_at);
            """
        )


def _normalize_score(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        score = float(value)
        return max(0.0, min(5.0, score))
    except (TypeError, ValueError):
        return None


def calculate_overall_score(
    technical_score: Any,
    communication_score: Any,
    problem_solving_score: Any,
) -> Optional[float]:
    """Average the supplied interview scores on a 1-5 scale."""
    values = [
        _normalize_score(technical_score),
        _normalize_score(communication_score),
        _normalize_score(problem_solving_score),
    ]
    values = [value for value in values if value is not None]

    if not values:
        return None

    return round(sum(values) / len(values), 2)


def create_interview(
    *,
    job_fingerprint: str,
    resume_name: str,
    candidate_name: str,
    interview_round: str,
    interview_type: str,
    scheduled_date: str,
    scheduled_time: str,
    duration_minutes: int = 60,
    interviewer: str = "",
    meeting_link: str = "",
    status: str = "Scheduled",
    recommendation: str = "Pending",
    feedback: str = "",
    internal_notes: str = "",
    db_path: str = DB_PATH,
) -> int:
    """Create a new scheduled interview and return its ID."""
    initialize_interview_database(db_path)

    if interview_round not in INTERVIEW_ROUNDS:
        raise ValueError("Invalid interview round.")

    if interview_type not in INTERVIEW_TYPES:
        raise ValueError("Invalid interview type.")

    if status not in INTERVIEW_STATUSES:
        raise ValueError("Invalid interview status.")

    if recommendation not in RECOMMENDATION_OPTIONS:
        raise ValueError("Invalid recommendation.")

    duration_minutes = int(duration_minutes)
    if duration_minutes < 15 or duration_minutes > 480:
        raise ValueError("Interview duration must be between 15 and 480 minutes.")

    now = _utc_now()
    candidate_key = make_candidate_key(job_fingerprint, resume_name)

    with get_interview_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO interviews (
                candidate_key,
                job_fingerprint,
                resume_name,
                candidate_name,
                interview_round,
                interview_type,
                scheduled_date,
                scheduled_time,
                duration_minutes,
                interviewer,
                meeting_link,
                status,
                recommendation,
                feedback,
                internal_notes,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_key,
                job_fingerprint,
                resume_name,
                candidate_name,
                interview_round,
                interview_type,
                scheduled_date,
                scheduled_time,
                duration_minutes,
                str(interviewer or "").strip(),
                str(meeting_link or "").strip(),
                status,
                recommendation,
                str(feedback or "").strip(),
                str(internal_notes or "").strip(),
                now,
                now,
            ),
        )

        interview_id = int(cursor.lastrowid)

        conn.execute(
            """
            INSERT INTO interview_history (
                interview_id,
                event_type,
                event_detail,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                interview_id,
                "created",
                f"{interview_round} scheduled for {scheduled_date} {scheduled_time}",
                now,
            ),
        )

    return interview_id


def get_interview(
    interview_id: int,
    db_path: str = DB_PATH,
) -> Optional[Dict[str, Any]]:
    initialize_interview_database(db_path)

    with get_interview_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM interviews WHERE id = ?",
            (int(interview_id),),
        ).fetchone()

    return dict(row) if row else None


def update_interview(
    interview_id: int,
    *,
    interview_round: Optional[str] = None,
    interview_type: Optional[str] = None,
    scheduled_date: Optional[str] = None,
    scheduled_time: Optional[str] = None,
    duration_minutes: Optional[int] = None,
    interviewer: Optional[str] = None,
    meeting_link: Optional[str] = None,
    status: Optional[str] = None,
    recommendation: Optional[str] = None,
    feedback: Optional[str] = None,
    internal_notes: Optional[str] = None,
    technical_score: Any = None,
    communication_score: Any = None,
    problem_solving_score: Any = None,
    db_path: str = DB_PATH,
) -> Optional[Dict[str, Any]]:
    """Update an interview and append an audit-history entry."""
    initialize_interview_database(db_path)

    existing = get_interview(interview_id, db_path)
    if existing is None:
        return None

    final_round = interview_round or existing["interview_round"]
    final_type = interview_type or existing["interview_type"]
    final_status = status or existing["status"]
    final_recommendation = recommendation or existing["recommendation"]

    if final_round not in INTERVIEW_ROUNDS:
        raise ValueError("Invalid interview round.")
    if final_type not in INTERVIEW_TYPES:
        raise ValueError("Invalid interview type.")
    if final_status not in INTERVIEW_STATUSES:
        raise ValueError("Invalid interview status.")
    if final_recommendation not in RECOMMENDATION_OPTIONS:
        raise ValueError("Invalid recommendation.")

    final_duration = (
        int(duration_minutes)
        if duration_minutes is not None
        else int(existing["duration_minutes"])
    )

    if final_duration < 15 or final_duration > 480:
        raise ValueError("Interview duration must be between 15 and 480 minutes.")

    final_technical = (
        _normalize_score(technical_score)
        if technical_score is not None
        else existing["technical_score"]
    )
    final_communication = (
        _normalize_score(communication_score)
        if communication_score is not None
        else existing["communication_score"]
    )
    final_problem_solving = (
        _normalize_score(problem_solving_score)
        if problem_solving_score is not None
        else existing["problem_solving_score"]
    )

    final_overall = calculate_overall_score(
        final_technical,
        final_communication,
        final_problem_solving,
    )

    # Preserve an existing overall score when no score fields are being touched.
    if (
        technical_score is None
        and communication_score is None
        and problem_solving_score is None
        and existing["overall_score"] is not None
    ):
        final_overall = existing["overall_score"]

    now = _utc_now()

    changed_fields = []

    comparable = {
        "interview_round": final_round,
        "interview_type": final_type,
        "scheduled_date": scheduled_date or existing["scheduled_date"],
        "scheduled_time": scheduled_time or existing["scheduled_time"],
        "duration_minutes": final_duration,
        "interviewer": (
            interviewer if interviewer is not None
            else existing["interviewer"]
        ),
        "meeting_link": (
            meeting_link if meeting_link is not None
            else existing["meeting_link"]
        ),
        "status": final_status,
        "recommendation": final_recommendation,
        "feedback": (
            feedback if feedback is not None
            else existing["feedback"]
        ),
        "internal_notes": (
            internal_notes if internal_notes is not None
            else existing["internal_notes"]
        ),
        "technical_score": final_technical,
        "communication_score": final_communication,
        "problem_solving_score": final_problem_solving,
        "overall_score": final_overall,
    }

    for key, value in comparable.items():
        if str(existing.get(key)) != str(value):
            changed_fields.append(key)

    with get_interview_connection(db_path) as conn:
        conn.execute(
            """
            UPDATE interviews
            SET
                interview_round = ?,
                interview_type = ?,
                scheduled_date = ?,
                scheduled_time = ?,
                duration_minutes = ?,
                interviewer = ?,
                meeting_link = ?,
                status = ?,
                technical_score = ?,
                communication_score = ?,
                problem_solving_score = ?,
                overall_score = ?,
                recommendation = ?,
                feedback = ?,
                internal_notes = ?,
                updated_at = ?
            WHERE id = ?
            """,
            (
                comparable["interview_round"],
                comparable["interview_type"],
                comparable["scheduled_date"],
                comparable["scheduled_time"],
                comparable["duration_minutes"],
                str(comparable["interviewer"] or "").strip(),
                str(comparable["meeting_link"] or "").strip(),
                comparable["status"],
                comparable["technical_score"],
                comparable["communication_score"],
                comparable["problem_solving_score"],
                comparable["overall_score"],
                comparable["recommendation"],
                str(comparable["feedback"] or "").strip(),
                str(comparable["internal_notes"] or "").strip(),
                now,
                int(interview_id),
            ),
        )

        if changed_fields:
            detail = ", ".join(changed_fields)
            conn.execute(
                """
                INSERT INTO interview_history (
                    interview_id,
                    event_type,
                    event_detail,
                    created_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    int(interview_id),
                    "updated",
                    detail,
                    now,
                ),
            )

    return get_interview(interview_id, db_path)


def list_interviews(
    job_fingerprint: str,
    *,
    candidate_resume: Optional[str] = None,
    status: Optional[str] = None,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    """Return interviews newest/schedule-first for one JD."""
    initialize_interview_database(db_path)

    query = """
        SELECT *
        FROM interviews
        WHERE job_fingerprint = ?
    """
    params: List[Any] = [job_fingerprint]

    if candidate_resume:
        query += " AND resume_name = ?"
        params.append(candidate_resume)

    if status and status != "All":
        query += " AND status = ?"
        params.append(status)

    query += """
        ORDER BY scheduled_date ASC,
                 scheduled_time ASC,
                 id DESC
    """

    with get_interview_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def get_candidate_interview_history(
    job_fingerprint: str,
    resume_name: str,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    return list_interviews(
        job_fingerprint,
        candidate_resume=resume_name,
        db_path=db_path,
    )


def get_interview_event_history(
    interview_id: int,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    initialize_interview_database(db_path)

    with get_interview_connection(db_path) as conn:
        rows = conn.execute(
            """
            SELECT event_type, event_detail, created_at
            FROM interview_history
            WHERE interview_id = ?
            ORDER BY id DESC
            """,
            (int(interview_id),),
        ).fetchall()

    return [dict(row) for row in rows]


def get_interview_statistics(
    job_fingerprint: str,
    db_path: str = DB_PATH,
) -> Dict[str, Any]:
    """Return useful recruiter interview metrics for a JD."""
    interviews = list_interviews(job_fingerprint, db_path=db_path)

    stats = {
        "total": len(interviews),
        "scheduled": 0,
        "completed": 0,
        "cancelled": 0,
        "rescheduled": 0,
        "pending_feedback": 0,
        "proceed": 0,
        "hold": 0,
        "reject": 0,
    }

    for interview in interviews:
        status = interview.get("status")
        if status == "Scheduled":
            stats["scheduled"] += 1
        elif status == "Completed":
            stats["completed"] += 1
        elif status == "Cancelled":
            stats["cancelled"] += 1
        elif status == "Rescheduled":
            stats["rescheduled"] += 1

        recommendation = interview.get("recommendation")
        if status == "Completed" and recommendation == "Pending":
            stats["pending_feedback"] += 1
        elif recommendation == "Proceed":
            stats["proceed"] += 1
        elif recommendation == "Hold":
            stats["hold"] += 1
        elif recommendation == "Reject":
            stats["reject"] += 1

    return stats


def generate_ics_event(interview: Dict[str, Any]) -> str:
    """Generate a lightweight RFC-style .ics event."""
    date_text = str(interview.get("scheduled_date", "")).strip()
    time_text = str(interview.get("scheduled_time", "")).strip()

    try:
        start = datetime.fromisoformat(f"{date_text}T{time_text}")
    except ValueError:
        raise ValueError("Invalid scheduled date/time for calendar export.")

    duration = int(interview.get("duration_minutes") or 60)
    end = start.fromtimestamp(start.timestamp())
    end = end.replace()  # preserve naive datetime
    from datetime import timedelta
    end = start + timedelta(minutes=duration)

    start_stamp = start.strftime("%Y%m%dT%H%M%S")
    end_stamp = end.strftime("%Y%m%dT%H%M%S")

    summary = (
        f"{interview.get('interview_round', 'Interview')} - "
        f"{interview.get('candidate_name', 'Candidate')}"
    )

    description_parts = [
        f"Type: {interview.get('interview_type', '')}",
        f"Interviewer: {interview.get('interviewer', '')}",
        f"Status: {interview.get('status', '')}",
    ]

    meeting_link = str(interview.get("meeting_link") or "").strip()
    if meeting_link:
        description_parts.append(f"Meeting link: {meeting_link}")

    description = "\\n".join(description_parts)

    # Basic ICS escaping.
    def esc(value: str) -> str:
        return (
            str(value)
            .replace("\\", "\\\\")
            .replace(";", "\\;")
            .replace(",", "\\,")
            .replace("\n", "\\n")
        )

    uid = f"resume-screening-{interview.get('id', 'event')}@local"

    return (
        "BEGIN:VCALENDAR\r\n"
        "VERSION:2.0\r\n"
        "PRODID:-//AI Resume Screening Platform//EN\r\n"
        "BEGIN:VEVENT\r\n"
        f"UID:{esc(uid)}\r\n"
        f"DTSTAMP:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}\r\n"
        f"DTSTART:{start_stamp}\r\n"
        f"DTEND:{end_stamp}\r\n"
        f"SUMMARY:{esc(summary)}\r\n"
        f"DESCRIPTION:{esc(description)}\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )


def delete_interview(
    interview_id: int,
    db_path: str = DB_PATH,
) -> bool:
    """Delete one interview and its event history."""
    initialize_interview_database(db_path)

    with get_interview_connection(db_path) as conn:
        cursor = conn.execute(
            "DELETE FROM interviews WHERE id = ?",
            (int(interview_id),),
        )

    return cursor.rowcount > 0


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as temp_dir:
        test_db = str(Path(temp_dir) / "test_recruiter.db")
        initialize_database(test_db)
        initialize_interview_database(test_db)

        interview_id = create_interview(
            job_fingerprint="demo-job",
            resume_name="Adhyan_Sharma.pdf",
            candidate_name="Adhyan Sharma",
            interview_round="Technical Round",
            interview_type="Video",
            scheduled_date="2026-09-23",
            scheduled_time="15:00",
            interviewer="Hiring Manager",
            meeting_link="https://example.com/meeting",
            db_path=test_db,
        )

        assert interview_id > 0

        update_interview(
            interview_id,
            status="Completed",
            technical_score=4,
            communication_score=5,
            problem_solving_score=4,
            recommendation="Proceed",
            feedback="Strong technical discussion.",
            db_path=test_db,
        )

        item = get_interview(interview_id, test_db)
        assert item is not None
        assert item["overall_score"] == 4.33
        assert item["recommendation"] == "Proceed"

        stats = get_interview_statistics("demo-job", test_db)
        assert stats["completed"] == 1
        assert stats["proceed"] == 1

        print("Interview management engine is working correctly.")
