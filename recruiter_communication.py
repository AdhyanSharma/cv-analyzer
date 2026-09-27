"""
Recruiter Communication Engine
------------------------------

Generates personalized recruiter emails for:
- Interview invitations
- Interview reminders
- Interview follow-ups
- Selection
- Rejection
- Hold / additional-information requests

Also stores generated messages in the shared SQLite recruiter database.

No external dependencies.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import quote

from recruiter_store import DB_PATH, initialize_database, make_candidate_key


TEMPLATE_OPTIONS = [
    "Interview Invitation",
    "Interview Reminder",
    "Interview Follow-up",
    "Selection",
    "Rejection",
    "On Hold",
]

COMMUNICATION_TYPES = {
    "Interview Invitation": "interview_invitation",
    "Interview Reminder": "interview_reminder",
    "Interview Follow-up": "interview_follow_up",
    "Selection": "selection",
    "Rejection": "rejection",
    "On Hold": "hold",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def get_communication_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    initialize_database(db_path)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    return conn


def initialize_communication_database(db_path: str = DB_PATH) -> None:
    initialize_database(db_path)

    with get_communication_connection(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS communications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_key TEXT NOT NULL,
                job_fingerprint TEXT NOT NULL,
                resume_name TEXT NOT NULL,
                candidate_name TEXT NOT NULL,
                candidate_email TEXT DEFAULT '',
                communication_type TEXT NOT NULL,
                subject TEXT NOT NULL,
                body TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_communications_candidate
                ON communications(candidate_key, created_at);

            CREATE INDEX IF NOT EXISTS idx_communications_job
                ON communications(job_fingerprint, created_at);
            """
        )


def _clean(value: Any, fallback: str = "") -> str:
    text = str(value or "").strip()
    return text if text else fallback


def build_mailto_url(
    recipient: str,
    subject: str,
    body: str,
) -> str:
    """
    Build a mailto URL for opening the user's default mail application.

    This does not send email automatically.
    """
    recipient = _clean(recipient)
    return (
        "mailto:"
        + quote(recipient, safe="@,")
        + "?subject="
        + quote(subject, safe="")
        + "&body="
        + quote(body, safe="")
    )


def generate_communication(
    template: str,
    *,
    candidate_name: str,
    candidate_email: str = "",
    role_title: str = "the position",
    company_name: str = "our company",
    recruiter_name: str = "Recruitment Team",
    interview_round: str = "",
    interview_date: str = "",
    interview_time: str = "",
    duration_minutes: Any = 60,
    interviewer: str = "",
    meeting_link: str = "",
    feedback: str = "",
    overall_score: Any = "",
) -> Dict[str, str]:
    """Generate subject/body from a communication template."""
    if template not in TEMPLATE_OPTIONS:
        raise ValueError(f"Unknown communication template: {template}")

    name = _clean(candidate_name, "Candidate")
    role = _clean(role_title, "the position")
    company = _clean(company_name, "our company")
    recruiter = _clean(recruiter_name, "Recruitment Team")
    round_name = _clean(interview_round, "interview")
    date = _clean(interview_date, "the scheduled date")
    time = _clean(interview_time, "the scheduled time")
    interviewer_name = _clean(interviewer, recruiter)

    try:
        duration = int(duration_minutes)
    except (TypeError, ValueError):
        duration = 60

    link = _clean(meeting_link)
    extra_feedback = _clean(feedback)
    score_text = _clean(overall_score)

    if template == "Interview Invitation":
        subject = f"Interview Invitation — {role}"
        body = (
            f"Dear {name},\n\n"
            f"Thank you for your interest in {role} at {company}.\n\n"
            f"We would like to invite you to the {round_name} for the {role} position. "
            f"The interview is scheduled for {date} at {time} and is expected to take "
            f"approximately {duration} minutes.\n\n"
            f"Interviewer: {interviewer_name}\n"
        )
        if link:
            body += f"Meeting link: {link}\n"
        body += (
            "\nPlease let us know if you need any clarification or require a different "
            "time slot.\n\n"
            f"Best regards,\n{recruiter}\n{company}"
        )

    elif template == "Interview Reminder":
        subject = f"Reminder: {round_name} — {role}"
        body = (
            f"Dear {name},\n\n"
            f"This is a quick reminder about your upcoming {round_name} for the "
            f"{role} position at {company}.\n\n"
            f"Date: {date}\n"
            f"Time: {time}\n"
            f"Duration: {duration} minutes\n"
            f"Interviewer: {interviewer_name}\n"
        )
        if link:
            body += f"Meeting link: {link}\n"
        body += (
            "\nPlease join a few minutes early and keep any relevant project or technical "
            "material available.\n\n"
            f"Best regards,\n{recruiter}"
        )

    elif template == "Interview Follow-up":
        subject = f"Thank You for Interviewing — {role}"
        body = (
            f"Dear {name},\n\n"
            f"Thank you for taking the time to attend the {round_name} for the {role} "
            f"position at {company}.\n\n"
            "We appreciate the opportunity to learn more about your experience and "
            "background. Our team will review the discussion and follow up with you "
            "regarding the next steps.\n\n"
        )
        if extra_feedback:
            body += f"Additional note: {extra_feedback}\n\n"
        body += f"Best regards,\n{recruiter}"

    elif template == "Selection":
        subject = f"Congratulations — Selected for {role}"
        body = (
            f"Dear {name},\n\n"
            f"We are pleased to let you know that you have been selected to move forward "
            f"for the {role} position at {company}.\n\n"
            "Congratulations on reaching this stage. Our team will share the next steps "
            "and any required documentation with you shortly.\n\n"
        )
        if score_text:
            body += f"Interview score recorded by the recruiting team: {score_text}/5.\n\n"
        body += (
            f"Please reply to this email if you have any questions.\n\n"
            f"Best regards,\n{recruiter}"
        )

    elif template == "Rejection":
        subject = f"Update on Your Application — {role}"
        body = (
            f"Dear {name},\n\n"
            f"Thank you for your interest in the {role} position at {company} and for "
            "the time you invested in our recruitment process.\n\n"
            "After careful consideration, we will not be moving forward with your "
            "application for this position at this time.\n\n"
        )
        if extra_feedback:
            body += f"Feedback: {extra_feedback}\n\n"
        body += (
            "We appreciate your effort and wish you the very best in your job search.\n\n"
            f"Best regards,\n{recruiter}"
        )

    else:  # On Hold
        subject = f"Application Update — {role}"
        body = (
            f"Dear {name},\n\n"
            f"Thank you for your interest in the {role} position at {company}.\n\n"
            "We are currently keeping your application on hold while we complete the "
            "next stage of our review. We will contact you when there is an update.\n\n"
        )
        if extra_feedback:
            body += f"Additional note: {extra_feedback}\n\n"
        body += f"Best regards,\n{recruiter}"

    return {
        "template": template,
        "communication_type": COMMUNICATION_TYPES[template],
        "recipient": _clean(candidate_email),
        "subject": subject.strip(),
        "body": body.strip(),
    }


def save_communication(
    *,
    job_fingerprint: str,
    resume_name: str,
    candidate_name: str,
    candidate_email: str,
    communication: Dict[str, str],
    db_path: str = DB_PATH,
) -> int:
    """Persist one generated communication and return its ID."""
    initialize_communication_database(db_path)

    now = _utc_now()
    candidate_key = make_candidate_key(job_fingerprint, resume_name)

    with get_communication_connection(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO communications (
                candidate_key,
                job_fingerprint,
                resume_name,
                candidate_name,
                candidate_email,
                communication_type,
                subject,
                body,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                candidate_key,
                job_fingerprint,
                resume_name,
                candidate_name,
                candidate_email,
                communication.get("communication_type", "custom"),
                communication.get("subject", ""),
                communication.get("body", ""),
                now,
            ),
        )

    return int(cursor.lastrowid)


def list_communications(
    job_fingerprint: str,
    *,
    resume_name: Optional[str] = None,
    db_path: str = DB_PATH,
) -> List[Dict[str, Any]]:
    """Return saved communications, newest first."""
    initialize_communication_database(db_path)

    query = """
        SELECT *
        FROM communications
        WHERE job_fingerprint = ?
    """
    params: List[Any] = [job_fingerprint]

    if resume_name:
        query += " AND resume_name = ?"
        params.append(resume_name)

    query += " ORDER BY id DESC"

    with get_communication_connection(db_path) as conn:
        rows = conn.execute(query, params).fetchall()

    return [dict(row) for row in rows]


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as temp_dir:
        test_db = str(Path(temp_dir) / "test_recruiter.db")

        message = generate_communication(
            "Interview Invitation",
            candidate_name="Adhyan Sharma",
            candidate_email="adhyan@example.com",
            role_title="AI Engineer",
            company_name="Example Technologies",
            recruiter_name="Recruitment Team",
            interview_round="Technical Round",
            interview_date="2026-09-25",
            interview_time="15:00",
            interviewer="Technical Lead",
            meeting_link="https://example.com/meeting",
        )

        assert "Adhyan Sharma" in message["body"]
        assert "Technical Round" in message["body"]

        communication_id = save_communication(
            job_fingerprint="demo-job",
            resume_name="Adhyan_Sharma.pdf",
            candidate_name="Adhyan Sharma",
            candidate_email="adhyan@example.com",
            communication=message,
            db_path=test_db,
        )

        assert communication_id > 0

        history = list_communications(
            "demo-job",
            resume_name="Adhyan_Sharma.pdf",
            db_path=test_db,
        )

        assert len(history) == 1

        mailto = build_mailto_url(
            "adhyan@example.com",
            message["subject"],
            message["body"],
        )

        assert mailto.startswith("mailto:")

        print("Recruiter communication engine is working correctly.")
