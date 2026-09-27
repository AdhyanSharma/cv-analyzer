"""V12 smoke tests: SQLAlchemy repository + relationships + application lifecycle."""
from __future__ import annotations

import os
import tempfile


def main() -> None:
    from platform_store import PlatformStore

    password_hash = "pbkdf2_sha256$310000$test$test"

    with tempfile.TemporaryDirectory() as tmp:
        url = f"sqlite:///{tmp.replace(os.sep, '/')}/v12.db"
        db = PlatformStore(database_url=url, auto_create=True)
        try:
            user = db.create_user("user@example.com", "Test Recruiter", password_hash)
            assert user["email"] == "user@example.com"

            job = db.create_job(
                user["id"],
                "AI Engineer",
                "DemoCo",
                "Build ML APIs and retrieval systems for an AI platform.",
            )
            assert job["status"] == "open"
            assert len(db.list_jobs(user["id"])) == 1

            candidate = db.upsert_candidate(
                user["id"],
                name="Test Candidate",
                email="candidate@example.com",
                phone="12345",
                linkedin="https://linkedin.com/in/test",
                github="https://github.com/test",
                resume_filename="candidate.pdf",
                resume_text="Python SQL FastAPI machine learning",
            )

            application = db.add_application(
                job["id"],
                candidate["id"],
                {"ats_score": 82.5, "matched_skills": ["Python", "SQL"]},
            )
            assert application["status"] == "new"
            assert application["analysis"]["ats_score"] == 82.5

            updated = db.update_application_status(
                user["id"], job["id"], candidate["id"], "shortlisted"
            )
            assert updated is not None
            assert updated["status"] == "shortlisted"

            rows = db.list_applications(user["id"], job["id"])
            assert len(rows) == 1
            assert rows[0]["candidate_id"] == candidate["id"]

            db.update_job(user["id"], job["id"], status="closed")
            assert db.get_job(user["id"], job["id"])["status"] == "closed"
        finally:
            db.dispose()

    print("V12 smoke tests passed.")


if __name__ == "__main__":
    main()
