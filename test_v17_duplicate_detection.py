import hashlib

from platform_store import PlatformStore


def test_exact_resume_hash_reuses_candidate_and_application(tmp_path):
    db_path = tmp_path / "duplicate_test.db"
    store = PlatformStore(db_path=str(db_path), auto_create=True)

    user = store.create_user(
        "duplicate-test@example.com",
        "Duplicate Test Recruiter",
        "test-password-hash",
    )
    job = store.create_job(
        user["id"],
        "Python AI Engineer",
        "Test Company",
        "Python, FastAPI, PostgreSQL",
    )

    raw_resume = b"same resume bytes"
    resume_hash = hashlib.sha256(raw_resume).hexdigest()

    first = store.upsert_candidate(
        user["id"],
        name="Candidate One",
        email="",
        phone="",
        linkedin="",
        github="",
        resume_filename="resume.pdf",
        resume_text="Python FastAPI",
        resume_hash=resume_hash,
    )

    second = store.upsert_candidate(
        user["id"],
        name="Candidate One",
        email="",
        phone="",
        linkedin="",
        github="",
        resume_filename="resume-copy.pdf",
        resume_text="Python FastAPI",
        resume_hash=resume_hash,
    )

    assert first["id"] == second["id"]

    app1 = store.add_application(
        job["id"],
        first["id"],
        {"ats": {"ats_score": 80}},
    )
    app2 = store.add_application(
        job["id"],
        second["id"],
        {"ats": {"ats_score": 80}},
    )

    assert app1["application_id"] == app2["application_id"]

    candidates = store.list_applications(user["id"], job["id"])
    assert len(candidates) == 1
