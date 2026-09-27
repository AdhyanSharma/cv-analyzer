"""Smoke tests for V11 auth, jobs and persistence."""
from __future__ import annotations

import os
import tempfile

from platform_store import PlatformStore
from security import create_access_token, decode_access_token, hash_password, verify_password


def main() -> None:
    password = "StrongPass123!"
    encoded = hash_password(password)
    assert verify_password(password, encoded)
    assert not verify_password("WrongPass123!", encoded)

    token = create_access_token("user-123", "user@example.com")
    payload = decode_access_token(token)
    assert payload["sub"] == "user-123"

    with tempfile.TemporaryDirectory() as tmp:
        store = PlatformStore(os.path.join(tmp, "test.db"))
        user = store.create_user("user@example.com", "Test Recruiter", encoded)
        job = store.create_job(user["id"], "AI Engineer", "DemoCo", "Build ML APIs and retrieval systems for an AI platform.")
        assert job["status"] == "open"
        jobs = store.list_jobs(user["id"])
        assert len(jobs) == 1
        updated = store.update_job(user["id"], job["id"], status="closed")
        assert updated["status"] == "closed"
        assert store.delete_job(user["id"], job["id"])

    print("V11 smoke tests passed.")


if __name__ == "__main__":
    main()
