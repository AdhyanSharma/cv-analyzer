from __future__ import annotations

import os
import tempfile


def main() -> None:
    os.environ.setdefault("JWT_SECRET", "V15-test-secret-with-at-least-32-bytes-1234")
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "v15.db")
        from platform_store import PlatformStore
        from responsible_ai import evaluate_threshold

        store = PlatformStore(db_path=db, auto_create=True)
        user = store.create_user("v15@test.local", "V15 Tester", "hash")
        job = store.create_job(user["id"], "AI Engineer", "Test Co", "Python and machine learning role for validation.")
        c1 = store.upsert_candidate(user["id"], name="A", email="a@test.local", phone="", linkedin="", github="", resume_filename="a.pdf", resume_text="Python")
        c2 = store.upsert_candidate(user["id"], name="B", email="b@test.local", phone="", linkedin="", github="", resume_filename="b.pdf", resume_text="SQL")
        store.add_application(job["id"], c1["id"], {"ats_score": 80})
        store.add_application(job["id"], c2["id"], {"ats_score": 30})
        store.create_audit_log(user["id"], action="test_event", entity_type="job", entity_id=job["id"], metadata={"sample": 2})
        rows = store.list_applications(user["id"], job["id"])
        metrics = evaluate_threshold(rows, {c1["id"]: "qualified", c2["id"]: "not_qualified"}, 50)
        assert metrics["sample_size"] == 2
        assert metrics["true_positive"] == 1
        assert metrics["true_negative"] == 1
        assert metrics["accuracy"] == 100.0
        run = store.create_evaluation_run(user["id"], job["id"], metrics)
        assert store.list_evaluation_runs(user["id"], job["id"])[0]["id"] == run["id"]
        assert store.list_audit_logs(user["id"])[0]["action"] == "test_event"
        store.dispose()
    print("V15 smoke tests passed.")


if __name__ == "__main__":
    main()
