# V14 Setup — Evaluation, Audit & Responsible-AI Telemetry

V14 extends the V13 React + FastAPI recruiter platform with:

- Audit logs for successful recruiter actions (login/register, job changes, screening completion, status changes, evaluation runs).
- Configurable threshold evaluation using recruiter-provided labels (`qualified` / `not_qualified`) with accuracy, precision, recall, F1 and confusion-matrix counts.
- Persistent evaluation run history.
- React UI for Evaluation and Audit log.
- Responsible-AI notice in the recruiter workspace.

## 1. Backend files

Copy the `backend\` contents into `C:\Users\ASUS\CV Analyzer` and keep the existing AI modules from V9/V10/V13.

## 2. Environment

```powershell
$env:JWT_SECRET="AdhyanCVAnalyzer-V11-JWT-Secret-2026-SecureKey"
$env:DATABASE_URL="sqlite:///./platform_data.db"
$env:AUTO_CREATE_DB="true"
```

## 3. Apply the V14 migration for PostgreSQL

From the `backend` directory when using PostgreSQL:

```powershell
alembic -c alembic.ini upgrade head
```

For local SQLite development with `AUTO_CREATE_DB=true`, the new tables are created automatically. For migration-controlled environments, use the Alembic command above.

## 4. Backend checks

```powershell
python -m py_compile api.py database.py models.py platform_store.py responsible_ai.py v14_schemas.py
python test_v14.py
```

## 5. Start API

```powershell
python -m uvicorn api:app --reload --port 8000
```

## 6. Frontend

```powershell
cd ..\frontend
npm install
Copy-Item .env.example .env
npm run dev
```

Open `http://127.0.0.1:5173`.

## Evaluation workflow

1. Open **Evaluation & Audit** in the sidebar.
2. Select a job.
3. Assign recruiter labels to candidates; use **Exclude** for candidates you do not want in the run.
4. Set the diagnostic ATS threshold.
5. Run evaluation.
6. Review the stored metrics and confusion-matrix counts.
7. Use **Audit log** to inspect recorded actions.

The evaluation layer is diagnostic. It does not change candidate status and does not create an automatic hiring decision.
