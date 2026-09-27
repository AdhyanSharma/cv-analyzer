# CV Analyzer — V14

AI Resume Screening Platform with React + FastAPI + SQLAlchemy/Alembic and explainable recruiter decision support.

## V14 additions

- Audit logs for successful recruiter activity.
- Persistent evaluation runs.
- Recruiter-labeled threshold diagnostics: accuracy, precision, recall, F1 and confusion counts.
- Responsible-AI notice explaining that platform signals support human review and that missing detections do not prove absence of a skill.
- React **Evaluation & Audit** workspace.

## Project layout

```text
backend/
  api.py
  backend_service.py
  database.py
  models.py
  platform_store.py
  security.py
  v14_schemas.py
  responsible_ai.py
  alembic.ini
  alembic/
frontend/
  src/
  package.json
```

See `V14_SETUP.md` for local setup.
