# CV Analyzer V15 — Docker + Production Setup

V15 packages the existing FastAPI backend, AI pipeline, React recruiter UI, and PostgreSQL database into a repeatable Docker Compose deployment.

## 1. What V15 changes

- FastAPI container with Alembic migrations.
- PostgreSQL 17 container with persistent volume.
- React/Vite production build served by nginx.
- nginx proxies `/api/*` to FastAPI, so the browser can use a same-origin API in Docker.
- `/health` is a liveness probe.
- `/health/ready` verifies database connectivity.
- `migrate_sqlite_to_postgres.py` now carries V14 audit/evaluation records as well as V11 core records.

## 2. Windows setup

Extract V15. Copy the **entire extracted V15 folder contents** into:

`C:\Users\ASUS\CV Analyzer`

This does not replace your existing top-level AI modules; the V15 archive adds/updates the backend/frontend/deployment files.

Open PowerShell in the project root:

```powershell
cd "C:\Users\ASUS\CV Analyzer"
Copy-Item ".env.example" ".env" -Force
```

Edit `.env` and set a strong password and JWT secret. The JWT secret should be at least 32 random characters.

## 3. Start the full stack

```powershell
docker compose up --build -d
```

Then inspect:

```powershell
docker compose ps
docker compose logs -f api
```

Expected services:

- `db` healthy
- `api` healthy
- `frontend` healthy

Open:

- Frontend: `http://127.0.0.1:8080`
- FastAPI docs: `http://127.0.0.1:8000/docs`
- Readiness: `http://127.0.0.1:8080/health/ready`

## 4. Stop the stack

```powershell
docker compose down
```

The PostgreSQL volume is retained. To remove database data too:

```powershell
docker compose down -v
```

## 5. Existing SQLite → PostgreSQL migration

V15's migration script includes users, jobs, candidates, applications, audit logs, and evaluation runs.

First start only PostgreSQL:

```powershell
docker compose up -d db
```

Then use the Python environment that has V15 dependencies installed and run:

```powershell
python backend\migrate_sqlite_to_postgres.py `
  --sqlite sqlite:///./platform_data.db `
  --postgres postgresql+psycopg://cv_admin:YOUR_PASSWORD@localhost:5432/cv_analyzer
```

For host-side migration, expose PostgreSQL temporarily or use a local PostgreSQL instance. The standard Compose file intentionally keeps the database internal to the Docker network.

## 6. Useful commands

```powershell
docker compose ps
docker compose logs -f api
docker compose logs -f frontend
docker compose exec api alembic current
docker compose exec api alembic history
docker compose exec db psql -U $env:POSTGRES_USER -d $env:POSTGRES_DB
```

## 7. Production notes

- Replace all example secrets before deployment.
- Keep PostgreSQL private behind the Compose/network boundary.
- Use HTTPS at the public reverse proxy/load balancer.
- Store `.env` outside source control.
- Back up the PostgreSQL volume.
- Keep `AUTO_CREATE_DB=false`; schema changes should go through Alembic.
- The V15 Docker image includes the existing AI modules from the project root. When you add/rename an AI module, rebuild the API image.

## 8. Local non-Docker development

Your existing V13/V14 commands still work. V15 adds Docker as a repeatable deployment path rather than removing local development.
