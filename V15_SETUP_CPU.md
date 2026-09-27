# V15 CPU-Optimized Docker Setup

This variant keeps the V15 application stack unchanged but installs CPU-only PyTorch in the API image before the rest of the Python dependencies. The goal is to avoid downloading the large NVIDIA CUDA runtime packages when GPU passthrough is not being used.

## Replace the project files

Extract this ZIP, then copy the full contents into:

`C:\Users\ASUS\CV Analyzer`

Use one PowerShell command:

```powershell
Copy-Item "C:\Users\ASUS\Downloads\CV_Analyzer_V15_Docker_CPU_Optimized\*" "C:\Users\ASUS\CV Analyzer" -Recurse -Force
```

## Configure `.env`

Keep your PostgreSQL settings and use a long JWT secret. Example:

```text
POSTGRES_DB=cv_analyzer
POSTGRES_USER=cv_admin
POSTGRES_PASSWORD=CvAnalyzerPG2026SecurePass
JWT_SECRET=AdhyanCVAnalyzer-V15-Production-Secret-2026-LongKey
JWT_EXPIRES_HOURS=8
AUTO_CREATE_DB=false
CORS_ORIGINS=http://localhost:8080,http://127.0.0.1:8080
UVICORN_WORKERS=2
DATABASE_URL=postgresql+psycopg://cv_admin:CvAnalyzerPG2026SecurePass@db:5432/cv_analyzer
```

## Clean the failed build cache

```powershell
docker compose down --remove-orphans
docker builder prune -af
```

## Build

```powershell
docker compose build --no-cache
```

## Start

```powershell
docker compose up -d
```

## Verify

```powershell
docker compose ps
docker compose logs api --tail 100
```

Open:

- `http://127.0.0.1:8080` — recruiter UI
- `http://127.0.0.1:8000/docs` — API docs
- `http://127.0.0.1:8080/health/ready` — readiness
