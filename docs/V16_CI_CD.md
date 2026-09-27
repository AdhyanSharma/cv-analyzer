# V16 — CI/CD + Automated Testing

## What V16 adds

V16 turns the V15 production stack into a project with repeatable engineering checks:

- Backend API smoke tests with `pytest`
- React production build validation
- Docker Compose configuration validation
- API and frontend Docker image build validation
- GitHub Actions CI on pushes and pull requests
- GitHub Container Registry publication on `main` and version tags

## GitHub Actions

### `ci.yml`

Runs for pushes and pull requests against:

- `main`
- `master`
- `develop`

Pipeline:

1. PostgreSQL 17 service starts.
2. Python 3.11 dependencies are installed.
3. Backend tests run.
4. Node 20 dependencies are installed.
5. React/Vite production build runs.
6. Docker Compose is validated.
7. API and frontend images are built.

### `docker-publish.yml`

Runs on:

- Push to `main`
- Version tags such as `v16.0.0`
- Manual workflow dispatch

Images are published to GitHub Container Registry:

- `ghcr.io/<owner>/cv-analyzer-api`
- `ghcr.io/<owner>/cv-analyzer-frontend`

The workflow uses the built-in GitHub Actions token, so no Docker Hub credentials are required.

## Important V15 compatibility note

The backend health contract is pinned to the verified V15 service:

```json
{
  "status": "ok",
  "service": "cv-analyzer-api",
  "version": "v15"
}
```

When the application version changes later, update the assertion in:

`backend/tests/test_health.py`

## Local validation

From the project root:

```powershell
.\scripts\run_ci_local.ps1
```

or run the checks manually:

```powershell
python -m pytest backend/tests -q

cd frontend
npm ci
npm run build
cd ..

docker compose config --quiet
docker compose build api frontend
```

## Recommended GitHub setup

After pushing these files:

1. Open the repository's **Actions** tab.
2. Confirm the `CI` workflow appears.
3. Push a commit or open a pull request.
4. Wait for all three CI jobs to pass.
5. Merge to `main`.
6. The Docker publication workflow will build and push the two images to GHCR.

## Portfolio impact

The project now demonstrates more than model and application development. It also contains automated verification and container delivery, which are practical software-engineering capabilities for an AI application.
