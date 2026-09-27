# V17 Production Architecture

```text
                           ┌──────────────────────┐
                           │       Browser        │
                           └──────────┬───────────┘
                                      │ HTTPS
                                      v
                           ┌──────────────────────┐
                           │ Render Frontend      │
                           │ React + Nginx        │
                           └──────────┬───────────┘
                                      │ HTTPS REST
                                      v
                           ┌──────────────────────┐
                           │ Render API           │
                           │ FastAPI + Uvicorn    │
                           └──────────┬───────────┘
                                      │
                         ┌────────────┴────────────┐
                         │                         │
                         v                         v
                ┌────────────────┐       ┌─────────────────┐
                │ Render Postgres│       │ Gemini API      │
                │                │       │ optional        │
                └────────────────┘       └─────────────────┘
```

## CI/CD flow

```text
Developer
    |
    | git push
    v
GitHub
    |
    v
GitHub Actions
    |
    +-- pytest
    +-- npm build
    +-- docker compose validation
    +-- docker image builds
    |
    | all checks pass
    v
Render
    |
    +-- API deploy
    +-- Frontend deploy
    +-- health checks
    |
    v
Public application
```
