# CV Analyzer V15

Production-oriented Docker Compose package for the existing AI recruitment platform.

Start with:

```powershell
Copy-Item ".env.example" ".env" -Force
docker compose up --build -d
```

Frontend: http://127.0.0.1:8080
API docs: http://127.0.0.1:8000/docs
Readiness: http://127.0.0.1:8080/health/ready

See `V15_SETUP.md` for Windows setup, migration, backup and deployment notes.

## CPU-optimized build

The CPU-optimized Docker variant installs CPU-only PyTorch before the remaining Python dependencies so the API image does not pull the large CUDA/NVIDIA runtime stack.
