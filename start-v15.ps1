# V15 one-command Windows launcher
$ErrorActionPreference = "Stop"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created .env from .env.example. Edit POSTGRES_PASSWORD and JWT_SECRET before first production-style start." -ForegroundColor Yellow
    exit 1
}

docker compose up --build -d
Write-Host "V15 started." -ForegroundColor Green
Write-Host "Frontend : http://127.0.0.1:8080"
Write-Host "API      : http://127.0.0.1:8000/docs"
Write-Host "Health   : http://127.0.0.1:8080/health/ready"
