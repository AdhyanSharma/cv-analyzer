$ErrorActionPreference = "Stop"

Write-Host "=== CV Analyzer V16 local CI ===" -ForegroundColor Cyan

Write-Host "`n[1/4] Backend tests" -ForegroundColor Yellow
python -m pytest backend/tests -q

Write-Host "`n[2/4] Frontend build" -ForegroundColor Yellow
Push-Location frontend
try {
    if (Test-Path package-lock.json) {
        npm ci
    } else {
        npm install
    }
    npm run build
}
finally {
    Pop-Location
}

Write-Host "`n[3/4] Docker Compose validation" -ForegroundColor Yellow
docker compose config --quiet

Write-Host "`n[4/4] Docker images" -ForegroundColor Yellow
docker compose build api frontend

Write-Host "`nV16 local CI checks completed successfully." -ForegroundColor Green
