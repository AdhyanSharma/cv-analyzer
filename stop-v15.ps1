$ErrorActionPreference = "Stop"
docker compose down
Write-Host "V15 containers stopped. PostgreSQL data remains in the Docker volume." -ForegroundColor Green
