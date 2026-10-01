@echo off
setlocal
title CV Analyzer V18.3

cd /d "C:\Users\ASUS\CV Analyzer"

echo.
echo =============================================
echo       CV ANALYZER V18.3 - LAUNCHING
echo =============================================
echo.
echo Checking Docker engine...

docker info >nul 2>&1
if not errorlevel 1 goto START_PROJECT

echo Docker is not running. Starting Docker Desktop...
start "" "%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"

echo Waiting for Docker engine...
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ok=$false; for($i=0;$i -lt 30;$i++){Start-Sleep -Seconds 2; docker info *> $null; if($LASTEXITCODE -eq 0){$ok=$true;break}}; if(-not $ok){exit 1}"

if errorlevel 1 goto DOCKER_ERROR

:START_PROJECT
echo Docker engine is ready.
echo.
echo Starting CV Analyzer...
docker compose up -d

if errorlevel 1 goto COMPOSE_ERROR

echo.
echo Services started.
docker compose ps

echo.
echo Opening CV Analyzer...
timeout /t 3 /nobreak >nul
start "" "http://localhost:8081"

echo.
echo =============================================
echo       CV ANALYZER IS RUNNING
echo       http://localhost:8081
echo =============================================
echo.
pause
exit /b 0

:DOCKER_ERROR
echo.
echo ERROR: Docker engine did not become ready.
pause
exit /b 1

:COMPOSE_ERROR
echo.
echo ERROR: Docker Compose failed.
docker compose ps
pause
exit /b 1
