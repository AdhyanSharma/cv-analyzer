@echo off
setlocal EnableExtensions EnableDelayedExpansion
title CV Analyzer V18.3 - Launch

echo.
echo =============================================
echo          CV ANALYZER V18.3 - LAUNCHING
echo =============================================
echo.

set "PROJECT=C:\Users\ASUS\CV Analyzer"
set "DOCKER_DESKTOP=%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"

cd /d "%PROJECT%" || goto project_error

echo Checking Docker engine...
docker info >nul 2>&1
if not errorlevel 1 goto docker_ready

echo Docker engine is not running.
echo Starting Docker Desktop...

if not exist "%DOCKER_DESKTOP%" goto docker_error

start "" "%DOCKER_DESKTOP%"

echo Waiting for Docker engine...
set /a COUNT=0

:WAIT_DOCKER
timeout /t 3 /nobreak >nul
docker info >nul 2>&1
if not errorlevel 1 goto docker_ready

set /a COUNT+=1
if !COUNT! GEQ 20 goto docker_timeout
goto WAIT_DOCKER

:docker_ready
echo Docker engine is ready.
echo.

echo Starting CV Analyzer services...
docker compose up -d
if errorlevel 1 goto compose_error

echo.
echo Checking service status...
docker compose ps

echo.
echo Waiting for services to initialize...
timeout /t 5 /nobreak >nul

echo Opening CV Analyzer...
start "" "http://localhost:8081"

echo.
echo =============================================
echo       CV ANALYZER IS STARTING
echo =============================================
echo Frontend: http://localhost:8081
echo API:      http://localhost:8000
echo Health:   http://localhost:8000/health
echo =============================================
echo.
pause
exit /b 0

:project_error
echo ERROR: Project folder not found:
echo %PROJECT%
pause
exit /b 1

:docker_error
echo ERROR: Docker Desktop was not found at:
echo %DOCKER_DESKTOP%
pause
exit /b 1

:docker_timeout
echo.
echo ERROR: Docker engine did not become ready within 60 seconds.
echo Please open Docker Desktop and try again.
pause
exit /b 1

:compose_error
echo.
echo ERROR: Docker Compose could not start CV Analyzer.
echo.
docker compose ps
pause
exit /b 1
