@echo off
setlocal

title CV Analyzer - Launch
set "PROJECT=C:\Users\ASUS\CV Analyzer"
set "DOCKER_DESKTOP=%LOCALAPPDATA%\Programs\DockerDesktop\Docker Desktop.exe"
set "FRONTEND_URL=http://localhost:8081"

cd /d "%PROJECT%" || (
  echo Project folder not found: %PROJECT%
  pause
  exit /b 1
)

echo.
echo =============================================
echo          CV ANALYZER V18.3 - LAUNCHING
echo =============================================
echo.

echo Checking Docker engine...
docker info >nul 2>&1
if errorlevel 1 (
  echo Docker engine is not running. Starting Docker Desktop...
  if not exist "%DOCKER_DESKTOP%" (
    echo Docker Desktop not found:
    echo %DOCKER_DESKTOP%
    pause
    exit /b 1
  )
  start "" "%DOCKER_DESKTOP%"

  set /a ATTEMPTS=0
  :WAIT_DOCKER
  timeout /t 3 /nobreak >nul
  docker info >nul 2>&1
  if not errorlevel 1 goto DOCKER_READY
  set /a ATTEMPTS+=1
  if %ATTEMPTS% GEQ 20 goto DOCKER_TIMEOUT
  goto WAIT_DOCKER
)

:DOCKER_READY
echo Docker engine ready.
echo Starting CV Analyzer containers...
docker compose up -d
if errorlevel 1 (
  echo.
  echo Docker Compose failed.
  docker compose ps
  pause
  exit /b 1
)

echo.
echo Waiting for the website to respond...
set /a SITE_ATTEMPTS=0
:WAIT_SITE
timeout /t 2 /nobreak >nul
curl.exe -fsS "%FRONTEND_URL%" >nul 2>&1
if not errorlevel 1 goto SITE_READY
set /a SITE_ATTEMPTS+=1
if %SITE_ATTEMPTS% GEQ 30 goto SITE_TIMEOUT
goto WAIT_SITE

:SITE_READY
echo CV Analyzer is ready.
start "" "%FRONTEND_URL%"

echo.
echo =============================================
echo Frontend: %FRONTEND_URL%
echo API:      http://localhost:8000
echo Health:   http://localhost:8000/health
echo =============================================
echo.
docker compose ps
pause
exit /b 0

:DOCKER_TIMEOUT
echo.
echo Docker engine did not become ready within 60 seconds.
echo Open Docker Desktop and try again.
pause
exit /b 1

:SITE_TIMEOUT
echo.
echo Containers started, but the frontend did not respond within 60 seconds.
echo Check: docker compose ps
echo Check logs: docker compose logs -f frontend
start "" "%FRONTEND_URL%"
pause
exit /b 1
