FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System libraries needed by the scientific/ML Python stack.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements-api-v15-cpu.txt /app/requirements.txt

# Install the CPU-only PyTorch build first. This prevents pip from pulling the
# much larger CUDA/NVIDIA dependency stack into the production image.
RUN python -m pip install --upgrade pip \
    && python -m pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && python -m pip install -r /app/requirements.txt

# Reuse the existing CV Analyzer AI pipeline from the project root first.
# Important: copy these BEFORE backend infrastructure so the V15 backend files
# below override any older root-level api.py/database.py/etc.
COPY *.py /app/

# Backend infrastructure (V15 wins over any older root-level files).
COPY backend/*.py /app/
COPY backend/alembic /app/alembic
COPY backend/alembic.ini /app/alembic.ini

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=8s --start-period=40s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=5).read()"

CMD ["sh", "-c", "alembic upgrade head && uvicorn api:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS:-2}"]
