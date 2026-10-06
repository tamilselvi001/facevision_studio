# syntax=docker/dockerfile:1

# ---------------------------------------------------------------- frontend --
# Built in a Node stage; the runtime image never needs Node at all.
FROM node:20-slim AS frontend

WORKDIR /build
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
RUN npm run build


# ----------------------------------------------------------------- runtime --
FROM python:3.11-slim

# opencv-python-headless has no GUI dependencies, so no libgl1/libglib2.0-0 and
# no build toolchain are needed. curl is only here for the healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root uid 1000.
RUN useradd --create-home --uid 1000 app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    HOME=/home/app \
    OMP_NUM_THREADS=1 \
    ONNX_THREADS=1 \
    PORT=10000

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

# Models (FaceNet ONNX fp16 ~44 MB + YuNet ~228 KB) ship inside the image, so a
# cold start never waits on a download.
COPY --chown=app:app backend/ ./backend/
# The compiled SPA sits where config.py looks for it first.
COPY --from=frontend --chown=app:app /build/dist ./backend/static

USER app

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# Single worker: model memory is the constraint, and duplicating it across
# workers is the fastest way to be OOM-killed on a 512 MB instance.
CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 65"]
