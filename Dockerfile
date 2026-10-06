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

# libgl1 + libglib2.0-0 are required by opencv-python, which DeepFace and
# retina-face depend on directly (the headless build would not satisfy them).
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root uid 1000: required for writes on Hugging Face Spaces and good
# practice everywhere else.
RUN useradd --create-home --uid 1000 app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    HOME=/home/app \
    DEEPFACE_HOME=/home/app \
    MPLCONFIGDIR=/tmp/mpl \
    PORT=7860

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY --chown=app:app backend/ ./backend/
# The compiled SPA sits where config.py looks for it first.
COPY --from=frontend --chown=app:app /build/dist ./backend/static

USER app

# Bake the weights into the image so the first request never waits on a
# download. Set PREFETCH_RETINAFACE=1 to include the 118 MB RetinaFace model.
ARG PREFETCH_RETINAFACE=0
ENV PREFETCH_RETINAFACE=${PREFETCH_RETINAFACE}
RUN python backend/scripts/prefetch_models.py

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT}/api/health" || exit 1

# Single worker: the models are the memory cost, and duplicating them across
# workers is the fastest way to be OOM-killed on a free instance.
CMD ["sh", "-c", "exec uvicorn backend.app.main:app --host 0.0.0.0 --port ${PORT} --workers 1 --timeout-keep-alive 65"]
