#!/usr/bin/env bash
set -euo pipefail
export PYTHONUTF8=1 PYTHONIOENCODING=utf-8
exec python -m uvicorn backend.app.main:app --reload --port 8000
