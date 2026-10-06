"""Runtime configuration.

Every value can be overridden with an environment variable so the same image
runs locally, on Hugging Face Spaces, Render, Railway or Fly without a rebuild.
"""

from __future__ import annotations

import os
from pathlib import Path

APP_NAME = "FaceVision Studio"
APP_VERSION = "1.1.0"

# backend/app/config.py -> backend/
BACKEND_ROOT = Path(__file__).resolve().parent.parent
# -> repository root
PROJECT_ROOT = BACKEND_ROOT.parent


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env_str(name, str(default)))
    except ValueError:
        return default


def _env_bool(name: str, default: bool) -> bool:
    return _env_str(name, "1" if default else "0").lower() in {"1", "true", "yes", "on"}


def _env_path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    return Path(raw).expanduser().resolve() if raw and raw.strip() else default


# --- Storage -----------------------------------------------------------------
# GALLERY_DIR can point at a mounted disk so enrolled people survive redeploys.
GALLERY_DIR = _env_path("GALLERY_DIR", BACKEND_ROOT / "gallery")

# Where the compiled Vite bundle lives. The Docker image copies it next to the
# backend; a local checkout keeps it under frontend/dist.
_DIST_CANDIDATES = [
    _env_path("FRONTEND_DIST", BACKEND_ROOT / "static"),
    PROJECT_ROOT / "frontend" / "dist",
]
FRONTEND_DIST = next((p for p in _DIST_CANDIDATES if (p / "index.html").exists()), None)

# --- HTTP --------------------------------------------------------------------
# Same-origin deployments need no CORS at all; "*" keeps a split
# frontend/backend deployment working out of the box.
_origins = _env_str("ALLOWED_ORIGINS", "*")
ALLOWED_ORIGINS = ["*"] if _origins == "*" else [o.strip() for o in _origins.split(",") if o.strip()]
ALLOW_CREDENTIALS = ALLOWED_ORIGINS != ["*"]

MAX_UPLOAD_BYTES = _env_int("MAX_UPLOAD_MB", 12) * 1024 * 1024

# --- Vision ------------------------------------------------------------------
# Inputs are fitted inside this box before any model runs. This bounds both
# latency and peak memory, which matters on small free-tier containers.
MAX_IMAGE_DIM = _env_int("MAX_IMAGE_DIM", 1600)

# Cosine similarity a FaceNet embedding must reach to claim an identity.
# DeepFace's reference cosine *distance* threshold for FaceNet is 0.40,
# i.e. a similarity of 0.60.
RECOGNITION_THRESHOLD = _env_float("RECOGNITION_THRESHOLD", 0.60)

# Combined grayscale+edge correlation a template match must reach.
TEMPLATE_MIN_SCORE = _env_float("TEMPLATE_MIN_SCORE", 0.60)

# Deep-detector confidence floor.
DETECTION_MIN_CONFIDENCE = _env_float("DETECTION_MIN_CONFIDENCE", 0.35)

# Deep detector backends. Measured on an 820x400 photo, warm:
#   yunet       0.015 s   228 KB  ONNX via OpenCV - ships with the image
#   retinaface  14.6  s   118 MB  needs TensorFlow (~1 GB RAM); optional extra
# RetinaFace has better recall on hard poses but cannot run on a free-tier
# container, so YuNet is the default and RetinaFace is opt-in for local work.
DETECTOR_BACKENDS = ("yunet", "retinaface")
_requested_backend = _env_str("DETECTOR_BACKEND", "yunet").lower()
DETECTOR_BACKEND = _requested_backend if _requested_backend in DETECTOR_BACKENDS else "yunet"

# RetinaFace pulls in TensorFlow, so it is off unless explicitly enabled.
ENABLE_RETINAFACE = _env_bool("ENABLE_RETINAFACE", False)

# ONNX Runtime threads. 1 suits a shared/fractional vCPU; raise it on real cores.
ONNX_THREADS = _env_int("ONNX_THREADS", 1)

# Set DEEP_MODELS_ENABLED=0 for a classical-CV-only deployment that loads no
# neural weights at all (Viola-Jones and template matching stay fully working).
DEEP_MODELS_ENABLED = _env_bool("DEEP_MODELS_ENABLED", True)

# Build FaceNet gallery embeddings in a background thread at startup so the
# first recognition request is not the one that pays for it.
WARM_UP_ON_STARTUP = _env_bool("WARM_UP_ON_STARTUP", True)

# Historic folder spellings are still displayed under their corrected name.
NAME_ALIASES = {
    "Elon Mask": "Elon Musk",
    "Sundhar Pichai": "Sundar Pichai",
}
