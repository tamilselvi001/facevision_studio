"""FastAPI application: JSON API plus the compiled frontend on one origin."""

from __future__ import annotations

import base64
import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import config
from .vision import (
    DeepModelsUnavailable,
    clamp_box,
    deep_models_available,
    deep_models_error,
    detect_deep,
    detect_viola_jones,
    draw_detections,
    face_embedding,
    fit_within,
    list_gallery,
    model_report,
    nms,
    recognize_embedding,
    retinaface_available,
    template_match,
    warm_up,
)

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
)
log = logging.getLogger("facevision")

METHODS = {
    "robust": "Deep detector (configurable backend)",
    "viola-jones": "Viola-Jones / Haar cascade (strict)",
    "template": "Template matching (multi-scale)",
    "facenet": "Deep detector + FaceNet recognition",
}

RECOGNITION_MODEL = "FaceNet (128-d, ONNX)"

DETECTOR_LABELS = {
    "yunet": "YuNet (ONNX)",
    "retinaface": "RetinaFace (TensorFlow)",
    "viola-jones": "Viola-Jones",
}

# Measured warm latency on an 820x400 photo; surfaced so the UI can warn before
# anyone picks the slow one.
DETECTOR_PROFILE = {
    "yunet": {"label": "YuNet", "speed": "fastest", "approx_ms": 15, "weights_mb": 0.23},
    "retinaface": {"label": "RetinaFace", "speed": "slow", "approx_ms": 14500, "weights_mb": 118},
}

_warm_up_state: dict[str, Any] = {"started": False, "done": False, "result": None}


def _warm_up_background() -> None:
    try:
        _warm_up_state["result"] = warm_up()
    except Exception as exc:  # pragma: no cover
        _warm_up_state["result"] = {"deep_models": False, "error": str(exc)}
    finally:
        _warm_up_state["done"] = True
        log.info("Warm-up finished: %s", _warm_up_state["result"])


@asynccontextmanager
async def lifespan(_: FastAPI):
    config.GALLERY_DIR.mkdir(parents=True, exist_ok=True)
    log.info("%s %s", config.APP_NAME, config.APP_VERSION)
    log.info("Gallery: %s -> %s", config.GALLERY_DIR, list_gallery() or "empty")
    log.info("Frontend bundle: %s", config.FRONTEND_DIST or "not built (API-only mode)")
    if config.WARM_UP_ON_STARTUP and config.DEEP_MODELS_ENABLED:
        _warm_up_state["started"] = True
        # Daemon thread: never block the health check on a 200 MB model load.
        threading.Thread(target=_warm_up_background, name="warm-up", daemon=True).start()
    yield


app = FastAPI(
    title=f"{config.APP_NAME} API",
    version=config.APP_VERSION,
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
    openapi_url="/api/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.ALLOWED_ORIGINS,
    allow_credentials=config.ALLOW_CREDENTIALS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


@app.exception_handler(DeepModelsUnavailable)
async def _deep_unavailable_handler(_: Request, exc: DeepModelsUnavailable):
    return JSONResponse({"detail": str(exc)}, status_code=503)


@app.exception_handler(ValueError)
async def _value_error_handler(_: Request, exc: ValueError):
    return JSONResponse({"detail": str(exc)}, status_code=400)


@app.exception_handler(Exception)
async def _unhandled_handler(_: Request, exc: Exception):  # pragma: no cover
    log.exception("Unhandled error")
    return JSONResponse(
        {"detail": f"{type(exc).__name__}: {exc}"},
        status_code=500,
    )


# --- Helpers -----------------------------------------------------------------


async def read_upload(upload: UploadFile, field: str) -> np.ndarray:
    raw = await upload.read()
    if not raw:
        raise HTTPException(400, f"'{field}' is empty.")
    if len(raw) > config.MAX_UPLOAD_BYTES:
        limit = config.MAX_UPLOAD_BYTES // (1024 * 1024)
        raise HTTPException(413, f"'{field}' exceeds the {limit} MB upload limit.")
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, f"'{field}' is not a decodable image (JPG, PNG, WebP or BMP).")
    return image


def jpg_data_uri(image: np.ndarray, quality: int = 90) -> str:
    ok, buffer = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Could not encode the annotated image.")
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def _backend_available(key: str) -> bool:
    if key == "retinaface":
        return retinaface_available()
    return True


def filter_boxes(boxes, shape):
    """Clip, drop negligible detections, then suppress duplicates."""
    height, width = shape[:2]
    minimum_area = max(400, int(width * height * 0.00035))
    clean, scores = [], []
    for box in boxes:
        x, y, w, h = clamp_box(box, shape)
        if w * h >= minimum_area:
            clean.append((x, y, w, h))
            scores.append(float(w * h))
    return nms(clean, scores, threshold=0.40)


# --- API ---------------------------------------------------------------------


@app.get("/api/health")
def health():
    """Cheap liveness probe; never touches TensorFlow."""
    return {
        "status": "ok",
        "version": config.APP_VERSION,
        "gallery_people": list_gallery(),
    }


@app.get("/api/status")
def status():
    """Capability report the UI uses to label modes honestly."""
    return {
        "status": "ok",
        "version": config.APP_VERSION,
        "deep_models_enabled": config.DEEP_MODELS_ENABLED,
        "deep_models_ready": deep_models_available() if config.DEEP_MODELS_ENABLED else False,
        "deep_models_error": deep_models_error(),
        "recognition_model": RECOGNITION_MODEL,
        "warming_up": _warm_up_state["started"] and not _warm_up_state["done"],
        "gallery_people": list_gallery(),
        "methods": METHODS,
        "detector_backend": config.DETECTOR_BACKEND,
        "detector_backends": [
            {"key": key, **DETECTOR_PROFILE[key], "available": _backend_available(key)}
            for key in config.DETECTOR_BACKENDS
        ],
        "models": model_report(),
        "limits": {
            "max_upload_mb": config.MAX_UPLOAD_BYTES // (1024 * 1024),
            "max_image_dim": config.MAX_IMAGE_DIM,
        },
        "thresholds": {
            "recognition_similarity": config.RECOGNITION_THRESHOLD,
            "template_score": config.TEMPLATE_MIN_SCORE,
            "detection_confidence": config.DETECTION_MIN_CONFIDENCE,
        },
    }


@app.get("/api/gallery")
def gallery():
    people = list_gallery()
    return {"people": people, "count": len(people)}


def _run_analysis(
    frame: np.ndarray,
    method: str,
    recognition: bool,
    template_image: np.ndarray | None,
    filename: str | None,
    backend: str | None = None,
) -> dict:
    """All blocking CV/TensorFlow work; executed off the event loop."""
    started = time.perf_counter()
    original_h, original_w = frame.shape[:2]
    frame, scale = fit_within(frame, config.MAX_IMAGE_DIM)

    template_score: float | None = None
    fallback_used = False

    if method == "viola-jones":
        boxes = detect_viola_jones(frame)
        detector = "Viola-Jones / Haar cascade (strict)"
    elif method == "template":
        if template_image is None:
            raise HTTPException(400, "Template mode requires a reference template image.")
        boxes, template_score = template_match(frame, template_image)
        detector = f"Template matching (best {template_score:.3f})"
    else:
        requested = backend or config.DETECTOR_BACKEND
        boxes, used = detect_deep(frame, requested)
        fallback_used = used == "viola-jones" and requested != "viola-jones"
        detector = (
            f"Viola-Jones ({requested} unavailable)"
            if fallback_used
            else DETECTOR_LABELS.get(used, used)
        )

    boxes = filter_boxes(boxes, frame.shape)

    is_template = method == "template"
    label = "template match" if is_template else "face"
    people = list_gallery()
    run_recognition = recognition and not is_template

    items: list[dict[str, Any]] = []
    for index, (x, y, w, h) in enumerate(boxes, start=1):
        item: dict[str, Any] = {
            "id": index,
            "box": {"x": x, "y": y, "w": w, "h": h},
            "detection": label,
            "identity": "Recognition disabled",
            "match": None,
        }
        if is_template:
            item["identity"] = "Template match"
            item["match"] = {"score": round(float(template_score or 0.0), 4)}
        elif not run_recognition:
            item["identity"] = "Detected face"
        elif not people:
            item["identity"] = "No enrolled references"
        else:
            try:
                match = recognize_embedding(face_embedding(frame[y:y + h, x:x + w]))
                item["identity"] = match["identity"]
                item["match"] = match
            except DeepModelsUnavailable as exc:
                item["identity"] = "Recognition unavailable"
                item["match"] = {"error": str(exc)}
            except Exception as exc:
                log.warning("Recognition failed for box %s: %s", index, exc)
                item["identity"] = "Recognition failed"
                item["match"] = {"error": str(exc)}
        items.append(item)

    annotated = draw_detections(frame.copy(), items)

    return {
        "filename": filename,
        "method": method,
        "detector": detector,
        "detector_backend": "viola-jones" if method == "viola-jones" else (
            None if method == "template" else (backend or config.DETECTOR_BACKEND)
        ),
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "source_width": int(original_w),
        "source_height": int(original_h),
        "resized": scale < 1.0,
        "fallback_used": fallback_used,
        "recognition_ran": run_recognition and bool(people),
        "gallery_people": people,
        "count": len(items),
        "result_kind": "matches" if is_template else "faces",
        "result_label": label,
        "template_score": round(float(template_score), 4) if template_score is not None else None,
        "elapsed_ms": int((time.perf_counter() - started) * 1000),
        "faces": items,
        "image": jpg_data_uri(annotated),
    }


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    method: str = Form("robust"),
    recognition: bool = Form(True),
    detector: str = Form(""),
    template: UploadFile | None = File(None),
):
    method = (method or "robust").strip().lower()
    if method not in METHODS:
        raise HTTPException(400, f"Unknown method '{method}'. Expected one of {sorted(METHODS)}.")
    # FaceNet *is* the recognition mode, so the toggle cannot switch it off.
    if method == "facenet":
        recognition = True

    backend = (detector or "").strip().lower() or config.DETECTOR_BACKEND
    if backend not in config.DETECTOR_BACKENDS:
        raise HTTPException(
            400,
            f"Unknown detector '{backend}'. Expected one of {list(config.DETECTOR_BACKENDS)}.",
        )

    frame = await read_upload(image, "image")
    template_image = None
    if method == "template":
        if template is None:
            raise HTTPException(400, "Template mode requires a reference template image.")
        template_image = await read_upload(template, "template")

    payload = await run_in_threadpool(
        _run_analysis, frame, method, recognition, template_image, image.filename, backend
    )
    return JSONResponse(payload)


@app.post("/api/template-match")
async def template_match_api(
    image: UploadFile = File(...),
    template: UploadFile = File(...),
):
    frame = await read_upload(image, "image")
    template_image = await read_upload(template, "template")
    payload = await run_in_threadpool(
        _run_analysis, frame, "template", False, template_image, image.filename
    )
    return JSONResponse(payload)


@app.post("/api/warm-up")
async def warm_up_api():
    """Load models on demand; useful right after a cold start."""
    result = await run_in_threadpool(warm_up)
    _warm_up_state["done"] = True
    _warm_up_state["result"] = result
    return result


# --- Frontend ----------------------------------------------------------------


class SpaStaticFiles(StaticFiles):
    """Static files for the SPA.

    ``html=True`` already falls back to index.html for unknown paths, which is
    what a client-side router wants -- but it would also answer an unmatched
    /api/... request with a page of HTML. Those must stay a JSON 404 so API
    clients see a real error instead of parsing markup.
    """

    async def get_response(self, path: str, scope):
        # scope["path"] is the real URL; `path` is already OS-normalised, so on
        # Windows it arrives as "api\\thing" and a "api/" prefix test would miss.
        url_path = scope.get("path", "/")
        is_api = url_path == "/api" or url_path.startswith("/api/")
        if is_api:
            return JSONResponse({"detail": f"No API route for {url_path}"}, status_code=404)
        try:
            return await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            # Let genuine asset 404s stand so a missing image is visible, but
            # hand route-like paths to the SPA.
            if exc.status_code != 404 or "." in url_path.rsplit("/", 1)[-1]:
                raise
            return await super().get_response("index.html", scope)


_FAVICON = base64.b64decode(
    # 32x32 transparent PNG; avoids a 404 when no icon file is supplied.
    "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAAIGNIUk0AAHolAACAgwAA+f8AAIDp"
    "AAB1MAAA6mAAADqYAAAXb5JfxUYAAAAJcEhZcwAACxMAAAsTAQCanBgAAAAEZ0FNQQAAsY58+1GT"
    "AAAAIElEQVR42mNgYGD4z4AFMOJTMKqAwaFCRgaGUQWjCggqAAC8/wX7lKu0AAAAAElFTkSuQmCC"
)


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    if config.FRONTEND_DIST:
        for name in ("favicon.svg", "favicon.ico", "favicon.png"):
            candidate = config.FRONTEND_DIST / name
            if candidate.exists():
                return FileResponse(candidate)
    return Response(content=_FAVICON, media_type="image/png")


if config.FRONTEND_DIST:
    app.mount("/", SpaStaticFiles(directory=str(config.FRONTEND_DIST), html=True), name="frontend")
else:

    @app.get("/", include_in_schema=False)
    def root():
        return {
            "name": config.APP_NAME,
            "version": config.APP_VERSION,
            "mode": "API only - no frontend bundle found",
            "hint": "Run 'npm run build' in frontend/, or use the Vite dev server.",
            "docs": "/api/docs",
        }
