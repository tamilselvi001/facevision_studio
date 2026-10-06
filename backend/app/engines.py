"""Model runtimes.

Everything here runs on ONNX Runtime and OpenCV. TensorFlow is never imported,
which is what keeps the service inside a 512 MB free-tier container.

The FaceNet graph is the exact model DeepFace ships, converted to ONNX and
stored at float16. Embeddings were verified against the TensorFlow original on
every bundled reference image: cosine agreement 0.9999977 or better, i.e.
identical to floating-point noise. The preprocessing below is a faithful port
of DeepFace's own, which matters more than it looks -- it feeds the network
**BGR** pixels (DeepFace flips to RGB and back) and pads to preserve aspect
ratio rather than stretching. Getting either wrong drops agreement to ~0.6.
"""

from __future__ import annotations

import logging
import threading

import cv2
import numpy as np

from . import config

log = logging.getLogger("facevision.engines")

MODELS_DIR = config.BACKEND_ROOT / "models"
FACENET_PATH = MODELS_DIR / "facenet.onnx"
YUNET_PATH = MODELS_DIR / "yunet.onnx"

FACENET_INPUT = 160


class ModelUnavailable(RuntimeError):
    """A required model file or runtime is missing."""


_lock = threading.Lock()
_facenet = None
_facenet_input = None
_facenet_error: str | None = None


def _onnxruntime():
    try:
        import onnxruntime as ort  # noqa: PLC0415 - deferred so startup stays fast

        return ort
    except Exception as exc:  # pragma: no cover
        raise ModelUnavailable(f"onnxruntime is not installed: {exc}") from exc


def facenet_session():
    """Load the FaceNet ONNX graph once. Thread-safe, caches failures."""
    global _facenet, _facenet_input, _facenet_error

    if _facenet is not None:
        return _facenet
    if _facenet_error is not None:
        raise ModelUnavailable(_facenet_error)

    with _lock:
        if _facenet is not None:
            return _facenet
        if _facenet_error is not None:
            raise ModelUnavailable(_facenet_error)
        try:
            if not FACENET_PATH.exists():
                raise ModelUnavailable(f"FaceNet model missing at {FACENET_PATH}")
            ort = _onnxruntime()
            options = ort.SessionOptions()
            # One thread per session keeps memory predictable on a shared vCPU.
            options.intra_op_num_threads = config.ONNX_THREADS
            options.inter_op_num_threads = 1
            options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            log.info("Loading FaceNet (ONNX) from %s", FACENET_PATH)
            _facenet = ort.InferenceSession(
                str(FACENET_PATH), options, providers=["CPUExecutionProvider"]
            )
            _facenet_input = _facenet.get_inputs()[0].name
            log.info("FaceNet ready.")
            return _facenet
        except Exception as exc:
            _facenet_error = f"FaceNet could not be loaded: {exc}"
            log.warning(_facenet_error)
            raise ModelUnavailable(_facenet_error) from exc


def facenet_available() -> bool:
    try:
        facenet_session()
        return True
    except ModelUnavailable:
        return False


def facenet_error() -> str | None:
    return _facenet_error


def preprocess_face(face: np.ndarray, size: int = FACENET_INPUT) -> np.ndarray:
    """Port of DeepFace's resize_image: BGR, aspect-preserving, zero-padded, /255."""
    if face is None or face.size == 0:
        raise ValueError("Empty face crop.")
    if face.ndim != 3:
        raise ValueError(f"Expected a 3-channel image, got shape {face.shape}.")

    factor = min(size / face.shape[0], size / face.shape[1])
    target = (max(1, int(face.shape[1] * factor)), max(1, int(face.shape[0] * factor)))
    resized = cv2.resize(face, target)

    pad_h = size - resized.shape[0]
    pad_w = size - resized.shape[1]
    padded = np.pad(
        resized,
        ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w - pad_w // 2), (0, 0)),
        "constant",
    )
    if padded.shape[:2] != (size, size):
        padded = cv2.resize(padded, (size, size))

    batch = padded.astype(np.float32)
    if batch.max() > 1:
        batch = batch / 255.0
    return batch[None, ...]


def face_embedding(face: np.ndarray) -> np.ndarray:
    """128-dimensional FaceNet embedding for a face crop."""
    session = facenet_session()
    output = session.run(None, {_facenet_input: preprocess_face(face)})
    return np.asarray(output[0][0], dtype=np.float32)


# --- YuNet detection ---------------------------------------------------------

_yunet = None
_yunet_error: str | None = None
_yunet_lock = threading.Lock()


def yunet_detector():
    """OpenCV's YuNet detector: 228 KB, no TensorFlow, milliseconds per image."""
    global _yunet, _yunet_error

    if _yunet is not None:
        return _yunet
    if _yunet_error is not None:
        raise ModelUnavailable(_yunet_error)

    with _yunet_lock:
        if _yunet is not None:
            return _yunet
        try:
            if not YUNET_PATH.exists():
                raise ModelUnavailable(f"YuNet model missing at {YUNET_PATH}")
            _yunet = cv2.FaceDetectorYN.create(
                str(YUNET_PATH),
                "",
                (320, 320),
                config.DETECTION_MIN_CONFIDENCE,
                0.3,
                5000,
            )
            log.info("YuNet ready.")
            return _yunet
        except Exception as exc:
            _yunet_error = f"YuNet could not be loaded: {exc}"
            log.warning(_yunet_error)
            raise ModelUnavailable(_yunet_error) from exc


def yunet_available() -> bool:
    try:
        yunet_detector()
        return True
    except ModelUnavailable:
        return False


def detect_yunet(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Detect faces. Returns (x, y, w, h) boxes clipped to the image."""
    detector = yunet_detector()
    height, width = image.shape[:2]
    with _yunet_lock:
        # setInputSize mutates detector state, so one caller at a time.
        detector.setInputSize((width, height))
        _, faces = detector.detect(image)

    if faces is None:
        return []

    boxes = []
    for row in faces:
        x, y, w, h = (int(round(v)) for v in row[:4])
        if w <= 0 or h <= 0:
            continue
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        boxes.append((x, y, max(1, min(w, width - x)), max(1, min(h, height - y))))
    return boxes


# --- Optional TensorFlow backend --------------------------------------------
# RetinaFace has the best recall on hard poses but needs TensorFlow (~1 GB) and
# takes ~14.5 s per image on CPU, so it is unusable on a free tier. It stays
# available for local work when the optional extras are installed.

_deepface = None
_deepface_error: str | None = None


def deepface_module():
    global _deepface, _deepface_error
    if _deepface is not None:
        return _deepface
    if _deepface_error is not None:
        raise ModelUnavailable(_deepface_error)
    if not config.ENABLE_RETINAFACE:
        _deepface_error = "RetinaFace is disabled (ENABLE_RETINAFACE=0)."
        raise ModelUnavailable(_deepface_error)
    with _lock:
        try:
            from deepface import DeepFace  # noqa: PLC0415

            _deepface = DeepFace
            return _deepface
        except Exception as exc:
            _deepface_error = (
                "RetinaFace needs the optional TensorFlow extras "
                f"(pip install -r backend/requirements-retinaface.txt): {exc}"
            )
            raise ModelUnavailable(_deepface_error) from exc


def retinaface_available() -> bool:
    if not config.ENABLE_RETINAFACE:
        return False
    try:
        deepface_module()
        return True
    except ModelUnavailable:
        return False


def detect_retinaface(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    deepface = deepface_module()
    height, width = image.shape[:2]
    with _lock:
        faces = deepface.extract_faces(
            img_path=image,
            detector_backend="retinaface",
            enforce_detection=False,
            align=True,
        )
    boxes = []
    for item in faces:
        area = item.get("facial_area") or {}
        w, h = int(area.get("w", 0)), int(area.get("h", 0))
        if w <= 0 or h <= 0 or float(item.get("confidence", 0.0)) < config.DETECTION_MIN_CONFIDENCE:
            continue
        if w >= width * 0.99 and h >= height * 0.99:
            continue
        x = max(0, min(int(area.get("x", 0)), width - 1))
        y = max(0, min(int(area.get("y", 0)), height - 1))
        boxes.append((x, y, max(1, min(w, width - x)), max(1, min(h, height - y))))
    return boxes


def model_report() -> dict:
    """What this instance can actually do right now."""
    return {
        "facenet": facenet_available(),
        "facenet_error": facenet_error(),
        "yunet": yunet_available(),
        "retinaface": retinaface_available(),
        "models_dir": str(MODELS_DIR),
    }
