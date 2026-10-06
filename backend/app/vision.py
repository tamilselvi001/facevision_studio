"""Detection, recognition and template-matching primitives.

Deep models are imported lazily and guarded by a lock: TensorFlow model
construction is not safe to run concurrently, and importing it eagerly would
delay startup past most platforms' health-check window.
"""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

import cv2
import numpy as np

from . import config

log = logging.getLogger("facevision.vision")

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class DeepModelsUnavailable(RuntimeError):
    """Raised when the TensorFlow stack cannot be used for this request."""


# --- Lazy, serialized access to DeepFace -------------------------------------

_deepface_lock = threading.Lock()
_deepface_module = None
_deepface_error: str | None = None


def _deepface():
    """Import DeepFace once, caching both success and failure."""
    global _deepface_module, _deepface_error

    if _deepface_module is not None:
        return _deepface_module
    if _deepface_error is not None:
        raise DeepModelsUnavailable(_deepface_error)
    if not config.DEEP_MODELS_ENABLED:
        _deepface_error = "Deep models are disabled (DEEP_MODELS_ENABLED=0)."
        raise DeepModelsUnavailable(_deepface_error)

    with _deepface_lock:
        if _deepface_module is not None:
            return _deepface_module
        if _deepface_error is not None:
            raise DeepModelsUnavailable(_deepface_error)
        try:
            log.info("Importing DeepFace / TensorFlow (first use)...")
            from deepface import DeepFace  # noqa: PLC0415 - intentionally deferred

            _deepface_module = DeepFace
            log.info("DeepFace ready.")
            return _deepface_module
        except Exception as exc:  # pragma: no cover - environment dependent
            _deepface_error = f"DeepFace/TensorFlow could not be loaded: {exc}"
            log.warning(_deepface_error)
            raise DeepModelsUnavailable(_deepface_error) from exc


def deep_models_available() -> bool:
    """Whether deep detection/recognition can serve a request right now."""
    if not config.DEEP_MODELS_ENABLED:
        return False
    try:
        _deepface()
        return True
    except DeepModelsUnavailable:
        return False


def deep_models_error() -> str | None:
    return _deepface_error


# TensorFlow graph building and the Keras model registry are not thread-safe,
# and parallel inference would multiply peak memory on a small container.
_inference_lock = threading.Lock()


# --- Haar cascades -----------------------------------------------------------

_cascade_cache: dict[str, cv2.CascadeClassifier] = {}


def _cascade(filename: str) -> cv2.CascadeClassifier:
    cached = _cascade_cache.get(filename)
    if cached is not None:
        return cached
    path = cv2.data.haarcascades + filename
    classifier = cv2.CascadeClassifier(path)
    if classifier.empty():
        raise RuntimeError(f"OpenCV cascade file could not be loaded: {path}")
    _cascade_cache[filename] = classifier
    return classifier


# --- Geometry helpers --------------------------------------------------------


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def nms(boxes, scores, threshold: float = 0.30):
    """Greedy non-maximum suppression; keeps the highest-scoring overlaps."""
    if not boxes:
        return []
    order = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
    kept: list[tuple[int, int, int, int]] = []
    for idx in order:
        box = tuple(boxes[int(idx)])
        if all(_iou(box, other) < threshold for other in kept):
            kept.append(box)  # type: ignore[arg-type]
    return kept


def clamp_box(box, shape):
    """Clip a box to the image and guarantee a positive width/height."""
    height, width = shape[:2]
    x, y, w, h = (int(round(v)) for v in box)
    x = max(0, min(x, width - 1))
    y = max(0, min(y, height - 1))
    w = max(1, min(w, width - x))
    h = max(1, min(h, height - y))
    return x, y, w, h


def fit_within(image: np.ndarray, max_dim: int) -> tuple[np.ndarray, float]:
    """Downscale so the longest side is <= max_dim. Returns (image, scale)."""
    height, width = image.shape[:2]
    longest = max(height, width)
    if longest <= max_dim:
        return image, 1.0
    scale = max_dim / float(longest)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


# --- Viola-Jones -------------------------------------------------------------


def detect_viola_jones(image: np.ndarray):
    """Strict Haar-cascade detector with multi-pass consensus.

    Haar cascades are fast but prone to texture false positives. The frontal
    cascade runs on two grayscale variants and candidates must be geometrically
    plausible *and* contain eye structure, which is what keeps shirts, flowers
    and posters from being reported as faces. This favours precision over
    recall by design; use RetinaFace for difficult poses.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    face_cascade = _cascade("haarcascade_frontalface_default.xml")
    eye_cascade = _cascade("haarcascade_eye.xml")

    height, width = gray.shape[:2]
    min_side = max(36, int(min(height, width) * 0.055))
    params = {
        "scaleFactor": 1.08,
        "minNeighbors": 8,
        "minSize": (min_side, min_side),
    }

    pass_a = [tuple(map(int, b)) for b in face_cascade.detectMultiScale(equalized, **params)]
    pass_b = [tuple(map(int, b)) for b in face_cascade.detectMultiScale(enhanced, **params)]

    eye_cache: dict[tuple[int, int, int, int], bool] = {}

    def has_eye_structure(box) -> bool:
        cached = eye_cache.get(box)
        if cached is not None:
            return cached
        x, y, w, h = box
        roi = equalized[y:y + max(1, int(h * 0.68)), x:x + w]
        found = False
        if roi.size:
            eyes = eye_cascade.detectMultiScale(
                roi,
                scaleFactor=1.08,
                minNeighbors=4,
                minSize=(max(8, w // 10), max(8, h // 12)),
            )
            found = len(eyes) >= 1
        eye_cache[box] = found
        return found

    candidates: list[tuple[int, int, int, int]] = []
    scores: list[float] = []
    for box in pass_a + pass_b:
        x, y, w, h = box
        aspect = w / max(h, 1)
        area_ratio = (w * h) / float(width * height)
        if not (0.62 <= aspect <= 1.45) or area_ratio < 0.0025:
            continue
        if not has_eye_structure(box):
            continue
        # Agreement between both grayscale variants is strong evidence.
        both = any(_iou(box, other) >= 0.25 for other in pass_b) and any(
            _iou(box, other) >= 0.25 for other in pass_a
        )
        candidates.append(box)
        scores.append((2.0 if both else 1.0) * 1e6 + w * h)

    return nms(candidates, scores, threshold=0.30)


# --- RetinaFace --------------------------------------------------------------


def detect_deep(image: np.ndarray, backend: str | None = None):
    """Deep face detection via DeepFace.

    ``backend`` is one of config.DETECTOR_BACKENDS. Returns
    ``(boxes, backend_actually_used)``; on any failure it degrades to the
    Viola-Jones cascade rather than erroring the whole request.
    """
    backend = (backend or config.DETECTOR_BACKEND).lower()
    if backend not in config.DETECTOR_BACKENDS:
        backend = config.DETECTOR_BACKEND

    try:
        deepface = _deepface()
    except DeepModelsUnavailable:
        return detect_viola_jones(image), "viola-jones"

    try:
        with _inference_lock:
            faces = deepface.extract_faces(
                img_path=image,
                detector_backend=backend,
                enforce_detection=False,
                align=True,
            )
    except Exception as exc:
        log.warning("%s failed, using Viola-Jones: %s", backend, exc)
        return detect_viola_jones(image), "viola-jones"

    boxes = []
    for item in faces:
        area = item.get("facial_area") or {}
        w = int(area.get("w", 0))
        h = int(area.get("h", 0))
        confidence = float(item.get("confidence", 0.0))
        if w <= 0 or h <= 0 or confidence < config.DETECTION_MIN_CONFIDENCE:
            continue
        # DeepFace reports the whole frame when the detector finds nothing.
        if w >= image.shape[1] * 0.99 and h >= image.shape[0] * 0.99:
            continue
        boxes.append(clamp_box((area.get("x", 0), area.get("y", 0), w, h), image.shape))

    return boxes, backend


# --- FaceNet embeddings ------------------------------------------------------


def face_embedding(face: np.ndarray) -> np.ndarray:
    deepface = _deepface()
    if face.size == 0:
        raise ValueError("Empty face crop.")
    with _inference_lock:
        reps = deepface.represent(
            img_path=face,
            model_name="Facenet",
            detector_backend="skip",
            enforce_detection=False,
            align=False,
            normalization="base",
        )
    if not reps:
        raise RuntimeError("FaceNet returned no embedding.")
    return np.asarray(reps[0]["embedding"], dtype=np.float32)


def reference_face(image: np.ndarray) -> np.ndarray:
    """Detect and crop the most prominent face from a gallery reference image."""
    boxes, _ = detect_deep(image)
    if not boxes:
        boxes = detect_viola_jones(image)
    if not boxes:
        raise RuntimeError("No usable face found in gallery reference.")
    x, y, w, h = max(boxes, key=lambda b: b[2] * b[3])
    return image[y:y + h, x:x + w].copy()


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0.0:
        return -1.0
    return float(np.dot(a, b) / denominator)


# --- Gallery -----------------------------------------------------------------

CACHE_VERSION = 4
_gallery_lock = threading.Lock()


def canonical_name(folder_name: str) -> str:
    return config.NAME_ALIASES.get(folder_name, folder_name)


def list_gallery(gallery: Path | None = None) -> list[str]:
    gallery = gallery or config.GALLERY_DIR
    if not gallery.exists():
        return []
    return sorted(
        {
            canonical_name(p.name)
            for p in gallery.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        }
    )


def gallery_embeddings(gallery: Path | None = None) -> dict[str, dict]:
    """Build (and incrementally cache) FaceNet embeddings for every reference.

    The cache is keyed on file size + mtime, so replacing an image invalidates
    only that entry. A read-only filesystem is tolerated: the work is simply
    redone next start.
    """
    gallery = gallery or config.GALLERY_DIR
    gallery.mkdir(parents=True, exist_ok=True)
    cache_path = gallery / ".embeddings.json"

    with _gallery_lock:
        data = {"version": CACHE_VERSION, "people": {}}
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("version") == CACHE_VERSION and isinstance(cached.get("people"), dict):
                    data = cached
            except Exception:
                log.warning("Ignoring unreadable embedding cache at %s", cache_path)

        people: dict[str, dict] = data["people"]
        changed = False
        folders = {
            p.name: p for p in gallery.iterdir() if p.is_dir() and not p.name.startswith(".")
        }

        for stale in [name for name in people if name not in folders]:
            del people[stale]
            changed = True

        for folder_name, folder in sorted(folders.items()):
            refs: dict = people.setdefault(folder_name, {})
            present: set[str] = set()
            for image_path in sorted(folder.iterdir()):
                if image_path.suffix.lower() not in SUPPORTED_SUFFIXES:
                    continue
                present.add(image_path.name)
                stat = image_path.stat()
                signature = f"{stat.st_size}:{stat.st_mtime_ns}"
                existing = refs.get(image_path.name)
                if existing and existing.get("signature") == signature and "embedding" in existing:
                    continue
                try:
                    bgr = cv2.imread(str(image_path))
                    if bgr is None:
                        log.warning("Unreadable gallery image: %s", image_path)
                        continue
                    bgr, _ = fit_within(bgr, config.MAX_IMAGE_DIM)
                    embedding = face_embedding(reference_face(bgr))
                    refs[image_path.name] = {
                        "signature": signature,
                        "embedding": embedding.tolist(),
                    }
                    changed = True
                    log.info("Enrolled %s/%s", folder_name, image_path.name)
                except DeepModelsUnavailable:
                    raise
                except Exception as exc:
                    log.warning("Skipping %s: %s", image_path, exc)

            for removed in [name for name in refs if name not in present]:
                del refs[removed]
                changed = True

        if changed:
            try:
                cache_path.write_text(json.dumps(data), encoding="utf-8")
            except OSError as exc:
                log.warning("Could not persist embedding cache: %s", exc)

        return {name: dict(refs) for name, refs in people.items()}


def recognize_embedding(embedding: np.ndarray, gallery: Path | None = None) -> dict:
    data = gallery_embeddings(gallery)
    best_name, best_score, best_ref = "Unknown", -1.0, None
    for folder_name, refs in data.items():
        for filename, ref in refs.items():
            score = _cosine(embedding, np.asarray(ref["embedding"], dtype=np.float32))
            if score > best_score:
                best_name, best_score, best_ref = canonical_name(folder_name), score, filename

    threshold = config.RECOGNITION_THRESHOLD
    matched = best_score >= threshold
    return {
        "identity": best_name if matched else "Unknown",
        "closest": best_name,
        "similarity": round(float(max(best_score, 0.0)), 4),
        "distance": round(float(1.0 - best_score), 4),
        "threshold": threshold,
        "matched": matched,
        "reference": best_ref,
    }


def warm_up() -> dict:
    """Load models and pre-compute gallery embeddings. Safe to call twice."""
    status = {"deep_models": False, "people": [], "error": None}
    try:
        _deepface()
        status["deep_models"] = True
        # A tiny synthetic face exercises both model loads.
        probe = np.full((160, 160, 3), 127, dtype=np.uint8)
        face_embedding(probe)
        gallery_embeddings()
    except DeepModelsUnavailable as exc:
        status["error"] = str(exc)
    except Exception as exc:  # pragma: no cover
        status["error"] = f"Warm-up incomplete: {exc}"
    status["people"] = list_gallery()
    return status


# --- Template matching -------------------------------------------------------


def _prepare_gray(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    return cv2.GaussianBlur(gray, (3, 3), 0)


def _edge_map(gray: np.ndarray) -> np.ndarray:
    return cv2.Canny(gray, 50, 150)


def template_match(image: np.ndarray, template: np.ndarray) -> tuple[list, float]:
    """Scale-tolerant template matching for a reference patch.

    Plain ``cv2.matchTemplate`` assumes the patch has the same pixel scale as
    the target, so a crop that was resized before upload can legitimately score
    ~0.48 and be thrown away. This searches a bounded scale pyramid and
    combines grayscale correlation (identity) with edge correlation (which
    rejects flat or uniformly textured regions).

    Returns ``([box], score)`` or ``([], best_score)`` when nothing clears the
    confidence floor -- a weak match is reported as "no match" rather than
    drawn as an arbitrary box.
    """
    target_gray = _prepare_gray(image)
    template_gray = _prepare_gray(template)

    template_h, template_w = template_gray.shape[:2]
    if template_h < 18 or template_w < 18:
        raise ValueError(
            "Template is too small. Use a crop containing the face and some surrounding detail."
        )

    # Bound work for large targets; keep the mapping back to original pixels.
    target_gray, target_scale = fit_within(target_gray, 1400)
    height, width = target_gray.shape[:2]

    max_template = 420
    if max(template_h, template_w) > max_template:
        template_gray, _ = fit_within(template_gray, max_template)
    template_h, template_w = template_gray.shape[:2]

    if template_h >= height or template_w >= width:
        raise ValueError("Template must be smaller than the target image.")

    target_edges = _edge_map(target_gray)
    best_score, best_box = -1.0, None

    # Covers the usual screenshot/browser resizing range without the cost of a
    # dense pyramid.
    scales = (0.55, 0.65, 0.75, 0.85, 0.95, 1.00, 1.08, 1.18, 1.30, 1.45, 1.60)
    for scale in scales:
        scaled_w = max(12, int(template_w * scale))
        scaled_h = max(12, int(template_h * scale))
        if scaled_w >= width or scaled_h >= height:
            continue
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
        patch = cv2.resize(template_gray, (scaled_w, scaled_h), interpolation=interpolation)

        _, gray_score, _, location = cv2.minMaxLoc(
            cv2.matchTemplate(target_gray, patch, cv2.TM_CCOEFF_NORMED)
        )
        _, edge_score, _, _ = cv2.minMaxLoc(
            cv2.matchTemplate(target_edges, _edge_map(patch), cv2.TM_CCOEFF_NORMED)
        )

        score = 0.72 * float(gray_score) + 0.28 * float(edge_score)
        if score > best_score:
            best_score = score
            best_box = (int(location[0]), int(location[1]), scaled_w, scaled_h)

    if best_box is None or best_score < config.TEMPLATE_MIN_SCORE:
        return [], max(0.0, float(best_score))

    inverse = 1.0 / target_scale
    box = clamp_box([v * inverse for v in best_box], image.shape)
    return [box], float(best_score)


# --- Annotation --------------------------------------------------------------

_BOX_COLOR = (160, 220, 70)  # BGR
_LABEL_BG = (28, 20, 15)
_LABEL_FG = (242, 245, 235)


def draw_detections(image: np.ndarray, results) -> np.ndarray:
    """Draw boxes and labels, scaled to the image and clamped inside it."""
    height, width = image.shape[:2]
    # Keep strokes and text legible on both thumbnails and large photographs.
    font_scale = max(0.42, min(0.95, min(height, width) / 900.0))
    thickness = max(1, int(round(min(height, width) / 450.0)))
    font = cv2.FONT_HERSHEY_SIMPLEX

    for item in results:
        box = item["box"]
        x, y, w, h = box["x"], box["y"], box["w"], box["h"]
        cv2.rectangle(image, (x, y), (x + w, y + h), _BOX_COLOR, thickness + 1)

        label = _label_for(item)
        if not label:
            continue

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        pad = max(4, int(text_h * 0.35))
        band_h = text_h + baseline + pad * 2
        band_w = min(width, text_w + pad * 2)

        # Prefer above the box; drop below when there is no room.
        top = y - band_h
        if top < 0:
            top = min(y + h, height - band_h)
        top = max(0, top)
        left = max(0, min(x, width - band_w))

        cv2.rectangle(image, (left, top), (left + band_w, top + band_h), _LABEL_BG, -1)
        cv2.putText(
            image,
            label,
            (left + pad, top + text_h + pad),
            font,
            font_scale,
            _LABEL_FG,
            thickness,
            cv2.LINE_AA,
        )
    return image


def _label_for(item) -> str:
    identity = item.get("identity") or "Face"
    match = item.get("match") or {}
    score = match.get("score")
    if score is not None:
        return f"{identity} {float(score) * 100:.0f}%"
    similarity = match.get("similarity")
    if similarity is not None:
        return f"{identity} {float(similarity) * 100:.0f}%"
    return str(identity)
