"""Download model weights at image-build time.

Without this the first request after a cold start pays for a download, which on
a free container reads as a hang. Run with PREFETCH_RETINAFACE=1 to also pull
the 118 MB RetinaFace network.
"""

from __future__ import annotations

import os
import sys

import numpy as np

# Keep TensorFlow quiet and the logger encodable on any console.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")


def main() -> int:
    from deepface import DeepFace

    probe = np.full((320, 320, 3), 127, dtype=np.uint8)

    backends = ["yunet", "ssd"]
    if os.getenv("PREFETCH_RETINAFACE", "0").lower() in {"1", "true", "yes", "on"}:
        backends.append("retinaface")

    failures: list[str] = []

    for backend in backends:
        try:
            DeepFace.extract_faces(
                img_path=probe,
                detector_backend=backend,
                enforce_detection=False,
                align=False,
            )
            print(f"[prefetch] detector '{backend}' ready")
        except Exception as exc:
            failures.append(f"{backend}: {exc}")
            print(f"[prefetch] detector '{backend}' FAILED: {exc}", file=sys.stderr)

    try:
        DeepFace.represent(
            img_path=np.full((160, 160, 3), 127, dtype=np.uint8),
            model_name="Facenet",
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
        print("[prefetch] FaceNet ready")
    except Exception as exc:
        failures.append(f"Facenet: {exc}")
        print(f"[prefetch] FaceNet FAILED: {exc}", file=sys.stderr)

    if failures:
        # Do not fail the build: the app degrades gracefully and can still
        # download a missing model on first use.
        print(f"[prefetch] {len(failures)} model(s) unavailable at build time", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
