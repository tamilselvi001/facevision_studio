"""Check the bundled ONNX models load and produce sane output.

Run from the repository root:  python backend/scripts/verify_models.py
Used by CI; also a quick way to confirm a deployment's models are intact.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.app import engines  # noqa: E402


def main() -> int:
    failures = []

    for name, path in (("FaceNet", engines.FACENET_PATH), ("YuNet", engines.YUNET_PATH)):
        if path.exists():
            print(f"[ok]   {name} present ({path.stat().st_size / 1e6:.1f} MB)")
        else:
            failures.append(f"{name} missing at {path}")
            print(f"[FAIL] {name} missing at {path}")

    try:
        embedding = engines.face_embedding(np.full((160, 160, 3), 127, dtype=np.uint8))
        if embedding.shape != (128,):
            failures.append(f"FaceNet returned shape {embedding.shape}, expected (128,)")
        print(f"[ok]   FaceNet embedding shape {embedding.shape}, norm {np.linalg.norm(embedding):.3f}")
    except Exception as exc:
        failures.append(f"FaceNet inference failed: {exc}")
        print(f"[FAIL] FaceNet inference: {exc}")

    try:
        engines.detect_yunet(np.full((240, 320, 3), 127, dtype=np.uint8))
        print("[ok]   YuNet inference")
    except Exception as exc:
        failures.append(f"YuNet inference failed: {exc}")
        print(f"[FAIL] YuNet inference: {exc}")

    print(f"\n{'FAILED: ' + '; '.join(failures) if failures else 'All model checks passed.'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
