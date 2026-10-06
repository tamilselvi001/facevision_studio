"""End-to-end smoke test: boots the ASGI app in-process and exercises the API.

Run from the repository root:  python backend/scripts/smoke_test.py

Classical modes are asserted strictly. Deep modes are reported but do not fail
the run, so the suite stays green on a machine or CI job without the weights.
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
os.environ.setdefault("WARM_UP_ON_STARTUP", "0")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from backend.app.main import app  # noqa: E402

SAMPLE = ROOT / "frontend" / "public" / "sample" / "group_faces.jpg"
TEMPLATE = ROOT / "frontend" / "public" / "sample" / "sundhar_face_template.jpg"

passed: list[str] = []
failed: list[str] = []
skipped: list[str] = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        passed.append(name)
        print(f"  PASS  {name}")
    else:
        failed.append(name)
        print(f"  FAIL  {name} {detail}")


def image_part(path: Path, field: str = "image"):
    return {field: (path.name, io.BytesIO(path.read_bytes()), "image/jpeg")}


def main() -> int:
    if not SAMPLE.exists():
        print(f"Sample image missing: {SAMPLE}")
        return 1

    with TestClient(app) as client:
        print("\n[health]")
        response = client.get("/api/health")
        check("GET /api/health is 200", response.status_code == 200, str(response.status_code))
        check("health reports ok", response.json().get("status") == "ok")

        print("\n[status]")
        response = client.get("/api/status")
        status = response.json()
        check("GET /api/status is 200", response.status_code == 200)
        check("status lists detector backends", bool(status.get("detector_backends")))
        check("status lists methods", len(status.get("methods", {})) == 4)
        deep_ready = bool(status.get("deep_models_ready"))
        print(f"  INFO  deep models ready: {deep_ready}")
        print(f"  INFO  gallery: {status.get('gallery_people')}")

        print("\n[routing]")
        check("unknown /api/ path is 404", client.get("/api/zzz").status_code == 404)
        check(
            "unknown /api/ path returns JSON",
            client.get("/api/zzz").headers.get("content-type", "").startswith("application/json"),
        )

        print("\n[viola-jones]")
        response = client.post(
            "/api/analyze",
            files=image_part(SAMPLE),
            data={"method": "viola-jones", "recognition": "false"},
        )
        check("viola-jones is 200", response.status_code == 200, response.text[:160])
        if response.status_code == 200:
            payload = response.json()
            check("viola-jones finds 2 faces", payload["count"] == 2, f"got {payload['count']}")
            check("viola-jones returns an image", payload["image"].startswith("data:image/jpeg"))
            check("boxes are inside the frame", all(
                box["box"]["x"] + box["box"]["w"] <= payload["width"]
                and box["box"]["y"] + box["box"]["h"] <= payload["height"]
                for box in payload["faces"]
            ))

        print("\n[template]")
        files = image_part(SAMPLE)
        files.update(image_part(TEMPLATE, "template"))
        response = client.post("/api/analyze", files=files, data={"method": "template"})
        check("template is 200", response.status_code == 200, response.text[:160])
        if response.status_code == 200:
            payload = response.json()
            check("template finds 1 match", payload["count"] == 1, f"got {payload['count']}")
            check(
                "template score is high",
                (payload["template_score"] or 0) > 0.85,
                f"got {payload['template_score']}",
            )

        print("\n[validation]")
        check(
            "unknown method is 400",
            client.post("/api/analyze", files=image_part(SAMPLE), data={"method": "bogus"}).status_code == 400,
        )
        check(
            "unknown detector is 400",
            client.post(
                "/api/analyze", files=image_part(SAMPLE), data={"detector": "bogus"}
            ).status_code == 400,
        )
        check(
            "template mode without a template is 400",
            client.post("/api/analyze", files=image_part(SAMPLE), data={"method": "template"}).status_code == 400,
        )
        check(
            "non-image upload is 400",
            client.post(
                "/api/analyze",
                files={"image": ("x.txt", io.BytesIO(b"not an image"), "text/plain")},
                data={"method": "viola-jones"},
            ).status_code == 400,
        )

        print("\n[deep models]")
        if deep_ready:
            response = client.post(
                "/api/analyze",
                files=image_part(SAMPLE),
                data={"method": "facenet", "detector": "yunet"},
            )
            check("facenet/yunet is 200", response.status_code == 200, response.text[:160])
            if response.status_code == 200:
                payload = response.json()
                check("yunet finds 2 faces", payload["count"] == 2, f"got {payload['count']}")
                identities = [face["identity"] for face in payload["faces"]]
                print(f"  INFO  identities: {identities}")
                if payload.get("gallery_people"):
                    check(
                        "a gallery identity was recognised",
                        any(name in identities for name in payload["gallery_people"]),
                        f"got {identities}",
                    )
        else:
            skipped.append("deep model checks")
            print("  SKIP  deep models unavailable in this environment")

    print(f"\n{len(passed)} passed, {len(failed)} failed, {len(skipped)} skipped")
    if failed:
        print("Failed: " + ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
