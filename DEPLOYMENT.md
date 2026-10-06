# Deployment guide

The repository builds into **one container** serving both the FastAPI API and
the compiled React frontend on a single origin. No CORS setup, no second
service, no separate frontend host.

---

## Why Render, and why not Hugging Face

The deciding constraint used to be memory: TensorFlow plus the FaceNet weights
need roughly **1 GB resident**, and free tiers give 512 MB.

That constraint is gone. FaceNet now runs on **ONNX Runtime** instead of
TensorFlow — the same model graph, exported to ONNX at float16 — so the whole
service peaks around **265 MB**. It fits a free 512 MB instance with headroom.

| Host | Free RAM | Card required | All features? |
|---|---|---|---|
| **Render** | 512 MB | **No** | **Yes** ✅ |
| Hugging Face Spaces | 16 GB | — | Docker/Gradio are **PRO-only since 2026** ❌ |
| Koyeb | — | Yes, since Feb 2026 | — |
| Fly.io / Railway / Cloud Run | varies | Yes | — |

**Use Render.** It is free, needs no credit card, and runs every feature.

---

## Deploy to Render

### 1. Push to GitHub

```bash
git remote add origin https://github.com/<you>/facevision_studio.git
git push -u origin main
```

On the password prompt, paste a **Personal Access Token** from
<https://github.com/settings/tokens> (scope: `repo`), not your account password.

### 2. Create the service

1. Sign up at <https://render.com> (GitHub login works; no card needed).
2. **New** → **Blueprint**.
3. Connect your GitHub account and pick the `facevision_studio` repo.
4. Render reads [`render.yaml`](render.yaml) and configures everything itself.
5. Click **Apply**.

> Prefer doing it manually? **New → Web Service** → pick the repo → set
> **Runtime: Docker**, **Plan: Free**, **Health check path: `/api/health`**.

### 3. Wait for the build

First build takes roughly **4–7 minutes**. There is no TensorFlow to download
and the models are committed to the repo, so nothing is fetched at runtime.

When the status turns **Live**, your app is at:

```
https://facevision-studio.onrender.com
```

(Render appends a suffix if the name is taken.)

### 4. Updating

```bash
git add -A && git commit -m "your change" && git push
```

`autoDeploy: true` in `render.yaml` means Render rebuilds on every push to
`main`.

---

## Free-tier behaviour you should expect

- **Sleeps after 15 minutes** with no traffic, and takes **~50 seconds** to wake
  on the next request. The first visitor after a quiet period waits; everyone
  after that does not.
- **750 instance-hours per month**, enough for one service running continuously
  (a month is ~744 hours).
- Shared, fractional CPU. `ONNX_THREADS=1` is set deliberately — more threads on
  a fractional core makes things slower, not faster.

If the wake delay matters, Render's paid Starter plan removes it.

---

## Any other Docker host

```bash
docker build -t facevision-studio .
docker run -p 10000:10000 facevision-studio
```

Open <http://localhost:10000>.

---

## Environment variables

Every setting has a working default; see [`.env.example`](.env.example).

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `10000` | Bind port |
| `ALLOWED_ORIGINS` | `*` | Only needed for a split-origin deployment |
| `DETECTOR_BACKEND` | `yunet` | `yunet` or `retinaface` |
| `ENABLE_RETINAFACE` | `0` | `1` enables the optional TensorFlow backend |
| `DEEP_MODELS_ENABLED` | `1` | `0` runs classical CV only, loads no weights |
| `WARM_UP_ON_STARTUP` | `1` | Load models in the background at boot |
| `ONNX_THREADS` | `1` | Raise on a host with real dedicated cores |
| `RECOGNITION_THRESHOLD` | `0.60` | Cosine similarity needed to claim an identity |
| `TEMPLATE_MIN_SCORE` | `0.60` | Correlation floor for a template match |
| `MAX_IMAGE_DIM` | `1600` | Inputs are fitted inside this box first |
| `MAX_UPLOAD_MB` | `12` | Rejected above this size |
| `GALLERY_DIR` | `backend/gallery` | Point at a mounted disk to persist enrolments |

---

## Models

Both model files are committed, so builds are reproducible and cold starts
never wait on a download.

| File | Size | Purpose |
|---|---|---|
| `backend/models/facenet.onnx` | ~44 MB | FaceNet 128-d embeddings, float16 |
| `backend/models/yunet.onnx` | ~228 KB | Face detection |

`facenet.onnx` is DeepFace's own FaceNet graph converted with `tf2onnx` and
quantised to float16. Embeddings were verified against the TensorFlow original
on every bundled reference image — cosine agreement **0.9999977 or better**.

Verify them on any machine or deployment:

```bash
python backend/scripts/verify_models.py
```

### The optional RetinaFace backend

RetinaFace needs TensorFlow (~1 GB resident) and takes ~14.5 s per image on
CPU, so it is excluded from the deployed image. For local comparison:

```bash
pip install -r backend/requirements-retinaface.txt
ENABLE_RETINAFACE=1 python -m uvicorn backend.app.main:app --port 8000
```

It then appears as a selectable backend in the UI. Without it, the dropdown
shows it as *not installed* rather than failing at request time.

---

## Health checks

| Path | Use |
|---|---|
| `/api/health` | Liveness. Never loads models; answers immediately. |
| `/api/status` | Readiness detail: models loaded, gallery, thresholds. |

Point the platform's health check at `/api/health`. `/api/status` or `/` would
make the service look down during warm-up.

---

## Persisting the gallery

Enrolled identities are folders of images under `backend/gallery/`, baked into
the image, so a redeploy restores exactly what is in git.

To let people enrol without a rebuild, mount a disk and point `GALLERY_DIR` at
it. Embeddings cache in `.embeddings.json` beside the images and recompute only
when a file's size or mtime changes.

---

## Troubleshooting

**Build fails on `npm ci`** — `frontend/package-lock.json` must be committed.

**`FaceNet model missing`** — `backend/models/facenet.onnx` was not committed.
Check it is not caught by `.gitignore` (the ignore rules deliberately allow
`backend/models/*.onnx`).

**Service killed during startup** — out of memory. Confirm `ONNX_THREADS=1` and
that `ENABLE_RETINAFACE` is `0`; enabling RetinaFace pulls in TensorFlow and
will not fit 512 MB.

**First request after idle is slow** — the free instance was asleep. Expect
~50 seconds, once.

**`UnicodeEncodeError` locally on Windows** — only affects the optional
RetinaFace extras, whose logger prints emoji the cp1252 console cannot encode.
Run with `PYTHONUTF8=1` or use the `run-dev` scripts.
