# Deployment guide

The repository builds into **one container** that serves both the FastAPI API
and the compiled React frontend on a single origin. No CORS configuration, no
second service, no separate frontend host.

---

## Which free host should I use?

The deciding constraint is memory. TensorFlow plus the FaceNet weights need
roughly **1 GB resident**.

| Host | Free RAM | All four features? | Notes |
|---|---|---|---|
| **Hugging Face Spaces** | **16 GB** | **Yes** | Docker SDK, no card required. **Recommended.** |
| Render | 512 MB | No — classical modes only | Deep models are OOM-killed on free. Needs the 2 GB `standard` plan. |
| Railway | ~512 MB trial | No | Trial credit expires. |
| Fly.io | 256 MB default | No | Needs a paid, larger VM. |

**Use Hugging Face Spaces.** It is the only free tier that runs detection,
template matching *and* FaceNet recognition without compromise.

---

## Option A — Hugging Face Spaces (recommended)

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "FaceVision Studio"
git branch -M main
git remote add origin https://github.com/<you>/facevision-studio.git
git push -u origin main
```

### 2. Create the Space

Go to <https://huggingface.co/new-space>:

- **Space SDK**: `Docker` → `Blank`
- **Hardware**: `CPU basic` (free)
- **Visibility**: Public

### 3. Push the code to the Space

The Space is its own git repository. Add it as a second remote:

```bash
git remote add space https://huggingface.co/spaces/<you>/facevision-studio
git push space main
```

When prompted for a password, use a **write token** from
<https://huggingface.co/settings/tokens>.

> The `README.md` front matter (`sdk: docker`, `app_port: 7860`) is what tells
> the Space how to run. Keep it at the top of the file.

### 4. Automate it (optional)

`.github/workflows/sync-to-hf-space.yml` mirrors GitHub to the Space on every
push to `main`, so afterwards `git push origin main` is the only command you
need. Configure it once in **Settings → Secrets and variables → Actions**:

| Kind | Name | Value |
|---|---|---|
| Secret | `HF_TOKEN` | a Hugging Face **write** token |
| Variable | `HF_SPACE` | `<you>/facevision-studio` |

### 5. First build

The first build takes roughly 6–10 minutes (TensorFlow is a large wheel). The
build also pre-downloads YuNet, SSD and FaceNet weights, so the first request
is fast rather than hanging on a download.

To include RetinaFace in the image (adds ~118 MB), set a build argument in the
Space settings or change the Dockerfile default:

```dockerfile
ARG PREFETCH_RETINAFACE=1
```

Otherwise RetinaFace is downloaded on first use, which is still supported.

---

## Option B — Render

`render.yaml` is a ready blueprint, but note the memory ceiling.

**Free tier** — ships with `DEEP_MODELS_ENABLED=0`. Viola–Jones detection and
template matching work completely; the two deep modes are reported as
unavailable in the UI rather than crashing. This is a deliberate, honest
degradation.

**Full features** — edit `render.yaml`:

```yaml
plan: standard          # 2 GB RAM
envVars:
  - key: DEEP_MODELS_ENABLED
    value: 1
```

Then: New → Blueprint → connect the repo. Render reads `render.yaml`
automatically.

Free instances sleep after 15 minutes idle and take ~50 seconds to wake.

---

## Option C — Any Docker host

```bash
docker build -t facevision-studio .
docker run -p 7860:7860 facevision-studio
```

Open <http://localhost:7860>.

Build with RetinaFace baked in:

```bash
docker build --build-arg PREFETCH_RETINAFACE=1 -t facevision-studio .
```

---

## Environment variables

Every setting has a working default; see `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `PORT` | `7860` | Bind port |
| `ALLOWED_ORIGINS` | `*` | Only needed for a split-origin deployment |
| `DETECTOR_BACKEND` | `yunet` | `yunet`, `ssd` or `retinaface` |
| `DEEP_MODELS_ENABLED` | `1` | `0` runs classical-CV only, no TensorFlow |
| `WARM_UP_ON_STARTUP` | `1` | Load models in the background at boot |
| `RECOGNITION_THRESHOLD` | `0.60` | Cosine similarity needed to claim an identity |
| `TEMPLATE_MIN_SCORE` | `0.60` | Correlation floor for a template match |
| `MAX_IMAGE_DIM` | `1600` | Inputs are fitted inside this box first |
| `MAX_UPLOAD_MB` | `12` | Rejected above this size |
| `GALLERY_DIR` | `backend/gallery` | Point at a mounted disk to persist enrolments |

---

## Detector backends

Measured on the bundled 820×400 sample, warm, 2 vCPU:

| Backend | Time/image | Weights | Faces found |
|---|---|---|---|
| `yunet` | **~15 ms** | 233 KB | 2 |
| `ssd` | ~35 ms | 10.7 MB | 2 |
| `retinaface` | ~14.5 s | 118 MB | 2 |

All three locate the same faces on this sample and produce the same FaceNet
identification. RetinaFace has higher recall on hard poses and heavy occlusion,
which is why it stays available — but it is roughly a thousand times slower on
CPU, so **YuNet is the default**. The picker in the UI lets anyone compare them
directly.

---

## Persisting the gallery

Enrolled identities live as folders of images under `backend/gallery/`. They are
baked into the image, so a redeploy restores exactly what is in git.

To let people enrol without a rebuild, mount a volume and point `GALLERY_DIR` at
it. Embeddings are cached in `.embeddings.json` beside the images and recomputed
only when a file's size or mtime changes.

---

## Health checks

| Path | Use |
|---|---|
| `/api/health` | Liveness. Never touches TensorFlow, answers immediately even while models load. |
| `/api/status` | Readiness detail: which models are loaded, gallery contents, thresholds. |

Point your platform's health check at `/api/health`. Using `/api/status` or `/`
would make the service look down during the model warm-up.

---

## Troubleshooting

**Build fails on `npm ci`** — `frontend/package-lock.json` must be committed.

**`ImportError: libGL.so.1`** — the runtime image needs `libgl1` and
`libglib2.0-0`. Both are installed in the Dockerfile; they are required because
DeepFace depends on `opencv-python` rather than the headless build.

**Container is killed during startup** — out of memory. Either raise the
instance size or set `DEEP_MODELS_ENABLED=0`.

**First deep request is slow** — weights are loading. `/api/status` reports
`warming_up: true` and the UI shows a banner. Baking weights in at build time
(the default) avoids the download portion.

**`UnicodeEncodeError` running locally on Windows** — DeepFace's logger prints
emoji that the cp1252 console cannot encode. Run with `PYTHONUTF8=1`, or use the
`run-dev` scripts which set it for you. Linux containers are unaffected.
