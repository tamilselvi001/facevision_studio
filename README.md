# FaceVision Studio

A computer-vision workbench that puts classical and modern face analysis side by
side, so you can see not just *what* each method answers but *where it breaks*.

WEB SERVICE AVAILABLE ON : https://facevision-studio.onrender.com
may not have some models for its free deployment limitations!

Four mechanisms, one image, one click:

| Mode | Question it answers | Engine |
|---|---|---|
| **Deep Detection** | Where are the faces? | YuNet (ONNX) · RetinaFace (optional) |
| **Viola–Jones** | Where are the faces? (classically) | Haar cascade + eye validation |
| **Template Matching** | Where does *this patch* occur? | Multi-scale normalised correlation |
| **FaceNet** | *Who* is this? | FaceNet embeddings vs. an enrolled gallery |

> **Detection** asks “where is a face?” · **Template matching** asks “where does
> this reference patch occur?” · **Recognition** asks “which enrolled identity is
> this?” These are not interchangeable, and the UI never pretends they are.

---

## Quick start

### Docker (closest to production)

```bash
docker build -t facevision-studio .
docker run -p 7860:7860 facevision-studio
```

Open <http://localhost:7860>.

### Local development

Two terminals. **Backend:**

```bash
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1        # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cd ..
python -m uvicorn backend.app.main:app --reload --port 8000
```

**Frontend:**

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:5173>. Vite proxies `/api` to port 8000, so no API URL is
ever hardcoded.

---

## Deploying

**Free hosting with every feature working: [Render](https://render.com).**
No credit card, 512 MB, Docker. The whole service peaks around 265 MB because
FaceNet runs on ONNX Runtime rather than TensorFlow, so everything fits.

Full step-by-step instructions are in **[DEPLOYMENT.md](DEPLOYMENT.md)**.

> Hugging Face Spaces is no longer an option: Docker and Gradio Spaces became
> PRO-only in 2026. Only Static Spaces remain free, and this app needs a server.

---

## Choosing a detector

Measured on the bundled 820×400 sample, warm:

| Backend | Time/image | Weights | Needs TensorFlow | Faces found |
|---|---|---|---|---|
| `yunet` *(default)* | **~15 ms** | 228 KB | No | 2 |
| `retinaface` | ~14.5 s | 118 MB | Yes (~1 GB RAM) | 2 |

Both find the same faces here and yield the same identification. RetinaFace has
better recall on hard poses and occlusion, so it stays available — but at
roughly a thousand times the cost on CPU it cannot run on a free tier. It is an
optional local extra:

```bash
pip install -r backend/requirements-retinaface.txt
ENABLE_RETINAFACE=1 python -m uvicorn backend.app.main:app --port 8000
```

---

## Enrolling your own identities

Add one folder per person under `backend/gallery/`, then restart:

```
backend/gallery/
  Ada Lovelace/
    ada1.jpg
    ada2.jpg
  Alan Turing/
    alan1.jpg
```

Use clear, mostly frontal photographs. Each reference is detected, cropped and
embedded once; results are cached in `.embeddings.json` and recomputed only when
a file changes. A face is labelled with an identity only when cosine similarity
reaches `RECOGNITION_THRESHOLD` (default **0.60**) — otherwise it is reported as
**Unknown** rather than guessed.

The bundled gallery contains Elon Musk and Sundar Pichai. Historic misspelled
folder names (`Elon Mask`, `Sundhar Pichai`) are still mapped to the corrected
display names, so an older gallery keeps working.

---

## API

Interactive explorer at **`/api/docs`**.

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/health` | GET | Liveness. Never loads models; safe as a platform health check. |
| `/api/status` | GET | Model readiness, gallery, thresholds, limits, backends. |
| `/api/gallery` | GET | Enrolled identities. |
| `/api/analyze` | POST | Detection, template matching and recognition. |
| `/api/template-match` | POST | Template matching only. |
| `/api/warm-up` | POST | Force model loading. |

`POST /api/analyze` takes multipart form data:

| Field | Type | Notes |
|---|---|---|
| `image` | file | **Required.** JPG, PNG, WebP or BMP. |
| `method` | text | `robust` (default), `viola-jones`, `template`, `facenet`. |
| `detector` | text | `yunet` (default), `ssd`, `retinaface`. |
| `recognition` | bool | Default `true`. Forced on for `facenet`, off for `template`. |
| `template` | file | Required when `method=template`. |

```bash
curl -X POST http://localhost:7860/api/analyze \
  -F "image=@photo.jpg" \
  -F "method=facenet" \
  -F "detector=yunet"
```

---

## How each method works

### Deep detection
A single-stage network returns boxes and landmarks. Candidates below
`DETECTION_MIN_CONFIDENCE` are dropped, as is the whole-frame box some backends
emit when they find nothing. If the deep stack is unavailable the request
degrades to Viola–Jones and says so in the response rather than failing.

### Viola–Jones
Haar-like features over an integral image with an AdaBoost cascade. This build
runs the frontal cascade on two grayscale variants (histogram-equalised and
CLAHE), requires Haar eye structure inside each candidate, checks aspect ratio
and minimum area, then applies IoU non-maximum suppression. It deliberately
favours precision over recall, which is what stops clothing and foliage being
reported as faces.

### Template matching
A single-scale `matchTemplate` collapses the moment the patch is resized — a
browser-resized crop can legitimately score ~0.48 and be discarded. This build
searches a bounded pyramid from 0.55× to 1.60× and blends grayscale correlation
(72%) with Canny edge correlation (28%), the latter rejecting flat or uniformly
textured regions. Below `TEMPLATE_MIN_SCORE` it reports *no match* instead of
drawing an arbitrary box — and it reports a **template match**, never a "face
detected", because template matching is not a face detector.

### FaceNet recognition
Each detected face is embedded into a 128-dimensional FaceNet vector and
compared by cosine similarity against every enrolled reference. The closest
identity is returned with its score; below threshold the result is **Unknown**.

This is the **same FaceNet graph DeepFace ships**, exported to ONNX and stored
at float16 so it runs on ONNX Runtime with no TensorFlow. Embeddings were
checked against the TensorFlow original on every bundled reference image:
cosine agreement **0.9999977 or better**, i.e. identical to floating-point
noise. The practical effect is that the model needs ~265 MB instead of ~1 GB
and loads in 0.3 s instead of 20 s, which is what makes free hosting possible.

Preprocessing is a faithful port of DeepFace's own and is load-bearing: the
network is fed **BGR** pixels (DeepFace converts to RGB and back) with
aspect-preserving zero padding rather than a stretch. Getting either wrong
drops agreement to ~0.6.

---

## Configuration

All settings are environment variables with production-sane defaults — see
[`.env.example`](.env.example) and the table in [DEPLOYMENT.md](DEPLOYMENT.md).

---

## Testing

```bash
pip install -r backend/requirements-dev.txt
python backend/scripts/smoke_test.py
```

Boots the app in-process and exercises health, status, routing, both classical
modes, input validation and — when the weights are present — the deep pipeline.
Deep checks are skipped, not failed, in an environment without them. CI runs
this on every push alongside a frontend build.

---

## Privacy

Uploaded images are held in memory for the duration of the request and are never
written to disk. Only images you place in the gallery folder persist.

---

## Project layout

```
backend/
  app/
    config.py      environment-driven settings
    engines.py     ONNX model loading (FaceNet, YuNet); no TensorFlow
    main.py        FastAPI routes, static SPA serving
    vision.py      geometry, classical CV, gallery
  models/
    facenet.onnx   FaceNet 128-d, float16 (~44 MB)
    yunet.onnx     YuNet detector (~228 KB)
  gallery/         one folder per enrolled person
  scripts/         model verification, smoke test
frontend/
  src/main.jsx     the whole UI
  public/sample/   bundled demo images
Dockerfile         node build stage -> python runtime
render.yaml        Render blueprint (free tier)
```

## License

MIT — see [LICENSE](LICENSE).
