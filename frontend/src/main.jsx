import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import {
  Activity, Aperture, BrainCircuit, ChevronRight, CircleHelp, Cpu, Download,
  FileImage, Github, Grid2X2, Info, Loader2, Moon, RotateCcw, ScanFace,
  ShieldCheck, Sparkles, Sun, Upload, UsersRound, X
} from "lucide-react";
import "./styles.css";

// Empty string => same origin. Set VITE_API_BASE only when the API is deployed
// separately from the frontend; in dev, vite.config.js proxies /api.
const API = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");
const REPO_URL = import.meta.env.VITE_REPO_URL ?? "https://github.com";

const api = (path) => `${API}${path}`;

const MECHANISMS = [
  {
    key: "robust",
    title: "Deep Detection",
    eyebrow: "ROBUST DETECTION",
    icon: ScanFace,
    needsDeep: true,
    text: "A modern deep detector locates multiple faces and estimates facial landmarks, handling scale and pose far better than a classical cascade.",
    detail:
      "YuNet is a 228 KB detector that runs through OpenCV's ONNX runtime and returns in milliseconds; it is the default and needs no TensorFlow. RetinaFace has higher recall on hard poses, but it is a 118 MB TensorFlow network that takes roughly 15 seconds per image on CPU, so it is an optional local extra rather than part of the deployed build. On this project's sample photograph both find the same faces and yield the same identification.",
    strength: "Accurate across scale and pose, milliseconds per image",
    limitation: "RetinaFace needs TensorFlow and is ~1000x slower"
  },
  {
    key: "viola-jones",
    title: "Viola–Jones",
    eyebrow: "CLASSICAL DETECTION",
    icon: Activity,
    needsDeep: false,
    text: "Haar-like features, integral images and an AdaBoost cascade give fast frontal-face detection at very low computational cost.",
    detail:
      "This build runs the frontal cascade over two grayscale variants (histogram-equalised and CLAHE), requires Haar eye structure inside each candidate, and applies IoU non-maximum suppression. It deliberately favours precision over recall, so textured regions such as clothing or foliage are rejected rather than reported as faces.",
    strength: "Fast, no model weights, fully explainable",
    limitation: "Frontal poses only; misses difficult angles"
  },
  {
    key: "template",
    title: "Template Matching",
    eyebrow: "CLASSICAL CV",
    icon: Grid2X2,
    needsDeep: false,
    text: "A reference patch is slid over the image using normalised correlation. Transparent and educational, but tied to the appearance of that one patch.",
    detail:
      "A single-scale matchTemplate call collapses as soon as the patch is resized. This build searches a bounded scale pyramid from 0.55× to 1.60× and blends grayscale correlation (72%) with Canny edge correlation (28%), then rejects anything under the score floor instead of drawing an arbitrary box.",
    strength: "Finds a known patch, scale-tolerantly",
    limitation: "Not a face detector; appearance-specific"
  },
  {
    key: "facenet",
    title: "FaceNet",
    eyebrow: "RECOGNITION",
    icon: BrainCircuit,
    needsDeep: true,
    text: "Each detected face becomes an embedding vector. Recognition compares that vector against your enrolled reference gallery by cosine similarity.",
    detail:
      "Faces are detected, then embedded with FaceNet into a 128-dimensional vector. This is the same FaceNet graph DeepFace ships, converted to ONNX so it runs without TensorFlow — embeddings agree with the original to seven decimal places. An identity is claimed only when cosine similarity clears the threshold; anything below is reported as Unknown rather than guessed at.",
    strength: "Answers 'who is this?' not just 'where?'",
    limitation: "Needs enrolled reference images"
  }
];

const SAMPLE_IMAGE = "/sample/group_faces.jpg";
const SAMPLE_TEMPLATE = "/sample/sundhar_face_template.jpg";

async function fetchAsFile(url, name) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not load ${url}`);
  const blob = await response.blob();
  return new File([blob], name, { type: blob.type || "image/jpeg" });
}

function useObjectUrl(file) {
  const [url, setUrl] = useState("");
  useEffect(() => {
    if (!file) {
      setUrl("");
      return undefined;
    }
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);
  return url;
}

function App() {
  const [dark, setDark] = useState(() => {
    try {
      const stored = localStorage.getItem("fv-theme");
      if (stored) return stored === "dark";
    } catch { /* storage can be unavailable */ }
    return true;
  });
  const [view, setView] = useState("workspace");
  const [file, setFile] = useState(null);
  const [fileSource, setFileSource] = useState("sample");
  const [template, setTemplate] = useState(null);
  const [templateSource, setTemplateSource] = useState("sample");
  const [method, setMethod] = useState("robust");
  const [recognition, setRecognition] = useState(true);
  const [detector, setDetector] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [drag, setDrag] = useState(false);
  const [status, setStatus] = useState(null);
  const [statusError, setStatusError] = useState(false);

  const previewUrl = useObjectUrl(file);
  const templateUrl = useObjectUrl(template);
  const pollRef = useRef(null);

  const selected = useMemo(
    () => MECHANISMS.find((m) => m.key === method) ?? MECHANISMS[0],
    [method]
  );
  const SelectedIcon = selected.icon;

  // --- theme -----------------------------------------------------------
  useEffect(() => {
    document.documentElement.dataset.theme = dark ? "dark" : "light";
    document.documentElement.style.colorScheme = dark ? "dark" : "light";
    try {
      localStorage.setItem("fv-theme", dark ? "dark" : "light");
    } catch { /* ignore */ }
  }, [dark]);

  // --- backend status ---------------------------------------------------
  const loadStatus = useCallback(async () => {
    try {
      const response = await fetch(api("/api/status"));
      if (!response.ok) throw new Error(String(response.status));
      const data = await response.json();
      setStatus(data);
      setStatusError(false);
      setDetector((current) => {
        const options = data.detector_backends ?? [];
        const usable = (key) => options.some((o) => o.key === key && o.available !== false);
        if (current && usable(current)) return current;
        if (usable(data.detector_backend)) return data.detector_backend;
        return options.find((o) => o.available !== false)?.key ?? "yunet";
      });
      return data;
    } catch {
      setStatusError(true);
      return null;
    }
  }, []);

  useEffect(() => {
    loadStatus();
  }, [loadStatus]);

  // Poll fast while models load on a cold start, then settle into a slow
  // heartbeat so the status pill still notices if the backend goes away.
  useEffect(() => {
    const interval = status?.warming_up ? 4000 : 30000;
    pollRef.current = setInterval(loadStatus, interval);
    return () => {
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = null;
    };
  }, [status?.warming_up, loadStatus]);

  // --- sample assets ----------------------------------------------------
  const loadSamples = useCallback(async (replaceImage = true) => {
    try {
      const [sampleImage, sampleTemplate] = await Promise.all([
        fetchAsFile(SAMPLE_IMAGE, "sample-group-faces.jpg"),
        fetchAsFile(SAMPLE_TEMPLATE, "sample-sundar-face.jpg")
      ]);
      if (replaceImage) {
        setFile(sampleImage);
        setFileSource("sample");
        setResult(null);
        setError("");
      }
      setTemplate(sampleTemplate);
      setTemplateSource("sample");
    } catch {
      // The workspace stays fully usable through the upload controls.
    }
  }, []);

  useEffect(() => {
    loadSamples(true);
  }, [loadSamples]);

  // --- actions ----------------------------------------------------------
  const chooseImage = (candidate) => {
    if (!candidate) return;
    if (!candidate.type.startsWith("image/")) {
      setError("That file is not an image. Choose a JPG, PNG or WebP.");
      return;
    }
    setFile(candidate);
    setFileSource("upload");
    setResult(null);
    setError("");
  };

  const chooseTemplate = (candidate) => {
    if (!candidate) return;
    if (!candidate.type.startsWith("image/")) {
      setError("The template must be an image file.");
      return;
    }
    setTemplate(candidate);
    setTemplateSource("upload");
    setError("");
  };

  const analyze = async () => {
    if (!file || busy) return;
    if (method === "template" && !template) {
      setError("Template Matching needs a reference template image.");
      return;
    }
    setBusy(true);
    setError("");
    try {
      const form = new FormData();
      form.append("image", file);
      form.append("method", method);
      form.append("recognition", String(recognition));
      if (detector) form.append("detector", detector);
      if (method === "template" && template) form.append("template", template);

      const response = await fetch(api("/api/analyze"), { method: "POST", body: form });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || `Request failed with status ${response.status}.`);
      }
      setResult(payload);
      loadStatus();
    } catch (e) {
      // A fetch that never reached the server throws TypeError, not an HTTP
      // error. That means the backend died or was never started, which is not
      // a fault of the selected method — say so explicitly instead of
      // surfacing a bare "Failed to fetch".
      const networkDown =
        e instanceof TypeError || /failed to fetch|networkerror|load failed/i.test(e.message);
      if (networkDown) {
        setError(
          "Could not reach the API — the request never left the browser. The backend is not running or has stopped. Start it with: python -m uvicorn backend.app.main:app --port 8000"
        );
        loadStatus();
      } else {
        setError(e.message);
      }
    } finally {
      setBusy(false);
    }
  };

  const deepReady = status?.deep_models_ready ?? false;
  const warming = status?.warming_up ?? false;
  const deepBlocked = Boolean(status) && !deepReady && selected.needsDeep;
  const people = status?.gallery_people ?? [];
  const recognitionLocked = method === "template" || method === "facenet";
  const usesDeepDetector = method === "robust" || method === "facenet";
  const activeDetector = useMemo(
    () => (status?.detector_backends ?? []).find((d) => d.key === detector) ?? null,
    [status, detector]
  );

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><ScanFace size={20} /></div>
          <div><strong>FaceVision</strong><span>Studio</span></div>
        </div>

        <nav>
          {[
            ["workspace", "Workspace"],
            ["methods", "Methods"],
            ["docs", "Documentation"]
          ].map(([key, label]) => (
            <button
              key={key}
              type="button"
              className={view === key ? "active" : ""}
              onClick={() => setView(key)}
            >
              {label}
            </button>
          ))}
        </nav>

        <div className="top-actions">
          <StatusPill status={status} statusError={statusError} />
          <button
            className="icon-btn"
            onClick={() => setDark(!dark)}
            title={dark ? "Switch to light theme" : "Switch to dark theme"}
            aria-label="Toggle colour theme"
          >
            {dark ? <Sun size={17} /> : <Moon size={17} />}
          </button>
          <a
            className="github"
            href={REPO_URL}
            target="_blank"
            rel="noreferrer noopener"
            aria-label="Source code"
          >
            <Github size={17} />
          </a>
        </div>
      </header>

      <main>
        <section className="hero">
          <div>
            <div className="eyebrow"><Sparkles size={14} /> COMPUTER VISION WORKBENCH</div>
            <h1>See the face.<br /><em>Understand the mechanism.</em></h1>
            <p>
              Detect multiple faces, inspect classical computer vision side by side with a modern
              detector, and run genuine FaceNet recognition against an enrolled reference gallery.
            </p>
          </div>
          <div className="hero-meta">
            <span><ShieldCheck size={15} /> Images are processed per request, never stored</span>
            <span><Cpu size={15} /> Python · OpenCV · DeepFace · FaceNet</span>
            <span><UsersRound size={15} /> {people.length} enrolled {people.length === 1 ? "identity" : "identities"}</span>
          </div>
        </section>

        {warming && (
          <div className="banner">
            <Loader2 size={16} className="spin" />
            <div>
              <b>Loading models…</b>
              <span>
                RetinaFace and FaceNet weights are being prepared. Classical modes work right now;
                deep modes become available in a moment.
              </span>
            </div>
          </div>
        )}

        {statusError && (
          <div className="banner warn">
            <CircleHelp size={16} />
            <div>
              <b>API unreachable</b>
              <span>The backend did not answer /api/status. Locally, start the FastAPI server.</span>
            </div>
          </div>
        )}

        {view === "workspace" && (
          <section className="workspace">
            <aside className="sidebar">
              <div className="side-label">ANALYSIS MODE</div>
              {MECHANISMS.map((m) => {
                const Icon = m.icon;
                const unavailable = m.needsDeep && Boolean(status) && !deepReady;
                return (
                  <button
                    key={m.key}
                    type="button"
                    className={`method ${method === m.key ? "selected" : ""}`}
                    onClick={() => { setMethod(m.key); setResult(null); setError(""); }}
                  >
                    <div className="method-icon"><Icon size={18} /></div>
                    <div>
                      <small>{m.eyebrow}</small>
                      <b>{m.title}</b>
                      {unavailable && <i className="flag">{warming ? "loading" : "unavailable"}</i>}
                    </div>
                    {method === m.key && <ChevronRight size={16} />}
                  </button>
                );
              })}

              <div className="side-divider" />
              <div className="side-label">PIPELINE</div>
              <div className="pipeline">
                {pipelineFor(method, activeDetector?.label ?? "deep detector").map((step, index, all) => (
                  <React.Fragment key={step}>
                    <div className="pipe-step">
                      <span>{String(index + 1).padStart(2, "0")}</span> {step}
                    </div>
                    {index < all.length - 1 && <div className="pipe-line" />}
                  </React.Fragment>
                ))}
              </div>

              {people.length > 0 && (
                <>
                  <div className="side-divider" />
                  <div className="side-label">ENROLLED GALLERY</div>
                  <div className="people">
                    {people.map((person) => <span key={person}>{person}</span>)}
                  </div>
                </>
              )}
            </aside>

            <section className="canvas">
              <div className="canvas-head">
                <div><span className="crumb">Workspace</span><ChevronRight size={14} /><b>{selected.title}</b></div>
                <span className="engine">
                  <span className={`dot ${deepReady ? "" : warming ? "amber" : "grey"}`} />
                  {deepReady ? "Engine ready" : warming ? "Warming up" : "Classical only"}
                </span>
              </div>

              <div
                className={`dropzone ${drag ? "drag" : ""} ${file ? "has-file" : ""}`}
                onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
                onDragLeave={() => setDrag(false)}
                onDrop={(e) => { e.preventDefault(); setDrag(false); chooseImage(e.dataTransfer.files[0]); }}
              >
                {file ? (
                  <div className="file-selected">
                    {previewUrl
                      ? <img className="input-preview" src={previewUrl} alt="Selected input" />
                      : <div className="file-icon"><FileImage size={27} /></div>}
                    <div className="file-info">
                      <b>{file.name}</b>
                      <span>
                        {(file.size / 1024 / 1024).toFixed(2)} MB ·{" "}
                        {fileSource === "sample" ? "Built-in sample" : "Uploaded"} · ready
                      </span>
                    </div>
                    <label className="mini-upload">
                      Replace
                      <input type="file" accept="image/*" onChange={(e) => chooseImage(e.target.files[0])} />
                    </label>
                    <button
                      className="remove"
                      type="button"
                      aria-label="Clear selected image"
                      onClick={() => { setFile(null); setFileSource("upload"); setResult(null); }}
                    >
                      <X size={17} />
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="upload-icon"><Upload size={22} /></div>
                    <h3>Drop an image here</h3>
                    <p>or choose a JPG, PNG or WebP image (max {status?.limits?.max_upload_mb ?? 12} MB)</p>
                    <div className="dropzone-actions">
                      <label className="browse">
                        Browse files
                        <input type="file" accept="image/*" onChange={(e) => chooseImage(e.target.files[0])} />
                      </label>
                      <button className="browse" type="button" onClick={() => loadSamples(true)}>
                        <RotateCcw size={13} /> Use sample
                      </button>
                    </div>
                  </>
                )}
              </div>

              {method === "template" && (
                <div className="template-row">
                  {templateUrl && <img className="template-preview" src={templateUrl} alt="Reference template" />}
                  <div>
                    <b>Reference template</b>
                    <span>
                      {template
                        ? `${template.name} · ${templateSource === "sample" ? "built-in sample" : "uploaded"}`
                        : "Upload a face patch to search for visually similar regions."}
                    </span>
                  </div>
                  <label className="mini-upload">
                    {template ? "Replace" : "Choose template"}
                    <input type="file" accept="image/*" onChange={(e) => chooseTemplate(e.target.files[0])} />
                  </label>
                </div>
              )}

              {usesDeepDetector && (
                <div className="detector-row">
                  <div>
                    <b>Detector backend</b>
                    <span>
                      {activeDetector
                        ? `${activeDetector.label} · ~${
                            activeDetector.approx_ms >= 1000
                              ? `${(activeDetector.approx_ms / 1000).toFixed(1)} s`
                              : `${activeDetector.approx_ms} ms`
                          } per image · ${activeDetector.weights_mb} MB of weights`
                        : "Choose which deep model locates the faces."}
                    </span>
                  </div>
                  <select
                    value={detector}
                    onChange={(e) => setDetector(e.target.value)}
                    aria-label="Deep detector backend"
                  >
                    {(status?.detector_backends ?? []).map((option) => (
                      <option
                        key={option.key}
                        value={option.key}
                        disabled={option.available === false}
                      >
                        {option.label} ({option.speed})
                        {option.available === false ? " — not installed" : ""}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              {usesDeepDetector && activeDetector?.approx_ms >= 5000 && (
                <div className="notice">
                  <Info size={15} />
                  {activeDetector.label} is the most thorough backend but takes roughly{" "}
                  {(activeDetector.approx_ms / 1000).toFixed(0)} seconds per image on a free CPU
                  instance. Switch to YuNet for near-instant results.
                </div>
              )}

              <div className="controls">
                <label className={`toggle ${recognitionLocked ? "disabled" : ""}`}>
                  <input
                    type="checkbox"
                    disabled={recognitionLocked}
                    checked={recognitionLocked ? method === "facenet" : recognition}
                    onChange={(e) => setRecognition(e.target.checked)}
                  />
                  <span className="switch" />
                  <span>
                    {method === "template"
                      ? "Recognition not used in this mode"
                      : method === "facenet"
                        ? "Recognition always on"
                        : "Run recognition"}
                  </span>
                </label>
                <button className="analyze" disabled={!file || busy} onClick={analyze}>
                  {busy
                    ? <><span className="spinner" />Analyzing…</>
                    : <><ScanFace size={17} /> Analyze image</>}
                </button>
              </div>

              {deepBlocked && !warming && (
                <div className="notice">
                  <Info size={15} />
                  {selected.title} needs the deep model stack, which is not available on this
                  instance. Viola–Jones and Template Matching remain fully functional.
                </div>
              )}

              {busy && !deepReady && selected.needsDeep && (
                <div className="notice">
                  <Loader2 size={15} className="spin" />
                  First deep run loads ~210 MB of weights and can take up to a minute. Later runs are fast.
                </div>
              )}

              {error && <div className="error"><CircleHelp size={17} />{error}</div>}

              {result ? <ResultPanel result={result} /> : (
                <div className="empty-state">
                  <Aperture size={27} />
                  <b>Analysis output will appear here</b>
                  <span>Choose an image and select Analyze image to begin.</span>
                </div>
              )}
            </section>

            <aside className="inspector">
              <div className="side-label">METHOD NOTES</div>
              <div className="note-icon"><SelectedIcon size={20} /></div>
              <h2>{selected.title}</h2>
              <p>{selected.text}</p>

              <div className="spec">
                <div><span>Input</span><b>RGB image</b></div>
                <div><span>Output</span><b>{method === "template" ? "Match box" : "Face boxes"}</b></div>
                <div><span>Recognition</span><b>{method === "template" ? "N/A" : "FaceNet"}</b></div>
                <div>
                  <span>Threshold</span>
                  <b>
                    {method === "template"
                      ? status?.thresholds?.template_score ?? "0.60"
                      : status?.thresholds?.recognition_similarity ?? "0.60"}
                  </b>
                </div>
              </div>

              <div className="learn">
                <Info size={15} />
                <div>
                  <b>Recognition is not detection</b>
                  <span>
                    Detection answers “where is a face?”. Recognition answers “does this face match
                    an enrolled reference?”.
                  </span>
                </div>
              </div>
            </aside>
          </section>
        )}

        {view === "methods" && <MethodsView onPick={(key) => { setMethod(key); setView("workspace"); }} />}
        {view === "docs" && <DocsView status={status} />}
      </main>

      <footer>
        <span>FACEVISION STUDIO · CV LAB</span>
        <span>Built with OpenCV · DeepFace · FaceNet</span>
      </footer>
    </div>
  );
}

function pipelineFor(method, detectorLabel = "deep detector") {
  if (method === "template") {
    return ["Upload image", "Upload template", "Multi-scale correlation", "Report best match"];
  }
  if (method === "viola-jones") {
    return ["Upload image", "Grayscale + CLAHE", "Haar cascade + eye check", "Suppress overlaps"];
  }
  if (method === "robust") {
    return ["Upload image", `Detect with ${detectorLabel}`, "Filter + suppress", "Embed and compare"];
  }
  return ["Upload image", `Detect with ${detectorLabel}`, "Embed with FaceNet", "Compare gallery"];
}

function StatusPill({ status, statusError }) {
  if (statusError) return <div className="status"><span className="dot red" />API offline</div>;
  if (!status) return <div className="status"><span className="dot grey" />Connecting…</div>;
  if (status.warming_up) return <div className="status"><span className="dot amber" />Loading models</div>;
  if (status.deep_models_ready) return <div className="status"><span className="dot" />All models ready</div>;
  return <div className="status"><span className="dot grey" />Classical only</div>;
}

function ResultPanel({ result }) {
  const isTemplate = result.result_kind === "matches";
  const noun = isTemplate ? "match" : "face";
  const plural = isTemplate ? "matches" : "faces";
  const heading = `${result.count} ${result.count === 1 ? noun : plural} ${isTemplate ? "found" : "detected"}`;
  const score = result.template_score;

  return (
    <div className="results">
      <div className="result-head">
        <div>
          <span className="side-label">ANALYSIS RESULT</span>
          <h2>{heading}</h2>
        </div>
        <div className="result-stats">
          <span><b>{result.source_width}×{result.source_height}</b> source</span>
          {result.resized && <span><b>{result.width}×{result.height}</b> analysed</span>}
          <span><b>{result.elapsed_ms} ms</b></span>
          <span><b>{result.detector}</b></span>
        </div>
      </div>

      {result.fallback_used && (
        <div className="match-summary warn">
          RetinaFace was unavailable, so this run used the Viola–Jones cascade instead.
        </div>
      )}

      {isTemplate && (
        <div className="match-summary">
          Best normalised correlation: <b>{score != null ? `${(score * 100).toFixed(1)}%` : "—"}</b>.
          Scale variation between 0.55× and 1.60× is searched automatically.
        </div>
      )}

      {!isTemplate && !result.recognition_ran && (
        <div className="match-summary">
          Detection only — recognition was not run for this request.
        </div>
      )}

      <div className="result-image">
        <img src={result.image} alt="Annotated analysis result" />
      </div>

      <div className="result-actions">
        <a className="browse" href={result.image} download={`facevision-${result.method}.jpg`}>
          <Download size={13} /> Download annotated image
        </a>
      </div>

      <div className="face-list">
        {result.faces.length === 0 ? (
          <div className="no-results">
            No reliable {noun} found above the confidence threshold
            {isTemplate && score != null ? ` (best score ${(score * 100).toFixed(1)}%)` : ""}.
          </div>
        ) : (
          result.faces.map((face) => (
            <div className="face-row" key={face.id}>
              <div className="face-number">{String(face.id).padStart(2, "0")}</div>
              <div className="face-desc">
                <b>{face.identity}</b>
                <span>
                  Box {face.box.x},{face.box.y} · {face.box.w}×{face.box.h}
                  {face.match?.reference && face.match?.matched ? ` · ref ${face.match.reference}` : ""}
                  {face.match?.error ? ` · ${face.match.error}` : ""}
                </span>
              </div>
              {face.match?.similarity != null && (
                <div className="score">
                  <b>{(face.match.similarity * 100).toFixed(1)}%</b>
                  <span>{face.match.matched ? "similarity" : `below ${(face.match.threshold * 100).toFixed(0)}%`}</span>
                </div>
              )}
              {face.match?.score != null && (
                <div className="score">
                  <b>{(face.match.score * 100).toFixed(1)}%</b>
                  <span>match score</span>
                </div>
              )}
            </div>
          ))
        )}
      </div>
    </div>
  );
}

function MethodsView({ onPick }) {
  return (
    <section className="page">
      <h2 className="page-title">Methods</h2>
      <p className="page-lead">
        Four mechanisms, chosen to contrast classical computer vision with modern deep models.
        They are not interchangeable: two answer “where is a face?”, one answers “where does this
        patch occur?”, and one answers “who is this?”.
      </p>
      <div className="method-grid">
        {MECHANISMS.map((m) => {
          const Icon = m.icon;
          return (
            <article key={m.key} className="method-card">
              <div className="method-icon"><Icon size={18} /></div>
              <small>{m.eyebrow}</small>
              <h3>{m.title}</h3>
              <p>{m.detail}</p>
              <dl>
                <div><dt>Strength</dt><dd>{m.strength}</dd></div>
                <div><dt>Limitation</dt><dd>{m.limitation}</dd></div>
              </dl>
              <button type="button" className="browse" onClick={() => onPick(m.key)}>
                Try {m.title} <ChevronRight size={13} />
              </button>
            </article>
          );
        })}
      </div>
    </section>
  );
}

function DocsView({ status }) {
  const recognitionThreshold = status?.thresholds?.recognition_similarity ?? 0.6;
  const templateThreshold = status?.thresholds?.template_score ?? 0.6;
  return (
    <section className="page">
      <h2 className="page-title">Documentation</h2>

      <div className="doc-block">
        <h3>Detection, matching and recognition are different questions</h3>
        <ul>
          <li><b>Detection</b> — “Where is a face?” Answered by RetinaFace or Viola–Jones.</li>
          <li><b>Template matching</b> — “Where does this reference patch visually occur?” It is not a face detector; it finds that specific patch.</li>
          <li><b>Recognition</b> — “Which enrolled identity is this detected face most similar to?” Answered by FaceNet embeddings plus a cosine threshold.</li>
        </ul>
      </div>

      <div className="doc-block">
        <h3>How a result is decided</h3>
        <p>
          A detected face is embedded into a FaceNet vector and compared with every enrolled
          reference by cosine similarity. An identity is claimed only when the best similarity
          reaches <b>{Number(recognitionThreshold).toFixed(2)}</b>; below that the face is reported
          as <b>Unknown</b> rather than guessed. Template matching reports a match only above a
          combined score of <b>{Number(templateThreshold).toFixed(2)}</b>.
        </p>
      </div>

      <div className="doc-block">
        <h3>Enrolling your own identities</h3>
        <p>
          Add a folder per person under <code>backend/gallery/</code> containing one or more clear,
          mostly frontal photographs, then restart the server. Embeddings are cached on disk and
          recomputed only when a file changes.
        </p>
        <pre>{`backend/gallery/
  Ada Lovelace/
    ada1.jpg
    ada2.jpg`}</pre>
      </div>

      <div className="doc-block">
        <h3>HTTP API</h3>
        <table className="api-table">
          <thead><tr><th>Endpoint</th><th>Method</th><th>Purpose</th></tr></thead>
          <tbody>
            <tr><td><code>/api/health</code></td><td>GET</td><td>Liveness probe; never loads models</td></tr>
            <tr><td><code>/api/status</code></td><td>GET</td><td>Model readiness, gallery, thresholds, limits</td></tr>
            <tr><td><code>/api/gallery</code></td><td>GET</td><td>Enrolled identities</td></tr>
            <tr><td><code>/api/analyze</code></td><td>POST</td><td>Detection, matching and recognition</td></tr>
            <tr><td><code>/api/template-match</code></td><td>POST</td><td>Template matching only</td></tr>
            <tr><td><code>/api/warm-up</code></td><td>POST</td><td>Force model loading</td></tr>
            <tr><td><code>/api/docs</code></td><td>GET</td><td>Interactive OpenAPI explorer</td></tr>
          </tbody>
        </table>
      </div>

      <div className="doc-block">
        <h3>Privacy</h3>
        <p>
          Uploaded images are held in memory for the duration of the request and are never written
          to disk. Only the images you place in the gallery folder persist.

          Done by Tamilselvi R to demonstrate the capabilities of FaceVision Studio. The source code is available on GitHub. Done for an educational purpose and to showcase the features of FaceVision Studio.
        </p>
      </div>
    </section>
  );
}

createRoot(document.getElementById("root")).render(<App />);
