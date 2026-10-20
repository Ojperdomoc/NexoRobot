/**
 * Nexo · Inspección de soldadura
 * Lógica de la aplicación: cámara, detección en vivo, captura, informe PDF.
 */
import {
  SolderDetector,
  filterDetections,
  evaluateAgainstGroundTruth,
  iou,
} from './detector.js';
import {
  CLASS_INFO,
  SEVERITY_COLORS,
  SEVERITY_LABELS,
  severityOf,
  summaryOf,
  buildRecommendations,
  PROCESS_ADVICE,
} from './ipc.js';
import { buildInspectionReport } from './report.js';

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

const SAMPLES = [
  { id: 'placa-demo-01', label: 'Placa de control · vista general' },
  { id: 'placa-demo-02', label: 'Módulo de potencia · macro' },
  { id: 'placa-demo-03', label: 'Placa mixta SMT / THT' },
];

const HISTORY_KEY = 'nexo-soldadura-historial-v1';
const SETTINGS_KEY = 'nexo-soldadura-ajustes-v1';

const el = {};
const state = {
  detector: null,
  modelReady: false,
  modelError: null,
  mode: 'idle', // idle | live | capture
  stream: null,
  videoDevices: [],
  torchOn: false,
  torchAvailable: false,
  rawDetections: [],
  detections: [],
  prevLive: [],
  liveDets: [],
  capture: null, // {canvas, url, width, height}
  sourceSize: { w: 0, h: 0 },
  inference: { ms: 0, tiles: 0, tileSize: 0, source: '' },
  fps: 0,
  inferCount: 0,
  inferWindowStart: 0,
  busy: false,
  lastProgress: 0,
  settings: {
    conf: 0.35,
    zoom: 1,
    showOk: true,
    warnOn: true,
    failOn: true,
    showLabels: true,
    onlyDefects: false,
    ipcClass: 2,
  },
  folio: '',
  sample: null,
  groundTruth: null,
  activeTab: 'detections',
};

/* ─────────────────────────── utilidades ─────────────────────────── */

function toast(message, type = 'info', ms = 3800) {
  const t = el.toast;
  t.textContent = message;
  t.className = `toast ${type === 'error' ? 'error' : ''}`;
  t.hidden = false;
  clearTimeout(toast._timer);
  toast._timer = setTimeout(() => {
    t.hidden = true;
  }, ms);
}

function announce(msg) {
  el.announcer.textContent = msg;
}

function folioFor(date = new Date()) {
  const p = (n) => String(n).padStart(2, '0');
  const rand = Math.random().toString(36).slice(2, 5).toUpperCase();
  return `NEX-${date.getFullYear()}${p(date.getMonth() + 1)}${p(date.getDate())}-${p(date.getHours())}${p(date.getMinutes())}-${rand}`;
}

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function hexToRgba(hex, alpha) {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${alpha})`;
}

function loadSettings() {
  try {
    const raw = JSON.parse(localStorage.getItem(SETTINGS_KEY) || 'null');
    if (raw) Object.assign(state.settings, raw);
  } catch { /* ignore */ }
}

function persistSettings() {
  try {
    localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings));
  } catch { /* ignore */ }
}

/* ─────────────────────────── modelo ─────────────────────────── */

async function waitForOrt(timeoutMs = 8000) {
  const t0 = performance.now();
  while (!window.ort && performance.now() - t0 < timeoutMs) {
    await new Promise((r) => setTimeout(r, 100));
  }
  return window.ort;
}

async function initModel() {
  setModelStatus('loading', 'Cargando modelo…');
  try {
    const ort = await waitForOrt();
    if (!ort) throw new Error('No se cargó ONNX Runtime (revisa vendor/ort.wasm.min.js)');
    const url = new URL('modelos/nexo-solder.onnx', document.baseURI).href;
    const metaUrl = new URL('modelos/nexo-solder.json', document.baseURI).href;
    const detector = await SolderDetector.load({
      modelUrl: url,
      metaUrl,
      wasmPaths: new URL('vendor/', document.baseURI).href,
    });
    // calentamiento con un tensor sintético
    const warm = document.createElement('canvas');
    warm.width = warm.height = detector.size;
    warm.getContext('2d').fillRect(0, 0, detector.size, detector.size);
    try {
      await detector.analyzeImage(warm, { tile: detector.size });
    } catch { /* opcional */ }
    state.detector = detector;
    state.modelReady = true;
    state.modelMeta = detector.meta;
    el.modelInput.textContent = `${detector.size} × ${detector.size} px`;
    el.modelChip.textContent = 'listo';
    el.modelChip.className = 'chip tiny ok';
    if (detector.meta.params) el.modelParams.textContent = `${(detector.meta.params / 1e6).toFixed(2)} M`;
    setModelStatus('ready', `Modelo listo · ${detector.size}px`);
    el.btnCapture.disabled = state.mode !== 'live';
    announce('Modelo de detección cargado. La cámara está lista para usarse.');
  } catch (err) {
    console.error(err);
    state.modelError = err;
    setModelStatus('error', 'Modelo no disponible');
    el.modelChip.textContent = 'error';
    el.modelChip.className = 'chip tiny fail';
    toast(
      'No se pudo cargar el modelo de detección. Verifica que existan inspeccion/modelos/nexo-solder.onnx y los archivos vendor/.',
      'error',
      7000,
    );
  }
}

function setModelStatus(kind, text) {
  el.modelStatusText.textContent = text;
  el.modelStatusDot.className = `dot ${kind}`;
}

/* ─────────────────────────── cámara ─────────────────────────── */

async function listCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    state.videoDevices = devices.filter((d) => d.kind === 'videoinput');
    el.cameraSelect.innerHTML = state.videoDevices
      .map(
        (d, i) =>
          `<option value="${d.deviceId}">${d.label || `Cámara ${i + 1}`}</option>`,
      )
      .join('');
    el.cameraSelect.hidden = state.videoDevices.length < 2;
  } catch (e) {
    console.warn(e);
  }
}

async function startCamera(deviceId) {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast('Este navegador no permite acceder a la cámara. Usa “Subir foto” o una muestra.', 'error', 6000);
    return;
  }
  if (state.stream) stopCamera();
  const video = el.video;
  const constraints = {
    audio: false,
    video: deviceId
      ? { deviceId: { exact: deviceId }, width: { ideal: 1920 }, height: { ideal: 1080 } }
      : {
          facingMode: { ideal: 'environment' },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
        },
  };
  try {
    state.stream = await navigator.mediaDevices.getUserMedia(constraints);
  } catch (err) {
    console.error(err);
    const msg =
      err.name === 'NotAllowedError'
        ? 'Permiso de cámara denegado. Habilítalo en el navegador para inspeccionar en vivo.'
        : err.name === 'NotFoundError'
          ? 'No se encontró ninguna cámara conectada.'
          : 'No se pudo abrir la cámara. Prueba con “Subir foto”.';
    toast(msg, 'error', 6500);
    return;
  }
  video.srcObject = state.stream;
  video.hidden = false;
  el.captured.hidden = true;
  el.stagePlaceholder.hidden = true;
  try {
    await video.play();
  } catch { /* algunos navegadores requieren gesto */ }
  await new Promise((resolve) => {
    if (video.videoWidth) return resolve();
    video.onloadedmetadata = () => resolve();
  });
  state.mode = 'live';
  state.capture = null;
  state.rawDetections = [];
  state.detections = [];
  state.prevLive = [];
  setStageAspect(video.videoWidth, video.videoHeight);
  el.hud.hidden = false;
  el.hudLive.hidden = false;
  el.stageToolbar.hidden = false;
  el.btnStopCam.hidden = false;
  el.btnCapture.hidden = false;
  el.btnNewCapture.hidden = true;
  el.btnCapture.disabled = !state.modelReady;
  el.btnTorch.hidden = !state.torchAvailable;
  await listCameras();
  try {
    const track = state.stream.getVideoTracks()[0];
    updateTorchAvailability(track);
    const id = track.getSettings?.().deviceId;
    if (id) el.cameraSelect.value = id;
  } catch { /* ignore */ }
  refreshResults();
  liveLoop();
  announce('Cámara activa. Analizando la soldadura en vivo.');
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
  }
  el.video.srcObject = null;
  el.video.hidden = true;
  if (el.liveCanvas) el.liveCanvas.hidden = true;
  el.hud.hidden = true;
  el.stageToolbar.hidden = true;
  if (!state.capture) {
    state.mode = 'idle';
    el.stagePlaceholder.hidden = false;
    el.btnCapture.disabled = true;
  }
}

function updateTorchAvailability(track) {
  try {
    const caps = track.getCapabilities?.();
    state.torchAvailable = !!(caps && 'torch' in caps);
    el.btnTorch.hidden = !state.torchAvailable;
  } catch {
    state.torchAvailable = false;
    el.btnTorch.hidden = true;
  }
}

async function toggleTorch() {
  if (!state.stream) return;
  const track = state.stream.getVideoTracks()[0];
  try {
    state.torchOn = !state.torchOn;
    await track.applyConstraints({ advanced: [{ torch: state.torchOn }] });
    el.btnTorch.classList.toggle('active', state.torchOn);
  } catch (e) {
    toast('La linterna no está disponible en esta cámara.', 'error');
    state.torchOn = false;
  }
}

/**
 * Ventana de análisis en vivo, en píxeles nativos de la cámara.
 * zoom = 1 -> 384 px (escala de entrenamiento); >1 amplía; <1 abre el plano.
 */
function liveRegion() {
  const vw = el.video.videoWidth;
  const vh = el.video.videoHeight;
  const S = state.detector?.size || 384;
  const zoom = Math.max(0.5, state.settings.zoom);
  const side = Math.round(clamp(S / zoom, 160, Math.min(vw, vh)));
  return {
    x: Math.round((vw - side) / 2),
    y: Math.round((vh - side) / 2),
    w: side,
    h: side,
    size: side,
  };
}

async function liveLoop() {
  if (state.mode !== 'live') return;
  if (document.hidden) {
    setTimeout(liveLoop, 400);
    return;
  }
  if (!state.detector || state.busy || el.video.readyState < 2) {
    setTimeout(liveLoop, 120);
    return;
  }
  state.busy = true;
  try {
    const region = liveRegion();
    renderLiveCanvas(region);
    const res = await state.detector.analyzeRegion(el.video, region);
    const visible = {
      ok: state.settings.showOk && !state.settings.onlyDefects,
      warn: state.settings.warnOn,
      fail: state.settings.failOn,
    };
    const filt = filterDetections(res.detections, {
      conf: state.settings.conf,
      minSize: 4,
      visible,
    });
    state.liveDets = stabilize(filt);
    state.inference.ms = res.ms;
    trackInference();
    draw();
  } catch (err) {
    console.error(err);
    state.busy = false;
    return;
  }
  state.busy = false;
  setTimeout(liveLoop, 30);
}

function stabilize(dets) {
  const prev = state.prevLive;
  const used = new Set();
  const next = [];
  for (const d of dets) {
    let best = 0;
    let bi = -1;
    prev.forEach((p, i) => {
      if (used.has(i) || p.className !== d.className) return;
      const v = iou(d, p);
      if (v > best && v > 0.25) {
        best = v;
        bi = i;
      }
    });
    if (bi >= 0) {
      used.add(bi);
      const p = prev[bi];
      next.push({
        ...d,
        x: p.x * 0.45 + d.x * 0.55,
        y: p.y * 0.45 + d.y * 0.55,
        w: p.w * 0.45 + d.w * 0.55,
        h: p.h * 0.45 + d.h * 0.55,
        hits: (p.hits || 1) + 1,
      });
    } else {
      next.push({ ...d, hits: 1 });
    }
  }
  state.prevLive = next;
  return next.filter((d) => d.hits >= 2);
}

function trackInference() {
  const now = performance.now();
  if (!state.inferWindowStart) state.inferWindowStart = now;
  state.inferCount++;
  const dt = now - state.inferWindowStart;
  if (dt > 1000) {
    state.fps = (state.inferCount * 1000) / dt;
    state.inferCount = 0;
    state.inferWindowStart = now;
  }
}

/* ─────────────────────────── dibujo ─────────────────────────── */

function setStageAspect(w, h) {
  if (!w || !h) return;
  const stage = el.stage;
  const maxRatio = 2.2;
  const ratio = clamp(w / h, 0.6, maxRatio);
  stage.style.aspectRatio = `${ratio}`;
}

/**
 * Vista en vivo: la ventana analizada se dibuja ampliada en #liveCanvas y las
 * cajas se proyectan con la misma transformación, así que vemos exactamente lo
 * que evalúa el modelo.
 */
function liveViewRect(region) {
  const W = el.stage.clientWidth;
  const H = el.stage.clientHeight;
  const s = Math.min(W / region.w, H / region.h);
  return { x: (W - region.w * s) / 2, y: (H - region.h * s) / 2, w: region.w * s, h: region.h * s, s };
}

function renderLiveCanvas(region) {
  const rect = liveViewRect(region);
  const canvas = el.liveCanvas;
  if (!canvas) return;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const pw = Math.max(1, Math.round(rect.w * dpr));
  const ph = Math.max(1, Math.round(rect.h * dpr));
  if (canvas.width !== pw || canvas.height !== ph) {
    canvas.width = pw;
    canvas.height = ph;
  }
  canvas.style.width = `${rect.w}px`;
  canvas.style.height = `${rect.h}px`;
  canvas.style.left = `${rect.x}px`;
  canvas.style.top = `${rect.y}px`;
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(el.video, region.x, region.y, region.w, region.h, 0, 0, pw, ph);
  canvas.hidden = false;
}

function mediaRect() {
  const stage = el.stage;
  const W = stage.clientWidth;
  const H = stage.clientHeight;
  if (state.mode === 'live' && el.video.videoWidth) {
    return liveViewRect(liveRegion());
  }
  const sw = state.sourceSize.w;
  const sh = state.sourceSize.h;
  if (!sw || !sh) return { x: 0, y: 0, w: W, h: H };
  const scale = Math.min(W / sw, H / sh);
  const w = sw * scale;
  const h = sh * scale;
  return { x: (W - w) / 2, y: (H - h) / 2, w, h };
}

function draw() {
  const canvas = el.overlay;
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const W = el.stage.clientWidth;
  const H = el.stage.clientHeight;
  if (canvas.width !== Math.round(W * dpr) || canvas.height !== Math.round(H * dpr)) {
    canvas.width = Math.round(W * dpr);
    canvas.height = Math.round(H * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  if (!state.sourceSize.w) return;
  const rect = mediaRect();
  const isLive = state.mode === 'live';
  const region = isLive ? liveRegion() : null;
  const srcW = isLive ? region.w : state.sourceSize.w;
  const srcH = isLive ? region.h : state.sourceSize.h;
  const offX = isLive ? region.x : 0;
  const offY = isLive ? region.y : 0;
  const sx = rect.w / srcW;
  const sy = rect.h / srcH;

  const dets = isLive ? state.liveDets.filter((d) => d.hits >= 2) : state.detections;

  // marco del visor en vivo
  if (isLive) {
    ctx.save();
    ctx.strokeStyle = 'rgba(111,227,182,0.35)';
    ctx.lineWidth = 1;
    ctx.strokeRect(rect.x + 0.5, rect.y + 0.5, rect.w - 1, rect.h - 1);
    const c = 20;
    ctx.lineWidth = 2.5;
    ctx.strokeStyle = 'rgba(111,227,182,0.9)';
    const corner = (x, y, dx, dy) => {
      ctx.beginPath();
      ctx.moveTo(x + dx * c, y);
      ctx.lineTo(x, y);
      ctx.lineTo(x, y + dy * c);
      ctx.stroke();
    };
    corner(rect.x, rect.y, 1, 1);
    corner(rect.x + rect.w, rect.y, -1, 1);
    corner(rect.x, rect.y + rect.h, 1, -1);
    corner(rect.x + rect.w, rect.y + rect.h, -1, -1);
    ctx.restore();
  }

  // cajas
  const showLabels = state.settings.showLabels;
  const sorted = dets.slice().sort((a, b) => (a.severity === 'fail' ? -1 : 1) - (b.severity === 'fail' ? -1 : 1));
  for (const d of sorted) {
    const color = SEVERITY_COLORS[d.severity] || '#888';
    const x = rect.x + (d.x - offX) * sx;
    const y = rect.y + (d.y - offY) * sy;
    const w = d.w * sx;
    const h = d.h * sy;
    if (w < 3 || h < 3) continue;
    ctx.lineWidth = d.severity === 'fail' ? 2.2 : 1.7;
    ctx.strokeStyle = color;
    ctx.shadowColor = hexToRgba(color, 0.55);
    ctx.shadowBlur = d.severity === 'fail' ? 10 : 5;
    ctx.fillStyle = hexToRgba(color, d.severity === 'ok' ? 0.07 : 0.12);
    ctx.beginPath();
    const r = Math.min(4, w / 4, h / 4);
    ctx.roundRect(x, y, w, h, r);
    ctx.fill();
    ctx.stroke();
    ctx.shadowBlur = 0;

    if (showLabels && w >= 22) {
      const label = `${(CLASS_INFO[d.className]?.label || d.className)} · ${Math.round(d.score * 100)}%`;
      ctx.font = '600 10.5px Inter, system-ui, sans-serif';
      const tw = ctx.measureText(label).width + 10;
      const th = 15;
      let lx = x;
      let ly = y - th - 3;
      if (ly < rect.y + 2) ly = y + h + 3;
      if (lx + tw > rect.x + rect.w) lx = rect.x + rect.w - tw;
      if (lx < rect.x) lx = rect.x;
      ctx.fillStyle = hexToRgba(color, 0.92);
      ctx.beginPath();
      ctx.roundRect(lx, ly, tw, th, 3);
      ctx.fill();
      ctx.fillStyle = d.severity === 'fail' ? '#fff' : '#08120f';
      ctx.textBaseline = 'middle';
      ctx.fillText(label, lx + 5, ly + th / 2 + 0.5);
    }
  }

  // HUD de conteo
  if (el.hudCount) {
    el.hudCount.textContent = dets.length;
    el.hudMs.textContent = `${state.inference.ms || 0} ms`;
    el.hudFps.textContent = isLive ? state.fps.toFixed(1) : '—';
  }
}

/* ─────────────────────────── captura / análisis ─────────────────────────── */

function captureFrame() {
  const video = el.video;
  if (!video.videoWidth) return null;
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height);
  return canvas;
}

async function onCapture() {
  if (!state.modelReady || !state.detector) {
    toast('El modelo todavía no está listo.', 'error');
    return;
  }
  if (state.mode !== 'live' || !state.stream) {
    toast('Activa la cámara para capturar, o carga una foto.', 'error');
    return;
  }
  const canvas = captureFrame();
  if (!canvas) {
    toast('No se pudo leer el fotograma de la cámara.', 'error');
    return;
  }
  setCaptureFromCanvas(canvas, 'Cámara');
  await analyzeCurrentCapture();
}

function setCaptureFromCanvas(canvas, sourceLabel) {
  state.capture = {
    canvas,
    url: canvas.toDataURL('image/jpeg', 0.9),
    width: canvas.width,
    height: canvas.height,
  };
  el.captured.src = state.capture.url;
  el.captured.hidden = false;
  el.video.hidden = true;
  if (el.liveCanvas) el.liveCanvas.hidden = true;
  el.stagePlaceholder.hidden = true;
  el.hud.hidden = false;
  el.hudLive.hidden = true;
  el.stageToolbar.hidden = true;
  el.btnNewCapture.hidden = !state.stream;
  state.mode = 'capture';
  state.sourceSize = { w: canvas.width, h: canvas.height };
  setStageAspect(canvas.width, canvas.height);
  state.inference.source = sourceLabel;
  state.folio = state.folio || folioFor();
}

async function analyzeCurrentCapture() {
  if (!state.capture || !state.detector) return;
  setProgress(0, 'Preparando el análisis…', true);
  el.btnPdf.disabled = true;
  el.btnJson.disabled = true;
  try {
    const { canvas } = state.capture;
    const maxDim = 2200;
    let work = canvas;
    let factor = 1;
    if (Math.max(canvas.width, canvas.height) > maxDim) {
      factor = maxDim / Math.max(canvas.width, canvas.height);
      work = document.createElement('canvas');
      work.width = Math.round(canvas.width * factor);
      work.height = Math.round(canvas.height * factor);
      work.getContext('2d').drawImage(canvas, 0, 0, work.width, work.height);
    }
    const tile = chooseTileSize(work.width, work.height, 9);
    const res = await state.detector.analyzeImage(work, {
      tile,
      overlap: 0.22,
      onProgress: (f, ms) => {
        setProgress(f * 0.96, `${Math.round(f * 100)} % · ${ms} ms`, true);
      },
    });
    setProgress(0.99, 'Consolidando detecciones…', true);
    const dets = res.detections.map((d) => ({
      ...d,
      x: d.x / factor,
      y: d.y / factor,
      w: d.w / factor,
      h: d.h / factor,
      cx: (d.x + d.w / 2) / factor,
      cy: (d.y + d.h / 2) / factor,
    }));
    state.rawDetections = dets;
    state.inference = {
      ms: res.ms,
      tiles: res.tiles,
      tileSize: res.tileSize,
      source: state.inference.source || 'Fotografía',
    };
    refreshResults();
    setProgress(1, 'Listo', true);
    setTimeout(() => setProgress(0, '', false), 240);
    announce(`Análisis completado: ${state.detections.length} uniones detectadas.`);
    if (!dets.length) {
      toast('No se detectaron uniones. Acércate más a la placa o mejora la iluminación.', 'error', 5200);
    } else {
      toast(`Análisis completado: ${state.detections.length} uniones · ${res.ms} ms.`);
    }
  } catch (err) {
    console.error(err);
    setProgress(0, '', false);
    toast('Ocurrió un error durante el análisis de la imagen.', 'error', 6000);
  }
  updateActionState();
}

/**
 * Las teselas se analizan a escala nativa: el modelo se entrenó con recortes de
 * 512 px y analiza a 384 px, así que teselas mayores perderían detalle.
 */
function chooseTileSize(w, h, maxTiles) {
  const S = state.detector?.size || 384;
  return Math.max(96, Math.min(S, Math.max(w, h)));
}

function setProgress(frac, text, visible) {
  el.progress.hidden = !visible;
  if (!visible) return;
  el.progressBar.style.width = `${Math.round(clamp(frac, 0, 1) * 100)}%`;
  el.progressText.textContent = text || '';
  el.progressTitle.textContent = 'Analizando la soldadura…';
}

/* ─────────────────────────── resultados ─────────────────────────── */

function refreshResults() {
  const visible = {
    ok: state.settings.showOk && !state.settings.onlyDefects,
    warn: state.settings.warnOn,
    fail: state.settings.failOn,
  };
  state.detections = filterDetections(state.rawDetections, {
    conf: state.settings.conf,
    minSize: 3,
    visible,
  }).sort((a, b) => b.score - a.score);
  renderSummary();
  renderTable();
  renderRecommendations();
  updateActionState();
  draw();
}

function renderSummary() {
  const s = summaryOf(state.detections);
  el.countOk.textContent = s.ok;
  el.countWarn.textContent = s.warn;
  el.countFail.textContent = s.fail;
  el.totalJoints.textContent = s.total;
  el.gaugePct.textContent = `${s.total ? s.pctOk : 0} %`;
  const color = s.total ? SEVERITY_COLORS[s.verdict] : '#2a4a52';
  el.gauge.style.setProperty('--pct', s.total ? s.pctOk : 0);
  el.gauge.style.setProperty('--ring', color);
  const chip = el.verdictChip;
  chip.className = 'chip';
  if (!s.total) {
    chip.textContent = 'Sin datos';
    el.verdictNote.textContent = 'Captura una imagen para obtener el veredicto.';
  } else if (s.verdict === 'fail') {
    chip.textContent = 'No conforme';
    chip.classList.add('fail');
    el.verdictNote.textContent = `${s.fail} unión(es) requieren retrabajo antes de continuar. Revisa la pestaña de recomendaciones.`;
  } else if (s.verdict === 'warn') {
    chip.textContent = 'Con observaciones';
    chip.classList.add('warn');
    el.verdictNote.textContent = `${s.warn} unión(es) cumplen de forma aceptable pero muestran indicios de proceso. Vigila la tendencia del lote.`;
  } else {
    chip.textContent = 'Conforme';
    chip.classList.add('ok');
    el.verdictNote.textContent = 'Todas las uniones detectadas cumplen el criterio visual con el umbral actual.';
  }
  el.tabCount.textContent = s.total;
}

function renderTable() {
  const body = el.detectionsBody;
  const dets = state.detections;
  if (!dets.length) {
    body.innerHTML =
      '<tr class="empty-row"><td colspan="6">Aún no hay detecciones. Captura una imagen para comenzar.</td></tr>';
    return;
  }
  const rows = dets.slice(0, 300).map((d, i) => {
    const info = CLASS_INFO[d.className] || {};
    const color = SEVERITY_COLORS[d.severity];
    return `<tr>
      <td class="idx">${i + 1}</td>
      <td><span class="sev"><span class="dot" style="background:${color}"></span>${SEVERITY_LABELS[d.severity]}</span></td>
      <td>${info.label || d.className}</td>
      <td><span class="conf"><span class="conf-bar"><i style="width:${Math.round(d.score * 100)}%;background:${color}"></i></span>${Math.round(d.score * 100)} %</span></td>
      <td class="pos">${Math.round(d.cx)}, ${Math.round(d.cy)}</td>
      <td class="ref">${(info.references?.[0] || 'IPC-A-610G').replace('IPC-A-610G ', '')}</td>
    </tr>`;
  });
  body.innerHTML = rows.join('');
}

function renderRecommendations() {
  const container = el.recommendationsList;
  if (!state.rawDetections.length) {
    container.innerHTML = '<p class="muted">Las recomendaciones se generan a partir de los hallazgos de la inspección.</p>';
    return;
  }
  const { items, general } = buildRecommendations(state.detections, {
    ipcClass: state.settings.ipcClass,
  });
  const cards = items.map((item) => {
    const color = SEVERITY_COLORS[item.severity];
    return `<article class="rec sev-${item.severity}">
      <div class="rec-head">
        <span class="dot" style="background:${color}"></span>
        <h4>${item.title}</h4>
        <span class="rec-count">${item.count} unión(es) · ${SEVERITY_LABELS[item.severity]}</span>
      </div>
      <p>${item.action}</p>
      <ul>${item.causes.slice(0, 3).map((c) => `<li>${c}</li>`).join('')}</ul>
      <div class="refs">${item.references.join(' · ')}</div>
    </article>`;
  });
  const generalCards = general
    .map((g) => `<article class="rec general"><p>${g}</p></article>`)
    .join('');
  const advice = PROCESS_ADVICE.map(
    (a) => `<article class="rec general"><div class="rec-head"><h4>${a.title}</h4></div><p>${a.text}</p></article>`,
  ).join('');
  container.innerHTML = `${cards || '<p class="muted">No hay hallazgos por encima del umbral.</p>'}
    <h3 class="section-sub">Acciones generales</h3>${generalCards}
    <h3 class="section-sub">Buenas prácticas de proceso</h3>${advice}`;
}

function updateActionState() {
  const live = state.mode === 'live';
  el.btnCapture.disabled = !(live && state.modelReady);
  el.btnCapture.hidden = !live;
  el.btnNewCapture.hidden = live || !state.stream;
  el.btnPdf.disabled = !state.capture || !state.modelReady;
  el.btnJson.disabled = !state.capture;
  el.btnValidate.hidden = !state.groundTruth;
}

/* ─────────────────────────── entrada de imágenes ─────────────────────────── */

async function loadImageFile(file) {
  if (!file) return;
  if (!file.type.startsWith('image/')) {
    toast('El archivo debe ser una imagen (JPG, PNG o WebP).', 'error');
    return;
  }
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.onload = async () => {
    URL.revokeObjectURL(url);
    stopCamera();
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext('2d', { willReadFrequently: false }).drawImage(img, 0, 0);
    state.groundTruth = null;
    state.sample = null;
    setCaptureFromCanvas(canvas, `Archivo · ${file.name}`);
    await analyzeCurrentCapture();
  };
  img.onerror = () => toast('No se pudo leer la imagen.', 'error');
  img.src = url;
}

async function loadSample(sample) {
  try {
    toast(`Cargando ${sample.label}…`);
    const url = new URL(`muestras/${sample.id}.jpg`, document.baseURI).href;
    const img = new Image();
    img.crossOrigin = 'anonymous';
    const loaded = new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
    });
    img.src = url;
    await loaded;
    stopCamera();
    const canvas = document.createElement('canvas');
    canvas.width = img.naturalWidth;
    canvas.height = img.naturalHeight;
    canvas.getContext('2d').drawImage(img, 0, 0);
    state.sample = sample;
    state.groundTruth = null;
    try {
      const gtUrl = new URL(`muestras/gt/${sample.id}.json`, document.baseURI).href;
      const res = await fetch(gtUrl);
      if (res.ok) state.groundTruth = await res.json();
    } catch { /* sin etiquetas */ }
    setCaptureFromCanvas(canvas, `Muestra · ${sample.label}`);
    state.folio = folioFor();
    await analyzeCurrentCapture();
  } catch (err) {
    console.error(err);
    toast('No se encontró la imagen de muestra. Genera las muestras con training/make_samples.py.', 'error', 6000);
  }
}

function validateAgainstGroundTruth() {
  if (!state.groundTruth) return;
  const gt = state.groundTruth.detections || state.groundTruth;
  const metrics = evaluateAgainstGroundTruth(state.detections, gt, 0.5);
  const pct = (v) => `${(v * 100).toFixed(1)} %`;
  toast(
    `Precisión ${pct(metrics.precision)} · Exhaustividad ${pct(metrics.recall)} · F1 ${pct(metrics.f1)} · Clase correcta ${pct(metrics.classAccuracy)} (${metrics.tp} aciertos, ${metrics.fp} falsos positivos, ${metrics.fn} omitidas).`,
    'info',
    9000,
  );
}

/* ─────────────────────────── informe ─────────────────────────── */

function renderAnnotatedCanvas(maxWidth = 1600, withLabels = true) {
  if (!state.capture) return null;
  const src = state.capture.canvas;
  const scale = Math.min(1, maxWidth / src.width);
  const canvas = document.createElement('canvas', { willReadFrequently: false });
  canvas.width = Math.round(src.width * scale);
  canvas.height = Math.round(src.height * scale);
  const ctx = canvas.getContext('2d', { alpha: false });
  if (!ctx) return canvas;
  ctx.drawImage(src, 0, 0, canvas.width, canvas.height);
  const dets = state.detections;
  ctx.textBaseline = 'middle';
  for (const d of dets) {
    const color = SEVERITY_COLORS[d.severity] || '#888';
    const x = d.x * scale;
    const y = d.y * scale;
    const w = d.w * scale;
    const h = d.h * scale;
    ctx.lineWidth = Math.max(1.4, Math.min(4, w * 0.04));
    ctx.strokeStyle = color;
    ctx.fillStyle = hexToRgba(color, d.severity === 'ok' ? 0.06 : 0.12);
    ctx.beginPath();
    ctx.roundRect(x, y, w, h, Math.min(5, w / 4, h / 4));
    ctx.fill();
    ctx.stroke();
    if (withLabels && w > 26) {
      const label = `${(CLASS_INFO[d.className]?.label || d.className)} ${Math.round(d.score * 100)}%`;
      ctx.font = `600 ${Math.max(9, Math.min(13, w * 0.22))}px Inter, system-ui, sans-serif`;
      const tw = ctx.measureText(label).width + 8;
      const th = Math.max(13, Math.min(18, h * 0.7));
      let lx = x;
      let ly = y - th - 2;
      if (ly < 0) ly = y + h + 2;
      if (lx + tw > canvas.width) lx = canvas.width - tw;
      ctx.fillStyle = hexToRgba(color, 0.92);
      ctx.beginPath();
      ctx.roundRect(lx, ly, tw, th, 3);
      ctx.fill();
      ctx.fillStyle = d.severity === 'fail' ? '#fff' : '#08120f';
      ctx.fillText(label, lx + 4, ly + th / 2 + 0.5);
    }
  }
  return canvas;
}

function reportMeta() {
  const source = state.inference.source || (state.sample ? state.sample.label : 'Cámara');
  const device = el.cameraSelect.selectedOptions?.[0]?.textContent || '—';
  return {
    folio: state.folio || folioFor(),
    date: new Date(),
    boardId: el.boardId.value.trim(),
    operator: el.operator.value.trim(),
    ipcClass: Number(el.ipcClass.value),
    notes: el.notes.value.trim(),
    source,
    device: state.mode === 'live' || state.inference.source === 'Cámara' ? device : 'Archivo local',
    resolution: `${state.capture?.width || 0} × ${state.capture?.height || 0} px`,
    conf: state.settings.conf,
    inferenceMs: state.inference.ms,
    tiles: state.inference.tiles,
  };
}

function buildPdf() {
  const jsPDF = window.jspdf?.jsPDF;
  if (!jsPDF) {
    toast('No se cargó el generador de PDF.', 'error');
    return null;
  }
  const canvas = renderAnnotatedCanvas(1600, true);
  const imageDataUrl = canvas.toDataURL('image/jpeg', 0.86);
  const summary = summaryOf(state.detections);
  const recommendations = buildRecommendations(state.detections, {
    ipcClass: Number(el.ipcClass.value),
  });
  const meta = reportMeta();
  state.folio = meta.folio;
  return buildInspectionReport({
    jsPDF,
    imageDataUrl,
    summary,
    detections: state.detections,
    recommendations,
    meta,
    model: {
      name: 'Nexo SolderNet (YOLO anchor-free)',
      size: state.detector?.size,
      params: state.detector?.meta?.params,
    },
  });
}

async function exportPdf() {
  if (!state.capture) {
    toast('Captura o carga una imagen antes de generar el informe.', 'error');
    return;
  }
  try {
    const doc = buildPdf();
    if (!doc) return;
    const folio = state.folio || folioFor();
    doc.save(`inspeccion-soldadura-${folio}.pdf`);
    saveHistory();
    toast('Informe PDF generado.');
  } catch (err) {
    console.error(err);
    toast('No se pudo generar el informe PDF.', 'error', 6000);
  }
}

function exportJson() {
  if (!state.capture) return;
  const payload = {
    folio: state.folio,
    fecha: new Date().toISOString(),
    placa: el.boardId.value.trim(),
    operador: el.operator.value.trim(),
    claseIPC: Number(el.ipcClass.value),
    notas: el.notes.value.trim(),
    modelo: {
      nombre: 'Nexo SolderNet',
      clases: state.detector?.classes,
      entrada: state.detector?.size,
    },
    resolucion: [state.capture.width, state.capture.height],
    inferencia: state.inference,
    resumen: summaryOf(state.detections),
    detecciones: state.detections.map((d) => ({
      clase: d.className,
      severidad: d.severity,
      confianza: Number(d.score.toFixed(4)),
      x: Math.round(d.x),
      y: Math.round(d.y),
      ancho: Math.round(d.w),
      alto: Math.round(d.h),
    })),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = `inspeccion-${state.folio}.json`;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

/* ─────────────────────────── historial ─────────────────────────── */

function readHistory() {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveHistory() {
  try {
    const canvas = renderAnnotatedCanvas(1000, true);
    const thumb = canvas.toDataURL('image/jpeg', 0.72);
    const record = {
      id: state.folio,
      folio: state.folio,
      date: new Date().toISOString(),
      boardId: el.boardId.value.trim(),
      operator: el.operator.value.trim(),
      ipcClass: Number(el.ipcClass.value),
      notes: el.notes.value.trim(),
      summary: summaryOf(state.detections),
      detectionCount: state.detections.length,
      width: canvas.width,
      height: canvas.height,
      image: thumb,
      detections: state.detections.slice(0, 400),
      inference: state.inference,
      source: state.inference.source,
    };
    const list = readHistory().filter((r) => r.folio !== record.folio);
    list.unshift(record);
    localStorage.setItem(HISTORY_KEY, JSON.stringify(list.slice(0, 6)));
    renderHistory();
  } catch (err) {
    console.warn('No se pudo guardar el historial', err);
  }
}

function renderHistory() {
  const list = readHistory();
  if (!list.length) {
    el.historyList.innerHTML = '<p class="muted">Todavía no hay inspecciones guardadas en este navegador.</p>';
    return;
  }
  el.historyList.innerHTML = list
    .map((r) => {
      const d = new Date(r.date);
      const dateStr = d.toLocaleString('es-CO', { dateStyle: 'medium', timeStyle: 'short' });
      const s = r.summary || {};
      return `<article class="history-item" data-id="${r.folio}">
        <img src="${r.image}" alt="Miniatura de la inspección ${r.folio}" />
        <div>
          <h4>${r.boardId || 'PCB sin identificar'}</h4>
          <p>${dateStr} · ${r.folio}</p>
          <p><span class="dot ok"></span> ${s.ok || 0} · <span class="dot warn"></span> ${s.warn || 0} · <span class="dot fail"></span> ${s.fail || 0}</p>
        </div>
        <div class="history-actions">
          <button class="btn tiny ghost" data-action="open">Ver</button>
          <button class="btn tiny ghost" data-action="pdf">PDF</button>
        </div>
      </article>`;
    })
    .join('');
  el.historyList.querySelectorAll('.history-item').forEach((node) => {
    const id = node.dataset.id;
    node.querySelector('[data-action="open"]').addEventListener('click', () => openHistory(id));
    node.querySelector('[data-action="pdf"]').addEventListener('click', () => exportHistoryPdf(id));
  });
}

function loadHistoryRecord(id) {
  const record = readHistory().find((r) => r.folio === id);
  if (!record) return null;
  return record;
}

async function openHistory(id) {
  const r = loadHistoryRecord(id);
  if (!r) return;
  const img = new Image();
  const loaded = new Promise((res, rej) => {
    img.onload = res;
    img.onerror = rej;
  });
  img.src = r.image;
  await loaded;
  stopCamera();
  const canvas = document.createElement('canvas');
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  canvas.getContext('2d').drawImage(img, 0, 0);
  state.capture = {
    canvas,
    url: r.image,
    width: canvas.width,
    height: canvas.height,
  };
  el.captured.src = r.image;
  el.captured.hidden = false;
  el.video.hidden = true;
  if (el.liveCanvas) el.liveCanvas.hidden = true;
  el.stagePlaceholder.hidden = true;
  el.hud.hidden = false;
  el.hudLive.hidden = true;
  el.stageToolbar.hidden = true;
  el.btnNewCapture.hidden = !state.stream;
  state.mode = 'capture';
  state.sourceSize = { w: canvas.width, h: canvas.height };
  state.folio = r.folio;
  state.rawDetections = r.detections || [];
  state.inference = r.inference || { ms: 0, tiles: 1, source: r.source };
  el.boardId.value = r.boardId || '';
  el.operator.value = r.operator || '';
  el.notes.value = r.notes || '';
  el.ipcClass.value = String(r.ipcClass || 2);
  state.settings.ipcClass = Number(r.ipcClass || 2);
  setStageAspect(canvas.width, canvas.height);
  refreshResults();
  toast(`Inspección ${r.folio} cargada desde el historial.`);
}

function exportHistoryPdf(id) {
  const r = loadHistoryRecord(id);
  if (!r) return;
  const jsPDF = window.jspdf?.jsPDF;
  if (!jsPDF) return;
  const summary = r.summary || summaryOf(r.detections);
  const recommendations = buildRecommendations(r.detections, { ipcClass: r.ipcClass || 2 });
  const doc = buildInspectionReport({
    jsPDF,
    imageDataUrl: r.image,
    summary,
    detections: r.detections || [],
    recommendations,
    meta: {
      folio: r.folio,
      date: new Date(r.date),
      boardId: r.boardId,
      operator: r.operator,
      ipcClass: r.ipcClass,
      notes: r.notes,
      source: r.source,
      resolution: `${r.width} × ${r.height} px`,
      conf: 0.35,
      inferenceMs: r.inference?.ms,
      tiles: r.inference?.tiles,
    },
    model: { name: 'Nexo SolderNet' },
  });
  doc.save(`inspeccion-soldadura-${r.folio}.pdf`);
}

/* ─────────────────────────── UI general ─────────────────────────── */

function switchTab(name) {
  state.activeTab = name;
  $$('.tab').forEach((t) => {
    const on = t.dataset.tab === name;
    t.classList.toggle('active', on);
    t.setAttribute('aria-selected', String(on));
  });
  $$('.panel').forEach((p) => p.classList.toggle('active', p.id === `panel-${name}`));
}

function resetInspection() {
  state.rawDetections = [];
  state.detections = [];
  state.prevLive = [];
  state.liveDets = [];
  state.capture = null;
  state.folio = '';
  state.groundTruth = null;
  state.sample = null;
  state.inference = { ms: 0, tiles: 0, source: '' };
  state.sourceSize = { w: 0, h: 0 };
  el.captured.hidden = true;
  el.captured.removeAttribute('src');
  el.btnValidate.hidden = true;
  el.btnNewCapture.hidden = true;
  el.hud.hidden = true;
  state.mode = 'idle';
  if (state.stream) {
    startCamera(el.cameraSelect.value || undefined);
  } else {
    el.stagePlaceholder.hidden = false;
    el.btnCapture.disabled = true;
    setStageAspect(16, 9);
  }
  el.boardId.value = '';
  el.operator.value = '';
  el.notes.value = '';
  refreshResults();
  draw();
  announce('Inspección reiniciada.');
}

function openSampleMenu(anchor) {
  let menu = $('#sampleMenu');
  if (menu) {
    menu.remove();
    return;
  }
  menu = document.createElement('div');
  menu.id = 'sampleMenu';
  menu.className = 'card sample-menu';
  menu.innerHTML = `<strong>Muestras incluidas</strong>${SAMPLES.map(
    (s) => `<button class="btn ghost block" data-sample="${s.id}">${s.label}</button>`,
  ).join('')}<p class="fineprint">Imágenes sintéticas de PCB con defectos etiquetados para probar la app.</p>`;
  document.body.appendChild(menu);
  const rect = anchor.getBoundingClientRect();
  const top = Math.min(window.innerHeight - 240, rect.bottom + 8);
  menu.style.top = `${top}px`;
  menu.style.left = `${clamp(rect.left, 12, window.innerWidth - 300)}px`;
  menu.querySelectorAll('[data-sample]').forEach((b) => {
    b.addEventListener('click', () => {
      const s = SAMPLES.find((x) => x.id === b.dataset.sample);
      menu.remove();
      if (s) loadSample(s);
    });
  });
  setTimeout(() => {
    const close = (ev) => {
      if (!menu.contains(ev.target)) {
        menu.remove();
        document.removeEventListener('click', close);
      }
    };
    document.addEventListener('click', close);
  }, 0);
}

function wireEvents() {
  el.btnStartCam.addEventListener('click', () => startCamera(el.cameraSelect.value || undefined));
  el.btnStopCam?.addEventListener('click', () => {
    stopCamera();
    toast('Cámara detenida.');
  });
  el.btnCapture.addEventListener('click', onCapture);
  el.btnTorch.addEventListener('click', toggleTorch);
  [el.btnUpload, el.btnUpload2].forEach((b) => b.addEventListener('click', () => el.fileInput.click()));
  el.fileInput.addEventListener('change', (e) => {
    const f = e.target.files?.[0];
    loadImageFile(f);
    e.target.value = '';
  });
  [el.btnSamples, el.btnSample2].forEach((b) =>
    b.addEventListener('click', (ev) => {
      ev.stopPropagation();
      openSampleMenu(b);
    }),
  );
  el.cameraSelect.addEventListener('change', () => startCamera(el.cameraSelect.value));
  el.btnNewCapture.addEventListener('click', () => {
    state.mode = 'idle';
    el.captured.hidden = true;
    el.capture = null;
    state.rawDetections = [];
    state.detections = [];
    refreshResults();
    startCamera(el.cameraSelect.value || undefined);
  });
  el.btnValidate.addEventListener('click', validateAgainstGroundTruth);

  el.zoomRange.addEventListener('input', () => {
    state.settings.zoom = Number(el.zoomRange.value);
    el.zoomValue.textContent = `${state.settings.zoom.toFixed(1)}×`;
    draw();
    if (state.mode === 'live' && el.video.videoWidth) renderLiveCanvas(liveRegion());
  });
  el.confRange.addEventListener('input', () => {
    state.settings.conf = Number(el.confRange.value);
    el.confValue.textContent = `${Math.round(state.settings.conf * 100)} %`;
    refreshResults();
    persistSettings();
  });
  el.toggleOk.addEventListener('change', () => {
    state.settings.showOk = el.toggleOk.checked;
    refreshResults();
    persistSettings();
  });
  el.toggleWarn.addEventListener('change', () => {
    state.settings.warnOn = el.toggleWarn.checked;
    refreshResults();
    persistSettings();
  });
  el.toggleFail.addEventListener('change', () => {
    state.settings.failOn = el.toggleFail.checked;
    refreshResults();
    persistSettings();
  });
  el.showLabels.addEventListener('change', () => {
    state.settings.showLabels = el.showLabels.checked;
    draw();
    persistSettings();
  });
  el.onlyDefects.addEventListener('change', () => {
    state.settings.onlyDefects = el.onlyDefects.checked;
    refreshResults();
    persistSettings();
  });
  el.ipcClass.addEventListener('change', () => {
    state.settings.ipcClass = Number(el.ipcClass.value);
    renderRecommendations();
    persistSettings();
  });
  [el.boardId, el.operator, el.notes].forEach((input) =>
    input.addEventListener('input', () => {
      state.folio = state.folio || '';
    }),
  );

  el.btnPdf.addEventListener('click', exportPdf);
  el.btnJson.addEventListener('click', exportJson);
  el.btnReset.addEventListener('click', resetInspection);
  el.btnClearHistory.addEventListener('click', () => {
    localStorage.removeItem(HISTORY_KEY);
    renderHistory();
    toast('Historial borrado.');
  });
  $$('.tab').forEach((t) => t.addEventListener('click', () => switchTab(t.dataset.tab)));

  [el.btnHelp, el.btnAbout].forEach((b) =>
    b.addEventListener('click', () => {
      el.helpModal.hidden = false;
      el.btnCloseHelp.focus();
    }),
  );
  el.btnCloseHelp.addEventListener('click', () => {
    el.helpModal.hidden = true;
  });
  el.helpModal.addEventListener('click', (e) => {
    if (e.target === el.helpModal) el.helpModal.hidden = true;
  });

  window.addEventListener('resize', () => {
    draw();
    if (state.mode === 'live' && el.video.videoWidth) renderLiveCanvas(liveRegion());
  });
  new ResizeObserver(() => draw()).observe(el.stage);
  document.addEventListener('keydown', (e) => {
    if (e.target.matches('input, textarea, select')) return;
    if (e.key === 'Escape') el.helpModal.hidden = true;
    if (e.key === ' ' && state.mode === 'live') {
      e.preventDefault();
      onCapture();
    }
    if (e.key.toLowerCase() === 'p' && state.capture) exportPdf();
    if (e.key.toLowerCase() === 'r') resetInspection();
    if (e.key === '?') el.helpModal.hidden = false;
  });
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden && state.mode === 'live') liveLoop();
  });
}

function applyStoredSettings() {
  const s = state.settings;
  el.confRange.value = String(s.conf);
  el.confValue.textContent = `${Math.round(s.conf * 100)} %`;
  el.zoomRange.value = String(s.zoom);
  el.zoomValue.textContent = `${Number(s.zoom).toFixed(1)}×`;
  el.toggleOk.checked = s.showOk;
  el.toggleWarn.checked = s.warnOn;
  el.toggleFail.checked = s.failOn;
  el.showLabels.checked = s.showLabels;
  el.onlyDefects.checked = s.onlyDefects;
  el.ipcClass.value = String(s.ipcClass);
}

function cacheEls() {
  Object.assign(el, {
    stage: $('#stage'),
    video: $('#video'),
    liveCanvas: $('#liveCanvas'),
    captured: $('#captured'),
    overlay: $('#overlay'),
    stagePlaceholder: $('#stagePlaceholder'),
    hud: $('#hud'),
    hudFps: $('#hudFps'),
    hudMs: $('#hudMs'),
    hudCount: $('#hudCount'),
    hudLive: $('#hudLive'),
    stageToolbar: $('#stageToolbar'),
    cameraSelect: $('#cameraSelect'),
    zoomRange: $('#zoomRange'),
    zoomValue: $('#zoomValue'),
    btnStartCam: $('#btnStartCam'),
    btnStopCam: $('#btnStopCam'),
    btnCapture: $('#btnCapture'),
    btnTorch: $('#btnTorch'),
    btnUpload: $('#btnUpload'),
    btnUpload2: $('#btnUpload2'),
    fileInput: $('#fileInput'),
    btnSamples: $('#btnSamples'),
    btnSample2: $('#btnSample2'),
    btnNewCapture: $('#btnNewCapture'),
    btnValidate: $('#btnValidate'),
    progress: $('#progress'),
    progressBar: $('#progressBar'),
    progressText: $('#progressText'),
    progressTitle: $('#progressTitle'),
    modelStatus: $('#modelStatus'),
    modelStatusDot: $('#modelStatusDot'),
    modelStatusText: $('#modelStatusText'),
    modelChip: $('#modelChip'),
    modelInput: $('#modelInput'),
    modelParams: $('#modelParams'),
    verdictChip: $('#verdictChip'),
    verdictNote: $('#verdictNote'),
    gauge: $('#gauge'),
    gaugePct: $('#gaugePct'),
    countOk: $('#countOk'),
    countWarn: $('#countWarn'),
    countFail: $('#countFail'),
    totalJoints: $('#totalJoints'),
    tabCount: $('#tabCount'),
    confRange: $('#confRange'),
    confValue: $('#confValue'),
    toggleOk: $('#toggleOk'),
    toggleWarn: $('#toggleWarn'),
    toggleFail: $('#toggleFail'),
    showLabels: $('#showLabels'),
    onlyDefects: $('#onlyDefects'),
    ipcClass: $('#ipcClass'),
    boardId: $('#boardId'),
    operator: $('#operator'),
    notes: $('#notes'),
    btnPdf: $('#btnPdf'),
    btnJson: $('#btnJson'),
    btnReset: $('#btnReset'),
    detectionsBody: $('#detectionsBody'),
    recommendationsList: $('#recommendationsList'),
    historyList: $('#historyList'),
    btnClearHistory: $('#btnClearHistory'),
    helpModal: $('#helpModal'),
    btnHelp: $('#btnHelp'),
    btnAbout: $('#btnAbout'),
    btnCloseHelp: $('#btnCloseHelp'),
    toast: $('#toast'),
    announcer: $('#announcer'),
  });
}

async function init() {
  cacheEls();
  loadSettings();
  applyStoredSettings();
  wireEvents();
  renderHistory();
  setStageAspect(16, 9);
  state.folio = folioFor();
  await initModel();
  if (navigator.mediaDevices?.enumerateDevices) {
    navigator.mediaDevices.enumerateDevices().then(() => listCameras()).catch(() => {});
  }
  draw();
}

document.addEventListener('DOMContentLoaded', init);

// Hook de depuración: permite inspeccionar y probar la app desde la consola.
window.nexoApp = {
  state,
  refreshResults,
  draw,
  summaryOf,
  renderTable,
  renderRecommendations,
  buildPdf,
  reportMeta,
  setCaptureFromCanvas,
  renderAnnotatedCanvas,
};
