/**
 * Detector de uniones de soldadura en el navegador.
 *
 * Envuelve ONNX Runtime Web (WASM) y el modelo YOLO-style exportado desde
 * training/train.py.  Todo el pre/post-proceso se hace en JS:
 *   - región cuadrada de la fuente -> tensor NCHW normalizado
 *   - decodificación anchor-free con strides 8/16/32
 *   - supresión de no máximos (agnóstica) y mapeo a coordenadas de la imagen
 */

const SEVERITY_OF = (name) => (name === 'ok' ? 'ok' : name.startsWith('warn') ? 'warn' : 'fail');

export class SolderDetector {
  constructor(session, meta, ort) {
    this.session = session;
    this.meta = meta;
    this.ort = ort;
    this.size = meta.size || 384;
    this.inputName = meta.inputs?.[0] || 'images';
    this.outputNames = meta.outputs || ['out0', 'out1', 'out2'];
    this.strides = meta.strides || [8, 16, 32];
    this.classes = meta.classes || ['ok', 'warn_dull', 'warn_coverage', 'fail_cold',
      'fail_insufficient', 'fail_nonwet', 'fail_void', 'fail_bridge', 'fail_missing'];
    this.mean = meta.mean || [0.485, 0.456, 0.406];
    this.std = meta.std || [0.229, 0.224, 0.225];
    this._canvas = document.createElement('canvas');
    this._canvas.width = this.size;
    this._canvas.height = this.size;
    this._ctx = this._canvas.getContext('2d', { willReadFrequently: true });
  }

  /**
   * Carga el modelo ONNX y su side-car JSON.
   * @param {{modelUrl:string, metaUrl:string, wasmPaths?:string}} opts
   */
  static async load({ modelUrl, metaUrl, wasmPaths, threads }) {
    const ort = window.ort;
    if (!ort) throw new Error('ONNX Runtime no está disponible (¿falta vendor/ort.wasm.min.js?)');
    if (wasmPaths && ort.env?.wasm) ort.env.wasm.wasmPaths = wasmPaths;
    if (ort.env?.wasm) {
      // Los hilos de WASM requieren SharedArrayBuffer, que solo existe en
      // páginas aisladas (COOP/COEP). Fuera de ese caso se usa 1 hilo para
      // no provocar un error de inicialización seguido de reintento.
      const isolated = typeof crossOriginIsolated !== 'undefined' && crossOriginIsolated;
      const hw = navigator.hardwareConcurrency || 2;
      const wanted = isolated ? Math.floor(hw / 2) : 1;
      ort.env.wasm.numThreads = Math.max(1, Math.min(threads || 4, wanted));
      ort.env.wasm.simd = true;
      ort.env.wasm.proxy = false;
      ort.env.logLevel = 'error';
    }
    const metaRes = await fetch(metaUrl, { cache: 'force-cache' });
    if (!metaRes.ok) throw new Error(`No se pudo cargar ${metaUrl}`);
    const meta = await metaRes.json();

    let lastError = null;
    for (const threads2 of [ort.env?.wasm?.numThreads || 1, 1]) {
      try {
        ort.env.wasm.numThreads = threads2;
        const session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ['wasm'],
          graphOptimizationLevel: 'all',
        });
        return new SolderDetector(session, meta, ort);
      } catch (err) {
        lastError = err;
      }
    }
    throw lastError || new Error('No se pudo inicializar el modelo');
  }

  /**
   * Ejecuta una pasada sobre una región de la fuente.
   * Si la región es menor que S se rellena (pad) en lugar de ampliarla, para
   * mantener la escala nativa con la que se entrenó el modelo.
   */
  async _infer(source, sx, sy, sw, sh) {
    const S = this.size;
    const ctx = this._ctx;
    ctx.fillStyle = '#727c76';
    ctx.fillRect(0, 0, S, S);
    const scale = Math.min(1, S / Math.max(sw, sh));
    const dw = Math.max(1, Math.round(sw * scale));
    const dh = Math.max(1, Math.round(sh * scale));
    const dx = Math.floor((S - dw) / 2);
    const dy = Math.floor((S - dh) / 2);
    ctx.drawImage(source, sx, sy, sw, sh, dx, dy, dw, dh);
    const { data } = ctx.getImageData(0, 0, S, S);
    const n = S * S;
    const input = new Float32Array(3 * n);
    const [m0, m1, m2] = this.mean;
    const [s0, s1, s2] = this.std;
    for (let i = 0, p = 0; i < n; i++, p += 4) {
      input[i] = (data[p] / 255 - m0) / s0;
      input[n + i] = (data[p + 1] / 255 - m1) / s1;
      input[2 * n + i] = (data[p + 2] / 255 - m2) / s2;
    }
    const feeds = { [this.inputName]: new this.ort.Tensor('float32', input, [1, 3, S, S]) };
    const results = await this.session.run(feeds);
    const outs = this.outputNames.map((name) => results[name]);
    // píxeles del tensor -> píxeles de la región (misma escala en X e Y)
    const inv = 1 / (scale || 1);
    const dets = this._decode(outs, inv, inv);
    return dets
      .map((d) => ({
        ...d,
        x: d.x - dx * inv,
        y: d.y - dy * inv,
        w: Math.min(d.w, sw),
        h: Math.min(d.h, sh),
      }))
      .filter((d) => d.x + d.w > -8 && d.y + d.h > -8 && d.x < sw + 8 && d.y < sh + 8)
      .map((d) => ({
        ...d,
        x: Math.max(0, d.x),
        y: Math.max(0, d.y),
      }));
  }

  _decode(outs, scaleX, scaleY = scaleX) {
    const dets = [];
    const nc = this.classes.length;
    outs.forEach((tensor, li) => {
      const stride = this.strides[li];
      const [, C, H, W] = tensor.dims;
      const d = tensor.data;
      const plane = H * W;
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          const idx = y * W + x;
          let best = 0;
          let bestCls = -1;
          for (let c = 0; c < nc; c++) {
            const v = 1 / (1 + Math.exp(-d[(4 + c) * plane + idx]));
            if (v > best) {
              best = v;
              bestCls = c;
            }
          }
          if (best < 0.08) continue;
          const ox = 1 / (1 + Math.exp(-d[idx]));
          const oy = 1 / (1 + Math.exp(-d[plane + idx]));
          const tw = Math.max(-6, Math.min(6, d[2 * plane + idx]));
          const th = Math.max(-6, Math.min(6, d[3 * plane + idx]));
          const cx = (x + ox) * stride;
          const cy = (y + oy) * stride;
          const bw = Math.exp(tw) * stride;
          const bh = Math.exp(th) * stride;
          dets.push({
            className: this.classes[bestCls],
            severity: SEVERITY_OF(this.classes[bestCls]),
            score: best,
            x: (cx - bw / 2) * scaleX,
            y: (cy - bh / 2) * scaleY,
            w: bw * scaleX,
            h: bh * scaleY,
          });
        }
      }
    });
    return dets;
  }

  /**
   * Analiza una imagen completa con teselas solapadas.
   * @param {CanvasImageSource & {width:number,height:number}} source
   */
  async analyzeImage(source, opts = {}) {
    const width = source.naturalWidth || source.videoWidth || source.width;
    const height = source.naturalHeight || source.videoHeight || source.height;
    const overlap = opts.overlap ?? 0.25;
    const onProgress = opts.onProgress || (() => {});
    // nunca se analiza por encima de la resolución de entrenamiento
    const tile = opts.tile || this.size;
    const tileSize = Math.max(96, Math.min(tile, this.size, Math.max(width, height)));
    const step = Math.max(64, Math.round(tileSize * (1 - overlap)));
    const xs = [];
    const ys = [];
    for (let x = 0; x + tileSize <= width + 1; x += step) xs.push(Math.min(x, width - tileSize));
    if (!xs.length) xs.push(0);
    if (xs[xs.length - 1] + tileSize < width) xs.push(width - tileSize);
    for (let y = 0; y + tileSize <= height + 1; y += step) ys.push(Math.min(y, height - tileSize));
    if (!ys.length) ys.push(0);
    if (ys[ys.length - 1] + tileSize < height) ys.push(height - tileSize);

    const all = [];
    const total = xs.length * ys.length;
    let done = 0;
    const t0 = performance.now();
    for (const y of ys) {
      for (const x of xs) {
        const sx = Math.max(0, Math.min(width - tileSize, x));
        const sy = Math.max(0, Math.min(height - tileSize, y));
        const dets = await this._infer(source, sx, sy, tileSize, tileSize);
        for (const d of dets) {
          d.x += sx;
          d.y += sy;
          all.push(d);
        }
        done++;
        // cede el hilo entre teselas para que el navegador pinte la barra
        // de progreso y la interfaz siga respondiendo mientras analiza
        await new Promise((resolve) => setTimeout(resolve, 0));
        onProgress(done / total, Math.round(performance.now() - t0));
      }
    }
    const merged = nms(all, opts.nmsIou ?? 0.45, opts.maxDetections ?? 600);
    for (const d of merged) {
      d.cx = d.x + d.w / 2;
      d.cy = d.y + d.h / 2;
    }
    return {
      detections: merged,
      tiles: total,
      tileSize,
      width,
      height,
      ms: Math.round(performance.now() - t0),
    };
  }

  /** Analiza una región de la imagen (modo cámara en vivo). */
  async analyzeRegion(source, region) {
    const t0 = performance.now();
    const sw = region.w || region.size;
    const sh = region.h || region.size;
    const dets = await this._infer(source, region.x, region.y, sw, sh);
    for (const d of dets) {
      d.x += region.x;
      d.y += region.y;
      d.cx = d.x + d.w / 2;
      d.cy = d.y + d.h / 2;
    }
    const merged = nms(dets, 0.45, 300);
    return { detections: merged, ms: Math.round(performance.now() - t0) };
  }
}

export function iou(a, b) {
  const ax2 = a.x + a.w;
  const ay2 = a.y + a.h;
  const bx2 = b.x + b.w;
  const by2 = b.y + b.h;
  const iw = Math.max(0, Math.min(ax2, bx2) - Math.max(a.x, b.x));
  const ih = Math.max(0, Math.min(ay2, by2) - Math.max(a.y, b.y));
  const inter = iw * ih;
  const union = a.w * a.h + b.w * b.h - inter;
  return union > 0 ? inter / union : 0;
}

/** Supresión de no máximos agnóstica de clase (una unión = una caja). */
export function nms(dets, iouThr = 0.45, maxOut = 300) {
  if (dets.length <= 1) return dets.slice(0, maxOut);
  const sorted = dets.slice().sort((a, b) => b.score - a.score);
  const keep = [];
  const suppressed = new Uint8Array(sorted.length);
  for (let i = 0; i < sorted.length; i++) {
    if (suppressed[i]) continue;
    const a = sorted[i];
    keep.push(a);
    if (keep.length >= maxOut) break;
    for (let j = i + 1; j < sorted.length; j++) {
      if (suppressed[j]) continue;
      if (iou(a, sorted[j]) > iouThr) suppressed[j] = 1;
    }
  }
  return keep;
}

/** Filtra por umbral de confianza, tamaño mínimo y clases visibles. */
export function filterDetections(dets, opts = {}) {
  const conf = opts.conf ?? 0.35;
  const minSize = opts.minSize ?? 4;
  const visible = opts.visible || { ok: true, warn: true, fail: true };
  return dets.filter(
    (d) =>
      d.score >= conf &&
      d.w >= minSize &&
      d.h >= minSize &&
      visible[d.severity] !== false,
  );
}

/** Compara detecciones contra etiquetas de verdad (muestras incluidas). */
export function evaluateAgainstGroundTruth(dets, gt, iouThr = 0.5) {
  const used = new Set();
  let tp = 0;
  let clsOk = 0;
  for (const d of dets) {
    let best = -1;
    let bestIou = iouThr;
    gt.forEach((g, i) => {
      if (used.has(i)) return;
      const v = iou(d, { x: g.x, y: g.y, w: g.w, h: g.h });
      if (v > bestIou) {
        bestIou = v;
        best = i;
      }
    });
    if (best >= 0) {
      used.add(best);
      tp++;
      if (gt[best].className === d.className) clsOk++;
    }
  }
  const precision = dets.length ? tp / dets.length : 0;
  const recall = gt.length ? tp / gt.length : 0;
  return {
    tp,
    fp: dets.length - tp,
    fn: gt.length - tp,
    precision,
    recall,
    f1: precision + recall ? (2 * precision * recall) / (precision + recall) : 0,
    classAccuracy: tp ? clsOk / tp : 0,
  };
}
