/**
 * Generación del informe PDF (jsPDF) a partir del resultado de la inspección.
 * A4 vertical, unidades mm.
 */
import { CLASS_INFO, SEVERITY_COLORS, SEVERITY_LABELS, PROCESS_ADVICE, clusterJoints } from './ipc.js';

const PAGE_W = 210;
const PAGE_H = 297;
const M = 14;                 // margen
const CONTENT_W = PAGE_W - M * 2;

const INK = [15, 32, 38];
const MUTED = [92, 112, 118];
const LINE = [214, 226, 224];
const ACCENT = [13, 122, 110];
const DARK = [10, 30, 33];
const SOFT = [241, 246, 244];

// WinAnsi (Helvetica) no incluye varios símbolos: se transliteran.
const TRANSLITERATE = [
  [/≥/g, '>='],
  [/≤/g, '<='],
  [/≈/g, '~'],
  [/→/g, '->'],
  [/●/g, 'o'],
  [/▲/g, '^'],
  [/▼/g, 'v'],
  [/✓/g, 'OK'],
  [/⚠/g, '!'],
  [/[\u2018\u2019]/g, "'"],
  [/[\u201C\u201D]/g, '"'],
  [/…/g, '...'],
];
function safeText(value) {
  let out = String(value);
  for (const [re, rep] of TRANSLITERATE) out = out.replace(re, rep);
  return out;
}

const hexToRgb = (hex) => {
  const n = parseInt(hex.replace('#', ''), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
};

function fmtDate(d) {
  return new Intl.DateTimeFormat('es-CO', {
    dateStyle: 'long',
    timeStyle: 'short',
  }).format(d);
}

export function buildInspectionReport(payload) {
  const {
    jsPDF,
    imageDataUrl,
    imageSize = null,
    details = [],
    summary,
    detections,
    recommendations,
    meta = {},
    model = {},
  } = payload;
  const doc = new jsPDF({ unit: 'mm', format: 'a4', compress: true });
  doc.setProperties({
    title: `Informe de inspección de soldadura ${meta.folio || ''}`.trim(),
    subject: 'Inspección visual de soldadura asistida por IA · IPC-A-610G',
    author: 'Nexo · Visión de soldadura',
    creator: 'NexoRobot',
  });

  // jsPDF sólo emite caracteres WinAnsi; se transliteran antes de dibujar.
  const rawText = doc.text.bind(doc);
  doc.text = (text, x, yy, opts) => rawText(
    Array.isArray(text) ? text.map(safeText) : safeText(text), x, yy, opts);

  let y = 0;
  const state = { page: 1 };

  const setColor = (c) => doc.setTextColor(c[0], c[1], c[2]);
  const fill = (c) => doc.setFillColor(c[0], c[1], c[2]);
  const stroke = (c) => doc.setDrawColor(c[0], c[1], c[2]);

  const footer = () => {
    const p = doc.getCurrentPageInfo().pageNumber;
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7);
    setColor(MUTED);
    doc.text('Nexo · Inspección visual de soldadura · Referencia IPC-A-610G', M, PAGE_H - 8);
    doc.text(`Página ${p}`, PAGE_W - M, PAGE_H - 8, { align: 'right' });
    stroke(LINE);
    doc.setLineWidth(0.2);
    doc.line(M, PAGE_H - 11, PAGE_W - M, PAGE_H - 11);
  };

  const newPage = () => {
    footer();
    doc.addPage();
    state.page++;
    y = M + 4;
    setColor(INK);
  };

  const ensure = (h) => {
    if (y + h > PAGE_H - 18) newPage();
  };

  const paragraph = (text, { size = 9, color = MUTED, lh = 4.2, pad = 0 } = {}) => {
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(size);
    setColor(color);
    const lines = doc.splitTextToSize(text, CONTENT_W - pad);
    ensure(lines.length * lh + 2);
    doc.text(lines, M + pad / 2, y);
    y += lines.length * lh;
  };

  const heading = (text, { size = 11, color = INK, gap = 3.5 } = {}) => {
    ensure(14);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(size);
    setColor(color);
    doc.text(text.toUpperCase(), M, y);
    y += 2;
    stroke(LINE);
    doc.setLineWidth(0.3);
    doc.line(M, y, PAGE_W - M, y);
    y += gap;
  };

  // ─────────────────────────── cabecera ───────────────────────────
  fill(DARK);
  doc.rect(0, 0, PAGE_W, 30, 'F');
  fill(ACCENT);
  doc.rect(0, 30, PAGE_W, 1.4, 'F');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(15);
  doc.setTextColor(240, 250, 246);
  doc.text('INFORME DE INSPECCIÓN DE SOLDADURA', M, 13);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  doc.setTextColor(160, 210, 198);
  doc.text('Nexo · Visión artificial para ensamble electrónico · IPC-A-610G', M, 19);
  doc.text(`Generado ${fmtDate(meta.date instanceof Date ? meta.date : new Date())}`, M, 24);

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(240, 250, 246);
  doc.text(`FOLIO ${meta.folio || 'S/N'}`, PAGE_W - M, 13, { align: 'right' });
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  doc.setTextColor(160, 210, 198);
  doc.text(`Clase IPC ${meta.ipcClass ?? 2}`, PAGE_W - M, 19, { align: 'right' });
  doc.text(`${detections.length} uniones analizadas`, PAGE_W - M, 24, { align: 'right' });

  y = 40;

  // ─────────────────────────── datos ───────────────────────────
  const rows = [
    ['Placa / PCB', meta.boardId || 'No especificada', 'Operador', meta.operator || 'No especificado'],
    ['Fuente de imagen', meta.source || 'Cámara', 'Resolución', meta.resolution || '—'],
    ['Cámara / dispositivo', meta.device || '—', 'Umbral de confianza', `${Math.round((meta.conf ?? 0.35) * 100)} %`],
    ['Modelo', model.name || 'Nexo SolderNet', 'Inferencia', meta.inferenceMs ? `${meta.inferenceMs} ms · ${meta.tiles || 1} tesela(s)` : '—'],
  ];
  doc.setFontSize(8.6);
  const colW = CONTENT_W / 2;
  rows.forEach((r) => {
    fill(SOFT);
    doc.roundedRect(M, y - 4.2, CONTENT_W, 10.5, 1.6, 1.6, 'F');
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(7.2);
    setColor(MUTED);
    doc.text(r[0].toUpperCase(), M + 3, y);
    doc.text(r[2].toUpperCase(), M + colW + 3, y);
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(9);
    setColor(INK);
    doc.text(String(r[1]).slice(0, 46), M + 3, y + 4.2);
    doc.text(String(r[3]).slice(0, 46), M + colW + 3, y + 4.2);
    y += 12;
  });
  y += 3;

  // ─────────────────────────── veredicto ───────────────────────────
  const verdictColor = hexToRgb(SEVERITY_COLORS[summary.verdict] || SEVERITY_COLORS.fail);
  const verdictText =
    summary.verdict === 'fail'
      ? 'NO CONFORME · requiere retrabajo'
      : summary.verdict === 'warn'
        ? 'CONFORME CON OBSERVACIONES'
        : summary.verdict === 'ok'
          ? 'CONFORME · sin defectos detectados'
          : 'SIN DETECCIONES';

  fill(SOFT);
  doc.roundedRect(M, y, CONTENT_W, 26, 2.5, 2.5, 'F');
  fill(verdictColor);
  doc.roundedRect(M, y, 3.2, 26, 1.2, 1.2, 'F');
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(12.5);
  setColor(INK);
  doc.text(verdictText, M + 9, y + 10);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8.4);
  setColor(MUTED);
  doc.text(
    `${summary.pctOk} % de las uniones analizadas cumple el criterio de aceptación. ` +
      `${summary.fail} no conformes · ${summary.warn} observaciones · ${summary.total} uniones.`,
    M + 9,
    y + 17,
  );

  // barra de proporciones
  const barY = y + 20.5;
  const barW = CONTENT_W - 18;
  let bx = M + 9;
  [['ok', summary.ok / Math.max(summary.total, 1)], ['warn', summary.warn / Math.max(summary.total, 1)],
    ['fail', summary.fail / Math.max(summary.total, 1)]].forEach(([sev, frac]) => {
    if (frac <= 0) return;
    fill(hexToRgb(SEVERITY_COLORS[sev]));
    doc.rect(bx, barY, Math.max(0.6, barW * frac), 2.2, 'F');
    bx += barW * frac;
  });
  y += 33;

  // ─────────────────────────── imagen ───────────────────────────
  heading('Evidencia fotográfica');
  if (imageDataUrl) {
    const img = doc.getImageProperties(imageDataUrl);
    const maxW = CONTENT_W;
    const maxH = 105;
    let w = maxW;
    let h = (img.height / img.width) * w;
    if (h > maxH) {
      h = maxH;
      w = (img.width / img.height) * h;
    }
    ensure(h + 12);
    const x = M + (CONTENT_W - w) / 2;
    stroke(LINE);
    doc.setLineWidth(0.3);
    doc.rect(x - 0.5, y - 0.5, w + 1, h + 1);
    doc.addImage(imageDataUrl, 'JPEG', x, y, w, h, undefined, 'FAST');

    // zonas que se amplían en la página siguiente
    const srcW = imageSize?.w || img.width;
    const srcH = imageSize?.h || img.height;
    if (details.length && srcW && srcH) {
      details.forEach((d) => {
        const rx = x + (d.x / srcW) * w;
        const ry = y + (d.y / srcH) * h;
        const rw = (d.w / srcW) * w;
        const rh = (d.h / srcH) * h;
        doc.setDrawColor(255, 255, 255);
        doc.setLineWidth(0.45);
        doc.rect(rx, ry, rw, rh, 'S');
        doc.setDrawColor(10, 30, 33);
        doc.setLineWidth(0.2);
        doc.rect(rx - 0.35, ry - 0.35, rw + 0.7, rh + 0.7, 'S');
        doc.setFillColor(255, 255, 255);
        doc.rect(rx, ry, 9, 4.6, 'F');
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(6.6);
        doc.setTextColor(10, 30, 33);
        doc.text(d.label, rx + 1.2, ry + 3.3);
      });
    }
    y += h + 5;
    doc.setFont('helvetica', 'italic');
    doc.setFontSize(7.6);
    setColor(MUTED);
    doc.text(
      'Recuadros verdes: conformes · naranjas: aceptables con observación · rojos: no conformes. La caja encierra la unión evaluada.',
      PAGE_W / 2,
      y,
      { align: 'center', maxWidth: CONTENT_W },
    );
    y += 7;
  }

  // ─────────────────── ampliaciones (detalle 1:1) ───────────────────
  if (details.length) {
    heading('Vistas ampliadas de las zonas críticas');
    paragraph(
      'Recortes a escala nativa (×2 al renderizar) de las zonas con más uniones no conformes. ' +
        'Cada recorte corresponde al rectángulo blanco marcado con su etiqueta sobre la evidencia fotográfica.',
      { size: 8, color: MUTED, lh: 4 },
    );
    y += 1;

    const gap = 6;
    const cellW = (CONTENT_W - gap) / 2;
    const cellH = cellW * 0.75;
    const rowH = cellH + 12;
    let rowY = y;
    details.forEach((d, i) => {
      const col = i % 2;
      if (col === 0) {
        ensure(rowH);
        rowY = y;
      }
      const cx = M + col * (cellW + gap);
      doc.setDrawColor(LINE[0], LINE[1], LINE[2]);
      doc.setLineWidth(0.25);
      doc.rect(cx - 0.4, rowY - 0.4, cellW + 0.8, cellH + 0.8, 'S');
      doc.addImage(d.dataUrl, 'JPEG', cx, rowY, cellW, cellH, undefined, 'FAST');
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(7.4);
      setColor(INK);
      doc.text(`${d.label} · zona (${d.x}, ${d.y})`, cx, rowY + cellH + 3.4);
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(7);
      setColor(MUTED);
      doc.text(
        `${d.fails} no conformes · ${d.warns} observaciones · ${d.total} uniones en el recorte`,
        cx,
        rowY + cellH + 7,
      );
      if (col === 1 || i === details.length - 1) y = rowY + rowH;
    });
    y += 2;
  }

  // ─────────────────────────── hallazgos ───────────────────────────
  const groups = Object.entries(
    detections.reduce((acc, d) => {
      acc[d.className] = acc[d.className] || { count: 0, score: 0 };
      acc[d.className].count++;
      acc[d.className].score += d.score;
      return acc;
    }, {}),
  )
    .map(([className, v]) => ({
      className,
      count: v.count,
      score: v.score / v.count,
      severity: (CLASS_INFO[className] || {}).severity || 'fail',
      label: (CLASS_INFO[className] || {}).label || className,
    }))
    .sort((a, b) => {
      const order = { fail: 0, warn: 1, ok: 2 };
      return order[a.severity] - order[b.severity] || b.count - a.count;
    });

  heading('Resumen de hallazgos');
  // cabecera de tabla
  const cols = [
    { key: 'sev', label: 'Severidad', x: M + 2, w: 30 },
    { key: 'label', label: 'Hallazgo', x: M + 34, w: 78 },
    { key: 'count', label: 'Uniones', x: M + 116, w: 18, align: 'right' },
    { key: 'conf', label: 'Confianza', x: M + 138, w: 20, align: 'right' },
    { key: 'ref', label: 'Referencia', x: M + 162, w: 34 },
  ];
  fill(DARK);
  doc.rect(M, y - 4, CONTENT_W, 7, 'F');
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7.4);
  doc.setTextColor(235, 245, 242);
  cols.forEach((c) => doc.text(c.label.toUpperCase(), c.x, y + 0.4, { align: c.align || 'left' }));
  y += 9;

  groups.forEach((g, i) => {
    ensure(9);
    const c = hexToRgb(SEVERITY_COLORS[g.severity]);
    if (i % 2 === 0) {
      fill([248, 251, 250]);
      doc.rect(M, y - 5, CONTENT_W, 8.4, 'F');
    }
    fill(c);
    doc.circle(M + 4, y - 1.2, 1.5, 'F');
    doc.setFont('helvetica', 'normal');
    doc.setFontSize(8);
    setColor(INK);
    doc.text(SEVERITY_LABELS[g.severity], M + 8, y);
    doc.text(g.label.slice(0, 42), M + 34, y);
    doc.text(String(g.count), M + 134, y, { align: 'right' });
    doc.text(`${Math.round(g.score * 100)} %`, M + 158, y, { align: 'right' });
    const ref = (CLASS_INFO[g.className] || {}).references?.[0] || 'IPC-A-610G';
    doc.setFontSize(7.4);
    setColor(MUTED);
    doc.text(ref.replace('IPC-A-610G ', ''), M + 162, y);
    y += 9;
  });
  y += 4;

  // ─────────────────────────── recomendaciones ───────────────────────────
  if (recommendations?.items?.length) {
    heading('Recomendaciones para mejorar la PCB');
    recommendations.items.forEach((item) => {
      const c = hexToRgb(SEVERITY_COLORS[item.severity]);
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(9.2);
      const title = `${item.count} × ${item.title}`;
      const titleLines = doc.splitTextToSize(title, CONTENT_W - 8);
      ensure(titleLines.length * 4.6 + 8);
      fill(c);
      doc.rect(M, y - 3.2, 1.8, titleLines.length * 4.6 + 2, 'F');
      doc.setTextColor(c[0], c[1], c[2]);
      doc.text(titleLines, M + 5, y + 0.6);
      y += titleLines.length * 4.6 + 1;
      paragraph(item.action, { size: 8.4, color: INK, lh: 4, pad: 10 });
      if (item.references?.length) {
        doc.setFont('helvetica', 'italic');
        doc.setFontSize(7.2);
        setColor(MUTED);
        ensure(5);
        doc.text(item.references.join(' · '), M + 5, y + 1.5);
        y += 5.5;
      }
      y += 3.5;
    });

    if (recommendations.general?.length) {
      heading('Acciones generales');
      recommendations.general.forEach((t) => {
        ensure(8);
        fill(ACCENT);
        doc.circle(M + 1.6, y - 1.2, 0.9, 'F');
        paragraph(t, { size: 8.2, color: INK, lh: 4, pad: 8 });
        y += 1.6;
      });
    }
  }

  // ─────────────────────────── componentes ───────────────────────────
  if (detections.length >= 4) {
    const clusters = clusterJoints(detections)
      .filter((c) => c.detections.length >= 2)
      .slice(0, 16);
    if (clusters.length) {
      heading('Componentes con observaciones');
      paragraph(
        'Agrupación automática de uniones próximas. Permite ubicar el componente que concentra los hallazgos y priorizar el retrabajo.',
        { size: 8, lh: 4 },
      );
      y += 1;
      const cc = [
        { label: 'Componente', x: M + 2 },
        { label: 'Región (px)', x: M + 34 },
        { label: 'Uniones', x: M + 92, align: 'right' },
        { label: 'OK', x: M + 108, align: 'center' },
        { label: 'OBS', x: M + 119, align: 'center' },
        { label: 'NC', x: M + 130, align: 'center' },
        { label: 'Veredicto', x: M + 142 },
      ];
      fill(DARK);
      doc.rect(M, y - 4, CONTENT_W, 6.4, 'F');
      doc.setFont('helvetica', 'bold');
      doc.setFontSize(7);
      doc.setTextColor(235, 245, 242);
      cc.forEach((c) => doc.text(c.label.toUpperCase(), c.x, y + 0.2, { align: c.align || 'left' }));
      y += 7.5;
      clusters.forEach((cl, i) => {
        ensure(7);
        if (i % 2 === 0) {
          fill([248, 251, 250]);
          doc.rect(M, y - 4, CONTENT_W, 6.6, 'F');
        }
        doc.setFont('helvetica', 'normal');
        doc.setFontSize(7.6);
        setColor(INK);
        doc.text(`C-${String(i + 1).padStart(2, '0')}`, M + 2, y);
        setColor(MUTED);
        doc.setFontSize(7.2);
        doc.text(`${Math.round(cl.box.x)}, ${Math.round(cl.box.y)} · ${Math.round(cl.box.w)}×${Math.round(cl.box.h)}`,
          M + 34, y);
        doc.setFontSize(7.6);
        setColor(INK);
        doc.text(String(cl.detections.length), M + 92, y, { align: 'right' });
        [['ok', M + 108], ['warn', M + 119], ['fail', M + 130]].forEach(([sev, x]) => {
          const col = hexToRgb(SEVERITY_COLORS[sev]);
          doc.setFont('helvetica', 'bold');
          doc.setTextColor(col[0], col[1], col[2]);
          doc.text(String(cl.counts[sev]), x, y, { align: 'center' });
        });
        const col = hexToRgb(SEVERITY_COLORS[cl.verdict]);
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(7.4);
        doc.setTextColor(col[0], col[1], col[2]);
        doc.text(SEVERITY_LABELS[cl.verdict], M + 142, y);
        y += 7;
      });
      y += 3;
    }
  }

  // ─────────────────────────── uniones prioritarias ───────────────────────────
  const worst = detections
    .filter((d) => d.severity !== 'ok')
    .sort((a, b) => b.score - a.score || (a.severity === 'fail' ? -1 : 1))
    .slice(0, 24);
  if (worst.length) {
    heading('Uniones prioritarias');
    paragraph(
      'Coordenadas en píxeles de la imagen original, ordenadas por confianza. Sirven para localizar la unión en la placa o en el archivo de imagen.',
      { size: 8, lh: 4 },
    );
    y += 1;
    const c2 = [
      { label: '#', x: M + 2 },
      { label: 'Tipo', x: M + 12 },
      { label: 'Severidad', x: M + 62 },
      { label: 'X, Y', x: M + 98 },
      { label: 'Conf.', x: M + 140, align: 'right' },
      { label: 'IPC', x: M + 152 },
    ];
    fill(DARK);
    doc.rect(M, y - 4, CONTENT_W, 6.4, 'F');
    doc.setFont('helvetica', 'bold');
    doc.setFontSize(7);
    doc.setTextColor(235, 245, 242);
    c2.forEach((c) => doc.text(c.label.toUpperCase(), c.x, y + 0.2, { align: c.align || 'left' }));
    y += 7.5;
    worst.forEach((d, i) => {
      ensure(7);
      if (i % 2 === 0) {
        fill([248, 251, 250]);
        doc.rect(M, y - 4, CONTENT_W, 6.6, 'F');
      }
      const c = hexToRgb(SEVERITY_COLORS[d.severity]);
      doc.setFont('helvetica', 'normal');
      doc.setFontSize(7.4);
      setColor(INK);
      doc.text(String(i + 1), M + 2, y);
      doc.text(((CLASS_INFO[d.className] || {}).label || d.className).slice(0, 30), M + 12, y);
      fill(c);
      doc.circle(M + 64, y - 1, 1.3, 'F');
      setColor(MUTED);
      doc.text(`${Math.round(d.x)}, ${Math.round(d.y)}`, M + 98, y);
      doc.text(`${Math.round(d.score * 100)} %`, M + 140, y, { align: 'right' });
      const ref = (CLASS_INFO[d.className] || {}).references?.[0] || '';
      doc.setFontSize(6.8);
      doc.text(ref.replace('IPC-A-610G ', ''), M + 152, y);
      y += 7;
    });
    y += 2;
  }

  // ─────────────────────────── notas del operador ───────────────────────────
  if (meta.notes) {
    heading('Notas del operador');
    paragraph(meta.notes, { size: 8.4, color: INK, lh: 4.2 });
    y += 3;
  }

  // ─────────────────────────── metodología ───────────────────────────
  heading('Metodología y alcance');
  paragraph(
    'La inspección se realizó con un detector de objetos convolucional (YOLO-style, anchor-free) ejecutado ' +
      'íntegramente en el navegador sobre la fotografía capturada. Cada unión detectada se clasifica según su ' +
      'apariencia visual en conforme (verde), aceptable con observación (naranja) o no conforme (rojo). ' +
      'Los criterios de aceptación y las referencias normativas provienen de IPC-A-610G, «Acceptability of ' +
      'Electronic Assemblies» (octubre de 2017), secciones 5.1, 5.2, 7.3.5 y 8.3.',
    { size: 8.2, lh: 4, color: MUTED },
  );
  y += 2;
  paragraph(
    'Limitaciones: esta herramienta es una ayuda de inspección visual y no sustituye la verificación humana ' +
      'ni los ensayos eléctricos o de rayos X. El modelo fue entrenado con imágenes sintéticas de placas y ' +
      'puede degradarse con iluminación especular, desenfoque, escalas muy distintas o acabados poco habituales. ' +
      'Las uniones ocultas (BGA, BTC) no son evaluables mediante inspección visual superficial. Confirme cada ' +
      'defecto antes de retrabajar la placa.',
    { size: 8.2, lh: 4, color: MUTED },
  );
  y += 4;

  ensure(24);
  fill(SOFT);
  doc.roundedRect(M, y, CONTENT_W, 20, 2, 2, 'F');
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  setColor(INK);
  doc.text('Firma del operador', M + 4, y + 14);
  doc.text('Revisión de calidad', M + CONTENT_W / 2 + 4, y + 14);
  stroke(LINE);
  doc.line(M + 4, y + 10, M + 60, y + 10);
  doc.line(M + CONTENT_W / 2 + 4, y + 10, M + CONTENT_W / 2 + 60, y + 10);

  footer();
  return doc;
}

/** Consejos generales de proceso, reutilizables en el informe o en la UI. */
export function processAdvice() {
  return PROCESS_ADVICE;
}
