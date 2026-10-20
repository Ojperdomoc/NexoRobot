/**
 * Base de conocimiento IPC-A-610G (Acceptability of Electronic Assemblies).
 *
 * Cada clase del detector se corresponde con un criterio visual del estándar.
 * Las referencias (§) apuntan a la revisión G de octubre de 2017:
 *   §5.1  Requisitos de aceptabilidad de soldadura
 *   §5.2  Anomalías de soldadura
 *   §7.3.5 Uniones en through-hole (Tabla 7-4)
 *   §8.3  Uniones SMT
 */

export const SEVERITY_COLORS = {
  ok: '#22c55e',
  warn: '#f59e0b',
  fail: '#ef4444',
};

export const SEVERITY_LABELS = {
  ok: 'Conforme',
  warn: 'Aceptable · observar',
  fail: 'No conforme',
};

export const SEVERITY_ORDER = { fail: 0, warn: 1, ok: 2 };

/** Información por clase del modelo (índice = salida del ONNX). */
export const CLASS_INFO = {
  ok: {
    severity: 'ok',
    label: 'Conforme',
    short: 'Filete correcto',
    criteria:
      'Filete cóncavo con borde afinado sobre el terminal, ángulo de humectación ≤ 90° y contorno del pin distinguible. La soldadura moja el terminal y la superficie de la PCB (§5.1, Fig. 5-1; Tabla 7-4; §8.3).',
    causes: [
      'Perfil térmico y volumen de pasta dentro de ventana',
      'Superficies limpias y bien humectables',
    ],
    action:
      'Sin acción correctiva. Mantener los parámetros de proceso y usar la pieza como referencia de patrón.',
    references: ['IPC-A-610G §5.1', 'Tabla 7-4', '§8.3.1'],
  },
  warn_dull: {
    severity: 'warn',
    label: 'Superficie mate o granulosa',
    short: 'Aceptable, vigilar',
    criteria:
      'Las aleaciones libres de plomo y el enfriamiento lento en placas de gran masa pueden producir soldadura opaca, gris o granulosa. Es aceptable si la humectación y el filete cumplen (§5.1). No confundir con unión fría (§5.2.5).',
    causes: [
      'Enfriamiento lento o gradiente térmico bajo',
      'Aleación SAC con granulación normal',
      'Residuos de flux no limpios',
      'Impurezas o baño envejecido',
    ],
    action:
      'Verificar el perfil de enfriamiento (2–4 °C/s en SAC305), la antigüedad del baño/pasta y los residuos de flux. Registrar como indicador de proceso (process indicator) y vigilar la tendencia del lote.',
    references: ['IPC-A-610G §5.1', '§5.2.5'],
  },
  warn_coverage: {
    severity: 'warn',
    label: 'Cobertura reducida',
    short: 'Aceptable, vigilar',
    criteria:
      'Cobertura de soldadura por debajo del objetivo pero que aún satisface los mínimos: solape final (J) ≥ 50 % del terminal, ancho de unión (C) ≥ 50 % del pad, humectación evidente (§8.3.1.8, §8.3.1.3, Tabla 8-1). En through-hole, llenado vertical ≥ 75 % (Tabla 7-4, Clase 2/3).',
    causes: [
      'Volumen de pasta bajo (apertura de stencil desgastada)',
      'Planitud del pad o humedad del componente',
      'Poco tiempo/temperatura sobre el pad',
      'Separación excesiva del pin respecto al pad',
    ],
    action:
      'Ajustar el volumen de pasta (relación área/aspecto del stencil), verificar planitud de pads y humedad. En hand soldering, aumentar el tiempo de contacto o la potencia térmica. Inspeccionar la tendencia del lote.',
    references: ['IPC-A-610G §8.3.1.3', '§8.3.1.8', 'Tabla 7-4'],
  },
  fail_cold: {
    severity: 'fail',
    label: 'Unión fría',
    short: 'Defecto',
    criteria:
      'Superficie gris, porosa y con humectación incompleta por calor insuficiente, impurezas o limpieza deficiente. Es defecto para Clase 1, 2 y 3 (§5.2.5, Fig. 5-23).',
    causes: [
      'Temperatura o tiempo de reflujo insuficientes',
      'Precalentamiento o masa térmica mal compensada',
      'Oxidación o contaminación de las superficies',
      'Hierro de soldar con punta fría o potencia insuficiente',
    ],
    action:
      'Corregir el perfil térmico (medir con termopares en la unión), subir la temperatura de la punta (350–380 °C SnPb / 360–400 °C SAC) y el tiempo de contacto, usar flux apropiado y re-trabajar la unión. Verificar la calibración del equipo.',
    references: ['IPC-A-610G §5.2.5'],
  },
  fail_insufficient: {
    severity: 'fail',
    label: 'Soldadura insuficiente',
    short: 'Defecto',
    criteria:
      'El filete no cubre el mínimo requerido: llenado vertical menor a 75 % en through-hole (Tabla 7-4), solape final (J) menor a 50 % del terminal o ancho de unión (C) menor a 50 % del pad en SMT (§8.3.1.3, §8.3.1.8). Terminal visible sin soldar.',
    causes: [
      'Volumen de pasta o aporte de estaño bajo',
      'Apertura de stencil obstruida o desgastada',
      'Pin mal asentado o pad con diseño inadecuado',
      'Calor insuficiente que impide el flujo',
    ],
    action:
      'Aumentar el volumen de pasta (revisar relación de área del stencil > 0.66), comprobar la dosificación y el asentamiento del componente, y corregir el perfil térmico. Re-trabajar con aporte de soldadura y flux.',
    references: ['IPC-A-610G Tabla 7-4', '§8.3.1.3', '§8.3.1.8'],
  },
  fail_nonwet: {
    severity: 'fail',
    label: 'Falta de humectación',
    short: 'Defecto',
    criteria:
      'La soldadura no forma enlace metálico con el metal base; el estaño retrocede dejando montículos irregulares y superficies sin mojar (nonwetting / dewetting). Defecto Clase 1, 2, 3 (§5.2.4, §5.2.6).',
    causes: [
      'Oxidación o contaminación de pads y terminales',
      'Pasta o flux vencidos / mal almacenados',
      'Perfil de precalentamiento inadecuado',
      'Acabado superficial (OSP/ENIG) degradado',
    ],
    action:
      'Revisar almacenamiento y vida útil de pasta y componentes, limpieza previa y atmósfera del horno. Puede requerir limpieza y re-trabajo completo de la unión; verificar el acabado superficial del lote.',
    references: ['IPC-A-610G §5.2.4', '§5.2.6'],
  },
  fail_void: {
    severity: 'fail',
    label: 'Vacíos o poros',
    short: 'Defecto',
    criteria:
      'Pinholes, blowholes o vacíos que reducen la sección de la unión por debajo del mínimo aceptable (§5.2.2). Los vacíos pequeños que no comprometen la unión son aceptables como indicador de proceso.',
    causes: [
      'Humedad en PCB, pasta o componente',
      'Rampa de precalentamiento demasiado rápida',
      'Pasta con exceso de óxidos o mal mezclada',
      'Atrapamiento de aire en el stencil',
    ],
    action:
      'Hornear PCB y componentes según humedad, suavizar la rampa de precalentamiento y revisar la pasta. Para área array, evaluar soldadura en vacío o perfil con mayor tiempo de líquido.',
    references: ['IPC-A-610G §5.2.2'],
  },
  fail_bridge: {
    severity: 'fail',
    label: 'Puente de soldadura',
    short: 'Defecto crítico',
    criteria:
      'Soldadura que une conductores que no deben estar conectados. Defecto Clase 1, 2 y 3: cortocircuito potencial (§5.2.7.2).',
    causes: [
      'Exceso de pasta o de estaño',
      'Stencil con aperturas demasiado juntas o desalineadas',
      'Presión de colocación excesiva',
      'Altura de componente insuficiente',
    ],
    action:
      'Reducir el volumen de pasta y revisar la apertura/separación del stencil, la presión del pick-and-place y la altura del componente. Retirar el puente con malla de desoldadura y verificar el aislamiento eléctrico.',
    references: ['IPC-A-610G §5.2.7.2'],
  },
  fail_missing: {
    severity: 'fail',
    label: 'Sin soldadura',
    short: 'Defecto crítico',
    criteria:
      'Ausencia de soldadura en una unión requerida: no existe conexión eléctrica y no cumple ninguno de los criterios de §5.1.',
    causes: [
      'Apertura de stencil obstruida',
      'Dosificación o impresión de pasta fallida',
      'Componente desplazado o caído',
      'Error de selección en el programa de montaje',
    ],
    action:
      'Verificar la impresión de pasta, la limpieza del stencil y el programa de colocación. Re-trabajar la unión y revisar las piezas del mismo lote.',
    references: ['IPC-A-610G §5.1', 'Tabla 7-4'],
  },
};

export function severityOf(className) {
  const info = CLASS_INFO[className];
  if (info) return info.severity;
  if (className === 'ok') return 'ok';
  if (className && className.startsWith('warn')) return 'warn';
  return 'fail';
}

export function classLabel(className) {
  const info = CLASS_INFO[className];
  if (info) return info.label;
  return className.replace(/_/g, ' ');
}

export function summaryOf(detections) {
  const counts = { ok: 0, warn: 0, fail: 0 };
  for (const d of detections) counts[severityOf(d.className)]++;
  const total = detections.length;
  const pct = (n) => (total ? Math.round((n / total) * 1000) / 10 : 0);
  return {
    total,
    ...counts,
    pctOk: pct(counts.ok),
    pctWarn: pct(counts.warn),
    pctFail: pct(counts.fail),
    verdict: counts.fail > 0 ? 'fail' : counts.warn > 0 ? 'warn' : total > 0 ? 'ok' : 'none',
  };
}

/** Agrupa las detecciones por clase y ordena por severidad y cantidad. */
export function groupFindings(detections) {
  const map = new Map();
  for (const d of detections) {
    const key = d.className;
    if (!map.has(key)) {
      map.set(key, {
        className: key,
        severity: severityOf(key),
        count: 0,
        confidence: 0,
        examples: [],
      });
    }
    const g = map.get(key);
    g.count++;
    g.confidence += d.score;
    if (g.examples.length < 3) g.examples.push(d);
  }
  return [...map.values()]
    .map((g) => ({ ...g, confidence: g.confidence / g.count }))
    .sort((a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity] || b.count - a.count);
}

/** Recomendaciones priorizadas a partir de las detecciones. */
export function buildRecommendations(detections, { ipcClass = 2 } = {}) {
  const groups = groupFindings(detections);
  const items = groups
    .filter((g) => g.severity !== 'ok')
    .map((g) => {
      const info = CLASS_INFO[g.className] || {};
      return {
        className: g.className,
        severity: g.severity,
        count: g.count,
        title: info.label || g.className,
        criteria: info.criteria || '',
        causes: info.causes || [],
        action: info.action || '',
        references: info.references || [],
      };
    });

  const general = [];
  if (items.some((i) => i.severity === 'fail')) {
    general.push(
      'Aislar y re-trabajar las uniones marcadas en rojo antes de continuar el ensamble; documentar la causa raíz en el registro de calidad.',
    );
  }
  if (items.some((i) => i.className === 'fail_bridge' || i.className === 'fail_missing')) {
    general.push(
      'Verificar continuidad eléctrica de las zonas con puentes o uniones sin soldar antes del retrabajo: hay riesgo de cortocircuito o circuito abierto.',
    );
  }
  if (items.some((i) => i.className === 'warn_coverage' || i.className === 'fail_insufficient')) {
    general.push(
      'Revisar el volumen de pasta y la apertura del stencil (relación de área ≥ 0.66) en las zonas con cobertura reducida.',
    );
  }
  if (items.some((i) => i.className === 'warn_dull')) {
    general.push(
      'Registrar la superficie mate como indicador de proceso y revisar la curva de enfriamiento del perfil de reflujo.',
    );
  }
  if (!items.length) {
    general.push(
      'No se detectaron anomalías por encima del umbral configurado. Mantener el monitoreo por muestreo según el plan de calidad.',
    );
  }
  general.push(
    `Criterio de aceptación aplicado: Clase ${ipcClass} de IPC-A-610G. Ajustar la severidad de las observaciones si el producto se inspecciona contra otra clase.`,
  );
  return { items, general };
}

/** Consejos de proceso independientes de la detección (sección "cómo mejorar"). */
export const PROCESS_ADVICE = [
  {
    title: 'Perfil térmico',
    text: 'Documenta el perfil de reflujo con termopares en la unión (rampa 1–3 °C/s, tiempo sobre líquidus 45–90 s, pico 20–30 °C sobre el líquidus). Un perfil verificado elimina la mayoría de uniones frías e insuficientes.',
  },
  {
    title: 'Volumen de pasta',
    text: 'Mide la altura y el área de depósito. Aperturas con relación de área > 0.66 y relaciones de aspecto > 1.5 imprimen de forma estable; revisa el esténcil cada 50 000 ciclos.',
  },
  {
    title: 'Higiene y almacenamiento',
    text: 'La pasta debe conservarse refrigerada y atemperarse antes de usar. Las superficies oxidadas producen falta de humectación: revisa la fecha de vencimiento y el acabado superficial del PCB.',
  },
  {
    title: 'Diseño para manufactura (DFM)',
    text: 'Ajusta el diseño de pads, el desahogo térmico y la separación entre pines para evitar exceso de pasta y puentes. El diseño correcto es más barato que el retrabajo.',
  },
  {
    title: 'Inspección y trazabilidad',
    text: 'Registra los resultados por lote y por clase IPC. Las tendencias en las observaciones naranjas anticipan defectos rojos; atiéndelas antes de que se conviertan en rechazo.',
  },
];


/**
 * Agrupa uniones próximas para presentarlas como componentes en el informe.
 * Devuelve [{ box, detections, severities, verdict }].
 */
export function clusterJoints(detections) {
  const used = new Set();
  const clusters = [];
  const center = (d) => ({ x: d.x + d.w / 2, y: d.y + d.h / 2 });
  const area = (d) => d.w * d.h;
  const sorted = detections.slice().sort((a, b) => area(b) - area(a));
  for (const seed of sorted) {
    if (used.has(seed)) continue;
    const group = [seed];
    used.add(seed);
    const c = center(seed);
    const reach = Math.max(seed.w, seed.h) * 0.85;
    for (const o of detections) {
      if (used.has(o)) continue;
      const oc = center(o);
      if (Math.hypot(oc.x - c.x, oc.y - c.y) <= reach) {
        group.push(o);
        used.add(o);
        for (const p of detections) {
          if (used.has(p) || p === seed) continue;
          const pc = center(p);
          if (Math.hypot(pc.x - c.x, pc.y - c.y) <= reach * 1.05) {
            group.push(p);
            used.add(p);
          }
        }
      }
    }
    const xs = group.map((d) => d.x);
    const ys = group.map((d) => d.y);
    const xe = group.map((d) => d.x + d.w);
    const ye = group.map((d) => d.y + d.h);
    const x0 = Math.min(...xs);
    const y0 = Math.min(...ys);
    const x1 = Math.max(...xe);
    const y1 = Math.max(...ye);
    const counts = { ok: 0, warn: 0, fail: 0 };
    for (const d of group) counts[severityOf(d.className)]++;
    clusters.push({
      box: { x: x0, y: y0, w: x1 - x0, h: y1 - y0 },
      detections: group,
      counts,
      verdict: counts.fail ? 'fail' : counts.warn ? 'warn' : 'ok',
    });
  }
  return clusters.sort((a, b) => a.box.y - b.box.y || a.box.x - b.box.x);
}

export default {
  CLASS_INFO,
  SEVERITY_COLORS,
  SEVERITY_LABELS,
  severityOf,
  classLabel,
  summaryOf,
  groupFindings,
  buildRecommendations,
  clusterJoints,
  PROCESS_ADVICE,
};
