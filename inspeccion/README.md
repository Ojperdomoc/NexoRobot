# Inspección de soldadura con IA · Nexo

Aplicación web que abre la cámara (o una fotografía), detecta las **uniones de
soldadura de una PCB** con un modelo YOLO ejecutado en el navegador y las marca
según su conformidad con **IPC-A-610G**:

| Color | Significado | Ejemplo de criterio |
| --- | --- | --- |
| 🟢 Verde | **Conforme** | filete cóncavo, humectación ≤ 90°, cobertura completa (§5.1) |
| 🟠 Naranja | **Aceptable con observación** | superficie mate/granulosa o cobertura mínima aceptable (§5.1, §8.3.1, Tabla 7-4) |
| 🔴 Rojo | **No conforme** | unión fría, soldadura insuficiente, falta de humectación, vacíos, puente, sin soldadura (§5.2) |

Además permite **capturar la foto** (con las cajas dibujadas) y **descargar un
informe PDF** con la evidencia, el resumen de hallazgos, las uniones
prioritarias, los componentes con observaciones y **recomendaciones concretas**
para mejorar el proceso, cada una citando el apartado del estándar.

## Uso

1. Abre `inspeccion/index.html` desde un servidor HTTP (no `file://`).
2. **Activar cámara** → se analiza en vivo la ventana central (384 px nativos,
   ampliables con el control *Ampliación*).
3. **Capturar y analizar** ejecuta un análisis por teselas de la foto completa
   (máxima resolución efectiva).
4. Ajusta la **confianza mínima** y los filtros de severidad si hace falta.
5. Completa *Placa*, *Operador* y *Clase IPC* y pulsa **Descargar PDF**.

Sin cámara puedes usar **Subir foto** o **Muestra** (tres placas de ejemplo con
etiquetas, útiles para probar y para el botón *Validar vs. etiquetas*).

Atajos: `Espacio` capturar · `P` PDF · `R` reiniciar · `?` ayuda.

## Arquitectura

```
inspeccion/
├── index.html            interfaz (una sola página, sin build)
├── assets/
│   ├── app.js            cámara, captura, render, historial, exportación
│   ├── detector.js       ONNX Runtime Web + teselas + NMS + métricas
│   ├── ipc.js            base de conocimiento IPC-A-610G y recomendaciones
│   ├── report.js         informe PDF (jsPDF)
│   ├── app.css           diseño
│   └── nexo-solder-card.md  tarjeta del modelo
├── modelos/
│   ├── nexo-solder.onnx  detector (≈1.2 MB)
│   ├── nexo-solder.json  clases, strides, normalización, tamaño de entrada
│   └── metrics.json      métricas de validación
├── muestras/             fotos de ejemplo + etiquetas de verdad (gt/)
└── vendor/               ort.min.js, ort-wasm-simd-threaded.wasm, jspdf
```

Todo se ejecuta en el cliente: **ninguna imagen se envía a un servidor** y no
hace falta backend ni proceso de build. La app funciona también en GitHub Pages
(ver el workflow del repositorio).

### Modelo

`nexo-solder.onnx` es un detector anchor-free tipo YOLO (backbone CNN con
convoluciones separables, FPN ligero y tres cabezas a *stride* 8/16/32), 0.30 M
de parámetros, entrada 384×384 y **9 clases** (conforme + 2 observaciones +
6 defectos). Se entrenó con un renderizador sintético de PCB
(`training/pcb_synth.py`) y se evalúa con el mismo pipeline de teselas que usa
la app (`training/eval_onnx.py`).

Detalles, métricas y limitaciones: [`assets/nexo-solder-card.md`](assets/nexo-solder-card.md).

## Servir en local

```bash
python3 tools/serve_preview.py  # desde la raíz del repositorio
# http://localhost:8080/inspeccion/   (la raíz redirige aquí)
```

El script escucha en `0.0.0.0`, sirve `.wasm` con el MIME correcto y desactiva
la caché. También vale cualquier servidor estático, por ejemplo
`python3 -m http.server 8080` desde la raíz del repositorio.

La cámara requiere **HTTPS** o `localhost`. En producción, GitHub Pages ya
publica la carpeta `inspeccion/` mediante el workflow del repositorio.

## Privacidad

* La inferencia y el PDF se generan en el dispositivo.
* El historial (últimas 6 inspecciones) se guarda en `localStorage` con una
  miniatura de la imagen; se puede borrar desde la pestaña *Historial*.
* No hay analítica ni peticiones externas: todas las dependencias están en
  `vendor/` y el modelo en `modelos/`.
