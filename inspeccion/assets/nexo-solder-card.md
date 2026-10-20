# Nexo SolderNet · tarjeta del modelo

Detector de uniones de soldadura en placas de circuito impreso, diseñado para
ejecutarse **íntegramente en el navegador** (ONNX Runtime Web · WebAssembly) sin
enviar imágenes a ningún servidor.

## Arquitectura

| Elemento | Valor |
| --- | --- |
| Tipo | Detector de objetos anchor-free estilo YOLO (una etapa) |
| Backbone | CNN MobileNet-style con convoluciones separables en profundidad |
| Cuello | FPN ligero (P8, P16, P32) |
| Cabezas | 3 niveles, decoupled (clasificación + regresión), 4 + *nc* canales |
| Parámetros | ≈ 0.30 M |
| Entrada | 384 × 384 px, RGB, normalización ImageNet |
| Salidas | `out0` 48×48, `out1` 24×24, `out2` 12×12 (4 + 9 canales) |
| Pesos | `nexo-solder.onnx` ≈ 1.2 MB (fp32) |

Pre/post-proceso implementado en `detector.js`:

1. Recorte cuadrado de la fuente y redimensionado a 384 × 384.
2. Normalización NCHW: `(x/255 − mean) / std`.
3. Decodificación: `sigmoid` para el centro, `exp` para ancho/alto, *stride* 8/16/32.
4. Umbral de confianza bajo (0.08) y **NMS agnóstica de clase** (IoU 0.45), de modo
   que cada unión física conserva una sola caja, con la clase más probable.
5. En fotos grandes se analiza por **teselas solapadas** (22 %) para conservar
   resolución efectiva; en vivo se usa una región central ampliada (zoom digital).

## Clases (9)

| id | clase | severidad visual | criterio IPC-A-610G |
| --- | --- | --- | --- |
| 0 | `ok` | verde | filete cóncavo, humectación ≤ 90° (§5.1) |
| 1 | `warn_dull` | naranja | superficie mate/granulosa aceptable (§5.1) |
| 2 | `warn_coverage` | naranja | cobertura mínima pero aceptable (§8.3.1, Tabla 7-4) |
| 3 | `fail_cold` | rojo | unión fría (§5.2.5) |
| 4 | `fail_insufficient` | rojo | soldadura insuficiente (§8.3.1.8, Tabla 7-4) |
| 5 | `fail_nonwet` | rojo | no humectación / dewetting (§5.2.4, §5.2.6) |
| 6 | `fail_void` | rojo | vacíos, pinholes, blowholes (§5.2.2) |
| 7 | `fail_bridge` | rojo | puente de soldadura (§5.2.7.2) |
| 8 | `fail_missing` | rojo | sin soldadura |

## Datos de entrenamiento

* **Sintéticos**: renderizador procedural de PCB (`training/pcb_synth.py`) con siete
  tipos de componente (chip 2 terminales, SOIC/QFN, electrolítico, conector, THT),
  vías, serigrafía, pistas, texturas de máscara, iluminación especular y
  aumentos de cámara (balance de blancos, exposición, desenfoque, JPEG).
* **~4 500 fotogramas** de 512 × 512 (recortes de ~380 placas) + validación
  independiente de 40 placas y 3 fotografías de ejemplo con etiquetas.
* Clases desbalanceadas a propósito (aprox. 54 % conformes / 26 % observaciones /
  20 % defectos), con las condiciones de fallo repartidas entre las seis
  categorías rojas.
* Entrenamiento: 16 épocas, AdamW + OneCycle (lr 3e-3), 384 px, batch 9,
  ~95 min en CPU. Función de pérdida: focal BCE en la clasificación + GIoU
  ponderada por IoU en la regresión, normalizada por el número de positivos.

## Métricas de validación (placas completas, pipeline con teselas)

Se evalúan las 20 placas de validación completas con el mismo pipeline que usa la
app (teselas de 384–900 px, NMS, confianza 0.35). Ver `inspeccion/modelos/metrics.json`.

| Métrica | Valor (confianza 0.35, IoU 0.5) |
| --- | --- |
| Precisión | 0.63 |
| Exhaustividad | 0.93 |
| **F1** | **0.75** |
| Exactitud de clase (entre aciertos) | 0.82 |
| Latencia WASM (1 hilo, navegador real) | ~0.8–1.3 s por foto (20–24 teselas) · ~40–80 ms por fotograma en vivo |

Por clase (placas completas):

| clase | P | R | F1 |
| --- | --- | --- | --- |
| ok | 0.67 | 0.97 | 0.79 |
| warn_dull | 0.66 | 0.93 | 0.77 |
| warn_coverage | 0.69 | 0.94 | 0.79 |
| fail_cold | 0.58 | 0.90 | 0.71 |
| fail_insufficient | 0.32 | 0.73 | 0.44 |
| fail_nonwet | 0.82 | 0.84 | 0.83 |
| fail_void | 0.38 | 0.81 | 0.51 |
| fail_bridge | 0.41 | 0.52 | 0.46 |
| fail_missing | 0.62 | 0.85 | 0.72 |

Validación de recortes (misma distribución que el entrenamiento): P 0.71 / R 0.72 / F1 0.72.
Fotos de ejemplo incluidas en la app (F1): 0.93 · 0.83 · 0.86.

## Limitaciones

* Entrenado con imágenes **sintéticas**: la apariencia de las uniones es
  realista pero no cubre todas las variantes industriales (pastas, acabados,
  reflujo en vacío, inspección por rayos X).
* Sensible a iluminación especular fuerte, desenfoque de movimiento y escalas
  muy distintas a las de entrenamiento (unión < 8 px o > 400 px).
* Las uniones ocultas (BGA, BTC, QFN con almohadilla térmica) **no** son
  evaluables por inspección visual superficial.
* Es una **ayuda** de inspección: no sustituye la verificación humana ni los
  ensayos eléctricos/funcionales.
* Las clases con menos ejemplos reales de soldadura son las más débiles en
  precisión: `fail_insufficient` (P 0.32), `fail_void` (0.38) y
  `fail_bridge` (0.41) generan falsos positivos, sobre todo a baja confianza.
* El generador sintético etiqueta ahora cada puente con **una sola caja** que
  abarca ambas uniones (antes marcaba además la unión mojada como conforme); el
  modelo publicado se entrenó antes de ese ajuste, así que regenerar el dataset
  y reentrenar puede variar ligeramente las métricas de `fail_bridge`.

## Reproducir

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch numpy pillow onnx onnxruntime onnxscript pypdf pymupdf

python training/dataset.py --out training/data --boards 380 --val-boards 40
python training/train.py --data training/data --epochs 16 --size 384 \
    --out inspeccion/modelos/nexo-solder.onnx
python training/fix_valfull.py                # etiquetas de placas completas
python training/eval_onnx.py --model inspeccion/modelos/nexo-solder.onnx
python training/make_samples.py               # fotos de ejemplo de la app
```
