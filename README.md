# NexoRobot

**Nexo** is a self-contained, browser-based 3D playground for a humanoid robot. The published site needs no build step, backend, or third-party runtime assets: the complete experience is in `index.html`.

## Inspección de soldadura con IA (`inspeccion/`)

A second, independent app in this repository: it opens the camera, detects PCB
solder joints with a tiny YOLO-style model running **in the browser** (ONNX
Runtime Web / WASM) and labels each joint against **IPC-A-610G**:

* 🟢 green — conforming joint
* 🟠 orange — acceptable with observation (process indicator)
* 🔴 red — non-conforming (cold joint, insufficient solder, non-wetting, voids,
  bridge, missing solder)

It can capture the photo and generate a **PDF report** with the annotated image,
the findings summary, the priority joints, the components with observations and
actionable recommendations that cite the standard.

Open <https://ojperdomoc.github.io/NexoRobot/inspeccion/> (or
`http://localhost:8080/inspeccion/` when serving the repo locally). Everything —
inference and PDF — runs on the device; no image leaves the browser.
See [`inspeccion/README.md`](inspeccion/README.md) for details.

### Modelo y entrenamiento

The detector (0.30 M parameters, 384×384 input, 9 classes) is trained from
procedurally rendered PCB photos, because real solder-joint datasets are not
freely available:

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch numpy pillow onnx onnxruntime onnxscript

python training/dataset.py --out training/data --boards 380 --val-boards 40
python training/train.py --data training/data --epochs 16 --size 384 \
    --out inspeccion/modelos/nexo-solder.onnx
python training/fix_valfull.py     # full-board labels for evaluation
python training/eval_onnx.py       # browser-equivalent tiled evaluation
python training/make_samples.py    # demo photos for the web app
```

`training/pcb_synth.py` renders the synthetic boards (chips, SOIC/QFN,
electrolytic capacitors, connectors, through-hole parts, silkscreen, copper,
specular highlights and camera effects) with per-joint labels for *ok*,
*warn* and six defect types.

## GitHub Pages

The repository includes `.github/workflows/deploy-pages.yml`, which deploys the static entry point from `main` with the official GitHub Pages Actions workflow.

1. In the repository, open **Settings → Pages**.
2. Under **Build and deployment**, choose **GitHub Actions** as the source.
3. Merge or push this change to `main`. The **Deploy GitHub Pages** workflow will publish it. You can also run it manually from the **Actions** tab.

Once deployed, the project site is available at:

- <https://ojperdomoc.github.io/NexoRobot/>

The old `Nexo.html` URL remains as a lightweight redirect so existing links continue to reach the site.

### Local preview

Serve the repository directory over HTTP rather than opening the file directly:

```bash
python3 -m http.server 8080
```

Then visit <http://localhost:8080/>. A secure HTTPS origin, such as GitHub Pages, is required for the Meta Quest/WebXR experience.

## Interaction

- Use the action buttons or keys **1–5** to interact with Nexo.
- Use **W A S D** or the arrow keys to move.
- Click the ground to choose a destination, and use the command box for instructions such as `sígueme`, `baila`, or `trae el núcleo`.

## Deployment notes

- `index.html` is the canonical Pages entry point.
- `.nojekyll` prevents accidental Jekyll processing if the Pages publishing mode is ever changed.
- The sharing QR/link is derived from the current HTTPS URL, so it works for the GitHub Pages project path and custom domains without a hard-coded tunnel URL.
