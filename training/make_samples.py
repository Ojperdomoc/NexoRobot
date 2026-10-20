"""Generate the demo PCB photos bundled with the web app.

    python training/make_samples.py

Writes inspeccion/muestras/placa-demo-0N.jpg and inspeccion/muestras/gt/...json.
The JSON holds the ground-truth boxes so the app can show precision/recall for
the demo images (button "Validar vs. etiquetas").
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcb_synth import CLASSES, add_camera_effects, render_board

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "inspeccion", "muestras")

SAMPLES = [
    # (id, seed, width, height, zoom, density, label)
    ("placa-demo-01", 313, 1600, 1100, 1.15, 1.0, "Placa de control · vista general"),
    ("placa-demo-02", 4711, 1500, 1100, 1.85, 0.85, "Módulo de potencia · macro"),
    ("placa-demo-03", 90210, 1680, 1120, 1.45, 1.15, "Placa mixta SMT / THT"),
]


def severity_of(name: str) -> str:
    if name == "ok":
        return "ok"
    return "warn" if name.startswith("warn") else "fail"


def class_name(joint) -> str | None:
    if joint.quality == "ok":
        return "ok"
    name = f"{joint.quality}_{joint.defect}"
    return name if name in CLASSES else None


def main():
    os.makedirs(os.path.join(OUT, "gt"), exist_ok=True)
    for sid, seed, w, h, zoom, density, label in SAMPLES:
        im, joints = render_board(seed, w, h, S=1, zoom=zoom, density=density)
        im = add_camera_effects(im, seed, jpeg_quality=90)
        if im.size != (w, h):
            im = im.resize((w, h))
        path = os.path.join(OUT, sid + ".jpg")
        im.save(path, quality=88, optimize=True)
        dets = []
        for j in joints:
            name = class_name(j)
            if not name:
                continue
            x0, y0, x1, y1 = j.box
            dets.append({
                "className": name,
                "severity": severity_of(name),
                "x": round(float(x0), 2),
                "y": round(float(y0), 2),
                "w": round(float(x1 - x0), 2),
                "h": round(float(y1 - y0), 2),
            })
        with open(os.path.join(OUT, "gt", sid + ".json"), "w") as f:
            json.dump({
                "id": sid,
                "label": label,
                "width": w,
                "height": h,
                "source": "render sintético (training/pcb_synth.py)",
                "detections": dets,
            }, f, indent=1)
        counts = {}
        for d in dets:
            counts[d["severity"]] = counts.get(d["severity"], 0) + 1
        print(sid, im.size, len(dets), counts, f"{os.path.getsize(path)/1024:.0f} KB")


if __name__ == "__main__":
    main()
