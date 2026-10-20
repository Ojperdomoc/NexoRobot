"""Regenerate the full-board validation labels (they must use width/height).

Early versions of dataset.py normalised the y coordinate by the width, which is
only correct for square crops.  This script re-renders the same deterministic
board seeds and rewrites images/labels/valfull with the correct normalisation.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataset import yolo_lines  # noqa: E402
from pcb_synth import add_camera_effects, render_board  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def main(val_boards=48, offset=0):
    img_dir = os.path.join(DATA, "images", "valfull")
    lbl_dir = os.path.join(DATA, "labels", "valfull")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for k in range(max(6, val_boards // 6)):
        seed = offset + 90000 + k + 1
        rng = np.random.default_rng(seed * 7919 + 13)
        W = int(rng.choice([1024, 1280, 1536]))
        H = int(rng.choice([768, 896, 1024]))
        zoom = float(rng.choice([0.85, 1.0, 1.15, 1.35, 1.6, 1.85]))
        im, joints = render_board(seed, W, H, S=1, zoom=zoom,
                                  density=float(rng.uniform(0.55, 1.25)))
        im = add_camera_effects(im, seed, jpeg_quality=int(rng.integers(78, 94)))
        name = f"board_{seed:05d}"
        im.save(os.path.join(img_dir, name + ".jpg"), quality=90)
        with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
            f.write("\n".join(yolo_lines(joints, W, H)))
        print(name, im.size, len(joints), flush=True)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--val-boards", type=int, default=48)
    ap.add_argument("--offset", type=int, default=0)
    a = ap.parse_args()
    main(a.val_boards, a.offset)
