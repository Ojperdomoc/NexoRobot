"""Build the synthetic training set for the PCB solder-joint detector.

Usage:
    python training/dataset.py --out training/data --boards 500 --crops 14

Outputs:
    training/data/images/train/*.jpg   512x512 crops
    training/data/labels/train/*.txt   YOLO labels (class cx cy w h, normalised)
    training/data/images/val/*.jpg
    training/data/labels/val/*.txt
    training/data/images/valfull/*.jpg  full board photos, for evaluation / demos
    training/data/labels/valfull/*.txt

Classes (see pcb_synth.CLASSES):
    0 ok                4 fail_insufficient
    1 warn_dull         5 fail_nonwet
    2 warn_coverage     6 fail_void
    3 fail_cold         7 fail_bridge
                        8 fail_missing
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcb_synth import (CLASS_ID, CLASSES, QUALITIES, add_camera_effects, render_board,
                       square_crop)


def class_index(joint) -> int | None:
    if joint.quality == "ok":
        return CLASS_ID["ok"]
    name = f"{joint.quality}_{joint.defect}"
    return CLASS_ID.get(name)


def yolo_lines(joints, size, height=None):
    """YOLO labels normalised by the image size (width, height)."""
    W = size
    H = height or size
    out = []
    seen = set()
    for j in joints:
        cid = class_index(j)
        if cid is None:
            continue
        x0, y0, x1, y1 = j.box
        cx, cy = (x0 + x1) / 2 / W, (y0 + y1) / 2 / H
        w, h = (x1 - x0) / W, (y1 - y0) / H
        if w <= 0.004 or h <= 0.004:
            continue
        key = (cid, round(cx * W / 2), round(cy * H / 2))
        if key in seen:
            continue
        seen.add(key)
        out.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    return out


def build_split(args, split: str, seeds):
    img_dir = os.path.join(args.out, "images", split)
    lbl_dir = os.path.join(args.out, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    stats = {c: 0 for c in CLASSES}
    n_crop = 0
    full_dir = os.path.join(args.out, "images", "valfull")
    full_lbl = os.path.join(args.out, "labels", "valfull")
    if split == "val":
        os.makedirs(full_dir, exist_ok=True)
        os.makedirs(full_lbl, exist_ok=True)

    for k, seed in enumerate(seeds):
        rng = np.random.default_rng(seed * 7919 + 13)
        W = int(rng.choice([1024, 1280, 1536]))
        H = int(rng.choice([768, 896, 1024]))
        zoom = float(rng.choice([0.85, 1.0, 1.15, 1.35, 1.6, 1.85]))
        im, joints = render_board(seed, W, H, S=1, zoom=zoom,
                                  density=float(rng.uniform(0.55, 1.25)))
        im = add_camera_effects(im, seed, jpeg_quality=int(rng.integers(78, 94)))
        if not joints:
            continue

        if split == "val" and k < max(6, len(seeds) // 6):
            name = f"board_{seed:05d}"
            im.save(os.path.join(full_dir, name + ".jpg"), quality=90)
            lines = yolo_lines(joints, W, H)
            with open(os.path.join(full_lbl, name + ".txt"), "w") as f:
                f.write("\n".join(lines))
            for j in joints:
                stats[CLASSES[class_index(j)] if class_index(j) is not None else "ok"] += 1

        centres = []
        for _ in range(args.crops):
            if rng.random() < 0.7:
                j = joints[int(rng.integers(0, len(joints)))]
                cx = int((j.box[0] + j.box[2]) / 2 + rng.normal(0, 70))
                cy = int((j.box[1] + j.box[3]) / 2 + rng.normal(0, 70))
            else:
                cx, cy = int(rng.integers(0, W)), int(rng.integers(0, H))
            centres.append((cx, cy))
        for cx, cy in centres:
            crop, cj = square_crop(im, joints, cx, cy, args.size)
            lines = yolo_lines(cj, args.size)
            if len(lines) < 2:
                continue
            n_crop += 1
            name = f"{split}_{seed:05d}_{n_crop:06d}"
            crop.save(os.path.join(img_dir, name + ".jpg"), quality=91)
            with open(os.path.join(lbl_dir, name + ".txt"), "w") as f:
                f.write("\n".join(lines))
            for j in cj:
                ci = class_index(j)
                if ci is not None:
                    stats[CLASSES[ci]] += 1
    return split, n_crop, stats


def _worker(payload):
    args, split, seeds = payload
    return build_split(args, split, seeds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="training/data")
    ap.add_argument("--boards", type=int, default=400)
    ap.add_argument("--val-boards", type=int, default=40)
    ap.add_argument("--crops", type=int, default=12)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--seed-offset", type=int, default=0)
    args = ap.parse_args()

    train_seeds = [args.seed_offset + i for i in range(1, args.boards + 1)]
    val_seeds = [args.seed_offset + 90000 + i for i in range(1, args.val_boards + 1)]
    chunks = []
    n = max(1, args.workers)
    for i in range(n):
        chunks.append((args, "train", train_seeds[i::n]))
    chunks.append((args, "val", val_seeds))
    with Pool(n) as pool:
        results = pool.map(_worker, chunks)

    total = {}
    for split, n_crop, stats in results:
        print(split, "crops:", n_crop, stats, flush=True)
        for q, v in stats.items():
            total[q] = total.get(q, 0) + v
    meta = {
        "classes": CLASSES,
        "num_crops": sum(r[1] for r in results),
        "joint_counts": total,
        "source": "synthetic PCB renderer (training/pcb_synth.py)",
    }
    with open(os.path.join(args.out, "dataset.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("dataset:", json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
