"""Export a trained checkpoint to ONNX (used by the app and by eval_onnx.py).

    python training/export_checkpoint.py --ckpt inspeccion/modelos/nexo-solder.pt \
        --out inspeccion/modelos/nexo-solder.onnx --size 384
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import NUM_CLASSES, SolderNet  # noqa: E402
from train import export_onnx  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="inspeccion/modelos/nexo-solder.pt")
    ap.add_argument("--out", default="inspeccion/modelos/nexo-solder.onnx")
    ap.add_argument("--size", type=int, default=384)
    ap.add_argument("--width", type=float, default=1.0)
    args = ap.parse_args()
    model = SolderNet(NUM_CLASSES, args.width)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    export_onnx(model, args.out, args.size)
    # side-car additions used by the UI
    meta_path = os.path.splitext(args.out)[0] + ".json"
    meta = json.load(open(meta_path))
    params = sum(p.numel() for p in model.parameters())
    meta["params"] = int(params)
    meta["arch"] = "nexo-soldernet"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print("exported", args.out, "params", params, "size", args.size)
    print("bytes", os.path.getsize(args.out))


if __name__ == "__main__":
    main()
