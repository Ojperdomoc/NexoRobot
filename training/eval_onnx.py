"""Evaluate the exported ONNX model exactly like the web app does.

Runs the tiled browser-style pipeline (tile crops -> ONNX -> decode -> NMS)
over the full-board validation images and reports precision / recall / F1 and
per-class statistics.  Optionally writes annotated previews.

    python training/eval_onnx.py --model inspeccion/modelos/nexo-solder.onnx
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcb_synth import CLASSES  # noqa: E402

COLORS = {
    "ok": (34, 197, 94),
    "warn": (245, 158, 11),
    "fail": (239, 68, 68),
}


def severity(name):
    return "ok" if name == "ok" else "warn" if name.startswith("warn") else "fail"


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def nms(boxes, scores, iou_thr=0.45, max_out=600):
    if len(boxes) == 0:
        return []
    order = scores.argsort()[::-1]
    keep = []
    suppressed = np.zeros(len(order), dtype=bool)
    for i in range(len(order)):
        if suppressed[i]:
            continue
        a = order[i]
        keep.append(a)
        if len(keep) >= max_out:
            break
        b = order[i + 1:]
        if len(b) == 0:
            break
        xx0 = np.maximum(boxes[a, 0], boxes[b, 0])
        yy0 = np.maximum(boxes[a, 1], boxes[b, 1])
        xx1 = np.minimum(boxes[a, 2], boxes[b, 2])
        yy1 = np.minimum(boxes[a, 3], boxes[b, 3])
        iw = np.clip(xx1 - xx0, 0, None)
        ih = np.clip(yy1 - yy0, 0, None)
        inter = iw * ih
        aa = (boxes[a, 2] - boxes[a, 0]) * (boxes[a, 3] - boxes[a, 1])
        bb = (boxes[b, 2] - boxes[b, 0]) * (boxes[b, 3] - boxes[b, 1])
        iou = inter / np.maximum(aa + bb - inter, 1e-9)
        suppressed[i + 1:][iou > iou_thr] = True
    return keep


class Detector:
    def __init__(self, model, meta):
        self.sess = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
        self.meta = meta
        self.size = meta["size"]
        self.strides = meta["strides"]
        self.classes = meta["classes"]
        self.mean = np.array(meta["mean"], dtype=np.float32)
        self.std = np.array(meta["std"], dtype=np.float32)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [o.name for o in self.sess.get_outputs()]

    def preprocess(self, im: Image.Image) -> np.ndarray:
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
        a = (a - self.mean) / self.std
        return np.transpose(a, (2, 0, 1))[None].astype(np.float32)

    def infer_tile(self, im: Image.Image):
        outs = self.sess.run(self.output_names, {self.input_name: self.preprocess(im)})
        dets = []
        scale = im.width / self.size
        for li, o in enumerate(outs):
            stride = self.strides[li]
            _, C, H, W = o.shape
            o = o[0]
            plane = H * W
            probs = sigmoid(o[4:].reshape(C - 4, plane)).T  # (HW, nc)
            best = probs.argmax(1)
            score = probs.max(1)
            m = score > 0.08
            if not m.any():
                continue
            idx = np.nonzero(m)[0]
            ys, xs = np.divmod(idx, W)
            ox = sigmoid(o[0].reshape(-1)[idx])
            oy = sigmoid(o[1].reshape(-1)[idx])
            tw = np.clip(o[2].reshape(-1)[idx], -6, 6)
            th = np.clip(o[3].reshape(-1)[idx], -6, 6)
            cx = (xs + ox) * stride
            cy = (ys + oy) * stride
            bw = np.exp(tw) * stride
            bh = np.exp(th) * stride
            for k in range(len(idx)):
                dets.append({
                    "className": self.classes[int(best[idx[k]])],
                    "score": float(score[idx[k]]),
                    "x": float((cx[k] - bw[k] / 2) * scale),
                    "y": float((cy[k] - bh[k] / 2) * scale),
                    "w": float(bw[k] * scale),
                    "h": float(bh[k] * scale),
                })
        return dets

    def analyze(self, im: Image.Image, tile=None, overlap=0.22):
        W, H = im.size
        tile = tile or choose_tile(W, H, 9, self.size)
        tile = int(min(max(96, tile), self.size, max(W, H)))
        step = max(64, int(round(tile * (1 - overlap))))
        xs = list(range(0, max(1, W - tile + 1), step)) or [0]
        if xs[-1] + tile < W:
            xs.append(max(0, W - tile))
        ys = list(range(0, max(1, H - tile + 1), step)) or [0]
        if ys[-1] + tile < H:
            ys.append(max(0, H - tile))
        all_dets = []
        for y in ys:
            for x in xs:
                sx = min(max(0, x), max(0, W - tile))
                sy = min(max(0, y), max(0, H - tile))
                crop = im.crop((sx, sy, sx + tile, sy + tile))
                for d in self.infer_tile(crop):
                    d["x"] += sx
                    d["y"] += sy
                    all_dets.append(d)
        boxes = np.array([[d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]] for d in all_dets])
        scores = np.array([d["score"] for d in all_dets])
        keep = nms(boxes, scores) if len(all_dets) else []
        return [all_dets[i] for i in keep], tile


def choose_tile(w, h, max_tiles, model_size=384):
    """Las teselas se analizan a escala nativa (<= entrada del modelo)."""
    return int(max(96, min(model_size, max(w, h))))


def iou_xywh(a, b):
    ax2, ay2 = a["x"] + a["w"], a["y"] + a["h"]
    bx2, by2 = b["x"] + b["w"], b["y"] + b["h"]
    iw = max(0.0, min(ax2, bx2) - max(a["x"], b["x"]))
    ih = max(0.0, min(ay2, by2) - max(a["y"], b["y"]))
    inter = iw * ih
    u = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / u if u > 0 else 0.0


def evaluate(dets, gt, conf=0.35, iou_thr=0.5):
    dets = [d for d in dets if d["score"] >= conf]
    used = np.zeros(len(gt), dtype=bool)
    tp = np.zeros(len(CLASSES))
    fp = np.zeros(len(CLASSES))
    cls_ok = np.zeros(len(CLASSES))
    for d in sorted(dets, key=lambda x: -x["score"]):
        best, best_iou = -1, iou_thr
        for i, g in enumerate(gt):
            if used[i]:
                continue
            v = iou_xywh(d, {"x": g["x"], "y": g["y"], "w": g["w"], "h": g["h"]})
            if v > best_iou:
                best, best_iou = i, v
        ci = CLASSES.index(d["className"])
        if best >= 0:
            used[best] = True
            tp[ci] += 1
            if gt[best]["className"] == d["className"]:
                cls_ok[ci] += 1
        else:
            fp[ci] += 1
    fn = np.zeros(len(CLASSES))
    for i, g in enumerate(gt):
        if not used[i]:
            fn[CLASSES.index(g["className"])] += 1
    p = tp.sum() / max(tp.sum() + fp.sum(), 1)
    r = tp.sum() / max(tp.sum() + fn.sum(), 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    per = {}
    for i, c in enumerate(CLASSES):
        pp = tp[i] / max(tp[i] + fp[i], 1e-9)
        rr = tp[i] / max(tp[i] + fn[i], 1e-9)
        per[c] = {
            "p": round(float(pp), 3),
            "r": round(float(rr), 3),
            "f1": round(float(2 * pp * rr / max(pp + rr, 1e-9)), 3),
            "support": int(tp[i] + fn[i]),
            "cls_acc": round(float(cls_ok[i] / max(tp[i], 1e-9)), 3),
        }
    return {
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
        "detections": len(dets),
        "gt": len(gt),
        "per_class": per,
    }


def annotate(im: Image.Image, dets, gt, out_path, conf=0.35):
    scale = 1.0
    canvas = im.copy().convert("RGB")
    d = ImageDraw.Draw(canvas)
    for g in gt:
        d.rectangle([g["x"] * scale, g["y"] * scale, (g["x"] + g["w"]) * scale, (g["y"] + g["h"]) * scale],
                    outline=(90, 90, 90), width=1)
    for det in dets:
        if det["score"] < conf:
            continue
        col = COLORS[severity(det["className"])]
        d.rectangle([det["x"], det["y"], det["x"] + det["w"], det["y"] + det["h"]], outline=col, width=2)
        d.text((det["x"] + 2, max(0, det["y"] - 9)), det["className"], fill=col)
    canvas.save(out_path, quality=88)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="inspeccion/modelos/nexo-solder.onnx")
    ap.add_argument("--data", default="training/data")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--tiles", type=int, default=9)
    ap.add_argument("--out", default="/tmp/eval")
    args = ap.parse_args()
    meta = json.load(open(os.path.splitext(args.model)[0] + ".json"))
    det = Detector(args.model, meta)
    os.makedirs(args.out, exist_ok=True)
    img_dir = os.path.join(args.data, "images", "valfull")
    lbl_dir = os.path.join(args.data, "labels", "valfull")
    names = sorted(n[:-4] for n in os.listdir(img_dir) if n.endswith(".jpg"))[: args.limit]
    agg = {"precision": [], "recall": [], "f1": []}
    totals = {"tp": 0, "fp": 0, "fn": 0}
    for name in names:
        im = Image.open(os.path.join(img_dir, name + ".jpg")).convert("RGB")
        W, H = im.size
        gt = []
        with open(os.path.join(lbl_dir, name + ".txt")) as f:
            for line in f:
                p = line.split()
                if len(p) != 5:
                    continue
                c, cx, cy, w, h = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                gt.append({"className": CLASSES[c], "x": (cx - w / 2) * W, "y": (cy - h / 2) * H,
                           "w": w * W, "h": h * H})
        dets, tile = det.analyze(im, tile=choose_tile(W, H, args.tiles))
        res = evaluate(dets, gt, conf=args.conf)
        print(name, f"tile={tile}", json.dumps({k: res[k] for k in ("precision", "recall", "f1")}),
              "dets", res["detections"], "gt", res["gt"], flush=True)
        for k in agg:
            agg[k].append(res[k])
        annotate(im, dets, gt, os.path.join(args.out, name + "_pred.jpg"), conf=args.conf)
        # per-class aggregation
        for c, st in res["per_class"].items():
            pass
    print("mean:", {k: round(float(np.mean(v)), 4) for k, v in agg.items()})
    # final per-class over all boards
    full = {"tp": np.zeros(len(CLASSES)), "fp": np.zeros(len(CLASSES)), "fn": np.zeros(len(CLASSES))}
    for name in names:
        im = Image.open(os.path.join(img_dir, name + ".jpg")).convert("RGB")
        W, H = im.size
        gt = []
        with open(os.path.join(lbl_dir, name + ".txt")) as f:
            for line in f:
                p = line.split()
                if len(p) != 5:
                    continue
                c, cx, cy, w, h = int(p[0]), float(p[1]), float(p[2]), float(p[3]), float(p[4])
                gt.append({"className": CLASSES[c], "x": (cx - w / 2) * W, "y": (cy - h / 2) * H,
                           "w": w * W, "h": h * H})
        dets, _ = det.analyze(im, tile=choose_tile(W, H, args.tiles))
        dets = [d for d in dets if d["score"] >= args.conf]
        used = np.zeros(len(gt), dtype=bool)
        for d in sorted(dets, key=lambda x: -x["score"]):
            best, bi = -1, 0.5
            for i, g in enumerate(gt):
                if used[i]:
                    continue
                v = iou_xywh(d, {"x": g["x"], "y": g["y"], "w": g["w"], "h": g["h"]})
                if v > bi:
                    best, bi = i, v
            ci = CLASSES.index(d["className"])
            if best >= 0:
                used[best] = True
                full["tp"][ci] += 1
            else:
                full["fp"][ci] += 1
        for i, g in enumerate(gt):
            if not used[i]:
                full["fn"][CLASSES.index(g["className"])] += 1
    print("\nper-class (conf %.2f):" % args.conf)
    for i, c in enumerate(CLASSES):
        tp, fp, fn = full["tp"][i], full["fp"][i], full["fn"][i]
        p = tp / max(tp + fp, 1e-9)
        r = tp / max(tp + fn, 1e-9)
        f1 = 2 * p * r / max(p + r, 1e-9)
        print(f"  {c:18s} P {p:.3f} R {r:.3f} F1 {f1:.3f}  n={int(tp+fn)}")
    tp = full["tp"].sum(); fp = full["fp"].sum(); fn = full["fn"].sum()
    p = tp / max(tp + fp, 1e-9); r = tp / max(tp + fn, 1e-9)
    print(f"  overall            P {p:.3f} R {r:.3f} F1 {2*p*r/max(p+r,1e-9):.3f}")


if __name__ == "__main__":
    main()
