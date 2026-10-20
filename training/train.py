"""Train the solder-joint detector and export it to ONNX for the web app.

    python training/train.py --data training/data --epochs 26 --size 448

The exported model is written to inspeccion/modelos/nexo-solder.onnx plus a
JSON side-car with class names, colours and the input size used for training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import (NUM_CLASSES, STRIDES, SolderNet, box_iou, build_targets, decode,
                   level_for_size)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class JointDataset(Dataset):
    def __init__(self, root: str, split: str, size: int, train: bool, limit: int = 0):
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        self.names = sorted(n[:-4] for n in os.listdir(self.img_dir) if n.endswith(".jpg"))
        if limit:
            random.Random(1).shuffle(self.names)
            self.names = self.names[:limit]
        self.size = size
        self.train = train

    def __len__(self):
        return len(self.names)

    def _load(self, i):
        name = self.names[i]
        im = Image.open(os.path.join(self.img_dir, name + ".jpg")).convert("RGB")
        labels = []
        with open(os.path.join(self.lbl_dir, name + ".txt")) as f:
            for line in f:
                p = line.split()
                if len(p) == 5:
                    labels.append([float(v) for v in p])
        return im, np.array(labels, dtype=np.float32).reshape(-1, 5)

    def __getitem__(self, i):
        im, labels = self._load(i)
        W, H = im.size
        scale = self.size / W
        if self.train:
            scale *= float(np.random.uniform(0.82, 1.18))
        nw, nh = max(32, int(round(W * scale))), max(32, int(round(H * scale)))
        im = im.resize((nw, nh), Image.BICUBIC)

        boxes = labels[:, 1:5] * np.array([W, H, W, H], dtype=np.float32)
        boxes *= scale

        if self.train:
            # colour / lens augmentations
            if random.random() < 0.8:
                im = ImageEnhance.Brightness(im).enhance(random.uniform(0.75, 1.25))
            if random.random() < 0.8:
                im = ImageEnhance.Contrast(im).enhance(random.uniform(0.8, 1.3))
            if random.random() < 0.6:
                im = ImageEnhance.Color(im).enhance(random.uniform(0.7, 1.35))
            if random.random() < 0.35:
                im = im.filter(ImageFilter.GaussianBlur(random.uniform(0.3, 1.4)))
            if random.random() < 0.3:
                a = np.array(im).astype(np.float32)
                a += np.random.normal(0, random.uniform(1.5, 6), a.shape)
                im = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
            if random.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
                boxes[:, 0] = nw - boxes[:, 0]
            if random.random() < 0.3:
                im = im.transpose(Image.FLIP_TOP_BOTTOM)
                boxes[:, 1] = nh - boxes[:, 1]

        # letterbox into size x size
        canvas = Image.new("RGB", (self.size, self.size), (114, 124, 118))
        ox = (self.size - nw) // 2
        oy = (self.size - nh) // 2
        canvas.paste(im, (ox, oy))
        boxes[:, 0] += ox
        boxes[:, 1] += oy

        arr = np.asarray(canvas, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        arr = np.transpose(arr, (2, 0, 1))

        keep = ((boxes[:, 0] > 2) & (boxes[:, 0] < self.size - 2) &
                (boxes[:, 1] > 2) & (boxes[:, 1] < self.size - 2) &
                (boxes[:, 2] > 3) & (boxes[:, 3] > 3))
        labels = labels[keep]
        boxes = boxes[keep]
        tgt = np.concatenate([labels[:, :1], boxes], axis=1).astype(np.float32)
        return torch.from_numpy(arr), torch.from_numpy(tgt), name_id(i)


def name_id(i):
    return i


def collate(batch):
    xs, ys, ids = zip(*batch)
    return torch.stack(xs), list(ys), ids


def make_targets(targets, size, device):
    return build_targets(targets, size, device)


def focal_bce(logits, target, alpha=0.7, gamma=2.0):
    p = logits.sigmoid()
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * target + (1 - alpha) * (1 - target)
        loss = alpha_t * loss
    return loss


def compute_loss(outs, targets, size, device):
    tgt_levels = make_targets(targets, size, device)
    cls_sum = 0.0
    reg_sum = 0.0
    n_pos = sum(int((t[1] > 0.5).sum()) for t in tgt_levels)
    norm = max(n_pos, 1)
    for li, (pred, (cls_t, pos, box_t)) in enumerate(zip(outs, tgt_levels)):
        stride = STRIDES[li]
        B, C, H, W = pred.shape
        cls_logits = pred[:, 4:].permute(0, 2, 3, 1)          # B,H,W,nc
        cls_t = cls_t * 1.0
        cls_sum = cls_sum + focal_bce(cls_logits, cls_t).sum()

        boxes, _ = decode(pred, stride)
        boxes = boxes.permute(0, 2, 3, 1)                     # B,H,W,4
        m = pos > 0.5
        if m.any():
            pb = boxes[m]
            tb = torch.stack([
                (box_t[..., 0] + torch.arange(W, device=device)[None, None, :])[m] * stride,
                (box_t[..., 1] + torch.arange(H, device=device)[None, :, None])[m] * stride,
                box_t[..., 2][m].exp() * stride,
                box_t[..., 3][m].exp() * stride,
            ], 1)
            iou = box_iou(pb.detach(), tb).diagonal().clamp(0, 1)
            giou = giou_loss(pb, tb)
            reg_sum = reg_sum + (giou * (0.5 + iou)).sum()
    return cls_sum / norm, reg_sum / norm


def giou_loss(pred, target, eps=1e-7):
    px0, py0 = pred[:, 0] - pred[:, 2] / 2, pred[:, 1] - pred[:, 3] / 2
    px1, py1 = pred[:, 0] + pred[:, 2] / 2, pred[:, 1] + pred[:, 3] / 2
    tx0, ty0 = target[:, 0] - target[:, 2] / 2, target[:, 1] - target[:, 3] / 2
    tx1, ty1 = target[:, 0] + target[:, 2] / 2, target[:, 1] + target[:, 3] / 2
    iw = (torch.minimum(px1, tx1) - torch.maximum(px0, tx0)).clamp(min=0)
    ih = (torch.minimum(py1, ty1) - torch.maximum(py0, ty0)).clamp(min=0)
    inter = iw * ih
    ap = (px1 - px0).clamp(min=0) * (py1 - py0).clamp(min=0)
    at = (tx1 - tx0).clamp(min=0) * (ty1 - ty0).clamp(min=0)
    union = ap + at - inter + eps
    cw = torch.maximum(px1, tx1) - torch.minimum(px0, tx0)
    ch = torch.maximum(py1, ty1) - torch.minimum(py0, ty0)
    enclose = cw.clamp(min=0) * ch.clamp(min=0) + eps
    iou = inter / union
    return 1 - iou + (enclose - union) / enclose


@torch.no_grad()
def evaluate(model, loader, size, device, conf=0.25, iou_thr=0.5):
    model.eval()
    stats = {"tp": np.zeros(NUM_CLASSES), "fp": np.zeros(NUM_CLASSES), "fn": np.zeros(NUM_CLASSES)}
    for x, targets, _ in loader:
        x = x.to(device)
        outs = model(x)
        preds = decode_all(outs, conf)
        for b in range(x.shape[0]):
            gt = targets[b].numpy()
            pr = preds[b]
            _match(pr, gt, stats, iou_thr)
    model.train()
    tp, fp, fn = stats["tp"], stats["fp"], stats["fn"]
    prec = tp / np.maximum(tp + fp, 1e-9)
    rec = tp / np.maximum(tp + fn, 1e-9)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    overall = {
        "precision": float(tp.sum() / max(tp.sum() + fp.sum(), 1)),
        "recall": float(tp.sum() / max(tp.sum() + fn.sum(), 1)),
        "f1": float(2 * tp.sum() / max(2 * tp.sum() + fp.sum() + fn.sum(), 1)),
    }
    per_class = {str(i): {"p": float(prec[i]), "r": float(rec[i]), "f1": float(f1[i]),
                          "n": int(tp[i] + fn[i])} for i in range(NUM_CLASSES)}
    return overall, per_class


def nms_numpy(boxes, scores, iou_thr=0.5):
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx0 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy0 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx1 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy1 = np.minimum(boxes[i, 3], boxes[rest, 3])
        w = np.clip(xx1 - xx0, 0, None)
        h = np.clip(yy1 - yy0, 0, None)
        inter = w * h
        a = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        b = (boxes[rest, 2] - boxes[rest, 0]) * (boxes[rest, 3] - boxes[rest, 1])
        iou = inter / np.maximum(a + b - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


@torch.no_grad()
def decode_all(outs, conf=0.25):
    """Decode raw heads into per-image [class, score, x0, y0, x1, y1] arrays."""
    B = outs[0].shape[0]
    per = []
    for b in range(B):
        rows = []
        for li, pred in enumerate(outs):
            s = STRIDES[li]
            p = pred[b:b + 1]
            boxes, cls = decode(p, s)
            prob = cls.sigmoid()
            score, cls_id = prob.max(1)
            score = score[0].cpu().numpy()
            cls_id = cls_id[0].cpu().numpy()
            bx = boxes[0].cpu().numpy()
            m = score > conf
            if not m.any():
                continue
            cx, cy, w, h = bx[:, m]
            rows.append(np.stack([cls_id[m], score[m], cx - w / 2, cy - h / 2,
                                  cx + w / 2, cy + h / 2], 1))
        if rows:
            rows = np.concatenate(rows, 0)
            keep = []
            for c in np.unique(rows[:, 0]):
                idx = np.where(rows[:, 0] == c)[0]
                k = nms_numpy(rows[idx, 2:6], rows[idx, 1], 0.55)
                keep.extend(idx[k].tolist())
            rows = rows[np.array(keep)]
            per.append(rows)
        else:
            per.append(np.zeros((0, 6), dtype=np.float32))
    return per


def _match(pred, gt, stats, iou_thr):
    used = np.zeros(len(gt), dtype=bool)
    for row in pred:
        c = int(row[0])
        best, best_iou = -1, iou_thr
        px = np.array([row[2], row[3], row[4], row[5]])
        for gi in range(len(gt)):
            if used[gi]:
                continue
            gc = int(gt[gi, 0])
            gx = np.array([gt[gi, 1] - gt[gi, 3] / 2, gt[gi, 2] - gt[gi, 4] / 2,
                           gt[gi, 1] + gt[gi, 3] / 2, gt[gi, 2] + gt[gi, 4] / 2])
            iw = max(0.0, min(px[2], gx[2]) - max(px[0], gx[0]))
            ih = max(0.0, min(px[3], gx[3]) - max(px[1], gx[1]))
            inter = iw * ih
            a = (px[2] - px[0]) * (px[3] - px[1])
            b = (gx[2] - gx[0]) * (gx[3] - gx[1])
            iou = inter / max(a + b - inter, 1e-9)
            if gc == c and iou > best_iou:
                best, best_iou = gi, iou
        if best >= 0:
            used[best] = True
            stats["tp"][c] += 1
        else:
            stats["fp"][c] += 1
    for gi in range(len(gt)):
        if not used[gi]:
            stats["fn"][int(gt[gi, 0])] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="training/data")
    ap.add_argument("--out", default="inspeccion/modelos/nexo-solder.onnx")
    ap.add_argument("--epochs", type=int, default=26)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--width", type=float, default=1.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--resume", default="")
    args = ap.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.set_num_threads(os.cpu_count() or 1)
    device = "cpu"

    ds = JointDataset(args.data, "train", args.size, True, args.limit)
    val = JointDataset(args.data, "val", args.size, False, max(200, args.limit // 8) if args.limit else 600)
    print(f"train crops: {len(ds)}  val crops: {len(val)}", flush=True)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, collate_fn=collate,
                    num_workers=args.workers, drop_last=True)
    vl = DataLoader(val, batch_size=args.batch, shuffle=False, collate_fn=collate,
                    num_workers=args.workers)

    model = SolderNet(NUM_CLASSES, args.width)
    if args.resume and os.path.exists(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location="cpu"))
        print("resumed", args.resume, flush=True)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"params: {nparam/1e6:.3f}M", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = max(1, len(dl)) * args.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps,
                                                pct_start=0.15)
    model.train()
    best = -1.0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    ckpt = os.path.join(os.path.dirname(args.out) or ".", "nexo-solder.pt")
    step = 0
    t0 = time.time()
    for ep in range(args.epochs):
        agg = [0.0, 0.0]
        for x, targets, _ in dl:
            outs = model(x)
            l_cls, l_reg = compute_loss(outs, targets, args.size, device)
            loss = l_cls + 2.0 * l_reg
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            if step < steps - 1:
                sched.step()
            step += 1
            agg[0] += float(l_cls)
            agg[1] += float(l_reg)
            if step % 25 == 0:
                el = time.time() - t0
                print(f"ep {ep+1}/{args.epochs} step {step}/{steps} "
                      f"cls {agg[0]/25:.3f} reg {agg[1]/25:.3f} "
                      f"{el/step*1000:.0f}ms/it", flush=True)
                agg = [0.0, 0.0]
        overall, per = evaluate(model, vl, args.size, device)
        print(f"== epoch {ep+1} val: {overall}", flush=True)
        print("   per class:", {k: round(v['f1'], 3) for k, v in per.items()}, flush=True)
        score = overall["f1"]
        if score > best:
            best = score
            torch.save(model.state_dict(), ckpt)
            with open(os.path.join(os.path.dirname(args.out) or ".", "metrics.json"), "w") as f:
                json.dump({"overall": overall, "per_class": per, "epoch": ep + 1,
                           "size": args.size, "params": nparam}, f, indent=2)
            print("   saved checkpoint", flush=True)

    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    export_onnx(model, args.out, args.size)
    print("exported", args.out, flush=True)


def export_onnx(model, path, size):
    dummy = torch.randn(1, 3, size, size)
    out = model(dummy)
    torch.onnx.export(
        model, dummy, path,
        input_names=["images"],
        output_names=[f"out{i}" for i in range(len(out))],
        dynamic_axes={"images": {2: "h", 3: "w"},
                      **{f"out{i}": {2: "oh{i}", 3: "ow{i}"} for i in range(len(out))}},
        opset_version=13,
        do_constant_folding=True,
        dynamo=False,
    )
    meta = {
        "classes": ["ok", "warn_dull", "warn_coverage", "fail_cold", "fail_insufficient",
                    "fail_nonwet", "fail_void", "fail_bridge", "fail_missing"],
        "strides": list(STRIDES),
        "size": size,
        "mean": MEAN.tolist(),
        "std": STD.tolist(),
        "inputs": ["images"],
        "outputs": [f"out{i}" for i in range(len(out))],
    }
    with open(os.path.splitext(path)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
