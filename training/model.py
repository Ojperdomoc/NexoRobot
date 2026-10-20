"""Compact anchor-free detector for PCB solder joints (PyTorch).

The network is a MobileNet-style backbone with a light FPN and YOLO-style
decoupled heads at strides 8, 16 and 32.  It is small enough to run in the
browser through ONNX Runtime Web (WASM) and is exported to
../../inspeccion/modelos/nexo-solder.onnx by train.py.
"""
from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 9
STRIDES = (8, 16, 32)


def _bn(c: int) -> nn.BatchNorm2d:
    return nn.BatchNorm2d(c, eps=1e-3, momentum=0.03)


class ConvBNAct(nn.Sequential):
    def __init__(self, cin, cout, k=3, s=1, act=True):
        layers = [nn.Conv2d(cin, cout, k, s, k // 2, bias=False), _bn(cout)]
        if act:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class DWConv(nn.Sequential):
    """Depthwise separable convolution."""

    def __init__(self, cin, cout, k=3, s=1):
        super().__init__(
            nn.Conv2d(cin, cin, k, s, k // 2, groups=cin, bias=False),
            _bn(cin),
            nn.SiLU(inplace=True),
            nn.Conv2d(cin, cout, 1, 1, 0, bias=False),
            _bn(cout),
            nn.SiLU(inplace=True),
        )


class Block(nn.Sequential):
    """Inverted-residual-ish block: DW conv + residual when shapes match."""

    def __init__(self, c: int):
        super().__init__(DWConv(c, c), DWConv(c, c))

    def forward(self, x):
        return x + super().forward(x)


class Neck(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.lat = ConvBNAct(cin, cout, 1)
        self.smooth = DWConv(cout, cout)

    def forward(self, x, up=None):
        x = self.lat(x)
        if up is not None:
            x = x + F.interpolate(up, size=x.shape[-2:], mode="nearest")
        return self.smooth(x)


class Head(nn.Module):
    """Decoupled classification / regression head with 5+nc outputs."""

    def __init__(self, c: int, nc: int, width: int = 64):
        super().__init__()
        self.cls = nn.Sequential(DWConv(c, width), nn.Conv2d(width, nc, 1))
        self.reg = nn.Sequential(DWConv(c, width), nn.Conv2d(width, 4, 1))
        # start with a low objectness prior
        nn.init.constant_(self.cls[-1].bias, -4.0)
        nn.init.zeros_(self.reg[-1].weight)
        nn.init.constant_(self.reg[-1].bias, 0.0)

    def forward(self, x):
        return self.cls(x), self.reg(x)


class SolderNet(nn.Module):
    def __init__(self, nc: int = NUM_CLASSES, width: float = 1.0):
        super().__init__()
        w = lambda v: max(8, int(v * width))
        self.nc = nc
        self.stem = nn.Sequential(ConvBNAct(3, w(8), 3, 2), DWConv(w(8), w(32), 3, 2))     # /4
        self.s4 = Block(w(24))
        self.p4 = DWConv(w(32), w(24))
        self.s8 = nn.Sequential(DWConv(w(24), w(48), 3, 2), Block(w(48)), Block(w(48)))
        self.s16 = nn.Sequential(DWConv(w(48), w(96), 3, 2), Block(w(96)), Block(w(96)))
        self.s32 = nn.Sequential(DWConv(w(96), w(160), 3, 2), Block(w(160)), Block(w(160)))
        c = w(96)
        self.n32 = Neck(w(160), c)
        self.n16 = Neck(w(96), c)
        self.n8 = Neck(w(48), c)
        self.h8 = Head(c, nc)
        self.h16 = Head(c, nc)
        self.h32 = Head(c, nc)

    def forward(self, x):
        x = self.s4(self.p4(self.stem(x)))      # /4
        f8 = self.s8(x)
        f16 = self.s16(f8)
        f32 = self.s32(f16)
        p32 = self.n32(f32)
        p16 = self.n16(f16, p32)
        p8 = self.n8(f8, p16)
        outs = []
        for head, f in ((self.h8, p8), (self.h16, p16), (self.h32, p32)):
            cls, reg = head(f)
            outs.append(torch.cat([reg, cls], 1))   # [B, 4+nc, H, W]
        return outs


def decode(pred: torch.Tensor, stride: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return boxes (cx, cy, w, h in input px) and class logits for one level."""
    B, C, H, W = pred.shape
    reg = pred[:, :4]
    cls = pred[:, 4:]
    ys = torch.arange(H, device=pred.device, dtype=pred.dtype)
    xs = torch.arange(W, device=pred.device, dtype=pred.dtype)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    ox = reg[:, 0].sigmoid()
    oy = reg[:, 1].sigmoid()
    tw = reg[:, 2].clamp(-6, 6)
    th = reg[:, 3].clamp(-6, 6)
    cx = (gx + ox) * stride
    cy = (gy + oy) * stride
    bw = tw.exp() * stride
    bh = th.exp() * stride
    return torch.stack([cx, cy, bw, bh], 1), cls


def box_iou(a, b) -> torch.Tensor:
    """IoU between [N,4] and [M,4] boxes in (cx,cy,w,h)."""
    ax0, ay0 = a[:, 0] - a[:, 2] / 2, a[:, 1] - a[:, 3] / 2
    ax1, ay1 = a[:, 0] + a[:, 2] / 2, a[:, 1] + a[:, 3] / 2
    bx0, by0 = b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2
    bx1, by1 = b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2
    ix0 = torch.maximum(ax0[:, None], bx0[None])
    iy0 = torch.maximum(ay0[:, None], by0[None])
    ix1 = torch.minimum(ax1[:, None], bx1[None])
    iy1 = torch.minimum(ay1[:, None], by1[None])
    iw = (ix1 - ix0).clamp(min=0)
    ih = (iy1 - iy0).clamp(min=0)
    inter = iw * ih
    area_a = (ax1 - ax0).clamp(min=0) * (ay1 - ay0).clamp(min=0)
    area_b = (bx1 - bx0).clamp(min=0) * (by1 - by0).clamp(min=0)
    return inter / (area_a[:, None] + area_b[None] - inter + 1e-7)


def level_for_size(size: float) -> int:
    if size < 26:
        return 0
    if size < 72:
        return 1
    return 2


def build_targets(targets: List[torch.Tensor], img_size: int, device):
    """Assign ground-truth boxes to grid cells.

    targets: list of [N,5] tensors (class, cx, cy, w, h) in pixels.
    Returns per level: cls target [B,H,W,nc] float, pos mask, box target [B,H,W,4]
    in (gx, gy, log(w), log(h)) parameterisation.
    """
    B = len(targets)
    out = []
    for li, s in enumerate(STRIDES):
        H = W = img_size // s
        cls_t = torch.zeros(B, H, W, NUM_CLASSES, device=device)
        pos = torch.zeros(B, H, W, device=device)
        box_t = torch.zeros(B, H, W, 4, device=device)
        for b, t in enumerate(targets):
            if t.numel() == 0:
                continue
            t = t.to(device)
            for row in t:
                c = int(row[0].item())
                cx, cy, bw, bh = [float(v) for v in row[1:5]]
                if level_for_size(max(bw, bh)) != li:
                    continue
                gx, gy = cx / s, cy / s
                ix, iy = int(gx), int(gy)
                if not (0 <= ix < W and 0 <= iy < H):
                    continue
                cls_t[b, iy, ix, c] = 1.0
                pos[b, iy, ix] = 1.0
                box_t[b, iy, ix, 0] = gx - ix
                box_t[b, iy, ix, 1] = gy - iy
                box_t[b, iy, ix, 2] = math.log(max(bw, 1.0) / s)
                box_t[b, iy, ix, 3] = math.log(max(bh, 1.0) / s)
        out.append((cls_t, pos, box_t))
    return out
