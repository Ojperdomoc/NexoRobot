"""Procedural PCB renderer.

Generates synthetic photos of printed circuit boards with labelled solder
joints in three visual quality classes:

    ok    - conforming joint (shiny, concave fillet, full wetting)
    warn  - acceptable but marginal / process indicator (dull, low coverage,
            excess solder, small void)
    fail  - defect (cold joint, insufficient solder, non-wetting, bridge,
            missing solder, large void)

The renderer is used to build the training set for the tiny YOLO detector
shipped with the web application (see train_yolo.py) and to create the
example board photos in ../inspeccion/assets/.

It is deliberately dependency-light: numpy + Pillow only.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = "/usr/share/fonts/truetype/dejavu"
FONT_BOLD = f"{FONT_DIR}/DejaVuSans-Bold.ttf"
FONT_REG = f"{FONT_DIR}/DejaVuSans.ttf"
FONT_MONO = f"{FONT_DIR}/DejaVuSansMono-Bold.ttf"

QUALITIES = ("ok", "warn", "fail")

# Detection classes: the first block is the IPC-A-610G verdict, the suffix
# names the visual condition.  The app maps each class to green / orange / red.
CLASSES = [
    "ok",                 # 0  conforme
    "warn_dull",          # 1  aceptable, superficie mate/granulosa (5.1)
    "warn_coverage",      # 2  aceptable, cobertura reducida (process indicator)
    "fail_cold",          # 3  unión fría (5.2.5)
    "fail_insufficient",  # 4  soldadura insuficiente / sin mojado completo (5.2.4)
    "fail_nonwet",        # 5  no humectación / dewetting (5.2.4, 5.2.6)
    "fail_void",          # 6  vacíos, poros, blowholes (5.2.2)
    "fail_bridge",        # 7  puente de soldadura (5.2.7.2)
    "fail_missing",       # 8  sin soldadura
]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}
SEVERITY = {name: ("ok" if name == "ok" else "warn" if name.startswith("warn") else "fail")
            for name in CLASSES}

MASK_COLORS = [
    (23, 78, 52),    # classic green
    (18, 66, 44),    # deep green
    (16, 58, 58),    # teal
    (22, 48, 72),    # blue
    (30, 34, 38),    # black
    (74, 26, 28),    # red
]

SILK = (232, 234, 226)
GOLD = (196, 162, 92)
GOLD_LIGHT = (232, 205, 146)
SILVER = (188, 192, 196)
SOLDER = (176, 180, 184)


@dataclass
class Joint:
    box: Tuple[float, float, float, float]  # x0, y0, x1, y1 (pixels)
    quality: str
    defect: str = "none"
    angle: float = 0.0  # 0 horizontal terminal axis, 90 vertical


@dataclass
class Component:
    kind: str
    box: Tuple[float, float, float, float]
    joints: List[Joint] = field(default_factory=list)
    refdes: str = ""
    body_color: Tuple[int, int, int] = (26, 26, 28)
    pins: int = 0


def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, max(6, int(size)))


def _rounded(draw: ImageDraw.ImageDraw, box, radius, fill=None, outline=None, width=1):
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return
    r = max(0, int(min(radius, (x1 - x0) / 2, (y1 - y0) / 2)))
    draw.rounded_rectangle([x0, y0, x1, y1], radius=r, fill=fill, outline=outline, width=width)


def _shade(height: np.ndarray, base, light=(-0.45, -0.55, 0.72), spec=42.0,
           spec_gain=0.9, ambient=0.55, rough=1.0, metallic=0.55):
    """Shade a height map into an RGB float image (values 0..255)."""
    gy, gx = np.gradient(height.astype(np.float32) * rough)
    nz = np.ones_like(gx) * 0.9
    n = np.stack([-gx, -gy, nz], axis=-1)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-6
    l = np.array(light, dtype=np.float32)
    l /= np.linalg.norm(l)
    ndl = np.clip((n * l).sum(-1), 0, 1)
    v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half = l + v
    half /= np.linalg.norm(half)
    ndh = np.clip((n * half).sum(-1), 0, 1)
    specular = (ndh ** spec) * spec_gain
    # metal tint modulated by lighting, plus a cool rim to fake reflections
    base_arr = np.array(base, dtype=np.float32)
    shade = 0.55 + 0.55 * ndl
    rim = np.clip(1.0 - n[..., 2], 0, 1) * 0.25
    rgb = base_arr[None, None, :] * shade[..., None]
    rgb += specular[..., None] * (255 - base_arr)[None, None, :] * 0.85
    rgb += rim[..., None] * np.array([60, 70, 80], dtype=np.float32)
    rgb = 255 - (255 - rgb) * (1.0 - 0.0)
    out = rgb * (ambient + (1 - ambient) * 0.9) + base_arr * 0.12
    return np.clip(out, 0, 255)


def _blob_height(w: int, h: int, spread: float, height_scale: float, rng: np.random.Generator,
                 roughness: float = 0.0, irregular: float = 0.0) -> np.ndarray:
    """A smooth meniscus-like height field used for solder fillets."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = (xx + 0.5) / max(w, 1)
    v = (yy + 0.5) / max(h, 1)
    # profile along x: rounded plateau that falls off at the wetting edge
    sx = np.clip((u - 0.5) / max(spread, 1e-3) + 0.5, 0, 1)
    edge = np.clip(np.sin(np.pi * np.clip(sx, 0, 1)), 0, 1) ** 0.6
    # across the width: rounded shoulders
    sy = np.clip((v - 0.5) / 0.5 + 0.5, 0, 1)
    shoulder = np.clip(np.sin(np.pi * np.clip(sy, 0, 1)), 0, 1) ** 0.35
    height = height_scale * edge * shoulder
    if irregular > 0:
        low = rng.normal(0, 1, (max(2, h // 5), max(2, w // 5)))
        low = np.array(Image.fromarray(((low - low.min()) / (np.ptp(low) + 1e-6) * 255).astype(np.uint8))
                       .resize((w, h), Image.BICUBIC), dtype=np.float32) / 255.0
        height *= (1.0 - irregular) + irregular * (0.45 + 0.9 * low)
    if roughness > 0:
        height += rng.normal(0, roughness, (h, w)).astype(np.float32)
    return height


def _paste(img: np.ndarray, patch: np.ndarray, x: int, y: int, alpha: np.ndarray | None = None):
    """Alpha composite a float RGB patch onto an RGB uint8 image."""
    h, w = patch.shape[:2]
    H, W = img.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - x, y0 - y
    p = patch[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)].astype(np.float32)
    a = 1.0 if alpha is None else alpha[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)][..., None]
    region = img[y0:y1, x0:x1].astype(np.float32)
    img[y0:y1, x0:x1] = np.clip(region * (1 - a) + p * a, 0, 255).astype(np.uint8)


def _fillet_field(w: int, h: int, rng: np.random.Generator, reach: float,
                  width: float, shiny: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Height + alpha of a solder meniscus hugging the left (component) edge.

    ``reach``  how far the solder flows over the land (0..1 of the patch width)
    ``width``  fraction of the patch height that is wetted
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    u = xx / max(w - 1, 1)                 # 0 at the component edge, 1 at the land tip
    v = (yy / max(h - 1, 1)) * 2 - 1       # -1..1 across the land
    # meniscus: tall at the component, feathered where it meets the land
    prof = np.clip(1 - (u / max(reach, 1e-3)) ** 1.15, 0, 1) ** 0.55
    # rounded shoulders; the wetting boundary is a smooth arc
    shoulder = np.clip(1 - (np.abs(v) / max(width, 1e-3)) ** 2.0, 0, 1) ** 0.38
    hgt = prof * shoulder
    # rounded crest so the highlight rolls across the bead
    crest = np.exp(-(((u - 0.30) / 0.26) ** 2 + ((np.abs(v) - 0.28) / 0.42) ** 2))
    hgt = hgt * (0.85 + 0.5 * crest)
    if shiny:
        hgt += rng.normal(0, 0.004, (h, w))
    else:
        low = rng.normal(0.0, 1.0, (max(2, h // 5), max(2, w // 5)))
        low = np.array(Image.fromarray(((low - low.min()) / (np.ptp(low) + 1e-6) * 255)
                                       .astype(np.uint8)).resize((w, h), Image.BICUBIC),
                       dtype=np.float32) / 255.0
        hgt = hgt * (0.82 + 0.36 * low) + rng.normal(0, 0.012, (h, w))
    # never let solder extend past the land tip
    hgt *= np.clip((1.0 - u) * 26 + 0.15, 0, 1)
    alpha = np.clip(hgt * 3.4, 0, 1) * np.clip(u * 14 + 0.35, 0, 1)
    return hgt, alpha


def _solder_patch(w: int, h: int, quality: str, rng: np.random.Generator,
                  spread: float, roughness: float, irregular: float,
                  seed_color=(186, 190, 194)) -> Tuple[np.ndarray, np.ndarray]:
    """Metallic RGB + alpha for one solder fillet."""
    shiny = quality == "ok"
    hgt, alpha = _fillet_field(w, h, rng, spread, 1.0, shiny)
    if quality == "warn":
        rgb = _shade(hgt, (170, 171, 173), spec=18, spec_gain=0.22,
                     ambient=0.74, rough=3.0, metallic=0.35)
        rgb = rgb * 0.93 + 7
    elif quality == "fail":
        rgb = _shade(hgt, (104, 105, 106), spec=5, spec_gain=0.08,
                     ambient=0.95, rough=3.4, metallic=0.2)
        rgb = rgb * 0.84 + 4
    else:
        rgb = _shade(hgt, seed_color, spec=90, spec_gain=0.40,
                     ambient=0.55, rough=1.8, metallic=0.8)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        u = xx / max(w - 1, 1)
        v = (yy / max(h - 1, 1)) * 2 - 1
        spec = np.exp(-(((u - 0.28) / 0.17) ** 2 + ((v + 0.30) / 0.5) ** 2))
        rgb += spec[..., None] * np.array([78, 80, 82], dtype=np.float32)
    rgb = np.clip(rgb, 0, 255)
    # contact shadow where the solder meets the component body
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    ao = np.clip(1 - xx / (w * 0.12), 0, 1) * 0.40
    rgb *= (1 - ao[..., None])
    return rgb, alpha


def draw_trace_layer(board: np.ndarray, rng: np.random.Generator, S: int, density: int = 40):
    """Subtle copper traces under the solder mask."""
    H, W = board.shape[:2]
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    for _ in range(density):
        x = rng.integers(0, W)
        y = rng.integers(0, H)
        col = (110, 190, 150, int(rng.integers(10, 26)))
        for _ in range(rng.integers(1, 4)):
            x2 = x + rng.integers(-320, 320) * S
            d.line([(x, y), (x2, y)], fill=col, width=int(rng.integers(2, 7) * S))
            x = x2
            y2 = y + rng.integers(-260, 260) * S
            d.line([(x, y), (x, y2)], fill=col, width=int(rng.integers(2, 7) * S))
            y = y2
    return Image.alpha_composite(Image.fromarray(board).convert("RGBA"), layer)


def _draw_pad(board, rng, S, box, kind="smd"):
    """Draw a gold/silver land with a soft metallic sheen."""
    x0, y0, x1, y1 = [int(v) for v in box]
    if x1 - x0 < 4 or y1 - y0 < 4:
        return
    pad_img = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    d = ImageDraw.Draw(pad_img)
    w, h = pad_img.size
    radius = max(1, int(min(w, h) * 0.18))
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius, fill=GOLD + (255,))
    # top-left highlight and bottom shadow
    if h > 6:
        d.rounded_rectangle([1, 1, w - 2, max(3, int(h * 0.45))], radius=radius,
                            fill=GOLD_LIGHT + (110,))
    d.line([(radius, h - 2), (w - radius, h - 2)], fill=(110, 84, 44, 160), width=1)
    pad_img = pad_img.filter(ImageFilter.GaussianBlur(0.45 * S))
    base = Image.fromarray(board).convert("RGBA")
    base.alpha_composite(pad_img, (x0, y0))
    board[:] = np.array(base.convert("RGB"))


def draw_solder_joint(board, joint: Joint, rng, S, body_side=1):
    """Render one joint with a quality-dependent appearance."""
    x0, y0, x1, y1 = joint.box
    pad = max(2, int(1.6 * S))
    gx0, gy0 = int(x0) - pad, int(y0) - pad
    gx1, gy1 = int(x1) + pad, int(y1) + pad
    w, h = max(6, gx1 - gx0), max(6, gy1 - gy0)

    q, defect = joint.quality, joint.defect
    if defect == "missing":
        return

    spread = 0.98
    if q == "warn":
        if defect == "coverage":
            spread = rng.uniform(0.55, 0.72)
        else:
            spread = rng.uniform(0.86, 1.0)
    elif q == "fail":
        if defect == "insufficient":
            spread = rng.uniform(0.30, 0.48)
        elif defect in ("nonwet", "cold"):
            spread = rng.uniform(0.42, 0.62)
    seed = (184 + int(rng.integers(-6, 7)), 188 + int(rng.integers(-6, 7)),
            192 + int(rng.integers(-6, 7)))

    vertical = joint.angle >= 45
    if vertical:
        rgb, alpha = _solder_patch(h, w, q, rng, spread, 0, 0, seed)
        rgb = np.transpose(rgb, (1, 0, 2))
        alpha = np.transpose(alpha, (1, 0))
        if body_side < 0:
            rgb = rgb[:, ::-1].copy()
            alpha = alpha[:, ::-1].copy()
    else:
        rgb, alpha = _solder_patch(w, h, q, rng, spread, 0, 0, seed)
        if body_side < 0:
            rgb = rgb[:, ::-1].copy()
            alpha = alpha[:, ::-1].copy()

    # non-wetting: ragged, oxidised edges leaving exposed land
    if defect == "nonwet":
        yy, xx = np.mgrid[0:alpha.shape[0], 0:alpha.shape[1]].astype(np.float32)
        noise = rng.normal(0, 1, alpha.shape).astype(np.float32)
        noise = np.array(Image.fromarray(((noise - noise.min()) / (np.ptp(noise) + 1e-6) * 255)
                                         .astype(np.uint8)).filter(ImageFilter.GaussianBlur(1.6)),
                         dtype=np.float32) / 255.0
        alpha = np.clip(alpha * (0.35 + 0.9 * noise), 0, 1)

    _paste(board, rgb, gx0, gy0, alpha)

    # insufficient solder leaves part of the metallic termination dry
    if defect == "insufficient":
        strip = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(strip)
        tw = max(2, int(w * 0.22))
        if body_side > 0:
            d.rectangle([0, int(h * 0.08), tw, int(h * 0.92)], fill=(206, 208, 210, 205))
            d.line([(tw - 1, int(h * 0.1)), (tw - 1, int(h * 0.9))], fill=(120, 120, 122, 160), width=1)
        else:
            d.rectangle([w - tw, int(h * 0.08), w, int(h * 0.92)], fill=(206, 208, 210, 205))
            d.line([(w - tw + 1, int(h * 0.1)), (w - tw + 1, int(h * 0.9))], fill=(120, 120, 122, 160), width=1)
        base = Image.fromarray(board).convert("RGBA")
        base.alpha_composite(strip, (gx0, gy0))
        board[:] = np.array(base.convert("RGB"))

    # voids / pinholes / cold crust
    if defect in ("void", "cold") and rng.random() < (0.95 if defect == "void" else 0.4):
        hole = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(hole)
        hr = max(2, int(min(w, h) * rng.uniform(0.10, 0.20)))
        hx = int(w * rng.uniform(0.28, 0.55))
        hy = int(h * rng.uniform(0.3, 0.7))
        d.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=(52, 54, 56, 250))
        d.ellipse([hx - hr * 0.65, hy - hr * 0.65, hx + hr * 0.65, hy + hr * 0.65],
                  fill=(22, 23, 24, 255))
        base = Image.fromarray(board).convert("RGBA")
        base.alpha_composite(hole, (gx0, gy0))
        board[:] = np.array(base.convert("RGB"))


def draw_bridge(board, a: Joint, b: Joint, rng, S):
    """Solder bridge between two adjacent joints."""
    ax = (a.box[0] + a.box[2]) / 2
    ay = (a.box[1] + a.box[3]) / 2
    bx = (b.box[0] + b.box[2]) / 2
    by = (b.box[1] + b.box[3]) / 2
    m = int(8 * S)
    x0, x1 = int(min(ax, bx)) - m, int(max(ax, bx)) + m
    y0, y1 = int(min(ay, by)) - m, int(max(ay, by)) + m
    w, h = max(3, x1 - x0), max(3, y1 - y0)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    seg = ((bx - ax) ** 2 + (by - ay) ** 2) + 1e-6
    t = np.clip(((xx - (ax - x0)) * (bx - ax) + (yy - (ay - y0)) * (by - ay)) / seg, 0, 1)
    px = (ax - x0) + t * (bx - ax)
    py = (ay - y0) + t * (by - ay)
    aa = max(a.box[2] - a.box[0], a.box[3] - a.box[1]) * 0.5
    ab = max(b.box[2] - b.box[0], b.box[3] - b.box[1]) * 0.5
    dist = np.sqrt((xx - px) ** 2 + (yy - py) ** 2)
    da = np.sqrt((xx - (ax - x0)) ** 2 + (yy - (ay - y0)) ** 2)
    db = np.sqrt((xx - (bx - x0)) ** 2 + (yy - (by - y0)) ** 2)
    radius = max(2.0, min(aa, ab) * 0.42)
    corridor = np.clip(1 - (dist - radius) / (radius * 0.7), 0, 1)
    corridor *= np.clip((da - aa * 0.55) / 2.0, 0, 1)
    corridor *= np.clip((db - ab * 0.55) / 2.0, 0, 1)
    height = corridor * 0.85 + rng.normal(0, 0.02, (h, w))
    rgb = _shade(height, (182, 186, 190), spec=80, spec_gain=0.4, ambient=0.55, rough=1.6)
    alpha = np.clip(corridor * 1.8, 0, 1)
    _paste(board, rgb, x0, y0, alpha)


def _chip_component(board, rng, S, x, y, w, h, refdes, quality_pool):
    """Two-terminal SMD chip (resistor / capacitor)."""
    body = (x, y, x + w, y + h)
    term_w = max(2, int(w * 0.20))
    # land pads extend beyond the body
    pad_ext = max(1, int(w * 0.12))
    _draw_pad(board, rng, S, (x - pad_ext, y - h * 0.12, x + term_w + pad_ext, y + h * 1.12))
    _draw_pad(board, rng, S, (x + w - term_w - pad_ext, y - h * 0.12, x + w + pad_ext, y + h * 1.12))
    layer = Image.new("RGBA", board.shape[1::-1], (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    body_col = rng.choice([(24, 24, 26), (32, 30, 28), (48, 44, 38), (58, 50, 42), (30, 36, 34)])
    _rounded(d, body, max(1, int(1.4 * S)), fill=tuple(body_col) + (255,))
    # metallic end caps
    for ex in (x - 1, x + w - term_w):
        d.rectangle([ex, y, ex + term_w, y + h], fill=(206, 208, 210, 255))
        d.line([(ex + term_w * 0.35, y), (ex + term_w * 0.35, y + h)], fill=(150, 152, 154, 170), width=max(1, S))
    # top sheen
    d.rectangle([x + term_w, y + 1, x + w - term_w, y + max(2, int(h * 0.28))],
                fill=(255, 255, 255, 18))
    if rng.random() < 0.5:
        f = _font(FONT_BOLD, max(6, int(h * 0.5)))
        label = rng.choice(["103", "472", "104", "220", "1R0", "10K", "4K7", "100n"])
        d.text((x + term_w + 3 * S, y + h * 0.22), label, font=f, fill=(220, 220, 220, 200))
    base = Image.fromarray(board).convert("RGBA")
    base.alpha_composite(layer)
    board[:] = np.array(base.convert("RGB"))
    # joints
    joints = []
    for i, jx in enumerate((x + term_w * 0.35, x + w - term_w * 0.35)):
        jw = term_w * 1.55
        jh = h * 0.98
        box = (jx - jw / 2, y + h / 2 - jh / 2, jx + jw / 2, y + h / 2 + jh / 2)
        q, defect = quality_pool(rng)
        j = Joint(box=box, quality=q, defect=defect, angle=0)
        draw_solder_joint(board, j, rng, S, body_side=1 if i == 0 else -1)
        joints.append(j)
    return Component("chip", body, joints, refdes)


def _ic_component(board, rng, S, x, y, w, h, refdes, quality_pool, qfp=False):
    layer = Image.new("RGBA", board.shape[1::-1], (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    pins_per_side = max(3, int(w / (9 * S))) if not qfp else max(4, int(w / (10 * S)))
    pin_len = 7 * S
    # pads + pins
    joints = []
    if not qfp:
        for side in (0, 1):
            for i in range(pins_per_side):
                px = x + (i + 0.5) * w / pins_per_side
                pbox = (px - 2.2 * S, y - pin_len if side == 0 else y + h,
                        px + 2.2 * S, y if side == 0 else y + h + pin_len)
                _draw_pad(board, rng, S, pbox)
                d.rectangle([px - 1.1 * S, pbox[1], px + 1.1 * S, pbox[3]],
                            fill=(198, 200, 202, 255))
                jb = (px - 2.6 * S, pbox[1] - 1.5 * S, px + 2.6 * S, pbox[3] + 1.5 * S)
                q, defect = quality_pool(rng)
                j = Joint(box=jb, quality=q, defect=defect, angle=90)
                joints.append(j)
        for side in (0, 1):
            for i in range(max(2, pins_per_side // 2)):
                py = y + (i + 0.5) * h / max(2, pins_per_side // 2)
                pbox = (x - pin_len if side == 0 else x + w, py - 2.2 * S,
                        x if side == 0 else x + w + pin_len, py + 2.2 * S)
                _draw_pad(board, rng, S, pbox)
                d.rectangle([pbox[0], py - 1.1 * S, pbox[2], py + 1.1 * S],
                            fill=(198, 200, 202, 255))
                jb = (pbox[0] - 1.5 * S, py - 2.6 * S, pbox[2] + 1.5 * S, py + 2.6 * S)
                q, defect = quality_pool(rng)
                joints.append(Joint(box=jb, quality=q, defect=defect, angle=0))
    body = (x, y, x + w, y + h)
    _rounded(d, body, max(1, int(2 * S)), fill=(22, 22, 24, 255))
    d.rectangle([x + 2, y + 2, x + w - 2, y + max(3, int(h * 0.18))], fill=(70, 72, 78, 40))
    d.ellipse([x + 4 * S, y + 4 * S, x + 9 * S, y + 9 * S], fill=(190, 192, 194, 235))
    f = _font(FONT_BOLD, max(6, int(h * 0.16)))
    d.text((x + w * 0.24, y + h * 0.44), rng.choice(["STM32", "TPS6213", "LM358", "NE555", "AMS1117", "MCP2301"]),
           font=f, fill=(176, 178, 180, 210))
    base = Image.fromarray(board).convert("RGBA")
    base.alpha_composite(layer)
    board[:] = np.array(base.convert("RGB"))
    for j in joints:
        draw_solder_joint(board, j, rng, S, body_side=1)
    return Component("ic", body, joints, refdes, pins=pins_per_side)


def _electrolytic(board, rng, S, cx, cy, r, refdes, quality_pool):
    layer = Image.new("RGBA", board.shape[1::-1], (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # pads
    _draw_pad(board, rng, S, (cx - r * 0.55, cy + r * 0.35, cx - r * 0.15, cy + r * 1.05))
    _draw_pad(board, rng, S, (cx + r * 0.15, cy + r * 0.35, cx + r * 0.55, cy + r * 1.05))
    d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(36, 40, 48, 255))
    d.ellipse([cx - r * 0.92, cy - r * 0.92, cx + r * 0.92, cy + r * 0.9], fill=(44, 50, 60, 255))
    d.pieslice([cx - r * 0.92, cy - r * 0.92, cx + r * 0.92, cy + r * 0.9], 200, 340,
               fill=(190, 194, 202, 255))
    d.ellipse([cx - r * 0.95, cy - r * 0.95, cx + r * 0.95, cy + r * 0.95],
              outline=(180, 184, 190, 140), width=max(1, int(S)))
    d.rectangle([cx - r * 0.72, cy - r * 0.2, cx + r * 0.2, cy + r * 0.62], fill=(16, 18, 22, 220))
    f = _font(FONT_BOLD, max(7, int(r * 0.24)))
    d.text((cx - r * 0.66, cy - r * 0.12), "470", font=f, fill=(210, 212, 214, 220))
    base = Image.fromarray(board).convert("RGBA")
    base.alpha_composite(layer)
    board[:] = np.array(base.convert("RGB"))
    joints = []
    for i, jx in enumerate((cx - r * 0.35, cx + r * 0.35)):
        q, defect = quality_pool(rng)
        jb = (jx - 4 * S, cy + r * 0.38, jx + 4 * S, cy + r * 1.02)
        j = Joint(box=jb, quality=q, defect=defect, angle=90)
        draw_solder_joint(board, j, rng, S, body_side=1)
        joints.append(j)
    return Component("cap", (cx - r, cy - r, cx + r, cy + r), joints, refdes)


def _connector(board, rng, S, x, y, w, h, refdes, quality_pool, pins=6):
    layer = Image.new("RGBA", board.shape[1::-1], (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    _rounded(d, (x, y, x + w, y + h), int(2 * S), fill=(28, 30, 34, 255))
    for i in range(pins):
        px = x + (i + 0.5) * w / pins
        _draw_pad(board, rng, S, (px - w / pins * 0.16, y - 0.55 * h, px + w / pins * 0.16, y + 0.1 * h))
        d.rectangle([px - w / pins * 0.13, y + 2, px + w / pins * 0.13, y + h * 0.35],
                    fill=(206, 174, 110, 255))
        d.rectangle([px - w / pins * 0.13, y + 2, px + w / pins * 0.13, y + h * 0.08],
                    fill=(244, 220, 168, 255))
    base = Image.fromarray(board).convert("RGBA")
    base.alpha_composite(layer)
    board[:] = np.array(base.convert("RGB"))
    joints = []
    for i in range(pins):
        px = x + (i + 0.5) * w / pins
        q, defect = quality_pool(rng)
        jb = (px - w / pins * 0.2, y - 0.6 * h, px + w / pins * 0.2, y + 0.02 * h)
        j = Joint(box=jb, quality=q, defect=defect, angle=90)
        draw_solder_joint(board, j, rng, S, body_side=1)
        joints.append(j)
    return Component("connector", (x, y, x + w, y + h), joints, refdes)


def _via(board, rng, S, x, y, r):
    layer = Image.new("RGBA", board.shape[1::-1], (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    d.ellipse([x - r, y - r, x + r, y + r], fill=GOLD + (255,))
    d.ellipse([x - r * 0.55, y - r * 0.55, x + r * 0.55, y + r * 0.55], fill=(14, 32, 26, 255))
    base = Image.fromarray(board).convert("RGBA")
    base.alpha_composite(layer)
    board[:] = np.array(base.convert("RGB"))


def _text(board, rng, S, x, y, text, size, color=SILK, alpha=185, bold=False, rotate=0):
    layer = Image.new("RGBA", board.shape[1::-1], (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    f = _font(FONT_BOLD if bold else FONT_REG, size)
    d.text((x, y), text, font=f, fill=tuple(color) + (alpha,))
    if rotate:
        layer = layer.rotate(rotate, resample=Image.BICUBIC, center=(x, y))
    base = Image.fromarray(board).convert("RGBA")
    base.alpha_composite(layer)
    board[:] = np.array(base.convert("RGB"))


def _tht_resistor(board, rng, S, cx, cy, length, refdes, qpool, vertical=False):
    """Axial resistor / diode with round plated pads and a meniscus each end."""
    body_l = length * 0.58
    body_r = max(5, int(length * 0.10))
    pad_r = max(5, int(length * 0.085))
    endpoints = [(-length / 2 + pad_r * 1.1, 0), (length / 2 - pad_r * 1.1, 0)]

    def rot(dx, dy):
        return (cx + dx, cy + dy) if not vertical else (cx - dy, cy + dx)

    layer = Image.new("RGBA", board.shape[1::-1], (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    # pads
    for dx, dy in endpoints:
        px, py = rot(dx, dy)
        _draw_pad(board, rng, S, (px - pad_r * 1.5, py - pad_r * 1.5, px + pad_r * 1.5, py + pad_r * 1.5))
        d.ellipse([px - pad_r, py - pad_r, px + pad_r, py + pad_r], fill=(150, 120, 66, 255))
        d.ellipse([px - pad_r * 0.42, py - pad_r * 0.42, px + pad_r * 0.42, py + pad_r * 0.42],
                  fill=(12, 30, 24, 255))
    # leads
    lead_w = max(2, int(pad_r * 0.55))
    for dx, dy in endpoints:
        px, py = rot(dx, dy)
        bx, by = rot(np.sign(dx) * body_l / 2, 0)
        d.line([(px, py), (bx, by)], fill=(176, 178, 180, 255), width=lead_w)
        d.line([(px, py), (bx, by)], fill=(214, 216, 218, 120), width=max(1, lead_w // 3))
    # body
    body_col = rng.choice([(196, 176, 132), (52, 66, 104), (176, 150, 108), (40, 96, 84)])
    if vertical:
        d.rounded_rectangle([cx - body_r, cy - body_l / 2, cx + body_r, cy + body_l / 2],
                            radius=max(3, body_r // 2), fill=tuple(body_col) + (255,))
        d.rectangle([cx - body_r + 2, cy - body_l / 2 + 2, cx - body_r + body_r * 0.7, cy + body_l / 2 - 2],
                    fill=(255, 255, 255, 26))
        for i in range(rng.integers(3, 5)):
            yy = cy - body_l / 2 + body_l * (0.18 + 0.17 * i)
            col = rng.choice([(60, 40, 30), (150, 120, 40), (30, 30, 30), (200, 200, 190)])
            d.rectangle([cx - body_r, yy, cx + body_r, yy + max(2, body_l * 0.06)], fill=tuple(col) + (255,))
    else:
        d.rounded_rectangle([cx - body_l / 2, cy - body_r, cx + body_l / 2, cy + body_r],
                            radius=max(3, body_r // 2), fill=tuple(body_col) + (255,))
        d.rectangle([cx - body_l / 2 + 2, cy - body_r + 2, cx + body_l / 2 - 2, cy - body_r + body_r * 0.7],
                    fill=(255, 255, 255, 26))
        for i in range(rng.integers(3, 5)):
            xx = cx - body_l / 2 + body_l * (0.18 + 0.17 * i)
            col = rng.choice([(60, 40, 30), (150, 120, 40), (30, 30, 30), (200, 200, 190)])
            d.rectangle([xx, cy - body_r, xx + max(2, body_l * 0.06), cy + body_r], fill=tuple(col) + (255,))
    base = Image.fromarray(board).convert("RGBA")
    base.alpha_composite(layer)
    board[:] = np.array(base.convert("RGB"))
    joints = []
    for dx, dy in endpoints:
        px, py = rot(dx, dy)
        q, defect = qpool(rng)
        pad = pad_r * 1.45
        jb = (px - pad, py - pad, px + pad, py + pad)
        j = Joint(box=jb, quality=q, defect=defect, angle=90 if vertical else 0)
        draw_solder_joint(board, j, rng, S, body_side=1)
        joints.append(j)
    box = (cx - (body_r if vertical else body_l / 2), cy - (body_l / 2 if vertical else body_r),
           cx + (body_r if vertical else body_l / 2), cy + (body_l / 2 if vertical else body_r))
    return Component("tht", box, joints, refdes)


def _silk_outline(board, box, color=SILK, alpha=120, dashed=True):
    layer = Image.new("RGBA", board.shape[1::-1], (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    x0, y0, x1, y1 = [int(v) for v in box]
    if dashed:
        step = 7
        for x in range(x0, x1, step * 2):
            d.line([(x, y0), (min(x + step, x1), y0)], fill=tuple(color) + (alpha,), width=1)
            d.line([(x, y1), (min(x + step, x1), y1)], fill=tuple(color) + (alpha,), width=1)
        for y in range(y0, y1, step * 2):
            d.line([(x0, y), (x0, min(y + step, y1))], fill=tuple(color) + (alpha,), width=1)
            d.line([(x1, y), (x1, min(y + step, y1))], fill=tuple(color) + (alpha,), width=1)
    else:
        d.rectangle([x0, y0, x1, y1], outline=tuple(color) + (alpha,), width=1)
    base = Image.fromarray(board).convert("RGBA")
    base.alpha_composite(layer)
    board[:] = np.array(base.convert("RGB"))


def render_board(seed: int, W: int = 1500, H: int = 1000, S: int = 2,
                 density: float = 1.0, zoom: float = 1.0) -> Tuple[Image.Image, List[Joint]]:
    """Render one synthetic board photo (W x H) and return its joint boxes."""
    rng = np.random.default_rng(seed)
    z = zoom

    mask = MASK_COLORS[int(rng.integers(0, len(MASK_COLORS)))]
    base = np.zeros((H, W, 3), dtype=np.float32)
    base[:] = np.array(mask, dtype=np.float32)
    low = rng.normal(0, 1, (H // 40 + 2, W // 40 + 2))
    low = np.array(Image.fromarray(((low - low.min()) / (np.ptp(low) + 1e-6) * 255).astype(np.uint8))
                   .resize((W, H), Image.BICUBIC), dtype=np.float32) / 255.0
    base *= (0.82 + 0.36 * low[..., None])
    fib = rng.normal(0, 1, (H, 2)).repeat(W // 2 + 1, axis=1)[:, :W]
    base += fib[..., None] * 1.4
    base += rng.normal(0, 2.2, (H, W, 1))
    board = np.clip(base, 0, 255).astype(np.uint8)
    board = np.array(draw_trace_layer(board, rng, 1, density=int(34 * density)).convert("RGB"))

    for _ in range(int(50 * density)):
        _via(board, rng, 1, int(rng.integers(30, W - 30)), int(rng.integers(30, H - 30)),
             int(rng.integers(3, 7)))
    for _ in range(3):
        fx, fy = int(rng.integers(40, W - 40)), int(rng.integers(40, H - 40))
        _draw_pad(board, rng, 1, (fx - 11, fy - 11, fx + 11, fy + 11))

    joints: List[Joint] = []
    components: List[Component] = []
    placed: List[Tuple[float, float, float, float]] = []
    ref_i = [0]

    def qpool(r):
        p = r.random()
        if p < 0.54:
            return "ok", "none"
        if p < 0.80:
            return "warn", ("dull" if r.random() < 0.55 else "coverage")
        return "fail", str(r.choice(["cold", "insufficient", "nonwet", "void", "bridge", "missing"]))

    def ref(prefix):
        ref_i[0] += 1
        return f"{prefix}{ref_i[0]}"

    def free(box, pad=6):
        for b in placed:
            if box[0] < b[2] + pad and box[2] + pad > b[0] and box[1] < b[3] + pad and box[3] + pad > b[1]:
                return False
        return True

    attempts = int(260 * density)
    for _ in range(attempts):
        kind = rng.random()
        if kind < 0.44:      # chip
            w = int(rng.integers(42, 110) * z)
            h = int(rng.integers(20, 46) * z)
            x = int(rng.integers(30, max(31, W - w - 30)))
            y = int(rng.integers(30, max(31, H - h - 30)))
            if not free((x - 12, y - 12, x + w + 12, y + h + 12)):
                continue
            placed.append((x - 12, y - 12, x + w + 12, y + h + 12))
            comp = _chip_component(board, rng, 1, x, y, w, h,
                                   ref("R" if rng.random() < 0.55 else "C"), qpool)
            _silk_outline(board, (x - 6, y - 4, x + w + 6, y + h + 10))
            components.append(comp)
        elif kind < 0.62:    # SOIC / QFN
            w = int(rng.integers(86, 190) * z)
            h = int(rng.integers(54, 110) * z)
            x = int(rng.integers(40, max(41, W - w - 40)))
            y = int(rng.integers(40, max(41, H - h - 40)))
            if not free((x - 22, y - 22, x + w + 22, y + h + 22)):
                continue
            placed.append((x - 22, y - 22, x + w + 22, y + h + 22))
            comp = _ic_component(board, rng, 1, x, y, w, h, ref("U"), qpool)
            _silk_outline(board, (x - 12, y - 12, x + w + 12, y + h + 12))
            components.append(comp)
        elif kind < 0.74:    # electrolytic
            r = int(rng.integers(17, 34) * z)
            x = int(rng.integers(40, max(41, W - 2 * r - 40)))
            y = int(rng.integers(40, max(41, H - 2 * r - 40)))
            if not free((x - 6, y - 6, x + 2 * r + 6, y + 2 * r + 6)):
                continue
            placed.append((x - 6, y - 6, x + 2 * r + 6, y + 2 * r + 6))
            comp = _electrolytic(board, rng, 1, x + r, y + r, r, ref("C"), qpool)
            _silk_outline(board, (x - 3, y - 3, x + 2 * r + 3, y + 2 * r + 3), dashed=False)
            components.append(comp)
        elif kind < 0.86:    # connector
            w = int(rng.integers(70, 165) * z)
            h = int(rng.integers(34, 64) * z)
            x = int(rng.integers(40, max(41, W - w - 40)))
            y = int(rng.integers(40, max(41, H - h - 40)))
            if not free((x - 10, y - 30, x + w + 10, y + h + 10)):
                continue
            placed.append((x - 10, y - 30, x + w + 10, y + h + 10))
            comp = _connector(board, rng, 1, x, y, w, h, ref("J"), qpool,
                              pins=int(rng.integers(3, 9)))
            _silk_outline(board, (x - 5, y - 26, x + w + 5, y + h + 5))
            components.append(comp)
        else:                # through-hole resistor
            length = int(rng.integers(80, 165) * z)
            vertical = bool(rng.random() < 0.45)
            bw = int(length * 0.30)
            x = int(rng.integers(60, max(61, W - 60)))
            y = int(rng.integers(60, max(61, H - 60)))
            hx = bw if vertical else length // 2 + 8
            hy = length // 2 + 8 if vertical else bw
            if not free((x - hx, y - hy, x + hx, y + hy)):
                continue
            placed.append((x - hx, y - hy, x + hx, y + hy))
            comp = _tht_resistor(board, rng, 1, x, y, length, ref("R"), qpool, vertical=vertical)
            _silk_outline(board, (x - hx + 3, y - hy + 3, x + hx - 3, y + hy - 3))
            components.append(comp)

    # silkscreen
    _text(board, rng, 1, 40, H - 52, f"NEXO  PCB-{int(rng.integers(100, 999))}  REV {chr(65 + int(rng.integers(0, 5)))}",
          max(9, int(17 * min(z, 1.6))), alpha=170, bold=True)
    _text(board, rng, 1, max(10, W - 300), H - 48, "IPC-A-610G  CLASS 2",
          max(8, int(13 * min(z, 1.6))), alpha=145)
    for comp in components:
        if comp.kind == "connector" or rng.random() < 0.6:
            f = _font(FONT_REG, max(6, int(11 * min(z, 1.5))))
            layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            d = ImageDraw.Draw(layer)
            d.text((comp.box[0] + 2, comp.box[3] + 3), comp.refdes, font=f, fill=SILK + (145,))
            base_im = Image.fromarray(board).convert("RGBA")
            base_im.alpha_composite(layer)
            board[:] = np.array(base_im.convert("RGB"))

    # bridges between close joints of neighbouring parts
    all_joints = [j for c in components for j in c.joints]
    bridges = [j for j in all_joints if j.defect == "bridge"]
    for j in bridges[:3]:
        cand = [o for o in all_joints if o is not j and o.defect != "bridge" and
                abs(o.box[0] - j.box[0]) < 60 * z and abs(o.box[1] - j.box[1]) < 50 * z]
        if not cand:
            continue
        other = min(cand, key=lambda o: abs(o.box[0] - j.box[0]) + abs(o.box[1] - j.box[1]))
        draw_bridge(board, j, other, rng, 1)
        # el fallo es el puente (caja que cubre ambas uniones), no una unión suelta
        j.quality, j.defect = "ok", "none"
        box = (min(j.box[0], other.box[0]) - 3, min(j.box[1], other.box[1]) - 3,
               max(j.box[2], other.box[2]) + 3, max(j.box[3], other.box[3]) + 3)
        if other.defect == "bridge":
            other.quality, other.defect = "ok", "none"
        joints.append(Joint(box=box, quality="fail", defect="bridge"))

    joints.extend(all_joints)

    # lighting: gentle directional sheen, vignette and a lens blur
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    gx = (xx / W - 0.45) * 0.6
    gy = (yy / H - 0.4) * 0.6
    sheen = np.clip(1.05 - 0.28 * (gx ** 2 + gy ** 2) * 3, 0.7, 1.08)
    board = np.clip(board.astype(np.float32) * sheen[..., None], 0, 255).astype(np.uint8)
    d2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    vig = np.clip(1.0 - 0.20 * d2, 0.72, 1.0)
    board = np.clip(board.astype(np.float32) * vig[..., None], 0, 255).astype(np.uint8)
    im = Image.fromarray(board).filter(ImageFilter.GaussianBlur(0.55))
    arr = np.array(im).astype(np.float32)
    bright = np.clip((arr.mean(-1) - 190) / 65, 0, 1)
    glow_im = Image.fromarray((bright * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(3.0))
    glow = np.array(glow_im, dtype=np.float32) / 255.0
    arr = np.clip(arr + glow[..., None] * np.array([20, 22, 24]), 0, 255)
    arr += rng.normal(0, 3.0, arr.shape)
    im = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    if S != 1:
        im = im.resize((W // S, H // S), Image.LANCZOS)
        sc = 1.0 / S
        joints = [Joint((j.box[0] * sc, j.box[1] * sc, j.box[2] * sc, j.box[3] * sc),
                        j.quality, j.defect, j.angle) for j in joints]

    W2, H2 = im.size
    joints = [j for j in joints if j.box[0] > 2 and j.box[1] > 2 and
              j.box[2] < W2 - 2 and j.box[3] < H2 - 2]
    return im, joints


def add_camera_effects(im: Image.Image, seed: int, jpeg_quality: int = 86) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.array(im).astype(np.float32)
    arr[..., 0] *= rng.uniform(0.97, 1.05)
    arr[..., 2] *= rng.uniform(0.96, 1.04)
    arr = arr * rng.uniform(0.88, 1.12) + rng.uniform(-8, 8)
    arr = (arr - 128) * rng.uniform(0.94, 1.08) + 128
    arr = np.clip(arr, 0, 255)
    im = Image.fromarray(arr.astype(np.uint8))
    if rng.random() < 0.35:
        im = im.filter(ImageFilter.GaussianBlur(float(rng.uniform(0.3, 1.1))))
    import io
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=int(jpeg_quality), subsampling=1)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def square_crop(im: Image.Image, joints: List[Joint], cx: int, cy: int, size: int):
    """Crop a square and remap the joints (clipped, centre inside the crop)."""
    x0 = max(0, min(im.width - size, int(cx) - size // 2))
    y0 = max(0, min(im.height - size, int(cy) - size // 2))
    crop = im.crop((x0, y0, x0 + size, y0 + size))
    out = []
    for j in joints:
        bx0 = max(j.box[0], x0) - x0
        by0 = max(j.box[1], y0) - y0
        bx1 = min(j.box[2], x0 + size) - x0
        by1 = min(j.box[3], y0 + size) - y0
        if bx1 - bx0 < 4 or by1 - by0 < 4:
            continue
        ccx = (j.box[0] + j.box[2]) / 2 - x0
        ccy = (j.box[1] + j.box[3]) / 2 - y0
        if not (0 <= ccx < size and 0 <= ccy < size):
            continue
        out.append(Joint((bx0, by0, bx1, by1), j.quality, j.defect, j.angle))
    return crop, out


if __name__ == "__main__":
    im, joints = render_board(7, 1500, 1000)
    im = add_camera_effects(im, 7)
    im.save("/tmp/synth_test.png")
    print(im.size, len(joints), {q: sum(j.quality == q for j in joints) for q in QUALITIES})
