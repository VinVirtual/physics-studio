"""Overlays drawn on each rendered frame: contact-force arrows, centre-of-mass markers, motor-effort
bars, the time and step counter, an event timeline, captions and the result card.

The maths (arrow length and clipping, projection, bar fractions, label placement) are plain functions
so they can be tested without drawing. Drawing is done at 2x on a transparent layer and scaled down,
which gives smooth edges with Pillow.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .camera import Camera, project

FONTS = Path(__file__).resolve().parents[2] / "assets" / "fonts"
SS = 2                       # supersampling factor for the overlay layer

# colour tokens (RGB). Force is the only red; effort uses teal/amber for sign; pass/fail are semantic.
INK = (21, 29, 44)
PAPER = (246, 243, 236)
MUTED = (118, 124, 136)
FORCE = (222, 62, 42)
POS = (36, 150, 136)
NEG = (226, 150, 48)
PASS = (40, 140, 84)
FAIL = (196, 48, 48)
HILITE = (255, 196, 61)


@lru_cache(maxsize=64)
def font(kind: str, size: int, weight: str = "Regular") -> ImageFont.FreeTypeFont:
    path = FONTS / ("JetBrainsMono.ttf" if kind == "mono" else "Geist.ttf")
    f = ImageFont.truetype(str(path), size)
    try:
        f.set_variation_by_name(weight)
    except (OSError, ValueError):
        pass
    return f


# --- maths --------------------------------------------------------------------------------------

def force_tip(pos, f, scale: float, max_len: float) -> tuple[np.ndarray, bool]:
    """World end point of a force arrow drawn from pos: length |f| * scale metres, capped at max_len.
    Returns (tip, clipped)."""
    pos = np.asarray(pos, float)
    f = np.asarray(f, float)
    n = float(np.linalg.norm(f))
    if n == 0:
        return pos.copy(), False
    length = n * scale
    if length > max_len:
        return pos + f / n * max_len, True
    return pos + f * scale, False


def arrow_polygon(x0: float, y0: float, x1: float, y1: float, width: float, head: float) -> list[tuple[float, float]]:
    """Outline of an arrow from (x0, y0) to (x1, y1): a shaft of the given width and a triangular head.
    The head never takes more than 60% of a short arrow."""
    dx, dy = x1 - x0, y1 - y0
    L = math.hypot(dx, dy)
    if L < 1e-6:
        return []
    ux, uy = dx / L, dy / L
    nx, ny = -uy, ux
    h = min(head, 0.6 * L)
    hw = max(width * 1.9, h * 0.62)
    bx, by = x1 - ux * h, y1 - uy * h
    w = width / 2
    return [(x0 + nx * w, y0 + ny * w), (bx + nx * w, by + ny * w), (bx + nx * hw, by + ny * hw), (x1, y1),
            (bx - nx * hw, by - ny * hw), (bx - nx * w, by - ny * w), (x0 - nx * w, y0 - ny * w)]


def bar_fraction(value: float, limit: float) -> float:
    """Motor effort as a signed fraction of its limit, clipped to -1..1 (0 when there is no limit)."""
    if not limit or not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value / limit))


def fmt_force(n: float) -> str:
    return f"{n:.1f} N" if n < 100 else f"{n:.0f} N"


def visible(uvd: np.ndarray, width: int, height: int, margin: float = 0) -> bool:
    u, v, dep = uvd
    return dep > 0 and -margin <= u <= width + margin and -margin <= v <= height + margin


def wrap_words(words: list[str], fnt, max_w: float) -> list[list[int]]:
    """Greedy line breaking of word indices so each line fits max_w pixels."""
    lines, cur, w = [], [], 0.0
    space = fnt.getlength(" ")
    for i, word in enumerate(words):
        ww = fnt.getlength(word)
        if cur and w + space + ww > max_w:
            lines.append(cur)
            cur, w = [], 0.0
        w = ww if not cur else w + space + ww
        cur.append(i)
    if cur:
        lines.append(cur)
    return lines


@dataclass
class Layout:
    width: int
    height: int
    pad: int
    hud: tuple[int, int]            # top-left of the time card
    effort: tuple[int, int, int]    # x, y, width of the motor panel
    caption_y: int                  # baseline region top for captions
    vertical: bool


def layout(width: int, height: int) -> Layout:
    pad = round(min(width, height) * 0.035)
    vertical = height > width
    if vertical:
        return Layout(width, height, pad, (pad, pad), (pad, pad + round(150 * width / 720), width - 2 * pad),
                      round(height * 0.84), True)
    pw = round(width * 0.24)
    return Layout(width, height, pad, (pad, pad), (width - pad - pw, pad, pw), round(height * 0.855), False)


# --- drawing ------------------------------------------------------------------------------------

class Painter:
    def __init__(self, width: int, height: int) -> None:
        self.w, self.h = width, height
        self.L = layout(width, height)
        self.s = min(width, height) / 720          # 1.0 at 720p
        self.layer = Image.new("RGBA", (width * SS, height * SS), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.layer)

    def px(self, v: float) -> int:
        return round(v * SS * self.s)

    def f(self, kind: str, size: float, weight: str = "Regular"):
        return font(kind, max(8, round(size * SS * self.s)), weight)

    def card(self, x: float, y: float, w: float, h: float, alpha: int = 226, fill=PAPER) -> None:
        X, Y, W, H = (round(v * SS) for v in (x, y, w, h))
        r = self.px(10)
        self.d.rounded_rectangle((X + self.px(2), Y + self.px(3), X + W + self.px(2), Y + H + self.px(3)), r,
                                 fill=(0, 0, 0, 38))
        self.d.rounded_rectangle((X, Y, X + W, Y + H), r, fill=(*fill, alpha))

    def text(self, x: float, y: float, s: str, fnt, fill, anchor: str = "la", alpha: int = 255) -> None:
        self.d.text((x * SS, y * SS), s, font=fnt, fill=(*fill, alpha), anchor=anchor)

    def finish(self, base: Image.Image) -> Image.Image:
        small = self.layer.resize((self.w, self.h), Image.LANCZOS)
        out = base.convert("RGBA")
        out.alpha_composite(small)
        return out.convert("RGB")


def draw_forces(p: Painter, cam: Camera, forces: list[dict], scale: float, min_force: float, max_len: float) -> None:
    items = sorted((f for f in forces if f["n"] >= min_force), key=lambda f: -f["n"])
    labelled = 0
    for f in items:
        tip, clipped = force_tip(f["pos"], f["f"], scale, max_len)
        uv = project(cam, [f["pos"], tip])
        if not (visible(uv[0], p.w, p.h, 40) and uv[1][2] > 0):
            continue
        (x0, y0), (x1, y1) = uv[0][:2], uv[1][:2]
        if math.hypot(x1 - x0, y1 - y0) < 3:
            continue
        poly = arrow_polygon(x0 * SS, y0 * SS, x1 * SS, y1 * SS, p.px(4.2), p.px(15))
        if not poly:
            continue
        outline = arrow_polygon(x0 * SS, y0 * SS, x1 * SS, y1 * SS, p.px(4.2) + p.px(3), p.px(15) + p.px(3))
        p.d.polygon(outline, fill=(255, 255, 255, 210))
        p.d.polygon(poly, fill=(*FORCE, 255))
        p.d.ellipse((x0 * SS - p.px(3.5), y0 * SS - p.px(3.5), x0 * SS + p.px(3.5), y0 * SS + p.px(3.5)),
                    fill=(*FORCE, 255), outline=(255, 255, 255, 230), width=p.px(1.5))
        if labelled < 2 and f["n"] >= 0.2:
            labelled += 1
            label = fmt_force(f["n"]) + ("+" if clipped else "")
            fnt = p.f("mono", 14, "Bold")
            tw, th = fnt.getlength(label) / SS, 18 * p.s
            L = math.hypot(x1 - x0, y1 - y0)
            ux, uy = (x1 - x0) / L, (y1 - y0) / L
            # the label sits just beyond the arrow head, centred on the arrow's line
            cx = x1 + ux * (tw / 2 + 10 * p.s)
            cy = y1 + uy * (th / 2 + 8 * p.s)
            bx, by = cx - tw / 2, cy - th / 2
            p.d.rounded_rectangle(((bx - 5 * p.s) * SS, (by - 2 * p.s) * SS, (bx + tw + 5 * p.s) * SS, (by + th) * SS),
                                  p.px(5), fill=(*FORCE, 235))
            p.text(cx, cy - 1 * p.s, label, fnt, (255, 255, 255), anchor="mm")


def draw_com(p: Painter, cam: Camera, coms: list[dict], floor_z_of) -> None:
    for c in coms:
        pos = np.array(c["pos"])
        foot = np.array([pos[0], pos[1], floor_z_of(pos[0], pos[1])])
        uv = project(cam, [pos, foot])
        if not visible(uv[0], p.w, p.h):
            continue
        x, y = uv[0][0] * SS, uv[0][1] * SS
        if uv[1][2] > 0 and c["label"] in ("robot CoM", "system CoM"):
            fx, fy = uv[1][0] * SS, uv[1][1] * SS
            n = max(1, int(math.hypot(fx - x, fy - y) / p.px(9)))
            for i in range(0, n, 2):
                a, b = i / n, min(1, (i + 1) / n)
                p.d.line((x + (fx - x) * a, y + (fy - y) * a, x + (fx - x) * b, y + (fy - y) * b),
                         fill=(*INK, 150), width=p.px(1.6))
            p.d.ellipse((fx - p.px(3), fy - p.px(3), fx + p.px(3), fy + p.px(3)), fill=(*INK, 170))
        r = p.px(8 if c["label"] in ("robot CoM", "system CoM") else 6)
        p.d.ellipse((x - r - p.px(1.5), y - r - p.px(1.5), x + r + p.px(1.5), y + r + p.px(1.5)), fill=(255, 255, 255, 235))
        com_glyph(p, x, y, r)
        if c["label"] in ("robot CoM", "system CoM"):
            p.d.text((x + p.px(13), y - p.px(8)), "CoM", font=p.f("sans", 13, "SemiBold"), fill=(*INK, 255),
                     stroke_width=p.px(2.5), stroke_fill=(255, 255, 255, 235))


def com_glyph(p: Painter, x: float, y: float, r: float) -> None:
    """The centre-of-mass symbol (a circle with two filled quarters), at pixel x, y (already scaled)."""
    p.d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255, 255), outline=(*INK, 255), width=p.px(1.5))
    p.d.pieslice((x - r, y - r, x + r, y + r), 0, 90, fill=(*INK, 255))
    p.d.pieslice((x - r, y - r, x + r, y + r), 180, 270, fill=(*INK, 255))


def draw_hud(p: Painter, t: float, step: int, dt: float, speed: float, engine: str,
             legend_forces: str | None = "contact force", legend_com: bool = True) -> None:
    x, y = p.L.hud
    w = 300 * p.s
    legend = legend_forces or legend_com
    h = (132 if legend else 104) * p.s
    p.card(x, y, w, h)
    if legend:
        ly = y + 112 * p.s
        p.d.line(((x + 16 * p.s) * SS, (ly - 12 * p.s) * SS, (x + w - 16 * p.s) * SS, (ly - 12 * p.s) * SS),
                 fill=(*INK, 30), width=p.px(1))
        lx = x + 16 * p.s
        small = p.f("sans", 12, "Medium")
        if legend_forces:
            p.d.polygon(arrow_polygon(lx * SS, ly * SS, (lx + 24 * p.s) * SS, ly * SS, p.px(3.2), p.px(9)), fill=(*FORCE, 255))
            p.text(lx + 31 * p.s, ly, legend_forces, small, INK, anchor="lm")
            lx += 31 * p.s + small.getlength(legend_forces) / SS + 18 * p.s
        if legend_com:
            com_glyph(p, (lx + 7 * p.s) * SS, ly * SS, p.px(6.5))
            p.text(lx + 19 * p.s, ly, "centre of mass", small, INK, anchor="lm")
    p.text(x + 16 * p.s, y + 13 * p.s, "SIMULATION", p.f("sans", 11.5, "Bold"), FORCE)
    p.text(x + 16 * p.s + p.f("sans", 11.5, "Bold").getlength("SIMULATION  ") / SS, y + 13 * p.s, engine,
           p.f("sans", 11.5, "Medium"), MUTED)
    p.text(x + 16 * p.s, y + 31 * p.s, f"{t:6.3f}", p.f("mono", 34, "Medium"), INK)
    p.text(x + 16 * p.s + p.f("mono", 34, "Medium").getlength(f"{t:6.3f}") / SS + 6 * p.s, y + 45 * p.s, "s",
           p.f("sans", 16, "Medium"), MUTED)
    p.text(x + 16 * p.s, y + 78 * p.s, f"step {step:>6d} · Δt {dt * 1000:g} ms" + (f" · {speed:g}× speed" if speed != 1 else ""),
           p.f("mono", 12.5), MUTED)


def draw_title(p: Painter, title: str, alpha: float) -> None:
    if not title or alpha <= 0:
        return
    x = p.L.hud[0]
    y = p.L.hud[1] + 144 * p.s
    fnt = p.f("sans", 22, "SemiBold")
    w = fnt.getlength(title) / SS + 32 * p.s
    p.card(x, y, w, 44 * p.s, alpha=round(226 * alpha), fill=INK)
    p.text(x + 16 * p.s, y + 11 * p.s, title, fnt, PAPER, alpha=round(255 * alpha))


def draw_effort(p: Painter, torques: list[dict]) -> None:
    if not torques:
        return
    x, y, w = p.L.effort
    row = 27 * p.s
    h = 40 * p.s + row * len(torques)
    p.card(x, y, w, h)
    p.text(x + 14 * p.s, y + 12 * p.s, "MOTOR EFFORT", p.f("sans", 11.5, "Bold"), INK)
    p.text(x + w - 14 * p.s, y + 12 * p.s, "of limit", p.f("sans", 11.5), MUTED, anchor="ra")
    name_w = min(w * 0.36, 112 * p.s)
    bx0 = x + 14 * p.s + name_w
    bx1 = x + w - 62 * p.s
    mid = (bx0 + bx1) / 2
    for i, tq in enumerate(torques):
        ry = y + 36 * p.s + i * row
        name = tq["name"].replace("_", " ")
        fnt = p.f("sans", 12.5, "Medium")
        while fnt.getlength(name) / SS > name_w - 6 * p.s and len(name) > 4:
            name = name[:-2] + "…"
        p.text(x + 14 * p.s, ry + 2 * p.s, name, fnt, INK)
        frac = bar_fraction(tq["value"], tq["limit"])
        p.d.rounded_rectangle((bx0 * SS, (ry + 6 * p.s) * SS, bx1 * SS, (ry + 14 * p.s) * SS), p.px(4), fill=(*INK, 26))
        p.d.line((mid * SS, (ry + 3 * p.s) * SS, mid * SS, (ry + 17 * p.s) * SS), fill=(*INK, 90), width=p.px(1))
        end = mid + frac * (bx1 - bx0) / 2
        if abs(end - mid) > 0.5:
            a, b = sorted((mid, end))
            p.d.rounded_rectangle((a * SS, (ry + 6 * p.s) * SS, b * SS, (ry + 14 * p.s) * SS), p.px(4),
                                  fill=(*(POS if frac >= 0 else NEG), 255))
        p.text(x + w - 14 * p.s, ry + 2 * p.s, f"{abs(frac) * 100:3.0f}%", p.f("mono", 12), INK if abs(frac) < 0.995 else FAIL, anchor="ra")


def draw_timeline(p: Painter, t: float, total: float, events: list[dict], current: str | None) -> None:
    y = p.h - 7 * p.s
    p.d.rectangle((0, (y - 1 * p.s) * SS, p.w * SS, p.h * SS), fill=(*INK, 120))
    x = min(1.0, t / total) * p.w if total > 0 else 0
    p.d.rectangle((0, (y - 1 * p.s) * SS, x * SS, p.h * SS), fill=(*HILITE, 255))
    for e in events:
        ex = min(1.0, e["t"] / total) * p.w
        p.d.rectangle(((ex - 1.2 * p.s) * SS, (y - 7 * p.s) * SS, (ex + 1.2 * p.s) * SS, p.h * SS),
                      fill=(*(PAPER if e["t"] > t else HILITE), 230))
    if current:
        fnt = p.f("sans", 12.5, "SemiBold")
        tw = fnt.getlength(current) / SS
        cx = min(max(x, tw / 2 + 12 * p.s), p.w - tw / 2 - 12 * p.s)
        p.d.rounded_rectangle(((cx - tw / 2 - 9 * p.s) * SS, (y - 36 * p.s) * SS, (cx + tw / 2 + 9 * p.s) * SS, (y - 13 * p.s) * SS),
                              p.px(11), fill=(*INK, 225))
        p.text(cx, y - 31 * p.s, current, fnt, PAPER, anchor="ma")


def draw_caption(p: Painter, words: list[dict], t: float) -> None:
    if not words:
        return
    fnt = p.f("sans", 27 if not p.L.vertical else 31, "SemiBold")
    max_w = p.w * (0.78 if not p.L.vertical else 0.86) * SS
    texts = [w["w"] for w in words]
    lines = wrap_words(texts, fnt, max_w)
    lh = fnt.size * 1.28
    total_h = lh * len(lines)
    y0 = p.L.caption_y * SS - total_h / 2
    widest = max(sum(fnt.getlength(texts[i]) for i in ln) + fnt.getlength(" ") * (len(ln) - 1) for ln in lines)
    cx = p.w * SS / 2
    padx, pady = p.px(20), p.px(11)
    p.d.rounded_rectangle((cx - widest / 2 - padx, y0 - pady, cx + widest / 2 + padx, y0 + total_h + pady * 0.6),
                          p.px(14), fill=(*INK, 214))
    for li, ln in enumerate(lines):
        lw = sum(fnt.getlength(texts[i]) for i in ln) + fnt.getlength(" ") * (len(ln) - 1)
        x = cx - lw / 2
        for i in ln:
            said = t >= words[i]["t"]
            col = HILITE if said and (i + 1 >= len(words) or t < words[i + 1]["t"]) else (PAPER if said else (190, 194, 204))
            p.d.text((x, y0 + li * lh), texts[i], font=fnt, fill=(*col, 255))
            x += fnt.getlength(texts[i]) + fnt.getlength(" ")


def draw_result(p: Painter, success: bool | None, headline: str, detail: str, alpha: float) -> None:
    if alpha <= 0:
        return
    col = PASS if success else (FAIL if success is False else INK)
    fh = p.f("sans", 30, "Bold")
    fd = p.f("sans", 17, "Medium")
    w = max(fh.getlength(headline), fd.getlength(detail)) / SS + 56 * p.s
    h = 96 * p.s
    x = (p.w - w) / 2
    y = p.h * (0.40 if not p.L.vertical else 0.44) - h / 2
    p.card(x, y, w, h, alpha=round(236 * alpha))
    p.d.rounded_rectangle((x * SS, y * SS, (x + 8 * p.s) * SS, (y + h) * SS), p.px(4), fill=(*col, round(255 * alpha)))
    p.text(x + 28 * p.s, y + 16 * p.s, headline, fh, col, alpha=round(255 * alpha))
    p.text(x + 28 * p.s, y + 60 * p.s, detail, fd, INK, alpha=round(255 * alpha))
