"""
ASCII character palettes, 6 curated color themes, color utilities,
and cross-platform glyph atlas generation.
"""

from __future__ import annotations

import colorsys
import os
import platform
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Karakter ASCII yang digunakan (indeks tetap untuk performa NumPy & GPU atlas)
GLYPH_CHARS: str = " .,:;'+*~o%#@/|\\-_=()v^·"
CHAR_TO_ID: Dict[str, int] = {ch: idx for idx, ch in enumerate(GLYPH_CHARS)}

# Ramp karakter berdasarkan kepadatan visual (dari lembut/ujung ke padat/inti)
PETAL_DENSE_CHARS = ["@", "#", "%", "o", "*", "+", ";", ":", "."]
STEM_CHARS = ["|", ":", ".", "·", ";"]
LEAF_CHARS = ["*", "+", ":", ".", "·"]
PARTICLE_CHARS = [".", "·", "*", "+", "'", "~", ":"]
TIE_CHARS = ["~", "=", "-", "*", ":"]

RGB = Tuple[int, int, int]

THEMES: Dict[str, Dict] = {
    "Wildflower": {
        "bg": (0, 0, 0),
        "stem": [(28, 130, 105), (72, 215, 165), (120, 248, 198)],
        "leaf": [(34, 150, 125), (85, 228, 182), (165, 250, 200)],
        "core": [(255, 252, 185), (242, 245, 120), (165, 242, 135)],
        "petals": [
            # Referensi utama gambar: kuning-hijau bercahaya ke mint & emerald-cyan
            [(248, 245, 145), (96, 224, 176), (48, 188, 165)],
            # Coral meadow
            [(255, 232, 140), (255, 132, 118), (225, 76, 112)],
            # Sky azure wildflower
            [(255, 246, 175), (110, 215, 255), (65, 145, 245)],
            # Wild lavender
            [(255, 240, 170), (190, 145, 255), (135, 95, 235)],
            # Marigold gold
            [(255, 252, 205), (255, 195, 72), (240, 120, 48)],
            # Seafoam mint
            [(240, 255, 180), (125, 245, 215), (55, 195, 185)],
        ],
        "particles": [(255, 248, 170), (110, 242, 195), (150, 225, 255), (255, 175, 195)],
    },
    "Pink Garden": {
        "bg": (4, 1, 5),
        "stem": [(45, 125, 95), (95, 190, 145), (145, 230, 185)],
        "leaf": [(52, 145, 110), (115, 210, 165), (175, 240, 205)],
        "core": [(255, 245, 200), (255, 205, 145), (255, 165, 175)],
        "petals": [
            [(255, 235, 220), (255, 138, 178), (225, 58, 128)],
            [(255, 240, 230), (255, 170, 200), (240, 95, 155)],
            [(255, 225, 180), (255, 155, 145), (235, 85, 115)],
            [(255, 235, 245), (235, 135, 220), (185, 70, 175)],
        ],
        "particles": [(255, 215, 230), (255, 150, 190), (255, 240, 195)],
    },
    "Purple Dreams": {
        "bg": (3, 1, 8),
        "stem": [(40, 115, 125), (78, 175, 175), (130, 220, 210)],
        "leaf": [(48, 135, 145), (92, 195, 190), (150, 235, 225)],
        "core": [(255, 248, 195), (235, 205, 255), (185, 150, 255)],
        "petals": [
            [(255, 240, 215), (195, 128, 255), (125, 62, 225)],
            [(240, 235, 255), (160, 145, 255), (95, 82, 235)],
            [(255, 225, 245), (225, 125, 245), (165, 55, 205)],
            [(225, 245, 255), (155, 175, 255), (110, 95, 240)],
        ],
        "particles": [(220, 180, 255), (165, 200, 255), (255, 235, 190)],
    },
    "Blue Night": {
        "bg": (1, 3, 9),
        "stem": [(25, 105, 135), (58, 170, 198), (115, 225, 235)],
        "leaf": [(30, 120, 150), (70, 190, 215), (140, 238, 245)],
        "core": [(245, 255, 255), (185, 240, 255), (120, 215, 255)],
        "petals": [
            [(235, 252, 255), (110, 210, 255), (48, 120, 245)],
            [(220, 248, 255), (90, 235, 235), (40, 165, 215)],
            [(240, 240, 255), (145, 165, 255), (72, 95, 235)],
            [(255, 250, 215), (130, 225, 255), (55, 150, 235)],
        ],
        "particles": [(165, 235, 255), (110, 185, 255), (235, 250, 255)],
    },
    "Rainbow Garden": {
        "bg": (0, 0, 0),
        "stem": [(40, 155, 110), (90, 225, 155), (155, 250, 195)],
        "leaf": [(48, 170, 125), (105, 235, 175), (175, 255, 210)],
        "core": [(255, 255, 225), (255, 245, 145), (255, 215, 175)],
        "petals": [
            [(255, 245, 180), (255, 105, 115), (235, 45, 95)],
            [(255, 248, 180), (255, 180, 75), (245, 115, 45)],
            [(255, 255, 200), (145, 245, 110), (55, 210, 145)],
            [(235, 255, 220), (85, 230, 245), (55, 145, 255)],
            [(255, 235, 255), (175, 130, 255), (120, 65, 245)],
            [(255, 235, 245), (255, 125, 215), (220, 65, 175)],
        ],
        "particles": [
            (255, 140, 160),
            (255, 225, 130),
            (130, 255, 180),
            (130, 220, 255),
            (215, 150, 255),
        ],
    },
    "Sunset Flowers": {
        "bg": (5, 1, 2),
        "stem": [(62, 115, 78), (115, 175, 98), (170, 215, 120)],
        "leaf": [(72, 128, 86), (132, 190, 108), (192, 228, 138)],
        "core": [(255, 252, 205), (255, 220, 115), (255, 170, 75)],
        "petals": [
            [(255, 248, 185), (255, 175, 68), (235, 72, 52)],
            [(255, 238, 175), (255, 125, 88), (205, 42, 92)],
            [(255, 245, 195), (255, 205, 85), (245, 128, 42)],
            [(255, 230, 195), (245, 115, 135), (175, 52, 125)],
        ],
        "particles": [(255, 230, 140), (255, 155, 95), (255, 115, 135)],
    },
}

THEME_NAMES: List[str] = list(THEMES.keys())


def shift_rgb(rgb: RGB, hue_delta: float, sat_boost: float = 1.0) -> RGB:
    """Menggeser hue warna RGB secara halus dalam ruang HSV."""
    r, g, b = rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)
    h = (h + hue_delta) % 1.0
    s = max(0.0, min(1.0, s * sat_boost))
    nr, ng, nb = colorsys.hsv_to_rgb(h, s, v)
    return int(nr * 255), int(ng * 255), int(nb * 255)


def gradient_3_vec(
    c0: RGB, c1: RGB, c2: RGB, t: np.ndarray
) -> np.ndarray:
    """
    Interpolasi 3 titik warna untuk array NumPy t in [0, 1].
    Mengembalikan array float32 bentuk (N, 3) bernilai [0..255].
    """
    t = np.clip(t, 0.0, 1.0).astype(np.float32)
    a = np.array(c0, dtype=np.float32)
    b = np.array(c1, dtype=np.float32)
    c = np.array(c2, dtype=np.float32)

    out = np.empty((t.shape[0], 3), dtype=np.float32)
    low = t < 0.5
    if np.any(low):
        u = (t[low] * 2.0)[:, None]
        out[low] = a * (1.0 - u) + b * u
    high = ~low
    if np.any(high):
        u = ((t[high] - 0.5) * 2.0)[:, None]
        out[high] = b * (1.0 - u) + c * u
    return out


# Bitmap 5x7 fallback agar tetap tajam bahkan tanpa font sistem (.ttf)
_FALLBACK_5X7: Dict[str, List[str]] = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "01100", "00100", "01000"],
    ":": ["00000", "01100", "01100", "00000", "01100", "01100", "00000"],
    ";": ["00000", "01100", "01100", "00000", "01100", "00100", "01000"],
    "'": ["00100", "00100", "01000", "00000", "00000", "00000", "00000"],
    "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
    "*": ["00000", "10101", "01110", "11111", "01110", "10101", "00000"],
    "~": ["00000", "00000", "01101", "10010", "00000", "00000", "00000"],
    "o": ["00000", "00000", "01110", "10001", "10001", "01110", "00000"],
    "%": ["11001", "11010", "00100", "00100", "01011", "10011", "00000"],
    "#": ["01010", "11111", "01010", "01010", "11111", "01010", "00000"],
    "@": ["01110", "10001", "10111", "10101", "10111", "10000", "01110"],
    "/": ["00001", "00010", "00100", "00100", "01000", "10000", "00000"],
    "|": ["00100", "00100", "00100", "00100", "00100", "00100", "00100"],
    "\\": ["10000", "01000", "00100", "00100", "00010", "00001", "00000"],
    "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
    "_": ["00000", "00000", "00000", "00000", "00000", "00000", "11111"],
    "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
    "(": ["00010", "00100", "01000", "01000", "01000", "00100", "00010"],
    ")": ["01000", "00100", "00010", "00010", "00010", "00100", "01000"],
    "v": ["00000", "00000", "10001", "10001", "01010", "00100", "00000"],
    "^": ["00100", "01010", "10001", "00000", "00000", "00000", "00000"],
    "·": ["00000", "00000", "00100", "01110", "00100", "00000", "00000"],
}


def find_monospace_font(size: int) -> Optional[ImageFont.ImageFont]:
    """Mencari font monospace sistem di Windows, macOS, Linux, dan Android."""
    candidates: List[str] = []
    sys_name = platform.system()
    if sys_name == "Windows":
        win_dir = os.environ.get("WINDIR", "C:\\Windows")
        fonts_dir = os.path.join(win_dir, "Fonts")
        candidates.extend([
            os.path.join(fonts_dir, "consola.ttf"),
            os.path.join(fonts_dir, "lucon.ttf"),
            os.path.join(fonts_dir, "cour.ttf"),
        ])
    elif sys_name == "Darwin":
        candidates.extend([
            "/System/Library/Fonts/Menlo.ttc",
            "/System/Library/Fonts/Monaco.ttf",
            "/Library/Fonts/Courier New.ttf",
        ])
    else:
        candidates.extend([
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
            "/system/fonts/DroidSansMono.ttf",
            "/system/fonts/RobotoMono-Regular.ttf",
            "/system/fonts/CutiveMono.ttf",
        ])

    for path in candidates:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return None


def rasterize_glyph(char: str, cell_size: int = 32) -> np.ndarray:
    """
    Merender 1 karakter menjadi alpha mask float32 (cell_size, cell_size) [0..1]
    dengan sedikit kehalusan tepi.
    """
    font = find_monospace_font(int(cell_size * 0.78))
    img = Image.new("L", (cell_size, cell_size), 0)
    if font is not None:
        draw = ImageDraw.Draw(img)
        try:
            bbox = draw.textbbox((0, 0), char, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
            ox = (cell_size - tw) * 0.5 - bbox[0]
            oy = (cell_size - th) * 0.5 - bbox[1]
            draw.text((ox, oy), char, fill=255, font=font)
        except Exception:
            font = None

    if font is None:
        pattern = _FALLBACK_5X7.get(char, _FALLBACK_5X7["."])
        scale = max(1, cell_size // 9)
        ox = (cell_size - 5 * scale) // 2
        oy = (cell_size - 7 * scale) // 2
        draw = ImageDraw.Draw(img)
        for ry, row in enumerate(pattern):
            for rx, bit in enumerate(row):
                if bit == "1":
                    x0 = ox + rx * scale
                    y0 = oy + ry * scale
                    draw.rectangle([x0, y0, x0 + scale - 1, y0 + scale - 1], fill=255)

    return np.asarray(img, dtype=np.float32) / 255.0


def build_glyph_atlas(cell_size: int = 40) -> Tuple[np.ndarray, Dict[int, Tuple[int, int, int, int]]]:
    """
    Membangun RGBA texture atlas untuk semua karakter + 1 sprite soft glow.
    Dipakai oleh Kivy Android renderer agar 1 texture cukup untuk semua glyph.
    """
    cols = 6
    total_items = len(GLYPH_CHARS) + 1
    rows = (total_items + cols - 1) // cols
    atlas_w = cols * cell_size
    atlas_h = rows * cell_size

    atlas = Image.new("RGBA", (atlas_w, atlas_h), (255, 255, 255, 0))
    rects: Dict[int, Tuple[int, int, int, int]] = {}

    for idx, ch in enumerate(GLYPH_CHARS):
        col = idx % cols
        row = idx // cols
        x0 = col * cell_size
        y0 = row * cell_size

        alpha_arr = rasterize_glyph(ch, cell_size=cell_size)
        core_img = Image.fromarray(np.clip(alpha_arr * 255.0, 0, 255).astype(np.uint8), mode="L")
        halo_img = core_img.filter(ImageFilter.GaussianBlur(radius=max(1.2, cell_size * 0.07)))
        combined = np.clip(
            np.asarray(core_img, dtype=np.float32) + np.asarray(halo_img, dtype=np.float32) * 0.55,
            0,
            255,
        ).astype(np.uint8)

        glyph_rgba = Image.new("RGBA", (cell_size, cell_size), (255, 255, 255, 0))
        glyph_rgba.putalpha(Image.fromarray(combined, mode="L"))
        atlas.paste(glyph_rgba, (x0, y0))
        rects[idx] = (x0, y0, cell_size, cell_size)

    glow_idx = len(GLYPH_CHARS)
    col = glow_idx % cols
    row = glow_idx // cols
    x0 = col * cell_size
    y0 = row * cell_size
    yy, xx = np.mgrid[0:cell_size, 0:cell_size]
    cx = (cell_size - 1) * 0.5
    cy = (cell_size - 1) * 0.5
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (cell_size * 0.48)
    glow_alpha = np.clip(1.0 - dist, 0.0, 1.0) ** 2.2
    glow_rgba = Image.new("RGBA", (cell_size, cell_size), (255, 255, 255, 0))
    glow_rgba.putalpha(Image.fromarray((glow_alpha * 255).astype(np.uint8), mode="L"))
    atlas.paste(glow_rgba, (x0, y0))
    rects[-1] = (x0, y0, cell_size, cell_size)

    return np.asarray(atlas, dtype=np.uint8), rects
