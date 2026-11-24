"""
Procedural ASCII wildflower bouquet generator.
Produces vectorized NumPy geometry for stems, dot-matrix leaves,
bouquet wrap ties, and multi-species ASCII flower heads.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .ascii_art import (
    CHAR_TO_ID,
    LEAF_CHARS,
    PETAL_DENSE_CHARS,
    STEM_CHARS,
    TIE_CHARS,
)


@dataclass
class FlowerData:
    species: str
    palette_idx: int
    base_pos: np.ndarray        # (2,) titik pangkal batang
    ctrl_pos: np.ndarray        # (2,) control point kurva Bezier
    head_pos: np.ndarray        # (2,) posisi kepala bunga
    stem_t: np.ndarray          # (S,) parameter sepanjang batang [0..1]
    stem_offset_x: np.ndarray   # (S,) lebar dot-matrix batang
    stem_chars: np.ndarray      # (S,) char_id
    stem_twinkle: np.ndarray    # (S,) fase kedip
    leaf_anchor_t: np.ndarray   # (L,) t penempelan di batang
    leaf_local_xy: np.ndarray   # (L, 2) offset relatif terhadap titik batang
    leaf_radial: np.ndarray     # (L,) posisi [0..1] pangkal->ujung daun
    leaf_chars: np.ndarray      # (L,) char_id
    leaf_twinkle: np.ndarray    # (L,) fase kedip
    petal_local_xy: np.ndarray  # (P, 2) koordinat lokal bunga
    petal_radial: np.ndarray    # (P,) jarak ternormalisasi [0..1] dari pusat bunga
    petal_edge: np.ndarray      # (P,) intensitas tepi kelopak [0..1]
    petal_scatter: np.ndarray   # (P, 2) arah sebar saat animasi pembentukan kelopak
    petal_chars: np.ndarray     # (P,) char_id
    petal_twinkle: np.ndarray   # (P,) fase kedip
    bloom_delay: float          # waktu tunda munculnya bunga (detik)
    stem_duration: float        # durasi pertumbuhan batang
    bloom_duration: float       # durasi mekar kelopak
    wind_phase: float
    wind_amp: float
    scale: float
    rotation: float


@dataclass
class Bouquet:
    seed: int
    flowers: List[FlowerData]
    tie_xy: np.ndarray          # (T, 2) simpul pita/ikatan ASCII di pangkal buket
    tie_chars: np.ndarray       # (T,) char_id
    tie_twinkle: np.ndarray     # (T,)


def _bezier_eval(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, t: np.ndarray) -> np.ndarray:
    u = 1.0 - t
    return (u * u)[:, None] * p0 + (2.0 * u * t)[:, None] * p1 + (t * t)[:, None] * p2


def _generate_petal_cloud(
    rng: np.random.Generator,
    species: str,
    radius: float,
    grid_step: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Menghasilkan awan karakter ASCII tersusun rapi di atas kisi dot-matrix
    untuk menciptakan tekstur kelopak organik seperti gambar referensi.
    """
    n_petals = int(rng.integers(5, 8))
    rot_offset = float(rng.uniform(0, math.pi * 2))

    lim = radius * 1.25
    coords = np.arange(-lim, lim + grid_step * 0.5, grid_step, dtype=np.float32)
    gx, gy = np.meshgrid(coords, coords)
    px = gx.ravel()
    py = gy.ravel()

    r = np.sqrt(px * px + py * py)
    theta = np.arctan2(py, px) - rot_offset

    if species == "reference_star":
        n_petals = int(rng.choice([6, 7]))
        wave = np.abs(np.cos(theta * n_petals * 0.5)) ** 0.72
        notches = 0.08 * np.cos(theta * n_petals * 1.5)
        max_r = radius * (0.28 + 0.72 * wave + notches)
    elif species == "cosmos":
        n_petals = int(rng.integers(7, 10))
        lobe = np.maximum(0.0, np.cos(theta * n_petals)) ** 0.45
        tip_dent = 1.0 - 0.12 * (np.cos(theta * n_petals * 2.0) ** 2)
        max_r = radius * (0.22 + 0.78 * lobe * tip_dent)
    elif species == "aster":
        n_petals = int(rng.integers(11, 16))
        spokes = np.abs(np.cos(theta * n_petals * 0.5)) ** 1.65
        max_r = radius * (0.25 + 0.78 * spokes)
    elif species == "bell":
        bell_mask = (py > -radius * 0.55) & (
            np.abs(px) < radius * (0.35 + 0.55 * ((py + radius * 0.6) / (radius * 1.5)).clip(0, 1))
        )
        fluted = np.abs(np.sin(px / max(radius, 1e-4) * math.pi * 2.5)) * 0.22
        max_r = np.where(bell_mask, radius * (0.85 + fluted), radius * 0.14)
    else:  # "lavender"
        pts_x: List[float] = []
        pts_y: List[float] = []
        pts_r: List[float] = []
        pts_e: List[float] = []
        tiers = 9
        for k in range(tiers):
            ty = -radius * 1.15 + k * (radius * 2.1 / max(1, tiers - 1))
            side = -1.0 if k % 2 == 0 else 1.0
            cx = side * radius * (0.18 + 0.08 * math.sin(k))
            for _ in range(18):
                ang = float(rng.uniform(0, math.tau))
                rr = float(rng.uniform(0.0, radius * 0.28))
                pts_x.append(cx + math.cos(ang) * rr)
                pts_y.append(ty + math.sin(ang) * rr * 0.8)
                pts_r.append(min(1.0, (rr / (radius * 0.28)) * 0.7 + (k / tiers) * 0.3))
                pts_e.append(rr / (radius * 0.28))
        xy = np.column_stack([pts_x, pts_y]).astype(np.float32)
        radial = np.array(pts_r, dtype=np.float32)
        edge = np.array(pts_e, dtype=np.float32)
        chars = _assign_petal_chars(rng, radial, edge)
        scatter = _build_scatter_vectors(rng, xy)
        twinkle = rng.uniform(0.0, math.tau, size=len(xy)).astype(np.float32)
        return xy, radial, edge, scatter, chars, twinkle

    inside = r <= max_r
    radial_raw = np.where(max_r > 1e-5, r / max_r, 0.0)

    stipple_prob = np.where(radial_raw < 0.28, 0.96, 0.84 - 0.18 * radial_raw)
    keep = inside & (rng.random(len(px)) < stipple_prob)

    core_mask = r < radius * 0.16
    keep = keep | core_mask

    xy = np.column_stack([px[keep], py[keep]]).astype(np.float32)
    radial = np.clip(radial_raw[keep], 0.0, 1.0).astype(np.float32)
    edge = np.clip((radial - 0.72) / 0.28, 0.0, 1.0).astype(np.float32)

    chars = _assign_petal_chars(rng, radial, edge)
    scatter = _build_scatter_vectors(rng, xy)
    twinkle = rng.uniform(0.0, math.tau, size=len(xy)).astype(np.float32)
    return xy, radial, edge, scatter, chars, twinkle


def _assign_petal_chars(
    rng: np.random.Generator, radial: np.ndarray, edge: np.ndarray
) -> np.ndarray:
    n = len(radial)
    ids = np.empty(n, dtype=np.int32)
    ramp = PETAL_DENSE_CHARS
    ramp_len = len(ramp)
    for i in range(n):
        density = 1.0 - radial[i] * 0.78 + edge[i] * 0.25 + float(rng.uniform(-0.16, 0.16))
        idx = int(np.clip((1.0 - density) * (ramp_len - 1), 0, ramp_len - 1))
        ids[i] = CHAR_TO_ID[ramp[idx]]
    return ids


def _build_scatter_vectors(rng: np.random.Generator, xy: np.ndarray) -> np.ndarray:
    angles = np.arctan2(xy[:, 1], xy[:, 0]) + rng.uniform(-0.85, 0.85, size=len(xy))
    dists = rng.uniform(0.06, 0.20, size=len(xy))
    return np.column_stack([np.cos(angles) * dists, np.sin(angles) * dists]).astype(np.float32)


def _generate_stem_and_leaves(
    rng: np.random.Generator,
    p0: np.ndarray,
    p1: np.ndarray,
    p2: np.ndarray,
    grid_step: float,
    flower_scale: float,
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    num_steps = int(max(36, min(85, np.linalg.norm(p2 - p0) / (grid_step * 0.85))))
    t_vals = np.linspace(0.0, 1.0, num_steps, dtype=np.float32)

    stem_t_list: List[float] = []
    stem_ox_list: List[float] = []
    stem_ch_list: List[int] = []

    for i, t in enumerate(t_vals):
        if rng.random() < 0.08:
            continue
        ch = STEM_CHARS[0] if i % 3 != 0 else str(rng.choice(STEM_CHARS[1:]))
        stem_t_list.append(float(t))
        stem_ox_list.append(0.0)
        stem_ch_list.append(CHAR_TO_ID[ch])

        if t < 0.72 and rng.random() < 0.78:
            stem_t_list.append(float(t))
            stem_ox_list.append(grid_step * 0.55)
            stem_ch_list.append(CHAR_TO_ID[str(rng.choice([":", ".", "|"]))])

    stem_t = np.array(stem_t_list, dtype=np.float32)
    stem_ox = np.array(stem_ox_list, dtype=np.float32)
    stem_chars = np.array(stem_ch_list, dtype=np.int32)
    stem_twinkle = rng.uniform(0.0, math.tau, size=len(stem_t)).astype(np.float32)

    leaf_anchor: List[float] = []
    leaf_xy: List[Tuple[float, float]] = []
    leaf_radial: List[float] = []
    leaf_chars: List[int] = []

    n_leaves = int(rng.integers(2, 5))
    start_side = int(rng.choice([-1, 1]))
    for l_idx in range(n_leaves):
        attach_t = float(0.24 + (l_idx / max(1, n_leaves)) * 0.48 + rng.uniform(-0.04, 0.04))
        attach_t = float(np.clip(attach_t, 0.18, 0.78))
        side = start_side if l_idx % 2 == 0 else -start_side

        leaf_len = flower_scale * float(rng.uniform(0.11, 0.17))
        leaf_arch = flower_scale * float(rng.uniform(0.025, 0.05))
        leaf_width = flower_scale * float(rng.uniform(0.018, 0.028))

        rows = 16
        for u_i in range(rows):
            u = u_i / max(1, rows - 1)
            cx = side * u * leaf_len
            cy = -math.sin(u * math.pi * 0.85) * leaf_arch + (u * u) * leaf_arch * 0.35
            half_w = math.sin(u * math.pi) * leaf_width
            cols = max(1, int(round(half_w / (grid_step * 0.55))))
            for v_i in range(-cols, cols + 1):
                if u > 0.7 and abs(v_i) > 0 and rng.random() < 0.55:
                    continue
                if rng.random() < 0.18:
                    continue
                lx = cx + float(rng.uniform(-grid_step * 0.15, grid_step * 0.15))
                ly = cy + v_i * grid_step * 0.62
                leaf_anchor.append(attach_t)
                leaf_xy.append((lx, ly))
                leaf_radial.append(u)
                ch = LEAF_CHARS[min(len(LEAF_CHARS) - 1, int(u * len(LEAF_CHARS)))]
                leaf_chars.append(CHAR_TO_ID[ch])

            if 0.25 < u < 0.85 and rng.random() < 0.45:
                leaf_anchor.append(attach_t)
                leaf_xy.append((
                    cx + side * float(rng.uniform(-0.01, 0.025)),
                    cy + float(rng.uniform(-0.028, 0.028)),
                ))
                leaf_radial.append(min(1.0, u + 0.2))
                leaf_chars.append(CHAR_TO_ID[str(rng.choice([".", "·"]))])

    if not leaf_anchor:
        leaf_anchor = [0.5]
        leaf_xy = [(0.0, 0.0)]
        leaf_radial = [0.0]
        leaf_chars = [CHAR_TO_ID["."]]

    return (
        stem_t,
        stem_ox,
        stem_chars,
        stem_twinkle,
        np.array(leaf_anchor, dtype=np.float32),
        np.array(leaf_xy, dtype=np.float32),
        np.array(leaf_radial, dtype=np.float32),
        np.array(leaf_chars, dtype=np.int32),
        rng.uniform(0.0, math.tau, size=len(leaf_anchor)).astype(np.float32),
    )


def generate_bouquet(
    seed: int,
    palette_count: int = 6,
    flower_count: int = 13,
    quality_scale: float = 1.0,
) -> Bouquet:
    """
    Menghasilkan buket bunga liar (wildflower bouquet) dalam ruang virtual
    ternormalisasi: lebar ~[-0.42, +0.42], tinggi ~[-0.44, +0.44].
    """
    rng = np.random.default_rng(seed)
    grid_step = 0.0085 / max(0.62, min(1.25, quality_scale))

    flowers: List[FlowerData] = []
    tie_center = np.array([0.0, 0.28], dtype=np.float32)
    species_pool = ["reference_star", "cosmos", "aster", "bell", "lavender"]

    for i in range(flower_count):
        if i == 0:
            species = "reference_star"
            hx = float(rng.uniform(-0.04, 0.04))
            hy = float(rng.uniform(-0.27, -0.21))
            f_scale = float(rng.uniform(1.08, 1.24))
            palette_idx = 0
            bloom_delay = 0.35
        else:
            frac = (i - 1) / max(1, flower_count - 2)
            fan_angle = (frac - 0.5) * math.pi * 0.78 + float(rng.uniform(-0.12, 0.12))
            reach = float(rng.uniform(0.32, 0.57))
            hx = math.sin(fan_angle) * reach * 0.68
            hy = 0.20 - math.cos(fan_angle) * reach * 0.92 + float(rng.uniform(-0.04, 0.04))
            hy = float(np.clip(hy, -0.35, 0.05))
            species = str(rng.choice(species_pool, p=[0.30, 0.24, 0.20, 0.14, 0.12]))
            f_scale = float(rng.uniform(0.62, 1.15))
            palette_idx = (i + int(rng.integers(0, 3))) % max(1, palette_count)
            bloom_delay = float(0.45 + (i / flower_count) * 2.1 + rng.uniform(0.0, 0.35))

        base_x = hx * 0.12 + float(rng.uniform(-0.018, 0.018))
        base_y = float(rng.uniform(0.38, 0.43))
        base_pos = np.array([base_x, base_y], dtype=np.float32)
        head_pos = np.array([hx, hy], dtype=np.float32)

        ctrl_x = tie_center[0] + hx * 0.24 + float(rng.uniform(-0.03, 0.03))
        ctrl_y = 0.16 + float(rng.uniform(-0.05, 0.06))
        ctrl_pos = np.array([ctrl_x, ctrl_y], dtype=np.float32)

        petal_radius = 0.088 * f_scale
        p_xy, p_rad, p_edge, p_scat, p_ch, p_tw = _generate_petal_cloud(
            rng, species, petal_radius, grid_step
        )

        (
            st_t, st_ox, st_ch, st_tw,
            lf_anc, lf_xy, lf_rad, lf_ch, lf_tw,
        ) = _generate_stem_and_leaves(
            rng, base_pos, ctrl_pos, head_pos, grid_step, f_scale
        )

        flowers.append(
            FlowerData(
                species=species,
                palette_idx=palette_idx,
                base_pos=base_pos,
                ctrl_pos=ctrl_pos,
                head_pos=head_pos,
                stem_t=st_t,
                stem_offset_x=st_ox,
                stem_chars=st_ch,
                stem_twinkle=st_tw,
                leaf_anchor_t=lf_anc,
                leaf_local_xy=lf_xy,
                leaf_radial=lf_rad,
                leaf_chars=lf_ch,
                leaf_twinkle=lf_tw,
                petal_local_xy=p_xy,
                petal_radial=p_rad,
                petal_edge=p_edge,
                petal_scatter=p_scat,
                petal_chars=p_ch,
                petal_twinkle=p_tw,
                bloom_delay=bloom_delay,
                stem_duration=float(rng.uniform(1.5, 2.2)),
                bloom_duration=float(rng.uniform(1.4, 2.1)),
                wind_phase=float(rng.uniform(0.0, math.tau)),
                wind_amp=float(rng.uniform(0.85, 1.25)),
                scale=f_scale,
                rotation=float(rng.uniform(-0.22, 0.22)),
            )
        )

    flowers.sort(key=lambda f: (f.head_pos[1], f.scale))

    tie_pts: List[Tuple[float, float]] = []
    tie_ch: List[int] = []
    for k in range(34):
        u = (k / 33.0) * 2.0 - 1.0
        tx = u * 0.052
        ty = 0.285 + math.sin(u * math.pi * 2.0) * 0.008
        tie_pts.append((tx, ty))
        tie_ch.append(CHAR_TO_ID[TIE_CHARS[k % len(TIE_CHARS)]])
        if k % 3 == 0:
            tie_pts.append((tx * 0.85, ty + 0.012))
            tie_ch.append(CHAR_TO_ID[":"])
    for j in range(1, 10):
        tie_pts.append((-0.025 - j * 0.005, 0.295 + j * 0.009))
        tie_ch.append(CHAR_TO_ID["/"])
        tie_pts.append((0.025 + j * 0.005, 0.295 + j * 0.009))
        tie_ch.append(CHAR_TO_ID["\\"])

    tie_xy = np.array(tie_pts, dtype=np.float32)
    tie_chars = np.array(tie_ch, dtype=np.int32)
    tie_twinkle = rng.uniform(0.0, math.tau, size=len(tie_pts)).astype(np.float32)

    return Bouquet(
        seed=seed,
        flowers=flowers,
        tie_xy=tie_xy,
        tie_chars=tie_chars,
        tie_twinkle=tie_twinkle,
    )
