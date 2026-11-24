"""
Unified animation engine for Wildflower Generative Art.
Computes wind sway, stem-upward growth, leaf unfurling, petal assembly,
smooth color shifting, twinkling, and particle motion in vectorized NumPy arrays.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .ascii_art import THEMES, THEME_NAMES, gradient_3_vec, shift_rgb
from .flowers import Bouquet, _bezier_eval, generate_bouquet
from .particles import ParticleField


@dataclass
class FrameData:
    bg_color: Tuple[int, int, int]
    xy: np.ndarray          # (N, 2) koordinat virtual
    char_ids: np.ndarray    # (N,) int32 indeks di GLYPH_CHARS
    rgb: np.ndarray         # (N, 3) uint8 [0..255]
    alpha: np.ndarray       # (N,) float32 [0..1]
    size_scale: np.ndarray  # (N,) float32 skala relatif
    glow: np.ndarray        # (N,) float32 intensitas pendar cahaya [0..1]


def _smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


class WildflowerSimulation:
    def __init__(
        self,
        seed: int = 2026,
        theme_name: str = "Wildflower",
        quality_scale: float = 1.0,
        max_particles: int = 420,
    ) -> None:
        self.seed = seed
        self.theme_name = theme_name if theme_name in THEMES else THEME_NAMES[0]
        self.quality_scale = quality_scale
        self.speed: float = 1.0
        self.particle_density: float = 1.0
        self.paused: bool = False
        self.time: float = 0.0
        self._gust_impulse: float = 0.0

        self._rebuild_bouquet()
        self.particles = ParticleField(max_particles=max_particles, seed=seed + 99)

    def _rebuild_bouquet(self) -> None:
        theme = THEMES[self.theme_name]
        flower_count = 13 if self.quality_scale >= 0.85 else 11
        self.bouquet: Bouquet = generate_bouquet(
            seed=self.seed,
            palette_count=len(theme["petals"]),
            flower_count=flower_count,
            quality_scale=self.quality_scale,
        )

    def clone(self) -> "WildflowerSimulation":
        """Membuat salinan independen untuk proses export GIF/MP4 di background thread."""
        return copy.deepcopy(self)

    def reset(self) -> None:
        self.time = 0.0
        self._gust_impulse = 0.0

    def randomize(self, new_seed: Optional[int] = None) -> int:
        if new_seed is None:
            new_seed = int(np.random.default_rng().integers(1000, 999999))
        self.seed = new_seed
        self._rebuild_bouquet()
        self.particles = ParticleField(
            max_particles=self.particles.max_particles, seed=new_seed + 99
        )
        self.time = 0.0
        return self.seed

    def next_theme(self) -> str:
        idx = THEME_NAMES.index(self.theme_name)
        self.theme_name = THEME_NAMES[(idx + 1) % len(THEME_NAMES)]
        return self.theme_name

    def set_theme(self, name: str) -> None:
        if name in THEMES:
            self.theme_name = name

    def toggle_pause(self) -> bool:
        self.paused = not self.paused
        return self.paused

    def set_paused(self, paused: bool) -> None:
        self.paused = paused

    def change_speed(self, delta: float) -> float:
        self.speed = float(np.clip(round(self.speed + delta, 2), 0.25, 3.0))
        return self.speed

    def change_density(self, delta: float) -> float:
        self.particle_density = float(np.clip(round(self.particle_density + delta, 2), 0.20, 2.0))
        return self.particle_density

    def trigger_gust(self, strength: float = 0.045) -> None:
        self._gust_impulse = float(np.clip(self._gust_impulse + strength, -0.09, 0.09))

    def step(self, dt: float) -> None:
        if self.paused:
            return
        eff_dt = min(dt, 0.05) * self.speed
        self.time += eff_dt
        self._gust_impulse *= math.exp(-eff_dt * 1.8)
        ambient_wind = math.sin(self.time * 1.15) * 0.015 + self._gust_impulse
        self.particles.step(eff_dt, self.time, ambient_wind)

    def _wind_offset(self, x_pos: float) -> float:
        t = self.time
        return (
            0.015 * math.sin(t * 1.35 + x_pos * 2.4)
            + 0.007 * math.sin(t * 2.75 - x_pos * 1.8)
            + self._gust_impulse
        )

    def build_frame(self) -> FrameData:
        """Menghasilkan seluruh titik glyph terurut untuk dirender pada frame saat ini."""
        theme = THEMES[self.theme_name]
        t_now = self.time

        hue_shift_global = math.sin(t_now * 0.22) * 0.045

        xy_parts = []
        ch_parts = []
        rgb_parts = []
        alpha_parts = []
        scale_parts = []
        glow_parts = []

        # 1. Partikel latar belakang
        p_xy, p_ch, p_rgb, p_a, p_sc = self.particles.sample(
            self.particle_density, t_now, theme["particles"]
        )
        xy_parts.append(p_xy)
        ch_parts.append(p_ch)
        rgb_parts.append(p_rgb)
        alpha_parts.append(p_a * 0.85)
        scale_parts.append(p_sc * 0.88)
        glow_parts.append(p_a * 0.65)

        stem_c = theme["stem"]
        leaf_c = theme["leaf"]
        core_c = theme["core"]
        petal_palettes = theme["petals"]

        # 2. Setiap bunga: batang -> daun -> kelopak
        for f in self.bouquet.flowers:
            stem_prog = _smoothstep((t_now - f.bloom_delay * 0.35) / f.stem_duration)
            if stem_prog <= 0.005:
                continue

            sway = self._wind_offset(float(f.head_pos[0])) * f.wind_amp
            bent_head = np.array([f.head_pos[0] + sway, f.head_pos[1] + abs(sway) * 0.08], dtype=np.float32)
            bent_ctrl = np.array([f.ctrl_pos[0] + sway * 0.32, f.ctrl_pos[1]], dtype=np.float32)

            # --- BATANG ---
            st_mask = f.stem_t <= stem_prog
            if np.any(st_mask):
                st_t = f.stem_t[st_mask]
                st_pts = _bezier_eval(f.base_pos, bent_ctrl, bent_head, st_t)
                st_pts[:, 0] += f.stem_offset_x[st_mask]

                st_rgb = gradient_3_vec(stem_c[0], stem_c[1], stem_c[2], st_t)
                tw = 0.72 + 0.28 * np.sin(t_now * 2.6 + f.stem_twinkle[st_mask])
                st_alpha = np.clip(tw * np.clip((stem_prog - st_t) * 8.0, 0.2, 1.0), 0.15, 1.0)

                xy_parts.append(st_pts)
                ch_parts.append(f.stem_chars[st_mask])
                rgb_parts.append(st_rgb)
                alpha_parts.append(st_alpha.astype(np.float32))
                scale_parts.append(np.full(len(st_t), 0.95, dtype=np.float32))
                glow_parts.append((st_alpha * 0.28).astype(np.float32))

            # --- DAUN ---
            lf_mask = f.leaf_anchor_t <= (stem_prog - 0.04)
            if np.any(lf_mask):
                anc_t = f.leaf_anchor_t[lf_mask]
                anc_pts = _bezier_eval(f.base_pos, bent_ctrl, bent_head, anc_t)
                lf_grow = np.clip((stem_prog - anc_t) * 3.8, 0.0, 1.0)
                flutter = math.sin(t_now * 3.1 + f.wind_phase) * 0.018

                l_local = f.leaf_local_xy[lf_mask].copy()
                ca, sa = math.cos(flutter), math.sin(flutter)
                rx = l_local[:, 0] * ca - l_local[:, 1] * sa
                ry = l_local[:, 0] * sa + l_local[:, 1] * ca
                l_pos = anc_pts + np.column_stack([rx, ry]) * lf_grow[:, None]

                lf_rad = f.leaf_radial[lf_mask]
                lf_rgb = gradient_3_vec(leaf_c[0], leaf_c[1], leaf_c[2], lf_rad)
                tw_lf = 0.72 + 0.28 * np.sin(t_now * 2.8 + f.leaf_twinkle[lf_mask])
                lf_alpha = np.clip(tw_lf * lf_grow * (1.0 - lf_rad * 0.22), 0.1, 1.0)

                xy_parts.append(l_pos.astype(np.float32))
                ch_parts.append(f.leaf_chars[lf_mask])
                rgb_parts.append(lf_rgb)
                alpha_parts.append(lf_alpha.astype(np.float32))
                scale_parts.append(np.full(len(anc_t), 0.88, dtype=np.float32))
                glow_parts.append((lf_alpha * 0.35).astype(np.float32))

            # --- KELOPAK BUNGA ---
            bloom_start = f.bloom_delay * 0.35 + f.stem_duration * 0.65
            bloom_prog = _smoothstep((t_now - bloom_start) / f.bloom_duration)
            if bloom_prog > 0.01:
                head_angle = f.rotation + sway * 1.85
                ca, sa = math.cos(head_angle), math.sin(head_angle)

                assemble = (1.0 - bloom_prog) ** 1.65
                scale_pop = 0.18 + 0.82 * bloom_prog
                local = f.petal_local_xy * scale_pop + f.petal_scatter * assemble

                breath = 1.0 + 0.025 * math.sin(t_now * 1.9 + f.wind_phase) * bloom_prog
                local = local * breath

                px = bent_head[0] + local[:, 0] * ca - local[:, 1] * sa
                py = bent_head[1] + local[:, 0] * sa + local[:, 1] * ca
                p_pos = np.column_stack([px, py]).astype(np.float32)

                pal = petal_palettes[f.palette_idx % len(petal_palettes)]
                flower_hue_shift = hue_shift_global + 0.028 * math.sin(t_now * 0.48 + f.wind_phase)
                c_inner = shift_rgb(core_c[0] if f.species == "reference_star" else pal[0], flower_hue_shift * 0.4)
                c_mid = shift_rgb(pal[1], flower_hue_shift)
                c_outer = shift_rgb(pal[2], flower_hue_shift * 1.2)

                color_wave = np.clip(
                    f.petal_radial + 0.07 * np.sin(t_now * 1.4 - f.petal_radial * 5.0 + f.wind_phase),
                    0.0,
                    1.0,
                )
                p_rgb = gradient_3_vec(c_inner, c_mid, c_outer, color_wave)

                tw_p = 0.74 + 0.26 * np.sin(t_now * 3.4 + f.petal_twinkle)
                core_boost = np.clip(1.28 - f.petal_radial * 0.45, 0.78, 1.25)
                p_alpha = np.clip(bloom_prog * tw_p * core_boost, 0.0, 1.0).astype(np.float32)

                p_glow = np.clip(
                    bloom_prog * (0.92 * (1.0 - f.petal_radial) ** 1.4 + 0.45 * f.petal_edge + 0.22),
                    0.0,
                    1.0,
                ).astype(np.float32)

                xy_parts.append(p_pos)
                ch_parts.append(f.petal_chars)
                rgb_parts.append(p_rgb)
                alpha_parts.append(p_alpha)
                scale_parts.append(np.full(len(p_pos), 0.96 + 0.14 * (1.0 - f.petal_radial), dtype=np.float32))
                glow_parts.append(p_glow)

        # 3. Simpul pita/ikatan buket di bagian bawah
        tie_prog = _smoothstep((t_now - 0.5) / 1.4)
        if tie_prog > 0.01:
            tie_sway = self._wind_offset(0.0) * 0.18
            tie_pos = self.bouquet.tie_xy.copy()
            tie_pos[:, 0] += tie_sway
            pal0 = petal_palettes[0]
            tie_rgb = gradient_3_vec(stem_c[1], pal0[1], core_c[0], np.linspace(0, 1, len(tie_pos)))
            tie_a = (0.7 + 0.3 * np.sin(t_now * 2.2 + self.bouquet.tie_twinkle)) * tie_prog
            xy_parts.append(tie_pos)
            ch_parts.append(self.bouquet.tie_chars)
            rgb_parts.append(tie_rgb)
            alpha_parts.append(tie_a.astype(np.float32))
            scale_parts.append(np.full(len(tie_pos), 0.92, dtype=np.float32))
            glow_parts.append((tie_a * 0.4).astype(np.float32))

        all_xy = np.concatenate(xy_parts, axis=0)
        all_ch = np.concatenate(ch_parts, axis=0)
        all_rgb = np.clip(np.concatenate(rgb_parts, axis=0), 0, 255).astype(np.uint8)
        all_alpha = np.concatenate(alpha_parts, axis=0)
        all_scale = np.concatenate(scale_parts, axis=0)
        all_glow = np.concatenate(glow_parts, axis=0)

        return FrameData(
            bg_color=theme["bg"],
            xy=all_xy,
            char_ids=all_ch,
            rgb=all_rgb,
            alpha=all_alpha,
            size_scale=all_scale,
            glow=all_glow,
        )
