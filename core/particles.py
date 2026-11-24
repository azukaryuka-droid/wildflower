"""
Ambient drifting ASCII particles (pollen, stardust, firefly sparks).
"""

from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np

from .ascii_art import CHAR_TO_ID, PARTICLE_CHARS


class ParticleField:
    def __init__(self, max_particles: int = 420, seed: int = 42) -> None:
        self.max_particles = max_particles
        self.rng = np.random.default_rng(seed)

        # Ruang partikel lebih lebar dari buket agar mengisi layar widescreen maupun portrait
        self.pos = np.empty((max_particles, 2), dtype=np.float32)
        self.vel = np.empty((max_particles, 2), dtype=np.float32)
        self.phase = self.rng.uniform(0.0, math.tau, size=max_particles).astype(np.float32)
        self.freq = self.rng.uniform(0.7, 2.3, size=max_particles).astype(np.float32)
        self.size_scale = self.rng.uniform(0.6, 1.25, size=max_particles).astype(np.float32)
        self.color_idx = self.rng.integers(0, 12, size=max_particles, dtype=np.int32)
        self.char_ids = np.array(
            [CHAR_TO_ID[PARTICLE_CHARS[i % len(PARTICLE_CHARS)]] for i in range(max_particles)],
            dtype=np.int32,
        )
        self.reset_positions()

    def reset_positions(self) -> None:
        self.pos[:, 0] = self.rng.uniform(-0.78, 0.78, size=self.max_particles)
        self.pos[:, 1] = self.rng.uniform(-0.50, 0.50, size=self.max_particles)
        self.vel[:, 0] = self.rng.uniform(-0.015, 0.015, size=self.max_particles)
        self.vel[:, 1] = self.rng.uniform(-0.032, -0.008, size=self.max_particles)

    def step(self, dt: float, time_sec: float, wind_gust: float) -> None:
        if dt <= 0.0:
            return
        # Gerakan mengapung perlahan + gelombang sinus + dorongan angin
        drift_x = np.sin(time_sec * 0.8 + self.phase) * 0.012 + wind_gust * 0.16
        drift_y = np.cos(time_sec * 0.5 + self.phase) * 0.004

        self.pos[:, 0] += (self.vel[:, 0] + drift_x) * dt
        self.pos[:, 1] += (self.vel[:, 1] + drift_y) * dt

        # Wrap-around halus
        top_out = self.pos[:, 1] < -0.52
        if np.any(top_out):
            count = int(np.sum(top_out))
            self.pos[top_out, 1] = 0.51
            self.pos[top_out, 0] = self.rng.uniform(-0.78, 0.78, size=count)

        left_out = self.pos[:, 0] < -0.82
        if np.any(left_out):
            self.pos[left_out, 0] = 0.80
        right_out = self.pos[:, 0] > 0.82
        if np.any(right_out):
            self.pos[right_out, 0] = -0.80

    def sample(
        self,
        density: float,
        time_sec: float,
        palette: List[Tuple[int, int, int]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Mengembalikan (xy, char_ids, rgb_float, alpha, size_scale) untuk partikel aktif."""
        active = int(np.clip(round(self.max_particles * 0.55 * density), 12, self.max_particles))
        xy = self.pos[:active]
        cids = self.char_ids[:active]

        twinkle = 0.45 + 0.55 * np.sin(time_sec * self.freq[:active] + self.phase[:active])
        edge_fade = np.clip((0.52 - np.abs(xy[:, 1])) * 4.5, 0.0, 1.0)
        alpha = np.clip(twinkle * edge_fade * 0.88, 0.05, 0.95).astype(np.float32)

        pal_arr = np.array(palette, dtype=np.float32)
        rgb = pal_arr[self.color_idx[:active] % len(pal_arr)]
        return xy, cids, rgb, alpha, self.size_scale[:active]
