"""
Software frame renderer (NumPy + Pillow) with additive bloom/glow,
and exporters for PNG, GIF, and MP4.
"""

from __future__ import annotations

import os
import subprocess
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

from .animation import FrameData, WildflowerSimulation
from .ascii_art import GLYPH_CHARS, rasterize_glyph

_RESAMPLE_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR
_PALETTE_ADAPTIVE = getattr(Image, "Palette", Image).ADAPTIVE


def project_virtual_to_screen(
    xy: np.ndarray, width: int, height: int
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Memproyeksikan koordinat virtual buket (-0.45..0.45) ke piksel layar (width, height)
    secara proporsional baik untuk Portrait (HP) maupun Landscape (Desktop).
    """
    scale = min(width * 0.90 / 0.84, height * 0.90 / 0.88)
    px = width * 0.5 + xy[:, 0] * scale
    py = height * 0.5 + xy[:, 1] * scale
    return px, py, scale


class SoftwareRenderer:
    """
    Merender FrameData ke gambar RGB PIL menggunakan glyph mask yang sudah di-rasterize
    dan ditambahkan efek bloom Gaussian untuk hasil PNG/GIF/MP4 yang tajam & bercahaya.
    """

    def __init__(self, glyph_px: int = 16) -> None:
        self.glyph_px = glyph_px
        self._masks: Dict[int, np.ndarray] = {}
        self._build_masks(glyph_px)

    def _build_masks(self, glyph_px: int) -> None:
        self.glyph_px = glyph_px
        self._masks.clear()
        for idx, ch in enumerate(GLYPH_CHARS):
            self._masks[idx] = rasterize_glyph(ch, cell_size=glyph_px)

    def render_image(self, frame: FrameData, width: int, height: int) -> Image.Image:
        target_glyph = max(8, int(round(min(width, height) * 0.015)))
        if abs(target_glyph - self.glyph_px) >= 3:
            self._build_masks(target_glyph)

        canvas = np.zeros((height, width, 3), dtype=np.float32)
        canvas[:, :] = np.array(frame.bg_color, dtype=np.float32)

        glow_canvas = np.zeros((height, width, 3), dtype=np.float32)

        px, py, _ = project_virtual_to_screen(frame.xy, width, height)
        g = self.glyph_px
        half = g // 2

        ix = np.round(px).astype(np.int32) - half
        iy = np.round(py).astype(np.int32) - half

        rgb_f = frame.rgb.astype(np.float32)
        alphas = frame.alpha
        glows = frame.glow
        cids = frame.char_ids

        for i in range(len(ix)):
            a = float(alphas[i])
            if a <= 0.02:
                continue
            x0 = int(ix[i])
            y0 = int(iy[i])
            x1 = x0 + g
            y1 = y0 + g
            if x1 <= 0 or y1 <= 0 or x0 >= width or y0 >= height:
                continue

            sx0 = max(0, -x0)
            sy0 = max(0, -y0)
            sx1 = g - max(0, x1 - width)
            sy1 = g - max(0, y1 - height)

            dx0 = max(0, x0)
            dy0 = max(0, y0)
            dx1 = min(width, x1)
            dy1 = min(height, y1)

            mask = self._masks[int(cids[i])][sy0:sy1, sx0:sx1, None] * a
            col = rgb_f[i]

            canvas[dy0:dy1, dx0:dx1] += mask * col
            gw = float(glows[i])
            if gw > 0.15:
                glow_canvas[dy0:dy1, dx0:dx1] += mask * col * (gw * 0.85)

        gw_w = max(32, width // 4)
        gw_h = max(32, height // 4)
        glow_img = Image.fromarray(np.clip(glow_canvas, 0, 255).astype(np.uint8), mode="RGB")
        glow_small = glow_img.resize((gw_w, gw_h), _RESAMPLE_BILINEAR)
        glow_blur = glow_small.filter(ImageFilter.GaussianBlur(radius=3.5))
        glow_up = glow_blur.resize((width, height), _RESAMPLE_BILINEAR)

        final_rgb = np.clip(
            canvas + np.asarray(glow_up, dtype=np.float32) * 0.85,
            0,
            255,
        ).astype(np.uint8)
        return Image.fromarray(final_rgb, mode="RGB")


def export_png(
    sim: WildflowerSimulation,
    filepath: str,
    width: int = 1920,
    height: int = 1080,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    renderer = SoftwareRenderer()
    img = renderer.render_image(sim.build_frame(), width, height)
    img.save(filepath, format="PNG", optimize=True)
    return filepath


def export_gif(
    sim_clone: WildflowerSimulation,
    filepath: str,
    width: int = 720,
    height: int = 720,
    fps: int = 18,
    duration: float = 6.5,
    from_start: bool = True,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    if from_start:
        sim_clone.reset()
    sim_clone.set_paused(False)

    renderer = SoftwareRenderer()
    total_frames = max(12, int(round(fps * duration)))
    dt = 1.0 / fps
    frames: List[Image.Image] = []

    for i in range(total_frames):
        sim_clone.step(dt)
        img = renderer.render_image(sim_clone.build_frame(), width, height)
        frames.append(img.convert("P", palette=_PALETTE_ADAPTIVE, colors=128))
        if progress_cb is not None:
            progress_cb((i + 1) / total_frames)

    frame_ms = int(round(1000 / fps))
    frames[0].save(
        filepath,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=True,
    )
    return filepath


def export_mp4(
    sim_clone: WildflowerSimulation,
    filepath: str,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    duration: float = 8.0,
    from_start: bool = True,
    progress_cb: Optional[Callable[[float], None]] = None,
) -> str:
    """
    Mengekspor animasi ke MP4 menggunakan imageio-ffmpeg, OpenCV, atau CLI ffmpeg.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    width -= width % 2
    height -= height % 2

    if from_start:
        sim_clone.reset()
    sim_clone.set_paused(False)

    renderer = SoftwareRenderer()
    total_frames = max(15, int(round(fps * duration)))
    dt = 1.0 / fps

    try:
        import imageio.v2 as imageio  # type: ignore

        writer = imageio.get_writer(
            filepath,
            fps=fps,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )
        try:
            for i in range(total_frames):
                sim_clone.step(dt)
                img = renderer.render_image(sim_clone.build_frame(), width, height)
                writer.append_data(np.asarray(img))
                if progress_cb is not None:
                    progress_cb((i + 1) / total_frames)
        finally:
            writer.close()
        return filepath
    except Exception:
        pass

    try:
        import cv2  # type: ignore

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        vw = cv2.VideoWriter(filepath, fourcc, fps, (width, height))
        if vw.isOpened():
            try:
                for i in range(total_frames):
                    sim_clone.step(dt)
                    img = renderer.render_image(sim_clone.build_frame(), width, height)
                    bgr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
                    vw.write(bgr)
                    if progress_cb is not None:
                        progress_cb((i + 1) / total_frames)
            finally:
                vw.release()
            return filepath
    except Exception:
        pass

    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{width}x{height}",
            "-pix_fmt", "rgb24",
            "-r", str(fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            filepath,
        ]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None
        for i in range(total_frames):
            sim_clone.step(dt)
            img = renderer.render_image(sim_clone.build_frame(), width, height)
            proc.stdin.write(np.asarray(img).tobytes())
            if progress_cb is not None:
                progress_cb((i + 1) / total_frames)
        proc.stdin.close()
        ret = proc.wait()
        if ret == 0:
            return filepath
    except Exception:
        pass

    raise RuntimeError(
        "Tidak ditemukan codec MP4 (install 'imageio[ffmpeg]' atau 'ffmpeg' di PATH)."
    )
