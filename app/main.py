"""
Wildflower Generative Art - Desktop Application (Pygame)
Mendukung Windows 10/11, Linux, dan macOS.
Resolusi default 1920x1080, resizable, fullscreen, 60 FPS, kontrol keyboard & mouse.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pygame

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from core.animation import FrameData, WildflowerSimulation
from core.ascii_art import GLYPH_CHARS, rasterize_glyph
from core.rendering import export_gif, export_mp4, export_png, project_virtual_to_screen


def get_exports_dir() -> str:
    """Mengembalikan path folder exports/ yang aman untuk mode Python maupun PyInstaller EXE."""
    if getattr(sys, "frozen", False):
        base = os.path.dirname(sys.executable)
    else:
        base = ROOT_DIR
    path = os.path.join(base, "exports")
    os.makedirs(path, exist_ok=True)
    return path


class PygameGlyphCache:
    """
    Cache permukaan Pygame berkuantisasi warna agar ribuan karakter ASCII
    dapat di-blit sekaligus pada 60 FPS tanpa membebani CPU.
    """

    def __init__(self, glyph_px: int = 18) -> None:
        self.glyph_px = glyph_px
        self._base_alpha: Dict[int, np.ndarray] = {}
        self._tinted_cache: Dict[Tuple[int, int, int, int], pygame.Surface] = {}
        self._rebuild_base(glyph_px)

    def _rebuild_base(self, glyph_px: int) -> None:
        self.glyph_px = max(8, glyph_px)
        self._base_alpha.clear()
        self._tinted_cache.clear()
        for idx, ch in enumerate(GLYPH_CHARS):
            alpha_f = rasterize_glyph(ch, cell_size=self.glyph_px)
            self._base_alpha[idx] = np.clip(alpha_f * 255.0, 0, 255).astype(np.uint8)

    def ensure_size(self, glyph_px: int) -> None:
        if abs(glyph_px - self.glyph_px) >= 3:
            self._rebuild_base(glyph_px)

    def get_surface(self, char_id: int, r: int, g: int, b: int) -> pygame.Surface:
        qr = r >> 4
        qg = g >> 4
        qb = b >> 4
        key = (char_id, qr, qg, qb)
        surf = self._tinted_cache.get(key)
        if surf is not None:
            return surf

        if len(self._tinted_cache) > 12000:
            self._tinted_cache.clear()

        rr = min(255, qr * 16 + 8)
        gg = min(255, qg * 16 + 8)
        bb = min(255, qb * 16 + 8)

        sz = self.glyph_px
        surf = pygame.Surface((sz, sz), pygame.SRCALPHA)
        rgb_arr = pygame.surfarray.pixels3d(surf)
        rgb_arr[:, :, 0] = rr
        rgb_arr[:, :, 1] = gg
        rgb_arr[:, :, 2] = bb
        del rgb_arr

        alpha_arr = pygame.surfarray.pixels_alpha(surf)
        alpha_arr[:, :] = self._base_alpha[char_id].T
        del alpha_arr

        self._tinted_cache[key] = surf
        return surf


class DesktopHUD:
    def __init__(self) -> None:
        self.visible = True
        self.buttons: List[Tuple[str, str, pygame.Rect]] = []
        self.font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 15, bold=True)
        self.info_font = pygame.font.SysFont("consolas,dejavusansmono,monospace", 14)

    def layout(self, width: int, height: int, sim: WildflowerSimulation) -> None:
        self.buttons.clear()
        if not self.visible:
            return

        labels = [
            ("pause", "START" if sim.paused else "PAUSE"),
            ("reset", "RESET"),
            ("random", "RANDOM"),
            ("theme", f"THEME: {sim.theme_name}"),
            ("spd_dn", "SPD -"),
            ("spd_up", f"SPD+ ({sim.speed:.2f}x)"),
            ("den_dn", "DEN -"),
            ("den_up", f"DEN+ ({sim.particle_density:.2f}x)"),
            ("png", "SAVE PNG"),
            ("gif", "EXPORT GIF"),
            ("mp4", "EXPORT MP4"),
            ("full", "FULLSCREEN"),
        ]

        gap = 8
        btn_h = 34
        bar_y = height - btn_h - 14

        total_w = 0
        widths: List[int] = []
        for _, text in labels:
            tw = self.font.size(text)[0] + 22
            widths.append(tw)
            total_w += tw + gap

        scale_fit = min(1.0, (width - 40) / max(1, total_w))
        cur_x = max(14, int((width - total_w * scale_fit) * 0.5))

        for (action, text), bw in zip(labels, widths):
            w = max(56, int(bw * scale_fit))
            self.buttons.append((action, text, pygame.Rect(cur_x, bar_y, w, btn_h)))
            cur_x += w + max(4, int(gap * scale_fit))

    def draw(
        self,
        screen: pygame.Surface,
        sim: WildflowerSimulation,
        fps: float,
        status_msg: str,
    ) -> None:
        w, h = screen.get_size()

        if status_msg:
            toast = self.info_font.render(status_msg, True, (220, 255, 225))
            tw, th = toast.get_size()
            bg = pygame.Surface((tw + 24, th + 12), pygame.SRCALPHA)
            bg.fill((12, 18, 22, 200))
            pygame.draw.rect(bg, (80, 220, 175, 160), bg.get_rect(), 1, border_radius=6)
            screen.blit(bg, ((w - tw - 24) // 2, 18))
            screen.blit(toast, ((w - tw) // 2, 24))

        if not self.visible:
            return

        top_text = (
            f"WILDFLOWER GENERATIVE ART  |  Theme: {sim.theme_name}  |  "
            f"Seed: {sim.seed}  |  FPS: {fps:.0f}  |  [H] Hide UI"
        )
        top_surf = self.info_font.render(top_text, True, (185, 215, 210))
        screen.blit(top_surf, (18, 14))

        bar_rect = pygame.Rect(10, h - 56, w - 20, 48)
        bar_surf = pygame.Surface(bar_rect.size, pygame.SRCALPHA)
        bar_surf.fill((10, 14, 18, 195))
        pygame.draw.rect(bar_surf, (70, 180, 150, 95), bar_surf.get_rect(), 1, border_radius=10)
        screen.blit(bar_surf, bar_rect.topleft)

        mouse_pos = pygame.mouse.get_pos()
        for _, text, rect in self.buttons:
            hovered = rect.collidepoint(mouse_pos)
            fill_col = (42, 110, 96, 220) if hovered else (24, 34, 42, 210)
            border_col = (120, 245, 200) if hovered else (70, 145, 130)
            btn_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
            btn_surf.fill(fill_col)
            pygame.draw.rect(btn_surf, border_col, btn_surf.get_rect(), 1, border_radius=6)
            screen.blit(btn_surf, rect.topleft)

            lbl = self.font.render(text, True, (238, 252, 248))
            lx = rect.x + (rect.width - lbl.get_width()) // 2
            ly = rect.y + (rect.height - lbl.get_height()) // 2
            screen.blit(lbl, (lx, ly))


class WildflowerDesktopApp:
    def __init__(
        self,
        width: int = 1920,
        height: int = 1080,
        theme_name: str = "Wildflower",
        seed: int = 2026,
    ) -> None:
        pygame.init()
        pygame.display.set_caption("WILDFLOWER GENERATIVE ART")

        info = pygame.display.Info()
        init_w = min(width, max(960, info.current_w))
        init_h = min(height, max(540, info.current_h))
        if width == 1920 and height == 1080 and info.current_w >= 1920 and info.current_h >= 1080:
            init_w, init_h = 1920, 1080

        self.fullscreen = False
        self.windowed_size = (init_w, init_h)
        self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE | pygame.DOUBLEBUF)
        self.clock = pygame.time.Clock()

        self.sim = WildflowerSimulation(seed=seed, theme_name=theme_name, quality_scale=1.0)
        self.glyph_cache = PygameGlyphCache(glyph_px=18)
        self.hud = DesktopHUD()
        self.bloom_surf: Optional[pygame.Surface] = None

        self.status_msg: str = ""
        self.status_until: float = 0.0
        self.exporting: bool = False

    def set_status(self, msg: str, seconds: float = 3.5) -> None:
        self.status_msg = msg
        self.status_until = time.time() + seconds

    def toggle_fullscreen(self) -> None:
        self.fullscreen = not self.fullscreen
        if self.fullscreen:
            self.windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN | pygame.DOUBLEBUF)
        else:
            self.screen = pygame.display.set_mode(self.windowed_size, pygame.RESIZABLE | pygame.DOUBLEBUF)

    def _start_export(self, kind: str) -> None:
        if self.exporting:
            self.set_status("Proses ekspor masih berjalan, mohon tunggu...")
            return

        exports_dir = get_exports_dir()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_copy = self.sim.clone()

        if kind == "png":
            path = os.path.join(exports_dir, f"wildflower_{stamp}.png")
            self.exporting = True
            self.set_status("Menyimpan PNG 1920x1080...", 10.0)

            def job_png() -> None:
                try:
                    export_png(sim_copy, path, width=1920, height=1080)
                    self.set_status(f"PNG tersimpan: {os.path.basename(path)}", 4.5)
                except Exception as exc:
                    self.set_status(f"Gagal simpan PNG: {exc}", 5.0)
                finally:
                    self.exporting = False

            threading.Thread(target=job_png, daemon=True).start()

        elif kind == "gif":
            path = os.path.join(exports_dir, f"wildflower_{stamp}.gif")
            self.exporting = True
            self.set_status("Membuat GIF (0%)...", 60.0)

            def job_gif() -> None:
                try:
                    export_gif(
                        sim_copy,
                        path,
                        width=720,
                        height=720,
                        fps=18,
                        duration=6.5,
                        progress_cb=lambda p: self.set_status(f"Membuat GIF ({int(p * 100)}%)...", 60.0),
                    )
                    self.set_status(f"GIF tersimpan: {os.path.basename(path)}", 5.0)
                except Exception as exc:
                    self.set_status(f"Gagal ekspor GIF: {exc}", 5.0)
                finally:
                    self.exporting = False

            threading.Thread(target=job_gif, daemon=True).start()

        elif kind == "mp4":
            path = os.path.join(exports_dir, f"wildflower_{stamp}.mp4")
            self.exporting = True
            self.set_status("Mengekspor MP4 (0%)...", 90.0)

            def job_mp4() -> None:
                try:
                    export_mp4(
                        sim_copy,
                        path,
                        width=1280,
                        height=720,
                        fps=30,
                        duration=8.0,
                        progress_cb=lambda p: self.set_status(f"Mengekspor MP4 ({int(p * 100)}%)...", 90.0),
                    )
                    self.set_status(f"MP4 tersimpan: {os.path.basename(path)}", 5.0)
                except Exception as exc:
                    self.set_status(f"Gagal ekspor MP4: {exc}", 6.0)
                finally:
                    self.exporting = False

            threading.Thread(target=job_mp4, daemon=True).start()

    def _handle_action(self, action: str) -> None:
        if action == "pause":
            paused = self.sim.toggle_pause()
            self.set_status("Animasi Dijeda (PAUSE)" if paused else "Animasi Berjalan (START)")
        elif action == "reset":
            self.sim.reset()
            self.sim.set_paused(False)
            self.set_status("Reset animasi dari awal")
        elif action == "random":
            seed = self.sim.randomize()
            self.set_status(f"Buket acak baru (Seed: {seed})")
        elif action == "theme":
            name = self.sim.next_theme()
            self.set_status(f"Tema warna: {name}")
        elif action == "spd_dn":
            spd = self.sim.change_speed(-0.25)
            self.set_status(f"Kecepatan animasi: {spd:.2f}x")
        elif action == "spd_up":
            spd = self.sim.change_speed(0.25)
            self.set_status(f"Kecepatan animasi: {spd:.2f}x")
        elif action == "den_dn":
            den = self.sim.change_density(-0.25)
            self.set_status(f"Kepadatan partikel: {den:.2f}x")
        elif action == "den_up":
            den = self.sim.change_density(0.25)
            self.set_status(f"Kepadatan partikel: {den:.2f}x")
        elif action == "png":
            self._start_export("png")
        elif action == "gif":
            self._start_export("gif")
        elif action == "mp4":
            self._start_export("mp4")
        elif action == "full":
            self.toggle_fullscreen()

    def _render_frame(self, frame: FrameData) -> None:
        w, h = self.screen.get_size()
        self.screen.fill(frame.bg_color)

        glyph_px = max(10, int(round(min(w, h) * 0.0155)))
        self.glyph_cache.ensure_size(glyph_px)
        half = glyph_px // 2

        px, py, _ = project_virtual_to_screen(frame.xy, w, h)
        ix = np.round(px).astype(np.int32) - half
        iy = np.round(py).astype(np.int32) - half

        rgb = frame.rgb
        alphas = frame.alpha
        cids = frame.char_ids
        glows = frame.glow

        blit_list = []
        glow_list = []

        bw = max(64, w // 4)
        bh = max(64, h // 4)
        if self.bloom_surf is None or self.bloom_surf.get_size() != (bw, bh):
            self.bloom_surf = pygame.Surface((bw, bh))
        self.bloom_surf.fill((0, 0, 0))

        for i in range(len(ix)):
            a = float(alphas[i])
            if a <= 0.03:
                continue
            x0 = int(ix[i])
            y0 = int(iy[i])
            if x0 < -glyph_px or y0 < -glyph_px or x0 >= w or y0 >= h:
                continue

            r = int(rgb[i, 0] * a)
            g = int(rgb[i, 1] * a)
            b = int(rgb[i, 2] * a)
            surf = self.glyph_cache.get_surface(int(cids[i]), r, g, b)
            blit_list.append((surf, (x0, y0)))

            if glows[i] > 0.25 and i % 2 == 0:
                glow_list.append((surf, (x0 // 4, y0 // 4)))

        if blit_list:
            self.screen.blits(blit_list, doreturn=False)

        if glow_list:
            self.bloom_surf.blits(glow_list, doreturn=False)
            up = pygame.transform.smoothscale(self.bloom_surf, (w, h))
            self.screen.blit(up, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

    def run(self) -> None:
        running = True
        while running:
            dt = self.clock.tick(60) / 1000.0
            w, h = self.screen.get_size()
            self.hud.layout(w, h, self.sim)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE and not self.fullscreen:
                    self.windowed_size = (max(640, event.w), max(400, event.h))
                    self.screen = pygame.display.get_surface()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE and self.fullscreen:
                        self.toggle_fullscreen()
                    elif event.key == pygame.K_SPACE:
                        self._handle_action("pause")
                    elif event.key == pygame.K_r:
                        self._handle_action("reset")
                    elif event.key in (pygame.K_g, pygame.K_RETURN):
                        self._handle_action("random")
                    elif event.key == pygame.K_c:
                        self._handle_action("theme")
                    elif event.key in (pygame.K_F11, pygame.K_f):
                        self._handle_action("full")
                    elif event.key == pygame.K_h:
                        self.hud.visible = not self.hud.visible
                    elif event.key == pygame.K_LEFTBRACKET:
                        self._handle_action("spd_dn")
                    elif event.key == pygame.K_RIGHTBRACKET:
                        self._handle_action("spd_up")
                    elif event.key == pygame.K_MINUS:
                        self._handle_action("den_dn")
                    elif event.key == pygame.K_EQUALS:
                        self._handle_action("den_up")
                    elif event.key == pygame.K_s:
                        self._handle_action("png")
                    elif event.key == pygame.K_e:
                        self._handle_action("gif")
                    elif event.key == pygame.K_m:
                        self._handle_action("mp4")
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    clicked_btn = False
                    if self.hud.visible:
                        for action, _, rect in self.hud.buttons:
                            if rect.collidepoint(event.pos):
                                self._handle_action(action)
                                clicked_btn = True
                                break
                    if not clicked_btn:
                        direction = 1.0 if event.pos[0] < w * 0.5 else -1.0
                        self.sim.trigger_gust(direction * 0.045)

            self.sim.step(dt)
            frame = self.sim.build_frame()
            self._render_frame(frame)

            active_status = self.status_msg if time.time() < self.status_until else ""
            self.hud.draw(self.screen, self.sim, self.clock.get_fps(), active_status)
            pygame.display.flip()

        pygame.quit()


def main(theme: str = "Wildflower", seed: int = 2026) -> None:
    app = WildflowerDesktopApp(width=1920, height=1080, theme_name=theme, seed=seed)
    app.run()


if __name__ == "__main__":
    main()
