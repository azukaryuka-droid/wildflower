"""
Wildflower Generative Art - Android / Mobile Application (Kivy)
Kompatibel dengan Pydroid 3 & Buildozer APK.
Mendukung layar sentuh, rotasi Portrait/Landscape otomatis,
optimasi performa untuk Android kelas menengah, tanpa keyboard/mouse.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from kivy.app import App
from kivy.clock import Clock
from kivy.core.window import Window
from kivy.graphics import Color, Rectangle
from kivy.graphics.texture import Texture
from kivy.metrics import dp
from kivy.uix.button import Button
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.widget import Widget

from core.animation import FrameData, WildflowerSimulation
from core.ascii_art import build_glyph_atlas
from core.rendering import export_gif, export_mp4, export_png, project_virtual_to_screen


def get_android_exports_dir(app_instance: Optional[App] = None) -> str:
    """Menentukan direktori penyimpanan yang dapat ditulis di Android / Pydroid 3."""
    candidates = [
        "/sdcard/Pictures/WildflowerArt",
        os.path.join(ROOT_DIR, "exports"),
    ]
    if app_instance is not None:
        candidates.append(os.path.join(app_instance.user_data_dir, "exports"))

    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            test_file = os.path.join(path, ".write_test")
            with open(test_file, "w", encoding="utf-8") as fh:
                fh.write("ok")
            os.remove(test_file)
            return path
        except Exception:
            continue

    fallback = os.path.join(ROOT_DIR, "exports")
    os.makedirs(fallback, exist_ok=True)
    return fallback


class WildflowerCanvasWidget(Widget):
    """
    Kanvas OpenGL ES Kivy berkecepatan tinggi dengan 1 Texture Atlas dan
    pre-allocated Rectangle instruction pool.
    """

    def __init__(self, sim: WildflowerSimulation, on_tap_cb=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self.sim = sim
        self.on_tap_cb = on_tap_cb

        self.atlas_tex: Optional[Texture] = None
        self.tex_coords_map: Dict[int, List[float]] = {}
        self.max_pool = 3200
        self.pool_colors: List[Color] = []
        self.pool_rects: List[Rectangle] = []
        self.bg_color_instr: Optional[Color] = None
        self.bg_rect_instr: Optional[Rectangle] = None

        self._touch_start_x = 0.0
        self._touch_moved = False

        self._init_atlas_and_pool()
        self.bind(pos=self._update_bg, size=self._update_bg)

    def _init_atlas_and_pool(self) -> None:
        atlas_rgba, rects = build_glyph_atlas(cell_size=40)
        h, w, _ = atlas_rgba.shape

        tex = Texture.create(size=(w, h), colorfmt="rgba")
        flipped = np.ascontiguousarray(np.flipud(atlas_rgba))
        tex.blit_buffer(flipped.tobytes(), colorfmt="rgba", bufferfmt="ubyte")
        self.atlas_tex = tex

        for cid, (rx, ry, rw, rh) in rects.items():
            u0 = rx / w
            u1 = (rx + rw) / w
            v0 = 1.0 - (ry + rh) / h
            v1 = 1.0 - ry / h
            self.tex_coords_map[cid] = [u0, v0, u1, v0, u1, v1, u0, v1]

        with self.canvas:
            self.bg_color_instr = Color(0, 0, 0, 1)
            self.bg_rect_instr = Rectangle(pos=self.pos, size=self.size)

            for _ in range(self.max_pool):
                c = Color(1, 1, 1, 0)
                r = Rectangle(
                    texture=self.atlas_tex,
                    pos=(-100, -100),
                    size=(0, 0),
                    tex_coords=self.tex_coords_map[0],
                )
                self.pool_colors.append(c)
                self.pool_rects.append(r)

    def _update_bg(self, *_args) -> None:
        if self.bg_rect_instr is not None:
            self.bg_rect_instr.pos = self.pos
            self.bg_rect_instr.size = self.size

    def on_touch_down(self, touch):
        if self.collide_point(*touch.pos):
            self._touch_start_x = touch.x
            self._touch_moved = False
            return True
        return super().on_touch_down(touch)

    def on_touch_move(self, touch):
        if self.collide_point(*touch.pos):
            dx = touch.x - self._touch_start_x
            if abs(dx) > dp(18):
                self._touch_moved = True
                self.sim.trigger_gust((dx / max(1.0, self.width)) * 0.08)
                self._touch_start_x = touch.x
            return True
        return super().on_touch_move(touch)

    def on_touch_up(self, touch):
        if self.collide_point(*touch.pos):
            if not self._touch_moved and self.on_tap_cb is not None:
                self.on_tap_cb()
            return True
        return super().on_touch_up(touch)

    def render_frame(self, frame: FrameData) -> None:
        w, h = self.size
        if w < 10 or h < 10:
            return

        if self.bg_color_instr is not None:
            bg = frame.bg_color
            self.bg_color_instr.rgba = (bg[0] / 255.0, bg[1] / 255.0, bg[2] / 255.0, 1.0)

        px, py, _ = project_virtual_to_screen(frame.xy, int(w), int(h))
        ky = h - py + self.y
        kx = px + self.x

        glyph_sz = max(dp(7), min(w, h) * 0.017)
        rgb = frame.rgb
        alphas = frame.alpha
        cids = frame.char_ids
        scales = frame.size_scale
        glows = frame.glow

        total = len(kx)
        idx = 0

        # 1. Layer glow lembut
        glow_coords = self.tex_coords_map[-1]
        for i in range(0, total, 3):
            if idx >= self.max_pool - 120:
                break
            gw = float(glows[i])
            a = float(alphas[i])
            if gw < 0.35 or a < 0.15:
                continue
            gsz = glyph_sz * 3.1
            half_g = gsz * 0.5
            self.pool_colors[idx].rgba = (
                rgb[i, 0] / 255.0,
                rgb[i, 1] / 255.0,
                rgb[i, 2] / 255.0,
                min(0.28, gw * a * 0.30),
            )
            rect = self.pool_rects[idx]
            rect.pos = (float(kx[i]) - half_g, float(ky[i]) - half_g)
            rect.size = (gsz, gsz)
            rect.tex_coords = glow_coords
            idx += 1

        # 2. Layer karakter ASCII utama
        for i in range(total):
            if idx >= self.max_pool:
                break
            a = float(alphas[i])
            if a <= 0.03:
                continue
            sz = glyph_sz * float(scales[i])
            half_sz = sz * 0.5
            self.pool_colors[idx].rgba = (
                rgb[i, 0] / 255.0,
                rgb[i, 1] / 255.0,
                rgb[i, 2] / 255.0,
                a,
            )
            rect = self.pool_rects[idx]
            rect.pos = (float(kx[i]) - half_sz, float(ky[i]) - half_sz)
            rect.size = (sz, sz)
            rect.tex_coords = self.tex_coords_map[int(cids[i])]
            idx += 1

        for k in range(idx, self.max_pool):
            if self.pool_colors[k].a == 0:
                break
            self.pool_colors[k].a = 0.0
            self.pool_rects[k].size = (0, 0)


class TouchButton(Button):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.background_normal = ""
        self.background_color = (0.09, 0.14, 0.17, 0.88)
        self.color = (0.90, 0.99, 0.96, 1.0)
        self.font_size = dp(12)
        self.bold = True
        self.halign = "center"


class WildflowerAndroidApp(App):
    def __init__(self, theme_name: str = "Wildflower", seed: int = 2026, **kwargs) -> None:
        super().__init__(**kwargs)
        self.initial_theme = theme_name
        self.initial_seed = seed
        self.sim: Optional[WildflowerSimulation] = None
        self.canvas_widget: Optional[WildflowerCanvasWidget] = None
        self.status_label: Optional[Label] = None
        self.controls_panel: Optional[GridLayout] = None
        self.hud_visible: bool = True
        self.is_fullscreen: bool = False
        self.exporting: bool = False
        self.status_clear_at: float = 0.0
        self.btn_pause: Optional[TouchButton] = None
        self.btn_speed: Optional[TouchButton] = None
        self.btn_density: Optional[TouchButton] = None
        self.btn_theme: Optional[TouchButton] = None

    def build(self):
        self.title = "WILDFLOWER GENERATIVE ART"
        self.sim = WildflowerSimulation(
            seed=self.initial_seed,
            theme_name=self.initial_theme,
            quality_scale=0.80,
            max_particles=260,
        )

        root = FloatLayout()
        self.canvas_widget = WildflowerCanvasWidget(
            self.sim,
            on_tap_cb=self.toggle_hud,
            size_hint=(1, 1),
            pos_hint={"x": 0, "y": 0},
        )
        root.add_widget(self.canvas_widget)

        self.status_label = Label(
            text=f"Theme: {self.sim.theme_name} | Tap layar: UI | Geser: Angin",
            size_hint=(0.94, None),
            height=dp(36),
            pos_hint={"center_x": 0.5, "top": 0.985},
            color=(0.75, 0.96, 0.90, 0.95),
            font_size=dp(12),
            halign="center",
            valign="middle",
        )
        self.status_label.bind(size=lambda inst, val: setattr(inst, "text_size", val))
        root.add_widget(self.status_label)

        self.controls_panel = GridLayout(
            cols=4,
            spacing=dp(6),
            padding=dp(8),
            size_hint=(0.98, None),
            height=dp(156),
            pos_hint={"center_x": 0.5, "y": 0.01},
        )

        self.btn_pause = TouchButton(text="PAUSE", on_release=lambda *_: self._act_pause())
        btn_reset = TouchButton(text="RESET", on_release=lambda *_: self._act_reset())
        btn_rand = TouchButton(text="RANDOM", on_release=lambda *_: self._act_random())
        self.btn_theme = TouchButton(
            text=f"TEMA\n{self.sim.theme_name.split()[0]}",
            on_release=lambda *_: self._act_theme(),
        )

        self.btn_speed = TouchButton(
            text=f"SPEED\n{self.sim.speed:.2f}x",
            on_release=lambda *_: self._act_speed_cycle(),
        )
        self.btn_density = TouchButton(
            text=f"PARTIKEL\n{self.sim.particle_density:.2f}x",
            on_release=lambda *_: self._act_density_cycle(),
        )
        btn_full = TouchButton(text="FULL\nSCREEN", on_release=lambda *_: self._act_fullscreen())
        btn_png = TouchButton(text="SAVE\nPNG", on_release=lambda *_: self._start_export("png"))

        btn_gif = TouchButton(text="EXPORT\nGIF", on_release=lambda *_: self._start_export("gif"))
        btn_mp4 = TouchButton(text="EXPORT\nMP4", on_release=lambda *_: self._start_export("mp4"))
        btn_spd_dn = TouchButton(text="SPD -", on_release=lambda *_: self._act_speed(-0.25))
        btn_den_dn = TouchButton(text="DEN -", on_release=lambda *_: self._act_density(-0.25))

        for widget in [
            self.btn_pause, btn_reset, btn_rand, self.btn_theme,
            self.btn_speed, self.btn_density, btn_full, btn_png,
            btn_gif, btn_mp4, btn_spd_dn, btn_den_dn,
        ]:
            self.controls_panel.add_widget(widget)

        root.add_widget(self.controls_panel)

        Window.bind(size=self._on_window_resize)
        self._on_window_resize(Window, Window.size)

        Clock.schedule_interval(self._update, 1.0 / 50.0)
        return root

    def _on_window_resize(self, _window, size: Tuple[int, int]) -> None:
        if self.controls_panel is None:
            return
        w, h = size
        is_landscape = w > h
        if is_landscape:
            self.controls_panel.cols = 6
            self.controls_panel.height = dp(104)
        else:
            self.controls_panel.cols = 4
            self.controls_panel.height = dp(164)

    def _set_status(self, msg: str, duration: float = 3.5) -> None:
        def apply(_dt=None) -> None:
            if self.status_label is not None:
                self.status_label.text = msg
            self.status_clear_at = time.time() + duration

        Clock.schedule_once(apply, 0)

    def toggle_hud(self) -> None:
        self.hud_visible = not self.hud_visible
        if self.controls_panel is not None:
            self.controls_panel.opacity = 1.0 if self.hud_visible else 0.0
            self.controls_panel.disabled = not self.hud_visible
        if self.status_label is not None:
            self.status_label.opacity = 1.0 if self.hud_visible else 0.0

    def _act_pause(self) -> None:
        assert self.sim is not None and self.btn_pause is not None
        paused = self.sim.toggle_pause()
        self.btn_pause.text = "START" if paused else "PAUSE"
        self._set_status("Animasi Dijeda" if paused else "Animasi Berjalan")

    def _act_reset(self) -> None:
        assert self.sim is not None and self.btn_pause is not None
        self.sim.reset()
        self.sim.set_paused(False)
        self.btn_pause.text = "PAUSE"
        self._set_status("Reset animasi dari awal")

    def _act_random(self) -> None:
        assert self.sim is not None
        seed = self.sim.randomize()
        self._set_status(f"Buket acak baru (Seed: {seed})")

    def _act_theme(self) -> None:
        assert self.sim is not None and self.btn_theme is not None
        name = self.sim.next_theme()
        self.btn_theme.text = f"TEMA\n{name.split()[0]}"
        self._set_status(f"Tema: {name}")

    def _act_speed_cycle(self) -> None:
        assert self.sim is not None and self.btn_speed is not None
        nxt = self.sim.speed + 0.25
        if nxt > 2.5:
            nxt = 0.5
        self.sim.speed = round(nxt, 2)
        self.btn_speed.text = f"SPEED\n{self.sim.speed:.2f}x"
        self._set_status(f"Kecepatan: {self.sim.speed:.2f}x")

    def _act_speed(self, delta: float) -> None:
        assert self.sim is not None and self.btn_speed is not None
        spd = self.sim.change_speed(delta)
        self.btn_speed.text = f"SPEED\n{spd:.2f}x"
        self._set_status(f"Kecepatan: {spd:.2f}x")

    def _act_density_cycle(self) -> None:
        assert self.sim is not None and self.btn_density is not None
        nxt = self.sim.particle_density + 0.25
        if nxt > 2.0:
            nxt = 0.25
        self.sim.particle_density = round(nxt, 2)
        self.btn_density.text = f"PARTIKEL\n{self.sim.particle_density:.2f}x"
        self._set_status(f"Partikel: {self.sim.particle_density:.2f}x")

    def _act_density(self, delta: float) -> None:
        assert self.sim is not None and self.btn_density is not None
        den = self.sim.change_density(delta)
        self.btn_density.text = f"PARTIKEL\n{den:.2f}x"
        self._set_status(f"Partikel: {den:.2f}x")

    def _act_fullscreen(self) -> None:
        self.is_fullscreen = not self.is_fullscreen
        try:
            Window.fullscreen = "auto" if self.is_fullscreen else False
        except Exception:
            pass
        if self.is_fullscreen:
            self.toggle_hud()

    def _start_export(self, kind: str) -> None:
        assert self.sim is not None
        if self.exporting:
            self._set_status("Proses ekspor masih berjalan...")
            return

        out_dir = get_android_exports_dir(self)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sim_copy = self.sim.clone()
        self.exporting = True
        is_landscape = Window.width > Window.height

        if kind == "png":
            path = os.path.join(out_dir, f"wildflower_{stamp}.png")
            out_w = 1920 if is_landscape else 1080
            out_h = 1080 if is_landscape else 1920
            self._set_status("Menyimpan PNG HD...", 10.0)

            def job() -> None:
                try:
                    export_png(sim_copy, path, width=out_w, height=out_h)
                    self._set_status(f"PNG disimpan: {path}", 5.0)
                except Exception as exc:
                    self._set_status(f"Gagal PNG: {exc}", 5.0)
                finally:
                    self.exporting = False

            threading.Thread(target=job, daemon=True).start()

        elif kind == "gif":
            path = os.path.join(out_dir, f"wildflower_{stamp}.gif")
            self._set_status("Membuat GIF (0%)...", 60.0)

            def job() -> None:
                try:
                    export_gif(
                        sim_copy,
                        path,
                        width=540,
                        height=720,
                        fps=15,
                        duration=5.5,
                        progress_cb=lambda p: self._set_status(f"Membuat GIF ({int(p * 100)}%)...", 60.0),
                    )
                    self._set_status(f"GIF disimpan: {path}", 5.0)
                except Exception as exc:
                    self._set_status(f"Gagal GIF: {exc}", 5.0)
                finally:
                    self.exporting = False

            threading.Thread(target=job, daemon=True).start()

        elif kind == "mp4":
            path = os.path.join(out_dir, f"wildflower_{stamp}.mp4")
            out_w = 1280 if is_landscape else 720
            out_h = 720 if is_landscape else 1280
            self._set_status("Mengekspor MP4 (0%)...", 90.0)

            def job() -> None:
                try:
                    export_mp4(
                        sim_copy,
                        path,
                        width=out_w,
                        height=out_h,
                        fps=24,
                        duration=6.0,
                        progress_cb=lambda p: self._set_status(f"Mengekspor MP4 ({int(p * 100)}%)...", 90.0),
                    )
                    self._set_status(f"MP4 disimpan: {path}", 5.0)
                except Exception as exc:
                    self._set_status(f"MP4 butuh ffmpeg: {exc}", 6.0)
                finally:
                    self.exporting = False

            threading.Thread(target=job, daemon=True).start()

    def _update(self, dt: float) -> None:
        if self.sim is None or self.canvas_widget is None:
            return
        self.sim.step(dt)
        frame = self.sim.build_frame()
        self.canvas_widget.render_frame(frame)

        if (
            self.status_label is not None
            and self.status_clear_at > 0
            and time.time() > self.status_clear_at
        ):
            self.status_clear_at = 0.0
            self.status_label.text = f"Theme: {self.sim.theme_name} | Seed: {self.sim.seed}"


def main(theme: str = "Wildflower", seed: int = 2026) -> None:
    WildflowerAndroidApp(theme_name=theme, seed=seed).run()


if __name__ == "__main__":
    main()
