"""
WILDFLOWER GENERATIVE ART - Unified Entry Point
Otomatis memilih versi Desktop (Pygame) atau Android (Kivy),
serta menyediakan mode --export-demo untuk membuat screenshot, GIF, dan MP4 GitHub.
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)


def is_android_env() -> bool:
    android_vars = ("ANDROID_ARGUMENT", "ANDROID_PRIVATE", "ANDROID_BOOTLOGO", "PYDROID_RPC")
    if any(var in os.environ for var in android_vars):
        return True
    if os.path.isdir("/data/data/ru.iiec.pydroid3") or os.path.isdir("/data/user/0/ru.iiec.pydroid3"):
        return True
    return False


def run_export_demo(theme: str, seed: int) -> None:
    from core.animation import WildflowerSimulation
    from core.rendering import export_gif, export_mp4, export_png

    assets_dir = os.path.join(ROOT_DIR, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    sim = WildflowerSimulation(seed=seed, theme_name=theme, quality_scale=1.0)
    # Majukan simulasi ke saat buket mekar penuh untuk screenshot utama
    for _ in range(260):
        sim.step(1.0 / 60.0)

    png_path = os.path.join(assets_dir, "screenshot.png")
    print(f"[1/3] Membuat screenshot HD -> {png_path}")
    export_png(sim, png_path, width=1920, height=1080)

    gif_path = os.path.join(assets_dir, "demo.gif")
    print(f"[2/3] Membuat demo GIF untuk GitHub mobile -> {gif_path}")
    export_gif(
        sim.clone(),
        gif_path,
        width=640,
        height=640,
        fps=18,
        duration=6.5,
        from_start=True,
        progress_cb=lambda p: print(f"  GIF progress: {int(p * 100)}%", end="\r"),
    )
    print("\n  GIF selesai.")

    mp4_path = os.path.join(assets_dir, "demo.mp4")
    print(f"[3/3] Membuat demo MP4 HD -> {mp4_path}")
    try:
        export_mp4(
            sim.clone(),
            mp4_path,
            width=1280,
            height=720,
            fps=30,
            duration=8.0,
            from_start=True,
            progress_cb=lambda p: print(f"  MP4 progress: {int(p * 100)}%", end="\r"),
        )
        print("\n  MP4 selesai.")
    except RuntimeError as exc:
        print(f"\n  Lewati MP4: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wildflower Generative Art Launcher")
    parser.add_argument(
        "--mode",
        choices=["auto", "desktop", "android"],
        default="auto",
        help="Pilih mode aplikasi (default: auto-detect)",
    )
    parser.add_argument("--theme", default="Wildflower", help="Tema warna awal")
    parser.add_argument("--seed", type=int, default=2026, help="Seed acak buket")
    parser.add_argument(
        "--export-demo",
        action="store_true",
        help="Generate assets/screenshot.png, assets/demo.gif, dan assets/demo.mp4",
    )
    args = parser.parse_args()

    if args.export_demo:
        run_export_demo(args.theme, args.seed)
        return

    mode = args.mode
    if mode == "auto":
        mode = "android" if is_android_env() else "desktop"

    if mode == "android":
        from android.main_android import main as run_android

        run_android(theme=args.theme, seed=args.seed)
    else:
        from desktop.main_desktop import main as run_desktop

        run_desktop(theme=args.theme, seed=args.seed)


if __name__ == "__main__":
    main()
