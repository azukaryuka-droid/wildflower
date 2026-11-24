# WILDFLOWER GENERATIVE ART 🌸✨

Karya seni digital generatif berbahasa **Python murni** yang menampilkan buket bunga liar (*wildflower bouquet*) tersusun dari karakter **ASCII, titik dot-matrix, simbol, garis, dan partikel bercahaya**.

---

## 🎥 Demo & Screenshot

> Jalankan `python main.py --export-demo` terlebih dahulu sebelum melakukan `git push` agar `assets/demo.gif`, `assets/demo.mp4`, dan `assets/screenshot.png` langsung tampil di halaman GitHub (termasuk saat dibuka dari HP).

![Wildflower Generative Art Demo](assets/demo.gif)

![Wildflower Generative Art Screenshot](assets/screenshot.png)

🎬 Video kualitas penuh: [`assets/demo.mp4`](assets/demo.mp4)

---

## ✨ Fitur Utama

- **Arsitektur Core Bersama (`core/`)**: Satu algoritma animasi & geometri untuk Desktop dan Android.
- **Animasi Organik 9 Lapis**:
  1. Batang tumbuh bertahap dari pangkal bawah buket.
  2. Daun *dot-matrix* melengkung mengikuti pertumbuhan batang.
  3. Karakter ASCII berkumpul membentuk kelopak bunga secara bertahap.
  4. Kelopak mekar berlapis dengan inti bercahaya.
  5. Partikel kecil bergerak perlahan seperti serbuk sari / kunang-kunang.
  6. Perubahan warna halus (*smooth HSV hue wave*).
  7. Efek cahaya *additive bloom* dan kedipan karakter (*twinkle*).
  8. Ayunan alami seluruh buket dan daun seperti tertiup angin.
  9. Berjalan terus-menerus (*infinite loop*).
- **6 Tema Warna Pilihan**:
  - `Wildflower` *(kuning-hijau bercahaya, mint, coral, sky azure, lavender)*
  - `Pink Garden`
  - `Purple Dreams`
  - `Blue Night`
  - `Rainbow Garden`
  - `Sunset Flowers`
- **Ekspor Karya**: Simpan frame ke **PNG (1920x1080)**, animasi **GIF**, dan video **MP4**.

---

## 📦 Daftar Library yang Diperlukan

| Library | Kegunaan |
| :--- | :--- |
| `numpy` | Komputasi vektorisasi animasi & geometri bunga |
| `pillow` | Rasterisasi glyph, efek Gaussian bloom, & ekspor PNG/GIF |
| `pygame` | Aplikasi Desktop 60 FPS (Windows, Linux, macOS) |
| `kivy` | Aplikasi sentuh Android (Pydroid 3 & Buildozer APK) |
| `imageio` & `imageio-ffmpeg` | Ekspor animasi ke format video MP4 |

---

## 💻 Cara Instalasi & Menjalankan di Laptop (Windows / Linux / macOS)

### 1. Instal Python
Unduh dan pasang **Python 3.10+** dari [python.org](https://www.python.org/downloads/).
*(Di Windows, centang **"Add Python to PATH"** saat instalasi).*

### 2. Clone Repository & Pasang Dependensi
```bash
git clone https://github.com/USERNAME/wildflower.git
cd wildflower
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi Desktop
```bash
python main.py
```

### 4. Generate Demo GIF, MP4, & Screenshot untuk GitHub
```bash
python main.py --export-demo
```

### ⌨️ Kontrol Keyboard & Mouse (Desktop)
- `Klik Kiri Kanvas`: Meniupkan embusan angin
- `Spasi`: **START / PAUSE**
- `R`: **RESET** animasi
- `G` atau `Enter`: **RANDOM FLOWERS** (susunan buket acak baru)
- `C`: **CHANGE COLORS** (ganti tema)
- `[` / `]`: Kurangi / tambah **ANIMATION SPEED**
- `-` / `=`: Kurangi / tambah **PARTICLE DENSITY**
- `F` atau `F11`: **FULLSCREEN**
- `H`: Sembunyikan / tampilkan HUD tombol
- `S`: **SAVE IMAGE (PNG 1920x1080)**
- `E`: **EXPORT GIF**
- `M`: **EXPORT MP4**

---

## 📱 Cara Menjalankan di Android (Pydroid 3)

1. Pasang aplikasi **Pydroid 3** dari Google Play Store.
2. Buka menu **Pip** di Pydroid 3, lalu instal:
   - `numpy`
   - `pillow`
   - `kivy`
3. Salin folder `wildflower/` ke penyimpanan internal HP.
4. Buka file `wildflower/main.py` (atau `wildflower/android/main_android.py`) di Pydroid 3, lalu tekan tombol **Play (▶)**.

### Kontrol Layar Sentuh (Android)
- **Ketuk (Tap) layar**: Sembunyikan/tampilkan panel tombol sentuh (mode Zen).
- **Geser (Swipe) kiri/kanan**: Meniupkan angin ke buket bunga.
- **Rotasi Portrait / Landscape**: Tata letak tombol otomatis menyesuaikan orientasi layar.

---

## 🤖 Cara Membuat File APK Android (Buildozer)

Gunakan Linux atau WSL2 (Windows Subsystem for Linux):

```bash
pip install buildozer cython
sudo apt update
sudo apt install -y git zip unzip openjdk-17-jdk autoconf libtool pkg-config zlib1g-dev libncurses5-dev libncursesw5-dev cmake libffi-dev libssl-dev

cd wildflower
buildozer -v android debug
```
File APK hasil kompilasi akan berada di folder `wildflower/bin/`.

---

## 🖥️ Cara Membuat Aplikasi EXE (Windows)

```bash
pip install pyinstaller
cd wildflower
pyinstaller --noconfirm --onefile --windowed --name "WildflowerGenerativeArt" --add-data "core;core" --add-data "desktop;desktop" desktop/main_desktop.py
```
File `WildflowerGenerativeArt.exe` akan tersedia di folder `dist/`.
