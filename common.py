"""Fungsi bersama untuk semua langkah pipeline."""
import sys
from pathlib import Path

import numpy as np

# Judul YouTube sering berisi emoji; di Windows keluaran bawaan cp1252 tidak bisa
# menampilkannya dan script berhenti. Paksa UTF-8, karakter yang tak bisa ditulis diganti.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
import yaml

EKSTENSI_VIDEO = {".mp4", ".webm", ".mkv", ".mov"}
EKSTENSI_FOTO = {".jpg", ".jpeg", ".png", ".webp"}


def muat_config(path="config.yaml"):
    """Baca config.yaml; daftar tokoh dibaca dari tokoh.csv (kolom: id, nama, kategori, jk)."""
    import csv
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    tokoh_csv = Path(path).parent / "tokoh.csv"
    if tokoh_csv.exists():
        with open(tokoh_csv, encoding="utf-8") as f:
            cfg["tokoh"] = [{"id": r["id"].strip(), "nama": r["nama"].strip(),
                             "kategori": (r.get("kategori") or "").strip().lower(),
                             "jk": (r.get("jk") or "").strip().upper()}
                            for r in csv.DictReader(f) if r.get("id", "").strip()]
    cfg.setdefault("tokoh", [])
    return cfg


_FFMPEG = None


def ffmpeg_exe():
    """Path ffmpeg: dari PATH kalau ada, kalau tidak dari paket imageio-ffmpeg (ffmpeg bawaan
    Python, tidak perlu dipasang terpisah di Windows). Paket dipasang otomatis bila belum ada."""
    global _FFMPEG
    if _FFMPEG:
        return _FFMPEG
    import shutil
    import subprocess
    _FFMPEG = shutil.which("ffmpeg")
    if _FFMPEG:
        return _FFMPEG
    try:
        import imageio_ffmpeg
    except ImportError:
        print("ffmpeg belum ada; memasang paket imageio-ffmpeg (sekali saja)...", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "imageio-ffmpeg"], check=False)
        try:
            import imageio_ffmpeg
        except ImportError:
            raise RuntimeError("ffmpeg tidak ditemukan dan imageio-ffmpeg gagal dipasang. "
                               "Jalankan: python -m pip install imageio-ffmpeg")
    _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return _FFMPEG


def format_dash(path):
    """True kalau file MP4 berformat DASH (fragmented MP4 dari YouTube). Format ini terbaca
    OpenCV/ffmpeg, tetapi thumbnail Windows, Windows Media Player, dan browser sering gagal."""
    try:
        with open(path, "rb") as f:
            kepala = f.read(12)
    except OSError:
        return False
    return kepala[4:8] == b"ftyp" and kepala[8:12] == b"dash"


def remux_mp4(path):
    """Kemas ulang DASH menjadi MP4 biasa tanpa encode ulang (ffmpeg -c copy), di tempat.
    Kembalikan keterangan proses, atau None kalau tidak perlu. RuntimeError kalau gagal."""
    import os
    import subprocess
    path = Path(path)
    if not format_dash(path):
        return None
    ffmpeg = ffmpeg_exe()
    sementara = path.with_name(path.stem + ".remux.mp4")
    hasil = subprocess.run([ffmpeg, "-v", "error", "-y", "-i", str(path), "-map", "0", "-c", "copy",
                            "-movflags", "+faststart", str(sementara)], capture_output=True, text=True)
    if hasil.returncode != 0 or not sementara.exists() or sementara.stat().st_size == 0:
        sementara.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg gagal remux {path.name}: {hasil.stderr.strip()[:300]}")
    os.replace(sementara, path)
    return "remux DASH -> MP4 biasa: ffmpeg -map 0 -c copy -movflags +faststart (tanpa encode ulang)"


def folder(cfg, *bagian, buat=True):
    p = Path(cfg["folder_kerja"]).joinpath(*bagian)
    if buat:
        p.mkdir(parents=True, exist_ok=True)
    return p


_APP = None


PERANGKAT_AKTIF = None  # diisi setelah model dimuat, ikut dicatat di log forensik


def _pengaturan_perangkat():
    try:
        return (muat_config().get("perangkat") or {})
    except Exception:  # noqa: BLE001
        return {}


def provider_onnx(mode=None):
    """Daftar execution provider ONNX Runtime. GPU (CUDA) dipakai kalau tersedia, kecuali
    config `perangkat.inferensi: cpu`. Kembalikan (providers, keterangan)."""
    import onnxruntime as ort
    mode = (mode or _pengaturan_perangkat().get("inferensi") or "auto").lower()
    tersedia = ort.get_available_providers()
    if mode != "cpu" and "CUDAExecutionProvider" in tersedia:
        # kalau torch sudah dimuat, ONNX Runtime memakai DLL CUDA milik torch (dokumentasi ORT);
        # memuat DLL kedua kalinya bisa bentrok
        if hasattr(ort, "preload_dlls") and "torch" not in sys.modules:
            try:  # muat DLL CUDA/cuDNN dari paket pip nvidia-*
                ort.preload_dlls()
            except Exception:  # noqa: BLE001
                pass
        return [("CUDAExecutionProvider", {"device_id": 0, "cudnn_conv_algo_search": "HEURISTIC"}),
                "CPUExecutionProvider"], "cuda"
    if mode == "gpu":
        raise RuntimeError("perangkat.inferensi = gpu, tetapi CUDAExecutionProvider tidak tersedia "
                           f"(onnxruntime {ort.__version__}, provider: {tersedia}). "
                           "Pasang onnxruntime-gpu lewat JALANKAN.bat.")
    return ["CPUExecutionProvider"], "cpu"


def penganalisis_wajah():
    """RetinaFace (deteksi) + ArcFace (pengenalan) dari InsightFace, di GPU kalau ada."""
    global _APP, PERANGKAT_AKTIF
    if _APP is None:
        from insightface.app import FaceAnalysis
        providers, mode = provider_onnx()
        _APP = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                            providers=providers)
        _APP.prepare(ctx_id=0 if mode == "cuda" else -1, det_size=(640, 640))
        # cek provider yang BENAR-BENAR dipakai (CUDA bisa diam-diam jatuh ke CPU)
        dipakai = set()
        for m in getattr(_APP, "models", {}).values():
            sesi = getattr(m, "session", None)
            if sesi is not None:
                dipakai.update(sesi.get_providers()[:1])
        PERANGKAT_AKTIF = "GPU (CUDA)" if dipakai == {"CUDAExecutionProvider"} else \
            f"CPU ({', '.join(sorted(dipakai)) or 'tidak diketahui'})"
        print(f"[InsightFace] berjalan di {PERANGKAT_AKTIF}", flush=True)
        if mode == "cuda" and "CUDA" not in PERANGKAT_AKTIF:
            print("[InsightFace] PERINGATAN: CUDA tersedia tapi model jatuh ke CPU. "
                  "Jalankan cek_alat.py untuk detailnya.", flush=True)
    return _APP


def args_encoder():
    """Argumen ffmpeg untuk encoding klip. Default libx264 (CPU) supaya kompresi semua klip
    seragam; ini penting untuk dataset deteksi deepfake karena artefak kompresi ikut dipelajari
    model. `perangkat.encoder_video: h264_nvenc` memakai encoder GPU NVIDIA (lebih cepat)."""
    enc = (_pengaturan_perangkat().get("encoder_video") or "libx264").lower()
    if enc == "h264_nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p5", "-tune", "hq", "-rc", "vbr", "-cq", "18",
                "-b:v", "0", "-pix_fmt", "yuv420p"], "h264_nvenc cq 18 preset p5"
    return ["-c:v", "libx264", "-crf", "18", "-preset", "fast", "-pix_fmt", "yuv420p"], "libx264 crf 18"


def lebar_wajah(face):
    return float(face.bbox[2] - face.bbox[0])


def wajah_terbesar(faces):
    if not faces:
        return None
    return max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))


def frontal(face, batas=0.2):
    """Cek kasar wajah menghadap depan: hidung harus kira-kira di tengah kedua mata."""
    mata_kiri, mata_kanan, hidung = face.kps[0], face.kps[1], face.kps[2]
    jarak = mata_kanan[0] - mata_kiri[0]
    if jarak <= 0:
        return False
    r = (hidung[0] - mata_kiri[0]) / jarak
    return batas < r < 1 - batas


def embedding_referensi(cfg, tokoh_id):
    """Rata-rata embedding ArcFace dari foto di data/referensi/<tokoh_id>/."""
    import cv2
    app = penganalisis_wajah()
    embs = []
    for p in sorted(folder(cfg, "referensi", tokoh_id, buat=False).glob("*")):
        if p.suffix.lower() not in EKSTENSI_FOTO:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        f = wajah_terbesar(app.get(img))
        if f is not None:
            embs.append(f.normed_embedding)
    if not embs:
        return None
    m = np.mean(embs, axis=0)
    return m / np.linalg.norm(m)


def crop_wajah(img, face, margin, ukuran):
    """Crop persegi di sekitar wajah (dengan margin), lalu resize."""
    import cv2
    x1, y1, x2, y2 = face.bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * margin / 2
    l, t, r, b = int(cx - s), int(cy - s), int(cx + s), int(cy + s)
    h, w = img.shape[:2]
    pad = max(0, -l, -t, r - w, b - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        l, t, r, b = l + pad, t + pad, r + pad, b + pad
    return cv2.resize(img[t:b, l:r], (ukuran, ukuran), interpolation=cv2.INTER_AREA)
