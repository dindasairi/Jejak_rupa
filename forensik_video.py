"""Mesin aplikasi forensik deepfake: memeriksa video barang bukti dengan model Xception + LSTM
yang dibangun di langkah 6, mengikuti alur yang sama dengan pipeline dataset.

Alur per barang bukti (semua tercatat di log berantai milik kasus, sama seperti log_forensik.jsonl):
  1. Akuisisi    : hitung MD5/SHA-1/SHA-256 file asal, salin ke folder kasus (salinan kerja,
                   dijadikan read-only), hitung ulang hash salinan dan pastikan identik.
  2. Metadata    : properti video (OpenCV), info stream + tag kontainer (FFmpeg), temuan metadata.
  3. Analisis    : verifikasi hash salinan -> video dipotong menjadi segmen `durasi_segmen_detik`
                   (sama dengan durasi klip dataset, 10 dtk) -> 20 frame per segmen dengan cara yang
                   sama dengan langkah 5 (baku: 20 frame berurutan 25 fps di tengah segmen) ->
                   MTCNN dengan parameter langkah 5 (margin 1.3, 224x224, prob >= 0.95, filter tepi
                   hitam) -> wajah dilacak per orang (IoU antarframe, identitas antarsegmen dengan
                   ArcFace) -> urutan wajah per orang per segmen -> Xception + LSTM -> skor fake.
  4. Grad-CAM    : setiap segmen yang melewati ambang + segmen tertinggi tiap orang (07_gradcam.py),
                   ditandai di garis waktu supaya temuan visual dan skor waktu bisa dicocokkan.
  5. Laporan     : HTML + PDF (laporan_forensik.py).

Folder kasus: data/kasus_forensik/<nomor_kasus>/
  kasus.json, log_forensik.jsonl
  bukti/BB-001/salinan/<nama file>        salinan kerja (read-only)
  original files/BB-001/<nama file>       salinan file asli apa adanya (read-only, hash diverifikasi)
  bukti/BB-001/bukti.json                 hash, asal, metadata
  bukti/BB-001/analisis/<waktu>/          hasil.json, manifest.csv, wajah/, gradcam/, kunci/
  laporan/laporan_<waktu>.html|.pdf
"""
import csv
import getpass
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np

import forensik
from bahasa import T, kes, tr
from common import ffmpeg_exe, muat_config
from forensik import catat, sekarang, sha256_file

AKAR = Path("data") / "kasus_forensik"
EKSTENSI = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".3gp", ".wmv", ".flv", ".mpg", ".mpeg", ".ts"}
ATURAN_FRAME = {"otomatis": "sesuai model", "merata": "merata sepanjang segmen (aturan lama)",
                "berurutan": "20 frame berurutan 25 fps (aturan baru)"}
BAKU = {"aturan_frame": "otomatis", "durasi_segmen_detik": 10, "frame_per_segmen": 20, "maks_segmen": 120, "maks_orang": 3,
        "iou_lacak": 0.3, "ambang_identitas": 0.45, "zona_ragu": 0.1, "gradcam_maks": 60,
        "gradcam_cakupan": "lewat_ambang", "pisah_jejak": True, "ambang_ganti_orang": 0.35,
        "ambang_ganti_adegan": 0.5, "ambang": "fpr5",
        "ambang_rendah": "val"}  # >= ambang (batas toleransi 5%) = terindikasi; Youden s.d. batas = perlu diperiksa
KESIMPULAN = {
    "TERINDIKASI": "Terindikasi deepfake",
    "RAGU": "Perlu pemeriksaan lanjutan",
    "TIDAK": "Tidak terindikasi deepfake",
    "TIDAK_ADA_WAJAH": "Tidak dapat dianalisis (wajah tidak cukup)",
}
_PERINGKAT = {"TERINDIKASI": 3, "RAGU": 2, "TIDAK": 1, "TIDAK_ADA_WAJAH": 0}


def pengaturan(cfg=None):
    cfg = cfg or muat_config()
    p = dict(BAKU)
    p.update(cfg.get("forensik") or {})
    return p


def muat_modul(nama_file, nama):
    spec = importlib.util.spec_from_file_location(nama, Path(__file__).with_name(nama_file))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tulis_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _baca_json(path, baku=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return baku


def _isi_cache_hash(path, sha256):
    """Supaya forensik.catat tidak menghitung ulang SHA-256 file besar yang baru saja di-hash."""
    p = Path(path)
    forensik._CACHE_HASH[(str(p.resolve()), p.stat().st_mtime_ns, p.stat().st_size)] = sha256


def slug(teks):
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(teks).strip()).strip("._")
    return s[:60] or "kasus"


# =================================================================== kasus
def folder_kasus(kasus_id):
    d = AKAR / kasus_id
    if not (d / "kasus.json").exists():
        raise FileNotFoundError(T(f"Kasus {kasus_id} tidak ditemukan", f"Case {kasus_id} not found"))
    return d


def buat_kasus(nomor, judul, pemeriksa, instansi="", keterangan=""):
    if not str(nomor).strip() or not str(pemeriksa).strip():
        raise ValueError(T("Nomor kasus dan nama pemeriksa wajib diisi", "Case number and examiner name are required"))
    kid = slug(nomor)
    d = AKAR / kid
    if (d / "kasus.json").exists():
        raise FileExistsError(T(f"Kasus dengan nomor {nomor} sudah ada", f"A case with number {nomor} already exists"))
    (d / "bukti").mkdir(parents=True, exist_ok=True)
    utc, wib = sekarang()
    data = {"id": kid, "nomor": str(nomor).strip(), "judul": str(judul).strip(), "pemeriksa": str(pemeriksa).strip(),
            "instansi": str(instansi).strip(), "keterangan": str(keterangan).strip(),
            "dibuat_utc": utc, "dibuat_wib": wib, "akun_os": getpass.getuser(), "komputer": platform.node()}
    _tulis_json(d / "kasus.json", data)
    catat("kasus_dibuat", d / "kasus.json", folder_kerja=d, info={"nomor": data["nomor"], "pemeriksa": data["pemeriksa"]})
    return data


def daftar_kasus():
    hasil = []
    if AKAR.exists():
        for f in sorted(AKAR.glob("*/kasus.json")):
            k = _baca_json(f, {})
            k["jumlah_bukti"] = len(list((f.parent / "bukti").glob("BB-*/bukti.json")))
            hasil.append(k)
    return sorted(hasil, key=lambda k: k.get("dibuat_utc", ""), reverse=True)


def analisis_terakhir(folder_bukti):
    runs = sorted((Path(folder_bukti) / "analisis").glob("*/hasil.json"))
    return _baca_json(runs[-1]) if runs else None


def ringkas_analisis(a):
    top = max(a["ringkasan_orang"], key=lambda o: o["skor_rata"], default=None)
    ek = a.get("ekstraksi", {})
    sesuai = a.get("aturan_frame", {}).get("sesuai_model")
    if sesuai is None:  # analisis lama (sebelum pilihan aturan): bandingkan dengan aturan latih modelnya
        try:
            sesuai = ek.get("mode_frame", "merata") == aturan_frame_model(a["model"]["folder"])["mode_frame"]
        except Exception:  # noqa: BLE001
            sesuai = True
    return {"id": a["id"], "waktu_wib": a["waktu_wib"], "model": a["model"]["nama"],
            "skenario": a["model"].get("skenario"), "nama_skenario": a["model"].get("nama_skenario"),
            "aturan_frame": ek.get("mode_frame", "merata"), "fps_sampel": ek.get("fps_sampel"),
            "aturan_sesuai_model": sesuai,
            "ambang": a["ambang"], "ambang_rendah": a.get("ambang_rendah"), "status": a["status"], "kesimpulan": a["kesimpulan"],
            "skor_rata_maks": top["skor_rata"] if top else None, "orang_skor_tertinggi": top["orang"] if top else None,
            "orang": [{k: o[k] for k in ("orang", "skor_rata", "skor_maks", "segmen_dianalisis", "status")}
                      for o in a["ringkasan_orang"]], "jumlah_segmen": a["jumlah_segmen"],
            "durasi_proses_detik": a["durasi_proses_detik"], "gabungan": a.get("gabungan"),
            "anggota": [x["nama"] for x in a["model"].get("anggota", [])],
            "skor_rata_model": top.get("skor_rata_model") if top else None}


def riwayat_analisis(folder_bukti):
    hasil = []
    for f in sorted((Path(folder_bukti) / "analisis").glob("*/hasil.json")):
        a = _baca_json(f)
        if a:
            hasil.append(ringkas_analisis(a))
    return hasil


def muat_analisis(kasus_id, bb, analisis_id):
    a = _baca_json(folder_kasus(kasus_id) / "bukti" / bb / "analisis" / analisis_id / "hasil.json")
    if not a:
        raise FileNotFoundError(T(f"Analisis {analisis_id} untuk {bb} tidak ditemukan",
                                  f"Analysis {analisis_id} for {bb} not found"))
    return a


def muat_kasus(kasus_id):
    d = folder_kasus(kasus_id)
    kasus = _baca_json(d / "kasus.json")
    bukti = []
    for f in sorted((d / "bukti").glob("BB-*/bukti.json")):
        b = _baca_json(f, {})
        a = analisis_terakhir(f.parent)
        b["analisis"] = a
        b["riwayat_analisis"] = riwayat_analisis(f.parent)
        b["jumlah_analisis"] = len(b["riwayat_analisis"])
        b["pratinjau"] = info_pratinjau(f.parent)
        b["perlu_pratinjau"] = perlu_pratinjau(b)
        bukti.append(b)
    laporan = []
    for p in sorted((d / "laporan").glob("laporan_*.*"), reverse=True):
        if p.suffix in (".html", ".pdf"):
            laporan.append({"file": p.as_posix(), "nama": p.name, "ukuran_byte": p.stat().st_size,
                            "waktu": datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")})
    return {"kasus": kasus, "bukti": bukti, "laporan": laporan, "log": forensik.baca_log(d)}


# ================================================================ akuisisi
def hash_file(path, kemajuan=None):
    h = {k: hashlib.new(k) for k in ("md5", "sha1", "sha256")}
    total, n = max(1, Path(path).stat().st_size), 0
    with open(path, "rb") as f:
        for blok in iter(lambda: f.read(4 << 20), b""):
            for x in h.values():
                x.update(blok)
            n += len(blok)
            if kemajuan:
                kemajuan(n / total)
    return {k: v.hexdigest() for k, v in h.items()}


def tambah_bukti(kasus_id, sumber, keterangan="", log=print, kemajuan=None):
    d = folder_kasus(kasus_id)
    src = Path(str(sumber).strip().strip('"'))
    if not src.is_file():
        raise FileNotFoundError(T(f"File tidak ditemukan: {src}", f"File not found: {src}"))
    if src.suffix.lower() not in EKSTENSI:
        raise ValueError(T(f"{src.name}: bukan file video yang didukung ({', '.join(sorted(EKSTENSI))})",
                           f"{src.name}: not a supported video file ({', '.join(sorted(EKSTENSI))})"))
    ada = sorted((d / "bukti").glob("BB-*"))
    no = 1 + max([int(p.name[3:]) for p in ada if p.name[3:].isdigit()] or [0])
    bb = f"BB-{no:03d}"
    fb = d / "bukti" / bb
    (fb / "salinan").mkdir(parents=True, exist_ok=True)
    utc0, wib0 = sekarang()
    st0 = src.stat()
    log(T(f"[{bb}] Menghitung hash file asal: {src}", f"[{bb}] Hashing source file: {src}"))
    h_asal = hash_file(src, lambda f: kemajuan and kemajuan(0.3 * f, T("hash file asal", "hashing source file")))
    # salinan file asli (arsip, tidak pernah dibuka untuk analisis) di <kasus>/original files/<BB>/
    ori = d / "original files" / bb / src.name
    ori.parent.mkdir(parents=True, exist_ok=True)
    log(T(f"[{bb}] Menyalin file asli ke: {ori}", f"[{bb}] Copying original file to: {ori}"))
    shutil.copyfile(src, ori)
    shutil.copystat(src, ori)
    h_ori = hash_file(ori, lambda f: kemajuan and kemajuan(0.3 + 0.25 * f, T("hash salinan file asli", "hashing original copy")))
    if h_ori != h_asal:
        log(T(f"[{bb}] ! HASH TIDAK COCOK: salinan di original files berbeda dari file asal",
              f"[{bb}] ! HASH MISMATCH: the copy in original files differs from the source file"))
    os.chmod(ori, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    dst = fb / "salinan" / src.name
    log(T(f"[{bb}] Menyalin ke salinan kerja: {dst}", f"[{bb}] Copying to working copy: {dst}"))
    shutil.copyfile(src, dst)
    shutil.copystat(src, dst)
    log(T(f"[{bb}] Menghitung hash salinan", f"[{bb}] Hashing working copy"))
    h_salin = hash_file(dst, lambda f: kemajuan and kemajuan(0.55 + 0.25 * f, T("hash salinan kerja", "hashing working copy")))
    cocok = h_asal == h_salin and h_asal == h_ori
    if h_asal != h_salin:
        log(T(f"[{bb}] ! HASH TIDAK COCOK: salinan kerja berbeda dari file asal",
              f"[{bb}] ! HASH MISMATCH: the working copy differs from the source file"))
    os.chmod(dst, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    if kemajuan:
        kemajuan(0.85, "metadata")
    log(T(f"[{bb}] Membaca metadata", f"[{bb}] Reading metadata"))
    meta = metadata_video(dst)
    utc, wib = sekarang()
    data = {"id": bb, "kasus": kasus_id, "nama_file": src.name, "keterangan": str(keterangan).strip(),
            "asal": {"path": str(src.resolve()), "ukuran_byte": st0.st_size,
                     "diubah": datetime.fromtimestamp(st0.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
                     "dibuat": datetime.fromtimestamp(st0.st_ctime).strftime("%Y-%m-%d %H:%M:%S"),
                     "hash": h_asal},
            "salinan": {"file": dst.as_posix(), "hash": h_salin, "read_only": True},
            "file_asli": {"file": ori.as_posix(), "hash": h_ori, "read_only": True},
            "hash_cocok": cocok, "akuisisi_mulai_wib": wib0, "akuisisi_selesai_wib": wib,
            "akuisisi_utc": utc, "operator": getpass.getuser(), "komputer": platform.node(),
            "metadata": meta}
    _tulis_json(fb / "bukti.json", data)
    _isi_cache_hash(dst, h_salin["sha256"])
    catat("salinan_file_asli", ori, folder_kerja=d, info={
        "bukti": bb, "path_asal": str(src.resolve()), "hash_asal": h_asal, "hash_salinan": h_ori,
        "hash_cocok": h_ori == h_asal})
    catat("akuisisi_bukti", dst, folder_kerja=d, info={
        "bukti": bb, "path_asal": str(src.resolve()), "hash_asal": h_asal, "hash_salinan": h_salin,
        "hash_cocok": cocok, "keterangan": data["keterangan"]})
    catat("metadata_bukti", fb / "bukti.json", induk=[dst], folder_kerja=d, info={"bukti": bb})
    log(T(f"[{bb}] Selesai. SHA-256 {h_salin['sha256']} ({'cocok' if cocok else 'TIDAK COCOK'})",
          f"[{bb}] Done. SHA-256 {h_salin['sha256']} ({'match' if cocok else 'MISMATCH'})"))
    return data


# ================================================================ video pratinjau (pemutaran saja)
KODEK_PEMUTAR = ("h264", "vp8", "vp9", "av1")  # kodek yang bisa diputar WebView/browser


def perlu_pratinjau(b):
    """True kalau kodek salinan tidak bisa diputar pemutar aplikasi (mis. MPEG-4 Part 2 / mp4v pada Celeb-DF)."""
    kodek = str((b.get("metadata") or {}).get("kodek_video") or "").split(" ")[0].lower()
    return bool(kodek) and kodek not in KODEK_PEMUTAR


def info_pratinjau(folder_bukti):
    j = Path(folder_bukti) / "pratinjau" / "pratinjau.json"
    if not j.exists():
        return None
    d = _baca_json(j, {})
    return d if d.get("file") and Path(d["file"]).exists() else None


def buat_pratinjau(kasus_id, bb, log=print, kemajuan=None):
    """Konversi salinan kerja ke H.264 HANYA untuk ditonton di aplikasi. File ini bukan barang bukti: analisis,
    hash, dan laporan tetap memakai salinan kerja asli. Proses dan hash-nya dicatat di log kasus."""
    d = folder_kasus(kasus_id)
    fb = d / "bukti" / bb
    b = _baca_json(fb / "bukti.json")
    ada = info_pratinjau(fb)
    if ada:
        log(T(f"[{bb}] Video pratinjau sudah ada: {ada['file']}", f"[{bb}] Preview video already exists: {ada['file']}"))
        return ada
    integ = periksa_integritas(kasus_id, bb, log=log)
    if integ["status"] != "COCOK":
        raise RuntimeError(T(f"Integritas salinan {integ['status']}: pratinjau tidak dibuat.",
                             f"Working copy integrity {integ['status']}: preview not created."))
    src = Path(b["salinan"]["file"])
    keluar = fb / "pratinjau" / f"{src.stem}_pratinjau_h264.mp4"
    keluar.parent.mkdir(parents=True, exist_ok=True)
    perintah = [ffmpeg_exe(), "-v", "error", "-y", "-i", str(src), "-map", "0:v:0", "-map", "0:a?",
                "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(keluar)]
    log(T(f"[{bb}] Membuat video pratinjau H.264 (kodek asli: {b['metadata'].get('kodek_video')})",
          f"[{bb}] Creating H.264 preview video (original codec: {b['metadata'].get('kodek_video')})"))
    if kemajuan:
        kemajuan(0.3, T("konversi H.264", "H.264 conversion"))
    r = subprocess.run(perintah, capture_output=True, text=True)
    if r.returncode != 0 or not keluar.exists():
        raise RuntimeError(T("ffmpeg gagal membuat pratinjau: ", "ffmpeg failed to create the preview: ")
                           + r.stderr.strip()[:300])
    os.chmod(keluar, stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
    utc, wib = sekarang()
    data = {"file": keluar.as_posix(), "sha256": sha256_file(keluar), "dibuat_wib": wib,
            "sumber": src.as_posix(), "sha256_sumber": b["salinan"]["hash"]["sha256"],
            "proses": "ffmpeg libx264 crf 18 veryfast, yuv420p, audio AAC 128k, faststart",
            "keterangan": "Hanya untuk pemutaran di aplikasi. Bukan barang bukti; analisis memakai salinan kerja."}
    _tulis_json(keluar.parent / "pratinjau.json", data)
    catat("pratinjau_pemutaran", keluar, induk=[src], folder_kerja=d,
          info={"bukti": bb, "proses": data["proses"], "keterangan": data["keterangan"]})
    log(T(f"[{bb}] Video pratinjau selesai: {keluar.name}", f"[{bb}] Preview video done: {keluar.name}"))
    return data


def periksa_integritas(kasus_id, bb, log=print):
    d = folder_kasus(kasus_id)
    b = _baca_json(d / "bukti" / bb / "bukti.json")
    p = Path(b["salinan"]["file"])
    if not p.exists():
        return {"status": "HILANG", "file": p.as_posix()}
    h = hash_file(p)
    status = "COCOK" if h == b["salinan"]["hash"] else "BERUBAH"
    log(T(f"[{bb}] Integritas salinan: {status}", f"[{bb}] Working copy integrity: {status}"))
    return {"status": status, "file": p.as_posix(), "hash_sekarang": h, "hash_tercatat": b["salinan"]["hash"],
            "waktu": sekarang()[1]}


# ================================================================ metadata
def metadata_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fourcc = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    terbaca = cap.isOpened()
    cap.release()
    kodek_cv = "".join(chr((fourcc >> 8 * i) & 0xFF) for i in range(4)).strip("\x00 ") if fourcc else ""
    info_ff, tag = "", {}
    try:
        r = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=60)
        baris = r.stderr.splitlines()
        mulai = next((i for i, b in enumerate(baris) if b.startswith("Input #")), 0)
        info_ff = "\n".join(b for b in baris[mulai:] if not b.startswith("At least one output"))
        r = subprocess.run([ffmpeg_exe(), "-v", "error", "-i", str(path), "-f", "ffmetadata", "-"],
                           capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
        for b in r.stdout.splitlines():
            if "=" in b and not b.startswith(";"):
                k, v = b.split("=", 1)
                tag[k.strip()] = v.strip()
    except Exception as err:  # noqa: BLE001
        info_ff = f"(gagal membaca dengan FFmpeg: {err})"
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", info_ff)
    durasi = (int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])) if m else (n / fps if fps else 0)
    m = re.search(r"bitrate: (\d+) kb/s", info_ff)
    kodek = re.search(r"Video: ([^,\n]+)", info_ff)
    audio = re.search(r"Audio: ([^,\n]+)", info_ff)
    temuan = []
    semua_tag = " ".join(f"{k}={v}" for k, v in tag.items())
    if "SINTETIS" in semua_tag.upper():
        temuan.append({"tingkat": "tinggi", "teks": "Tag metadata menyatakan video SINTETIS (video fake buatan "
                                                     "pipeline dataset ini)."})
    enc = tag.get("encoder", "")
    if enc.lower().startswith("lavf"):
        temuan.append({"tingkat": "info", "teks": f"Kontainer ditulis oleh FFmpeg/libavformat ({enc}): file pernah "
                                                  "diproses ulang perangkat lunak (umum pada unduhan media sosial, "
                                                  "bukan bukti manipulasi)."})
    if not any(k.lower().endswith("creation_time") for k in tag):
        temuan.append({"tingkat": "info", "teks": "Tidak ada creation_time di kontainer: metadata waktu rekam "
                                                  "tidak tersedia atau sudah dihapus."})
    perangkat = {k: v for k, v in tag.items() if any(s in k.lower() for s in ("make", "model", "software", "location"))}
    if perangkat:
        temuan.append({"tingkat": "info", "teks": "Metadata perangkat/perangkat lunak: " +
                                                  ", ".join(f"{k}={v}" for k, v in perangkat.items())})
    if not terbaca or n <= 0:
        temuan.append({"tingkat": "tinggi", "teks": "OpenCV tidak bisa membaca frame video; file mungkin rusak "
                                                     "atau kodeknya tidak didukung."})
    return {"lebar": w, "tinggi": h, "fps": round(fps, 3), "jumlah_frame": n, "durasi_detik": round(durasi, 3),
            "kodek_video": kodek[1].strip() if kodek else kodek_cv, "audio": audio[1].strip() if audio else "",
            "bitrate_kbps": int(m[1]) if m else None, "tag": tag, "temuan": temuan, "info_ffmpeg": info_ff}


# ================================================================ analisis
class Pembaca:
    """Membaca frame berindeks naik secara berurutan (grab) tanpa menyimpan seluruh video di memori."""
    def __init__(self, path):
        import cv2
        self.cap = cv2.VideoCapture(str(path))
        self.i = 0

    def ambil(self, idxs):
        hasil = {}
        for t in sorted(set(idxs)):
            while self.i < t:
                if not self.cap.grab():
                    return hasil
                self.i += 1
            ok, fr = self.cap.read()
            self.i += 1
            if ok:
                hasil[t] = fr
            else:
                return hasil
        return hasil

    def tutup(self):
        self.cap.release()


def bagi_segmen(n, fps, p, e, indeks_frame):
    fps = fps if fps and fps > 0 else 25.0
    L = max(1, int(round(p["durasi_segmen_detik"] * fps)))
    batas = list(range(0, n, L)) + [n]
    seg = [(a, b) for a, b in zip(batas[:-1], batas[1:])]
    if len(seg) >= 2 and seg[-1][1] - seg[-1][0] < L / 2:  # sisa pendek digabung ke segmen sebelumnya
        seg[-2:] = [(seg[-2][0], seg[-1][1])]
    disampel = False
    if len(seg) > p["maks_segmen"]:
        pilih = np.linspace(0, len(seg) - 1, p["maks_segmen"]).round().astype(int)
        seg, disampel = [seg[i] for i in sorted(set(pilih))], True
    keluar = []
    for k, (a, b) in enumerate(seg, 1):
        idx = indeks_frame(a, b, fps, {**e, "frame_per_video": p["frame_per_segmen"]})
        keluar.append({"no": k, "frame_awal": a, "frame_akhir": b - 1, "mulai_detik": round(a / fps, 2),
                       "selesai_detik": round(b / fps, 2), "idx": idx})
    return keluar, disampel


def lacak(urut, det, iou_min, iou):
    """Gabungkan deteksi antarframe menjadi jejak per orang (IoU dengan kotak terakhir, lalu jarak pusat)."""
    jejak = []
    for i in urut:
        terpakai = set()
        for d in sorted(det.get(i, []), key=lambda d: -d.lebar):
            terbaik, nilai = None, 0.0
            for j, t in enumerate(jejak):
                if j in terpakai or i in t["det"]:
                    continue
                v = iou(t["kotak"], d.bbox)
                if v < iou_min:  # gerakan besar (2 fps): izinkan bila pusat dekat & ukuran mirip
                    cx, cy = (d.bbox[0] + d.bbox[2]) / 2, (d.bbox[1] + d.bbox[3]) / 2
                    tx, ty = (t["kotak"][0] + t["kotak"][2]) / 2, (t["kotak"][1] + t["kotak"][3]) / 2
                    lb = t["kotak"][2] - t["kotak"][0]
                    if ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5 < 0.6 * lb and 0.6 < d.lebar / max(lb, 1) < 1.6:
                        v = iou_min
                    else:
                        continue
                if v > nilai:
                    terbaik, nilai = j, v
            if terbaik is None:
                jejak.append({"det": {i: d}, "kotak": d.bbox})
                terpakai.add(len(jejak) - 1)
            else:
                jejak[terbaik]["det"][i] = d
                jejak[terbaik]["kotak"] = d.bbox
                terpakai.add(terbaik)
    return jejak


def _histogram(img):
    import cv2
    hsv = cv2.cvtColor(cv2.resize(img, (160, 90)), cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
    return cv2.normalize(h, h).flatten()


def ganti_adegan(urut, frames, ambang):
    """Frame (indeks) tempat adegan berganti: korelasi histogram HSV dengan frame sebelumnya < ambang."""
    import cv2
    potong, lalu = set(), None
    for i in urut:
        h = _histogram(frames[i])
        if lalu is not None and cv2.compareHist(lalu, h, cv2.HISTCMP_CORREL) < ambang:
            potong.add(i)
        lalu = h
    return potong


def pecah_jejak(jejak, frames, idn, potong_adegan, ambang_orang, log=None):
    """Potong jejak wajah bila adegan berganti atau wajahnya berubah menjadi orang lain (ArcFace), supaya satu
    urutan yang dibaca LSTM tidak mencampur dua orang / dua adegan. Kembalikan daftar jejak baru; setiap potongan
    menyimpan alasan pemotongannya di "dipisah"."""
    hasil = []
    for t in jejak:
        urut = sorted(t["det"])
        emb = {}
        if idn:
            for i in urut:
                try:
                    emb[i] = idn.emb(frames[i], t["det"][i])
                except Exception:  # noqa: BLE001
                    pass
        bagian, sekarang, acuan, alasan = [], [urut[0]], [], None
        if urut[0] in emb:
            acuan.append(emb[urut[0]])
        for i in urut[1:]:
            sebab = None
            if i in potong_adegan:
                sebab = "pergantian adegan"
            elif i in emb and acuan:
                ref = np.mean(acuan, axis=0)
                ref = ref / (np.linalg.norm(ref) + 1e-9)
                if float(np.dot(emb[i], ref)) < ambang_orang:
                    sebab = "wajah berganti orang"
            if sebab:
                bagian.append((sekarang, alasan))
                sekarang, acuan, alasan = [i], [], sebab
            else:
                sekarang.append(i)
            if i in emb:
                acuan.append(emb[i])
        bagian.append((sekarang, alasan))
        if len(bagian) > 1 and log:
            log(T(f"    jejak wajah dipotong menjadi {len(bagian)} bagian (", f"    face track split into {len(bagian)} parts (")
                + ", ".join(sorted(tr(a) for a in {a for _, a in bagian if a})) + ")")
        for idx, (bg, al) in enumerate(bagian):
            alasan_bg = sorted({a for _, a in bagian if a}) if len(bagian) > 1 else []
            hasil.append({"det": {i: t["det"][i] for i in bg}, "kotak": t["det"][bg[-1]].bbox,
                          "dipisah": alasan_bg, "bagian": f"{idx + 1}/{len(bagian)}" if len(bagian) > 1 else None})
    return hasil


class Model:
    """Model Xception + LSTM hasil langkah 6, dimuat sekali dan dipakai ulang."""
    def __init__(self, folder_model):
        import torch
        self.torch = torch
        self.folder = Path(folder_model)
        self.L = muat_modul("06_latih_xception.py", "latih_xception")
        self.G = muat_modul("07_gradcam.py", "gradcam")
        simpan = torch.load(self.folder / "terbaik.pt", map_location="cpu", weights_only=False)
        p = simpan["pengaturan"]
        self.p = p
        self.perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.L.XceptionLSTM(p["model"], p["lstm_hidden"], p["lstm_lapis"], p["lstm_dua_arah"],
                                         p["dropout"], pralatih=False).to(self.perangkat)
        self.model.load_state_dict(simpan["model"])
        self.model.eval()
        self.tf = self.L.transformasi(p["augmentasi"], False, self.model.rata, self.model.simpangan)
        metrik = _baca_json(self.folder / "metrik.json", {})
        uji = {s: {k: round(float(v), 4) for k, v in u["video@0.5"].items() if k in ("auc", "akurasi", "f1", "presisi", "recall", "n")}
               for s, u in (metrik.get("uji") or {}).items() if "video@0.5" in u}
        gab = metrik_keseluruhan(self.folder, metrik)
        if gab:
            uji = {"keseluruhan": gab, **uji}
        self.info = {"folder": self.folder.as_posix(), "nama": self.folder.name,
                     "sha256_bobot": sha256_file(self.folder / "terbaik.pt"),
                     "arsitektur": simpan.get("arsitektur", "Xception + LSTM"), "backbone": p["model"],
                     "skenario": simpan.get("skenario"), "nama_skenario": metrik.get("nama_skenario", ""),
                     "epoch_terbaik": simpan.get("epoch"), "val_auc": round(float(simpan.get("val_auc", float("nan"))), 4),
                     "ambang_val": round(float(simpan.get("ambang_val", 0.5)), 4),
                     "ambang_kandidat": metrik.get("ambang_kandidat") or {},
                     "panjang_urutan": p["panjang_urutan"], "min_frame": p["min_frame"],
                     "lstm_hidden": p["lstm_hidden"], "lstm_lapis": p["lstm_lapis"], "lstm_dua_arah": p["lstm_dua_arah"],
                     "uji": uji, "torch": torch.__version__,
                     "perangkat": torch.cuda.get_device_name(0) if self.perangkat.type == "cuda" else "CPU"}

    def _tensor(self, files):
        from PIL import Image
        T = int(self.p["panjang_urutan"])
        if len(files) > T:
            files = [files[i] for i in np.linspace(0, len(files) - 1, T).round().astype(int)]
        fr = [self.tf(Image.open(f).convert("RGB")) for f in files]
        n = len(fr)
        return self.torch.stack(fr + [fr[-1]] * (T - n)), n, files

    def skor(self, files):
        torch = self.torch
        x, n, _ = self._tensor(files)
        with torch.no_grad(), torch.autocast(device_type=self.perangkat.type, dtype=torch.float16,
                                             enabled=self.perangkat.type == "cuda"):
            logit = self.model(x[None].to(self.perangkat), torch.tensor([n]))
        return float(torch.sigmoid(logit.float()).item())

    def gradcam(self, files):
        x, n, files = self._tensor(files)
        skor, cam = self.G.gradcam_video(self.model, x, n, self.perangkat)
        return skor, cam, files


def _pengaturan_ekstraksi(cfg):
    L5 = muat_modul("05_ekstrak_wajah.py", "ekstrak_wajah")
    e = dict(L5.BAKU)
    e.update(cfg.get("ekstraksi") or {})
    return L5, e


class Penganalisis:
    """Detektor wajah + model, dimuat sekali per aplikasi."""
    def __init__(self, log=print):
        self.cfg = muat_config()
        self.L5, self.e = _pengaturan_ekstraksi(self.cfg)
        self._pend, self._idn, self._model = None, None, {}
        self.log = log

    @property
    def pend(self):
        if self._pend is None:
            self.log(T("Memuat detektor wajah MTCNN...", "Loading MTCNN face detector..."))
            self._pend = self.L5.Pendeteksi(self.e)
        return self._pend

    def idn(self):
        if self._idn is None:
            try:
                self._idn = self.L5.Identitas()
            except Exception as err:  # noqa: BLE001
                self.log(T(f"! ArcFace tidak tersedia ({err}); orang antarsegmen dibedakan dari ukuran wajah saja",
                           f"! ArcFace not available ({err}); people across segments are told apart by face size only"))
                self._idn = False
        return self._idn

    def model(self, folder_model):
        k = Path(folder_model).as_posix()
        if k not in self._model:
            self.log(T(f"Memuat model {Path(folder_model).name}...", f"Loading model {Path(folder_model).name}..."))
            self._model[k] = Model(folder_model)
        return self._model[k]


NAMA_AMBANG = {"fpr5": "batas toleransi 5%", "fpr10": "batas toleransi 10%", "val": "Youden",
               "youden": "Youden", "f1": "F1 terbaik di val"}
# batas toleransi x% = ambang terendah yang membuat paling banyak x% video asli di data validasi melewatinya
TOLERANSI = {"fpr5": 5, "fpr10": 10}


def pilih_ambang(info, pilihan, log=print, bb=""):
    """Ambang keputusan: dari kandidat yang dihitung di data val saat training/evaluasi, atau angka tetap."""
    kand = info.get("ambang_kandidat") or {}
    pilihan = str(pilihan)
    if pilihan in ("val", "youden"):
        return float(kand.get("youden", info["ambang_val"])), "ambang Youden dari data val"
    if pilihan in kand:
        if pilihan in TOLERANSI:
            return float(kand[pilihan]), (f"batas toleransi {TOLERANSI[pilihan]}%: paling banyak {TOLERANSI[pilihan]}% "
                                          "video asli di data validasi melewati batas ini")
        return float(kand[pilihan]), f"ambang {NAMA_AMBANG.get(pilihan, pilihan)} dari data val"
    try:
        return float(pilihan), "ambang tetap"
    except ValueError:
        log(T(f"[{bb}] ! model belum punya ambang '{pilihan}' (evaluasi ulang model untuk menghitungnya); memakai 0,5",
              f"[{bb}] ! the model has no '{pilihan}' threshold yet (re-evaluate the model to compute it); using 0.5"))
        return 0.5, f"ambang tetap 0,5 (ambang '{pilihan}' belum tersedia untuk model ini)"


def _kesimpulan(rata, maks, ambang, zona, rendah=None):
    """Dua ambang (rendah tidak None): rata-rata >= batas toleransi -> terindikasi; rata-rata >= ambang Youden atau ada
    segmen >= batas toleransi -> perlu pemeriksaan lanjutan; selain itu tidak terindikasi.
    Satu ambang (rendah None): aturan lama dengan zona ragu di sekitar ambang."""
    if rata >= ambang:
        return "TERINDIKASI"
    if rendah is not None:
        return "RAGU" if (rata >= rendah or maks >= ambang) else "TIDAK"
    # zona ragu hanya di BAWAH ambang dan paling lebar separuh ambang: dengan ambang kecil (mis. Youden 0,02),
    # zona tetap 0,1 membuat skor ~0 ikut "perlu pemeriksaan lanjutan"
    if maks >= ambang or (ambang - rata) < min(zona, ambang / 2):
        return "RAGU"
    return "TIDAK"


def _gambar_kunci(img, wajah_per_orang, keluar, judul):
    import cv2
    im = img.copy()
    warna = [(80, 200, 255), (120, 255, 120), (255, 160, 80), (200, 120, 255), (255, 255, 120)]
    for orang, d in wajah_per_orang:
        x1, y1, x2, y2 = [int(v) for v in d.bbox]
        c = warna[(int(orang[1:]) - 1) % len(warna)]
        t = max(2, im.shape[1] // 400)
        cv2.rectangle(im, (x1, y1), (x2, y2), c, t)
        cv2.putText(im, orang, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, max(0.7, im.shape[1] / 1200), c, t)
    s = 640 / max(im.shape[1], 1)
    if s < 1:
        im = cv2.resize(im, (640, int(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
    cv2.putText(im, judul, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    keluar.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(keluar), im, [cv2.IMWRITE_JPEG_QUALITY, 88])


def waktu_tepat(detik):
    """mm:ss.ss (atau hh:mm:ss.ss) untuk menandai frame."""
    detik = float(detik or 0)
    j, sisa = divmod(detik, 3600)
    mnt, s = divmod(sisa, 60)
    return (f"{int(j):02d}:" if j else "") + f"{int(mnt):02d}:{s:05.2f}"


def bobot_frame(cam):
    """Bobot tiap frame dari peta Grad-CAM (dinormalisasi bersama satu segmen): rata-rata intensitas frame dibagi
    rata-rata frame tertinggi (1 = frame paling berpengaruh pada skor segmen), dan kontribusi % dari total."""
    rata = np.array([float(np.mean(c)) for c in cam])
    maks, total = float(rata.max()) if len(rata) else 0.0, float(rata.sum())
    return ([float(r / maks) if maks > 0 else 0.0 for r in rata],
            [float(100 * r / total) if total > 0 else 0.0 for r in rata], [float(np.max(c)) for c in cam])


def _gambar_gradcam(G, files, frames, cam, keluar, judul, sub, maks_kolom=10, sel=150, detik=None, bobot=None,
                    lewat=True):
    from PIL import Image, ImageDraw
    idx = np.linspace(0, len(files) - 1, min(maks_kolom, len(files))).round().astype(int)
    lebar, tj = 12 + len(idx) * (sel + 4), 54
    kanvas = Image.new("RGB", (lebar, tj + 2 * (sel + 4) + (40 if detik is not None else 24)), "white")
    gb = ImageDraw.Draw(kanvas)
    gb.text((8, 6), judul, font=G.huruf(16), fill=(20, 20, 20))
    gb.text((8, 30), sub, font=G.huruf(12), fill=(80, 80, 80))
    for j, k in enumerate(idx):
        asli, panas = G.tumpuk(files[k], cam[k], sel)
        x = 6 + j * (sel + 4)
        kanvas.paste(Image.fromarray(asli), (x, tj))
        kanvas.paste(Image.fromarray(panas), (x, tj + sel + 4))
        if detik is not None:  # waktu tepat di video + bobot, sama dengan yang tampil di garis waktu aplikasi
            gb.text((x + 2, tj + 2 * (sel + 4) + 2), f"{waktu_tepat(detik[k])} (frame {frames[k]})", font=G.huruf(12),
                    fill=(40, 40, 40))
            gb.text((x + 2, tj + 2 * (sel + 4) + 18), T("bobot relatif", "relative weight") + f" {bobot[k]:.2f}", font=G.huruf(12),
                    fill=(200, 40, 40) if bobot[k] >= 0.999 and lewat else (90, 90, 90))
        else:
            gb.text((x + 2, tj + 2 * (sel + 4) + 4), f"frame {frames[k]}", font=G.huruf(12), fill=(90, 90, 90))
    keluar.parent.mkdir(parents=True, exist_ok=True)
    kanvas.save(keluar)


def skala_terpadu(skor, yakin, peka):
    """Petakan skor satu model ke skala bersama berdasarkan ambang model itu sendiri, supaya model dengan sebaran
    skor berbeda bisa digabung: dua ambang -> [0, peka) = 0-0,33, [peka, yakin) = 0,33-0,67, >= yakin = 0,67-1;
    satu ambang -> [0, ambang) = 0-0,5, >= ambang = 0,5-1. Urutan skor di dalam satu model tidak berubah."""
    skor = float(skor)
    if peka is not None:
        if skor < peka:
            return 0.33 * skor / max(peka, 1e-9)
        if skor < yakin:
            return 0.33 + 0.34 * (skor - peka) / max(yakin - peka, 1e-9)
        return 0.67 + 0.33 * (skor - yakin) / max(1 - yakin, 1e-9)
    if skor < yakin:
        return 0.5 * skor / max(yakin, 1e-9)
    return 0.5 + 0.5 * (skor - yakin) / max(1 - yakin, 1e-9)


def analisis_bukti(kasus_id, bb, folder_model, penganalisis, opsi=None, log=print, kemajuan=None):
    """folder_model: satu folder model, atau daftar folder untuk analisis GABUNGAN (ensemble): wajah dideteksi dan
    dilacak sekali, lalu setiap model menilai crop dengan profilnya sendiri; skor tiap model dipetakan ke skala
    terpadu (berdasarkan ambang masing-masing) lalu digabung dengan skor tertinggi ('maks') atau rata-rata."""
    import cv2
    t0 = time.time()
    d = folder_kasus(kasus_id)
    fb = d / "bukti" / bb
    b = _baca_json(fb / "bukti.json")
    p = pengaturan(penganalisis.cfg)
    p.update({k: v for k, v in (opsi or {}).items() if v is not None and k in BAKU})
    e, L5 = dict(penganalisis.e), penganalisis.L5
    kemajuan = kemajuan or (lambda f, t="": None)
    folders = list(folder_model) if isinstance(folder_model, (list, tuple)) else [folder_model]
    gabung = (opsi or {}).get("gabungan", "maks") if len(folders) > 1 else None

    # 1. integritas salinan
    salinan = Path(b["salinan"]["file"])
    log(T(f"[{bb}] Verifikasi integritas salinan sebelum analisis...",
          f"[{bb}] Verifying working copy integrity before analysis..."))
    integ = periksa_integritas(kasus_id, bb, log=lambda m: None)
    if integ["status"] != "COCOK":
        raise RuntimeError(T(f"Integritas salinan {integ['status']}: analisis dihentikan. Lakukan akuisisi ulang.",
                             f"Working copy integrity {integ['status']}: analysis stopped. Acquire the evidence again."))
    _isi_cache_hash(salinan, integ["hash_sekarang"]["sha256"])
    log(T(f"[{bb}] Integritas COCOK (SHA-256 {integ['hash_sekarang']['sha256']})",
          f"[{bb}] Integrity MATCH (SHA-256 {integ['hash_sekarang']['sha256']})"))

    # 2. model, ambang, profil crop
    anggota = []
    for f in folders:
        m = penganalisis.model(f)
        yakin, jenis_y = pilih_ambang(m.info, p["ambang"], log, bb)
        peka, jenis_p = None, None
        if str(p.get("ambang_rendah") or "tidak") not in ("tidak", "none", ""):
            peka, jenis_p = pilih_ambang(m.info, p["ambang_rendah"], log, bb)
            if peka >= yakin:
                log(T(f"[{bb}] ! {m.info['nama']}: ambang Youden ({peka:.3f}) tidak lebih kecil dari batas toleransi "
                      f"({yakin:.3f}); memakai satu ambang",
                      f"[{bb}] ! {m.info['nama']}: Youden threshold ({peka:.3f}) is not below the tolerance limit "
                      f"({yakin:.3f}); using a single threshold"))
                peka, jenis_p = None, None
        at = aturan_frame_model(m.folder)
        m.info["aturan_frame_latih"] = at["mode_frame"]
        anggota.append({"m": m, "nama": m.info["nama"], "yakin": yakin, "peka": peka, "jenis_yakin": jenis_y,
                        "jenis_peka": jenis_p, "aturan": at, "profil": (float(at["margin"]), int(at["ukuran"]))})
    m0 = anggota[0]["m"]
    # aturan pengambilan frame (sama untuk semua model; pada gabungan diambil dari model pertama)
    aturan_model = anggota[0]["aturan"]
    pilihan = p.get("aturan_frame", "otomatis")
    if pilihan in ("merata", "berurutan"):
        e["mode_frame"] = pilihan
        if pilihan == "berurutan":
            e["fps_sampel"] = aturan_model.get("fps_sampel") or e.get("fps_sampel", 25)
    else:
        e["mode_frame"], e["fps_sampel"] = aturan_model["mode_frame"], aturan_model.get("fps_sampel", 25)
    for a_ in anggota:
        a_["sesuai"] = a_["aturan"]["mode_frame"] == e["mode_frame"]
        if not a_["sesuai"]:
            log(T(f"[{bb}] ! {a_['nama']} dilatih dengan {ATURAN_FRAME[a_['aturan']['mode_frame']]}, berbeda dengan "
                  f"aturan yang dipakai ({ATURAN_FRAME[e['mode_frame']]}): skornya untuk perbandingan saja",
                  f"[{bb}] ! {a_['nama']} was trained with {tr(ATURAN_FRAME[a_['aturan']['mode_frame']])}, unlike the "
                  f"rule in use ({tr(ATURAN_FRAME[e['mode_frame']])}): its score is for comparison only"))
    sesuai = all(a_["sesuai"] for a_ in anggota)
    e["margin"], e["ukuran"] = anggota[0]["profil"]
    info_aturan = {"dipilih": pilihan, "dipakai": e["mode_frame"], "fps_sampel": e.get("fps_sampel"),
                   "aturan_model": aturan_model["mode_frame"], "sumber_aturan_model": aturan_model["sumber"],
                   "sesuai_model": sesuai}
    profil_list = sorted({a_["profil"] for a_ in anggota}, key=lambda x: (x != anggota[0]["profil"], x))
    log(T(f"[{bb}] Crop wajah: ", f"[{bb}] Face crop: ") + ", ".join(f"margin {mg}, {uk}x{uk}" for mg, uk in profil_list)
        + T(" (sesuai model)", " (as trained)"))
    log(T(f"[{bb}] Aturan frame: {ATURAN_FRAME[e['mode_frame']]}", f"[{bb}] Frame rule: {tr(ATURAN_FRAME[e['mode_frame']])}")
        + (T(" (otomatis, sesuai model)", " (automatic, as trained)") if pilihan == "otomatis" else ""))
    min_frame = max(int(a_["m"].p["min_frame"]) for a_ in anggota)
    for a_ in anggota:
        log(f"[{bb}] Model {a_['nama']} ({a_['m'].info['arsitektur']}, " + T("skenario", "scenario")
            + f" {a_['m'].info['skenario']}, crop m{a_['profil'][0]}/{a_['profil'][1]}), " + (
                T(f"batas toleransi {a_['yakin']:.3f}, ambang Youden {a_['peka']:.3f}",
                  f"tolerance limit {a_['yakin']:.3f}, Youden threshold {a_['peka']:.3f}") if a_["peka"] is not None
                else T(f"ambang {a_['yakin']:.3f}", f"threshold {a_['yakin']:.3f}"))
            + T(f", perangkat {a_['m'].info['perangkat']}", f", device {a_['m'].info['perangkat']}"))
    if gabung:
        dua = all(a_["peka"] is not None for a_ in anggota)
        ambang, rendah = (0.67, 0.33) if dua else (0.5, None)
        jenis_ambang = (f"zona terpadu gabungan {len(anggota)} model ({'skor tertinggi' if gabung == 'maks' else 'rata-rata'}): "
                        "skor tiap model dipetakan menurut ambang masing-masing")
        jenis_rendah = "zona terpadu: ambang Youden masing-masing model" if dua else None
        log(T(f"[{bb}] ANALISIS GABUNGAN {len(anggota)} model, digabung dengan "
              f"{'skor tertinggi' if gabung == 'maks' else 'rata-rata skor'} pada skala terpadu "
              f"(batas toleransi {ambang}, ambang Youden {rendah})",
              f"[{bb}] ENSEMBLE ANALYSIS of {len(anggota)} models, combined by "
              f"{'highest score' if gabung == 'maks' else 'mean score'} on a unified scale "
              f"(tolerance limit {ambang}, Youden threshold {rendah})"))
    else:
        ambang, rendah = anggota[0]["yakin"], anggota[0]["peka"]
        jenis_ambang, jenis_rendah = anggota[0]["jenis_yakin"], anggota[0]["jenis_peka"]

    # 3. segmen
    meta = b["metadata"]
    n, fps = int(meta.get("jumlah_frame") or 0), float(meta.get("fps") or 0)
    if n <= 0:
        raise RuntimeError(T("Jumlah frame video tidak terbaca", "The video frame count cannot be read"))
    segmen, disampel = bagi_segmen(n, fps, p, e, L5.indeks_frame)
    fps_ef = fps if fps > 0 else 25.0
    waktu = datetime.now().strftime("%Y%m%d_%H%M%S")
    fa = fb / "analisis" / waktu
    (fa / "wajah").mkdir(parents=True, exist_ok=True)
    log(T(f"[{bb}] {len(segmen)} segmen x {p['frame_per_segmen']} frame (segmen {p['durasi_segmen_detik']} dtk"
          f"{', disampel merata' if disampel else ''})",
          f"[{bb}] {len(segmen)} segments x {p['frame_per_segmen']} frames ({p['durasi_segmen_detik']} s segments"
          f"{', sampled evenly' if disampel else ''})"))

    pend = penganalisis.pend
    idn = penganalisis.idn()
    galeri = []  # [(orang, embedding)]
    baris, frame_kunci = [], {}
    pembaca = Pembaca(salinan)
    try:
        for si, sg in enumerate(segmen):
            kemajuan(0.05 + 0.8 * si / len(segmen), T("segmen", "segment") + f" {sg['no']}/{len(segmen)}")
            frames = pembaca.ambil(sg["idx"])
            urut = [i for i in sg["idx"] if i in frames]
            if not urut:
                continue
            det = dict(zip(urut, pend.deteksi([frames[i] for i in urut])))
            jejak = lacak(urut, det, p["iou_lacak"], L5.iou)
            if p.get("pisah_jejak", True):
                potong = ganti_adegan(urut, frames, float(p.get("ambang_ganti_adegan", 0.5)))
                jejak = pecah_jejak([j for j in jejak if len(j["det"]) >= 2], frames, idn, potong,
                                    float(p.get("ambang_ganti_orang", 0.35)), log)
            jejak = [j for j in jejak if len(j["det"]) >= min_frame]
            jejak = sorted(jejak, key=lambda j: -np.mean([x.lebar for x in j["det"].values()]))[:int(p["maks_orang"])]
            # identitas antarsegmen dulu (ArcFace pada wajah terbesar di jejak), lalu satu jejak per orang per segmen:
            # bila dua potongan jatuh ke orang yang sama, yang lebih panjang dipakai
            dipakai = {}
            for k, j in enumerate(jejak):
                orang = None
                if idn:
                    i_ref = max(j["det"], key=lambda i: j["det"][i].lebar)
                    try:
                        v = idn.emb(frames[i_ref], j["det"][i_ref])
                        sim = [(float(np.dot(v, g)), o) for o, g in galeri]
                        s_best, o_best = max(sim, default=(-1, None))
                        if s_best >= p["ambang_identitas"]:
                            orang = o_best
                        else:
                            orang = f"P{len(galeri) + 1}"
                            galeri.append((orang, v))
                    except Exception as err:  # noqa: BLE001
                        log(T(f"  ! ArcFace gagal ({err})", f"  ! ArcFace failed ({err})"))
                if orang is None:
                    orang = f"P{k + 1}"
                j["orang"] = orang
                if orang not in dipakai or len(j["det"]) > len(dipakai[orang]["det"]):
                    if orang in dipakai:
                        log(T(f"    {orang}: potongan lebih pendek ({len(dipakai[orang]['det'])} frame) tidak dipakai",
                              f"    {orang}: shorter part ({len(dipakai[orang]['det'])} frames) not used"))
                    dipakai[orang] = j
                else:
                    log(T(f"    {orang}: potongan lebih pendek ({len(j['det'])} frame) tidak dipakai",
                          f"    {orang}: shorter part ({len(j['det'])} frames) not used"))
            jejak = [j for j in jejak if dipakai.get(j["orang"]) is j]
            sg["jumlah_wajah_maks"] = max((len(det[i]) for i in urut), default=0)
            kunci = []
            for k, j in enumerate(jejak):
                orang = j["orang"]
                # crop untuk setiap profil (margin, ukuran) yang dibutuhkan model
                per_profil = {}
                for pi, (mg, uk) in enumerate(profil_list):
                    sub = f"S{sg['no']:03d}_{orang}" + ("" if pi == 0 else f"/m{int(round(mg * 10))}_{uk}")
                    folder_w = fa / "wajah" / sub
                    folder_w.mkdir(parents=True, exist_ok=True)
                    files, fr_idx, lebar, hitam = [], [], [], 0
                    for i in sorted(j["det"]):
                        dd = j["det"][i]
                        g = L5.crop(frames[i], dd, mg, uk)
                        if e.get("maks_porsi_hitam") and float(np.mean(g.max(axis=2) <= 16)) > e["maks_porsi_hitam"]:
                            hitam += 1
                            continue
                        f = folder_w / f"{i:06d}.png"
                        cv2.imwrite(str(f), g)
                        files.append(f.as_posix())
                        fr_idx.append(i)
                        lebar.append(dd.lebar)
                    per_profil[(mg, uk)] = {"file": files, "frame": fr_idx, "lebar": lebar, "hitam": hitam}
                utama = per_profil[profil_list[0]]
                r = {"segmen": sg["no"], "mulai_detik": sg["mulai_detik"], "selesai_detik": sg["selesai_detik"],
                     "orang": orang, "jumlah_wajah": len(utama["file"]), "dilewati_tepi_hitam": utama["hitam"],
                     "lebar_wajah_rata": round(float(np.mean(utama["lebar"])), 1) if utama["lebar"] else 0,
                     "prob_deteksi_rata": round(float(np.mean([j["det"][i].prob for i in utama["frame"]])), 4) if utama["frame"] else 0,
                     "frame": utama["frame"], "file": utama["file"], "skor": None,
                     "dipisah": j.get("dipisah") or []}
                skor_model, terpadu = {}, {}
                for a_ in anggota:
                    pp = per_profil[a_["profil"]]
                    if len(pp["file"]) >= int(a_["m"].p["min_frame"]):
                        sk_ = round(a_["m"].skor(pp["file"]), 4)
                        skor_model[a_["nama"]] = sk_
                        terpadu[a_["nama"]] = round(skala_terpadu(sk_, a_["yakin"], a_["peka"]), 4)
                if gabung:
                    r["skor_model"], r["skor_terpadu_model"] = skor_model, terpadu
                    if terpadu:
                        r["skor"] = round(max(terpadu.values()) if gabung == "maks" else float(np.mean(list(terpadu.values()))), 4)
                        r["model_penentu"] = max(terpadu, key=terpadu.get)
                    r["file_per_model"] = {a_["nama"]: per_profil[a_["profil"]]["file"] for a_ in anggota}
                    r["frame_per_model"] = {a_["nama"]: per_profil[a_["profil"]]["frame"] for a_ in anggota}
                elif skor_model:
                    r["skor"] = skor_model[anggota[0]["nama"]]
                baris.append(r)
                mid = sorted(j["det"])[len(j["det"]) // 2]
                kunci.append((orang, j["det"][mid], mid))
            if kunci:
                i_mid = kunci[0][2]
                out = fa / "kunci" / f"S{sg['no']:03d}.jpg"
                _gambar_kunci(frames[i_mid], [(o, dd) for o, dd, _ in kunci], out,
                              T("Segmen", "Segment") + f" {sg['no']} - {_jam(i_mid / fps_ef)}")
                frame_kunci[sg["no"]] = out.as_posix()
            ringkas = ", ".join(
                (f"{r['orang']}={r['skor']:.3f}" + (" [" + ", ".join(f"{k_}={v_:.3f}" for k_, v_ in r["skor_model"].items()) + "]"
                                                    if gabung else "")) if r["skor"] is not None else f"{r['orang']}=-"
                for r in baris if r["segmen"] == sg["no"])
            log(T("  segmen", "  segment") + f" {sg['no']:>3} {_jam(sg['mulai_detik'])}-{_jam(sg['selesai_detik'])}: "
                + (ringkas or T("tidak ada wajah cukup", "not enough faces")))
    finally:
        pembaca.tutup()

    # 4. ringkasan per orang
    orang_ = {}
    for r in baris:
        if r["skor"] is not None:
            orang_.setdefault(r["orang"], []).append(r)
    ringkasan = []
    for o, rs in sorted(orang_.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        sk = np.array([r["skor"] for r in rs])
        st = _kesimpulan(float(sk.mean()), float(sk.max()), ambang, p["zona_ragu"], rendah)
        ro = {"orang": o, "segmen_dianalisis": len(rs), "skor_rata": round(float(sk.mean()), 4),
              "skor_median": round(float(np.median(sk)), 4), "skor_maks": round(float(sk.max()), 4),
              "segmen_melewati_ambang": int((sk >= ambang).sum()),
              "porsi_melewati_ambang": round(float((sk >= ambang).mean()), 3),
              "segmen_melewati_rendah": int((sk >= rendah).sum()) if rendah is not None else None,
              "segmen_tertinggi": int(rs[int(sk.argmax())]["segmen"]),
              "status": st, "kesimpulan": KESIMPULAN[st]}
        if gabung:
            ro["skor_rata_model"] = {a_["nama"]: round(float(np.mean([r["skor_model"][a_["nama"]] for r in rs
                                                                       if a_["nama"] in r.get("skor_model", {})])), 4)
                                     for a_ in anggota if any(a_["nama"] in r.get("skor_model", {}) for r in rs)}
        ringkasan.append(ro)
    status = max((r["status"] for r in ringkasan), key=lambda s: _PERINGKAT[s], default="TIDAK_ADA_WAJAH")

    # 5. Grad-CAM, sinkron dengan garis waktu. Cakupan (forensik.gradcam_cakupan):
    #    lewat_ambang = setiap segmen yang melewati ambang + segmen tertinggi tiap orang (pembanding bila tidak ada
    #    yang lewat); tertinggi = segmen tertinggi tiap orang lalu skor tertinggi lain; semua = setiap segmen.
    #    Dibatasi gradcam_maks gambar. Pada gabungan memakai model yang paling menentukan skor segmen tersebut.
    kemajuan(0.88, "Grad-CAM")
    ada_skor = [r for r in baris if r["skor"] is not None]
    cakupan = str(p.get("gradcam_cakupan") or "lewat_ambang")
    batas_g = rendah if rendah is not None else ambang  # segmen >= Youden sudah masuk zona pemeriksaan
    maks_g = int(p["gradcam_maks"])
    tertinggi = [next(r for r in ada_skor if r["orang"] == o["orang"] and r["segmen"] == o["segmen_tertinggi"])
                 for o in ringkasan]
    id_t = {id(r) for r in tertinggi}
    if cakupan == "semua":
        lain = [r for r in ada_skor if id(r) not in id_t]
    elif cakupan == "tertinggi":
        lain = sorted([r for r in ada_skor if id(r) not in id_t], key=lambda r: -r["skor"])[:max(0, maks_g - len(tertinggi))]
    else:
        lain = [r for r in ada_skor if id(r) not in id_t and r["skor"] >= batas_g]
    if len(tertinggi) + len(lain) > maks_g:
        log(T(f"  ! {len(tertinggi) + len(lain)} segmen memenuhi syarat Grad-CAM; dibatasi {maks_g} (skor tertinggi). "
              "Naikkan forensik.gradcam_maks di config.yaml bila perlu.",
              f"  ! {len(tertinggi) + len(lain)} segments qualify for Grad-CAM; limited to {maks_g} (highest scores). "
              "Raise forensik.gradcam_maks in config.yaml if needed."))
        lain = sorted(lain, key=lambda r: -r["skor"])[:max(0, maks_g - len(tertinggi))]
    pilih = sorted(tertinggi + lain, key=lambda r: (r["mulai_detik"], r["orang"]))
    log(T(f"  Grad-CAM untuk {len(pilih)} segmen ({cakupan})", f"  Grad-CAM for {len(pilih)} segments ({cakupan})"))
    gradcam = []
    per_nama = {a_["nama"]: a_ for a_ in anggota}
    for r in pilih:
        try:
            a_ = per_nama[r.get("model_penentu") or anggota[0]["nama"]]
            f_in = r["file_per_model"][a_["nama"]] if gabung else r["file"]
            fr_in = r["frame_per_model"][a_["nama"]] if gabung else r["frame"]
            skor_g, cam, files = a_["m"].gradcam(f_in)
            frs = [fr_in[f_in.index(f)] for f in files]
            out = fa / "gradcam" / f"S{r['segmen']:03d}_{r['orang']}.png"
            # per frame: waktu tepat di video, bobot Grad-CAM, dan gambar overlay sendiri (untuk pemutar interaktif)
            dtk = [round(int(fr_) / fps_ef, 3) for fr_ in frs]
            bobot, kontrib, puncak = bobot_frame(cam)
            from PIL import Image as _Img
            folder_f = fa / "gradcam" / f"S{r['segmen']:03d}_{r['orang']}"
            folder_f.mkdir(parents=True, exist_ok=True)
            detail = []
            for k_, (fw_, fr_) in enumerate(zip(files, frs)):
                _, panas_ = a_["m"].G.tumpuk(fw_, cam[k_], 224)
                f_cam = folder_f / f"{int(fr_):06d}.jpg"
                _Img.fromarray(panas_).save(f_cam, quality=90)
                detail.append({"frame": int(fr_), "detik": dtk[k_], "waktu": waktu_tepat(dtk[k_]),
                               "bobot": round(bobot[k_], 4), "kontribusi_persen": round(kontrib[k_], 2),
                               "puncak": round(puncak[k_], 4), "file_wajah": Path(fw_).as_posix(),
                               "file_cam": f_cam.as_posix()})
            _gambar_gradcam(a_["m"].G, files, frs, cam, out,
                            f"{bb} · " + T("segmen", "segment") + f" S{r['segmen']} ({_jam(r['mulai_detik'])}-{_jam(r['selesai_detik'])}) · {r['orang']}"
                            + (f" · model {a_['nama']}" if gabung else ""),
                            T(f"Skor segmen {r['skor']:.3f}, ", f"Segment score {r['skor']:.3f}, ")
                            + (T(f"ambang Youden {rendah:.3f}, batas toleransi {ambang:.3f}",
                                 f"Youden threshold {rendah:.3f}, tolerance limit {ambang:.3f}")
                               if rendah is not None else T(f"ambang {ambang:.3f}", f"threshold {ambang:.3f}"))
                            + T(f" (sama dengan titik S{r['segmen']} di garis waktu). Skor fake model {skor_g:.3f}. "
                                "Baris atas: wajah. Baris bawah: Grad-CAM, merah = paling memengaruhi skor.",
                                f" (same as point S{r['segmen']} on the timeline). Model fake score {skor_g:.3f}. "
                                "Top row: faces. Bottom row: Grad-CAM, red = most influence on the score.")
                            + ("" if r["skor"] >= batas_g else T(
                                " Segmen di bawah ambang: bobot relatif hanya menunjukkan perhatian model, bukan tanda "
                                "manipulasi.", " Segment below the threshold: relative weight only shows the model's "
                                "attention, not a sign of manipulation.")),
                            detik=dtk, bobot=bobot, lewat=r["skor"] >= batas_g)
            gradcam.append({"frame_detail": detail, "segmen": r["segmen"], "orang": r["orang"], "skor": round(skor_g, 4), "file": out.as_posix(),
                            "model": a_["nama"], "mulai_detik": r["mulai_detik"], "selesai_detik": r["selesai_detik"],
                            "skor_segmen": round(float(r["skor"]), 4), "melewati_ambang": bool(r["skor"] >= batas_g),
                            "alasan": ("skor tertinggi orang ini" if id(r) in id_t else
                                       "melewati batas toleransi" if r["skor"] >= ambang else
                                       "di zona pemeriksaan" if r["skor"] >= batas_g else "skor tinggi")})
            log(T("  Grad-CAM segmen", "  Grad-CAM segment") + f" {r['segmen']} {r['orang']}" + (f" ({a_['nama']})" if gabung else "") + f": {out.name}")
        except Exception as err:  # noqa: BLE001
            log(T(f"  ! Grad-CAM segmen {r['segmen']} {r['orang']} gagal: {err}",
                  f"  ! Grad-CAM segment {r['segmen']} {r['orang']} failed: {err}"))

    # 6. simpan hasil + manifest + log
    kemajuan(0.95, T("menyimpan hasil", "saving results"))
    utc, wib = sekarang()
    if gabung:
        info_model = dict(m0.info)
        info_model.update({"nama": "GABUNGAN " + " + ".join(a_["nama"] for a_ in anggota),
                           "arsitektur": f"Gabungan {len(anggota)} model Xception + LSTM "
                                         f"({'skor tertinggi' if gabung == 'maks' else 'rata-rata'}, skala terpadu)",
                           "sha256_bobot": ", ".join(a_["m"].info["sha256_bobot"] for a_ in anggota),
                           "anggota": [{**{k: a_["m"].info.get(k) for k in ("nama", "folder", "skenario", "nama_skenario",
                                                                              "val_auc", "sha256_bobot", "uji")},
                                        "crop": f"m{a_['profil'][0]}/{a_['profil'][1]}", "ambang_yakin": a_["yakin"],
                                        "ambang_peka": a_["peka"], "jenis_ambang_yakin": a_["jenis_yakin"],
                                        "jenis_ambang_peka": a_["jenis_peka"]} for a_ in anggota],
                           "cara_gabung": gabung})
    else:
        info_model = m0.info
    hasil = {"id": waktu, "bukti": bb, "kasus": kasus_id, "file_bukti": salinan.as_posix(),
             "sha256_bukti": integ["hash_sekarang"]["sha256"], "waktu_wib": wib, "waktu_utc": utc,
             "durasi_proses_detik": round(time.time() - t0, 1), "operator": getpass.getuser(),
             "komputer": platform.node(), "model": info_model, "ambang": round(ambang, 4),
             "jenis_ambang": jenis_ambang,
             "ambang_rendah": round(rendah, 4) if rendah is not None else None, "jenis_ambang_rendah": jenis_rendah,
             "skema_ambang": "dua" if rendah is not None else "satu", "gabungan": gabung,
             "pengaturan": p, "aturan_frame": info_aturan, "ekstraksi": {k: e[k] for k in ("frame_per_video", "mode_frame", "fps_sampel", "ukuran", "margin", "ukuran_wajah_min",
                                                              "prob_min", "ambang_mtcnn", "maks_porsi_hitam") if k in e},
             "segmen_disampel": disampel, "jumlah_segmen": len(segmen),
             "segmen": [{k: v for k, v in s.items() if k != "idx"} | {"frame_kunci": frame_kunci.get(s["no"])}
                        for s in segmen],
             "hasil": baris, "ringkasan_orang": ringkasan, "status": status, "kesimpulan": KESIMPULAN[status],
             "gradcam": gradcam, "gradcam_cakupan": cakupan}
    _tulis_json(fa / "hasil.json", hasil)
    with open(fa / "manifest.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "sha256", "ukuran_byte"])
        for q in sorted(fa.rglob("*")):
            if q.is_file() and q.name != "manifest.csv":
                w.writerow([q.as_posix(), sha256_file(q), q.stat().st_size])
    info = {"bukti": bb, "model": info_model["nama"], "sha256_bobot_model": info_model["sha256_bobot"],
            "ambang": ambang, "ambang_rendah": rendah, "gabungan": gabung,
            "status": status, "kesimpulan": KESIMPULAN[status],
            "ringkasan_orang": [{k: o[k] for k in ("orang", "skor_rata", "skor_maks", "segmen_dianalisis", "status")}
                                for o in ringkasan]}
    catat("analisis_deepfake", fa / "hasil.json", induk=[salinan] + [a_["m"].folder / "terbaik.pt" for a_ in anggota],
          folder_kerja=d, info=info)
    catat("manifest_analisis", fa / "manifest.csv", induk=[fa / "hasil.json"], folder_kerja=d, info={"bukti": bb})
    log(T(f"[{bb}] Selesai dalam {time.time() - t0:.0f} dtk. Kesimpulan: ",
          f"[{bb}] Finished in {time.time() - t0:.0f} s. Conclusion: ") + kes(status).upper())
    kemajuan(1.0, T("selesai", "done"))
    return hasil


def _jam(detik):
    detik = float(detik or 0)
    j, s = divmod(int(detik), 3600)
    mnt, s = divmod(s, 60)
    return f"{j:02d}:{mnt:02d}:{s:02d}" if j else f"{mnt:02d}:{s:02d}"


# ================================================================ model
def metrik_keseluruhan(folder_model, mj):
    """Metrik test seluruh dataset (Indonesia + Celeb-DF sebagai satu kesatuan) pada ambang 0,5: dari metrik.json
    (model baru) atau dihitung dari prediksi_<sumber>.csv (model lama)."""
    if mj.get("uji_gabungan"):
        m = mj["uji_gabungan"]["video@0.5"]
        return {k: round(float(m[k]), 4) for k in ("auc", "akurasi", "f1", "presisi", "recall", "n")}
    file = sorted(Path(folder_model).glob("prediksi_*.csv"))
    if len(file) < 2:
        return None
    y, s = [], []
    for f in file:
        with open(f, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                y.append(int(r["label"]))
                s.append(float(r["skor"]))
    p = [int(v >= 0.5) for v in s]
    tp = sum(a == 1 and b == 1 for a, b in zip(y, p))
    fp = sum(a == 0 and b == 1 for a, b in zip(y, p))
    fn = sum(a == 1 and b == 0 for a, b in zip(y, p))
    neg = sorted(v for a, v in zip(y, s) if a == 0)
    pos = [v for a, v in zip(y, s) if a == 1]
    import bisect
    auc = sum(bisect.bisect_left(neg, v) + 0.5 * (bisect.bisect_right(neg, v) - bisect.bisect_left(neg, v))
              for v in pos) / max(1, len(pos) * len(neg))
    return {"auc": round(auc, 4), "akurasi": round(sum(a == b for a, b in zip(y, p)) / len(y), 4),
            "f1": round(2 * tp / max(1, 2 * tp + fp + fn), 4), "presisi": round(tp / max(1, tp + fp), 4),
            "recall": round(tp / max(1, tp + fn), 4), "n": len(y)}


def aturan_frame_model(folder_model, mj=None):
    """Aturan pengambilan frame yang dipakai saat model dilatih. Model baru menyimpannya di metrik.json
    (bagian ekstraksi); model lama disimpulkan dari waktu: dilatih sebelum ekstraksi terakhir = parameter
    ekstraksi sebelumnya (merata), sesudahnya = pengaturan ekstraksi sekarang."""
    cfg = muat_config()
    e_kini = {"mode_frame": "merata", "fps_sampel": 25, **(cfg.get("ekstraksi") or {})}
    akar = Path("data") / e_kini.get("folder", "wajah_mtcnn")
    mj = mj if mj is not None else _baca_json(Path(folder_model) / "metrik.json", {})
    for par in (mj.get("ekstraksi") or {}).values():
        return {"mode_frame": par.get("mode_frame", "merata"), "fps_sampel": par.get("fps_sampel"),
                "margin": par.get("margin", 1.3), "ukuran": par.get("ukuran", 224), "folder": par.get("folder"),
                "sumber": "metrik.json model"}
    # model tanpa catatan ekstraksi dibuat sebelum profil crop bisa diubah: margin 1.3, 224 px
    lama = {"margin": 1.3, "ukuran": 224, "folder": "wajah_mtcnn"}
    waktu_param = [f.stat().st_mtime for f in (Path("data") / "wajah_mtcnn").glob("parameter_*.json")]
    pt = Path(folder_model) / "terbaik.pt"
    if waktu_param and pt.exists() and pt.stat().st_mtime < max(waktu_param):
        return {"mode_frame": "merata", "fps_sampel": None, **lama,
                "sumber": "disimpulkan: model dilatih sebelum ekstraksi terakhir"}
    return {"mode_frame": "berurutan", "fps_sampel": 25, **lama,
            "sumber": "disimpulkan: model dilatih setelah ekstraksi terakhir"}


def daftar_model():
    hasil = []
    for pt in sorted(Path("data/model").glob("*/terbaik.pt")):
        mj = _baca_json(pt.parent / "metrik.json", {})
        uji = mj.get("uji") or {}
        if not uji or "video@0.5" not in next(iter(uji.values())):
            continue  # model lama (per frame) / training belum selesai
        at = aturan_frame_model(pt.parent, mj)
        hasil.append({"folder": pt.parent.as_posix(), "nama": pt.parent.name, "skenario": mj.get("skenario"),
                      "nama_skenario": mj.get("nama_skenario"), "arsitektur": mj.get("arsitektur"),
                      "val_auc": mj.get("val_auc"), "ambang_val": mj.get("ambang_val"),
                      "ambang_kandidat": mj.get("ambang_kandidat") or {},
                      "epoch_terbaik": mj.get("epoch_terbaik"), "aturan_frame": at["mode_frame"],
                      "fps_sampel": at.get("fps_sampel"), "sumber_aturan": at["sumber"],
                      "margin": at["margin"], "ukuran": at["ukuran"], "folder_data": at.get("folder"),
                      "uji": {**({"keseluruhan": g_} if (g_ := metrik_keseluruhan(pt.parent, mj)) else {}),
                              **{s: {k: u["video@0.5"].get(k) for k in ("auc", "akurasi", "f1")} for s, u in uji.items()}},
                      "auc_per_metode": {s: u.get("auc_per_metode", {}) for s, u in uji.items()},
                      "waktu": datetime.fromtimestamp(pt.stat().st_mtime).strftime("%Y-%m-%d %H:%M"),
                      "_t": pt.stat().st_mtime})
    hasil.sort(key=lambda m: -m["_t"])  # terbaru dulu
    for m in hasil:
        m.pop("_t")
    return hasil


def verifikasi_kasus(kasus_id):
    d = folder_kasus(kasus_id)
    utuh, masalah, per_file = forensik.verifikasi(d)
    return {"rantai_utuh": utuh, "masalah": masalah, "file": per_file, "waktu": sekarang()[1],
            "jumlah_entri": len(forensik.baca_log(d))}
