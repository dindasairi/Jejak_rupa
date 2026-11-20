"""Langkah 6 — Latih dan evaluasi detektor deepfake Xception + LSTM (PyTorch + timm, GPU).

Model: setiap frame wajah (224x224 dari langkah 5) -> Xception pralatih ImageNet (timm
`legacy_xception`) -> vektor fitur 2048 -> LSTM membaca urutan frame -> dense -> skor fake per video.
LSTM menangkap ketidakkonsistenan antarframe yang tidak terlihat dari satu frame saja.

Unit data = URUTAN wajah utama satu video: dari tiap frame dipilih wajah terbesar, diurutkan menurut
waktu (maks `panjang_urutan` frame). Label urutan = label mayoritas wajah-wajah tersebut.

Skenario (data latih):
  A  celebdf     : Celeb-DF v2 saja (baseline)
  B  gabungan    : Celeb-DF v2 + Indonesia
  C  indonesia   : Indonesia saja
  D  + wild      : Celeb-DF v2 + Indonesia + in the wild (video beredar di internet, 01w_in_the_wild.py)
  E  id + wild   : Indonesia + in the wild
Checkpoint dipilih dengan val AUC dari sumber data latih yang sama (A memakai val Celeb-DF saja).
Setiap model selalu diuji pada test Indonesia DAN test Celeb-DF, di tingkat video.

Augmentasi (hanya data latih, config.yaml bagian pelatihan.augmentasi): flip horizontal, rotasi,
filter warna + penajaman + grayscale, gaussian blur, opsional JPEG. Parameter acaknya SAMA untuk semua
frame dalam satu urutan, supaya LSTM tidak melihat "goyangan" buatan antarframe.

Grad-CAM: setelah training, contoh peta perhatian dibuat otomatis dari test set (lihat 07_gradcam.py).

  python 06_latih_xception.py --skenario B
  python 06_latih_xception.py --skenario A --epoch 1 --batas 200       # uji cepat (200 video per split)
  python 06_latih_xception.py --evaluasi data/model/B_20261008_1530     # evaluasi ulang model

Hasil: data/model/<skenario>_<waktu>/ terbaik.pt, riwayat.csv, metrik.json, ringkasan.md,
       prediksi_<sumber>.csv, gradcam/; model dicatat di log forensik.
"""
import argparse
import json
import math
import os
import random
import re
import time
from datetime import datetime
from pathlib import Path

import torch  # noqa: I001  (impor torch sebelum modul lain yang memakai CUDA)
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from common import folder, muat_config

SKENARIO = {"A": ["celebdf"], "B": ["celebdf", "indonesia"], "C": ["indonesia"],
            "D": ["celebdf", "indonesia", "wild"], "E": ["indonesia", "wild"]}
NAMA_SKENARIO = {"A": "Celeb-DF saja", "B": "Celeb-DF + Indonesia", "C": "Indonesia saja",
                 "D": "Celeb-DF + Indonesia + in the wild", "E": "Indonesia + in the wild"}
SUMBER = ("indonesia", "celebdf", "wild")

BAKU = {"model": "legacy_xception", "epoch": 10, "batch": 4, "lr": 1e-4, "lr_backbone": 1e-5,
        "weight_decay": 1e-4, "bekukan_epoch_awal": 1, "sabar": 3, "seimbang_metode": True,
        "pengaman": {"suhu_maks_gpu": 83, "suhu_lanjut_gpu": 72, "jeda_per_batch_ms": 0, "interval_pantau_detik": 10}, "workers": 4, "workers_eval": 2, "seed": 42,
        "sampel_per_epoch": None, "amp": True,
        "panjang_urutan": 20, "min_frame": 8, "lstm_hidden": 512, "lstm_lapis": 1, "lstm_dua_arah": False,
        "dropout": 0.5, "gradcam_per_kategori": 3,
        "augmentasi": {"flip": 0.5, "rotasi_derajat": 15, "rotasi_p": 0.5,
                       "filter_p": 0.5, "kecerahan": 0.2, "kontras": 0.2, "saturasi": 0.2, "hue": 0.05,
                       "ketajaman_p": 0.2, "grayscale_p": 0.05,
                       "blur_p": 0.3, "blur_kernel": 5, "blur_sigma": [0.1, 2.0],
                       "jpeg_p": 0.0, "jpeg_kualitas": [40, 95],
                       "downscale_p": 0.0, "downscale_skala": [0.35, 0.75],
                       "noise_p": 0.0, "noise_sigma": [0.005, 0.03]}}


# ------------------------------------------------------------------- data
def muat_metadata(cfg, sumber):
    akar = folder(cfg, (cfg.get("ekstraksi") or {}).get("folder", "wajah_mtcnn"), buat=False)
    p = akar / f"metadata_{sumber}.csv"
    if not p.exists():
        raise SystemExit(f"{p} belum ada. Jalankan langkah 5 (05_ekstrak_wajah.py) untuk {sumber}.")
    m = pd.read_csv(p, dtype={"video_id": str, "identitas": str, "metode": str})
    m = m.drop_duplicates("file", keep="last")
    return m[m["file"].map(lambda f: Path(f).exists())]


def bangun_urutan(meta, panjang, min_frame):
    """Satu urutan per video: wajah terbesar di tiap frame, urut waktu, maks `panjang` frame
    (diambil merata bila lebih). Kembalikan DataFrame satu baris per video."""
    utama = meta.sort_values("lebar_wajah", ascending=False).drop_duplicates(["video_id", "frame"])
    hasil, dibuang = [], 0
    for vid, g in utama.groupby("video_id", sort=True):
        g = g.sort_values("frame")
        if len(g) < min_frame:
            dibuang += 1
            continue
        if len(g) > panjang:
            g = g.iloc[np.linspace(0, len(g) - 1, panjang).round().astype(int)]
        r0 = g.iloc[0]
        hasil.append({"video_id": vid, "file": list(g["file"]), "frame": list(g["frame"]),
                      "label": int(g["label_wajah"].mean() >= 0.5), "label_video": int(r0["label_video"]),
                      "porsi_wajah_palsu": round(float(g["label_wajah"].mean()), 3),
                      "split": r0["split"], "sumber": r0["sumber"], "metode": r0["metode"],
                      "identitas": r0["identitas"], "panjang": len(g)})
    return pd.DataFrame(hasil), dibuang


class KompresiJPEG:
    def __init__(self, kualitas):
        self.kualitas = kualitas

    def __call__(self, img):
        import io
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=random.randint(*self.kualitas))
        buf.seek(0)
        return Image.open(buf).convert("RGB")


class BlurGaussian:
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        return img.filter(ImageFilter.GaussianBlur(random.uniform(*self.sigma)))


class SkalaTurunNaik:
    """Perkecil lalu perbesar kembali (meniru video yang diperkecil platform/aplikasi pesan lalu ditonton penuh).
    Faktor skala diambil dari `random`, jadi sama untuk semua frame satu video."""
    def __init__(self, skala):
        self.skala = skala

    def __call__(self, img):
        f = random.uniform(*self.skala)
        w, h = img.size
        kecil = img.resize((max(8, round(w * f)), max(8, round(h * f))), Image.BILINEAR)
        return kecil.resize((w, h), Image.BICUBIC)


class NoiseGauss:
    """Noise gaussian ringan (sensor kamera HP, cahaya redup). Besarnya (sigma) sama untuk semua frame satu video,
    tetapi pola butirnya berbeda tiap frame seperti noise sungguhan."""
    def __init__(self, sigma):
        self.sigma = sigma

    def __call__(self, img):
        s = random.uniform(*self.sigma)
        a = np.asarray(img, dtype=np.float32) / 255.0
        a = a + np.random.default_rng().normal(0.0, s, a.shape).astype(np.float32)
        return Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8))


def langkah_augmentasi(a):
    """Urutan augmentasi data latih (dipakai training DAN galeri contoh_augmentasi.py)."""
    import torchvision.transforms as T
    langkah = [
        T.RandomHorizontalFlip(a["flip"]),
        T.RandomApply([T.RandomRotation(a["rotasi_derajat"], interpolation=T.InterpolationMode.BILINEAR)],
                      p=a["rotasi_p"]),
        T.RandomApply([T.ColorJitter(a["kecerahan"], a["kontras"], a["saturasi"], a["hue"])], p=a["filter_p"]),
        T.RandomAdjustSharpness(2.0, p=a["ketajaman_p"]),
        T.RandomGrayscale(a["grayscale_p"]),
        T.RandomApply([BlurGaussian(a["blur_sigma"])], p=a["blur_p"]),
    ]
    if a.get("downscale_p"):
        langkah.append(T.RandomApply([SkalaTurunNaik(a["downscale_skala"])], p=a["downscale_p"]))
    if a.get("noise_p"):
        langkah.append(T.RandomApply([NoiseGauss(a["noise_sigma"])], p=a["noise_p"]))
    if a.get("jpeg_p"):  # kompresi terakhir, seperti video yang diunggah ulang
        langkah.append(T.RandomApply([KompresiJPEG(a["jpeg_kualitas"])], p=a["jpeg_p"]))
    return langkah


def augmentasi_tunggal(a):
    """Setiap augmentasi dipaksa aktif (peluang 1) dengan parameter yang sama, untuk ilustrasi."""
    import torchvision.transforms as T
    hasil = [
        ("Flip horizontal", T.RandomHorizontalFlip(1.0)),
        (f"Rotasi ±{a['rotasi_derajat']}°", T.RandomRotation(a["rotasi_derajat"],
                                                              interpolation=T.InterpolationMode.BILINEAR)),
        ("Filter warna", T.ColorJitter(a["kecerahan"], a["kontras"], a["saturasi"], a["hue"])),
        ("Penajaman", T.RandomAdjustSharpness(2.0, p=1.0)),
        ("Grayscale", T.Grayscale(num_output_channels=3)),
        (f"Gaussian blur σ{a['blur_sigma'][0]}-{a['blur_sigma'][1]}", BlurGaussian(a["blur_sigma"])),
    ]
    # ditampilkan selalu (ilustrasi), walau di config.yaml peluangnya 0 dan bisa dinyalakan per training
    hasil.append((f"JPEG q{a['jpeg_kualitas'][0]}-{a['jpeg_kualitas'][1]}", KompresiJPEG(a["jpeg_kualitas"])))
    hasil.append((f"Skala {a['downscale_skala'][0]}-{a['downscale_skala'][1]}x", SkalaTurunNaik(a["downscale_skala"])))
    hasil.append((f"Noise σ{a['noise_sigma'][0]}-{a['noise_sigma'][1]}", NoiseGauss(a["noise_sigma"])))
    return hasil


def pengaturan_augmentasi(cfg):
    return {**BAKU["augmentasi"], **((cfg.get("pelatihan") or {}).get("augmentasi") or {})}


def transformasi(a, latih, rata, simpangan):
    import torchvision.transforms as T
    akhir = [T.ToTensor(), T.Normalize(rata, simpangan)]
    if not latih:
        return T.Compose(akhir)
    return T.Compose(langkah_augmentasi(a) + akhir)


class DataUrutan(Dataset):
    """Satu item = tensor (T, 3, H, W) + panjang + label. Augmentasi identik untuk seluruh frame."""
    def __init__(self, tabel, tf, panjang, latih):
        self.file = tabel["file"].tolist()
        self.label = tabel["label"].astype(np.float32).tolist()
        self.tf, self.T, self.latih = tf, panjang, latih

    def __len__(self):
        return len(self.file)

    def __getitem__(self, i):
        benih = int.from_bytes(os.urandom(4), "little") if self.latih else 0
        frame = []
        for f in self.file[i]:
            if self.latih:  # parameter acak yang sama untuk setiap frame
                random.seed(benih)
                torch.manual_seed(benih)
            frame.append(self.tf(Image.open(f).convert("RGB")))
        n = len(frame)
        x = torch.stack(frame + [frame[-1]] * (self.T - n))  # isi sisa dengan frame terakhir (di-mask)
        return x, torch.tensor(n), torch.tensor(self.label[i])


# ------------------------------------------------------------------ model
class XceptionLSTM(nn.Module):
    def __init__(self, nama, hidden, lapis, dua_arah, dropout, pralatih=True):
        super().__init__()
        import timm
        from timm.data import resolve_data_config
        try:
            self.tulang = timm.create_model(nama, pretrained=pralatih, num_classes=0, global_pool="")
        except Exception as err:  # noqa: BLE001
            raise SystemExit(f"Gagal memuat model timm '{nama}': {err}")
        c = self.tulang.num_features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(dropout)
        self.lstm = nn.LSTM(c, hidden, num_layers=lapis, batch_first=True, bidirectional=dua_arah,
                            dropout=dropout if lapis > 1 else 0.0)
        self.kepala = nn.Linear(hidden * (2 if dua_arah else 1), 1)
        dc = resolve_data_config({}, model=self.tulang)
        self.rata, self.simpangan = dc["mean"], dc["std"]
        self.peta = None  # peta fitur konvolusi terakhir (untuk Grad-CAM)

    def forward(self, x, panjang, simpan_peta=False):
        b, t = x.shape[:2]
        peta = self.tulang.forward_features(x.flatten(0, 1))           # (B*T, C, h, w)
        if simpan_peta:
            peta.retain_grad()
            self.peta = peta
        fitur = self.drop(self.pool(peta).flatten(1)).view(b, t, -1)   # (B, T, C)
        kemas = nn.utils.rnn.pack_padded_sequence(fitur.float(), panjang.cpu(), batch_first=True,
                                                  enforce_sorted=False)
        _, (h, _) = self.lstm(kemas)
        akhir = torch.cat([h[-2], h[-1]], 1) if self.lstm.bidirectional else h[-1]
        return self.kepala(self.drop(akhir)).squeeze(1)


# ----------------------------------------------------------------- metrik
def metrik_biner(y, skor, ambang):
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, precision_score,
                                 recall_score, roc_auc_score)
    y = np.asarray(y).astype(int)
    pred = (np.asarray(skor) >= ambang).astype(int)
    hasil = {"n": int(len(y)), "n_palsu": int(y.sum()), "ambang": round(float(ambang), 4),
             "akurasi": float(accuracy_score(y, pred)),
             # rata-rata recall kelas asli dan palsu: tidak terpengaruh perbandingan jumlah asli/palsu data uji
             "akurasi_seimbang": float(balanced_accuracy_score(y, pred)) if len(set(y)) == 2 else float("nan"),
             "f1": float(f1_score(y, pred, zero_division=0)),
             "presisi": float(precision_score(y, pred, zero_division=0)),
             "recall": float(recall_score(y, pred, zero_division=0))}
    hasil["auc"] = float(roc_auc_score(y, skor)) if len(set(y)) == 2 else float("nan")
    return hasil


def ambang_youden(y, skor):
    from sklearn.metrics import roc_curve
    if len(set(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, skor)
    return float(min(1.0, thr[int(np.argmax(tpr - fpr))]))


def ambang_kandidat(y, skor):
    """Beberapa pilihan ambang dari data VAL:
    youden = maksimalkan TPR - FPR; fpr5 / fpr10 = ambang terendah yang membuat video asli salah dituduh fake
    paling banyak 5% / 10% (konservatif, cocok untuk forensik); f1 = F1 tertinggi."""
    from sklearn.metrics import roc_curve
    y, skor = np.asarray(y).astype(int), np.asarray(skor, dtype=float)
    if len(set(y)) < 2:
        return {"youden": 0.5, "fpr5": 0.5, "fpr10": 0.5, "f1": 0.5}
    fpr, tpr, thr = roc_curve(y, skor)
    thr = np.minimum(thr, 1.0)
    hasil = {"youden": float(thr[int(np.argmax(tpr - fpr))])}
    for nama, batas in (("fpr5", 0.05), ("fpr10", 0.10)):
        ok = np.where(fpr <= batas)[0]
        hasil[nama] = float(thr[ok[int(np.argmax(tpr[ok]))]])
    p_ = y.sum()
    f1 = [2 * t_ * p_ / max(1e-9, t_ * p_ + p_ + f_ * (len(y) - p_)) for t_, f_ in zip(tpr, fpr)]
    hasil["f1"] = float(thr[int(np.argmax(f1))])
    return {k: round(v, 4) for k, v in hasil.items()}


def evaluasi(pred, ambang_val):
    """Metrik tingkat video pada ambang 0.5 dan ambang dari val, plus AUC per metode fake."""
    from sklearn.metrics import roc_auc_score
    hasil = {"video@0.5": metrik_biner(pred["label"], pred["skor"], 0.5),
             "video@val": metrik_biner(pred["label"], pred["skor"], ambang_val)}
    asli = pred[pred["label"] == 0]
    hasil["auc_per_metode"] = {
        m: float(roc_auc_score([0] * len(asli) + [1] * len(f), list(asli["skor"]) + list(f["skor"])))
        for m, f in pred[pred["label"] == 1].groupby("metode") if len(asli)}
    return hasil


def _galat_memori(err):
    t = str(err)
    return "1455" in t or "shared file mapping" in t or "paging file" in t.lower() or "DataLoader worker" in t


@torch.no_grad()
def prediksi(model, loader, perangkat, amp):
    model.eval()
    try:
        skor = []
        for x, n, _ in loader:
            x = x.to(perangkat, non_blocking=True)
            with torch.autocast(device_type=perangkat.type, dtype=torch.float16, enabled=amp):
                skor.append(torch.sigmoid(model(x, n).float()).cpu())
    except RuntimeError as err:
        if not _galat_memori(err) or loader.num_workers == 0:
            raise
        # Windows: paging file / memori bersama habis (error 1455) -> ulangi tanpa proses worker
        print("  ! memori virtual Windows (paging file) tidak cukup untuk worker DataLoader; "
              "prediksi diulang tanpa worker (lebih lambat, hasil sama)", flush=True)
        return prediksi(model, DataLoader(loader.dataset, batch_size=loader.batch_size, shuffle=False,
                                          num_workers=0), perangkat, amp)
    return torch.cat(skor).numpy() if skor else np.array([])


# ------------------------------------------------------------- pemantauan
class Pemantau:
    """Mencatat suhu/daya GPU (nvidia-smi) dan CPU ke pemantauan.csv setiap beberapa detik. Setiap baris langsung
    ditulis ke disk, sehingga bila PC mati mendadak baris terakhir menunjukkan kondisi tepat sebelum mati."""
    KOLOM = ["waktu", "epoch", "batch", "suhu_gpu", "daya_w", "batas_daya_w", "util_gpu", "clock_mhz", "vram_mb",
             "kipas", "pembatas", "cpu_persen", "ram_persen"]

    def __init__(self, path, interval=10):
        import shutil
        import threading
        self.path, self.interval = Path(path), interval
        self.posisi = {"epoch": 0, "batch": 0}
        self.suhu = None
        self.smi = shutil.which("nvidia-smi")
        self._berhenti = threading.Event()
        baru = not self.path.exists()
        self.f = open(self.path, "a", encoding="utf-8")
        if baru:
            self.f.write(",".join(self.KOLOM) + "\n")
        try:
            import psutil
        except ImportError:  # untuk mencatat beban CPU/RAM; dipasang sekali
            import subprocess
            import sys
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "psutil"], check=False)
        try:
            import psutil
            self.psutil = psutil
            psutil.cpu_percent(None)
        except ImportError:
            self.psutil = None
        self.t = threading.Thread(target=self._loop, daemon=True)
        self.t.start()

    def baca_gpu(self):
        if not self.smi:
            return {}
        import subprocess
        q = ("temperature.gpu,power.draw,power.limit,utilization.gpu,clocks.sm,memory.used,fan.speed,"
             "clocks_throttle_reasons.active")
        try:
            out = subprocess.run([self.smi, f"--query-gpu={q}", "--format=csv,noheader,nounits"], capture_output=True,
                                 text=True, timeout=10).stdout.strip().splitlines()[0].split(",")
            nilai = [x.strip() for x in out]
            return dict(zip(["suhu_gpu", "daya_w", "batas_daya_w", "util_gpu", "clock_mhz", "vram_mb", "kipas",
                             "pembatas"], nilai))
        except Exception:  # noqa: BLE001
            return {}

    def _loop(self):
        while not self._berhenti.is_set():
            g_ = self.baca_gpu()
            try:
                self.suhu = float(g_.get("suhu_gpu")) if g_.get("suhu_gpu") not in (None, "", "[N/A]") else None
            except ValueError:
                self.suhu = None
            baris = {"waktu": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), **self.posisi, **g_}
            if self.psutil:
                baris["cpu_persen"] = self.psutil.cpu_percent(None)
                baris["ram_persen"] = self.psutil.virtual_memory().percent
            self.f.write(",".join(str(baris.get(k, "")) for k in self.KOLOM) + "\n")
            self.f.flush()
            try:
                os.fsync(self.f.fileno())
            except OSError:
                pass
            self._berhenti.wait(self.interval)

    def tunggu_dingin(self, maks, lanjut, cetak=print):
        """Jeda training bila GPU terlalu panas, sampai turun ke `lanjut` derajat."""
        if not maks or self.suhu is None or self.suhu < maks:
            return
        cetak(f"  ! suhu GPU {self.suhu:.0f}°C >= {maks}°C: training dijeda sampai turun ke {lanjut}°C", flush=True)
        mulai = time.time()
        while self.suhu is not None and self.suhu > lanjut and time.time() - mulai < 1800:
            time.sleep(5)
        cetak(f"  lanjut (suhu {self.suhu}°C, jeda {time.time() - mulai:.0f} dtk)", flush=True)

    def tutup(self):
        self._berhenti.set()
        self.t.join(timeout=15)
        self.f.close()


def simpan_atomik(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------ utama
def buat_loader(tabel, tf, p, latih):
    ds = DataUrutan(tabel, tf, p["panjang_urutan"], latih)
    # Di Windows setiap worker adalah proses baru yang memuat PyTorch + CUDA (~2-3 GB memori virtual).
    # Worker latih dibuat sekali (persistent); val/test memakai sedikit worker yang langsung ditutup,
    # supaya jumlah proses sekaligus tidak menghabiskan paging file (error 1455).
    w = int(p["workers"]) if latih else int(p.get("workers_eval", min(2, int(p["workers"]))))
    kw = dict(batch_size=p["batch"], num_workers=w, pin_memory=True,
              persistent_workers=latih and w > 0, prefetch_factor=2 if w > 0 else None)
    if latih:
        # seimbangkan label dan sumber supaya kelas/sumber minor tidak tenggelam; bila seimbang_metode,
        # di dalam kelompok fake setiap metode juga mendapat porsi sama (metode sulit lebih sering terlihat)
        kunci = list(zip(tabel["sumber"], tabel["label"]))
        hitung = pd.Series(kunci).value_counts()
        bobot = [1.0 / hitung[k] for k in kunci]
        if p.get("seimbang_metode", True):
            km = list(zip(tabel["sumber"], tabel["label"], tabel["metode"]))
            hm = pd.Series(km).value_counts()
            jm = {k: tabel[(tabel["sumber"] == k[0]) & (tabel["label"] == k[1])]["metode"].nunique() for k in hitung.index}
            bobot = [1.0 / (jm[(s_, l_)] * hm[(s_, l_, m_)]) if l_ == 1 else w
                     for (s_, l_, m_), w in zip(km, bobot)]
        n = int(p["sampel_per_epoch"] or len(ds))
        return DataLoader(ds, sampler=WeightedRandomSampler(bobot, n, replacement=True), **kw)
    return DataLoader(ds, shuffle=False, **kw)


def _bergiliran(df, kolom, seed):
    """Acak (seed tetap), lalu urutkan bergiliran antarnilai `kolom` supaya head(n) tersebar rata."""
    if df.empty:
        return df
    d = df.sample(frac=1, random_state=seed)
    d = d.assign(_urut=d.groupby(kolom).cumcount())
    return d.sort_values("_urut", kind="stable").drop(columns="_urut")


def pilih_data_latih(tabel, aturan, seed):
    """Kurangi data latih SATU sumber (undersampling). `aturan`: "semua", "seimbang" (palsu = jumlah
    asli), atau {"asli": n, "palsu": n}. Video asli dipilih merata per tokoh/identitas; video palsu
    mendapat kuota sama per metode (sisa kuota metode yang datanya kurang dialihkan ke metode lain),
    dan di dalam tiap metode merata per identitas. Kembalikan (tabel terpilih, ringkasan)."""
    asli, palsu = tabel[tabel["label_video"] == 0], tabel[tabel["label_video"] == 1]
    if aturan in (None, "", "semua"):
        n_a, n_p = len(asli), len(palsu)
    elif aturan == "seimbang":
        n_a, n_p = len(asli), min(len(palsu), len(asli))
    else:
        n_a = min(len(asli), int(aturan.get("asli") or len(asli)))
        n_p = min(len(palsu), int(aturan.get("palsu") or len(palsu)))
    pilih_a = _bergiliran(asli, "identitas", seed).head(n_a)
    grup = {m: _bergiliran(g, "identitas", seed) for m, g in palsu.groupby("metode")}
    kuota, sisa = {m: 0 for m in grup}, n_p
    while sisa > 0:
        aktif = [m for m in grup if kuota[m] < len(grup[m])]
        if not aktif:
            break
        tiap = max(1, sisa // len(aktif))
        for m in aktif:
            tambah = min(tiap, len(grup[m]) - kuota[m], sisa)
            kuota[m] += tambah
            sisa -= tambah
            if sisa <= 0:
                break
    pilih_p = pd.concat([grup[m].head(k) for m, k in kuota.items()]) if grup else palsu.head(0)
    ringkas = {"aturan": aturan or "semua", "asli": int(len(pilih_a)), "palsu": int(len(pilih_p)),
               "tersedia_asli": int(len(asli)), "tersedia_palsu": int(len(palsu)),
               "palsu_per_metode": {m: int(k) for m, k in kuota.items()}}
    return pd.concat([pilih_a, pilih_p]).sort_index(), ringkas


def batasi(tabel, n, seed):
    """Ambil maks n video per tabel, seimbang asli/palsu (uji cepat)."""
    if not n or len(tabel) <= n:
        return tabel
    bagian = [g.sample(min(len(g), n // 2), random_state=seed) for _, g in tabel.groupby("label")]
    return pd.concat(bagian).sort_index()


def atur_latih(model, beku, p):
    for prm in model.tulang.parameters():
        prm.requires_grad = not beku
    model.tulang.train(not beku)  # BatchNorm backbone tidak berubah saat dibekukan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skenario", choices=list(SKENARIO), default="B")
    ap.add_argument("--epoch", type=int)
    ap.add_argument("--batch", type=int, help="jumlah VIDEO per batch (tiap video = panjang_urutan frame)")
    ap.add_argument("--lr", type=float)
    ap.add_argument("--panjang-urutan", type=int, dest="panjang_urutan")
    ap.add_argument("--lanjut", help="folder model yang trainingnya terputus (PC mati dsb.): lanjutkan dari epoch "
                                     "terakhir yang selesai")
    ap.add_argument("--data-wajah-paksa", dest="data_wajah_paksa", action="store_true",
                    help="evaluasi dengan data wajah pilihan walau profil crop tercatat berbeda (untuk memperbaiki catatan)")
    ap.add_argument("--data-wajah", dest="data_wajah", help="folder hasil langkah 5 di data/ (baku config ekstraksi.folder)")
    ap.add_argument("--sabar", type=int, help="early stopping: berhenti bila val AUC tidak membaik sekian epoch (0 = mati)")
    ap.add_argument("--batas", type=int, help="batasi jumlah video per split (uji cepat)")
    ap.add_argument("--evaluasi", help="folder model yang sudah dilatih: hanya evaluasi test")
    ap.add_argument("--tanpa-gradcam", action="store_true")
    ap.add_argument("--tanpa-jendela", dest="tanpa_jendela", action="store_true",
                    help="lewati evaluasi multi-jendela walaupun metadata_<sumber>_jendela.csv ada")
    ap.add_argument("--augmentasi", help="JSON penimpa pengaturan augmentasi, mis. '{\"noise_p\": 0.3}' (peluang 0 = mati)")
    ap.add_argument("--jumlah-latih", dest="jumlah_latih",
                    help='JSON jumlah data LATIH per sumber, mis. \'{"celebdf": "seimbang", "indonesia": '
                         '{"asli": 500, "palsu": 500}}\' ("semua" = tanpa pengurangan). Val dan test tidak diubah.')
    ap.add_argument("--langkah-penuh", dest="langkah_penuh", action="store_true",
                    help="jumlah langkah per epoch disamakan dengan data latih penuh (perbandingan adil)")
    ap.add_argument("--h264", action="store_true",
                    help="tambahkan salinan data latih hasil kompresi ulang H.264 (metadata_<sumber>_h264.csv dari langkah 5)")
    args = ap.parse_args()

    cfg = muat_config()
    lanjut = None
    if args.lanjut:
        lanjut = torch.load(Path(args.lanjut) / "terakhir.pt", map_location="cpu", weights_only=False)
        args.skenario, args.data_wajah, args.batas = lanjut["skenario"], lanjut["data_wajah"], lanjut.get("batas")
        print(f"Melanjutkan {args.lanjut} dari epoch {lanjut['epoch'] + 1}", flush=True)
    if args.evaluasi and not args.data_wajah:
        # evaluasi ulang memakai data wajah yang sama dengan saat model dilatih (profil crop harus sama!)
        try:
            sm = torch.load(Path(args.evaluasi) / "terbaik.pt", map_location="cpu", weights_only=False)
            args.data_wajah = sm.get("data_wajah")
            del sm
        except Exception:  # noqa: BLE001
            pass
        if not args.data_wajah:
            try:
                mj = json.loads((Path(args.evaluasi) / "metrik.json").read_text(encoding="utf-8"))
                args.data_wajah = next((x.get("folder") for x in (mj.get("ekstraksi") or {}).values() if x.get("folder")), None)
            except (OSError, ValueError):
                pass
    if args.data_wajah:
        cfg.setdefault("ekstraksi", {})["folder"] = args.data_wajah
    print(f"Data wajah: data/{(cfg.get('ekstraksi') or {}).get('folder', 'wajah_mtcnn')}", flush=True)
    p = {**BAKU, **(cfg.get("pelatihan") or {})}
    p["augmentasi"] = pengaturan_augmentasi(cfg)
    for k in ("epoch", "batch", "lr", "panjang_urutan", "sabar"):
        if getattr(args, k) is not None:
            p[k] = getattr(args, k)
    if args.augmentasi:
        p["augmentasi"] = {**p["augmentasi"], **json.loads(args.augmentasi)}
    p["data_h264"] = bool(args.h264)
    p["jumlah_latih"] = json.loads(args.jumlah_latih) if args.jumlah_latih else {}
    p["langkah_seperti_penuh"] = bool(args.langkah_penuh)
    if lanjut:
        p = lanjut["pengaturan"]  # pengaturan harus sama persis dengan awal training
    pg = {"suhu_maks_gpu": 83, "suhu_lanjut_gpu": 72, "jeda_per_batch_ms": 0, "interval_pantau_detik": 10,
          **(p.get("pengaman") or {})}
    random.seed(p["seed"]), np.random.seed(p["seed"]), torch.manual_seed(p["seed"])
    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = bool(p["amp"]) and perangkat.type == "cuda"
    print(f"[Torch {torch.__version__}] perangkat: "
          f"{torch.cuda.get_device_name(0) if perangkat.type == 'cuda' else 'CPU (lambat!)'}", flush=True)

    if args.evaluasi:
        keluar = Path(args.evaluasi)
        simpan = torch.load(keluar / "terbaik.pt", map_location="cpu", weights_only=False)
        p.update(simpan["pengaturan"])
        skenario = simpan["skenario"]
    else:
        skenario = args.skenario
        keluar = Path(args.lanjut) if lanjut else folder(cfg, "model", f"{skenario}_{datetime.now():%Y%m%d_%H%M}")

    akar_meta = folder(cfg, (cfg.get("ekstraksi") or {}).get("folder", "wajah_mtcnn"), buat=False)
    urutan = {}
    for s in SUMBER:
        if (akar_meta / f"metadata_{s}.csv").exists():
            urutan[s], dibuang = bangun_urutan(muat_metadata(cfg, s), p["panjang_urutan"], p["min_frame"])
            if dibuang:
                print(f"  {s}: {dibuang} video dilewati (wajah < {p['min_frame']} frame)")
    perlu = SKENARIO[skenario]
    for s in perlu:
        if s not in urutan or urutan[s].empty:
            raise SystemExit(f"Skenario {skenario} butuh wajah {s} dari langkah 5. Jalankan langkah 5 dulu.")
    bagian_latih, ringkas_latih = [], {}
    for s in perlu:
        t_ = urutan[s][urutan[s]["split"] == "train"]
        aturan_s = (p.get("jumlah_latih") or {}).get(s, "semua")
        if aturan_s not in (None, "", "semua"):
            t_, ringkas_latih[s] = pilih_data_latih(t_, aturan_s, p["seed"])
            r_ = ringkas_latih[s]
            print(f"  {s}: data latih dikurangi -> {r_['asli']} asli (dari {r_['tersedia_asli']}), "
                  f"{r_['palsu']} palsu (dari {r_['tersedia_palsu']}); palsu per metode: "
                  + ", ".join(f"{m} {k}" for m, k in r_["palsu_per_metode"].items()), flush=True)
        bagian_latih.append(t_)
    latih = pd.concat(bagian_latih, ignore_index=True)
    if ringkas_latih:
        p["jumlah_latih_terpakai"] = ringkas_latih
        n_penuh = sum(int((urutan[s]["split"] == "train").sum()) for s in perlu)
        if p.get("langkah_seperti_penuh") and not p.get("sampel_per_epoch"):
            p["sampel_per_epoch"] = n_penuh
            print(f"  langkah per epoch disamakan dengan data penuh: {n_penuh} urutan", flush=True)
        if not args.evaluasi:
            latih[["video_id", "sumber", "label_video", "metode", "identitas"]].to_csv(
                keluar / "data_latih_terpilih.csv", index=False)
    if p.get("data_h264") and not args.evaluasi:
        # salinan kompresi ulang H.264 (hanya split train): menambah ragam, jumlah langkah per epoch tetap sama
        tambahan = []
        for s in perlu:
            if (akar_meta / f"metadata_{s}_h264.csv").exists():
                u_, _ = bangun_urutan(muat_metadata(cfg, f"{s}_h264"), p["panjang_urutan"], p["min_frame"])
                u_ = u_[u_["split"] == "train"]
                if s in ringkas_latih:  # hanya salinan H.264 dari video latih yang terpilih
                    terpilih = set(latih.loc[latih["sumber"] == s, "video_id"])
                    u_ = u_[u_["video_id"].map(lambda v: re.sub(r"_crf\d+(_s\d+)?$", "", v)).isin(terpilih)]
                tambahan.append(u_)
            else:
                print(f"  ! {s}: belum ada data kompresi ulang H.264 (langkah 5 opsi H.264), dilewati", flush=True)
        if tambahan:
            if not p.get("sampel_per_epoch"):
                p["sampel_per_epoch"] = len(latih)
            n0 = len(latih)
            latih = pd.concat([latih, *tambahan], ignore_index=True)
            print(f"  + {len(latih) - n0} urutan kompresi ulang H.264 ditambahkan ke data latih "
                  f"(sampel per epoch tetap {p['sampel_per_epoch']})", flush=True)
    val = pd.concat([urutan[s][urutan[s]["split"] == "val"] for s in perlu], ignore_index=True)
    latih, val = batasi(latih, args.batas, p["seed"]), batasi(val, args.batas, p["seed"])
    uji = {s: batasi(u[u["split"] == "test"].reset_index(drop=True), args.batas, p["seed"]).reset_index(drop=True)
           for s, u in urutan.items()}

    model = XceptionLSTM(p["model"], p["lstm_hidden"], p["lstm_lapis"], p["lstm_dua_arah"], p["dropout"],
                         pralatih=not args.evaluasi).to(perangkat)
    tf_latih = transformasi(p["augmentasi"], True, model.rata, model.simpangan)
    tf_uji = transformasi(p["augmentasi"], False, model.rata, model.simpangan)
    if args.evaluasi:  # cegah evaluasi dengan crop yang berbeda dari saat training
        try:
            from PIL import Image as _Im
            contoh = next(f for t in uji.values() for r in t["file"] for f in r)
            ukuran_data = _Im.open(contoh).size[0]
            mj_lama = json.loads((keluar / "metrik.json").read_text(encoding="utf-8")) if (keluar / "metrik.json").exists() else {}
            uk_latih = next((x.get("ukuran") for x in (mj_lama.get("ekstraksi") or {}).values()), None)
            if uk_latih and int(uk_latih) != ukuran_data and not args.data_wajah_paksa:
                raise SystemExit(f"Data wajah berukuran {ukuran_data} px, padahal model tercatat dilatih dengan {uk_latih} px. "
                                 "Pilih data wajah yang sama dengan saat training (atau --data-wajah-paksa bila yakin).")
        except StopIteration:
            pass
    print(f"Skenario {skenario} ({NAMA_SKENARIO[skenario]}), Xception + LSTM, urutan {p['panjang_urutan']} frame: "
          f"latih {len(latih)} video ({int(latih['label'].sum())} palsu), val {len(val)}, "
          + ", ".join(f"test {s} {len(t)}" for s, t in uji.items()), flush=True)

    if not args.evaluasi:
        dl_latih = buat_loader(latih, tf_latih, p, True)
        dl_val = buat_loader(val, tf_uji, p, False)
        kriteria = nn.BCEWithLogitsLoss()
        param_tulang = list(model.tulang.parameters())
        id_tulang = {id(x) for x in param_tulang}
        param_lain = [x for x in model.parameters() if id(x) not in id_tulang]
        opt = torch.optim.AdamW([{"params": param_tulang, "lr": p["lr_backbone"]},
                                 {"params": param_lain, "lr": p["lr"]}], weight_decay=p["weight_decay"])
        total_langkah = p["epoch"] * len(dl_latih)
        jadwal = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(1, total_langkah)))))
        skala = torch.amp.GradScaler("cuda", enabled=amp)
        terbaik, sabar, riwayat, ep_awal = -1.0, 0, [], 1
        if lanjut:
            model.load_state_dict(lanjut["model"])
            opt.load_state_dict(lanjut["opt"])
            jadwal.load_state_dict(lanjut["jadwal"])
            skala.load_state_dict(lanjut["skala"])
            terbaik, sabar, riwayat, ep_awal = lanjut["terbaik"], lanjut["sabar"], lanjut["riwayat"], lanjut["epoch"] + 1
            del lanjut
        pemantau = Pemantau(keluar / "pemantauan.csv", pg["interval_pantau_detik"]) if perangkat.type == "cuda" else None
        print(f"Pengaman: jeda bila GPU >= {pg['suhu_maks_gpu']}°C (lanjut <= {pg['suhu_lanjut_gpu']}°C)"
              + (f", jeda {pg['jeda_per_batch_ms']} ms per batch" if pg["jeda_per_batch_ms"] else "")
              + "; suhu/daya dicatat di pemantauan.csv. Bila PC mati: lanjutkan dari menu Latih & evaluasi.", flush=True)
        berhenti = False
        for ep in range(ep_awal, p["epoch"] + 1):
            if berhenti:
                break
            beku = ep <= p["bekukan_epoch_awal"]
            model.train()
            atur_latih(model, beku, p)
            t0, total_loss, n, benar = time.time(), 0.0, 0, 0
            for b, (x, panjang, y) in enumerate(dl_latih, 1):
                if pemantau:
                    pemantau.posisi.update(epoch=ep, batch=b)
                    if b % 5 == 0:
                        pemantau.tunggu_dingin(pg["suhu_maks_gpu"], pg["suhu_lanjut_gpu"], print)
                if pg["jeda_per_batch_ms"]:
                    time.sleep(pg["jeda_per_batch_ms"] / 1000)
                x, y = x.to(perangkat, non_blocking=True), y.to(perangkat, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=perangkat.type, dtype=torch.float16, enabled=amp):
                    logit = model(x, panjang).float()
                    loss = kriteria(logit, y)
                skala.scale(loss).backward()
                skala.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                skala.step(opt)
                skala.update()
                jadwal.step()
                total_loss += loss.item() * len(y)
                benar += int(((logit.detach() > 0).float() == y).sum().item())
                n += len(y)
                if b % 50 == 0:
                    print(f"  epoch {ep} batch {b}/{len(dl_latih)} loss {total_loss / n:.4f}", flush=True)
            sv = prediksi(model, dl_val, perangkat, amp)
            m_val = metrik_biner(val["label"], sv, 0.5)
            pv = np.clip(np.asarray(sv, dtype=float), 1e-7, 1 - 1e-7)
            yv = val["label"].to_numpy().astype(float)
            loss_val = float(-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)))
            baris = {"epoch": ep, "backbone_beku": beku, "loss_latih": round(total_loss / max(1, n), 5),
                     "loss_val": round(loss_val, 5), "akurasi_latih": round(benar / max(1, n), 4),
                     "val_auc": round(m_val["auc"], 4), "val_akurasi": round(m_val["akurasi"], 4),
                     "val_f1": round(m_val["f1"], 4), "detik": round(time.time() - t0)}
            riwayat.append(baris)
            print(f"Epoch {ep}/{p['epoch']}: loss latih {baris['loss_latih']} (akurasi {baris['akurasi_latih']}), "
                  f"loss val {baris['loss_val']}, val AUC {baris['val_auc']}, "
                  f"akurasi {baris['val_akurasi']}, F1 {baris['val_f1']} ({baris['detik']} dtk)", flush=True)
            pd.DataFrame(riwayat).to_csv(keluar / "riwayat.csv", index=False)
            skor_pilih = m_val["auc"] if not math.isnan(m_val["auc"]) else -baris["loss_latih"]
            if skor_pilih > terbaik:
                terbaik, sabar = skor_pilih, 0
                ambang_val = ambang_youden(val["label"].astype(int), sv)
                torch.save({"model": model.state_dict(), "pengaturan": p, "skenario": skenario, "epoch": ep,
                            "ambang_val": ambang_val, "val_auc": m_val["auc"],
                            "arsitektur": "Xception + LSTM",
                            "data_wajah": (cfg.get("ekstraksi") or {}).get("folder", "wajah_mtcnn")}, keluar / "terbaik.pt")
                print(f"  -> checkpoint terbaik disimpan (ambang val {ambang_val:.3f})", flush=True)
            elif not beku:
                sabar += 1
                if p["sabar"] and sabar >= p["sabar"]:
                    print(f"Berhenti lebih awal: val AUC tidak membaik {p['sabar']} epoch.", flush=True)
                    berhenti = True
            # titik lanjut: bila PC mati, training bisa diteruskan dari epoch berikutnya
            simpan_atomik({"model": model.state_dict(), "opt": opt.state_dict(), "jadwal": jadwal.state_dict(),
                           "skala": skala.state_dict(), "epoch": ep, "terbaik": terbaik, "sabar": sabar,
                           "riwayat": riwayat, "pengaturan": p, "skenario": skenario, "batas": args.batas,
                           "data_wajah": (cfg.get("ekstraksi") or {}).get("folder", "wajah_mtcnn")},
                          keluar / "terakhir.pt")
        if pemantau:
            pemantau.tutup()
        (keluar / "terakhir.pt").unlink(missing_ok=True)  # training selesai: titik lanjut tidak diperlukan lagi

    simpan = torch.load(keluar / "terbaik.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(simpan["model"])
    # ambang dihitung ulang dari skor VAL model terbaik (juga saat evaluasi ulang dengan data yang benar)
    sv = prediksi(model, buat_loader(val, tf_uji, p, False), perangkat, amp)
    val.assign(skor=sv)[["video_id", "label", "metode", "sumber", "skor"]].to_csv(keluar / "prediksi_val.csv", index=False)
    kandidat = ambang_kandidat(val["label"], sv)
    ambang_val = kandidat["youden"]
    print("Ambang dari val: " + ", ".join(f"{k} {v:.3f}" for k, v in kandidat.items()), flush=True)
    ekstraksi_dipakai = {}
    for s_ in urutan:
        try:
            ekstraksi_dipakai[s_] = json.loads((akar_meta / f"parameter_{s_}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    hasil = {"skenario": skenario, "nama_skenario": NAMA_SKENARIO[skenario], "arsitektur": "Xception + LSTM",
             "ekstraksi": ekstraksi_dipakai,
             "epoch_terbaik": simpan["epoch"], "val_auc": simpan["val_auc"], "ambang_val": ambang_val,
             "ambang_kandidat": kandidat,
             "pengaturan": p, "torch": torch.__version__, "uji": {}}
    semua_pt = []
    for s, t in uji.items():
        sk = prediksi(model, buat_loader(t, tf_uji, p, False), perangkat, amp)
        pt = t.assign(skor=sk)
        pt[["video_id", "label", "label_video", "porsi_wajah_palsu", "metode", "identitas", "panjang", "skor"]] \
            .to_csv(keluar / f"prediksi_{s}.csv", index=False)
        hasil["uji"][s] = evaluasi(pt, ambang_val)
        hasil["uji"][s]["video@fpr5"] = metrik_biner(pt["label"], pt["skor"], kandidat["fpr5"])
        semua_pt.append(pt.assign(sumber_uji=s))
    if len(semua_pt) > 1:  # semua data uji digabung (Indonesia + Celeb-DF) sebagai satu set
        gab_ = pd.concat(semua_pt, ignore_index=True)
        hasil["uji_gabungan"] = evaluasi(gab_, ambang_val)
        hasil["uji_gabungan"]["video@fpr5"] = metrik_biner(gab_["label"], gab_["skor"], kandidat["fpr5"])
        hasil["uji_gabungan"]["sumber"] = list(hasil["uji"])

    # evaluasi multi-jendela: beberapa jendela 20 frame per video uji (langkah 5 opsi jendela uji), skor video =
    # rata-rata jendela. Pembanding: jendela tengah saja pada video yang sama. Keduanya memakai label resmi video.
    n_j = int((cfg.get("evaluasi_jendela") or {}).get("jumlah", 5))
    hasil_j, gab_j = {}, []
    for s in ([] if args.tanpa_jendela or args.batas else list(uji)):
        if not (akar_meta / f"metadata_{s}_jendela.csv").exists():
            continue
        uj, _ = bangun_urutan(muat_metadata(cfg, f"{s}_jendela"), p["panjang_urutan"], p["min_frame"])
        uj = uj[uj["split"] == "test"].reset_index(drop=True) if not uj.empty else uj
        if uj.empty:
            continue
        print(f"Evaluasi multi-jendela {s}: {len(uj)} jendela", flush=True)
        uj = uj.assign(skor=prediksi(model, buat_loader(uj, tf_uji, p, False), perangkat, amp))
        uj["video_induk"] = uj["video_id"].str.replace(r"_w\d+$", "", regex=True)
        uj["jendela"] = uj["video_id"].str.extract(r"_w(\d+)$")[0].astype(int)
        uj[["video_induk", "jendela", "label_video", "metode", "identitas", "panjang", "skor"]].to_csv(
            keluar / f"jendela_prediksi_{s}.csv", index=False)
        rata = uj.groupby("video_induk").agg(skor=("skor", "mean"), label=("label_video", "first"),
                                             metode=("metode", "first"), n_jendela=("skor", "size"))
        tengah = uj[uj["jendela"] == n_j // 2].set_index("video_induk")["skor"]
        sama = rata.index.intersection(tengah.index)  # video yang punya jendela tengah: perbandingan adil
        r_ = rata.loc[sama].reset_index()
        t_ = r_.assign(skor=tengah.loc[sama].values)
        hasil_j[s] = {"n_video": int(len(sama)), "jendela_rata_rata": float(rata["n_jendela"].mean()),
                      "satu_jendela": evaluasi(t_, ambang_val), "rata_jendela": evaluasi(r_, ambang_val)}
        for k_, d_ in (("satu_jendela", t_), ("rata_jendela", r_)):
            hasil_j[s][k_]["video@fpr5"] = metrik_biner(d_["label"], d_["skor"], kandidat["fpr5"])
        gab_j.append((t_, r_))
    if hasil_j:
        if len(gab_j) > 1:
            t_ = pd.concat([x for x, _ in gab_j], ignore_index=True)
            r_ = pd.concat([y for _, y in gab_j], ignore_index=True)
            hasil_j["keseluruhan"] = {"n_video": int(len(r_)), "satu_jendela": evaluasi(t_, ambang_val),
                                      "rata_jendela": evaluasi(r_, ambang_val)}
        hasil["uji_jendela"] = {"jumlah_jendela": n_j, "label": "label resmi video",
                                "catatan": "skor video = rata-rata skor semua jendela; pembanding = jendela tengah saja "
                                           "pada video yang sama; ambang dari data val satu jendela", **hasil_j}
    (keluar / "metrik.json").write_text(json.dumps(hasil, indent=2, ensure_ascii=False), encoding="utf-8")

    def sel(m):
        return (f"{m['auc']:.3f} | {m['akurasi']:.3f} | {m.get('akurasi_seimbang', float('nan')):.3f} | "
                f"{m['f1']:.3f} | {m['presisi']:.3f} | {m['recall']:.3f}")
    md = [f"# Skenario {skenario}: {NAMA_SKENARIO[skenario]}", "",
          f"Xception ({p['model']}) + LSTM (hidden {p['lstm_hidden']}, {p['lstm_lapis']} lapis"
          f"{', dua arah' if p['lstm_dua_arah'] else ''}), urutan {p['panjang_urutan']} frame, "
          f"epoch terbaik {simpan['epoch']}, val AUC {simpan['val_auc']:.3f}, ambang val {ambang_val:.3f}", "",
          "| Data uji (tingkat video) | Ambang | AUC | Akurasi | Akurasi seimbang | F1 | Presisi | Recall |",
          "|---|---|---|---|---|---|---|---|"]
    for s, h in ([("KESELURUHAN", hasil["uji_gabungan"])] if "uji_gabungan" in hasil else []) + list(hasil["uji"].items()):
        md += [f"| {s} | 0.5 | {sel(h['video@0.5'])} |", f"| {s} | Youden val ({ambang_val:.3f}) | {sel(h['video@val'])} |",
               f"| {s} | FPR val ≤ 5% ({kandidat['fpr5']:.3f}) | {sel(h['video@fpr5'])} |"]
    if p.get("jumlah_latih_terpakai"):
        md += ["", "Data latih (undersampling): " + "; ".join(
            f"{s} {r['asli']} asli / {r['palsu']} palsu (tersedia {r['tersedia_asli']} / {r['tersedia_palsu']})"
            for s, r in p["jumlah_latih_terpakai"].items())]
    for s, h in hasil["uji"].items():
        if h["auc_per_metode"]:
            md += ["", f"AUC per metode ({s}): " + ", ".join(f"{m} {a:.3f}" for m, a in h["auc_per_metode"].items())]
    if hasil.get("uji_jendela"):
        hj = hasil["uji_jendela"]

        def sel_j(m):
            return (f"{m['auc']:.3f} | {m['akurasi']:.3f} | {m.get('akurasi_seimbang', float('nan')):.3f} | "
                    f"{m['f1']:.3f}")
        md += ["", f"## Evaluasi multi-jendela ({hj['jumlah_jendela']} jendela x {p['panjang_urutan']} frame per video uji)",
               "", "Skor video = rata-rata skor semua jendela. Pembanding = jendela tengah saja pada video yang sama. "
               "Keduanya memakai label resmi video, jadi angkanya bisa sedikit berbeda dari tabel di atas.", "",
               "| Data uji | Cara | Ambang | AUC | Akurasi | Akurasi seimbang | F1 |", "|---|---|---|---|---|---|---|"]
        for s_, h_ in ([("KESELURUHAN", hj["keseluruhan"])] if "keseluruhan" in hj else []) + \
                [(k, v) for k, v in hj.items() if isinstance(v, dict) and k != "keseluruhan"]:
            for cara, kunci in (("1 jendela (tengah)", "satu_jendela"), (f"rata-rata {hj['jumlah_jendela']} jendela", "rata_jendela")):
                md += [f"| {s_} ({h_['n_video']} video) | {cara} | 0.5 | {sel_j(h_[kunci]['video@0.5'])} |",
                       f"| {s_} ({h_['n_video']} video) | {cara} | Youden val ({ambang_val:.3f}) | {sel_j(h_[kunci]['video@val'])} |"]
    (keluar / "ringkasan.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n" + "\n".join(md), flush=True)

    if not args.tanpa_gradcam and p.get("gradcam_per_kategori"):
        try:
            import importlib.util
            spec = importlib.util.spec_from_file_location("gradcam", Path(__file__).with_name("07_gradcam.py"))
            G = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(G)
            G.contoh_dari_prediksi(model, keluar, uji, tf_uji, perangkat, int(p["gradcam_per_kategori"]))
        except Exception as err:  # noqa: BLE001
            print(f"! Grad-CAM gagal dibuat: {err}")

    try:  # grafik untuk tesis: loss, akurasi, confusion matrix, metrik per kelas, ROC, distribusi skor
        import grafik_model
        for a_ in ("0.5", "fpr5"):
            grafik_model.buat_semua(keluar, a_, log=lambda m: print(m, flush=True))
    except Exception as err:  # noqa: BLE001
        print(f"! Grafik gagal dibuat: {err}")

    from forensik import catat
    catat("6_latih_model" if not args.evaluasi else "6_evaluasi_model", keluar / "terbaik.pt",
          induk=[akar_meta / f"metadata_{s}.csv" for s in urutan], folder_kerja=cfg["folder_kerja"],
          info={"skenario": skenario, "arsitektur": "Xception + LSTM", "epoch_terbaik": simpan["epoch"],
                "pengaturan": p,
                "ringkasan_uji": {s: {"auc_video": h["video@0.5"]["auc"], "akurasi": h["video@0.5"]["akurasi"],
                                      "f1": h["video@0.5"]["f1"]} for s, h in hasil["uji"].items()}})
    print(f"\nHasil tersimpan di {keluar}")


if __name__ == "__main__":
    main()
