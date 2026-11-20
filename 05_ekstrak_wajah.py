"""Langkah 5 — Ekstraksi wajah dengan MTCNN untuk pelatihan Xception.

Spesifikasi (config.yaml bagian `ekstraksi`): 20 frame BERURUTAN per video pada 25 fps (satu jendela
0,8 detik di tengah video; `mode_frame: berurutan`, `fps_sampel: 25`; video ber-fps lain diambil pada
waktu yang setara 25 fps) supaya gerak bibir antarframe terbaca (penting untuk deteksi lipsync), setiap wajah
dideteksi MTCNN (facenet-pytorch, GPU), di-crop persegi dengan margin, lalu di-resize 224x224.
Indonesia dan Celeb-DF v2 diproses dengan cara yang SAMA PERSIS supaya perbandingan adil.

Label PER WAJAH:
  * video asli  -> semua wajah berlabel asli (0)
  * video fake  -> hanya wajah yang identitasnya berubah dibanding frame yang sama di video aslinya
                   (dicocokkan posisi, identitas dibandingkan dengan ArcFace) berlabel palsu (1);
                   wajah lain tetap asli (0); wajah yang tak bisa dipastikan dilewati.
  * video fake LIPSYNC (metode berisi "lipsync") -> identitas tidak berubah, jadi ArcFace tidak bisa
                   dipakai; wajah TERBESAR di frame (yang dipilih FaceFusion, face selector large-small)
                   berlabel palsu (1), wajah lain asli (0).

Celeb-DF v2: split test mengikuti List_of_testing_videos.txt resmi. Dari sisanya, sebagian kecil
identitas dijadikan val (untuk memilih checkpoint). Video fake train/val Celeb-DF diambil acak
sebanyak `rasio_fake_per_asli` x jumlah video asli supaya seimbang; test diambil semua.

  python 05_ekstrak_wajah.py                         # Indonesia + Celeb-DF
  python 05_ekstrak_wajah.py --sumber indonesia
  python 05_ekstrak_wajah.py --sumber celebdf --batas 20   # uji cepat
  python 05_ekstrak_wajah.py --sumber wild                 # dataset in the wild (01w_in_the_wild.py)

Hasil (satu folder per video):
       data/wajah_mtcnn/<sumber>/<split>/<asli|fake>/<video_id>/<frame>_<k>_<asli|palsu>.png
         - folder asli/fake = jenis VIDEO
         - akhiran nama file = label WAJAH (di video fake bisa ada wajah orang lain yang tetap asli)
       data/wajah_mtcnn/metadata_<sumber>.csv (satu baris per wajah)
"""
import argparse
import csv
import json
import random
import time
from collections import Counter
from pathlib import Path

# torch HARUS diimpor sebelum onnxruntime/insightface: ONNX Runtime lalu memakai DLL CUDA milik torch
import torch  # noqa: I001
import cv2
import numpy as np

from common import folder, muat_config
from forensik import catat, sha256_file

KOLOM = ["file", "sha256", "video_id", "sha256_video", "frame", "wajah_ke", "label_wajah", "label_video",
         "split", "sumber", "metode", "identitas", "prob_deteksi", "lebar_wajah"]

BAKU = {"folder": "wajah_mtcnn", "frame_per_video": 20, "mode_frame": "berurutan", "fps_sampel": 25, "ukuran": 224, "margin": 1.3, "ukuran_wajah_min": 50,
        "prob_min": 0.95, "ambang_mtcnn": [0.6, 0.7, 0.7], "batch_frame": 10, "format": "png",
        "ambang_identitas_berubah": 0.6, "maks_porsi_hitam": 0.3}
BAKU_CDF = {"folder": "data/archive_Celeb-dfv2", "porsi_val": 0.1, "rasio_fake_per_asli": 1.0, "seed": 42}


# ---------------------------------------------------------------- daftar video
def daftar_indonesia(cfg):
    with open(folder(cfg) / "split_tokoh.json", encoding="utf-8") as f:
        tokoh_ke_split = {t: s for s, ts in json.load(f).items() for t in ts}
    hasil, hilang = [], 0
    with open(folder(cfg) / "klip_wajah.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = Path(r["file"].replace("\\", "/"))
            if not p.exists() or r["tokoh_id"] not in tokoh_ke_split:
                hilang += 1
                continue
            hasil.append(dict(sumber="indonesia", split=tokoh_ke_split[r["tokoh_id"]], label=0,
                              video_id=r["klip_id"], file=str(p), metode="asli", pasangan=None,
                              identitas=r["tokoh_id"]))
    belum = 0
    with open(folder(cfg) / "rencana_fake.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            p = Path(r["file_keluaran"].replace("\\", "/"))
            if not p.exists():
                belum += 1
                continue
            hasil.append(dict(sumber="indonesia", split=r["split"], label=1, video_id=r["fake_id"], file=str(p),
                              metode=r["metode"], pasangan=r["video_target"].replace("\\", "/"),
                              identitas=r["tokoh_target"]))
    if hilang:
        print(f"! {hilang} video asli tidak ada filenya / tokoh tanpa split, dilewati")
    if belum:
        print(f"! {belum} video fake di rencana belum ada filenya, dilewati")
    return hasil


def daftar_wild(cfg):
    """Klip in the wild (01w_in_the_wild.py): split sudah ditentukan per kelompok orang/video sumber."""
    p = folder(cfg) / "in_the_wild" / "klip.csv"
    if not p.exists():
        raise SystemExit("data/in_the_wild/klip.csv belum ada. Jalankan menu Dataset in the wild (unduh + potong) dulu.")
    hasil, hilang = [], 0
    with open(p, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q = Path(r["file"].replace("\\", "/"))
            if not q.exists():
                hilang += 1
                continue
            hasil.append(dict(sumber="wild", split=r["split"], label=int(r["label"]), video_id=r["klip_id"], file=str(q),
                              metode=r["metode"], pasangan=None, identitas=r["identitas"],
                              wajah_fake=r.get("wajah_fake") or "terbesar"))
    if hilang:
        print(f"! {hilang} klip in the wild tidak ada filenya, dilewati")
    return hasil


def daftar_celebdf(root, c):
    root = Path(root)
    lst = root / "List_of_testing_videos.txt"
    if not lst.exists():
        raise SystemExit(f"{lst} tidak ditemukan. Periksa celebdf.folder di config.yaml")
    uji = {b.split()[1] for b in lst.read_text().splitlines() if len(b.split()) == 2}
    rng = random.Random(c["seed"])
    video = []
    for sub, label in [("Celeb-real", 0), ("YouTube-real", 0), ("Celeb-synthesis", 1)]:
        for p in sorted((root / sub).glob("*.mp4")):
            b = p.stem.split("_")
            pasangan, identitas = None, (b[0] if sub != "YouTube-real" else f"yt_{p.stem}")
            if label and len(b) == 3:  # id0_id16_0000 -> target id0, video asli Celeb-real/id0_0000
                kandidat = root / "Celeb-real" / f"{b[0]}_{b[2]}.mp4"
                pasangan = str(kandidat) if kandidat.exists() else None
            video.append(dict(sumber="celebdf", split="test" if f"{sub}/{p.name}" in uji else "train",
                              label=label, video_id=f"{sub}_{p.stem}", file=str(p), metode=sub,
                              pasangan=pasangan, identitas=identitas))
    # val: sebagian identitas (bukan video) dari non-test, supaya identitas val tidak muncul di train
    ids = sorted({v["identitas"] for v in video if v["split"] == "train"})
    rng.shuffle(ids)
    val_ids = set(ids[:max(1, round(len(ids) * c["porsi_val"]))])
    for v in video:
        if v["split"] == "train" and v["identitas"] in val_ids:
            v["split"] = "val"
    # seimbangkan fake di train/val (test tetap lengkap sesuai daftar resmi)
    hasil = [v for v in video if v["split"] == "test" or v["label"] == 0]
    rasio = c.get("rasio_fake_per_asli")
    for s in ("train", "val"):
        fake = [v for v in video if v["split"] == s and v["label"] == 1]
        if rasio:
            n = int(round(sum(1 for v in video if v["split"] == s and v["label"] == 0) * float(rasio)))
            rng.shuffle(fake)
            fake = sorted(fake[:n], key=lambda v: v["video_id"])
        hasil += fake
    return sorted(hasil, key=lambda v: (v["split"], v["video_id"]))


# --------------------------------------------------- kompresi ulang H.264 (augmentasi)
BAKU_H264 = {"crf": [23, 28, 32, 36], "salinan_per_video": 1, "skala": [1.0, 0.75, 0.5], "seed": 42}


def rencana_h264(daftar, k, seed):
    """Salinan kompresi ulang untuk video TRAIN saja: tiap video mendapat `salinan_per_video` kombinasi
    (CRF, skala resolusi) acak tetapi tetap (seed), sehingga bisa dilanjutkan dan diulang dengan hasil sama."""
    rng = random.Random(seed)
    hasil = []
    for v in sorted((v for v in daftar if v["split"] == "train"), key=lambda v: v["video_id"]):
        pilihan = [(c, f) for c in k["crf"] for f in k["skala"]]
        for crf, f in rng.sample(pilihan, min(int(k["salinan_per_video"]), len(pilihan))):
            hasil.append({**v, "video_id": f"{v['video_id']}_crf{crf}" + (f"_s{int(round(f * 100))}" if f < 1 else ""),
                          "asal_video": v["file"], "crf": crf, "skala": float(f)})
    return hasil


def rencana_jendela(daftar, n_jendela):
    """Beberapa jendela per video TEST untuk evaluasi multi-jendela: video dibagi menjadi `n_jendela` bagian sama
    panjang dan dari tengah setiap bagian diambil satu jendela (aturan frame sama dengan ekstraksi utama).
    video_id diberi akhiran _w<k>; jendela tengah (k = n // 2 bila n ganjil) sama dengan jendela ekstraksi utama."""
    hasil = []
    for v in sorted((v for v in daftar if v["split"] == "test"), key=lambda v: v["video_id"]):
        for k in range(int(n_jendela)):
            hasil.append({**v, "video_id": f"{v['video_id']}_w{k}", "jendela": (k, int(n_jendela))})
    return hasil


def encode_ulang(v, tmp_dir):
    """Encode ulang dengan libx264 (seperti unggahan ulang ke WhatsApp/TikTok); file sementara dihapus setelah
    wajahnya diekstrak. Audio dibuang."""
    import subprocess

    from common import ffmpeg_exe
    Path(tmp_dir).mkdir(parents=True, exist_ok=True)
    out = Path(tmp_dir) / f"{v['video_id']}.mp4"
    # skala < 1: resolusi diturunkan (seperti kompresi aplikasi pesan); ukuran genap untuk yuv420p
    vf = ["-vf", f"scale=trunc(iw*{v['skala']}/2)*2:trunc(ih*{v['skala']}/2)*2"] if v["skala"] < 1 else []
    r = subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", v["asal_video"], *vf, "-an", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", str(v["crf"]), "-pix_fmt", "yuv420p", str(out)],
                       capture_output=True, text=True)
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"ffmpeg gagal: {r.stderr.strip()[:200]}")
    return out


# ------------------------------------------------------------- utilitas wajah
class Deteksi:
    """Wajah hasil MTCNN (atribut mirip InsightFace supaya bisa dipakai ArcFace)."""
    def __init__(self, bbox, prob, kps):
        self.bbox, self.prob, self.kps = np.asarray(bbox, dtype=np.float32), float(prob), np.asarray(kps, np.float32)

    @property
    def lebar(self):
        return float(self.bbox[2] - self.bbox[0])


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    luas = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / luas if luas > 0 else 0.0


def ambil_frame(path, idxs):
    """Baca frame berindeks tertentu secara berurutan (lebih cepat & akurat daripada seek)."""
    idxs = sorted(set(idxs))
    cap = cv2.VideoCapture(str(path))
    hasil, i, k = {}, 0, 0
    while k < len(idxs):
        if not cap.grab():
            break
        if i == idxs[k]:
            ok, fr = cap.retrieve()
            if ok:
                hasil[i] = fr
            k += 1
        i += 1
    cap.release()
    return hasil


def jumlah_frame(path):
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w, h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return n, w, h


def fps_video(path):
    cap = cv2.VideoCapture(str(path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0)
    cap.release()
    return fps if 1 <= fps <= 240 else 25.0


def indeks_frame(a, b, fps, e):
    """Indeks frame dalam rentang [a, b): `merata` = tersebar sepanjang rentang; `berurutan` = satu jendela
    `frame_per_video` frame pada `fps_sampel` (25 fps) di tengah rentang."""
    k = int(e["frame_per_video"])
    n = b - a
    if n <= 0:
        return []
    if e.get("mode_frame", "merata") != "berurutan":
        return sorted(set(np.linspace(a, b - 1, k).round().astype(int).tolist()))
    langkah = max(1.0, fps / float(e.get("fps_sampel", 25)))
    rentang = langkah * (k - 1)
    if rentang >= n:  # video lebih pendek dari jendela: ambil semua yang ada
        return sorted(set(np.linspace(a, b - 1, min(k, n)).round().astype(int).tolist()))
    awal = a + (n - 1 - rentang) / 2
    return sorted(set(int(round(awal + i * langkah)) for i in range(k)))


class Pendeteksi:
    def __init__(self, e):
        from facenet_pytorch import MTCNN
        self.perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mtcnn = MTCNN(keep_all=True, select_largest=False, post_process=False, device=self.perangkat,
                           min_face_size=max(20, int(e["ukuran_wajah_min"] * 0.8)),
                           thresholds=list(e["ambang_mtcnn"]))
        self.e = e
        print(f"[MTCNN] berjalan di {'GPU (CUDA) ' + torch.cuda.get_device_name(0) if self.perangkat.type == 'cuda' else 'CPU'}",
              flush=True)

    def deteksi(self, frames_bgr):
        """List frame BGR (ukuran sama) -> list daftar Deteksi per frame."""
        keluar = []
        b = int(self.e["batch_frame"])
        for i in range(0, len(frames_bgr), b):
            rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames_bgr[i:i + b]]
            with torch.no_grad():
                boxes, probs, pts = self.mtcnn.detect(rgb, landmarks=True)
            for bx, pr, pt in zip(boxes, probs, pts):
                wajah = []
                if bx is not None:
                    for bb, p, k in zip(bx, pr, pt):
                        d = Deteksi(bb, p, k)
                        if p is not None and d.prob >= self.e["prob_min"] and d.lebar >= self.e["ukuran_wajah_min"]:
                            wajah.append(d)
                keluar.append(wajah)
        return keluar


class Identitas:
    """ArcFace (InsightFace buffalo_l) memakai landmark MTCNN untuk alignment.
    Dimuat saat pertama dibutuhkan (hanya untuk video fake)."""
    def __init__(self):
        self.rec = None

    def emb(self, img, d):
        if self.rec is None:
            from insightface.app.common import Face

            from common import penganalisis_wajah
            self.rec = penganalisis_wajah().models["recognition"]
            self.Face = Face
        f = self.Face(bbox=d.bbox, kps=d.kps, det_score=d.prob)
        v = self.rec.get(img, f)
        return v / (np.linalg.norm(v) + 1e-9)


def crop(img, d, margin, ukuran):
    x1, y1, x2, y2 = d.bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    s = max(x2 - x1, y2 - y1) * margin / 2
    l, t, r, b = int(round(cx - s)), int(round(cy - s)), int(round(cx + s)), int(round(cy + s))
    h, w = img.shape[:2]
    pad = max(0, -l, -t, r - w, b - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT)
        l, t, r, b = l + pad, t + pad, r + pad, b + pad
    potong = img[t:b, l:r]
    interp = cv2.INTER_AREA if potong.shape[0] >= ukuran else cv2.INTER_CUBIC
    return cv2.resize(potong, (ukuran, ukuran), interpolation=interp)


def label_fake(d, f_img, wajah_asli, a_img, jumlah, skala, e, idn):
    """1 = dimanipulasi, 0 = tidak tersentuh, None = tidak bisa dipastikan."""
    if wajah_asli is None:  # tak ada video asli pembanding
        return 1 if jumlah == 1 else None
    if not wajah_asli:
        return None
    sx, sy = skala
    kotak = [d.bbox[0] * sx, d.bbox[1] * sy, d.bbox[2] * sx, d.bbox[3] * sy]
    cocok = max(wajah_asli, key=lambda r: iou(kotak, r.bbox))
    if iou(kotak, cocok.bbox) < 0.3 and not (jumlah == 1 and len(wajah_asli) == 1):
        return None
    sim = float(np.dot(idn.emb(f_img, d), idn.emb(a_img, cocok)))
    return 1 if sim < e["ambang_identitas_berubah"] else 0


# ------------------------------------------------------------------ ekstraksi
def ekstrak(v, pend, idn, e, akar):
    n, w, h = jumlah_frame(v["file"])
    if n <= 0:
        return [], Counter({"video_gagal": 1})
    if v.get("jendela"):  # evaluasi multi-jendela: satu jendela di tengah bagian ke-k dari n bagian
        k_, n_ = v["jendela"]
        idxs = indeks_frame(int(round(n * k_ / n_)), int(round(n * (k_ + 1) / n_)), fps_video(v["file"]), e)
    else:
        idxs = indeks_frame(0, n, fps_video(v["file"]), e)
    frames = ambil_frame(v["file"], idxs)
    urut = [i for i in idxs if i in frames]
    if not urut:
        return [], Counter({"video_gagal": 1})
    det = dict(zip(urut, pend.deteksi([frames[i] for i in urut])))

    asli_det, asli_frm, skala = {}, {}, (1.0, 1.0)
    punya_pasangan = False
    lipsync = v["label"] == 1 and "lipsync" in str(v.get("metode", ""))
    # in the wild: tak ada video asli pembanding -> wajah terbesar (baku) atau semua wajah berlabel fake
    wf = v.get("wajah_fake") if v["label"] == 1 else None
    terbesar = lipsync or wf == "terbesar"
    if v["label"] == 1 and not lipsync and v["pasangan"] and Path(v["pasangan"]).exists():
        na, wa, ha = jumlah_frame(v["pasangan"])
        if na > 0:
            peta = {i: min(na - 1, round(i * na / n)) for i in urut}
            asli_frm = ambil_frame(v["pasangan"], list(peta.values()))
            ada = [i for i in urut if peta[i] in asli_frm]
            hasil_a = pend.deteksi([asli_frm[peta[i]] for i in ada]) if ada else []
            asli_det = {i: (asli_frm[peta[i]], d) for i, d in zip(ada, hasil_a)}
            punya_pasangan = True
            if w and h and wa and ha:
                skala = (wa / w, ha / h)

    folder_video = akar / ("fake" if v["label"] else "asli") / v["video_id"]
    folder_video.mkdir(parents=True, exist_ok=True)
    hash_video = sha256_file(v.get("asal_video") or v["file"])
    baris, stat = [], Counter()
    for i in urut:
        img = frames[i]
        wajah = det[i]
        for k, d in enumerate(wajah):
            if v["label"] == 0:
                lab = 0
            elif wf == "semua":
                lab = 1
            elif terbesar:  # lipsync / in the wild: wajah terbesar = wajah yang dimanipulasi
                lab = 1 if d is max(wajah, key=lambda x: x.lebar) else 0
            else:
                if punya_pasangan:
                    a_img, a_wajah = asli_det.get(i, (None, []))
                else:
                    a_img, a_wajah = None, None  # tanpa pembanding: hanya frame berwajah tunggal
                lab = label_fake(d, img, a_wajah, a_img, len(wajah), skala, e, idn)
            if lab is None:
                stat["dilewati"] += 1
                continue
            gambar = crop(img, d, e["margin"], e["ukuran"])
            # tepi hitam besar (wajah di pinggir frame video 2b / di luar frame) -> lewati,
            # supaya model tidak belajar "bidang hitam" sebagai ciri
            if e.get("maks_porsi_hitam") and float(np.mean(gambar.max(axis=2) <= 16)) > e["maks_porsi_hitam"]:
                stat["tepi_hitam"] += 1
                continue
            out = folder_video / f"{i:05d}_{k}_{'palsu' if lab else 'asli'}.{e['format']}"
            cv2.imwrite(str(out), gambar)
            stat["palsu" if lab else "asli"] += 1
            baris.append({"file": out.as_posix(), "sha256": sha256_file(out), "video_id": v["video_id"],
                          "sha256_video": hash_video, "frame": i, "wajah_ke": k, "label_wajah": lab,
                          "label_video": v["label"], "split": v["split"], "sumber": v["sumber"],
                          "metode": v["metode"], "identitas": v["identitas"],
                          "prob_deteksi": round(d.prob, 4), "lebar_wajah": round(d.lebar)})
        if not wajah:
            stat["frame_tanpa_wajah"] += 1
    return baris, stat


def parameter_hasil(e, c, nama):
    """Pengaturan yang memengaruhi isi hasil ekstraksi (batch_frame hanya soal kecepatan)."""
    p = {k: v for k, v in e.items() if k != "batch_frame"}
    if nama.endswith("_jendela"):
        from common import muat_config
        p["jendela_uji"] = int((muat_config().get("evaluasi_jendela") or {}).get("jumlah", 5))
    if nama.endswith("_h264"):
        from common import muat_config
        p["kompresi_ulang"] = {**BAKU_H264, **(muat_config().get("kompresi_ulang") or {})}
    if nama.startswith("celebdf"):
        p["celebdf"] = {k: c[k] for k in ("porsi_val", "rasio_fake_per_asli", "seed")}
    return json.loads(json.dumps(p))


def mulai_ulang(cfg, e, nama, alasan):
    """Hapus hasil ekstraksi lama satu sumber; metadata dicatat dihapus di log forensik."""
    import shutil

    from forensik import catat_hapus
    akar = folder(cfg, e["folder"])
    meta = akar / f"metadata_{nama}.csv"
    if meta.exists():
        catat_hapus(meta, alasan, cfg["folder_kerja"])
    n = sum(1 for _ in (akar / nama).rglob("*.png")) if (akar / nama).exists() else 0
    shutil.rmtree(akar / nama, ignore_errors=True)
    for f in (akar / f"selesai_{nama}.txt", akar / f"parameter_{nama}.json"):
        f.unlink(missing_ok=True)
    print(f"[{nama}] hasil lama dihapus ({n} gambar wajah). Alasan: {alasan}", flush=True)


def periksa_parameter(cfg, e, c, nama, paksa_ulang):
    """Cegah hasil lama dan baru tercampur bila pengaturan ekstraksi berubah."""
    akar = folder(cfg, e["folder"])
    p_path = akar / f"parameter_{nama}.json"
    sekarang = parameter_hasil(e, c, nama)
    ada_hasil = (akar / f"metadata_{nama}.csv").exists()
    if paksa_ulang and ada_hasil:
        mulai_ulang(cfg, e, nama, "mulai ulang ekstraksi atas permintaan pengguna")
    elif ada_hasil:
        lama = json.loads(p_path.read_text(encoding="utf-8")) if p_path.exists() else None
        if lama != sekarang:
            beda = sorted(k for k in set(sekarang) | set(lama or {}) if (lama or {}).get(k) != sekarang.get(k))
            raise SystemExit(
                f"[{nama}] Pengaturan ekstraksi berbeda dengan hasil yang sudah ada "
                f"({', '.join(beda) if lama else 'hasil lama tanpa catatan pengaturan'}). Supaya data tidak "
                "tercampur, jalankan ulang dengan pilihan 'Mulai ulang' (--mulai-ulang): hasil lama dihapus "
                "otomatis dan penghapusannya dicatat di log forensik.")
    p_path.write_text(json.dumps(sekarang, indent=2, ensure_ascii=False), encoding="utf-8")


def proses(cfg, e, daftar, nama, pend, idn, batas=None):
    akar_semua = folder(cfg, e["folder"])
    meta = akar_semua / f"metadata_{nama}.csv"
    selesai_path = akar_semua / f"selesai_{nama}.txt"
    selesai = set(selesai_path.read_text(encoding="utf-8").split("\n")) if selesai_path.exists() else set()
    tugas = [v for v in daftar if v["video_id"] not in selesai]
    if batas:
        # uji cepat: ambil bergiliran dari tiap (split, asli/fake) supaya label palsu ikut teruji
        kelompok = {}
        for v in tugas:
            kelompok.setdefault((v["split"], v["label"]), []).append(v)
        antrian, pilih = [kelompok[k] for k in sorted(kelompok, key=lambda k: (k[1], k[0]))], []
        while len(pilih) < batas and any(antrian):
            for q in antrian:
                if q and len(pilih) < batas:
                    pilih.append(q.pop(0))
        tugas = pilih
    print(f"\n== {nama}: {len(daftar)} video ({dict(Counter((v['split'], v['label']) for v in daftar))}); "
          f"{len(tugas)} belum diproses", flush=True)
    tulis_header = not meta.exists()
    total = Counter()
    t0 = time.time()
    with open(meta, "a", newline="", encoding="utf-8") as fm, open(selesai_path, "a", encoding="utf-8") as fs:
        w = csv.DictWriter(fm, fieldnames=KOLOM)
        if tulis_header:
            w.writeheader()
        for n, v in enumerate(tugas, 1):
            tmp = None
            try:
                if v.get("asal_video"):  # salinan kompresi ulang H.264
                    tmp = encode_ulang(v, akar_semua / "_tmp_h264")
                    v = {**v, "file": str(tmp)}
                baris, stat = ekstrak(v, pend, idn, e, akar_semua / nama / v["split"])
                if tmp:
                    for b_ in baris:
                        b_["sha256_video"] = f"kompresi ulang crf{v['crf']} skala {v['skala']} dari {b_['sha256_video'][:16]}"
            except Exception as err:  # noqa: BLE001
                print(f"  ! {v['video_id']}: {err}")
                total["video_gagal"] += 1
                continue
            finally:
                if tmp:
                    Path(tmp).unlink(missing_ok=True)
            w.writerows(baris)
            fm.flush()
            fs.write(v["video_id"] + "\n")
            fs.flush()
            total.update(stat)
            if n % 20 == 0 or n == len(tugas):
                laju = (time.time() - t0) / n
                print(f"  {nama}: {n}/{len(tugas)} video ({laju:.1f} dtk/video, sisa ~{laju * (len(tugas) - n) / 60:.0f} menit)",
                      flush=True)
    if meta.exists():
        catat("5_ekstrak_wajah", meta, folder_kerja=cfg["folder_kerja"], info={
            "sumber": nama, "detektor": "MTCNN (facenet-pytorch)", "wajah_asli_baru": total["asli"],
            "wajah_palsu_baru": total["palsu"], "dilewati": total["dilewati"],
            "dilewati_tepi_hitam": total["tepi_hitam"], "parameter": e,
            "catatan": "hash tiap crop wajah dan video asalnya ada di kolom sha256 metadata",
        })
    print(f"{nama}: +{total['asli']} wajah asli, +{total['palsu']} wajah palsu, {total['dilewati']} dilewati, "
          f"{total['tepi_hitam']} dilewati karena tepi hitam, {total['frame_tanpa_wajah']} frame tanpa wajah, "
          f"{total['video_gagal']} video gagal -> {meta}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sumber", nargs="+", choices=["indonesia", "celebdf", "wild"],
                    default=["indonesia", "celebdf"])
    ap.add_argument("--celebdf", help="folder Celeb-DF v2 (default: celebdf.folder di config.yaml)")
    ap.add_argument("--batas", type=int, help="proses paling banyak N video per sumber (uji)")
    ap.add_argument("--margin", type=float, help="perbesaran kotak wajah (baku config ekstraksi.margin, mis. 1.3 / 1.6)")
    ap.add_argument("--ukuran", type=int, help="ukuran crop persegi (baku config ekstraksi.ukuran, mis. 224 / 299)")
    ap.add_argument("--folder", help="folder hasil di data/ (baku config ekstraksi.folder); profil crop berbeda "
                                     "sebaiknya di folder berbeda supaya hasil lama tetap ada")
    ap.add_argument("--mulai-ulang", action="store_true",
                    help="hapus hasil ekstraksi lama sumber terpilih lalu mulai dari awal")
    ap.add_argument("--h264", action="store_true",
                    help="buat juga data latih kompresi ulang H.264 (video TRAIN di-encode ulang pada CRF acak, "
                         "lalu wajahnya diekstrak ke metadata_<sumber>_h264.csv); val/test tidak disentuh")
    ap.add_argument("--h264-saja", dest="h264_saja", action="store_true",
                    help="hanya buat data kompresi ulang H.264 (sumber aslinya sudah diekstrak)")
    ap.add_argument("--jendela-uji", dest="jendela_uji", action="store_true",
                    help="ekstrak juga beberapa jendela per video TEST (evaluasi_jendela.jumlah, baku 5) ke "
                         "metadata_<sumber>_jendela.csv untuk evaluasi multi-jendela; train/val tidak disentuh")
    ap.add_argument("--jendela-saja", dest="jendela_saja", action="store_true",
                    help="hanya ekstrak jendela tambahan video test (sumber utama sudah diekstrak)")
    args = ap.parse_args()

    cfg = muat_config()
    e = {**BAKU, **(cfg.get("ekstraksi") or {})}
    for k in ("margin", "ukuran", "folder"):
        if getattr(args, k) is not None:
            e[k] = getattr(args, k)
    print(f"Profil crop: margin {e['margin']}, ukuran {e['ukuran']}x{e['ukuran']}, folder data/{e['folder']}", flush=True)
    c = {**BAKU_CDF, **(cfg.get("celebdf") or {})}
    k264 = {**BAKU_H264, **(cfg.get("kompresi_ulang") or {})}
    buat_h264 = args.h264 or args.h264_saja
    buat_jendela = args.jendela_uji or args.jendela_saja
    n_jendela = int((cfg.get("evaluasi_jendela") or {}).get("jumlah", 5))
    hanya_tambahan = args.h264_saja or args.jendela_saja
    for nama in args.sumber:  # periksa dulu sebelum memuat model
        if buat_jendela:
            periksa_parameter(cfg, e, c, f"{nama}_jendela", args.mulai_ulang)
        if not hanya_tambahan:
            periksa_parameter(cfg, e, c, nama, args.mulai_ulang)
        if buat_h264:
            periksa_parameter(cfg, e, c, f"{nama}_h264", args.mulai_ulang)
    if buat_h264:
        print(f"Kompresi ulang H.264 data latih: CRF {k264['crf']}, skala resolusi {k264['skala']}, "
              f"{k264['salinan_per_video']} salinan per video", flush=True)
    pend = Pendeteksi(e)
    idn = Identitas()
    daftar = {}
    if "indonesia" in args.sumber:
        daftar["indonesia"] = daftar_indonesia(cfg)
    if "celebdf" in args.sumber:
        daftar["celebdf"] = daftar_celebdf(args.celebdf or c["folder"], c)
    if "wild" in args.sumber:
        daftar["wild"] = daftar_wild(cfg)
    if buat_jendela:
        print(f"Evaluasi multi-jendela: {n_jendela} jendela per video test", flush=True)
    for nama, d in daftar.items():
        if not hanya_tambahan:
            proses(cfg, e, d, nama, pend, idn, args.batas)
        if buat_jendela:
            proses(cfg, e, rencana_jendela(d, n_jendela), f"{nama}_jendela", pend, idn, args.batas)
        if buat_h264:
            proses(cfg, e, rencana_h264(d, k264, k264["seed"]), f"{nama}_h264", pend, idn, args.batas)


if __name__ == "__main__":
    main()
