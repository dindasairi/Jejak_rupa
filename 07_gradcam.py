"""Grad-CAM untuk model Xception + LSTM: bagian wajah mana yang paling memengaruhi skor fake.

Gradien skor video (keluaran LSTM) dirambatkan balik ke peta fitur konvolusi terakhir Xception di
SETIAP frame. Bobot kanal = rata-rata gradien; CAM = ReLU(jumlah bobot x peta fitur), dinormalisasi
bersama untuk seluruh frame satu video (supaya intensitas antarframe bisa dibandingkan), lalu
ditumpuk di atas wajah.

Contoh otomatis (dipanggil di akhir training): dari tiap data uji diambil video yang paling yakin
untuk kategori TP (fake terdeteksi), TN (asli benar), FP (asli dikira fake), FN (fake lolos).

  python 07_gradcam.py --model data/model/B_20261008_2300 --contoh            # buat ulang contoh
  python 07_gradcam.py --model data/model/B_20261008_2300 --video-id fake_0123 # satu video

Hasil: data/model/<model>/gradcam/<sumber>/<kategori>_<video_id>.png
"""
import argparse
import importlib.util
import json
from pathlib import Path

import torch  # noqa: I001
import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

KATEGORI = {"TP": "fake terdeteksi (benar)", "TN": "asli (benar)",
            "FP": "asli dikira fake (salah)", "FN": "fake lolos (salah)"}


def modul_latih():
    p = Path(__file__).with_name("06_latih_xception.py")
    spec = importlib.util.spec_from_file_location("latih_xception", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def gradcam_video(model, x, n, perangkat):
    """x: (T,3,H,W) tensor ter-normalisasi, n: jumlah frame valid. Kembalikan (skor, cam (n,H,W))."""
    model.eval()
    x = x[None].to(perangkat).requires_grad_(True)
    with torch.enable_grad(), torch.backends.cudnn.flags(enabled=False):  # LSTM backward di mode eval
        logit = model(x.float(), torch.tensor([n]), simpan_peta=True)
        model.zero_grad(set_to_none=True)
        logit.sum().backward()
    a = model.peta[:n].detach().float()
    g = model.peta.grad[:n].detach().float()
    w = g.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((w * a).sum(1))                                  # (n, h, w)
    cam = cam / (cam.max() + 1e-8)                                    # normalisasi bersama satu video
    cam = torch.nn.functional.interpolate(cam[:, None], size=x.shape[-2:], mode="bilinear",
                                          align_corners=False)[:, 0]
    model.peta = None
    return float(torch.sigmoid(logit).item()), cam.cpu().numpy()


def tumpuk(file, cam, ukuran):
    img = np.array(Image.open(file).convert("RGB").resize((ukuran, ukuran)))
    cam = cv2.resize(np.clip(cam, 0, 1).astype(np.float32), (ukuran, ukuran), interpolation=cv2.INTER_LINEAR)
    panas = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return img, (0.55 * img + 0.45 * panas).astype(np.uint8)


def huruf(ukuran):
    for nama in ("arial.ttf", "DejaVuSans.ttf", "segoeui.ttf"):
        try:
            return ImageFont.truetype(nama, ukuran)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=ukuran)
    except TypeError:
        return ImageFont.load_default()


def gambar_video(baris, skor, cam, keluar, judul, maks_kolom=10, sel=150):
    """Grid: baris atas wajah asli, baris bawah Grad-CAM, untuk maks_kolom frame yang tersebar."""
    files = baris["file"][:len(cam)]
    idx = np.linspace(0, len(files) - 1, min(maks_kolom, len(files))).round().astype(int)
    lebar, tinggi_judul = 10 + len(idx) * (sel + 4), 54
    kanvas = Image.new("RGB", (lebar, tinggi_judul + 2 * (sel + 4) + 24), "white")
    gb = ImageDraw.Draw(kanvas)
    f1, f2 = huruf(16), huruf(12)
    gb.text((8, 6), judul, font=f1, fill=(20, 20, 20))
    gb.text((8, 30), f"label: {'PALSU' if baris['label'] else 'ASLI'} · skor fake {skor:.3f} · metode {baris['metode']}"
            f" · {baris['panjang']} frame", font=f2, fill=(80, 80, 80))
    for j, k in enumerate(idx):
        asli, panas = tumpuk(files[k], cam[k], sel)
        x = 6 + j * (sel + 4)
        kanvas.paste(Image.fromarray(asli), (x, tinggi_judul))
        kanvas.paste(Image.fromarray(panas), (x, tinggi_judul + sel + 4))
        gb.text((x + 2, tinggi_judul + 2 * (sel + 4) + 4), f"frame {baris['frame'][k]}", font=f2, fill=(90, 90, 90))
    keluar.parent.mkdir(parents=True, exist_ok=True)
    kanvas.save(keluar)
    return keluar


def buat_untuk(model, baris, tf, perangkat, keluar, judul, panjang):
    import torch as _t
    frame = [tf(Image.open(f).convert("RGB")) for f in baris["file"]]
    n = len(frame)
    x = _t.stack(frame + [frame[-1]] * (panjang - n))
    skor, cam = gradcam_video(model, x, n, perangkat)
    return gambar_video(baris, skor, cam, keluar, judul)


def contoh_dari_prediksi(model, folder_model, uji, tf, perangkat, per_kategori=3, panjang=None):
    """Ambil video paling yakin per kategori (TP/TN/FP/FN) dari prediksi_<sumber>.csv."""
    folder_model = Path(folder_model)
    panjang = panjang or max(len(r) for t in uji.values() for r in t["file"])
    daftar = []
    for s, tabel in uji.items():
        pf = folder_model / f"prediksi_{s}.csv"
        if not pf.exists() or tabel.empty:
            continue
        pred = pd.read_csv(pf, dtype={"video_id": str})
        pred["kategori"] = np.where(pred["label"] == 1, np.where(pred["skor"] >= 0.5, "TP", "FN"),
                                    np.where(pred["skor"] >= 0.5, "FP", "TN"))
        per_id = {r["video_id"]: r for _, r in tabel.iterrows()}
        for kat in KATEGORI:
            g = pred[pred["kategori"] == kat]
            g = g.sort_values("skor", ascending=kat in ("TN", "FN")) if kat in ("TN", "FN") else \
                g.sort_values("skor", ascending=False)
            for _, r in g.head(per_kategori).iterrows():
                b = per_id.get(r["video_id"])
                if b is None:
                    continue
                out = folder_model / "gradcam" / s / f"{kat}_{r['video_id']}.png"
                buat_untuk(model, b, tf, perangkat, out, f"{s} · {kat}: {KATEGORI[kat]} · {r['video_id']}", panjang)
                daftar.append({"sumber": s, "kategori": kat, "video_id": r["video_id"], "skor": r["skor"],
                               "file": out.as_posix()})
    if daftar:
        pd.DataFrame(daftar).to_csv(folder_model / "gradcam" / "indeks.csv", index=False)
    print(f"Grad-CAM: {len(daftar)} gambar di {folder_model / 'gradcam'}", flush=True)
    return daftar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="folder model (berisi terbaik.pt)")
    ap.add_argument("--video-id", nargs="+", help="buat Grad-CAM untuk video tertentu")
    ap.add_argument("--contoh", action="store_true", help="buat ulang contoh TP/TN/FP/FN dari data uji")
    ap.add_argument("--per-kategori", type=int, default=3)
    args = ap.parse_args()

    from common import folder, muat_config
    L = modul_latih()
    cfg = muat_config()
    folder_model = Path(args.model)
    simpan = torch.load(folder_model / "terbaik.pt", map_location="cpu", weights_only=False)
    p = simpan["pengaturan"]
    perangkat = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = L.XceptionLSTM(p["model"], p["lstm_hidden"], p["lstm_lapis"], p["lstm_dua_arah"], p["dropout"],
                           pralatih=False).to(perangkat)
    model.load_state_dict(simpan["model"])
    tf = L.transformasi(p["augmentasi"], False, model.rata, model.simpangan)
    try:  # data wajah yang dipakai saat model dilatih (profil crop bisa berbeda dengan config sekarang)
        mj = json.loads((folder_model / "metrik.json").read_text(encoding="utf-8"))
        f_latih = next((x.get("folder") for x in (mj.get("ekstraksi") or {}).values() if x.get("folder")), None)
        if f_latih:
            cfg.setdefault("ekstraksi", {})["folder"] = f_latih
    except (OSError, ValueError):
        pass
    akar = folder(cfg, (cfg.get("ekstraksi") or {}).get("folder", "wajah_mtcnn"), buat=False)
    urutan = {}
    for s in getattr(L, "SUMBER", ("indonesia", "celebdf")):
        if (akar / f"metadata_{s}.csv").exists():
            urutan[s], _ = L.bangun_urutan(L.muat_metadata(cfg, s), p["panjang_urutan"], p["min_frame"])
    if args.contoh:
        uji = {s: u[u["split"] == "test"] for s, u in urutan.items()}
        contoh_dari_prediksi(model, folder_model, uji, tf, perangkat, args.per_kategori, p["panjang_urutan"])
    for vid in args.video_id or []:
        for s, u in urutan.items():
            b = u[u["video_id"] == vid]
            if len(b):
                out = buat_untuk(model, b.iloc[0], tf, perangkat, folder_model / "gradcam" / s / f"pilihan_{vid}.png",
                                 f"{s} · {vid}", p["panjang_urutan"])
                print(f"Grad-CAM {vid}: {out}")
                break
        else:
            print(f"! video {vid} tidak ditemukan di hasil ekstraksi")


if __name__ == "__main__":
    main()
