"""Aplikasi Forensik Deepfake (desktop): Python + web view.

Backend: server HTTP lokal (hanya 127.0.0.1) yang memanggil forensik_video.py dan laporan_forensik.py.
Tampilan: ui_forensik/index.html di jendela desktop pywebview (WebView2/Edge di Windows). Bila pywebview
tidak tersedia, aplikasi dibuka di browser bawaan.

  python aplikasi_forensik.py              # jendela desktop
  python aplikasi_forensik.py --browser    # buka di browser
"""
import argparse
import json
import mimetypes
import os
import queue
import socket
import sys
import threading
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

os.chdir(Path(__file__).resolve().parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))

import bahasa as BH  # noqa: E402
import forensik_video as FV  # noqa: E402
import laporan_forensik as LF  # noqa: E402
from bahasa import T  # noqa: E402

JUDUL = "Forensik Deepfake - Xception + LSTM"
UI = Path("ui_forensik") / "index.html"
JENDELA = {"w": None}


# ============================================================ pekerjaan latar
class Pekerjaan:
    def __init__(self, jenis, judul, fungsi):
        self.id, self.jenis, self.judul, self.fungsi = uuid.uuid4().hex[:10], jenis, judul, fungsi
        self.status, self.progres, self.tahap, self.log, self.hasil, self.galat = "antri", 0.0, "", [], None, None
        self.mulai = self.selesai = None

    def tulis(self, pesan):
        self.log.append(f"{time.strftime('%H:%M:%S')}  {pesan}")
        print(pesan, flush=True)

    def maju(self, f, tahap=""):
        self.progres = float(max(0.0, min(1.0, f)))
        if tahap:
            self.tahap = tahap

    def ringkas(self, dari=0):
        return {"id": self.id, "jenis": self.jenis, "judul": self.judul, "status": self.status,
                "progres": self.progres, "tahap": self.tahap, "log": self.log[dari:], "jumlah_log": len(self.log),
                "hasil": self.hasil, "galat": self.galat, "mulai": self.mulai, "selesai": self.selesai}


PEKERJAAN, ANTRIAN = {}, queue.Queue()
SISTEM = {"torch": None, "gpu": None, "status": "memuat", "penganalisis": None}


def pekerja():
    while True:
        p = ANTRIAN.get()
        p.status, p.mulai = "berjalan", time.strftime("%H:%M:%S")
        try:
            p.hasil = p.fungsi(p)
            p.status = "selesai"
            p.maju(1.0, "selesai")
        except Exception as err:  # noqa: BLE001
            p.status, p.galat = "gagal", str(err)
            p.tulis(T("GAGAL: ", "FAILED: ") + str(err))
            print(traceback.format_exc(), flush=True)
        p.selesai = time.strftime("%H:%M:%S")


def antre(jenis, judul, fungsi):
    p = Pekerjaan(jenis, judul, fungsi)
    PEKERJAAN[p.id] = p
    ANTRIAN.put(p)
    return p


def penganalisis(p):
    if SISTEM["penganalisis"] is None:
        SISTEM["penganalisis"] = FV.Penganalisis(log=p.tulis)
    SISTEM["penganalisis"].log = p.tulis
    return SISTEM["penganalisis"]


def cek_sistem():
    try:
        import torch
        SISTEM["torch"] = torch.__version__
        SISTEM["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        SISTEM["status"] = "siap"
    except Exception as err:  # noqa: BLE001
        SISTEM["status"] = f"PyTorch: {err}"


# ================================================================ API
def pilih_file():
    jenis = "Video (*.mp4;*.avi;*.mov;*.mkv;*.webm;*.m4v;*.3gp;*.wmv;*.flv;*.mpg;*.mpeg;*.ts)"
    w = JENDELA["w"]
    if w is not None:
        import webview
        mode = getattr(getattr(webview, "FileDialog", None), "OPEN", None) or webview.OPEN_DIALOG
        hasil = w.create_file_dialog(mode, allow_multiple=True, file_types=(jenis, T("Semua file (*.*)", "All files (*.*)")))
        return list(hasil or [])
    import tkinter as tk
    from tkinter import filedialog
    akar = tk.Tk()
    akar.withdraw()
    akar.attributes("-topmost", True)
    hasil = filedialog.askopenfilenames(title=T("Pilih video barang bukti", "Select evidence video"),
                                        filetypes=[("Video", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v *.3gp *.wmv *.flv "
                                                             "*.mpg *.mpeg *.ts"), (T("Semua file", "All files"), "*.*")])
    akar.destroy()
    return list(hasil)


def buka_path(path):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    if os.name == "nt":
        os.startfile(str(p.resolve()))  # noqa: S606
    else:
        import webbrowser
        webbrowser.open(p.resolve().as_uri())
    return True


def api(nama, d):
    if nama == "bahasa":
        return {"bahasa": BH.atur(d.get("bahasa", "id"))}
    if nama == "info":
        return {"bahasa": BH.BAHASA, "kamus_data": BH.kamus_data(),
                "sistem": {k: v for k, v in SISTEM.items() if k != "penganalisis"}, "kasus": FV.daftar_kasus(),
                "model": FV.daftar_model(), "pengaturan": FV.pengaturan(), "pengguna": os.environ.get("USERNAME")
                or os.environ.get("USER", ""), "ekstensi": sorted(FV.EKSTENSI)}
    if nama == "kasus_baru":
        return FV.buat_kasus(d.get("nomor", ""), d.get("judul", ""), d.get("pemeriksa", ""), d.get("instansi", ""),
                             d.get("keterangan", ""))
    if nama == "kasus":
        return FV.muat_kasus(d["kasus"])
    if nama == "pilih_file":
        return pilih_file()
    if nama == "buka":
        return buka_path(d["path"])
    if nama == "tambah_bukti":
        files, kid, ket = [f for f in d.get("files", []) if str(f).strip()], d["kasus"], d.get("keterangan", "")
        if not files:
            raise ValueError(T("Pilih minimal satu file video", "Select at least one video file"))

        def kerja(p):
            hasil = []
            for i, f in enumerate(files):
                def maju(x, t="", i=i):
                    p.maju((i + x) / len(files), f"{Path(f).name}: {t}")
                b_ = FV.tambah_bukti(kid, f, ket, log=p.tulis, kemajuan=maju)
                if FV.perlu_pratinjau(b_):  # mis. Celeb-DF (mp4v) tidak bisa diputar WebView
                    try:
                        FV.buat_pratinjau(kid, b_["id"], log=p.tulis)
                    except Exception as err:  # noqa: BLE001
                        p.tulis(T("! pratinjau gagal dibuat: ", "! preview could not be created: ") + str(err))
                hasil.append(b_["id"])
            return {"bukti": hasil}
        return antre("akuisisi", T(f"Akuisisi {len(files)} file", f"Acquiring {len(files)} file(s)"), kerja).ringkas()
    if nama == "analisis":
        kid, bbs = d["kasus"], d["bukti"]
        models = d["model"] if isinstance(d["model"], list) else [d["model"]]
        if not bbs or not models:
            raise ValueError(T("Pilih minimal satu barang bukti dan satu model", "Select at least one evidence item and one model"))
        opsi = dict(d.get("opsi") or {})
        cara = d.get("cara") or "terpisah"
        if cara in ("maks", "rata") and len(models) >= 2:
            opsi["gabungan"] = cara
            tugas = [(bb, list(models)) for bb in bbs]
        else:
            tugas = [(bb, m) for m in models for bb in bbs]

        def kerja(p):
            pa = penganalisis(p)
            hasil = []
            for i, (bb, m) in enumerate(tugas):
                def maju(x, t="", i=i, bb=bb):
                    p.maju((i + x) / len(tugas), f"{bb} ({i + 1}/{len(tugas)}): {t}")
                nm = " + ".join(Path(x).name for x in m) if isinstance(m, list) else Path(m).name
                p.tulis(T(f"=== {bb} dengan model {nm} ===", f"=== {bb} with model {nm} ==="))
                hasil.append({"bukti": bb, "model": nm,
                              "status": FV.analisis_bukti(kid, bb, m, pa, opsi, log=p.tulis, kemajuan=maju)["status"]})
            return {"hasil": hasil}
        judul = T("Analisis ", "Analysis ") + ", ".join(bbs) + (
            T(f" (gabungan {len(models)} model)", f" (ensemble of {len(models)} models)") if isinstance(tugas[0][1], list) else
            T(f" x {len(models)} model", f" x {len(models)} models") if len(models) > 1 else "")
        return antre("analisis", judul, kerja).ringkas()
    if nama == "analisis_detail":
        return FV.muat_analisis(d["kasus"], d["bukti"], d["id"])
    if nama == "laporan":
        kid, bbs, mdl = d["kasus"], d.get("bukti") or None, d.get("model")
        return antre("laporan", T("Membuat laporan", "Creating report"), lambda p: LF.buat_laporan(kid, bbs, log=p.tulis, kemajuan=p.maju,
                                                                              model=mdl)).ringkas()
    if nama == "pratinjau":
        kid, bb = d["kasus"], d["bukti"]
        return antre("pratinjau", T(f"Video pratinjau {bb}", f"Preview video {bb}"),
                     lambda p: FV.buat_pratinjau(kid, bb, log=p.tulis, kemajuan=p.maju)).ringkas()
    if nama == "integritas":
        kid, bb = d["kasus"], d["bukti"]
        return antre("integritas", T(f"Verifikasi hash {bb}", f"Hash verification {bb}"),
                     lambda p: FV.periksa_integritas(kid, bb, log=p.tulis)).ringkas()
    if nama == "verifikasi":
        kid = d["kasus"]

        def kerja(p):
            p.tulis(T("Memverifikasi rantai log dan hash semua file kasus...", "Verifying the log chain and hashes of all case files..."))
            v = FV.verifikasi_kasus(kid)
            n_ok = sum(f['status'] == 'COCOK' for f in v['file'])
            p.tulis(T(f"Rantai {'UTUH' if v['rantai_utuh'] else 'TIDAK UTUH'}; {n_ok}/{len(v['file'])} file cocok",
                      f"Chain {'INTACT' if v['rantai_utuh'] else 'BROKEN'}; {n_ok}/{len(v['file'])} files match"))
            for m in v["masalah"]:
                p.tulis(f"! {BH.tr(m)}")
            return v
        return antre("verifikasi", T("Verifikasi kasus", "Case verification"), kerja).ringkas()
    if nama == "pekerjaan":
        p = PEKERJAAN.get(d.get("id"))
        if p is None:
            raise KeyError(T("pekerjaan tidak ditemukan", "job not found"))
        return p.ringkas(int(d.get("dari", 0)))
    if nama == "daftar_pekerjaan":
        return [p.ringkas(len(p.log)) for p in list(PEKERJAAN.values())[-20:]]
    raise KeyError(T(f"API tidak dikenal: {nama}", f"Unknown API: {nama}"))


IZIN_FILE = [Path("data/kasus_forensik").resolve(), Path("data/model").resolve(), Path("ui_forensik").resolve()]


class Penangan(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _kirim(self, kode, isi, jenis="application/json; charset=utf-8", header=None):
        if isinstance(isi, str):
            isi = isi.encode("utf-8")
        self.send_response(kode)
        self.send_header("Content-Type", jenis)
        self.send_header("Content-Length", str(len(isi)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (header or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(isi)

    def _host_aman(self):
        h = (self.headers.get("Host") or "").split(":")[0]
        return h in ("127.0.0.1", "localhost")

    def do_GET(self):
        if not self._host_aman():
            return self._kirim(403, "{}")
        u = urlparse(self.path)
        if u.path in ("/", "/index.html"):
            return self._kirim(200, UI.read_bytes(), "text/html; charset=utf-8")
        if u.path == "/file":
            return self._file(unquote(parse_qs(u.query).get("p", [""])[0]))
        self._kirim(404, "{}")

    do_HEAD = do_GET

    def _file(self, path):
        p = Path(path).resolve()
        if not any(p.is_relative_to(a) for a in IZIN_FILE) or not p.is_file():
            return self._kirim(404, "{}")
        jenis = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        ukuran = p.stat().st_size
        rentang = self.headers.get("Range")
        if rentang and rentang.startswith("bytes="):
            a, _, b = rentang[6:].partition("-")
            a = int(a or 0)
            b = min(int(b) if b else ukuran - 1, ukuran - 1, a + (8 << 20) - 1)
            with open(p, "rb") as f:
                f.seek(a)
                isi = f.read(b - a + 1)
            return self._kirim(206, isi, jenis, {"Content-Range": f"bytes {a}-{b}/{ukuran}", "Accept-Ranges": "bytes"})
        self._kirim(200, p.read_bytes(), jenis, {"Accept-Ranges": "bytes"})

    def do_POST(self):
        if not self._host_aman():
            return self._kirim(403, "{}")
        u = urlparse(self.path)
        if not u.path.startswith("/api/"):
            return self._kirim(404, "{}")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            d = json.loads(self.rfile.read(n) or b"{}")
            hasil = api(u.path[5:], d)
            self._kirim(200, json.dumps({"ok": True, "data": hasil}, ensure_ascii=False, default=str))
        except Exception as err:  # noqa: BLE001
            print(traceback.format_exc(), flush=True)
            self._kirim(200, json.dumps({"ok": False, "galat": str(err)}, ensure_ascii=False))


def port_bebas(awal=8601):
    for port in range(awal, awal + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(T("Tidak ada port bebas", "No free port"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--browser", action="store_true", help="buka di browser, bukan jendela desktop")
    ap.add_argument("--port", type=int)
    ap.add_argument("--tanpa-buka", action="store_true", help="jalankan server saja (untuk uji)")
    args = ap.parse_args()

    port = args.port or port_bebas()
    server = ThreadingHTTPServer(("127.0.0.1", port), Penangan)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    threading.Thread(target=pekerja, daemon=True).start()
    threading.Thread(target=cek_sistem, daemon=True).start()
    url = f"http://127.0.0.1:{port}/"
    print(T(f"{JUDUL} berjalan di {url}", f"{JUDUL} running at {url}"), flush=True)

    if args.tanpa_buka:
        threading.Event().wait()
    if not args.browser:
        try:
            import webview
            JENDELA["w"] = webview.create_window(JUDUL, url, width=1440, height=900, min_size=(1100, 700),
                                                 background_color="#1b1d21")
            webview.start()
            return
        except Exception as err:  # noqa: BLE001
            print(T(f"pywebview tidak bisa dipakai ({err}); membuka di browser.",
                    f"pywebview cannot be used ({err}); opening in the browser."), flush=True)
    import webbrowser
    webbrowser.open(url)
    print(T("Tutup jendela ini untuk menghentikan aplikasi.", "Close this window to stop the application."), flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
