"""Pencatatan bukti digital (chain of custody) untuk seluruh pipeline.

Setiap file yang masuk atau dihasilkan pipeline dicatat di data/log_forensik.jsonl:
siapa, kapan (UTC dan WIB), dari mana (URL / file induk beserta hash-nya),
proses apa dengan parameter apa, versi alat, ukuran, dan SHA-256 file.

Log bersifat append-only dan berantai: setiap entri menyimpan hash entri
sebelumnya, sehingga entri yang diubah atau dihapus akan terdeteksi saat verifikasi.

Perintah:
  python forensik.py verifikasi            cek rantai log + hash semua file
  python forensik.py catat-fake --alat "SimSwap (commit abc123)"
                                           catat video fake yang sudah dibuat
  python forensik.py hapus data/asli/x/klip.mp4 --alasan "wajah salah orang"
                                           hapus file sambil mencatat hash dan alasannya
  python forensik.py rapikan --alasan "dihapus manual lewat File Explorer"
                                           catat semua file HILANG sebagai dihapus
  python forensik.py laporan               buat laporan HTML + CSV di data/laporan/
"""
import argparse
import csv
import getpass
import hashlib
import html
import json
import platform
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

WIB = timezone(timedelta(hours=7), "WIB")
_CACHE_HASH = {}
_VERSI = None


# ------------------------------------------------------------------ dasar
def sha256_file(path, pakai_cache=True):
    path = Path(path)
    kunci = (str(path.resolve()), path.stat().st_mtime_ns, path.stat().st_size)
    if pakai_cache and kunci in _CACHE_HASH:
        return _CACHE_HASH[kunci]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blok in iter(lambda: f.read(1 << 20), b""):
            h.update(blok)
    _CACHE_HASH[kunci] = h.hexdigest()
    return _CACHE_HASH[kunci]


def sekarang():
    t = datetime.now(timezone.utc)
    return t.isoformat(timespec="seconds"), t.astimezone(WIB).strftime("%Y-%m-%d %H:%M:%S WIB")


def versi_alat():
    global _VERSI
    if _VERSI is None:
        _VERSI = {"python": platform.python_version(), "os": platform.platform()}
        try:
            import yt_dlp
            _VERSI["yt-dlp"] = yt_dlp.version.__version__
        except Exception:
            pass
        try:
            import insightface
            _VERSI["insightface"] = insightface.__version__
        except Exception:
            pass
        try:
            import onnxruntime
            _VERSI["onnxruntime"] = onnxruntime.__version__
        except Exception:
            pass
        try:
            import cv2
            _VERSI["opencv"] = cv2.__version__
        except Exception:
            pass
        try:
            from common import ffmpeg_exe
            out = subprocess.run([ffmpeg_exe(), "-version"], capture_output=True, text=True).stdout
            _VERSI["ffmpeg"] = out.split("\n")[0].split(" ")[2]
        except Exception:
            pass
    try:  # perangkat inferensi baru diketahui setelah model wajah dimuat
        import common
        if common.PERANGKAT_AKTIF:
            _VERSI["inferensi_wajah"] = common.PERANGKAT_AKTIF
    except ImportError:
        pass
    return _VERSI


def _path_log(folder_kerja="data"):
    p = Path(folder_kerja) / "log_forensik.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _hash_entri(entri):
    isi = {k: v for k, v in entri.items() if k != "hash_entri"}
    return hashlib.sha256(json.dumps(isi, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def baca_log(folder_kerja="data"):
    p = _path_log(folder_kerja)
    if not p.exists():
        return []
    return [json.loads(b) for b in p.read_text(encoding="utf-8").splitlines() if b.strip()]


def catat(langkah, file, induk=(), info=None, sumber_url=None, folder_kerja="data"):
    """Tambahkan satu entri ke log. `induk` = daftar path file asal."""
    log = baca_log(folder_kerja)
    utc, wib = sekarang()
    file = Path(file)
    entri = {
        "no": len(log) + 1,
        "waktu_utc": utc,
        "waktu_wib": wib,
        "langkah": langkah,
        "file": file.as_posix(),
        "sha256": sha256_file(file),
        "ukuran_byte": file.stat().st_size,
        "sumber_url": sumber_url,
        "induk": [{"file": Path(p).as_posix(), "sha256": sha256_file(p)} for p in induk if p],
        "info": info or {},
        "operator": getpass.getuser(),
        "komputer": platform.node(),
        "versi_alat": versi_alat(),
        "hash_entri_sebelumnya": log[-1]["hash_entri"] if log else "0" * 64,
    }
    entri["hash_entri"] = _hash_entri(entri)
    with open(_path_log(folder_kerja), "a", encoding="utf-8") as f:
        f.write(json.dumps(entri, ensure_ascii=False) + "\n")
    return entri


def entri_terakhir_per_file(folder_kerja="data"):
    return {e["file"]: e for e in baca_log(folder_kerja)}


# ------------------------------------------------------------- verifikasi
def verifikasi(folder_kerja="data"):
    """Kembalikan (rantai_utuh, daftar_masalah, hasil_per_file)."""
    log = baca_log(folder_kerja)
    masalah = []
    sebelumnya = "0" * 64
    for e in log:
        if e.get("hash_entri_sebelumnya") != sebelumnya:
            masalah.append(f"Entri #{e.get('no')}: rantai terputus (ada entri dihapus/disisipkan)")
        if _hash_entri(e) != e.get("hash_entri"):
            masalah.append(f"Entri #{e.get('no')}: isi entri berubah setelah dicatat")
        sebelumnya = e.get("hash_entri")
    rantai_utuh = not masalah

    hasil = []
    for f, e in entri_terakhir_per_file(folder_kerja).items():
        p = Path(f)
        if e["langkah"] == "hapus" and not p.exists():
            status = "DIHAPUS"  # dihapus dengan sengaja dan tercatat
        elif not p.exists():
            status = "HILANG"
        elif sha256_file(p, pakai_cache=False) == e["sha256"]:
            status = "COCOK"
        else:
            status = "BERUBAH"
        hasil.append({"file": f, "langkah": e["langkah"], "sha256_tercatat": e["sha256"],
                      "status": status})
        if status not in ("COCOK", "DIHAPUS"):
            masalah.append(f"{f}: {status}")
    return rantai_utuh, masalah, hasil


# ----------------------------------------------------------- video fake
def catat_fake(alat, folder_kerja="data", hanya=None):
    """Catat video fake yang ada di rencana tapi belum tercatat di log.
    `hanya` = daftar fake_id; kalau diisi, hanya fake tersebut yang dicatat."""
    rencana = Path(folder_kerja) / "rencana_fake.csv"
    if not rencana.exists():
        return 0
    sudah = entri_terakhir_per_file(folder_kerja)
    n = 0
    with open(rencana, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            out = Path(r["file_keluaran"])
            if hanya is not None and r["fake_id"] not in hanya:
                continue
            if not out.exists():
                continue
            lama = sudah.get(out.as_posix())
            if lama and lama["langkah"] != "hapus":
                # sudah tercatat. Hanya unggahan eksplisit (hanya=...) dengan isi baru yang
                # dicatat ulang; pencatatan massal tidak boleh "memutihkan" file yang berubah.
                if hanya is None or sha256_file(out, pakai_cache=False) == lama["sha256"]:
                    continue
            induk = [r["video_target"]]
            if Path(r["contoh_klip_wajah"]).exists():
                induk.append(r["contoh_klip_wajah"])
            catat("4_buat_fake", out, induk=induk, folder_kerja=folder_kerja, info={
                "fake_id": r["fake_id"], "metode": r["metode"], "alat": alat,
                "split": r["split"], "tokoh_target": r["tokoh_target"],
                "tokoh_wajah": r["tokoh_wajah"],
            })
            n += 1
    return n


# ---------------------------------------------------------------- hapus
CSV_DAFTAR = ("unduhan.csv", "klip_asli.csv", "klip_wajah.csv")


def catat_hapus(file, alasan, folder_kerja="data"):
    """Hapus satu file hasil pipeline dengan tercatat: hash, alasan, dan operator disimpan
    di log sebelum file dihapus, lalu barisnya dibuang dari unduhan.csv / klip_asli.csv /
    klip_wajah.csv (video mentah yang dihapus jadi bisa diunduh ulang).

    Kalau file sudah tidak ada (misalnya terhapus manual) tapi tercatat di log, penghapusannya
    tetap dicatat memakai hash terakhir yang tercatat, dengan keterangan bahwa file sudah hilang.
    Kembalikan entri log."""
    p = Path(str(file).strip().strip('"').replace("\\", "/"))
    if not alasan.strip():
        raise ValueError("Alasan penghapusan wajib diisi")
    lama = entri_terakhir_per_file(folder_kerja).get(p.as_posix())
    if not p.exists():
        if lama is None:
            raise FileNotFoundError(f"{p} tidak ada dan tidak tercatat di log")
        if lama["langkah"] == "hapus":
            raise FileNotFoundError(f"{p} sudah tercatat dihapus (entri #{lama['no']})")
    baris_dibuang = {}
    for nama in CSV_DAFTAR:
        c = Path(folder_kerja) / nama
        if not c.exists():
            continue
        with open(c, encoding="utf-8", newline="") as f:
            rd = csv.DictReader(f)
            kolom, baris = rd.fieldnames, list(rd)
        sisa = [r for r in baris if Path(r.get("file", "").replace("\\", "/")).as_posix() != p.as_posix()]
        if len(sisa) != len(baris):
            baris_dibuang[nama] = [r for r in baris if r not in sisa]
            with open(c, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=kolom)
                w.writeheader()
                w.writerows(sisa)
    info = {"alasan": alasan.strip(), "baris_csv_dibuang": baris_dibuang}
    if p.exists():
        e = catat("hapus", p, folder_kerja=folder_kerja, info=info)
        p.unlink()  # baris CSV yang dibuang tersimpan di info entri ini
        return e
    # file sudah hilang sebelum dicatat: pakai hash terakhir yang tercatat
    info["keterangan"] = "file sudah tidak ada saat penghapusan dicatat; hash dari entri #%d" % lama["no"]
    log = baca_log(folder_kerja)
    utc, wib = sekarang()
    entri = {
        "no": len(log) + 1, "waktu_utc": utc, "waktu_wib": wib, "langkah": "hapus",
        "file": p.as_posix(), "sha256": lama["sha256"], "ukuran_byte": lama["ukuran_byte"],
        "sumber_url": lama.get("sumber_url"), "induk": [], "info": info,
        "operator": getpass.getuser(), "komputer": platform.node(), "versi_alat": versi_alat(),
        "hash_entri_sebelumnya": log[-1]["hash_entri"] if log else "0" * 64,
    }
    entri["hash_entri"] = _hash_entri(entri)
    with open(_path_log(folder_kerja), "a", encoding="utf-8") as f:
        f.write(json.dumps(entri, ensure_ascii=False) + "\n")
    return entri


def rapikan_hilang(alasan, folder_kerja="data"):
    """Catat semua file berstatus HILANG (dihapus di luar pipeline, misalnya lewat File Explorer)
    sebagai dihapus, dengan hash terakhir yang tercatat, dan buang barisnya dari CSV.
    Kembalikan daftar file yang dicatat."""
    hasil = []
    for f, e in entri_terakhir_per_file(folder_kerja).items():
        if e["langkah"] != "hapus" and not Path(f).exists():
            catat_hapus(f, alasan, folder_kerja)
            hasil.append(f)
    return hasil


# --------------------------------------------------------------- laporan
def tanggal_akses(e):
    """Tanggal akses: yang dinyatakan pengguna (video manual) atau saat diunduh otomatis."""
    return (e["info"].get("tanggal_akses_dinyatakan") or e["info"].get("waktu_akses_asli")
            or e["waktu_wib"])


def rantai_asal(file, indeks, log=None):
    """Telusuri dari file akhir ke sumber asalnya (URL). Kalau `log` diberikan, entri unduhan
    asli sebelum remux ikut ditampilkan."""
    per_no = {e["no"]: e for e in log} if log else {}
    jalur, saat_ini, dilihat = [], file, set()
    while saat_ini in indeks and saat_ini not in dilihat:
        dilihat.add(saat_ini)
        e = indeks[saat_ini]
        jalur.append(e)
        asal = per_no.get(e["info"].get("entri_unduh_asal"))
        if asal:
            jalur.append(asal)
        if not e["induk"]:
            break
        saat_ini = e["induk"][0]["file"]
    return jalur


def buat_laporan(folder_kerja="data", judul="Laporan Forensik Digital Dataset Deepfake Wajah Indonesia"):
    out_dir = Path(folder_kerja) / "laporan"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = baca_log(folder_kerja)
    indeks = entri_terakhir_per_file(folder_kerja)
    rantai_utuh, masalah, hasil = verifikasi(folder_kerja)
    utc, wib = sekarang()
    stempel = datetime.now(WIB).strftime("%Y%m%d_%H%M%S")

    # CSV: satu baris per file
    csv_path = out_dir / f"daftar_bukti_{stempel}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["no", "langkah", "file", "sha256", "ukuran_byte", "waktu_wib", "tanggal_akses",
                    "sumber_url",
                    "judul", "kanal", "tanggal_unggah", "induk_file", "induk_sha256", "operator",
                    "status_verifikasi"])
        status = {h["file"]: h["status"] for h in hasil}
        for e in indeks.values():
            w.writerow([e["no"], e["langkah"], e["file"], e["sha256"], e["ukuran_byte"],
                        e["waktu_wib"], tanggal_akses(e) if e.get("sumber_url") else "",
                        e.get("sumber_url") or "", e["info"].get("judul", ""),
                        e["info"].get("kanal", ""), e["info"].get("tanggal_unggah", ""),
                        ";".join(i["file"] for i in e["induk"]),
                        ";".join(i["sha256"] for i in e["induk"]), e["operator"],
                        status.get(e["file"], "")])

    esc = lambda x: html.escape(str(x if x is not None else ""))
    sumber = [e for e in indeks.values() if e["langkah"] in ("1_unduh", "1_tambah_manual", "1_remux")]
    jumlah = {}
    for e in indeks.values():
        jumlah[e["langkah"]] = jumlah.get(e["langkah"], 0) + 1

    baris_sumber = "".join(
        f"<tr><td>{esc(e['info'].get('judul'))}<br><a href='{esc(e.get('sumber_url'))}'>"
        f"{esc(e.get('sumber_url'))}</a></td><td>{esc(e['info'].get('kanal'))}</td>"
        f"<td>{esc(e['info'].get('tanggal_unggah'))}</td><td>{esc(tanggal_akses(e))}</td>"
        f"<td class='h'>{esc(e['sha256'])}</td></tr>" for e in sumber)

    akhir = [e for e in indeks.values() if e["langkah"] in ("2b_pisah_wajah", "4_buat_fake")]
    blok_rantai = []
    for e in akhir:
        langkah_html = "".join(
            f"<li><b>{esc(x['langkah'])}</b> — {esc(x['file'])}<br>"
            f"<span class='h'>SHA-256 {esc(x['sha256'])}</span><br>"
            f"{esc(x['waktu_wib'])}{(' — ' + esc(x['sumber_url'])) if x.get('sumber_url') else ''}"
            f"</li>" for x in rantai_asal(e["file"], indeks, log))
        blok_rantai.append(f"<details><summary>{esc(e['file'])}</summary><ol>{langkah_html}</ol></details>")

    baris_verif = "".join(
        f"<tr class='{h['status'].lower()}'><td>{esc(h['file'])}</td><td>{esc(h['langkah'])}</td>"
        f"<td>{esc(h['status'])}</td></tr>" for h in hasil if h["status"] != "COCOK")
    foto_ref = [e for e in indeks.values() if e["langkah"] == "0_foto_referensi"]
    baris_foto = "".join(
        f"<tr><td>{esc(e['info'].get('tokoh_id'))}</td><td>{esc(Path(e['file']).name)}</td>"
        f"<td>{esc(e['info'].get('kanal'))}<br><a href='{esc(e.get('sumber_url'))}'>{esc(e.get('sumber_url'))}</a></td>"
        f"<td>{esc(e['info'].get('lisensi'))}</td><td>{esc(e['info'].get('pembuat'))}</td>"
        f"<td>{esc(e['waktu_wib'])}</td><td class='h'>{esc(e['sha256'])}</td></tr>" for e in foto_ref)
    dihapus = [e for e in log if e["langkah"] == "hapus"]
    baris_hapus = "".join(
        f"<tr><td>{esc(e['file'])}</td><td>{esc(e['waktu_wib'])}</td><td>{esc(e['operator'])}</td>"
        f"<td>{esc(e['info'].get('alasan'))}</td><td class='h'>{esc(e['sha256'])}</td></tr>" for e in dihapus)
    log_path = _path_log(folder_kerja)
    hash_log = sha256_file(log_path, pakai_cache=False) if log_path.exists() else "-"
    versi = log[-1]["versi_alat"] if log else versi_alat()

    isi = f"""<!doctype html><html lang="id"><head><meta charset="utf-8">
<title>{esc(judul)}</title><style>
body{{font-family:"Source Serif 4",Georgia,serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#1d2433;line-height:1.5}}
h1{{font-size:1.6rem}} h2{{font-size:1.2rem;margin-top:2rem;border-bottom:2px solid #1F3A5F}}
table{{border-collapse:collapse;width:100%;font-size:.85rem}} td,th{{border:1px solid #c9ced8;padding:.35rem;vertical-align:top;text-align:left}}
th{{background:#e9edf3}} .h{{font-family:Consolas,monospace;font-size:.75rem;word-break:break-all}}
.ok{{color:#1b6b3a;font-weight:bold}} .gagal{{color:#a1261b;font-weight:bold}} .hilang td,.berubah td{{background:#fbe9e7}}
details{{margin:.3rem 0}} summary{{cursor:pointer}} @media print{{details{{display:block}}}}
</style></head><body>
<h1>{esc(judul)}</h1>
<p>Dibuat: {esc(wib)} ({esc(utc)} UTC) oleh {esc(getpass.getuser())} di {esc(platform.node())}</p>

<h2>Ringkasan integritas</h2>
<p>Rantai log: <span class="{'ok' if rantai_utuh else 'gagal'}">{'UTUH' if rantai_utuh else 'TIDAK UTUH'}</span><br>
File diperiksa: {len(hasil)} — cocok {sum(h['status']=='COCOK' for h in hasil)},
berubah {sum(h['status']=='BERUBAH' for h in hasil)}, hilang {sum(h['status']=='HILANG' for h in hasil)},
dihapus dengan tercatat {sum(h['status']=='DIHAPUS' for h in hasil)}<br>
Jumlah entri log: {len(log)}<br>SHA-256 file log saat laporan dibuat: <span class="h">{esc(hash_log)}</span></p>
{'<table><tr><th>File</th><th>Langkah</th><th>Status</th></tr>' + baris_verif + '</table>' if baris_verif else ''}
{''.join(f'<p class="gagal">{esc(m)}</p>' for m in masalah if m.startswith('Entri'))}

<h2>Jumlah item per tahap</h2>
<table><tr><th>Tahap</th><th>Jumlah file</th></tr>
{''.join(f'<tr><td>{esc(k)}</td><td>{v}</td></tr>' for k, v in sorted(jumlah.items()))}</table>

<h2>Sumber bukti asal ({len(sumber)} video)</h2>
<table><tr><th>Judul dan URL</th><th>Kanal</th><th>Tanggal unggah</th><th>Tanggal akses</th><th>SHA-256</th></tr>
{baris_sumber}</table>

<h2>Rantai asal setiap sampel ({len(akhir)})</h2>
<p>Dari file sampel akhir ditelusuri mundur sampai ke URL sumber.</p>
{''.join(blok_rantai)}

<h2>Foto referensi tokoh ({len(foto_ref)})</h2>
<table><tr><th>Tokoh</th><th>File</th><th>Sumber</th><th>Lisensi</th><th>Pembuat</th><th>Diambil</th><th>SHA-256</th></tr>
{baris_foto}</table>

<h2>File yang dihapus dengan sengaja ({len(dihapus)})</h2>
<table><tr><th>File</th><th>Waktu</th><th>Operator</th><th>Alasan</th><th>SHA-256 saat dihapus</th></tr>
{baris_hapus}</table>

<h2>Lingkungan dan alat</h2>
<table>{''.join(f'<tr><td>{esc(k)}</td><td>{esc(v)}</td></tr>' for k, v in versi.items())}</table>
<p>Daftar lengkap per file: {esc(csv_path.name)}. Log mentah: log_forensik.jsonl.</p>
</body></html>"""
    html_path = out_dir / f"laporan_forensik_{stempel}.html"
    html_path.write_text(isi, encoding="utf-8")
    return html_path, csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="perintah", required=True)
    sub.add_parser("verifikasi")
    cf = sub.add_parser("catat-fake")
    cf.add_argument("--alat", required=True, help="nama + versi tool pembuat fake")
    rp = sub.add_parser("rapikan", help="catat semua file HILANG sebagai dihapus")
    rp.add_argument("--alasan", required=True)
    hp = sub.add_parser("hapus")
    hp.add_argument("file", nargs="+")
    hp.add_argument("--alasan", required=True)
    sub.add_parser("laporan")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from common import muat_config
    fk = muat_config()["folder_kerja"]

    if args.perintah == "verifikasi":
        utuh, masalah, hasil = verifikasi(fk)
        print(f"Rantai log: {'UTUH' if utuh else 'TIDAK UTUH'}; {len(hasil)} file diperiksa")
        for m in masalah:
            print("  !", m)
        if not masalah:
            print("Semua file cocok dengan hash yang tercatat.")
    elif args.perintah == "catat-fake":
        print(f"{catat_fake(args.alat, fk)} video fake baru dicatat")
    elif args.perintah == "rapikan":
        daftar = rapikan_hilang(args.alasan, fk)
        print(f"{len(daftar)} file HILANG dicatat sebagai dihapus dan barisnya dibuang dari CSV.")
    elif args.perintah == "hapus":
        for f in args.file:
            try:
                e = catat_hapus(f, args.alasan, fk)
            except (FileNotFoundError, ValueError) as err:
                print(f"  ! {err}")
                continue
            print(f"Dihapus dan tercatat: {f} (SHA-256 {e['sha256'][:16]}...)")
    else:
        h, c = buat_laporan(fk)
        print(f"Laporan: {h}\nDaftar bukti: {c}")


if __name__ == "__main__":
    main()
