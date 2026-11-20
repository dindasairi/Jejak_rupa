"""Laporan pemeriksaan forensik (HTML mandiri + PDF) untuk aplikasi forensik deepfake.

HTML berisi semua gambar (base64) sehingga bisa dibuka tanpa folder kasus. PDF dicetak dari HTML yang
sama memakai Microsoft Edge / Google Chrome mode headless (tidak perlu pustaka tambahan).
Kedua file dicatat di log berantai kasus.
"""
import base64
import html
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import forensik
from forensik import catat, sekarang
from bahasa import T, kes, tr
from forensik_video import _jam, folder_kasus, muat_analisis, muat_kasus, verifikasi_kasus

ATURAN = {"merata": "20 frame merata sepanjang segmen (aturan lama)", "berurutan": "20 frame berurutan 25 fps (aturan baru)"}
WARNA = {"TERINDIKASI": "#c62828", "RAGU": "#e08a00", "TIDAK": "#2e7d32", "TIDAK_ADA_WAJAH": "#666"}
WARNA_ORANG = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]
SUMBER = {"indonesia": ("Indonesia", "Indonesia"), "celebdf": ("Celeb-DF", "Celeb-DF"), "wild": ("in the wild", "in the wild"),
          "keseluruhan": ("keseluruhan (Indonesia + Celeb-DF)", "overall (Indonesia + Celeb-DF)")}
KEGIATAN = {"kasus_dibuat": ("Kasus dibuat", "Case created"),
            "salinan_file_asli": ("Salinan file asli (original files)", "Original file copy (original files)"),
            "akuisisi_bukti": ("Akuisisi barang bukti (salinan kerja)", "Evidence acquisition (working copy)"),
            "metadata_bukti": ("Pencatatan metadata bukti", "Evidence metadata recorded"),
            "pratinjau_pemutaran": ("Video pratinjau untuk pemutaran", "Preview video for playback"),
            "analisis_deepfake": ("Analisis deepfake", "Deepfake analysis"),
            "manifest_analisis": ("Manifest hasil analisis", "Analysis result manifest"),
            "laporan_html": ("Laporan HTML", "HTML report"), "laporan_pdf": ("Laporan PDF", "PDF report")}


def _sumber(k):
    v = SUMBER.get(k)
    return T(*v) if v else k


def _kegiatan(k):
    v = KEGIATAN.get(k)
    return T(*v) if v else k


def _aturan(k, cadangan=None):
    v = ATURAN.get(k)
    return tr(v) if v else cadangan


def e(x):
    return html.escape(str(x if x is not None else "-"))


def gambar(path, lebar="100%"):
    p = Path(path) if path else None
    if not p or not p.exists():
        return ""
    jenis = "jpeg" if p.suffix.lower() in (".jpg", ".jpeg") else "png"
    data = base64.b64encode(p.read_bytes()).decode()
    return f'<img src="data:image/{jenis};base64,{data}" style="width:{lebar}">'


def lencana(status):
    return (f'<span class="lencana" style="background:{WARNA.get(status, "#666")}">'
            f'{e(kes(status)).upper()}</span>')


def ukuran(n):
    n = float(n or 0)
    for s in ("B", "KB", "MB", "GB"):
        if n < 1024 or s == "GB":
            return f"{n:.0f} {s}" if s == "B" else f"{n:.2f} {s}"
        n /= 1024


def svg_garis_waktu(a, lebar=700, tinggi=210):
    """Skor fake per segmen per orang terhadap waktu, dengan garis ambang."""
    rs = [r for r in a["hasil"] if r["skor"] is not None]
    if not rs:
        return "<p><i>" + T("Tidak ada segmen dengan wajah yang cukup.", "No segment has enough faces.") + "</i></p>"
    durasi = max(s["selesai_detik"] for s in a["segmen"]) or 1
    kiri, kanan, atas, bawah = 44, 12, 12, 30
    W, H = lebar - kiri - kanan, tinggi - atas - bawah

    def x(t):
        return kiri + W * t / durasi

    def y(v):
        return atas + H * (1 - v)
    out = [f'<svg viewBox="0 0 {lebar} {tinggi}" width="100%" xmlns="http://www.w3.org/2000/svg" '
           'font-family="Arial" font-size="10">',
           f'<rect x="{kiri}" y="{atas}" width="{W}" height="{H}" fill="#fafafa" stroke="#bbb"/>']
    for v in (0, 0.25, 0.5, 0.75, 1):
        out.append(f'<line x1="{kiri}" x2="{kiri + W}" y1="{y(v)}" y2="{y(v)}" stroke="#e3e3e3"/>'
                   f'<text x="{kiri - 6}" y="{y(v) + 3}" text-anchor="end" fill="#555">{v:.2f}</text>')
    langkah = max(1, round(durasi / 8))
    t = 0
    while t <= durasi:
        out.append(f'<text x="{x(t)}" y="{tinggi - 12}" text-anchor="middle" fill="#555">{_jam(t)}</text>')
        t += langkah
    amb, rd = a["ambang"], a.get("ambang_rendah")
    out.append(f'<line x1="{kiri}" x2="{kiri + W}" y1="{y(amb)}" y2="{y(amb)}" stroke="#c62828" '
               f'stroke-dasharray="5,4"/><text x="{kiri + W - 4}" y="{y(amb) - 4}" text-anchor="end" '
               f'fill="#c62828">{T("batas toleransi", "tolerance limit") if rd is not None else T("ambang", "threshold")} {amb:.3f}</text>')
    if rd is not None:
        out.append(f'<line x1="{kiri}" x2="{kiri + W}" y1="{y(rd)}" y2="{y(rd)}" stroke="#e08a00" '
                   f'stroke-dasharray="2,3"/><text x="{kiri + W - 4}" y="{y(rd) + 11}" text-anchor="end" '
                   f'fill="#e08a00">{T("ambang Youden", "Youden threshold")} {rd:.3f}</text>')
    orang = [o["orang"] for o in a["ringkasan_orang"]]
    for k, o in enumerate(orang):
        c = WARNA_ORANG[k % len(WARNA_ORANG)]
        pts = sorted((r["mulai_detik"], r["selesai_detik"], r["skor"]) for r in rs if r["orang"] == o)
        gc_o = {(g["segmen"]) for g in (a.get("gradcam") or []) if g["orang"] == o}
        seg_o = {(r["mulai_detik"], r["selesai_detik"]): r["segmen"] for r in rs if r["orang"] == o}
        if len(pts) > 1:
            out.append('<polyline fill="none" stroke="%s" stroke-width="1.5" points="%s"/>' % (
                c, " ".join(f"{x((m + s) / 2):.1f},{y(v):.1f}" for m, s, v in pts)))
        for m, s, v in pts:
            out.append(f'<line x1="{x(m):.1f}" x2="{x(s):.1f}" y1="{y(v):.1f}" y2="{y(v):.1f}" stroke="{c}" '
                       f'stroke-width="3" opacity="0.45"/><circle cx="{x((m + s) / 2):.1f}" cy="{y(v):.1f}" r="3" fill="{c}"/>')
            no = seg_o.get((m, s))
            if no in gc_o:  # titik yang punya gambar Grad-CAM: lingkaran + label S<n> = judul gambar di bawah
                out.append(f'<circle cx="{x((m + s) / 2):.1f}" cy="{y(v):.1f}" r="6.5" fill="none" stroke="#111" '
                           f'stroke-width="1.2"/><text x="{x((m + s) / 2):.1f}" y="{y(v) - 9:.1f}" text-anchor="middle" '
                           f'font-size="9" fill="#111">S{no}</text>')
        out.append(f'<rect x="{kiri + 8 + k * 60}" y="{atas + 6}" width="10" height="10" fill="{c}"/>'
                   f'<text x="{kiri + 22 + k * 60}" y="{atas + 15}" fill="#333">{e(o)}</text>')
    out.append(f'<text x="12" y="{atas + H / 2}" transform="rotate(-90 12 {atas + H / 2})" text-anchor="middle" '
               f'fill="#555">{T("skor fake", "fake score")}</text></svg>')
    return "".join(out)


CSS = """
@page { size: A4; margin: 16mm 14mm 18mm 14mm; }
body { font-family: "Segoe UI", Arial, sans-serif; color: #1b1b1b; font-size: 10.5pt; line-height: 1.45;
       max-width: 900px; margin: 0 auto; padding: 16px; background: #fff; }
h1 { font-size: 17pt; margin: 0; } h2 { font-size: 13pt; border-bottom: 2px solid #1f3a5f; padding-bottom: 3px;
     margin-top: 26px; color: #1f3a5f; } h3 { font-size: 11.5pt; margin: 18px 0 6px; }
table { border-collapse: collapse; width: 100%; margin: 6px 0 10px; font-size: 9.5pt; }
th, td { border: 1px solid #c9c9c9; padding: 4px 6px; text-align: left; vertical-align: top; }
th { background: #eef2f7; width: 28%; } table.data th { width: auto; }
.kop { border-bottom: 3px double #1f3a5f; padding-bottom: 8px; margin-bottom: 10px; display: flex;
       justify-content: space-between; align-items: flex-end; gap: 12px; }
.kop small { color: #555; } .mono { font-family: Consolas, "Courier New", monospace; font-size: 8.5pt;
       word-break: break-all; }
.lencana { color: #fff; padding: 2px 8px; border-radius: 3px; font-weight: 700; font-size: 9pt; white-space: nowrap; }
.kotak { border: 1px solid #c9c9c9; border-left: 5px solid #1f3a5f; padding: 8px 12px; background: #f7f9fc;
         margin: 8px 0; } .peringatan { border-left-color: #e08a00; background: #fff8ec; }
.ok { color: #2e7d32; font-weight: 700; } .salah { color: #c62828; font-weight: 700; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; } figure { margin: 6px 0 12px; }
figcaption { font-size: 8.5pt; color: #555; } .putus { page-break-before: always; }
.ttd { margin-top: 30px; display: flex; justify-content: flex-end; } .ttd div { width: 260px; text-align: center; }
pre { white-space: pre-wrap; font-size: 8pt; background: #f4f4f4; padding: 6px; border: 1px solid #ddd; }
@media print { body { padding: 0; } h2, h3 { page-break-after: avoid; } figure, table { page-break-inside: avoid; } }
"""


PERINGKAT = {"TERINDIKASI": 3, "RAGU": 2, "TIDAK": 1, "TIDAK_ADA_WAJAH": 0}


def _daftar(item, maks=6, satuan=None):
    """'a', 'a dan b', 'a, b, dan c' (English: 'a and b', 'a, b, and c'); sisanya diringkas 'dan n <satuan> lain'."""
    satuan = satuan or T("segmen", "segments")
    item = list(item)
    if len(item) > maks:
        item = item[:maks] + [T(f"{len(item) - maks} {satuan} lain", f"{len(item) - maks} other {satuan}")]
    dan = T("dan", "and")
    if len(item) <= 2:
        return f" {dan} ".join(item)
    return ", ".join(item[:-1]) + f", {dan} " + item[-1]


def _seg(no, seg_w):
    sg = seg_w.get(no, {})
    return f'S{no} ({_jam(sg.get("mulai_detik", 0))}-{_jam(sg.get("selesai_detik", 0))})'


def narasi_bukti(b, a):
    """Paragraf temuan dan penjelasan keputusan, disusun dari angka hasil analisis (tanpa tambahan informasi lain)."""
    seg_w = {sg["no"]: sg for sg in a["segmen"]}
    amb, rd = a["ambang"], a.get("ambang_rendah")
    zona = float((a.get("pengaturan") or {}).get("zona_ragu", 0.1))
    m = b["metadata"]
    orang = sorted(a["ringkasan_orang"], key=lambda o: (-PERINGKAT.get(o["status"], 0), o["orang"]))
    dianalisis = [r for r in a["hasil"] if r["skor"] is not None]
    seg_wajah = sorted({r["segmen"] for r in dianalisis})
    nama_bawah = T("ambang Youden", "the Youden threshold") if rd is not None else T("ambang", "the threshold")
    nama_atas = T("batas toleransi", "the tolerance limit") if rd is not None else T("ambang", "the threshold")

    # ---- temuan
    t = [T(f'Video berdurasi {_jam(m["durasi_detik"])} dibagi menjadi {a["jumlah_segmen"]} segmen. ',
           f'The {_jam(m["durasi_detik"])} video was divided into {a["jumlah_segmen"]} segments. ')
         + ((T(f'Wajah yang cukup untuk dianalisis ditemukan di {len(seg_wajah)} segmen, milik {len(orang)} orang ',
               f'Faces sufficient for analysis were found in {len(seg_wajah)} segments, belonging to {len(orang)} '
               f'{"person" if len(orang) == 1 else "people"} ')
             + f'({_daftar(o["orang"] for o in orang)}). ') if orang else
            T('Tidak ada segmen dengan wajah yang cukup untuk dianalisis. ', 'No segment has enough faces for analysis. '))]
    for o in orang:
        rs = sorted((r for r in dianalisis if r["orang"] == o["orang"]), key=lambda r: r["segmen"])
        bawah = rd if rd is not None else amb
        lewat = [r for r in rs if r["skor"] >= bawah]
        atas = [r for r in rs if r["skor"] >= amb]
        k = T(f'{o["orang"]} muncul di {o["segmen_dianalisis"]} segmen dengan skor rata-rata {o["skor_rata"]:.3f} '
              f'dan skor tertinggi {o["skor_maks"]:.3f} di {_seg(o["segmen_tertinggi"], seg_w)}. ',
              f'{o["orang"]} appears in {o["segmen_dianalisis"]} segments with a mean score of {o["skor_rata"]:.3f} '
              f'and a highest score of {o["skor_maks"]:.3f} in {_seg(o["segmen_tertinggi"], seg_w)}. ')
        if lewat:
            k += T(f'Segmen yang mencapai {nama_bawah} {bawah:.3f} ada {len(lewat)} dari {len(rs)}, yaitu '
                   f'{_daftar(_seg(r["segmen"], seg_w) for r in lewat)}. ',
                   f'{len(lewat)} of {len(rs)} segments reach {nama_bawah} of {bawah:.3f}: '
                   f'{_daftar(_seg(r["segmen"], seg_w) for r in lewat)}. ')
            if rd is not None:
                k += (T(f'Dari segmen tersebut, {len(atas)} juga mencapai batas toleransi {amb:.3f}',
                        f'Of these, {len(atas)} also reach the tolerance limit of {amb:.3f}')
                      + (f' ({_daftar(_seg(r["segmen"], seg_w) for r in atas)}). ' if atas else '. '))
        else:
            k += T(f'Tidak ada segmennya yang mencapai {nama_bawah} {bawah:.3f}. ',
                   f'None of its segments reach {nama_bawah} of {bawah:.3f}. ')
        g_ = [g for g in a.get("gradcam") or [] if g["orang"] == o["orang"] and g.get("frame_detail")
              and g.get("skor_segmen", g["skor"]) >= (rd if rd is not None else amb)]
        if g_:
            puncak = max(g_, key=lambda g: g.get("skor_segmen", g["skor"]))
            d = max(puncak["frame_detail"], key=lambda d: d["bobot"])
            k += T(f'Pada Grad-CAM segmen S{puncak["segmen"]}, frame yang paling diperhatikan model berada di '
                   f'{d.get("waktu") or _jam(d["detik"])} (frame {d["frame"]}). ',
                   f'In the Grad-CAM of segment S{puncak["segmen"]}, the frame the model attended to most is at '
                   f'{d.get("waktu") or _jam(d["detik"])} (frame {d["frame"]}). ')
        dp = [r for r in rs if r.get("dipisah")]
        if dp:
            sebab = f' {T("dan", "and")} '.join(sorted({tr(x) for r in dp for x in r["dipisah"]}))
            k += T(f'Urutan wajah {o["orang"]} di {_daftar(_seg(r["segmen"], seg_w) for r in dp)} dipotong karena '
                   f'{sebab}, jadi model hanya membaca bagian yang berisi satu orang dalam satu adegan. ',
                   f'The face sequence of {o["orang"]} in {_daftar(_seg(r["segmen"], seg_w) for r in dp)} was split '
                   f'because of {sebab}, so the model only reads parts containing one person in one scene. ')
        t.append(k)
    tinggi = [x for x in m.get("temuan") or [] if x.get("tingkat") == "tinggi"]
    if m.get("temuan"):
        t.append(T(f'Pemeriksaan metadata mencatat {len(m["temuan"])} temuan pendukung',
                   f'The metadata examination recorded {len(m["temuan"])} supporting findings')
                 + (T(f', {len(tinggi)} di antaranya ditandai penting', f', {len(tinggi)} of them marked important')
                    if tinggi else '')
                 + T('. Temuan metadata tidak dipakai untuk menentukan kesimpulan. ',
                     '. Metadata findings are not used to determine the conclusion. '))
    h_akuisisi = b["salinan"]["hash"]["sha256"]
    t.append(T('SHA-256 salinan kerja saat analisis identik dengan hash saat akuisisi, jadi file yang dianalisis sama '
               'dengan file yang diakuisisi.',
               'The SHA-256 of the working copy at analysis is identical to the hash at acquisition, so the analyzed '
               'file is the acquired file.') if a["sha256_bukti"] == h_akuisisi else
             T('SHA-256 salinan kerja saat analisis BERBEDA dari hash saat akuisisi. Hasil analisis ini tidak dapat '
               'dipakai sebelum perbedaan tersebut dijelaskan.',
               'The SHA-256 of the working copy at analysis DIFFERS from the hash at acquisition. This analysis cannot '
               'be used until the difference is explained.'))

    # ---- penjelasan keputusan
    if rd is None:
        p = [T(f'Ambang keputusan untuk analisis ini adalah {amb:.3f} ({e(tr(a["jenis_ambang"]))}). ',
               f'The decision threshold for this analysis is {amb:.3f} ({e(tr(a["jenis_ambang"]))}). ')]
    else:
        p = [T(f'Analisis ini memakai dua batas dari data validasi model: ambang Youden {rd:.3f} '
               f'({e(tr(a.get("jenis_ambang_rendah")))}) dan batas toleransi {amb:.3f} ({e(tr(a["jenis_ambang"]))}). ',
               f'This analysis uses two limits from the model validation data: the Youden threshold of {rd:.3f} '
               f'({e(tr(a.get("jenis_ambang_rendah")))}) and the tolerance limit of {amb:.3f} ({e(tr(a["jenis_ambang"]))}). ')]
    for o in orang:
        r_, mx, st = o["skor_rata"], o["skor_maks"], o["status"]
        if st == "TERINDIKASI":
            p.append(T(f'Skor rata-rata {o["orang"]} ({r_:.3f}) mencapai {nama_atas} {amb:.3f}, sehingga {o["orang"]} '
                       f'dinyatakan <i>{kes(st)}</i>. ',
                       f'The mean score of {o["orang"]} ({r_:.3f}) reaches {nama_atas} of {amb:.3f}, so {o["orang"]} '
                       f'is marked <i>{kes(st)}</i>. '))
        elif st == "RAGU":
            if rd is not None and r_ >= rd:
                alasan = T(f'mencapai ambang Youden {rd:.3f} tetapi di bawah batas toleransi {amb:.3f}',
                           f'reaches the Youden threshold of {rd:.3f} but is below the tolerance limit of {amb:.3f}')
            elif mx >= amb:
                alasan = T(f'di bawah {nama_bawah} {(rd if rd is not None else amb):.3f}, tetapi ada '
                           f'{o["segmen_melewati_ambang"]} segmen yang mencapai {nama_atas} {amb:.3f} dengan skor '
                           f'tertinggi {mx:.3f} di S{o["segmen_tertinggi"]}',
                           f'is below {nama_bawah} of {(rd if rd is not None else amb):.3f}, but '
                           f'{o["segmen_melewati_ambang"]} segments reach {nama_atas} of {amb:.3f}, with the highest '
                           f'score of {mx:.3f} in S{o["segmen_tertinggi"]}')
            else:
                alasan = T(f'di bawah ambang {amb:.3f} dengan selisih hanya {amb - r_:.3f}, lebih kecil dari batas zona '
                           f'ragu {min(zona, amb / 2):.3f}',
                           f'is below the threshold of {amb:.3f} by only {amb - r_:.3f}, less than the uncertainty '
                           f'zone of {min(zona, amb / 2):.3f}')
            p.append(T(f'Skor rata-rata {o["orang"]} ({r_:.3f}) {alasan}, sehingga {o["orang"]} dinyatakan '
                       f'<i>{kes(st)}</i>. ',
                       f'The mean score of {o["orang"]} ({r_:.3f}) {alasan}, so {o["orang"]} is marked '
                       f'<i>{kes(st)}</i>. '))
        elif st == "TIDAK":
            nilai = rd if rd is not None else amb
            p.append(T(f'Skor rata-rata {o["orang"]} ({r_:.3f}) dan skor segmen tertingginya ({mx:.3f}) berada di bawah '
                       f'{nama_bawah} {nilai:.3f}, sehingga {o["orang"]} dinyatakan <i>{kes(st)}</i>. ',
                       f'The mean score of {o["orang"]} ({r_:.3f}) and its highest segment score ({mx:.3f}) are below '
                       f'{nama_bawah} of {nilai:.3f}, so {o["orang"]} is marked <i>{kes(st)}</i>. '))
    if orang:
        berat = orang[0]
        sama = [o["orang"] for o in orang if o["status"] == berat["status"]]
        p.append(T(f'Kesimpulan barang bukti mengikuti kesimpulan orang yang paling berat, yaitu {_daftar(sama)}, '
                   f'sehingga barang bukti ini dinyatakan <b>{e(kes(a["status"]))}</b>.',
                   f'The evidence verdict follows the most severe verdict among the people, namely {_daftar(sama)}, '
                   f'so this evidence item is marked <b>{e(kes(a["status"]))}</b>.'))
        if a["status"] in ("TERINDIKASI", "RAGU"):
            p.append(T(' Segmen yang disebut di atas adalah bagian video yang perlu diperiksa lebih dulu secara visual '
                       'dan dengan pemeriksaan lain sebelum kesimpulan dipakai.',
                       ' The segments listed above are the parts of the video to examine first, visually and with other '
                       'methods, before the verdict is used.'))
    else:
        p.append(T('Karena tidak ada wajah yang cukup, model tidak menghasilkan skor dan barang bukti ini dinyatakan ',
                   'Because there are not enough faces, the model produced no score and this evidence item is marked ')
                 + f'<b>{e(kes(a["status"]))}</b>.')
    return (f'<h3>{T("Temuan", "Findings")}</h3><p>' + "".join(t) + f'</p><h3>{T("Penjelasan keputusan", "Decision explanation")}'
            '</h3><p>' + "".join(p) + "</p>")


def _tabel(baris):
    return "<table>" + "".join(f"<tr><th>{e(k)}</th><td>{v}</td></tr>" for k, v in baris) + "</table>"


def bagian_bukti(b, nomor):
    a = b.get("analisis")
    m = b["metadata"]
    h1, h2 = b["asal"]["hash"], b["salinan"]["hash"]
    identik, beda = T("identik", "identical"), T("BERBEDA", "DIFFERENT")
    s = [f'<h2 class="putus">{nomor}. {T("Barang bukti", "Evidence")} {e(b["id"])}: {e(b["nama_file"])}</h2>',
         f'<h3>{T("Akuisisi dan hash", "Acquisition and hashes")}</h3>',
         _tabel([(T("Nama file", "File name"), e(b["nama_file"])), (T("Keterangan", "Notes"), e(b.get("keterangan") or "-")),
                 (T("Lokasi asal", "Source location"), f'<span class="mono">{e(b["asal"]["path"])}</span>'),
                 (T("Ukuran", "Size"), f'{ukuran(b["asal"]["ukuran_byte"])} ({b["asal"]["ukuran_byte"]:,} byte)'),
                 (T("Waktu ubah / buat (sistem file asal)", "Modified / created time (source file system)"),
                  f'{e(b["asal"]["diubah"])} / {e(b["asal"]["dibuat"])}'),
                 (T("Waktu akuisisi", "Acquisition time"),
                  f'{e(b["akuisisi_mulai_wib"])} {T("s.d.", "to")} {e(b["akuisisi_selesai_wib"])}'),
                 (T("Operator / komputer", "Operator / computer"), f'{e(b["operator"])} / {e(b["komputer"])}'),
                 (T("Salinan kerja (read-only)", "Working copy (read-only)"), f'<span class="mono">{e(b["salinan"]["file"])}</span>')]
                + ([(T("Salinan file asli (original files, read-only)", "Original file copy (original files, read-only)"),
                     f'<span class="mono">{e(b["file_asli"]["file"])}</span> (SHA-256 '
                     + (T("identik dengan file asal", "identical to the source file") if b["file_asli"]["hash"] == h1
                        else f"<b class=salah>{beda}</b>") + ')')]
                   if b.get("file_asli") else [])),
         f'<table class="data"><tr><th>{T("Algoritma", "Algorithm")}</th><th>{T("File asal", "Source file")}</th>'
         f'<th>{T("Salinan kerja", "Working copy")}</th><th>Status</th></tr>']
    for k in ("md5", "sha1", "sha256"):
        ok = h1[k] == h2[k]
        s.append(f'<tr><td>{ {"md5": "MD5", "sha1": "SHA-1", "sha256": "SHA-256"}[k]}</td><td class="mono">{h1[k]}</td><td class="mono">{h2[k]}</td>'
                 f'<td class="{"ok" if ok else "salah"}">{identik if ok else beda}</td></tr>')
    s.append("</table>")
    s += [f'<h3>{T("Metadata video", "Video metadata")}</h3>',
          _tabel([(T("Resolusi", "Resolution"), f'{m["lebar"]} x {m["tinggi"]} {T("piksel", "pixels")}'),
                  ("Frame rate", f'{m["fps"]} fps'),
                  (T("Jumlah frame", "Frame count"), f'{m["jumlah_frame"]:,}'),
                  (T("Durasi", "Duration"), f'{_jam(m["durasi_detik"])} ({m["durasi_detik"]} {T("dtk", "s")})'),
                  (T("Kodek video", "Video codec"), e(m["kodek_video"])),
                  ("Audio", e(m["audio"] or T("tidak ada", "none"))),
                  ("Bitrate", f'{m["bitrate_kbps"]} kb/s' if m.get("bitrate_kbps") else "-")]
                 + [(f"Tag: {k}", e(v)) for k, v in list(m["tag"].items())[:15]])]
    if m["temuan"]:
        s.append(f'<div class="kotak peringatan"><b>{T("Temuan metadata", "Metadata findings")}</b> '
                 f'{T("(pendukung, bukan penentu kesimpulan)", "(supporting, not decisive)")}:<ul>'
                 + "".join(f'<li>{"<b>[" + T("penting", "important") + "]</b> " if t["tingkat"] == "tinggi" else ""}{e(tr(t["teks"]))}</li>'
                           for t in m["temuan"]) + "</ul></div>")
    if not a:
        s.append(f'<div class="kotak peringatan">{T("Barang bukti ini belum dianalisis.", "This evidence item has not been analyzed.")}</div>')
        return "".join(s)
    mi = a["model"]
    ambang_teks = (T(f'ambang Youden {a["ambang_rendah"]:.3f} ({e(tr(a.get("jenis_ambang_rendah")))}) dan batas toleransi '
                     f'{a["ambang"]:.3f} ({e(tr(a["jenis_ambang"]))})',
                     f'Youden threshold {a["ambang_rendah"]:.3f} ({e(tr(a.get("jenis_ambang_rendah")))}) and tolerance limit '
                     f'{a["ambang"]:.3f} ({e(tr(a["jenis_ambang"]))})') if a.get("ambang_rendah") is not None
                   else f'{a["ambang"]:.3f} ({e(tr(a["jenis_ambang"]))})')
    frame_teks = (T(f'{a["jumlah_segmen"]} segmen, masing-masing ', f'{a["jumlah_segmen"]} segments, each with ')
                  + _aturan(a.get("ekstraksi", {}).get("mode_frame"), f'{a["pengaturan"]["frame_per_segmen"]} frame')
                  + ("" if a.get("aturan_frame", {}).get("sesuai_model", True)
                     else '. <b class="salah">' + T("Aturan ini berbeda dengan saat model dilatih.",
                                                    "This rule differs from the one used in training.") + '</b>'))
    s += [f'<h3>{T("Hasil analisis deteksi deepfake", "Deepfake detection result")}</h3>',
          f'<div class="kotak">{T("Kesimpulan", "Conclusion")}: {lencana(a["status"])}</div>',
          _tabel([(T("Waktu analisis", "Analysis time"), e(a["waktu_wib"])),
                  ("Model", f'<b>{e(mi["nama"])}</b>, {e(mi["arsitektur"])}, {T("skenario", "scenario")} {e(mi["skenario"])} '
                            f'{e(tr(mi.get("nama_skenario")))}'),
                  (T("Ambang keputusan", "Decision thresholds"), ambang_teks),
                  (T("Segmen dan frame", "Segments and frames"), frame_teks),
                  (T("Durasi proses", "Processing time"), T(f'{a["durasi_proses_detik"]} detik di {e(mi["perangkat"])}',
                                                             f'{a["durasi_proses_detik"]} seconds on {e(mi["perangkat"])}')),
                  (T("SHA-256 salinan kerja saat analisis", "Working copy SHA-256 at analysis"),
                   f'<span class="mono">{e(a["sha256_bukti"])}</span><br>'
                   + ('<span class="ok">' + T("identik dengan hash saat akuisisi", "identical to the hash at acquisition")
                      + '</span>' if a["sha256_bukti"] == h2["sha256"]
                      else '<span class="salah">' + T("BERBEDA dari hash saat akuisisi", "DIFFERENT from the hash at acquisition")
                      + '</span>'))]),
          narasi_bukti(b, a)]
    ang = mi.get("anggota") or []
    if a.get("gabungan") and ang:
        dua = a.get("ambang_rendah") is not None
        maks = a["gabungan"] == "maks"
        s.append('<div class="kotak"><b>' + T(f'Analisis gabungan dari {len(ang)} model', f'Ensemble analysis of {len(ang)} models')
                 + '</b>, ' + T("digabung dengan ", "combined by ")
                 + (T("skor tertinggi", "highest score") if maks else T("rata-rata", "mean")) + '. '
                 + T('Wajah dideteksi dan dilacak sekali, lalu setiap model membaca crop wajah sesuai profilnya sendiri. '
                     'Sebaran skor setiap model berbeda, jadi skor mentah dipetakan dulu ke skala terpadu berdasarkan '
                     'ambang model itu sendiri, baru digabung. ',
                     'Faces are detected and tracked once, then each model reads the face crop with its own profile. '
                     'Each model has a different score distribution, so raw scores are first mapped to a unified scale '
                     'based on that model\'s own thresholds, then combined. ')
                 + (T('Di bawah ambang Youden menjadi 0 sampai 0,33, antara ambang Youden dan batas toleransi menjadi 0,33 '
                      'sampai 0,67, dan di atas batas toleransi menjadi 0,67 sampai 1. Skor gabungan 0,67 atau lebih berarti '
                      'minimal satu model melewati batas toleransinya sendiri (pada penggabungan skor tertinggi).',
                      'Below the Youden threshold maps to 0 to 0.33, between the Youden threshold and the tolerance limit '
                      'maps to 0.33 to 0.67, and above the tolerance limit maps to 0.67 to 1. A combined score of 0.67 or '
                      'more means at least one model passed its own tolerance limit (with highest-score combination).')
                    if dua else
                    T('Di bawah ambang menjadi 0 sampai 0,5 dan di atas ambang menjadi 0,5 sampai 1. Skor gabungan 0,5 atau '
                      'lebih berarti minimal satu model melewati ambangnya sendiri (pada penggabungan skor tertinggi).',
                      'Below the threshold maps to 0 to 0.5 and above it maps to 0.5 to 1. A combined score of 0.5 or more '
                      'means at least one model passed its own threshold (with highest-score combination).'))
                 + f'<table class="data"><tr><th>{T("Model anggota", "Member model")}</th><th>Crop</th>'
                 + (f'<th>{T("Batas toleransi", "Tolerance limit")}</th><th>{T("Ambang Youden", "Youden threshold")}</th>'
                    if dua else f'<th>{T("Ambang", "Threshold")}</th>')
                 + f'<th>{T("AUC data uji keseluruhan", "Overall test AUC")}</th></tr>'
                 + "".join(_baris_anggota(x, dua) for x in ang) + "</table></div>")
    if a["ringkasan_orang"]:
        dua = a.get("ambang_rendah") is not None
        s.append(f'<table class="data"><tr><th>{T("Orang", "Person")}</th><th>{T("Segmen dianalisis", "Segments analyzed")}</th>'
                 f'<th>{T("Skor rata-rata", "Mean score")}</th><th>Median</th><th>{T("Maks (segmen)", "Max (segment)")}</th>'
                 f'<th>{T("Segmen", "Segments")} &ge; '
                 + (T('batas toleransi', 'tolerance limit') if dua else T('ambang', 'threshold')) + '</th>'
                 + (f'<th>{T("Segmen", "Segments")} &ge; {T("ambang Youden", "Youden threshold")}</th>' if dua else "")
                 + ("".join(f"<th>{T('Skor mentah rata', 'Raw mean score')} {e(x['nama'])}</th>" for x in ang)
                    if a.get("gabungan") else "")
                 + f'<th>{T("Kesimpulan", "Conclusion")}</th></tr>')
        for o in a["ringkasan_orang"]:
            s.append(f'<tr><td><b>{e(o["orang"])}</b></td><td>{o["segmen_dianalisis"]}</td><td>{o["skor_rata"]:.3f}</td>'
                     f'<td>{o["skor_median"]:.3f}</td><td>{o["skor_maks"]:.3f} (S{o["segmen_tertinggi"]})</td>'
                     f'<td>{o["segmen_melewati_ambang"]} ({o["porsi_melewati_ambang"] * 100:.0f}%)</td>'
                     + (f'<td>{o.get("segmen_melewati_rendah", "-")}</td>' if dua else "")
                     + ("".join(f'<td>{_angka((o.get("skor_rata_model") or {}).get(x["nama"]))}</td>' for x in ang)
                        if a.get("gabungan") else "")
                     + f'<td>{lencana(o["status"])}</td></tr>')
        s.append("</table>")
    s.append("<figure>" + svg_garis_waktu(a) + "<figcaption>"
             + T("Skor fake setiap orang per segmen sepanjang video (0 = asli, 1 = fake). ",
                 "Fake score of each person per segment across the video (0 = real, 1 = fake). ")
             + (T("Garis merah adalah batas toleransi dan garis oranye adalah ambang Youden. ",
                  "The red line is the tolerance limit and the orange line is the Youden threshold. ")
                if a.get("ambang_rendah") is not None else
                T("Garis merah putus-putus adalah ambang. ", "The dashed red line is the threshold. "))
             + T("Titik berlingkaran dengan label S<i>n</i> memiliki gambar Grad-CAM berlabel sama di bagian Grad-CAM.",
                 "Circled points labeled S<i>n</i> have a Grad-CAM image with the same label in the Grad-CAM section.")
             + "</figcaption></figure>")
    kunci = {sg["no"]: sg.get("frame_kunci") for sg in a["segmen"]}
    fig = []
    for o in a["ringkasan_orang"]:
        f = kunci.get(o["segmen_tertinggi"])
        if f:
            fig.append(f'<figure>{gambar(f)}<figcaption>'
                       + T(f'Frame kunci segmen S{o["segmen_tertinggi"]}, segmen dengan skor tertinggi untuk {e(o["orang"])}. '
                           'Kotak menandai wajah yang dianalisis.',
                           f'Key frame of segment S{o["segmen_tertinggi"]}, the highest-scoring segment for {e(o["orang"])}. '
                           'The box marks the analyzed face.')
                       + '</figcaption></figure>')
    if fig:
        s.append('<div class="grid2">' + "".join(fig[:4]) + "</div>")
    if a["gradcam"]:
        seg_w = {sg["no"]: sg for sg in a["segmen"]}
        skor_seg = {(r["segmen"], r["orang"]): r["skor"] for r in a["hasil"] if r["skor"] is not None}

        def info_g(g):
            sg = seg_w.get(g["segmen"], {})
            sk = g.get("skor_segmen", skor_seg.get((g["segmen"], g["orang"]), g["skor"]))
            return g.get("mulai_detik", sg.get("mulai_detik", 0)), g.get("selesai_detik", sg.get("selesai_detik", 0)), sk
        urut = sorted(a["gradcam"], key=lambda g: (info_g(g)[0], g["orang"]))
        ada_alasan = any(g.get("alasan") for g in urut)
        bawah_g = a["ambang_rendah"] if a.get("ambang_rendah") is not None else a["ambang"]

        def posisi(sk):
            if a.get("ambang_rendah") is None:
                return f'{"&ge;" if sk >= a["ambang"] else "&lt;"} {T("ambang", "threshold")} {a["ambang"]:.3f}'
            if sk >= a["ambang"]:
                return f'&ge; {T("batas toleransi", "tolerance limit")} {a["ambang"]:.3f}'
            if sk >= bawah_g:
                return T(f'zona pemeriksaan ({bawah_g:.3f} s.d. {a["ambang"]:.3f})',
                         f'examination zone ({bawah_g:.3f} to {a["ambang"]:.3f})')
            return f'&lt; {T("ambang Youden", "Youden threshold")} {bawah_g:.3f}'
        dua = a.get("ambang_rendah") is not None
        cak = {"lewat_ambang": (T("setiap segmen yang mencapai ambang Youden", "every segment that reaches the Youden threshold")
                                if dua else T("setiap segmen yang melewati ambang", "every segment above the threshold"))
                               + T(", ditambah segmen dengan skor tertinggi tiap orang", ", plus the highest-scoring segment of each person"),
               "tertinggi": T("segmen dengan skor tertinggi", "the highest-scoring segments"),
               "semua": T("setiap segmen", "every segment")}
        s.append("<h3>Grad-CAM</h3><p>" + T("Grad-CAM dibuat untuk ", "Grad-CAM was created for ")
                 + cak.get(a.get("gradcam_cakupan"), T("segmen dengan skor tertinggi setiap orang", "the highest-scoring segment of each person"))
                 + T(". Label S<i>n</i>, rentang waktu, dan skor segmen di tabel dan gambar berikut sama dengan titik "
                     "berlabel pada garis waktu, sehingga setiap peta bisa dicocokkan dengan posisinya di video.</p>"
                     "<p>Pada setiap gambar, baris atas berisi wajah yang dibaca model dan baris bawah berisi peta Grad-CAM. "
                     "Warna merah menandai area yang paling memengaruhi skor fake. ",
                     ". The S<i>n</i> labels, time ranges, and segment scores in the table and figures below match the "
                     "labeled points on the timeline, so each map can be matched to its position in the video.</p>"
                     "<p>In each figure, the top row shows the faces read by the model and the bottom row shows the "
                     "Grad-CAM map. Red marks the areas with the most influence on the fake score. ")
                 + (T("Di bawah setiap frame tertera waktunya di video dan bobot relatifnya. Bobot dihitung per segmen, "
                      "jadi setiap segmen selalu punya satu frame dengan bobot 1.00, termasuk segmen yang dinilai asli. Bobot "
                      "menunjukkan frame yang paling diperhatikan model, sedangkan yang menentukan fake atau tidak adalah "
                      "skor segmen. ",
                      "Below each frame are its time in the video and its relative weight. Weights are computed per "
                      "segment, so every segment always has one frame with a weight of 1.00, including segments judged "
                      "real. The weight shows the frame the model attended to most; the segment score decides fake or not. ")
                    if any(g.get("frame_detail") for g in urut) else
                    T("Analisis ini dibuat dengan versi aplikasi sebelumnya, jadi waktu dan bobot per frame belum tersedia. ",
                      "This analysis was made with an earlier version of the application, so per-frame times and weights are not available. "))
                 + T("Peta ini menunjukkan bagian wajah yang dipakai model untuk memutuskan, bukan bukti manipulasi yang "
                     "berdiri sendiri.</p>",
                     "The map shows which parts of the face the model used to decide; it is not standalone evidence of "
                     "manipulation.</p>"))
        s.append(f'<table class="data"><tr><th>Label</th><th>{T("Waktu", "Time")}</th><th>{T("Orang", "Person")}</th>'
                 f'<th>{T("Skor segmen", "Segment score")}</th><th>{T("Posisi terhadap ambang", "Position relative to thresholds")}</th>'
                 + (f'<th>{T("Alasan dipilih", "Reason selected")}</th>' if ada_alasan else '') + '</tr>' + "".join(
                     f'<tr><td><b>S{g["segmen"]}</b></td><td>{_jam(m_)}-{_jam(s_)}</td><td>{e(g["orang"])}</td>'
                     f'<td>{sk:.3f}</td><td>{posisi(sk)}</td>'
                     + (f'<td>{e(tr(g.get("alasan")) or "-")}</td>' if ada_alasan else '') + '</tr>'
                     for g in urut for m_, s_, sk in [info_g(g)]) + "</table>")
        for g in urut:
            m_, s_, sk = info_g(g)
            s.append(f'<figure>{gambar(g["file"])}<figcaption><b>S{g["segmen"]}</b> ({_jam(m_)}-{_jam(s_)}), '
                     f'{e(g["orang"])}, {T("skor segmen", "segment score")} {sk:.3f} ({posisi(sk)})'
                     + (f', model {e(g["model"])}' if a.get("gabungan") and g.get("model") else "")
                     + (T('. Frame dengan bobot relatif tertinggi: ', '. Frames with the highest relative weight: ') + ", ".join(
                         f'{d.get("waktu") or _jam(d["detik"])} ({d["bobot"]:.2f})'
                         for d in sorted(g["frame_detail"], key=lambda d: -d["bobot"])[:3])
                        + (T(". Segmen ini di bawah ambang, jadi bobot tersebut bukan tanda manipulasi",
                             ". This segment is below the threshold, so these weights are not a sign of manipulation")
                           if sk < bawah_g else "")
                        if g.get("frame_detail") else "")
                     + '.</figcaption></figure>')
    R = b.get("riwayat_analisis") or []
    if len(R) > 1:
        orang = sorted({o["orang"] for r in R for o in r["orang"]})
        s.append("<div style='page-break-inside:avoid'><h3>" + T("Perbandingan semua analisis", "Comparison of all analyses")
                 + "</h3><p>" + T("Barang bukti ini dianalisis lebih dari sekali dengan model atau aturan pengambilan frame "
                                  "yang berbeda. Kesimpulan di atas memakai analisis ",
                                  "This evidence item was analyzed more than once with different models or frame rules. "
                                  "The conclusion above uses the analysis of ")
                 + e(a["waktu_wib"]) + " (model " + e(a["model"]["nama"]) + ").</p>"
                 f'<table class="data"><tr><th>{T("Waktu", "Time")}</th><th>Model</th><th>{T("Data latih", "Training data")}</th>'
                 f'<th>{T("Aturan frame", "Frame rule")}</th><th>{T("Ambang", "Thresholds")}</th>'
                 + "".join(f"<th>{T('Skor rata', 'Mean score')} {e(o)}</th>" for o in orang)
                 + f"<th>{T('Kesimpulan', 'Conclusion')}</th></tr>")
        for r in R:
            sel = "".join(
                "<td>" + next((f"{x['skor_rata']:.3f}" for x in r["orang"] if x["orang"] == o), "-") + "</td>" for o in orang)
            s.append(f'<tr><td>{e(r["waktu_wib"])}</td><td>{e(r["model"])}</td><td>{e(r.get("skenario"))} '
                     f'{e(tr(r.get("nama_skenario") or ""))}</td><td>{e(_aturan(r["aturan_frame"], r["aturan_frame"]))}'
                     f'{"" if r.get("aturan_sesuai_model", True) else T(" (≠ saat latih)", " (≠ training)")}</td><td>{r["ambang"]:.3f}'
                     + (f" / {r['ambang_rendah']:.3f}" if r.get("ambang_rendah") is not None else "") + '</td>'
                     f'{sel}<td>{lencana(r["status"])}</td></tr>')
        s.append("</table></div>")
    s.append(f'<h3>{T("Rincian per segmen", "Per-segment details")}</h3><table class="data"><tr><th>{T("Segmen", "Segment")}</th>'
             f'<th>{T("Waktu", "Time")}</th><th>{T("Orang", "Person")}</th><th>{T("Wajah", "Faces")}</th>'
             f'<th>{T("Lebar wajah", "Face width")}</th><th>{T("Skor", "Score")}'
             + (T(" gabungan", " (ensemble)") if a.get("gabungan") else "") + '</th>'
             + ("".join(f"<th>{e(x['nama'])}</th>" for x in ang) if a.get("gabungan") else "")
             + f'<th>Grad-CAM</th><th>{T("Catatan", "Notes")}</th></tr>')
    ada_gc = {(g["segmen"], g["orang"]) for g in a.get("gradcam") or []}
    for r in a["hasil"]:
        rd = a.get("ambang_rendah")
        sk = "-" if r["skor"] is None else (f'<b class="salah">{r["skor"]:.3f}</b>' if r["skor"] >= a["ambang"]
                                            else f'<b style="color:#e08a00">{r["skor"]:.3f}</b>'
                                            if rd is not None and r["skor"] >= rd else f'{r["skor"]:.3f}')
        lebar_ = f"{r['lebar_wajah_rata']:.0f} px" if r["jumlah_wajah"] else "-"
        s.append(f'<tr><td>{r["segmen"]}</td><td>{_jam(r["mulai_detik"])}-{_jam(r["selesai_detik"])}</td>'
                 f'<td>{e(r["orang"])}</td><td>{r["jumlah_wajah"]}</td><td>{lebar_}</td><td>{sk}</td>'
                 + ("".join(f'<td>{_angka((r.get("skor_model") or {}).get(x["nama"]))}</td>' for x in ang) if a.get("gabungan") else "")
                 + (f'<td><b>S{r["segmen"]}</b></td>' if (r["segmen"], r["orang"]) in ada_gc else "<td>-</td>")
                 + (f'<td>{T("urutan dipotong", "sequence split")}: {e(", ".join(tr(x) for x in r["dipisah"]))}</td>'
                    if r.get("dipisah") else "<td>-</td>") + '</tr>')
    s.append("</table>")
    return "".join(s)


def _angka(v):
    return "-" if v is None else f"{float(v):.3f}"


def _baris_anggota(x, dua=True):
    auc = (x.get("uji") or {}).get("keseluruhan", {}).get("auc")
    return (f'<tr><td>{e(x["nama"])}</td><td>{e(x.get("crop"))}</td><td>{_angka(x.get("ambang_yakin"))}</td>'
            + (f'<td>{_angka(x.get("ambang_peka"))}</td>' if dua else "") + f'<td>{_angka(auc)}</td></tr>')


METODOLOGI_ID = (
    "<ol><li><b>Akuisisi.</b> Aplikasi menghitung hash MD5, SHA-1, dan SHA-256 file asal, lalu menyalinnya ke folder "
    "kasus sebagai salinan kerja dan, untuk barang bukti yang tabel akuisisinya mencantumkan salinan file asli, "
    "juga ke folder <i>original files</i> sebagai arsip. Setiap salinan dibuat read-only dan di-hash ulang untuk "
    "memastikan isinya identik dengan file asal. Semua pemeriksaan memakai salinan kerja; file asal tidak "
    "diubah.</li>"
    "<li><b>Verifikasi integritas.</b> Sebelum analisis, aplikasi menghitung ulang SHA-256 salinan kerja. Analisis "
    "hanya berjalan bila hasilnya sama dengan hash saat akuisisi.</li>"
    "<li><b>Segmentasi.</b> Video dibagi menjadi segmen 10 detik, sama dengan durasi klip data latih. Dari setiap "
    "segmen diambil 20 frame dengan aturan yang sama seperti saat model dilatih. Aturan baru mengambil 20 frame "
    "berurutan pada 25 fps di tengah segmen supaya gerak bibir terbaca. Aturan lama mengambil 20 frame merata "
    "sepanjang segmen. Aturan yang dipakai tercatat di setiap hasil analisis.</li>"
    "<li><b>Deteksi wajah.</b> MTCNN mendeteksi wajah dengan parameter yang sama seperti saat dataset dibuat: crop "
    "persegi dengan margin dan ukuran sesuai profil data latih model (misalnya margin 1,6 dan 299 x 299 piksel), "
    "probabilitas deteksi minimal 0,95, dan wajah dengan tepi hitam besar dilewati. Wajah setiap orang dilacak di "
    "dalam segmen dan dicocokkan antarsegmen dengan ArcFace.</li>"
    "<li><b>Klasifikasi.</b> Xception membaca fitur setiap frame wajah, lalu LSTM membaca hubungan antarframe dan "
    "menghasilkan skor fake 0 sampai 1 untuk setiap orang di setiap segmen.</li>"
    "<li><b>Keputusan.</b> Keputusan memakai dua batas dari data validasi model. <i>Ambang Youden</i> adalah ambang "
    "yang memberi selisih terbesar antara tingkat deteksi video fake dan tingkat salah tuduh pada video asli. "
    "<i>Batas toleransi 5%</i> adalah batas terendah yang dilewati paling banyak 5% video asli di data validasi. "
    "Seseorang dinyatakan <i>Terindikasi</i> bila skor rata-ratanya mencapai batas toleransi. Bila skor rata-ratanya "
    "berada dari ambang Youden sampai di bawah batas toleransi, atau ada segmen yang mencapai batas toleransi, "
    "statusnya <i>Perlu pemeriksaan lanjutan</i>. Bila skor rata-ratanya di bawah ambang Youden dan tidak ada "
    "segmen yang mencapai batas toleransi, statusnya <i>Tidak terindikasi</i>. Kesimpulan barang bukti mengikuti "
    "kesimpulan orang yang paling berat. Analisis yang dibuat dengan versi aplikasi sebelumnya bisa memakai ambang "
    "lain; nilai dan jenis ambang setiap analisis tercatat di hasil barang bukti masing-masing.</li>"
    "<li><b>Grad-CAM.</b> Gradien skor dirambatkan balik ke peta fitur Xception untuk menunjukkan area wajah yang "
    "paling memengaruhi skor. Grad-CAM dibuat untuk setiap segmen yang mencapai ambang Youden dan untuk segmen "
    "dengan skor tertinggi setiap orang; analisis dari versi aplikasi sebelumnya hanya memuat segmen tertinggi. Bobot "
    "relatif sebuah frame adalah rata-rata intensitas peta Grad-CAM frame itu, dibagi nilai tertinggi di segmen yang "
    "sama, sehingga bobot menunjukkan perhatian model dan bukan tingkat kepalsuan.</li>"
    "<li><b>Pemotongan urutan wajah.</b> Bila di dalam satu segmen adegan berganti (korelasi histogram warna antar"
    "frame berurutan turun di bawah 0,5) atau wajah berubah menjadi orang lain (kemiripan ArcFace terhadap "
    "bagian sebelumnya di bawah 0,35), urutan wajah dipotong. Setiap potongan dicocokkan ulang ke orang yang "
    "sesuai, dan potongan dengan wajah kurang dari batas minimum frame model tidak dianalisis.</li></ol>")

METODOLOGI_EN = (
    "<ol><li><b>Acquisition.</b> The application computes MD5, SHA-1, and SHA-256 hashes of the source file, then "
    "copies it into the case folder as a working copy and, for evidence whose acquisition table lists an original "
    "file copy, also into the <i>original files</i> folder as an archive. Each copy is made read-only and hashed "
    "again to confirm it is identical to the source file. All examination uses the working copy; the source file "
    "is never changed.</li>"
    "<li><b>Integrity verification.</b> Before analysis, the application recomputes the SHA-256 of the working copy. "
    "The analysis only runs if it matches the hash at acquisition.</li>"
    "<li><b>Segmentation.</b> The video is divided into 10-second segments, the same length as the training clips. "
    "From each segment, 20 frames are taken with the same rule used in training. The new rule takes 20 consecutive "
    "frames at 25 fps in the middle of the segment so lip movement can be read. The old rule takes 20 frames evenly "
    "across the segment. The rule used is recorded in every analysis result.</li>"
    "<li><b>Face detection.</b> MTCNN detects faces with the same parameters used to build the dataset: a square "
    "crop with the margin and size of the model's training profile (for example a margin of 1.6 and 299 x 299 "
    "pixels), a minimum detection probability of 0.95, and faces with large black borders skipped. Each person's "
    "face is tracked within the segment and matched across segments with ArcFace.</li>"
    "<li><b>Classification.</b> Xception reads the features of each face frame, then an LSTM reads the relationship "
    "between frames and produces a fake score from 0 to 1 for each person in each segment.</li>"
    "<li><b>Decision.</b> The decision uses two limits from the model's validation data. The <i>Youden threshold</i> "
    "gives the largest difference between the fake detection rate and the false accusation rate on real videos. "
    "The <i>5% tolerance limit</i> is the lowest score exceeded by at most 5% of real videos in the validation data. "
    "A person is marked <i>Deepfake indicated</i> when their mean score reaches the tolerance limit. When the mean "
    "score is from the Youden threshold up to below the tolerance limit, or any segment reaches the tolerance limit, "
    "the status is <i>Further examination required</i>. When the mean score is below the Youden threshold and no "
    "segment reaches the tolerance limit, the status is <i>Not indicated as deepfake</i>. The evidence verdict follows "
    "the most severe verdict among the people. Analyses made with earlier versions of the application may use other "
    "thresholds; the value and type of each threshold are recorded in each evidence result.</li>"
    "<li><b>Grad-CAM.</b> The score gradient is propagated back to the Xception feature maps to show the face areas "
    "with the most influence on the score. Grad-CAM is created for every segment that reaches the Youden threshold "
    "and for the highest-scoring segment of each person; analyses from earlier versions only include the highest "
    "segment. The relative weight of a frame is the mean intensity of its Grad-CAM map divided by the highest value "
    "in the same segment, so the weight shows model attention, not how fake the frame is.</li>"
    "<li><b>Face sequence splitting.</b> When the scene changes within a segment (the color histogram correlation "
    "between consecutive frames drops below 0.5) or the face changes to another person (ArcFace similarity to the "
    "previous part below 0.35), the face sequence is split. Each part is matched again to the right person, and "
    "parts with fewer faces than the model's minimum frame count are not analyzed.</li></ol>")


def bagian_metodologi(model_dipakai, dua_ambang=False):
    s = [f'<h2 class="putus">{T("Metodologi dan keterbatasan", "Methodology and limitations")}</h2>',
         T(METODOLOGI_ID, METODOLOGI_EN)]
    for mi in model_dipakai.values():
        if mi.get("anggota"):
            s.append(f"<h3>{e(mi['nama'])}</h3><p>" + T("Gabungan model dengan ", "Model ensemble with ")
                     + (T("skor tertinggi", "highest score") if mi.get("cara_gabung") == "maks" else T("rata-rata", "mean"))
                     + T(" pada skala terpadu. Skor setiap model dipetakan menurut ambang model itu sendiri sebelum "
                         "digabung.</p>",
                         " on a unified scale. Each model's score is mapped by that model's own thresholds before "
                         "combining.</p>"))
            for x in mi["anggota"]:
                baris = [("Model", e(x["nama"])),
                         (T("Data latih", "Training data"), f'{T("Skenario", "Scenario")} {e(x.get("skenario"))} {e(tr(x.get("nama_skenario")))}'),
                         (T("Crop wajah", "Face crop"), e(x.get("crop"))), (T("AUC validasi", "Validation AUC"), _angka(x.get("val_auc"))),
                         ((T("Ambang Youden / batas toleransi", "Youden threshold / tolerance limit"),
                           f'{_angka(x.get("ambang_peka"))} ({e(tr(x.get("jenis_ambang_peka")))}) / '
                           f'{_angka(x.get("ambang_yakin"))} ({e(tr(x.get("jenis_ambang_yakin")))})')
                          if x.get("ambang_peka") is not None else
                          (T("Ambang", "Threshold"), f'{_angka(x.get("ambang_yakin"))} ({e(tr(x.get("jenis_ambang_yakin")))})')),
                         (T("SHA-256 bobot model", "Model weights SHA-256"), f'<span class="mono">{e(x.get("sha256_bobot"))}</span>')]
                for sumber, u in (x.get("uji") or {}).items():
                    baris.append((T(f"Kinerja pada data uji {_sumber(sumber)}", f"Performance on {_sumber(sumber)} test data"),
                                  T(f'AUC {_angka(u.get("auc"))}, akurasi {_angka(u.get("akurasi"))}',
                                    f'AUC {_angka(u.get("auc"))}, accuracy {_angka(u.get("akurasi"))}')))
                s.append(f"<h4>{T('Anggota', 'Member')} {e(x['nama'])}</h4>" + _tabel(baris))
            continue
        baris = [("Model", e(mi["nama"])),
                 (T("Arsitektur", "Architecture"), f'{e(mi["arsitektur"])} (backbone {e(mi["backbone"])}, LSTM '
                  f'{mi["lstm_hidden"]} x {mi["lstm_lapis"]} {T("lapis", "layer(s)")}'
                  f'{T(", dua arah", ", bidirectional") if mi["lstm_dua_arah"] else ""}, '
                  f'{T("urutan", "sequence of")} {mi["panjang_urutan"]} frame)'),
                 (T("Data latih", "Training data"), f'{T("Skenario", "Scenario")} {e(mi["skenario"])} {e(tr(mi.get("nama_skenario")))}'),
                 (T("Epoch terbaik / AUC validasi / ambang Youden", "Best epoch / validation AUC / Youden threshold"),
                  f'{mi["epoch_terbaik"]} / {mi["val_auc"]} / {mi["ambang_val"]}'),
                 (T("SHA-256 bobot model", "Model weights SHA-256"), f'<span class="mono">{e(mi["sha256_bobot"])}</span>'),
                 (T("PyTorch / perangkat", "PyTorch / device"), f'{e(mi["torch"])} / {e(mi["perangkat"])}')]
        for sumber, u in mi.get("uji", {}).items():
            n_ = int(u["n"]) if u.get("n") is not None else "-"
            baris.append((T(f"Kinerja pada data uji {_sumber(sumber)}", f"Performance on {_sumber(sumber)} test data"),
                          T(f'AUC {u.get("auc", float("nan")):.3f}, akurasi {u.get("akurasi", float("nan")):.3f}, '
                            f'F1 {u.get("f1", float("nan")):.3f} (n = {n_}, ambang 0,5)',
                            f'AUC {u.get("auc", float("nan")):.3f}, accuracy {u.get("akurasi", float("nan")):.3f}, '
                            f'F1 {u.get("f1", float("nan")):.3f} (n = {n_}, threshold 0.5)')))
        s.append(f"<h3>Model {e(mi['nama'])}</h3>" + _tabel(baris))
    s.append('<div class="kotak peringatan">' + T(
        "<b>Keterbatasan.</b> Hasil di laporan ini adalah keluaran model statistik, jadi merupakan <b>indikasi</b> dan "
        "bukan pembuktian yang berdiri sendiri. Model hanya mempelajari teknik manipulasi yang ada di data latihnya "
        "(lihat baris Data latih di atas): Celeb-DF v2 dan/atau video buatan FaceFusion dengan face swap hyperswap, "
        "inswapper, simswap, dan ghost serta lip-sync wav2lip. Akurasi bisa turun pada teknik lain seperti reenactment, "
        "wajah sintetis penuh, atau video dari model difusi, dan pada video beresolusi rendah, kompresi berat, wajah "
        "kecil, tertutup, atau menyamping, serta pencahayaan ekstrem. Angka kinerja di atas diukur pada data uji "
        "penelitian dan bisa berbeda pada video lain. Kesimpulan perlu dikonfirmasi dengan pemeriksaan lain: analisis "
        "visual frame per frame, audio, asal unggahan, metadata, dan keterangan saksi.",
        "<b>Limitations.</b> The results in this report are the output of a statistical model, so they are an "
        "<b>indication</b> and not standalone proof. The model only learned the manipulation techniques in its training "
        "data (see the Training data row above): Celeb-DF v2 and/or videos made with FaceFusion using hyperswap, "
        "inswapper, simswap, and ghost face swap, and wav2lip lip-sync. Accuracy can drop on other techniques such as "
        "reenactment, fully synthetic faces, or diffusion-generated video, and on low-resolution video, heavy "
        "compression, small, occluded, or turned faces, and extreme lighting. The performance figures above were "
        "measured on the research test data and may differ on other videos. The verdict should be confirmed with other "
        "examinations: frame-by-frame visual analysis, audio, upload origin, metadata, and witness statements.")
             + "</div>")
    return "".join(s)


def bagian_integritas(kasus_id, data):
    v = verifikasi_kasus(kasus_id)
    n_ok = sum(1 for f in v["file"] if f["status"] == "COCOK")
    s = [f'<h2 class="putus">{T("Verifikasi integritas dan rantai bukti (chain of custody)", "Integrity verification and chain of custody")}</h2>',
         '<div class="kotak">' + T(f'Diverifikasi {e(v["waktu"])}: rantai log {v["jumlah_entri"]} entri ',
                                   f'Verified {e(v["waktu"])}: log chain of {v["jumlah_entri"]} entries ')
         + f'<span class="{"ok" if v["rantai_utuh"] else "salah"}">'
         + (T("UTUH", "INTACT") if v["rantai_utuh"] else T("TIDAK UTUH", "BROKEN")) + '</span>; '
         + T(f'{n_ok} dari {len(v["file"])} file cocok dengan hash yang tercatat.',
             f'{n_ok} of {len(v["file"])} files match their recorded hashes.')
         + ("<ul>" + "".join(f"<li class=salah>{e(tr(m))}</li>" for m in v["masalah"]) + "</ul>" if v["masalah"] else "")
         + "</div>",
         "<p>" + T("Setiap entri log menyimpan hash entri sebelumnya, jadi entri yang diubah atau dihapus akan terdeteksi "
                   "saat verifikasi. Hash entri terakhir: ",
                   "Each log entry stores the hash of the previous entry, so an edited or deleted entry is detected "
                   "during verification. Hash of the last entry: ")
         + "<span class=mono>" + e(data["log"][-1]["hash_entri"] if data["log"] else "-") + "</span></p>",
         f'<table class="data"><tr><th>No</th><th>{T("Waktu", "Time")}</th><th>{T("Kegiatan", "Activity")}</th>'
         f'<th>File</th><th>SHA-256</th><th>Operator</th></tr>']
    for x in data["log"]:
        s.append(f'<tr><td>{x["no"]}</td><td>{e(x["waktu_wib"])}</td><td>{e(_kegiatan(x["langkah"]))}</td>'
                 f'<td class="mono">{e(x["file"].split("/kasus_forensik/", 1)[-1])}</td>'
                 f'<td class="mono">{e(x["sha256"])}</td><td>{e(x["operator"])}@{e(x["komputer"])}</td></tr>')
    s.append("</table>")
    return "".join(s)


def buat_html(kasus_id, bb_ids=None, model=None):
    data = muat_kasus(kasus_id)
    if model:  # kesimpulan memakai analisis terbaru dari model yang dipilih (bila bukti pernah dianalisis dengannya)
        for b in data["bukti"]:
            r = [x for x in b.get("riwayat_analisis") or []
                 if (x.get("gabungan") if model == "@gabungan" else x["model"] == model)]
            if r and (not b.get("analisis") or b["analisis"]["id"] != r[-1]["id"]):
                b["analisis"] = muat_analisis(kasus_id, b["id"], r[-1]["id"])
    k = data["kasus"]
    bukti = [b for b in data["bukti"] if not bb_ids or b["id"] in bb_ids]
    utc, wib = sekarang()
    no_lap = f"{T('LAP', 'REP')}/{k['id']}/{datetime.now():%Y%m%d%H%M%S}"
    s = [f'<!doctype html><html lang="{T("id", "en")}"><head><meta charset="utf-8"><title>'
         f'{T("Laporan forensik", "Forensic report")} {e(k["nomor"])}</title>'
         f'<meta name="viewport" content="width=device-width,initial-scale=1"><style>{CSS}</style></head><body>',
         f'<div class="kop"><div><small>{T("LAPORAN PEMERIKSAAN FORENSIK DIGITAL", "DIGITAL FORENSIC EXAMINATION REPORT")}</small>'
         f'<h1>{T("Deteksi Manipulasi Wajah (Deepfake) pada Barang Bukti Video", "Facial Manipulation (Deepfake) Detection in Video Evidence")}'
         f'</h1></div><div style="text-align:right"><small>{T("No. laporan", "Report no.")}</small><br>'
         f'<b class="mono">{e(no_lap)}</b><br><small>{e(wib)}</small></div></div>',
         f'<h2>1. {T("Identitas pemeriksaan", "Examination details")}</h2>',
         _tabel([(T("Nomor kasus", "Case number"), e(k["nomor"])), (T("Judul / perihal", "Title / subject"), e(k["judul"])),
                 (T("Pemeriksa", "Examiner"), e(k["pemeriksa"])),
                 (T("Instansi", "Institution"), e(k.get("instansi") or "-")), (T("Keterangan", "Notes"), e(k.get("keterangan") or "-")),
                 (T("Kasus dibuat", "Case created"), e(k["dibuat_wib"])),
                 (T("Laporan dibuat", "Report created"), T(f"{e(wib)} oleh {e(k['pemeriksa'])}", f"{e(wib)} by {e(k['pemeriksa'])}")),
                 (T("Perangkat lunak", "Software"), T("Aplikasi Forensik Deepfake (Xception + LSTM + Grad-CAM), pipeline dataset "
                                                      "deepfake wajah Indonesia",
                                                      "Deepfake Forensics Application (Xception + LSTM + Grad-CAM), Indonesian "
                                                      "face deepfake dataset pipeline"))]),
         f'<h2>2. {T("Ringkasan hasil", "Summary of results")}</h2>',
         f'<table class="data"><tr><th>{T("Bukti", "Evidence")}</th><th>{T("Nama file", "File name")}</th>'
         f'<th>{T("SHA-256 (salinan)", "SHA-256 (copy)")}</th><th>{T("Orang", "People")}</th>'
         f'<th>{T("Skor rata-rata tertinggi", "Highest mean score")}</th><th>{T("Kesimpulan", "Conclusion")}</th></tr>']
    for b in bukti:
        a = b.get("analisis")
        if a:
            top = max(a["ringkasan_orang"], key=lambda o: o["skor_rata"], default=None)
            teks_top = "{:.3f} ({})".format(top["skor_rata"], e(top["orang"])) if top else "-"
            s.append(f'<tr><td>{e(b["id"])}</td><td>{e(b["nama_file"])}</td><td class="mono">{b["salinan"]["hash"]["sha256"]}</td>'
                     f'<td>{len(a["ringkasan_orang"])}</td><td>{teks_top}</td>'
                     f'<td>{lencana(a["status"])}</td></tr>')
        else:
            s.append(f'<tr><td>{e(b["id"])}</td><td>{e(b["nama_file"])}</td><td class="mono">{b["salinan"]["hash"]["sha256"]}</td>'
                     f'<td>-</td><td>-</td><td>{T("belum dianalisis", "not analyzed")}</td></tr>')
    hitung = {}
    for b in bukti:
        st_ = (b.get("analisis") or {}).get("status", "BELUM")
        hitung[st_] = hitung.get(st_, 0) + 1
    nama_st = [("TERINDIKASI", T("terindikasi deepfake", "deepfake indicated")),
               ("RAGU", T("perlu pemeriksaan lanjutan", "further examination required")),
               ("TIDAK", T("tidak terindikasi", "not indicated")),
               ("TIDAK_ADA_WAJAH", T("tidak dapat dianalisis karena wajah tidak cukup", "cannot be analyzed (not enough faces)")),
               ("BELUM", T("belum dianalisis", "not analyzed"))]
    s.append("</table><p>" + T(f"Laporan ini mencakup {len(bukti)} barang bukti: ", f"This report covers {len(bukti)} evidence items: ")
             + _daftar([f"{hitung[k]} {t}" for k, t in nama_st if hitung.get(k)], maks=10) + ". "
             + (T("Barang bukti yang terindikasi atau perlu pemeriksaan lanjutan: ",
                  "Evidence items indicated or requiring further examination: ")
                + _daftar([f'{b["id"]} ({e(b["nama_file"])})' for b in bukti
                           if (b.get("analisis") or {}).get("status") in ("TERINDIKASI", "RAGU")], maks=10,
                          satuan=T("barang bukti", "evidence items")) + ". "
                if hitung.get("TERINDIKASI") or hitung.get("RAGU") else "")
             + T("Temuan dan penjelasan keputusan untuk setiap barang bukti ada di bagian masing-masing.</p>",
                 "Findings and decision explanations for each evidence item are in its own section.</p>"))
    s.append("<p>" + T("Kesimpulan di tabel ini adalah <b>indikasi</b> dari model pembelajaran mesin. Dasar dan "
                       "batasannya dijelaskan di bagian Metodologi dan keterbatasan.",
                       "The conclusions in this table are an <b>indication</b> from a machine learning model. Their basis "
                       "and limitations are explained in the Methodology and limitations section.") + "</p>")
    model_dipakai = {}
    for i, b in enumerate(bukti, 3):
        s.append(bagian_bukti(b, i))
        if b.get("analisis"):
            model_dipakai[b["analisis"]["model"]["nama"]] = b["analisis"]["model"]
    s.append(bagian_metodologi(model_dipakai, any((b.get("analisis") or {}).get("ambang_rendah") is not None
                                                  for b in bukti)))
    s.append(bagian_integritas(kasus_id, data))
    s.append(f'<div class="ttd"><div>{e(wib.split(" ")[0])}<br>{T("Pemeriksa", "Examiner")},<br><br><br><br>'
             f'<b>{e(k["pemeriksa"])}</b><br><small>{e(k.get("instansi") or "")}</small></div></div></body></html>')
    isi = "".join(s)
    if T("id", "en") == "en":  # bentuk tunggal untuk angka 1 ("1 segments" -> "1 segment")
        isi = re.sub(r"(?<![\d.,])1 (segment|model|file|evidence item|supporting finding|other segment)s\b", r"1 \1", isi)
    return isi, no_lap


def cari_browser():
    kandidat = []
    for env in ("PROGRAMFILES(X86)", "PROGRAMFILES", "LOCALAPPDATA"):
        akar = os.environ.get(env)
        if akar:
            kandidat += [Path(akar) / "Microsoft/Edge/Application/msedge.exe",
                         Path(akar) / "Google/Chrome/Application/chrome.exe"]
    for n in ("msedge", "chrome", "google-chrome", "chromium", "chromium-browser"):
        w = shutil.which(n)
        if w:
            kandidat.append(Path(w))
    kandidat += sorted(Path("/opt/pw-browsers").glob("chromium-*/chrome-linux/chrome"))
    return next((p for p in kandidat if p.exists()), None)


def cetak_pdf(html_path, pdf_path):
    b = cari_browser()
    if not b:
        raise RuntimeError(T("Microsoft Edge / Google Chrome tidak ditemukan untuk mencetak PDF",
                             "Microsoft Edge / Google Chrome not found for printing the PDF"))
    with tempfile.TemporaryDirectory() as profil:
        perintah = [str(b), "--headless=new", "--disable-gpu", "--no-first-run", "--disable-extensions",
                    f"--user-data-dir={profil}", "--no-pdf-header-footer", "--print-to-pdf-no-header",
                    f"--print-to-pdf={Path(pdf_path).resolve()}", Path(html_path).resolve().as_uri()]
        if os.name != "nt":
            perintah.insert(1, "--no-sandbox")
        subprocess.run(perintah, capture_output=True, timeout=180)
    if not Path(pdf_path).exists() or Path(pdf_path).stat().st_size == 0:
        raise RuntimeError(T("Browser gagal mencetak PDF", "The browser failed to print the PDF"))
    return pdf_path


def buat_laporan(kasus_id, bb_ids=None, log=print, kemajuan=None, model=None):
    d = folder_kasus(kasus_id)
    kemajuan = kemajuan or (lambda f, t="": None)
    log(T("Menyusun laporan dan memverifikasi rantai bukti...", "Building the report and verifying the chain of custody..."))
    kemajuan(0.1, T("menyusun HTML", "building HTML"))
    isi, no_lap = buat_html(kasus_id, bb_ids, model)
    if model:
        log(T(f"Kesimpulan memakai analisis terbaru dengan model {model} (bila ada untuk bukti tersebut)",
              f"Conclusions use the latest analysis with model {model} (when available for the evidence)"))
    (d / "laporan").mkdir(exist_ok=True)
    dasar = d / "laporan" / f"laporan_{datetime.now():%Y%m%d_%H%M%S}"
    p_html = dasar.with_suffix(".html")
    p_html.write_text(isi, encoding="utf-8")
    catat("laporan_html", p_html, folder_kerja=d, info={"no_laporan": no_lap, "bukti": bb_ids or "semua",
                                                         "bahasa": T("id", "en")})
    log(T(f"Laporan HTML: {p_html}", f"HTML report: {p_html}"))
    kemajuan(0.6, T("mencetak PDF", "printing PDF"))
    hasil = {"html": p_html.as_posix(), "pdf": None, "no_laporan": no_lap}
    try:
        p_pdf = dasar.with_suffix(".pdf")
        cetak_pdf(p_html, p_pdf)
        catat("laporan_pdf", p_pdf, induk=[p_html], folder_kerja=d, info={"no_laporan": no_lap})
        hasil["pdf"] = p_pdf.as_posix()
        log(T(f"Laporan PDF: {p_pdf}", f"PDF report: {p_pdf}"))
    except Exception as err:  # noqa: BLE001
        log(T(f"! PDF gagal dibuat ({err}); laporan HTML tetap tersedia dan bisa dicetak dari browser.",
              f"! PDF could not be created ({err}); the HTML report is still available and can be printed from a browser."))
    kemajuan(1.0, T("selesai", "done"))
    _ = forensik  # log ditulis lewat forensik.catat
    return hasil
