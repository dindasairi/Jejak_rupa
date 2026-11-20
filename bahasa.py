"""Bahasa aplikasi forensik (Indonesia / English).

- T(id, en)  : teks yang dibuat sekarang (log, pesan galat, gambar, laporan) dalam bahasa aktif.
- tr(teks)   : terjemahkan teks data yang tersimpan dalam bahasa Indonesia (jenis ambang, alasan, temuan
               metadata, dsb.) ke bahasa aktif. Data di hasil.json/bukti.json selalu disimpan dalam bahasa
               Indonesia supaya rantai bukti tidak bergantung pada bahasa tampilan.
- kes(status): nama kesimpulan dari kode status.
Bahasa aktif disimpan di data/pengaturan_aplikasi.json dan diubah dari tombol bahasa di aplikasi.
"""
import json
import re
from pathlib import Path

_FILE = Path("data") / "pengaturan_aplikasi.json"
BAHASA = "id"

KESIMPULAN = {
    "TERINDIKASI": ("Terindikasi deepfake", "Deepfake indicated"),
    "RAGU": ("Perlu pemeriksaan lanjutan", "Further examination required"),
    "TIDAK": ("Tidak terindikasi deepfake", "Not indicated as deepfake"),
    "TIDAK_ADA_WAJAH": ("Tidak dapat dianalisis (wajah tidak cukup)", "Cannot be analyzed (not enough faces)"),
}

# teks data tersimpan (bahasa Indonesia) -> English. Dipakai juga oleh tampilan (dikirim lewat API "info").
TEPAT = {
    # jenis ambang
    "ambang Youden dari data val": "Youden threshold from validation data",
    "ambang tetap": "fixed threshold",
    "zona terpadu: ambang Youden masing-masing model": "unified zone: each model's Youden threshold",
    "zona terpadu: ambang peka masing-masing model": "unified zone: each model's Youden threshold",
    # alasan Grad-CAM
    "skor tertinggi orang ini": "highest score for this person",
    "melewati batas toleransi": "reaches the tolerance limit",
    "di zona pemeriksaan": "in the examination zone",
    "melewati ambang": "above the threshold",
    "skor tinggi": "high score",
    # pemotongan urutan wajah
    "pergantian adegan": "scene change",
    "wajah berganti orang": "face changed to another person",
    # aturan pengambilan frame
    "sesuai model": "as trained",
    "merata sepanjang segmen (aturan lama)": "evenly across the segment (old rule)",
    "20 frame berurutan 25 fps (aturan baru)": "20 consecutive frames at 25 fps (new rule)",
    "20 frame merata sepanjang segmen (aturan lama)": "20 frames evenly across the segment (old rule)",
    "frame merata (aturan lama)": "evenly spaced frames (old rule)",
    # skenario model
    "Celeb-DF saja": "Celeb-DF only",
    "Celeb-DF + Indonesia": "Celeb-DF + Indonesia",
    "Indonesia saja": "Indonesia only",
    "Celeb-DF + Indonesia + in the wild": "Celeb-DF + Indonesia + in the wild",
    "Indonesia + in the wild": "Indonesia + in the wild",
    # temuan metadata
    "Tag metadata menyatakan video SINTETIS (video fake buatan pipeline dataset ini).":
        "A metadata tag marks the video as SYNTHETIC (a fake video made by this dataset pipeline).",
    "Tidak ada creation_time di kontainer: metadata waktu rekam tidak tersedia atau sudah dihapus.":
        "No creation_time in the container: recording time metadata is missing or was removed.",
    "OpenCV tidak bisa membaca frame video; file mungkin rusak atau kodeknya tidak didukung.":
        "OpenCV cannot read the video frames; the file may be damaged or its codec is not supported.",
    "Hanya untuk pemutaran di aplikasi. Bukan barang bukti; analisis memakai salinan kerja.":
        "For playback in the application only. Not evidence; analysis uses the working copy.",
    "Tidak ada": "None",
}
POLA = [
    (r"^batas toleransi (\d+)%: paling banyak \1% video asli di data validasi melewati batas ini$",
     r"tolerance limit \1%: at most \1% of real videos in the validation data exceed this limit"),
    (r"^ambang tetap 0,5 \(ambang '(.+)' belum tersedia untuk model ini\)$",
     r"fixed threshold 0.5 (threshold '\1' is not available for this model)"),
    (r"^ambang (.+) dari data val$", r"\1 threshold from validation data"),
    (r"^zona terpadu gabungan (\d+) model \(skor tertinggi\): skor tiap model dipetakan menurut ambang masing-masing$",
     r"unified zone, ensemble of \1 models (highest score): each model's score is mapped by its own thresholds"),
    (r"^zona terpadu gabungan (\d+) model \(rata-rata\): skor tiap model dipetakan menurut ambang masing-masing$",
     r"unified zone, ensemble of \1 models (mean): each model's score is mapped by its own thresholds"),
    (r"^zona terpadu gabungan (\d+) model \(skor tertinggi\): skor tiap model dipetakan menurut ambang yakin masing-masing$",
     r"unified zone, ensemble of \1 models (highest score): each model's score is mapped by its own thresholds"),
    (r"^zona terpadu gabungan (\d+) model \(rata-rata\): skor tiap model dipetakan menurut ambang yakin masing-masing$",
     r"unified zone, ensemble of \1 models (mean): each model's score is mapped by its own thresholds"),
    (r"^Kontainer ditulis oleh FFmpeg/libavformat \((.*)\): file pernah diproses ulang perangkat lunak \(umum pada "
     r"unduhan media sosial, bukan bukti manipulasi\)\.$",
     r"Container written by FFmpeg/libavformat (\1): the file was re-processed by software (common for social media "
     r"downloads, not evidence of manipulation)."),
    (r"^Metadata perangkat/perangkat lunak: (.*)$", r"Device/software metadata: \1"),
    (r"^Entri #(\d+): rantai terputus \(ada entri dihapus/disisipkan\)$",
     r"Entry #\1: chain broken (an entry was deleted or inserted)"),
    (r"^Entri #(\d+): isi entri berubah setelah dicatat$", r"Entry #\1: entry content changed after it was logged"),
    (r"^(.+): (HILANG|BERUBAH)$", None),  # ditangani di tr() supaya nama file tidak ikut diterjemahkan
]
STATUS_FILE = {"COCOK": "MATCH", "BERUBAH": "CHANGED", "HILANG": "MISSING", "DIHAPUS": "DELETED"}


def muat():
    global BAHASA
    try:
        BAHASA = json.loads(_FILE.read_text(encoding="utf-8")).get("bahasa", "id")
    except (OSError, ValueError):
        BAHASA = "id"
    return BAHASA


def atur(b):
    global BAHASA
    BAHASA = "en" if str(b).lower().startswith("en") else "id"
    try:
        _FILE.parent.mkdir(parents=True, exist_ok=True)
        lama = json.loads(_FILE.read_text(encoding="utf-8")) if _FILE.exists() else {}
        _FILE.write_text(json.dumps({**lama, "bahasa": BAHASA}, indent=2), encoding="utf-8")
    except (OSError, ValueError):
        pass
    return BAHASA


def en():
    return BAHASA == "en"


def T(id_teks, en_teks):
    return en_teks if BAHASA == "en" else id_teks


def kes(status):
    p = KESIMPULAN.get(status)
    return (p[1] if BAHASA == "en" else p[0]) if p else str(status)


def tr(teks):
    """Terjemahkan teks data berbahasa Indonesia (bila dikenal) ke bahasa aktif."""
    if BAHASA != "en" or not isinstance(teks, str):
        return teks
    if teks in TEPAT:
        return TEPAT[teks]
    for pola, ganti in POLA:
        m = re.match(pola, teks)
        if m and ganti:
            return m.expand(ganti)
        if m and pola.startswith("^(.+): (HILANG"):
            return f"{m[1]}: {STATUS_FILE.get(m[2], m[2])}"
    return teks


def kamus_data():
    """Kamus data untuk tampilan (JavaScript): pola regex memakai $1 sebagai pengganti."""
    return {"tepat": TEPAT, "pola": [[p, g.replace("\\1", "$1")] for p, g in POLA if g],
            "status_file": STATUS_FILE, "kesimpulan": {k: v[1] for k, v in KESIMPULAN.items()}}


muat()
