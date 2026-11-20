@echo off
setlocal EnableExtensions
chcp 65001 >nul
cd /d "%~dp0"
title Forensik Deepfake

set "VPY=.venv\Scripts\python.exe"

rem ================= 1. Lingkungan Python (.venv), dibuat sekali =================
if exist "%VPY%" goto :venv_ada
set "PY="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY py -%%V -c "import sys" >nul 2>nul && set "PY=py -%%V"
)
if not defined PY python -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" >nul 2>nul && set "PY=python"
if not defined PY (
    echo [!] Butuh Python 3.10 sampai 3.12. Pasang dari https://www.python.org lalu jalankan file ini lagi.
    pause
    exit /b 1
)
echo Membuat lingkungan Python di folder .venv ...
%PY% -m venv .venv || goto :gagal

:venv_ada
rem ================= 2. Dependensi, dipasang sekali =================
if exist ".venv\terpasang" goto :jalankan
echo Memasang dependensi, hanya sekali dan bisa makan waktu cukup lama...
"%VPY%" -m pip install -q --upgrade pip
where nvidia-smi >nul 2>nul
if errorlevel 1 (
    echo GPU NVIDIA tidak ditemukan: memasang PyTorch versi CPU.
    "%VPY%" -m pip install torch torchvision || goto :gagal
    "%VPY%" -m pip install onnxruntime || goto :gagal
) else (
    echo GPU NVIDIA ditemukan: memasang PyTorch CUDA 13.0 dan onnxruntime-gpu, unduhan beberapa GB.
    "%VPY%" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130 || goto :gagal
    "%VPY%" -m pip install "onnxruntime-gpu[cuda,cudnn]" || "%VPY%" -m pip install onnxruntime || goto :gagal
)
"%VPY%" -m pip install -r requirements.txt || goto :gagal
"%VPY%" -m pip install facenet-pytorch --no-deps || goto :gagal
"%VPY%" -m pip install pywebview || echo [!] pywebview gagal dipasang; aplikasi akan dibuka di browser.
echo ok> ".venv\terpasang"

:jalankan
echo Membuka aplikasi Forensik Deepfake. Jendela ini menampilkan log dan harus tetap terbuka.
"%VPY%" aplikasi_forensik.py %*
if errorlevel 1 pause
exit /b 0

:gagal
echo [!] Pemasangan gagal. Lihat pesan di atas. InsightFace di Windows kadang butuh Microsoft C++ Build Tools.
pause
exit /b 1
