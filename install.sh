#!/bin/bash
# ============================================================
# install.sh — Python 3.10 | PyTorch 2.1.1+cu118 | CUDA 11.8
# Cara pakai: chmod +x install.sh && ./install.sh
# ============================================================
set -e
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIBS_DIR="$ROOT_DIR/libs"

echo ""
echo "=============================================="
echo "  Head Detection Project — Install Script"
echo "  PyTorch 2.1.1 | CUDA 11.8 | Python 3.10"
echo "=============================================="

echo ""
echo "[Check] Verifikasi environment..."
python -c "
import sys, torch
assert sys.version_info[:2] == (3,10), f'Python 3.10 diperlukan, ditemukan {sys.version}'
assert torch.__version__.startswith('2.1.1'), f'PyTorch 2.1.1 diperlukan, ditemukan {torch.__version__}'
assert torch.version.cuda == '11.8', f'CUDA 11.8 diperlukan, ditemukan {torch.version.cuda}'
print(f'  Python  : {sys.version.split()[0]}  OK')
print(f'  PyTorch : {torch.__version__}  OK')
print(f'  CUDA    : {torch.version.cuda}  OK')
"

echo ""
echo "[1/4] Installing causal-conv1d..."
cd "$LIBS_DIR/causal-conv1d"
pip install -e . --no-build-isolation
echo "  OK"

echo ""
echo "[2/4] Installing mamba-ssm..."
cd "$LIBS_DIR/mamba-1p1p1"
pip install -e . --no-build-isolation
echo "  OK"

echo ""
echo "[3/4] Installing Detectron2 (Vim fork)..."
cd "$LIBS_DIR/detectron2_vim"
pip install -e . --no-build-isolation
echo "  OK"

echo ""
echo "[4/4] Installing Python dependencies..."
cd "$ROOT_DIR/project"
pip install -r requirements.txt
echo "  OK"

echo ""
echo "[Verify] Mengecek semua import..."
python -c "
import torch
from mamba_ssm.modules.mamba_simple import Mamba
from causal_conv1d import causal_conv1d_fn
import detectron2
from detectron2.modeling import VisionMambaDet
print('  torch        :', torch.__version__)
print('  mamba_ssm    : OK')
print('  causal_conv1d: OK')
print('  detectron2   :', detectron2.__version__)
print('  Semua import berhasil!')
"

echo ""
echo "=============================================="
echo "  Instalasi selesai!"
echo ""
echo "  Langkah selanjutnya:"
echo "  1. Taruh pretrained di:"
echo "     project/checkpoints/vim_tiny_pretrained.pth"
echo ""
echo "  2. Taruh dataset di:"
echo "     project/data/train/images/  + annotations.json"
echo "     project/data/val/images/    + annotations.json"
echo ""
echo "  3. Edit NUM_TRAIN_IMAGES di:"
echo "     project/configs/head_detection_baseline.py"
echo ""
echo "  4. Training:"
echo "     cd project"
echo "     python tools/train_net.py \\"
echo "         --config-file configs/head_detection_baseline.py \\"
echo "         --num-gpus 1"
echo "=============================================="
