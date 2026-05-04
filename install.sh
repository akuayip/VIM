#!/usr/bin/env bash
# install.sh — install project dependencies using the current Python environment
# Note: This script does NOT create or activate conda environments. Manage conda/python yourself.
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
LIBS_DIR="$ROOT_DIR/libs"

section() {
	echo
	echo "=================================================="
	echo "$1"
	echo "=================================================="
}

install_editable() {
	local package_dir="$1"
	local package_name="$2"
	if [ -d "$package_dir" ]; then
		echo "- Installing $package_name"
		python -m pip install --no-build-isolation -e "$package_dir"
	else
		echo "- Skipping $package_name (folder not found)"
	fi
}

section "Head Detection Project — Install Script"
echo "This script installs packages into the currently active Python environment."
echo "Please activate the target conda/env first."

echo
echo "Python runtime:"
python -c "import sys,platform; print(sys.executable, platform.python_version())"

section "[0/7] Preparing base build tools"
echo "Installing pip, setuptools, wheel, and packaging so editable/CUDA builds work in a fresh env."
python -m pip install --upgrade pip
python -m pip install --upgrade "setuptools>=68,<81" wheel packaging

echo "Checking pkg_resources availability..."
if python - <<'PY'
import pkg_resources
print("pkg_resources OK:", pkg_resources.__file__)
PY
then
	echo "- pkg_resources is available"
else
	echo "- pkg_resources missing, reinstalling setuptools"
	python -m pip install --upgrade "setuptools>=68,<81"
fi

section "[1/7] Installing PyTorch if needed"
echo "This project expects torch 2.1.1 + cu118."
if python - <<'PY'
import importlib.util
import sys
sys.exit(0 if importlib.util.find_spec('torch') is not None else 1)
PY
then
	echo "- PyTorch already installed"
else
	echo "- PyTorch not found, installing torch==2.1.1 / torchvision==0.16.1 / torchaudio==2.1.1"
	python -m pip install "torch==2.1.1" "torchvision==0.16.1" "torchaudio==2.1.1" --index-url https://download.pytorch.org/whl/cu118
fi

section "[2/7] Installing core Python dependencies"
echo "Pins numpy to 1.26.2 and installs fvcore + transformers 4.35.2 for compatibility."
python -m pip install "numpy==1.26.2" fvcore==0.1.5.post20221221
python -m pip install "transformers==4.35.2"

section "[3/7] Installing project requirements"
if [ -f "$ROOT_DIR/project/requirements.txt" ]; then
	echo "- Found project/requirements.txt"
	python -m pip install -r "$ROOT_DIR/project/requirements.txt"
else
	echo "- project/requirements.txt not found, skipping"
fi

section "[4/7] Installing editable libraries from libs"
install_editable "$LIBS_DIR/causal-conv1d" "causal-conv1d"
install_editable "$LIBS_DIR/mamba-1p1p1" "mamba-1p1p1"
install_editable "$LIBS_DIR/detectron2_vim" "detectron2_vim"

section "[5/7] Verifying important imports"
python - <<'PY'
import sys, importlib
print('PYTHON:', sys.executable, sys.version.split()[0])
try:
	import numpy as np
	print('NUMPY:', np.__version__)
except Exception as e:
	print('NUMPY: ERROR', e)
try:
	import torch
	print('TORCH:', torch.__version__, 'CUDA:', getattr(torch.version, 'cuda', None))
except Exception as e:
	print('TORCH: ERROR', e)
try:
	import pkg_resources
	print('PKG_RESOURCES: OK')
except Exception as e:
	print('PKG_RESOURCES: ERROR', e)
print('FVCORE:', importlib.util.find_spec('fvcore') is not None)
print('transformers:', importlib.util.find_spec('transformers') is not None)
print('detectron2:', importlib.util.find_spec('detectron2') is not None)
PY

section "[6/7] Checking dependency consistency"
echo "Running pip check (warnings only)."
python -m pip check || true

section "[7/7] Done"
echo "Installation completed in the current environment."

section "Next steps"
echo "1) Put pretrained weights in: project/checkpoints/vim_tiny_pretrained.pth"
echo "2) Train from project/: python tools/train_net.py --config-file configs/head_detection_baseline.py --num-gpus 1"
echo "3) If you create a new env, just re-run this script after activating it."
