#!/usr/bin/env bash
set -euo pipefail

BASE="/home/omego/Documents/Hybrid_Stego_Eva/maxSRM_EC"
VENV="${BASE}/venv"

mkdir -p "${BASE}"/{scripts,runs,features,logs,pairs}

python3 -m venv "${VENV}"
source "${VENV}/bin/activate"

python -m pip install --upgrade pip wheel setuptools

python -m pip install \
  numpy \
  pandas \
  scikit-learn \
  pillow \
  imageio \
  tqdm \
  joblib \
  psutil \
  sealwatch

echo "Environment ready."
echo "Activate with:"
echo "source ${VENV}/bin/activate"
