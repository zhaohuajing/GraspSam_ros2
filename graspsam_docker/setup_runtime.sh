#!/usr/bin/env bash
set -euo pipefail

echo "=========================================="
echo "[GraspSAM] Self-healing runtime setup"
echo "=========================================="

############################################
# 0. Load conda correctly (non-interactive)
############################################
if [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  echo "[ERROR] conda.sh not found"
  exit 1
fi

conda activate GraspSAM

############################################
# 1. Hard guard: remove ghost torch installs
############################################
echo "[1/8] Cleaning ghost PyTorch installs (if any)"

SITE_PKGS="$CONDA_PREFIX/lib/python3.8/site-packages"

rm -rf \
  "$SITE_PKGS/torch" \
  "$SITE_PKGS/torch-"*.dist-info \
  "$SITE_PKGS/torchvision" \
  "$SITE_PKGS/torchvision-"*.dist-info \
  "$SITE_PKGS/torchaudio"* \
  "$SITE_PKGS/triton"* || true

############################################
# 2. Install correct PyTorch (cu117)
############################################
echo "[2/8] Installing PyTorch 2.0.1 + CUDA 11.7"

pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu117 \
  torch==2.0.1 \
  torchvision==0.15.2 \
  torchaudio==2.0.2

############################################
# 3. Verify PyTorch ABI
############################################
echo "[3/8] Verifying PyTorch ABI"

python - <<'PY'
import torch, sys
assert torch.__version__.startswith("2.0.1"), torch.__version__
assert torch.version.cuda == "11.7", torch.version.cuda
assert torch.cuda.is_available(), "CUDA not available"
print("✓ torch OK:", torch.__version__, "cuda", torch.version.cuda)
PY

############################################
# 4. Fix GLIBCXX runtime (hard enforcement)
############################################
echo "[4/8] Fixing GLIBCXX runtime (hard enforcement)"

conda install -y -c conda-forge libstdcxx-ng libgcc-ng

# Force conda runtime FIRST
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

echo "Checking conda libstdc++:"
strings "$CONDA_PREFIX/lib/libstdc++.so.6" | grep GLIBCXX_3.4.29 || {
  echo "[ERROR] Conda libstdc++ missing GLIBCXX_3.4.29"
  exit 1
}

echo "Checking system libstdc++ (expected to be older):"
strings /lib/x86_64-linux-gnu/libstdc++.so.6 | grep GLIBCXX_3.4.29 || true

echo "GLIBCXX runtime enforced via LD_LIBRARY_PATH"


############################################
# 5. Ensuring libGL runtime (safe apt fix)
############################################
echo "[5/8] Ensuring libGL runtime"

# Remove broken NVIDIA apt repos (safe in containers)
rm -f /etc/apt/sources.list.d/cuda.list \
      /etc/apt/sources.list.d/nvidia-ml.list || true

apt-get update || true

apt-get install -y \
    ca-certificates \
    libgl1 \
    libglib2.0-0

ldconfig

# Verify libGL
ldconfig -p | grep libGL.so.1 || {
  echo "[ERROR] libGL.so.1 not found"
  exit 1
}

echo "libGL runtime OK"


############################################
# 6. Install GroundingDINO (editable)
############################################
echo "[6/8] Installing GroundingDINO"

GDINO_DIR="$HOME/graspnet_ws/src/graspsam_ros2/compare_GraspSAM/GroundingDINO"

if [[ ! -f "$GDINO_DIR/setup.py" ]]; then
  echo "[ERROR] GroundingDINO not found at:"
  echo "        $GDINO_DIR"
  exit 1
fi

cd "$GDINO_DIR"

pip uninstall -y groundingdino || true
pip install -e .

############################################
# 7. Build CUDA extension (in-place)
############################################
echo "[7/8] Building GroundingDINO CUDA extension"

export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9
export CUDA_HOME=/usr/local/cuda
export FORCE_CUDA=1

python setup.py build_ext --inplace

############################################
# 8. Final sanity check
############################################
echo "[8/8] Final sanity check"

python - <<'PY'
import torch
from groundingdino.models import build_model
print("✓ GroundingDINO build_model OK")
print("✓ Environment fully ready")
PY

echo "=========================================="
echo " GraspSAM runtime setup COMPLETE"
echo "=========================================="

# MANUALLY TYPE IN TERMINAL:
# source /opt/conda/etc/profile.d/conda.sh
# conda activate GraspSAM
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
# echo $LD_LIBRARY_PATH