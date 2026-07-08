#!/usr/bin/env bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate GraspSAM

pip uninstall -y torch torchvision torchaudio groundingdino || true

pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu117 \
  torch==2.0.1 \
  torchvision==0.15.2 \
  torchaudio==2.0.2

pip install open3d

apt-get update
apt-get install -y libgl1 libglib2.0-0 ca-certificates

rm -f /etc/apt/sources.list.d/cuda.list \
      /etc/apt/sources.list.d/nvidia-ml.list || true
apt-get update

conda install -y -c conda-forge libstdcxx-ng libgcc-ng
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
apt-get update
apt-get install -y \
    libgl1 \
    libglib2.0-0

apt-get update


cd ~/graspnet_ws/src/graspsam_ros2/compare_GraspSAM/GroundingDINO
pip install -e .
python setup.py build_ext --inplace

python - <<'PY'
import torch
from groundingdino.models import build_model
print("Environment OK")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
PY

cd ~/graspnet_ws/src/graspsam_ros2/compare_GraspSAM