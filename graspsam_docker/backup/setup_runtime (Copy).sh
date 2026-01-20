#!/usr/bin/env bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate GraspSAM

echo "[1/4] Install C++ runtime"
conda install -y -c conda-forge libstdcxx-ng libgcc-ng

echo "[2/4] Install conda activation hook"
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
cat << 'EOF' > $CONDA_PREFIX/etc/conda/activate.d/ld_library_path.sh
#!/usr/bin/env bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF
chmod +x $CONDA_PREFIX/etc/conda/activate.d/ld_library_path.sh

echo "[3/4] Verify GLIBCXX"
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29

# echo "[4/5] Install GroundingDINO"
# cd ~/graspnet_ws/src/graspsam_ros2/compare_GraspSAM/GroundingDINO
# pip install -e .
# python setup.py build_ext --inplace

echo "[4/4] Sanity check"
python - <<'PY'
from groundingdino.models import build_model
print("Environment OK")
PY
