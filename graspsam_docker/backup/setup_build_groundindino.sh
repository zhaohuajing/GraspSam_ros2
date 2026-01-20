#!/usr/bin/env bash


# NOTE: Run once per image or only if GroundingDINO source changes

set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate GraspSAM

cd ~/graspnet_ws/src/graspsam_ros2/compare_GraspSAM/GroundingDINO
pip install -e .
python setup.py build_ext --inplace
