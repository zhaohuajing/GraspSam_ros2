# GraspSAM + GroundingDINO Docker Setup

## Overview

This repository provides a **reproducible Docker-based environment** for running
GraspSAM with GroundingDINO, CUDA acceleration, and a Conda-managed Python stack.

The setup has been validated with:
- Ubuntu 20.04
- CUDA 11.7
- PyTorch 2.0.1 + torchvision 0.15.2
- Python 3.8 (Conda)
- NVIDIA GPU (Docker + nvidia-container-toolkit)

---

## Prerequisites

Ensure the host system has:

- Docker >= 20.x
- NVIDIA GPU driver compatible with CUDA 11.7
- NVIDIA Container Toolkit installed

Check GPU access:
```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:11.7.1-base-ubuntu20.04 nvidia-smi
```

---

## Build Docker Image

From the repository root (where `Dockerfile` is located):

```bash
docker build -t graspsam:cu117 .
```

The image includes:
- Conda environment `GraspSAM`
- CUDA-matched PyTorch
- Required system libraries (`libgl1`, `libglib2.0-0`, etc.)
- Python dependencies from `requirements.workable.txt`
- Editable install of **GroundingDINO** with compiled CUDA extensions

GroundingDINO should be cloned separately:
https://github.com/IDEA-Research/GroundingDINO


---

## Run Docker Container

Use the provided helper script:

```bash
./run_docker.sh
```

This script:
- Removes any old container with the same name
- Mounts the workspace and dataset folders (modify `./run_docker.sh` and set to your customer defined paths as needed) into the container
- Sets the working directory to `compare_GraspSAM`
- Launches an interactive shell

---

## Runtime Setup (Required After Container Start)

Due to a known interaction between:
- Conda-provided libstdc++ (required by PIL / OpenCV),
- System libstdc++ inside the Docker base image,
- CUDA / PyTorch / GroundingDINO C++ extensions,
activating the Conda environment alone is not sufficient. Specifically, `conda activate GraspSAM` may fall back to system `libstdc++.so.6`.

To ensure the correct Conda libraries are used at runtime, an additional setup script must be executed.

Recommended Workflow (One Command):

After starting the container with:
```bash
./run_docker.sh
```
run the following once per container session:
```bash
source setup_runtime.sh
```

This script:
- Activates the Conda environment
- Ensures compatible libstdc++ and libgcc are installed
- Forces the dynamic linker to prefer Conda libraries
- Verifies GroundingDINO can be imported successfully

Note: You must source the setup_runtime.sh script (i.e., `source setup_runtime.sh`), NOT execute it (i.e., `./setup_runtime.sh`). Otherwise, it runs in a subshell, and the `export LD_LIBRARY_PATH=...` and `conda activate GraspSAM` commands will NOT persist after the script exits.

---


## Verify Installation

Run the following checks:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())

from groundingdino.models import build_model
from groundingdino.models.GroundingDINO import ms_deform_attn

print("GroundingDINO build_model OK")
print("ms_deform_attn OK")
PY
```

Expected output:
- CUDA available: `True`
- No missing shared library errors
- No failed C++ extension warnings

---

## Notes on Dependencies

- **Do not reinstall packages manually inside the container**
- All dependencies are baked into the Docker image
- `requirements.workable.txt` is the authoritative Python dependency list
- Freeze files (`pip_freeze_*`, `conda_list_*`) are for reference only

---

## Common Issues

- Packages missing after reboot → Docker image not rebuilt
- `groundingdino` only imports inside repo → editable install missing
- `libGL.so.1` error → system libraries missing (handled in Dockerfile)

---

## Status

This environment is:
- Restart-safe
- CUDA-enabled
- Reproducible
- Ready for GraspSAM + GroundingDINO development
