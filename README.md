
# GraspSAM ROS 2 Server (Docker-backed)

## Overview

This package provides a ROS 2 service wrapper for GraspSAM, allowing GraspSAM inference to be triggered from ROS 2 while running the actual model inside a persistent Docker container via `subprocess`.

The design intentionally **decouples**:
- the ROS 2 runtime (Ubuntu 24.04 / ROS 2 Jazzy, host), and
- the GraspSAM runtime (Ubuntu 20.04, CUDA, Conda, PyTorch, GroundingDINO, etc., inside Docker)

This avoids dependency conflicts while enabling seamless integration with ROS 2 planning pipelines (e.g., MoveIt).


## High-Level Architecture

```bash
ROS 2 Client
   |
   |  (RunGraspSAM.srv)
   v
GraspSAM ROS 2 Server [graspsam_server.py]
   |
   |  subprocess [docker exec]
   v
Persistent Docker Container
   |
   |  conda activate GraspSAM
   |  python eval.py / train.py
   v
Grasp Outputs [JSON / NPY / PNG]
   |
   v
Returned to ROS 2 [paths + parsed grasps]

```

Key design choices:
- Docker container is persistent (sleep infinity), not launched per request
- Host workspace is bind-mounted into the container
- ROS 2 never imports ML dependencies directly
- All heavy libraries (CUDA, Torch, PIL, GL, etc.) live inside Docker

### Repository Structure (Relevant Parts)
```bash
graspsam_ros2/
├── graspsam_ros2/            # ROS 2 Python package
│   ├── graspsam_server.py
│   └── graspsam_server_minimal.py
├── srv/
│   └── RunGraspSAM.srv
├── msg/
│   └── Grasp.msg
├── compare_GraspSAM/         # Original GraspSAM codebase
│   ├── eval.py
│   ├── train.py
│   └── grasp_outputs/
├── GraspSam_docker/
│   └── run_docker.sh
```

---

## Provided ROS Interfaces

### Service: `graspsam_ros2/srv/RunGraspSAM`
### Request:
```bash
string dataset_root          # relative to compare_GraspSAM/
string checkpoint_path       # e.g. ./pretrained_checkpoint/mobile_sam.pt
string sam_encoder_type      # vit_t, vit_b, vit_h, ...
int32  no_grasps             # top-K grasps
bool   seen_set              # optional dataset split flag
```
### Response:
```bash
bool   success
string message               # stdout/stderr summary
string output_dir            # host path to grasp_outputs/sampleX
graspsam_ros2/msg/Grasp[] grasps   # parsed grasp results (if enabled)
```
### Message: `graspsam_ros2/msg/Grasp`

---

### Running the Server

```bash
source ~/graspnet_ws/install/setup.bash
ros2 run graspsam_ros2 graspsam_server.py
```
You should see:
```
[INFO] [graspsam_server]: GraspSAM server ready (runs eval.py inside Docker).
```

### Calling the Service
Example:
```bash
ros2 service call /run_graspsam graspsam_ros2/srv/RunGraspSAM "{
  dataset_root: './datasets/Jacquard_Samples/Samples/1a9fa4c269cfcc1b738e43095496b061/',
  checkpoint_path: './pretrained_checkpoint/mobile_sam.pt',
  sam_encoder_type: 'vit_t',
  no_grasps: 10,
  seen_set: false
}"
```

Outputs are written to:
```compare_GraspSAM/grasp_outputs/<sample_id>/```

### Output Layout
GraspSAM writes outputs under:

```bash
compare_GraspSAM/grasp_outputs/
  sampleX/
    note.txt
    sample_0_grasps.json
    sample_0_grasps.npy
    sample_0_maps.npz
    sample_0.png
    ...

```
---


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
- Mounts `~/graspnet_ws` into the container
- Sets the working directory to `compare_GraspSAM`
- Launches an interactive shell

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
