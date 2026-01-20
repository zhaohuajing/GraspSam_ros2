FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV FORCE_CUDA=1

# --------------------
# System dependencies
# --------------------
RUN apt-get update && apt-get install -y \
    git \
    wget \
    ca-certificates \
    build-essential \
    ninja-build \
    libgl1 \
    libglib2.0-0 \
    tree \
    && rm -rf /var/lib/apt/lists/*

# --------------------
# Miniconda
# --------------------
ENV CONDA_DIR=/opt/conda
RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh \
    -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

ENV PATH=$CONDA_DIR/bin:$PATH
RUN conda init bash

# --------------------
# Conda env
# --------------------
RUN conda create -y -n GraspSAM python=3.8 && \
    conda clean -afy

SHELL ["bash", "-lc"]

# --------------------
# PyTorch (CUDA pinned)
# --------------------
RUN conda activate GraspSAM && \
    python -m pip install --upgrade pip && \
    python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu117 \
      torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 
      
RUN python -m pip install ultralytics

# --------------------
# Copy project
# --------------------
WORKDIR /root/graspnet_ws/src/GraspSam_ros2/compare_GraspSAM
COPY . .

# --------------------
# Python deps (safe layer)
# --------------------
RUN conda activate GraspSAM && \
    python -m pip install --no-cache-dir \
      -r requirements/requirements.workable.txt

# --------------------
# GroundingDINO (C++/CUDA)
# --------------------
WORKDIR /root/graspnet_ws/src/GraspSam_ros2/compare_GraspSAM/GroundingDINO
RUN conda activate GraspSAM && \
    export CC=/usr/bin/gcc-9 && \
    export CXX=/usr/bin/g++-9 && \
    python -m pip install -e .

# --------------------
# Final
# --------------------
WORKDIR /root/graspnet_ws/src/GraspSam_ros2/compare_GraspSAM
CMD ["bash"]
