#FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git wget ca-certificates bzip2 \
    build-essential \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Miniforge (no Anaconda ToS issues) + conda-forge-only
RUN wget -q https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -O /tmp/miniforge.sh && \
    bash /tmp/miniforge.sh -b -p /opt/conda && \
    rm -f /tmp/miniforge.sh

ENV PATH=/opt/conda/bin:$PATH

RUN conda config --system --set channel_priority strict && \
    conda config --system --remove channels defaults || true && \
    conda config --system --add channels conda-forge

# Create env (Python 3.8)
RUN conda create -y -n GraspSAM python=3.8 pip && conda clean -afy

# Make this env the default python/pip without needing "conda activate"
ENV CONDA_DEFAULT_ENV=GraspSAM
ENV PATH=/opt/conda/envs/GraspSAM/bin:$PATH

# Copy requirements (IMPORTANT: build context must include compare_GraspSAM/)
COPY compare_GraspSAM/requirements/requirements.clean.txt compare_GraspSAM/requirements.cleaner.txt

# Patch requirements for Python 3.8 + remove packages we will install via conda-forge
# - MultiScaleDeformableAttention has no pip wheel -> must NOT be pip-installed
# - Many pinned versions (matplotlib 3.8.3, pandas 2.2.1, sklearn 1.4.x, skimage 0.22, scipy 1.12, pyarrow 17)
#   are not available for py38 -> install compatible builds from conda-forge instead.
RUN sed -i \
    -e 's/^contourpy==1\.2\.0$/contourpy==1.1.1/' \
    -e 's/^lightning==2\.4\.0$/lightning==2.0.9.post0/' \
    -e 's/^pytorch-lightning==2\.4\.0$/pytorch-lightning==2.0.9.post0/' \
    compare_GraspSAM/requirements/requirements.cleaner.txt && \
    sed -i \
    -e '/^MultiScaleDeformableAttention==/d' \
    -e '/^imagecodecs==/d' \
    -e '/^matplotlib==/d' \
    -e '/^pandas==/d' \
    -e '/^scikit-image==/d' \
    -e '/^scikit-learn==/d' \
    -e '/^scipy==/d' \
    -e '/^pyarrow==/d' \
    -e '/^opencv-python==/d' \
    -e '/^opencv-python-headless==/d' \
    -e '/^ray==/d' \
    -e '/^tifffile==/d' \
    -e '/^albumentations==/d' \
    compare_GraspSAM/requirements/requirements.cleaner.txt  
    

# Install "common offenders" from conda-forge (py38-compatible)
# Pin ranges chosen to stay on versions that still publish py38 builds.
RUN conda install -y -n GraspSAM \
      "numpy<1.25" \
      "scipy<1.11" \
      "pandas<2.1" \
      "scikit-learn<1.4" \
      "scikit-image<0.22" \
      "matplotlib<3.8" \
      opencv \
      imagecodecs \
      tifffile \
      "ray-default<2.11" \
      albumentations \
    && conda clean -afy  

# # Install the remaining python deps (after patching)
# RUN python -m pip install -r /tmp/requirements.clean.txt

RUN apt-get update && apt-get install -y \
    build-essential git wget ca-certificates bzip2 \
    ninja-build cmake \
    && rm -rf /var/lib/apt/lists/*


# after creating and activating the env, do:
RUN /opt/conda/bin/conda run -n GraspSAM python -m pip uninstall -y torch torchvision torchaudio || true && \
    /opt/conda/bin/conda run -n GraspSAM python -m pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cu117 \
      torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2

ENV LD_LIBRARY_PATH=/opt/conda/envs/GraspSAM/lib:/opt/conda/lib:${LD_LIBRARY_PATH}
# LD_LIBRARY_PATH=/opt/conda/envs/GraspSAM/lib:/opt/conda/lib:${LD_LIBRARY_PATH} python -c "..."

# To fix OpenCV / libbGL.so.1 import error
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y tree && rm -rf /var/lib/apt/lists/*

# Install GroundingDINO from source
WORKDIR /root/graspnet_ws/src/GraspSam_ros2/compare_GraspSAM/GroundingDINO
RUN pip install -e .


# Default: drop into an interactive shell with env active (PATH already points to env)
WORKDIR /workspace
CMD ["bash", "-lc", "python --version && which python && exec bash"]
