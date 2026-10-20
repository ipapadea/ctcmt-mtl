# shift-detection-tta runtime image (single-task + MTL/CT-CMT extension).
#
# Pins match tools/install/setup_venv.sh exactly:
#   Python 3.9 / PyTorch 1.11.0 cu113 / cuDNN 8
#   mmengine @62f9504  mmcv==2.0.0rc4  mmdet==3.0.0rc5
#   mmcls @3ff80f5     mmtrack @9e4cb98
#
# The repo source is NOT baked in — bind-mount it at /workspace/shift-detection-tta
# so edits on the host take effect immediately (see docker/run.sh).

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FORCE_CUDA=1 \
    MMCV_WITH_OPS=1 \
    TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6+PTX" \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# -------- System deps (image/video libs + build tools for mmcv ops) ---------
# NVIDIA CUDA APT repos in this base image have an expired signing key and we
# don't need CUDA-from-apt anyway, so drop those lists before apt update.
RUN rm -f /etc/apt/sources.list.d/cuda*.list /etc/apt/sources.list.d/nvidia*.list \
 && apt-get update && apt-get install -y --no-install-recommends \
        git wget curl ca-certificates build-essential ninja-build \
        ffmpeg libsm6 libxext6 libgl1 libglib2.0-0 \
        openssh-client bzip2 unzip \
 && rm -rf /var/lib/apt/lists/*

# -------- Miniconda w/ Python 3.9 -------------------------------------------
ENV CONDA_DIR=/opt/conda \
    PATH=/opt/conda/bin:$PATH
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-2-Linux-x86_64.sh -o /tmp/mc.sh \
 && bash /tmp/mc.sh -b -p $CONDA_DIR \
 && rm /tmp/mc.sh \
 && conda clean -afy

# -------- PyTorch 1.11.0 cu113 (matches conda install in setup_venv.sh) -----
RUN pip install --upgrade pip setuptools wheel openmim \
 && pip install \
        torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
        --extra-index-url https://download.pytorch.org/whl/cu113

# -------- OpenMMLab stack (order matters: engine -> cv -> det -> track) -----
# --no-build-isolation is required because the pinned mmengine / mmtrack / mmcls
# commits import ``pkg_resources`` in their setup.py, which is not present in
# pip's isolated build env.
RUN pip install --no-build-isolation \
      "git+https://github.com/open-mmlab/mmengine.git@62f9504d701251db763f56658436fd23a586fe25" \
 && mim install "mmcv==2.0.0rc4" \
 && mim install "mmdet==3.0.0rc5" \
 && pip install --no-build-isolation \
      "git+https://github.com/open-mmlab/mmclassification.git@3ff80f5047fe3f3780a05d387f913dd02999611d" \
      "git+https://github.com/open-mmlab/mmtracking.git@9e4cb98a3cdac749242cd8decb3a172058d4fd6e" \
      "git+https://github.com/JonathonLuiten/TrackEval.git" \
      "git+https://github.com/scalabel/scalabel.git"

# -------- Repo runtime deps (pinned in requirements/) -----------------------
COPY requirements/ /tmp/requirements/
COPY requirements.txt /tmp/requirements.txt
RUN cd /tmp && pip install -r requirements.txt \
 && rm -rf /tmp/requirements /tmp/requirements.txt

# `transformers` is imported eagerly by no_adaptation_yolox_dino_adapter.py
# (prior single-task code) but is not listed in requirements.txt. Pin below
# 4.44 to keep torch 1.x support (later versions require torch >= 2.1).
# `cityscapesscripts` is needed for `createTrainIdLabelImgs` (seg mask prep).
# `yapf<0.40` because mmengine 0.8 calls FormatCode(..., verify=True) which
# was removed in yapf 0.40+.
RUN pip install "transformers==4.36.2" "cityscapesscripts==2.2.2" "yapf<0.40"

# Where the repo will be bind-mounted at runtime.
WORKDIR /workspace/shift-detection-tta

# Make the bind-mounted repo importable without pip install (no entry_points).
ENV PYTHONPATH=/workspace/shift-detection-tta:${PYTHONPATH}

# Datasets bind-mount root (override with `-e DATA_ROOT=... -v <host>:<DATA_ROOT>`).
ENV DATA_ROOT=/workspace/data

CMD ["bash"]
