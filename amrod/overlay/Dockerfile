# AMROD reproduction env (Decision 1B for D2 pipeline + 2b for corruption gen).
#
# What we pin EXACTLY (byte-equivalent to AMROD's corrupt.py output):
#   imagecorruptions==1.1.2, numpy==1.24.3, scipy==1.14.0, Pillow==10.4.0,
#   opencv-python==4.8.1.78  (scikit-image left unpinned since AMROD didn't pin it)
#
# What we deviate on (forced by hardware -- L40S/sm_89 has no torch 1.9+cu111
# kernels, and AMROD's own requirements.txt is internally inconsistent):
#   Python 3.11, torch 2.1.2+cu121 (matches our vgcmt container so any D2-vs-mmdet
#   delta is purely framework-level, not a torch-version confound).
#
# The AMROD Detectron2 fork is baked in at /opt/AMROD_d2 and pip installed there
# with sm_89 CUDA arch so custom C++ ops build for L40S.

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC \
    PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
    && add-apt-repository -y ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3.11-distutils \
        build-essential git curl ca-certificates \
        libglib2.0-0 libsm6 libxext6 libxrender-dev libgl1 \
        libjpeg-dev libpng-dev \
        ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && curl -fsSL https://bootstrap.pypa.io/get-pip.py | python3.11 \
    && python -m pip install --upgrade pip setuptools wheel

RUN pip install \
        torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 \
        --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
        numpy==1.24.3 \
        scipy==1.14.0 \
        Pillow==10.4.0 \
        opencv-python==4.8.1.78 \
        imagecorruptions==1.1.2 \
        pycocotools==2.0.7 \
        pandas==2.0.3 \
        matplotlib==3.7.3 \
        tqdm==4.66.1 \
        cityscapesScripts==2.2.2 \
        Shapely==2.0.5 \
        fvcore==0.1.5.post20221221 \
        iopath==0.1.9 \
        cloudpickle==3.0.0 \
        omegaconf==2.3.0 \
        hydra-core==1.3.2 \
        termcolor==2.4.0 \
        tabulate==0.9.0 \
        yacs \
        black==24.8.0 \
        openpyxl==3.1.5

# Bake AMROD's Detectron2 fork into the image at /opt/AMROD_d2 (survives the
# runtime bind-mount over /workspace/AMROD).
COPY detectron2 /opt/AMROD_d2/detectron2
RUN cd /opt/AMROD_d2/detectron2 && \
    TORCH_CUDA_ARCH_LIST="8.9" FORCE_CUDA=1 \
    pip install --no-build-isolation -e .

ARG USER_UID=1000
ARG USER_GID=1000
RUN groupadd -g ${USER_GID} amrod || true \
    && useradd -m -u ${USER_UID} -g ${USER_GID} -s /bin/bash amrod || true \
    && chown -R ${USER_UID}:${USER_GID} /opt/AMROD_d2

WORKDIR /workspace/AMROD

ENV TORCH_CUDA_ARCH_LIST="8.9" \
    FORCE_CUDA=1 \
    PYTHONPATH=/workspace/AMROD

CMD ["/bin/bash"]
