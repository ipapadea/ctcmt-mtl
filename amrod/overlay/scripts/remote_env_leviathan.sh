#!/usr/bin/env bash
set -Eeo pipefail   # no -u: conda activation hooks (gcc) read unset variables

# Builds the conda environment used on the remote machine ("leviathan", no docker, NVIDIA
# driver 470 = CUDA 11.4).  The local docker image uses torch 2.1.2+cu121, which needs a
# driver >= 525, so here torch 2.1.2+cu118 is used (same torch/torchvision versions, other
# CUDA runtime).  Run ON the remote machine:
#
#   bash /mnt/data_storage/ilias/AMROD/scripts/remote_env_leviathan.sh
#
# Result: /mnt/data_storage/ilias/envs/amrod (conda prefix) with the AMROD detectron2 fork
# built for sm_86 (RTX 3090).  Package versions are the ones pinned in the Dockerfile.

ROOT="${ROOT:-/mnt/data_storage/ilias/AMROD}"
ENV="${ENV:-/mnt/data_storage/ilias/envs/amrod}"
source "$HOME/anaconda3/etc/profile.d/conda.sh"

if [[ ! -x "$ENV/bin/python" ]]; then
  conda create -y -p "$ENV" python=3.11
fi
conda activate "$ENV"
python -m pip install --upgrade pip "setuptools<70" wheel

pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118

# nvcc 11.8 + headers + a host compiler it accepts (the system has nvcc-less CUDA 11.1 and gcc 7.5)
conda install -y -p "$ENV" -c "nvidia/label/cuda-11.8.0" cuda-toolkit
conda install -y -p "$ENV" -c conda-forge "gxx_linux-64=11" ninja

pip install \
  numpy==1.24.3 scipy==1.14.0 Pillow==10.4.0 opencv-python==4.8.1.78 \
  imagecorruptions==1.1.2 pycocotools==2.0.7 pandas==2.0.3 matplotlib==3.7.3 tqdm==4.66.1 \
  cityscapesScripts==2.2.2 Shapely==2.0.5 fvcore==0.1.5.post20221221 iopath==0.1.9 \
  cloudpickle==3.0.0 omegaconf==2.3.0 hydra-core==1.3.2 termcolor==2.4.0 tabulate==0.9.0 \
  yacs black==24.8.0 openpyxl==3.1.5

export CUDA_HOME="$ENV"
export PATH="$ENV/bin:$PATH"
export CC="$ENV/bin/x86_64-conda-linux-gnu-gcc" CXX="$ENV/bin/x86_64-conda-linux-gnu-g++"
export TORCH_CUDA_ARCH_LIST="8.6" FORCE_CUDA=1
cd "$ROOT/detectron2"
pip install "setuptools<70"   # torch 2.1 cpp_extension needs pkg_resources
pip install --no-build-isolation -e .

python - <<'EOF'
import torch, torchvision, detectron2
from detectron2 import _C  # compiled ops
print("torch", torch.__version__, "tv", torchvision.__version__, "cuda ok:", torch.cuda.is_available(),
      "gpus:", torch.cuda.device_count(), "detectron2:", detectron2.__file__)
x = torch.randn(2, 3, device="cuda"); print("cuda matmul ok:", (x @ x.T).shape)
EOF
echo "ENV BUILD DONE"
