#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${1:-$PWD}"

cd "$REPO_DIR"

python -m pip install --upgrade pip setuptools wheel \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

# Install CUDA 11.8 wheels from mirror first so pip does not fall back to
# the default PyTorch host during dependency resolution.
python -m pip install \
  "torch==2.3.1+cu118" \
  "torchvision==0.18.1+cu118" \
  "torchaudio==2.3.1+cu118" \
  --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu118 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 120

# Install the remaining Python packages from the Tsinghua PyPI mirror only.
grep -vi "segment-anything" requirements-gpu.txt \
  | grep -Ev "^(--index-url|--extra-index-url|torch|torchvision|torchaudio)" \
  | python -m pip install -r /dev/stdin \
      -i https://pypi.tuna.tsinghua.edu.cn/simple \
      --timeout 120

# segment-anything is available on PyPI, so this avoids a direct GitHub clone.
python -m pip install "segment-anything==1.0" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 120
