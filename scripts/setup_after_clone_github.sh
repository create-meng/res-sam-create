#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"

cd "$REPO_DIR"

echo "[1/5] Installing CUDA 11.8 torch wheels..."
python -m pip install \
  "torch==2.3.1+cu118" \
  "torchvision==0.18.1+cu118" \
  "torchaudio==2.3.1+cu118" \
  --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu118 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 120

echo "[2/5] Installing remaining Python dependencies..."
grep -vi "segment-anything" requirements-gpu.txt \
  | grep -Ev "^(--index-url|--extra-index-url|torch|torchvision|torchaudio)" \
  | python -m pip install -r /dev/stdin \
      -i https://pypi.tuna.tsinghua.edu.cn/simple \
      --timeout 120

echo "[3/5] Installing segment-anything..."
python -m pip install "segment-anything==1.0" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 120

echo "[4/5] Ensuring git-lfs is available..."
if ! git lfs version >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    conda install -c conda-forge git-lfs -y
  else
    echo "git-lfs is not available and conda was not found."
    echo "Please install git-lfs first, then rerun this script."
    exit 1
  fi
fi

echo "[5/5] Pulling SAM checkpoint via Git LFS..."
git lfs install
git lfs pull

echo
echo "GitHub post-clone setup completed."
