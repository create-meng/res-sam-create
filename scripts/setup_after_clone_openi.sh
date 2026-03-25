#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SAM_WEIGHT_URL="${2:-https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth}"

cd "$REPO_DIR"

echo "[1/6] Installing CUDA 11.8 torch wheels..."
python -m pip install \
  "torch==2.3.1+cu118" \
  "torchvision==0.18.1+cu118" \
  "torchaudio==2.3.1+cu118" \
  --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu118 \
  --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 120

echo "[2/6] Installing remaining Python dependencies..."
grep -vi "segment-anything" requirements-gpu.txt \
  | grep -Ev "^(--index-url|--extra-index-url|torch|torchvision|torchaudio)" \
  | python -m pip install -r /dev/stdin \
      -i https://pypi.tuna.tsinghua.edu.cn/simple \
      --timeout 120

echo "[3/6] Installing segment-anything..."
python -m pip install "segment-anything==1.0" \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --timeout 120

echo "[4/6] Ensuring git-lfs is available..."
if ! git lfs version >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    conda install -c conda-forge git-lfs -y
  else
    echo "git-lfs is not available and conda was not found."
    echo "Please install git-lfs first, then rerun this script."
    exit 1
  fi
fi

echo "[5/6] Pulling all LFS objects except the SAM checkpoint..."
git lfs install
git lfs pull --exclude="sam/sam_vit_b_01ec64.pth"

echo "[6/6] Downloading the SAM checkpoint directly..."
mkdir -p sam
curl -L "$SAM_WEIGHT_URL" -o sam/sam_vit_b_01ec64.pth

echo
echo "OpenI post-clone setup completed."
