#!/usr/bin/env bash
# OpenI 镜像未必托管约 1.2GB 的 SAM ViT-L 权重；若 LFS 不可用，请用默认 URL 经 aria2 下载到 sam/sam_vit_l_0b3195.pth。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SAM_WEIGHT_URL="${2:-https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth}"
STATE_DIR="$REPO_DIR/.setup_state"
STATE_FILE="$STATE_DIR/openi_after_clone.done"

cd "$REPO_DIR"
mkdir -p "$STATE_DIR"

mark_done() {
  local step="$1"
  printf '%s\n' "$step" >> "$STATE_FILE"
}

is_done() {
  local step="$1"
  test -f "$STATE_FILE" && grep -Fxq "$step" "$STATE_FILE"
}

echo "[1/6] Installing CUDA 11.8 torch wheels..."
if is_done "torch"; then
  echo "  - already done, skipping"
else
  python -m pip install \
    "torch==2.3.1+cu118" \
    "torchvision==0.18.1+cu118" \
    "torchaudio==2.3.1+cu118" \
    --index-url https://mirrors.nju.edu.cn/pytorch/whl/cu118 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple \
    --timeout 120
  mark_done "torch"
fi

echo "[2/6] Installing remaining Python dependencies..."
if is_done "deps"; then
  echo "  - already done, skipping"
else
  grep -vi "segment-anything" requirements-gpu.txt \
    | grep -Ev "^(--index-url|--extra-index-url|torch|torchvision|torchaudio)" \
    | python -m pip install -r /dev/stdin \
        -i https://pypi.tuna.tsinghua.edu.cn/simple \
        --extra-index-url https://pypi.org/simple \
        --timeout 120
  mark_done "deps"
fi

echo "[3/6] Installing segment-anything..."
if is_done "segment_anything"; then
  echo "  - already done, skipping"
else
  python -m pip install "segment-anything==1.0" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://pypi.org/simple \
    --timeout 120
  mark_done "segment_anything"
fi

echo "[4/6] Ensuring git-lfs is available..."
if is_done "git_lfs"; then
  echo "  - already done, skipping"
else
  if ! git lfs version >/dev/null 2>&1; then
    if command -v conda >/dev/null 2>&1; then
      conda install -c conda-forge git-lfs -y
    else
      echo "git-lfs is not available and conda was not found."
      echo "Please install git-lfs first, then rerun this script."
      exit 1
    fi
  fi
  mark_done "git_lfs"
fi

echo "[5/6] Pulling all LFS objects except the SAM checkpoint..."
if is_done "lfs_pull"; then
  echo "  - already done, skipping"
else
  git lfs install
  git lfs pull --exclude="sam/sam_vit_l_0b3195.pth"
  mark_done "lfs_pull"
fi

echo "[6/6] Downloading the SAM checkpoint directly..."
if is_done "sam_weight" && [ -s sam/sam_vit_l_0b3195.pth ]; then
  echo "  - already done, skipping"
else
  # 权重必须与 sam/sam.py 同级：仓库根下的 sam/sam_vit_l_0b3195.pth（克隆后已有 sam/，勿放到别的路径）
  if [ ! -d sam ] || [ ! -f sam/sam.py ]; then
    echo "ERROR: expected repo-root sam/ with sam/sam.py (clone layout). cwd=$(pwd)" >&2
    exit 1
  fi
  apt-get update && apt-get install -y aria2
  aria2c -x 16 -s 16 -k 1M \
    -o sam_vit_l_0b3195.pth \
    -d sam \
    "$SAM_WEIGHT_URL"
  mark_done "sam_weight"
fi

echo
echo "OpenI post-clone setup completed."
