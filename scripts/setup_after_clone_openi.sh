#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${1:-$(cd "$SCRIPT_DIR/.." && pwd)}"
SAM_WEIGHT_URL="${2:-https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth}"
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
  git lfs pull --exclude="sam/sam_vit_b_01ec64.pth"
  mark_done "lfs_pull"
fi

echo "[6/6] Downloading the SAM checkpoint directly..."
if is_done "sam_weight" && [ -s sam/sam_vit_b_01ec64.pth ]; then
  echo "  - already done, skipping"
else
  mkdir -p sam
  curl -L "$SAM_WEIGHT_URL" -o sam/sam_vit_b_01ec64.pth
  mark_done "sam_weight"
fi

echo
echo "OpenI post-clone setup completed."
