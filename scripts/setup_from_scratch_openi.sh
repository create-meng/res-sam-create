#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${1:-https://openi.pcl.ac.cn/create-meng/res-sam.git}"
TARGET_DIR="${2:-$PWD/res-sam}"
SAM_WEIGHT_URL="${3:-https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth}"

echo "[1/2] Cloning repo from OpenI with LFS smudge disabled..."
export GIT_LFS_SKIP_SMUDGE=1
if [ -d "$TARGET_DIR/.git" ]; then
  echo "Repo already exists at $TARGET_DIR, skipping clone."
  echo "Updating existing repo..."
  git -C "$TARGET_DIR" pull
else
  git clone "$REPO_URL" "$TARGET_DIR"
fi

echo "[2/2] Running OpenI post-clone setup..."
bash "$TARGET_DIR/scripts/setup_after_clone_openi.sh" "$TARGET_DIR" "$SAM_WEIGHT_URL"

echo
echo "OpenI full setup completed."
