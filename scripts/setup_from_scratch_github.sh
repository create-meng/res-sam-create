#!/usr/bin/env bash
set -euo pipefail

REPO_URL="${1:-https://github.com/create-meng/res-sam-create.git}"
TARGET_DIR="${2:-$PWD/res-sam-create}"

echo "[1/2] Cloning repo from GitHub..."
if [ -d "$TARGET_DIR/.git" ]; then
  echo "Repo already exists at $TARGET_DIR, skipping clone."
else
  git clone "$REPO_URL" "$TARGET_DIR"
fi

echo "Updating repo..."
git -C "$TARGET_DIR" pull

echo "[2/2] Running GitHub post-clone setup..."
bash "$TARGET_DIR/scripts/setup_after_clone_github.sh" "$TARGET_DIR"

echo
echo "GitHub full setup completed."
