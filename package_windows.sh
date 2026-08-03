#!/usr/bin/env bash
# Windows distribution packer (only runtime-required files)
# Usage:
#   bash package_windows.sh
#   bash package_windows.sh Your-Custom-Name

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATE_TAG="$(date +%Y-%m-%d)"
DEFAULT_PACKAGE_NAME="GPT-SoVITS-Inference-Windows-${DATE_TAG}"
PACKAGE_NAME="${1:-$DEFAULT_PACKAGE_NAME}"

TMP_ROOT="$SCRIPT_DIR/.pack_tmp"
STAGE_DIR="$TMP_ROOT/$PACKAGE_NAME"
ZIP_PATH="$SCRIPT_DIR/${PACKAGE_NAME}.zip"

echo "=========================================="
echo "Windows distribution packing started"
echo "Project dir : $SCRIPT_DIR"
echo "Output file : $ZIP_PATH"
echo "=========================================="

rm -rf "$TMP_ROOT"
mkdir -p "$STAGE_DIR"

# 1) Copy core runtime directories
cp -R "$SCRIPT_DIR/GPT_SoVITS" "$STAGE_DIR/"
cp -R "$SCRIPT_DIR/tools" "$STAGE_DIR/"

# 2) Copy runtime scripts and docs
cp "$SCRIPT_DIR/requirements.txt" "$STAGE_DIR/"
cp "$SCRIPT_DIR/start_inference.bat" "$STAGE_DIR/"
cp "$SCRIPT_DIR/setup_windows.bat" "$STAGE_DIR/"
cp "$SCRIPT_DIR/clean_for_sharing.bat" "$STAGE_DIR/"
cp "$SCRIPT_DIR/README.md" "$STAGE_DIR/"
cp "$SCRIPT_DIR/window.md" "$STAGE_DIR/"

# 3) Create required runtime directories (keep empty dirs in package)
mkdir -p "$STAGE_DIR/output/batch_result"
mkdir -p "$STAGE_DIR/GPT_weights_v2"
mkdir -p "$STAGE_DIR/SoVITS_weights_v2"
mkdir -p "$STAGE_DIR/GPT_weights"
mkdir -p "$STAGE_DIR/SoVITS_weights"

# 4) Remove files that should not be distributed
find "$STAGE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "$STAGE_DIR" -type f \( -name "*.pyc" -o -name "*.pyo" -o -name "*.log" -o -name "*.tmp" \) -delete
find "$STAGE_DIR/output" -type f -delete

# 5) Build zip
rm -f "$ZIP_PATH"
(
  cd "$TMP_ROOT"
  zip -r "$ZIP_PATH" "$PACKAGE_NAME" >/dev/null
)

# 6) Clean temp
rm -rf "$TMP_ROOT"

echo "=========================================="
echo "Pack completed: $ZIP_PATH"
echo "=========================================="
