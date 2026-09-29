#!/usr/bin/env bash
# =============================================================================
# Project: RAW ISP | Task: Download SID Sony Dataset
# Timestamp: 2026-09-20 | Author: Guozh | Version: Data-SID-Sony-v0.1.0
# =============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="$PROJECT_ROOT/data/raw/SID"
TARGET="$TARGET_DIR/Sony2025.zip"
URL="https://storage.googleapis.com/isl-datasets/SID/Sony2025.zip"

mkdir -p "$TARGET_DIR"
available_kb="$(df -Pk "$TARGET_DIR" | awk 'NR==2 {print $4}')"
if [ "$available_kb" -lt 35000000 ]; then
  echo "ERROR: less than 35 GB free in $TARGET_DIR" >&2
  exit 1
fi

echo "Downloading/resuming: $URL"
echo "Target: $TARGET"
echo "Use Ctrl-C to pause; rerun this script to resume."
wget -c --show-progress -O "$TARGET" "$URL"

echo "Download command finished. Verify the archive before extraction."
ls -lh "$TARGET"
