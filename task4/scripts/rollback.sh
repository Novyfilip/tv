#!/bin/bash
# rollback.sh — Roll back to a specific model version
#
# Usage:
#   ./scripts/rollback.sh <model_filename>
#
# Example:
#   ./scripts/rollback.sh model_20240115_120000.pkl

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models}"
TARGET_MODEL="${1:?Usage: rollback.sh <model_filename>}"

MODEL_PATH="${MODEL_DIR}/${TARGET_MODEL}"

if [ ! -f "${MODEL_PATH}" ]; then
    echo "ERROR: Model file not found: ${MODEL_PATH}"
    echo "Available models:"
    ls -la "${MODEL_DIR}"/model_*.pkl 2>/dev/null || echo "  (none)"
    exit 1
fi

echo "Rolling back to: ${TARGET_MODEL}"
echo "${TARGET_MODEL}" > "${MODEL_DIR}/latest.txt"
echo "Done. Current model is now: ${TARGET_MODEL}"
