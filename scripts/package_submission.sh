#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-dist}"
PACKAGE_NAME="${2:-contest_agent_submission}"

mkdir -p "$OUT_DIR"
python scripts/package_submission.py "$OUT_DIR" "$PACKAGE_NAME"
