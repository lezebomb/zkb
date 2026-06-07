#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8080}"
python scripts/smoke_test.py "$BASE_URL"
