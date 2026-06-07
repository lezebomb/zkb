#!/usr/bin/env bash
set -euo pipefail

echo "Optional pre-race helper."
echo "This script is intentionally inert unless you export explicit model URLs."
echo "Do not run downloads during the official timed evaluation."

python - <<'PY'
from __future__ import annotations

import os
from pathlib import Path
from urllib import request

downloads = [
    ("CLASSIFY_MODEL_URL", Path("models/classifier.onnx")),
    ("DETECT_MODEL_URL", Path("models/detector.onnx")),
    ("OCR_MODEL_URL", Path("models/ocr/model.bin")),
]

for env_name, target in downloads:
    url = os.getenv(env_name, "").strip()
    if not url:
        print(f"{env_name}: skipped")
        continue
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {env_name} -> {target}")
    request.urlretrieve(url, target)
PY
