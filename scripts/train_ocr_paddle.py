from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contest_agent.training.common import env_bool


def module_ok(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def main() -> int:
    parser = argparse.ArgumentParser(description="Dry-run or launch PaddleOCR recognition training.")
    parser.add_argument("--config", default="configs/ocr/paddleocr_rec_template.yml")
    parser.add_argument("--data", default="data/processed/ocr/synthetic")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-run", action="store_true")
    args = parser.parse_args()
    config = Path(args.config)
    labels = Path(args.data) / "labels.txt"
    print("ocr training plan")
    print(f"config={config} labels={labels}")
    checks = {
        "paddleocr": module_ok("paddleocr"),
        "paddle": module_ok("paddle"),
        "config": config.exists(),
        "labels": labels.exists(),
    }
    for key, ok in checks.items():
        print(f"{key}: {'OK' if ok else 'WARN'}")
    if args.dry_run or not (args.allow_run or env_bool("RUN_FULL_TRAIN", False)):
        print("dry-run/default: not launching PaddleOCR training")
        print("suggested: python -m paddle.distributed.launch tools/train.py -c configs/ocr/paddleocr_rec_template.yml")
        return 0
    if not all(checks.values()):
        print("cannot run PaddleOCR training until dependencies, config and labels are ready")
        return 2
    cmd = [sys.executable, "-m", "paddle.distributed.launch", "tools/train.py", "-c", str(config)]
    return subprocess.call(cmd, cwd=os.getcwd())


if __name__ == "__main__":
    raise SystemExit(main())

