from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".ultralytics"))
(ROOT / ".ultralytics").mkdir(parents=True, exist_ok=True)

from contest_agent.training.common import env_bool


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a local Ultralytics YOLO detect model.")
    parser.add_argument("--model", default="yolo11n.pt", help="Ultralytics model name or local .pt path")
    parser.add_argument("--output", default="models/detect/yolo11n.pt")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        print(f"exists: {output}")
        print("Before competition set ALLOW_MODEL_AUTO_DOWNLOAD=false and confirm /debug/status model_exists=true.")
        return 0

    allowed = args.allow_download or env_bool("ALLOW_MODEL_AUTO_DOWNLOAD", False)
    if not allowed:
        print("model missing and auto download is disabled")
        print(f"manual: place {args.model} at {output}")
        print("or run with --allow-download / ALLOW_MODEL_AUTO_DOWNLOAD=true before competition")
        return 0

    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"ultralytics import failed: {exc}")
        print("install: pip install -r requirements-detect.txt")
        return 2

    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        model = YOLO(args.model)
        candidates = [Path(args.model)]
        ckpt_path = getattr(model, "ckpt_path", None)
        if ckpt_path:
            candidates.append(Path(ckpt_path))
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                shutil.copy2(candidate, output)
                print(f"copied: {candidate} -> {output}")
                break
        else:
            try:
                model.save(str(output))
                print(f"saved model to {output}")
            except Exception as exc:
                print(f"download/load succeeded but could not save weight file: {exc}")
                return 3
    except Exception as exc:
        print(f"YOLO model preparation failed: {exc}")
        return 3

    print("Before competition set ALLOW_MODEL_AUTO_DOWNLOAD=false and confirm /debug/status model_exists=true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
