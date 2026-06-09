from __future__ import annotations

import importlib.util
import platform
import sys
from pathlib import Path


def has_module(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def line(status: str, name: str, detail: str = "") -> None:
    suffix = f" - {detail}" if detail else ""
    print(f"[{status}] {name}{suffix}")


def main() -> int:
    line("OK", "python", f"{platform.python_version()} ({sys.executable})")
    for module, package in [("ultralytics", "requirements-detect.txt"), ("torch", "requirements-train.txt"), ("torchvision", "requirements-train.txt"), ("paddleocr", "requirements-ocr.txt"), ("paddle", "requirements-ocr.txt")]:
        line("OK" if has_module(module) else "WARN", module, "installed" if has_module(module) else f"install {package} if needed")
    if has_module("torch"):
        try:
            import torch

            line("OK" if torch.cuda.is_available() else "WARN", "gpu", "cuda available" if torch.cuda.is_available() else "CPU only")
        except Exception as exc:
            line("WARN", "gpu", str(exc))
    else:
        line("WARN", "gpu", "torch not installed")
    for path in ["models/detect/yolo11n.pt", "data/processed/detect", "data/processed/classify", "data/processed/ocr/synthetic"]:
        p = Path(path)
        line("OK" if p.exists() else "WARN", path, "exists" if p.exists() else "missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

