from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert reviewed detection data into YOLO layout placeholder.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--format", choices=["yolo"], default="yolo")
    args = parser.parse_args()
    src = Path(args.input)
    out = Path(args.output)
    if not src.exists():
        print(f"input missing: {src}")
        return 2
    out.mkdir(parents=True, exist_ok=True)
    print("YOLO format is already the project target. Use prepare_detect_dataset.py --mode custom for validation/copy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

