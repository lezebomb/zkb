from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contest_agent.training.ocr_synthetic import generate_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic OCR images and labels.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--count", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.count < 1:
        raise SystemExit("--count must be positive")
    out = generate_dataset(args.output, args.count, seed=args.seed)
    print(f"generated synthetic OCR dataset: {out}")
    print(f"labels: {out / 'labels.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

