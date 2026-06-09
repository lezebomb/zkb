from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contest_agent.training.classify_data import create_template, create_tiny_placeholder, prepare_custom


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare ImageFolder data for 8 contest scene classes.")
    parser.add_argument("--mode", choices=["template", "custom", "tiny-smoke"], required=True)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.mode == "template":
        out = create_template(args.output)
    elif args.mode == "tiny-smoke":
        out = create_tiny_placeholder(args.output)
    else:
        if not args.input:
            raise SystemExit("--input is required for custom mode")
        out = prepare_custom(args.input, args.output)
    print(f"prepared classify dataset: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

