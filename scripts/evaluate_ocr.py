from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contest_agent.evaluation.ocr_eval import evaluate_ocr


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate OCR predictions on labels.txt data.")
    parser.add_argument("--pred-backend", default="api", choices=["api", "fallback"])
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="runs/eval/ocr_metrics.json")
    args = parser.parse_args()
    metrics = evaluate_ocr(args.data, args.output, backend=args.pred_backend)
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

