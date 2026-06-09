from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contest_agent.evaluation.classify_eval import evaluate_classify


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a local classifier.pt on ImageFolder val data.")
    parser.add_argument("--model", default="models/classify/classifier.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="runs/eval/classify_metrics.json")
    args = parser.parse_args()
    metrics = evaluate_classify(args.model, args.data, args.output)
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

