from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contest_agent.evaluation.contest_api_eval import evaluate_contest_api


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate the running contest HTTP API.")
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--classify-data")
    parser.add_argument("--detect-data")
    parser.add_argument("--ocr-data")
    parser.add_argument("--output", default="runs/eval/api_eval_report.json")
    args = parser.parse_args()
    report = evaluate_contest_api(args.base_url, args.classify_data, args.detect_data, args.ocr_data, args.output)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

