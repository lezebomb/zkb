from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contest_agent.evaluation.detect_eval import evaluate_detect


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate YOLO detect with center-hit proxy metric.")
    parser.add_argument("--model", default="models/detect/best.pt")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="runs/eval/detect_metrics.json")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    metrics = evaluate_detect(args.model, args.data, args.output, conf=args.conf, imgsz=args.imgsz, device=args.device)
    print(metrics)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

