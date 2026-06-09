from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contest_agent.training.common import env_bool


def main() -> int:
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO for contest detect labels.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="smoke")
    parser.add_argument("--export", default="models/detect/best.pt")
    parser.add_argument("--allow-long-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    print("detect training plan")
    print(f"model={args.model} data={args.data} epochs={args.epochs} imgsz={args.imgsz} batch={args.batch} device={args.device}")
    if args.dry_run:
        print("dry-run: no training executed")
        return 0

    if args.epochs > 3 and not (args.allow_long_run or env_bool("RUN_FULL_TRAIN", False)):
        print("refusing long detect training because RUN_FULL_TRAIN=false; pass --allow-long-run or set RUN_FULL_TRAIN=true")
        return 2
    if not Path(args.data).exists():
        print(f"data yaml missing: {args.data}")
        return 2
    if not Path(args.model).exists():
        print(f"model missing: {args.model}")
        print("prepare it with scripts/prepare_detect_model.py or install requirements-detect.txt")
        return 2
    try:
        from ultralytics import YOLO
    except Exception as exc:
        print(f"ultralytics import failed: {exc}")
        print("install: pip install -r requirements-detect.txt")
        return 2

    model = YOLO(args.model)
    result = model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, device=args.device, project=args.project, name=args.name)
    save_dir = Path(getattr(result, "save_dir", Path(args.project) / args.name))
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"training finished but best.pt was not found: {best}")
        return 3
    target = Path(args.export)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, target)
    print(f"exported: {target}")
    print("DETECT_BACKEND=ultralytics")
    print(f"MODEL_DETECT_PATH={target.as_posix()}")
    print("ALLOW_MODEL_AUTO_DOWNLOAD=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

