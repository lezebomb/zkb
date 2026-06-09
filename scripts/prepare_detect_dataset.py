from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contest_agent.training.common import env_bool
from contest_agent.training.detect_data import prepare_coco8, prepare_coco_subset, prepare_custom


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare YOLO detect datasets for contest classes.")
    parser.add_argument("--mode", choices=["coco8", "coco-subset", "custom"], required=True)
    parser.add_argument("--input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--classes", nargs="*", default=[])
    parser.add_argument("--max-images-per-class", type=int, default=500)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    allow = args.allow_download or env_bool("ALLOW_DATASET_DOWNLOAD", False)

    if args.mode == "coco8":
        yaml_path = prepare_coco8(args.output, allow_download=allow)
    elif args.mode == "coco-subset":
        if not allow:
            print("COCO full/subset download is disabled; generating workspace and yaml only.")
        yaml_path = prepare_coco_subset(args.output, args.classes, args.max_images_per_class, allow_download=allow)
    else:
        if not args.input:
            raise SystemExit("--input is required for custom mode")
        yaml_path = prepare_custom(args.input, args.output)
    print(f"data yaml: {yaml_path}")
    print("note: 台灯 is not covered by COCO; prepare extra data or a second-stage strategy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

