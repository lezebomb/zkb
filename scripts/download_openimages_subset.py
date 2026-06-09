from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contest_agent.training.common import env_bool, ensure_dir, write_text
from contest_agent.training.detect_data import contest_data_yaml


OI_TO_CN = {
    "Mobile phone": "手机",
    "Laptop": "笔记本电脑",
    "Mug": "水杯",
    "Dog": "狗",
    "Couch": "沙发",
    "Sofa": "沙发",
    "Table lamp": "台灯",
    "Desk lamp": "台灯",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download/prepare guarded Open Images detect subset.")
    parser.add_argument("--classes", nargs="*", required=True)
    parser.add_argument("--max-samples-per-class", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    out = ensure_dir(args.output)
    for rel in ("images/train", "images/val", "labels/train", "labels/val"):
        ensure_dir(out / rel)
    write_text(out / "data.yaml", contest_data_yaml(out))
    allowed = args.allow_download or env_bool("ALLOW_DATASET_DOWNLOAD", False)
    if not allowed:
        write_text(out / "download_todo.md", "Download blocked. Re-run with --allow-download after license review.\n")
        print(f"download disabled; workspace created: {out}")
        return 0
    try:
        import fiftyone.zoo as foz
    except Exception as exc:
        print(f"fiftyone import failed: {exc}")
        print("install: pip install -r requirements-data.txt")
        return 2
    valid = [name for name in args.classes if name in OI_TO_CN]
    skipped = [name for name in args.classes if name not in OI_TO_CN]
    if skipped:
        print(f"skipping unmapped/uncertain classes: {skipped}")
    if not valid:
        print("no mapped Open Images classes requested")
        return 2
    try:
        dataset = foz.load_zoo_dataset(
            "open-images-v7",
            split="validation",
            label_types=["detections"],
            classes=valid,
            max_samples=min(args.max_samples_per_class * len(valid), 1000),
            dataset_dir=str(out / "fiftyone_cache"),
        )
        write_text(out / "README.md", f"Downloaded via FiftyOne: {dataset.name}\nClasses: {', '.join(valid)}\nConvert to YOLO with scripts/convert_detection_dataset.py if needed.\n")
    except Exception as exc:
        print(f"Open Images download failed: {exc}")
        write_text(out / "download_todo.md", f"FiftyOne download failed: {exc}\nTry fewer classes or manual reviewed data.\n")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

