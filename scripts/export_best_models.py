from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contest_agent.training.common import sha256_file, utc_now_iso


def add_file(manifest: list[dict], model_type: str, path: Path, source: str, env: dict[str, str], eval_metrics_path: str | None = None) -> None:
    if not path.exists() or not path.is_file():
        return
    manifest.append(
        {
            "model_type": model_type,
            "path": path.as_posix(),
            "created_at": utc_now_iso(),
            "source_run": source,
            "eval_metrics_path": eval_metrics_path,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "recommended_env": env,
        }
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect best model artifacts into models/ and write manifest.")
    parser.add_argument("--detect-run", default="runs/detect")
    parser.add_argument("--detect-output", default="models/detect/best.pt")
    parser.add_argument("--classify-source", default="models/classify/classifier.pt")
    parser.add_argument("--ocr-dir", default="models/ocr/paddleocr")
    parser.add_argument("--detect-metrics", default="runs/eval/detect_metrics.json")
    parser.add_argument("--classify-metrics", default="runs/eval/classify_metrics.json")
    args = parser.parse_args()
    manifest: list[dict] = []

    detect_root = Path(args.detect_run)
    candidates = sorted(detect_root.glob("**/weights/best.pt"), key=lambda p: p.stat().st_mtime, reverse=True) if detect_root.exists() else []
    if candidates:
        target = Path(args.detect_output)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidates[0], target)
        add_file(manifest, "detect", target, str(candidates[0]), {"DETECT_BACKEND": "ultralytics", "MODEL_DETECT_PATH": target.as_posix()}, args.detect_metrics)
        print(f"detect exported: {target}")

    classify = Path(args.classify_source)
    add_file(manifest, "classify", classify, str(classify), {"CLASSIFY_BACKEND": "local", "MODEL_CLASSIFY_PATH": classify.as_posix()}, args.classify_metrics)
    ocr = Path(args.ocr_dir)
    if all((ocr / name).exists() for name in ("det", "rec", "cls")):
        manifest.append({"model_type": "ocr", "path": ocr.as_posix(), "created_at": utc_now_iso(), "source_run": str(ocr), "eval_metrics_path": "runs/eval/ocr_metrics.json", "sha256": None, "size_bytes": None, "recommended_env": {"OCR_BACKEND": "paddleocr", "MODEL_OCR_PATH": ocr.as_posix()}})
    else:
        print(f"ocr dirs not complete: {ocr}")

    out = Path("models/model_manifest.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"models": manifest}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {out}")
    print("Recommended .env:")
    print("DETECT_BACKEND=ultralytics\nMODEL_DETECT_PATH=models/detect/best.pt")
    print("CLASSIFY_BACKEND=local\nMODEL_CLASSIFY_PATH=models/classify/classifier.pt")
    print("OCR_BACKEND=paddleocr\nMODEL_OCR_PATH=models/ocr/paddleocr")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
