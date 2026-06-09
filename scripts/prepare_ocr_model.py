from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contest_agent.training.common import env_bool, write_text


def ready(output: Path) -> bool:
    return all((output / name).exists() for name in ("det", "rec", "cls"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare local PaddleOCR model directories.")
    parser.add_argument("--output", default="models/ocr/paddleocr")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    output = Path(args.output)

    if ready(output):
        print(f"ready: {output}")
        print("Before competition keep models local and set ALLOW_MODEL_AUTO_DOWNLOAD=false.")
        return 0

    allowed = args.allow_download or env_bool("ALLOW_MODEL_AUTO_DOWNLOAD", False)
    if not allowed:
        print("PaddleOCR model dirs are missing and auto download is disabled")
        print(f"manual: prepare det/ rec/ cls/ under {output}")
        print("or run with --allow-download before competition to warm PaddleOCR cache")
        return 0

    try:
        from paddleocr import PaddleOCR
    except Exception as exc:
        print(f"paddleocr import failed: {exc}")
        print("install: pip install -r requirements-ocr.txt")
        return 2

    try:
        PaddleOCR(use_angle_cls=True, lang="ch")
    except Exception as exc:
        print(f"PaddleOCR initialization failed: {exc}")
        print("Some PaddleOCR versions require different constructor args; check docs/model_setup_ocr.md.")
        return 3

    output.mkdir(parents=True, exist_ok=True)
    write_text(
        output / "README.md",
        "PaddleOCR cache was warmed. Copy actual det/rec/cls inference model directories here before competition.\n",
    )
    print("PaddleOCR initialized. Now localize det/rec/cls model directories under:", output)
    print("Before competition keep ALLOW_MODEL_AUTO_DOWNLOAD=false and confirm /debug/status model_dirs_exist=true.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

