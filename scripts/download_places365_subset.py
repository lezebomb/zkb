from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from contest_agent.training.classify_data import CLASSIFY_LABELS, create_template
from contest_agent.training.common import env_bool, write_text


EN_TO_CN = {
    "office": "办公室",
    "park": "公园",
    "street": "街道",
    "urban_street": "街道",
    "shopping_mall": "商场",
    "mall": "商场",
    "kitchen": "厨房",
    "bedroom": "卧室",
    "library": "图书馆",
    "gymnasium": "体育馆",
    "gym": "体育馆",
    "indoor_sports": "体育馆",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a guarded Places365 8-class subset workspace.")
    parser.add_argument("--classes", nargs="*", default=list(EN_TO_CN))
    parser.add_argument("--max-images-per-class", type=int, default=200)
    parser.add_argument("--output", required=True)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    out = create_template(args.output)
    allowed = args.allow_download or env_bool("ALLOW_DATASET_DOWNLOAD", False)
    todo = [
        "# Places365 subset TODO",
        "",
        "Full Places365 is not downloaded by default.",
        f"Requested classes: {', '.join(args.classes)}",
        f"Max images per class: {args.max_images_per_class}",
        "Put reviewed images into train/<中文类名>/ and val/<中文类名>/.",
        "Do not crawl search engines or unclear-license images.",
    ]
    if not allowed:
        todo.append("Download blocked: pass --allow-download or ALLOW_DATASET_DOWNLOAD=true after license review.")
        write_text(out / "download_todo.md", "\n".join(todo) + "\n")
        print(f"download disabled; template created: {out}")
        return 0
    todo.append("Automatic Places365 download is not implemented safely here; use official data after reviewing terms.")
    write_text(out / "download_todo.md", "\n".join(todo) + "\n")
    print("allow-download set, but no unclear web crawling is performed. Use official Places365 source manually.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

