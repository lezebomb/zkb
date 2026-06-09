from __future__ import annotations

import json
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

from contest_agent.training.common import ensure_dir, write_text


CLASSIFY_LABELS = ["办公室", "公园", "街道", "商场", "厨房", "卧室", "图书馆", "体育馆"]

CLASSIFY_SEARCH_HINTS = {
    "办公室": ["office"],
    "公园": ["park"],
    "街道": ["street", "urban street"],
    "商场": ["shopping mall", "mall"],
    "厨房": ["kitchen"],
    "卧室": ["bedroom"],
    "图书馆": ["library"],
    "体育馆": ["gymnasium", "gym", "sports hall"],
}


def create_template(output: Path | str) -> Path:
    out = Path(output)
    for split in ("train", "val"):
        for label in CLASSIFY_LABELS:
            ensure_dir(out / split / label)
    write_text(out / "class_names.json", json.dumps(CLASSIFY_LABELS, ensure_ascii=False, indent=2) + "\n")
    write_text(out / "README.md", "ImageFolder template for 8 contest scene classes. Add real images before competition training.\n")
    return out


def prepare_custom(input_dir: Path | str, output: Path | str) -> Path:
    src = Path(input_dir)
    if not src.exists():
        raise FileNotFoundError(f"classify input does not exist: {src}")
    out = Path(output)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out)
    create_template(out)
    return out


def create_tiny_placeholder(output: Path | str, images_per_class: int = 2) -> Path:
    out = create_template(output)
    colors = [(210, 80, 80), (90, 160, 90), (80, 120, 210), (210, 170, 70)]
    for split in ("train", "val"):
        count = images_per_class if split == "train" else 1
        for idx, label in enumerate(CLASSIFY_LABELS):
            for image_idx in range(count):
                image = Image.new("RGB", (224, 224), colors[(idx + image_idx) % len(colors)])
                draw = ImageDraw.Draw(image)
                draw.rectangle((20, 20, 204, 204), outline=(255, 255, 255), width=3)
                draw.text((32, 98), f"class {idx}", fill=(255, 255, 255))
                image.save(out / split / label / f"placeholder_{image_idx:02d}.jpg", quality=90)
    write_text(
        out / "README.md",
        "Tiny synthetic placeholder dataset for smoke tests only. Do not use it for competition scoring.\n",
    )
    return out


def imagefolder_has_images(root: Path | str) -> bool:
    base = Path(root)
    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return any(path.is_file() and path.suffix.lower() in suffixes for path in base.rglob("*"))

