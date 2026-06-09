from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from contest_agent.image_io import ImageRecord


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def write_json(path: Path | str, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return target


def make_image_record(path: Path | str) -> ImageRecord:
    source = Path(path)
    with Image.open(source) as image:
        rgb = image.convert("RGB")
        array = np.array(rgb)
    return ImageRecord(
        pil_image=rgb,
        np_image=array,
        width=rgb.width,
        height=rgb.height,
        sha256="",
        source_summary=str(source),
        mime_type=source.suffix.lstrip(".").lower(),
    )


def timed_call(fn):
    started = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - started) * 1000.0


def image_files(root: Path | str) -> list[Path]:
    base = Path(root)
    if not base.exists():
        return []
    return sorted(path for path in base.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)

