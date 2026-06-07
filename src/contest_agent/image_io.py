from __future__ import annotations

import base64
import hashlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Tuple
from urllib import request

import numpy as np
from PIL import Image

from contest_agent.config import Settings
from contest_agent.schemas import ImagePayload
from contest_agent.utils import summarize_text


class ImageLoadError(RuntimeError):
    pass


@dataclass
class ImageRecord:
    pil_image: Image.Image
    np_image: np.ndarray
    width: int
    height: int
    sha256: str
    source_summary: str
    mime_type: str


def _enforce_size_limit(data: bytes, settings: Settings) -> None:
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(data) > max_bytes:
        raise ImageLoadError(f"image payload exceeds MAX_IMAGE_SIZE_MB={settings.max_image_size_mb}")


def _load_bytes_from_url(url: str, settings: Settings) -> bytes:
    req = request.Request(url, headers={"User-Agent": "contest-agent/0.2.0"})
    with request.urlopen(req, timeout=settings.image_download_timeout_seconds) as resp:
        data = resp.read()
    _enforce_size_limit(data, settings)
    return data


def _load_bytes_from_base64(value: str, settings: Settings) -> bytes:
    payload = value.split(",", 1)[1] if "," in value and "base64" in value[:64].lower() else value
    try:
        data = base64.b64decode(payload, validate=True)
    except Exception as exc:  # pragma: no cover - base64 error types vary
        raise ImageLoadError("invalid base64 image payload") from exc
    _enforce_size_limit(data, settings)
    return data


def _load_bytes_from_path(value: str, settings: Settings) -> bytes:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    if not path.exists():
        raise ImageLoadError(f"image path does not exist: {value}")
    data = path.read_bytes()
    _enforce_size_limit(data, settings)
    return data


def _decode_image(data: bytes) -> Tuple[Image.Image, np.ndarray, str]:
    try:
        with Image.open(BytesIO(data)) as image:
            image_format = (image.format or "unknown").lower()
            rgb = image.convert("RGB")
            array = np.array(rgb)
    except Exception as exc:  # pragma: no cover - Pillow error types vary
        raise ImageLoadError("failed to decode image bytes") from exc
    mime_type = image_format
    return rgb, array, mime_type


def load_image(image_payload: ImagePayload, settings: Settings) -> ImageRecord:
    image_format = (image_payload.format or "url").strip().lower()
    source_summary = summarize_text(image_payload.data, max_length=96)

    if image_format == "url":
        image_bytes = _load_bytes_from_url(image_payload.data, settings)
    elif image_format == "base64":
        image_bytes = _load_bytes_from_base64(image_payload.data, settings)
    elif image_format == "path":
        image_bytes = _load_bytes_from_path(image_payload.data, settings)
    else:
        raise ImageLoadError(f"unsupported image format: {image_format}")

    pil_image, np_image, mime_type = _decode_image(image_bytes)
    sha256 = hashlib.sha256(image_bytes).hexdigest()
    width, height = pil_image.size
    return ImageRecord(
        pil_image=pil_image,
        np_image=np_image,
        width=width,
        height=height,
        sha256=sha256,
        source_summary=f"{image_format}:{source_summary}",
        mime_type=mime_type,
    )
