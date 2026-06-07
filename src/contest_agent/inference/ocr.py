from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from contest_agent.config import Settings
from contest_agent.inference.base import OCRBackend
from contest_agent.inference.fallback import FallbackOCRBackend


class LocalOCRBackend(OCRBackend):
    def __init__(self, model_path: Path, logger: logging.Logger | None = None) -> None:
        self.model_path = model_path
        self.logger = logger or logging.getLogger(__name__)
        self.fallback = FallbackOCRBackend(self.logger)
        self.available = model_path.exists()
        if not self.available:
            self.logger.warning("OCR model not found at %s, falling back.", model_path)

    def predict(self, image: Any, meta: dict[str, Any]) -> dict[str, Any]:
        if not self.available:
            return self.fallback.predict(image, meta)
        self.logger.warning("Local OCR backend is a placeholder; using fallback until real model is wired.")
        return self.fallback.predict(image, meta)


def build_ocr_backend(settings: Settings, logger: logging.Logger | None = None) -> OCRBackend:
    if settings.ocr_backend == "local":
        return LocalOCRBackend(settings.model_ocr_path, logger)
    return FallbackOCRBackend(logger)
