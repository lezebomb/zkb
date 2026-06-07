from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from contest_agent.config import Settings
from contest_agent.inference.base import OCRBackend
from contest_agent.inference.fallback import FallbackOCRBackend
from contest_agent.postprocess.ocr_text import merge_ocr_lines, truncate_text


def _box_anchor(box: Any, fallback_index: int) -> tuple[float, float]:
    if isinstance(box, (list, tuple)) and box:
        points = [point for point in box if isinstance(point, (list, tuple)) and len(point) >= 2]
        if points:
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            return min(xs), min(ys)
    return float(fallback_index), 0.0


def _is_old_paddle_line(item: Any) -> bool:
    if not isinstance(item, (list, tuple)) or len(item) < 2:
        return False
    text_part = item[1]
    if isinstance(text_part, (list, tuple)) and text_part and isinstance(text_part[0], str):
        return True
    return False


def _collect_text_lines(node: Any, lines: list[tuple[float, float, str]]) -> None:
    if node is None:
        return

    if isinstance(node, dict):
        texts = node.get("rec_texts") or node.get("texts") or node.get("text")
        polys = node.get("dt_polys") or node.get("polys") or node.get("boxes")
        if isinstance(texts, str):
            lines.append((0.0, 0.0, texts))
            return
        if isinstance(texts, list):
            for index, text in enumerate(texts):
                if not isinstance(text, str):
                    continue
                anchor_x, anchor_y = _box_anchor(polys[index], index) if isinstance(polys, list) and index < len(polys) else (float(index), 0.0)
                lines.append((anchor_x, anchor_y, text))
            return
        for value in node.values():
            _collect_text_lines(value, lines)
        return

    if _is_old_paddle_line(node):
        text_part = node[1]
        text_value = text_part[0] if isinstance(text_part, (list, tuple)) else ""
        if isinstance(text_value, str):
            anchor_x, anchor_y = _box_anchor(node[0], len(lines))
            lines.append((anchor_x, anchor_y, text_value))
        return

    if isinstance(node, (list, tuple)):
        for item in node:
            _collect_text_lines(item, lines)


class PaddleOCRBackend(OCRBackend):
    def __init__(self, settings: Settings, logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)
        self.fallback = FallbackOCRBackend(settings, self.logger)
        self._engine: Any | None = None
        self._load_attempted = False

    def _model_dirs_exist(self) -> bool:
        required = [self.settings.ocr_det_model_dir, self.settings.ocr_rec_model_dir]
        if self.settings.ocr_use_angle_cls:
            required.append(self.settings.ocr_cls_model_dir)
        return all(path.exists() for path in required)

    def _load_engine(self) -> Any | None:
        if self._load_attempted:
            return self._engine
        self._load_attempted = True

        if not self._model_dirs_exist() and not self.settings.allow_model_auto_download:
            self.logger.warning("PaddleOCR model directories missing and ALLOW_MODEL_AUTO_DOWNLOAD=false; falling back.")
            return None

        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            self.logger.warning("PaddleOCR import failed (%s); falling back.", exc)
            return None

        kwargs: dict[str, Any] = {
            "use_angle_cls": self.settings.ocr_use_angle_cls,
            "lang": self.settings.ocr_lang,
            "show_log": False,
        }
        if self.settings.ocr_det_model_dir.exists():
            kwargs["det_model_dir"] = str(self.settings.ocr_det_model_dir)
        if self.settings.ocr_rec_model_dir.exists():
            kwargs["rec_model_dir"] = str(self.settings.ocr_rec_model_dir)
        if self.settings.ocr_use_angle_cls and self.settings.ocr_cls_model_dir.exists():
            kwargs["cls_model_dir"] = str(self.settings.ocr_cls_model_dir)
        if not self._model_dirs_exist() and self.settings.allow_model_auto_download:
            self.logger.warning("Allowing PaddleOCR auto-download for pre-race preparation only.")

        try:
            self._engine = PaddleOCR(**kwargs)
        except Exception as exc:
            self.logger.warning("Failed to initialize PaddleOCR (%s); falling back.", exc)
            self._engine = None
        return self._engine

    def predict(self, image: Any, meta: dict[str, Any]) -> dict[str, Any]:
        engine = self._load_engine()
        if engine is None:
            return self.fallback.predict(image, meta)

        try:
            raw_result = engine.ocr(image.np_image, cls=self.settings.ocr_use_angle_cls)
        except Exception as exc:
            self.logger.warning("PaddleOCR inference failed (%s); falling back.", exc)
            return self.fallback.predict(image, meta)

        text_lines: list[tuple[float, float, str]] = []
        _collect_text_lines(raw_result, text_lines)
        merged = merge_ocr_lines(text_lines)
        raw_text = truncate_text(merged, self.settings.ocr_max_text_length)
        return {"text": raw_text, "raw_text": raw_text}


class LocalOCRBackend(OCRBackend):
    def __init__(self, model_path: Path, settings: Settings, logger: logging.Logger | None = None) -> None:
        self.model_path = model_path
        self.logger = logger or logging.getLogger(__name__)
        self.fallback = FallbackOCRBackend(settings, self.logger)
        self.available = model_path.exists()
        if not self.available:
            self.logger.warning("OCR model not found at %s, falling back.", model_path)

    def predict(self, image: Any, meta: dict[str, Any]) -> dict[str, Any]:
        if not self.available:
            return self.fallback.predict(image, meta)
        self.logger.warning("Local OCR backend is a placeholder; using fallback until real model is wired.")
        return self.fallback.predict(image, meta)


def build_ocr_backend(settings: Settings, logger: logging.Logger | None = None) -> OCRBackend:
    if settings.ocr_backend in {"local", "paddleocr"}:
        return PaddleOCRBackend(settings, logger)
    return FallbackOCRBackend(settings, logger)
