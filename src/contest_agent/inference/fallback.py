from __future__ import annotations

import logging
from typing import Any

from contest_agent.config import Settings
from contest_agent.inference.base import ClassifierBackend, DetectorBackend, OCRBackend
from contest_agent.postprocess.labels import get_legal_class_names
from contest_agent.postprocess.ocr_text import normalize_text_for_output
from contest_agent.utils import stable_choice, stable_random, stable_token


def _extract_expected_text(meta: dict[str, Any]) -> str:
    expected = meta.get("expected")
    if isinstance(expected, dict) and isinstance(expected.get("text"), str):
        return expected["text"]

    samples = meta.get("samples")
    if isinstance(samples, list) and samples:
        first = samples[0]
        if isinstance(first, dict):
            inner = first.get("expected")
            if isinstance(inner, dict) and isinstance(inner.get("text"), str):
                return inner["text"]
    return ""


class FallbackClassifierBackend(ClassifierBackend):
    def predict(self, image: Any, meta: dict[str, Any]) -> dict[str, Any]:
        legal_labels = get_legal_class_names(meta, "classify")
        label = stable_choice(legal_labels, image.sha256)
        brightness = float(image.np_image.mean()) / 255.0 if image.np_image.size else 0.5
        score = round(0.45 + min(brightness, 0.5), 4)
        return {"label": label, "score": score}


class FallbackDetectorBackend(DetectorBackend):
    def predict(self, image: Any, meta: dict[str, Any]) -> dict[str, Any]:
        legal_labels = get_legal_class_names(meta, "detect")
        rnd = stable_random(f"{image.sha256}:{image.width}x{image.height}")
        target_count = 1 if len(legal_labels) <= 1 else 2
        targets: list[dict[str, Any]] = []
        used_labels: set[str] = set()

        for idx in range(target_count):
            label_candidates = [label for label in legal_labels if label not in used_labels] or legal_labels
            label = rnd.choice(label_candidates)
            used_labels.add(label)
            frac_x = 0.35 + 0.3 * idx
            frac_y = 0.45 + 0.1 * (idx % 2)
            targets.append(
                {
                    "label": label,
                    "cx": round(image.width * frac_x, 3),
                    "cy": round(image.height * frac_y, 3),
                    "score": round(0.6 + 0.15 * rnd.random(), 4),
                }
            )
        return {"targets": targets}


class FallbackOCRBackend(OCRBackend):
    def __init__(self, settings: Settings, logger: logging.Logger | None = None) -> None:
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)

    def predict(self, image: Any, meta: dict[str, Any]) -> dict[str, Any]:
        expected = _extract_expected_text(meta) if self.settings.mock_allow_expected_text else ""
        raw_text = expected or f"DEMO-{stable_token(image.sha256, 8)}"
        normalized = normalize_text_for_output(raw_text, meta.get("normalize_rules"))
        self.logger.info("ocr_fallback raw_text=%r normalized_text=%r", raw_text, normalized)
        return {"text": normalized, "raw_text": raw_text}
