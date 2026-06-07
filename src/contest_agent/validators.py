from __future__ import annotations

from typing import Any

from contest_agent.postprocess.detection import normalize_detect_result
from contest_agent.postprocess.labels import (
    choose_fallback_label,
    get_legal_class_names,
    normalize_classify_label,
)
from contest_agent.postprocess.ocr_text import normalize_text_for_output


def validate_classify_result(result: dict[str, Any] | None, meta: dict[str, Any]) -> dict[str, Any]:
    legal_labels = get_legal_class_names(meta, "classify")
    raw_label = ""
    if isinstance(result, dict):
        raw_label = str(result.get("label", "")).strip()
    normalized = normalize_classify_label(raw_label)
    if normalized not in legal_labels:
        normalized = choose_fallback_label(legal_labels, "classify")
    validated: dict[str, Any] = {"label": normalized}
    if isinstance(result, dict) and "score" in result:
        try:
            validated["score"] = round(float(result["score"]), 4)
        except (TypeError, ValueError):
            pass
    return validated


def validate_detect_result(
    result: dict[str, Any] | None,
    meta: dict[str, Any],
    image_width: int,
    image_height: int,
    *,
    allow_empty: bool = True,
    fill_when_empty: bool = False,
) -> dict[str, Any]:
    legal_labels = get_legal_class_names(meta, "detect")
    normalized = normalize_detect_result(result, legal_labels, image_width, image_height)
    targets = normalized["targets"]
    if not targets and fill_when_empty:
        normalized["targets"] = [
            {
                "label": choose_fallback_label(legal_labels, "detect"),
                "cx": round(max(0.0, (image_width - 1) / 2.0), 3),
                "cy": round(max(0.0, (image_height - 1) / 2.0), 3),
                "score": 0.01,
            }
        ]
    elif not targets and not allow_empty:
        raise ValueError("detect result contains no valid targets")
    return normalized


def validate_ocr_result(result: dict[str, Any] | None, meta: dict[str, Any]) -> dict[str, Any]:
    raw_text = ""
    if isinstance(result, dict):
        raw_text = str(result.get("text", result.get("content", "")))
    normalized = normalize_text_for_output(raw_text, meta.get("normalize_rules"))
    return {"text": normalized}
