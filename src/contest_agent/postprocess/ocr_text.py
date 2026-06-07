from __future__ import annotations

import re
import unicodedata
from typing import Any


DEFAULT_PUNCTUATION_MAP = {
    "：": ":",
    "％": "%",
    "，": ",",
    "。": ".",
    "（": "(",
    "）": ")",
}

DEFAULT_CONFUSION_MAP = {
    "O": "0",
    "o": "0",
    "I": "1",
    "l": "1",
    "S": "5",
}


def _to_halfwidth(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def merge_ocr_lines(lines: list[tuple[float, float, str]]) -> str:
    if not lines:
        return ""
    ordered = sorted(lines, key=lambda item: (round(item[1] / 10.0), item[1], item[0]))
    return " ".join(text for _, _, text in ordered if text).strip()


def truncate_text(text: str, max_length: int) -> str:
    if max_length <= 0:
        return ""
    return text[:max_length]


def normalize_text_for_output(text: Any, normalize_rules: dict[str, Any] | None = None) -> str:
    rules = normalize_rules or {}
    value = _to_halfwidth("" if text is None else str(text))

    punctuation_map = DEFAULT_PUNCTUATION_MAP.copy()
    extra_map = rules.get("punctuation_map")
    if isinstance(extra_map, dict):
        punctuation_map.update({str(k): str(v) for k, v in extra_map.items()})
    for src, dst in punctuation_map.items():
        value = value.replace(src, dst)

    value = value.strip()

    if rules.get("trim_space", False):
        value = re.sub(r"\s+", "", value)
    else:
        value = re.sub(r"[ \t]+", " ", value)

    if rules.get("case_insensitive", False):
        value = value.lower()

    if rules.get("apply_confusion_map", False):
        confusion_map = DEFAULT_CONFUSION_MAP.copy()
        extra_confusion_map = rules.get("confusion_map")
        if isinstance(extra_confusion_map, dict):
            confusion_map.update({str(k): str(v) for k, v in extra_confusion_map.items()})
        value = "".join(confusion_map.get(ch, ch) for ch in value)

    return value
