from __future__ import annotations

from typing import Any


CLASSIFY_LABELS = [
    "办公室",
    "公园",
    "街道",
    "商场",
    "厨房",
    "卧室",
    "图书馆",
    "体育馆",
]

DETECT_LABELS = [
    "人",
    "汽车",
    "自行车",
    "手机",
    "水杯",
    "笔记本电脑",
    "台灯",
    "沙发",
    "狗",
]

CLASSIFY_LABEL_ALIASES = {
    "office": "办公室",
    "park": "公园",
    "street": "街道",
    "mall": "商场",
    "shopping mall": "商场",
    "kitchen": "厨房",
    "bedroom": "卧室",
    "library": "图书馆",
    "gym": "体育馆",
    "gymnasium": "体育馆",
    "stadium": "体育馆",
}

DETECT_LABEL_ALIASES = {
    "person": "人",
    "people": "人",
    "man": "人",
    "woman": "人",
    "car": "汽车",
    "vehicle": "汽车",
    "bicycle": "自行车",
    "bike": "自行车",
    "cell phone": "手机",
    "mobile phone": "手机",
    "phone": "手机",
    "cup": "水杯",
    "mug": "水杯",
    "laptop": "笔记本电脑",
    "notebook": "笔记本电脑",
    "desk lamp": "台灯",
    "lamp": "台灯",
    "sofa": "沙发",
    "couch": "沙发",
    "dog": "狗",
}


def _extract_sample_meta_class_names(meta: dict[str, Any]) -> list[str] | None:
    samples = meta.get("samples")
    if not isinstance(samples, list) or not samples:
        return None
    first = samples[0]
    if not isinstance(first, dict):
        return None
    inner_meta = first.get("meta")
    if not isinstance(inner_meta, dict):
        return None
    class_names = inner_meta.get("class_names")
    if not isinstance(class_names, list):
        return None
    return [str(item) for item in class_names if str(item).strip()]


def get_legal_class_names(meta: dict[str, Any], task_type: str) -> list[str]:
    sample_override = _extract_sample_meta_class_names(meta)
    if sample_override:
        return sample_override

    class_names = meta.get("class_names")
    if isinstance(class_names, list) and class_names:
        return [str(item) for item in class_names if str(item).strip()]

    if task_type == "classify":
        return list(CLASSIFY_LABELS)
    if task_type == "detect":
        return list(DETECT_LABELS)
    return []


def _normalize_alias(label: str, aliases: dict[str, str]) -> str:
    compact = " ".join(str(label).strip().lower().split())
    return aliases.get(compact, str(label).strip())


def normalize_classify_label(label: str) -> str:
    return _normalize_alias(label, CLASSIFY_LABEL_ALIASES)


def normalize_detect_label(label: str) -> str:
    return _normalize_alias(label, DETECT_LABEL_ALIASES)


def choose_fallback_label(legal_labels: list[str], task_type: str) -> str:
    if legal_labels:
        return legal_labels[0]
    if task_type == "classify":
        return CLASSIFY_LABELS[0]
    if task_type == "detect":
        return DETECT_LABELS[0]
    return ""
