from contest_agent.postprocess.detection import normalize_detect_result
from contest_agent.postprocess.labels import (
    CLASSIFY_LABELS,
    DETECT_LABELS,
    get_legal_class_names,
    normalize_classify_label,
    normalize_detect_label,
)
from contest_agent.postprocess.ocr_text import normalize_text_for_output

__all__ = [
    "CLASSIFY_LABELS",
    "DETECT_LABELS",
    "get_legal_class_names",
    "normalize_classify_label",
    "normalize_detect_label",
    "normalize_detect_result",
    "normalize_text_for_output",
]
