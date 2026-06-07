from contest_agent.inference.classifier import build_classifier_backend
from contest_agent.inference.detector import build_detector_backend
from contest_agent.inference.ocr import build_ocr_backend

__all__ = [
    "build_classifier_backend",
    "build_detector_backend",
    "build_ocr_backend",
]
