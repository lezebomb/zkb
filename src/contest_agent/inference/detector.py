from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from contest_agent.config import Settings
from contest_agent.inference.base import DetectorBackend
from contest_agent.inference.fallback import FallbackDetectorBackend
from contest_agent.postprocess.labels import get_legal_class_names, normalize_detect_label
from contest_agent.utils import clamp


os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".ultralytics"))


class UltralyticsDetectorBackend(DetectorBackend):
    def __init__(self, model_path: Path | str, settings: Settings, logger: logging.Logger | None = None) -> None:
        self.model_path = Path(model_path)
        self.settings = settings
        self.logger = logger or logging.getLogger(__name__)
        self.fallback = FallbackDetectorBackend()
        self._model: Any | None = None
        self._load_attempted = False

    def _resolve_model_source(self) -> str | None:
        if self.model_path.exists():
            return str(self.model_path)
        if not self.settings.allow_model_auto_download:
            self.logger.warning(
                "Detector model not found at %s and ALLOW_MODEL_AUTO_DOWNLOAD=false; falling back.",
                self.model_path,
            )
            return None
        model_name = self.model_path.name
        self.logger.warning(
            "Detector model path %s not found; allowing Ultralytics auto-download for pre-race preparation using %s.",
            self.model_path,
            model_name,
        )
        return model_name

    def _load_model(self) -> Any | None:
        if self._load_attempted:
            return self._model
        self._load_attempted = True

        model_source = self._resolve_model_source()
        if model_source is None:
            return None

        try:
            from ultralytics import YOLO
        except Exception as exc:
            self.logger.warning("Ultralytics import failed (%s); falling back.", exc)
            return None

        try:
            self._model = YOLO(model_source)
        except Exception as exc:
            self.logger.warning("Failed to load Ultralytics detector from %s (%s); falling back.", model_source, exc)
            self._model = None
        return self._model

    def predict(self, image: Any, meta: dict[str, Any]) -> dict[str, Any]:
        model = self._load_model()
        if model is None:
            return self.fallback.predict(image, meta)

        legal_labels = get_legal_class_names(meta, "detect")
        try:
            results = model.predict(
                image.np_image,
                conf=self.settings.detect_score_threshold,
                imgsz=self.settings.detect_imgsz,
                device=self.settings.detect_device,
                max_det=self.settings.detect_max_targets,
                save=False,
                verbose=False,
            )
        except Exception as exc:
            self.logger.warning("Ultralytics detect inference failed (%s); falling back.", exc)
            return self.fallback.predict(image, meta)

        targets: list[dict[str, Any]] = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue

            names = getattr(result, "names", None) or getattr(model, "names", None) or {}
            xyxy_values = getattr(boxes, "xyxy", None)
            cls_values = getattr(boxes, "cls", None)
            conf_values = getattr(boxes, "conf", None)
            if xyxy_values is None or cls_values is None or conf_values is None:
                continue

            for xyxy, cls_idx, score in zip(xyxy_values.tolist(), cls_values.tolist(), conf_values.tolist()):
                score_value = float(score)
                if score_value < self.settings.detect_score_threshold:
                    continue

                class_index = int(cls_idx)
                if isinstance(names, dict):
                    raw_name = names.get(class_index, str(class_index))
                elif isinstance(names, list) and 0 <= class_index < len(names):
                    raw_name = names[class_index]
                else:
                    raw_name = str(class_index)

                label = normalize_detect_label(str(raw_name))
                if label not in legal_labels:
                    continue

                x1, y1, x2, y2 = [float(value) for value in xyxy]
                targets.append(
                    {
                        "label": label,
                        "cx": round(clamp((x1 + x2) / 2.0, 0.0, max(0.0, float(image.width - 1))), 3),
                        "cy": round(clamp((y1 + y2) / 2.0, 0.0, max(0.0, float(image.height - 1))), 3),
                        "score": round(score_value, 4),
                    }
                )

        targets.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        return {"targets": targets[: self.settings.detect_max_targets]}


class LocalDetectorBackend(UltralyticsDetectorBackend):
    pass


def build_detector_backend(settings: Settings, logger: logging.Logger | None = None) -> DetectorBackend:
    if settings.detect_backend in {"local", "ultralytics"}:
        return UltralyticsDetectorBackend(settings.model_detect_path, settings, logger)
    return FallbackDetectorBackend()
