from __future__ import annotations

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Any

from contest_agent.config import Settings
from contest_agent.inference.base import ClassifierBackend
from contest_agent.inference.fallback import FallbackClassifierBackend
from contest_agent.postprocess.labels import CLASSIFY_LABELS


os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")


class TorchClassifierBackend(ClassifierBackend):
    def __init__(self, model_path: Path, logger: logging.Logger | None = None) -> None:
        self.model_path = Path(model_path)
        self.class_names_path = self.model_path.parent / "class_names.json"
        self.logger = logger or logging.getLogger(__name__)
        self.fallback = FallbackClassifierBackend()
        self._load_attempted = False
        self._model: Any | None = None
        self._classes: list[str] = []
        self._constant_label: str | None = None
        self._torch: Any | None = None
        self._transform: Any | None = None

    def _read_class_names(self) -> list[str]:
        if self.class_names_path.exists():
            try:
                raw = json.loads(self.class_names_path.read_text(encoding="utf-8"))
                if isinstance(raw, list) and raw:
                    return [str(item) for item in raw]
            except Exception as exc:
                self.logger.warning("failed to read class names from %s: %s", self.class_names_path, exc)
        return list(CLASSIFY_LABELS)

    def _try_load_constant_checkpoint(self) -> bool:
        try:
            with self.model_path.open("rb") as handle:
                checkpoint = pickle.load(handle)
        except Exception:
            return False
        if not isinstance(checkpoint, dict):
            return False
        classes = checkpoint.get("classes")
        if isinstance(classes, list) and classes:
            self._classes = [str(item) for item in classes]
        else:
            self._classes = self._read_class_names()
        label = checkpoint.get("constant_label") or (self._classes[0] if self._classes else None)
        if label:
            self._constant_label = str(label)
            return True
        return False

    def _build_model(self, model_name: str, num_classes: int) -> Any:
        from torch import nn
        from torchvision import models

        if model_name == "resnet18":
            model = models.resnet18(weights=None)
            model.fc = nn.Linear(model.fc.in_features, num_classes)
            return model
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model

    def _load(self) -> bool:
        if self._load_attempted:
            return self._constant_label is not None or self._model is not None
        self._load_attempted = True
        if not self.model_path.exists():
            self.logger.warning("Classifier model not found at %s, falling back.", self.model_path)
            return False

        try:
            import torch
            from torchvision import transforms
        except Exception as exc:
            if self._try_load_constant_checkpoint():
                self.logger.warning("torch import failed (%s); using constant classifier checkpoint.", exc)
                return True
            self.logger.warning("torch/torchvision import failed (%s); falling back.", exc)
            return False

        try:
            checkpoint = torch.load(self.model_path, map_location="cpu")
            if not isinstance(checkpoint, dict):
                raise ValueError("checkpoint is not a dict")
            classes = checkpoint.get("classes")
            self._classes = [str(item) for item in classes] if isinstance(classes, list) and classes else self._read_class_names()
            model_name = str(checkpoint.get("model", "mobilenet_v3_small"))
            model = self._build_model(model_name, len(self._classes))
            state = checkpoint.get("state_dict")
            if state:
                model.load_state_dict(state, strict=True)
            model.eval()
            self._model = model
            self._torch = torch
            self._transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
            return True
        except Exception as exc:
            if self._try_load_constant_checkpoint():
                self.logger.warning("torch checkpoint load failed (%s); using constant classifier checkpoint.", exc)
                return True
            self.logger.warning("failed to load classifier checkpoint %s (%s); falling back.", self.model_path, exc)
            return False

    def predict(self, image: Any, meta: dict[str, Any]) -> dict[str, Any]:
        if not self._load():
            return self.fallback.predict(image, meta)
        if self._constant_label is not None:
            return {"label": self._constant_label, "score": 1.0}
        try:
            tensor = self._transform(image.pil_image.convert("RGB")).unsqueeze(0)
            with self._torch.no_grad():
                logits = self._model(tensor)
                probs = self._torch.softmax(logits, dim=1)[0]
                score, idx = self._torch.max(probs, dim=0)
            class_index = int(idx.item())
            label = self._classes[class_index] if 0 <= class_index < len(self._classes) else self._classes[0]
            return {"label": label, "score": float(score.item())}
        except Exception as exc:
            self.logger.warning("classifier inference failed (%s); falling back.", exc)
            return self.fallback.predict(image, meta)


LocalClassifierBackend = TorchClassifierBackend


def build_classifier_backend(settings: Settings, logger: logging.Logger | None = None) -> ClassifierBackend:
    if settings.classify_backend in {"local", "torch"}:
        return TorchClassifierBackend(settings.model_classify_path, logger)
    return FallbackClassifierBackend()
