from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

if False:  # pragma: no cover
    from contest_agent.image_io import ImageRecord


class ClassifierBackend(ABC):
    @abstractmethod
    def predict(self, image: "ImageRecord", meta: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class DetectorBackend(ABC):
    @abstractmethod
    def predict(self, image: "ImageRecord", meta: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class OCRBackend(ABC):
    @abstractmethod
    def predict(self, image: "ImageRecord", meta: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError
