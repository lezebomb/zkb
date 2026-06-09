from __future__ import annotations

import json
import pickle
from pathlib import Path

from fastapi.testclient import TestClient

from contest_agent.app import create_app
from contest_agent.config import get_settings
from contest_agent.image_io import ImageRecord
from contest_agent.inference.classifier import TorchClassifierBackend


def test_torch_classifier_backend_loads_constant_checkpoint(tmp_path: Path, ensure_sample_image: Path) -> None:
    model = tmp_path / "classifier.pt"
    classes = ["办公室", "公园", "街道", "商场", "厨房", "卧室", "图书馆", "体育馆"]
    model.write_bytes(pickle.dumps({"constant_label": "办公室", "classes": classes}))
    (tmp_path / "class_names.json").write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    from PIL import Image
    import numpy as np

    image = Image.open(ensure_sample_image).convert("RGB")
    record = ImageRecord(image, np.array(image), image.width, image.height, "", "test", "jpg")
    result = TorchClassifierBackend(model).predict(record, {"class_names": classes})
    assert result["label"] == "办公室"
    assert result["score"] == 1.0


def test_infer_classify_uses_local_checkpoint(monkeypatch, tmp_path: Path, ensure_sample_image: Path) -> None:
    model = tmp_path / "classifier.pt"
    classes = ["办公室", "公园", "街道", "商场", "厨房", "卧室", "图书馆", "体育馆"]
    model.write_bytes(pickle.dumps({"constant_label": "办公室", "classes": classes}))
    (tmp_path / "class_names.json").write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("CLASSIFY_BACKEND", "local")
    monkeypatch.setenv("MODEL_CLASSIFY_PATH", str(model))
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        response = client.post("/infer", json={"request_id": "cls-local", "session_id": "s", "task_type": "classify", "image": {"format": "path", "data": str(ensure_sample_image)}, "meta": {"class_names": classes}})
    get_settings.cache_clear()
    payload = response.json()
    assert payload["ok"] is True
    assert payload["result"]["label"] == "办公室"

