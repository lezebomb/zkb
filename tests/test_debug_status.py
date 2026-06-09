from __future__ import annotations

import json
import pickle
from pathlib import Path

from fastapi.testclient import TestClient

from contest_agent.app import create_app
from contest_agent.config import get_settings


def test_debug_status_reports_model_metadata_without_loading(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "classifier.pt"
    model.write_bytes(pickle.dumps({"constant_label": "办公室", "classes": ["办公室"]}))
    (tmp_path / "class_names.json").write_text(json.dumps(["办公室"], ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("CLASSIFY_BACKEND", "local")
    monkeypatch.setenv("MODEL_CLASSIFY_PATH", str(model))
    get_settings.cache_clear()
    app = create_app()
    with TestClient(app) as client:
        payload = client.get("/debug/status").json()
    get_settings.cache_clear()
    assert payload["backends"]["classify"]["model_exists"] is True
    assert payload["backends"]["classify"]["class_names_exists"] is True
    assert payload["backends"]["classify"]["model_size_mb"] is not None
    assert "dirs" in payload["backends"]["ocr"]

