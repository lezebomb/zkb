from __future__ import annotations

from base64 import b64encode
from pathlib import Path

from fastapi.testclient import TestClient

from contest_agent.app import create_app
from contest_agent.config import get_settings


def _sample_path(ensure_sample_image: Path) -> str:
    return str(ensure_sample_image)


def test_infer_ocr_returns_text_field(client, ensure_sample_image: Path) -> None:
    response = client.post(
        "/infer",
        json={
            "request_id": "eval-ocr-1",
            "session_id": "team-7",
            "task_type": "ocr",
            "image": {"format": "path", "data": _sample_path(ensure_sample_image)},
            "meta": {
                "language_hint": "zh",
                "normalize_rules": {"trim_space": True, "case_insensitive": False},
            },
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert isinstance(payload["result"]["text"], str)


def test_infer_ocr_does_not_read_expected_text_by_default(client, ensure_sample_image: Path) -> None:
    response = client.post(
        "/infer",
        json={
            "request_id": "eval-ocr-secret",
            "session_id": "team-7",
            "task_type": "ocr",
            "image": {"format": "path", "data": _sample_path(ensure_sample_image)},
            "meta": {
                "expected": {"text": "SECRET_ANSWER"},
                "normalize_rules": {"trim_space": True, "case_insensitive": False},
            },
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["result"]["text"] != "SECRET_ANSWER"


def test_paddleocr_missing_deps_or_models_falls_back_safely(monkeypatch, ensure_sample_image: Path) -> None:
    monkeypatch.setenv("OCR_BACKEND", "paddleocr")
    monkeypatch.setenv("ALLOW_MODEL_AUTO_DOWNLOAD", "false")
    monkeypatch.setenv("OCR_DET_MODEL_DIR", "models/missing-det")
    monkeypatch.setenv("OCR_REC_MODEL_DIR", "models/missing-rec")
    monkeypatch.setenv("OCR_CLS_MODEL_DIR", "models/missing-cls")
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as current_client:
        response = current_client.post(
            "/infer",
            json={
                "request_id": "eval-ocr-paddle-fallback",
                "session_id": "team-7",
                "task_type": "ocr",
                "image": {"format": "path", "data": _sample_path(ensure_sample_image)},
                "meta": {"normalize_rules": {"trim_space": True, "case_insensitive": False}},
            },
        )

    get_settings.cache_clear()
    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert isinstance(payload["result"]["text"], str)


def test_infer_ocr_accepts_base64_input(client, ensure_sample_image: Path) -> None:
    encoded = b64encode(Path(_sample_path(ensure_sample_image)).read_bytes()).decode("ascii")
    response = client.post(
        "/infer",
        json={
            "request_id": "eval-ocr-b64",
            "session_id": "team-7",
            "task_type": "ocr",
            "image": {"format": "base64", "data": encoded},
            "meta": {"normalize_rules": {"trim_space": True, "case_insensitive": False}},
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is True


def test_infer_invalid_task_type_returns_json_failure(client, ensure_sample_image: Path) -> None:
    response = client.post(
        "/infer",
        json={
            "request_id": "eval-unknown-1",
            "session_id": "team-7",
            "task_type": "keypoint",
            "image": {"format": "path", "data": _sample_path(ensure_sample_image)},
            "meta": {},
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["request_id"] == "eval-unknown-1"
    assert payload["task_type"] == "keypoint"
    assert payload["ok"] is False


def test_image_read_failure_returns_json_failure(client) -> None:
    response = client.post(
        "/infer",
        json={
            "request_id": "eval-ocr-2",
            "session_id": "team-7",
            "task_type": "ocr",
            "image": {"format": "path", "data": "tests/fixtures/missing.jpg"},
            "meta": {},
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is False
    assert "image load failed" in payload["message"]
