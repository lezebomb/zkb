from __future__ import annotations

from pathlib import Path


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
