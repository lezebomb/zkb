from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from contest_agent.app import create_app
from contest_agent.config import get_settings


DETECT_LABELS = ["人", "汽车", "自行车", "手机", "水杯", "笔记本电脑", "台灯", "沙发", "狗"]


def _sample_path(ensure_sample_image: Path) -> str:
    return str(ensure_sample_image)


def test_infer_detect_returns_pixel_targets(client, ensure_sample_image: Path) -> None:
    response = client.post(
        "/infer",
        json={
            "request_id": "eval-detect-1",
            "session_id": "team-7",
            "task_type": "detect",
            "image": {"format": "path", "data": _sample_path(ensure_sample_image)},
            "meta": {
                "difficulty": "L2",
                "coord_mode": "pixel",
                "class_names": DETECT_LABELS,
                "image_width": 320,
                "image_height": 240,
            },
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is True
    targets = payload["result"]["targets"]
    assert targets
    for target in targets:
        assert target["label"] in DETECT_LABELS
        assert 0 <= target["cx"] <= 319
        assert 0 <= target["cy"] <= 239


def test_infer_detect_invalid_coord_mode_returns_json_failure(client, ensure_sample_image: Path) -> None:
    response = client.post(
        "/infer",
        json={
            "request_id": "eval-detect-2",
            "session_id": "team-7",
            "task_type": "detect",
            "image": {"format": "path", "data": _sample_path(ensure_sample_image)},
            "meta": {"coord_mode": "normalized"},
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is False
    assert "coord_mode" in payload["message"]


def test_ultralytics_missing_model_falls_back_safely(monkeypatch, ensure_sample_image: Path) -> None:
    monkeypatch.setenv("DETECT_BACKEND", "ultralytics")
    monkeypatch.setenv("MODEL_DETECT_PATH", "models/does-not-exist.pt")
    monkeypatch.setenv("ALLOW_MODEL_AUTO_DOWNLOAD", "false")
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as current_client:
        response = current_client.post(
            "/infer",
            json={
                "request_id": "eval-detect-3",
                "session_id": "team-7",
                "task_type": "detect",
                "image": {"format": "path", "data": _sample_path(ensure_sample_image)},
                "meta": {
                    "coord_mode": "pixel",
                    "class_names": DETECT_LABELS,
                    "image_width": 320,
                    "image_height": 240,
                },
            },
        )

    get_settings.cache_clear()
    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert "targets" in payload["result"]
