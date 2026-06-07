from __future__ import annotations

from pathlib import Path


def _sample_path(ensure_sample_image: Path) -> str:
    return str(ensure_sample_image)


def test_infer_classify_returns_valid_label(client, ensure_sample_image: Path) -> None:
    response = client.post(
        "/infer",
        json={
            "request_id": "eval-classify-1",
            "session_id": "team-7",
            "task_type": " classify ",
            "image": {"format": "path", "data": _sample_path(ensure_sample_image)},
            "meta": {
                "difficulty": "L1",
                "class_names": ["办公室", "公园", "街道", "商场", "厨房", "卧室", "图书馆", "体育馆"],
            },
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["request_id"] == "eval-classify-1"
    assert payload["task_type"] == "classify"
    assert payload["ok"] is True
    assert payload["result"]["label"] in ["办公室", "公园", "街道", "商场", "厨房", "卧室", "图书馆", "体育馆"]
    assert payload["message"] == ""


def test_infer_classify_meta_missing_does_not_crash(client, ensure_sample_image: Path) -> None:
    response = client.post(
        "/infer",
        json={
            "request_id": "eval-classify-2",
            "session_id": "team-7",
            "task_type": "classify",
            "image": {"format": "path", "data": _sample_path(ensure_sample_image)},
        },
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["result"]["label"]
