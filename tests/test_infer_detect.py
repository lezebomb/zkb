from __future__ import annotations

from pathlib import Path


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
                "class_names": ["人", "汽车", "自行车", "手机", "水杯", "笔记本电脑", "台灯", "沙发", "狗"],
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
        assert target["label"] in ["人", "汽车", "自行车", "手机", "水杯", "笔记本电脑", "台灯", "沙发", "狗"]
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
