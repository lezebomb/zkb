from __future__ import annotations

import http.server
import socketserver
import sys
import threading
from base64 import b64encode
from pathlib import Path
from types import SimpleNamespace

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


def test_detect_accepts_url_image_input(client, ensure_sample_image: Path, tmp_path: Path) -> None:
    served_image = tmp_path / "served.jpg"
    served_image.write_bytes(Path(_sample_path(ensure_sample_image)).read_bytes())

    class QuietHandler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            return

    handler = lambda *args, **kwargs: QuietHandler(*args, directory=str(tmp_path), **kwargs)
    with socketserver.TCPServer(("127.0.0.1", 0), handler) as server:
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            response = client.post(
                "/infer",
                json={
                    "request_id": "eval-detect-url",
                    "session_id": "team-7",
                    "task_type": "detect",
                    "image": {"format": "url", "data": f"http://127.0.0.1:{port}/served.jpg"},
                    "meta": {
                        "coord_mode": "pixel",
                        "class_names": DETECT_LABELS,
                        "image_width": 320,
                        "image_height": 240,
                    },
                },
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)

    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is True
    assert payload["message"] == ""
    assert "targets" in payload["result"]
    for target in payload["result"]["targets"]:
        assert 0 <= target["cx"] <= 319
        assert 0 <= target["cy"] <= 239


def test_detect_with_fake_ultralytics_backend(monkeypatch, ensure_sample_image: Path, tmp_path: Path) -> None:
    class FakeTensor:
        def __init__(self, values):
            self._values = values

        def tolist(self):
            return self._values

    class FakeBoxes:
        xyxy = FakeTensor([[10.0, 20.0, 110.0, 220.0]])
        cls = FakeTensor([0])
        conf = FakeTensor([0.93])

    class FakeResult:
        boxes = FakeBoxes()
        names = {0: "person"}

    class FakeYOLO:
        last_kwargs = None

        def __init__(self, source: str):
            self.source = source
            self.names = {0: "person"}

        def predict(self, image, **kwargs):
            FakeYOLO.last_kwargs = kwargs
            return [FakeResult()]

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=FakeYOLO))
    model_path = tmp_path / "fake.pt"
    model_path.write_bytes(b"fake")
    monkeypatch.setenv("DETECT_BACKEND", "ultralytics")
    monkeypatch.setenv("MODEL_DETECT_PATH", str(model_path))
    monkeypatch.setenv("DETECT_SCORE_THRESHOLD", "0.25")
    monkeypatch.setenv("DETECT_MAX_TARGETS", "50")
    monkeypatch.setenv("DETECT_DEVICE", "cpu")
    monkeypatch.setenv("DETECT_IMGSZ", "640")
    get_settings.cache_clear()

    app = create_app()
    with TestClient(app) as current_client:
        response = current_client.post(
            "/infer",
            json={
                "request_id": "eval-detect-fake-yolo",
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
    assert payload["result"]["targets"][0]["label"] == "人"
    assert payload["result"]["targets"][0]["cx"] == 60.0
    assert payload["result"]["targets"][0]["cy"] == 120.0
    assert FakeYOLO.last_kwargs["save"] is False
    assert FakeYOLO.last_kwargs["device"] == "cpu"
    assert FakeYOLO.last_kwargs["imgsz"] == 640


def test_detect_accepts_base64_image_input(client, ensure_sample_image: Path) -> None:
    encoded = b64encode(Path(_sample_path(ensure_sample_image)).read_bytes()).decode("ascii")
    response = client.post(
        "/infer",
        json={
            "request_id": "eval-detect-b64",
            "session_id": "team-7",
            "task_type": "detect",
            "image": {"format": "base64", "data": encoded},
            "meta": {
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
