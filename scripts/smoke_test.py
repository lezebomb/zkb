from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path
from urllib import request

from PIL import Image, ImageDraw


CLASSIFY_LABELS = ["办公室", "公园", "街道", "商场", "厨房", "卧室", "图书馆", "体育馆"]
DETECT_LABELS = ["人", "汽车", "自行车", "手机", "水杯", "笔记本电脑", "台灯", "沙发", "狗"]


def _create_sample_image(path: Path) -> None:
    image = Image.new("RGB", (320, 240), color=(244, 247, 250))
    draw = ImageDraw.Draw(image)
    draw.rectangle((40, 50, 160, 180), outline=(20, 80, 160), width=4)
    draw.text((55, 95), "Valve 12.8%", fill=(10, 10, 10))
    image.save(path, format="JPEG")


def _post_json(base_url: str, payload: dict) -> dict:
    req = request.Request(
        url=f"{base_url}/infer",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8080"
    tmp_dir = Path(tempfile.mkdtemp(prefix="contest-agent-"))
    try:
        sample_path = tmp_dir / "sample.jpg"
        _create_sample_image(sample_path)

        print(f"[1/4] GET {base_url}/health")
        with request.urlopen(f"{base_url}/health", timeout=10) as resp:
            health = json.loads(resp.read().decode("utf-8"))
        print(json.dumps(health, ensure_ascii=False, indent=2))
        assert health["status"] == "ok"

        print("[2/4] POST classify")
        classify = _post_json(
            base_url,
            {
                "request_id": "smoke-classify-1",
                "session_id": "local-smoke",
                "task_type": "classify",
                "image": {"format": "path", "data": str(sample_path)},
                "meta": {"difficulty": "L1", "class_names": CLASSIFY_LABELS},
            },
        )
        print(json.dumps(classify, ensure_ascii=False, indent=2))
        assert classify["ok"] is True
        assert classify["result"]["label"] in CLASSIFY_LABELS

        print("[3/4] POST detect")
        detect = _post_json(
            base_url,
            {
                "request_id": "smoke-detect-1",
                "session_id": "local-smoke",
                "task_type": "detect",
                "image": {"format": "path", "data": str(sample_path)},
                "meta": {
                    "difficulty": "L1",
                    "coord_mode": "pixel",
                    "image_width": 320,
                    "image_height": 240,
                    "class_names": DETECT_LABELS,
                },
            },
        )
        print(json.dumps(detect, ensure_ascii=False, indent=2))
        assert detect["ok"] is True
        assert "targets" in detect["result"]

        print("[4/4] POST ocr")
        ocr = _post_json(
            base_url,
            {
                "request_id": "smoke-ocr-1",
                "session_id": "local-smoke",
                "task_type": "ocr",
                "image": {"format": "path", "data": str(sample_path)},
                "meta": {
                    "difficulty": "L1",
                    "language_hint": "zh",
                    "normalize_rules": {"trim_space": True, "case_insensitive": False},
                },
            },
        )
        print(json.dumps(ocr, ensure_ascii=False, indent=2))
        assert ocr["ok"] is True
        assert "text" in ocr["result"]

        print("Smoke test passed.")
        return 0
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
