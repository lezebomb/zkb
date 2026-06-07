from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


@pytest.fixture(scope="session", autouse=True)
def ensure_sample_image() -> Path:
    fixtures_dir = ROOT_DIR / "tests" / "fixtures"
    fixtures_dir.mkdir(parents=True, exist_ok=True)
    sample_path = fixtures_dir / "sample.jpg"
    if not sample_path.exists():
        image = Image.new("RGB", (320, 240), color=(242, 245, 248))
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 40, 150, 170), outline=(33, 90, 180), width=4)
        draw.text((44, 94), "DEMO 12.8%", fill=(10, 10, 10))
        image.save(sample_path, format="JPEG")
    return sample_path


@pytest.fixture()
def client() -> TestClient:
    from contest_agent.app import create_app
    from contest_agent.config import get_settings

    get_settings.cache_clear()
    test_app = create_app()
    with TestClient(test_app) as current_client:
        yield current_client
    get_settings.cache_clear()
