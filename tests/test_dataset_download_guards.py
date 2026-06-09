from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=60)


def test_places365_download_guard(tmp_path: Path) -> None:
    out = tmp_path / "places"
    result = run_script("scripts/download_places365_subset.py", "--classes", "office", "park", "--output", str(out))
    assert result.returncode == 0
    assert (out / "download_todo.md").exists()


def test_openimages_download_guard(tmp_path: Path) -> None:
    out = tmp_path / "openimages"
    result = run_script("scripts/download_openimages_subset.py", "--classes", "Mobile phone", "Desk lamp", "--output", str(out))
    assert result.returncode == 0
    assert (out / "download_todo.md").exists()
    assert (out / "data.yaml").exists()
