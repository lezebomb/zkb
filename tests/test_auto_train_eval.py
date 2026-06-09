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


def test_auto_train_eval_scripts_dry_run(tmp_path: Path) -> None:
    commands = [
        ("scripts/auto_train_eval_classify.py", "--data", str(tmp_path), "--rounds", "3", "--max-minutes", "1", "--dry-run"),
        ("scripts/auto_train_eval_detect.py", "--base-model", "models/detect/yolo11n.pt", "--data", "coco8.yaml", "--rounds", "3", "--max-minutes", "1", "--dry-run"),
    ]
    for command in commands:
        result = run_script(*command)
        assert result.returncode == 0, result.stderr + result.stdout
        assert "dry-run" in result.stdout
