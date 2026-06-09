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


def test_verify_training_setup_runs_without_deps() -> None:
    result = run_script("scripts/verify_training_setup.py")
    assert result.returncode == 0
    assert "python" in result.stdout.lower()


def test_generate_synthetic_ocr_dataset_count_5(tmp_path: Path) -> None:
    out = tmp_path / "ocr"
    result = run_script("scripts/generate_synthetic_ocr_dataset.py", "--output", str(out), "--count", "5", "--seed", "7")
    assert result.returncode == 0, result.stderr + result.stdout
    assert (out / "labels.txt").exists()
    assert len((out / "labels.txt").read_text(encoding="utf-8").strip().splitlines()) == 5
    assert len(list((out / "images").glob("*.png"))) == 5


def test_prepare_classify_dataset_template(tmp_path: Path) -> None:
    out = tmp_path / "contest8"
    result = run_script("scripts/prepare_classify_dataset.py", "--mode", "template", "--output", str(out))
    assert result.returncode == 0, result.stderr + result.stdout
    assert len([path for path in (out / "train").iterdir() if path.is_dir()]) == 8
    assert len([path for path in (out / "val").iterdir() if path.is_dir()]) == 8


def test_training_scripts_dry_run() -> None:
    commands = [
        ("scripts/train_detect_yolo.py", "--model", "models/detect/yolo11n.pt", "--data", "configs/detect/coco8_contest.yaml", "--dry-run"),
        ("scripts/train_classify.py", "--data", "data/processed/classify/contest8", "--dry-run"),
        ("scripts/train_ocr_paddle.py", "--dry-run"),
    ]
    for command in commands:
        result = run_script(*command)
        assert result.returncode == 0, result.stderr + result.stdout
        assert "dry-run" in result.stdout.lower()
