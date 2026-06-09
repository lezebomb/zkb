from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

from contest_agent.training.classify_data import create_tiny_placeholder
from contest_agent.training.ocr_synthetic import generate_dataset

ROOT = Path(__file__).resolve().parents[1]


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src")
    return subprocess.run([sys.executable, *args], cwd=ROOT, env=env, text=True, encoding="utf-8", errors="replace", capture_output=True, timeout=60)


def test_evaluate_classify_and_ocr_scripts(tmp_path: Path) -> None:
    data = create_tiny_placeholder(tmp_path / "cls")
    model = tmp_path / "classifier.pt"
    classes = [p.name for p in sorted((data / "train").iterdir()) if p.is_dir()]
    model.write_bytes(pickle.dumps({"constant_label": classes[0], "classes": classes}))
    (tmp_path / "class_names.json").write_text(json.dumps(classes, ensure_ascii=False), encoding="utf-8")
    cls_out = tmp_path / "classify_metrics.json"
    result = run_script("scripts/evaluate_classify.py", "--model", str(model), "--data", str(data / "val"), "--output", str(cls_out))
    assert result.returncode == 0, result.stderr + result.stdout
    assert "accuracy" in cls_out.read_text(encoding="utf-8")

    ocr_data = generate_dataset(tmp_path / "ocr", 3, seed=1)
    ocr_out = tmp_path / "ocr_metrics.json"
    result = run_script("scripts/evaluate_ocr.py", "--pred-backend", "fallback", "--data", str(ocr_data), "--output", str(ocr_out))
    assert result.returncode == 0, result.stderr + result.stdout
    assert "exact_match_after_normalize" in ocr_out.read_text(encoding="utf-8")


def test_evaluate_detect_missing_model_writes_report(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    root = tmp_path / "detect"
    (root / "labels" / "val").mkdir(parents=True)
    data_yaml.write_text(f"path: {root.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n  0: 人\n", encoding="utf-8")
    out = tmp_path / "detect_metrics.json"
    result = run_script("scripts/evaluate_detect.py", "--model", str(tmp_path / "missing.pt"), "--data", str(data_yaml), "--output", str(out))
    assert result.returncode == 0, result.stderr + result.stdout
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "proxy_center_hit_score" in payload
