from __future__ import annotations

import zipfile
import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("package_submission", ROOT / "scripts" / "package_submission.py")
assert SPEC and SPEC.loader
package_submission = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(package_submission)


def test_package_submission_includes_requirements_and_excludes_work_dirs(tmp_path: Path) -> None:
    zip_path, sha_path, _ = package_submission.build_package(tmp_path, "test_submission", ROOT)
    assert zip_path.exists()
    assert sha_path.exists()
    with zipfile.ZipFile(zip_path) as zf:
        names = set(zf.namelist())
    required = {
        "requirements-detect.txt",
        "requirements-ocr.txt",
        "requirements-all.txt",
        "configs/detect/contest_detect.yaml",
        "src/contest_agent/app.py",
        "scripts/train_detect_yolo.py",
    }
    assert required.issubset(names)
    assert not any(name.startswith(".git/") for name in names)
    assert not any(name.startswith(".venv/") for name in names)
    assert not any(name.startswith("runs/") for name in names)
    assert not any(name.startswith("data/raw/") for name in names)
    assert not any(name.startswith("data/processed/") for name in names)
