from __future__ import annotations

import hashlib
import shutil
import zipfile
from pathlib import Path
import sys


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
    package_name = sys.argv[2] if len(sys.argv) > 2 else "contest_agent_submission"
    root = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    for pattern in ("__pycache__", ".pytest_cache"):
        for path in root.rglob(pattern):
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)

    log_dir = root / "logs"
    if log_dir.exists():
        for path in log_dir.glob("*.log*"):
            try:
                if path.stat().st_size > 5 * 1024 * 1024:
                    path.unlink()
            except FileNotFoundError:
                pass

    package_targets = [
        "src",
        "scripts",
        "docs",
        "tests",
        "models",
        "README.md",
        "AGENTS.md",
        "requirements.txt",
        "pyproject.toml",
        ".env.example",
    ]

    model_total_size = 0
    models_dir = root / "models"
    if models_dir.exists():
        for model_file in models_dir.rglob("*"):
            if model_file.is_file():
                model_total_size += model_file.stat().st_size

    zip_path = out_dir / f"{package_name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for target in package_targets:
            path = root / target
            if not path.exists():
                continue
            if path.is_file():
                zf.write(path, arcname=path.relative_to(root))
                continue
            for child in path.rglob("*"):
                if child.is_dir():
                    continue
                if ".venv" in child.parts:
                    continue
                zf.write(child, arcname=child.relative_to(root))

    zip_size = zip_path.stat().st_size
    limit = 2 * 1024 * 1024 * 1024
    if zip_size > limit:
        raise SystemExit(f"Packaged zip exceeds 2GB limit: {zip_size} bytes")

    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    sha_path = zip_path.with_suffix(".zip.sha256")
    sha_path.write_text(f"{sha256}  {zip_path.name}\n", encoding="utf-8")

    print(f"Model payload size: {model_total_size} bytes")
    print(f"Zip package: {zip_path}")
    print(f"Zip size: {zip_size} bytes")
    print(f"SHA256: {sha256}")
    print(f"SHA256 file: {sha_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
