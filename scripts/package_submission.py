from __future__ import annotations

import hashlib
import shutil
import sys
import zipfile
from pathlib import Path


PACKAGE_TARGETS = [
    "requirements.txt",
    "requirements-detect.txt",
    "requirements-ocr.txt",
    "requirements-train.txt",
    "requirements-all.txt",
    "configs",
    "src",
    "scripts",
    "docs",
    "tests",
    "models",
    "README.md",
    "AGENTS.md",
    ".env.example",
    "pyproject.toml",
]

EXCLUDED_NAMES = {".git", ".venv", "dist", "runs", "__pycache__", ".pytest_cache"}
EXCLUDED_PREFIXES = {Path("data/raw"), Path("data/processed")}
MAX_ZIP_BYTES = 2 * 1024 * 1024 * 1024


def _is_excluded(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    if any(part in EXCLUDED_NAMES for part in rel.parts):
        return True
    if any(rel == prefix or prefix in rel.parents for prefix in EXCLUDED_PREFIXES):
        return True
    if path.is_file() and path.suffix.lower() == ".log" and path.stat().st_size > 5 * 1024 * 1024:
        return True
    return False


def _cleanup(root: Path) -> None:
    log_dir = root / "logs"
    if log_dir.exists():
        for path in log_dir.glob("*.log*"):
            try:
                if path.stat().st_size > 5 * 1024 * 1024:
                    path.unlink()
            except FileNotFoundError:
                pass


def build_package(out_dir: Path, package_name: str, root: Path | None = None) -> tuple[Path, Path, str]:
    root = (root or Path.cwd()).resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    _cleanup(root)

    models_dir = root / "models"
    model_total_size = sum(path.stat().st_size for path in models_dir.rglob("*") if path.is_file()) if models_dir.exists() else 0

    zip_path = out_dir / f"{package_name}.zip"
    if zip_path.exists():
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for target in PACKAGE_TARGETS:
            path = root / target
            if not path.exists() or _is_excluded(path, root):
                continue
            if path.is_file():
                zf.write(path, arcname=path.relative_to(root))
                continue
            for child in path.rglob("*"):
                if child.is_dir() or _is_excluded(child, root):
                    continue
                zf.write(child, arcname=child.relative_to(root))

    zip_size = zip_path.stat().st_size
    if zip_size > MAX_ZIP_BYTES:
        raise SystemExit(f"Packaged zip exceeds 2GB limit: {zip_size} bytes")

    sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
    sha_path = zip_path.with_suffix(".zip.sha256")
    sha_path.write_text(f"{sha256}  {zip_path.name}\n", encoding="utf-8")

    print(f"Model payload size: {model_total_size} bytes")
    return zip_path, sha_path, sha256


def main() -> int:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("dist")
    package_name = sys.argv[2] if len(sys.argv) > 2 else "contest_agent_submission"
    zip_path, sha_path, sha256 = build_package(out_dir, package_name)
    print(f"Zip package: {zip_path}")
    print(f"Zip size: {zip_path.stat().st_size} bytes")
    print(f"SHA256: {sha256}")
    print(f"SHA256 file: {sha_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
