from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

from contest_agent import __version__


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_LOG_FILE = ROOT_DIR / "logs" / "app.log"


def _load_env_file() -> None:
    env_path = ROOT_DIR / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_host: str
    app_port: int
    app_version: str
    offline_mode: bool
    allow_model_auto_download: bool
    mock_allow_expected_text: bool
    model_classify_path: Path
    model_detect_path: Path
    model_ocr_path: Path
    classify_backend: str
    detect_backend: str
    ocr_backend: str
    detect_score_threshold: float
    detect_max_targets: int
    detect_empty_fallback: bool
    detect_device: str
    detect_imgsz: int
    ocr_use_angle_cls: bool
    ocr_lang: str
    ocr_det_model_dir: Path
    ocr_rec_model_dir: Path
    ocr_cls_model_dir: Path
    ocr_max_text_length: int
    image_download_timeout_seconds: int
    max_image_size_mb: int
    log_level: str
    log_file: Path
    supported_tasks: tuple[str, ...]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_env_file()
    return Settings(
        app_name="contestant-algo-test-service",
        app_host=os.getenv("APP_HOST", "0.0.0.0"),
        app_port=_get_int("APP_PORT", 8080),
        app_version=os.getenv("APP_VERSION", __version__),
        offline_mode=_get_bool("OFFLINE_MODE", True),
        allow_model_auto_download=_get_bool("ALLOW_MODEL_AUTO_DOWNLOAD", False),
        mock_allow_expected_text=_get_bool("MOCK_ALLOW_EXPECTED_TEXT", False),
        model_classify_path=ROOT_DIR / os.getenv("MODEL_CLASSIFY_PATH", "models/classifier.onnx"),
        model_detect_path=ROOT_DIR / os.getenv("MODEL_DETECT_PATH", "models/yolo11n.pt"),
        model_ocr_path=ROOT_DIR / os.getenv("MODEL_OCR_PATH", "models/ocr"),
        classify_backend=os.getenv("CLASSIFY_BACKEND", "fallback").strip().lower(),
        detect_backend=os.getenv("DETECT_BACKEND", "fallback").strip().lower(),
        ocr_backend=os.getenv("OCR_BACKEND", "fallback").strip().lower(),
        detect_score_threshold=_get_float("DETECT_SCORE_THRESHOLD", 0.25),
        detect_max_targets=_get_int("DETECT_MAX_TARGETS", 50),
        detect_empty_fallback=_get_bool("DETECT_EMPTY_FALLBACK", False),
        detect_device=os.getenv("DETECT_DEVICE", "cpu").strip(),
        detect_imgsz=_get_int("DETECT_IMGSZ", 640),
        ocr_use_angle_cls=_get_bool("OCR_USE_ANGLE_CLS", True),
        ocr_lang=os.getenv("OCR_LANG", "ch").strip(),
        ocr_det_model_dir=ROOT_DIR / os.getenv("OCR_DET_MODEL_DIR", "models/paddleocr/det"),
        ocr_rec_model_dir=ROOT_DIR / os.getenv("OCR_REC_MODEL_DIR", "models/paddleocr/rec"),
        ocr_cls_model_dir=ROOT_DIR / os.getenv("OCR_CLS_MODEL_DIR", "models/paddleocr/cls"),
        ocr_max_text_length=_get_int("OCR_MAX_TEXT_LENGTH", 128),
        image_download_timeout_seconds=_get_int("IMAGE_DOWNLOAD_TIMEOUT_SECONDS", 5),
        max_image_size_mb=_get_int("MAX_IMAGE_SIZE_MB", 20),
        log_level=os.getenv("LOG_LEVEL", "INFO").strip().upper(),
        log_file=ROOT_DIR / os.getenv("LOG_FILE", str(DEFAULT_LOG_FILE.relative_to(ROOT_DIR))),
        supported_tasks=("classify", "ocr", "detect"),
    )
