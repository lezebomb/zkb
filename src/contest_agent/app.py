from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from pydantic import ValidationError

from contest_agent.config import Settings, get_settings
from contest_agent.image_io import ImageLoadError, load_image
from contest_agent.inference import (
    build_classifier_backend,
    build_detector_backend,
    build_ocr_backend,
)
from contest_agent.logging_utils import configure_logging, summarize_result
from contest_agent.schemas import HealthResponse, InferRequest
from contest_agent.utils import elapsed_ms, normalize_task_type
from contest_agent.validators import (
    validate_classify_result,
    validate_detect_result,
    validate_ocr_result,
)


def _display_path(settings: Settings, path: Any) -> str:
    try:
        return str(Path(path).resolve().relative_to(settings.model_detect_path.parents[1]))
    except Exception:
        try:
            return str(Path(path))
        except Exception:
            return str(path)


def _ocr_model_dirs_exist(settings: Settings) -> bool:
    required = [settings.ocr_det_model_dir, settings.ocr_rec_model_dir]
    if settings.ocr_use_angle_cls:
        required.append(settings.ocr_cls_model_dir)
    return all(path.exists() for path in required)


def _file_size_mb(path: Path) -> float | None:
    try:
        if path.exists() and path.is_file():
            return round(path.stat().st_size / (1024 * 1024), 4)
    except OSError:
        return None
    return None


def _manifest_sha256(settings: Settings, model_path: Path) -> str | None:
    manifest_path = settings.model_detect_path.parents[1] / "model_manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    rel_candidates = {str(model_path), str(model_path.as_posix())}
    try:
        rel_candidates.add(str(model_path.resolve().relative_to(settings.model_detect_path.parents[1]).as_posix()))
    except Exception:
        pass
    for item in manifest.get("models", []):
        if isinstance(item, dict) and str(item.get("path")) in rel_candidates:
            sha = item.get("sha256")
            return str(sha) if sha else None
    return None


def _backend_status(settings: Settings) -> dict[str, Any]:
    classify_effective = "fallback"
    detect_effective = "fallback"
    ocr_effective = "fallback"

    if settings.detect_backend in {"local", "ultralytics"}:
        detect_effective = "ultralytics_or_fallback"
    if settings.ocr_backend in {"local", "paddleocr"}:
        ocr_effective = "paddleocr_or_fallback"

    return {
        "classify": {
            "configured": settings.classify_backend,
            "effective": classify_effective,
            "model_path": _display_path(settings, settings.model_classify_path),
            "model_exists": settings.model_classify_path.exists(),
            "class_names_path": _display_path(settings, settings.model_classify_path.parent / "class_names.json"),
            "class_names_exists": (settings.model_classify_path.parent / "class_names.json").exists(),
            "model_size_mb": _file_size_mb(settings.model_classify_path),
        },
        "detect": {
            "configured": settings.detect_backend,
            "effective": detect_effective,
            "model_path": _display_path(settings, settings.model_detect_path),
            "model_exists": settings.model_detect_path.exists(),
            "model_size_mb": _file_size_mb(settings.model_detect_path),
            "manifest_sha256": _manifest_sha256(settings, settings.model_detect_path),
        },
        "ocr": {
            "configured": settings.ocr_backend,
            "effective": ocr_effective,
            "model_path": _display_path(settings, settings.model_ocr_path),
            "model_dirs_exist": _ocr_model_dirs_exist(settings),
            "dirs": {
                "det": settings.ocr_det_model_dir.exists(),
                "rec": settings.ocr_rec_model_dir.exists(),
                "cls": settings.ocr_cls_model_dir.exists(),
            },
        },
    }


def _extract_request_envelope(raw_body: bytes) -> tuple[str, str]:
    if not raw_body:
        return "", ""
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception:
        return "", ""
    if not isinstance(payload, dict):
        return "", ""
    request_id = str(payload.get("request_id", "")).strip()
    task_type = normalize_task_type(str(payload.get("task_type", ""))) if payload.get("task_type") else ""
    return request_id, task_type


def _failure_response(
    request_id: str,
    task_type: str,
    started_at: float,
    message: str,
    *,
    status_code: int = 200,
) -> JSONResponse:
    return JSONResponse(
        status_code=status_code,
        content={
            "request_id": request_id,
            "task_type": task_type,
            "ok": False,
            "result": None,
            "elapsed_ms": elapsed_ms(started_at),
            "message": message,
        },
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    app_settings = settings or get_settings()
    logger = configure_logging(app_settings)

    app = FastAPI(title=app_settings.app_name, version=app_settings.app_version)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    classifier_backend = build_classifier_backend(app_settings, logger)
    detector_backend = build_detector_backend(app_settings, logger)
    ocr_backend = build_ocr_backend(app_settings, logger)

    @app.exception_handler(RequestValidationError)
    async def request_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        raw_body = await request.body()
        request_id, task_type = _extract_request_envelope(raw_body)
        return _failure_response(request_id, task_type, time.perf_counter(), f"request validation error: {exc.errors()}")

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        raw_body = await request.body()
        request_id, task_type = _extract_request_envelope(raw_body)
        logger.exception("unhandled_exception path=%s", request.url.path)
        return _failure_response(request_id, task_type, time.perf_counter(), f"internal error: {exc}")

    @app.get("/health", response_model=HealthResponse)
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "supported_tasks": list(app_settings.supported_tasks),
            "service": app_settings.app_name,
            "version": app_settings.app_version,
            "bridge_mode": "mock-local",
        }

    @app.get("/debug/status")
    async def debug_status() -> dict[str, Any]:
        return {
            "service": app_settings.app_name,
            "version": app_settings.app_version,
            "offline_mode": app_settings.offline_mode,
            "allow_model_auto_download": app_settings.allow_model_auto_download,
            "backends": _backend_status(app_settings),
        }

    @app.options("/{path:path}")
    async def options_handler(path: str) -> Response:
        return Response(status_code=204)

    @app.post("/infer")
    async def infer(request: Request) -> JSONResponse:
        started_at = time.perf_counter()
        raw_body = await request.body()
        request_id, raw_task_type = _extract_request_envelope(raw_body)

        content_type = request.headers.get("content-type", "")
        if "application/json" not in content_type.lower():
            return _failure_response(
                request_id,
                raw_task_type,
                started_at,
                "Content-Type must be application/json",
            )

        try:
            payload_dict = json.loads(raw_body.decode("utf-8"))
        except Exception:
            return _failure_response(request_id, raw_task_type, started_at, "request body is not valid JSON")

        try:
            payload = InferRequest.model_validate(payload_dict)
        except ValidationError as exc:
            return _failure_response(
                request_id,
                raw_task_type,
                started_at,
                f"invalid request fields: {exc.errors()}",
            )

        task_type = normalize_task_type(payload.task_type)
        infer_t_max_ms = int(payload.meta.get("infer_T_max_ms", 0) or 0)

        if task_type not in app_settings.supported_tasks:
            return _failure_response(
                payload.request_id,
                task_type,
                started_at,
                f"不支持的 task_type: {task_type}，当前仅支持 {list(app_settings.supported_tasks)}",
            )

        if task_type == "detect" and "coord_mode" in payload.meta:
            coord_mode = str(payload.meta.get("coord_mode", "")).strip().lower()
            if coord_mode != "pixel":
                return _failure_response(
                    payload.request_id,
                    task_type,
                    started_at,
                    f"coord_mode must be pixel, got: {payload.meta.get('coord_mode')}",
                )

        try:
            image = load_image(payload.image, app_settings)
        except ImageLoadError as exc:
            return _failure_response(payload.request_id, task_type, started_at, f"image load failed: {exc}")

        try:
            if task_type == "classify":
                raw_result = classifier_backend.predict(image, payload.meta)
                result = validate_classify_result(raw_result, payload.meta)
            elif task_type == "detect":
                raw_result = detector_backend.predict(image, payload.meta)
                result = validate_detect_result(
                    raw_result,
                    payload.meta,
                    int(payload.meta.get("image_width") or image.width),
                    int(payload.meta.get("image_height") or image.height),
                    allow_empty=True,
                    fill_when_empty=app_settings.detect_empty_fallback,
                )
            else:
                raw_result = ocr_backend.predict(image, payload.meta)
                result = validate_ocr_result(raw_result, payload.meta)
        except Exception as exc:
            logger.exception("inference_failed request_id=%s task_type=%s", payload.request_id, task_type)
            return _failure_response(payload.request_id, task_type, started_at, f"inference failed: {exc}")

        response_content = {
            "request_id": payload.request_id,
            "task_type": task_type,
            "ok": True,
            "result": result,
            "elapsed_ms": elapsed_ms(started_at),
            "message": "",
        }

        difficulty = payload.meta.get("difficulty", "")
        logger.info(
            "request_id=%s task_type=%s difficulty=%s image=%s image_size=%sx%s infer_T_max_ms=%s elapsed_ms=%s ok=%s message=%r result=%s",
            payload.request_id,
            task_type,
            difficulty,
            payload.image.format,
            image.width,
            image.height,
            infer_t_max_ms,
            response_content["elapsed_ms"],
            response_content["ok"],
            response_content["message"],
            summarize_result(result),
        )
        if infer_t_max_ms and response_content["elapsed_ms"] > infer_t_max_ms:
            logger.warning(
                "soft_timeout_exceeded request_id=%s task_type=%s allowed_ms=%s actual_ms=%s",
                payload.request_id,
                task_type,
                infer_t_max_ms,
                response_content["elapsed_ms"],
            )

        return JSONResponse(status_code=200, content=response_content)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    current_settings = get_settings()
    uvicorn.run(
        "contest_agent.app:app",
        host=current_settings.app_host,
        port=current_settings.app_port,
        reload=False,
    )
