from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

from contest_agent.config import Settings
from contest_agent.utils import summarize_text


def configure_logging(settings: Settings) -> logging.Logger:
    logger = logging.getLogger("contest_agent")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level, logging.INFO))
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = RotatingFileHandler(
        filename=settings.log_file,
        maxBytes=2 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.propagate = False
    return logger


def summarize_result(result: dict[str, Any] | None) -> str:
    if not result:
        return "null"
    if "label" in result:
        return f"label={result.get('label')}"
    if "text" in result:
        return f"text={summarize_text(str(result.get('text', '')), max_length=48)}"
    if "targets" in result:
        targets = result.get("targets")
        if isinstance(targets, list):
            labels = [str(target.get("label", "?")) for target in targets[:3] if isinstance(target, dict)]
            return f"targets={len(targets)} labels={labels}"
    return summarize_text(str(result), max_length=80)
