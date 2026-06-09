from __future__ import annotations

from contest_agent.training.dataset_registry import registry_entries, render_manifest


def test_registry_generates_manifest() -> None:
    manifest = render_manifest()
    assert "datasets:" in manifest
    assert "COCO" in manifest
    assert "Synthetic OCR" in manifest


def test_registry_covers_all_tasks() -> None:
    entries = registry_entries()
    tasks = {entry["task"] for entry in entries}
    assert {"detect", "classify", "ocr"}.issubset(tasks)


def test_large_datasets_are_not_default_downloads() -> None:
    entries = registry_entries()
    large = [entry for entry in entries if entry["name"] in {"COCO", "Open Images", "LVIS", "Places365", "PaddleOCR official datasets"}]
    assert large
    for entry in large:
        assert entry["default_download"] is False or entry["needs_manual_review"] is True

