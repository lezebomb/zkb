from __future__ import annotations

from pathlib import Path
from typing import Any

from contest_agent.evaluation.common import make_image_record, timed_call, write_json
from contest_agent.config import get_settings
from contest_agent.inference.fallback import FallbackOCRBackend
from contest_agent.postprocess.ocr_text import normalize_text_for_output
from contest_agent.validators import validate_ocr_result


def _char_accuracy(pred: str, gt: str) -> float:
    if not gt:
        return 1.0 if not pred else 0.0
    matches = sum(1 for a, b in zip(pred, gt) if a == b)
    return matches / max(len(gt), len(pred), 1)


def evaluate_ocr(data_dir: Path | str, output: Path | str, backend: str = "fallback") -> dict[str, Any]:
    base = Path(data_dir)
    labels = base / "labels.txt"
    rows = []
    if labels.exists():
        for line in labels.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rel, text = line.split("\t", 1)
            rows.append((base / rel, text))
    ocr = FallbackOCRBackend(get_settings())
    exact = 0
    empty = 0
    char_scores: list[float] = []
    latencies: list[float] = []
    rules = {"trim_space": True, "case_insensitive": False}
    for image_path, gt_text in rows:
        record = make_image_record(image_path)
        raw, latency = timed_call(lambda: ocr.predict(record, {"normalize_rules": rules}))
        pred = validate_ocr_result(raw, {"normalize_rules": rules})["text"]
        gt = normalize_text_for_output(gt_text, rules)
        latencies.append(latency)
        if not pred:
            empty += 1
        if pred == gt:
            exact += 1
        char_scores.append(_char_accuracy(pred, gt))
    total = len(rows)
    metrics = {
        "backend": backend,
        "samples": total,
        "exact_match_after_normalize": round(exact / total, 6) if total else 0.0,
        "char_accuracy": round(sum(char_scores) / len(char_scores), 6) if char_scores else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "empty_text_rate": round(empty / total, 6) if total else 0.0,
    }
    write_json(output, metrics)
    return metrics
