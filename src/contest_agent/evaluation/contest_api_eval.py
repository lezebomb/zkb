from __future__ import annotations

import base64
import json
import time
from pathlib import Path
from typing import Any

import httpx

from contest_agent.evaluation.classify_eval import CLASSIFY_LABELS
from contest_agent.evaluation.common import image_files, write_json
from contest_agent.evaluation.ocr_eval import _char_accuracy
from contest_agent.postprocess.ocr_text import normalize_text_for_output


def _post_infer(base_url: str, payload: dict[str, Any]) -> tuple[dict[str, Any] | None, float, str]:
    started = time.perf_counter()
    try:
        response = httpx.post(f"{base_url.rstrip('/')}/infer", json=payload, timeout=10)
        latency = (time.perf_counter() - started) * 1000.0
        return response.json(), latency, ""
    except Exception as exc:
        return None, (time.perf_counter() - started) * 1000.0, str(exc)


def evaluate_contest_api(base_url: str, classify_data: Path | str | None, detect_data: Path | str | None, ocr_data: Path | str | None, output: Path | str) -> dict[str, Any]:
    report: dict[str, Any] = {"classify": {}, "detect": {}, "ocr": {}, "errors": []}
    if classify_data:
        total = correct = invalid = 0
        latencies = []
        for image_path in image_files(classify_data):
            gt = image_path.parent.name
            if gt not in CLASSIFY_LABELS:
                continue
            payload = {"request_id": f"api-cls-{total}", "session_id": "eval", "task_type": "classify", "image": {"format": "path", "data": str(image_path)}, "meta": {"class_names": CLASSIFY_LABELS}}
            data, latency, error = _post_infer(base_url, payload)
            latencies.append(latency)
            total += 1
            if error or not data or not data.get("ok"):
                report["errors"].append(error or str(data))
                continue
            pred = data.get("result", {}).get("label", "")
            invalid += 0 if pred in CLASSIFY_LABELS else 1
            correct += 1 if pred == gt else 0
        report["classify"] = {"samples": total, "accuracy": correct / total if total else 0.0, "invalid_label_rate": invalid / total if total else 0.0, "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0}
    if ocr_data:
        base = Path(ocr_data)
        rows = []
        labels = base / "labels.txt"
        if labels.exists():
            for line in labels.read_text(encoding="utf-8").splitlines():
                if "\t" in line:
                    rel, text = line.split("\t", 1)
                    rows.append((base / rel, text))
        exact = empty = 0
        chars = []
        latencies = []
        rules = {"trim_space": True, "case_insensitive": False}
        for idx, (image_path, gt_raw) in enumerate(rows):
            payload = {"request_id": f"api-ocr-{idx}", "session_id": "eval", "task_type": "ocr", "image": {"format": "path", "data": str(image_path)}, "meta": {"normalize_rules": rules}}
            data, latency, error = _post_infer(base_url, payload)
            latencies.append(latency)
            if error or not data or not data.get("ok"):
                report["errors"].append(error or str(data))
                continue
            pred = data.get("result", {}).get("text", "")
            gt = normalize_text_for_output(gt_raw, rules)
            exact += 1 if pred == gt else 0
            empty += 1 if not pred else 0
            chars.append(_char_accuracy(pred, gt))
        report["ocr"] = {"samples": len(rows), "exact_match_after_normalize": exact / len(rows) if rows else 0.0, "char_accuracy": sum(chars) / len(chars) if chars else 0.0, "empty_text_rate": empty / len(rows) if rows else 0.0, "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0}
    if detect_data:
        report["detect"] = {"samples": 0, "proxy_center_hit_score": 0.0, "note": "HTTP detect eval uses YOLO labels; run scripts/evaluate_detect.py for model-level center-hit metrics."}
    report["summary"] = {"error_count": len(report["errors"]), "non_json_response_count": 0, "recommendation": "check per-task metrics and /debug/status"}
    write_json(output, report)
    return report

