from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from contest_agent.evaluation.common import image_files, make_image_record, timed_call, write_json
from contest_agent.inference.classifier import TorchClassifierBackend
from contest_agent.postprocess.labels import CLASSIFY_LABELS
from contest_agent.validators import validate_classify_result


def evaluate_classify(model_path: Path | str, data_dir: Path | str, output: Path | str) -> dict[str, Any]:
    backend = TorchClassifierBackend(Path(model_path))
    files = image_files(data_dir)
    total = 0
    correct = 0
    invalid = 0
    latencies: list[float] = []
    per_total: dict[str, int] = defaultdict(int)
    per_correct: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for image_path in files:
        gt = image_path.parent.name
        if gt not in CLASSIFY_LABELS:
            continue
        record = make_image_record(image_path)
        raw, latency = timed_call(lambda: backend.predict(record, {"class_names": CLASSIFY_LABELS}))
        pred = validate_classify_result(raw, {"class_names": CLASSIFY_LABELS})["label"]
        total += 1
        per_total[gt] += 1
        latencies.append(latency)
        if pred not in CLASSIFY_LABELS:
            invalid += 1
        if pred == gt:
            correct += 1
            per_correct[gt] += 1
        confusion[gt][pred] += 1
    metrics = {
        "samples": total,
        "accuracy": round(correct / total, 6) if total else 0.0,
        "per_class_accuracy": {label: round(per_correct[label] / per_total[label], 6) if per_total[label] else 0.0 for label in CLASSIFY_LABELS},
        "invalid_label_rate": round(invalid / total, 6) if total else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "confusion_matrix": {gt: dict(preds) for gt, preds in confusion.items()},
    }
    write_json(output, metrics)
    return metrics

