from __future__ import annotations

from collections import defaultdict
import os
from pathlib import Path
from typing import Any

from PIL import Image

from contest_agent.postprocess.labels import DETECT_LABELS, normalize_detect_label
from contest_agent.training.detect_data import DETECT_LABELS as TRAIN_DETECT_LABELS
from contest_agent.evaluation.common import timed_call, write_json


os.environ.setdefault("YOLO_CONFIG_DIR", str(Path.cwd() / ".ultralytics"))


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml

        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        data: dict[str, Any] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" in line and not line.startswith(" "):
                key, value = line.split(":", 1)
                data[key.strip()] = value.strip()
        return data


def _label_files(data_yaml: Path) -> list[Path]:
    data = _parse_simple_yaml(data_yaml)
    root = Path(str(data.get("path") or data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve() if str(data.get("path", "")).startswith(".") else Path.cwd() / root
    label_roots = [root / "labels" / "val", root / "labels" / "train"]
    files: list[Path] = []
    for label_root in label_roots:
        if label_root.exists():
            files.extend(sorted(label_root.glob("*.txt")))
            break
    return files


def _image_for_label(label_path: Path) -> Path | None:
    parts = list(label_path.parts)
    try:
        idx = parts.index("labels")
        parts[idx] = "images"
    except ValueError:
        return None
    stem_path = Path(*parts).with_suffix("")
    for suffix in (".jpg", ".jpeg", ".png", ".bmp"):
        candidate = stem_path.with_suffix(suffix)
        if candidate.exists():
            return candidate
    return None


def _gt_boxes(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, x, y, w, h = int(float(parts[0])), *[float(v) for v in parts[1:5]]
        label = TRAIN_DETECT_LABELS[cls] if 0 <= cls < len(TRAIN_DETECT_LABELS) else str(cls)
        boxes.append({"label": label, "x1": (x - w / 2) * width, "y1": (y - h / 2) * height, "x2": (x + w / 2) * width, "y2": (y + h / 2) * height})
    return boxes


def _center_hit(pred: dict[str, Any], gt: dict[str, Any]) -> bool:
    return pred.get("label") == gt.get("label") and gt["x1"] <= float(pred.get("cx", -1)) <= gt["x2"] and gt["y1"] <= float(pred.get("cy", -1)) <= gt["y2"]


def evaluate_detect(model_path: Path | str, data_yaml: Path | str, output: Path | str, conf: float = 0.25, imgsz: int = 320, device: str = "cpu") -> dict[str, Any]:
    labels = _label_files(Path(data_yaml))
    total_gt = 0
    hits = 0
    per_total: dict[str, int] = defaultdict(int)
    per_hit: dict[str, int] = defaultdict(int)
    confs: list[float] = []
    latencies: list[float] = []
    empty = 0
    invalid = 0
    error = ""
    model = None
    try:
        from ultralytics import YOLO

        if Path(model_path).exists():
            model = YOLO(str(model_path))
        else:
            error = f"model missing: {model_path}"
    except Exception as exc:
        error = f"ultralytics unavailable: {exc}"
    for label_path in labels:
        image_path = _image_for_label(label_path)
        if image_path is None:
            continue
        with Image.open(image_path) as image:
            width, height = image.size
        gt_boxes = _gt_boxes(label_path, width, height)
        total_gt += len(gt_boxes)
        for gt in gt_boxes:
            per_total[gt["label"]] += 1
        preds: list[dict[str, Any]] = []
        if model is not None:
            try:
                results, latency = timed_call(lambda: model.predict(str(image_path), conf=conf, imgsz=imgsz, device=device, save=False, verbose=False))
                latencies.append(latency)
                for result in results or []:
                    boxes = getattr(result, "boxes", None)
                    names = getattr(result, "names", None) or getattr(model, "names", {})
                    if boxes is None:
                        continue
                    for xyxy, cls_idx, score in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist()):
                        raw = names.get(int(cls_idx), str(cls_idx)) if isinstance(names, dict) else str(cls_idx)
                        label = normalize_detect_label(raw)
                        if label not in DETECT_LABELS:
                            invalid += 1
                            continue
                        x1, y1, x2, y2 = [float(v) for v in xyxy]
                        confs.append(float(score))
                        preds.append({"label": label, "cx": (x1 + x2) / 2, "cy": (y1 + y2) / 2, "score": float(score)})
            except Exception as exc:
                error = f"predict failed: {exc}"
        if not preds:
            empty += 1
        used = set()
        for pred in preds:
            for idx, gt in enumerate(gt_boxes):
                if idx in used:
                    continue
                if _center_hit(pred, gt):
                    used.add(idx)
                    hits += 1
                    per_hit[gt["label"]] += 1
                    break
    metrics = {
        "samples": len(labels),
        "gt_boxes": total_gt,
        "proxy_center_hit_score": round(hits / total_gt, 6) if total_gt else 0.0,
        "per_class_center_hit": {label: round(per_hit[label] / per_total[label], 6) if per_total[label] else 0.0 for label in TRAIN_DETECT_LABELS},
        "mean_confidence": round(sum(confs) / len(confs), 6) if confs else 0.0,
        "avg_latency_ms": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "empty_prediction_rate": round(empty / len(labels), 6) if labels else 0.0,
        "invalid_label_rate": round(invalid / max(1, invalid + len(confs)), 6),
        "error": error,
    }
    write_json(output, metrics)
    return metrics
