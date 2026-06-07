from __future__ import annotations

from typing import Any

from contest_agent.postprocess.labels import choose_fallback_label, normalize_detect_label
from contest_agent.utils import clamp


def _is_normalized_box(values: list[float]) -> bool:
    return values and min(values) >= 0.0 and max(values) <= 1.0


def _box_to_center(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float]:
    if _is_normalized_box([x1, y1, x2, y2]) and image_width > 1 and image_height > 1:
        x1 *= image_width
        x2 *= image_width
        y1 *= image_height
        y2 *= image_height
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return cx, cy


def _extract_center_from_box_like(item: dict[str, Any], image_width: int, image_height: int) -> tuple[float, float] | None:
    xyxy = item.get("xyxy")
    if isinstance(xyxy, (list, tuple)) and len(xyxy) == 4:
        values = [float(v) for v in xyxy]
        return _box_to_center(values[0], values[1], values[2], values[3], image_width, image_height)

    box = item.get("box")
    if isinstance(box, dict):
        if all(key in box for key in ("x1", "y1", "x2", "y2")):
            return _box_to_center(
                float(box["x1"]),
                float(box["y1"]),
                float(box["x2"]),
                float(box["y2"]),
                image_width,
                image_height,
            )
        if all(key in box for key in ("x", "y", "w", "h")):
            x = float(box["x"])
            y = float(box["y"])
            w = float(box["w"])
            h = float(box["h"])
            if _is_normalized_box([x, y, w, h]) and image_width > 1 and image_height > 1:
                x *= image_width
                y *= image_height
                w *= image_width
                h *= image_height
            return x + w / 2.0, y + h / 2.0

    bbox = item.get("bbox")
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        values = [float(v) for v in bbox]
        x, y, w, h = values
        if _is_normalized_box(values) and image_width > 1 and image_height > 1:
            x *= image_width
            y *= image_height
            w *= image_width
            h *= image_height
        return x + w / 2.0, y + h / 2.0

    return None


def _coerce_target(item: dict[str, Any], image_width: int, image_height: int) -> dict[str, float | str] | None:
    label = item.get("label") or item.get("class_name") or item.get("name")
    if not label:
        return None

    if "cx" in item and "cy" in item:
        cx = float(item["cx"])
        cy = float(item["cy"])
        if 0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0 and image_width > 1 and image_height > 1:
            cx *= image_width
            cy *= image_height
    else:
        center = _extract_center_from_box_like(item, image_width, image_height)
        if center is None:
            return None
        cx, cy = center

    return {
        "label": normalize_detect_label(str(label)),
        "cx": clamp(cx, 0.0, max(0.0, float(image_width - 1))),
        "cy": clamp(cy, 0.0, max(0.0, float(image_height - 1))),
        "score": float(item.get("score", item.get("confidence", 1.0))),
    }


def normalize_detect_result(
    result: dict[str, Any] | None,
    legal_labels: list[str],
    image_width: int,
    image_height: int,
) -> dict[str, Any]:
    raw_result = result or {}
    items: list[dict[str, Any]] = []
    for key in ("targets", "boxes", "detections"):
        value = raw_result.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, dict))

    normalized: list[dict[str, Any]] = []
    for item in items:
        target = _coerce_target(item, image_width, image_height)
        if not target:
            continue
        label = str(target["label"])
        if legal_labels:
            if label not in legal_labels:
                continue
        else:
            target["label"] = choose_fallback_label(legal_labels, "detect")
        normalized.append(
            {
                "label": str(target["label"]),
                "cx": round(float(target["cx"]), 3),
                "cy": round(float(target["cy"]), 3),
                "score": round(float(target["score"]), 4),
            }
        )

    return {"targets": normalized}
