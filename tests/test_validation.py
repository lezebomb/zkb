from __future__ import annotations

from contest_agent.postprocess.ocr_text import normalize_text_for_output
from contest_agent.validators import (
    validate_classify_result,
    validate_detect_result,
    validate_ocr_result,
)


def test_validate_classify_repairs_invalid_label() -> None:
    result = validate_classify_result({"label": "office"}, {"class_names": ["办公室", "公园"]})
    assert result["label"] == "办公室"


def test_validate_detect_converts_normalized_box_to_pixel_center() -> None:
    result = validate_detect_result(
        {"boxes": [{"label": "cell phone", "xyxy": [0.25, 0.5, 0.5, 0.75], "score": 0.9}]},
        {"class_names": ["手机"]},
        640,
        480,
    )
    target = result["targets"][0]
    assert target["label"] == "手机"
    assert target["cx"] == 240.0
    assert target["cy"] == 300.0


def test_validate_ocr_applies_normalize_rules() -> None:
    result = validate_ocr_result(
        {"text": " 阀 位 １２.８％ "},
        {"normalize_rules": {"trim_space": True, "case_insensitive": False}},
    )
    assert result["text"] == "阀位12.8%"


def test_normalize_text_preserves_case_when_requested() -> None:
    assert normalize_text_for_output(" Valve A ", {"trim_space": False, "case_insensitive": False}) == "Valve A"


def test_missing_fields_return_json_not_html(client) -> None:
    response = client.post("/infer", json={"task_type": "ocr"})
    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is False
    assert "invalid request fields" in payload["message"]


def test_non_json_content_type_returns_json_failure(client) -> None:
    response = client.post(
        "/infer",
        content="not-json",
        headers={"Content-Type": "text/plain"},
    )
    payload = response.json()
    assert response.status_code == 200
    assert payload["ok"] is False
    assert "Content-Type" in payload["message"]
