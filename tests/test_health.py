from __future__ import annotations


def test_health_returns_expected_contract(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["supported_tasks"] == ["classify", "ocr", "detect"]


def test_options_preflight_is_available(client) -> None:
    response = client.options(
        "/infer",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code in {200, 204}
    assert response.headers["access-control-allow-origin"] == "*"
