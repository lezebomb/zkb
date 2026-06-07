from __future__ import annotations


def test_health_returns_expected_contract(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["supported_tasks"] == ["classify", "ocr", "detect"]
    assert payload["bridge_mode"] == "mock-local"


def test_debug_status_returns_backend_summary(client) -> None:
    response = client.get("/debug/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["service"] == "contestant-algo-test-service"
    assert payload["offline_mode"] is True
    assert "backends" in payload
    assert payload["backends"]["detect"]["configured"] in {"fallback", "local", "ultralytics"}


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
