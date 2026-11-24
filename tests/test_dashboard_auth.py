"""Tests for dashboard authentication helpers (F-03)."""

import pytest

from src.dashboard_auth import (
    LOOPBACK_HOSTS,
    check_request_allowed,
    check_websocket_allowed,
)


TOKEN = "test-token-abc-123"


# ---------------- HTTP auth ----------------


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1"])
def test_http_allows_loopback_with_correct_token(host):
    ok, status, _ = check_request_allowed(
        host, f"Bearer {TOKEN}", TOKEN
    )
    assert ok is True
    assert status == 200


def test_http_rejects_non_loopback():
    ok, status, reason = check_request_allowed(
        "192.168.1.10", f"Bearer {TOKEN}", TOKEN
    )
    assert ok is False
    assert status == 403
    assert "localhost" in reason.lower()


def test_http_rejects_when_token_not_configured():
    ok, status, reason = check_request_allowed(
        "127.0.0.1", f"Bearer {TOKEN}", None
    )
    assert ok is False
    assert status == 503
    assert "configure" in reason.lower()


def test_http_rejects_when_token_empty():
    ok, status, _ = check_request_allowed("127.0.0.1", "Bearer x", "")
    assert ok is False
    assert status == 503


def test_http_rejects_missing_header():
    ok, status, _ = check_request_allowed("127.0.0.1", None, TOKEN)
    assert ok is False
    assert status == 401


def test_http_rejects_wrong_scheme():
    ok, status, _ = check_request_allowed(
        "127.0.0.1", f"Basic {TOKEN}", TOKEN
    )
    assert ok is False
    assert status == 401


def test_http_rejects_wrong_token():
    ok, status, _ = check_request_allowed(
        "127.0.0.1", "Bearer wrong-token", TOKEN
    )
    assert ok is False
    assert status == 401


def test_http_rejects_empty_bearer():
    ok, status, _ = check_request_allowed("127.0.0.1", "Bearer ", TOKEN)
    assert ok is False
    assert status == 401


# ---------------- WebSocket auth ----------------


def test_ws_allows_loopback_with_correct_token():
    ok, code, _ = check_websocket_allowed(
        "127.0.0.1", TOKEN, TOKEN
    )
    assert ok is True
    assert code == 1000


def test_ws_rejects_non_loopback():
    ok, code, _ = check_websocket_allowed(
        "10.0.0.5", TOKEN, TOKEN
    )
    assert ok is False
    assert code == 1008


def test_ws_rejects_missing_token():
    ok, code, _ = check_websocket_allowed("127.0.0.1", None, TOKEN)
    assert ok is False
    assert code == 1008


def test_ws_rejects_wrong_token():
    ok, code, _ = check_websocket_allowed(
        "127.0.0.1", "wrong", TOKEN
    )
    assert ok is False
    assert code == 1008


def test_ws_rejects_when_token_not_configured():
    ok, code, _ = check_websocket_allowed("127.0.0.1", TOKEN, "")
    assert ok is False
    assert code == 1011