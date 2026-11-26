"""Focused M-15 test: dashboard risk-configuration endpoint."""
import asyncio
import importlib.util
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException


MAIN_PATH = (
    Path(__file__).resolve().parent.parent
    / "dashboard" / "backend" / "main.py"
)


@pytest.fixture(scope="module")
def main_mod():
    assert MAIN_PATH.is_file(), f"dashboard main.py not found at {MAIN_PATH}"
    spec = importlib.util.spec_from_file_location(
        "_dashboard_main_under_test", MAIN_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _config_stub(*, max_risk=2.0):
    cfg = MagicMock()
    cfg.trading.mode = "demo"
    cfg.risk = MagicMock(
        default_risk_percent=1.0,
        max_risk_percent=max_risk,
        max_open_positions=3,
        daily_loss_limit_percent=5.0,
        max_account_drawdown_percent=10.0,
        max_total_open_risk_percent=4.0,
        emergency_drawdown_percent=10.0,
        compound_enabled=True,
    )
    return cfg


def _run(main_mod, request, *, cfg=None):
    """Run endpoint with in-memory override store; return (result, store)."""
    cfg = cfg or _config_stub()
    store = {}

    def _read(env):
        return dict(store.get(env, {}))

    def _write(env, data):
        store[env] = dict(data)

    with patch.object(main_mod, "config", cfg), \
         patch.object(main_mod, "_read_risk_overrides", side_effect=_read), \
         patch.object(main_mod, "_write_risk_overrides", side_effect=_write):
        try:
            return asyncio.run(main_mod.update_risk_config(request)), store
        except HTTPException as exc:
            return exc, store


def test_tightening_risk_percent_accepted(main_mod):
    req = main_mod.RiskConfigRequest(risk_percent=0.5)
    result, store = _run(main_mod, req)
    assert not isinstance(result, HTTPException), (
        result.detail if isinstance(result, HTTPException) else ""
    )
    assert result["success"] is True
    assert result["applied_changes"]["default_risk_percent"] == 0.5
    assert store["demo"]["default_risk_percent"] == 0.5


def test_no_change_request_returns_current(main_mod):
    req = main_mod.RiskConfigRequest()
    result, store = _run(main_mod, req)
    assert result["success"] is True
    assert result["message"] == "No changes requested."
    assert store == {}


def test_weakening_rejected_without_confirmation(main_mod):
    # Current default_risk_percent is 1.0; 1.5 is weakening.
    req = main_mod.RiskConfigRequest(risk_percent=1.5)
    result, store = _run(main_mod, req)
    assert isinstance(result, HTTPException)
    assert result.status_code == 400
    assert "confirm" in str(result.detail).lower()
    assert store == {}  # no partial write


def test_weakening_accepted_with_confirmation(main_mod):
    req = main_mod.RiskConfigRequest(
        risk_percent=1.5, confirm="I_UNDERSTAND"
    )
    result, store = _run(main_mod, req)
    assert not isinstance(result, HTTPException), (
        result.detail if isinstance(result, HTTPException) else ""
    )
    assert result["success"] is True
    assert result["applied_changes"]["default_risk_percent"] == 1.5
    assert store["demo"]["default_risk_percent"] == 1.5


def test_risk_percent_above_cap_rejected_even_with_confirmation(main_mod):
    # Cap is 2.0; 3.0 exceeds cap.
    req = main_mod.RiskConfigRequest(
        risk_percent=3.0, confirm="I_UNDERSTAND"
    )
    result, store = _run(main_mod, req)
    assert isinstance(result, HTTPException)
    assert result.status_code == 400
    assert store == {}


def test_invalid_negative_risk_percent_rejected(main_mod):
    req = main_mod.RiskConfigRequest(risk_percent=-0.5)
    result, store = _run(main_mod, req)
    assert isinstance(result, HTTPException)
    assert result.status_code == 400
    assert store == {}


def test_nan_risk_percent_rejected(main_mod):
    req = main_mod.RiskConfigRequest(risk_percent=float("nan"))
    result, store = _run(main_mod, req)
    assert isinstance(result, HTTPException)
    assert result.status_code == 400
    assert store == {}


def test_max_positions_below_one_rejected(main_mod):
    req = main_mod.RiskConfigRequest(max_positions=0)
    result, store = _run(main_mod, req)
    assert isinstance(result, HTTPException)
    assert result.status_code == 400


def test_disabling_compounding_is_weakening(main_mod):
    # Turning compounding OFF requires confirmation.
    req = main_mod.RiskConfigRequest(compound_enabled=False)
    result, store = _run(main_mod, req)
    assert isinstance(result, HTTPException)
    assert result.status_code == 400

    req2 = main_mod.RiskConfigRequest(
        compound_enabled=False, confirm="I_UNDERSTAND"
    )
    result2, store2 = _run(main_mod, req2)
    assert not isinstance(result2, HTTPException), (
        result2.detail if isinstance(result2, HTTPException) else ""
    )
    assert result2["success"] is True
    assert store2["demo"]["compound_enabled"] is False


def test_write_error_fails_closed(main_mod):
    req = main_mod.RiskConfigRequest(risk_percent=0.5)
    cfg = _config_stub()

    with patch.object(main_mod, "config", cfg), \
         patch.object(main_mod, "_read_risk_overrides", return_value={}), \
         patch.object(
             main_mod, "_write_risk_overrides",
             side_effect=RuntimeError("db down"),
         ):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(main_mod.update_risk_config(req))
        assert excinfo.value.status_code == 500