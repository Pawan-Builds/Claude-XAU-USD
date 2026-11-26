"""Focused M-13 test: LIVE authorization at the connector boundary.

LIVE is NEVER enabled at runtime by this test. All checks are
unit-level: _assert_live_opening_authorized() only decides
authorization; it does not place a trade.
"""
import os
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------
# property freshness
# ---------------------------------------------------------------------

def test_live_confirmation_property_reads_fresh_env():
    """M-13: the confirmation must be read fresh on every access,
    not captured at class-definition time (which is before
    load_dotenv ran)."""
    from src.config_loader import config

    with patch.dict(
        os.environ, {"LIVE_TRADING_CONFIRMATION": "value_a"}, clear=False
    ):
        assert config.live_trading_confirmation == "value_a"

    with patch.dict(
        os.environ, {"LIVE_TRADING_CONFIRMATION": "value_b"}, clear=False
    ):
        assert config.live_trading_confirmation == "value_b"

    # Unset -> empty string; never None (property always returns str).
    env = dict(os.environ)
    env.pop("LIVE_TRADING_CONFIRMATION", None)
    with patch.dict(os.environ, env, clear=True):
        assert config.live_trading_confirmation == ""


# ---------------------------------------------------------------------
# boundary enforces exact match
# ---------------------------------------------------------------------

def _make_live_connector():
    """Fresh connector instance with LIVE-ready mock state.

    Skips __init__ so we don't trigger config.mt5 validation.
    """
    from src.mt5_connector import MT5Connector
    connector = MT5Connector.__new__(MT5Connector)
    connector.mt5_config = MagicMock()
    connector.mt5_config.server = "TEST-SERVER"
    connector._account_info = MagicMock(
        login=12345, server="TEST-SERVER"
    )
    return connector


def _make_live_config(confirmation):
    """Mock config satisfying every boundary check except #2."""
    cfg = MagicMock()
    cfg.trading.mode = "LIVE"
    cfg.live_trading_enabled = True
    cfg.live_trading_confirmation = confirmation
    cfg.live_allowed_account = 12345
    cfg.safety.demo_only = False
    return cfg


@pytest.mark.parametrize(
    "confirmation,should_authorize",
    [
        (None, False),                             # missing
        ("", False),                               # empty
        ("x", False),                              # non-empty wrong
        ("I_ACCEPT", False),                       # prefix only
        ("I_ACCEPT_LIVE_TRADING_RISK_EXTRA", False),  # suffix appended
        ("i_accept_live_trading_risk", False),     # wrong case
        ("I_ACCEPT_LIVE_TRADING_RISK", True),      # exact
        ("  I_ACCEPT_LIVE_TRADING_RISK  ", True),  # padded exact
    ],
)
def test_live_boundary_requires_exact_confirmation(
    confirmation, should_authorize
):
    from src.config_loader import LIVE_CONFIRMATION_PHRASE

    connector = _make_live_connector()
    cfg = _make_live_config(confirmation)

    # Stub the checks that come AFTER the confirmation check so the
    # "authorized" cases can reach the return-None path without
    # touching real kill-switch / reconciliation state.
    with patch("src.mt5_connector.config", cfg), \
         patch(
             "src.risk_manager.risk_manager.get_kill_switch_state_strict",
             return_value=False,
         ), \
         patch(
             "src.order_executor.order_executor.has_unresolved_reconciliation",
             return_value=False,
         ):
        result = connector._assert_live_opening_authorized()

    if should_authorize:
        assert result is None, f"expected authorization, got: {result}"
    else:
        assert result is not None, "expected rejection, got None"
        assert "confirmation" in result.lower()
        # The literal must not leak into the rejection text.
        assert LIVE_CONFIRMATION_PHRASE not in result


# ---------------------------------------------------------------------
# DEMO short-circuits without invoking any LIVE check
# ---------------------------------------------------------------------

def test_demo_mode_boundary_bypasses_confirmation():
    from src.mt5_connector import MT5Connector

    connector = MT5Connector.__new__(MT5Connector)
    connector.mt5_config = MagicMock()
    connector._account_info = MagicMock()

    cfg = MagicMock()
    cfg.trading.mode = "DEMO"

    # Even with an empty confirmation, DEMO must return None
    # (short-circuit) — the LIVE check is not invoked for DEMO.
    cfg.live_trading_confirmation = ""

    with patch("src.mt5_connector.config", cfg):
        result = connector._assert_live_opening_authorized()

    assert result is None


# ---------------------------------------------------------------------
# Startup and boundary constants cannot drift
# ---------------------------------------------------------------------

def test_startup_and_boundary_use_same_phrase():
    import importlib.util
    from pathlib import Path

    from src.config_loader import (
        LIVE_CONFIRMATION_PHRASE,
        PROJECT_ROOT,
    )

    # main.py can live at the repo root OR under src/. Locate it
    # deterministically under PROJECT_ROOT rather than assuming a
    # specific location.
    candidates = [
        PROJECT_ROOT / "main.py",
        PROJECT_ROOT / "src" / "main.py",
    ]
    main_path = next((p for p in candidates if p.is_file()), None)
    if main_path is None:
        # Fallback: recursive search limited to files literally
        # named main.py so we never silently pick up an unrelated
        # module of the same import name.
        hits = [
            p for p in PROJECT_ROOT.rglob("main.py")
            if p.is_file()
        ]
        assert hits, (
            f"main.py not found under {PROJECT_ROOT}; "
            f"checked {[str(p) for p in candidates]}"
        )
        # Prefer the shallowest match.
        main_path = sorted(hits, key=lambda p: len(p.parts))[0]

    spec = importlib.util.spec_from_file_location(
        "_bot_main_under_test", main_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert hasattr(module, "TradingAgent"), (
        f"Loaded {main_path} but it has no TradingAgent attribute."
    )
    assert (
        module.TradingAgent.LIVE_CONFIRMATION
        == LIVE_CONFIRMATION_PHRASE
    ), (
        f"Startup phrase {module.TradingAgent.LIVE_CONFIRMATION!r} "
        f"!= shared constant {LIVE_CONFIRMATION_PHRASE!r} "
        f"(main.py at {main_path})"
    )