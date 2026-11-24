"""Shared pytest fixtures for the trading-bot test suite."""

import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Optional
from unittest.mock import MagicMock

import pytest

# Ensure project root is on sys.path when tests are run from tests/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class StubGateResult:
    ok: bool
    reason: str = "OK"
    checks: Optional[Dict[str, bool]] = None

    def __post_init__(self):
        if self.checks is None:
            self.checks = {}

    def __bool__(self) -> bool:
        return self.ok


class StubGate:
    """Drop-in replacement for EntryGate in tests."""

    def __init__(self, ok: bool = True, reason: str = "OK"):
        self.ok = ok
        self.reason = reason
        self.calls = {"check": 0, "probe": 0}

    def check(self):
        self.calls["check"] += 1
        return StubGateResult(ok=self.ok, reason=self.reason)

    def probe(self):
        self.calls["probe"] += 1
        return StubGateResult(ok=self.ok, reason=self.reason)


@pytest.fixture
def stub_gate_factory():
    return StubGate


@pytest.fixture
def sample_signal():
    """Minimal duck-typed signal matching what order_executor expects."""
    return SimpleNamespace(
        timestamp=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
        symbol="GOLD",
        direction=SimpleNamespace(value="BUY"),
        strategy=SimpleNamespace(value="liquidity_sweep_fvg"),
        entry_price=2000.00,
        sl_price=1995.00,
        tp_price=2010.00,
        risk_percent=1.0,
        risk_reward=2.0,
        lot_size=0.0,
        confirmations=[],
        entry_tf="m5",
        setup_tf="m15",
        structure_tf="h1",
        bias_tf="h4",
        fingerprint=None,
    )


@pytest.fixture
def sample_order_result():
    from src.mt5_connector import OrderResult

    return OrderResult(
        success=True,
        ticket=9001,
        order_id=9001,
        volume=0.10,
        price=2000.00,
        retcode=10009,
        comment="DONE",
        deal_id=8001,
    )


@pytest.fixture
def sample_position():
    return {
        "ticket": 9001,
        "magic": 123456789,
        "magic_number": 123456789,
        "symbol": "GOLD",
        "type": "BUY",
        "volume": 0.10,
        "price_open": 2000.00,
        "price_current": 2000.00,
        "sl": 1995.00,
        "tp": 2010.00,
        "swap": 0.0,
        "profit": 0.0,
        "commission": -0.70,
        "comment": "SMC_liquidity_sweep_fvg",
        "time_open": datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
        "time_update": datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
    }