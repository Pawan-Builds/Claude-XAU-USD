"""Tests for F-06: ClosedTrade reconstruction from deal history."""

from datetime import datetime, timezone, timedelta

import pytest

from tests.test_position_recovery import (  # noqa: F401 — fixture is shared
    harness,  # noqa: F401
    _entry_deal, _exit_deal,
)


def test_realised_pl_uses_deal_history_not_stale_row(harness):
    """ClosedTrade.profit/swap/commission must reflect deal history."""
    from src.database_manager import ExecutedOrder, OpenPosition, ClosedTrade

    harness.db.add(ExecutedOrder(
        order_request_id=None, ticket=6001, magic_number=123456789,
        symbol="GOLD", order_type="BUY", volume=0.1,
        price_open=2000.0, price_current=2000.0,
        sl=1995.0, tp=2010.0,
        swap=0.0, profit=0.0, commission=0.0,  # <— stale values
        comment="SMC_liquidity_sweep_fvg",
        time_open=datetime.now(timezone.utc) - timedelta(hours=2),
        environment="demo",
    ))
    harness.db.add(OpenPosition(
        ticket=6001, magic_number=123456789, symbol="GOLD",
        direction="BUY", volume=0.1, price_open=2000.0,
        price_current=2000.0, sl=1995.0, tp=2010.0, swap=0.0,
        profit=0.0, commission=0.0, strategy="liquidity_sweep_fvg",
        time_open=datetime.now(timezone.utc) - timedelta(hours=2),
        environment="demo", initial_sl=1995.0, initial_tp=2010.0,
        initial_risk_distance=5.0, remaining_volume=0.1,
        highest_r=0.0, lowest_r=0.0,
    ))

    harness.mt5.get_history_deals.return_value = [
        {**_entry_deal(6001, price=2000.0), "commission": -0.70},
        {**_exit_deal(6001, price=2012.0), "commission": -0.70,
         "profit": 120.0, "swap": -0.50},
    ]

    harness.manager.reconstruct_closed_trade(6001, exit_reason="TP")

    ct = harness.db.query(ClosedTrade)[0]
    assert ct.profit == 120.0
    assert ct.swap == -0.50
    assert ct.commission == -1.40
    assert abs(ct.net_profit - 118.10) < 0.01


def test_partial_exits_sum_all_deals(harness):
    """Two exit deals must be summed (partial close then final close)."""
    from src.database_manager import ExecutedOrder, ClosedTrade

    harness.db.add(ExecutedOrder(
        order_request_id=None, ticket=6002, magic_number=123456789,
        symbol="GOLD", order_type="BUY", volume=0.2,
        price_open=2000.0, price_current=2000.0,
        sl=1995.0, tp=2010.0,
        swap=0.0, profit=0.0, commission=0.0,
        comment="SMC_test",
        time_open=datetime.now(timezone.utc) - timedelta(hours=2),
        environment="demo",
    ))

    entry = {**_entry_deal(6002, price=2000.0, volume=0.2),
             "commission": -1.40}
    exit_1 = {**_exit_deal(6002, price=2005.0, volume=0.1),
              "profit": 50.0, "commission": -0.70,
              "time": datetime.now(timezone.utc) - timedelta(hours=1)}
    exit_2 = {**_exit_deal(6002, price=2010.0, volume=0.1),
              "profit": 100.0, "commission": -0.70,
              "time": datetime.now(timezone.utc) - timedelta(minutes=5)}

    harness.mt5.get_history_deals.return_value = [entry, exit_1, exit_2]

    harness.manager.reconstruct_closed_trade(6002, exit_reason="TP")

    ct = harness.db.query(ClosedTrade)[0]
    assert abs(ct.profit - 150.0) < 0.01
    # Weighted-average close: (2005 * 0.1 + 2010 * 0.1) / 0.2 = 2007.5
    assert abs(ct.price_close - 2007.5) < 0.01
    # Total commission: -1.40 - 0.70 - 0.70 = -2.80
    assert abs(ct.commission - (-2.80)) < 0.01


def test_sell_realised_r_uses_correct_sign(harness):
    from src.database_manager import ExecutedOrder, ClosedTrade

    harness.db.add(ExecutedOrder(
        order_request_id=None, ticket=6003, magic_number=123456789,
        symbol="GOLD", order_type="SELL", volume=0.1,
        price_open=2000.0, price_current=2000.0,
        sl=2005.0, tp=1990.0,
        swap=0.0, profit=0.0, commission=0.0,
        comment="SMC_test",
        time_open=datetime.now(timezone.utc) - timedelta(hours=2),
        environment="demo",
    ))

    entry = {**_entry_deal(6003, price=2000.0),
             "type": "SELL", "commission": -0.70}
    exit_ = {**_exit_deal(6003, price=1990.0),
             "type": "SELL", "profit": 100.0, "commission": -0.70}

    harness.mt5.get_history_deals.return_value = [entry, exit_]

    harness.manager.reconstruct_closed_trade(6003, exit_reason="TP")

    ct = harness.db.query(ClosedTrade)[0]
    # R = (entry - exit) / initial_risk = (2000 - 1990) / 5 = 2.0
    assert abs(ct.realized_r - 2.0) < 0.01
    assert ct.direction == "SELL"