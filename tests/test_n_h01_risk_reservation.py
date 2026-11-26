"""Focused N-H01: persistent risk reservation across concurrent entries."""
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# RiskManager.try_reserve (semantics only; DB lock is tested via mock)
# ---------------------------------------------------------------------------

def _rm():
    from src.risk_manager import RiskManager
    return RiskManager()


def _patch_db_atomic(side_effect_or_return):
    """Return (patcher, mock_method) for db_manager.reserve_risk_atomic."""
    if callable(side_effect_or_return):
        m = MagicMock(side_effect=side_effect_or_return)
    else:
        m = MagicMock(return_value=side_effect_or_return)
    return patch(
        "src.risk_manager.db_manager.reserve_risk_atomic", m
    ), m


def test_try_reserve_rejects_negative_risk():
    rm = _rm()
    ok, reason = rm.try_reserve("fp", rm.ENVIRONMENT, -1.0, 10000.0)
    assert ok is False
    assert "Non-positive" in reason or "Invalid" in reason or "negative" in reason


def test_try_reserve_rejects_zero_equity():
    rm = _rm()
    ok, reason = rm.try_reserve("fp", rm.ENVIRONMENT, 10.0, 0.0)
    assert ok is False


def test_try_reserve_rejects_env_mismatch():
    rm = _rm()
    ok, reason = rm.try_reserve("fp", "not-the-env", 10.0, 10000.0)
    assert ok is False
    assert "Environment" in reason


def test_try_reserve_rejects_broker_query_failure():
    rm = _rm()
    with patch(
        "src.risk_manager.mt5_connector.get_all_positions",
        side_effect=RuntimeError("boom"),
    ):
        ok, reason = rm.try_reserve("fp", rm.ENVIRONMENT, 10.0, 10000.0)
    assert ok is False


def test_try_reserve_passes_through_to_atomic():
    rm = _rm()
    rm.risk_config = MagicMock(max_total_open_risk_percent=4.0)

    with patch(
        "src.risk_manager.mt5_connector.get_all_positions",
        return_value=[],
    ), patch(
        "src.risk_manager.mt5_connector.get_all_orders",
        return_value=[],
    ), patch.object(
        rm,
        "_calculate_account_open_positions_risk",
        return_value=100.0,
    ), patch.object(
        rm,
        "_calculate_account_pending_orders_risk",
        return_value=0.0,
    ):
        patcher, mock_atomic = _patch_db_atomic((True, None))
        with patcher:
            ok, reason = rm.try_reserve(
                "fp", rm.ENVIRONMENT, 50.0, 10_000.0
            )

    assert ok is True
    assert reason is None
    kwargs = mock_atomic.call_args.kwargs
    assert kwargs["fingerprint"] == "fp"
    assert kwargs["environment"] == rm.ENVIRONMENT
    assert kwargs["risk_amount"] == 50.0
    assert kwargs["equity"] == 10_000.0
    assert kwargs["open_risk"] == 100.0
    assert kwargs["pending_risk"] == 0.0
    assert kwargs["cap"] == pytest.approx(10_000.0 * 4.0 / 100.0)


def test_try_reserve_propagates_rejection_reason():
    rm = _rm()
    rm.risk_config = MagicMock(max_total_open_risk_percent=4.0)
    with patch(
        "src.risk_manager.mt5_connector.get_all_positions",
        return_value=[],
    ), patch(
        "src.risk_manager.mt5_connector.get_all_orders",
        return_value=[],
    ), patch.object(
        rm, "_calculate_account_open_positions_risk", return_value=350.0,
    ), patch.object(
        rm, "_calculate_account_pending_orders_risk", return_value=0.0,
    ):
        patcher, _ = _patch_db_atomic(
            (False, "Reservation rejected: total would be 400.5 > 400")
        )
        with patcher:
            ok, reason = rm.try_reserve(
                "fp", rm.ENVIRONMENT, 50.5, 10_000.0
            )
    assert ok is False
    assert "Reservation rejected" in reason


# ---------------------------------------------------------------------------
# _check_total_risk includes active reservations
# ---------------------------------------------------------------------------

def test_check_total_risk_includes_reservations():
    rm = _rm()
    rm.risk_config = MagicMock(max_total_open_risk_percent=4.0)

    positions = [{"ticket": 1}]
    # open 350, pending 0, reserved 40 → 390 <= 400 → pass
    with patch.object(
        rm, "_calculate_account_open_positions_risk", return_value=350.0,
    ), patch.object(
        rm, "_calculate_account_pending_orders_risk", return_value=0.0,
    ), patch.object(
        rm, "get_active_reservation_risk", return_value=40.0,
    ), patch(
        "src.risk_manager.mt5_connector.get_all_orders",
        return_value=[],
    ):
        assert rm._check_total_risk(10_000.0, positions) is True

    # reserved pushes over cap
    with patch.object(
        rm, "_calculate_account_open_positions_risk", return_value=350.0,
    ), patch.object(
        rm, "_calculate_account_pending_orders_risk", return_value=0.0,
    ), patch.object(
        rm, "get_active_reservation_risk", return_value=60.0,
    ), patch(
        "src.risk_manager.mt5_connector.get_all_orders",
        return_value=[],
    ):
        assert rm._check_total_risk(10_000.0, positions) is False


def test_check_total_risk_fails_closed_when_reservations_unreadable():
    rm = _rm()
    rm.risk_config = MagicMock(max_total_open_risk_percent=4.0)
    with patch.object(
        rm, "_calculate_account_open_positions_risk", return_value=100.0,
    ), patch.object(
        rm, "_calculate_account_pending_orders_risk", return_value=0.0,
    ), patch.object(
        rm, "get_active_reservation_risk", return_value=None,
    ), patch(
        "src.risk_manager.mt5_connector.get_all_orders",
        return_value=[],
    ):
        assert rm._check_total_risk(10_000.0, []) is False


# ---------------------------------------------------------------------------
# OrderExecutor release constants
# ---------------------------------------------------------------------------

def test_releasable_statuses_cover_terminal_states():
    from src.order_executor import _RESERVATION_RELEASABLE_STATUSES
    for s in (
        "filled", "filled_closed", "unresolved_closed",
        "rejected", "rejected_notfound",
        "cancelled", "expired", "superseded",
    ):
        assert s in _RESERVATION_RELEASABLE_STATUSES

    # Holding statuses must NOT release.
    for s in (
        "pending", "sent", "placed",
        "filled_unconfirmed", "uncertain",
    ):
        assert s not in _RESERVATION_RELEASABLE_STATUSES


# ---------------------------------------------------------------------------
# OrderExecutor._mark_request_status releases on terminal
# ---------------------------------------------------------------------------

def test_mark_request_status_releases_on_filled():
    from src.order_executor import OrderExecutor
    oe = OrderExecutor()

    fake_request = MagicMock()
    fake_request.fingerprint = "fp-xyz"

    class _Sess:
        def __init__(self):
            self.request = fake_request
        def query(self, *a):
            s = self
            class _Q:
                def filter(self, *a, **k):
                    return self
                def first(self):
                    return s.request
            return _Q()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    with patch(
        "src.order_executor.db_manager.get_session",
        return_value=_Sess(),
    ), patch.object(
        oe, "_release_reservation",
    ) as mock_rel:
        oe._mark_request_status(42, "filled")

    mock_rel.assert_called_once()
    args = mock_rel.call_args.args
    assert args[0] == "fp-xyz"
    assert "filled" in args[1]


def test_mark_request_status_does_not_release_on_placed():
    from src.order_executor import OrderExecutor
    oe = OrderExecutor()

    fake_request = MagicMock()
    fake_request.fingerprint = "fp-xyz"

    class _Sess:
        def query(self, *a):
            class _Q:
                def filter(self, *a, **k):
                    return self
                def first(self):
                    return fake_request
            return _Q()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    with patch(
        "src.order_executor.db_manager.get_session",
        return_value=_Sess(),
    ), patch.object(
        oe, "_release_reservation",
    ) as mock_rel:
        oe._mark_request_status(42, "placed")

    mock_rel.assert_not_called()


# ---------------------------------------------------------------------------
# Orphan reconciliation
# ---------------------------------------------------------------------------

def test_reconcile_orphaned_reservations_releases_when_no_holding_request():
    from src.order_executor import OrderExecutor
    oe = OrderExecutor()

    r1 = MagicMock(fingerprint="fp-1", environment=oe.ENVIRONMENT)
    r2 = MagicMock(fingerprint="fp-2", environment=oe.ENVIRONMENT)

    class _Sess:
        def query(self, *args):
            # Route by target model, since both queries run inside
            # the same `with get_session()` block in the source.
            target = None
            for a in args:
                cls = getattr(a, "class_", None)
                if cls is not None:
                    target = cls.__name__
                elif hasattr(a, "__name__"):
                    target = a.__name__

            class _Q:
                def filter(self, *a, **k):
                    return self

                def all(self):
                    if target == "RiskReservation":
                        return [r1, r2]
                    if target == "OrderRequest":
                        # Only fp-2 has a holding OrderRequest.
                        return [("fp-2",)]
                    return []

            return _Q()

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    with patch(
        "src.order_executor.db_manager.get_session",
        return_value=_Sess(),
    ), patch.object(oe, "_release_reservation") as mock_rel:
        oe._reconcile_orphaned_reservations()

    released_fps = {c.args[0] for c in mock_rel.call_args_list}
    assert released_fps == {"fp-1"}, (
        f"expected only fp-1 (orphan) to be released, got {released_fps}"
    )
    # Confirm fp-2 was NOT released.
    for call in mock_rel.call_args_list:
        assert call.args[0] != "fp-2"

# ---------------------------------------------------------------------------
# execute_signal: reservation call ordered before _save_signal
# ---------------------------------------------------------------------------

def test_execute_signal_reserves_before_save_signal():
    from src.order_executor import OrderExecutor
    oe = OrderExecutor()

    signal = MagicMock()
    signal.symbol = "GOLD"
    signal.direction.value = "BUY"
    signal.entry_price = 2000.0
    signal.sl_price = 1990.0
    signal.tp_price = 2020.0
    signal.risk_percent = 1.0
    signal.strategy.value = "s"
    signal.confirmations = []
    signal.timestamp = datetime.utcnow()
    signal.entry_tf = "m5"
    signal.risk_reward = 2.0
    signal.fingerprint = "fp-test"

    gate = MagicMock()
    gate.check.return_value = MagicMock(ok=True, token="tok")

    size = MagicMock(lot_size=0.1, risk_amount=50.0, equity=10_000.0)
    validation = MagicMock(valid=True, reason="")

    order_of_operations = []

    fake_rm = MagicMock()

    def _fake_reserve(**kwargs):
        order_of_operations.append("reserve")
        return True, None

    fake_rm.try_reserve.side_effect = _fake_reserve

    def _fake_save_signal(*a, **k):
        order_of_operations.append("save_signal")
        return 1

    with patch(
        "src.order_executor.mt5_connector"
    ) as mock_conn, patch(
        "src.order_executor.signal_validator"
    ) as mock_sv, patch(
        "src.order_executor.risk_manager", fake_rm
    ), patch.object(
        oe, "_save_signal", side_effect=_fake_save_signal
    ), patch.object(
        oe, "_save_order_request", return_value=99
    ), patch.object(
        oe, "_prepare_order_request", return_value={}
    ), patch.object(
        oe, "_is_duplicate", return_value=False
    ), patch.object(
        oe, "_update_order_request"
    ), patch.object(
        oe, "_finalise_market_entry"
    ), patch.object(
        oe, "_generate_fingerprint", return_value="fp-test"
    ):
        mock_conn.is_connected.return_value = True
        mock_conn.get_fresh_tick.return_value = {
            "bid": 1999.5, "ask": 2000.0,
        }
        mock_conn.get_symbol_info.return_value = MagicMock(
            point=0.01, digits=2, name="GOLD",
        )
        mock_conn.send_order.return_value = MagicMock(
            success=True, uncertain=False, order_id=1234, ticket=1234,
            volume=0.1, price=2000.0, retcode=0, comment="",
            error_code=None, error_message=None,
        )
        mock_sv.validate_signal.return_value = validation
        fake_rm.calculate_position_size.return_value = size

        oe.execute_signal(signal, market_entry=True, gate=gate)

    assert order_of_operations == ["reserve", "save_signal"], (
        f"expected reserve before save_signal, got {order_of_operations}"
    )
    fake_rm.try_reserve.assert_called_once()


def test_execute_signal_releases_on_signal_save_failure():
    from src.order_executor import OrderExecutor
    oe = OrderExecutor()

    signal = MagicMock()
    signal.symbol = "GOLD"
    signal.direction.value = "BUY"
    signal.entry_price = 2000.0
    signal.sl_price = 1990.0
    signal.tp_price = 2020.0
    signal.risk_percent = 1.0
    signal.strategy.value = "s"
    signal.confirmations = []
    signal.timestamp = datetime.utcnow()
    signal.entry_tf = "m5"
    signal.risk_reward = 2.0
    signal.fingerprint = "fp-x"

    gate = MagicMock()
    gate.check.return_value = MagicMock(ok=True, token="tok")

    size = MagicMock(lot_size=0.1, risk_amount=50.0, equity=10_000.0)
    validation = MagicMock(valid=True, reason="")

    fake_rm = MagicMock()
    fake_rm.try_reserve.return_value = (True, None)

    with patch(
        "src.order_executor.mt5_connector"
    ) as mock_conn, patch(
        "src.order_executor.signal_validator"
    ) as mock_sv, patch(
        "src.order_executor.risk_manager", fake_rm
    ), patch.object(
        oe, "_save_signal", side_effect=RuntimeError("boom")
    ), patch.object(
        oe, "_is_duplicate", return_value=False
    ), patch.object(
        oe, "_generate_fingerprint", return_value="fp-x"
    ):
        mock_conn.is_connected.return_value = True
        mock_conn.get_fresh_tick.return_value = {
            "bid": 1999.5, "ask": 2000.0,
        }
        mock_conn.get_symbol_info.return_value = MagicMock(
            point=0.01, digits=2, name="GOLD",
        )
        mock_sv.validate_signal.return_value = validation
        fake_rm.calculate_position_size.return_value = size

        result = oe.execute_signal(signal, market_entry=True, gate=gate)

    assert result.success is False
    fake_rm.release_reservation.assert_called()
    args = fake_rm.release_reservation.call_args
    assert args.args[0] == "fp-x"