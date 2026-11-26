"""Focused L-02 test: reconstruct_closed_trade deletes OpenPosition once."""
import inspect
from unittest.mock import MagicMock, patch

from src.trade_manager import TradeManager


def test_reconstruct_closed_trade_has_single_delete():
    """The duplicate delete block must have been removed."""
    src = inspect.getsource(TradeManager.reconstruct_closed_trade)
    # One delete call from the authoritative block.
    assert src.count(".delete()") == 1, (
        "reconstruct_closed_trade must contain exactly one "
        "OpenPosition delete; found "
        f"{src.count('.delete()')}"
    )
    # One exception handler for that delete.
    assert src.count(
        'logger.exception(\n                "Could not delete OpenPosition'
    ) == 1


def test_reconstruct_closed_trade_idempotent_when_row_exists():
    """Preserved: existing ClosedTrade short-circuits; no delete occurs."""
    tm = TradeManager()

    with patch("src.trade_manager.db_manager") as db:
        db.query_one.side_effect = lambda model, **kw: (
            MagicMock() if getattr(model, "__name__", "") == "ClosedTrade"
            else None
        )
        assert tm.reconstruct_closed_trade(123) is True
        # No session was opened because we returned before any write.
        assert not db.get_session.called


def test_reconstruct_closed_trade_delete_happens_once_on_full_path():
    """Exercise the path once and confirm exactly one delete session."""
    tm = TradeManager()

    fake_deals = [
        {
            "entry": "IN", "time": __import__("datetime").datetime(
                2024, 1, 1, tzinfo=__import__("datetime").timezone.utc,
            ),
            "price": 2000.0, "volume": 0.1, "type": "BUY",
            "symbol": "GOLD", "magic": 1, "profit": 0.0,
            "swap": 0.0, "commission": 0.0,
        },
        {
            "entry": "OUT", "time": __import__("datetime").datetime(
                2024, 1, 1, 1, tzinfo=__import__("datetime").timezone.utc,
            ),
            "price": 2010.0, "volume": 0.1, "type": "BUY",
            "symbol": "GOLD", "magic": 1, "profit": 100.0,
            "swap": 0.0, "commission": 0.0,
        },
    ]

    delete_calls = []

    class _Q:
        def __init__(self, tracker):
            self._t = tracker
        def filter(self, *a, **k):
            return self
        def delete(self):
            self._t.append(1)

    class _Session:
        def __init__(self):
            self.committed = False
        def query(self, model):
            # Only count OpenPosition deletes.
            if getattr(model, "__name__", "") == "OpenPosition":
                return _Q(delete_calls)
            return MagicMock()
        def add(self, *a, **k):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            return False

    with patch("src.trade_manager.db_manager") as db, \
         patch.object(
             tm, "_deals_for_ticket", return_value=fake_deals
         ), \
         patch.object(tm, "remove_position") as rm:

        # First call: no ClosedTrade yet, no ExecutedOrder.
        db.query_one.return_value = None
        db.get_session.side_effect = lambda: _Session()

        ok = tm.reconstruct_closed_trade(1, exit_reason="TEST")

    assert ok is True
    assert len(delete_calls) == 1, (
        f"OpenPosition must be deleted exactly once; got {len(delete_calls)}"
    )
    rm.assert_called_once_with(1)