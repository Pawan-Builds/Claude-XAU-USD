"""Focused N-H02: peak equity must never silently reset to current."""
import math
from unittest.mock import MagicMock, patch

from src.risk_manager import RiskManager


# ---------------------------------------------------------------------------
# Minimal test doubles
# ---------------------------------------------------------------------------

def _account(equity):
    a = MagicMock()
    a.equity = equity
    return a


def _peak_row(value):
    r = MagicMock()
    r.value = value
    return r


class _Row:
    """Single-column row surrogate; supports ``row[0]``."""

    def __init__(self, value):
        self._value = value

    def __getitem__(self, index):
        if index != 0:
            raise IndexError(index)
        return self._value


class _Q:
    """Tiny query-object double; routes by target class name."""

    def __init__(self, session, target):
        self.s = session
        self.target = target

    def filter(self, *a, **k):
        return self

    def filter_by(self, *a, **k):
        return self

    def order_by(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def first(self):
        if self.target == "AccountSnapshot":
            if self.s.snap_exc is not None:
                raise self.s.snap_exc
            if self.s.snap_max is None:
                return None
            return _Row(self.s.snap_max)
        if self.target == "AppSettings":
            # Peak-update write path.
            row = MagicMock()
            row.value = None
            return row
        return None

    def count(self):
        if self.s.count_exc is not None:
            raise self.s.count_exc
        if self.target == "ExecutedOrder":
            return self.s.exec_count
        if self.target == "ClosedTrade":
            return self.s.closed_count
        return 0


class _Session:
    """Minimal session double."""

    def __init__(
        self,
        *,
        snap_max,
        exec_count,
        closed_count,
        snap_exc=None,
        count_exc=None,
        write_raises=False,
    ):
        self.snap_max = snap_max
        self.exec_count = exec_count
        self.closed_count = closed_count
        self.snap_exc = snap_exc
        self.count_exc = count_exc
        self.write_raises = write_raises

    def query(self, *args):
        target = None
        for a in args:
            cls = getattr(a, "class_", None)
            if cls is not None:
                target = cls.__name__
            elif hasattr(a, "__name__"):
                target = a.__name__
        return _Q(self, target)

    def add(self, obj):
        if self.write_raises:
            raise RuntimeError("write failed")

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _run_peak(
    rm,
    *,
    peak_row=None,
    peak_read_exc=None,
    snap_max=None,
    exec_count=0,
    closed_count=0,
    snap_exc=None,
    count_exc=None,
    write_raises=False,
    account=None,
):
    """Drive RiskManager._get_peak_equity with a mocked DB/account."""

    class _DB:
        def query_one(self, model, **kw):
            if peak_read_exc is not None:
                raise peak_read_exc
            return peak_row

        def get_session(self):
            return _Session(
                snap_max=snap_max,
                exec_count=exec_count,
                closed_count=closed_count,
                snap_exc=snap_exc,
                count_exc=count_exc,
                write_raises=write_raises,
            )

    with patch("src.risk_manager.db_manager", _DB()), \
         patch("src.risk_manager.mt5_connector") as mock_conn:
        mock_conn.get_account_info.return_value = account
        return rm._get_peak_equity()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

# 1. Peak read failure -> fail closed (None), never reset to current
def test_peak_read_failure_returns_none():
    rm = RiskManager()
    out = _run_peak(
        rm,
        peak_read_exc=RuntimeError("db down"),
        account=_account(90_000.0),
    )
    assert out is None


# 2. Persisted peak valid, above current -> return stored peak
def test_valid_peak_above_current_returned_unchanged():
    rm = RiskManager()
    out = _run_peak(
        rm,
        peak_row=_peak_row("100000"),
        account=_account(90_000.0),
    )
    assert out == 100_000.0


# 3. Persisted peak valid, below current -> ratchets upward
def test_valid_peak_ratchets_up():
    rm = RiskManager()
    out = _run_peak(
        rm,
        peak_row=_peak_row("90000"),
        account=_account(100_000.0),
    )
    assert out == 100_000.0


# 4. Corrupt peak value -> fail closed, do NOT re-seed
def test_corrupt_peak_returns_none_not_reseed():
    rm = RiskManager()
    out = _run_peak(
        rm,
        peak_row=_peak_row("garbage"),
        snap_max=90_000.0,        # snapshot exists; must NOT be used
        account=_account(90_000.0),
    )
    assert out is None


# 5. Empty peak value -> fail closed
def test_empty_peak_value_returns_none():
    rm = RiskManager()
    out = _run_peak(
        rm,
        peak_row=_peak_row(""),
        account=_account(90_000.0),
    )
    assert out is None


# 6. Non-positive peak value -> fail closed
def test_non_positive_peak_returns_none():
    rm = RiskManager()
    out = _run_peak(
        rm,
        peak_row=_peak_row("0"),
        account=_account(90_000.0),
    )
    assert out is None


# 7. No peak row + snapshot query raises -> fail closed
def test_snapshot_query_failure_returns_none():
    rm = RiskManager()
    out = _run_peak(
        rm,
        peak_row=None,
        snap_exc=RuntimeError("snapshot db error"),
        account=_account(90_000.0),
    )
    assert out is None


# 8. No peak row + snapshot exists -> seed from snapshot max
def test_no_peak_seeds_from_snapshot():
    rm = RiskManager()
    out = _run_peak(
        rm,
        peak_row=None,
        snap_max=105_000.0,
        account=_account(90_000.0),
    )
    assert out == 105_000.0


# 9. No peak, no snapshots, trade history exists -> fail closed
def test_no_peak_no_snapshots_but_trades_returns_none():
    rm = RiskManager()
    out = _run_peak(
        rm,
        peak_row=None,
        snap_max=None,
        exec_count=5,
        account=_account(90_000.0),
    )
    assert out is None


# 10. No peak, no snapshots, trade-history count raises -> fail closed
def test_trade_count_failure_returns_none():
    rm = RiskManager()
    out = _run_peak(
        rm,
        peak_row=None,
        snap_max=None,
        count_exc=RuntimeError("db error"),
        account=_account(90_000.0),
    )
    assert out is None


# 11. Genuine first-ever -> seeds from current equity
def test_first_ever_seeds_from_current():
    rm = RiskManager()
    out = _run_peak(
        rm,
        peak_row=None,
        snap_max=None,
        exec_count=0,
        closed_count=0,
        account=_account(100_000.0),
    )
    assert out == 100_000.0


# 12. First-ever but peak write fails -> fail closed
def test_first_ever_write_failure_returns_none():
    rm = RiskManager()
    out = _run_peak(
        rm,
        peak_row=None,
        snap_max=None,
        exec_count=0,
        closed_count=0,
        write_raises=True,
        account=_account(100_000.0),
    )
    assert out is None


# 13. Drawdown uses trustworthy peak -> real drawdown detected
def test_max_drawdown_uses_stored_peak():
    rm = RiskManager()
    rm.risk_config = MagicMock(
        max_account_drawdown_percent=5.0,
        emergency_drawdown_percent=10.0,
    )
    with patch.object(
        rm, "_get_peak_equity", return_value=100_000.0
    ):
        assert rm._check_max_drawdown(90_000.0) is False   # 10% > 5%
        assert rm._check_max_drawdown(96_000.0) is True    # 4%  < 5%


# 14. Drawdown with unavailable peak -> fail closed (False)
def test_max_and_emergency_drawdown_fail_closed():
    rm = RiskManager()
    rm.risk_config = MagicMock(
        max_account_drawdown_percent=5.0,
        emergency_drawdown_percent=10.0,
    )
    with patch.object(rm, "_get_peak_equity", return_value=None):
        assert rm._check_max_drawdown(90_000.0) is False
        assert rm._check_emergency_drawdown(90_000.0) is False