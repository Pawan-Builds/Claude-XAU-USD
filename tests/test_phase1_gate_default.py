"""Phase 1 / C-03 regression: an EntryGate that was not given live
pause/shutdown getters must refuse to authorize anything."""

from src.entry_gate import EntryGate, entry_gate as module_gate


def test_module_gate_is_unconfigured():
    assert module_gate._is_paused_getter is None
    assert module_gate._is_shutdown_getter is None
    assert module_gate._configured is False


def test_module_gate_check_fails_closed():
    result = module_gate.check()
    assert result.ok is False
    assert "not configured" in result.reason.lower()


def test_module_gate_probe_fails_closed():
    result = module_gate.probe()
    assert result.ok is False
    assert "not configured" in result.reason.lower()


def test_configured_gate_reports_configured():
    gate = EntryGate(
        is_paused_getter=lambda: False,
        is_shutdown_getter=lambda: False,
    )
    assert gate._configured is True


def test_gate_becomes_configured_after_setters():
    gate = EntryGate()
    assert gate._configured is False
    gate.set_paused_getter(lambda: False)
    assert gate._configured is False  # only one getter set
    gate.set_shutdown_getter(lambda: False)
    assert gate._configured is True