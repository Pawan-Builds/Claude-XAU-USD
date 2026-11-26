"""Focused L-01 test: each EntryGate setter has exactly one definition."""
import inspect
from unittest.mock import MagicMock

from src.entry_gate import EntryGate


def _source() -> str:
    return inspect.getsource(EntryGate)


def test_set_paused_getter_defined_once():
    src = _source()
    assert src.count("def set_paused_getter(") == 1, (
        "set_paused_getter must have exactly one authoritative definition"
    )


def test_set_shutdown_getter_defined_once():
    src = _source()
    assert src.count("def set_shutdown_getter(") == 1, (
        "set_shutdown_getter must have exactly one authoritative definition"
    )


def test_setters_update_private_attributes():
    gate = EntryGate()
    assert gate._is_paused_getter is None
    assert gate._is_shutdown_getter is None

    p = MagicMock(return_value=False)
    s = MagicMock(return_value=False)
    gate.set_paused_getter(p)
    gate.set_shutdown_getter(s)

    assert gate._is_paused_getter is p
    assert gate._is_shutdown_getter is s

    # Single authoritative definition must actually be the one invoked.
    gate._is_paused_getter()
    p.assert_called_once()


def test_configured_reflects_setters():
    gate = EntryGate()
    assert gate._configured is False
    gate.set_paused_getter(lambda: False)
    assert gate._configured is False
    gate.set_shutdown_getter(lambda: False)
    assert gate._configured is True