"""M-07 regression tests: entry_mode configuration.

Asserts:
  * entry_mode defaults to "planned";
  * normalize is case/whitespace-insensitive;
  * unsupported values fail closed (including the reserved-but-
    unimplemented "immediate", and the previously advertised
    "limit"/"market" values);
  * validate_mode_configuration() runs the fail-closed check, so the
    startup path (main.TradingAgent._verify_selected_mode) catches
    bad entry_mode before MT5 connects;
  * the obsolete entry_order_type key is gone from both
    settings.yaml and config_loader.py;
  * the Option 1 planned-entry workflow in main is preserved:
    reconfirm_plan -> entry gate -> market-entry execute_signal.
"""

import re
from pathlib import Path

import pytest
import yaml

from src.config_loader import Config


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _write_config(tmp_path, entry_mode_value=None):
    """Write a minimal DEMO config that Config._load_config() accepts."""
    trading = {"mode": "DEMO"}
    if entry_mode_value is not None:
        trading["entry_mode"] = entry_mode_value
    data = {"trading": trading, "demo": {}, "live": {}}
    path = tmp_path / "settings.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_entry_mode_defaults_to_planned(tmp_path):
    cfg = Config(str(_write_config(tmp_path)))
    assert cfg.trading.entry_mode == "planned"
    assert cfg.validate_entry_mode() is True


def test_entry_mode_accepts_planned_explicitly(tmp_path):
    cfg = Config(str(_write_config(tmp_path, entry_mode_value="planned")))
    assert cfg.trading.entry_mode == "planned"
    cfg.validate_entry_mode()


def test_entry_mode_normalizes_case_and_whitespace(tmp_path):
    cfg = Config(str(_write_config(tmp_path, entry_mode_value="  PLANNED  ")))
    assert cfg.trading.entry_mode == "planned"
    cfg.validate_entry_mode()


@pytest.mark.parametrize(
    "bad_value",
    ["immediate", "market", "limit", "Planned-Immediate", "foo"],
)
def test_entry_mode_fails_closed_on_unsupported(tmp_path, bad_value):
    # An explicit empty string must not silently fall back to default.
    cfg = Config(str(_write_config(tmp_path, entry_mode_value=bad_value)))
    with pytest.raises(RuntimeError, match="entry_mode"):
        cfg.validate_entry_mode()


def test_validate_mode_configuration_calls_validate_entry_mode(tmp_path):
    """The startup path must fail closed on bad entry_mode before any
    credential/MT5 work."""
    cfg = Config(str(_write_config(tmp_path, entry_mode_value="immediate")))
    with pytest.raises(RuntimeError, match="entry_mode"):
        cfg.validate_mode_configuration("DEMO")


def test_settings_yaml_uses_entry_mode_not_entry_order_type():
    settings = yaml.safe_load(
        (PROJECT_ROOT / "config" / "settings.yaml").read_text(
            encoding="utf-8"
        )
    )
    trading = settings.get("trading", {})
    assert "entry_order_type" not in trading, (
        "settings.yaml still contains the misleading entry_order_type "
        "key; M-07 requires it to be replaced by entry_mode."
    )
    assert trading.get("entry_mode") == "planned", (
        "settings.yaml must set trading.entry_mode: planned"
    )


def test_config_loader_has_no_entry_order_type_references():
    """The obsolete key must be gone from *code*.

    A docstring may still mention the old name while explaining the
    rename, so match only real code usages: keyword argument, dataclass
    field, and quoted string literal.
    """
    src = (PROJECT_ROOT / "src" / "config_loader.py").read_text(
        encoding="utf-8"
    )
    assert "entry_order_type=" not in src, (
        "config_loader.py still passes entry_order_type as a kwarg"
    )
    assert "entry_order_type: str" not in src, (
        "config_loader.py still declares entry_order_type on a dataclass"
    )
    assert '"entry_order_type"' not in src, (
        "config_loader.py still reads the string literal "
        '"entry_order_type"'
    )
    assert "'entry_order_type'" not in src, (
        "config_loader.py still reads the string literal "
        "'entry_order_type'"
    )


def test_main_process_trade_plans_preserves_option_1_workflow():
    """Behaviour guard for M-07 Option 1.

    The planned-entry workflow must remain:
      wait for planned price
        -> reconfirm_plan(plan)
        -> _entry_gate.check()
        -> execute_signal(..., market_entry=True, gate=...)
    """
    src = (PROJECT_ROOT / "src" / "main.py").read_text(encoding="utf-8")
    start = src.index("async def _process_trade_plans")
    rest = src[start + len("async def _process_trade_plans"):]
    m = re.search(r"\n    (?:async )?def ", rest)
    assert m is not None, "could not bound _process_trade_plans body"
    body = rest[: m.start()]

    assert "reconfirm_plan" in body, "setup reconfirmation missing"
    assert "_entry_gate.check()" in body, "entry gate call missing"
    assert "market_entry=True" in body, "market execution missing"
    assert "execute_signal" in body, "executor call missing"

    exec_idx = body.index("execute_signal")
    assert body.index("reconfirm_plan") < exec_idx, (
        "reconfirm_plan must run before execute_signal"
    )
    assert body.index("_entry_gate.check()") < exec_idx, (
        "entry gate must run before execute_signal"
    )