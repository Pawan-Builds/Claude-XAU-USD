"""Phase 2C / H-01: TradePlan reconfirmation must reject stale setups."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _plan(
    direction="BUY",
    strategy="liquidity_sweep_fvg",
    entry=2000.0,
    sl=1990.0,
    tp=2020.0,
    symbol="GOLD",
):
    return SimpleNamespace(
        direction=direction,
        strategy=strategy,
        entry_price=entry,
        sl_price=sl,
        tp_price=tp,
        symbol=symbol,
        status="WAITING",
    )


def _zone(bottom, top):
    return SimpleNamespace(bottom=bottom, top=top)


@pytest.fixture
def analyzers(monkeypatch):
    """Replace the four analyzers on the smc_strategy module namespace."""
    from src import smc_strategy as smc_mod

    ms = MagicMock(name="market_structure")
    ms.get_higher_tf_bias.return_value = "bullish"
    ms.get_structure_tf_bias.return_value = "bullish"
    ms.is_bos_bullish.return_value = False
    ms.is_bos_bearish.return_value = False
    ms.is_choch_bullish.return_value = False
    ms.is_choch_bearish.return_value = False

    ob = MagicMock(name="order_blocks")
    ob.get_active_bullish_obs.return_value = []
    ob.get_active_bearish_obs.return_value = []

    fvg = MagicMock(name="fair_value_gaps")
    fvg.get_active_bullish_fvgs.return_value = []
    fvg.get_active_bearish_fvgs.return_value = []

    monkeypatch.setattr(smc_mod, "market_structure", ms)
    monkeypatch.setattr(smc_mod, "order_blocks", ob)
    monkeypatch.setattr(smc_mod, "fair_value_gaps", fvg)

    return SimpleNamespace(ms=ms, ob=ob, fvg=fvg, engine=smc_mod.smc_engine)


# ---------- common preconditions ----------


def test_reconfirm_blocks_when_h4_bias_flips(analyzers):
    analyzers.ms.get_higher_tf_bias.return_value = "bearish"
    ok, reason = analyzers.engine.reconfirm_plan(_plan(direction="BUY"))
    assert ok is False
    assert "H4 bias" in reason


def test_reconfirm_blocks_when_sl_wrong_side(analyzers):
    # BUY with SL above entry is invalid.
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(direction="BUY", sl=2005.0, entry=2000.0, tp=2020.0),
    )
    assert ok is False
    assert "SL" in reason


def test_reconfirm_blocks_invalid_direction(analyzers):
    ok, reason = analyzers.engine.reconfirm_plan(_plan(direction="HOLD"))
    assert ok is False
    assert "direction" in reason.lower()


def test_reconfirm_blocks_unknown_strategy(analyzers):
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="mystery_strategy"),
    )
    assert ok is False
    assert "Unknown strategy" in reason


def test_reconfirm_blocks_non_numeric_prices(analyzers):
    plan = _plan()
    plan.entry_price = "not-a-number"
    ok, reason = analyzers.engine.reconfirm_plan(plan)
    assert ok is False
    assert "numeric" in reason.lower()


# ---------- liquidity_sweep_fvg ----------


def test_reconfirm_sweep_fvg_passes_when_fvg_intact(analyzers):
    analyzers.fvg.get_active_bullish_fvgs.return_value = [
        _zone(bottom=1995.0, top=2002.0),
    ]
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="liquidity_sweep_fvg", entry=2000.0),
    )
    assert ok is True, reason


def test_reconfirm_sweep_fvg_blocks_when_fvg_mitigated(analyzers):
    analyzers.fvg.get_active_bullish_fvgs.return_value = []
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="liquidity_sweep_fvg"),
    )
    assert ok is False
    assert "FVG" in reason


def test_reconfirm_sweep_fvg_blocks_when_fvg_does_not_cover_entry(
    analyzers,
):
    # FVG is 2010..2020, entry is 2000 → not covered.
    analyzers.fvg.get_active_bullish_fvgs.return_value = [
        _zone(bottom=2010.0, top=2020.0),
    ]
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="liquidity_sweep_fvg", entry=2000.0),
    )
    assert ok is False
    assert "covers" in reason.lower()


# ---------- liquidity_sweep_choch ----------


def test_reconfirm_sweep_choch_passes_when_choch_still_confirmed(analyzers):
    analyzers.ms.is_choch_bullish.return_value = True
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="liquidity_sweep_choch", sl=1990.0, entry=2000.0),
    )
    assert ok is True, reason


def test_reconfirm_sweep_choch_blocks_when_choch_gone(analyzers):
    analyzers.ms.is_choch_bullish.return_value = False
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="liquidity_sweep_choch"),
    )
    assert ok is False
    assert "CHoCH" in reason


# ---------- bos_order_block ----------


def test_reconfirm_bos_ob_passes_when_all_intact(analyzers):
    analyzers.ms.get_structure_tf_bias.return_value = "bullish"
    analyzers.ms.is_bos_bullish.return_value = True
    analyzers.ob.get_active_bullish_obs.return_value = [
        _zone(bottom=1995.0, top=2005.0),
    ]
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="bos_order_block", entry=2000.0),
    )
    assert ok is True, reason


def test_reconfirm_bos_ob_blocks_when_h1_flips(analyzers):
    analyzers.ms.get_structure_tf_bias.return_value = "bearish"
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="bos_order_block"),
    )
    assert ok is False
    assert "H1 bias" in reason


def test_reconfirm_bos_ob_blocks_when_bos_gone(analyzers):
    analyzers.ms.is_bos_bullish.return_value = False
    analyzers.ob.get_active_bullish_obs.return_value = [
        _zone(bottom=1995.0, top=2005.0),
    ]
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="bos_order_block"),
    )
    assert ok is False
    assert "BOS" in reason


def test_reconfirm_bos_ob_blocks_when_ob_mitigated(analyzers):
    analyzers.ms.is_bos_bullish.return_value = True
    analyzers.ob.get_active_bullish_obs.return_value = []
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="bos_order_block"),
    )
    assert ok is False
    assert "OB" in reason


# ---------- choch_order_block ----------


def test_reconfirm_choch_ob_passes_when_intact(analyzers):
    analyzers.ms.is_choch_bullish.return_value = True
    analyzers.ob.get_active_bullish_obs.return_value = [
        _zone(bottom=1995.0, top=2005.0),
    ]
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="choch_order_block"),
    )
    assert ok is True, reason


def test_reconfirm_choch_ob_blocks_when_choch_gone(analyzers):
    analyzers.ms.is_choch_bullish.return_value = False
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="choch_order_block"),
    )
    assert ok is False
    assert "CHoCH" in reason


# ---------- displacement_fvg ----------


def test_reconfirm_displacement_fvg_passes_when_fvg_intact(analyzers):
    analyzers.fvg.get_active_bullish_fvgs.return_value = [
        _zone(bottom=1995.0, top=2002.0),
    ]
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="displacement_fvg"),
    )
    assert ok is True, reason


def test_reconfirm_displacement_fvg_blocks_when_fvg_gone(analyzers):
    analyzers.fvg.get_active_bullish_fvgs.return_value = []
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="displacement_fvg"),
    )
    assert ok is False
    assert "FVG" in reason


# ---------- liquidity_sweep_ob_fvg ----------


def test_reconfirm_sweep_ob_fvg_passes_when_ob_or_fvg_covers(analyzers):
    analyzers.ob.get_active_bullish_obs.return_value = [
        _zone(bottom=1995.0, top=2005.0),
    ]
    analyzers.fvg.get_active_bullish_fvgs.return_value = [
        _zone(bottom=1998.0, top=2002.0),
    ]
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="liquidity_sweep_ob_fvg"),
    )
    assert ok is True, reason


def test_reconfirm_sweep_ob_fvg_blocks_when_neither_covers(analyzers):
    analyzers.ob.get_active_bullish_obs.return_value = [
        _zone(bottom=2010.0, top=2020.0),
    ]
    analyzers.fvg.get_active_bullish_fvgs.return_value = [
        _zone(bottom=2010.0, top=2020.0),
    ]
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(strategy="liquidity_sweep_ob_fvg", entry=2000.0),
    )
    assert ok is False
    assert "covers" in reason.lower() or "covered" in reason.lower()


# ---------- SELL direction ----------


def test_reconfirm_sell_blocks_when_h4_bullish(analyzers):
    # H4 = bullish, SELL plan → blocked.
    analyzers.ms.get_higher_tf_bias.return_value = "bullish"
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(
            direction="SELL", strategy="liquidity_sweep_fvg",
            entry=2000.0, sl=2010.0, tp=1980.0,
        ),
    )
    assert ok is False
    assert "H4 bias" in reason


def test_reconfirm_sell_sweep_fvg_passes_when_bearish_intact(analyzers):
    analyzers.ms.get_higher_tf_bias.return_value = "bearish"
    analyzers.ms.get_structure_tf_bias.return_value = "bearish"
    analyzers.fvg.get_active_bearish_fvgs.return_value = [
        _zone(bottom=1998.0, top=2005.0),
    ]
    ok, reason = analyzers.engine.reconfirm_plan(
        _plan(
            direction="SELL", strategy="liquidity_sweep_fvg",
            entry=2000.0, sl=2010.0, tp=1980.0,
        ),
    )
    assert ok is True, reason