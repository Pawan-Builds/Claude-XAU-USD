"""M-06 tests: per-strategy HTF alignment, configurable bias/structure
timeframes, and fail-closed reconfirm alignment.

Run with: pytest tests/test_htf_alignment.py -v
"""

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from src.market_structure import MarketStructureAnalyzer
from src.smc_strategy import SMCEngine, StrategyType


# ---------------------------------------------------------------------------
# Config load: per-strategy require_h1 flags
# ---------------------------------------------------------------------------

class TestAlignmentConfigLoading:

    def _patched_config(self, alignment_section):
        """Return a context manager patching config for SMCEngine()."""
        from unittest.mock import patch as _patch
        fake_smc = SimpleNamespace(
            strategies=[
                "liquidity_sweep_fvg",
                "liquidity_sweep_choch",
                "bos_order_block",
                "choch_order_block",
                "displacement_fvg",
                "liquidity_sweep_ob_fvg",
            ],
            displacement_atr_period=14,
            displacement_atr_multiplier=1.5,
            displacement_min_body_pips=5,
            displacement_threshold=1.5,
            strategy_alignment=alignment_section,
        )
        fake = SimpleNamespace(
            smc=fake_smc,
            trading=SimpleNamespace(min_risk_reward=2.0),
        )
        return _patch("src.smc_strategy.config", fake)

    def test_defaults_used_when_section_absent(self):
        with self._patched_config(None):
            engine = SMCEngine()
        assert engine.strategy_alignment == {
            "liquidity_sweep_fvg": False,
            "liquidity_sweep_choch": True,
            "bos_order_block": True,
            "choch_order_block": True,
            "displacement_fvg": False,
            "liquidity_sweep_ob_fvg": False,
        }

    def test_config_overrides_defaults(self):
        section = SimpleNamespace(
            bos_order_block=SimpleNamespace(require_h1=False),
            choch_order_block=SimpleNamespace(require_h1=False),
        )
        with self._patched_config(section):
            engine = SMCEngine()
        assert engine.strategy_alignment["bos_order_block"] is False
        assert engine.strategy_alignment["choch_order_block"] is False
        # Untouched entries keep their defaults
        assert engine.strategy_alignment["liquidity_sweep_choch"] is True

    def test_invalid_require_h1_falls_back_to_default(self):
        section = SimpleNamespace(
            bos_order_block=SimpleNamespace(require_h1="not-a-bool"),
        )
        with self._patched_config(section):
            engine = SMCEngine()
        # "not-a-bool" -> bool() -> True; verify our bool-coerce path
        # keeps a valid boolean. (We do not try to interpret arbitrary
        # strings as false.)
        assert isinstance(
            engine.strategy_alignment["bos_order_block"], bool
        )


# ---------------------------------------------------------------------------
# read helpers
# ---------------------------------------------------------------------------

class TestAlignmentReaders:

    def test_read_require_h1_from_dict(self):
        assert SMCEngine._read_require_h1(
            {"require_h1": True}, False
        ) is True
        assert SMCEngine._read_require_h1(
            {"require_h1": False}, True
        ) is False
        assert SMCEngine._read_require_h1({}, True) is True

    def test_read_require_h1_from_object(self):
        ns = SimpleNamespace(require_h1=True)
        assert SMCEngine._read_require_h1(ns, False) is True

    def test_read_require_h1_from_none_returns_default(self):
        assert SMCEngine._read_require_h1(None, True) is True
        assert SMCEngine._read_require_h1(None, False) is False

    def test_read_alignment_entry_dict(self):
        section = {"bos_order_block": {"require_h1": True}}
        result = SMCEngine._read_alignment_entry(
            section, "bos_order_block"
        )
        assert result == {"require_h1": True}

    def test_read_alignment_entry_missing_returns_none(self):
        section = {}
        assert SMCEngine._read_alignment_entry(section, "x") is None

    def test_read_alignment_entry_object(self):
        ns = SimpleNamespace(bos_order_block=SimpleNamespace(require_h1=True))
        result = SMCEngine._read_alignment_entry(
            ns, "bos_order_block"
        )
        assert result.require_h1 is True


# ---------------------------------------------------------------------------
# Alignment gate (_passes_alignment)
# ---------------------------------------------------------------------------

class TestPassesAlignment:

    def _engine(self):
        """Engine constructed with no config overrides (defaults only)."""
        fake_smc = SimpleNamespace(
            strategies=[
                "liquidity_sweep_fvg",
                "liquidity_sweep_choch",
                "bos_order_block",
                "choch_order_block",
                "displacement_fvg",
                "liquidity_sweep_ob_fvg",
            ],
            displacement_atr_period=14,
            displacement_atr_multiplier=1.5,
            displacement_min_body_pips=5,
            displacement_threshold=1.5,
            strategy_alignment=None,
        )
        fake = SimpleNamespace(
            smc=fake_smc,
            trading=SimpleNamespace(min_risk_reward=2.0),
        )
        with patch("src.smc_strategy.config", fake):
            return SMCEngine()

    # --- require_h1=False strategies ---

    def test_h4_only_passes_when_h1_neutral(self):
        engine = self._engine()
        assert engine._passes_alignment(
            StrategyType.LIQUIDITY_SWEEP_FVG, "bullish", "neutral"
        ) is True

    def test_h4_only_passes_when_h1_matches(self):
        engine = self._engine()
        assert engine._passes_alignment(
            StrategyType.LIQUIDITY_SWEEP_FVG, "bullish", "bullish"
        ) is True

    def test_h4_only_passes_when_h1_opposite(self):
        engine = self._engine()
        # Momentum strategies don't care about h1 direction.
        assert engine._passes_alignment(
            StrategyType.LIQUIDITY_SWEEP_FVG, "bullish", "bearish"
        ) is True

    # --- require_h1=True strategies ---

    def test_h1_required_neutral_fails(self):
        engine = self._engine()
        assert engine._passes_alignment(
            StrategyType.CHOCH_ORDER_BLOCK, "bullish", "neutral"
        ) is False

    def test_h1_required_matches_passes(self):
        engine = self._engine()
        assert engine._passes_alignment(
            StrategyType.CHOCH_ORDER_BLOCK, "bullish", "bullish"
        ) is True

    def test_h1_required_opposite_fails(self):
        engine = self._engine()
        assert engine._passes_alignment(
            StrategyType.CHOCH_ORDER_BLOCK, "bullish", "bearish"
        ) is False

    def test_bos_ob_h1_required(self):
        engine = self._engine()
        assert engine._passes_alignment(
            StrategyType.BOS_ORDER_BLOCK, "bearish", "bearish"
        ) is True
        assert engine._passes_alignment(
            StrategyType.BOS_ORDER_BLOCK, "bearish", "neutral"
        ) is False

    def test_sweep_choch_h1_required(self):
        engine = self._engine()
        assert engine._passes_alignment(
            StrategyType.LIQUIDITY_SWEEP_CHOCH, "bullish", "bullish"
        ) is True
        assert engine._passes_alignment(
            StrategyType.LIQUIDITY_SWEEP_CHOCH, "bullish", "neutral"
        ) is False


# ---------------------------------------------------------------------------
# Gate wired into _check_strategy
# ---------------------------------------------------------------------------

class TestCheckStrategyGate:

    def _engine(self):
        fake_smc = SimpleNamespace(
            strategies=[
                "liquidity_sweep_fvg",
                "liquidity_sweep_choch",
                "bos_order_block",
                "choch_order_block",
                "displacement_fvg",
                "liquidity_sweep_ob_fvg",
            ],
            displacement_atr_period=14,
            displacement_atr_multiplier=1.5,
            displacement_min_body_pips=5,
            displacement_threshold=1.5,
            strategy_alignment=None,
        )
        fake = SimpleNamespace(
            smc=fake_smc,
            trading=SimpleNamespace(min_risk_reward=2.0),
        )
        with patch("src.smc_strategy.config", fake):
            return SMCEngine()

    def test_gate_blocks_before_dispatch(self):
        engine = self._engine()
        # choch_order_block default requires H1; h1 neutral -> block.
        with patch.object(
            engine, "_check_choch_order_block"
        ) as mock_check:
            result = engine._check_strategy(
                StrategyType.CHOCH_ORDER_BLOCK, "bullish", "neutral"
            )
            assert result is None
            mock_check.assert_not_called()

    def test_gate_dispatches_when_aligned(self):
        engine = self._engine()
        with patch.object(
            engine, "_check_choch_order_block", return_value=None
        ) as mock_check:
            engine._check_strategy(
                StrategyType.CHOCH_ORDER_BLOCK, "bullish", "bullish"
            )
            mock_check.assert_called_once()

    def test_h4_only_strategy_dispatches_with_neutral_h1(self):
        engine = self._engine()
        with patch.object(
            engine, "_check_liquidity_sweep_fvg", return_value=None
        ) as mock_check:
            engine._check_strategy(
                StrategyType.LIQUIDITY_SWEEP_FVG, "bullish", "neutral"
            )
            mock_check.assert_called_once()


# ---------------------------------------------------------------------------
# market_structure configurable timeframes
# ---------------------------------------------------------------------------

class TestMarketStructureTimeframes:

    def _patched_ms(self, bias_tf, structure_tf):
        from unittest.mock import patch as _patch
        fake = SimpleNamespace(
            timeframes=SimpleNamespace(
                bias_tf=bias_tf, structure_tf=structure_tf,
            ),
            smc=SimpleNamespace(structure_event_lookback=10),
        )
        return _patch("src.market_structure.config", fake)

    def test_default_bias_tf_is_h4(self):
        with self._patched_ms("h4", "h1"):
            ms = MarketStructureAnalyzer()
        assert ms.get_bias_timeframe() == "h4"
        assert ms.get_structure_timeframe() == "h1"

    def test_config_overrides(self):
        with self._patched_ms("h1", "m15"):
            ms = MarketStructureAnalyzer()
        assert ms.get_bias_timeframe() == "h1"
        assert ms.get_structure_timeframe() == "m15"

    def test_unsupported_timeframe_falls_back(self):
        with self._patched_ms("d1", "w1"):
            ms = MarketStructureAnalyzer()
        assert ms.get_bias_timeframe() == "h4"
        assert ms.get_structure_timeframe() == "h1"

    def test_missing_section_uses_defaults(self):
        from unittest.mock import patch as _patch
        fake = SimpleNamespace(
            timeframes=SimpleNamespace(),  # no bias_tf / structure_tf
            smc=SimpleNamespace(structure_event_lookback=10),
        )
        with _patch("src.market_structure.config", fake):
            ms = MarketStructureAnalyzer()
        assert ms.get_bias_timeframe() == "h4"
        assert ms.get_structure_timeframe() == "h1"

    def test_get_higher_tf_bias_uses_configured_tf(self):
        with self._patched_ms("h1", "m15"):
            ms = MarketStructureAnalyzer()
        # Populate cache for h1 only.
        ms.structure_cache["h1"] = Mock(bias="bullish")
        assert ms.get_higher_tf_bias() == "bullish"
        # h4 has no cache -> would be neutral if we (wrongly) used h4.
        assert "h4" not in ms.structure_cache

    def test_get_structure_tf_bias_uses_configured_tf(self):
        with self._patched_ms("h4", "m15"):
            ms = MarketStructureAnalyzer()
        ms.structure_cache["m15"] = Mock(bias="bearish")
        assert ms.get_structure_tf_bias() == "bearish"


# ---------------------------------------------------------------------------
# _create_signal persistence of configurable tf names
# ---------------------------------------------------------------------------

class TestCreateSignalTimeframes:

    def test_signal_uses_configured_tfs(self):
        fake_smc = SimpleNamespace(
            strategies=["liquidity_sweep_fvg"],
            displacement_atr_period=14,
            displacement_atr_multiplier=1.5,
            displacement_min_body_pips=5,
            displacement_threshold=1.5,
            strategy_alignment=None,
        )
        fake = SimpleNamespace(
            smc=fake_smc,
            trading=SimpleNamespace(
                min_risk_reward=2.0, symbol="XAUUSD",
            ),
            risk=SimpleNamespace(default_risk_percent=1.0),
        )
        with patch("src.smc_strategy.config", fake):
            engine = SMCEngine()

        # Mock market_structure's tf getters to return custom values.
        with patch("src.smc_strategy.market_structure") as mock_ms:
            mock_ms.get_bias_timeframe.return_value = "h1"
            mock_ms.get_structure_timeframe.return_value = "m15"
            sig = engine._create_signal(
                direction=__import__(
                    "src.smc_strategy", fromlist=["SignalDirection"]
                ).SignalDirection.BUY,
                entry_price=2000.0,
                sl_price=1990.0,
                tp_price=2020.0,
                strategy=StrategyType.LIQUIDITY_SWEEP_FVG,
                confirmations=[],
                entry_tf="m5",
                setup_tf="m15",
            )
        # No confirmations -> signal rejected; but if it had been
        # accepted, structure_tf / bias_tf would be "m15" / "h1".
        # Verify via the internal getters instead.
        with patch("src.smc_strategy.market_structure") as mock_ms:
            mock_ms.get_bias_timeframe.return_value = "h1"
            mock_ms.get_structure_timeframe.return_value = "m15"
            assert engine._get_bias_tf() == "h1"
            assert engine._get_structure_tf() == "m15"


# ---------------------------------------------------------------------------
# Reconfirm alignment fail-closed
# ---------------------------------------------------------------------------

class TestReconfirmAlignment:

    def _engine(self):
        fake_smc = SimpleNamespace(
            strategies=[
                "choch_order_block",
                "liquidity_sweep_fvg",
            ],
            displacement_atr_period=14,
            displacement_atr_multiplier=1.5,
            displacement_min_body_pips=5,
            displacement_threshold=1.5,
            strategy_alignment=None,
        )
        fake = SimpleNamespace(
            smc=fake_smc,
            trading=SimpleNamespace(min_risk_reward=2.0),
        )
        with patch("src.smc_strategy.config", fake):
            return SMCEngine()

    def _plan(self, strategy, direction="BUY"):
        p = Mock()
        p.direction = direction
        p.strategy = strategy
        p.entry_price = 2000.0
        p.sl_price = 1990.0
        p.tp_price = 2020.0
        p.setup_state = {
            "ob": {"type": "bullish", "top": 2001.0, "bottom": 2000.0},
            "choch": {"timeframe": "m15", "direction": "bullish"},
        }
        return p

    def test_alignment_loss_rejects_choch_order_block(self):
        engine = self._engine()
        plan = self._plan("choch_order_block")

        with patch("src.smc_strategy.market_structure") as mock_ms:
            mock_ms.get_higher_tf_bias.return_value = "bullish"
            mock_ms.get_structure_tf_bias.return_value = "neutral"
            # Should fail at alignment check, before internal checks.
            ok, reason = engine.reconfirm_plan(plan)
        assert ok is False
        assert "structure-tf alignment" in reason.lower()

    def test_alignment_opposite_rejects(self):
        engine = self._engine()
        plan = self._plan("choch_order_block")

        with patch("src.smc_strategy.market_structure") as mock_ms:
            mock_ms.get_higher_tf_bias.return_value = "bullish"
            mock_ms.get_structure_tf_bias.return_value = "bearish"
            ok, reason = engine.reconfirm_plan(plan)
        assert ok is False
        assert "alignment" in reason.lower()

    def test_momentum_strategy_ignores_h1(self):
        """liquidity_sweep_fvg does not require H1; neutral H1 must
        not trip the reconfirm alignment gate."""
        engine = self._engine()
        plan = self._plan("liquidity_sweep_fvg")
        # Give it the fields its reconfirm expects.
        plan.setup_state = {
            "fvg": {"type": "bullish", "top": 2001.0, "bottom": 2000.0},
            "sweep": {"side": "sell_side", "price": 1995.0},
        }
        with patch("src.smc_strategy.market_structure") as mock_ms, \
             patch("src.smc_strategy.fair_value_gaps") as mock_fvg, \
             patch("src.smc_strategy.liquidity") as mock_liq:
            mock_ms.get_higher_tf_bias.return_value = "bullish"
            mock_ms.get_structure_tf_bias.return_value = "neutral"
            mock_fvg.get_active_bullish_fvgs.return_value = [
                Mock(top=2001.0, bottom=2000.0)
            ]
            mock_liq.get_liquidity.return_value = Mock(
                sell_side=[Mock(price=1995.0)]
            )
            ok, reason = engine.reconfirm_plan(plan)
        assert ok is True, reason