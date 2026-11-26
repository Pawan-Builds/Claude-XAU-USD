"""SMC strategy engine.

Detects multi-timeframe SMC setups and creates validated trade
candidates. Actual entry execution and waiting for price to reach
an entry zone belong to the order-execution layer.

M-06 HTF alignment semantics:
- `analyze_all_strategies` computes `h4_bias` (value of
  `config.timeframes.bias_tf`, default h4) and `h1_bias` (value of
  `config.timeframes.structure_tf`, default h1). The historical
  parameter names are retained (D23) for compatibility, but they
  now refer to whichever timeframes the config selects.
- A per-strategy gate (`_passes_alignment`) enforces the
  `config.smc.strategy_alignment.<strategy>.require_h1` rule.
  Structural strategies require H4+H1 alignment; momentum
  strategies require H4 only.
"""

from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from src.smc_context import AnalyzerProxy
from src.market_data import market_data

# N-H04: proxies resolve to the current context's analyzer instance
# (backtest) or the module-level singleton (live). No method-body
# changes are required in this file — the proxies preserve the exact
# attribute-access syntax the rest of this module already uses.
market_structure = AnalyzerProxy("market_structure")
liquidity = AnalyzerProxy("liquidity")
order_blocks = AnalyzerProxy("order_blocks")
fair_value_gaps = AnalyzerProxy("fair_value_gaps")
premium_discount = AnalyzerProxy("premium_discount")
from src.config_loader import config
from src.mt5_connector import mt5_connector, MT5Connector
from src.logger import get_logger

logger = get_logger(__name__)


class SignalDirection(Enum):
    BUY = "BUY"
    SELL = "SELL"


class StrategyType(Enum):
    LIQUIDITY_SWEEP_FVG = "liquidity_sweep_fvg"
    LIQUIDITY_SWEEP_CHOCH = "liquidity_sweep_choch"
    BOS_ORDER_BLOCK = "bos_order_block"
    CHOCH_ORDER_BLOCK = "choch_order_block"
    DISPLACEMENT_FVG = "displacement_fvg"
    LIQUIDITY_SWEEP_OB_FVG = "liquidity_sweep_ob_fvg"


@dataclass
class Confirmation:
    name: str
    passed: bool
    details: str


@dataclass
class TradeSignal:
    timestamp: datetime
    symbol: str
    direction: SignalDirection
    entry_price: float
    sl_price: float
    tp_price: float
    lot_size: float
    risk_percent: float
    risk_reward: float
    strategy: StrategyType
    confirmations: List[Confirmation]
    timeframe_analysis: Dict[str, Any]
    entry_tf: str
    setup_tf: str
    structure_tf: str
    bias_tf: str
    fingerprint: str = ""
    # H-03: structured identity of the setup that produced this signal.
    # Persisted into TradePlan.setup_state and re-checked at reconfirm.
    setup_state: Dict[str, Any] = field(default_factory=dict)


class SMCEngine:
    """Multi-timeframe SMC setup detection engine."""
    # M-06: strategies that default to requiring H1 alignment.
    _DEFAULT_STRATEGIES_REQUIRING_H1 = frozenset({
        "bos_order_block",
        "choch_order_block",
        "liquidity_sweep_choch",
    })

    def __init__(self):
        self.strategies = list(config.smc.strategies)
        self.min_risk_reward = float(config.trading.min_risk_reward)
        self.point_value = 0.01
        self._refresh_displacement_config()
        # M-06: per-strategy require_h1 flags.
        self.strategy_alignment = self._load_strategy_alignment()

    def _load_strategy_alignment(self) -> Dict[str, bool]:
        """M-06: load per-strategy require_h1 flags.

        All known strategies are included with their D20 defaults,
        then overridden by config.smc.strategy_alignment.<name>.
        require_h1 where present. Missing/invalid entries keep the
        default.
        """
        all_strategies = [
            "liquidity_sweep_fvg",
            "liquidity_sweep_choch",
            "bos_order_block",
            "choch_order_block",
            "displacement_fvg",
            "liquidity_sweep_ob_fvg",
        ]
        defaults = {
            name: (name in self._DEFAULT_STRATEGIES_REQUIRING_H1)
            for name in all_strategies
        }
        result = dict(defaults)

        try:
            section = getattr(
                config.smc, "strategy_alignment", None
            )
        except Exception:
            logger.exception(
                "Could not read smc.strategy_alignment; using "
                "defaults."
            )
            return result

        if section is None:
            return result

        for name, default_val in defaults.items():
            try:
                entry = self._read_alignment_entry(section, name)
                if entry is None:
                    continue
                result[name] = self._read_require_h1(
                    entry, default_val
                )
            except Exception:
                logger.exception(
                    "Failed to read alignment for %s; using "
                    "default.", name,
                )

        return result

    @staticmethod
    def _read_alignment_entry(section, name):
        """Return the per-strategy alignment entry, dict or attribute."""
        if isinstance(section, dict):
            return section.get(name)
        return getattr(section, name, None)

    @staticmethod
    def _read_require_h1(entry, default: bool) -> bool:
        """Return require_h1 from an entry; fall back on invalid."""
        if entry is None:
            return default
        if isinstance(entry, dict):
            value = entry.get("require_h1", default)
        else:
            value = getattr(entry, "require_h1", default)
        if isinstance(value, bool):
            return value
        try:
            return bool(value)
        except Exception:
            return default

    def _passes_alignment(
        self,
        strategy: StrategyType,
        bias_tf_bias: str,
        structure_tf_bias: str,
    ) -> bool:
        """M-06: per-strategy HTF alignment gate.

        `bias_tf_bias`        = value of config.timeframes.bias_tf
                                (default h4).
        `structure_tf_bias`   = value of config.timeframes.structure_tf
                                (default h1).
        Per D23, the parameter names used in this file's callers
        remain `h4_bias` / `h1_bias`; here we document the actual
        semantics.

        H4/bias_tf must be bullish or bearish (enforced upstream by
        `analyze_all_strategies`). If the strategy requires H1
        alignment, the structure-TF bias must also be bullish or
        bearish and must match the bias-TF bias.
        """
        require_h1 = self.strategy_alignment.get(
            strategy.value, False
        )
        if not require_h1:
            return True

        if structure_tf_bias not in ("bullish", "bearish"):
            return False

        if bias_tf_bias != structure_tf_bias:
            return False

        return True

    def _refresh_displacement_config(self) -> None:
        """Load M-02 displacement parameters from config.smc.

        Falls back to legacy `displacement_threshold` for the multiplier
        if the new key is absent. All values validated; invalid values
        revert to the conservative defaults.
        """
        smc = config.smc

        try:
            ap = getattr(smc, "displacement_atr_period", 14)
            if (
                isinstance(ap, int)
                and not isinstance(ap, bool)
                and 1 <= ap <= 500
            ):
                self.displacement_atr_period = ap
            else:
                self.displacement_atr_period = 14
        except Exception:
            self.displacement_atr_period = 14

        try:
            am = getattr(
                smc,
                "displacement_atr_multiplier",
                getattr(smc, "displacement_threshold", 1.5),
            )
            am = float(am)
            if np.isfinite(am) and 0.0 < am <= 100.0:
                self.displacement_atr_multiplier = am
            else:
                self.displacement_atr_multiplier = 1.5
        except (TypeError, ValueError):
            self.displacement_atr_multiplier = 1.5

        try:
            mb = getattr(smc, "displacement_min_body_pips", 5.0)
            mb = float(mb)
            if np.isfinite(mb) and 0.0 <= mb <= 100000.0:
                self.displacement_min_body_pips = mb
            else:
                self.displacement_min_body_pips = 5.0
        except (TypeError, ValueError):
            self.displacement_min_body_pips = 5.0

    def _get_pip_size(self) -> float:
        """Broker-aware pip size for the current symbol.

        Distinct from `_get_point_value`, which returns the raw point.
        Used by displacement's absolute floor, which is pip-denominated.
        """
        try:
            info = mt5_connector.get_symbol_info()
        except Exception:
            info = None
        if info is not None:
            try:
                size = MT5Connector.pip_size_for(info)
                if isinstance(size, (int, float)) and size > 0:
                    return float(size)
            except Exception:
                pass
        return 0.01

    def _get_bias_tf(self) -> str:
        """M-06: return configured bias timeframe (default h4)."""
        try:
            return market_structure.get_bias_timeframe()
        except Exception:
            return "h4"

    def _get_structure_tf(self) -> str:
        """M-06: return configured structure timeframe (default h1)."""
        try:
            return market_structure.get_structure_timeframe()
        except Exception:
            return "h1"

    def _get_point_value(self) -> float:
        symbol_info = mt5_connector.get_symbol_info()
        if symbol_info and symbol_info.point > 0:
            self.point_value = float(symbol_info.point)
        return self.point_value

    def _get_safe_stop_distance(self, tf: str = "m15") -> float:
        atr = market_data.calculate_atr(tf, 14)
        point = self._get_point_value()

        if atr is None or atr <= 0:
            return 150 * point

        safe_distance = max(atr * 1.5, 100 * point)
        return safe_distance

    # ------------------------------------------------------------------
    # Public analysis
    # ------------------------------------------------------------------

    def analyze_all_strategies(self) -> List[TradeSignal]:
        signals: List[TradeSignal] = []

        try:
            market_structure.analyze_all_timeframes()
            liquidity.analyze_all_timeframes()
            order_blocks.analyze_all_timeframes()
            fair_value_gaps.analyze_all_timeframes()
            premium_discount.analyze_all_timeframes()

            h4_bias = market_structure.get_higher_tf_bias()
            h1_bias = market_structure.get_structure_tf_bias()

            if h4_bias not in ("bullish", "bearish"):
                logger.debug(
                    "No trade candidate: H4 bias is neutral or unavailable."
                )
                return signals

            for strategy_name in self.strategies:
                try:
                    strategy = StrategyType(strategy_name)
                except ValueError:
                    logger.warning(
                        "Unknown SMC strategy configured: %s", strategy_name
                    )
                    continue

                try:
                    signal = self._check_strategy(strategy, h4_bias, h1_bias)
                except Exception:
                    logger.exception(
                        "Strategy evaluation failed: %s", strategy.value
                    )
                    continue

                if signal is not None:
                    signals.append(signal)

            if signals:
                logger.info(
                    "SMC analysis produced %d candidate(s). H4=%s H1=%s",
                    len(signals), h4_bias, h1_bias,
                )
            else:
                logger.debug(
                    "No valid SMC candidate. H4=%s H1=%s", h4_bias, h1_bias
                )

            return signals

        except Exception:
            logger.exception("SMC analysis cycle failed.")
            return []

    def _check_strategy(
        self,
        strategy: StrategyType,
        h4_bias: str,
        h1_bias: str,
    ) -> Optional[TradeSignal]:

        # M-06: enforce per-strategy HTF alignment before dispatch.
        if not self._passes_alignment(strategy, h4_bias, h1_bias):
            logger.debug(
                "M-06 alignment gate blocked %s "
                "(bias_tf=%s structure_tf=%s).",
                strategy.value, h4_bias, h1_bias,
            )
            return None

        checks = {
            StrategyType.LIQUIDITY_SWEEP_FVG: self._check_liquidity_sweep_fvg,
            StrategyType.LIQUIDITY_SWEEP_CHOCH: self._check_liquidity_sweep_choch,
            StrategyType.BOS_ORDER_BLOCK: self._check_bos_order_block,
            StrategyType.CHOCH_ORDER_BLOCK: self._check_choch_order_block,
            StrategyType.DISPLACEMENT_FVG: self._check_displacement_fvg,
            StrategyType.LIQUIDITY_SWEEP_OB_FVG: self._check_liquidity_sweep_ob_fvg,
        }

        checker = checks.get(strategy)
        return checker(h4_bias, h1_bias) if checker else None
    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _latest_close(self, timeframe: str) -> Optional[float]:
        candle = market_data.get_latest_closed_candle(timeframe)
        if candle is None:
            return None
        value = float(candle.close)
        return value if np.isfinite(value) and value > 0 else None

    def _latest_closed_close(self, timeframe: str) -> Optional[float]:
        candles = market_data.get_closed_candles(timeframe, 1)
        if not candles:
            return None
        value = float(candles[-1].close)
        return value if np.isfinite(value) and value > 0 else None

    def _get_level(self, levels: Any, key: str) -> Optional[float]:
        if not isinstance(levels, dict):
            return None
        value = levels.get(key)
        if value is None:
            return None
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) and value > 0 else None

    def _make_confirmations(
        self,
        items: List[tuple],
    ) -> List[Confirmation]:
        return [
            Confirmation(name=name, passed=passed, details=details)
            for name, passed, details in items
        ]

    def _select_target(
        self,
        direction: SignalDirection,
        entry: float,
        candidates: Any,
        fallback_distance: float,
    ) -> float:

        valid_prices = []

        for item in candidates or []:
            try:
                price = float(item.price)
            except (AttributeError, TypeError, ValueError):
                continue

            if not np.isfinite(price) or price <= 0:
                continue

            if direction == SignalDirection.BUY and price > entry:
                valid_prices.append(price)
            elif direction == SignalDirection.SELL and price < entry:
                valid_prices.append(price)

        if valid_prices:
            if direction == SignalDirection.BUY:
                return min(valid_prices)
            return max(valid_prices)

        if direction == SignalDirection.BUY:
            return entry + fallback_distance
        return entry - fallback_distance

    def _find_sweep_price(
        self, timeframe: str, sweep_type: str
    ) -> Optional[float]:
        """H-03: return the price of the recent sweep matching sweep_type.

        Used only at plan-creation time to persist the swept level's
        identity. Returns None if not found (caller stores None; the
        reconfirmer will then skip the sweep-level existence check and
        rely on downstream conditions).
        """
        sweeps = liquidity.get_recent_sweep_details(timeframe)
        for sweep in sweeps:
            if sweep.get("type") == sweep_type:
                try:
                    return float(sweep["price"])
                except (TypeError, ValueError, KeyError):
                    return None
        return None

    def _create_signal(
        self,
        direction: SignalDirection,
        entry_price: float,
        sl_price: float,
        tp_price: float,
        strategy: StrategyType,
        confirmations: List[Confirmation],
        entry_tf: str,
        setup_tf: str,
        setup_state: Optional[Dict[str, Any]] = None,
    ) -> Optional[TradeSignal]:

        try:
            entry = float(entry_price)
            stop = float(sl_price)
            target = float(tp_price)
        except (TypeError, ValueError):
            logger.debug(
                "%s rejected: non-numeric entry/SL/TP.", strategy.value
            )
            return None

        if not all(np.isfinite(v) and v > 0 for v in (entry, stop, target)):
            logger.debug("%s rejected: invalid price values.", strategy.value)
            return None

        if direction == SignalDirection.BUY:
            risk = entry - stop
            reward = target - entry
        else:
            risk = stop - entry
            reward = entry - target

        if risk <= 0 or reward <= 0:
            logger.debug(
                "%s rejected: incorrect price geometry. direction=%s "
                "entry=%.3f SL=%.3f TP=%.3f",
                strategy.value, direction.value, entry, stop, target,
            )
            return None

        risk_reward = reward / risk

        if risk_reward < self.min_risk_reward:
            logger.debug(
                "%s rejected: RR %.2f is below configured minimum %.2f.",
                strategy.value, risk_reward, self.min_risk_reward,
            )
            return None

        if not confirmations or not all(c.passed for c in confirmations):
            logger.debug(
                "%s rejected: one or more confirmations failed.",
                strategy.value,
            )
            return None

        risk_percent = float(config.risk.default_risk_percent)

        return TradeSignal(
            timestamp=datetime.now(timezone.utc),
            symbol=config.trading.symbol,
            direction=direction,
            entry_price=entry,
            sl_price=stop,
            tp_price=target,
            lot_size=0.0,
            risk_percent=risk_percent,
            risk_reward=risk_reward,
            strategy=strategy,
            confirmations=confirmations,
            timeframe_analysis={
                "entry": entry,
                "stop_loss": stop,
                "take_profit": target,
                "risk_reward": risk_reward,
            },
            entry_tf=entry_tf,
            setup_tf=setup_tf,
            structure_tf=self._get_structure_tf(),
            bias_tf=self._get_bias_tf(),
            setup_state=setup_state or {},
        )
    # ------------------------------------------------------------------
    # Strategy 1: Liquidity sweep + FVG
    # ------------------------------------------------------------------

    def _check_liquidity_sweep_fvg(
        self, h4_bias: str, h1_bias: str,
    ) -> Optional[TradeSignal]:

        m15_liq = liquidity.get_liquidity("m15")
        m5_fvgs = fair_value_gaps.get_fvgs("m5")

        if not m15_liq or not m5_fvgs:
            return None

        safe_dist = self._get_safe_stop_distance("m15")

        if h4_bias == "bullish":
            swept = liquidity.was_sell_side_swept("m15")
            fvgs = fair_value_gaps.get_active_bullish_fvgs("m5")

            confirmations = self._make_confirmations([
                ("H4 bullish bias", True, "H4 bias is bullish."),
                ("M15 sell-side sweep", swept,
                 "Sell-side liquidity swept." if swept else "No sell-side sweep."),
                ("M5 bullish FVG", bool(fvgs),
                 f"{len(fvgs)} active bullish FVG(s)."),
            ])

            if not swept or not fvgs:
                return None

            fvg = fvgs[0]
            entry = float(fvg.bottom)
            stop = entry - safe_dist
            target = self._select_target(
                SignalDirection.BUY, entry,
                liquidity.get_buy_side_liquidity("h1"), safe_dist * 2,
            )

            return self._create_signal(
                SignalDirection.BUY, entry, stop, target,
                StrategyType.LIQUIDITY_SWEEP_FVG, confirmations, "m5", "m15",
                setup_state={
                    "strategy": "liquidity_sweep_fvg",
                    "direction": "BUY",
                    "h4_bias_at_creation": "bullish",
                    "fvg": {
                        "type": "bullish",
                        "top": float(fvg.top),
                        "bottom": float(fvg.bottom),
                    },
                    "sweep": {
                        "side": "sell_side",
                        "price": self._find_sweep_price(
                            "m15", "sell_side_sweep"
                        ),
                    },
                },
            )

        if h4_bias == "bearish":
            swept = liquidity.was_buy_side_swept("m15")
            fvgs = fair_value_gaps.get_active_bearish_fvgs("m5")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                ("M15 buy-side sweep", swept,
                 "Buy-side liquidity swept." if swept else "No buy-side sweep."),
                ("M5 bearish FVG", bool(fvgs),
                 f"{len(fvgs)} active bearish FVG(s)."),
            ])

            if not swept or not fvgs:
                return None

            fvg = fvgs[0]
            entry = float(fvg.top)
            stop = entry + safe_dist
            target = self._select_target(
                SignalDirection.SELL, entry,
                liquidity.get_sell_side_liquidity("h1"), safe_dist * 2,
            )

            return self._create_signal(
                SignalDirection.SELL, entry, stop, target,
                StrategyType.LIQUIDITY_SWEEP_FVG, confirmations, "m5", "m15",
                setup_state={
                    "strategy": "liquidity_sweep_fvg",
                    "direction": "SELL",
                    "h4_bias_at_creation": "bearish",
                    "fvg": {
                        "type": "bearish",
                        "top": float(fvg.top),
                        "bottom": float(fvg.bottom),
                    },
                    "sweep": {
                        "side": "buy_side",
                        "price": self._find_sweep_price(
                            "m15", "buy_side_sweep"
                        ),
                    },
                },
            )

        return None

    # ------------------------------------------------------------------
    # Strategy 2: Liquidity sweep + CHoCH
    # ------------------------------------------------------------------

    def _check_liquidity_sweep_choch(
        self, h4_bias: str, h1_bias: str,
    ) -> Optional[TradeSignal]:

        if not liquidity.get_liquidity("m15"):
            return None

        structure = market_structure.get_structure("m15")
        if structure is None:
            return None

        levels = structure.structure.structure_levels
        safe_dist = self._get_safe_stop_distance("m15")

        if h4_bias == "bullish":
            swept = liquidity.was_sell_side_swept("m15")
            choch = market_structure.is_choch_bullish("m15")

            confirmations = self._make_confirmations([
                ("H4 bullish bias", True, "H4 bias is bullish."),
                ("M15 sell-side sweep", swept,
                 "Sell-side liquidity swept." if swept else "No sell-side sweep."),
                ("M15 bullish CHoCH", choch,
                 "Bullish CHoCH confirmed." if choch else "No bullish CHoCH."),
            ])

            if not swept or not choch:
                return None

            entry = self._latest_closed_close("m5")
            if entry is None:
                return None

            recent_low = self._get_level(levels, "recent_low")
            if recent_low is not None and recent_low < entry:
                stop = recent_low - (safe_dist * 0.2)
            else:
                stop = entry - safe_dist

            target = self._select_target(
                SignalDirection.BUY, entry,
                liquidity.get_buy_side_liquidity("h1"), safe_dist * 2.5,
            )

            return self._create_signal(
                SignalDirection.BUY, entry, stop, target,
                StrategyType.LIQUIDITY_SWEEP_CHOCH, confirmations, "m5", "m15",
                setup_state={
                    "strategy": "liquidity_sweep_choch",
                    "direction": "BUY",
                    "h4_bias_at_creation": "bullish",
                    "sweep": {
                        "side": "sell_side",
                        "price": self._find_sweep_price(
                            "m15", "sell_side_sweep"
                        ),
                    },
                    "choch": {
                        "timeframe": "m15",
                        "direction": "bullish",
                    },
                },
            )

        if h4_bias == "bearish":
            swept = liquidity.was_buy_side_swept("m15")
            choch = market_structure.is_choch_bearish("m15")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                ("M15 buy-side sweep", swept,
                 "Buy-side liquidity swept." if swept else "No buy-side sweep."),
                ("M15 bearish CHoCH", choch,
                 "Bearish CHoCH confirmed." if choch else "No bearish CHoCH."),
            ])

            if not swept or not choch:
                return None

            entry = self._latest_closed_close("m5")
            if entry is None:
                return None

            recent_high = self._get_level(levels, "recent_high")
            if recent_high is not None and recent_high > entry:
                stop = recent_high + (safe_dist * 0.2)
            else:
                stop = entry + safe_dist

            target = self._select_target(
                SignalDirection.SELL, entry,
                liquidity.get_sell_side_liquidity("h1"), safe_dist * 2.5,
            )

            return self._create_signal(
                SignalDirection.SELL, entry, stop, target,
                StrategyType.LIQUIDITY_SWEEP_CHOCH, confirmations, "m5", "m15",
                setup_state={
                    "strategy": "liquidity_sweep_choch",
                    "direction": "SELL",
                    "h4_bias_at_creation": "bearish",
                    "sweep": {
                        "side": "buy_side",
                        "price": self._find_sweep_price(
                            "m15", "buy_side_sweep"
                        ),
                    },
                    "choch": {
                        "timeframe": "m15",
                        "direction": "bearish",
                    },
                },
            )

        return None

    # ------------------------------------------------------------------
    # Strategy 3: BOS + Order Block
    # ------------------------------------------------------------------

    def _check_bos_order_block(
        self, h4_bias: str, h1_bias: str,
    ) -> Optional[TradeSignal]:

        if market_structure.get_structure("h1") is None:
            return None

        obs = order_blocks.get_obs("m15")
        if not obs:
            return None

        safe_dist = self._get_safe_stop_distance("h1")

        if h4_bias == "bullish" and h1_bias == "bullish":
            bos = market_structure.is_bos_bullish("h1")
            bullish_obs = order_blocks.get_active_bullish_obs("m15")

            confirmations = self._make_confirmations([
                ("H4 bullish bias", True, "H4 bias is bullish."),
                ("H1 bullish bias", True, "H1 bias is bullish."),
                ("H1 bullish BOS", bos,
                 "Bullish BOS confirmed." if bos else "No bullish BOS."),
                ("M15 bullish OB", bool(bullish_obs),
                 f"{len(bullish_obs)} active bullish OB(s)."),
            ])

            if not bos or not bullish_obs:
                return None

            ob = bullish_obs[0]
            entry = float(ob.bottom)
            stop = entry - safe_dist
            target = self._select_target(
                SignalDirection.BUY, entry,
                liquidity.get_buy_side_liquidity("h1"), safe_dist * 2.5,
            )

            return self._create_signal(
                SignalDirection.BUY, entry, stop, target,
                StrategyType.BOS_ORDER_BLOCK, confirmations, "m15", "h1",
                setup_state={
                    "strategy": "bos_order_block",
                    "direction": "BUY",
                    "h4_bias_at_creation": "bullish",
                    "h1_bias_at_creation": "bullish",
                    "ob": {
                        "type": "bullish",
                        "top": float(ob.top),
                        "bottom": float(ob.bottom),
                    },
                    "bos": {
                        "timeframe": "h1",
                        "direction": "bullish",
                    },
                },
            )

        if h4_bias == "bearish" and h1_bias == "bearish":
            bos = market_structure.is_bos_bearish("h1")
            bearish_obs = order_blocks.get_active_bearish_obs("m15")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                ("H1 bearish bias", True, "H1 bias is bearish."),
                ("H1 bearish BOS", bos,
                 "Bearish BOS confirmed." if bos else "No bearish BOS."),
                ("M15 bearish OB", bool(bearish_obs),
                 f"{len(bearish_obs)} active bearish OB(s)."),
            ])

            if not bos or not bearish_obs:
                return None

            ob = bearish_obs[0]
            entry = float(ob.top)
            stop = entry + safe_dist
            target = self._select_target(
                SignalDirection.SELL, entry,
                liquidity.get_sell_side_liquidity("h1"), safe_dist * 2.5,
            )

            return self._create_signal(
                SignalDirection.SELL, entry, stop, target,
                StrategyType.BOS_ORDER_BLOCK, confirmations, "m15", "h1",
                setup_state={
                    "strategy": "bos_order_block",
                    "direction": "SELL",
                    "h4_bias_at_creation": "bearish",
                    "h1_bias_at_creation": "bearish",
                    "ob": {
                        "type": "bearish",
                        "top": float(ob.top),
                        "bottom": float(ob.bottom),
                    },
                    "bos": {
                        "timeframe": "h1",
                        "direction": "bearish",
                    },
                },
            )

        return None

    # ------------------------------------------------------------------
    # Strategy 4: CHoCH + Order Block
    # ------------------------------------------------------------------

    def _check_choch_order_block(
        self, h4_bias: str, h1_bias: str,
    ) -> Optional[TradeSignal]:

        if market_structure.get_structure("m15") is None:
            return None

        obs = order_blocks.get_obs("m5")
        if not obs:
            return None

        safe_dist = self._get_safe_stop_distance("m15")

        if h4_bias == "bullish":
            choch = market_structure.is_choch_bullish("m15")
            bullish_obs = order_blocks.get_active_bullish_obs("m5")

            confirmations = self._make_confirmations([
                ("H4 bullish bias", True, "H4 bias is bullish."),
                ("M15 bullish CHoCH", choch,
                 "Bullish CHoCH confirmed." if choch else "No bullish CHoCH."),
                ("M5 bullish OB", bool(bullish_obs),
                 f"{len(bullish_obs)} active bullish OB(s)."),
            ])

            if not choch or not bullish_obs:
                return None

            ob = bullish_obs[0]
            entry = float(ob.bottom)
            stop = entry - safe_dist
            target = self._select_target(
                SignalDirection.BUY, entry,
                liquidity.get_buy_side_liquidity("m15"), safe_dist * 2,
            )

            return self._create_signal(
                SignalDirection.BUY, entry, stop, target,
                StrategyType.CHOCH_ORDER_BLOCK, confirmations, "m5", "m15",
                setup_state={
                    "strategy": "choch_order_block",
                    "direction": "BUY",
                    "h4_bias_at_creation": "bullish",
                    "ob": {
                        "type": "bullish",
                        "top": float(ob.top),
                        "bottom": float(ob.bottom),
                    },
                    "choch": {
                        "timeframe": "m15",
                        "direction": "bullish",
                    },
                },
            )

        if h4_bias == "bearish":
            choch = market_structure.is_choch_bearish("m15")
            bearish_obs = order_blocks.get_active_bearish_obs("m5")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                ("M15 bearish CHoCH", choch,
                 "Bearish CHoCH confirmed." if choch else "No bearish CHoCH."),
                ("M5 bearish OB", bool(bearish_obs),
                 f"{len(bearish_obs)} active bearish OB(s)."),
            ])

            if not choch or not bearish_obs:
                return None

            ob = bearish_obs[0]
            entry = float(ob.top)
            stop = entry + safe_dist
            target = self._select_target(
                SignalDirection.SELL, entry,
                liquidity.get_sell_side_liquidity("m15"), safe_dist * 2,
            )

            return self._create_signal(
                SignalDirection.SELL, entry, stop, target,
                StrategyType.CHOCH_ORDER_BLOCK, confirmations, "m5", "m15",
                setup_state={
                    "strategy": "choch_order_block",
                    "direction": "SELL",
                    "h4_bias_at_creation": "bearish",
                    "ob": {
                        "type": "bearish",
                        "top": float(ob.top),
                        "bottom": float(ob.bottom),
                    },
                    "choch": {
                        "timeframe": "m15",
                        "direction": "bearish",
                    },
                },
            )

        return None

    # ------------------------------------------------------------------
    # Strategy 5: Displacement + FVG
    # ------------------------------------------------------------------

    def _check_displacement_fvg(
        self, h4_bias: str, h1_bias: str,
    ) -> Optional[TradeSignal]:

        from src.indicators import detect_displacement

        candles = market_data.get_closed_candles("m5", 30)
        if not candles:
            return None

        # Need atr_period+1 candles for ATR at any index i, plus
        # `lookback` more candles so the last `lookback` positions are
        # fully evaluable. +2 safety margin for off-by-one.
        required = self.displacement_atr_period + 10 + 2
        if len(candles) < required:
            logger.debug(
                "Insufficient M5 candles for displacement check: "
                "%d < %d", len(candles), required,
            )
            return None

        opens = np.asarray([c.open for c in candles], dtype=float)
        highs = np.asarray([c.high for c in candles], dtype=float)
        lows = np.asarray([c.low for c in candles], dtype=float)
        closes = np.asarray([c.close for c in candles], dtype=float)

        if not all(
            np.all(np.isfinite(arr))
            for arr in (opens, highs, lows, closes)
        ):
            return None

        displacements = detect_displacement(
            opens, highs, lows, closes,
            lookback=10,
            atr_period=self.displacement_atr_period,
            atr_multiplier=self.displacement_atr_multiplier,
            min_body_pips=self.displacement_min_body_pips,
            point_value=self._get_pip_size(),
        )

        if not displacements:
            return None

        recent = displacements[-1]
        if not recent:
            return None

        fvgs = fair_value_gaps.get_fvgs("m5")
        if not fvgs:
            return None

        displacement_direction = recent.get("direction")
        atr_ratio = float(recent.get("atr_ratio", 0.0))
        safe_dist = self._get_safe_stop_distance("m5")

        if h4_bias == "bullish" and displacement_direction == "bullish":
            bullish_fvgs = fair_value_gaps.get_active_bullish_fvgs("m5")

            confirmations = self._make_confirmations([
                ("H4 bullish bias", True, "H4 bias is bullish."),
                ("M5 bullish displacement", True,
                 f"Displacement ATR-ratio: {atr_ratio:.2f}."),
                ("M5 bullish FVG", bool(bullish_fvgs),
                 f"{len(bullish_fvgs)} active bullish FVG(s)."),
            ])

            if not bullish_fvgs:
                return None

            fvg = bullish_fvgs[0]
            entry = float(fvg.bottom)
            stop = entry - safe_dist
            target = self._select_target(
                SignalDirection.BUY, entry,
                liquidity.get_buy_side_liquidity("h1"), safe_dist * 2,
            )

            return self._create_signal(
                SignalDirection.BUY, entry, stop, target,
                StrategyType.DISPLACEMENT_FVG, confirmations, "m5", "m5",
                setup_state={
                    "strategy": "displacement_fvg",
                    "direction": "BUY",
                    "h4_bias_at_creation": "bullish",
                    "fvg": {
                        "type": "bullish",
                        "top": float(fvg.top),
                        "bottom": float(fvg.bottom),
                    },
                    "displacement": {
                        "direction": "bullish",
                        "atr_ratio_at_creation": atr_ratio,
                    },
                },
            )

        if h4_bias == "bearish" and displacement_direction == "bearish":
            bearish_fvgs = fair_value_gaps.get_active_bearish_fvgs("m5")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                ("M5 bearish displacement", True,
                 f"Displacement ATR-ratio: {atr_ratio:.2f}."),
                ("M5 bearish FVG", bool(bearish_fvgs),
                 f"{len(bearish_fvgs)} active bearish FVG(s)."),
            ])

            if not bearish_fvgs:
                return None

            fvg = bearish_fvgs[0]
            entry = float(fvg.top)
            stop = entry + safe_dist
            target = self._select_target(
                SignalDirection.SELL, entry,
                liquidity.get_sell_side_liquidity("h1"), safe_dist * 2,
            )

            return self._create_signal(
                SignalDirection.SELL, entry, stop, target,
                StrategyType.DISPLACEMENT_FVG, confirmations, "m5", "m5",
                setup_state={
                    "strategy": "displacement_fvg",
                    "direction": "SELL",
                    "h4_bias_at_creation": "bearish",
                    "fvg": {
                        "type": "bearish",
                        "top": float(fvg.top),
                        "bottom": float(fvg.bottom),
                    },
                    "displacement": {
                        "direction": "bearish",
                        "atr_ratio_at_creation": atr_ratio,
                    },
                },
            )

        return None

    # ------------------------------------------------------------------
    # Strategy 6: Liquidity sweep + OB + FVG
    # ------------------------------------------------------------------

    def _check_liquidity_sweep_ob_fvg(
        self, h4_bias: str, h1_bias: str,
    ) -> Optional[TradeSignal]:

        if not liquidity.get_liquidity("m15"):
            return None

        if not order_blocks.get_obs("m15"):
            return None

        if not fair_value_gaps.get_fvgs("m5"):
            return None

        safe_dist = self._get_safe_stop_distance("m15")
        alignment_tolerance = safe_dist * 0.5

        if h4_bias == "bullish":
            swept = liquidity.was_sell_side_swept("m15")
            bullish_obs = order_blocks.get_active_bullish_obs("m15")
            bullish_fvgs = fair_value_gaps.get_active_bullish_fvgs("m5")

            confirmations = self._make_confirmations([
                ("H4 bullish bias", True, "H4 bias is bullish."),
                ("M15 sell-side sweep", swept,
                 "Sell-side liquidity swept." if swept else "No sell-side sweep."),
                ("M15 bullish OB", bool(bullish_obs),
                 f"{len(bullish_obs)} active bullish OB(s)."),
                ("M5 bullish FVG", bool(bullish_fvgs),
                 f"{len(bullish_fvgs)} active bullish FVG(s)."),
            ])

            if not swept or not bullish_obs or not bullish_fvgs:
                return None

            for ob in bullish_obs:
                for fvg in bullish_fvgs:
                    ob_bottom = float(ob.bottom)
                    fvg_bottom = float(fvg.bottom)

                    if abs(ob_bottom - fvg_bottom) <= alignment_tolerance:
                        entry = fvg_bottom
                        stop = min(ob_bottom, fvg_bottom) - safe_dist
                        target = self._select_target(
                            SignalDirection.BUY, entry,
                            liquidity.get_buy_side_liquidity("h1"),
                            safe_dist * 2.5,
                        )

                        signal = self._create_signal(
                            SignalDirection.BUY, entry, stop, target,
                            StrategyType.LIQUIDITY_SWEEP_OB_FVG,
                            confirmations, "m5", "m15",
                            setup_state={
                                "strategy": "liquidity_sweep_ob_fvg",
                                "direction": "BUY",
                                "h4_bias_at_creation": "bullish",
                                "ob": {
                                    "type": "bullish",
                                    "top": float(ob.top),
                                    "bottom": float(ob.bottom),
                                },
                                "fvg": {
                                    "type": "bullish",
                                    "top": float(fvg.top),
                                    "bottom": float(fvg.bottom),
                                },
                                "sweep": {
                                    "side": "sell_side",
                                    "price": self._find_sweep_price(
                                        "m15", "sell_side_sweep"
                                    ),
                                },
                            },
                        )
                        if signal:
                            return signal

        elif h4_bias == "bearish":
            swept = liquidity.was_buy_side_swept("m15")
            bearish_obs = order_blocks.get_active_bearish_obs("m15")
            bearish_fvgs = fair_value_gaps.get_active_bearish_fvgs("m5")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                ("M15 buy-side sweep", swept,
                 "Buy-side liquidity swept." if swept else "No buy-side sweep."),
                ("M15 bearish OB", bool(bearish_obs),
                 f"{len(bearish_obs)} active bearish OB(s)."),
                ("M5 bearish FVG", bool(bearish_fvgs),
                 f"{len(bearish_fvgs)} active bearish FVG(s)."),
            ])

            if not swept or not bearish_obs or not bearish_fvgs:
                return None

            for ob in bearish_obs:
                for fvg in bearish_fvgs:
                    ob_top = float(ob.top)
                    fvg_top = float(fvg.top)

                    if abs(ob_top - fvg_top) <= alignment_tolerance:
                        entry = fvg_top
                        stop = max(ob_top, fvg_top) + safe_dist
                        target = self._select_target(
                            SignalDirection.SELL, entry,
                            liquidity.get_sell_side_liquidity("h1"),
                            safe_dist * 2.5,
                        )

                        signal = self._create_signal(
                            SignalDirection.SELL, entry, stop, target,
                            StrategyType.LIQUIDITY_SWEEP_OB_FVG,
                            confirmations, "m5", "m15",
                            setup_state={
                                "strategy": "liquidity_sweep_ob_fvg",
                                "direction": "SELL",
                                "h4_bias_at_creation": "bearish",
                                "ob": {
                                    "type": "bearish",
                                    "top": float(ob.top),
                                    "bottom": float(ob.bottom),
                                },
                                "fvg": {
                                    "type": "bearish",
                                    "top": float(fvg.top),
                                    "bottom": float(fvg.bottom),
                                },
                                "sweep": {
                                    "side": "buy_side",
                                    "price": self._find_sweep_price(
                                        "m15", "buy_side_sweep"
                                    ),
                                },
                            },
                        )
                        if signal:
                            return signal

        return None

    # ------------------------------------------------------------------
    # H-03: TradePlan reconfirmation
    # ------------------------------------------------------------------

    def reconfirm_plan(self, plan) -> tuple[bool, str]:
        """Reconfirm the structured setup that produced a TradePlan.

        Requires TradePlan.setup_state. Plans from pre-H-03 code carry
        no setup_state and fail closed.

        For every strategy, ALL recheckable mandatory conditions are
        re-verified against current closed-candle state:
          - H4 bias (all strategies)
          - H1 bias (bos_order_block)
          - FVG identity + entry coverage (sweep_fvg, displacement_fvg,
            sweep_ob_fvg)
          - OB identity + entry coverage (bos_ob, choch_ob, sweep_ob_fvg)
          - CHoCH direction (sweep_choch, choch_ob)
          - BOS direction (bos_ob)
          - Displacement direction (displacement_fvg)
          - Original M15 liquidity level still present in the current
            analysis (sweep strategies)
          - Composite: BOTH OB AND FVG still valid (sweep_ob_fvg) --
            the pre-H-03 reconfirmer incorrectly used OR.

        Historical one-time triggers (sweep, displacement) are not
        re-fired, but their identity is verified against persisted
        state. BOS/CHoCH carry only a direction, not an event id;
        direction is rechecked and a residual limitation is documented.
        """
        direction = str(getattr(plan, "direction", "")).upper()
        if direction not in ("BUY", "SELL"):
            return False, f"Invalid plan direction: {direction!r}"

        strategy = str(getattr(plan, "strategy", "")).lower()

        try:
            entry = float(plan.entry_price)
            sl = float(plan.sl_price)
            tp = float(plan.tp_price)
        except (TypeError, ValueError):
            return False, "Plan prices are not numeric."

        if min(entry, sl, tp) <= 0:
            return False, "Plan prices are not positive."

        setup_state = getattr(plan, "setup_state", None)
        if not isinstance(setup_state, dict) or not setup_state:
            return False, (
                "No structured setup_state; cannot reconfirm "
                "(pre-H-03 plan or corrupted state)."
            )

        h4_bias = market_structure.get_higher_tf_bias()
        h1_bias = market_structure.get_structure_tf_bias()
        required_h4 = "bullish" if direction == "BUY" else "bearish"

        if h4_bias != required_h4:
            return (
                False,
                f"H4 bias is {h4_bias!r}; plan requires {required_h4!r}.",
            )

        # M-06: recheck structure-TF alignment for strategies that
        # require it. Pre-M-06 plans for these strategies now fail
        # here if alignment has been lost, which is the intended
        # fail-closed behaviour of D20(c).
        if self.strategy_alignment.get(strategy, False):
            if h1_bias != required_h4:
                return (
                    False,
                    f"Structure-TF alignment lost: {h1_bias!r} "
                    f"(required {required_h4!r}).",
                )

        if direction == "BUY" and sl >= entry:
            return False, "SL is no longer below entry."
        if direction == "SELL" and sl <= entry:
            return False, "SL is no longer above entry."

        checks = {
            "liquidity_sweep_fvg": self._reconfirm_sweep_fvg,
            "liquidity_sweep_choch": self._reconfirm_sweep_choch,
            "bos_order_block": self._reconfirm_bos_ob,
            "choch_order_block": self._reconfirm_choch_ob,
            "displacement_fvg": self._reconfirm_displacement_fvg,
            "liquidity_sweep_ob_fvg": self._reconfirm_sweep_ob_fvg,
        }

        checker = checks.get(strategy)
        if checker is None:
            return False, f"Unknown strategy: {strategy!r}"

        try:
            return checker(
                direction, entry, sl, tp, h4_bias, h1_bias, setup_state,
            )
        except Exception as exc:
            logger.exception(
                "Reconfirmation raised for strategy %s: %s", strategy, exc,
            )
            return False, f"Reconfirmation error: {exc}"

    # ---------- tolerances and matching helpers ----------

    def _zone_tolerance(self) -> float:
        """Price tolerance for matching persisted FVG/OB ranges.

        FVG/OB detection is deterministic given the same candles, so
        only broker re-quote noise needs absorbing. Two points is
        tight enough to reject a *different* FVG/OB at a nearby price.
        """
        return max(self._get_point_value() * 2.0, 1e-8)

    def _liquidity_tolerance(self) -> float:
        """Price tolerance for matching persisted liquidity levels.

        Matches the analyzer's own clustering tolerance (2 pips), so
        re-averaging of a level after the window shifts cannot cause
        a spurious mismatch.
        """
        try:
            info = mt5_connector.get_symbol_info()
        except Exception:
            info = None
        if info is not None:
            pip = MT5Connector.pip_size_for(info)
            if isinstance(pip, (int, float)) and pip > 0:
                return float(pip) * 2.0
        return max(self._get_point_value() * 20.0, 1e-8)

    @staticmethod
    def _find_matching_zone(
        zones, persisted_top, persisted_bottom, tolerance,
    ):
        """Return the first zone whose price range matches the persisted range."""
        try:
            persisted_top = float(persisted_top)
            persisted_bottom = float(persisted_bottom)
        except (TypeError, ValueError):
            return None
        if (
            persisted_top <= 0
            or persisted_bottom <= 0
            or persisted_top < persisted_bottom
        ):
            return None
        for zone in zones:
            try:
                top = float(zone.top)
                bottom = float(zone.bottom)
            except (AttributeError, TypeError, ValueError):
                continue
            if (
                abs(top - persisted_top) <= tolerance
                and abs(bottom - persisted_bottom) <= tolerance
            ):
                return zone
        return None

    @staticmethod
    def _find_matching_liquidity_level(
        levels, persisted_price, tolerance,
    ):
        """Return the first level whose price matches the persisted price."""
        if persisted_price is None:
            return None
        try:
            persisted_price = float(persisted_price)
        except (TypeError, ValueError):
            return None
        if persisted_price <= 0:
            return None
        for level in levels:
            try:
                price = float(level.price)
            except (AttributeError, TypeError, ValueError):
                continue
            if abs(price - persisted_price) <= tolerance:
                return level
        return None

    @staticmethod
    def _zone_covers_entry(zones, entry: float, tol: float = 1e-6) -> bool:
        return any(
            float(z.bottom) - tol <= entry <= float(z.top) + tol
            for z in zones
        )

    def _check_sweep_level_present(
        self, setup_state, timeframe: str = "m15",
    ) -> tuple[bool, str]:
        """Verify the persisted swept level still exists in the current
        M15 liquidity state.

        A sweep is a one-time event (the `swept` flag is transient and
        reflects only the current candle). What CAN be rechecked is
        whether the level itself is still visible to the analyzer -- if
        the underlying liquidity structure has been restructured so far
        that the level no longer exists, the setup thesis is invalid.
        Aging out of the 100-candle lookback (25h) cannot occur before
        the 4h default plan expiry fires, so this fail-closed behaviour
        is safe.
        """
        sweep = (setup_state or {}).get("sweep") or {}
        persisted_price = sweep.get("price")
        if persisted_price is None:
            # No price was captured at creation; cannot verify. Rely on
            # downstream conditions instead of failing here.
            return True, "sweep level not captured"

        side = sweep.get("side")
        state = liquidity.get_liquidity(timeframe)
        if state is None:
            return False, f"{timeframe.upper()} liquidity state unavailable."

        if side == "sell_side":
            levels = state.sell_side
        elif side == "buy_side":
            levels = state.buy_side
        else:
            return False, f"Unknown persisted sweep side: {side!r}"

        tol = self._liquidity_tolerance()
        if self._find_matching_liquidity_level(
            levels, persisted_price, tol,
        ) is None:
            return (
                False,
                "Original M15 liquidity level no longer present "
                f"(price={persisted_price}).",
            )
        return True, "OK"

    # ---------- per-strategy reconfirmers ----------

    def _reconfirm_sweep_fvg(
        self, direction, entry, sl, tp, h4, h1, setup_state,
    ) -> tuple[bool, str]:
        persisted_fvg = setup_state.get("fvg") or {}

        if direction == "BUY":
            fvgs = fair_value_gaps.get_active_bullish_fvgs("m5")
            expected_type = "bullish"
        else:
            fvgs = fair_value_gaps.get_active_bearish_fvgs("m5")
            expected_type = "bearish"

        if not fvgs:
            return False, "M5 FVG mitigated or removed."

        if persisted_fvg.get("type") not in (None, expected_type):
            return False, (
                f"Persisted FVG type {persisted_fvg.get('type')!r} "
                f"does not match plan direction {direction}."
            )

        tol = self._zone_tolerance()
        matched = self._find_matching_zone(
            fvgs,
            persisted_fvg.get("top"),
            persisted_fvg.get("bottom"),
            tol,
        )
        if matched is None:
            return False, "Original M5 FVG no longer present."

        if not self._zone_covers_entry([matched], entry):
            return False, "M5 FVG no longer covers the planned entry zone."

        ok, reason = self._check_sweep_level_present(setup_state, "m15")
        if not ok:
            return False, reason

        return True, "OK"

    def _reconfirm_sweep_choch(
        self, direction, entry, sl, tp, h4, h1, setup_state,
    ) -> tuple[bool, str]:
        if direction == "BUY":
            if not market_structure.is_choch_bullish("m15"):
                return False, "M15 bullish CHoCH no longer confirmed."
        else:
            if not market_structure.is_choch_bearish("m15"):
                return False, "M15 bearish CHoCH no longer confirmed."

        ok, reason = self._check_sweep_level_present(setup_state, "m15")
        if not ok:
            return False, reason

        return True, "OK"

    def _reconfirm_bos_ob(
        self, direction, entry, sl, tp, h4, h1, setup_state,
    ) -> tuple[bool, str]:
        required_h1 = "bullish" if direction == "BUY" else "bearish"
        if h1 != required_h1:
            return False, f"H1 bias flipped to {h1!r}."

        if direction == "BUY":
            if not market_structure.is_bos_bullish("h1"):
                return False, "H1 bullish BOS no longer confirmed."
            obs = order_blocks.get_active_bullish_obs("m15")
            expected_type = "bullish"
        else:
            if not market_structure.is_bos_bearish("h1"):
                return False, "H1 bearish BOS no longer confirmed."
            obs = order_blocks.get_active_bearish_obs("m15")
            expected_type = "bearish"

        if not obs:
            return False, "M15 OB mitigated or removed."

        persisted_ob = setup_state.get("ob") or {}
        if persisted_ob.get("type") not in (None, expected_type):
            return False, (
                f"Persisted OB type {persisted_ob.get('type')!r} "
                f"does not match plan direction {direction}."
            )

        tol = self._zone_tolerance()
        matched = self._find_matching_zone(
            obs,
            persisted_ob.get("top"),
            persisted_ob.get("bottom"),
            tol,
        )
        if matched is None:
            return False, "Original M15 OB no longer present."

        if not self._zone_covers_entry([matched], entry):
            return False, "M15 OB no longer covers the planned entry zone."

        return True, "OK"

    def _reconfirm_choch_ob(
        self, direction, entry, sl, tp, h4, h1, setup_state,
    ) -> tuple[bool, str]:
        if direction == "BUY":
            if not market_structure.is_choch_bullish("m15"):
                return False, "M15 bullish CHoCH no longer confirmed."
            obs = order_blocks.get_active_bullish_obs("m5")
            expected_type = "bullish"
        else:
            if not market_structure.is_choch_bearish("m15"):
                return False, "M15 bearish CHoCH no longer confirmed."
            obs = order_blocks.get_active_bearish_obs("m5")
            expected_type = "bearish"

        if not obs:
            return False, "M5 OB mitigated or removed."

        persisted_ob = setup_state.get("ob") or {}
        if persisted_ob.get("type") not in (None, expected_type):
            return False, (
                f"Persisted OB type {persisted_ob.get('type')!r} "
                f"does not match plan direction {direction}."
            )

        tol = self._zone_tolerance()
        matched = self._find_matching_zone(
            obs,
            persisted_ob.get("top"),
            persisted_ob.get("bottom"),
            tol,
        )
        if matched is None:
            return False, "Original M5 OB no longer present."

        if not self._zone_covers_entry([matched], entry):
            return False, "M5 OB no longer covers the planned entry zone."

        return True, "OK"

    def _reconfirm_displacement_fvg(
        self, direction, entry, sl, tp, h4, h1, setup_state,
    ) -> tuple[bool, str]:
        from src.indicators import detect_displacement

        persisted_fvg = setup_state.get("fvg") or {}
        persisted_disp = setup_state.get("displacement") or {}
        expected_direction = (
            "bullish" if direction == "BUY" else "bearish"
        )

        candles = market_data.get_closed_candles("m5", 30)
        if not candles:
            return False, "Insufficient M5 candles for displacement check."

        required = self.displacement_atr_period + 10 + 2
        if len(candles) < required:
            return False, (
                f"Insufficient M5 candles for displacement check: "
                f"{len(candles)} < {required}."
            )

        opens = np.asarray([c.open for c in candles], dtype=float)
        highs = np.asarray([c.high for c in candles], dtype=float)
        lows = np.asarray([c.low for c in candles], dtype=float)
        closes = np.asarray([c.close for c in candles], dtype=float)

        if not all(
            np.all(np.isfinite(arr))
            for arr in (opens, highs, lows, closes)
        ):
            return False, "Non-finite M5 candle data for displacement check."

        displacements = detect_displacement(
            opens, highs, lows, closes,
            lookback=10,
            atr_period=self.displacement_atr_period,
            atr_multiplier=self.displacement_atr_multiplier,
            min_body_pips=self.displacement_min_body_pips,
            point_value=self._get_pip_size(),
        )
        if not displacements:
            return False, "Displacement no longer detected in recent window."

        latest = displacements[-1]
        if latest.get("direction") != expected_direction:
            return False, (
                f"Latest M5 displacement is "
                f"{latest.get('direction')!r}; plan requires "
                f"{expected_direction!r}."
            )

        if direction == "BUY":
            fvgs = fair_value_gaps.get_active_bullish_fvgs("m5")
        else:
            fvgs = fair_value_gaps.get_active_bearish_fvgs("m5")

        if not fvgs:
            return False, "M5 FVG mitigated or removed."

        if persisted_fvg.get("type") not in (None, expected_direction):
            return False, (
                f"Persisted FVG type {persisted_fvg.get('type')!r} "
                f"does not match plan direction {direction}."
            )

        tol = self._zone_tolerance()
        matched = self._find_matching_zone(
            fvgs,
            persisted_fvg.get("top"),
            persisted_fvg.get("bottom"),
            tol,
        )
        if matched is None:
            return False, "Original M5 FVG no longer present."

        if not self._zone_covers_entry([matched], entry):
            return False, "M5 FVG no longer covers the planned entry zone."

        if persisted_disp.get("direction") not in (
            None, expected_direction
        ):
            return False, (
                "Persisted displacement direction is inconsistent "
                "with plan direction."
            )

        return True, "OK"       

    def _reconfirm_sweep_ob_fvg(
        self, direction, entry, sl, tp, h4, h1, setup_state,
    ) -> tuple[bool, str]:
        # H-03: the pre-H-03 reconfirmer used OR between OB and FVG.
        # The original strategy required BOTH. This method requires
        # BOTH, matching the originating setup.

        persisted_ob = setup_state.get("ob") or {}
        persisted_fvg = setup_state.get("fvg") or {}

        if direction == "BUY":
            obs = order_blocks.get_active_bullish_obs("m15")
            fvgs = fair_value_gaps.get_active_bullish_fvgs("m5")
            expected_type = "bullish"
        else:
            obs = order_blocks.get_active_bearish_obs("m15")
            fvgs = fair_value_gaps.get_active_bearish_fvgs("m5")
            expected_type = "bearish"

        if not obs:
            return False, "M15 OB mitigated or removed."
        if not fvgs:
            return False, "M5 FVG mitigated or removed."

        tol = self._zone_tolerance()

        matched_ob = self._find_matching_zone(
            obs,
            persisted_ob.get("top"),
            persisted_ob.get("bottom"),
            tol,
        )
        if matched_ob is None:
            return False, "Original M15 OB no longer present."

        matched_fvg = self._find_matching_zone(
            fvgs,
            persisted_fvg.get("top"),
            persisted_fvg.get("bottom"),
            tol,
        )
        if matched_fvg is None:
            return False, "Original M5 FVG no longer present."

        if not self._zone_covers_entry([matched_ob], entry):
            return False, "M15 OB no longer covers the planned entry zone."
        if not self._zone_covers_entry([matched_fvg], entry):
            return False, "M5 FVG no longer covers the planned entry zone."

        ok, reason = self._check_sweep_level_present(setup_state, "m15")
        if not ok:
            return False, reason

        return True, "OK"


# Shared engine instance used by the trading agent.
smc_engine = SMCEngine()