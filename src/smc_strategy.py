"""SMC strategy engine.

Detects multi-timeframe SMC setups and creates validated trade candidates.
Actual entry execution and waiting for price to reach an entry zone belong
to the order-execution layer.
"""

from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

import numpy as np

from src.market_structure import market_structure
from src.liquidity import liquidity
from src.order_blocks import order_blocks
from src.fair_value_gaps import fair_value_gaps
from src.premium_discount import premium_discount
from src.market_data import market_data
from src.config_loader import config
from src.mt5_connector import mt5_connector
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


class SMCEngine:
    """Multi-timeframe SMC setup detection engine."""

    def __init__(self):
        self.strategies = list(config.smc.strategies)
        self.min_risk_reward = float(config.trading.min_risk_reward)
        # Default fallback point size
        self.point_value = 0.01 

    def _get_point_value(self) -> float:
        """Fetch the live point value from MT5 broker specs."""
        symbol_info = mt5_connector.get_symbol_info()
        if symbol_info and symbol_info.point > 0:
            self.point_value = float(symbol_info.point)
        return self.point_value

    def _get_safe_stop_distance(self, tf: str = "m15") -> float:
        """Calculate a volatility-based stop distance using ATR.
        
        Uses 1.5 * ATR(14) on the setup timeframe, with a hard floor
        to prevent the 10-point stop-loss disaster.
        """
        atr = market_data.calculate_atr(tf, 14)
        point = self._get_point_value()
        
        if atr is None or atr <= 0:
            # Fallback if ATR fails: 150 points ($1.50 on XM Gold)
            return 150 * point
            
        # Ensure stop is at least $1.00 (100 points) to clear spread
        safe_distance = max(atr * 1.5, 100 * point)
        return safe_distance

    # ------------------------------------------------------------------
    # Public analysis
    # ------------------------------------------------------------------

    def analyze_all_strategies(self) -> List[TradeSignal]:
        """Refresh analyzers and return all valid trade candidates."""

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
                    len(signals),
                    h4_bias,
                    h1_bias,
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
        """Close of the last CONFIRMED candle.

        Fixed: previously called ``market_data.get_latest_candle()``,
        which does not exist on MarketDataEngine. The method is
        currently unused by the six strategies (which read entry price
        from the FVG/OB zone, not from a candle close), but leaving a
        broken method is a landmine for future edits.
        """
        candle = market_data.get_latest_closed_candle(timeframe)
        if candle is None:
            return None

        value = float(candle.close)
        return value if np.isfinite(value) and value > 0 else None

    def _latest_closed_close(self, timeframe: str) -> Optional[float]:
        """Return the close of the last CONFIRMED (closed) candle."""
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
        """Select the nearest valid liquidity target in the trade direction."""

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
    ) -> Optional[TradeSignal]:
        """Validate candidate prices and create a TradeSignal."""

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
                strategy.value,
                direction.value,
                entry,
                stop,
                target,
            )
            return None

        risk_reward = reward / risk

        if risk_reward < self.min_risk_reward:
            logger.debug(
                "%s rejected: RR %.2f is below configured minimum %.2f.",
                strategy.value,
                risk_reward,
                self.min_risk_reward,
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
            structure_tf="h1",
            bias_tf="h4",
        )

    # ------------------------------------------------------------------
    # Strategy 1: Liquidity sweep + FVG
    # ------------------------------------------------------------------

    def _check_liquidity_sweep_fvg(
        self,
        h4_bias: str,
        h1_bias: str,
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
                (
                    "M15 sell-side sweep",
                    swept,
                    "Sell-side liquidity swept."
                    if swept
                    else "No sell-side sweep.",
                ),
                (
                    "M5 bullish FVG",
                    bool(fvgs),
                    f"{len(fvgs)} active bullish FVG(s).",
                ),
            ])

            if not swept or not fvgs:
                return None

            fvg = fvgs[0]
            entry = float(fvg.bottom)
            stop = entry - safe_dist
            target = self._select_target(
                SignalDirection.BUY,
                entry,
                liquidity.get_buy_side_liquidity("h1"),
                safe_dist * 2,
            )

            return self._create_signal(
                SignalDirection.BUY,
                entry,
                stop,
                target,
                StrategyType.LIQUIDITY_SWEEP_FVG,
                confirmations,
                "m5",
                "m15",
            )

        if h4_bias == "bearish":
            swept = liquidity.was_buy_side_swept("m15")
            fvgs = fair_value_gaps.get_active_bearish_fvgs("m5")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                (
                    "M15 buy-side sweep",
                    swept,
                    "Buy-side liquidity swept."
                    if swept
                    else "No buy-side sweep.",
                ),
                (
                    "M5 bearish FVG",
                    bool(fvgs),
                    f"{len(fvgs)} active bearish FVG(s).",
                ),
            ])

            if not swept or not fvgs:
                return None

            fvg = fvgs[0]
            entry = float(fvg.top)
            stop = entry + safe_dist
            target = self._select_target(
                SignalDirection.SELL,
                entry,
                liquidity.get_sell_side_liquidity("h1"),
                safe_dist * 2,
            )

            return self._create_signal(
                SignalDirection.SELL,
                entry,
                stop,
                target,
                StrategyType.LIQUIDITY_SWEEP_FVG,
                confirmations,
                "m5",
                "m15",
            )

        return None

    # ------------------------------------------------------------------
    # Strategy 2: Liquidity sweep + CHoCH
    # ------------------------------------------------------------------

    def _check_liquidity_sweep_choch(
        self,
        h4_bias: str,
        h1_bias: str,
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
                ("M15 sell-side sweep", swept, "Sell-side liquidity swept." if swept else "No sell-side sweep."),
                ("M15 bullish CHoCH", choch, "Bullish CHoCH confirmed." if choch else "No bullish CHoCH."),
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

            target = self._select_target(SignalDirection.BUY, entry, liquidity.get_buy_side_liquidity("h1"), safe_dist * 2.5)

            return self._create_signal(
                SignalDirection.BUY, entry, stop, target, StrategyType.LIQUIDITY_SWEEP_CHOCH, confirmations, "m5", "m15",
            )

        if h4_bias == "bearish":
            swept = liquidity.was_buy_side_swept("m15")
            choch = market_structure.is_choch_bearish("m15")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                ("M15 buy-side sweep", swept, "Buy-side liquidity swept." if swept else "No buy-side sweep."),
                ("M15 bearish CHoCH", choch, "Bearish CHoCH confirmed." if choch else "No bearish CHoCH."),
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

            target = self._select_target(SignalDirection.SELL, entry, liquidity.get_sell_side_liquidity("h1"), safe_dist * 2.5)

            return self._create_signal(
                SignalDirection.SELL, entry, stop, target, StrategyType.LIQUIDITY_SWEEP_CHOCH, confirmations, "m5", "m15",
            )

        return None

    # ------------------------------------------------------------------
    # Strategy 3: BOS + Order Block
    # ------------------------------------------------------------------

    def _check_bos_order_block(
        self,
        h4_bias: str,
        h1_bias: str,
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
                ("H1 bullish BOS", bos, "Bullish BOS confirmed." if bos else "No bullish BOS."),
                ("M15 bullish OB", bool(bullish_obs), f"{len(bullish_obs)} active bullish OB(s)."),
            ])

            if not bos or not bullish_obs:
                return None

            ob = bullish_obs[0]
            entry = float(ob.bottom)
            stop = entry - safe_dist
            target = self._select_target(SignalDirection.BUY, entry, liquidity.get_buy_side_liquidity("h1"), safe_dist * 2.5)

            return self._create_signal(
                SignalDirection.BUY, entry, stop, target, StrategyType.BOS_ORDER_BLOCK, confirmations, "m15", "h1",
            )

        if h4_bias == "bearish" and h1_bias == "bearish":
            bos = market_structure.is_bos_bearish("h1")
            bearish_obs = order_blocks.get_active_bearish_obs("m15")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                ("H1 bearish bias", True, "H1 bias is bearish."),
                ("H1 bearish BOS", bos, "Bearish BOS confirmed." if bos else "No bearish BOS."),
                ("M15 bearish OB", bool(bearish_obs), f"{len(bearish_obs)} active bearish OB(s)."),
            ])

            if not bos or not bearish_obs:
                return None

            ob = bearish_obs[0]
            entry = float(ob.top)
            stop = entry + safe_dist
            target = self._select_target(SignalDirection.SELL, entry, liquidity.get_sell_side_liquidity("h1"), safe_dist * 2.5)

            return self._create_signal(
                SignalDirection.SELL, entry, stop, target, StrategyType.BOS_ORDER_BLOCK, confirmations, "m15", "h1",
            )

        return None

    # ------------------------------------------------------------------
    # Strategy 4: CHoCH + Order Block
    # ------------------------------------------------------------------

    def _check_choch_order_block(
        self,
        h4_bias: str,
        h1_bias: str,
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
                ("M15 bullish CHoCH", choch, "Bullish CHoCH confirmed." if choch else "No bullish CHoCH."),
                ("M5 bullish OB", bool(bullish_obs), f"{len(bullish_obs)} active bullish OB(s)."),
            ])

            if not choch or not bullish_obs:
                return None

            ob = bullish_obs[0]
            entry = float(ob.bottom)
            stop = entry - safe_dist
            target = self._select_target(SignalDirection.BUY, entry, liquidity.get_buy_side_liquidity("m15"), safe_dist * 2)

            return self._create_signal(
                SignalDirection.BUY, entry, stop, target, StrategyType.CHOCH_ORDER_BLOCK, confirmations, "m5", "m15",
            )

        if h4_bias == "bearish":
            choch = market_structure.is_choch_bearish("m15")
            bearish_obs = order_blocks.get_active_bearish_obs("m5")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                ("M15 bearish CHoCH", choch, "Bearish CHoCH confirmed." if choch else "No bearish CHoCH."),
                ("M5 bearish OB", bool(bearish_obs), f"{len(bearish_obs)} active bearish OB(s)."),
            ])

            if not choch or not bearish_obs:
                return None

            ob = bearish_obs[0]
            entry = float(ob.top)
            stop = entry + safe_dist
            target = self._select_target(SignalDirection.SELL, entry, liquidity.get_sell_side_liquidity("m15"), safe_dist * 2)

            return self._create_signal(
                SignalDirection.SELL, entry, stop, target, StrategyType.CHOCH_ORDER_BLOCK, confirmations, "m5", "m15",
            )

        return None

    # ------------------------------------------------------------------
    # Strategy 5: Displacement + FVG
    # ------------------------------------------------------------------

    def _check_displacement_fvg(
        self,
        h4_bias: str,
        h1_bias: str,
    ) -> Optional[TradeSignal]:

        from src.indicators import detect_displacement

        # Use CLOSED candles to prevent repainting!
        candles = market_data.get_closed_candles("m5", 30)
        if not candles or len(candles) < 20:
            return None

        opens = np.asarray([c.open for c in candles], dtype=float)
        closes = np.asarray([c.close for c in candles], dtype=float)

        if not np.all(np.isfinite(opens)) or not np.all(np.isfinite(closes)):
            return None

        displacements = detect_displacement(
            opens, closes, lookback=10, threshold=1.5
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
        ratio = float(recent.get("ratio", 0.0))
        safe_dist = self._get_safe_stop_distance("m5")

        if h4_bias == "bullish" and displacement_direction == "bullish":
            bullish_fvgs = fair_value_gaps.get_active_bullish_fvgs("m5")

            confirmations = self._make_confirmations([
                ("H4 bullish bias", True, "H4 bias is bullish."),
                ("M5 bullish displacement", True, f"Displacement ratio: {ratio:.2f}."),
                ("M5 bullish FVG", bool(bullish_fvgs), f"{len(bullish_fvgs)} active bullish FVG(s)."),
            ])

            if not bullish_fvgs:
                return None

            fvg = bullish_fvgs[0]
            entry = float(fvg.bottom)
            stop = entry - safe_dist
            target = self._select_target(SignalDirection.BUY, entry, liquidity.get_buy_side_liquidity("h1"), safe_dist * 2)

            return self._create_signal(
                SignalDirection.BUY, entry, stop, target, StrategyType.DISPLACEMENT_FVG, confirmations, "m5", "m5",
            )

        if h4_bias == "bearish" and displacement_direction == "bearish":
            bearish_fvgs = fair_value_gaps.get_active_bearish_fvgs("m5")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                ("M5 bearish displacement", True, f"Displacement ratio: {ratio:.2f}."),
                ("M5 bearish FVG", bool(bearish_fvgs), f"{len(bearish_fvgs)} active bearish FVG(s)."),
            ])

            if not bearish_fvgs:
                return None

            fvg = bearish_fvgs[0]
            entry = float(fvg.top)
            stop = entry + safe_dist
            target = self._select_target(SignalDirection.SELL, entry, liquidity.get_sell_side_liquidity("h1"), safe_dist * 2)

            return self._create_signal(
                SignalDirection.SELL, entry, stop, target, StrategyType.DISPLACEMENT_FVG, confirmations, "m5", "m5",
            )

        return None

    # ------------------------------------------------------------------
    # Strategy 6: Liquidity sweep + OB + FVG
    # ------------------------------------------------------------------

    def _check_liquidity_sweep_ob_fvg(
        self,
        h4_bias: str,
        h1_bias: str,
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
                ("M15 sell-side sweep", swept, "Sell-side liquidity swept." if swept else "No sell-side sweep."),
                ("M15 bullish OB", bool(bullish_obs), f"{len(bullish_obs)} active bullish OB(s)."),
                ("M5 bullish FVG", bool(bullish_fvgs), f"{len(bullish_fvgs)} active bullish FVG(s)."),
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
                        target = self._select_target(SignalDirection.BUY, entry, liquidity.get_buy_side_liquidity("h1"), safe_dist * 2.5)

                        signal = self._create_signal(
                            SignalDirection.BUY, entry, stop, target, StrategyType.LIQUIDITY_SWEEP_OB_FVG, confirmations, "m5", "m15",
                        )
                        if signal:
                            return signal

        elif h4_bias == "bearish":
            swept = liquidity.was_buy_side_swept("m15")
            bearish_obs = order_blocks.get_active_bearish_obs("m15")
            bearish_fvgs = fair_value_gaps.get_active_bearish_fvgs("m5")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                ("M15 buy-side sweep", swept, "Buy-side liquidity swept." if swept else "No buy-side sweep."),
                ("M15 bearish OB", bool(bearish_obs), f"{len(bearish_obs)} active bearish OB(s)."),
                ("M5 bearish FVG", bool(bearish_fvgs), f"{len(bearish_fvgs)} active bearish FVG(s)."),
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
                        target = self._select_target(SignalDirection.SELL, entry, liquidity.get_sell_side_liquidity("h1"), safe_dist * 2.5)

                        signal = self._create_signal(
                            SignalDirection.SELL, entry, stop, target, StrategyType.LIQUIDITY_SWEEP_OB_FVG, confirmations, "m5", "m15",
                        )
                        if signal:
                            return signal

        return None

    # ------------------------------------------------------------------
    # H-01: TradePlan reconfirmation
    # ------------------------------------------------------------------

    def reconfirm_plan(self, plan) -> tuple[bool, str]:
        """Reconfirm that the SMC setup underlying a TradePlan is still valid.

        Called by ``TradingAgent._process_trade_plans`` immediately before
        the submission gate, using the freshly-cached closed-candle state.
        Returns ``(valid, reason)``. Any failure of the strategy-specific
        checks invalidates the plan — this method never fabricates a
        confirmation.
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

        # Common precondition: H4 bias must still match plan direction.
        h4_bias = market_structure.get_higher_tf_bias()
        h1_bias = market_structure.get_structure_tf_bias()
        required_h4 = "bullish" if direction == "BUY" else "bearish"

        if h4_bias != required_h4:
            return (
                False,
                f"H4 bias is {h4_bias!r}; plan requires {required_h4!r}.",
            )

        # Common precondition: SL still on the correct side of entry.
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
                direction, entry, sl, tp, h4_bias, h1_bias,
            )
        except Exception as exc:
            logger.exception(
                "Reconfirmation raised for strategy %s: %s", strategy, exc,
            )
            return False, f"Reconfirmation error: {exc}"

    # ---------- per-strategy reconfirmation checkers ----------
    # Each checker re-validates the specific confirmations that produced
    # the original plan, using the freshly-cached closed-candle state.

    @staticmethod
    def _zone_covers_entry(zones, entry: float, tol: float = 1e-6) -> bool:
        return any(
            float(z.bottom) - tol <= entry <= float(z.top) + tol
            for z in zones
        )

    def _reconfirm_sweep_fvg(
        self, direction, entry, sl, tp, h4, h1,
    ) -> tuple[bool, str]:
        if direction == "BUY":
            fvgs = fair_value_gaps.get_active_bullish_fvgs("m5")
        else:
            fvgs = fair_value_gaps.get_active_bearish_fvgs("m5")

        if not fvgs:
            return False, "M5 FVG mitigated or removed."

        if not self._zone_covers_entry(fvgs, entry):
            return False, "M5 FVG no longer covers the planned entry zone."

        return True, "OK"

    def _reconfirm_sweep_choch(
        self, direction, entry, sl, tp, h4, h1,
    ) -> tuple[bool, str]:
        if direction == "BUY":
            if not market_structure.is_choch_bullish("m15"):
                return False, "M15 bullish CHoCH no longer confirmed."
        else:
            if not market_structure.is_choch_bearish("m15"):
                return False, "M15 bearish CHoCH no longer confirmed."

        return True, "OK"

    def _reconfirm_bos_ob(
        self, direction, entry, sl, tp, h4, h1,
    ) -> tuple[bool, str]:
        required_h1 = "bullish" if direction == "BUY" else "bearish"
        if h1 != required_h1:
            return False, f"H1 bias flipped to {h1!r}."

        if direction == "BUY":
            if not market_structure.is_bos_bullish("h1"):
                return False, "H1 bullish BOS no longer confirmed."
            obs = order_blocks.get_active_bullish_obs("m15")
        else:
            if not market_structure.is_bos_bearish("h1"):
                return False, "H1 bearish BOS no longer confirmed."
            obs = order_blocks.get_active_bearish_obs("m15")

        if not obs:
            return False, "M15 OB mitigated or removed."

        if not self._zone_covers_entry(obs, entry):
            return False, "M15 OB no longer covers the planned entry zone."

        return True, "OK"

    def _reconfirm_choch_ob(
        self, direction, entry, sl, tp, h4, h1,
    ) -> tuple[bool, str]:
        if direction == "BUY":
            if not market_structure.is_choch_bullish("m15"):
                return False, "M15 bullish CHoCH no longer confirmed."
            obs = order_blocks.get_active_bullish_obs("m5")
        else:
            if not market_structure.is_choch_bearish("m15"):
                return False, "M15 bearish CHoCH no longer confirmed."
            obs = order_blocks.get_active_bearish_obs("m5")

        if not obs:
            return False, "M5 OB mitigated or removed."

        if not self._zone_covers_entry(obs, entry):
            return False, "M5 OB no longer covers the planned entry zone."

        return True, "OK"

    def _reconfirm_displacement_fvg(
        self, direction, entry, sl, tp, h4, h1,
    ) -> tuple[bool, str]:
        if direction == "BUY":
            fvgs = fair_value_gaps.get_active_bullish_fvgs("m5")
        else:
            fvgs = fair_value_gaps.get_active_bearish_fvgs("m5")

        if not fvgs:
            return False, "M5 FVG mitigated or removed."

        if not self._zone_covers_entry(fvgs, entry):
            return False, "M5 FVG no longer covers the planned entry zone."

        return True, "OK"

    def _reconfirm_sweep_ob_fvg(
        self, direction, entry, sl, tp, h4, h1,
    ) -> tuple[bool, str]:
        if direction == "BUY":
            obs = order_blocks.get_active_bullish_obs("m15")
            fvgs = fair_value_gaps.get_active_bullish_fvgs("m5")
        else:
            obs = order_blocks.get_active_bearish_obs("m15")
            fvgs = fair_value_gaps.get_active_bearish_fvgs("m5")

        if not obs:
            return False, "M15 OB mitigated or removed."
        if not fvgs:
            return False, "M5 FVG mitigated or removed."

        if not (
            self._zone_covers_entry(obs, entry)
            or self._zone_covers_entry(fvgs, entry)
        ):
            return False, (
                "Entry zone no longer covered by an active OB or FVG."
            )

        return True, "OK"


# Shared engine instance used by the trading agent.
smc_engine = SMCEngine()