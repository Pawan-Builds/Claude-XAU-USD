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


class SMCEngine:
    """Multi-timeframe SMC setup detection engine."""

    def __init__(self):
        self.strategies = list(config.smc.strategies)
        self.min_risk_reward = float(config.trading.min_risk_reward)
        self.point_value = 0.01

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
        candle = market_data.get_latest_candle(timeframe)
        if candle is None:
            return None

        value = float(candle.close)
        return value if np.isfinite(value) and value > 0 else None

    def _latest_closed_close(self, timeframe: str) -> Optional[float]:
        """Return the close of the last CONFIRMED (closed) candle.

        Using the forming candle's close causes repainting: the signal
        would be calculated against a bar that continues to change
        until it closes.
        """
        candles = market_data.get_candles(timeframe, 2)
        if not candles or len(candles) < 2:
            return None

        value = float(candles[-2].close)
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
            stop = entry - 10 * self.point_value
            target = self._select_target(
                SignalDirection.BUY,
                entry,
                liquidity.get_buy_side_liquidity("h1"),   # FIX: buy-side above
                50 * self.point_value,
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
            stop = entry + 10 * self.point_value
            target = self._select_target(
                SignalDirection.SELL,
                entry,
                liquidity.get_sell_side_liquidity("h1"),  # FIX: sell-side below
                50 * self.point_value,
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

        if h4_bias == "bullish":
            swept = liquidity.was_sell_side_swept("m15")
            choch = market_structure.is_choch_bullish("m15")

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
                    "M15 bullish CHoCH",
                    choch,
                    "Bullish CHoCH confirmed."
                    if choch
                    else "No bullish CHoCH.",
                ),
            ])

            if not swept or not choch:
                return None

            # Use the last CLOSED m5 candle — not the forming one.
            entry = self._latest_closed_close("m5")
            if entry is None:
                return None

            recent_low = self._get_level(levels, "recent_low")
            if recent_low is None or recent_low >= entry:
                return None

            stop = recent_low - 5 * self.point_value
            target = self._select_target(
                SignalDirection.BUY,
                entry,
                liquidity.get_buy_side_liquidity("h1"),   # FIX
                60 * self.point_value,
            )

            return self._create_signal(
                SignalDirection.BUY,
                entry,
                stop,
                target,
                StrategyType.LIQUIDITY_SWEEP_CHOCH,
                confirmations,
                "m5",
                "m15",
            )

        if h4_bias == "bearish":
            swept = liquidity.was_buy_side_swept("m15")
            choch = market_structure.is_choch_bearish("m15")

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
                    "M15 bearish CHoCH",
                    choch,
                    "Bearish CHoCH confirmed."
                    if choch
                    else "No bearish CHoCH.",
                ),
            ])

            if not swept or not choch:
                return None

            entry = self._latest_closed_close("m5")
            if entry is None:
                return None

            recent_high = self._get_level(levels, "recent_high")
            if recent_high is None or recent_high <= entry:
                return None

            stop = recent_high + 5 * self.point_value
            target = self._select_target(
                SignalDirection.SELL,
                entry,
                liquidity.get_sell_side_liquidity("h1"),  # FIX
                60 * self.point_value,
            )

            return self._create_signal(
                SignalDirection.SELL,
                entry,
                stop,
                target,
                StrategyType.LIQUIDITY_SWEEP_CHOCH,
                confirmations,
                "m5",
                "m15",
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

        if h4_bias == "bullish" and h1_bias == "bullish":
            bos = market_structure.is_bos_bullish("h1")
            bullish_obs = order_blocks.get_active_bullish_obs("m15")

            confirmations = self._make_confirmations([
                ("H4 bullish bias", True, "H4 bias is bullish."),
                ("H1 bullish bias", True, "H1 bias is bullish."),
                (
                    "H1 bullish BOS",
                    bos,
                    "Bullish BOS confirmed." if bos else "No bullish BOS.",
                ),
                (
                    "M15 bullish OB",
                    bool(bullish_obs),
                    f"{len(bullish_obs)} active bullish OB(s).",
                ),
            ])

            if not bos or not bullish_obs:
                return None

            ob = bullish_obs[0]
            entry = float(ob.bottom)
            stop = entry - 15 * self.point_value
            target = self._select_target(
                SignalDirection.BUY,
                entry,
                liquidity.get_buy_side_liquidity("h1"),   # FIX
                60 * self.point_value,
            )

            return self._create_signal(
                SignalDirection.BUY,
                entry,
                stop,
                target,
                StrategyType.BOS_ORDER_BLOCK,
                confirmations,
                "m15",
                "h1",
            )

        if h4_bias == "bearish" and h1_bias == "bearish":
            bos = market_structure.is_bos_bearish("h1")
            bearish_obs = order_blocks.get_active_bearish_obs("m15")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                ("H1 bearish bias", True, "H1 bias is bearish."),
                (
                    "H1 bearish BOS",
                    bos,
                    "Bearish BOS confirmed." if bos else "No bearish BOS.",
                ),
                (
                    "M15 bearish OB",
                    bool(bearish_obs),
                    f"{len(bearish_obs)} active bearish OB(s).",
                ),
            ])

            if not bos or not bearish_obs:
                return None

            ob = bearish_obs[0]
            entry = float(ob.top)
            stop = entry + 15 * self.point_value
            target = self._select_target(
                SignalDirection.SELL,
                entry,
                liquidity.get_sell_side_liquidity("h1"),  # FIX
                60 * self.point_value,
            )

            return self._create_signal(
                SignalDirection.SELL,
                entry,
                stop,
                target,
                StrategyType.BOS_ORDER_BLOCK,
                confirmations,
                "m15",
                "h1",
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

        if h4_bias == "bullish":
            choch = market_structure.is_choch_bullish("m15")
            bullish_obs = order_blocks.get_active_bullish_obs("m5")

            confirmations = self._make_confirmations([
                ("H4 bullish bias", True, "H4 bias is bullish."),
                (
                    "M15 bullish CHoCH",
                    choch,
                    "Bullish CHoCH confirmed."
                    if choch
                    else "No bullish CHoCH.",
                ),
                (
                    "M5 bullish OB",
                    bool(bullish_obs),
                    f"{len(bullish_obs)} active bullish OB(s).",
                ),
            ])

            if not choch or not bullish_obs:
                return None

            ob = bullish_obs[0]
            entry = float(ob.bottom)
            stop = entry - 10 * self.point_value
            target = self._select_target(
                SignalDirection.BUY,
                entry,
                liquidity.get_buy_side_liquidity("m15"),  # FIX
                40 * self.point_value,
            )

            return self._create_signal(
                SignalDirection.BUY,
                entry,
                stop,
                target,
                StrategyType.CHOCH_ORDER_BLOCK,
                confirmations,
                "m5",
                "m15",
            )

        if h4_bias == "bearish":
            choch = market_structure.is_choch_bearish("m15")
            bearish_obs = order_blocks.get_active_bearish_obs("m5")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                (
                    "M15 bearish CHoCH",
                    choch,
                    "Bearish CHoCH confirmed."
                    if choch
                    else "No bearish CHoCH.",
                ),
                (
                    "M5 bearish OB",
                    bool(bearish_obs),
                    f"{len(bearish_obs)} active bearish OB(s).",
                ),
            ])

            if not choch or not bearish_obs:
                return None

            ob = bearish_obs[0]
            entry = float(ob.top)
            stop = entry + 10 * self.point_value
            target = self._select_target(
                SignalDirection.SELL,
                entry,
                liquidity.get_sell_side_liquidity("m15"), # FIX
                40 * self.point_value,
            )

            return self._create_signal(
                SignalDirection.SELL,
                entry,
                stop,
                target,
                StrategyType.CHOCH_ORDER_BLOCK,
                confirmations,
                "m5",
                "m15",
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

        candles = market_data.get_candles("m5", 30)
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

        if h4_bias == "bullish" and displacement_direction == "bullish":
            bullish_fvgs = fair_value_gaps.get_active_bullish_fvgs("m5")

            confirmations = self._make_confirmations([
                ("H4 bullish bias", True, "H4 bias is bullish."),
                (
                    "M5 bullish displacement",
                    True,
                    f"Displacement ratio: {ratio:.2f}.",
                ),
                (
                    "M5 bullish FVG",
                    bool(bullish_fvgs),
                    f"{len(bullish_fvgs)} active bullish FVG(s).",
                ),
            ])

            if not bullish_fvgs:
                return None

            fvg = bullish_fvgs[0]
            entry = float(fvg.bottom)
            stop = entry - 10 * self.point_value
            target = self._select_target(
                SignalDirection.BUY,
                entry,
                liquidity.get_buy_side_liquidity("h1"),   # FIX
                30 * self.point_value,
            )

            return self._create_signal(
                SignalDirection.BUY,
                entry,
                stop,
                target,
                StrategyType.DISPLACEMENT_FVG,
                confirmations,
                "m5",
                "m5",
            )

        if h4_bias == "bearish" and displacement_direction == "bearish":
            bearish_fvgs = fair_value_gaps.get_active_bearish_fvgs("m5")

            confirmations = self._make_confirmations([
                ("H4 bearish bias", True, "H4 bias is bearish."),
                (
                    "M5 bearish displacement",
                    True,
                    f"Displacement ratio: {ratio:.2f}.",
                ),
                (
                    "M5 bearish FVG",
                    bool(bearish_fvgs),
                    f"{len(bearish_fvgs)} active bearish FVG(s).",
                ),
            ])

            if not bearish_fvgs:
                return None

            fvg = bearish_fvgs[0]
            entry = float(fvg.top)
            stop = entry + 10 * self.point_value
            target = self._select_target(
                SignalDirection.SELL,
                entry,
                liquidity.get_sell_side_liquidity("h1"),  # FIX
                30 * self.point_value,
            )

            return self._create_signal(
                SignalDirection.SELL,
                entry,
                stop,
                target,
                StrategyType.DISPLACEMENT_FVG,
                confirmations,
                "m5",
                "m5",
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

        alignment_tolerance = 20 * self.point_value

        if h4_bias == "bullish":
            swept = liquidity.was_sell_side_swept("m15")
            bullish_obs = order_blocks.get_active_bullish_obs("m15")
            bullish_fvgs = fair_value_gaps.get_active_bullish_fvgs("m5")

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
                    "M15 bullish OB",
                    bool(bullish_obs),
                    f"{len(bullish_obs)} active bullish OB(s).",
                ),
                (
                    "M5 bullish FVG",
                    bool(bullish_fvgs),
                    f"{len(bullish_fvgs)} active bullish FVG(s).",
                ),
            ])

            if not swept or not bullish_obs or not bullish_fvgs:
                return None

            for ob in bullish_obs:
                for fvg in bullish_fvgs:
                    ob_bottom = float(ob.bottom)
                    fvg_bottom = float(fvg.bottom)

                    if abs(ob_bottom - fvg_bottom) <= alignment_tolerance:
                        entry = fvg_bottom
                        stop = (
                            min(ob_bottom, fvg_bottom)
                            - 10 * self.point_value
                        )
                        target = self._select_target(
                            SignalDirection.BUY,
                            entry,
                            liquidity.get_buy_side_liquidity("h1"),   # FIX
                            60 * self.point_value,
                        )

                        signal = self._create_signal(
                            SignalDirection.BUY,
                            entry,
                            stop,
                            target,
                            StrategyType.LIQUIDITY_SWEEP_OB_FVG,
                            confirmations,
                            "m5",
                            "m15",
                        )
                        if signal:
                            return signal

        elif h4_bias == "bearish":
            swept = liquidity.was_buy_side_swept("m15")
            bearish_obs = order_blocks.get_active_bearish_obs("m15")
            bearish_fvgs = fair_value_gaps.get_active_bearish_fvgs("m5")

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
                    "M15 bearish OB",
                    bool(bearish_obs),
                    f"{len(bearish_obs)} active bearish OB(s).",
                ),
                (
                    "M5 bearish FVG",
                    bool(bearish_fvgs),
                    f"{len(bearish_fvgs)} active bearish FVG(s).",
                ),
            ])

            if not swept or not bearish_obs or not bearish_fvgs:
                return None

            for ob in bearish_obs:
                for fvg in bearish_fvgs:
                    ob_top = float(ob.top)
                    fvg_top = float(fvg.top)

                    if abs(ob_top - fvg_top) <= alignment_tolerance:
                        entry = fvg_top
                        stop = (
                            max(ob_top, fvg_top)
                            + 10 * self.point_value
                        )
                        target = self._select_target(
                            SignalDirection.SELL,
                            entry,
                            liquidity.get_sell_side_liquidity("h1"),  # FIX
                            60 * self.point_value,
                        )

                        signal = self._create_signal(
                            SignalDirection.SELL,
                            entry,
                            stop,
                            target,
                            StrategyType.LIQUIDITY_SWEEP_OB_FVG,
                            confirmations,
                            "m5",
                            "m15",
                        )
                        if signal:
                            return signal

        return None


# Shared engine instance used by the trading agent.
smc_engine = SMCEngine()