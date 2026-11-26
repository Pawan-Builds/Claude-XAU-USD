"""FastAPI backend for trading dashboard.

F-03: every endpoint except /api/health requires localhost + bearer
token. Every state-changing call routes through trading_controller.
"""

import asyncio
import hmac
import json
import math
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
from contextlib import asynccontextmanager

from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.config_loader import config
from src.mt5_connector import mt5_connector
from src.market_data import market_data
from src.market_structure import market_structure
from src.liquidity import liquidity
from src.order_blocks import order_blocks
from src.fair_value_gaps import fair_value_gaps
from src.premium_discount import premium_discount
from src.risk_manager import risk_manager
from src.trade_manager import trade_manager
from src.order_executor import order_executor
from src.session_filter import session_filter
from src.news_filter import news_filter
from src.telegram_bot import telegram_bot
from src.trading_controller import trading_controller
from src.dashboard_auth import require_auth, check_websocket_allowed
from src.database_manager import (
    db_manager, ClosedTrade, AccountSnapshot, OpenPosition,
    BacktestReport, BacktestConfig, TradePlan, AppSettings,
)
from src.backtesting.engine import run_backtest_async
from src.backtesting.reports import (
    get_report_summary, list_reports, compare_reports,
)
from src.logger import get_logger

logger = get_logger(__name__)


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(
            "WebSocket connected. Total: %d", len(self.active_connections)
        )

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(
            "WebSocket disconnected. Total: %d", len(self.active_connections)
        )

    async def broadcast(self, message: Dict):
        for connection in list(self.active_connections):
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


async def live_updates_task():
    while True:
        try:
            if manager.active_connections:
                update = await get_live_update()
                await manager.broadcast(update)
        except Exception as e:
            logger.error("Live update error: %s", e)
        await asyncio.sleep(2)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Dashboard backend starting...")
    asyncio.create_task(live_updates_task())
    yield
    logger.info("Dashboard backend shutting down...")


app = FastAPI(
    title="XAU/USD SMC Trading Dashboard",
    description="Real-time trading dashboard for SMC algorithmic trading system",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.dashboard.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------- Pydantic models ----------------

class TradeRequest(BaseModel):
    direction: str
    entry_price: float
    sl_price: float
    tp_price: float
    risk_percent: float = 1.0
    strategy: str = "manual"
    market_entry: bool = False


class BacktestRequest(BaseModel):
    name: str
    start_date: str
    end_date: str
    initial_balance: float = 10000
    risk_percent: float = 2.0
    min_risk_reward: float = 2.0
    timeframes: List[str] = ["m5", "m15", "h1", "h4"]
    sessions: List[str] = ["london", "new_york"]
    strategies: List[str] = ["liquidity_sweep_fvg"]
    spread_pips: float = 2.0
    slippage_pips: float = 1.0
    commission_per_lot: float = 7.0
    swap_enabled: bool = True


class RiskConfigRequest(BaseModel):
    risk_percent: Optional[float] = None
    max_positions: Optional[int] = None
    max_daily_loss: Optional[float] = None
    max_drawdown: Optional[float] = None
    compound_enabled: Optional[bool] = None
    # M-15: weakening a limit requires explicit confirmation.
    confirm: str = ""


class KillSwitchRequest(BaseModel):
    active: bool
    confirm: str = ""

_RISK_OVERRIDES_KEY_PREFIX = "risk_config_overrides_"


def _risk_overrides_key(env: str) -> str:
    return f"{_RISK_OVERRIDES_KEY_PREFIX}{env}"


def _read_risk_overrides(env: str) -> Dict[str, Any]:
    """M-15: read persisted risk overrides for the given environment."""
    try:
        row = db_manager.query_one(
            AppSettings, key=_risk_overrides_key(env)
        )
        if row is None or not row.value:
            return {}
        parsed = json.loads(row.value)
        if not isinstance(parsed, dict):
            return {}
        return parsed
    except Exception:
        logger.exception("Could not read risk config overrides.")
        return {}


def _write_risk_overrides(env: str, data: Dict[str, Any]) -> None:
    """M-15: persist overrides. Raises on failure so caller fails closed."""
    payload = json.dumps(data)
    with db_manager.get_session() as session:
        row = (
            session.query(AppSettings)
            .filter_by(key=_risk_overrides_key(env))
            .first()
        )
        if row is None:
            session.add(AppSettings(
                key=_risk_overrides_key(env), value=payload,
            ))
        else:
            row.value = payload


def _validate_risk_request(
    request: "RiskConfigRequest",
    current: Dict[str, Any],
) -> Tuple[Dict[str, Any], bool, List[str]]:
    """M-15: strict validation. Returns (changes, weakening, errors)."""
    changes: Dict[str, Any] = {}
    errors: List[str] = []
    weakening = False

    if request.risk_percent is not None:
        try:
            val = float(request.risk_percent)
        except (TypeError, ValueError):
            errors.append("risk_percent must be a finite number.")
            val = None
        if val is not None:
            if not math.isfinite(val):
                errors.append("risk_percent must be finite.")
            elif val <= 0:
                errors.append("risk_percent must be > 0.")
            else:
                cap = float(current.get("max_risk_percent") or 0.0)
                cur = float(current.get("default_risk_percent") or 0.0)
                if cap > 0 and val > cap:
                    errors.append(
                        f"risk_percent {val} exceeds configured cap "
                        f"{cap}. Edit config/settings.yaml to raise the cap."
                    )
                else:
                    changes["default_risk_percent"] = val
                    if cur > 0 and val > cur:
                        weakening = True

    if request.max_positions is not None:
        try:
            val = int(request.max_positions)
        except (TypeError, ValueError):
            errors.append("max_positions must be an integer.")
            val = None
        if val is not None:
            if val < 1:
                errors.append("max_positions must be >= 1.")
            elif val > 100:
                errors.append("max_positions cannot exceed 100.")
            else:
                cur = int(current.get("max_open_positions") or 0)
                changes["max_open_positions"] = val
                if cur > 0 and val > cur:
                    weakening = True

    if request.max_daily_loss is not None:
        try:
            val = float(request.max_daily_loss)
        except (TypeError, ValueError):
            errors.append("max_daily_loss must be a finite number.")
            val = None
        if val is not None:
            if not math.isfinite(val):
                errors.append("max_daily_loss must be finite.")
            elif val <= 0:
                errors.append("max_daily_loss must be > 0.")
            elif val > 100:
                errors.append("max_daily_loss cannot exceed 100%.")
            else:
                cur = float(current.get("daily_loss_limit_percent") or 0.0)
                changes["daily_loss_limit_percent"] = val
                if cur > 0 and val > cur:
                    weakening = True

    if request.max_drawdown is not None:
        try:
            val = float(request.max_drawdown)
        except (TypeError, ValueError):
            errors.append("max_drawdown must be a finite number.")
            val = None
        if val is not None:
            if not math.isfinite(val):
                errors.append("max_drawdown must be finite.")
            elif val <= 0:
                errors.append("max_drawdown must be > 0.")
            elif val > 100:
                errors.append("max_drawdown cannot exceed 100%.")
            else:
                cur = float(
                    current.get("max_account_drawdown_percent") or 0.0
                )
                changes["max_account_drawdown_percent"] = val
                if cur > 0 and val > cur:
                    weakening = True

    if request.compound_enabled is not None:
        if not isinstance(request.compound_enabled, bool):
            errors.append("compound_enabled must be a boolean.")
        else:
            cur = bool(current.get("compound_enabled") or False)
            changes["compound_enabled"] = request.compound_enabled
            if cur and not request.compound_enabled:
                # Turning compounding off means position size stops
                # adapting to reduced equity — a safety regression.
                weakening = True

    return changes, weakening, errors
# ---------------- Endpoints ----------------

@app.get("/api/health")
async def health_check():
    """Liveness probe; unauthenticated, leaks nothing beyond 'up'."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/status", dependencies=[Depends(require_auth)])
async def get_status():
    account = mt5_connector.get_account_info()
    controller_status = trading_controller.status()

    active_plans = []
    try:
        with db_manager.get_session() as session:
            plans = (
                session.query(TradePlan)
                .filter(
                    TradePlan.environment == config.trading.mode.lower(),
                    TradePlan.status == "WAITING",
                )
                .all()
            )
            active_plans = [
                {
                    "direction": p.direction,
                    "strategy": p.strategy,
                    "entry_price": p.entry_price,
                    "sl": p.sl_price,
                    "tp": p.tp_price,
                    "expiry": p.expiry_time.isoformat(),
                }
                for p in plans
            ]
    except Exception as e:
        logger.error("Failed to fetch active trade plans: %s", e)

    return {
        "bot_state": controller_status.state,
        "agent_registered": controller_status.agent_registered,
        "kill_switch_active": controller_status.kill_switch,
        "trading_mode": controller_status.mode,
        "mt5_connected": controller_status.mt5_connected,
        "account": (
            {
                "login": account.login,
                "server": account.server,
                "is_demo": account.is_demo,
                "trade_allowed": account.trade_allowed,
                "balance": account.balance,
                "equity": account.equity,
                "free_margin": account.free_margin,
                "margin_level": account.margin_level,
            }
            if account else None
        ),
        "market_data": {
            "symbol": mt5_connector.get_symbol_name(),
            "last_update": datetime.now(timezone.utc).isoformat(),
        },
        "trading_paused": controller_status.state == "paused",
        "active_plans": active_plans,
    }


@app.get("/api/account", dependencies=[Depends(require_auth)])
async def get_account():
    account = mt5_connector.get_account_info()
    if not account:
        raise HTTPException(status_code=503, detail="Account not connected")

    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    )
    env = config.trading.mode.lower()
    with db_manager.get_session() as session:
        trades = (
            session.query(ClosedTrade)
            .filter(
                ClosedTrade.environment == env,
                ClosedTrade.time_close >= today_start,
            )
            .all()
        )
    daily_pl = sum(t.net_profit for t in trades)
    positions = mt5_connector.get_positions() or []

    return {
        "login": account.login,
        "server": account.server,
        "name": account.name,
        "company": account.company,
        "currency": account.currency,
        "leverage": account.leverage,
        "balance": account.balance,
        "equity": account.equity,
        "margin": account.margin,
        "free_margin": account.free_margin,
        "margin_level": account.margin_level,
        "is_demo": account.is_demo,
        "trade_allowed": account.trade_allowed,
        "trade_expert": account.trade_expert,
        "daily_pl": daily_pl,
        "floating_pl": sum(p["profit"] for p in positions),
    }


@app.get("/api/market", dependencies=[Depends(require_auth)])
async def get_market():
    tick = market_data.get_current_tick()
    session_info = market_data.get_session_info()

    h4_structure = market_structure.get_structure("h4")
    h1_structure = market_structure.get_structure("h1")
    m15_structure = market_structure.get_structure("m15")
    m5_structure = market_structure.get_structure("m5")

    def _s(s):
        if not s:
            return {}
        return {
            "bias": s.bias,
            "bos": s.structure.bos_detected,
            "choch": s.structure.choch_detected,
        }

    return {
        "tick": tick,
        "session": session_info,
        "structure": {
            "h4": _s(h4_structure),
            "h1": _s(h1_structure),
            "m15": _s(m15_structure),
            "m5": _s(m5_structure),
        },
        "liquidity": {
            "m15_buy_swept": liquidity.was_buy_side_swept("m15"),
            "m15_sell_swept": liquidity.was_sell_side_swept("m15"),
            "h1_buy_levels": len(liquidity.get_buy_side_liquidity("h1")),
            "h1_sell_levels": len(liquidity.get_sell_side_liquidity("h1")),
        },
        "order_blocks": {
            "m15_bullish": len(order_blocks.get_active_bullish_obs("m15")),
            "m15_bearish": len(order_blocks.get_active_bearish_obs("m15")),
            "m5_bullish": len(order_blocks.get_active_bullish_obs("m5")),
            "m5_bearish": len(order_blocks.get_active_bearish_obs("m5")),
        },
        "fvgs": {
            "m5_bullish": len(fair_value_gaps.get_active_bullish_fvgs("m5")),
            "m5_bearish": len(fair_value_gaps.get_active_bearish_fvgs("m5")),
            "m15_bullish": len(fair_value_gaps.get_active_bullish_fvgs("m15")),
            "m15_bearish": len(fair_value_gaps.get_active_bearish_fvgs("m15")),
        },
        "premium_discount": {
            "h4": premium_discount.get_zone("h4"),
            "h1": premium_discount.get_zone("h1"),
            "m15": premium_discount.get_zone("m15"),
            "m5": premium_discount.get_zone("m5"),
        },
        "spread_pips": market_data.get_spread_pips(),
        "high_spread": market_data.is_high_spread(),
    }


@app.get("/api/positions", dependencies=[Depends(require_auth)])
async def get_positions():
    positions = mt5_connector.get_positions() or []

    result = []
    for pos in positions:
        pos_state = trade_manager.get_position_state(pos["ticket"])
        r_multiple = 0
        if pos_state and pos_state["current_sl"] > 0:
            entry = pos_state["entry_price"]
            sl = pos_state["current_sl"]
            if pos["type"] == "BUY":
                risk = entry - sl
                if risk > 0:
                    r_multiple = (pos["price_current"] - entry) / risk
            else:
                risk = sl - entry
                if risk > 0:
                    r_multiple = (entry - pos["price_current"]) / risk

        result.append({
            "ticket": pos["ticket"],
            "direction": pos["type"],
            "symbol": pos["symbol"],
            "volume": pos["volume"],
            "entry": pos["price_open"],
            "current": pos["price_current"],
            "sl": pos["sl"],
            "tp": pos["tp"],
            "profit": pos["profit"],
            "swap": pos["swap"],
            "commission": pos.get("commission", 0.0),
            "r_multiple": round(r_multiple, 2),
            "strategy": (pos.get("comment") or "").replace("SMC_", ""),
            "duration_hours": round(
                (datetime.now(timezone.utc) - pos["time_open"]).total_seconds()
                / 3600,
                1,
            ),
        })
    return {"positions": result, "count": len(result)}


@app.get("/api/history", dependencies=[Depends(require_auth)])
async def get_history(limit: int = 50):
    trades = db_manager.get_closed_trades(
        config.trading.mode.lower(), limit
    )
    result = [
        {
            "ticket": t.ticket,
            "direction": t.direction,
            "symbol": t.symbol,
            "entry_time": t.time_open.isoformat(),
            "exit_time": t.time_close.isoformat(),
            "entry_price": t.price_open,
            "exit_price": t.price_close,
            "sl": t.sl,
            "tp": t.tp,
            "volume": t.volume,
            "gross_profit": t.profit,
            "net_profit": t.net_profit,
            "realized_r": t.realized_r,
            "risk_percent": t.risk_percent,
            "strategy": t.strategy,
            "exit_reason": t.exit_reason,
            "duration_hours": t.duration_hours,
        }
        for t in trades
    ]
    return {"trades": result, "count": len(result)}


@app.get("/api/performance", dependencies=[Depends(require_auth)])
async def get_performance():
    env = config.trading.mode.lower()
    with db_manager.get_session() as session:
        trades = (
            session.query(ClosedTrade)
            .filter(ClosedTrade.environment == env)
            .all()
        )

    if not trades:
        return {"trades": 0, "message": "No trades yet"}

    total = len(trades)
    winners = [t for t in trades if t.net_profit > 0]
    losers = [t for t in trades if t.net_profit < 0]
    win_rate = len(winners) / total * 100
    gross_profit = sum(t.net_profit for t in winners)
    gross_loss = abs(sum(t.net_profit for t in losers))
    net_profit = sum(t.net_profit for t in trades)
    profit_factor = (
        gross_profit / gross_loss if gross_loss > 0 else float("inf")
    )

    snapshots = db_manager.get_account_snapshots(env, 500)
    equity_curve = [
        {
            "time": s.timestamp.isoformat(),
            "equity": s.equity,
            "balance": s.balance,
        }
        for s in reversed(snapshots)
    ]

    from collections import defaultdict
    monthly = defaultdict(float)
    equity = 10000
    for trade in sorted(trades, key=lambda x: x.time_close):
        key = trade.time_close.strftime("%Y-%m")
        ret = trade.net_profit / equity * 100
        monthly[key] += ret
        equity += trade.net_profit

    return {
        "total_trades": total,
        "winning_trades": len(winners),
        "losing_trades": len(losers),
        "win_rate": round(win_rate, 1),
        "profit_factor": (
            round(profit_factor, 2)
            if profit_factor != float("inf")
            else "inf"
        ),
        "net_profit": round(net_profit, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_win": round(gross_profit / len(winners), 2) if winners else 0,
        "avg_loss": round(gross_loss / len(losers), 2) if losers else 0,
        "expectancy": round(
            (win_rate / 100 * gross_profit / len(winners))
            - ((100 - win_rate) / 100 * gross_loss / len(losers)),
            2,
        ) if winners and losers else 0,
        "max_drawdown": 0,
        "equity_curve": equity_curve,
        "monthly_returns": dict(monthly),
    }


@app.get("/api/backtests", dependencies=[Depends(require_auth)])
async def get_backtests():
    return {"reports": list_reports(50)}


@app.get("/api/backtests/{report_id}", dependencies=[Depends(require_auth)])
async def get_backtest(report_id: int):
    summary = get_report_summary(report_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Report not found")
    return summary


@app.post("/api/backtests", dependencies=[Depends(require_auth)])
async def run_backtest(request: BacktestRequest):
    try:
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="Invalid date format. Use ISO format.",
        )

    bt_config = BacktestConfig(
        name=request.name,
        start_date=start_date,
        end_date=end_date,
        initial_balance=request.initial_balance,
        risk_percent=request.risk_percent,
        min_risk_reward=request.min_risk_reward,
        timeframes=request.timeframes,
        sessions=request.sessions,
        strategies=request.strategies,
        spread_pips=request.spread_pips,
        slippage_pips=request.slippage_pips,
        commission_per_lot=request.commission_per_lot,
        swap_enabled=request.swap_enabled,
        data_source="mt5",
    )

    try:
        config_id = db_manager.add(bt_config)
    except Exception as exc:
        logger.exception("Could not persist backtest config.")
        raise HTTPException(
            status_code=500,
            detail=f"Could not save backtest config: {exc}",
        )

    # Phase 4A: pass the integer ID (not the object) and let
    # run_backtest_async resolve the config itself.
    asyncio.create_task(run_backtest_async(config_id))
    return {"config_id": config_id, "status": "started"}


@app.post("/api/trade", dependencies=[Depends(require_auth)])
async def execute_trade(request: TradeRequest):
    """Manual trade; funnels through trading_controller -> agent gate."""
    from src.smc_strategy import (
        TradeSignal, SignalDirection, StrategyType, Confirmation,
    )

    try:
        direction = SignalDirection(request.direction.upper())
    except ValueError:
        raise HTTPException(
            status_code=400, detail="direction must be BUY or SELL",
        )

    strategy = StrategyType.LIQUIDITY_SWEEP_FVG
    for s in StrategyType:
        if s.value == request.strategy:
            strategy = s
            break

    signal = TradeSignal(
        timestamp=datetime.now(timezone.utc),
        symbol=config.trading.symbol,
        direction=direction,
        entry_price=request.entry_price,
        sl_price=request.sl_price,
        tp_price=request.tp_price,
        lot_size=0,
        risk_percent=request.risk_percent,
        risk_reward=0,
        strategy=strategy,
        confirmations=[Confirmation("Manual", True, "Manual trade")],
        timeframe_analysis={},
        entry_tf="m5",
        setup_tf="m15",
        structure_tf="h1",
        bias_tf="h4",
    )

    result = trading_controller.submit_manual_trade(signal)

    if result.success:
        try:
            await telegram_bot.notify_new_trade(signal, result)
        except Exception:
            logger.exception("Telegram notify_new_trade failed.")
        return {
            "success": True,
            "ticket": (
                result.order_result.ticket
                if result.order_result else None
            ),
            "pending": result.pending,
        }
    return {"success": False, "error": result.error}


@app.post("/api/control/pause", dependencies=[Depends(require_auth)])
async def pause_trading():
    return trading_controller.pause()


@app.post("/api/control/resume", dependencies=[Depends(require_auth)])
async def resume_trading():
    return trading_controller.resume()


@app.post("/api/control/shutdown", dependencies=[Depends(require_auth)])
async def shutdown_bot():
    return trading_controller.shutdown()


@app.post(
    "/api/control/close_position/{ticket}",
    dependencies=[Depends(require_auth)],
)
async def close_position(ticket: int):
    return trading_controller.close_position(ticket)


@app.post("/api/control/close_all", dependencies=[Depends(require_auth)])
async def close_all_positions():
    return trading_controller.close_all()


@app.post(
    "/api/control/emergency_stop", dependencies=[Depends(require_auth)]
)
async def emergency_stop():
    return trading_controller.emergency_stop(reason="dashboard")


@app.post(
    "/api/control/clear_emergency", dependencies=[Depends(require_auth)]
)
async def clear_emergency():
    return trading_controller.clear_emergency()


@app.post(
    "/api/control/kill_switch", dependencies=[Depends(require_auth)]
)
async def set_kill_switch(request: KillSwitchRequest):
    """Explicit kill-switch control. Deactivation requires confirmation."""
    if not request.active and request.confirm != "I_UNDERSTAND":
        raise HTTPException(
            status_code=400,
            detail=(
                "To deactivate the kill switch, 'confirm' must be "
                "'I_UNDERSTAND'."
            ),
        )
    return trading_controller.set_kill_switch(
        request.active, reason="dashboard"
    )


@app.get("/api/risk", dependencies=[Depends(require_auth)])
async def get_risk_status():
    return risk_manager.check_risk_limits()


@app.post("/api/risk/config", dependencies=[Depends(require_auth)])
async def update_risk_config(request: RiskConfigRequest):
    """Update (tighten) risk configuration overrides.

    Security model
    --------------
    * Authentication: enforced by the standard bearer-token
      ``require_auth`` dependency — identical to every other
      state-changing endpoint here.
    * Authorization: only operators holding the dashboard token may
      call this endpoint (same boundary as pause / resume /
      kill-switch).
    * Validation: every field is bounds-checked. NaN, infinities,
      non-positive values, and values above the configured YAML cap
      are rejected with HTTP 400 *before* any write happens.
    * Direction: tightening is accepted directly. Weakening requires
      ``confirm='I_UNDERSTAND'`` (mirroring the kill-switch pattern).
    * Persistence: accepted changes are merged into ``AppSettings``
      under ``risk_config_overrides_<env>`` (durable + auditable),
      with read-back verification.
    * Effect: this endpoint **does not** hot-reload the running bot.
      ``risk_manager`` snapshots its config at construction and YAML
      remains authoritative. Apply overrides by editing
      ``config/settings.yaml`` and restarting.
    """
    env = config.trading.mode.lower()

    try:
        current_risk = config.risk
    except Exception:
        logger.exception("Could not read current risk config.")
        raise HTTPException(
            status_code=500,
            detail="Current risk configuration is unreadable.",
        )

    current = {
        "default_risk_percent": float(current_risk.default_risk_percent),
        "max_risk_percent": float(current_risk.max_risk_percent),
        "max_open_positions": int(current_risk.max_open_positions),
        "daily_loss_limit_percent": float(
            current_risk.daily_loss_limit_percent
        ),
        "max_account_drawdown_percent": float(
            current_risk.max_account_drawdown_percent
        ),
        "max_total_open_risk_percent": float(
            current_risk.max_total_open_risk_percent
        ),
        "emergency_drawdown_percent": float(
            current_risk.emergency_drawdown_percent
        ),
        "compound_enabled": bool(current_risk.compound_enabled),
    }

    changes, weakening, errors = _validate_risk_request(
        request, current
    )

    if errors:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid risk configuration.",
                "errors": errors,
            },
        )

    if not changes:
        return {
            "success": True,
            "message": "No changes requested.",
            "current": current,
        }

    if weakening and request.confirm != "I_UNDERSTAND":
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Weakening risk limits requires confirmation. "
                    "Set confirm='I_UNDERSTAND' to proceed. Preferred "
                    "path: edit config/settings.yaml and restart the "
                    "bot."
                ),
                "requested_changes": changes,
                "current": current,
            },
        )

    try:
        existing = _read_risk_overrides(env)
    except Exception:
        logger.exception("Could not read existing risk overrides.")
        raise HTTPException(
            status_code=500,
            detail="Could not read existing risk overrides.",
        )

    merged = {**existing, **changes}

    try:
        _write_risk_overrides(env, merged)
    except Exception:
        logger.exception("Could not persist risk config overrides.")
        raise HTTPException(
            status_code=500,
            detail="Risk configuration could not be persisted.",
        )

    persisted = _read_risk_overrides(env)
    if persisted != merged:
        logger.critical(
            "Risk override read-back mismatch | wrote=%s read=%s",
            merged, persisted,
        )
        raise HTTPException(
            status_code=500,
            detail="Risk configuration read-back verification failed.",
        )

    logger.warning(
        "RISK OVERRIDES UPDATED | env=%s | weakening=%s | "
        "changes=%s | persisted=%s",
        env, weakening, changes, persisted,
    )

    return {
        "success": True,
        "message": (
            "Overrides persisted to AppSettings. The running bot "
            "continues to use YAML-configured values until it is "
            "restarted with the corresponding YAML changes applied. "
            "This endpoint records operator intent and provides a "
            "validated, authenticated, audit-trailed write path — "
            "it does not hot-reload into the live process, "
            "preserving fail-closed semantics."
        ),
        "previous": current,
        "applied_changes": changes,
        "all_overrides": persisted,
    }


# ---------------- WebSocket ----------------

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    client_host = websocket.client.host if websocket.client else None
    provided = websocket.query_params.get("token", "")
    configured = config.dashboard.secret_key

    allowed, code, reason = check_websocket_allowed(
        client_host, provided, configured
    )
    if not allowed:
        logger.warning(
            "WebSocket refused | host=%s | reason=%s",
            client_host, reason,
        )
        await websocket.close(code=code)
        return

    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep-alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def get_live_update() -> Dict:
    tick = market_data.get_current_tick()
    positions = mt5_connector.get_positions() or []
    account = mt5_connector.get_account_info()
    controller = trading_controller.status()

    active_plans = []
    try:
        with db_manager.get_session() as session:
            plans = (
                session.query(TradePlan)
                .filter(
                    TradePlan.environment == config.trading.mode.lower(),
                    TradePlan.status == "WAITING",
                )
                .all()
            )
            active_plans = [
                {
                    "direction": p.direction,
                    "strategy": p.strategy,
                    "entry_price": p.entry_price,
                    "sl": p.sl_price,
                    "tp": p.tp_price,
                    "expiry": p.expiry_time.isoformat(),
                }
                for p in plans
            ]
    except Exception:
        pass

    pos_data = []
    for pos in positions:
        pos_state = trade_manager.get_position_state(pos["ticket"])
        r_mult = 0
        if pos_state and pos_state["current_sl"] > 0:
            entry = pos_state["entry_price"]
            sl = pos_state["current_sl"]
            if pos["type"] == "BUY":
                risk = entry - sl
                if risk > 0:
                    r_mult = (pos["price_current"] - entry) / risk
            else:
                risk = sl - entry
                if risk > 0:
                    r_mult = (entry - pos["price_current"]) / risk
        pos_data.append({
            "ticket": pos["ticket"],
            "direction": pos["type"],
            "volume": pos["volume"],
            "entry": pos["price_open"],
            "current": pos["price_current"],
            "sl": pos["sl"],
            "tp": pos["tp"],
            "profit": pos["profit"],
            "r_multiple": round(r_mult, 2),
        })

    return {
        "type": "live_update",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tick": tick,
        "account": {
            "balance": account.balance if account else 0,
            "equity": account.equity if account else 0,
            "free_margin": account.free_margin if account else 0,
        },
        "positions": pos_data,
        "bot_state": controller.state,
        "kill_switch_active": controller.kill_switch,
        "spread_pips": market_data.get_spread_pips(),
        "active_plans": active_plans,
    }


try:
    app.mount(
        "/",
        StaticFiles(directory="dashboard/frontend/dist", html=True),
        name="frontend",
    )
except RuntimeError:
    pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, host=config.dashboard.host, port=config.dashboard.port
    )