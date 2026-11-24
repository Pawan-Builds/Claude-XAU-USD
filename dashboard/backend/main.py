"""FastAPI backend for trading dashboard."""
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
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
from src.telegram_bot import telegram_bot, BotState
from src.database_manager import db_manager, ClosedTrade, AccountSnapshot, OpenPosition, BacktestReport, BacktestConfig, TradePlan
from src.backtesting.engine import run_backtest_async, BacktestConfig as BTConfig
from src.backtesting.reports import get_report_summary, list_reports, compare_reports
from src.logger import get_logger

logger = get_logger(__name__)


# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: Dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass


manager = ConnectionManager()


# Background task for live updates
async def live_updates_task():
    """Send live updates to connected WebSocket clients."""
    while True:
        try:
            if manager.active_connections:
                update = await get_live_update()
                await manager.broadcast(update)
        except Exception as e:
            logger.error(f"Live update error: {e}")

        await asyncio.sleep(2)  # Update every 2 seconds


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Dashboard backend starting...")
    asyncio.create_task(live_updates_task())
    yield
    # Shutdown
    logger.info("Dashboard backend shutting down...")


app = FastAPI(
    title="XAU/USD SMC Trading Dashboard",
    description="Real-time trading dashboard for SMC algorithmic trading system",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.dashboard.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class TradeRequest(BaseModel):
    direction: str  # BUY or SELL
    entry_price: float
    sl_price: float
    tp_price: float
    risk_percent: float = 2.0
    strategy: str = "manual"


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


# Authentication (simple for localhost)
async def verify_localhost(request: Request):
    client_host = request.client.host
    if client_host not in ["127.0.0.1", "localhost", "::1"]:
        raise HTTPException(status_code=403, detail="Access denied: localhost only")
    return True


# --- API Endpoints ---

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/api/status")
async def get_status():
    """Get overall system status."""
    mt5_connected = mt5_connector.is_connected()
    account = mt5_connector.get_account_info()
    bot_state = telegram_bot.get_state()

    # Fetch active trade plans
    active_plans = []
    try:
        with db_manager.get_session() as session:
            plans = session.query(TradePlan).filter(
                TradePlan.environment == config.trading.mode.lower(),
                TradePlan.status == "WAITING"
            ).all()
            active_plans = [
                {
                    "direction": p.direction,
                    "strategy": p.strategy,
                    "entry_price": p.entry_price,
                    "sl": p.sl_price,
                    "tp": p.tp_price,
                    "expiry": p.expiry_time.isoformat()
                } for p in plans
            ]
    except Exception as e:
        logger.error(f"Failed to fetch active trade plans: {e}")

    return {
        "bot_state": bot_state.value,
        "mt5_connected": mt5_connected,
        "account": {
            "login": account.login if account else None,
            "server": account.server if account else None,
            "is_demo": account.is_demo if account else None,
            "trade_allowed": account.trade_allowed if account else None,
            "balance": account.balance if account else 0,
            "equity": account.equity if account else 0,
            "free_margin": account.free_margin if account else 0,
            "margin_level": account.margin_level if account else 0,
        } if account else None,
        "market_data": {
            "symbol": mt5_connector.get_symbol_name(),
            "last_update": datetime.now(timezone.utc).isoformat()
        },
        "trading_paused": bot_state == BotState.PAUSED,
        "active_plans": active_plans
    }


@app.get("/api/account")
async def get_account():
    """Get account details."""
    account = mt5_connector.get_account_info()
    if not account:
        raise HTTPException(status_code=503, detail="Account not connected")

    # Daily P/L
    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    with db_manager.get_session() as session:
        trades = session.query(ClosedTrade).filter(
            ClosedTrade.environment == "demo",
            ClosedTrade.time_close >= today_start
        ).all()

    daily_pl = sum(t.net_profit for t in trades)

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
        "floating_pl": sum(p["profit"] for p in mt5_connector.get_positions())
    }


@app.get("/api/market")
async def get_market():
    """Get live market data."""
    tick = market_data.get_current_tick()
    session_info = market_data.get_session_info()

    # Get multi-timeframe analysis
    h4_structure = market_structure.get_structure("h4")
    h1_structure = market_structure.get_structure("h1")
    m15_structure = market_structure.get_structure("m15")
    m5_structure = market_structure.get_structure("m5")

    return {
        "tick": tick,
        "session": session_info,
        "structure": {
            "h4": {"bias": h4_structure.bias if h4_structure else "unknown",
                   "bos": h4_structure.structure.bos_detected if h4_structure else False,
                   "choch": h4_structure.structure.choch_detected if h4_structure else False} if h4_structure else {},
            "h1": {"bias": h1_structure.bias if h1_structure else "unknown",
                   "bos": h1_structure.structure.bos_detected if h1_structure else False,
                   "choch": h1_structure.structure.choch_detected if h1_structure else False} if h1_structure else {},
            "m15": {"bias": m15_structure.bias if m15_structure else "unknown",
                    "bos": m15_structure.structure.bos_detected if m15_structure else False,
                    "choch": m15_structure.structure.choch_detected if m15_structure else False} if m15_structure else {},
            "m5": {"bias": m5_structure.bias if m5_structure else "unknown",
                   "bos": m5_structure.structure.bos_detected if m5_structure else False,
                   "choch": m5_structure.structure.choch_detected if m5_structure else False} if m5_structure else {}
        },
        "liquidity": {
            "m15_buy_swept": liquidity.was_buy_side_swept("m15"),
            "m15_sell_swept": liquidity.was_sell_side_swept("m15"),
            "h1_buy_levels": len(liquidity.get_buy_side_liquidity("h1")),
            "h1_sell_levels": len(liquidity.get_sell_side_liquidity("h1"))
        },
        "order_blocks": {
            "m15_bullish": len(order_blocks.get_active_bullish_obs("m15")),
            "m15_bearish": len(order_blocks.get_active_bearish_obs("m15")),
            "m5_bullish": len(order_blocks.get_active_bullish_obs("m5")),
            "m5_bearish": len(order_blocks.get_active_bearish_obs("m5"))
        },
        "fvgs": {
            "m5_bullish": len(fair_value_gaps.get_active_bullish_fvgs("m5")),
            "m5_bearish": len(fair_value_gaps.get_active_bearish_fvgs("m5")),
            "m15_bullish": len(fair_value_gaps.get_active_bullish_fvgs("m15")),
            "m15_bearish": len(fair_value_gaps.get_active_bearish_fvgs("m15"))
        },
        "premium_discount": {
            "h4": premium_discount.get_zone("h4"),
            "h1": premium_discount.get_zone("h1"),
            "m15": premium_discount.get_zone("m15"),
            "m5": premium_discount.get_zone("m5")
        },
        "spread_pips": market_data.get_spread_pips(),
        "high_spread": market_data.is_high_spread()
    }


@app.get("/api/positions")
async def get_positions():
    """Get open positions."""
    positions = mt5_connector.get_positions()

    result = []
    for pos in positions:
        # Calculate R multiple
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
            "commission": pos["commission"],
            "r_multiple": round(r_multiple, 2),
            "strategy": pos.get("comment", "").replace("SMC_", ""),
            "duration_hours": round((datetime.now(timezone.utc) - pos["time_open"]).total_seconds() / 3600, 1)
        })

    return {"positions": result, "count": len(result)}


@app.get("/api/history")
async def get_history(limit: int = 50):
    """Get trade history."""
    trades = db_manager.get_closed_trades("demo", limit)

    result = []
    for trade in trades:
        result.append({
            "ticket": trade.ticket,
            "direction": trade.direction,
            "symbol": trade.symbol,
            "entry_time": trade.time_open.isoformat(),
            "exit_time": trade.time_close.isoformat(),
            "entry_price": trade.price_open,
            "exit_price": trade.price_close,
            "sl": trade.sl,
            "tp": trade.tp,
            "volume": trade.volume,
            "gross_profit": trade.profit,
            "net_profit": trade.net_profit,
            "realized_r": trade.realized_r,
            "risk_percent": trade.risk_percent,
            "strategy": trade.strategy,
            "exit_reason": trade.exit_reason,
            "duration_hours": trade.duration_hours
        })

    return {"trades": result, "count": len(result)}


@app.get("/api/performance")
async def get_performance():
    """Get performance statistics."""
    with db_manager.get_session() as session:
        trades = session.query(ClosedTrade).filter(
            ClosedTrade.environment == "demo"
        ).all()

    if not trades:
        return {"trades": 0, "message": "No trades yet"}

    total = len(trades)
    winners = [t for t in trades if t.net_profit > 0]
    losers = [t for t in trades if t.net_profit < 0]

    win_rate = len(winners) / total * 100
    gross_profit = sum(t.net_profit for t in winners)
    gross_loss = abs(sum(t.net_profit for t in losers))
    net_profit = sum(t.net_profit for t in trades)
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Equity curve
    snapshots = db_manager.get_account_snapshots("demo", 500)
    equity_curve = [
        {"time": s.timestamp.isoformat(), "equity": s.equity, "balance": s.balance}
        for s in reversed(snapshots)
    ]

    # Monthly returns
    from collections import defaultdict
    monthly = defaultdict(float)
    equity = 10000  # Starting equity
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
        "profit_factor": round(profit_factor, 2) if profit_factor != float('inf') else "inf",
        "net_profit": round(net_profit, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "avg_win": round(gross_profit / len(winners), 2) if winners else 0,
        "avg_loss": round(gross_loss / len(losers), 2) if losers else 0,
        "expectancy": round((win_rate/100 * gross_profit/len(winners)) - ((100-win_rate)/100 * gross_loss/len(losers)), 2) if winners and losers else 0,
        "max_drawdown": 0,  # Would need calculation
        "equity_curve": equity_curve,
        "monthly_returns": dict(monthly)
    }


@app.get("/api/backtests")
async def get_backtests():
    """List backtest reports."""
    reports = list_reports(50)
    return {"reports": reports}


@app.get("/api/backtests/{report_id}")
async def get_backtest(report_id: int):
    """Get backtest report details."""
    summary = get_report_summary(report_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Report not found")
    return summary


@app.post("/api/backtests")
async def run_backtest(request: BacktestRequest):
    """Run a new backtest."""
    try:
        start_date = datetime.fromisoformat(request.start_date)
        end_date = datetime.fromisoformat(request.end_date)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format.")

    # Create config
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
        data_source="mt5"
    )

    saved_config = db_manager.add(bt_config)

    # Run backtest (async)
    asyncio.create_task(run_backtest_async(saved_config))

    return {"config_id": saved_config.id, "status": "started"}


@app.post("/api/trade")
async def execute_trade(request: TradeRequest):
    """Execute a manual trade."""
    if telegram_bot.get_state() != telegram_bot.BotState.RUNNING:
        raise HTTPException(status_code=400, detail="Bot not running")

    # Create signal-like object
    from src.smc_strategy import TradeSignal, SignalDirection, StrategyType, Confirmation

    signal = TradeSignal(
        timestamp=datetime.now(timezone.utc),
        symbol=config.trading.symbol,
        direction=SignalDirection(request.direction),
        entry_price=request.entry_price,
        sl_price=request.sl_price,
        tp_price=request.tp_price,
        lot_size=0,
        risk_percent=request.risk_percent,
        risk_reward=0,
        strategy=StrategyType(request.strategy) if request.strategy in [s.value for s in StrategyType] else StrategyType.LIQUIDITY_SWEEP_FVG,
        confirmations=[Confirmation("Manual", True, "Manual trade")],
        timeframe_analysis={},
        entry_tf="m5",
        setup_tf="m15",
        structure_tf="h1",
        bias_tf="h4"
    )

    result = order_executor.execute_signal(signal)

    if result.success:
        await telegram_bot.notify_new_trade(signal, result)
        return {"success": True, "ticket": result.order_result.ticket}
    else:
        return {"success": False, "error": result.error}


@app.post("/api/control/pause")
async def pause_trading():
    """Pause trading."""
    telegram_bot._pause_trading()
    return {"status": "paused"}


@app.post("/api/control/resume")
async def resume_trading():
    """Resume trading."""
    telegram_bot._resume_trading()
    return {"status": "running"}


@app.post("/api/control/close_position/{ticket}")
async def close_position(ticket: int):
    """Close a specific position."""
    result = order_executor.close_position(ticket, "MANUAL_DASHBOARD")
    return {"success": result.success, "error": result.error_message}


@app.post("/api/control/close_all")
async def close_all_positions():
    """Close all positions."""
    results = order_executor.close_all_positions("MANUAL_DASHBOARD_ALL")
    return {"closed": sum(1 for r in results if r.success), "failed": sum(1 for r in results if not r.success)}


@app.post("/api/control/emergency_stop")
async def emergency_stop():
    """Emergency stop - pause and close all."""
    telegram_bot._state = telegram_bot.BotState.EMERGENCY
    telegram_bot._pause_trading()
    results = order_executor.close_all_positions("EMERGENCY")
    await telegram_bot.notify_emergency_shutdown("Emergency stop triggered from dashboard")
    return {"status": "emergency_stop", "closed": sum(1 for r in results if r.success)}


@app.post("/api/control/shutdown")
async def shutdown_bot():
    """Shutdown the bot."""
    telegram_bot._request_shutdown()
    return {"status": "shutdown_initiated"}


@app.get("/api/risk")
async def get_risk_status():
    """Get risk management status."""
    checks = risk_manager.check_risk_limits()
    return checks


@app.post("/api/risk/config")
async def update_risk_config(request: RiskConfigRequest):
    """Update risk configuration."""
    # This would update the config - for now just return current
    return {"message": "Risk config update not implemented", "current": config.risk}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for live updates."""
    await manager.connect(websocket)
    try:
        while True:
            # Keep connection alive
            data = await websocket.receive_text()
            # Echo back for ping/pong
            await websocket.send_text(data)
    except WebSocketDisconnect:
        manager.disconnect(websocket)


async def get_live_update() -> Dict:
    """Get live data for WebSocket broadcast."""
    tick = market_data.get_current_tick()
    positions = mt5_connector.get_positions()
    account = mt5_connector.get_account_info()

    # Fetch active plans for websocket
    active_plans = []
    try:
        with db_manager.get_session() as session:
            plans = session.query(TradePlan).filter(
                TradePlan.environment == config.trading.mode.lower(),
                TradePlan.status == "WAITING"
            ).all()
            active_plans = [
                {
                    "direction": p.direction,
                    "strategy": p.strategy,
                    "entry_price": p.entry_price,
                    "sl": p.sl_price,
                    "tp": p.tp_price,
                    "expiry": p.expiry_time.isoformat()
                } for p in plans
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
            "r_multiple": round(r_mult, 2)
        })

    return {
        "type": "live_update",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tick": tick,
        "account": {
            "balance": account.balance if account else 0,
            "equity": account.equity if account else 0,
            "free_margin": account.free_margin if account else 0
        },
        "positions": pos_data,
        "bot_state": telegram_bot.get_state().value,
        "spread_pips": market_data.get_spread_pips(),
        "active_plans": active_plans
    }


# Serve frontend static files
try:
    app.mount("/", StaticFiles(directory="dashboard/frontend/dist", html=True), name="frontend")
except RuntimeError:
    pass  # Frontend not built yet


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.dashboard.host, port=config.dashboard.port)