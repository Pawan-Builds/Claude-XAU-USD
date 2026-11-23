python -m dashboard.backend.main

# Set in .env: TRADING_MODE=DEMO + demo credentials
python -m src.main

# Set in .env: TRADING_MODE=LIVE + live credentials + LIVE_TRADING_ENABLED=true + LIVE_ALLOWED_ACCOUNT
python -m src.main --mode LIVE --confirm-live

# XAU/USD SMC Trading Agent - Dual Mode (DEMO/LIVE)

A complete autonomous XAU/USD algorithmic trading system using Smart Money Concepts (SMC), Python, and XM MetaTrader 5.

## 🎯 Dual-Mode Architecture

This system supports **both DEMO and LIVE trading modes** through configuration. No code changes required to switch modes.

### Mode Selection

```bash
# In .env file - change and restart
TRADING_MODE=DEMO   # or LIVE
```

---

## 📋 Configuration

### Environment Variables (.env)

```ini
# === TRADING MODE ===
# Select: DEMO or LIVE
TRADING_MODE=DEMO

# === DEMO ACCOUNT CREDENTIALS ===
DEMO_MT5_LOGIN=your_demo_login
DEMO_MT5_PASSWORD=your_demo_password
DEMO_MT5_SERVER=XMGlobal-MT5 5
DEMO_MT5_PATH=C:\Program Files\XAUUSD\terminal64.exe

# === LIVE ACCOUNT CREDENTIALS ===
LIVE_MT5_LOGIN=your_live_login
LIVE_MT5_PASSWORD=your_live_password
LIVE_MT5_SERVER=XMGlobal-MT5 5
LIVE_MT5_PATH=C:\Program Files\XAUUSD\terminal64.exe

# LIVE TRADING SAFETY ACTIVATION
# ALL must be true for LIVE trading:
# - TRADING_MODE=LIVE
# - LIVE_TRADING_ENABLED=true
# - LIVE_ALLOWED_ACCOUNT matches actual MT5 account
LIVE_TRADING_ENABLED=false
LIVE_ALLOWED_ACCOUNT=

# TELEGRAM
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# TRADING CONFIGURATION
RISK_PERCENT=2.0
MIN_RISK_REWARD=2.0

# DASHBOARD
DASHBOARD_HOST=127.0.0.1
DASHBOARD_PORT=8000
DASHBOARD_SECRET_KEY=change_this_in_production
```

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
cd dashboard/frontend && npm install && npm run build
```

### 2. Initialize Database
```bash
python -c "from src.database_manager import db_manager; db_manager.create_tables()"
```

### 3. Run DEMO Mode
```bash
# 1. Set TRADING_MODE=DEMO in .env
# 2. Add your XM demo credentials
# 4. Run the bot
python -m src.main

# 5. Run dashboard
python -m dashboard.backend.main
# Open http://localhost:8000
```

### 4. Run LIVE Mode
```bash
# 1. Set TRADING_MODE=LIVE in .env
# 2. Add your XM live credentials
# 3. CRITICAL: Set LIVE_TRADING_ENABLED=true
# 4. Set LIVE_ALLOWED_ACCOUNT=your_live_account_number
# 5. Run with explicit confirmation
python -m src.main --mode LIVE --confirm-live
```

---

## 🔒 LIVE Trading Safety Architecture

LIVE trading requires **ALL** of the following:

1. `TRADING_MODE=LIVE` in .env
2. `LIVE_TRADING_ENABLED=true` in .env
3. `LIVE_ALLOWED_ACCOUNT=your_live_account_number` (must match actual MT5 account)
5. Actual MT5 account must be verified as LIVE (not DEMO)
6. Explicit startup confirmation: `--confirm-live`

The bot will **STOP** if any condition fails:
- Account type mismatch (DEMO config with LIVE account or vice versa)
- Account number doesn't match LIVE_ALLOWED_ACCOUNT
- LIVE_TRADING_ENABLED is false
- Missing required configuration

---

## 📊 Dashboard

Access at `http://localhost:8000` after starting `dashboard/backend/main.py`

### Pages:
- **Dashboard** - Real-time market data, SMC analysis, positions, controls
- **Positions** - Open positions with P/L, R-multiple, strategy
- **History** - Trade history with filters, sorting, CSV export
- **Performance** - Equity curve, monthly returns, win/loss distribution
- **Backtesting** - Run backtests, view reports, compare strategies
- **Settings** - Configure risk, sessions, strategies, trade management

---

## 🤖 Telegram Bot Commands

| Command | Description |
|---------|-------------|
| `/start` | Bot status overview |
| `/status` | System status (MT5, account, connection) |
| `/balance` | Account balance, equity, daily P/L |
| `/positions` | Open positions with P/L |
| `/history` | Recent closed trades |
| `/pnl` | Performance statistics |
| `/pause` | Pause new entries |
| `/resume` | Resume trading (with safety checks) |
| `/mode` | Show current trading mode |
| `/risk` | Show risk limits status |
| `/shutdown` | Safely stop the bot |

**Telegram clearly identifies mode**: All notifications show `[DEMO]` or `[LIVE]` prefix

---

## 🧪 Backtesting

```bash
# Via Dashboard -> Backtesting page
# Or programmatically:
python -c "
from backtesting.engine import run_backtest_async
from src.database_manager import db_manager, BacktestConfig
from datetime import datetime

config = BacktestConfig(
    name='Test Backtest',
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31),
    initial_balance=10000,
    risk_percent=2.0,
    min_risk_reward=2.0
)
saved = db_manager.add(config)
report = run_backtest_async(saved)
print(f'Net Profit: \${report.net_profit:.2f}')
print(f'Win Rate: {report.win_rate:.1f}%')
print(f'Total Trades: {report.total_trades}')
"
```

### Reports Include:
- Net profit, gross profit/loss, profit factor
- Win rate, max drawdown, average win/loss
- Expectancy, Sharpe ratio, Sortino ratio
- Consecutive wins/losses
- Individual trade history with SMC confirmations

---

## 📁 Project Structure

```
xauusd_smc_agent/
├── config/
│   ├── settings.yaml       # Main configuration (dual-mode)
│   └── logging.yaml        # Logging configuration
├── src/
│   ├── main.py             # Trading agent with dual-mode startup
│   ├── config_loader.py    # Dual-mode config loader
│   ├── logger.py           # Logging + Telegram handler
│   ├── database_manager.py # SQLite persistence
│   ├── mt5_connector.py    # MT5 connection & trading
│   ├── market_data.py      # Live/historical data engine
│   ├── indicators.py       # SMC indicators
│   ├── market_structure.py # BOS, CHoCH, trend analysis
│   ├── liquidity.py        # Liquidity levels & sweeps
│   ├── order_blocks.py     # Order block detection
│   ├── fair_value_gaps.py  # FVG detection
│   ├── premium_discount.py # Premium/discount zones
│   ├── smc_strategy.py     # Strategy engine (6 strategies)
│   ├── signal_validator.py # Pre-trade validation
│   ├── risk_manager.py     # Position sizing & risk limits
│   ├── trade_manager.py    # SL/TP, break-even, trailing
│   ├── order_executor.py   # Order execution & persistence
│   ├── session_filter.py   # Trading session logic
│   ├── news_filter.py      # Economic calendar filter
│   └── telegram_bot.py     # Telegram notifications & control
├── backtesting/
│   ├── engine.py           # Backtesting engine
│   ├── metrics.py          # Performance calculations
│   └── reports.py          # Report generation & export
├── dashboard/
│   ├── backend/main.py     # FastAPI + WebSocket server
│   └── frontend/           # React dashboard
├── tests/
│   └── test_mt5_connector.py
├── data/historical/        # Historical data cache
├── database/               # SQLite database
├── logs/                   # Application logs
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🎯 SMC Strategies Implemented

1. **Liquidity Sweep + FVG** - M15 liquidity sweep + M5 FVG entry
2. **Liquidity Sweep + CHoCH** - M15 sweep + M15 CHoCH confirmation
3. **BOS + Order Block** - H1 BOS + M15 Order Block
4. **CHoCH + Order Block** - M15 CHoCH + M5 Order Block
5. **Displacement + FVG** - M5 displacement candle + M5 FVG
6. **Liquidity Sweep + OB + FVG** - Highest confluence (M15 sweep + OB + M5 FVG)

---

## ⚠️ Risk Management

| Limit | DEMO | LIVE | Configurable |
|-------|------|------|--------------|
| Max risk/trade | 2% | 2% (max 3%) | Yes |
| Max daily loss | 5% | 5% | Yes |
| Max drawdown | 10% | 10% | Yes |
| Max positions | 3 | 3 | Yes |
| Max daily trades | 20 | 20 | Yes |
| Max consecutive losses | 5 | 5 | Yes |
| Max holding time | 48h | 48h | Yes |
| Spread limit | 5 pips | 5 pips | Yes |

**Key Rules:**
- 2% equity-based risk per trade (maximum)
- Automatic equity compounding
- No martingale, grid, or averaging down
- Emergency shutdown at 15% drawdown
- Minimum 1:2 risk/reward required

---

## 🔧 Commands Reference

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python -c "from src.database_manager import db_manager; db_manager.create_tables()"

# Run trading agent (DEMO)
python -m src.main

# Run trading agent (LIVE) - requires explicit confirmation
python -m src.main --mode LIVE --confirm-live

# Run dashboard backend
python -m dashboard.backend.main

# Build frontend
cd dashboard/frontend && npm run build

# Run tests
pytest tests/ -v

# Run specific test
pytest tests/test_mt5_connector.py::TestSignalValidator::test_validate_signal_success -v
```

---

## ⚠️ Important Notes

1. **DEMO mode is default** - Bot starts in DEMO unless explicitly configured
2. **LIVE mode requires explicit activation** - Multiple safety layers prevent accidental live trading
3. **Account verification is mandatory** - Bot verifies actual MT5 account type matches selected mode
4. **Never bypass safety checks** - The architecture is designed to prevent accidental live trading
5. **Test thoroughly in DEMO** - Run for at least 1-3 months before considering LIVE
6. **Backtesting is separate** - Backtest results do not reflect live performance

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🤝 Support

For issues, check logs in `logs/trading.log` and `logs/errors.log`.