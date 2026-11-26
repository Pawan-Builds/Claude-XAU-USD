# Claude-XAU-USD

A project for [Claude-XAU-USD](https://github.com/Pawan-Builds/Claude-XAU-USD)

## Description

XAU/USD SMC (Smart Money Concepts) algorithmic trading platform for
MetaTrader 5. Multi-component architecture:

- `main.py` - async trading agent (analysis -> plan -> gate -> submit -> manage)
- `src/` - strategy, risk, execution, connector, persistence modules
- `dashboard/backend/main.py` - FastAPI dashboard backend
- `dashboard/frontend/` - React dashboard UI
- `config/settings.yaml` - strategy / risk / session configuration
- `.env` - credentials and runtime environment

Safety model: every new entry must pass the central fail-closed
`EntryGate`; every opening submission must also carry a single-use
gate token verified at the MT5 connector boundary. DEMO mode is the
default; LIVE is refused unless every LIVE guard passes.

## Prerequisites

- Python 3.11+
- MetaTrader 5 terminal (Windows) with the target broker account
- Python packages: `pip install -r requirements.txt`

## Configuration

1. **`config/settings.yaml`** - committed, non-secret: timeframes, SMC
   parameters, risk limits, session windows, trade management, dashboard
   host/port, safety flags (`safety.demo_only`).
2. **`.env`** - local, secret: credentials and environment.

Required `.env` keys for **DEMO**:
TRADING_MODE=DEMO
DEMO_MT5_LOGIN=<numeric>
DEMO_MT5_PASSWORD=<password>
DEMO_MT5_SERVER=<broker server>
DEMO_MT5_PATH=<full path to terminal64.exe>
DASHBOARD_SECRET_KEY=<long random string>


**LIVE requires ALL of the following**:
TRADING_MODE=LIVE
LIVE_MT5_LOGIN, LIVE_MT5_PASSWORD, LIVE_MT5_SERVER, LIVE_MT5_PATH
LIVE_TRADING_ENABLED=true
LIVE_ALLOWED_ACCOUNT=<numeric MT5 login>
LIVE_TRADING_CONFIRMATION=I_ACCEPT_LIVE_TRADING_RISK


and in `config/settings.yaml`:
safety:
demo_only: false


`LIVE_TRADING_CONFIRMATION` must match `I_ACCEPT_LIVE_TRADING_RISK`
exactly - both startup and the connector submission boundary enforce it.

## Startup

1. Activate the virtual environment.
2. Start the trading agent:
python main.py
A single-instance lock at `bot.lock` prevents concurrent agents.
3. Start the dashboard backend:
uvicorn dashboard.backend.main:app --host 127.0.0.1 --port 8000
Endpoints are localhost-only and require a Bearer token equal to
`DASHBOARD_SECRET_KEY`. The only unauthenticated endpoint is
`/api/health`.

## Shutdown

Send `SIGINT` or `SIGTERM`. Graceful shutdown stops new entries,
cancels bot-owned pending orders when
`trading.pending_orders_on_shutdown: "cancel"`, verifies via fresh
broker query that none remain, and releases the instance lock if this
process still owns it.

The Kill switch state and unresolved-reconciliation markers persist
in the `AppSettings` table across restarts. Reset the Kill switch via
dashboard `POST /api/control/kill_switch` with
`{"active": false, "confirm": "I_UNDERSTAND"}`.

## Recovery

On every startup, before enabling entries:

- Syncs live MT5 positions with `ExecutedOrder` / `OpenPosition` rows.
- Reconstructs missing `ClosedTrade` rows from MT5 deal history.
- Retries any order request whose prior outcome was uncertain.
- Blocks all new entries until reconciliation succeeds.

If broker state cannot be verified, entries stay paused. The agent
will not resume until `EntryGate.probe()` and
`TradeManager.reconcile_with_broker()` both succeed.

## Risk configuration

`POST /api/risk/config` persists risk overrides (risk %, max positions,
daily loss, max drawdown, compounding) in `AppSettings`:

- Tightening is accepted directly.
- Weakening requires `"confirm": "I_UNDERSTAND"`.
- Values above the YAML cap are always rejected; raise the cap in
`config/settings.yaml` instead.
- The running agent does not hot-reload overrides; edit YAML and
restart.

## Contributing

Internal project - see repository owner.

## License

See repository owner.