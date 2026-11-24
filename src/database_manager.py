"""Database models and manager using SQLAlchemy."""
import os
from datetime import datetime
from typing import Optional, List, Dict, Any
from contextlib import contextmanager

from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, Boolean,
    Text, Index, ForeignKey, Enum as SQLEnum, func, BigInteger,
    inspect, text
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session, relationship
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.dialects.sqlite import JSON

from src.config_loader import config
from src.logger import get_logger

logger = get_logger(__name__)

Base = declarative_base()


class HistoricalCandle(Base):
    __tablename__ = "historical_candles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    timeframe = Column(String(10), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    spread = Column(Float, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    __table_args__ = (
        Index("ix_candle_unique", "symbol", "timeframe", "timestamp", unique=True),
    )


class AccountSnapshot(Base):
    __tablename__ = "account_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    balance = Column(Float, nullable=False)
    equity = Column(Float, nullable=False)
    margin = Column(Float, default=0)
    free_margin = Column(Float, default=0)
    margin_level = Column(Float, default=0)
    floating_pl = Column(Float, default=0)
    daily_pl = Column(Float, default=0)
    environment = Column(String(20), nullable=False, default="demo")  # demo, backtest, paper


class TradingSignal(Base):
    __tablename__ = "trading_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    direction = Column(String(10), nullable=False)  # BUY, SELL
    entry_price = Column(Float, nullable=False)
    sl_price = Column(Float, nullable=False)
    tp_price = Column(Float, nullable=False)
    lot_size = Column(Float, nullable=False)
    risk_percent = Column(Float, nullable=False)
    risk_reward = Column(Float, nullable=False)
    strategy = Column(String(50), nullable=False)
    confirmations = Column(JSON, default=list)
    status = Column(String(20), default="pending")  # pending, executed, rejected, expired
    executed_order_id = Column(Integer, nullable=True)
    environment = Column(String(20), default="demo")
    fingerprint = Column(String(64), nullable=True, index=True)

class TradePlan(Base):
    __tablename__ = "trade_plans"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)  # BUY, SELL
    strategy = Column(String(50), nullable=False)
    entry_price = Column(Float, nullable=False)
    sl_price = Column(Float, nullable=False)
    tp_price = Column(Float, nullable=False)
    risk_reward = Column(Float, nullable=False)
    status = Column(String(20), default="WAITING")  # WAITING, EXECUTED, INVALIDATED, EXPIRED
    expiry_time = Column(DateTime, nullable=False)
    reason = Column(Text, nullable=True)
    invalidated_reason = Column(Text, nullable=True)
    environment = Column(String(20), default="demo")


class SMCDetection(Base):
    __tablename__ = "smc_detections"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    symbol = Column(String(20), nullable=False)
    timeframe = Column(String(10), nullable=False)
    detection_type = Column(String(30), nullable=False)  # BOS, CHoCH, FVG, OB, LIQUIDITY_SWEEP, etc.
    direction = Column(String(10), nullable=False)  # bullish, bearish
    price_level = Column(Float, nullable=False)
    start_time = Column(DateTime, nullable=True)
    end_time = Column(DateTime, nullable=True)
    details = Column(JSON, default=dict)
    is_active = Column(Boolean, default=True)
    mitigated_at = Column(DateTime, nullable=True)


class OrderRequest(Base):
    __tablename__ = "order_requests"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True, default=datetime.utcnow)
    signal_id = Column(Integer, ForeignKey("trading_signals.id"), nullable=True)
    magic_number = Column(Integer, nullable=False)
    symbol = Column(String(20), nullable=False)
    order_type = Column(String(20), nullable=False)  # BUY, SELL, BUY_LIMIT, SELL_LIMIT, etc.
    volume = Column(Float, nullable=False)
    price = Column(Float, nullable=True)
    sl = Column(Float, nullable=True)
    tp = Column(Float, nullable=True)
    comment = Column(String(100), nullable=True)
    request_data = Column(JSON, default=dict)
    response_data = Column(JSON, default=dict)
    status = Column(String(20), default="pending")  # pending, sent, filled, rejected, cancelled
    error_code = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)
    environment = Column(String(20), default="demo")
    fingerprint = Column(String(64), nullable=True, index=True)
    mt5_ticket = Column(BigInteger, nullable=True, index=True)
    expires_at = Column(DateTime, nullable=True)


class ExecutedOrder(Base):
    __tablename__ = "executed_orders"

    id = Column(Integer, primary_key=True, autoincrement=True)
    order_request_id = Column(Integer, ForeignKey("order_requests.id"), nullable=True)
    ticket = Column(BigInteger, nullable=False, unique=True, index=True)
    magic_number = Column(Integer, nullable=False)
    symbol = Column(String(20), nullable=False)
    order_type = Column(String(20), nullable=False)
    volume = Column(Float, nullable=False)
    price_open = Column(Float, nullable=False)
    price_current = Column(Float, default=0)
    sl = Column(Float, default=0)
    tp = Column(Float, default=0)
    swap = Column(Float, default=0)
    profit = Column(Float, default=0)
    commission = Column(Float, default=0)
    comment = Column(String(100), nullable=True)
    time_open = Column(DateTime, nullable=False, index=True)
    time_close = Column(DateTime, nullable=True, index=True)
    time_update = Column(DateTime, default=datetime.utcnow)
    reason = Column(String(30), nullable=True)  # SL, TP, MANUAL, etc.
    environment = Column(String(20), default="demo")


class OpenPosition(Base):
    __tablename__ = "open_positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket = Column(BigInteger, nullable=False, unique=True, index=True)
    magic_number = Column(Integer, nullable=False)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)  # BUY, SELL
    volume = Column(Float, nullable=False)
    price_open = Column(Float, nullable=False)
    price_current = Column(Float, default=0)
    sl = Column(Float, default=0)
    tp = Column(Float, default=0)
    swap = Column(Float, default=0)
    profit = Column(Float, default=0)
    commission = Column(Float, default=0)
    strategy = Column(String(50), nullable=True)
    time_open = Column(DateTime, nullable=False)
    time_update = Column(DateTime, default=datetime.utcnow)
    break_even_activated = Column(Boolean, default=False)
    trailing_active = Column(Boolean, default=False)
    partial_tp_taken = Column(JSON, default=list)
    initial_sl = Column(Float, nullable=False, default=0.0)
    initial_tp = Column(Float, nullable=False, default=0.0)
    initial_risk_distance = Column(Float, nullable=False, default=0.0)
    remaining_volume = Column(Float, nullable=False, default=0.0)
    highest_r = Column(Float, nullable=False, default=0.0)
    lowest_r = Column(Float, nullable=False, default=0.0)
    management_state = Column(JSON, nullable=True)
    environment = Column(String(20), default="demo")


class ClosedTrade(Base):
    __tablename__ = "closed_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket = Column(BigInteger, nullable=False, index=True)
    magic_number = Column(Integer, nullable=False)
    symbol = Column(String(20), nullable=False)
    direction = Column(String(10), nullable=False)
    volume = Column(Float, nullable=False)
    price_open = Column(Float, nullable=False)
    price_close = Column(Float, nullable=False)
    sl = Column(Float, default=0)
    tp = Column(Float, default=0)
    profit = Column(Float, nullable=False)
    swap = Column(Float, default=0)
    commission = Column(Float, default=0)
    net_profit = Column(Float, nullable=False)
    risk_percent = Column(Float, nullable=False)
    realized_r = Column(Float, nullable=False)
    strategy = Column(String(50), nullable=True)
    entry_reason = Column(Text, nullable=True)
    exit_reason = Column(String(30), nullable=True)  # SL, TP, MANUAL, BE, TRAILING, TIME
    time_open = Column(DateTime, nullable=False)
    time_close = Column(DateTime, nullable=False, index=True)
    duration_hours = Column(Float, default=0)
    environment = Column(String(20), default="demo")


class TradeManagementEvent(Base):
    __tablename__ = "trade_management_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticket = Column(BigInteger, nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    event_type = Column(String(30), nullable=False)  # SL_MODIFIED, TP_MODIFIED, BE_ACTIVATED, TRAILING_UPDATED, PARTIAL_TP
    old_value = Column(Float, nullable=True)
    new_value = Column(Float, nullable=True)
    details = Column(JSON, default=dict)
    environment = Column(String(20), default="demo")


class RiskCalculation(Base):
    __tablename__ = "risk_calculations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow)
    equity = Column(Float, nullable=False)
    risk_percent = Column(Float, nullable=False)
    sl_distance_pips = Column(Float, nullable=False)
    tick_value = Column(Float, nullable=False)
    tick_size = Column(Float, nullable=False)
    contract_size = Column(Float, nullable=False)
    calculated_lot = Column(Float, nullable=False)
    adjusted_lot = Column(Float, nullable=False)
    monetary_risk = Column(Float, nullable=False)
    environment = Column(String(20), default="demo")


class BacktestConfig(Base):
    __tablename__ = "backtest_configs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_balance = Column(Float, nullable=False)
    risk_percent = Column(Float, nullable=False)
    min_risk_reward = Column(Float, nullable=False)
    timeframes = Column(JSON, default=list)
    sessions = Column(JSON, default=list)
    strategies = Column(JSON, default=list)
    spread_pips = Column(Float, default=0)
    slippage_pips = Column(Float, default=0)
    commission_per_lot = Column(Float, default=0)
    swap_enabled = Column(Boolean, default=True)
    data_source = Column(String(20), default="mt5")
    created_at = Column(DateTime, default=datetime.utcnow)


class BacktestReport(Base):
    __tablename__ = "backtest_reports"

    id = Column(Integer, primary_key=True, autoincrement=True)
    config_id = Column(Integer, ForeignKey("backtest_configs.id"), nullable=True)
    start_date = Column(DateTime, nullable=False)
    end_date = Column(DateTime, nullable=False)
    initial_capital = Column(Float, nullable=False)
    final_capital = Column(Float, nullable=False)
    gross_profit = Column(Float, default=0)
    gross_loss = Column(Float, default=0)
    net_profit = Column(Float, default=0)
    net_return_pct = Column(Float, default=0)
    total_trades = Column(Integer, default=0)
    winning_trades = Column(Integer, default=0)
    losing_trades = Column(Integer, default=0)
    win_rate = Column(Float, default=0)
    profit_factor = Column(Float, default=0)
    max_drawdown = Column(Float, default=0)
    max_drawdown_pct = Column(Float, default=0)
    avg_risk_reward = Column(Float, default=0)
    avg_realized_r = Column(Float, default=0)
    expectancy = Column(Float, default=0)
    sharpe_ratio = Column(Float, default=0)
    sortino_ratio = Column(Float, default=0)
    recovery_factor = Column(Float, default=0)
    largest_win = Column(Float, default=0)
    largest_loss = Column(Float, default=0)
    avg_holding_hours = Column(Float, default=0)
    max_consecutive_wins = Column(Integer, default=0)
    max_consecutive_losses = Column(Integer, default=0)
    long_trades = Column(Integer, default=0)
    long_win_rate = Column(Float, default=0)
    long_profit = Column(Float, default=0)
    short_trades = Column(Integer, default=0)
    short_win_rate = Column(Float, default=0)
    short_profit = Column(Float, default=0)
    completed_at = Column(DateTime, default=datetime.utcnow)
    environment = Column(String(20), default="backtest")


class BacktestTrade(Base):
    __tablename__ = "backtest_trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(Integer, ForeignKey("backtest_reports.id"), nullable=False)
    ticket = Column(BigInteger, default=0)
    direction = Column(String(10), nullable=False)
    entry_time = Column(DateTime, nullable=False)
    exit_time = Column(DateTime, nullable=False)
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float, nullable=False)
    sl = Column(Float, default=0)
    tp = Column(Float, default=0)
    volume = Column(Float, nullable=False)
    risk_percent = Column(Float, default=0)
    realized_r = Column(Float, default=0)
    gross_profit = Column(Float, default=0)
    net_profit = Column(Float, default=0)
    strategy = Column(String(50), nullable=True)
    entry_reason = Column(Text, nullable=True)
    exit_reason = Column(String(30), nullable=True)
    bos_confirmed = Column(Boolean, default=False)
    choch_confirmed = Column(Boolean, default=False)
    liquidity_sweep = Column(Boolean, default=False)
    ob_confirmed = Column(Boolean, default=False)
    fvg_confirmed = Column(Boolean, default=False)
    session = Column(String(20), nullable=True)


class AppSettings(Base):
    __tablename__ = "app_settings"

    id = Column(Integer, primary_key=True, autoincrement=True)
    key = Column(String(100), nullable=False, unique=True)
    value = Column(Text, nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ErrorLog(Base):
    __tablename__ = "error_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    level = Column(String(20), nullable=False)
    logger = Column(String(100), nullable=False)
    message = Column(Text, nullable=False)
    traceback = Column(Text, nullable=True)
    context = Column(JSON, default=dict)
    resolved = Column(Boolean, default=False)


class DatabaseManager:
    """Manages database connections, sessions, and safe schema migrations."""

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or config.database.url
        self.engine = None
        self.SessionLocal = None
        self._init_engine()

    def _init_engine(self):
        """Initialize the database engine safely."""

        is_sqlite = self.database_url.startswith("sqlite://")

        if is_sqlite:
            db_path = self.database_url.replace("sqlite:///", "", 1)

            if db_path and db_path != ":memory:":
                directory = os.path.dirname(db_path)
                if directory:
                    os.makedirs(directory, exist_ok=True)

            self.engine = create_engine(
                self.database_url,
                echo=config.database.echo,
                connect_args={
                    "check_same_thread": False,
                    "timeout": 30
                },
                pool_pre_ping=True
            )
        else:
            self.engine = create_engine(
                self.database_url,
                echo=config.database.echo,
                pool_size=config.database.pool_size,
                max_overflow=config.database.max_overflow,
                pool_pre_ping=True
            )

        self.SessionLocal = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False
        )

        logger.info("Database engine initialized")

    def create_tables(self):
        """Create missing tables and apply safe additive migrations."""

        Base.metadata.create_all(bind=self.engine)
        self._migrate_schema()

        logger.info("Database tables created and schema verified")

    def _migrate_schema(self):
        """Additive, idempotent column migrations.

        FIX #27: every DDL statement and backfill runs inside a single
        `engine.begin()` transaction. Previously the account_snapshots
        index was created after the with-block exited, which raised
        ResourceClosedError: "This Connection is closed".
        """
        migrations_map = {
            "open_positions": {
                "initial_sl": "FLOAT NOT NULL DEFAULT 0.0",
                "initial_tp": "FLOAT NOT NULL DEFAULT 0.0",
                "initial_risk_distance": "FLOAT NOT NULL DEFAULT 0.0",
                "remaining_volume": "FLOAT NOT NULL DEFAULT 0.0",
                "highest_r": "FLOAT NOT NULL DEFAULT 0.0",
                "lowest_r": "FLOAT NOT NULL DEFAULT 0.0",
                "management_state": "JSON",
            },
            "order_requests": {
                "mt5_ticket": "BIGINT",
                "expires_at": "DATETIME",
                "fingerprint": "VARCHAR(64)",
            },
            "trading_signals": {
                "fingerprint": "VARCHAR(64)",
            },
        }

        inspector = inspect(self.engine)
        table_names = inspector.get_table_names()

        added_columns: List[str] = []

        with self.engine.begin() as connection:
            # 1. Additive column migrations.
            for table, columns in migrations_map.items():
                if table not in table_names:
                    continue

                existing = {
                    col["name"] for col in inspector.get_columns(table)
                }

                for col_name, col_def in columns.items():
                    if col_name in existing:
                        continue

                    connection.exec_driver_sql(
                        f'ALTER TABLE "{table}" ADD COLUMN '
                        f'"{col_name}" {col_def}'
                    )
                    added_columns.append(f"{table}.{col_name}")

            # 2. Backfill defaults for the new columns.
            if "open_positions" in table_names:
                connection.execute(text("""
                    UPDATE open_positions
                    SET initial_sl = sl
                    WHERE initial_sl = 0 AND sl != 0
                """))
                connection.execute(text("""
                    UPDATE open_positions
                    SET initial_tp = tp
                    WHERE initial_tp = 0 AND tp != 0
                """))
                connection.execute(text("""
                    UPDATE open_positions
                    SET remaining_volume = volume
                    WHERE remaining_volume = 0 AND volume > 0
                """))
                connection.execute(text("""
                    UPDATE open_positions
                    SET initial_risk_distance = ABS(price_open - initial_sl)
                    WHERE initial_risk_distance = 0
                      AND initial_sl != 0
                """))

            # 3. Optional compound index for daily-loss queries.
            if "account_snapshots" in table_names:
                try:
                    connection.exec_driver_sql(
                        "CREATE INDEX IF NOT EXISTS ix_snapshots_env_ts "
                        "ON account_snapshots (environment, timestamp)"
                    )
                except Exception:
                    logger.exception(
                        "Could not create ix_snapshots_env_ts (non-fatal)"
                    )
                    
            # 4. Indexes for duplicate-order prevention
            if "trading_signals" in table_names:
                try:
                    connection.exec_driver_sql(
                        "CREATE INDEX IF NOT EXISTS ix_trading_signals_fp "
                        "ON trading_signals (fingerprint)"
                    )
                except Exception:
                    logger.exception("Could not create trading_signals_fp index (non-fatal)")

            if "order_requests" in table_names:
                try:
                    connection.exec_driver_sql(
                        "CREATE INDEX IF NOT EXISTS ix_order_requests_fp "
                        "ON order_requests (fingerprint)"
                    )
                except Exception:
                    logger.exception("Could not create order_requests_fp index (non-fatal)")
            if "order_requests" in table_names:
                try:
                    connection.exec_driver_sql(
                        "CREATE INDEX IF NOT EXISTS "
                        "ix_order_requests_mt5_ticket "
                        "ON order_requests (mt5_ticket)"
                    )
                except Exception:
                    logger.exception(
                        "Could not create order_requests_mt5_ticket "
                        "index (non-fatal)"
                    )

        if added_columns:
            logger.info(
                "Database migration added columns: %s",
                ", ".join(added_columns),
            )
        else:
            logger.info("Database schema is already up to date")

    def drop_tables(self):
        """Drop all tables. Destructive: do not use during normal operation."""
        logger.warning("Dropping all database tables")
        Base.metadata.drop_all(bind=self.engine)

    @contextmanager
    def get_session(self):
        """Provide a session with automatic commit and rollback."""

        session = self.SessionLocal()

        try:
            yield session
            session.commit()

        except Exception:
            session.rollback()
            logger.exception("Database transaction failed")
            raise

        finally:
            session.close()

    def execute_raw(self, query: str, params: Optional[Dict] = None) -> List[Dict]:
        """Execute a parameterized SQL query."""

        with self.get_session() as session:
            result = session.execute(text(query), params or {})
            return [dict(row._mapping) for row in result]

    # ---------------- CRUD HELPERS ----------------

    def add(self, obj: Base) -> int:
        """Insert one record and return its database ID."""

        with self.get_session() as session:
            session.add(obj)
            session.flush()
            obj_id = obj.id

        return obj_id

    def add_all(self, objects: List[Base]) -> List[int]:
        """Insert multiple records and return their IDs."""

        with self.get_session() as session:
            session.add_all(objects)
            session.flush()
            obj_ids = [obj.id for obj in objects]

        return obj_ids

    def get(self, model: type, id: int) -> Optional[Base]:
        """Get a record and return it detached from the session."""

        with self.SessionLocal() as session:
            result = session.query(model).filter(model.id == id).first()

            if result is not None:
                session.expunge(result)

            return result

    def query(self, model: type, **filters) -> List[Base]:
        """Query records using keyword filters."""

        with self.get_session() as session:
            return session.query(model).filter_by(**filters).all()

    def query_one(self, model: type, **filters) -> Optional[Base]:
        """Query one record using keyword filters."""

        with self.get_session() as session:
            return session.query(model).filter_by(**filters).first()

    def update(self, model: type, id: int, **updates) -> Optional[Base]:
        """Update a record and return its updated state."""

        with self.get_session() as session:
            obj = session.query(model).filter(model.id == id).first()

            if obj is None:
                return None

            valid_columns = {column.name for column in model.__table__.columns}

            for key, value in updates.items():
                if key not in valid_columns:
                    raise ValueError(
                        f"Invalid column '{key}' for model {model.__name__}"
                    )

                setattr(obj, key, value)

            session.flush()
            session.refresh(obj)

            return obj

    def delete(self, model: type, id: int) -> bool:
        """Delete a record by ID."""

        with self.get_session() as session:
            obj = session.query(model).filter(model.id == id).first()

            if obj is None:
                return False

            session.delete(obj)
            return True

    # ---------------- SPECIFIC QUERIES ----------------

    def get_latest_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 100
    ) -> List[HistoricalCandle]:

        with self.get_session() as session:
            return session.query(HistoricalCandle).filter(
                HistoricalCandle.symbol == symbol,
                HistoricalCandle.timeframe == timeframe
            ).order_by(
                HistoricalCandle.timestamp.desc()
            ).limit(limit).all()

    def get_open_positions(
        self,
        environment: str = "demo"
    ) -> List[OpenPosition]:

        with self.get_session() as session:
            return session.query(OpenPosition).filter(
                OpenPosition.environment == environment
            ).all()

    def get_closed_trades(
        self,
        environment: str = "demo",
        limit: int = 100
    ) -> List[ClosedTrade]:

        with self.get_session() as session:
            return session.query(ClosedTrade).filter(
                ClosedTrade.environment == environment
            ).order_by(
                ClosedTrade.time_close.desc()
            ).limit(limit).all()

    def get_account_snapshots(
        self,
        environment: str = "demo",
        limit: int = 1000
    ) -> List[AccountSnapshot]:

        with self.get_session() as session:
            return session.query(AccountSnapshot).filter(
                AccountSnapshot.environment == environment
            ).order_by(
                AccountSnapshot.timestamp.desc()
            ).limit(limit).all()

    def get_backtest_reports(
        self,
        limit: int = 50
    ) -> List[BacktestReport]:

        with self.get_session() as session:
            return session.query(BacktestReport).order_by(
                BacktestReport.completed_at.desc()
            ).limit(limit).all()

    def get_backtest_trades(
        self,
        report_id: int
    ) -> List[BacktestTrade]:

        with self.get_session() as session:
            return session.query(BacktestTrade).filter(
                BacktestTrade.report_id == report_id
            ).order_by(
                BacktestTrade.entry_time
            ).all()

    def get_active_smc_detections(
        self,
        symbol: str,
        timeframe: str,
        detection_type: str
    ) -> List[SMCDetection]:

        with self.get_session() as session:
            return session.query(SMCDetection).filter(
                SMCDetection.symbol == symbol,
                SMCDetection.timeframe == timeframe,
                SMCDetection.detection_type == detection_type,
                SMCDetection.is_active.is_(True)
            ).all()

    def update_smc_mitigated(
        self,
        detection_id: int,
        mitigated_at: datetime
    ):
        """Mark an SMC detection as mitigated."""

        with self.get_session() as session:
            detection = session.query(SMCDetection).filter(
                SMCDetection.id == detection_id
            ).first()

            if detection:
                detection.is_active = False
                detection.mitigated_at = mitigated_at

    def close(self):
        """Dispose of the database connection pool."""

        if self.engine:
            self.engine.dispose()
            logger.info("Database engine disposed")

# Global database manager instance
db_manager = DatabaseManager()