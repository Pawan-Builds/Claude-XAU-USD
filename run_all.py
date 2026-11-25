"""Combined launcher for the Trading Agent and Dashboard Backend."""
import asyncio
import threading
import time
import uvicorn

from src.database_manager import db_manager
from src.logger import setup_logging
from src.main import TradingAgent
from dashboard.backend.main import app

async def _run_agent():
    agent = TradingAgent()
    await agent.start()

if __name__ == "__main__":
    setup_logging()
    
    # 1. Ensure DB is migrated
    db_manager.create_tables()
    
    # 2. Start the trading agent in a background thread
    print("Starting Trading Agent...")
    agent_thread = threading.Thread(target=lambda: asyncio.run(_run_agent()), daemon=True)
    agent_thread.start()
    
    # 3. Wait for the agent to connect to MT5 and register
    time.sleep(5) 
    
    # 4. Start the dashboard backend (which serves the frontend)
    print("Starting Dashboard Backend...")
    uvicorn.run(app, host="127.0.0.1", port=8000)