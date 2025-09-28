#!/usr/bin/env python3
"""
Start the Real-time Trading WebSocket Server
"""

import asyncio
import sys
import signal
import logging
from pathlib import Path

# Add the current directory to the path
sys.path.insert(0, str(Path(__file__).parent))

from realtime_trading_engine import (
    MarketDataFeed, 
    RealtimeBacktester, 
    RealtimePaperTrader,
    start_websocket_server
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def main():
    """Main entry point for the real-time trading server"""
    logger.info("Starting Real-time Trading WebSocket Server...")
    
    # Initialize components
    market_feed = MarketDataFeed()
    backtest_engine = RealtimeBacktester(market_feed)
    paper_trader = RealtimePaperTrader(market_feed)
    
    # Start market data feed
    asyncio.create_task(market_feed.start())
    logger.info("Market data feed started")
    
    # Start WebSocket server
    server_task = asyncio.create_task(
        start_websocket_server(
            market_feed=market_feed,
            backtest_engine=backtest_engine,
            paper_trader=paper_trader,
            host='0.0.0.0',
            port=9000
        )
    )
    
    logger.info("WebSocket server running on ws://localhost:9000")
    logger.info("Connect your frontend to start receiving real-time data")
    
    # Handle shutdown gracefully
    def signal_handler(sig, frame):
        logger.info("Shutting down server...")
        for task in asyncio.all_tasks():
            task.cancel()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # Keep the server running
        await server_task
    except asyncio.CancelledError:
        logger.info("Server shutdown complete")
    except Exception as e:
        logger.error(f"Server error: {e}")
        raise

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)