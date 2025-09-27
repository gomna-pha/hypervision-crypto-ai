#!/usr/bin/env python3
"""
HyperVision API Server
Production-grade REST and WebSocket API for the hyperbolic HFT platform
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
import asyncio
import json
import logging
import uvicorn
from pathlib import Path
import numpy as np

# Import our platform
from hyperbolic_hft_platform import (
    HyperVisionPlatform,
    MarketData,
    Order,
    ArbitrageOpportunity,
    OrderSide,
    OrderType,
    OrderStatus,
    ArbitrageType
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="HyperVision HFT API",
    description="Professional Hyperbolic HFT Platform API",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global platform instance
platform: Optional[HyperVisionPlatform] = None
websocket_clients: List[WebSocket] = []

# ==================== API Models ====================

class SystemStatus(BaseModel):
    status: str
    timestamp: datetime
    uptime_seconds: float
    connections: int
    platform_metrics: Dict[str, Any]

class TradingConfig(BaseModel):
    symbols: List[str]
    exchanges: List[str]
    max_position_size: float
    max_daily_loss: float
    min_profit_threshold: float
    auto_execute: bool

class ExecutionRequest(BaseModel):
    opportunity_id: str
    max_slippage: Optional[float] = 0.001
    timeout_ms: Optional[int] = 1000

class RiskParameters(BaseModel):
    var_confidence: float = Field(default=0.95, ge=0.9, le=0.99)
    max_drawdown: float = Field(default=0.1, ge=0.01, le=0.5)
    position_limit: float = Field(default=1000000, gt=0)
    daily_loss_limit: float = Field(default=50000, gt=0)

# ==================== API Endpoints ====================

@app.get("/")
async def root():
    """Serve the dashboard"""
    with open("hypervision_dashboard.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.get("/api/status")
async def get_status() -> SystemStatus:
    """Get system status"""
    global platform
    
    if platform is None:
        raise HTTPException(status_code=503, detail="Platform not initialized")
    
    return SystemStatus(
        status="online",
        timestamp=datetime.now(),
        uptime_seconds=0,  # Would track actual uptime
        connections=len(websocket_clients),
        platform_metrics=platform.performance_metrics
    )

@app.get("/api/metrics")
async def get_metrics():
    """Get platform metrics"""
    global platform
    
    if platform is None:
        return JSONResponse(content={"error": "Platform not initialized"}, status_code=503)
    
    metrics = {
        "performance": platform.performance_metrics,
        "execution": platform.executor.execution_stats,
        "risk": {
            "current_daily_pnl": platform.risk_manager.current_daily_pnl,
            "position_limits": platform.risk_manager.position_limits,
            "max_position_size": platform.risk_manager.max_position_size,
            "max_daily_loss": platform.risk_manager.max_daily_loss
        },
        "data_pipeline": {
            "market_buffer_size": len(platform.data_pipeline.market_buffer),
            "social_buffer_size": len(platform.data_pipeline.social_buffer),
            "macro_buffer_size": len(platform.data_pipeline.macro_buffer)
        }
    }
    
    return JSONResponse(content=metrics)

@app.get("/api/opportunities")
async def get_opportunities(limit: int = 10):
    """Get current arbitrage opportunities"""
    global platform
    
    if platform is None:
        return JSONResponse(content={"error": "Platform not initialized"}, status_code=503)
    
    # Get recent opportunities from history
    opportunities = list(platform.arbitrage_engine.opportunity_history)[-limit:]
    
    # Convert to JSON-serializable format
    opp_data = []
    for opp in opportunities:
        opp_data.append({
            "opportunity_id": opp.opportunity_id,
            "type": opp.arbitrage_type.value,
            "symbols": opp.symbols,
            "exchanges": opp.exchanges,
            "expected_profit": opp.expected_profit,
            "probability": opp.probability,
            "risk_score": opp.risk_score,
            "latency_window_ms": opp.latency_window_ms,
            "hyperbolic_distance": opp.hyperbolic_distance,
            "timestamp": opp.timestamp.isoformat(),
            "entry_prices": opp.entry_prices,
            "exit_prices": opp.exit_prices,
            "volume_limits": opp.volume_limits
        })
    
    return JSONResponse(content={"opportunities": opp_data})

@app.post("/api/execute")
async def execute_opportunity(request: ExecutionRequest):
    """Execute a specific arbitrage opportunity"""
    global platform
    
    if platform is None:
        return JSONResponse(content={"error": "Platform not initialized"}, status_code=503)
    
    # Find the opportunity
    opportunity = None
    for opp in platform.arbitrage_engine.opportunity_history:
        if opp.opportunity_id == request.opportunity_id:
            opportunity = opp
            break
    
    if opportunity is None:
        raise HTTPException(status_code=404, detail="Opportunity not found")
    
    # Execute the opportunity
    orders = await platform.executor.execute_arbitrage(opportunity)
    
    # Convert orders to JSON-serializable format
    orders_data = []
    for order in orders:
        orders_data.append({
            "order_id": order.order_id,
            "symbol": order.symbol,
            "exchange": order.exchange,
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": order.quantity,
            "price": order.price,
            "status": order.status.value,
            "filled_quantity": order.filled_quantity,
            "average_fill_price": order.average_fill_price,
            "fees": order.fees,
            "timestamp": order.timestamp.isoformat()
        })
    
    return JSONResponse(content={"orders": orders_data})

@app.get("/api/orders")
async def get_orders(limit: int = 50):
    """Get recent orders"""
    global platform
    
    if platform is None:
        return JSONResponse(content={"error": "Platform not initialized"}, status_code=503)
    
    orders = platform.executor.executed_orders[-limit:]
    
    orders_data = []
    for order in orders:
        orders_data.append({
            "order_id": order.order_id,
            "symbol": order.symbol,
            "exchange": order.exchange,
            "side": order.side.value,
            "type": order.order_type.value,
            "quantity": order.quantity,
            "price": order.price,
            "status": order.status.value,
            "filled_quantity": order.filled_quantity,
            "average_fill_price": order.average_fill_price,
            "fees": order.fees,
            "timestamp": order.timestamp.isoformat()
        })
    
    return JSONResponse(content={"orders": orders_data})

@app.get("/api/market-data")
async def get_market_data(symbol: Optional[str] = None):
    """Get current market data"""
    global platform
    
    if platform is None:
        return JSONResponse(content={"error": "Platform not initialized"}, status_code=503)
    
    market_data = list(platform.data_pipeline.market_buffer)
    
    if symbol:
        market_data = [d for d in market_data if d.symbol == symbol]
    
    # Convert to JSON-serializable format
    data = []
    for md in market_data[-100:]:  # Last 100 data points
        data.append({
            "timestamp": md.timestamp.isoformat(),
            "symbol": md.symbol,
            "exchange": md.exchange,
            "bid": md.bid,
            "ask": md.ask,
            "bid_volume": md.bid_volume,
            "ask_volume": md.ask_volume,
            "last_price": md.last_price,
            "volume_24h": md.volume_24h,
            "vwap": md.vwap,
            "spread": md.spread,
            "order_book_imbalance": md.order_book_imbalance
        })
    
    return JSONResponse(content={"market_data": data})

@app.post("/api/config")
async def update_config(config: TradingConfig):
    """Update trading configuration"""
    global platform
    
    if platform is None:
        return JSONResponse(content={"error": "Platform not initialized"}, status_code=503)
    
    # Update platform configuration
    platform.symbols = config.symbols
    platform.exchanges = config.exchanges
    platform.risk_manager.max_position_size = config.max_position_size
    platform.risk_manager.max_daily_loss = config.max_daily_loss
    
    return JSONResponse(content={"status": "Configuration updated"})

@app.post("/api/risk-params")
async def update_risk_parameters(params: RiskParameters):
    """Update risk management parameters"""
    global platform
    
    if platform is None:
        return JSONResponse(content={"error": "Platform not initialized"}, status_code=503)
    
    platform.risk_manager.max_position_size = params.position_limit
    platform.risk_manager.max_daily_loss = params.daily_loss_limit
    
    return JSONResponse(content={"status": "Risk parameters updated"})

@app.get("/api/embeddings")
async def get_embeddings():
    """Get current hyperbolic embeddings visualization data"""
    global platform
    
    if platform is None:
        return JSONResponse(content={"error": "Platform not initialized"}, status_code=503)
    
    # Generate sample embedding data for visualization
    # In production, this would return actual embedding coordinates
    embeddings = []
    for i in range(20):
        r = np.random.random() * 0.9  # Keep within Poincaré disk
        theta = np.random.random() * 2 * np.pi
        
        embeddings.append({
            "id": i,
            "x": r * np.cos(theta),
            "y": r * np.sin(theta),
            "symbol": platform.symbols[i % len(platform.symbols)] if platform.symbols else f"Asset_{i}",
            "cluster": i % 3,
            "risk_score": np.random.random()
        })
    
    return JSONResponse(content={"embeddings": embeddings})

# ==================== WebSocket Endpoints ====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    websocket_clients.append(websocket)
    
    try:
        while True:
            # Send updates every second
            await asyncio.sleep(1)
            
            if platform:
                update = {
                    "type": "metrics_update",
                    "data": {
                        "performance": platform.performance_metrics,
                        "execution": platform.executor.execution_stats,
                        "timestamp": datetime.now().isoformat()
                    }
                }
                await websocket.send_json(update)
            
    except WebSocketDisconnect:
        websocket_clients.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in websocket_clients:
            websocket_clients.remove(websocket)

async def broadcast_opportunity(opportunity: ArbitrageOpportunity):
    """Broadcast new opportunity to all WebSocket clients"""
    message = {
        "type": "new_opportunity",
        "data": {
            "opportunity_id": opportunity.opportunity_id,
            "type": opportunity.arbitrage_type.value,
            "symbols": opportunity.symbols,
            "expected_profit": opportunity.expected_profit,
            "probability": opportunity.probability,
            "timestamp": opportunity.timestamp.isoformat()
        }
    }
    
    for client in websocket_clients:
        try:
            await client.send_json(message)
        except:
            websocket_clients.remove(client)

async def broadcast_execution(orders: List[Order]):
    """Broadcast execution results to all WebSocket clients"""
    message = {
        "type": "execution_update",
        "data": {
            "orders": [
                {
                    "order_id": order.order_id,
                    "symbol": order.symbol,
                    "status": order.status.value,
                    "filled_quantity": order.filled_quantity,
                    "average_fill_price": order.average_fill_price
                }
                for order in orders
            ],
            "timestamp": datetime.now().isoformat()
        }
    }
    
    for client in websocket_clients:
        try:
            await client.send_json(message)
        except:
            websocket_clients.remove(client)

# ==================== Platform Integration ====================

async def start_platform():
    """Start the HyperVision platform"""
    global platform
    
    logger.info("Initializing HyperVision Platform...")
    platform = HyperVisionPlatform()
    
    # Start platform in background
    asyncio.create_task(platform.run())
    logger.info("Platform started successfully")

@app.on_event("startup")
async def startup_event():
    """Initialize platform on startup"""
    await start_platform()

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down platform...")
    # Cleanup code here

# ==================== Health Check ====================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "platform": "initialized" if platform else "not_initialized"
    }

# ==================== Main Entry Point ====================

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )