#!/usr/bin/env python3
"""
Paper Trading System for HyperVision Platform
Real-time simulation with virtual portfolio management
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
import pandas as pd
from collections import defaultdict, deque
import threading
import websocket
import redis
import pickle

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Data Models ====================

class OrderStatus(Enum):
    PENDING = "PENDING"
    OPEN = "OPEN"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"
    TRAILING_STOP = "TRAILING_STOP"

@dataclass
class PaperOrder:
    """Virtual order for paper trading"""
    order_id: str
    timestamp: datetime
    symbol: str
    side: str  # BUY or SELL
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PaperPosition:
    """Virtual position tracking"""
    symbol: str
    quantity: float
    average_entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    opened_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    
    def update_pnl(self, current_price: float):
        """Update unrealized P&L"""
        self.current_price = current_price
        self.unrealized_pnl = (current_price - self.average_entry_price) * self.quantity
        self.last_updated = datetime.now()

@dataclass
class PaperAccount:
    """Virtual trading account"""
    account_id: str
    initial_balance: float
    current_balance: float
    buying_power: float
    positions: Dict[str, PaperPosition] = field(default_factory=dict)
    orders: List[PaperOrder] = field(default_factory=list)
    trade_history: List[Dict] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    
    @property
    def total_equity(self) -> float:
        """Calculate total account equity"""
        positions_value = sum(
            pos.quantity * pos.current_price 
            for pos in self.positions.values()
        )
        return self.current_balance + positions_value
    
    @property
    def total_pnl(self) -> float:
        """Calculate total P&L"""
        unrealized = sum(pos.unrealized_pnl for pos in self.positions.values())
        realized = sum(pos.realized_pnl for pos in self.positions.values())
        return unrealized + realized

# ==================== Order Execution Simulator ====================

class OrderExecutionSimulator:
    """Simulates realistic order execution with market conditions"""
    
    def __init__(self, slippage_model: str = "realistic", 
                 commission_rate: float = 0.0002):
        self.slippage_model = slippage_model
        self.commission_rate = commission_rate
        self.market_impact_factor = 0.0001
        
    def simulate_market_order(self, order: PaperOrder, 
                             market_data: Dict) -> Tuple[float, float, float]:
        """Simulate market order execution"""
        bid = market_data.get('bid', 0)
        ask = market_data.get('ask', 0)
        last = market_data.get('last', (bid + ask) / 2)
        spread = ask - bid
        volume = market_data.get('volume', 10000)
        
        # Determine execution price based on side
        if order.side == "BUY":
            base_price = ask if ask > 0 else last * 1.0001
        else:
            base_price = bid if bid > 0 else last * 0.9999
        
        # Calculate slippage
        slippage = self._calculate_slippage(
            order.quantity, base_price, volume, spread
        )
        
        # Final execution price
        if order.side == "BUY":
            fill_price = base_price + slippage
        else:
            fill_price = base_price - slippage
        
        # Calculate commission
        commission = abs(order.quantity * fill_price * self.commission_rate)
        
        return fill_price, slippage, commission
    
    def simulate_limit_order(self, order: PaperOrder, 
                           market_data: Dict) -> Optional[Tuple[float, float, float]]:
        """Simulate limit order execution"""
        bid = market_data.get('bid', 0)
        ask = market_data.get('ask', 0)
        last = market_data.get('last', 0)
        
        # Check if limit order would fill
        if order.side == "BUY":
            if ask <= order.limit_price:
                # Order fills at limit price or better
                fill_price = min(ask, order.limit_price)
                slippage = 0  # No slippage on limit orders
                commission = abs(order.quantity * fill_price * self.commission_rate)
                return fill_price, slippage, commission
        else:  # SELL
            if bid >= order.limit_price:
                fill_price = max(bid, order.limit_price)
                slippage = 0
                commission = abs(order.quantity * fill_price * self.commission_rate)
                return fill_price, slippage, commission
        
        return None  # Order doesn't fill
    
    def simulate_stop_order(self, order: PaperOrder, 
                           market_data: Dict) -> Optional[Tuple[float, float, float]]:
        """Simulate stop order execution"""
        last = market_data.get('last', 0)
        
        # Check if stop is triggered
        if order.side == "BUY":
            if last >= order.stop_price:
                # Convert to market order
                return self.simulate_market_order(order, market_data)
        else:  # SELL
            if last <= order.stop_price:
                return self.simulate_market_order(order, market_data)
        
        return None  # Stop not triggered
    
    def _calculate_slippage(self, size: float, price: float, 
                           volume: float, spread: float) -> float:
        """Calculate realistic slippage"""
        if self.slippage_model == "zero":
            return 0
        elif self.slippage_model == "fixed":
            return price * 0.0001
        elif self.slippage_model == "realistic":
            # Market impact based on order size relative to volume
            size_impact = (size / volume) * self.market_impact_factor * price
            
            # Spread cost
            spread_cost = spread * 0.5
            
            # Random micro-structure noise
            noise = np.random.normal(0, price * 0.00001)
            
            return size_impact + spread_cost + noise
        else:
            return 0

# ==================== Paper Trading Engine ====================

class PaperTradingEngine:
    """Main paper trading engine with real-time simulation"""
    
    def __init__(self, initial_capital: float = 100000):
        self.accounts = {}
        self.executor = OrderExecutionSimulator()
        self.market_data_feed = {}
        self.is_running = False
        self.processing_thread = None
        self.order_queue = deque()
        self.initial_capital = initial_capital
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Risk management
        self.risk_limits = {
            'max_position_size': 10000,
            'max_daily_loss': 5000,
            'max_leverage': 2.0,
            'max_positions': 10
        }
        
    def create_account(self, user_id: str, 
                      initial_balance: Optional[float] = None) -> PaperAccount:
        """Create a new paper trading account"""
        account_id = f"paper_{user_id}_{uuid.uuid4().hex[:8]}"
        balance = initial_balance or self.initial_capital
        
        account = PaperAccount(
            account_id=account_id,
            initial_balance=balance,
            current_balance=balance,
            buying_power=balance * 2  # 2x leverage available
        )
        
        self.accounts[account_id] = account
        logger.info(f"Created paper account {account_id} with balance ${balance:,.2f}")
        
        return account
    
    async def place_order(self, account_id: str, order: PaperOrder) -> Dict[str, Any]:
        """Place a paper trading order"""
        if account_id not in self.accounts:
            return {"success": False, "error": "Account not found"}
        
        account = self.accounts[account_id]
        
        # Risk checks
        risk_check = self._check_risk_limits(account, order)
        if not risk_check['passed']:
            order.status = OrderStatus.REJECTED
            return {
                "success": False, 
                "error": risk_check['reason'],
                "order_id": order.order_id
            }
        
        # Add order to account
        order.status = OrderStatus.OPEN
        account.orders.append(order)
        
        # Queue for processing
        self.order_queue.append((account_id, order))
        
        logger.info(f"Order placed: {order.order_id} - {order.side} {order.quantity} {order.symbol}")
        
        return {
            "success": True,
            "order_id": order.order_id,
            "status": order.status.value
        }
    
    def _check_risk_limits(self, account: PaperAccount, 
                          order: PaperOrder) -> Dict[str, Any]:
        """Check if order passes risk limits"""
        # Check position size
        if order.quantity > self.risk_limits['max_position_size']:
            return {
                "passed": False,
                "reason": f"Order size exceeds limit of {self.risk_limits['max_position_size']}"
            }
        
        # Check number of positions
        if len(account.positions) >= self.risk_limits['max_positions']:
            if order.symbol not in account.positions:
                return {
                    "passed": False,
                    "reason": f"Maximum number of positions ({self.risk_limits['max_positions']}) reached"
                }
        
        # Check buying power
        estimated_cost = order.quantity * (order.limit_price or self.market_data_feed.get(
            order.symbol, {}).get('last', 0))
        
        if order.side == "BUY" and estimated_cost > account.buying_power:
            return {
                "passed": False,
                "reason": "Insufficient buying power"
            }
        
        # Check daily loss limit
        if account.total_pnl < -self.risk_limits['max_daily_loss']:
            return {
                "passed": False,
                "reason": "Daily loss limit reached"
            }
        
        return {"passed": True}
    
    async def process_orders(self):
        """Process pending orders against market data"""
        while self.is_running:
            if self.order_queue:
                account_id, order = self.order_queue.popleft()
                account = self.accounts[account_id]
                
                # Get current market data
                market_data = self.market_data_feed.get(order.symbol, {})
                
                if not market_data:
                    # Re-queue if no market data available
                    self.order_queue.append((account_id, order))
                    await asyncio.sleep(0.1)
                    continue
                
                # Process based on order type
                fill_result = None
                
                if order.order_type == OrderType.MARKET:
                    fill_result = self.executor.simulate_market_order(order, market_data)
                elif order.order_type == OrderType.LIMIT:
                    fill_result = self.executor.simulate_limit_order(order, market_data)
                elif order.order_type == OrderType.STOP:
                    fill_result = self.executor.simulate_stop_order(order, market_data)
                
                if fill_result:
                    fill_price, slippage, commission = fill_result
                    await self._execute_fill(account, order, fill_price, slippage, commission)
                else:
                    # Re-queue unfilled orders
                    if order.status == OrderStatus.OPEN:
                        self.order_queue.append((account_id, order))
            
            await asyncio.sleep(0.01)  # 10ms processing cycle
    
    async def _execute_fill(self, account: PaperAccount, order: PaperOrder,
                           fill_price: float, slippage: float, commission: float):
        """Execute order fill and update account"""
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.commission = commission
        order.slippage = slippage
        order.status = OrderStatus.FILLED
        
        # Update positions
        if order.side == "BUY":
            if order.symbol in account.positions:
                # Add to existing position
                position = account.positions[order.symbol]
                total_quantity = position.quantity + order.quantity
                total_cost = (position.quantity * position.average_entry_price + 
                             order.quantity * fill_price)
                position.quantity = total_quantity
                position.average_entry_price = total_cost / total_quantity
            else:
                # Create new position
                account.positions[order.symbol] = PaperPosition(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    average_entry_price=fill_price,
                    current_price=fill_price
                )
            
            # Update balance
            total_cost = order.quantity * fill_price + commission
            account.current_balance -= total_cost
            account.buying_power -= total_cost
            
        else:  # SELL
            if order.symbol in account.positions:
                position = account.positions[order.symbol]
                
                if position.quantity >= order.quantity:
                    # Calculate realized P&L
                    realized_pnl = (fill_price - position.average_entry_price) * order.quantity
                    position.realized_pnl += realized_pnl
                    
                    # Update position
                    position.quantity -= order.quantity
                    if position.quantity == 0:
                        del account.positions[order.symbol]
                    
                    # Update balance
                    proceeds = order.quantity * fill_price - commission
                    account.current_balance += proceeds
                    account.buying_power += proceeds
                else:
                    # Reject partial sell
                    order.status = OrderStatus.REJECTED
                    logger.warning(f"Rejected sell order - insufficient position")
                    return
        
        # Record trade
        trade_record = {
            'timestamp': datetime.now().isoformat(),
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.quantity,
            'fill_price': fill_price,
            'commission': commission,
            'slippage': slippage,
            'pnl': position.realized_pnl if order.side == "SELL" else 0
        }
        
        account.trade_history.append(trade_record)
        
        # Update performance metrics
        self.performance_tracker.record_trade(account_id, trade_record)
        
        logger.info(f"Order filled: {order.order_id} @ ${fill_price:.2f}")
    
    def update_market_data(self, symbol: str, data: Dict[str, float]):
        """Update market data feed"""
        self.market_data_feed[symbol] = {
            **data,
            'timestamp': datetime.now()
        }
        
        # Update position P&L
        for account in self.accounts.values():
            if symbol in account.positions:
                account.positions[symbol].update_pnl(data.get('last', 0))
    
    def get_account_summary(self, account_id: str) -> Dict[str, Any]:
        """Get account summary with performance metrics"""
        if account_id not in self.accounts:
            return {"error": "Account not found"}
        
        account = self.accounts[account_id]
        
        # Calculate metrics
        total_return = (account.total_equity - account.initial_balance) / account.initial_balance
        
        positions_summary = []
        for symbol, position in account.positions.items():
            positions_summary.append({
                'symbol': symbol,
                'quantity': position.quantity,
                'entry_price': position.average_entry_price,
                'current_price': position.current_price,
                'unrealized_pnl': position.unrealized_pnl,
                'realized_pnl': position.realized_pnl
            })
        
        return {
            'account_id': account_id,
            'initial_balance': account.initial_balance,
            'current_balance': account.current_balance,
            'total_equity': account.total_equity,
            'buying_power': account.buying_power,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_pnl': account.total_pnl,
            'positions': positions_summary,
            'open_orders': len([o for o in account.orders if o.status == OrderStatus.OPEN]),
            'total_trades': len(account.trade_history),
            'created_at': account.created_at.isoformat()
        }
    
    async def start(self):
        """Start paper trading engine"""
        self.is_running = True
        logger.info("Paper Trading Engine started")
        await self.process_orders()
    
    def stop(self):
        """Stop paper trading engine"""
        self.is_running = False
        logger.info("Paper Trading Engine stopped")

# ==================== Performance Tracking ====================

class PerformanceTracker:
    """Track and analyze paper trading performance"""
    
    def __init__(self):
        self.trades = defaultdict(list)
        self.daily_pnl = defaultdict(lambda: defaultdict(float))
        self.metrics = defaultdict(dict)
        
    def record_trade(self, account_id: str, trade: Dict):
        """Record a completed trade"""
        self.trades[account_id].append(trade)
        
        # Update daily P&L
        date = datetime.now().date()
        self.daily_pnl[account_id][date] += trade.get('pnl', 0)
        
    def calculate_metrics(self, account_id: str) -> Dict[str, float]:
        """Calculate performance metrics"""
        trades = self.trades.get(account_id, [])
        
        if not trades:
            return {}
        
        # Extract P&L values
        pnls = [t.get('pnl', 0) for t in trades]
        winning_trades = [p for p in pnls if p > 0]
        losing_trades = [p for p in pnls if p < 0]
        
        # Calculate metrics
        metrics = {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) if trades else 0,
            'total_pnl': sum(pnls),
            'avg_win': np.mean(winning_trades) if winning_trades else 0,
            'avg_loss': np.mean(losing_trades) if losing_trades else 0,
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
            'profit_factor': abs(sum(winning_trades) / sum(losing_trades)) 
                           if losing_trades else float('inf'),
            'expectancy': np.mean(pnls) if pnls else 0
        }
        
        # Calculate Sharpe ratio (simplified)
        if len(pnls) > 1:
            returns = pd.Series(pnls)
            metrics['sharpe_ratio'] = (returns.mean() / returns.std() * np.sqrt(252)) \
                                     if returns.std() > 0 else 0
        
        self.metrics[account_id] = metrics
        return metrics
    
    def get_equity_curve(self, account_id: str) -> List[Dict]:
        """Get equity curve data"""
        trades = self.trades.get(account_id, [])
        
        if not trades:
            return []
        
        equity_curve = []
        cumulative_pnl = 0
        
        for trade in trades:
            cumulative_pnl += trade.get('pnl', 0)
            equity_curve.append({
                'timestamp': trade['timestamp'],
                'cumulative_pnl': cumulative_pnl,
                'trade_pnl': trade.get('pnl', 0)
            })
        
        return equity_curve

# ==================== WebSocket API ====================

class PaperTradingWebSocketServer:
    """WebSocket server for real-time paper trading updates"""
    
    def __init__(self, engine: PaperTradingEngine, port: int = 8765):
        self.engine = engine
        self.port = port
        self.clients = set()
        
    async def handler(self, websocket, path):
        """Handle WebSocket connections"""
        self.clients.add(websocket)
        try:
            async for message in websocket:
                data = json.loads(message)
                response = await self.process_message(data)
                await websocket.send(json.dumps(response))
        finally:
            self.clients.remove(websocket)
    
    async def process_message(self, data: Dict) -> Dict:
        """Process incoming WebSocket messages"""
        msg_type = data.get('type')
        
        if msg_type == 'create_account':
            account = self.engine.create_account(
                data.get('user_id', 'anonymous')
            )
            return {
                'type': 'account_created',
                'account_id': account.account_id,
                'initial_balance': account.initial_balance
            }
        
        elif msg_type == 'place_order':
            order = PaperOrder(
                order_id=str(uuid.uuid4()),
                timestamp=datetime.now(),
                symbol=data['symbol'],
                side=data['side'],
                order_type=OrderType[data.get('order_type', 'MARKET')],
                quantity=data['quantity'],
                limit_price=data.get('limit_price'),
                stop_price=data.get('stop_price')
            )
            
            result = await self.engine.place_order(data['account_id'], order)
            return {'type': 'order_result', **result}
        
        elif msg_type == 'get_account':
            summary = self.engine.get_account_summary(data['account_id'])
            return {'type': 'account_summary', **summary}
        
        elif msg_type == 'get_performance':
            metrics = self.engine.performance_tracker.calculate_metrics(
                data['account_id']
            )
            return {'type': 'performance_metrics', **metrics}
        
        else:
            return {'type': 'error', 'message': 'Unknown message type'}
    
    async def broadcast_updates(self):
        """Broadcast account updates to all clients"""
        while True:
            for account_id, account in self.engine.accounts.items():
                update = {
                    'type': 'account_update',
                    'account_id': account_id,
                    'total_equity': account.total_equity,
                    'total_pnl': account.total_pnl,
                    'positions': len(account.positions)
                }
                
                dead_clients = set()
                for client in self.clients:
                    try:
                        await client.send(json.dumps(update))
                    except:
                        dead_clients.add(client)
                
                self.clients -= dead_clients
            
            await asyncio.sleep(1)  # Broadcast every second

# ==================== Main Entry Point ====================

async def main():
    """Main function to run paper trading system"""
    # Initialize paper trading engine
    engine = PaperTradingEngine(initial_capital=100000)
    
    # Create a demo account
    account = engine.create_account("demo_user")
    
    # Simulate some market data updates
    symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD']
    
    async def simulate_market_data():
        """Simulate market data feed"""
        while True:
            for symbol in symbols:
                base_price = 50000 if 'BTC' in symbol else 3000 if 'ETH' in symbol else 100
                engine.update_market_data(symbol, {
                    'bid': base_price * (1 - 0.0001),
                    'ask': base_price * (1 + 0.0001),
                    'last': base_price,
                    'volume': np.random.uniform(1000, 10000)
                })
            await asyncio.sleep(1)
    
    # Start tasks
    tasks = [
        asyncio.create_task(engine.start()),
        asyncio.create_task(simulate_market_data())
    ]
    
    logger.info("Paper Trading System running...")
    
    try:
        await asyncio.gather(*tasks)
    except KeyboardInterrupt:
        logger.info("Shutting down Paper Trading System")
        engine.stop()

if __name__ == "__main__":
    asyncio.run(main())