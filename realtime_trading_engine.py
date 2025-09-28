#!/usr/bin/env python3
"""
Real-Time Trading Engine for Backtesting and Paper Trading
Provides live market data, WebSocket streaming, and real-time execution
"""

import asyncio
import json
import websockets
import aiohttp
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
import uuid
from collections import deque
import time
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Real-Time Market Data ====================

class MarketDataFeed:
    """Real-time market data feed with WebSocket streaming"""
    
    def __init__(self):
        self.subscribers = set()
        self.market_data = {}
        self.order_books = {}
        self.trade_stream = deque(maxlen=1000)
        self.is_running = False
        
    async def connect_exchanges(self):
        """Connect to multiple exchange WebSocket feeds"""
        exchanges = {
            'binance': 'wss://stream.binance.com:9443/ws',
            'coinbase': 'wss://ws-feed.pro.coinbase.com',
            'kraken': 'wss://ws.kraken.com'
        }
        
        # For demo, we'll simulate real-time data
        await self.simulate_realtime_data()
        
    async def simulate_realtime_data(self):
        """Simulate real-time market data for demo"""
        symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD', 'MATIC/USD']
        
        while self.is_running:
            for symbol in symbols:
                # Generate realistic price movements
                if symbol not in self.market_data:
                    base_prices = {
                        'BTC/USD': 50000,
                        'ETH/USD': 3000,
                        'SOL/USD': 100,
                        'AVAX/USD': 35,
                        'MATIC/USD': 1.5
                    }
                    self.market_data[symbol] = {
                        'price': base_prices.get(symbol, 100),
                        'volume': 0
                    }
                
                # Realistic price movement (0.01% to 0.1% per tick)
                current = self.market_data[symbol]['price']
                change = np.random.normal(0, 0.001) * current
                new_price = current + change
                
                # Create market data update
                update = {
                    'type': 'market_data',
                    'symbol': symbol,
                    'timestamp': datetime.now().isoformat(),
                    'bid': new_price - (new_price * 0.0001),
                    'ask': new_price + (new_price * 0.0001),
                    'last': new_price,
                    'volume': np.random.uniform(100, 1000),
                    'volume_24h': np.random.uniform(1000000, 10000000),
                    'high_24h': new_price * 1.02,
                    'low_24h': new_price * 0.98,
                    'vwap': new_price * 0.999,
                    'spread': new_price * 0.0002,
                    'order_book_imbalance': np.random.uniform(-0.5, 0.5)
                }
                
                self.market_data[symbol] = {
                    'price': new_price,
                    'volume': update['volume'],
                    'data': update
                }
                
                # Generate order book
                self.order_books[symbol] = self.generate_order_book(symbol, new_price)
                
                # Broadcast to subscribers
                await self.broadcast(update)
                
                # Add to trade stream
                self.trade_stream.append({
                    'symbol': symbol,
                    'price': new_price,
                    'volume': update['volume'],
                    'timestamp': update['timestamp'],
                    'side': 'BUY' if change > 0 else 'SELL'
                })
            
            await asyncio.sleep(0.5)  # Update every 500ms
    
    def generate_order_book(self, symbol: str, mid_price: float) -> Dict:
        """Generate realistic order book"""
        levels = 10
        bids = []
        asks = []
        
        for i in range(levels):
            # Bids (buy orders)
            bid_price = mid_price - (mid_price * 0.0001 * (i + 1))
            bid_volume = np.random.uniform(0.1, 10) * (levels - i)
            bids.append([bid_price, bid_volume])
            
            # Asks (sell orders)
            ask_price = mid_price + (mid_price * 0.0001 * (i + 1))
            ask_volume = np.random.uniform(0.1, 10) * (levels - i)
            asks.append([ask_price, ask_volume])
        
        return {
            'bids': bids,
            'asks': asks,
            'timestamp': datetime.now().isoformat()
        }
    
    async def broadcast(self, data: Dict):
        """Broadcast data to all subscribers"""
        if self.subscribers:
            message = json.dumps(data)
            dead_subscribers = set()
            
            for subscriber in self.subscribers:
                try:
                    await subscriber.send(message)
                except:
                    dead_subscribers.add(subscriber)
            
            self.subscribers -= dead_subscribers
    
    async def subscribe(self, websocket):
        """Subscribe to market data feed"""
        self.subscribers.add(websocket)
        
        # Send current state
        for symbol, data in self.market_data.items():
            if 'data' in data:
                await websocket.send(json.dumps(data['data']))
    
    def unsubscribe(self, websocket):
        """Unsubscribe from market data feed"""
        self.subscribers.discard(websocket)
    
    async def start(self):
        """Start market data feed"""
        self.is_running = True
        await self.connect_exchanges()
    
    def stop(self):
        """Stop market data feed"""
        self.is_running = False

# ==================== Real-Time Backtesting ====================

class RealtimeBacktester:
    """Real-time backtesting with live market data"""
    
    def __init__(self, market_feed: MarketDataFeed):
        self.market_feed = market_feed
        self.strategies = {}
        self.backtest_results = {}
        self.is_running = False
        
    async def run_realtime_backtest(self, strategy_id: str, config: Dict):
        """Run backtest with real-time data streaming"""
        logger.info(f"Starting real-time backtest: {strategy_id}")
        
        # Initialize backtest state
        self.backtest_results[strategy_id] = {
            'config': config,
            'equity_curve': [],
            'trades': [],
            'metrics': {},
            'status': 'running',
            'start_time': datetime.now()
        }
        
        initial_capital = config.get('initial_capital', 100000)
        capital = initial_capital
        positions = {}
        
        # Subscribe to market data
        while self.is_running and strategy_id in self.backtest_results:
            # Get latest market data
            for symbol, market_data in self.market_feed.market_data.items():
                if 'data' not in market_data:
                    continue
                
                data = market_data['data']
                
                # Apply strategy logic
                signal = await self.apply_strategy(
                    config['strategy'], 
                    symbol, 
                    data, 
                    positions
                )
                
                if signal:
                    # Execute trade
                    trade = self.execute_backtest_trade(
                        signal, 
                        data, 
                        capital,
                        config
                    )
                    
                    if trade:
                        self.backtest_results[strategy_id]['trades'].append(trade)
                        capital += trade['pnl']
                        
                        # Update positions
                        if signal['action'] == 'BUY':
                            positions[symbol] = {
                                'quantity': signal['quantity'],
                                'entry_price': trade['price']
                            }
                        elif signal['action'] == 'SELL' and symbol in positions:
                            del positions[symbol]
                
                # Update equity curve
                total_value = capital
                for sym, pos in positions.items():
                    if sym in self.market_feed.market_data:
                        current_price = self.market_feed.market_data[sym]['price']
                        total_value += pos['quantity'] * current_price
                
                self.backtest_results[strategy_id]['equity_curve'].append({
                    'timestamp': datetime.now().isoformat(),
                    'value': total_value,
                    'capital': capital,
                    'positions_value': total_value - capital
                })
                
                # Calculate real-time metrics
                self.update_metrics(strategy_id, initial_capital)
                
                # Broadcast update
                await self.broadcast_backtest_update(strategy_id)
            
            await asyncio.sleep(1)  # Update every second
    
    async def apply_strategy(self, strategy_type: str, symbol: str, 
                            data: Dict, positions: Dict) -> Optional[Dict]:
        """Apply trading strategy and generate signals"""
        
        if strategy_type == 'momentum':
            # Simple momentum strategy
            if len(self.market_feed.trade_stream) > 20:
                recent_trades = list(self.market_feed.trade_stream)[-20:]
                prices = [t['price'] for t in recent_trades if t['symbol'] == symbol]
                
                if len(prices) >= 10:
                    sma_short = np.mean(prices[-5:])
                    sma_long = np.mean(prices[-10:])
                    
                    if sma_short > sma_long and symbol not in positions:
                        return {
                            'action': 'BUY',
                            'symbol': symbol,
                            'quantity': 1,
                            'strategy': 'momentum'
                        }
                    elif sma_short < sma_long and symbol in positions:
                        return {
                            'action': 'SELL',
                            'symbol': symbol,
                            'quantity': positions[symbol]['quantity'],
                            'strategy': 'momentum'
                        }
        
        elif strategy_type == 'mean_reversion':
            # Mean reversion strategy
            if symbol in self.market_feed.market_data:
                current_price = data['last']
                vwap = data['vwap']
                
                deviation = (current_price - vwap) / vwap
                
                if deviation < -0.01 and symbol not in positions:  # 1% below VWAP
                    return {
                        'action': 'BUY',
                        'symbol': symbol,
                        'quantity': 1,
                        'strategy': 'mean_reversion'
                    }
                elif deviation > 0.01 and symbol in positions:  # 1% above VWAP
                    return {
                        'action': 'SELL',
                        'symbol': symbol,
                        'quantity': positions[symbol]['quantity'],
                        'strategy': 'mean_reversion'
                    }
        
        elif strategy_type == 'arbitrage':
            # Arbitrage detection
            spread_pct = data['spread'] / data['last']
            if spread_pct > 0.002:  # 0.2% spread
                return {
                    'action': 'BUY' if np.random.random() > 0.5 else 'SELL',
                    'symbol': symbol,
                    'quantity': 0.5,
                    'strategy': 'arbitrage'
                }
        
        return None
    
    def execute_backtest_trade(self, signal: Dict, market_data: Dict, 
                              capital: float, config: Dict) -> Optional[Dict]:
        """Execute trade in backtest"""
        price = market_data['ask'] if signal['action'] == 'BUY' else market_data['bid']
        
        # Apply slippage
        slippage = config.get('slippage', 0.0001)
        if signal['action'] == 'BUY':
            price *= (1 + slippage)
        else:
            price *= (1 - slippage)
        
        # Calculate commission
        commission = price * signal['quantity'] * config.get('commission', 0.0002)
        
        # Calculate P&L for sells
        pnl = 0
        if signal['action'] == 'SELL':
            # This is simplified - in production would track entry prices
            pnl = signal['quantity'] * price - commission
        else:
            pnl = -(signal['quantity'] * price + commission)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'symbol': signal['symbol'],
            'action': signal['action'],
            'quantity': signal['quantity'],
            'price': price,
            'commission': commission,
            'slippage': price * slippage,
            'pnl': pnl,
            'strategy': signal['strategy']
        }
    
    def update_metrics(self, strategy_id: str, initial_capital: float):
        """Update real-time metrics"""
        result = self.backtest_results[strategy_id]
        
        if len(result['equity_curve']) < 2:
            return
        
        # Calculate returns
        equity_values = [e['value'] for e in result['equity_curve']]
        returns = pd.Series(equity_values).pct_change().dropna()
        
        # Calculate metrics
        total_return = (equity_values[-1] / initial_capital) - 1
        
        if len(returns) > 0:
            sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
            
            # Calculate max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Trade statistics
        trades = result['trades']
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(trades)
            
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in trades if t['pnl'] < 0]) if any(t['pnl'] < 0 for t in trades) else 0
            
            profit_factor = abs(sum(t['pnl'] for t in winning_trades) / sum(t['pnl'] for t in trades if t['pnl'] < 0)) \
                          if any(t['pnl'] < 0 for t in trades) else float('inf')
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        result['metrics'] = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(trades),
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'current_equity': equity_values[-1] if equity_values else initial_capital
        }
    
    async def broadcast_backtest_update(self, strategy_id: str):
        """Broadcast backtest updates to subscribers"""
        result = self.backtest_results[strategy_id]
        
        update = {
            'type': 'backtest_update',
            'strategy_id': strategy_id,
            'metrics': result['metrics'],
            'latest_equity': result['equity_curve'][-1] if result['equity_curve'] else None,
            'latest_trade': result['trades'][-1] if result['trades'] else None,
            'status': result['status']
        }
        
        await self.market_feed.broadcast(update)
    
    async def start(self):
        """Start real-time backtester"""
        self.is_running = True
    
    def stop(self):
        """Stop real-time backtester"""
        self.is_running = False

# ==================== Real-Time Paper Trading ====================

class RealtimePaperTrader:
    """Real-time paper trading with live execution"""
    
    def __init__(self, market_feed: MarketDataFeed):
        self.market_feed = market_feed
        self.accounts = {}
        self.orders = {}
        self.positions = {}
        self.execution_engine = ExecutionEngine(market_feed)
        
    async def create_account(self, user_id: str, initial_balance: float = 100000):
        """Create paper trading account"""
        account_id = f"paper_{user_id}_{uuid.uuid4().hex[:8]}"
        
        self.accounts[account_id] = {
            'user_id': user_id,
            'balance': initial_balance,
            'initial_balance': initial_balance,
            'equity': initial_balance,
            'buying_power': initial_balance * 2,  # 2x leverage
            'positions': {},
            'orders': [],
            'trades': [],
            'created_at': datetime.now(),
            'metrics': {
                'total_pnl': 0,
                'realized_pnl': 0,
                'unrealized_pnl': 0,
                'win_rate': 0,
                'total_trades': 0
            }
        }
        
        self.positions[account_id] = {}
        
        logger.info(f"Created paper account: {account_id}")
        return account_id
    
    async def place_order(self, account_id: str, order: Dict):
        """Place paper trading order"""
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        order_id = f"order_{uuid.uuid4().hex[:12]}"
        order['order_id'] = order_id
        order['account_id'] = account_id
        order['status'] = 'PENDING'
        order['timestamp'] = datetime.now()
        
        # Risk checks
        if not await self.check_risk_limits(account_id, order):
            order['status'] = 'REJECTED'
            order['reject_reason'] = 'Risk limit exceeded'
            return order
        
        # Add to order queue
        self.orders[order_id] = order
        self.accounts[account_id]['orders'].append(order)
        
        # Execute immediately for market orders
        if order['type'] == 'MARKET':
            await self.execute_order(order_id)
        
        return order
    
    async def execute_order(self, order_id: str):
        """Execute paper order with real-time prices"""
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        account_id = order['account_id']
        symbol = order['symbol']
        
        # Get real-time market data
        if symbol not in self.market_feed.market_data:
            order['status'] = 'REJECTED'
            order['reject_reason'] = 'Symbol not available'
            return
        
        market_data = self.market_feed.market_data[symbol]['data']
        
        # Execute based on order type
        if order['type'] == 'MARKET':
            fill_price = market_data['ask'] if order['side'] == 'BUY' else market_data['bid']
        elif order['type'] == 'LIMIT':
            # Check if limit price is met
            if order['side'] == 'BUY' and market_data['ask'] <= order['limit_price']:
                fill_price = order['limit_price']
            elif order['side'] == 'SELL' and market_data['bid'] >= order['limit_price']:
                fill_price = order['limit_price']
            else:
                return  # Order not filled yet
        else:
            return  # Other order types not implemented yet
        
        # Apply slippage
        slippage = self.calculate_slippage(order, market_data)
        fill_price += slippage
        
        # Calculate commission
        commission = fill_price * order['quantity'] * 0.0002
        
        # Update account
        account = self.accounts[account_id]
        
        if order['side'] == 'BUY':
            total_cost = fill_price * order['quantity'] + commission
            
            if total_cost > account['buying_power']:
                order['status'] = 'REJECTED'
                order['reject_reason'] = 'Insufficient buying power'
                return
            
            # Update position
            if symbol not in self.positions[account_id]:
                self.positions[account_id][symbol] = {
                    'quantity': 0,
                    'average_price': 0,
                    'realized_pnl': 0
                }
            
            position = self.positions[account_id][symbol]
            new_quantity = position['quantity'] + order['quantity']
            new_cost = (position['quantity'] * position['average_price'] + 
                       order['quantity'] * fill_price)
            
            position['quantity'] = new_quantity
            position['average_price'] = new_cost / new_quantity if new_quantity > 0 else 0
            
            account['balance'] -= total_cost
            account['buying_power'] -= total_cost
            
        else:  # SELL
            if symbol not in self.positions[account_id] or \
               self.positions[account_id][symbol]['quantity'] < order['quantity']:
                order['status'] = 'REJECTED'
                order['reject_reason'] = 'Insufficient position'
                return
            
            position = self.positions[account_id][symbol]
            
            # Calculate P&L
            pnl = (fill_price - position['average_price']) * order['quantity'] - commission
            position['realized_pnl'] += pnl
            account['metrics']['realized_pnl'] += pnl
            
            # Update position
            position['quantity'] -= order['quantity']
            if position['quantity'] == 0:
                del self.positions[account_id][symbol]
            
            proceeds = fill_price * order['quantity'] - commission
            account['balance'] += proceeds
            account['buying_power'] += proceeds
        
        # Record trade
        trade = {
            'trade_id': f"trade_{uuid.uuid4().hex[:12]}",
            'order_id': order_id,
            'symbol': symbol,
            'side': order['side'],
            'quantity': order['quantity'],
            'price': fill_price,
            'commission': commission,
            'slippage': slippage,
            'pnl': pnl if order['side'] == 'SELL' else 0,
            'timestamp': datetime.now()
        }
        
        account['trades'].append(trade)
        account['metrics']['total_trades'] += 1
        
        # Update order status
        order['status'] = 'FILLED'
        order['fill_price'] = fill_price
        order['fill_timestamp'] = datetime.now()
        
        # Update metrics
        await self.update_account_metrics(account_id)
        
        # Broadcast update
        await self.broadcast_account_update(account_id)
        
        logger.info(f"Executed order {order_id}: {order['side']} {order['quantity']} {symbol} @ {fill_price}")
    
    def calculate_slippage(self, order: Dict, market_data: Dict) -> float:
        """Calculate realistic slippage"""
        base_slippage = market_data['spread'] * 0.5
        size_impact = (order['quantity'] / market_data['volume']) * market_data['last'] * 0.001
        volatility_impact = np.random.normal(0, market_data['last'] * 0.0001)
        
        total_slippage = base_slippage + size_impact + volatility_impact
        
        return total_slippage if order['side'] == 'BUY' else -total_slippage
    
    async def check_risk_limits(self, account_id: str, order: Dict) -> bool:
        """Check if order passes risk limits"""
        account = self.accounts[account_id]
        
        # Position concentration limit
        if order['side'] == 'BUY':
            estimated_cost = order['quantity'] * self.market_feed.market_data.get(
                order['symbol'], {}).get('price', 0)
            
            if estimated_cost > account['buying_power'] * 0.5:  # Max 50% per position
                return False
        
        # Daily loss limit
        if account['metrics']['realized_pnl'] < -account['initial_balance'] * 0.05:  # 5% daily loss limit
            return False
        
        # Maximum positions
        if len(self.positions[account_id]) >= 10 and order['side'] == 'BUY':
            if order['symbol'] not in self.positions[account_id]:
                return False
        
        return True
    
    async def update_account_metrics(self, account_id: str):
        """Update account metrics in real-time"""
        account = self.accounts[account_id]
        
        # Calculate unrealized P&L
        unrealized_pnl = 0
        positions_value = 0
        
        for symbol, position in self.positions[account_id].items():
            if symbol in self.market_feed.market_data:
                current_price = self.market_feed.market_data[symbol]['price']
                unrealized = (current_price - position['average_price']) * position['quantity']
                unrealized_pnl += unrealized
                positions_value += current_price * position['quantity']
        
        account['metrics']['unrealized_pnl'] = unrealized_pnl
        account['metrics']['total_pnl'] = account['metrics']['realized_pnl'] + unrealized_pnl
        account['equity'] = account['balance'] + positions_value
        
        # Calculate win rate
        trades = account['trades']
        if trades:
            winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
            account['metrics']['win_rate'] = len(winning_trades) / len(trades)
    
    async def broadcast_account_update(self, account_id: str):
        """Broadcast account updates"""
        account = self.accounts[account_id]
        
        update = {
            'type': 'paper_trading_update',
            'account_id': account_id,
            'balance': account['balance'],
            'equity': account['equity'],
            'positions': len(self.positions[account_id]),
            'metrics': account['metrics'],
            'latest_trade': account['trades'][-1] if account['trades'] else None
        }
        
        await self.market_feed.broadcast(update)

# ==================== Execution Engine ====================

class ExecutionEngine:
    """Handles order execution logic"""
    
    def __init__(self, market_feed: MarketDataFeed):
        self.market_feed = market_feed
        
    async def execute_limit_order(self, order: Dict, market_data: Dict) -> Optional[float]:
        """Execute limit order if conditions are met"""
        if order['side'] == 'BUY' and market_data['ask'] <= order['limit_price']:
            return min(market_data['ask'], order['limit_price'])
        elif order['side'] == 'SELL' and market_data['bid'] >= order['limit_price']:
            return max(market_data['bid'], order['limit_price'])
        return None
    
    async def execute_stop_order(self, order: Dict, market_data: Dict) -> Optional[float]:
        """Execute stop order if triggered"""
        if order['side'] == 'BUY' and market_data['last'] >= order['stop_price']:
            return market_data['ask']
        elif order['side'] == 'SELL' and market_data['last'] <= order['stop_price']:
            return market_data['bid']
        return None

# ==================== WebSocket Server ====================

class RealtimeTradingServer:
    """WebSocket server for real-time trading"""
    
    def __init__(self):
        self.market_feed = MarketDataFeed()
        self.backtester = RealtimeBacktester(self.market_feed)
        self.paper_trader = RealtimePaperTrader(self.market_feed)
        
    async def handler(self, websocket, path):
        """Handle WebSocket connections"""
        await self.market_feed.subscribe(websocket)
        
        try:
            async for message in websocket:
                data = json.loads(message)
                response = await self.process_message(data, websocket)
                
                if response:
                    await websocket.send(json.dumps(response))
                    
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            self.market_feed.unsubscribe(websocket)
    
    async def process_message(self, data: Dict, websocket) -> Dict:
        """Process incoming messages"""
        msg_type = data.get('type')
        
        if msg_type == 'start_backtest':
            strategy_id = f"backtest_{uuid.uuid4().hex[:8]}"
            asyncio.create_task(
                self.backtester.run_realtime_backtest(strategy_id, data['config'])
            )
            return {'type': 'backtest_started', 'strategy_id': strategy_id}
        
        elif msg_type == 'stop_backtest':
            strategy_id = data['strategy_id']
            if strategy_id in self.backtester.backtest_results:
                self.backtester.backtest_results[strategy_id]['status'] = 'stopped'
            return {'type': 'backtest_stopped', 'strategy_id': strategy_id}
        
        elif msg_type == 'create_paper_account':
            account_id = await self.paper_trader.create_account(
                data.get('user_id', 'anonymous'),
                data.get('initial_balance', 100000)
            )
            return {'type': 'account_created', 'account_id': account_id}
        
        elif msg_type == 'place_order':
            result = await self.paper_trader.place_order(
                data['account_id'],
                data['order']
            )
            return {'type': 'order_placed', 'order': result}
        
        elif msg_type == 'get_account_info':
            account_id = data['account_id']
            if account_id in self.paper_trader.accounts:
                account = self.paper_trader.accounts[account_id]
                return {
                    'type': 'account_info',
                    'account': {
                        'balance': account['balance'],
                        'equity': account['equity'],
                        'positions': self.paper_trader.positions.get(account_id, {}),
                        'metrics': account['metrics']
                    }
                }
        
        return {'type': 'error', 'message': 'Unknown message type'}
    
    async def start(self):
        """Start the server"""
        # Start market data feed
        asyncio.create_task(self.market_feed.start())
        
        # Start backtester
        await self.backtester.start()
        
        # Start WebSocket server
        async with websockets.serve(self.handler, 'localhost', 8765):
            logger.info("Real-time trading server running on ws://localhost:8765")
            await asyncio.Future()  # Run forever

# ==================== Helper Functions ====================

async def start_websocket_server(market_feed, backtest_engine, paper_trader, host='0.0.0.0', port=9000):
    """Start the WebSocket server with provided components"""
    
    async def handler(websocket, path):
        """Handle WebSocket connections"""
        client_id = str(uuid.uuid4())
        logger.info(f"Client {client_id} connected")
        
        try:
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    # Process message based on action
                    if data['action'] == 'subscribe':
                        # Subscribe to market data
                        symbols = data['data'].get('symbols', [])
                        for symbol in symbols:
                            # Send current market data
                            market_data = await market_feed.get_latest(symbol)
                            await websocket.send(json.dumps({
                                'type': 'market_data',
                                'data': market_data
                            }))
                    
                    elif data['action'] == 'start_backtest':
                        # Start real-time backtest
                        strategy_id = data['data'].get('strategy_id')
                        config = data['data'].get('config', {})
                        
                        async def update_callback(results):
                            await websocket.send(json.dumps({
                                'type': 'backtest_update',
                                'data': results
                            }))
                        
                        await backtest_engine.run_realtime_backtest(
                            strategy_id, config, update_callback
                        )
                    
                    elif data['action'] == 'create_paper_account':
                        # Create paper trading account
                        account = await paper_trader.create_account(data['data'])
                        await websocket.send(json.dumps({
                            'type': 'paper_trading_update',
                            'data': account
                        }))
                    
                    elif data['action'] == 'place_order':
                        # Place paper trading order
                        account_id = data['data'].get('account_id')
                        order = data['data'].get('order')
                        result = await paper_trader.place_order(account_id, order)
                        await websocket.send(json.dumps({
                            'type': 'order_execution',
                            'data': result
                        }))
                    
                except json.JSONDecodeError:
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'error': 'Invalid JSON format'
                    }))
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    await websocket.send(json.dumps({
                        'type': 'error',
                        'error': str(e)
                    }))
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}")
    
    # Start the WebSocket server
    async with websockets.serve(handler, host, port):
        logger.info(f"WebSocket server running on ws://{host}:{port}")
        await asyncio.Future()  # Run forever

# ==================== Main Entry Point ====================

if __name__ == "__main__":
    server = RealtimeTradingServer()
    asyncio.run(server.start())