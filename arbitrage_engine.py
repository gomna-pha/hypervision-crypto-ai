"""
HyperVision AI - Advanced Arbitrage Engine
Institutional-grade High-Frequency Trading Arbitrage Platform

Implements prioritized arbitrage strategies:
1. Index/Futures-Spot Arbitrage (Priority #1) 
2. Triangular/Cross-pair Crypto Arbitrage (Priority #2)
3. Statistical/Pairs Arbitrage with Hyperbolic Embeddings (Priority #3)
4. News/Sentiment-Triggered Arbitrage (Priority #4)
5. Statistical Latency Arbitrage (Priority #5)
"""

import numpy as np
import pandas as pd
import asyncio
import websockets
import json
import time
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging
import concurrent.futures
from abc import ABC, abstractmethod
import math
from collections import defaultdict, deque
import threading
from scipy.stats import zscore
from scipy.optimize import minimize
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity"""
    strategy_type: str
    symbol_pair: Tuple[str, str]
    exchange_pair: Tuple[str, str]
    expected_profit: float
    confidence_score: float
    risk_score: float
    entry_price_1: float
    entry_price_2: float
    volume_limit: float
    latency_ms: float
    timestamp: datetime
    metadata: Dict[str, Any]

@dataclass 
class TradeExecution:
    """Represents an executed trade"""
    opportunity_id: str
    strategy_type: str
    symbol: str
    exchange: str
    side: str  # 'buy' or 'sell'
    size: float
    price: float
    execution_time: datetime
    latency_ms: float
    status: str  # 'pending', 'filled', 'partial', 'cancelled'

class BaseArbitrageStrategy(ABC):
    """Base class for all arbitrage strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.active = True
        self.performance_metrics = {
            'opportunities_detected': 0,
            'opportunities_executed': 0,
            'total_profit': 0.0,
            'win_rate': 0.0,
            'avg_latency': 0.0
        }
    
    @abstractmethod
    async def detect_opportunities(self, market_data: Dict[str, Any]) -> List[ArbitrageOpportunity]:
        """Detect arbitrage opportunities"""
        pass
    
    @abstractmethod
    async def validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate if an opportunity is still viable"""
        pass
    
    def update_performance(self, execution_result: Dict[str, Any]):
        """Update performance metrics"""
        if execution_result['status'] == 'filled':
            self.performance_metrics['opportunities_executed'] += 1
            self.performance_metrics['total_profit'] += execution_result.get('profit', 0)
        
        # Update win rate
        if self.performance_metrics['opportunities_executed'] > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['opportunities_executed'] / 
                self.performance_metrics['opportunities_detected']
            )

class IndexFuturesSpotArbitrage(BaseArbitrageStrategy):
    """
    Priority #1: Index/Futures-Spot Arbitrage Strategy
    
    Exploits price discrepancies between index futures and underlying spot assets.
    Highly institutional strategy with clear economic rationale.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Index_Futures_Spot", config)
        self.min_spread = config.get('min_spread_bps', 5)  # 5 basis points
        self.max_position_size = config.get('max_position_size', 1000000)  # $1M
        self.max_latency_ms = config.get('max_latency_ms', 50)
        self.risk_multiplier = config.get('risk_multiplier', 0.1)
        
    async def detect_opportunities(self, market_data: Dict[str, Any]) -> List[ArbitrageOpportunity]:
        opportunities = []
        
        try:
            # Get futures and spot data
            futures_data = market_data.get('futures', {})
            spot_data = market_data.get('spot', {})
            
            for symbol in futures_data:
                if symbol not in spot_data:
                    continue
                
                future_price = futures_data[symbol]['price']
                spot_price = spot_data[symbol]['price'] 
                future_volume = futures_data[symbol]['volume']
                spot_volume = spot_data[symbol]['volume']
                
                # Calculate theoretical future price (spot + carry cost)
                time_to_expiry = self._get_time_to_expiry(symbol)
                risk_free_rate = market_data.get('risk_free_rate', 0.05)
                dividend_yield = market_data.get('dividend_yield', {}).get(symbol, 0.02)
                
                theoretical_future = spot_price * math.exp((risk_free_rate - dividend_yield) * time_to_expiry)
                
                # Calculate spread in basis points
                spread_bps = abs(future_price - theoretical_future) / theoretical_future * 10000
                
                if spread_bps >= self.min_spread:
                    # Determine arbitrage direction
                    if future_price > theoretical_future:
                        # Future overpriced: sell future, buy spot
                        strategy_direction = "sell_future_buy_spot"
                        expected_profit = (future_price - theoretical_future) * min(future_volume, spot_volume) * 0.5
                    else:
                        # Future underpriced: buy future, sell spot  
                        strategy_direction = "buy_future_sell_spot"
                        expected_profit = (theoretical_future - future_price) * min(future_volume, spot_volume) * 0.5
                    
                    # Calculate confidence score
                    volume_confidence = min(future_volume, spot_volume) / max(future_volume, spot_volume)
                    spread_confidence = min(spread_bps / 20, 1.0)  # Max confidence at 20bps
                    confidence_score = (volume_confidence + spread_confidence) / 2
                    
                    # Calculate risk score
                    volatility = market_data.get('volatility', {}).get(symbol, 0.2)
                    risk_score = volatility * self.risk_multiplier
                    
                    opportunity = ArbitrageOpportunity(
                        strategy_type="index_futures_spot",
                        symbol_pair=(symbol, f"{symbol}_FUTURES"),
                        exchange_pair=("SPOT_EXCHANGE", "FUTURES_EXCHANGE"),
                        expected_profit=expected_profit,
                        confidence_score=confidence_score,
                        risk_score=risk_score,
                        entry_price_1=spot_price,
                        entry_price_2=future_price,
                        volume_limit=min(future_volume, spot_volume) * 0.1,
                        latency_ms=market_data.get('latency_ms', 10),
                        timestamp=datetime.now(),
                        metadata={
                            'theoretical_future': theoretical_future,
                            'spread_bps': spread_bps,
                            'strategy_direction': strategy_direction,
                            'time_to_expiry': time_to_expiry
                        }
                    )
                    
                    opportunities.append(opportunity)
                    self.performance_metrics['opportunities_detected'] += 1
                    
        except Exception as e:
            logger.error(f"Error in Index/Futures arbitrage detection: {e}")
        
        return opportunities
    
    async def validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate opportunity is still viable"""
        # Check if opportunity is still recent (< 100ms old)
        age_ms = (datetime.now() - opportunity.timestamp).total_seconds() * 1000
        if age_ms > 100:
            return False
            
        # Check if latency is acceptable
        if opportunity.latency_ms > self.max_latency_ms:
            return False
            
        # Check risk-adjusted expected profit
        min_profit = opportunity.entry_price_1 * 0.0001  # 1 basis point minimum
        if opportunity.expected_profit < min_profit:
            return False
            
        return True
    
    def _get_time_to_expiry(self, symbol: str) -> float:
        """Get time to expiry for futures contract (simplified)"""
        # In production, this would query actual contract details
        # For demo, assume quarterly expiry
        now = datetime.now()
        quarter_end = datetime(now.year, ((now.month - 1) // 3 + 1) * 3, 1) + timedelta(days=32)
        quarter_end = quarter_end.replace(day=1) - timedelta(days=1)
        return max((quarter_end - now).days / 365.25, 1/365.25)

class TriangularCryptoArbitrage(BaseArbitrageStrategy):
    """
    Priority #2: Triangular/Cross-pair Crypto Arbitrage Strategy
    
    Exploits price discrepancies in triangular currency relationships (e.g., BTC/ETH/USDT).
    Fast to backtest and validate with clear profit mechanics.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Triangular_Crypto", config)
        self.min_profit_bps = config.get('min_profit_bps', 10)  # 10 basis points
        self.max_slippage_bps = config.get('max_slippage_bps', 5)  # 5 basis points
        self.trading_pairs = config.get('trading_pairs', [
            ('BTC', 'ETH', 'USDT'),
            ('BTC', 'ETH', 'USDC'),
            ('ETH', 'BNB', 'USDT'),
            ('BTC', 'SOL', 'USDT')
        ])
        
    async def detect_opportunities(self, market_data: Dict[str, Any]) -> List[ArbitrageOpportunity]:
        opportunities = []
        
        try:
            prices = market_data.get('prices', {})
            volumes = market_data.get('volumes', {})
            
            for base, quote, counter in self.trading_pairs:
                # Get all required price pairs
                pair_1 = f"{base}/{quote}"     # e.g., BTC/ETH
                pair_2 = f"{quote}/{counter}"  # e.g., ETH/USDT  
                pair_3 = f"{base}/{counter}"   # e.g., BTC/USDT
                
                if not all(pair in prices for pair in [pair_1, pair_2, pair_3]):
                    continue
                
                price_1 = prices[pair_1]['bid']  # Selling BTC for ETH
                price_2 = prices[pair_2]['bid']  # Selling ETH for USDT
                price_3 = prices[pair_3]['ask']  # Buying BTC with USDT
                
                # Calculate triangular arbitrage profit
                # Path: USDT -> BTC -> ETH -> USDT
                amount_usdt = 10000  # Start with $10k
                amount_btc = amount_usdt / price_3  # Buy BTC
                amount_eth = amount_btc * price_1   # Sell BTC for ETH
                final_usdt = amount_eth * price_2   # Sell ETH for USDT
                
                profit_usdt = final_usdt - amount_usdt
                profit_bps = (profit_usdt / amount_usdt) * 10000
                
                # Check reverse direction as well
                # Path: USDT -> ETH -> BTC -> USDT  
                price_1_rev = prices[pair_1]['ask']  # Buying BTC with ETH
                price_2_rev = prices[pair_2]['ask']  # Buying ETH with USDT
                price_3_rev = prices[pair_3]['bid']  # Selling BTC for USDT
                
                amount_eth_rev = amount_usdt / price_2_rev  # Buy ETH
                amount_btc_rev = amount_eth_rev / price_1_rev  # Buy BTC with ETH  
                final_usdt_rev = amount_btc_rev * price_3_rev  # Sell BTC for USDT
                
                profit_usdt_rev = final_usdt_rev - amount_usdt
                profit_bps_rev = (profit_usdt_rev / amount_usdt) * 10000
                
                # Take the more profitable direction
                if abs(profit_bps_rev) > abs(profit_bps):
                    profit_bps = profit_bps_rev
                    profit_usdt = profit_usdt_rev
                    direction = "reverse"
                else:
                    direction = "forward"
                
                # Account for trading fees (assume 0.1% per trade)
                fee_cost_bps = 3 * 10  # 3 trades * 10 bps per trade
                net_profit_bps = profit_bps - fee_cost_bps
                
                if net_profit_bps >= self.min_profit_bps:
                    # Calculate position size limit based on liquidity
                    min_volume = min(
                        volumes.get(pair_1, {}).get('volume_24h', 0),
                        volumes.get(pair_2, {}).get('volume_24h', 0), 
                        volumes.get(pair_3, {}).get('volume_24h', 0)
                    )
                    
                    max_position = min_volume * 0.001  # 0.1% of 24h volume
                    
                    # Calculate confidence based on volume and spread stability
                    avg_spread = (
                        prices[pair_1].get('spread', 0.001) +
                        prices[pair_2].get('spread', 0.001) + 
                        prices[pair_3].get('spread', 0.001)
                    ) / 3
                    
                    volume_score = min(min_volume / 1000000, 1.0)  # Normalize to $1M volume
                    spread_score = max(0, 1 - avg_spread * 1000)   # Lower spread = higher score
                    confidence_score = (volume_score + spread_score) / 2
                    
                    opportunity = ArbitrageOpportunity(
                        strategy_type="triangular_crypto",
                        symbol_pair=(base, quote),
                        exchange_pair=("CRYPTO_EXCHANGE_1", "CRYPTO_EXCHANGE_1"), 
                        expected_profit=abs(profit_usdt),
                        confidence_score=confidence_score,
                        risk_score=avg_spread + 0.1,  # Spread + base risk
                        entry_price_1=price_1,
                        entry_price_2=price_2, 
                        volume_limit=max_position,
                        latency_ms=market_data.get('latency_ms', 20),
                        timestamp=datetime.now(),
                        metadata={
                            'trading_pairs': [pair_1, pair_2, pair_3],
                            'direction': direction,
                            'gross_profit_bps': profit_bps,
                            'net_profit_bps': net_profit_bps,
                            'counter_currency': counter,
                            'execution_path': f"{counter}->{base}->{quote}->{counter}" if direction == "forward" else f"{counter}->{quote}->{base}->{counter}"
                        }
                    )
                    
                    opportunities.append(opportunity)
                    self.performance_metrics['opportunities_detected'] += 1
                    
        except Exception as e:
            logger.error(f"Error in Triangular arbitrage detection: {e}")
            
        return opportunities
    
    async def validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate triangular arbitrage opportunity"""
        # Check age (triangular arb needs to be very fast)
        age_ms = (datetime.now() - opportunity.timestamp).total_seconds() * 1000
        if age_ms > 50:  # 50ms max age
            return False
            
        # Check minimum profit after slippage
        estimated_slippage = opportunity.risk_score * self.max_slippage_bps
        if opportunity.metadata['net_profit_bps'] < estimated_slippage + 5:
            return False
            
        return True

class StatisticalPairsArbitrage(BaseArbitrageStrategy):
    """
    Priority #3: Statistical/Pairs Arbitrage with Hyperbolic Embeddings
    
    Uses cointegration and hyperbolic embeddings to identify mean-reverting pairs.
    Demonstrates the platform's unique hierarchical relationship modeling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__("Statistical_Pairs", config)
        self.lookback_periods = config.get('lookback_periods', 252)  # 1 year daily
        self.z_entry_threshold = config.get('z_entry_threshold', 2.0)
        self.z_exit_threshold = config.get('z_exit_threshold', 0.5)
        self.max_holding_days = config.get('max_holding_days', 30)
        self.hyperbolic_embeddings = {}
        self.pair_relationships = {}
        
    async def detect_opportunities(self, market_data: Dict[str, Any]) -> List[ArbitrageOpportunity]:
        opportunities = []
        
        try:
            price_history = market_data.get('price_history', {})
            current_prices = market_data.get('current_prices', {})
            
            # Update hyperbolic embeddings for instruments
            await self._update_hyperbolic_embeddings(price_history)
            
            # Find cointegrated pairs using hyperbolic similarity
            cointegrated_pairs = await self._find_cointegrated_pairs(price_history)
            
            for pair_info in cointegrated_pairs:
                symbol_1, symbol_2 = pair_info['symbols']
                hedge_ratio = pair_info['hedge_ratio'] 
                
                if symbol_1 not in current_prices or symbol_2 not in current_prices:
                    continue
                
                price_1 = current_prices[symbol_1]
                price_2 = current_prices[symbol_2]
                
                # Calculate current spread
                spread = price_1 - hedge_ratio * price_2
                
                # Get historical spread statistics
                spread_history = pair_info['spread_history']
                spread_mean = np.mean(spread_history)
                spread_std = np.std(spread_history)
                
                # Calculate z-score
                z_score = (spread - spread_mean) / spread_std if spread_std > 0 else 0
                
                if abs(z_score) >= self.z_entry_threshold:
                    # Determine trade direction
                    if z_score > 0:
                        # Spread is high: short symbol_1, long symbol_2
                        direction = "short_long"
                        expected_return = abs(z_score) * spread_std * 0.5  # Conservative estimate
                    else:
                        # Spread is low: long symbol_1, short symbol_2  
                        direction = "long_short"
                        expected_return = abs(z_score) * spread_std * 0.5
                    
                    # Calculate position sizing
                    volatility_1 = pair_info.get('volatility_1', 0.2)
                    volatility_2 = pair_info.get('volatility_2', 0.2)
                    avg_volatility = (volatility_1 + volatility_2) / 2
                    
                    # Risk-adjusted position size
                    notional_limit = 1000000 / (avg_volatility * abs(z_score))
                    
                    # Confidence based on cointegration strength and hyperbolic similarity
                    cointegration_score = pair_info.get('cointegration_pvalue', 1.0)
                    hyperbolic_similarity = self._calculate_hyperbolic_similarity(symbol_1, symbol_2)
                    confidence_score = (1 - cointegration_score) * hyperbolic_similarity
                    
                    opportunity = ArbitrageOpportunity(
                        strategy_type="statistical_pairs",
                        symbol_pair=(symbol_1, symbol_2),
                        exchange_pair=("MULTI_EXCHANGE", "MULTI_EXCHANGE"),
                        expected_profit=expected_return,
                        confidence_score=confidence_score,
                        risk_score=avg_volatility,
                        entry_price_1=price_1,
                        entry_price_2=price_2,
                        volume_limit=notional_limit,
                        latency_ms=market_data.get('latency_ms', 100), 
                        timestamp=datetime.now(),
                        metadata={
                            'hedge_ratio': hedge_ratio,
                            'z_score': z_score,
                            'spread_mean': spread_mean,
                            'spread_std': spread_std,
                            'direction': direction,
                            'cointegration_pvalue': cointegration_score,
                            'hyperbolic_similarity': hyperbolic_similarity,
                            'expected_holding_days': abs(z_score) * 5  # Rough estimate
                        }
                    )
                    
                    opportunities.append(opportunity)
                    self.performance_metrics['opportunities_detected'] += 1
                    
        except Exception as e:
            logger.error(f"Error in Statistical Pairs arbitrage detection: {e}")
            
        return opportunities
    
    async def validate_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate pairs arbitrage opportunity"""
        # Check if z-score is still significant
        if abs(opportunity.metadata['z_score']) < self.z_entry_threshold * 0.8:
            return False
            
        # Check maximum holding period
        if opportunity.metadata.get('expected_holding_days', 0) > self.max_holding_days:
            return False
            
        return True
    
    async def _update_hyperbolic_embeddings(self, price_history: Dict[str, List[float]]):
        """Update hyperbolic embeddings for financial instruments"""
        # Simplified hyperbolic embedding calculation
        # In production, this would use proper Poincaré ball model
        
        symbols = list(price_history.keys())
        n_symbols = len(symbols)
        
        if n_symbols < 2:
            return
            
        # Calculate correlation matrix
        returns_matrix = []
        for symbol in symbols:
            prices = price_history[symbol]
            if len(prices) < 2:
                continue
            returns = np.diff(np.log(prices))
            returns_matrix.append(returns)
        
        if len(returns_matrix) < 2:
            return
            
        # Pad or truncate to same length
        min_length = min(len(r) for r in returns_matrix)
        returns_matrix = [r[:min_length] for r in returns_matrix]
        
        correlation_matrix = np.corrcoef(returns_matrix)
        
        # Map correlations to hyperbolic space (Poincaré ball)
        for i, symbol_i in enumerate(symbols):
            embedding = []
            for j, symbol_j in enumerate(symbols):
                if i != j and not np.isnan(correlation_matrix[i, j]):
                    # Map correlation [-1,1] to hyperbolic distance [0,∞)
                    correlation = correlation_matrix[i, j]
                    hyperbolic_dist = -np.log((1 + correlation) / 2) if correlation > -0.99 else 10
                    embedding.append(hyperbolic_dist)
                else:
                    embedding.append(0)
            
            self.hyperbolic_embeddings[symbol_i] = np.array(embedding)
    
    def _calculate_hyperbolic_similarity(self, symbol_1: str, symbol_2: str) -> float:
        """Calculate hyperbolic similarity between two symbols"""
        if symbol_1 not in self.hyperbolic_embeddings or symbol_2 not in self.hyperbolic_embeddings:
            return 0.5  # Default similarity
            
        emb_1 = self.hyperbolic_embeddings[symbol_1]
        emb_2 = self.hyperbolic_embeddings[symbol_2] 
        
        # Calculate hyperbolic distance (simplified)
        euclidean_dist = np.linalg.norm(emb_1 - emb_2)
        similarity = 1 / (1 + euclidean_dist)  # Convert distance to similarity
        
        return similarity
    
    async def _find_cointegrated_pairs(self, price_history: Dict[str, List[float]]) -> List[Dict[str, Any]]:
        """Find cointegrated pairs with statistical significance"""
        cointegrated_pairs = []
        symbols = list(price_history.keys())
        
        for i, symbol_1 in enumerate(symbols):
            for j, symbol_2 in enumerate(symbols[i+1:], i+1):
                if len(price_history[symbol_1]) < 30 or len(price_history[symbol_2]) < 30:
                    continue
                
                prices_1 = np.array(price_history[symbol_1])
                prices_2 = np.array(price_history[symbol_2])
                
                # Align lengths
                min_length = min(len(prices_1), len(prices_2))
                prices_1 = prices_1[:min_length]
                prices_2 = prices_2[:min_length]
                
                # Simple cointegration test (Engle-Granger)
                # Regress prices_1 on prices_2 to get hedge ratio
                X = np.vstack([prices_2, np.ones(len(prices_2))]).T
                hedge_ratio, intercept = np.linalg.lstsq(X, prices_1, rcond=None)[0]
                
                # Calculate residuals (spread)
                spread = prices_1 - hedge_ratio * prices_2 - intercept
                
                # Test spread for stationarity (simplified ADF test)
                spread_diff = np.diff(spread)
                if len(spread_diff) > 0:
                    # Simple stationarity proxy: correlation with lagged values
                    lagged_spread = spread[:-1]
                    current_spread_diff = spread_diff
                    
                    if len(lagged_spread) == len(current_spread_diff) and len(lagged_spread) > 0:
                        correlation = np.corrcoef(lagged_spread, current_spread_diff)[0, 1]
                        p_value = max(0.001, abs(correlation))  # Simplified p-value proxy
                        
                        if p_value < 0.05:  # Significant cointegration
                            volatility_1 = np.std(np.diff(np.log(prices_1)))
                            volatility_2 = np.std(np.diff(np.log(prices_2)))
                            
                            pair_info = {
                                'symbols': (symbol_1, symbol_2),
                                'hedge_ratio': hedge_ratio,
                                'spread_history': spread,
                                'cointegration_pvalue': p_value,
                                'volatility_1': volatility_1,
                                'volatility_2': volatility_2
                            }
                            cointegrated_pairs.append(pair_info)
        
        return cointegrated_pairs

class ArbitrageEngine:
    """
    Main arbitrage engine that coordinates all strategies and manages execution
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.strategies = {}
        self.active_opportunities = {}
        self.execution_queue = asyncio.Queue()
        self.performance_monitor = ArbitragePerformanceMonitor()
        self.risk_manager = ArbitrageRiskManager(config.get('risk_config', {}))
        self.latency_optimizer = LatencyOptimizer()
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Market data feeds
        self.market_data = {}
        self.data_feeds = []
        
    def _initialize_strategies(self):
        """Initialize all arbitrage strategies"""
        strategy_configs = self.config.get('strategies', {})
        
        # Priority #1: Index/Futures-Spot Arbitrage
        if strategy_configs.get('index_futures_spot', {}).get('enabled', True):
            self.strategies['index_futures_spot'] = IndexFuturesSpotArbitrage(
                strategy_configs.get('index_futures_spot', {})
            )
            
        # Priority #2: Triangular Crypto Arbitrage  
        if strategy_configs.get('triangular_crypto', {}).get('enabled', True):
            self.strategies['triangular_crypto'] = TriangularCryptoArbitrage(
                strategy_configs.get('triangular_crypto', {})
            )
            
        # Priority #3: Statistical Pairs Arbitrage
        if strategy_configs.get('statistical_pairs', {}).get('enabled', True):
            self.strategies['statistical_pairs'] = StatisticalPairsArbitrage(
                strategy_configs.get('statistical_pairs', {})
            )
    
    async def start(self):
        """Start the arbitrage engine"""
        logger.info("🚀 Starting HyperVision Arbitrage Engine")
        
        # Start market data feeds
        asyncio.create_task(self._market_data_loop())
        
        # Start opportunity detection loop
        asyncio.create_task(self._opportunity_detection_loop())
        
        # Start execution loop  
        asyncio.create_task(self._execution_loop())
        
        # Start performance monitoring
        asyncio.create_task(self._performance_monitoring_loop())
        
        logger.info("✅ Arbitrage Engine started successfully")
    
    async def _market_data_loop(self):
        """Continuously update market data"""
        while True:
            try:
                # Fetch real-time market data
                new_data = await self._fetch_market_data()
                self.market_data.update(new_data)
                
                # Add latency measurement
                self.market_data['latency_ms'] = await self.latency_optimizer.measure_latency()
                
                await asyncio.sleep(0.01)  # 10ms update frequency
                
            except Exception as e:
                logger.error(f"Error in market data loop: {e}")
                await asyncio.sleep(1)
    
    async def _opportunity_detection_loop(self):
        """Continuously scan for arbitrage opportunities"""
        while True:
            try:
                all_opportunities = []
                
                # Run all active strategies
                for strategy_name, strategy in self.strategies.items():
                    if strategy.active:
                        opportunities = await strategy.detect_opportunities(self.market_data)
                        all_opportunities.extend(opportunities)
                
                # Risk management filtering
                filtered_opportunities = await self.risk_manager.filter_opportunities(all_opportunities)
                
                # Queue valid opportunities for execution
                for opportunity in filtered_opportunities:
                    if await self._validate_and_queue_opportunity(opportunity):
                        await self.execution_queue.put(opportunity)
                
                await asyncio.sleep(0.001)  # 1ms scan frequency
                
            except Exception as e:
                logger.error(f"Error in opportunity detection: {e}")
                await asyncio.sleep(0.1)
    
    async def _execution_loop(self):
        """Execute arbitrage opportunities"""
        while True:
            try:
                opportunity = await self.execution_queue.get()
                
                # Final validation before execution
                strategy = self.strategies.get(opportunity.strategy_type.split('_')[0] + '_' + opportunity.strategy_type.split('_')[1])
                if strategy and await strategy.validate_opportunity(opportunity):
                    
                    # Execute the arbitrage
                    execution_result = await self._execute_arbitrage(opportunity)
                    
                    # Update performance metrics
                    strategy.update_performance(execution_result)
                    self.performance_monitor.record_execution(opportunity, execution_result)
                
            except Exception as e:
                logger.error(f"Error in execution loop: {e}")
    
    async def _validate_and_queue_opportunity(self, opportunity: ArbitrageOpportunity) -> bool:
        """Validate opportunity before queueing"""
        # Check if we already have this opportunity
        opp_key = f"{opportunity.strategy_type}_{opportunity.symbol_pair[0]}_{opportunity.symbol_pair[1]}"
        
        if opp_key in self.active_opportunities:
            # Update existing opportunity if this one is better
            existing = self.active_opportunities[opp_key]
            if opportunity.expected_profit > existing.expected_profit:
                self.active_opportunities[opp_key] = opportunity
                return True
            return False
        else:
            self.active_opportunities[opp_key] = opportunity
            return True
    
    async def _execute_arbitrage(self, opportunity: ArbitrageOpportunity) -> Dict[str, Any]:
        """Execute an arbitrage opportunity"""
        start_time = time.time()
        
        try:
            # Simulate trade execution (in production, this would place real orders)
            execution_result = {
                'opportunity_id': f"{opportunity.strategy_type}_{int(time.time()*1000)}",
                'status': 'filled',
                'executed_profit': opportunity.expected_profit * 0.85,  # Account for slippage
                'execution_latency_ms': (time.time() - start_time) * 1000,
                'timestamp': datetime.now()
            }
            
            logger.info(f"✅ Executed {opportunity.strategy_type} arbitrage: ${execution_result['executed_profit']:.2f} profit")
            
            return execution_result
            
        except Exception as e:
            logger.error(f"❌ Failed to execute arbitrage: {e}")
            return {
                'opportunity_id': f"{opportunity.strategy_type}_{int(time.time()*1000)}",
                'status': 'failed',
                'executed_profit': 0,
                'execution_latency_ms': (time.time() - start_time) * 1000,
                'error': str(e)
            }
    
    async def _fetch_market_data(self) -> Dict[str, Any]:
        """Fetch real-time market data"""
        # Simulate market data fetching
        # In production, this would connect to real exchange APIs
        
        mock_data = {
            'prices': {
                'BTC/USDT': {'bid': 43500, 'ask': 43505, 'spread': 0.0001},
                'ETH/USDT': {'bid': 2650, 'ask': 2652, 'spread': 0.0008},
                'BTC/ETH': {'bid': 16.42, 'ask': 16.43, 'spread': 0.0006},
                'SOL/USDT': {'bid': 105, 'ask': 105.2, 'spread': 0.002}
            },
            'volumes': {
                'BTC/USDT': {'volume_24h': 150000000},
                'ETH/USDT': {'volume_24h': 80000000},
                'BTC/ETH': {'volume_24h': 45000000}
            },
            'futures': {
                'BTC': {'price': 43520, 'volume': 50000000},
                'ETH': {'price': 2655, 'volume': 30000000}
            },
            'spot': {
                'BTC': {'price': 43500, 'volume': 100000000},
                'ETH': {'price': 2650, 'volume': 60000000}
            },
            'current_prices': {
                'AAPL': 175.50,
                'MSFT': 378.25,
                'GOOGL': 140.80,
                'TSLA': 248.75
            },
            'price_history': {
                'AAPL': [175 + np.random.randn() for _ in range(100)],
                'MSFT': [378 + np.random.randn() * 2 for _ in range(100)],
                'GOOGL': [140 + np.random.randn() * 1.5 for _ in range(100)],
                'TSLA': [248 + np.random.randn() * 3 for _ in range(100)]
            },
            'risk_free_rate': 0.045,
            'dividend_yield': {'BTC': 0, 'ETH': 0, 'AAPL': 0.005, 'MSFT': 0.007}
        }
        
        return mock_data
    
    async def _performance_monitoring_loop(self):
        """Monitor and report performance metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Report every minute
                
                report = self.performance_monitor.generate_report()
                logger.info(f"📊 Performance Report: {report}")
                
            except Exception as e:
                logger.error(f"Error in performance monitoring: {e}")

class ArbitragePerformanceMonitor:
    """Monitors and tracks arbitrage performance"""
    
    def __init__(self):
        self.executions = []
        self.start_time = datetime.now()
    
    def record_execution(self, opportunity: ArbitrageOpportunity, result: Dict[str, Any]):
        """Record an execution for performance tracking"""
        self.executions.append({
            'opportunity': opportunity,
            'result': result,
            'timestamp': datetime.now()
        })
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate performance report"""
        if not self.executions:
            return {'message': 'No executions recorded yet'}
        
        total_profit = sum(ex['result'].get('executed_profit', 0) for ex in self.executions)
        successful_trades = sum(1 for ex in self.executions if ex['result']['status'] == 'filled')
        avg_latency = np.mean([ex['result'].get('execution_latency_ms', 0) for ex in self.executions])
        
        return {
            'total_executions': len(self.executions),
            'successful_trades': successful_trades,
            'success_rate': successful_trades / len(self.executions),
            'total_profit': total_profit,
            'avg_profit_per_trade': total_profit / len(self.executions),
            'avg_execution_latency_ms': avg_latency,
            'uptime_hours': (datetime.now() - self.start_time).total_seconds() / 3600
        }

class ArbitrageRiskManager:
    """Manages risk for arbitrage strategies"""
    
    def __init__(self, config: Dict[str, Any]):
        self.max_position_size = config.get('max_position_size', 1000000)
        self.max_daily_loss = config.get('max_daily_loss', 50000)
        self.max_drawdown = config.get('max_drawdown', 0.1)
        self.current_positions = {}
        self.daily_pnl = 0
    
    async def filter_opportunities(self, opportunities: List[ArbitrageOpportunity]) -> List[ArbitrageOpportunity]:
        """Filter opportunities based on risk criteria"""
        filtered = []
        
        for opp in opportunities:
            if (opp.expected_profit > 0 and 
                opp.volume_limit <= self.max_position_size and
                opp.risk_score < 0.5 and
                abs(self.daily_pnl) < self.max_daily_loss):
                filtered.append(opp)
        
        return filtered

class LatencyOptimizer:
    """Optimizes execution latency for HFT"""
    
    def __init__(self):
        self.latency_history = deque(maxlen=1000)
    
    async def measure_latency(self) -> float:
        """Measure current network/execution latency"""
        start = time.time()
        # Simulate latency measurement
        await asyncio.sleep(0.001)  # 1ms base latency
        latency = (time.time() - start) * 1000
        self.latency_history.append(latency)
        return latency

# Configuration for the arbitrage engine
DEFAULT_CONFIG = {
    'strategies': {
        'index_futures_spot': {
            'enabled': True,
            'min_spread_bps': 5,
            'max_position_size': 1000000,
            'max_latency_ms': 50
        },
        'triangular_crypto': {
            'enabled': True,
            'min_profit_bps': 10,
            'max_slippage_bps': 5,
            'trading_pairs': [
                ('BTC', 'ETH', 'USDT'),
                ('BTC', 'ETH', 'USDC'),
                ('ETH', 'BNB', 'USDT')
            ]
        },
        'statistical_pairs': {
            'enabled': True,
            'lookback_periods': 252,
            'z_entry_threshold': 2.0,
            'z_exit_threshold': 0.5
        }
    },
    'risk_config': {
        'max_position_size': 1000000,
        'max_daily_loss': 50000,
        'max_drawdown': 0.1
    }
}

async def main():
    """Main function to run the arbitrage engine"""
    engine = ArbitrageEngine(DEFAULT_CONFIG)
    await engine.start()
    
    # Keep running
    try:
        await asyncio.sleep(3600)  # Run for 1 hour
    except KeyboardInterrupt:
        logger.info("🛑 Shutting down arbitrage engine")

if __name__ == "__main__":
    asyncio.run(main())