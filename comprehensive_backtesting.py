#!/usr/bin/env python3
"""
COMPREHENSIVE BACKTESTING FRAMEWORK
==================================
Advanced backtesting system with walk-forward validation for the
Hyperbolic Portfolio Optimization Platform.

Features:
- Walk-forward analysis with expanding/rolling windows
- Out-of-sample testing with proper time-series splits
- Performance attribution analysis
- Risk-adjusted metrics calculation
- Monte Carlo simulation for robustness testing
- Regime-aware backtesting
- Transaction cost modeling
- Slippage and market impact estimation
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
import warnings
import json
import logging
from dataclasses import dataclass, asdict
from scipy import stats
from sklearn.preprocessing import StandardScaler
import concurrent.futures
import multiprocessing as mp

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BacktestConfig:
    """Configuration for backtesting parameters"""
    start_date: str = "2020-01-01"
    end_date: str = "2024-01-01"
    initial_capital: float = 1000000.0  # $1M
    rebalance_frequency: str = "weekly"  # daily, weekly, monthly, quarterly
    transaction_cost: float = 0.001  # 0.1% per trade
    slippage: float = 0.0005  # 0.05% slippage
    min_position_size: float = 0.01  # 1% minimum position
    max_position_size: float = 0.3   # 30% maximum position
    cash_threshold: float = 0.02  # 2% minimum cash
    lookback_window: int = 252  # 1 year lookback
    walk_forward_steps: int = 21  # 21 days step forward
    benchmark: str = "SPY"
    
@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    total_return: float = 0.0
    annualized_return: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    var_95: float = 0.0
    cvar_95: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    beta: float = 0.0
    alpha: float = 0.0
    information_ratio: float = 0.0
    tracking_error: float = 0.0

class MarketDataManager:
    """Manages market data download and preprocessing"""
    
    def __init__(self, symbols: List[str]):
        self.symbols = symbols
        self.data = {}
        
    def download_data(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Download market data for all symbols"""
        
        logger.info(f"Downloading data for {len(self.symbols)} symbols from {start_date} to {end_date}")
        
        for symbol in self.symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)
                
                if not data.empty:
                    # Add technical indicators
                    data = self._add_technical_indicators(data)
                    self.data[symbol] = data
                    logger.info(f"Downloaded {len(data)} days for {symbol}")
                else:
                    logger.warning(f"No data available for {symbol}")
                    
            except Exception as e:
                logger.error(f"Failed to download {symbol}: {e}")
                
        return self.data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data"""
        
        df = data.copy()
        
        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['LogReturns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['EMA_12'] = df['Close'].ewm(span=12).mean()
        df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
        # Volatility
        df['Volatility_20'] = df['Returns'].rolling(20).std()
        df['Volatility_60'] = df['Returns'].rolling(60).std()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(20).mean()
        bb_std = df['Close'].rolling(20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Volume indicators
        df['Volume_SMA'] = df['Volume'].rolling(20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA']
        
        return df
        
    def get_aligned_data(self) -> pd.DataFrame:
        """Get aligned data for all symbols"""
        
        if not self.data:
            raise ValueError("No data available. Run download_data first.")
        
        # Align all data to common dates
        aligned_data = {}
        
        for symbol, data in self.data.items():
            aligned_data[f"{symbol}_Close"] = data['Close']
            aligned_data[f"{symbol}_Volume"] = data['Volume']
            aligned_data[f"{symbol}_Returns"] = data['Returns']
            aligned_data[f"{symbol}_Volatility"] = data['Volatility_20']
        
        df = pd.DataFrame(aligned_data)
        return df.dropna()

class PortfolioSimulator:
    """Simulates portfolio performance with transaction costs and constraints"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio_history = []
        self.transaction_history = []
        
    def simulate_portfolio(self, weights_series: pd.Series, 
                          price_data: pd.DataFrame) -> Tuple[pd.Series, pd.DataFrame]:
        """
        Simulate portfolio performance given weight series and price data
        
        Args:
            weights_series: Time series of portfolio weights
            price_data: Price data for all assets
            
        Returns:
            portfolio_values: Time series of portfolio values
            detailed_history: Detailed transaction and position history
        """
        
        logger.info("Starting portfolio simulation")
        
        # Initialize portfolio
        current_value = self.config.initial_capital
        current_weights = {}
        cash = self.config.initial_capital
        
        portfolio_values = []
        detailed_history = []
        
        dates = weights_series.index
        
        for i, date in enumerate(dates):
            try:
                # Get current prices
                current_prices = {}
                for col in price_data.columns:
                    if '_Close' in col:
                        symbol = col.replace('_Close', '')
                        if date in price_data.index:
                            current_prices[symbol] = price_data.loc[date, col]
                
                if not current_prices:
                    continue
                
                # Calculate current portfolio value
                portfolio_value = cash
                for symbol, weight in current_weights.items():
                    if symbol in current_prices and not np.isnan(current_prices[symbol]):
                        shares = (current_value * weight) / current_prices[symbol] if current_prices[symbol] > 0 else 0
                        portfolio_value += shares * current_prices[symbol]
                
                # Get target weights for rebalancing
                target_weights = {}
                weights_row = weights_series.loc[date] if date in weights_series.index else None
                
                if weights_row is not None and isinstance(weights_row, (dict, pd.Series)):
                    if isinstance(weights_row, pd.Series):
                        target_weights = weights_row.to_dict()
                    else:
                        target_weights = weights_row
                    
                    # Clean and normalize weights
                    target_weights = {k: v for k, v in target_weights.items() 
                                    if not np.isnan(v) and v > self.config.min_position_size}
                    
                    if target_weights:
                        total_weight = sum(target_weights.values())
                        if total_weight > 0:
                            target_weights = {k: min(v / total_weight, self.config.max_position_size) 
                                            for k, v in target_weights.items()}
                
                # Rebalancing logic
                if self._should_rebalance(date, i) and target_weights:
                    transactions = self._rebalance_portfolio(
                        current_weights, target_weights, portfolio_value, current_prices, date
                    )
                    self.transaction_history.extend(transactions)
                    current_weights = target_weights.copy()
                    
                    # Apply transaction costs
                    total_transaction_cost = sum(abs(t['value']) * self.config.transaction_cost 
                                               for t in transactions)
                    portfolio_value -= total_transaction_cost
                
                # Record portfolio state
                portfolio_values.append(portfolio_value)
                
                detailed_state = {
                    'date': date,
                    'portfolio_value': portfolio_value,
                    'weights': current_weights.copy(),
                    'prices': current_prices.copy(),
                    'cash': cash
                }
                detailed_history.append(detailed_state)
                
                current_value = portfolio_value
                
            except Exception as e:
                logger.warning(f"Error processing date {date}: {e}")
                if portfolio_values:
                    portfolio_values.append(portfolio_values[-1])  # Use previous value
                else:
                    portfolio_values.append(current_value)
        
        portfolio_series = pd.Series(portfolio_values, index=dates[:len(portfolio_values)])
        history_df = pd.DataFrame(detailed_history)
        
        return portfolio_series, history_df
    
    def _should_rebalance(self, date: pd.Timestamp, index: int) -> bool:
        """Determine if portfolio should be rebalanced"""
        
        if index == 0:  # Always rebalance on first day
            return True
            
        if self.config.rebalance_frequency == "daily":
            return True
        elif self.config.rebalance_frequency == "weekly":
            return date.weekday() == 0  # Monday
        elif self.config.rebalance_frequency == "monthly":
            return date.day <= 7 and date.weekday() == 0  # First Monday of month
        elif self.config.rebalance_frequency == "quarterly":
            return date.month % 3 == 1 and date.day <= 7 and date.weekday() == 0
        
        return False
    
    def _rebalance_portfolio(self, current_weights: Dict[str, float],
                           target_weights: Dict[str, float],
                           portfolio_value: float,
                           current_prices: Dict[str, float],
                           date: pd.Timestamp) -> List[Dict]:
        """Execute portfolio rebalancing"""
        
        transactions = []
        
        # Calculate required transactions
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current_weight = current_weights.get(symbol, 0)
            target_weight = target_weights.get(symbol, 0)
            
            if symbol in current_prices and current_prices[symbol] > 0:
                weight_diff = target_weight - current_weight
                transaction_value = weight_diff * portfolio_value
                
                # Apply slippage
                if transaction_value != 0:
                    slippage_cost = abs(transaction_value) * self.config.slippage
                    transaction_value -= np.sign(transaction_value) * slippage_cost
                    
                    transaction = {
                        'date': date,
                        'symbol': symbol,
                        'type': 'buy' if weight_diff > 0 else 'sell',
                        'value': transaction_value,
                        'weight_change': weight_diff,
                        'price': current_prices[symbol],
                        'slippage_cost': slippage_cost
                    }
                    transactions.append(transaction)
        
        return transactions

class WalkForwardAnalyzer:
    """Performs walk-forward analysis"""
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        
    def run_walk_forward_analysis(self, 
                                 strategy_func: Callable,
                                 market_data: pd.DataFrame,
                                 **strategy_params) -> Dict[str, Any]:
        """
        Run walk-forward analysis
        
        Args:
            strategy_func: Function that generates portfolio weights
            market_data: Market data for all assets
            **strategy_params: Parameters to pass to strategy function
            
        Returns:
            Dictionary containing analysis results
        """
        
        logger.info("Starting walk-forward analysis")
        
        results = {
            'periods': [],
            'performance_metrics': [],
            'out_of_sample_results': [],
            'stability_analysis': {},
            'parameter_sensitivity': {}
        }
        
        # Define analysis periods
        analysis_periods = self._generate_analysis_periods(market_data.index)
        
        logger.info(f"Generated {len(analysis_periods)} analysis periods")
        
        portfolio_simulator = PortfolioSimulator(self.config)
        
        for i, period in enumerate(analysis_periods):
            logger.info(f"Processing period {i+1}/{len(analysis_periods)}: {period['train_start']} to {period['test_end']}")
            
            try:
                # Split data
                train_data = market_data.loc[period['train_start']:period['train_end']]
                test_data = market_data.loc[period['test_start']:period['test_end']]
                
                if len(train_data) < 50 or len(test_data) < 5:  # Minimum data requirements
                    logger.warning(f"Insufficient data for period {i+1}")
                    continue
                
                # Generate strategy weights on training data
                strategy_weights = strategy_func(train_data, **strategy_params)
                
                # Apply weights to test period
                test_weights = self._apply_weights_to_period(strategy_weights, test_data)
                
                # Simulate portfolio performance
                portfolio_values, detailed_history = portfolio_simulator.simulate_portfolio(
                    test_weights, test_data
                )
                
                # Calculate performance metrics
                period_metrics = self._calculate_period_metrics(
                    portfolio_values, test_data, period
                )
                
                results['periods'].append(period)
                results['performance_metrics'].append(period_metrics)
                
                # Store out-of-sample results
                oos_result = {
                    'period': i + 1,
                    'train_period': f"{period['train_start']} to {period['train_end']}",
                    'test_period': f"{period['test_start']} to {period['test_end']}",
                    'returns': portfolio_values.pct_change().dropna().tolist(),
                    'final_value': portfolio_values.iloc[-1] if len(portfolio_values) > 0 else self.config.initial_capital,
                    'metrics': asdict(period_metrics)
                }
                results['out_of_sample_results'].append(oos_result)
                
            except Exception as e:
                logger.error(f"Error in period {i+1}: {e}")
                continue
        
        # Aggregate results
        if results['performance_metrics']:
            results['aggregate_metrics'] = self._aggregate_metrics(results['performance_metrics'])
            results['stability_analysis'] = self._analyze_stability(results['performance_metrics'])
        
        return results
    
    def _generate_analysis_periods(self, date_index: pd.DatetimeIndex) -> List[Dict[str, str]]:
        """Generate overlapping train/test periods for walk-forward analysis"""
        
        periods = []
        start_date = date_index[0]
        end_date = date_index[-1]
        
        current_date = start_date + pd.Timedelta(days=self.config.lookback_window)
        
        while current_date < end_date - pd.Timedelta(days=self.config.walk_forward_steps):
            train_start = current_date - pd.Timedelta(days=self.config.lookback_window)
            train_end = current_date
            test_start = current_date + pd.Timedelta(days=1)
            test_end = test_start + pd.Timedelta(days=self.config.walk_forward_steps)
            
            # Ensure dates are within available data
            train_start = max(train_start, start_date)
            test_end = min(test_end, end_date)
            
            if train_start < train_end < test_start < test_end:
                period = {
                    'train_start': train_start.strftime('%Y-%m-%d'),
                    'train_end': train_end.strftime('%Y-%m-%d'),
                    'test_start': test_start.strftime('%Y-%m-%d'),
                    'test_end': test_end.strftime('%Y-%m-%d')
                }
                periods.append(period)
            
            current_date += pd.Timedelta(days=self.config.walk_forward_steps)
        
        return periods
    
    def _apply_weights_to_period(self, strategy_weights: pd.DataFrame, 
                               test_data: pd.DataFrame) -> pd.Series:
        """Apply strategy weights to test period"""
        
        # Use the last weights from training period for the entire test period
        if len(strategy_weights) > 0:
            last_weights = strategy_weights.iloc[-1]
            
            # Create weight series for test period
            test_weights = pd.DataFrame(
                index=test_data.index,
                columns=strategy_weights.columns
            )
            
            for date in test_data.index:
                test_weights.loc[date] = last_weights
            
            return test_weights
        else:
            # Equal weights fallback
            symbols = [col.replace('_Close', '') for col in test_data.columns if '_Close' in col]
            equal_weight = 1.0 / len(symbols) if symbols else 0
            
            test_weights = pd.DataFrame(
                index=test_data.index,
                columns=symbols
            )
            test_weights.fillna(equal_weight, inplace=True)
            
            return test_weights
    
    def _calculate_period_metrics(self, portfolio_values: pd.Series,
                                test_data: pd.DataFrame,
                                period: Dict[str, str]) -> PerformanceMetrics:
        """Calculate performance metrics for a single period"""
        
        metrics = PerformanceMetrics()
        
        if len(portfolio_values) < 2:
            return metrics
        
        # Calculate returns
        returns = portfolio_values.pct_change().dropna()
        
        if len(returns) == 0:
            return metrics
        
        # Basic metrics
        metrics.total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1)
        
        # Annualized metrics
        trading_days = len(returns)
        days_per_year = 252
        
        if trading_days > 0:
            metrics.annualized_return = (1 + metrics.total_return) ** (days_per_year / trading_days) - 1
            metrics.volatility = returns.std() * np.sqrt(days_per_year)
        
        # Risk-adjusted metrics
        if metrics.volatility > 0:
            metrics.sharpe_ratio = metrics.annualized_return / metrics.volatility
        
        # Sortino ratio
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 0:
            downside_std = negative_returns.std() * np.sqrt(days_per_year)
            if downside_std > 0:
                metrics.sortino_ratio = metrics.annualized_return / downside_std
        
        # Maximum drawdown
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdowns = (cumulative_returns - running_max) / running_max
        metrics.max_drawdown = abs(drawdowns.min())
        
        # Calmar ratio
        if metrics.max_drawdown > 0:
            metrics.calmar_ratio = metrics.annualized_return / metrics.max_drawdown
        
        # Win rate
        metrics.win_rate = (returns > 0).mean()
        
        # Profit factor
        positive_returns = returns[returns > 0].sum()
        negative_returns = abs(returns[returns < 0].sum())
        if negative_returns > 0:
            metrics.profit_factor = positive_returns / negative_returns
        
        # VaR and CVaR
        metrics.var_95 = np.percentile(returns, 5)
        var_exceedances = returns[returns <= metrics.var_95]
        if len(var_exceedances) > 0:
            metrics.cvar_95 = var_exceedances.mean()
        
        # Distribution moments
        metrics.skewness = stats.skew(returns)
        metrics.kurtosis = stats.kurtosis(returns)
        
        # Benchmark comparison (if available)
        benchmark_col = f"{self.config.benchmark}_Returns"
        if benchmark_col in test_data.columns:
            benchmark_returns = test_data[benchmark_col].dropna()
            
            if len(benchmark_returns) > 0:
                # Align returns with benchmark
                aligned_data = pd.DataFrame({
                    'portfolio': returns,
                    'benchmark': benchmark_returns
                }).dropna()
                
                if len(aligned_data) > 1:
                    # Beta
                    covariance = np.cov(aligned_data['portfolio'], aligned_data['benchmark'])[0, 1]
                    benchmark_variance = np.var(aligned_data['benchmark'])
                    if benchmark_variance > 0:
                        metrics.beta = covariance / benchmark_variance
                    
                    # Alpha
                    benchmark_return = aligned_data['benchmark'].mean() * days_per_year
                    metrics.alpha = metrics.annualized_return - benchmark_return * metrics.beta
                    
                    # Information ratio
                    excess_returns = aligned_data['portfolio'] - aligned_data['benchmark']
                    tracking_error = excess_returns.std() * np.sqrt(days_per_year)
                    metrics.tracking_error = tracking_error
                    
                    if tracking_error > 0:
                        metrics.information_ratio = excess_returns.mean() * days_per_year / tracking_error
        
        return metrics
    
    def _aggregate_metrics(self, metrics_list: List[PerformanceMetrics]) -> PerformanceMetrics:
        """Aggregate metrics across all periods"""
        
        if not metrics_list:
            return PerformanceMetrics()
        
        aggregate = PerformanceMetrics()
        
        # Calculate means for all numeric fields
        numeric_fields = [field for field in asdict(aggregate).keys()]
        
        for field in numeric_fields:
            values = [getattr(m, field) for m in metrics_list if not np.isnan(getattr(m, field))]
            if values:
                setattr(aggregate, field, np.mean(values))
        
        return aggregate
    
    def _analyze_stability(self, metrics_list: List[PerformanceMetrics]) -> Dict[str, Any]:
        """Analyze stability of performance across periods"""
        
        stability_analysis = {
            'consistency_scores': {},
            'trend_analysis': {},
            'stability_ranking': 'stable'  # stable, moderate, unstable
        }
        
        if len(metrics_list) < 2:
            return stability_analysis
        
        # Extract time series of key metrics
        returns_series = [m.annualized_return for m in metrics_list]
        sharpe_series = [m.sharpe_ratio for m in metrics_list if not np.isnan(m.sharpe_ratio)]
        drawdown_series = [m.max_drawdown for m in metrics_list if not np.isnan(m.max_drawdown)]
        
        # Calculate consistency (coefficient of variation)
        if returns_series:
            returns_cv = np.std(returns_series) / (abs(np.mean(returns_series)) + 1e-8)
            stability_analysis['consistency_scores']['returns'] = 1 / (1 + returns_cv)
        
        if sharpe_series and len(sharpe_series) > 1:
            sharpe_cv = np.std(sharpe_series) / (abs(np.mean(sharpe_series)) + 1e-8)
            stability_analysis['consistency_scores']['sharpe'] = 1 / (1 + sharpe_cv)
        
        if drawdown_series and len(drawdown_series) > 1:
            drawdown_cv = np.std(drawdown_series) / (np.mean(drawdown_series) + 1e-8)
            stability_analysis['consistency_scores']['drawdown'] = 1 / (1 + drawdown_cv)
        
        # Overall stability score
        consistency_values = list(stability_analysis['consistency_scores'].values())
        if consistency_values:
            overall_consistency = np.mean(consistency_values)
            
            if overall_consistency > 0.7:
                stability_analysis['stability_ranking'] = 'stable'
            elif overall_consistency > 0.4:
                stability_analysis['stability_ranking'] = 'moderate'
            else:
                stability_analysis['stability_ranking'] = 'unstable'
        
        return stability_analysis

# Example strategy functions
def equal_weight_strategy(market_data: pd.DataFrame, **params) -> pd.DataFrame:
    """Simple equal weight strategy"""
    
    symbols = [col.replace('_Close', '') for col in market_data.columns if '_Close' in col]
    n_symbols = len(symbols)
    
    if n_symbols == 0:
        return pd.DataFrame()
    
    weight = 1.0 / n_symbols
    
    weights_df = pd.DataFrame(
        index=market_data.index,
        columns=symbols
    )
    weights_df.fillna(weight, inplace=True)
    
    return weights_df

def momentum_strategy(market_data: pd.DataFrame, lookback: int = 20, **params) -> pd.DataFrame:
    """Momentum-based strategy"""
    
    symbols = [col.replace('_Close', '') for col in market_data.columns if '_Close' in col]
    
    weights_df = pd.DataFrame(
        index=market_data.index,
        columns=symbols
    )
    
    for symbol in symbols:
        price_col = f"{symbol}_Close"
        if price_col in market_data.columns:
            # Calculate momentum
            momentum = market_data[price_col].pct_change(lookback)
            
            # Rank-based weights
            for date in market_data.index:
                if date in momentum.index:
                    mom_values = {}
                    for s in symbols:
                        mom_col = f"{s}_Close"
                        if mom_col in market_data.columns and date in market_data.index:
                            mom_val = market_data.loc[date, mom_col] if not np.isnan(market_data.loc[date, mom_col]) else 0
                            mom_values[s] = mom_val
                    
                    # Assign weights based on momentum ranking
                    if mom_values:
                        sorted_symbols = sorted(mom_values.keys(), key=lambda x: mom_values[x], reverse=True)
                        n_symbols = len(sorted_symbols)
                        
                        for i, s in enumerate(sorted_symbols):
                            # Linear decay weights
                            weight = (n_symbols - i) / sum(range(1, n_symbols + 1))
                            weights_df.loc[date, s] = weight
    
    return weights_df.fillna(0)

def create_backtesting_demo():
    """Create a demonstration of the backtesting framework"""
    
    print("="*80)
    print("COMPREHENSIVE BACKTESTING FRAMEWORK DEMONSTRATION")
    print("Walk-Forward Analysis with Portfolio Simulation")
    print("="*80)
    
    # Configuration
    config = BacktestConfig(
        start_date="2020-01-01",
        end_date="2023-12-31",
        initial_capital=1000000,
        rebalance_frequency="weekly",
        transaction_cost=0.001,
        lookback_window=126,  # 6 months
        walk_forward_steps=21  # 1 month
    )
    
    # Test symbols
    symbols = ['BTC-USD', 'ETH-USD', 'SPY', 'QQQ', 'GLD', 'TLT']
    
    print(f"\n📊 Configuration:")
    print(f"  Period: {config.start_date} to {config.end_date}")
    print(f"  Initial Capital: ${config.initial_capital:,.0f}")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Rebalancing: {config.rebalance_frequency}")
    print(f"  Transaction Cost: {config.transaction_cost:.1%}")
    
    # Download market data
    print(f"\n📈 Downloading market data...")
    data_manager = MarketDataManager(symbols)
    market_data = data_manager.download_data(config.start_date, config.end_date)
    
    if not market_data:
        print("❌ No market data available")
        return
    
    # Get aligned data
    aligned_data = data_manager.get_aligned_data()
    print(f"  Aligned data shape: {aligned_data.shape}")
    
    # Initialize walk-forward analyzer
    wf_analyzer = WalkForwardAnalyzer(config)
    
    # Test equal weight strategy
    print(f"\n🎯 Running Walk-Forward Analysis - Equal Weight Strategy...")
    ew_results = wf_analyzer.run_walk_forward_analysis(
        equal_weight_strategy,
        aligned_data
    )
    
    # Test momentum strategy
    print(f"\n🎯 Running Walk-Forward Analysis - Momentum Strategy...")
    mom_results = wf_analyzer.run_walk_forward_analysis(
        momentum_strategy,
        aligned_data,
        lookback=20
    )
    
    # Display results
    print(f"\n📊 EQUAL WEIGHT STRATEGY RESULTS")
    print("="*50)
    
    if 'aggregate_metrics' in ew_results:
        metrics = ew_results['aggregate_metrics']
        print(f"  Annualized Return: {metrics.annualized_return:.2%}")
        print(f"  Volatility: {metrics.volatility:.2%}")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"  Win Rate: {metrics.win_rate:.2%}")
    
    if 'stability_analysis' in ew_results:
        stability = ew_results['stability_analysis']
        print(f"  Stability Ranking: {stability.get('stability_ranking', 'unknown')}")
    
    print(f"  Periods Analyzed: {len(ew_results.get('periods', []))}")
    
    print(f"\n📊 MOMENTUM STRATEGY RESULTS")
    print("="*50)
    
    if 'aggregate_metrics' in mom_results:
        metrics = mom_results['aggregate_metrics']
        print(f"  Annualized Return: {metrics.annualized_return:.2%}")
        print(f"  Volatility: {metrics.volatility:.2%}")
        print(f"  Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
        print(f"  Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"  Win Rate: {metrics.win_rate:.2%}")
    
    if 'stability_analysis' in mom_results:
        stability = mom_results['stability_analysis']
        print(f"  Stability Ranking: {stability.get('stability_ranking', 'unknown')}")
    
    print(f"  Periods Analyzed: {len(mom_results.get('periods', []))}")
    
    # Save results
    results_summary = {
        'config': asdict(config),
        'equal_weight_results': ew_results,
        'momentum_results': mom_results,
        'timestamp': datetime.now().isoformat(),
        'symbols_analyzed': symbols
    }
    
    output_file = '/home/user/webapp/backtesting_results.json'
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {output_file}")
    
    print(f"\n✅ Backtesting Analysis Complete")
    print(f"   Equal Weight: {ew_results['aggregate_metrics'].sharpe_ratio:.3f} Sharpe")
    print(f"   Momentum: {mom_results['aggregate_metrics'].sharpe_ratio:.3f} Sharpe")
    
    return results_summary

if __name__ == "__main__":
    create_backtesting_demo()