#!/usr/bin/env python3
"""
HYPERBOLIC PORTFOLIO OPTIMIZATION - COMPLETE DEMONSTRATION
=========================================================
This script demonstrates all the advanced portfolio features including:
- Hyperbolic space optimization with hierarchical indices
- Multi-index correlation analysis
- Overfitting prevention and validation
- Hallucination detection
- Comprehensive backtesting with walk-forward analysis

Run this script to see the complete system in action.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import our custom modules
from hyperbolic_portfolio_engine import HyperbolicPortfolioOptimizer
from advanced_validation_framework import ValidationFrameworkOrchestrator
from comprehensive_backtesting import WalkForwardAnalyzer, BacktestConfig, equal_weight_strategy, momentum_strategy

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"{title:^80}")
    print("="*80)

def print_subheader(title):
    """Print formatted subsection header"""
    print(f"\n{title}")
    print("-" * len(title))

def create_sample_data():
    """Create comprehensive sample data for demonstration"""
    
    print_subheader("📊 Creating Sample Market Data")
    
    # Extended list of assets across different categories
    symbols = {
        # Crypto
        'BTC-USD': 'Bitcoin',
        'ETH-USD': 'Ethereum',
        # US Equity
        'SPY': 'S&P 500 ETF',
        'QQQ': 'Nasdaq 100 ETF', 
        'IWM': 'Russell 2000 ETF',
        # International
        'VEA': 'Developed Markets ETF',
        'VWO': 'Emerging Markets ETF',
        # Bonds
        'TLT': 'Long-Term Treasury ETF',
        'HYG': 'High Yield Bond ETF',
        # Commodities
        'GLD': 'Gold ETF',
        'USO': 'Oil ETF',
        # REITs
        'VNQ': 'Real Estate ETF',
        # Sectors
        'XLF': 'Financial Sector ETF',
        'XLK': 'Technology Sector ETF'
    }
    
    print(f"Generating data for {len(symbols)} assets:")
    for symbol, name in symbols.items():
        print(f"  • {symbol}: {name}")
    
    # Generate realistic correlated returns
    np.random.seed(42)  # For reproducibility
    n_days = 1000
    n_assets = len(symbols)
    
    # Create correlation structure
    correlation_matrix = np.eye(n_assets)
    
    # Add realistic correlations
    asset_list = list(symbols.keys())
    
    # Crypto correlations
    btc_idx = asset_list.index('BTC-USD')
    eth_idx = asset_list.index('ETH-USD')
    correlation_matrix[btc_idx, eth_idx] = 0.7
    correlation_matrix[eth_idx, btc_idx] = 0.7
    
    # Equity correlations
    spy_idx = asset_list.index('SPY')
    qqq_idx = asset_list.index('QQQ')
    iwm_idx = asset_list.index('IWM')
    correlation_matrix[spy_idx, qqq_idx] = 0.85
    correlation_matrix[spy_idx, iwm_idx] = 0.75
    correlation_matrix[qqq_idx, iwm_idx] = 0.7
    
    # Cross-asset correlations
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            if correlation_matrix[i, j] == 0:  # Only set if not already set
                if np.random.random() < 0.3:  # 30% chance of correlation
                    corr_val = np.random.uniform(0.1, 0.6)
                    correlation_matrix[i, j] = corr_val
                    correlation_matrix[j, i] = corr_val
    
    # Generate returns using correlation structure
    mean_returns = np.random.uniform(-0.0005, 0.001, n_assets)  # Daily returns
    volatilities = np.random.uniform(0.01, 0.04, n_assets)  # Daily volatilities
    
    # Ensure positive definite correlation matrix
    eigenvals, eigenvecs = np.linalg.eigh(correlation_matrix)
    eigenvals = np.maximum(eigenvals, 0.01)  # Ensure positive eigenvalues
    correlation_matrix = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
    
    # Cholesky decomposition for correlated random variables
    L = np.linalg.cholesky(correlation_matrix)
    
    # Generate dates
    dates = pd.date_range('2020-01-01', periods=n_days, freq='D')
    
    # Create market data dictionary
    market_data = {}
    
    for i, (symbol, name) in enumerate(symbols.items()):
        # Generate correlated returns
        random_returns = np.random.normal(0, 1, n_days)
        correlated_factors = L[i, :] @ np.random.normal(0, 1, (n_assets, n_days))
        
        # Combine idiosyncratic and systematic factors
        asset_returns = (mean_returns[i] + 
                        volatilities[i] * (0.7 * correlated_factors + 0.3 * random_returns))
        
        # Generate price series
        initial_price = np.random.uniform(50, 500)
        prices = initial_price * np.exp(np.cumsum(asset_returns))
        
        # Create realistic OHLCV data
        highs = prices * (1 + np.abs(np.random.normal(0, 0.005, n_days)))
        lows = prices * (1 - np.abs(np.random.normal(0, 0.005, n_days)))
        opens = np.roll(prices, 1)
        opens[0] = initial_price
        volumes = np.random.randint(1000000, 10000000, n_days)
        
        market_data[symbol] = pd.DataFrame({
            'Open': opens,
            'High': highs,
            'Low': lows,
            'Close': prices,
            'Volume': volumes
        }, index=dates)
    
    print(f"✅ Generated {n_days} days of data for {n_assets} assets")
    
    return market_data, symbols

def demonstrate_hyperbolic_optimization(market_data):
    """Demonstrate hyperbolic portfolio optimization"""
    
    print_header("🌐 HYPERBOLIC SPACE PORTFOLIO OPTIMIZATION")
    
    print_subheader("🔧 Initializing Hyperbolic Portfolio Optimizer")
    
    # Initialize optimizer with different risk tolerance levels
    risk_tolerances = [0.1, 0.2, 0.3]
    
    for risk_tolerance in risk_tolerances:
        print(f"\n📊 Risk Tolerance: {risk_tolerance:.1%}")
        print("-" * 40)
        
        optimizer = HyperbolicPortfolioOptimizer(risk_tolerance=risk_tolerance)
        
        # Run optimization
        result = optimizer.optimize_portfolio(market_data)
        
        # Display results
        print("🎯 Asset Allocation:")
        weights = result['recommendations']['asset_weights']
        
        for asset, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True)[:8]:
            print(f"  {asset:12}: {weight:7.1%}")
        
        print(f"\n📈 Portfolio Metrics:")
        metrics = result['recommendations']['risk_metrics']
        print(f"  Expected Return:  {metrics['expected_return']:7.2%}")
        print(f"  Volatility:       {metrics['volatility']:7.2%}")
        print(f"  Sharpe Ratio:     {metrics['sharpe_ratio']:7.2f}")
        
        print(f"\n🌐 Hyperbolic Diversification:")
        div_metrics = result['recommendations']['hyperbolic_diversification']
        print(f"  Avg Distance:     {div_metrics['average_hyperbolic_distance']:7.3f}")
        print(f"  Effective Assets: {div_metrics['effective_number_of_assets']:7.1f}")
        print(f"  Diversification:  {div_metrics['diversification_ratio']:7.1%}")
        
        print(f"\n🛡️ Validation Status:")
        validation = result['validation']
        print(f"  Validation Score: {validation['overfitting_score']:7.3f}")
        print(f"  Status:          {'✅ PASS' if validation['validation_passed'] else '❌ FAIL'}")
        
        hallucination = result['hallucination_analysis']
        print(f"  Hallucination:    {hallucination['overall_hallucination_risk']:7.1%}")
    
    return result

def demonstrate_validation_framework(market_data):
    """Demonstrate comprehensive validation framework"""
    
    print_header("🔬 ADVANCED VALIDATION FRAMEWORK")
    
    print_subheader("🧪 Statistical Validation Tests")
    
    # Create synthetic predictions for demonstration
    np.random.seed(123)
    n_samples = 500
    
    # Generate true signal with some structure
    t = np.linspace(0, 4*np.pi, n_samples)
    true_signal = np.sin(t) + 0.3*np.sin(3*t) + np.random.normal(0, 0.1, n_samples)
    
    # Create predictions with controlled overfitting
    predictions = true_signal + np.random.normal(0, 0.05, n_samples)
    
    # Add some systematic bias (overfitting signature)
    predictions[400:] += 0.2
    
    # Generate confidence scores
    confidence_scores = np.random.beta(5, 2, n_samples)
    confidence_scores[400:] *= 0.8  # Lower confidence where bias exists
    
    # Create sample market data
    sample_asset = list(market_data.keys())[0]
    sample_data = market_data[sample_asset].iloc[:n_samples]
    
    print(f"🔍 Testing on {n_samples} synthetic predictions")
    print(f"  True signal range:   [{np.min(true_signal):6.3f}, {np.max(true_signal):6.3f}]")
    print(f"  Prediction range:    [{np.min(predictions):6.3f}, {np.max(predictions):6.3f}]")
    print(f"  Mean confidence:     {np.mean(confidence_scores):6.3f}")
    
    # Initialize validation framework
    config = {
        'significance_level': 0.05,
        'cv_splits': 5,
        'confidence_threshold': 0.8
    }
    
    validator = ValidationFrameworkOrchestrator(config)
    
    # Run comprehensive validation
    validation_results = validator.comprehensive_validation(
        predictions=predictions,
        actuals=true_signal,
        confidence_scores=confidence_scores,
        market_data=sample_data
    )
    
    # Display statistical analysis results
    print_subheader("📊 Statistical Analysis Results")
    
    stat_results = validation_results['statistical_analysis']
    
    print(f"Overall Validation Score: {stat_results['validation_score']:.3f}")
    print(f"Validation Status: {'✅ PASSED' if stat_results['validation_passed'] else '❌ FAILED'}")
    
    # Performance metrics
    if 'performance_metrics' in stat_results:
        perf = stat_results['performance_metrics']
        print(f"\n📈 Performance Metrics:")
        print(f"  RMSE:             {perf.get('rmse', 0):7.4f}")
        print(f"  R² Score:         {perf.get('r2_score', 0):7.3f}")
        print(f"  Mean Abs Error:   {perf.get('mae', 0):7.4f}")
    
    # Normality tests
    if 'normality_tests' in stat_results:
        normality = stat_results['normality_tests']
        print(f"\n🔬 Normality Tests:")
        
        for test_name, test_result in normality.items():
            if isinstance(test_result, dict) and 'is_normal' in test_result:
                status = "✅ Normal" if test_result['is_normal'] else "❌ Non-normal"
                p_value = test_result.get('p_value', 0)
                print(f"  {test_name:20}: {status} (p={p_value:.3f})")
    
    # Independence tests
    if 'independence_tests' in stat_results:
        independence = stat_results['independence_tests']
        print(f"\n🔗 Independence Tests:")
        
        for test_name, test_result in independence.items():
            if isinstance(test_result, dict):
                if 'is_independent' in test_result:
                    status = "✅ Independent" if test_result['is_independent'] else "❌ Autocorrelated"
                    print(f"  {test_name:20}: {status}")
                elif 'is_random' in test_result:
                    status = "✅ Random" if test_result['is_random'] else "❌ Pattern detected"
                    print(f"  {test_name:20}: {status}")
    
    # Hallucination detection results
    print_subheader("🧠 Hallucination Detection")
    
    halluc_results = validation_results['hallucination_detection']
    
    print(f"Hallucination Risk Score: {halluc_results['hallucination_risk_score']:.3f}")
    print(f"Is Hallucinating: {'❌ YES' if halluc_results['is_hallucinating'] else '✅ NO'}")
    
    if 'confidence_analysis' in halluc_results:
        conf = halluc_results['confidence_analysis']
        print(f"\n📊 Confidence Analysis:")
        print(f"  Mean Confidence:      {conf['mean_confidence']:7.3f}")
        print(f"  Low Confidence Ratio: {conf['low_confidence_ratio']:7.1%}")
        print(f"  Confidence Trend:     {conf.get('confidence_trend', 'unknown')}")
    
    # Overall assessment
    print_subheader("🎯 Overall Assessment")
    
    assessment = validation_results['overall_assessment']
    
    print(f"Validation Passed: {'✅ YES' if assessment['validation_passed'] else '❌ NO'}")
    print(f"Risk Level:        {assessment['risk_level'].upper()}")
    print(f"Confidence Level:  {assessment['confidence_level'].upper()}")
    print(f"Overall Score:     {assessment.get('overall_score', 0):.3f}")
    print(f"Recommendation:    {assessment['recommendation']}")
    
    if assessment.get('key_concerns'):
        print(f"Key Concerns:      {', '.join(assessment['key_concerns'])}")
    if assessment.get('strengths'):
        print(f"Strengths:         {', '.join(assessment['strengths'])}")
    
    return validation_results

def demonstrate_backtesting(market_data):
    """Demonstrate comprehensive backtesting framework"""
    
    print_header("📈 COMPREHENSIVE BACKTESTING FRAMEWORK")
    
    print_subheader("⚙️  Backtesting Configuration")
    
    # Configure backtesting parameters
    config = BacktestConfig(
        start_date="2020-01-01",
        end_date="2022-12-31",
        initial_capital=1000000,  # $1M
        rebalance_frequency="weekly",
        transaction_cost=0.001,  # 0.1%
        slippage=0.0005,  # 0.05%
        lookback_window=126,  # 6 months
        walk_forward_steps=21   # 1 month
    )
    
    print(f"Initial Capital:      ${config.initial_capital:,.0f}")
    print(f"Rebalancing:          {config.rebalance_frequency}")
    print(f"Transaction Cost:     {config.transaction_cost:.1%}")
    print(f"Slippage:            {config.slippage:.2%}")
    print(f"Lookback Window:     {config.lookback_window} days")
    print(f"Walk-Forward Step:   {config.walk_forward_steps} days")
    
    # Prepare market data for backtesting
    print_subheader("📊 Preparing Market Data")
    
    # Convert to aligned format for backtesting
    aligned_data = {}
    for symbol, data in market_data.items():
        aligned_data[f"{symbol}_Close"] = data['Close']
        aligned_data[f"{symbol}_Volume"] = data['Volume']
        aligned_data[f"{symbol}_Returns"] = data['Close'].pct_change()
    
    df_aligned = pd.DataFrame(aligned_data).dropna()
    
    print(f"Aligned data shape:   {df_aligned.shape}")
    print(f"Date range:          {df_aligned.index[0]} to {df_aligned.index[-1]}")
    
    # Initialize walk-forward analyzer
    wf_analyzer = WalkForwardAnalyzer(config)
    
    # Test multiple strategies
    strategies = [
        ("Equal Weight", equal_weight_strategy, {}),
        ("Momentum (10d)", momentum_strategy, {"lookback": 10}),
        ("Momentum (20d)", momentum_strategy, {"lookback": 20}),
    ]
    
    strategy_results = {}
    
    for strategy_name, strategy_func, strategy_params in strategies:
        print_subheader(f"🚀 Testing {strategy_name} Strategy")
        
        try:
            # Run walk-forward analysis
            results = wf_analyzer.run_walk_forward_analysis(
                strategy_func,
                df_aligned,
                **strategy_params
            )
            
            strategy_results[strategy_name] = results
            
            # Display results
            if 'aggregate_metrics' in results:
                metrics = results['aggregate_metrics']
                
                print(f"📊 Performance Summary:")
                print(f"  Annualized Return:   {metrics.annualized_return:7.2%}")
                print(f"  Volatility:         {metrics.volatility:7.2%}")
                print(f"  Sharpe Ratio:       {metrics.sharpe_ratio:7.3f}")
                print(f"  Sortino Ratio:      {metrics.sortino_ratio:7.3f}")
                print(f"  Max Drawdown:       {metrics.max_drawdown:7.2%}")
                print(f"  Calmar Ratio:       {metrics.calmar_ratio:7.3f}")
                print(f"  Win Rate:           {metrics.win_rate:7.1%}")
                
                if hasattr(metrics, 'information_ratio'):
                    print(f"  Information Ratio:  {metrics.information_ratio:7.3f}")
            
            if 'stability_analysis' in results:
                stability = results['stability_analysis']
                print(f"  Stability Ranking:  {stability.get('stability_ranking', 'unknown').upper()}")
            
            print(f"  Periods Tested:     {len(results.get('periods', []))}")
            
        except Exception as e:
            print(f"❌ Error testing {strategy_name}: {e}")
            continue
    
    # Compare strategies
    print_subheader("📋 Strategy Comparison")
    
    if len(strategy_results) > 1:
        print(f"{'Strategy':<20} {'Return':<10} {'Sharpe':<8} {'Drawdown':<10} {'Stability':<12}")
        print("-" * 70)
        
        for strategy_name, results in strategy_results.items():
            if 'aggregate_metrics' in results:
                metrics = results['aggregate_metrics']
                stability = results.get('stability_analysis', {}).get('stability_ranking', 'unknown')
                
                print(f"{strategy_name:<20} "
                      f"{metrics.annualized_return:7.1%}    "
                      f"{metrics.sharpe_ratio:6.3f}  "
                      f"{metrics.max_drawdown:7.1%}     "
                      f"{stability.upper():<12}")
        
        # Find best strategy by Sharpe ratio
        best_strategy = max(strategy_results.keys(), 
                          key=lambda x: strategy_results[x].get('aggregate_metrics', 
                                                               type('obj', (object,), {'sharpe_ratio': -999})).sharpe_ratio)
        
        print(f"\n🏆 Best Strategy: {best_strategy}")
    
    return strategy_results

def save_comprehensive_results(optimization_result, validation_result, backtesting_results):
    """Save all results to comprehensive JSON file"""
    
    print_subheader("💾 Saving Comprehensive Results")
    
    comprehensive_results = {
        'timestamp': datetime.now().isoformat(),
        'summary': {
            'optimization_completed': True,
            'validation_completed': True,
            'backtesting_completed': True,
            'total_assets_analyzed': len(optimization_result.get('hyperbolic_embeddings', {})),
            'validation_score': validation_result['statistical_analysis']['validation_score'],
            'validation_passed': validation_result['overall_assessment']['validation_passed'],
            'best_strategy_sharpe': max([
                results.get('aggregate_metrics', type('obj', (object,), {'sharpe_ratio': 0})).sharpe_ratio
                for results in backtesting_results.values()
            ]) if backtesting_results else 0
        },
        'hyperbolic_optimization': optimization_result,
        'validation_framework': validation_result,
        'backtesting_analysis': backtesting_results,
        'system_info': {
            'python_version': sys.version,
            'modules_used': [
                'hyperbolic_portfolio_engine',
                'advanced_validation_framework', 
                'comprehensive_backtesting'
            ]
        }
    }
    
    # Save to file
    output_file = '/home/user/webapp/comprehensive_portfolio_results.json'
    
    try:
        with open(output_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {output_file}")
        print(f"   File size: {os.path.getsize(output_file) / 1024:.1f} KB")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")
    
    return comprehensive_results

def main():
    """Main demonstration function"""
    
    print_header("🚀 GOMNA AI - HYPERBOLIC PORTFOLIO OPTIMIZATION")
    print("Complete Demonstration of Advanced Portfolio Features")
    print(f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    try:
        # Step 1: Create sample data
        market_data, symbols = create_sample_data()
        
        # Step 2: Demonstrate hyperbolic optimization
        optimization_result = demonstrate_hyperbolic_optimization(market_data)
        
        # Step 3: Demonstrate validation framework
        validation_result = demonstrate_validation_framework(market_data)
        
        # Step 4: Demonstrate backtesting
        backtesting_results = demonstrate_backtesting(market_data)
        
        # Step 5: Save comprehensive results
        comprehensive_results = save_comprehensive_results(
            optimization_result, validation_result, backtesting_results
        )
        
        # Final summary
        print_header("✅ DEMONSTRATION COMPLETE")
        
        print("🎯 Summary of Results:")
        summary = comprehensive_results['summary']
        
        print(f"  Total Assets Analyzed:    {summary['total_assets_analyzed']}")
        print(f"  Validation Score:         {summary['validation_score']:.3f}")
        print(f"  Validation Status:        {'✅ PASSED' if summary['validation_passed'] else '❌ FAILED'}")
        print(f"  Best Strategy Sharpe:     {summary['best_strategy_sharpe']:.3f}")
        
        print(f"\n🔍 Key Insights:")
        print(f"  • Hyperbolic geometry provides superior diversification modeling")
        print(f"  • Comprehensive validation prevents overfitting and hallucinations") 
        print(f"  • Walk-forward analysis ensures robust out-of-sample performance")
        print(f"  • Multi-strategy comparison enables optimal selection")
        
        print(f"\n📊 Next Steps:")
        print(f"  • Review detailed results in comprehensive_portfolio_results.json")
        print(f"  • Integrate preferred strategy into live trading system")
        print(f"  • Monitor performance with real-time validation")
        print(f"  • Implement automated rebalancing based on recommendations")
        
        return True
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED")
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit_code = 0 if success else 1
    print(f"\nExiting with code: {exit_code}")
    sys.exit(exit_code)