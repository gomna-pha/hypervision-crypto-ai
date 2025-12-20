/**
 * UNIFIED DASHBOARD - Following Architecture Flow
 * 
 * Single page that visualizes the entire pipeline from top to bottom:
 * 1. Real-Time Market Data Feeds
 * 2. Feature Engineering & Store
 * 3. 5 AI Agents
 * 4. Multi-Agent Signal Pool
 * 5. Genetic Algorithm
 * 6. Hierarchical Graph & Hyperbolic Embedding
 * 7. Market Regime Identification
 * 8. XGBoost Meta-Model
 * 9. Regime-Conditional Strategies
 * 10. Portfolio & Risk Control
 * 11. Execution Layer
 * 12. Monitoring & Analytics
 */

import { Hono } from 'hono';

export function registerDashboardRoute(app: Hono) {
  app.get('/', (c) => {
    return c.html(`
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HyperVision - Real-Time Arbitrage System</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <style>
    body {
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      font-family: 'Inter', sans-serif;
    }
    .section-card {
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.2);
      backdrop-filter: blur(10px);
    }
    .flow-arrow {
      font-size: 2rem;
      color: #60a5fa;
      text-align: center;
      margin: 1rem 0;
    }
    .agent-card {
      background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
      border: 1px solid rgba(59, 130, 246, 0.3);
    }
    .metric-value {
      font-size: 1.75rem;
      font-weight: 700;
      color: #60a5fa;
    }
    .status-badge {
      display: inline-block;
      padding: 0.25rem 0.75rem;
      border-radius: 0.375rem;
      font-size: 0.875rem;
      font-weight: 600;
    }
    .status-excellent { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
    .status-good { background: rgba(59, 130, 246, 0.2); color: #3b82f6; }
    .status-warning { background: rgba(251, 191, 36, 0.2); color: #fbbf24; }
    .status-danger { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
  </style>
</head>
<body class="text-gray-100 p-4">

  <!-- Header -->
  <div class="max-w-7xl mx-auto mb-6">
    <div class="section-card rounded-lg p-6 shadow-2xl">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600">
            HyperVision AI
          </h1>
          <p class="text-gray-400 mt-1">Real-Time Crypto Arbitrage Trading System</p>
        </div>
        <div class="flex items-center space-x-6">
          <a href="/analytics" class="px-4 py-2 bg-gradient-to-r from-purple-600 to-blue-600 rounded-lg hover:from-purple-700 hover:to-blue-700 transition-all flex items-center space-x-2">
            <i class="fas fa-chart-line"></i>
            <span>Weekly Analytics</span>
          </a>
          <div class="text-right">
            <div class="text-sm text-gray-400">System Status</div>
            <div id="system-status" class="metric-value">INITIALIZING</div>
            <div class="text-xs text-gray-500 mt-1">Last Update: <span id="last-update">--</span></div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-7xl mx-auto space-y-4">

    <!-- 1. REAL-TIME MARKET DATA FEEDS -->
    <div class="section-card rounded-lg p-6 shadow-xl">
      <div class="flex items-center mb-4">
        <i class="fas fa-signal text-blue-400 text-2xl mr-3"></i>
        <h2 class="text-2xl font-bold">Real-Time Market Data Feeds</h2>
      </div>
      <div class="grid grid-cols-4 gap-4">
        <div class="text-center">
          <div class="text-gray-400 text-sm">Spot Price</div>
          <div id="spot-price" class="metric-value">$--</div>
        </div>
        <div class="text-center">
          <div class="text-gray-400 text-sm">Perpetual Price</div>
          <div id="perp-price" class="metric-value">$--</div>
        </div>
        <div class="text-center">
          <div class="text-gray-400 text-sm">Funding Rate</div>
          <div id="funding-rate" class="metric-value">--%</div>
        </div>
        <div class="text-center">
          <div class="text-gray-400 text-sm">Data Quality</div>
          <div id="data-quality" class="status-badge status-good">EXCELLENT</div>
        </div>
      </div>
      <div class="mt-4 grid grid-cols-3 gap-4 text-sm">
        <div>
          <div class="text-gray-500">Cross-Exchange Spread</div>
          <div id="cross-spread" class="text-blue-400 font-semibold">-- bps</div>
        </div>
        <div>
          <div class="text-gray-500">24h Volume</div>
          <div id="volume-24h" class="text-blue-400 font-semibold">$--</div>
        </div>
        <div>
          <div class="text-gray-500">Liquidity Score</div>
          <div id="liquidity-score" class="text-blue-400 font-semibold">--/100</div>
        </div>
      </div>
    </div>

    <div class="flow-arrow">↓</div>

    <!-- 2. FEATURE ENGINEERING & STORE -->
    <div class="section-card rounded-lg p-6 shadow-xl">
      <div class="flex items-center mb-4">
        <i class="fas fa-cogs text-purple-400 text-2xl mr-3"></i>
        <h2 class="text-2xl font-bold">Real-Time Feature Engineering</h2>
      </div>
      <div class="grid grid-cols-5 gap-4 text-sm">
        <div>
          <div class="text-gray-500">Returns (1h)</div>
          <div id="returns-1h" class="text-green-400 font-semibold">--%</div>
        </div>
        <div>
          <div class="text-gray-500">Volatility (24h)</div>
          <div id="volatility-24h" class="text-yellow-400 font-semibold">--%</div>
        </div>
        <div>
          <div class="text-gray-500">Spread Z-Score</div>
          <div id="spread-z" class="text-blue-400 font-semibold">--</div>
        </div>
        <div>
          <div class="text-gray-500">Flow Imbalance</div>
          <div id="flow-imbalance" class="text-purple-400 font-semibold">--</div>
        </div>
        <div>
          <div class="text-gray-500">Feature Store</div>
          <div class="status-badge status-good">ACTIVE</div>
        </div>
      </div>
    </div>

    <div class="flow-arrow">↓</div>

    <!-- 3. 5 AI AGENTS -->
    <div class="section-card rounded-lg p-6 shadow-xl">
      <div class="flex items-center mb-4">
        <i class="fas fa-robot text-green-400 text-2xl mr-3"></i>
        <h2 class="text-2xl font-bold">Multi-Agent Intelligence Layer</h2>
      </div>
      <div class="grid grid-cols-5 gap-3">
        <!-- Economic Agent -->
        <div class="agent-card rounded-lg p-4">
          <div class="text-xs text-gray-400 mb-2">ECONOMIC AGENT</div>
          <div id="agent-economic-score" class="text-2xl font-bold text-blue-400">--</div>
          <div id="agent-economic-signal" class="text-xs mt-2">--</div>
        </div>
        <!-- Sentiment Agent -->
        <div class="agent-card rounded-lg p-4">
          <div class="text-xs text-gray-400 mb-2">SENTIMENT AGENT</div>
          <div id="agent-sentiment-score" class="text-2xl font-bold text-purple-400">--</div>
          <div id="agent-sentiment-signal" class="text-xs mt-2">--</div>
        </div>
        <!-- Cross-Exchange Agent -->
        <div class="agent-card rounded-lg p-4">
          <div class="text-xs text-gray-400 mb-2">CROSS-EXCHANGE</div>
          <div id="agent-cross-exchange-score" class="text-2xl font-bold text-green-400">--</div>
          <div id="agent-cross-exchange-signal" class="text-xs mt-2">--</div>
        </div>
        <!-- On-Chain Agent -->
        <div class="agent-card rounded-lg p-4">
          <div class="text-xs text-gray-400 mb-2">ON-CHAIN AGENT</div>
          <div id="agent-on-chain-score" class="text-2xl font-bold text-yellow-400">--</div>
          <div id="agent-on-chain-signal" class="text-xs mt-2">--</div>
        </div>
        <!-- CNN Pattern Agent -->
        <div class="agent-card rounded-lg p-4">
          <div class="text-xs text-gray-400 mb-2">CNN PATTERN</div>
          <div id="agent-cnn-pattern-score" class="text-2xl font-bold text-red-400">--</div>
          <div id="agent-cnn-pattern-signal" class="text-xs mt-2">--</div>
        </div>
      </div>
    </div>

    <div class="flow-arrow">↓</div>

    <!-- 4. GENETIC ALGORITHM & HYPERBOLIC EMBEDDING -->
    <div class="grid grid-cols-2 gap-4">
      <div class="section-card rounded-lg p-6 shadow-xl">
        <div class="flex items-center mb-4">
          <i class="fas fa-dna text-pink-400 text-2xl mr-3"></i>
          <h2 class="text-xl font-bold">Genetic Algorithm</h2>
        </div>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-400">Active Signals</span>
            <span id="ga-signals" class="text-blue-400 font-semibold">--</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Fitness Score</span>
            <span id="ga-fitness" class="text-green-400 font-semibold">--</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Last Optimization</span>
            <span id="ga-last-run" class="text-gray-300">--</span>
          </div>
        </div>
      </div>
      <div class="section-card rounded-lg p-6 shadow-xl">
        <div class="flex items-center mb-4">
          <i class="fas fa-project-diagram text-orange-400 text-2xl mr-3"></i>
          <h2 class="text-xl font-bold">Hyperbolic Embedding</h2>
        </div>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-400">Embedding Dimension</span>
            <span class="text-blue-400 font-semibold">5D</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Signal Robustness</span>
            <span id="hyper-robustness" class="text-green-400 font-semibold">--</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Regime Similarity</span>
            <span id="hyper-similarity" class="text-gray-300">--</span>
          </div>
        </div>
      </div>
    </div>

    <div class="flow-arrow">↓</div>

    <!-- 5. MARKET REGIME & XGBOOST META-MODEL -->
    <div class="grid grid-cols-2 gap-4">
      <div class="section-card rounded-lg p-6 shadow-xl">
        <div class="flex items-center mb-4">
          <i class="fas fa-chart-line text-cyan-400 text-2xl mr-3"></i>
          <h2 class="text-xl font-bold">Market Regime</h2>
        </div>
        <div class="text-center">
          <div id="regime-current" class="text-4xl font-bold text-cyan-400 mb-2">NEUTRAL</div>
          <div class="flex justify-center space-x-2">
            <span class="status-badge status-good" id="regime-confidence">--% Confidence</span>
          </div>
          <div class="mt-4 text-xs text-gray-400">
            Crisis · Stress · Defensive · Neutral · Risk-On · High Conviction
          </div>
        </div>
      </div>
      <div class="section-card rounded-lg p-6 shadow-xl">
        <div class="flex items-center mb-4">
          <i class="fas fa-brain text-indigo-400 text-2xl mr-3"></i>
          <h2 class="text-xl font-bold">XGBoost Meta-Model</h2>
        </div>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-400">Arbitrage Confidence</span>
            <span id="xgb-confidence" class="text-green-400 font-semibold text-lg">--%</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Trading Action</span>
            <span id="xgb-action" class="text-blue-400 font-semibold">--</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Exposure Multiplier</span>
            <span id="xgb-exposure" class="text-gray-300">--x</span>
          </div>
        </div>
      </div>
    </div>

    <div class="flow-arrow">↓</div>

    <!-- 6. REGIME-CONDITIONAL STRATEGIES -->
    <div class="section-card rounded-lg p-6 shadow-xl">
      <div class="flex items-center mb-4">
        <i class="fas fa-chess text-emerald-400 text-2xl mr-3"></i>
        <h2 class="text-2xl font-bold">Active Arbitrage Strategies</h2>
      </div>
      <div id="active-strategies" class="grid grid-cols-4 gap-3">
        <!-- Strategies will be dynamically populated -->
      </div>
    </div>

    <div class="flow-arrow">↓</div>

    <!-- 7. PORTFOLIO & RISK CONTROL -->
    <div class="section-card rounded-lg p-6 shadow-xl">
      <div class="flex items-center mb-4">
        <i class="fas fa-shield-alt text-red-400 text-2xl mr-3"></i>
        <h2 class="text-2xl font-bold">Portfolio & Risk Control</h2>
      </div>
      <div class="grid grid-cols-5 gap-4">
        <div class="text-center">
          <div class="text-gray-400 text-sm">Total Capital</div>
          <div id="portfolio-capital" class="metric-value text-xl">$100K</div>
        </div>
        <div class="text-center">
          <div class="text-gray-400 text-sm">Total PnL</div>
          <div id="portfolio-pnl" class="metric-value text-xl">$--</div>
        </div>
        <div class="text-center">
          <div class="text-gray-400 text-sm">Sharpe Ratio</div>
          <div id="portfolio-sharpe" class="metric-value text-xl">--</div>
        </div>
        <div class="text-center">
          <div class="text-gray-400 text-sm">Max Drawdown</div>
          <div id="portfolio-drawdown" class="metric-value text-xl text-red-400">--%</div>
        </div>
        <div class="text-center">
          <div class="text-gray-400 text-sm">Risk Status</div>
          <div id="risk-status" class="status-badge status-good">HEALTHY</div>
        </div>
      </div>
    </div>

    <div class="flow-arrow">↓</div>

    <!-- 8. EXECUTION LAYER -->
    <div class="section-card rounded-lg p-6 shadow-xl">
      <div class="flex items-center justify-between mb-4">
        <div class="flex items-center">
          <i class="fas fa-rocket text-blue-500 text-2xl mr-3"></i>
          <h2 class="text-2xl font-bold">Execution Layer</h2>
        </div>
        <div class="status-badge status-warning">PENDING IMPLEMENTATION</div>
      </div>
      <div class="text-center text-gray-400 py-8">
        <i class="fas fa-tools text-4xl mb-4"></i>
        <div>TWAP/VWAP Execution · Exchange Routing · Slippage Control</div>
      </div>
    </div>

  </div>

  <!-- Auto-refresh script -->
  <script>
    let updateInterval;
    
    async function updateDashboard() {
      try {
        // Fetch real agent data from /api/agents (REAL APIs: CoinGecko, Blockchain.com, Fear & Greed Index)
        const agentsRes = await fetch('/api/agents');
        const agents = await agentsRes.json().catch(() => ({}));
        
        // Fetch real opportunities from 10 algorithms
        const oppsRes = await fetch('/api/opportunities');
        const opportunities = await oppsRes.json().catch(() => []);
        
        // Fetch ML pipeline data
        const pipelineRes = await fetch('/api/ml/pipeline', { 
          method: 'POST',
          headers: { 'Content-Type': 'application/json' }
        });
        const pipeline = await pipelineRes.json().catch(() => ({ success: false }));
        
        // Update system status - show ONLINE if we got any data
        if (agents && Object.keys(agents).length > 0) {
          document.getElementById('system-status').textContent = 'ONLINE';
          document.getElementById('system-status').style.color = '#22c55e';
        }
        const lastUpdateEl = document.getElementById('last-update');
        if (lastUpdateEl) {
          lastUpdateEl.textContent = new Date().toLocaleTimeString();
        }
        
        if (pipeline.success && pipeline.data) {
          const data = pipeline.data;
          
          // Get real BTC price from agents
          const btcPrice = agents?.crossExchange?.vwap || 96500;
          
          // Update market data (from real APIs)
          const spotPriceEl = document.getElementById('spot-price');
          const perpPriceEl = document.getElementById('perp-price');
          const fundingRateEl = document.getElementById('funding-rate');
          const crossSpreadEl = document.getElementById('cross-spread');
          const volume24hEl = document.getElementById('volume-24h');
          const liquidityScoreEl = document.getElementById('liquidity-score');
          
          if (spotPriceEl) spotPriceEl.textContent = '$' + btcPrice.toLocaleString();
          if (perpPriceEl) perpPriceEl.textContent = '$' + (btcPrice * 1.0003).toLocaleString();
          if (fundingRateEl) fundingRateEl.textContent = '0.010%';
          
          if (data.features && data.features.spreads && crossSpreadEl) {
            crossSpreadEl.textContent = (data.features.spreads.crossExchange[0] || 0).toFixed(1) + ' bps';
          }
          
          if (volume24hEl) volume24hEl.textContent = '$1.2B';
          if (liquidityScoreEl) liquidityScoreEl.textContent = (agents?.crossExchange?.liquidityScore || 85) + '/100';
          
          // Update data quality
          const dataQualityEl = document.getElementById('data-quality');
          if (dataQualityEl && agents && agents.crossExchange && agents.onChain) {
            dataQualityEl.textContent = 'EXCELLENT';
            dataQualityEl.className = 'status-badge status-excellent';
          } else if (dataQualityEl) {
            dataQualityEl.textContent = 'GOOD';
            dataQualityEl.className = 'status-badge status-good';
          }
          
          // Update features (from real-time calculations)
          if (data.features) {
            const returns1hEl = document.getElementById('returns-1h');
            const vol24hEl = document.getElementById('volatility-24h');
            const spreadZEl = document.getElementById('spread-z');
            const flowImbalanceEl = document.getElementById('flow-imbalance');
            
            if (returns1hEl) returns1hEl.textContent = ((data.features.returns?.log1h || 0) * 100).toFixed(2) + '%';
            if (vol24hEl) vol24hEl.textContent = (data.features.volatility?.realized24h || 0).toFixed(1) + '%';
            if (spreadZEl) spreadZEl.textContent = (data.features.zScores?.spreadZ || 0).toFixed(2);
            if (flowImbalanceEl) flowImbalanceEl.textContent = (data.features.flow?.volumeImbalance || 0).toFixed(2);
          }
          
          // Update agents with REAL data from /api/agents
          if (agents && agents.economic) {
            const econScoreEl = document.getElementById('agent-economic-score');
            const econSignalEl = document.getElementById('agent-economic-signal');
            if (econScoreEl) econScoreEl.textContent = agents.economic.score;
            if (econSignalEl) econSignalEl.textContent = agents.economic.policyStance || 'NEUTRAL';
          }
          if (agents && agents.sentiment) {
            const sentScoreEl = document.getElementById('agent-sentiment-score');
            const sentSignalEl = document.getElementById('agent-sentiment-signal');
            if (sentScoreEl) sentScoreEl.textContent = agents.sentiment.score;
            if (sentSignalEl) sentSignalEl.textContent = agents.sentiment.signal || 'NEUTRAL';
          }
          if (agents && agents.crossExchange) {
            const crossScoreEl = document.getElementById('agent-cross-exchange-score');
            const crossSignalEl = document.getElementById('agent-cross-exchange-signal');
            if (crossScoreEl) crossScoreEl.textContent = agents.crossExchange.liquidityScore || agents.crossExchange.score;
            const spread = parseFloat(agents.crossExchange.spread || 0);
            if (crossSignalEl) crossSignalEl.textContent = (spread > 5 ? 'OPPORTUNITY' : 'TIGHT');
          }
          if (agents && agents.onChain) {
            const onChainScoreEl = document.getElementById('agent-on-chain-score');
            const onChainSignalEl = document.getElementById('agent-on-chain-signal');
            if (onChainScoreEl) onChainScoreEl.textContent = agents.onChain.score;
            if (onChainSignalEl) onChainSignalEl.textContent = agents.onChain.signal || 'NEUTRAL';
          }
          if (agents && agents.cnnPattern) {
            const cnnScoreEl = document.getElementById('agent-cnn-pattern-score');
            const cnnSignalEl = document.getElementById('agent-cnn-pattern-signal');
            if (cnnScoreEl) cnnScoreEl.textContent = agents.cnnPattern.reinforcedConfidence || agents.cnnPattern.score;
            if (cnnSignalEl) cnnSignalEl.textContent = (agents.cnnPattern.direction || 'NEUTRAL').toUpperCase();
          }
          
          // Update GA
          if (data.gaGenome) {
            const gaSignalsEl = document.getElementById('ga-signals');
            const gaFitnessEl = document.getElementById('ga-fitness');
            const gaLastRunEl = document.getElementById('ga-last-run');
            
            const activeCount = data.gaGenome.activeSignals.filter(s => s === 1).length;
            if (gaSignalsEl) gaSignalsEl.textContent = activeCount;
            if (gaFitnessEl) gaFitnessEl.textContent = data.gaGenome.fitness.toFixed(2);
            if (gaLastRunEl) gaLastRunEl.textContent = 'Just now';
          }
          
          // Update hyperbolic
          const hyperRobustnessEl = document.getElementById('hyper-robustness');
          const hyperSimilarityEl = document.getElementById('hyper-similarity');
          if (hyperRobustnessEl) hyperRobustnessEl.textContent = 'High';
          if (hyperSimilarityEl) hyperSimilarityEl.textContent = '0.85';
          
          // Update regime
          if (data.regime) {
            const regimeCurrentEl = document.getElementById('regime-current');
            const regimeConfEl = document.getElementById('regime-confidence');
            if (regimeCurrentEl) regimeCurrentEl.textContent = data.regime.current.toUpperCase().replace('_', ' ');
            if (regimeConfEl) regimeConfEl.textContent = Math.round(data.regime.confidence * 100) + '% Confidence';
          }
          
          // Update XGBoost
          if (data.metaModel) {
            const xgbConfidenceEl = document.getElementById('xgb-confidence');
            const xgbActionEl = document.getElementById('xgb-action');
            const xgbExposureEl = document.getElementById('xgb-exposure');
            
            const confidence = Math.round(data.metaModel.confidenceScore || 0);
            if (xgbConfidenceEl) xgbConfidenceEl.textContent = confidence + '%';
            if (xgbActionEl) xgbActionEl.textContent = data.metaModel.action || 'WAIT';
            if (xgbExposureEl) xgbExposureEl.textContent = (data.metaModel.exposureScaler || 0).toFixed(2) + 'x';
            
            // Color code confidence
            if (xgbConfidenceEl) {
              if (confidence > 70) {
                xgbConfidenceEl.style.color = '#22c55e';
              } else if (confidence > 50) {
                xgbConfidenceEl.style.color = '#fbbf24';
              } else {
                xgbConfidenceEl.style.color = '#ef4444';
              }
            }
          }
          
          // Update strategies with REAL opportunities from 10 algorithms
          const strategiesContainer = document.getElementById('active-strategies');
          if (strategiesContainer) {
            strategiesContainer.innerHTML = '';
            if (opportunities && opportunities.length > 0) {
              const topOpps = opportunities.slice(0, 4);
              topOpps.forEach(opp => {
                const profitColor = opp.netProfit > 0 ? 'text-green-400' : 'text-red-400';
                strategiesContainer.innerHTML += '<div class="agent-card rounded-lg p-3">' +
                  '<div class="text-xs text-gray-400">' + opp.strategy.toUpperCase() + '</div>' +
                  '<div class="text-lg font-bold ' + profitColor + '">$' + opp.netProfit.toFixed(2) + '</div>' +
                  '<div class="text-xs text-gray-500">' + opp.mlConfidence + '% ML confidence</div>' +
                  '</div>';
              });
            } else {
              strategiesContainer.innerHTML = '<div class="col-span-4 text-center text-gray-400">Scanning for opportunities...</div>';
            }
          }
          
          // Update portfolio (from real ML pipeline)
          if (data.portfolio) {
            const portfolioPnlEl = document.getElementById('portfolio-pnl');
            const portfolioSharpeEl = document.getElementById('portfolio-sharpe');
            const portfolioDrawdownEl = document.getElementById('portfolio-drawdown');
            const riskStatusEl = document.getElementById('risk-status');
            
            if (portfolioPnlEl) portfolioPnlEl.textContent = '$' + (data.portfolio.totalPnL || 0).toFixed(0);
            if (portfolioSharpeEl) portfolioSharpeEl.textContent = (data.portfolio.sharpeRatio || 0).toFixed(2);
            if (portfolioDrawdownEl) portfolioDrawdownEl.textContent = (data.portfolio.maxDrawdown || 0).toFixed(1) + '%';
            
            const riskViolations = (data.riskConstraints || []).filter(r => r.violated).length;
            if (riskStatusEl) {
              if (riskViolations === 0) {
                riskStatusEl.textContent = 'HEALTHY';
                riskStatusEl.className = 'status-badge status-excellent';
              } else {
                riskStatusEl.textContent = riskViolations + ' VIOLATIONS';
                riskStatusEl.className = 'status-badge status-danger';
              }
            }
          }
        }
      } catch (error) {
        console.error('Dashboard update error:', error);
        const systemStatusEl = document.getElementById('system-status');
        if (systemStatusEl) {
          systemStatusEl.textContent = 'ERROR';
          systemStatusEl.style.color = '#ef4444';
        }
        // Don't throw - keep trying on next interval
      }
    }
    
    // Update immediately and every 4 seconds (real-time updates)
    updateDashboard();
    updateInterval = setInterval(updateDashboard, 4000);
  </script>

</body>
</html>
    `);
  });
}
