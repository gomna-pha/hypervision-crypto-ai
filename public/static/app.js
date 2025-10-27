// Trading Intelligence Platform - Frontend Logic

const API_BASE = '';

// Initialize dashboard
document.addEventListener('DOMContentLoaded', async () => {
  console.log('Trading Intelligence Platform loaded');
  await loadDashboard();
  await loadStrategies();
  
  // Auto-refresh every 30 seconds
  setInterval(loadDashboard, 30000);
});

// Load dashboard summary
async function loadDashboard() {
  try {
    const response = await axios.get(`${API_BASE}/api/dashboard/summary`);
    if (response.data.success) {
      const { dashboard } = response.data;
      
      // Update market regime
      if (dashboard.market_regime) {
        const regime = dashboard.market_regime;
        document.getElementById('regime-type').textContent = 
          regime.regime_type.toUpperCase().replace('_', ' ');
      }
      
      // Update strategy count
      document.getElementById('strategy-count').textContent = 
        dashboard.active_strategies || 5;
      
      // Update signal count
      document.getElementById('signal-count').textContent = 
        dashboard.recent_signals?.length || 0;
      
      // Update backtest count
      document.getElementById('backtest-count').textContent = 
        dashboard.recent_backtests?.length || 0;
      
      // Display recent signals
      if (dashboard.recent_signals && dashboard.recent_signals.length > 0) {
        displaySignals(dashboard.recent_signals);
      }
      
      // Display backtest results
      if (dashboard.recent_backtests && dashboard.recent_backtests.length > 0) {
        displayBacktests(dashboard.recent_backtests);
      }
    }
  } catch (error) {
    console.error('Error loading dashboard:', error);
  }
}

// Load trading strategies
async function loadStrategies() {
  try {
    const response = await axios.get(`${API_BASE}/api/strategies`);
    if (response.data.success) {
      const { strategies } = response.data;
      displayStrategies(strategies);
    }
  } catch (error) {
    console.error('Error loading strategies:', error);
  }
}

// Display strategies
function displayStrategies(strategies) {
  const container = document.getElementById('strategies-list');
  if (!strategies || strategies.length === 0) {
    container.innerHTML = '<p class="text-gray-400">No strategies available</p>';
    return;
  }
  
  container.innerHTML = strategies.map(strategy => `
    <div class="bg-gray-900 p-4 rounded-lg border border-gray-600 hover:border-blue-500 transition">
      <div class="flex items-center justify-between mb-2">
        <h3 class="font-bold text-lg">${strategy.strategy_name}</h3>
        <span class="px-3 py-1 rounded-full text-xs ${getStrategyTypeColor(strategy.strategy_type)}">
          ${strategy.strategy_type.toUpperCase()}
        </span>
      </div>
      <p class="text-gray-400 text-sm mb-3">${strategy.description}</p>
      <button onclick="generateSignal(${strategy.id}, '${strategy.strategy_name}')" 
              class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded text-sm w-full">
        <i class="fas fa-bolt mr-2"></i>Generate Signal
      </button>
    </div>
  `).join('');
}

// Get color for strategy type
function getStrategyTypeColor(type) {
  const colors = {
    'momentum': 'bg-blue-600',
    'mean_reversion': 'bg-green-600',
    'arbitrage': 'bg-purple-600',
    'sentiment': 'bg-pink-600',
    'factor': 'bg-yellow-600'
  };
  return colors[type] || 'bg-gray-600';
}

// Display signals
function displaySignals(signals) {
  const container = document.getElementById('signals-list');
  if (!signals || signals.length === 0) {
    container.innerHTML = '<p class="text-gray-400">No signals yet...</p>';
    return;
  }
  
  container.innerHTML = signals.map(signal => {
    const signalColor = signal.signal_type === 'buy' ? 'text-green-400' : 
                       signal.signal_type === 'sell' ? 'text-red-400' : 'text-gray-400';
    const icon = signal.signal_type === 'buy' ? 'fa-arrow-up' : 
                 signal.signal_type === 'sell' ? 'fa-arrow-down' : 'fa-minus';
    
    return `
      <div class="bg-gray-900 p-3 rounded-lg border border-gray-600">
        <div class="flex items-center justify-between">
          <div>
            <p class="font-bold ${signalColor}">
              <i class="fas ${icon} mr-2"></i>${signal.signal_type.toUpperCase()}
            </p>
            <p class="text-sm text-gray-400">${signal.symbol || 'BTC-USD'}</p>
          </div>
          <div class="text-right">
            <p class="text-sm">Strength: ${(signal.signal_strength * 100).toFixed(0)}%</p>
            <p class="text-xs text-gray-500">Conf: ${(signal.confidence * 100).toFixed(0)}%</p>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

// Display backtest results
function displayBacktests(backtests) {
  const container = document.getElementById('backtest-results');
  if (!backtests || backtests.length === 0) {
    container.innerHTML = '<p class="text-gray-400">No backtests run yet...</p>';
    return;
  }
  
  container.innerHTML = backtests.map(backtest => {
    const returnColor = backtest.total_return > 0 ? 'text-green-400' : 'text-red-400';
    
    return `
      <div class="bg-gray-900 p-4 rounded-lg border border-gray-600">
        <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div>
            <p class="text-gray-400 text-sm">Symbol</p>
            <p class="font-bold">${backtest.symbol}</p>
          </div>
          <div>
            <p class="text-gray-400 text-sm">Total Return</p>
            <p class="font-bold ${returnColor}">${backtest.total_return.toFixed(2)}%</p>
          </div>
          <div>
            <p class="text-gray-400 text-sm">Sharpe Ratio</p>
            <p class="font-bold">${backtest.sharpe_ratio?.toFixed(2) || 'N/A'}</p>
          </div>
          <div>
            <p class="text-gray-400 text-sm">Max Drawdown</p>
            <p class="font-bold text-red-400">${backtest.max_drawdown?.toFixed(2) || 'N/A'}%</p>
          </div>
        </div>
        <div class="mt-3 grid grid-cols-2 gap-4">
          <div>
            <p class="text-gray-400 text-sm">Win Rate</p>
            <p class="font-bold">${backtest.win_rate?.toFixed(1) || 'N/A'}%</p>
          </div>
          <div>
            <p class="text-gray-400 text-sm">Total Trades</p>
            <p class="font-bold">${backtest.total_trades || 0}</p>
          </div>
        </div>
      </div>
    `;
  }).join('');
}

// Generate signal for a strategy
async function generateSignal(strategyId, strategyName) {
  try {
    // Mock market data
    const mockData = {
      symbol: 'BTC-USD',
      market_data: {
        rsi: Math.random() * 100,
        momentum: (Math.random() - 0.5) * 10,
        sentiment: (Math.random() - 0.5) * 2,
        volatility: Math.random() * 0.8
      }
    };
    
    const response = await axios.post(`${API_BASE}/api/strategies/${strategyId}/signal`, mockData);
    
    if (response.data.success) {
      alert(`✅ Signal Generated!\n\nStrategy: ${strategyName}\nSignal: ${response.data.signal.signal_type.toUpperCase()}\nStrength: ${(response.data.signal.signal_strength * 100).toFixed(0)}%`);
      await loadDashboard();
    }
  } catch (error) {
    console.error('Error generating signal:', error);
    alert('❌ Error generating signal');
  }
}

// Request LLM analysis
async function requestAnalysis(analysisType) {
  const responseDiv = document.getElementById('llm-response');
  responseDiv.innerHTML = '<p class="text-blue-400"><i class="fas fa-spinner fa-spin mr-2"></i>Analyzing with LLM...</p>';
  
  try {
    const mockContext = {
      symbol: 'BTC-USD',
      rsi: Math.random() * 100,
      trend: Math.random() > 0.5 ? 'bullish' : 'bearish',
      volatility: Math.random() * 0.8,
      regime: ['bull', 'bear', 'sideways'][Math.floor(Math.random() * 3)],
      risk_level: Math.random() * 10,
      price: 45000 + Math.random() * 5000
    };
    
    const response = await axios.post(`${API_BASE}/api/llm/analyze`, {
      analysis_type: analysisType,
      symbol: 'BTC-USD',
      context: mockContext
    });
    
    if (response.data.success) {
      const analysis = response.data.analysis;
      responseDiv.innerHTML = `
        <div class="mb-3">
          <div class="flex items-center justify-between mb-2">
            <span class="px-3 py-1 rounded-full text-xs bg-blue-600">
              ${analysis.type.replace('_', ' ').toUpperCase()}
            </span>
            <span class="text-sm text-gray-400">
              Confidence: ${(analysis.confidence * 100).toFixed(0)}%
            </span>
          </div>
          <div class="h-2 bg-gray-700 rounded-full mb-3">
            <div class="h-2 bg-green-500 rounded-full" style="width: ${analysis.confidence * 100}%"></div>
          </div>
        </div>
        <p class="text-gray-200 leading-relaxed whitespace-pre-line">${analysis.response}</p>
        <p class="text-xs text-gray-500 mt-3">
          <i class="far fa-clock mr-1"></i>
          ${new Date(analysis.timestamp).toLocaleString()}
        </p>
      `;
    }
  } catch (error) {
    console.error('Error requesting analysis:', error);
    responseDiv.innerHTML = '<p class="text-red-400"><i class="fas fa-exclamation-triangle mr-2"></i>Error getting analysis</p>';
  }
}

// Run backtest
async function runBacktest() {
  if (!confirm('Run backtest for BTC-USD with Momentum strategy?')) return;
  
  try {
    const endDate = Date.now();
    const startDate = endDate - (30 * 24 * 60 * 60 * 1000); // 30 days ago
    
    // First, add some mock market data
    await addMockMarketData('BTC-USD', startDate, endDate);
    
    const response = await axios.post(`${API_BASE}/api/backtest/run`, {
      strategy_id: 1,
      symbol: 'BTC-USD',
      start_date: startDate,
      end_date: endDate,
      initial_capital: 10000
    });
    
    if (response.data.success) {
      const result = response.data.backtest;
      alert(`✅ Backtest Complete!\n\nInitial: $${result.initial_capital.toFixed(2)}\nFinal: $${result.final_capital.toFixed(2)}\nReturn: ${result.total_return.toFixed(2)}%\nSharpe: ${result.sharpe_ratio.toFixed(2)}\nDrawdown: ${result.max_drawdown.toFixed(2)}%\nTrades: ${result.total_trades}`);
      await loadDashboard();
    }
  } catch (error) {
    console.error('Error running backtest:', error);
    alert('❌ Error running backtest');
  }
}

// Add mock market data for backtesting
async function addMockMarketData(symbol, startDate, endDate) {
  try {
    const days = Math.floor((endDate - startDate) / (24 * 60 * 60 * 1000));
    let price = 45000;
    
    for (let i = 0; i < days; i++) {
      price += (Math.random() - 0.5) * 1000;
      const timestamp = startDate + (i * 24 * 60 * 60 * 1000);
      
      await axios.get(`${API_BASE}/api/market/data/${symbol}`);
    }
  } catch (error) {
    console.error('Error adding mock data:', error);
  }
}

// Initialize sample data on first load
async function initializeSampleData() {
  try {
    // Add sample economic indicator
    await axios.post(`${API_BASE}/api/economic/indicators`, {
      indicator_name: 'Federal Funds Rate',
      indicator_code: 'FEDFUNDS',
      value: 5.33,
      period: '2024-Q4',
      source: 'FRED'
    });
    
    // Detect market regime
    await axios.post(`${API_BASE}/api/market/regime`, {
      indicators: {
        volatility: 0.35,
        trend: 0.08,
        volume: 1.2
      }
    });
    
    console.log('Sample data initialized');
  } catch (error) {
    console.error('Error initializing sample data:', error);
  }
}

// Initialize on first load
setTimeout(initializeSampleData, 1000);
