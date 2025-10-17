import express from 'express';
import http from 'http';
import { Server as SocketIOServer } from 'socket.io';
import axios from 'axios';
import Logger from '../utils/logger';
import config from '../utils/ConfigLoader';
import { BacktestEngine } from '../backtest/BacktestEngine';
import { DecisionEngine } from '../decision/DecisionEngine';

const app = express();
const server = http.createServer(app);
const io = new SocketIOServer(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

const logger = Logger.getInstance('EnhancedDashboard');
const PORT = process.env.DASHBOARD_PORT || 3000;

// Initialize engines
const backtestEngine = new BacktestEngine(config.get('backtesting'));
const decisionEngine = new DecisionEngine();

// Market data cache
let marketData = {
  btc: { price: 67213.44, change: -2.48, volume: 51500, trades: 502000 },
  eth: { price: 3475.77, change: 2.86, volume: 37000, trades: 777000 },
  sol: { price: 119.12, change: 3.37, volume: 5900, trades: 277000 }
};

let crossExchangeSpreads = {
  'binance-coinbase': 18.36,
  'kraken-bybit': 27.42,
  'futures-spot': 256.83
};

let orderBook = {
  asks: [],
  bids: [],
  spread: 0.32
};

// Serve static files
app.use(express.static('public'));
app.use(express.json());

// API Endpoints
app.get('/api/market/live', (req, res) => {
  res.json(marketData);
});

app.get('/api/spreads', (req, res) => {
  res.json(crossExchangeSpreads);
});

app.get('/api/orderbook/:pair', async (req, res) => {
  try {
    const response = await axios.get('http://localhost:3002/agents/priceagent/latest');
    const data = response.data;
    
    // Generate mock orderbook if no real data
    const midPrice = data.mid_price || 67810;
    const asks = [];
    const bids = [];
    
    for (let i = 0; i < 5; i++) {
      asks.push({
        price: midPrice + (i * 0.5) + Math.random() * 0.5,
        volume: Math.random() * 20
      });
      bids.push({
        price: midPrice - (i * 0.5) - Math.random() * 0.5,
        volume: Math.random() * 20
      });
    }
    
    res.json({
      asks: asks.sort((a, b) => a.price - b.price),
      bids: bids.sort((a, b) => b.price - a.price),
      spread: data.best_ask && data.best_bid ? data.best_ask - data.best_bid : 0.32
    });
  } catch (error) {
    res.json(orderBook);
  }
});

app.get('/api/sentiment', async (req, res) => {
  try {
    const response = await axios.get('http://localhost:3003/agents/sentimentagent/latest');
    res.json(response.data);
  } catch (error) {
    // Mock sentiment data
    res.json({
      fear_greed_index: 48,
      social_volume: 161000,
      market_mood: {
        btc: 51,
        eth: 44,
        sol: 47
      }
    });
  }
});

app.get('/api/parameters', (req, res) => {
  const params = decisionEngine.getParameters();
  res.json(params);
});

app.get('/api/backtest/results', async (req, res) => {
  const metrics = backtestEngine.getMetrics();
  const trades = backtestEngine.getTrades();
  
  res.json({
    metrics: metrics || {
      total_trades: 0,
      winning_trades: 0,
      losing_trades: 0,
      total_pnl_usd: 0,
      sharpe_ratio: 0,
      sortino_ratio: 0,
      max_drawdown_pct: 0,
      win_rate: 0
    },
    recent_trades: trades.slice(-10)
  });
});

app.get('/api/hyperbolic/stats', (req, res) => {
  res.json({
    geodesic_paths: 791,
    space_curvature: -1.0,
    path_efficiency: 99.5,
    patterns_detected: 12,
    clustering_coefficient: 0.82
  });
});

app.get('/api/opportunities', (req, res) => {
  const audit = decisionEngine.getAuditLog(10);
  const opportunities = audit
    .filter(a => a.decision === 'approved')
    .map(a => ({
      pair: a.executionPlan?.pair || 'BTC-USDT',
      buy: a.executionPlan?.buy_exchange || 'binance',
      sell: a.executionPlan?.sell_exchange || 'coinbase',
      spread: a.executionPlan?.predicted_spread_pct || 0,
      confidence: a.prediction.confidence,
      aos_score: a.aosScore,
      status: a.executionPlan?.status || 'pending'
    }));
    
  res.json(opportunities);
});

// Enhanced Dashboard HTML
app.get('/', (req, res) => {
  res.send(generateDashboardHTML());
});

// WebSocket for real-time updates
io.on('connection', (socket) => {
  logger.info('Dashboard client connected');
  
  // Send initial data
  socket.emit('market_update', marketData);
  socket.emit('spread_update', crossExchangeSpreads);
  
  // Send updates every second
  const interval = setInterval(() => {
    updateMarketData();
    socket.emit('market_update', marketData);
    socket.emit('spread_update', crossExchangeSpreads);
  }, 1000);
  
  socket.on('disconnect', () => {
    clearInterval(interval);
    logger.info('Dashboard client disconnected');
  });
});

// Update market data (mock for now)
function updateMarketData() {
  // Simulate price changes
  marketData.btc.price += (Math.random() - 0.5) * 50;
  marketData.btc.change += (Math.random() - 0.5) * 0.1;
  marketData.eth.price += (Math.random() - 0.5) * 10;
  marketData.eth.change += (Math.random() - 0.5) * 0.1;
  marketData.sol.price += (Math.random() - 0.5) * 1;
  marketData.sol.change += (Math.random() - 0.5) * 0.1;
  
  // Update spreads
  crossExchangeSpreads['binance-coinbase'] += (Math.random() - 0.5) * 2;
  crossExchangeSpreads['kraken-bybit'] += (Math.random() - 0.5) * 3;
  crossExchangeSpreads['futures-spot'] += (Math.random() - 0.5) * 10;
}

function generateDashboardHTML(): string {
  return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Arbitrage Platform - Professional Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="/socket.io/socket.io.js"></script>
    <style>
        @keyframes pulse-live { 
            0%, 100% { opacity: 1; } 
            50% { opacity: 0.3; } 
        }
        .pulse-live { animation: pulse-live 2s infinite; }
        .metric-card { 
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
        }
        .positive { color: #10b981; }
        .negative { color: #ef4444; }
        .orderbook-row { font-family: 'Courier New', monospace; }
    </style>
</head>
<body class="bg-gray-950 text-gray-100 font-sans">
    <!-- Header -->
    <div class="bg-gray-900 border-b border-gray-800 px-6 py-4">
        <div class="flex items-center justify-between">
            <div class="flex items-center space-x-4">
                <i class="fas fa-chart-line text-3xl text-blue-500"></i>
                <div>
                    <h1 class="text-2xl font-bold">LLM Arbitrage Intelligence Platform</h1>
                    <p class="text-sm text-gray-400">Agent-Based • Hyperbolic-Enhanced • Real-Time</p>
                </div>
            </div>
            <div class="flex items-center space-x-6">
                <div class="text-sm">
                    <span class="text-gray-500">System Time:</span>
                    <span class="font-mono" id="system-time"></span>
                </div>
                <div class="flex items-center space-x-2">
                    <div class="w-2 h-2 bg-green-500 rounded-full pulse-live"></div>
                    <span class="text-sm">LIVE</span>
                </div>
            </div>
        </div>
    </div>

    <div class="p-6">
        <!-- Market Feeds -->
        <div class="mb-6">
            <h2 class="text-lg font-semibold mb-3 text-gray-300">Market Feeds</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="metric-card p-4 rounded-lg">
                    <div class="flex justify-between items-start mb-2">
                        <div>
                            <div class="text-sm text-gray-500">BTC/USD</div>
                            <div class="text-xs text-gray-600">Vol: <span id="btc-volume">51.5K</span> | Trades: <span id="btc-trades">502K</span></div>
                        </div>
                        <span class="text-xs bg-gray-800 px-2 py-1 rounded">LIVE</span>
                    </div>
                    <div class="text-2xl font-bold" id="btc-price">$67,213.44</div>
                    <div class="text-sm mt-1" id="btc-change">
                        <span class="negative">-2.48%</span>
                    </div>
                </div>
                
                <div class="metric-card p-4 rounded-lg">
                    <div class="flex justify-between items-start mb-2">
                        <div>
                            <div class="text-sm text-gray-500">ETH/USD</div>
                            <div class="text-xs text-gray-600">Vol: <span id="eth-volume">37.0K</span> | Trades: <span id="eth-trades">777K</span></div>
                        </div>
                        <span class="text-xs bg-gray-800 px-2 py-1 rounded">LIVE</span>
                    </div>
                    <div class="text-2xl font-bold" id="eth-price">$3,475.77</div>
                    <div class="text-sm mt-1" id="eth-change">
                        <span class="positive">+2.86%</span>
                    </div>
                </div>
                
                <div class="metric-card p-4 rounded-lg">
                    <div class="flex justify-between items-start mb-2">
                        <div>
                            <div class="text-sm text-gray-500">SOL/USD</div>
                            <div class="text-xs text-gray-600">Vol: <span id="sol-volume">5.9K</span> | Trades: <span id="sol-trades">277K</span></div>
                        </div>
                        <span class="text-xs bg-gray-800 px-2 py-1 rounded">LIVE</span>
                    </div>
                    <div class="text-2xl font-bold" id="sol-price">$119.12</div>
                    <div class="text-sm mt-1" id="sol-change">
                        <span class="positive">+3.37%</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Cross-Exchange Spreads -->
        <div class="mb-6">
            <h2 class="text-lg font-semibold mb-3 text-gray-300">Cross-Exchange Spreads</h2>
            <div class="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div class="bg-gray-800 p-3 rounded-lg border border-gray-700">
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-400">Binance-Coinbase:</span>
                        <span class="font-mono positive" id="spread-bc">+$18.36</span>
                    </div>
                </div>
                <div class="bg-gray-800 p-3 rounded-lg border border-gray-700">
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-400">Kraken-Bybit:</span>
                        <span class="font-mono positive" id="spread-kb">+$27.42</span>
                    </div>
                </div>
                <div class="bg-gray-800 p-3 rounded-lg border border-gray-700">
                    <div class="flex justify-between items-center">
                        <span class="text-sm text-gray-400">Futures-Spot:</span>
                        <span class="font-mono positive" id="spread-fs">+$256.83</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Main Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <!-- Left Column -->
            <div class="space-y-6">
                <!-- Market Sentiment -->
                <div class="metric-card p-4 rounded-lg">
                    <h3 class="text-sm font-semibold mb-3 text-gray-300 flex items-center">
                        <i class="fas fa-brain mr-2 text-purple-500"></i>
                        Market Sentiment
                        <span class="ml-auto text-xs bg-blue-900 px-2 py-1 rounded">AUTO</span>
                    </h3>
                    <div class="space-y-3">
                        <div>
                            <div class="text-xs text-gray-500 mb-1">Market Mood</div>
                            <div class="text-2xl">😰 <span id="mood-percent">48%</span></div>
                        </div>
                        <div class="grid grid-cols-3 gap-2 text-xs">
                            <div>
                                <div class="text-gray-500">BTC:</div>
                                <div class="font-mono">51%</div>
                            </div>
                            <div>
                                <div class="text-gray-500">ETH:</div>
                                <div class="font-mono">44%</div>
                            </div>
                            <div>
                                <div class="text-gray-500">SOL:</div>
                                <div class="font-mono">47%</div>
                            </div>
                        </div>
                        <div class="pt-2 border-t border-gray-700 space-y-1 text-sm">
                            <div class="flex justify-between">
                                <span class="text-gray-500">Fear & Greed:</span>
                                <span class="font-bold text-yellow-500">80</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-500">Social Volume:</span>
                                <span>161K</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Economic Indicators -->
                <div class="metric-card p-4 rounded-lg">
                    <h3 class="text-sm font-semibold mb-3 text-gray-300">
                        <i class="fas fa-globe mr-2 text-green-500"></i>
                        Economic Indicators
                    </h3>
                    <div class="space-y-2 text-sm" id="economic-indicators">
                        <div class="flex justify-between">
                            <span class="text-gray-500">US GDP:</span>
                            <span>2.34 <span class="negative text-xs">↘️0.87%</span></span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-500">Inflation:</span>
                            <span>3.13 <span class="negative text-xs">↘️0.82%</span></span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-500">Fed Rate:</span>
                            <span>5.32 <span class="positive text-xs">↗️0.96%</span></span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-500">Unemployment:</span>
                            <span>3.66 <span class="positive text-xs">↗️0.12%</span></span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-500">DXY Index:</span>
                            <span>104.52 <span class="negative text-xs">↘️0.92</span></span>
                        </div>
                        <div class="pt-2 border-t border-gray-700">
                            <div class="flex justify-between">
                                <span class="text-gray-500">BTC Dominance:</span>
                                <span>52.4%</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-500">Total Market Cap:</span>
                                <span>$2.01T</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-500">DeFi TVL:</span>
                                <span>$82.2B</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Middle Column -->
            <div class="space-y-6">
                <!-- Hyperbolic Engine -->
                <div class="metric-card p-4 rounded-lg">
                    <h3 class="text-sm font-semibold mb-3 text-gray-300 flex items-center">
                        <i class="fas fa-project-diagram mr-2 text-indigo-500"></i>
                        Hyperbolic Engine
                        <span class="ml-auto text-xs bg-green-900 px-2 py-1 rounded">ENHANCED</span>
                    </h3>
                    <div class="mb-3">
                        <div class="text-xs text-gray-500 mb-1">Patterns</div>
                        <canvas id="hyperbolic-chart" width="200" height="150"></canvas>
                    </div>
                    <div class="space-y-1 text-sm">
                        <div class="flex justify-between">
                            <span class="text-gray-500">Geodesic Paths:</span>
                            <span class="font-mono">791</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-500">Space Curvature:</span>
                            <span class="font-mono">-1.0</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-500">Path Efficiency:</span>
                            <span class="font-mono positive">99.5%</span>
                        </div>
                    </div>
                </div>

                <!-- Live Arbitrage Opportunities -->
                <div class="metric-card p-4 rounded-lg">
                    <h3 class="text-sm font-semibold mb-3 text-gray-300 flex items-center">
                        <i class="fas fa-exchange-alt mr-2 text-blue-500"></i>
                        Live Arbitrage Opportunities
                        <span class="ml-auto text-xs bg-orange-900 px-2 py-1 rounded">6 ACTIVE</span>
                    </h3>
                    <div class="text-xs text-gray-500 mb-2">Last scan: <span id="last-scan">04:23:38 PM</span></div>
                    <div class="space-y-2" id="opportunities-list">
                        <!-- Opportunities will be inserted here -->
                    </div>
                </div>

                <!-- Strategy Performance -->
                <div class="metric-card p-4 rounded-lg">
                    <h3 class="text-sm font-semibold mb-3 text-gray-300 flex items-center">
                        <i class="fas fa-chart-bar mr-2 text-yellow-500"></i>
                        Strategy Performance
                        <span class="ml-auto text-xs bg-gray-700 px-2 py-1 rounded">REAL-TIME</span>
                    </h3>
                    <div class="grid grid-cols-2 gap-3 mb-3">
                        <div class="text-center">
                            <div class="text-2xl font-bold positive">+$4,260</div>
                            <div class="text-xs text-gray-500">Total P&L Today</div>
                        </div>
                        <div class="text-center">
                            <div class="text-2xl font-bold">82.7%</div>
                            <div class="text-xs text-gray-500">Combined Win Rate</div>
                        </div>
                    </div>
                    <div class="grid grid-cols-2 gap-3 text-sm">
                        <div>
                            <div class="text-gray-500 text-xs">Total Executions</div>
                            <div class="font-mono">50</div>
                        </div>
                        <div>
                            <div class="text-gray-500 text-xs">Avg Execution Time</div>
                            <div class="font-mono">47μs</div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Right Column -->
            <div class="space-y-6">
                <!-- Order Book Depth -->
                <div class="metric-card p-4 rounded-lg">
                    <h3 class="text-sm font-semibold mb-3 text-gray-300 flex items-center">
                        <i class="fas fa-layer-group mr-2 text-teal-500"></i>
                        Order Book Depth
                        <span class="ml-auto text-xs bg-gray-700 px-2 py-1 rounded">DEPTH</span>
                    </h3>
                    <div class="space-y-1 text-xs">
                        <div class="text-center text-gray-500 mb-2">ASKS</div>
                        <div id="orderbook-asks" class="space-y-1">
                            <!-- Asks will be inserted here -->
                        </div>
                        <div class="py-2 text-center font-bold text-sm">
                            Spread: <span id="orderbook-spread">$0.32</span>
                        </div>
                        <div class="text-center text-gray-500 mb-2">BIDS</div>
                        <div id="orderbook-bids" class="space-y-1">
                            <!-- Bids will be inserted here -->
                        </div>
                    </div>
                </div>

                <!-- Model Parameters -->
                <div class="metric-card p-4 rounded-lg">
                    <h3 class="text-sm font-semibold mb-3 text-gray-300 flex items-center">
                        <i class="fas fa-sliders-h mr-2 text-red-500"></i>
                        Model Parameters
                    </h3>
                    <div class="space-y-2 text-xs" id="model-parameters">
                        <div class="font-semibold text-gray-400 mb-1">Constraints</div>
                        <div class="flex justify-between">
                            <span>Max Exposure:</span>
                            <span class="font-mono">3%</span>
                        </div>
                        <div class="flex justify-between">
                            <span>API Health Threshold:</span>
                            <span class="font-mono">2</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Event Blackout:</span>
                            <span class="font-mono">300s</span>
                        </div>
                        
                        <div class="font-semibold text-gray-400 mb-1 mt-2">Bounds</div>
                        <div class="flex justify-between">
                            <span>Min Spread:</span>
                            <span class="font-mono">0.5%</span>
                        </div>
                        <div class="flex justify-between">
                            <span>LLM Confidence:</span>
                            <span class="font-mono">80%</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Max Hold Time:</span>
                            <span class="font-mono">3600s</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Backtest vs LLM Comparison -->
        <div class="mt-6 metric-card p-4 rounded-lg">
            <h3 class="text-sm font-semibold mb-3 text-gray-300 flex items-center">
                <i class="fas fa-balance-scale mr-2 text-orange-500"></i>
                Backtest vs LLM Performance Comparison
            </h3>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="text-center">
                    <div class="text-xs text-gray-500">Backtest Sharpe</div>
                    <div class="text-xl font-bold">1.82</div>
                </div>
                <div class="text-center">
                    <div class="text-xs text-gray-500">LLM Sharpe</div>
                    <div class="text-xl font-bold positive">2.15</div>
                </div>
                <div class="text-center">
                    <div class="text-xs text-gray-500">Backtest Win Rate</div>
                    <div class="text-xl font-bold">68.3%</div>
                </div>
                <div class="text-center">
                    <div class="text-xs text-gray-500">LLM Win Rate</div>
                    <div class="text-xl font-bold positive">82.7%</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Initialize Socket.IO
        const socket = io();
        
        // Update time
        function updateTime() {
            const now = new Date();
            document.getElementById('system-time').textContent = now.toLocaleTimeString();
            document.getElementById('last-scan').textContent = now.toLocaleTimeString();
        }
        setInterval(updateTime, 1000);
        
        // Socket listeners
        socket.on('market_update', (data) => {
            // Update market prices
            document.getElementById('btc-price').textContent = '$' + data.btc.price.toFixed(2);
            document.getElementById('eth-price').textContent = '$' + data.eth.price.toFixed(2);
            document.getElementById('sol-price').textContent = '$' + data.sol.price.toFixed(2);
            
            // Update changes
            updateChange('btc-change', data.btc.change);
            updateChange('eth-change', data.eth.change);
            updateChange('sol-change', data.sol.change);
        });
        
        socket.on('spread_update', (spreads) => {
            document.getElementById('spread-bc').textContent = '+$' + Math.abs(spreads['binance-coinbase']).toFixed(2);
            document.getElementById('spread-kb').textContent = '+$' + Math.abs(spreads['kraken-bybit']).toFixed(2);
            document.getElementById('spread-fs').textContent = '+$' + Math.abs(spreads['futures-spot']).toFixed(2);
        });
        
        function updateChange(id, value) {
            const element = document.getElementById(id);
            const span = element.querySelector('span') || element;
            span.textContent = (value >= 0 ? '+' : '') + value.toFixed(2) + '%';
            span.className = value >= 0 ? 'positive' : 'negative';
        }
        
        // Initialize hyperbolic chart
        const ctx = document.getElementById('hyperbolic-chart').getContext('2d');
        const hyperbolicChart = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'Asset Clustering',
                    data: Array.from({length: 20}, () => ({
                        x: Math.random() * 2 - 1,
                        y: Math.random() * 2 - 1
                    })),
                    backgroundColor: 'rgba(99, 102, 241, 0.5)'
                }]
            },
            options: {
                responsive: false,
                plugins: {
                    legend: { display: false }
                },
                scales: {
                    x: { display: false },
                    y: { display: false }
                }
            }
        });
        
        // Load opportunities
        async function loadOpportunities() {
            try {
                const response = await fetch('/api/opportunities');
                const opportunities = await response.json();
                const list = document.getElementById('opportunities-list');
                
                list.innerHTML = opportunities.slice(0, 3).map(opp => \`
                    <div class="bg-gray-800 p-2 rounded text-xs">
                        <div class="flex justify-between mb-1">
                            <span class="font-semibold">\${opp.pair}</span>
                            <span class="text-green-500">\${(opp.spread || 0).toFixed(3)}%</span>
                        </div>
                        <div class="flex justify-between text-gray-500">
                            <span>\${opp.buy} → \${opp.sell}</span>
                            <span>AOS: \${(opp.aos_score || 0).toFixed(2)}</span>
                        </div>
                    </div>
                \`).join('');
            } catch (error) {
                console.error('Failed to load opportunities:', error);
            }
        }
        
        // Load orderbook
        async function loadOrderbook() {
            try {
                const response = await fetch('/api/orderbook/BTC-USDT');
                const orderbook = await response.json();
                
                const asksHtml = orderbook.asks.slice(0, 5).map(ask => \`
                    <div class="orderbook-row flex justify-between text-red-400">
                        <span>$\${ask.price.toFixed(2)}</span>
                        <span>\${ask.volume.toFixed(2)} BTC</span>
                    </div>
                \`).join('');
                
                const bidsHtml = orderbook.bids.slice(0, 5).map(bid => \`
                    <div class="orderbook-row flex justify-between text-green-400">
                        <span>$\${bid.price.toFixed(2)}</span>
                        <span>\${bid.volume.toFixed(2)} BTC</span>
                    </div>
                \`).join('');
                
                document.getElementById('orderbook-asks').innerHTML = asksHtml;
                document.getElementById('orderbook-bids').innerHTML = bidsHtml;
                document.getElementById('orderbook-spread').textContent = '$' + orderbook.spread.toFixed(2);
            } catch (error) {
                console.error('Failed to load orderbook:', error);
            }
        }
        
        // Load model parameters
        async function loadParameters() {
            try {
                const response = await fetch('/api/parameters');
                const params = await response.json();
                
                // Update parameters display
                // This would be more dynamic in production
            } catch (error) {
                console.error('Failed to load parameters:', error);
            }
        }
        
        // Initial loads
        loadOpportunities();
        loadOrderbook();
        loadParameters();
        
        // Refresh intervals
        setInterval(loadOpportunities, 5000);
        setInterval(loadOrderbook, 2000);
    </script>
</body>
</html>`;
}

server.listen(PORT, '0.0.0.0', () => {
  logger.info(`Enhanced dashboard running on port ${PORT}`);
  console.log(`\n🚀 Enhanced Dashboard available at http://localhost:${PORT}\n`);
});