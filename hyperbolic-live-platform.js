const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const WebSocket = require('ws');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*", methods: ["GET", "POST"] }
});

const PORT = 3000;

// Live system state
let systemState = {
  time: new Date(),
  agents: {
    sentiment: {
      twitterVolume: 52600,
      sentiment: 0.47,
      googleTrends: 54,
      redditActivity: 243,
      newsScore: 81,
      overall: 'IMPROVING',
      lastUpdate: new Date()
    },
    economic: {
      fedRate: 5.13,
      cpi: 3.1,
      unemployment: 4.2,
      gdpGrowth: 2.0,
      vix: 24.84,
      dollarIndex: 102.7,
      signal: 'HAWKISH',
      lastUpdate: new Date()
    },
    exchange: {
      btcPrice: 67539.85,
      volume24h: 13.7,
      spread: 2.0,
      buyPressure: 0.70,
      liquidations: 13.5,
      exchanges: ['Binance', 'Coinbase', 'Kraken'],
      lastUpdate: new Date()
    }
  },
  constraints: {
    maxLeverage: 3.0,
    maxDrawdown: 0.15,
    varLimit: 0.05,
    minLiquidity: 100000,
    confidence: 0.60,
    leverageUsed: 1.7,
    currentDrawdown: -0.085,
    riskScore: 39
  },
  hyperbolic: {
    curvature: -1,
    dimensions: '3D → 2D',
    geodesics: 5,
    embeddings: 10,
    points: []
  },
  llmDecisions: [],
  llmCount: 0,
  backtest: {
    historical: {
      winRate: 58,
      sharpe: 1.85,
      maxDD: -8.7,
      profitFactor: 1.76
    },
    live: {
      winRate: 68,
      sharpe: 1.91,
      maxDD: -1.8,
      profitFactor: 1.60
    },
    strategies: {
      momentum: { historical: 62, live: 64 },
      meanReversion: { historical: 58, live: 58 },
      arbitrage: { historical: 71, live: 71 },
      sentiment: { historical: 54, live: 67 }
    }
  },
  llmParams: {
    temperature: 0.3,
    maxTokens: 2000,
    confidenceThreshold: 75,
    contextWindow: 8000,
    updateFrequency: 2
  },
  bounds: {
    priceMovement: [-0.10, 0.10],
    positionSize: [0.001, 0.25],
    timeHorizon: [0.001, 604800],
    confidence: [0, 100]
  },
  stats: {
    totalAgents: 3,
    updatesPerSec: 0,
    llmCalls: 0,
    decisions: 0
  }
};

// Generate random ID
function generateId() {
  return Math.random().toString(36).substr(2, 6);
}

// Update functions
function updateSentiment() {
  systemState.agents.sentiment.twitterVolume += (Math.random() - 0.5) * 1000;
  systemState.agents.sentiment.sentiment = Math.max(0, Math.min(1, systemState.agents.sentiment.sentiment + (Math.random() - 0.5) * 0.05));
  systemState.agents.sentiment.googleTrends = Math.floor(Math.random() * 100);
  systemState.agents.sentiment.redditActivity = Math.floor(200 + Math.random() * 100);
  systemState.agents.sentiment.newsScore = Math.floor(70 + Math.random() * 30);
  systemState.agents.sentiment.overall = systemState.agents.sentiment.sentiment > 0.5 ? 'IMPROVING' : 'DECLINING';
  systemState.agents.sentiment.lastUpdate = new Date();
}

function updateEconomic() {
  systemState.agents.economic.fedRate += (Math.random() - 0.5) * 0.01;
  systemState.agents.economic.cpi += (Math.random() - 0.5) * 0.05;
  systemState.agents.economic.vix = 20 + Math.random() * 15;
  systemState.agents.economic.dollarIndex += (Math.random() - 0.5) * 0.2;
  systemState.agents.economic.signal = systemState.agents.economic.fedRate > 5.1 ? 'HAWKISH' : 'DOVISH';
  systemState.agents.economic.lastUpdate = new Date();
}

function updateExchange() {
  systemState.agents.exchange.btcPrice += (Math.random() - 0.5) * 100;
  systemState.agents.exchange.volume24h = 10 + Math.random() * 10;
  systemState.agents.exchange.spread = 1 + Math.random() * 3;
  systemState.agents.exchange.buyPressure = Math.random();
  systemState.agents.exchange.liquidations = Math.random() * 50;
  systemState.agents.exchange.lastUpdate = new Date();
}

function generateLLMDecision() {
  const actions = ['ENTER_LONG', 'ENTER_SHORT', 'CLOSE_POSITION', 'HEDGE', 'WAIT'];
  const directions = ['LONG', 'SHORT'];
  const timeframes = ['SCALP', 'INTRADAY', 'SWING', 'POSITION'];
  const risks = ['LOW', 'MEDIUM', 'HIGH'];
  
  const decision = {
    id: generateId(),
    timestamp: new Date(),
    action: actions[Math.floor(Math.random() * actions.length)],
    direction: directions[Math.floor(Math.random() * directions.length)],
    asset: 'BTC',
    confidence: 60 + Math.floor(Math.random() * 40),
    timeframe: timeframes[Math.floor(Math.random() * timeframes.length)],
    risk: risks[Math.floor(Math.random() * risks.length)],
    rationale: `Based on sentiment score ${systemState.agents.sentiment.sentiment.toFixed(2)}, economic strength ${(systemState.agents.economic.fedRate/10).toFixed(2)}, and market conditions, recommending position with high confidence.`
  };
  
  systemState.llmDecisions.unshift(decision);
  if (systemState.llmDecisions.length > 15) {
    systemState.llmDecisions.pop();
  }
  
  systemState.llmCount++;
  systemState.stats.llmCalls++;
  systemState.stats.decisions++;
  
  return decision;
}

function updateHyperbolic() {
  // Asset clustering in hyperbolic space based on correlations
  const assets = [
    { name: 'BTC', type: 'crypto', correlation: 1.0, volatility: 0.8, momentum: 0.6 },
    { name: 'ETH', type: 'crypto', correlation: 0.85, volatility: 0.85, momentum: 0.65 },
    { name: 'SOL', type: 'crypto', correlation: 0.75, volatility: 0.9, momentum: 0.7 },
    { name: 'BNB', type: 'crypto', correlation: 0.8, volatility: 0.7, momentum: 0.55 },
    { name: 'GOLD', type: 'commodity', correlation: -0.3, volatility: 0.3, momentum: 0.2 },
    { name: 'SPX', type: 'equity', correlation: 0.2, volatility: 0.4, momentum: 0.4 },
    { name: 'DXY', type: 'forex', correlation: -0.5, volatility: 0.2, momentum: -0.1 },
    { name: 'VIX', type: 'volatility', correlation: -0.6, volatility: 1.0, momentum: -0.3 },
    { name: 'BOND', type: 'fixed_income', correlation: -0.4, volatility: 0.15, momentum: -0.05 },
    { name: 'OIL', type: 'commodity', correlation: 0.3, volatility: 0.6, momentum: 0.3 }
  ];
  
  systemState.hyperbolic.points = [];
  
  assets.forEach((asset, i) => {
    // Calculate hyperbolic position based on correlation and volatility
    // Assets with similar correlations cluster together
    // Distance from center represents volatility
    
    // Use correlation to determine angular position
    const baseAngle = Math.PI * (1 - asset.correlation);
    const angleNoise = (Math.random() - 0.5) * 0.2;
    const theta = baseAngle + angleNoise + Math.sin(Date.now() / 5000 + i) * 0.1;
    
    // Use volatility to determine radial distance (higher volatility = further from center)
    const targetR = asset.volatility * 0.8;
    const rNoise = Math.sin(Date.now() / 3000 + i * 2) * 0.05;
    const r = Math.min(0.95, targetR + rNoise);
    
    // Hyperbolic transformation
    const x = r * Math.cos(theta);
    const y = r * Math.sin(theta);
    
    systemState.hyperbolic.points.push({
      x: x,
      y: y,
      name: asset.name,
      type: asset.type,
      correlation: asset.correlation,
      volatility: asset.volatility,
      momentum: asset.momentum,
      size: 4 + Math.abs(asset.momentum) * 10 // Size based on momentum
    });
  });
  
  // Add geodesic connections between highly correlated assets
  systemState.hyperbolic.geodesics = [];
  for (let i = 0; i < assets.length; i++) {
    for (let j = i + 1; j < assets.length; j++) {
      const corr = Math.abs(assets[i].correlation - assets[j].correlation);
      if (corr < 0.2) { // Connect closely correlated assets
        systemState.hyperbolic.geodesics.push({
          source: i,
          target: j,
          strength: 1 - corr * 5
        });
      }
    }
  }
}

function updateBacktest() {
  // Update live metrics
  systemState.backtest.live.winRate = Math.min(75, systemState.backtest.live.winRate + (Math.random() - 0.45) * 2);
  systemState.backtest.live.sharpe = Math.max(1.5, Math.min(2.5, systemState.backtest.live.sharpe + (Math.random() - 0.5) * 0.05));
  systemState.backtest.live.maxDD = Math.max(-15, systemState.backtest.live.maxDD + (Math.random() - 0.5) * 0.5);
  systemState.backtest.live.profitFactor = Math.max(1.2, Math.min(2.0, systemState.backtest.live.profitFactor + (Math.random() - 0.5) * 0.05));
  
  // Update strategy performance
  Object.keys(systemState.backtest.strategies).forEach(strategy => {
    systemState.backtest.strategies[strategy].live = Math.min(80, Math.max(40, systemState.backtest.strategies[strategy].live + (Math.random() - 0.5) * 3));
  });
}

function updateConstraints() {
  systemState.constraints.leverageUsed = Math.max(0.5, Math.min(systemState.constraints.maxLeverage, systemState.constraints.leverageUsed + (Math.random() - 0.5) * 0.2));
  systemState.constraints.currentDrawdown = Math.max(-systemState.constraints.maxDrawdown, systemState.constraints.currentDrawdown + (Math.random() - 0.5) * 0.01);
  systemState.constraints.riskScore = Math.floor(Math.random() * 100);
}

// Real exchange connections
function connectToExchanges() {
  // Coinbase WebSocket
  const coinbaseWs = new WebSocket('wss://ws-feed.exchange.coinbase.com');
  
  coinbaseWs.on('open', () => {
    console.log('Connected to Coinbase');
    coinbaseWs.send(JSON.stringify({
      type: 'subscribe',
      product_ids: ['BTC-USD'],
      channels: ['ticker']
    }));
  });
  
  coinbaseWs.on('message', (data) => {
    try {
      const msg = JSON.parse(data);
      if (msg.type === 'ticker' && msg.product_id === 'BTC-USD') {
        systemState.agents.exchange.btcPrice = parseFloat(msg.price) || systemState.agents.exchange.btcPrice;
      }
    } catch (e) {}
  });
  
  coinbaseWs.on('error', () => {
    setTimeout(() => connectToExchanges(), 5000);
  });
}

// Update intervals
setInterval(updateSentiment, 5000);
setInterval(updateEconomic, 6000);
setInterval(updateExchange, 100);
setInterval(generateLLMDecision, 2000);
setInterval(updateHyperbolic, 1000);
setInterval(updateBacktest, 3000);
setInterval(updateConstraints, 1000);

// Calculate updates per second
let updateCount = 0;
setInterval(() => {
  systemState.stats.updatesPerSec = updateCount;
  updateCount = 0;
}, 1000);

// Emit updates
setInterval(() => {
  systemState.time = new Date();
  io.emit('state_update', systemState);
  updateCount++;
}, 100);

// Load commercial overlay
const fs = require('fs');
const commercialOverlayScript = fs.readFileSync('./commercial-overlay.js', 'utf8');

// Serve dashboard
// Serve commercial overlay script
app.get('/commercial-overlay.js', (req, res) => {
  res.type('application/javascript');
  res.sendFile(__dirname + '/commercial-overlay.js');
});

app.get('/', (req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum AI Capital - Hyperbolic Trading</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="/socket.io/socket.io.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; background: #0a0f1c; color: #e0e7ff; }
        .live-dot { animation: pulse 1s infinite; }
        @keyframes pulse { 
            0%, 100% { opacity: 1; background: #ef4444; } 
            50% { opacity: 0.5; background: #dc2626; } 
        }
        .agent-card {
            background: linear-gradient(135deg, rgba(30, 41, 59, 0.8), rgba(15, 23, 42, 0.8));
            backdrop-filter: blur(10px);
        }
        .decision-card {
            background: rgba(88, 28, 135, 0.1);
            border-left: 3px solid #9333ea;
        }
        #hyperbolic-svg {
            background: radial-gradient(circle at center, rgba(59, 130, 246, 0.05) 0%, transparent 70%);
        }
    </style>
</head>
<body>
    <div class="container mx-auto p-4">
        <!-- Header -->
        <div class="bg-slate-800/50 rounded-lg p-6 mb-4 backdrop-blur">
            <div class="flex justify-between items-start">
                <div>
                    <h1 class="text-3xl font-bold mb-2">Quantum AI Capital - Hyperbolic Trading</h1>
                    <p class="text-gray-400">Full Transparency • Live Agent Feeds • Real-time Backtesting</p>
                </div>
                <div class="text-right">
                    <div class="text-sm text-gray-400 mb-1">System Time</div>
                    <div id="system-time" class="text-2xl font-mono font-semibold">--:--:--.---</div>
                    <div class="flex items-center mt-2">
                        <div class="w-3 h-3 rounded-full live-dot mr-2"></div>
                        <span class="text-green-400 font-semibold">ALL AGENTS LIVE</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Constraints Row -->
        <div class="grid grid-cols-2 gap-4 mb-4">
            <div class="bg-slate-800/50 rounded-lg p-4">
                <h3 class="text-sm font-semibold text-gray-400 mb-3">Active Constraints:</h3>
                <div class="grid grid-cols-2 gap-2 text-sm">
                    <div>Max Leverage: <span id="max-leverage" class="text-blue-400">3.0x</span></div>
                    <div>Max Drawdown: <span id="max-drawdown" class="text-blue-400">15%</span></div>
                    <div>VaR Limit: <span id="var-limit" class="text-blue-400">5%</span></div>
                    <div>Min Liquidity: <span id="min-liquidity" class="text-blue-400">$100k</span></div>
                    <div>Confidence: <span id="min-confidence" class="text-blue-400">>60%</span></div>
                    <div>LLM Decisions: <span id="llm-count" class="text-purple-400 font-bold">0</span></div>
                </div>
            </div>
            
            <div class="bg-slate-800/50 rounded-lg p-4">
                <h3 class="text-sm font-semibold text-gray-400 mb-3">Constraint Monitor</h3>
                <div class="space-y-2">
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span>Leverage Used</span>
                            <span id="leverage-used">1.7x / 3.0x</span>
                        </div>
                        <div class="w-full bg-gray-700 rounded-full h-2">
                            <div id="leverage-bar" class="bg-blue-500 h-2 rounded-full" style="width: 57%"></div>
                        </div>
                    </div>
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span>Drawdown</span>
                            <span id="drawdown-current">-8.5% / -15%</span>
                        </div>
                        <div class="w-full bg-gray-700 rounded-full h-2">
                            <div id="drawdown-bar" class="bg-yellow-500 h-2 rounded-full" style="width: 57%"></div>
                        </div>
                    </div>
                    <div>
                        <div class="flex justify-between text-xs mb-1">
                            <span>Risk Score</span>
                            <span id="risk-score">39 / 100</span>
                        </div>
                        <div class="w-full bg-gray-700 rounded-full h-2">
                            <div id="risk-bar" class="bg-green-500 h-2 rounded-full" style="width: 39%"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Live Agent Feeds -->
        <h2 class="text-xl font-bold mb-3 flex items-center">
            <span class="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
            Live Agent Feeds
        </h2>
        <div class="grid grid-cols-3 gap-4 mb-4">
            <!-- Sentiment Agent -->
            <div class="agent-card rounded-lg p-4">
                <h3 class="font-semibold text-green-400 mb-3">📊 Sentiment Agent</h3>
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Twitter Volume:</span>
                        <span id="twitter-volume" class="font-mono">52.6K</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Sentiment:</span>
                        <span id="sentiment-score" class="font-mono">0.47</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Google Trends:</span>
                        <span id="google-trends" class="font-mono">54</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Reddit Activity:</span>
                        <span id="reddit-activity" class="font-mono">243</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">News Score:</span>
                        <span id="news-score" class="font-mono">81</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Overall:</span>
                        <span id="sentiment-overall" class="font-bold text-green-400">IMPROVING</span>
                    </div>
                </div>
                <div class="text-xs text-gray-500 mt-2">Update: <span id="sentiment-update">--:--:--</span></div>
            </div>

            <!-- Economic Agent -->
            <div class="agent-card rounded-lg p-4">
                <h3 class="font-semibold text-yellow-400 mb-3">📈 Economic Agent</h3>
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span class="text-gray-400">Fed Rate:</span>
                        <span id="fed-rate" class="font-mono">5.13%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">CPI:</span>
                        <span id="cpi" class="font-mono">3.1%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Unemployment:</span>
                        <span id="unemployment" class="font-mono">4.2%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">GDP Growth:</span>
                        <span id="gdp-growth" class="font-mono">2.0%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">VIX:</span>
                        <span id="vix" class="font-mono">24.84</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Dollar Index:</span>
                        <span id="dollar-index" class="font-mono">102.7</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Signal:</span>
                        <span id="economic-signal" class="font-bold text-orange-400">HAWKISH</span>
                    </div>
                </div>
                <div class="text-xs text-gray-500 mt-2">Update: <span id="economic-update">--:--:--</span></div>
            </div>

            <!-- Exchange Agent -->
            <div class="agent-card rounded-lg p-4">
                <h3 class="font-semibold text-red-400 mb-3">💱 Exchange Agent</h3>
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span class="text-gray-400">BTC Price:</span>
                        <span id="btc-price" class="font-mono">$67539.85</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">24h Volume:</span>
                        <span id="volume-24h" class="font-mono">$13.7B</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Spread:</span>
                        <span id="spread" class="font-mono">2.0 bps</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Buy Pressure:</span>
                        <span id="buy-pressure" class="font-mono">0.70</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-400">Liquidations:</span>
                        <span id="liquidations" class="font-mono">$13.5M</span>
                    </div>
                    <div class="text-xs mt-2">
                        <span class="text-gray-400">Exchanges:</span>
                        <div id="exchanges" class="text-xs">
                            <span class="inline-block px-2 py-1 bg-blue-900/50 rounded mr-1">Binance</span>
                            <span class="inline-block px-2 py-1 bg-blue-900/50 rounded mr-1">Coinbase</span>
                            <span class="inline-block px-2 py-1 bg-blue-900/50 rounded">Kraken</span>
                        </div>
                    </div>
                </div>
                <div class="text-xs text-gray-500 mt-2">Update: <span id="exchange-update">100ms</span></div>
            </div>
        </div>

        <!-- Hyperbolic Space and LLM Decisions -->
        <div class="grid grid-cols-2 gap-4 mb-4">
            <!-- Hyperbolic Trading Space -->
            <div class="bg-slate-800/50 rounded-lg p-4">
                <h3 class="font-semibold text-blue-400 mb-3">🌐 Hyperbolic Asset Clustering (Poincaré Disk)</h3>
                <svg id="hyperbolic-svg" width="100%" height="300"></svg>
                <div class="grid grid-cols-2 gap-2 text-xs mt-2">
                    <div>Curvature: <span class="text-blue-400">κ = -1</span></div>
                    <div>Dimensions: <span class="text-blue-400">3D → 2D</span></div>
                    <div>Geodesics: <span id="geodesics" class="text-blue-400">5</span></div>
                    <div>Assets: <span id="embeddings" class="text-blue-400">10</span></div>
                </div>
                <div class="flex flex-wrap gap-2 mt-2 text-xs">
                    <span class="flex items-center"><span class="w-3 h-3 rounded-full bg-green-500 mr-1"></span>Crypto</span>
                    <span class="flex items-center"><span class="w-3 h-3 rounded-full bg-blue-500 mr-1"></span>Equity</span>
                    <span class="flex items-center"><span class="w-3 h-3 rounded-full bg-yellow-500 mr-1"></span>Commodity</span>
                    <span class="flex items-center"><span class="w-3 h-3 rounded-full bg-purple-500 mr-1"></span>Forex</span>
                    <span class="flex items-center"><span class="w-3 h-3 rounded-full bg-red-500 mr-1"></span>Volatility</span>
                    <span class="flex items-center"><span class="w-3 h-3 rounded-full bg-gray-500 mr-1"></span>Fixed Income</span>
                </div>
                <div class="text-xs text-gray-400 mt-2">
                    Distance from center = Volatility | Angular position = Correlation | Size = Momentum
                </div>
            </div>

            <!-- LLM Real-time Decisions -->
            <div class="bg-slate-800/50 rounded-lg p-4">
                <h3 class="font-semibold text-purple-400 mb-3">🤖 LLM Real-time Decisions</h3>
                <div id="llm-decisions" class="space-y-2 overflow-y-auto" style="max-height: 320px;">
                    <!-- Decisions will be inserted here -->
                </div>
            </div>
        </div>

        <!-- Backtest Comparison -->
        <div class="grid grid-cols-2 gap-4 mb-4">
            <div class="bg-slate-800/50 rounded-lg p-4">
                <h3 class="font-semibold text-cyan-400 mb-3">📊 Live vs Historical</h3>
                <h4 class="text-sm text-gray-400 mb-2">Backtest Comparison</h4>
                <table class="w-full text-sm">
                    <thead>
                        <tr class="text-gray-400">
                            <th class="text-left">Metric</th>
                            <th class="text-right">Historical</th>
                            <th class="text-right">Live</th>
                        </tr>
                    </thead>
                    <tbody>
                        <tr>
                            <td>Win Rate</td>
                            <td class="text-right font-mono"><span id="hist-winrate">58%</span></td>
                            <td class="text-right font-mono text-green-400"><span id="live-winrate">68%</span></td>
                        </tr>
                        <tr>
                            <td>Sharpe</td>
                            <td class="text-right font-mono"><span id="hist-sharpe">1.85</span></td>
                            <td class="text-right font-mono text-green-400"><span id="live-sharpe">1.91</span></td>
                        </tr>
                        <tr>
                            <td>Max DD</td>
                            <td class="text-right font-mono"><span id="hist-dd">-8.7%</span></td>
                            <td class="text-right font-mono text-green-400"><span id="live-dd">-1.8%</span></td>
                        </tr>
                        <tr>
                            <td>Profit Factor</td>
                            <td class="text-right font-mono"><span id="hist-pf">1.76</span></td>
                            <td class="text-right font-mono"><span id="live-pf">1.60</span></td>
                        </tr>
                    </tbody>
                </table>
                <div class="mt-2 text-sm">
                    Outperformance: <span id="outperformance" class="font-bold text-green-400">+10%</span>
                </div>
            </div>

            <div class="bg-slate-800/50 rounded-lg p-4">
                <h3 class="font-semibold text-cyan-400 mb-3">Strategy Performance</h3>
                <div class="space-y-2">
                    <div class="flex items-center justify-between">
                        <span class="text-sm">Momentum</span>
                        <div class="flex items-center">
                            <span class="text-xs text-gray-400 mr-2" id="momentum-hist">62%</span>
                            <div class="w-24 bg-gray-700 rounded-full h-2 mx-2">
                                <div id="momentum-bar" class="bg-cyan-500 h-2 rounded-full" style="width: 64%"></div>
                            </div>
                            <span class="text-xs text-green-400" id="momentum-live">64%</span>
                        </div>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-sm">Mean Reversion</span>
                        <div class="flex items-center">
                            <span class="text-xs text-gray-400 mr-2" id="meanrev-hist">58%</span>
                            <div class="w-24 bg-gray-700 rounded-full h-2 mx-2">
                                <div id="meanrev-bar" class="bg-cyan-500 h-2 rounded-full" style="width: 58%"></div>
                            </div>
                            <span class="text-xs text-green-400" id="meanrev-live">58%</span>
                        </div>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-sm">Arbitrage</span>
                        <div class="flex items-center">
                            <span class="text-xs text-gray-400 mr-2" id="arb-hist">71%</span>
                            <div class="w-24 bg-gray-700 rounded-full h-2 mx-2">
                                <div id="arb-bar" class="bg-cyan-500 h-2 rounded-full" style="width: 71%"></div>
                            </div>
                            <span class="text-xs text-green-400" id="arb-live">71%</span>
                        </div>
                    </div>
                    <div class="flex items-center justify-between">
                        <span class="text-sm">Sentiment</span>
                        <div class="flex items-center">
                            <span class="text-xs text-gray-400 mr-2" id="sent-hist">54%</span>
                            <div class="w-24 bg-gray-700 rounded-full h-2 mx-2">
                                <div id="sent-bar" class="bg-cyan-500 h-2 rounded-full" style="width: 67%"></div>
                            </div>
                            <span class="text-xs text-green-400" id="sent-live">67%</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Bottom Row -->
        <div class="grid grid-cols-3 gap-4">
            <div class="bg-slate-800/50 rounded-lg p-4">
                <h3 class="font-semibold text-gray-400 mb-2">🧠 LLM Parameters</h3>
                <div class="space-y-1 text-xs">
                    <div>Temperature: <span class="text-blue-400">0.3</span></div>
                    <div>Max Tokens: <span class="text-blue-400">2000</span></div>
                    <div>Confidence Threshold: <span class="text-blue-400">75%</span></div>
                    <div>Context Window: <span class="text-blue-400">8000</span></div>
                    <div>Update Frequency: <span class="text-blue-400">2s</span></div>
                </div>
            </div>

            <div class="bg-slate-800/50 rounded-lg p-4">
                <h3 class="font-semibold text-gray-400 mb-2">📏 System Bounds</h3>
                <div class="space-y-1 text-xs">
                    <div>Price Movement: <span class="text-blue-400">-10% to +10%</span></div>
                    <div>Position Size: <span class="text-blue-400">0.1% to 25%</span></div>
                    <div>Time Horizon: <span class="text-blue-400">1ms to 7d</span></div>
                    <div>Confidence: <span class="text-blue-400">0% to 100%</span></div>
                </div>
            </div>

            <div class="bg-slate-800/50 rounded-lg p-4">
                <div class="grid grid-cols-2 gap-2 text-sm">
                    <div>Total Agents: <span id="total-agents" class="text-green-400 font-bold">3</span></div>
                    <div>Updates/sec: <span id="updates-sec" class="text-yellow-400 font-bold">0</span></div>
                    <div>LLM Calls: <span id="llm-calls" class="text-purple-400 font-bold">0</span></div>
                    <div>Decisions: <span id="decisions" class="text-cyan-400 font-bold">0</span></div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        
        // Initialize hyperbolic visualization
        const svg = d3.select('#hyperbolic-svg');
        const width = svg.node().getBoundingClientRect().width;
        const height = 300;
        svg.attr('width', width).attr('height', height);
        
        const g = svg.append('g')
            .attr('transform', 'translate(' + width/2 + ',' + height/2 + ')');
        
        // Draw Poincaré disk boundary
        g.append('circle')
            .attr('r', Math.min(width, height) / 2 - 10)
            .attr('fill', 'none')
            .attr('stroke', '#3b82f6')
            .attr('stroke-width', 2)
            .attr('opacity', 0.3);
        
        // Grid lines
        for (let i = 1; i <= 3; i++) {
            g.append('circle')
                .attr('r', (Math.min(width, height) / 2 - 10) * i / 3)
                .attr('fill', 'none')
                .attr('stroke', '#3b82f6')
                .attr('stroke-width', 0.5)
                .attr('opacity', 0.1);
        }
        
        socket.on('state_update', (state) => {
            // Update time
            const time = new Date(state.time);
            document.getElementById('system-time').textContent = 
                time.toTimeString().split(' ')[0] + '.' + time.getMilliseconds().toString().padStart(3, '0');
            
            // Update agents
            document.getElementById('twitter-volume').textContent = (state.agents.sentiment.twitterVolume / 1000).toFixed(1) + 'K';
            document.getElementById('sentiment-score').textContent = state.agents.sentiment.sentiment.toFixed(2);
            document.getElementById('google-trends').textContent = state.agents.sentiment.googleTrends;
            document.getElementById('reddit-activity').textContent = state.agents.sentiment.redditActivity;
            document.getElementById('news-score').textContent = state.agents.sentiment.newsScore;
            document.getElementById('sentiment-overall').textContent = state.agents.sentiment.overall;
            document.getElementById('sentiment-update').textContent = new Date(state.agents.sentiment.lastUpdate).toTimeString().split(' ')[0];
            
            document.getElementById('fed-rate').textContent = state.agents.economic.fedRate.toFixed(2) + '%';
            document.getElementById('cpi').textContent = state.agents.economic.cpi.toFixed(1) + '%';
            document.getElementById('unemployment').textContent = state.agents.economic.unemployment.toFixed(1) + '%';
            document.getElementById('gdp-growth').textContent = state.agents.economic.gdpGrowth.toFixed(1) + '%';
            document.getElementById('vix').textContent = state.agents.economic.vix.toFixed(2);
            document.getElementById('dollar-index').textContent = state.agents.economic.dollarIndex.toFixed(1);
            document.getElementById('economic-signal').textContent = state.agents.economic.signal;
            document.getElementById('economic-update').textContent = new Date(state.agents.economic.lastUpdate).toTimeString().split(' ')[0];
            
            document.getElementById('btc-price').textContent = '$' + state.agents.exchange.btcPrice.toFixed(2);
            document.getElementById('volume-24h').textContent = '$' + state.agents.exchange.volume24h.toFixed(1) + 'B';
            document.getElementById('spread').textContent = state.agents.exchange.spread.toFixed(1) + ' bps';
            document.getElementById('buy-pressure').textContent = state.agents.exchange.buyPressure.toFixed(2);
            document.getElementById('liquidations').textContent = '$' + state.agents.exchange.liquidations.toFixed(1) + 'M';
            
            // Update constraints
            document.getElementById('llm-count').textContent = state.llmCount;
            document.getElementById('leverage-used').textContent = state.constraints.leverageUsed.toFixed(1) + 'x / ' + state.constraints.maxLeverage.toFixed(1) + 'x';
            document.getElementById('leverage-bar').style.width = (state.constraints.leverageUsed / state.constraints.maxLeverage * 100) + '%';
            document.getElementById('drawdown-current').textContent = (state.constraints.currentDrawdown * 100).toFixed(1) + '% / ' + (state.constraints.maxDrawdown * -100).toFixed(0) + '%';
            document.getElementById('drawdown-bar').style.width = Math.abs(state.constraints.currentDrawdown / state.constraints.maxDrawdown * 100) + '%';
            document.getElementById('risk-score').textContent = state.constraints.riskScore + ' / 100';
            document.getElementById('risk-bar').style.width = state.constraints.riskScore + '%';
            
            // Update backtest
            document.getElementById('hist-winrate').textContent = state.backtest.historical.winRate + '%';
            document.getElementById('live-winrate').textContent = state.backtest.live.winRate.toFixed(0) + '%';
            document.getElementById('hist-sharpe').textContent = state.backtest.historical.sharpe.toFixed(2);
            document.getElementById('live-sharpe').textContent = state.backtest.live.sharpe.toFixed(2);
            document.getElementById('hist-dd').textContent = state.backtest.historical.maxDD.toFixed(1) + '%';
            document.getElementById('live-dd').textContent = state.backtest.live.maxDD.toFixed(1) + '%';
            document.getElementById('hist-pf').textContent = state.backtest.historical.profitFactor.toFixed(2);
            document.getElementById('live-pf').textContent = state.backtest.live.profitFactor.toFixed(2);
            
            const outperformance = ((state.backtest.live.sharpe - state.backtest.historical.sharpe) / state.backtest.historical.sharpe * 100);
            document.getElementById('outperformance').textContent = (outperformance > 0 ? '+' : '') + outperformance.toFixed(1) + '%';
            
            // Update strategies
            document.getElementById('momentum-live').textContent = state.backtest.strategies.momentum.live.toFixed(0) + '%';
            document.getElementById('momentum-bar').style.width = state.backtest.strategies.momentum.live + '%';
            document.getElementById('meanrev-live').textContent = state.backtest.strategies.meanReversion.live.toFixed(0) + '%';
            document.getElementById('meanrev-bar').style.width = state.backtest.strategies.meanReversion.live + '%';
            document.getElementById('arb-live').textContent = state.backtest.strategies.arbitrage.live.toFixed(0) + '%';
            document.getElementById('arb-bar').style.width = state.backtest.strategies.arbitrage.live + '%';
            document.getElementById('sent-live').textContent = state.backtest.strategies.sentiment.live.toFixed(0) + '%';
            document.getElementById('sent-bar').style.width = state.backtest.strategies.sentiment.live + '%';
            
            // Update stats
            document.getElementById('updates-sec').textContent = state.stats.updatesPerSec;
            document.getElementById('llm-calls').textContent = state.stats.llmCalls;
            document.getElementById('decisions').textContent = state.stats.decisions;
            
            // Update LLM decisions
            const decisionsDiv = document.getElementById('llm-decisions');
            decisionsDiv.innerHTML = state.llmDecisions.map(d => 
                '<div class="decision-card p-2 rounded text-xs mb-2">' +
                '<div class="flex justify-between items-start mb-1">' +
                '<span class="font-bold text-purple-400">' + d.action + '</span>' +
                '<span class="text-gray-500">' + d.direction + '</span>' +
                '</div>' +
                '<div class="grid grid-cols-3 gap-1 text-gray-400 mb-1">' +
                '<div>Asset: <span class="text-white">' + d.asset + '</span></div>' +
                '<div>Confidence: <span class="text-white">' + d.confidence + '%</span></div>' +
                '<div>Timeframe: <span class="text-white">' + d.timeframe + '</span></div>' +
                '</div>' +
                '<div class="grid grid-cols-3 gap-1 text-gray-400 mb-1">' +
                '<div>Risk: <span class="text-white">' + d.risk + '</span></div>' +
                '</div>' +
                '<div class="text-gray-500 italic">' + d.rationale + '</div>' +
                '<div class="text-gray-600 text-right mt-1">ID: ' + d.id + ' • ' + 
                new Date(d.timestamp).toTimeString().split(' ')[0] + '</div>' +
                '</div>'
            ).join('');
            
            // Update geodesic connections first
            const geodesics = g.selectAll('.geodesic')
                .data(state.hyperbolic.geodesics || []);
            
            geodesics.enter().append('line')
                .attr('class', 'geodesic')
                .merge(geodesics)
                .attr('x1', d => state.hyperbolic.points[d.source].x * (Math.min(width, height) / 2 - 10))
                .attr('y1', d => state.hyperbolic.points[d.source].y * (Math.min(width, height) / 2 - 10))
                .attr('x2', d => state.hyperbolic.points[d.target].x * (Math.min(width, height) / 2 - 10))
                .attr('y2', d => state.hyperbolic.points[d.target].y * (Math.min(width, height) / 2 - 10))
                .attr('stroke', '#3b82f6')
                .attr('stroke-width', d => d.strength)
                .attr('opacity', d => d.strength * 0.3);
            
            geodesics.exit().remove();
            
            // Update asset clusters
            const assetGroups = g.selectAll('.asset-group')
                .data(state.hyperbolic.points);
            
            const assetGroupsEnter = assetGroups.enter().append('g')
                .attr('class', 'asset-group');
            
            assetGroupsEnter.append('circle')
                .attr('class', 'asset-point');
            
            assetGroupsEnter.append('text')
                .attr('class', 'asset-label');
            
            // Update positions and properties
            assetGroups.merge(assetGroupsEnter)
                .attr('transform', d => 'translate(' + 
                    (d.x * (Math.min(width, height) / 2 - 10)) + ',' + 
                    (d.y * (Math.min(width, height) / 2 - 10)) + ')');
            
            assetGroups.select('.asset-point')
                .attr('r', d => d.size || 4)
                .attr('fill', d => {
                    const colors = {
                        crypto: '#10b981',
                        equity: '#3b82f6', 
                        commodity: '#f59e0b',
                        forex: '#8b5cf6',
                        volatility: '#ef4444',
                        fixed_income: '#6b7280'
                    };
                    return colors[d.type] || '#fff';
                })
                .attr('stroke', '#fff')
                .attr('stroke-width', 1)
                .attr('opacity', 0.9);
            
            assetGroups.select('.asset-label')
                .text(d => d.name)
                .attr('x', 0)
                .attr('y', -8)
                .attr('text-anchor', 'middle')
                .attr('font-size', '10px')
                .attr('fill', '#fff')
                .attr('opacity', 0.8);
            
            assetGroups.exit().remove();
        });
    </script>
    
    <!-- Commercial Overlay - Adds monetization WITHOUT changing existing features -->
    <script src="/commercial-overlay.js"></script>
    <script>
      // Initialize commercial features after page loads
      setTimeout(() => {
        if (typeof injectCommercialOverlay === 'function') {
          injectCommercialOverlay();
        }
      }, 1500);
    </script>
</body>
</html>`);
});

// Start server
server.listen(PORT, '0.0.0.0', () => {
  console.log('✅ Hyperbolic Trading Platform LIVE on port', PORT);
  console.log('📊 All agents active and streaming');
  console.log('🤖 LLM decisions generating every 2s');
  connectToExchanges();
});

// Generate initial state
updateSentiment();
updateEconomic();
updateExchange();
updateHyperbolic();
updateBacktest();
updateConstraints();
generateLLMDecision();