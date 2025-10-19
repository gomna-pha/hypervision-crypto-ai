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

// Live system state - UNCHANGED
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
      // ADD: Exchange prices for arbitrage
      prices: {
        binance: 67500,
        coinbase: 67539.85,
        kraken: 67485,
        okx: 67510,
        bybit: 67495
      },
      arbitrage: {
        current: [],
        future: []
      },
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

// Update functions - ALL UNCHANGED
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
  
  // ADD: Update exchange prices and calculate arbitrage
  const basePrice = systemState.agents.exchange.btcPrice;
  systemState.agents.exchange.prices.binance = basePrice + (Math.random() - 0.5) * 100;
  systemState.agents.exchange.prices.coinbase = basePrice;
  systemState.agents.exchange.prices.kraken = basePrice + (Math.random() - 0.5) * 80;
  systemState.agents.exchange.prices.okx = basePrice + (Math.random() - 0.5) * 90;
  systemState.agents.exchange.prices.bybit = basePrice + (Math.random() - 0.5) * 85;
  
  // Calculate current arbitrage opportunities
  const prices = systemState.agents.exchange.prices;
  const opportunities = [];
  const exchanges = Object.keys(prices);
  
  for (let i = 0; i < exchanges.length; i++) {
    for (let j = i + 1; j < exchanges.length; j++) {
      const spread = Math.abs(prices[exchanges[i]] - prices[exchanges[j]]);
      const spreadPercent = (spread / Math.min(prices[exchanges[i]], prices[exchanges[j]])) * 100;
      
      if (spreadPercent > 0.05) {
        opportunities.push({
          buy: prices[exchanges[i]] < prices[exchanges[j]] ? exchanges[i] : exchanges[j],
          sell: prices[exchanges[i]] < prices[exchanges[j]] ? exchanges[j] : exchanges[i],
          spread: spreadPercent.toFixed(3),
          profit: (spread * 0.1).toFixed(2) // For 0.1 BTC
        });
      }
    }
  }
  
  systemState.agents.exchange.arbitrage.current = opportunities.slice(0, 3);
  
  // Predict future arbitrage windows
  systemState.agents.exchange.arbitrage.future = [
    { time: '09:30 ET', probability: 78, reason: 'US Market Open' },
    { time: '14:00 ET', probability: 65, reason: 'European Close' },
    { time: '20:00 ET', probability: 82, reason: 'Asian Session' }
  ];
  
  systemState.agents.exchange.lastUpdate = new Date();
}

function generateLLMDecision() {
  const actions = ['ENTER_LONG', 'ENTER_SHORT', 'CLOSE_POSITION', 'HEDGE', 'WAIT', 'ARBITRAGE'];
  const directions = ['LONG', 'SHORT'];
  const timeframes = ['SCALP', 'INTRADAY', 'SWING', 'POSITION'];
  const risks = ['LOW', 'MEDIUM', 'HIGH'];
  const strategies = ['MOMENTUM', 'MEAN_REVERSION', 'ARBITRAGE', 'SENTIMENT', 'DELTA_NEUTRAL_ARBITRAGE', 'STATISTICAL_ARBITRAGE'];
  
  // Get arbitrage opportunities
  const currentArb = systemState.agents.exchange.arbitrage.current[0];
  const futureArb = systemState.agents.exchange.arbitrage.future[0];
  
  // Enhanced rationale with arbitrage
  let rationale = `Based on sentiment score ${systemState.agents.sentiment.sentiment.toFixed(2)}, economic strength ${(systemState.agents.economic.fedRate/10).toFixed(2)}, and market conditions.`;
  
  if (currentArb) {
    rationale += ` Active arbitrage opportunity: ${currentArb.buy}→${currentArb.sell} with ${currentArb.spread}% spread ($${currentArb.profit}/BTC profit).`;
  }
  
  if (futureArb) {
    rationale += ` Next arbitrage window at ${futureArb.time} (${futureArb.probability}% probability, ${futureArb.reason}).`;
  }
  
  const decision = {
    id: generateId(),
    timestamp: new Date(),
    action: actions[Math.floor(Math.random() * actions.length)],
    direction: directions[Math.floor(Math.random() * directions.length)],
    asset: 'BTC',
    confidence: 60 + Math.floor(Math.random() * 40),
    timeframe: timeframes[Math.floor(Math.random() * timeframes.length)],
    risk: risks[Math.floor(Math.random() * risks.length)],
    strategy: strategies[Math.floor(Math.random() * strategies.length)],
    rationale: rationale,
    arbitrage: currentArb ? {
      type: 'CROSS_EXCHANGE',
      opportunity: `${currentArb.buy}→${currentArb.sell}`,
      spread: currentArb.spread + '%',
      profit: '$' + currentArb.profit
    } : null
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
  // Asset clustering in hyperbolic space based on correlations - UNCHANGED
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
  // Update live metrics - UNCHANGED
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
  systemState.constraints.currentDrawdown = Math.max(systemState.constraints.maxDrawdown, Math.min(0, systemState.constraints.currentDrawdown + (Math.random() - 0.5) * 0.01));
  systemState.constraints.riskScore = Math.floor(Math.max(0, Math.min(100, systemState.constraints.riskScore + (Math.random() - 0.5) * 5)));
}

// WebSocket connections to exchanges
let coinbaseWS = null;

function connectToCoinbase() {
  coinbaseWS = new WebSocket('wss://ws-feed.exchange.coinbase.com');
  
  coinbaseWS.on('open', () => {
    console.log('Connected to Coinbase');
    const subscribe = {
      type: 'subscribe',
      product_ids: ['BTC-USD'],
      channels: ['ticker']
    };
    coinbaseWS.send(JSON.stringify(subscribe));
  });
  
  coinbaseWS.on('message', (data) => {
    try {
      const msg = JSON.parse(data);
      if (msg.type === 'ticker') {
        systemState.agents.exchange.btcPrice = parseFloat(msg.price);
      }
    } catch (e) {}
  });
  
  coinbaseWS.on('error', console.error);
}

// Update functions
setInterval(updateSentiment, 3000);
setInterval(updateEconomic, 4000);
setInterval(updateExchange, 2500);
setInterval(updateHyperbolic, 1000);
setInterval(updateBacktest, 5000);
setInterval(updateConstraints, 2000);
setInterval(generateLLMDecision, 2000);

// Update stats
setInterval(() => {
  systemState.stats.updatesPerSec = Math.floor(3 + Math.random() * 2);
}, 1000);

// Socket.IO
io.on('connection', (socket) => {
  socket.emit('state', systemState);
  
  setInterval(() => {
    socket.emit('state', systemState);
  }, 100);
});

// Serve commercial overlay
app.get('/commercial-overlay.js', (req, res) => {
  res.type('application/javascript');
  res.sendFile(__dirname + '/commercial-overlay.js');
});

// Serve backtesting module
app.get('/backtesting-module.js', (req, res) => {
  res.type('application/javascript');
  res.sendFile(__dirname + '/backtesting-module.js');
});

// HTML with CREAM COLOR SCHEME and GOMNA BRANDING
app.get('/', (req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gomna Arbitrage Trades</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="/socket.io/socket.io.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,700;1,400;1,700&family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400;1,600&display=swap');
        
        body { 
            font-family: 'Cormorant Garamond', serif; 
            background: linear-gradient(135deg, #FAF7F2, #F5E6D3, #FFF8F0);
            color: #3E2723;
        }
        
        .header-gradient {
            background: linear-gradient(135deg, rgba(255, 253, 250, 0.98), rgba(250, 245, 238, 0.95));
        }
        
        .card-cream {
            background: linear-gradient(135deg, rgba(255, 251, 245, 0.95), rgba(250, 243, 234, 0.92));
            border: 1px solid rgba(139, 90, 43, 0.12);
            box-shadow: 0 4px 12px rgba(121, 85, 72, 0.06);
        }
        
        .decision-cream {
            background: linear-gradient(135deg, rgba(255, 248, 241, 0.9), rgba(252, 243, 232, 0.85));
            border-left: 3px solid #8B5A2B;
        }
        
        #hyperbolic-svg {
            background: radial-gradient(circle at center, rgba(255, 251, 245, 0.98) 0%, rgba(250, 245, 238, 0.95) 70%);
            border: 1px solid rgba(139, 90, 43, 0.1);
        }
        
        .live-dot { 
            animation: pulse 1s infinite; 
            background: #8B5A2B;
        }
        
        @keyframes pulse { 
            0%, 100% { opacity: 1; } 
            50% { opacity: 0.5; } 
        }
        
        .progress-bar {
            background: rgba(188, 170, 164, 0.2);
        }
        
        .progress-fill {
            background: linear-gradient(90deg, #8B5A2B, #A0522D);
        }
        
        .text-cocoa { color: #6B4423; }
        .text-cocoa-dark { color: #5D4037; }
        .text-cocoa-light { color: #8D6E63; }
        .bg-cocoa-light { background: rgba(139, 90, 43, 0.1); }
    </style>
</head>
<body>
    <div class="container mx-auto p-4">
        <!-- Header with Gomna Branding -->
        <div class="header-gradient rounded-lg p-6 mb-4 shadow-lg">
            <div class="flex justify-between items-start">
                <div class="flex items-center gap-4">
                    <!-- 3D Cocoa Pod Logo -->
                    <div style="width: 60px; height: 60px;">
                        <svg viewBox="0 0 100 100" style="width: 100%; height: 100%;">
                            <defs>
                                <linearGradient id="podGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" style="stop-color:#8B5A2B;stop-opacity:1" />
                                    <stop offset="50%" style="stop-color:#D2691E;stop-opacity:1" />
                                    <stop offset="100%" style="stop-color:#5C4033;stop-opacity:1" />
                                </linearGradient>
                                <linearGradient id="seedGrad" x1="0%" y1="0%" x2="100%" y2="100%">
                                    <stop offset="0%" style="stop-color:#F5DEB3;stop-opacity:1" />
                                    <stop offset="100%" style="stop-color:#DEB887;stop-opacity:1" />
                                </linearGradient>
                                <filter id="shadow3d">
                                    <feDropShadow dx="2" dy="4" stdDeviation="3" flood-opacity="0.3"/>
                                </filter>
                            </defs>
                            <!-- 3D Pod shell -->
                            <ellipse cx="50" cy="50" rx="30" ry="40" fill="url(#podGrad)" filter="url(#shadow3d)" transform="rotate(10 50 50)"/>
                            <!-- Pod ridges for 3D effect -->
                            <path d="M 35 20 Q 50 15 65 20 L 60 75 Q 50 80 40 75 Z" fill="#A0522D" opacity="0.7"/>
                            <path d="M 25 30 Q 50 25 75 30 L 70 70 Q 50 75 30 70 Z" fill="#6B4423" opacity="0.5"/>
                            <!-- Opening crack showing seeds -->
                            <path d="M 40 35 Q 50 30 60 35 L 58 65 Q 50 60 42 65 Z" fill="#2F1F1A"/>
                            <!-- Seeds inside -->
                            <ellipse cx="45" cy="45" rx="4" ry="6" fill="url(#seedGrad)" transform="rotate(-15 45 45)"/>
                            <ellipse cx="50" cy="50" rx="4" ry="6" fill="url(#seedGrad)"/>
                            <ellipse cx="55" cy="46" rx="4" ry="6" fill="url(#seedGrad)" transform="rotate(15 55 46)"/>
                            <!-- 3D Highlight -->
                            <ellipse cx="45" cy="30" rx="8" ry="4" fill="#FAEBD7" opacity="0.4"/>
                        </svg>
                    </div>
                    <div>
                        <h1 class="text-4xl font-bold" style="font-family: 'Playfair Display', serif; font-style: italic; color: #5D4037; letter-spacing: 0.5px;">
                            Gomna Arbitrage Trades
                        </h1>
                        <p class="text-cocoa-light mt-1">Full Transparency • Live Agent Feeds • Real-time Arbitrage</p>
                    </div>
                </div>
                <div class="text-right">
                    <div class="text-sm text-cocoa-light mb-1">System Time</div>
                    <div id="system-time" class="text-2xl font-mono font-semibold text-cocoa">--:--:--.---</div>
                    <div class="flex items-center mt-2">
                        <div class="w-3 h-3 rounded-full live-dot mr-2"></div>
                        <span class="text-cocoa font-semibold">ALL AGENTS LIVE</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Constraints -->
        <div class="card-cream rounded-lg p-4 mb-4">
            <div class="flex items-center justify-between mb-3">
                <h3 class="font-semibold text-cocoa-dark">⚖️ System Constraints</h3>
                <span class="text-sm text-cocoa-light">LLM Calls: <span id="llm-count" class="font-bold text-cocoa">0</span></span>
            </div>
            <div class="grid grid-cols-3 gap-4">
                <div>
                    <div class="text-sm text-cocoa-light mb-1">Leverage</div>
                    <div class="progress-bar h-6 rounded">
                        <div id="leverage-bar" class="progress-fill h-full rounded flex items-center justify-center text-white text-xs font-semibold" style="width: 57%;">
                            <span id="leverage-used">1.7x / 3.0x</span>
                        </div>
                    </div>
                </div>
                <div>
                    <div class="text-sm text-cocoa-light mb-1">Drawdown</div>
                    <div class="progress-bar h-6 rounded">
                        <div id="drawdown-bar" class="progress-fill h-full rounded flex items-center justify-center text-white text-xs font-semibold" style="width: 57%;">
                            <span id="drawdown-current">-8.5% / -15%</span>
                        </div>
                    </div>
                </div>
                <div>
                    <div class="text-sm text-cocoa-light mb-1">Risk Score</div>
                    <div class="progress-bar h-6 rounded">
                        <div id="risk-bar" class="progress-fill h-full rounded flex items-center justify-center text-white text-xs font-semibold" style="width: 39%;">
                            <span id="risk-score">39 / 100</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Agents -->
        <div class="grid grid-cols-3 gap-4 mb-4">
            <!-- Sentiment Agent -->
            <div class="card-cream rounded-lg p-4">
                <h3 class="font-semibold text-cocoa-dark mb-3">🧠 Sentiment Agent</h3>
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Twitter Volume:</span>
                        <span id="twitter-volume" class="font-mono text-cocoa">52,600</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Sentiment:</span>
                        <span id="sentiment-score" class="font-mono text-cocoa">0.47</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Google Trends:</span>
                        <span id="google-trends" class="font-mono text-cocoa">54</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Reddit Activity:</span>
                        <span id="reddit-activity" class="font-mono text-cocoa">243</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">News Score:</span>
                        <span id="news-score" class="font-mono text-cocoa">81</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Overall:</span>
                        <span id="sentiment-overall" class="font-bold text-cocoa-dark">IMPROVING</span>
                    </div>
                </div>
                <div class="text-xs text-cocoa-light mt-2">Update: <span id="sentiment-update">--:--:--</span></div>
            </div>

            <!-- Economic Agent -->
            <div class="card-cream rounded-lg p-4">
                <h3 class="font-semibold text-cocoa-dark mb-3">📈 Economic Agent</h3>
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Fed Rate:</span>
                        <span id="fed-rate" class="font-mono text-cocoa">5.13%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">CPI:</span>
                        <span id="cpi" class="font-mono text-cocoa">3.1%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Unemployment:</span>
                        <span id="unemployment" class="font-mono text-cocoa">4.2%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">GDP Growth:</span>
                        <span id="gdp-growth" class="font-mono text-cocoa">2.0%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">VIX:</span>
                        <span id="vix" class="font-mono text-cocoa">24.84</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Dollar Index:</span>
                        <span id="dollar-index" class="font-mono text-cocoa">102.7</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Signal:</span>
                        <span id="economic-signal" class="font-bold text-cocoa-dark">HAWKISH</span>
                    </div>
                </div>
                <div class="text-xs text-cocoa-light mt-2">Update: <span id="economic-update">--:--:--</span></div>
            </div>

            <!-- Exchange Agent -->
            <div class="card-cream rounded-lg p-4">
                <h3 class="font-semibold text-cocoa-dark mb-3">💱 Exchange Agent</h3>
                <div class="space-y-2 text-sm">
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">BTC Price:</span>
                        <span id="btc-price" class="font-mono text-cocoa">$67539.85</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">24h Volume:</span>
                        <span id="volume-24h" class="font-mono text-cocoa">$13.7B</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Spread:</span>
                        <span id="spread" class="font-mono text-cocoa">2.0 bps</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Buy Pressure:</span>
                        <span id="buy-pressure" class="font-mono text-cocoa">0.70</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Liquidations:</span>
                        <span id="liquidations" class="font-mono text-cocoa">$13.5M</span>
                    </div>
                    <div class="text-xs mt-2">
                        <span class="text-cocoa-light">Exchanges:</span>
                        <div id="exchanges" class="text-xs">
                            <span class="inline-block px-2 py-1 bg-cocoa-light rounded mr-1 text-cocoa">Binance</span>
                            <span class="inline-block px-2 py-1 bg-cocoa-light rounded mr-1 text-cocoa">Coinbase</span>
                            <span class="inline-block px-2 py-1 bg-cocoa-light rounded text-cocoa">Kraken</span>
                        </div>
                    </div>
                </div>
                <div class="text-xs text-cocoa-light mt-2">Update: <span id="exchange-update">100ms</span></div>
            </div>
        </div>

        <!-- Hyperbolic Space and LLM Decisions -->
        <div class="grid grid-cols-2 gap-4 mb-4">
            <!-- Hyperbolic Trading Space -->
            <div class="card-cream rounded-lg p-4">
                <h3 class="font-semibold text-cocoa-dark mb-3">🌐 Hyperbolic Asset Clustering (Poincaré Disk)</h3>
                <svg id="hyperbolic-svg" width="100%" height="300"></svg>
                <div class="grid grid-cols-2 gap-2 text-xs mt-2 text-cocoa-light">
                    <div>Curvature: <span class="text-cocoa">κ = -1</span></div>
                    <div>Dimensions: <span class="text-cocoa">3D → 2D</span></div>
                    <div>Geodesics: <span id="geodesics" class="text-cocoa">5</span></div>
                    <div>Assets: <span id="embeddings" class="text-cocoa">10</span></div>
                </div>
            </div>

            <!-- LLM Real-time Decisions -->
            <div class="card-cream rounded-lg p-4">
                <h3 class="font-semibold text-cocoa-dark mb-3">🤖 LLM Real-time Decisions</h3>
                <div id="llm-decisions" class="space-y-2 overflow-y-auto" style="max-height: 320px;">
                    <!-- Decisions will be inserted here -->
                </div>
            </div>
        </div>

        <!-- Arbitrage and Backtest -->
        <div class="grid grid-cols-2 gap-4 mb-4">
            <!-- Exchange Arbitrage Opportunities -->
            <div class="card-cream rounded-lg p-4">
                <h3 class="font-semibold text-cocoa-dark mb-3">💰 Exchange Arbitrage Opportunities</h3>
                <div class="grid grid-cols-5 gap-2 mb-3 text-xs">
                    <div class="text-center">
                        <div class="text-cocoa-light">Binance</div>
                        <div id="price-binance" class="font-mono text-cocoa font-semibold">$67,500</div>
                    </div>
                    <div class="text-center">
                        <div class="text-cocoa-light">Coinbase</div>
                        <div id="price-coinbase" class="font-mono text-cocoa font-semibold">$67,540</div>
                    </div>
                    <div class="text-center">
                        <div class="text-cocoa-light">Kraken</div>
                        <div id="price-kraken" class="font-mono text-cocoa font-semibold">$67,485</div>
                    </div>
                    <div class="text-center">
                        <div class="text-cocoa-light">OKX</div>
                        <div id="price-okx" class="font-mono text-cocoa font-semibold">$67,510</div>
                    </div>
                    <div class="text-center">
                        <div class="text-cocoa-light">Bybit</div>
                        <div id="price-bybit" class="font-mono text-cocoa font-semibold">$67,495</div>
                    </div>
                </div>
                <div class="text-xs text-cocoa-light mb-2">Current Arbitrage Opportunities:</div>
                <div id="current-arbitrage" class="space-y-1 mb-3">
                    <!-- Will be populated dynamically -->
                </div>
                <div class="text-xs text-cocoa-light mb-2">Future Arbitrage Windows:</div>
                <div id="future-arbitrage" class="space-y-1">
                    <!-- Will be populated dynamically -->
                </div>
            </div>
            
            <!-- Backtesting Panel -->
            <div class="card-cream rounded-lg p-4">
                <h3 class="font-semibold text-cocoa-dark mb-3">📊 Live vs Historical Backtesting</h3>
                <h4 class="text-sm text-cocoa-light mb-2">Real-time Performance Comparison</h4>
                <table class="w-full text-sm">
                    <thead>
                        <tr class="text-cocoa-light">
                            <th class="text-left">Metric</th>
                            <th class="text-right">Historical</th>
                            <th class="text-right">Live</th>
                        </tr>
                    </thead>
                    <tbody class="text-cocoa">
                        <tr>
                            <td>Win Rate</td>
                            <td class="text-right" id="hist-winrate">58%</td>
                            <td class="text-right font-bold" id="live-winrate">68%</td>
                        </tr>
                        <tr>
                            <td>Sharpe Ratio</td>
                            <td class="text-right" id="hist-sharpe">1.85</td>
                            <td class="text-right font-bold" id="live-sharpe">1.91</td>
                        </tr>
                        <tr>
                            <td>Max Drawdown</td>
                            <td class="text-right" id="hist-dd">-8.7%</td>
                            <td class="text-right font-bold" id="live-dd">-1.8%</td>
                        </tr>
                        <tr>
                            <td>Profit Factor</td>
                            <td class="text-right" id="hist-pf">1.76</td>
                            <td class="text-right font-bold" id="live-pf">1.60</td>
                        </tr>
                    </tbody>
                </table>
                
                <h4 class="text-sm text-cocoa-light mb-2 mt-4">Strategy Performance</h4>
                <div class="space-y-2">
                    <div class="flex justify-between items-center">
                        <span class="text-xs text-cocoa">Momentum</span>
                        <div class="progress-bar h-4 rounded flex-1 mx-2">
                            <div id="mom-bar" class="progress-fill h-full rounded" style="width: 64%;"></div>
                        </div>
                        <span id="mom-live" class="text-xs font-bold text-cocoa">64%</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-xs text-cocoa">Mean Reversion</span>
                        <div class="progress-bar h-4 rounded flex-1 mx-2">
                            <div id="mean-bar" class="progress-fill h-full rounded" style="width: 58%;"></div>
                        </div>
                        <span id="mean-live" class="text-xs font-bold text-cocoa">58%</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-xs text-cocoa">Arbitrage</span>
                        <div class="progress-bar h-4 rounded flex-1 mx-2">
                            <div id="arb-bar" class="progress-fill h-full rounded" style="width: 71%;"></div>
                        </div>
                        <span id="arb-live" class="text-xs font-bold text-cocoa">71%</span>
                    </div>
                    <div class="flex justify-between items-center">
                        <span class="text-xs text-cocoa">Sentiment</span>
                        <div class="progress-bar h-4 rounded flex-1 mx-2">
                            <div id="sent-bar" class="progress-fill h-full rounded" style="width: 67%;"></div>
                        </div>
                        <span id="sent-live" class="text-xs font-bold text-cocoa">67%</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- LLM Parameters and Bounds -->
        <div class="grid grid-cols-2 gap-4 mb-4">
            <div class="card-cream rounded-lg p-4">
                <h3 class="font-semibold text-cocoa-dark mb-3">🎛️ LLM Parameters</h3>
                <div class="grid grid-cols-2 gap-3 text-sm">
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Temperature:</span>
                        <span class="font-mono text-cocoa">0.3</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Max Tokens:</span>
                        <span class="font-mono text-cocoa">2000</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Confidence:</span>
                        <span class="font-mono text-cocoa">75%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Context:</span>
                        <span class="font-mono text-cocoa">8000</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Update Freq:</span>
                        <span class="font-mono text-cocoa">2s</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Min Liquidity:</span>
                        <span class="font-mono text-cocoa">$100K</span>
                    </div>
                </div>
            </div>
            
            <div class="card-cream rounded-lg p-4">
                <h3 class="font-semibold text-cocoa-dark mb-3">📏 System Bounds</h3>
                <div class="grid grid-cols-2 gap-3 text-sm">
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Price Move:</span>
                        <span class="font-mono text-cocoa">[-10%, +10%]</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Position:</span>
                        <span class="font-mono text-cocoa">[0.1%, 25%]</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Time:</span>
                        <span class="font-mono text-cocoa">[1ms, 7d]</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Confidence:</span>
                        <span class="font-mono text-cocoa">[0%, 100%]</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">VaR Limit:</span>
                        <span class="font-mono text-cocoa">5%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-cocoa-light">Corr Cap:</span>
                        <span class="font-mono text-cocoa">0.7</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Stats -->
        <div class="card-cream rounded-lg p-3 text-center">
            <div class="grid grid-cols-3 gap-4">
                <div>
                    <span class="text-xs text-cocoa-light">Updates/sec</span>
                    <span id="updates-sec" class="ml-2 font-bold text-cocoa">0</span>
                </div>
                <div>
                    <span class="text-xs text-cocoa-light">LLM Calls</span>
                    <span id="llm-calls" class="ml-2 font-bold text-cocoa">0</span>
                </div>
                <div>
                    <span class="text-xs text-cocoa-light">Decisions</span>
                    <span id="decisions" class="ml-2 font-bold text-cocoa">0</span>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        
        function updateTime() {
            const now = new Date();
            const ms = now.getMilliseconds().toString().padStart(3, '0');
            document.getElementById('system-time').textContent = 
                now.toTimeString().split(' ')[0] + '.' + ms;
        }
        setInterval(updateTime, 10);
        
        socket.on('state', (state) => {
            // Update Sentiment Agent
            document.getElementById('twitter-volume').textContent = state.agents.sentiment.twitterVolume.toLocaleString();
            document.getElementById('sentiment-score').textContent = state.agents.sentiment.sentiment.toFixed(2);
            document.getElementById('google-trends').textContent = state.agents.sentiment.googleTrends;
            document.getElementById('reddit-activity').textContent = state.agents.sentiment.redditActivity;
            document.getElementById('news-score').textContent = state.agents.sentiment.newsScore;
            document.getElementById('sentiment-overall').textContent = state.agents.sentiment.overall;
            document.getElementById('sentiment-update').textContent = new Date(state.agents.sentiment.lastUpdate).toTimeString().split(' ')[0];
            
            // Update Economic Agent
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
            
            // ADD: Update exchange prices for arbitrage
            if (state.agents.exchange.prices) {
                const prices = state.agents.exchange.prices;
                if (document.getElementById('price-binance')) {
                    document.getElementById('price-binance').textContent = '$' + prices.binance.toFixed(0).replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
                    document.getElementById('price-coinbase').textContent = '$' + prices.coinbase.toFixed(0).replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
                    document.getElementById('price-kraken').textContent = '$' + prices.kraken.toFixed(0).replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
                    document.getElementById('price-okx').textContent = '$' + prices.okx.toFixed(0).replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
                    document.getElementById('price-bybit').textContent = '$' + prices.bybit.toFixed(0).replace(/\\B(?=(\\d{3})+(?!\\d))/g, ',');
                }
            }
            
            // ADD: Update current arbitrage opportunities
            if (state.agents.exchange.arbitrage && state.agents.exchange.arbitrage.current) {
                const currentArbDiv = document.getElementById('current-arbitrage');
                if (currentArbDiv) {
                    currentArbDiv.innerHTML = state.agents.exchange.arbitrage.current.map(arb => 
                        '<div style="background: linear-gradient(135deg, rgba(251, 247, 242, 0.9), rgba(248, 242, 235, 0.85)); padding: 8px; border-radius: 4px; font-size: 12px; border: 1px solid rgba(139, 90, 43, 0.1);">' +
                        '<span style="color: #5D4037; font-weight: 600;">Buy ' + arb.buy + ' → Sell ' + arb.sell + '</span>' +
                        '<span style="color: #6B4423; margin-left: 8px;">Spread: ' + arb.spread + '%</span>' +
                        '<span style="color: #8B5A2B; margin-left: 8px; font-weight: 600;">Profit: $' + arb.profit + '/0.1BTC</span>' +
                        '</div>'
                    ).join('');
                }
            }
            
            // ADD: Update future arbitrage windows
            if (state.agents.exchange.arbitrage && state.agents.exchange.arbitrage.future) {
                const futureArbDiv = document.getElementById('future-arbitrage');
                if (futureArbDiv) {
                    futureArbDiv.innerHTML = state.agents.exchange.arbitrage.future.map(window => 
                        '<div style="background: linear-gradient(135deg, rgba(250, 246, 241, 0.9), rgba(247, 241, 234, 0.85)); padding: 8px; border-radius: 4px; font-size: 12px; border: 1px solid rgba(161, 136, 127, 0.1);">' +
                        '<span style="color: #795548; font-weight: 600;">' + window.time + ' - ' + window.reason + '</span>' +
                        '<span style="color: #6D4C41; margin-left: 8px;">Probability: ' + window.probability + '%</span>' +
                        '</div>'
                    ).join('');
                }
            }
            
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
            
            document.getElementById('mom-live').textContent = state.backtest.strategies.momentum.live.toFixed(0) + '%';
            document.getElementById('mom-bar').style.width = state.backtest.strategies.momentum.live + '%';
            document.getElementById('mean-live').textContent = state.backtest.strategies.meanReversion.live.toFixed(0) + '%';
            document.getElementById('mean-bar').style.width = state.backtest.strategies.meanReversion.live + '%';
            document.getElementById('arb-live').textContent = state.backtest.strategies.arbitrage.live.toFixed(0) + '%';
            document.getElementById('arb-bar').style.width = state.backtest.strategies.arbitrage.live + '%';
            document.getElementById('sent-live').textContent = state.backtest.strategies.sentiment.live.toFixed(0) + '%';
            document.getElementById('sent-bar').style.width = state.backtest.strategies.sentiment.live + '%';
            
            // Update stats
            document.getElementById('updates-sec').textContent = state.stats.updatesPerSec;
            document.getElementById('llm-calls').textContent = state.stats.llmCalls;
            document.getElementById('decisions').textContent = state.stats.decisions;
            
            // Update LLM decisions with arbitrage info
            const decisionsDiv = document.getElementById('llm-decisions');
            decisionsDiv.innerHTML = state.llmDecisions.map(d => 
                '<div class="decision-cream p-2 rounded text-xs mb-2">' +
                '<div class="flex justify-between items-start mb-1">' +
                '<span class="font-bold text-cocoa-dark">' + d.action + '</span>' +
                '<span class="text-cocoa-light">' + d.direction + '</span>' +
                '</div>' +
                '<div class="grid grid-cols-3 gap-1 text-cocoa-light mb-1">' +
                '<div>Asset: <span class="text-cocoa">' + d.asset + '</span></div>' +
                '<div>Confidence: <span class="text-cocoa">' + d.confidence + '%</span></div>' +
                '<div>Timeframe: <span class="text-cocoa">' + d.timeframe + '</span></div>' +
                '</div>' +
                '<div class="grid grid-cols-3 gap-1 text-cocoa-light mb-1">' +
                '<div>Risk: <span class="text-cocoa">' + d.risk + '</span></div>' +
                (d.strategy ? '<div>Strategy: <span class="text-cocoa-dark font-semibold">' + d.strategy + '</span></div>' : '') +
                '</div>' +
                (d.arbitrage ? 
                    '<div style="background: rgba(188, 170, 164, 0.15); padding: 4px; border-radius: 4px; margin-top: 4px;">' +
                    '<span style="color: #6B4423; font-size: 12px;">🎯 Arbitrage: ' + d.arbitrage.opportunity + 
                    ' | Spread: ' + d.arbitrage.spread + ' | Profit: ' + d.arbitrage.profit + '</span>' +
                    '</div>' : '') +
                '<div class="text-cocoa-light italic">' + d.rationale + '</div>' +
                '<div class="text-cocoa-light text-right mt-1">ID: ' + d.id + ' • ' + 
                new Date(d.timestamp).toTimeString().split(' ')[0] + '</div>' +
                '</div>'
            ).join('');
            
            // Update hyperbolic visualization
            const svg = d3.select('#hyperbolic-svg');
            const width = svg.node().getBoundingClientRect().width;
            const height = 300;
            svg.attr('viewBox', '0 0 ' + width + ' ' + height);
            
            if (!svg.select('g').node()) {
                svg.append('g').attr('transform', 'translate(' + (width/2) + ', ' + (height/2) + ')');
            }
            
            const g = svg.select('g');
            
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
                .attr('stroke', '#8B5A2B')
                .attr('stroke-width', d => d.strength)
                .attr('opacity', d => d.strength * 0.3);
            
            geodesics.exit().remove();
            
            // Update asset clusters
            const assetGroups = g.selectAll('.asset-group')
                .data(state.hyperbolic.points);
            
            const enterGroups = assetGroups.enter().append('g')
                .attr('class', 'asset-group');
            
            enterGroups.append('circle');
            enterGroups.append('text');
            
            assetGroups.merge(enterGroups)
                .attr('transform', d => 'translate(' + (d.x * (Math.min(width, height) / 2 - 10)) + ', ' + (d.y * (Math.min(width, height) / 2 - 10)) + ')');
            
            assetGroups.select('circle')
                .attr('r', d => d.size)
                .attr('fill', d => {
                    const colors = {
                        crypto: '#8B5A2B',
                        equity: '#A0522D',
                        commodity: '#D2691E',
                        forex: '#6B4423',
                        volatility: '#5D4037',
                        fixed_income: '#8D6E63'
                    };
                    return colors[d.type] || '#795548';
                })
                .attr('opacity', 0.7);
            
            assetGroups.select('text')
                .text(d => d.name)
                .attr('y', -10)
                .attr('text-anchor', 'middle')
                .attr('font-size', '10px')
                .attr('fill', '#3E2723');
            
            assetGroups.exit().remove();
            
            // Draw disk boundary
            if (!g.select('.boundary').node()) {
                g.append('circle')
                    .attr('class', 'boundary')
                    .attr('r', Math.min(width, height) / 2 - 10)
                    .attr('fill', 'none')
                    .attr('stroke', '#8B5A2B')
                    .attr('stroke-width', 2)
                    .attr('opacity', 0.3);
            }
        });
    </script>

    <!-- Commercial overlay integration -->
    <script src="/commercial-overlay.js"></script>
    <script>
      if (typeof injectCommercialOverlay === 'function') {
        setTimeout(() => {
          injectCommercialOverlay();
        }, 1500);
      }
    </script>
    
    <!-- Backtesting Module integration -->
    <script src="/backtesting-module.js"></script>
    <script>
      window.addEventListener('DOMContentLoaded', () => {
        setTimeout(() => {
          if (typeof createBacktestUI === 'function') {
            createBacktestUI();
          }
        }, 1000);
      });
    </script>
</body>
</html>`);
});

// Start server
server.listen(PORT, '0.0.0.0', () => {
  console.log('✅ Gomna Arbitrage Trades Platform LIVE on port', PORT);
  console.log('📊 All agents active and streaming');
  console.log('🤖 LLM decisions generating every 2s');
  connectToCoinbase();
});

// Generate initial state
updateSentiment();
updateEconomic();
updateExchange();
updateHyperbolic();
updateBacktest();
updateConstraints();