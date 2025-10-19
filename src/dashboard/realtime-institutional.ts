import express from 'express';
import http from 'http';
import { Server as SocketIOServer } from 'socket.io';
import Logger from '../utils/logger';
import { MultiFrequencyTradingEngine } from '../trading/MultiFrequencyTradingEngine';
import { AdvancedTradingStrategies } from '../strategies/AdvancedTradingStrategies';

const app = express();
const server = http.createServer(app);
const io = new SocketIOServer(server, {
  cors: { origin: "*", methods: ["GET", "POST"] }
});

const logger = Logger.getInstance('RealtimeInstitutionalDashboard');
const PORT = process.env.DASHBOARD_PORT || 3001;

// Initialize trading engines
const multiFreqEngine = new MultiFrequencyTradingEngine();
const advancedStrategies = new AdvancedTradingStrategies();

// Start all engines
async function startEngines() {
  await multiFreqEngine.start();
  await advancedStrategies.start();
  logger.info('All trading engines started successfully');
}

// Real-time data streaming
multiFreqEngine.on('hft_update', (data) => {
  io.emit('hft_update', data);
});

multiFreqEngine.on('hft_trade', (trade) => {
  io.emit('hft_trade', trade);
});

multiFreqEngine.on('medium_freq_update', (data) => {
  io.emit('medium_freq_update', data);
});

multiFreqEngine.on('low_freq_update', (data) => {
  io.emit('low_freq_update', data);
});

multiFreqEngine.on('tick', (data) => {
  io.emit('market_tick', data);
});

advancedStrategies.on('barra_update', (data) => {
  io.emit('barra_factors', data);
});

advancedStrategies.on('ml_prediction', (data) => {
  io.emit('ml_predictions', data);
});

advancedStrategies.on('pairs_update', (data) => {
  io.emit('pairs_trading', data);
});

advancedStrategies.on('portfolio_optimized', (data) => {
  io.emit('portfolio_update', data);
});

app.use(express.static('public'));
app.use(express.json());

// API endpoints
app.get('/api/hft/snapshot', (req, res) => {
  res.json(multiFreqEngine.getHFTData());
});

app.get('/api/medium/snapshot', (req, res) => {
  res.json(multiFreqEngine.getMediumFreqData());
});

app.get('/api/low/snapshot', (req, res) => {
  res.json(multiFreqEngine.getLowFreqData());
});

// Main dashboard HTML with real-time updates
app.get('/', (req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum AI Capital - Real-Time Multi-Frequency Trading</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">
    <script src="/socket.io/socket.io.js"></script>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300;400;500;600;700&display=swap');
        body { font-family: 'Inter', sans-serif; }
        .mono { font-family: 'JetBrains Mono', monospace; }
        
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        @keyframes flash { 0%, 100% { background-color: transparent; } 50% { background-color: rgba(34, 197, 94, 0.1); } }
        @keyframes slideUp { from { transform: translateY(10px); opacity: 0; } to { transform: translateY(0); opacity: 1; } }
        
        .pulse { animation: pulse 2s infinite; }
        .flash { animation: flash 0.5s; }
        .slide-up { animation: slideUp 0.3s ease-out; }
        
        .update-flash {
            animation: flash 0.3s;
        }
        
        .ticker {
            background: linear-gradient(90deg, #1e293b 0%, #0f172a 100%);
            overflow: hidden;
        }
        
        .live-indicator {
            position: relative;
            display: inline-block;
        }
        
        .live-indicator::before {
            content: '';
            position: absolute;
            top: 50%;
            left: -15px;
            transform: translateY(-50%);
            width: 8px;
            height: 8px;
            background: #10b981;
            border-radius: 50%;
            animation: pulse 1s infinite;
        }
        
        .timestamp {
            font-size: 10px;
            color: #64748b;
            font-family: 'JetBrains Mono', monospace;
        }
        
        .frequency-badge {
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
            text-transform: uppercase;
        }
        
        .freq-high { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
        .freq-medium { background: rgba(251, 146, 60, 0.2); color: #fb923c; }
        .freq-low { background: rgba(59, 130, 246, 0.2); color: #3b82f6; }
    </style>
</head>
<body class="bg-gray-950 text-gray-100">
    <!-- Header with Live Clock -->
    <div class="bg-gray-900 border-b border-gray-800 px-6 py-3">
        <div class="flex items-center justify-between">
            <div class="flex items-center space-x-4">
                <div class="relative">
                    <div class="w-12 h-12 bg-gradient-to-br from-green-500 to-blue-600 rounded-lg flex items-center justify-center">
                        <i class="fas fa-chart-line text-white text-xl"></i>
                    </div>
                    <div class="absolute -top-1 -right-1 w-3 h-3 bg-green-500 rounded-full pulse"></div>
                </div>
                <div>
                    <h1 class="text-2xl font-bold bg-gradient-to-r from-green-400 to-blue-400 bg-clip-text text-transparent">
                        Quantum AI Capital
                    </h1>
                    <p class="text-xs text-gray-400">Multi-Frequency Algorithmic Trading Platform</p>
                </div>
            </div>
            <div class="flex items-center space-x-6">
                <div class="text-center">
                    <div class="text-2xl font-bold mono live-indicator" id="live-clock">--:--:--</div>
                    <div class="timestamp" id="microseconds">μs: ------</div>
                </div>
                <div class="flex items-center space-x-2">
                    <div class="w-2 h-2 bg-green-500 rounded-full pulse"></div>
                    <span class="text-sm font-bold">LIVE TRADING</span>
                </div>
            </div>
        </div>
    </div>

    <!-- Frequency Status Bar -->
    <div class="bg-gray-900/50 px-6 py-2 border-b border-gray-800">
        <div class="flex items-center justify-between">
            <div class="flex space-x-4">
                <div class="flex items-center space-x-2">
                    <span class="frequency-badge freq-high">HFT</span>
                    <span class="text-xs">Updates: <span class="mono" id="hft-updates">0</span>/sec</span>
                    <span class="timestamp" id="hft-timestamp">--</span>
                </div>
                <div class="flex items-center space-x-2">
                    <span class="frequency-badge freq-medium">MED</span>
                    <span class="text-xs">Updates: <span class="mono" id="med-updates">0</span>/sec</span>
                    <span class="timestamp" id="med-timestamp">--</span>
                </div>
                <div class="flex items-center space-x-2">
                    <span class="frequency-badge freq-low">LOW</span>
                    <span class="text-xs">Updates: <span class="mono" id="low-updates">0</span>/30s</span>
                    <span class="timestamp" id="low-timestamp">--</span>
                </div>
            </div>
            <div>
                <span class="text-xs text-gray-500">Total Ticks:</span>
                <span class="mono text-green-400" id="total-ticks">0</span>
            </div>
        </div>
    </div>

    <div class="flex h-screen">
        <!-- Left Panel - High Frequency Trading -->
        <div class="w-1/3 p-4 space-y-4 overflow-y-auto border-r border-gray-800">
            <h2 class="text-lg font-bold text-gray-300 mb-2">
                <span class="frequency-badge freq-high">HFT</span>
                High-Frequency Trading (10ms)
            </h2>

            <!-- Live Order Book -->
            <div class="bg-gray-900/50 p-4 rounded-lg">
                <h3 class="text-sm font-semibold text-green-400 mb-3 flex justify-between">
                    <span>Live Order Book - BTC</span>
                    <span class="timestamp" id="ob-timestamp">--</span>
                </h3>
                <div class="grid grid-cols-2 gap-4">
                    <div>
                        <div class="text-xs text-gray-500 mb-1">BIDS</div>
                        <div id="bid-levels" class="space-y-1 text-xs mono">
                            <!-- Bid levels will be populated here -->
                        </div>
                    </div>
                    <div>
                        <div class="text-xs text-gray-500 mb-1">ASKS</div>
                        <div id="ask-levels" class="space-y-1 text-xs mono">
                            <!-- Ask levels will be populated here -->
                        </div>
                    </div>
                </div>
                <div class="mt-3 pt-3 border-t border-gray-700">
                    <div class="grid grid-cols-2 gap-2 text-xs">
                        <div>
                            <span class="text-gray-500">Spread:</span>
                            <span class="mono text-yellow-400" id="spread-bps">0.0</span> bps
                        </div>
                        <div>
                            <span class="text-gray-500">Imbalance:</span>
                            <span class="mono" id="imbalance">0.0%</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Recent HFT Trades -->
            <div class="bg-gray-900/50 p-4 rounded-lg">
                <h3 class="text-sm font-semibold text-yellow-400 mb-3">
                    Recent HFT Executions
                </h3>
                <div id="hft-trades" class="space-y-1 text-xs mono">
                    <!-- Recent trades will be populated here -->
                </div>
            </div>

            <!-- Microstructure Metrics -->
            <div class="bg-gray-900/50 p-4 rounded-lg">
                <h3 class="text-sm font-semibold text-purple-400 mb-3">
                    Market Microstructure
                </h3>
                <div class="space-y-2 text-xs">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Effective Spread:</span>
                        <span class="mono" id="eff-spread">0.0 bps</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Realized Vol:</span>
                        <span class="mono" id="realized-vol">0.0%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Flow Toxicity:</span>
                        <span class="mono" id="toxicity">0.0</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Market Impact:</span>
                        <span class="mono" id="impact">0.0000</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Liquidity Score:</span>
                        <span class="mono text-green-400" id="liquidity">0</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Center Panel - Medium Frequency Trading -->
        <div class="w-1/3 p-4 space-y-4 overflow-y-auto border-r border-gray-800">
            <h2 class="text-lg font-bold text-gray-300 mb-2">
                <span class="frequency-badge freq-medium">MED</span>
                Medium-Frequency Trading (1s)
            </h2>

            <!-- Live Price Ticker -->
            <div class="bg-gray-900/50 p-4 rounded-lg">
                <h3 class="text-sm font-semibold text-blue-400 mb-3">
                    Live Price Feed
                </h3>
                <div id="price-ticker" class="space-y-2">
                    <!-- Price tickers will be populated here -->
                </div>
            </div>

            <!-- Technical Indicators -->
            <div class="bg-gray-900/50 p-4 rounded-lg">
                <h3 class="text-sm font-semibold text-orange-400 mb-3">
                    Technical Indicators - BTC
                </h3>
                <div class="grid grid-cols-2 gap-2 text-xs">
                    <div>
                        <span class="text-gray-500">RSI:</span>
                        <span class="mono" id="rsi">50.0</span>
                    </div>
                    <div>
                        <span class="text-gray-500">MACD:</span>
                        <span class="mono" id="macd">0.00</span>
                    </div>
                    <div>
                        <span class="text-gray-500">VWAP:</span>
                        <span class="mono" id="vwap">$0</span>
                    </div>
                    <div>
                        <span class="text-gray-500">EMA20:</span>
                        <span class="mono" id="ema20">$0</span>
                    </div>
                    <div>
                        <span class="text-gray-500">Volume:</span>
                        <span class="mono" id="volume24h">0</span>
                    </div>
                    <div>
                        <span class="text-gray-500">Order Flow:</span>
                        <span class="mono" id="orderflow">0</span>
                    </div>
                </div>
                <div class="mt-3 pt-3 border-t border-gray-700">
                    <div class="text-center">
                        <span class="text-xs text-gray-500">Signal:</span>
                        <span class="font-bold text-lg" id="med-signal">HOLD</span>
                    </div>
                </div>
            </div>

            <!-- Active Medium-Freq Trades -->
            <div class="bg-gray-900/50 p-4 rounded-lg">
                <h3 class="text-sm font-semibold text-cyan-400 mb-3">
                    Active Signals
                </h3>
                <div id="medium-signals" class="space-y-2">
                    <!-- Signals will be populated here -->
                </div>
            </div>
        </div>

        <!-- Right Panel - Low Frequency & Portfolio -->
        <div class="w-1/3 p-4 space-y-4 overflow-y-auto">
            <h2 class="text-lg font-bold text-gray-300 mb-2">
                <span class="frequency-badge freq-low">LOW</span>
                Low-Frequency Trading (30s)
            </h2>

            <!-- Portfolio Allocation -->
            <div class="bg-gray-900/50 p-4 rounded-lg">
                <h3 class="text-sm font-semibold text-indigo-400 mb-3 flex justify-between">
                    <span>Portfolio Allocation</span>
                    <span class="timestamp" id="portfolio-timestamp">--</span>
                </h3>
                <div id="portfolio-allocation" class="space-y-2">
                    <!-- Allocations will be populated here -->
                </div>
                <div class="mt-3 pt-3 border-t border-gray-700">
                    <div class="grid grid-cols-2 gap-2 text-xs">
                        <div>
                            <span class="text-gray-500">Market Regime:</span>
                            <span class="font-bold" id="market-regime">--</span>
                        </div>
                        <div>
                            <span class="text-gray-500">Vol Regime:</span>
                            <span class="font-bold" id="vol-regime">--</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Swing Trades -->
            <div class="bg-gray-900/50 p-4 rounded-lg">
                <h3 class="text-sm font-semibold text-purple-400 mb-3">
                    Swing Trade Opportunities
                </h3>
                <div id="swing-trades" class="space-y-2">
                    <!-- Swing trades will be populated here -->
                </div>
            </div>

            <!-- Correlation Matrix -->
            <div class="bg-gray-900/50 p-4 rounded-lg">
                <h3 class="text-sm font-semibold text-green-400 mb-3">
                    Correlation Matrix
                </h3>
                <div id="correlation-matrix" class="space-y-1 text-xs">
                    <!-- Correlations will be populated here -->
                </div>
            </div>

            <!-- Performance Summary -->
            <div class="bg-gradient-to-br from-green-900/30 to-blue-900/30 p-4 rounded-lg border border-green-500/30">
                <h3 class="text-sm font-semibold text-white mb-3">
                    Live Performance Metrics
                </h3>
                <div class="grid grid-cols-2 gap-3 text-xs">
                    <div class="text-center p-2 bg-gray-900/50 rounded">
                        <div class="text-gray-500">HFT Trades</div>
                        <div class="text-lg font-bold text-green-400" id="hft-count">0</div>
                    </div>
                    <div class="text-center p-2 bg-gray-900/50 rounded">
                        <div class="text-gray-500">Med Signals</div>
                        <div class="text-lg font-bold text-yellow-400" id="med-count">0</div>
                    </div>
                    <div class="text-center p-2 bg-gray-900/50 rounded">
                        <div class="text-gray-500">Swing Trades</div>
                        <div class="text-lg font-bold text-blue-400" id="swing-count">0</div>
                    </div>
                    <div class="text-center p-2 bg-gray-900/50 rounded">
                        <div class="text-gray-500">Total Volume</div>
                        <div class="text-lg font-bold" id="total-volume">$0</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        
        // Update counters
        let hftUpdateCount = 0;
        let medUpdateCount = 0;
        let lowUpdateCount = 0;
        let totalTicks = 0;
        let hftTradeCount = 0;
        let totalVolume = 0;
        
        // Live clock with microseconds
        function updateClock() {
            const now = new Date();
            const hours = String(now.getHours()).padStart(2, '0');
            const minutes = String(now.getMinutes()).padStart(2, '0');
            const seconds = String(now.getSeconds()).padStart(2, '0');
            const ms = String(now.getMilliseconds()).padStart(3, '0');
            
            document.getElementById('live-clock').textContent = \`\${hours}:\${minutes}:\${seconds}.\${ms}\`;
            document.getElementById('microseconds').textContent = \`μs: \${Date.now()}\${ms}\`;
        }
        setInterval(updateClock, 10);
        
        // Format timestamp
        function formatTimestamp(timestamp) {
            const date = new Date(timestamp);
            return date.toTimeString().split(' ')[0] + '.' + String(date.getMilliseconds()).padStart(3, '0');
        }
        
        // HFT Updates
        socket.on('hft_update', (data) => {
            hftUpdateCount++;
            document.getElementById('hft-updates').textContent = Math.min(100, hftUpdateCount);
            document.getElementById('hft-timestamp').textContent = formatTimestamp(data.timestamp);
            
            // Update order book
            if (data.orderBooks && data.orderBooks.BTC) {
                const ob = data.orderBooks.BTC;
                document.getElementById('ob-timestamp').textContent = formatTimestamp(ob.timestamp);
                document.getElementById('spread-bps').textContent = ob.spread.toFixed(2);
                document.getElementById('imbalance').textContent = (ob.imbalance * 100).toFixed(1) + '%';
                
                // Update bid levels
                const bidLevels = document.getElementById('bid-levels');
                if (ob.depth && ob.depth.bids) {
                    bidLevels.innerHTML = ob.depth.bids.slice(0, 5).map(level => 
                        \`<div class="flex justify-between text-green-400">
                            <span>$\${level.price.toFixed(2)}</span>
                            <span>\${level.volume.toFixed(3)}</span>
                        </div>\`
                    ).join('');
                }
                
                // Update ask levels
                const askLevels = document.getElementById('ask-levels');
                if (ob.depth && ob.depth.asks) {
                    askLevels.innerHTML = ob.depth.asks.slice(0, 5).map(level => 
                        \`<div class="flex justify-between text-red-400">
                            <span>$\${level.price.toFixed(2)}</span>
                            <span>\${level.volume.toFixed(3)}</span>
                        </div>\`
                    ).join('');
                }
            }
            
            // Update microstructure
            if (data.microstructure) {
                const ms = data.microstructure;
                document.getElementById('eff-spread').textContent = ms.effectiveSpread.toFixed(2) + ' bps';
                document.getElementById('realized-vol').textContent = (ms.realizedVolatility * 100).toFixed(1) + '%';
                document.getElementById('toxicity').textContent = ms.orderFlowToxicity.toFixed(3);
                document.getElementById('impact').textContent = ms.marketImpact.toFixed(4);
                document.getElementById('liquidity').textContent = Math.round(ms.liquidityScore);
            }
            
            // Flash update indicator
            document.getElementById('ob-timestamp').classList.add('update-flash');
            setTimeout(() => document.getElementById('ob-timestamp').classList.remove('update-flash'), 300);
        });
        
        // HFT Trades
        socket.on('hft_trade', (trade) => {
            hftTradeCount++;
            document.getElementById('hft-count').textContent = hftTradeCount;
            totalVolume += trade.volume * trade.price;
            document.getElementById('total-volume').textContent = '$' + (totalVolume / 1000000).toFixed(2) + 'M';
            
            const tradesDiv = document.getElementById('hft-trades');
            const tradeEl = document.createElement('div');
            tradeEl.className = \`flex justify-between \${trade.side === 'BUY' ? 'text-green-400' : 'text-red-400'} slide-up\`;
            tradeEl.innerHTML = \`
                <span>\${trade.symbol}</span>
                <span>$\${trade.price.toFixed(2)}</span>
                <span>\${trade.volume.toFixed(3)}</span>
                <span class="timestamp">\${formatTimestamp(trade.timestamp)}</span>
            \`;
            
            tradesDiv.insertBefore(tradeEl, tradesDiv.firstChild);
            if (tradesDiv.children.length > 10) tradesDiv.removeChild(tradesDiv.lastChild);
        });
        
        // Market Ticks
        socket.on('market_tick', (data) => {
            totalTicks++;
            document.getElementById('total-ticks').textContent = totalTicks;
            
            // Update price ticker
            const tickerDiv = document.getElementById('price-ticker');
            tickerDiv.innerHTML = Object.entries(data.prices).map(([symbol, price]) => 
                \`<div class="flex justify-between items-center p-2 bg-gray-800/50 rounded">
                    <span class="font-semibold">\${symbol}</span>
                    <span class="mono text-lg">$\${price.toFixed(2)}</span>
                    <span class="timestamp">\${formatTimestamp(data.timestamp)}</span>
                </div>\`
            ).join('');
        });
        
        // Medium Frequency Updates
        socket.on('medium_freq_update', (data) => {
            medUpdateCount++;
            document.getElementById('med-updates').textContent = medUpdateCount;
            document.getElementById('med-timestamp').textContent = formatTimestamp(data.timestamp);
            document.getElementById('med-count').textContent = Object.keys(data.signals || {}).length;
            
            // Update indicators for BTC
            if (data.indicators && data.indicators.BTC) {
                const ind = data.indicators.BTC;
                document.getElementById('rsi').textContent = ind.rsi.toFixed(1);
                document.getElementById('macd').textContent = ind.macd.toFixed(3);
                document.getElementById('vwap').textContent = '$' + ind.vwap.toFixed(2);
                document.getElementById('ema20').textContent = '$' + ind.ema20.toFixed(2);
                document.getElementById('volume24h').textContent = (ind.volume24h / 1000000).toFixed(1) + 'M';
                document.getElementById('orderflow').textContent = (ind.orderFlow / 1000).toFixed(0) + 'K';
            }
            
            // Update signals
            if (data.signals) {
                const signalsDiv = document.getElementById('medium-signals');
                signalsDiv.innerHTML = Object.entries(data.signals).slice(0, 3).map(([symbol, signal]) => 
                    \`<div class="p-2 bg-gray-800/50 rounded">
                        <div class="flex justify-between items-center mb-1">
                            <span class="font-semibold">\${symbol}</span>
                            <span class="text-sm font-bold \${signal.action.includes('BUY') ? 'text-green-400' : signal.action.includes('SELL') ? 'text-red-400' : 'text-gray-400'}">
                                \${signal.action}
                            </span>
                        </div>
                        <div class="text-xs text-gray-400">
                            <div>Conf: \${(signal.confidence * 100).toFixed(0)}%</div>
                            <div>R:R: \${signal.riskReward.toFixed(2)}</div>
                        </div>
                    </div>\`
                ).join('');
                
                // Update main signal for BTC
                if (data.signals.BTC) {
                    const signal = data.signals.BTC.action;
                    document.getElementById('med-signal').textContent = signal;
                    document.getElementById('med-signal').className = 
                        \`font-bold text-lg \${signal.includes('BUY') ? 'text-green-400' : signal.includes('SELL') ? 'text-red-400' : 'text-gray-400'}\`;
                }
            }
        });
        
        // Low Frequency Updates
        socket.on('low_freq_update', (data) => {
            lowUpdateCount++;
            document.getElementById('low-updates').textContent = lowUpdateCount;
            document.getElementById('low-timestamp').textContent = formatTimestamp(data.timestamp);
            
            if (data.portfolio) {
                const portfolio = data.portfolio;
                document.getElementById('portfolio-timestamp').textContent = formatTimestamp(portfolio.timestamp);
                document.getElementById('market-regime').textContent = portfolio.marketRegime;
                document.getElementById('vol-regime').textContent = portfolio.volatilityRegime;
                
                // Update allocations
                const allocDiv = document.getElementById('portfolio-allocation');
                allocDiv.innerHTML = Object.entries(portfolio.targetAllocations).map(([symbol, target]) => {
                    const current = portfolio.currentAllocations[symbol];
                    const diff = target - current;
                    return \`<div class="flex justify-between items-center">
                        <span class="text-sm">\${symbol}</span>
                        <div class="text-right">
                            <div class="mono">\${(target * 100).toFixed(1)}%</div>
                            <div class="text-xs \${diff > 0 ? 'text-green-400' : 'text-red-400'}">
                                \${diff > 0 ? '+' : ''}\${(diff * 100).toFixed(1)}%
                            </div>
                        </div>
                    </div>\`;
                }).join('');
                
                // Update correlations
                if (portfolio.correlationMatrix) {
                    const corrDiv = document.getElementById('correlation-matrix');
                    corrDiv.innerHTML = Object.entries(portfolio.correlationMatrix).map(([pair, corr]) => 
                        \`<div class="flex justify-between">
                            <span>\${pair}:</span>
                            <span class="mono">\${corr.toFixed(3)}</span>
                        </div>\`
                    ).join('');
                }
            }
            
            // Update swing trades
            if (data.swingTrades) {
                document.getElementById('swing-count').textContent = data.swingTrades.length;
                const swingDiv = document.getElementById('swing-trades');
                swingDiv.innerHTML = data.swingTrades.slice(0, 3).map(trade => 
                    \`<div class="p-2 bg-gray-800/50 rounded">
                        <div class="flex justify-between items-center mb-1">
                            <span class="font-semibold">\${trade.symbol}</span>
                            <span class="text-sm font-bold \${trade.direction === 'LONG' ? 'text-green-400' : 'text-red-400'}">
                                \${trade.direction}
                            </span>
                        </div>
                        <div class="text-xs text-gray-400">
                            <div>Expected: \${trade.expectedMove}</div>
                            <div>Confidence: \${(trade.confidence * 100).toFixed(0)}%</div>
                            <div>Period: \${trade.holdingPeriod}</div>
                        </div>
                    </div>\`
                ).join('');
            }
        });
        
        // Reset counters every second for rate calculation
        setInterval(() => {
            document.getElementById('hft-updates').textContent = Math.min(100, hftUpdateCount);
            hftUpdateCount = 0;
        }, 1000);
    </script>
</body>
</html>`);
});

// WebSocket connection handler
io.on('connection', (socket) => {
  logger.info('Client connected to real-time feed');
  
  socket.on('disconnect', () => {
    logger.info('Client disconnected from real-time feed');
  });
});

// Start the server
startEngines().then(() => {
  server.listen(PORT, '0.0.0.0', () => {
    logger.info(`Real-Time Multi-Frequency Trading Dashboard running on port ${PORT}`);
    console.log(`
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║   QUANTUM AI CAPITAL - REAL-TIME MULTI-FREQUENCY TRADING    ║
    ║                                                              ║
    ║   🚀 Dashboard: http://localhost:${PORT}                        ║
    ║                                                              ║
    ║   Trading Frequencies:                                       ║
    ║   • HIGH: 10ms updates (100/sec) - HFT & Market Making     ║
    ║   • MEDIUM: 1s updates - Technical Signals                  ║
    ║   • LOW: 30s updates - Portfolio & Swing Trades            ║
    ║                                                              ║
    ║   Features:                                                 ║
    ║   • Live microsecond timestamps                            ║
    ║   • Real-time order book with 10 levels                    ║
    ║   • HFT trade execution feed                               ║
    ║   • Technical indicators with signals                       ║
    ║   • Portfolio rebalancing alerts                           ║
    ║   • Multi-asset correlation tracking                       ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    `);
  });
});

export default server;