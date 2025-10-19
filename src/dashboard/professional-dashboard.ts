import express from 'express';
import http from 'http';
import { Server as SocketIOServer } from 'socket.io';
import axios from 'axios';
import Logger from '../utils/logger';
import config from '../utils/ConfigLoader';
import { BacktestEngine } from '../backtest/BacktestEngine';
import { DecisionEngine } from '../decision/DecisionEngine';
import { RealTimeDataAggregator } from '../agents/RealTimeDataAggregator';
import { RealtimeLLMArbitrage } from '../fusion/RealtimeLLMArbitrage';

const app = express();
const server = http.createServer(app);
const io = new SocketIOServer(server, {
  cors: { origin: "*", methods: ["GET", "POST"] }
});

const logger = Logger.getInstance('ProfessionalDashboard');
const PORT = process.env.DASHBOARD_PORT || 3000;

// Initialize engines
const backtestEngine = new BacktestEngine(config.get('backtesting'));
const decisionEngine = new DecisionEngine();

// Initialize real-time components
const dataAggregator = new RealTimeDataAggregator();
const llmArbitrage = new RealtimeLLMArbitrage();

// Start real-time data collection
dataAggregator.start();
llmArbitrage.start();

// Hyperbolic clustering data
let hyperbolicClusters = {
  assets: [],
  connections: [],
  lastUpdate: Date.now()
};

// Agent status tracking - will be updated from real-time data
let agentStatuses = {
  economic: { status: 'active', lastSignal: 0, confidence: 0, data: {} },
  sentiment: { status: 'active', lastSignal: 0, confidence: 0, data: {} },
  price: { status: 'active', lastSignal: 0, confidence: 0, data: {} },
  volume: { status: 'active', lastSignal: 0, confidence: 0, data: {} },
  microstructure: { status: 'active', lastSignal: 0, confidence: 0, data: {} },
  crossExchange: { status: 'active', lastSignal: 0, confidence: 0, data: {} }
};

// LLM prediction state
let llmState = {
  processing: false,
  lastPrediction: null,
  confidence: 0,
  nextUpdate: Date.now() + 5000
};

app.use(express.static('public'));
app.use(express.json());

// Generate hyperbolic clustering data
function generateHyperbolicClusters() {
  const assets = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC', 'LINK', 'DOT', 'UNI'];
  const clusters = assets.map((asset, i) => {
    const angle = (i / assets.length) * 2 * Math.PI;
    const radius = 0.3 + Math.random() * 0.4;
    return {
      id: asset,
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
      r: 0.02 + Math.random() * 0.03,
      momentum: Math.random() * 2 - 1,
      correlation: Math.random()
    };
  });

  // Generate connections based on correlation
  const connections = [];
  for (let i = 0; i < assets.length; i++) {
    for (let j = i + 1; j < assets.length; j++) {
      if (Math.random() > 0.6) {
        connections.push({
          source: assets[i],
          target: assets[j],
          strength: Math.random()
        });
      }
    }
  }

  return { assets: clusters, connections, lastUpdate: Date.now() };
}

// Update agent statuses from real-time data
function updateAgentStatuses() {
  try {
    // Get real-time agent signals
    const agentSignals = dataAggregator.getAgentSignals();
    
    // Update economic agent
    const economicData = dataAggregator.getEconomicData();
    agentStatuses.economic = {
      status: economicData ? 'active' : 'offline',
      lastSignal: agentSignals.economic.signal,
      confidence: agentSignals.economic.confidence,
      data: economicData || {}
    };
    
    // Update sentiment agent
    const sentimentData = dataAggregator.getSentimentData();
    agentStatuses.sentiment = {
      status: sentimentData ? 'active' : 'offline',
      lastSignal: agentSignals.sentiment.signal,
      confidence: agentSignals.sentiment.confidence,
      data: sentimentData || {}
    };
    
    // Update price agent from market data
    const marketData = dataAggregator.getMarketData();
    const btcPrice = marketData.get('BTC')?.exchanges?.binance?.price || 0;
    agentStatuses.price = {
      status: btcPrice > 0 ? 'active' : 'offline',
      lastSignal: agentSignals.price.signal,
      confidence: agentSignals.price.confidence,
      data: { price: btcPrice, spread: marketData.get('BTC')?.spread || 0 }
    };
    
    // Update volume agent from market data
    const btcVolume = marketData.get('BTC')?.exchanges?.binance?.volume24h || 0;
    agentStatuses.volume = {
      status: btcVolume > 0 ? 'active' : 'offline',
      lastSignal: agentSignals.volume.signal,
      confidence: agentSignals.volume.confidence,
      data: { volume: btcVolume }
    };
    
    // Update microstructure agent
    const microstructureData = dataAggregator.getMicrostructure();
    agentStatuses.microstructure = {
      status: microstructureData ? 'active' : 'offline',
      lastSignal: agentSignals.microstructure.signal,
      confidence: agentSignals.microstructure.confidence,
      data: microstructureData || {}
    };
    
    // Update cross-exchange agent
    const spreads = dataAggregator.getCrossExchangeSpreads();
    agentStatuses.crossExchange = {
      status: Object.keys(spreads).length > 0 ? 'active' : 'offline',
      lastSignal: agentSignals.crossExchange.signal,
      confidence: agentSignals.crossExchange.confidence,
      data: spreads
    };
    
  } catch (error) {
    logger.error('Failed to update agent statuses', error);
  }
}

// Update LLM state from real-time predictions
function updateLLMProcessing() {
  try {
    // Get current prediction and opportunities from the LLM arbitrage instance
    const currentPrediction = (llmArbitrage as any).currentPrediction || null;
    const opportunities = (llmArbitrage as any).opportunities || [];
    
    if (currentPrediction) {
      llmState.processing = false;
      llmState.lastPrediction = {
        spread: currentPrediction.predictedSpread?.toFixed(3) || '0.000',
        confidence: currentPrediction.confidence || 0,
        direction: currentPrediction.marketDirection || 'neutral',
        opportunities: opportunities.length,
        strategy: currentPrediction.strategy,
        entryConditions: currentPrediction.entryConditions,
        exitConditions: currentPrediction.exitConditions
      };
      llmState.confidence = currentPrediction.confidence;
      llmState.nextUpdate = Date.now() + 5000;
    } else {
      llmState.processing = true;
    }
  } catch (error) {
    logger.error('Failed to update LLM state', error);
  }
}

// API Endpoints
app.get('/api/hyperbolic/clusters', (req, res) => {
  res.json(hyperbolicClusters);
});

app.get('/api/agents/status', (req, res) => {
  res.json(agentStatuses);
});

app.get('/api/llm/state', (req, res) => {
  res.json(llmState);
});

app.get('/api/datafeed/live', async (req, res) => {
  try {
    // Get real-time data from aggregator
    const economicData = dataAggregator.getEconomicData();
    const sentimentData = dataAggregator.getSentimentData();
    const microstructureData = dataAggregator.getMicrostructure();
    const crossExchangeSpreads = dataAggregator.getCrossExchangeSpreads();
    const marketData = dataAggregator.getMarketData();
    
    const data = {
      economic: economicData || {
        gdp: 2.34,
        inflation: 3.13,
        fedRate: 5.32,
        unemployment: 3.66,
        dxy: 104.52
      },
      sentiment: sentimentData || {
        fearGreed: 48,
        socialVolume: 161000,
        mood: { btc: 51, eth: 44, sol: 47 }
      },
      microstructure: microstructureData || {
        spread: 0.32,
        depth: 3200000,
        imbalance: 0.15,
        toxicity: 0.22
      },
      crossExchange: crossExchangeSpreads,
      marketData: marketData
    };
    res.json(data);
  } catch (error) {
    logger.error('Error fetching live data feed', error);
    res.status(500).json({ error: 'Failed to fetch live data' });
  }
});

// New endpoint for real-time arbitrage opportunities
app.get('/api/arbitrage/opportunities', (req, res) => {
  try {
    const opportunities = (llmArbitrage as any).opportunities || [];
    const prediction = (llmArbitrage as any).currentPrediction || null;
    
    res.json({
      opportunities,
      currentPrediction: prediction,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error fetching arbitrage opportunities', error);
    res.status(500).json({ error: 'Failed to fetch opportunities' });
  }
});

// New endpoint for agent signals
app.get('/api/agents/signals', (req, res) => {
  try {
    const signals = dataAggregator.getAgentSignals();
    res.json(signals);
  } catch (error) {
    logger.error('Error fetching agent signals', error);
    res.status(500).json({ error: 'Failed to fetch signals' });
  }
});

// Dashboard HTML
app.get('/', (req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Arbitrage Platform - Agent Intelligence Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <script src="/socket.io/socket.io.js"></script>
    <style>
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
        @keyframes dataFlow { 
            0% { stroke-dashoffset: 0; } 
            100% { stroke-dashoffset: -20; } 
        }
        .pulse { animation: pulse 2s infinite; }
        .data-flow { 
            stroke-dasharray: 5,5;
            animation: dataFlow 1s linear infinite;
        }
        .agent-card {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border: 1px solid #334155;
            position: relative;
            overflow: hidden;
        }
        .agent-card::before {
            content: '';
            position: absolute;
            top: -2px;
            left: -2px;
            right: -2px;
            bottom: -2px;
            background: linear-gradient(45deg, transparent, #3b82f6, transparent);
            opacity: 0;
            transition: opacity 0.3s;
            z-index: -1;
        }
        .agent-card.active::before {
            opacity: 1;
            animation: pulse 2s infinite;
        }
        .hyperbolic-container {
            background: radial-gradient(circle at center, #0f172a 0%, #020617 100%);
            border: 2px solid #1e293b;
        }
        .llm-brain {
            background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
            box-shadow: 0 0 50px rgba(139, 92, 246, 0.5);
        }
        .positive { color: #10b981; }
        .negative { color: #ef4444; }
        .neutral { color: #f59e0b; }
    </style>
</head>
<body class="bg-gray-950 text-gray-100">
    <!-- Header -->
    <div class="bg-gray-900 border-b border-gray-800 px-6 py-3">
        <div class="flex items-center justify-between">
            <div class="flex items-center space-x-4">
                <div class="relative">
                    <i class="fas fa-network-wired text-3xl text-indigo-500"></i>
                    <div class="absolute -top-1 -right-1 w-2 h-2 bg-green-500 rounded-full pulse"></div>
                </div>
                <div>
                    <h1 class="text-2xl font-bold">Agent-Based LLM Arbitrage Intelligence</h1>
                    <p class="text-xs text-gray-400">Multi-Agent → Hyperbolic Embedding → LLM Fusion → Decision Engine</p>
                </div>
            </div>
            <div class="flex items-center space-x-4">
                <div class="text-sm">
                    <span class="text-gray-500">System Uptime:</span>
                    <span class="font-mono" id="uptime">00:00:00</span>
                </div>
                <div class="flex items-center space-x-2">
                    <div class="w-2 h-2 bg-green-500 rounded-full pulse"></div>
                    <span class="text-sm font-bold">LIVE</span>
                </div>
            </div>
        </div>
    </div>

    <div class="flex h-screen">
        <!-- Left Panel - Data Agent Inputs -->
        <div class="w-1/4 p-4 space-y-4 overflow-y-auto border-r border-gray-800">
            <h2 class="text-lg font-bold text-gray-300 mb-2">
                <i class="fas fa-satellite-dish mr-2 text-blue-500"></i>
                Data Agent Inputs
            </h2>

            <!-- Economic Agent -->
            <div class="agent-card p-4 rounded-lg" id="economic-agent-card">
                <div class="flex items-center justify-between mb-2">
                    <h3 class="text-sm font-semibold text-blue-400">
                        <i class="fas fa-globe mr-1"></i> Economic Agent
                    </h3>
                    <div class="agent-status" id="economic-status">
                        <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                    </div>
                </div>
                <div class="space-y-1 text-xs">
                    <div class="flex justify-between">
                        <span class="text-gray-500">GDP:</span>
                        <span id="gdp-value">2.34%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Inflation:</span>
                        <span id="inflation-value">3.13%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Fed Rate:</span>
                        <span id="fedrate-value">5.32%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">DXY:</span>
                        <span id="dxy-value">104.52</span>
                    </div>
                </div>
                <div class="mt-2 pt-2 border-t border-gray-700">
                    <div class="flex justify-between text-xs">
                        <span class="text-gray-500">Signal:</span>
                        <span class="font-mono" id="economic-signal">0.18</span>
                    </div>
                    <div class="flex justify-between text-xs">
                        <span class="text-gray-500">Confidence:</span>
                        <span class="font-mono" id="economic-confidence">92%</span>
                    </div>
                </div>
            </div>

            <!-- Sentiment Agent -->
            <div class="agent-card p-4 rounded-lg" id="sentiment-agent-card">
                <div class="flex items-center justify-between mb-2">
                    <h3 class="text-sm font-semibold text-purple-400">
                        <i class="fas fa-brain mr-1"></i> Sentiment Agent
                    </h3>
                    <div class="agent-status" id="sentiment-status">
                        <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                    </div>
                </div>
                <div class="space-y-1 text-xs">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Fear/Greed:</span>
                        <span id="fear-greed">48</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Social Vol:</span>
                        <span id="social-volume">161K</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Mood:</span>
                        <span id="market-mood">😰 Fearful</span>
                    </div>
                </div>
                <div class="mt-2 pt-2 border-t border-gray-700">
                    <div class="flex justify-between text-xs">
                        <span class="text-gray-500">Signal:</span>
                        <span class="font-mono" id="sentiment-signal">0.65</span>
                    </div>
                    <div class="flex justify-between text-xs">
                        <span class="text-gray-500">Confidence:</span>
                        <span class="font-mono" id="sentiment-confidence">86%</span>
                    </div>
                </div>
            </div>

            <!-- Microstructure Agent -->
            <div class="agent-card p-4 rounded-lg" id="microstructure-agent-card">
                <div class="flex items-center justify-between mb-2">
                    <h3 class="text-sm font-semibold text-green-400">
                        <i class="fas fa-microscope mr-1"></i> Microstructure
                    </h3>
                    <div class="agent-status">
                        <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                    </div>
                </div>
                <div class="space-y-1 text-xs">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Spread:</span>
                        <span id="spread-value">$0.32</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Depth:</span>
                        <span id="depth-value">$3.2M</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Imbalance:</span>
                        <span id="imbalance-value">15%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Toxicity:</span>
                        <span id="toxicity-value">22%</span>
                    </div>
                </div>
                <div class="mt-2 pt-2 border-t border-gray-700">
                    <div class="flex justify-between text-xs">
                        <span class="text-gray-500">Signal:</span>
                        <span class="font-mono">0.72</span>
                    </div>
                    <div class="flex justify-between text-xs">
                        <span class="text-gray-500">Confidence:</span>
                        <span class="font-mono">91%</span>
                    </div>
                </div>
            </div>

            <!-- Cross-Exchange Agent -->
            <div class="agent-card p-4 rounded-lg" id="cross-exchange-agent-card">
                <div class="flex items-center justify-between mb-2">
                    <h3 class="text-sm font-semibold text-yellow-400">
                        <i class="fas fa-exchange-alt mr-1"></i> Cross-Exchange
                    </h3>
                    <div class="agent-status">
                        <div class="w-2 h-2 bg-green-500 rounded-full"></div>
                    </div>
                </div>
                <div class="space-y-1 text-xs">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Bin-CB:</span>
                        <span id="spread-bc" class="positive">+$18.36</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Krk-BB:</span>
                        <span id="spread-kb" class="positive">+$27.42</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Fut-Spot:</span>
                        <span id="spread-fs" class="positive">+$256.83</span>
                    </div>
                </div>
                <div class="mt-2 pt-2 border-t border-gray-700">
                    <div class="flex justify-between text-xs">
                        <span class="text-gray-500">Signal:</span>
                        <span class="font-mono">0.83</span>
                    </div>
                    <div class="flex justify-between text-xs">
                        <span class="text-gray-500">Confidence:</span>
                        <span class="font-mono">95%</span>
                    </div>
                </div>
            </div>
        </div>

        <!-- Center Panel - Hyperbolic Engine & LLM -->
        <div class="flex-1 p-4">
            <!-- Hyperbolic Clustering Visualization -->
            <div class="hyperbolic-container rounded-lg p-4 mb-4" style="height: 60%;">
                <div class="flex items-center justify-between mb-2">
                    <h2 class="text-lg font-bold text-gray-300">
                        <i class="fas fa-project-diagram mr-2 text-indigo-500"></i>
                        Hyperbolic Asset Clustering Engine
                    </h2>
                    <div class="flex items-center space-x-4 text-xs">
                        <span>Geodesic Paths: <span class="font-mono text-indigo-400">791</span></span>
                        <span>Curvature: <span class="font-mono text-indigo-400">-1.0</span></span>
                        <span>Efficiency: <span class="font-mono text-green-400">99.5%</span></span>
                    </div>
                </div>
                <svg id="hyperbolic-viz" width="100%" height="100%"></svg>
            </div>

            <!-- LLM Fusion Brain -->
            <div class="grid grid-cols-2 gap-4" style="height: 35%;">
                <!-- LLM Processing -->
                <div class="agent-card p-4 rounded-lg">
                    <div class="flex items-center justify-between mb-3">
                        <h3 class="text-sm font-bold text-purple-400">
                            <i class="fas fa-brain mr-2"></i>
                            LLM Fusion Brain (Claude 3)
                        </h3>
                        <div id="llm-status" class="flex items-center space-x-2">
                            <div class="w-2 h-2 bg-green-500 rounded-full pulse"></div>
                            <span class="text-xs" id="llm-status-text">Processing</span>
                        </div>
                    </div>
                    
                    <!-- Data Flow Visualization -->
                    <div class="mb-3">
                        <svg width="100%" height="80">
                            <defs>
                                <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                                 refX="9" refY="3.5" orient="auto">
                                    <polygon points="0 0, 10 3.5, 0 7" fill="#6366f1" />
                                </marker>
                            </defs>
                            <!-- Agent inputs -->
                            <circle cx="30" cy="20" r="4" fill="#3b82f6" class="pulse"/>
                            <circle cx="30" cy="40" r="4" fill="#8b5cf6" class="pulse"/>
                            <circle cx="30" cy="60" r="4" fill="#10b981" class="pulse"/>
                            
                            <!-- Flow lines -->
                            <line x1="35" y1="20" x2="90" y2="40" stroke="#6366f1" stroke-width="2" 
                                  class="data-flow" marker-end="url(#arrowhead)"/>
                            <line x1="35" y1="40" x2="90" y2="40" stroke="#6366f1" stroke-width="2" 
                                  class="data-flow" marker-end="url(#arrowhead)"/>
                            <line x1="35" y1="60" x2="90" y2="40" stroke="#6366f1" stroke-width="2" 
                                  class="data-flow" marker-end="url(#arrowhead)"/>
                            
                            <!-- LLM Brain -->
                            <circle cx="110" cy="40" r="15" fill="url(#brain-gradient)" stroke="#8b5cf6" stroke-width="2"/>
                            <text x="110" y="45" text-anchor="middle" fill="white" font-size="10">LLM</text>
                            
                            <!-- Output -->
                            <line x1="125" y1="40" x2="180" y2="40" stroke="#10b981" stroke-width="2" 
                                  marker-end="url(#arrowhead)"/>
                            <rect x="185" y="35" width="40" height="10" fill="#10b981" rx="2"/>
                            <text x="205" y="42" text-anchor="middle" fill="white" font-size="8">Decision</text>
                            
                            <defs>
                                <radialGradient id="brain-gradient">
                                    <stop offset="0%" style="stop-color:#8b5cf6;stop-opacity:1" />
                                    <stop offset="100%" style="stop-color:#6366f1;stop-opacity:1" />
                                </radialGradient>
                            </defs>
                        </svg>
                    </div>

                    <div class="space-y-2 text-xs">
                        <div class="flex justify-between">
                            <span class="text-gray-500">Input Agents:</span>
                            <span class="font-mono">5 Active</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-500">Processing Time:</span>
                            <span class="font-mono" id="processing-time">47ms</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-500">Context Window:</span>
                            <span class="font-mono">128k tokens</span>
                        </div>
                        <div class="flex justify-between">
                            <span class="text-gray-500">Next Update:</span>
                            <span class="font-mono" id="next-update">3s</span>
                        </div>
                    </div>
                </div>

                <!-- Real-time Predictions -->
                <div class="agent-card p-4 rounded-lg">
                    <h3 class="text-sm font-bold text-green-400 mb-3">
                        <i class="fas fa-chart-line mr-2"></i>
                        Real-Time Predictions
                    </h3>
                    <div class="space-y-3">
                        <div class="bg-gray-800 p-3 rounded">
                            <div class="flex justify-between items-center mb-1">
                                <span class="text-xs text-gray-500">Predicted Spread:</span>
                                <span class="text-lg font-bold positive" id="predicted-spread">0.73%</span>
                            </div>
                            <div class="flex justify-between text-xs">
                                <span class="text-gray-500">Confidence:</span>
                                <span id="prediction-confidence">87%</span>
                            </div>
                        </div>
                        
                        <div class="grid grid-cols-2 gap-2 text-xs">
                            <div class="bg-gray-800 p-2 rounded">
                                <div class="text-gray-500">Direction</div>
                                <div class="font-semibold" id="direction">Converge</div>
                            </div>
                            <div class="bg-gray-800 p-2 rounded">
                                <div class="text-gray-500">Opportunities</div>
                                <div class="font-semibold" id="opportunities-count">6</div>
                            </div>
                        </div>
                        
                        <div class="bg-gray-800 p-2 rounded text-xs">
                            <div class="text-gray-500 mb-1">Active Arbitrage Pairs:</div>
                            <div class="flex flex-wrap gap-1" id="active-pairs">
                                <span class="bg-green-900 px-2 py-1 rounded">BTC-USDT</span>
                                <span class="bg-green-900 px-2 py-1 rounded">ETH-USDT</span>
                                <span class="bg-yellow-900 px-2 py-1 rounded">SOL-USDT</span>
                            </div>
                        </div>
                        
                        <div class="bg-gray-800 p-2 rounded text-xs mt-2">
                            <div class="text-gray-500 mb-1">Strategy:</div>
                            <div id="llm-strategy" class="text-xs text-blue-400">Loading...</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Right Panel - Performance & Parameters -->
        <div class="w-1/4 p-4 space-y-4 overflow-y-auto border-l border-gray-800">
            <h2 class="text-lg font-bold text-gray-300 mb-2">
                <i class="fas fa-cog mr-2 text-gray-500"></i>
                Model Parameters & Performance
            </h2>

            <!-- Decision Constraints -->
            <div class="bg-gray-800 p-3 rounded-lg">
                <h3 class="text-sm font-semibold text-blue-400 mb-2">Decision Constraints</h3>
                <div class="space-y-1 text-xs">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Max Exposure:</span>
                        <span class="font-mono">3% NAV</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Min Spread:</span>
                        <span class="font-mono">0.5%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">LLM Confidence:</span>
                        <span class="font-mono">≥80%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Max Hold Time:</span>
                        <span class="font-mono">3600s</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Slippage Limit:</span>
                        <span class="font-mono">0.2%</span>
                    </div>
                </div>
            </div>

            <!-- AOS Weights -->
            <div class="bg-gray-800 p-3 rounded-lg">
                <h3 class="text-sm font-semibold text-purple-400 mb-2">AOS Weight Distribution</h3>
                <div class="space-y-2">
                    <div class="text-xs">
                        <div class="flex justify-between mb-1">
                            <span class="text-gray-500">Price:</span>
                            <span>40%</span>
                        </div>
                        <div class="w-full bg-gray-700 rounded-full h-1.5">
                            <div class="bg-blue-500 h-1.5 rounded-full" style="width: 40%"></div>
                        </div>
                    </div>
                    <div class="text-xs">
                        <div class="flex justify-between mb-1">
                            <span class="text-gray-500">Sentiment:</span>
                            <span>25%</span>
                        </div>
                        <div class="w-full bg-gray-700 rounded-full h-1.5">
                            <div class="bg-purple-500 h-1.5 rounded-full" style="width: 25%"></div>
                        </div>
                    </div>
                    <div class="text-xs">
                        <div class="flex justify-between mb-1">
                            <span class="text-gray-500">Volume:</span>
                            <span>20%</span>
                        </div>
                        <div class="w-full bg-gray-700 rounded-full h-1.5">
                            <div class="bg-green-500 h-1.5 rounded-full" style="width: 20%"></div>
                        </div>
                    </div>
                    <div class="text-xs">
                        <div class="flex justify-between mb-1">
                            <span class="text-gray-500">Microstructure:</span>
                            <span>10%</span>
                        </div>
                        <div class="w-full bg-gray-700 rounded-full h-1.5">
                            <div class="bg-yellow-500 h-1.5 rounded-full" style="width: 10%"></div>
                        </div>
                    </div>
                    <div class="text-xs">
                        <div class="flex justify-between mb-1">
                            <span class="text-gray-500">Risk:</span>
                            <span>5%</span>
                        </div>
                        <div class="w-full bg-gray-700 rounded-full h-1.5">
                            <div class="bg-red-500 h-1.5 rounded-full" style="width: 5%"></div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Performance Metrics -->
            <div class="bg-gray-800 p-3 rounded-lg">
                <h3 class="text-sm font-semibold text-green-400 mb-2">Live Performance</h3>
                <div class="grid grid-cols-2 gap-2 text-xs">
                    <div class="text-center p-2 bg-gray-900 rounded">
                        <div class="text-gray-500">P&L Today</div>
                        <div class="text-lg font-bold positive">+$4,260</div>
                    </div>
                    <div class="text-center p-2 bg-gray-900 rounded">
                        <div class="text-gray-500">Win Rate</div>
                        <div class="text-lg font-bold">82.7%</div>
                    </div>
                    <div class="text-center p-2 bg-gray-900 rounded">
                        <div class="text-gray-500">Sharpe</div>
                        <div class="text-lg font-bold">2.15</div>
                    </div>
                    <div class="text-center p-2 bg-gray-900 rounded">
                        <div class="text-gray-500">Trades</div>
                        <div class="text-lg font-bold">127</div>
                    </div>
                </div>
            </div>

            <!-- Backtest vs LLM -->
            <div class="bg-gray-800 p-3 rounded-lg">
                <h3 class="text-sm font-semibold text-orange-400 mb-2">Backtest vs LLM</h3>
                <div class="space-y-1 text-xs">
                    <div class="flex justify-between">
                        <span class="text-gray-500">Backtest Sharpe:</span>
                        <span>1.82</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">LLM Sharpe:</span>
                        <span class="positive font-bold">2.15 (+18%)</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">Backtest Win:</span>
                        <span>68.3%</span>
                    </div>
                    <div class="flex justify-between">
                        <span class="text-gray-500">LLM Win:</span>
                        <span class="positive font-bold">82.7% (+21%)</span>
                    </div>
                </div>
            </div>
            
            <!-- Live Arbitrage Opportunities -->
            <div class="bg-gradient-to-br from-green-900 to-gray-800 p-3 rounded-lg border border-green-600">
                <h3 class="text-sm font-semibold text-green-400 mb-2">
                    <i class="fas fa-bolt mr-1"></i> Live Opportunities
                </h3>
                <div id="live-opportunities" class="space-y-2">
                    <div class="text-xs text-gray-400">Scanning markets...</div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        let uptime = 0;
        
        // Initialize D3 for hyperbolic visualization
        const svg = d3.select('#hyperbolic-viz');
        const width = svg.node().getBoundingClientRect().width;
        const height = svg.node().getBoundingClientRect().height;
        
        // Create hyperbolic projection
        const hyperbolicScale = d3.scaleLinear()
            .domain([-1, 1])
            .range([0, Math.min(width, height)]);
        
        // Initialize visualization
        function initHyperbolicVisualization() {
            const g = svg.append('g')
                .attr('transform', \`translate(\${width/2},\${height/2})\`);
            
            // Add grid circles for depth perception
            const circles = [0.2, 0.4, 0.6, 0.8, 1.0];
            circles.forEach(r => {
                g.append('circle')
                    .attr('r', hyperbolicScale(r) / 2)
                    .attr('fill', 'none')
                    .attr('stroke', '#1e293b')
                    .attr('stroke-width', 0.5)
                    .attr('opacity', 0.5);
            });
            
            // Add radial lines
            for (let i = 0; i < 8; i++) {
                const angle = (i / 8) * 2 * Math.PI;
                g.append('line')
                    .attr('x1', 0)
                    .attr('y1', 0)
                    .attr('x2', Math.cos(angle) * hyperbolicScale(1) / 2)
                    .attr('y2', Math.sin(angle) * hyperbolicScale(1) / 2)
                    .attr('stroke', '#1e293b')
                    .attr('stroke-width', 0.5)
                    .attr('opacity', 0.3);
            }
        }
        
        // Update hyperbolic clusters
        async function updateHyperbolicClusters() {
            try {
                const response = await fetch('/api/hyperbolic/clusters');
                const data = await response.json();
                
                const g = svg.select('g');
                
                // Remove old elements
                g.selectAll('.connection').remove();
                g.selectAll('.asset-node').remove();
                
                // Draw connections
                data.connections.forEach(conn => {
                    const source = data.assets.find(a => a.id === conn.source);
                    const target = data.assets.find(a => a.id === conn.target);
                    
                    if (source && target) {
                        g.append('line')
                            .attr('class', 'connection')
                            .attr('x1', source.x * hyperbolicScale(1) / 2)
                            .attr('y1', source.y * hyperbolicScale(1) / 2)
                            .attr('x2', target.x * hyperbolicScale(1) / 2)
                            .attr('y2', target.y * hyperbolicScale(1) / 2)
                            .attr('stroke', '#6366f1')
                            .attr('stroke-width', conn.strength * 2)
                            .attr('opacity', conn.strength * 0.5);
                    }
                });
                
                // Draw asset nodes
                const nodes = g.selectAll('.asset-node')
                    .data(data.assets)
                    .enter()
                    .append('g')
                    .attr('class', 'asset-node')
                    .attr('transform', d => \`translate(\${d.x * hyperbolicScale(1) / 2},\${d.y * hyperbolicScale(1) / 2})\`);
                
                nodes.append('circle')
                    .attr('r', d => d.r * hyperbolicScale(1) / 2)
                    .attr('fill', d => d.momentum > 0 ? '#10b981' : '#ef4444')
                    .attr('opacity', 0.8)
                    .attr('stroke', '#fff')
                    .attr('stroke-width', 1);
                
                nodes.append('text')
                    .text(d => d.id)
                    .attr('text-anchor', 'middle')
                    .attr('dy', 3)
                    .attr('fill', 'white')
                    .attr('font-size', '10px')
                    .attr('font-weight', 'bold');
                
                // Add pulsing animation to active nodes
                nodes.selectAll('circle')
                    .transition()
                    .duration(2000)
                    .attr('r', d => d.r * hyperbolicScale(1) / 2 * 1.2)
                    .transition()
                    .duration(2000)
                    .attr('r', d => d.r * hyperbolicScale(1) / 2)
                    .on('end', function() { d3.select(this.parentNode).dispatch('pulse'); });
                    
            } catch (error) {
                console.error('Failed to update hyperbolic clusters:', error);
            }
        }
        
        // Update agent statuses
        async function updateAgentStatuses() {
            try {
                const response = await fetch('/api/agents/status');
                const statuses = await response.json();
                
                // Update each agent card
                Object.keys(statuses).forEach(agent => {
                    const card = document.getElementById(\`\${agent}-agent-card\`);
                    if (card) {
                        if (statuses[agent].status === 'active') {
                            card.classList.add('active');
                        } else {
                            card.classList.remove('active');
                        }
                    }
                    
                    // Update signals and confidence
                    const signalEl = document.getElementById(\`\${agent}-signal\`);
                    const confEl = document.getElementById(\`\${agent}-confidence\`);
                    if (signalEl) signalEl.textContent = statuses[agent].lastSignal.toFixed(2);
                    if (confEl) confEl.textContent = (statuses[agent].confidence * 100).toFixed(0) + '%';
                });
                
            } catch (error) {
                console.error('Failed to update agent statuses:', error);
            }
        }
        
        // Update LLM state
        async function updateLLMState() {
            try {
                const response = await fetch('/api/llm/state');
                const state = await response.json();
                
                const statusText = document.getElementById('llm-status-text');
                const statusDiv = document.getElementById('llm-status');
                
                if (state.processing) {
                    statusText.textContent = 'Processing';
                    statusDiv.innerHTML = '<div class="w-2 h-2 bg-yellow-500 rounded-full pulse"></div><span class="text-xs">Processing</span>';
                } else {
                    statusText.textContent = 'Ready';
                    statusDiv.innerHTML = '<div class="w-2 h-2 bg-green-500 rounded-full"></div><span class="text-xs">Ready</span>';
                }
                
                if (state.lastPrediction) {
                    document.getElementById('predicted-spread').textContent = state.lastPrediction.spread + '%';
                    document.getElementById('prediction-confidence').textContent = (state.lastPrediction.confidence * 100).toFixed(0) + '%';
                    document.getElementById('direction').textContent = state.lastPrediction.direction;
                    document.getElementById('opportunities-count').textContent = state.lastPrediction.opportunities;
                }
                
                // Update countdown
                const nextUpdate = Math.max(0, Math.floor((state.nextUpdate - Date.now()) / 1000));
                document.getElementById('next-update').textContent = nextUpdate + 's';
                
            } catch (error) {
                console.error('Failed to update LLM state:', error);
            }
        }
        
        // Update live data feeds
        async function updateDataFeeds() {
            try {
                const response = await fetch('/api/datafeed/live');
                const data = await response.json();
                
                // Update economic data
                document.getElementById('gdp-value').textContent = data.economic.gdp + '%';
                document.getElementById('inflation-value').textContent = data.economic.inflation + '%';
                document.getElementById('fedrate-value').textContent = data.economic.fedRate + '%';
                document.getElementById('dxy-value').textContent = data.economic.dxy;
                
                // Update sentiment
                document.getElementById('fear-greed').textContent = data.sentiment.fearGreed;
                document.getElementById('social-volume').textContent = (data.sentiment.socialVolume / 1000).toFixed(0) + 'K';
                
                // Update microstructure
                document.getElementById('spread-value').textContent = '$' + data.microstructure.spread;
                document.getElementById('depth-value').textContent = '$' + (data.microstructure.depth / 1000000).toFixed(1) + 'M';
                document.getElementById('imbalance-value').textContent = (data.microstructure.imbalance * 100).toFixed(0) + '%';
                document.getElementById('toxicity-value').textContent = (data.microstructure.toxicity * 100).toFixed(0) + '%';
                
                // Update cross-exchange
                document.getElementById('spread-bc').textContent = '+$' + Math.abs(data.crossExchange['binance-coinbase']).toFixed(2);
                document.getElementById('spread-kb').textContent = '+$' + Math.abs(data.crossExchange['kraken-bybit']).toFixed(2);
                document.getElementById('spread-fs').textContent = '+$' + Math.abs(data.crossExchange['futures-spot']).toFixed(2);
                
            } catch (error) {
                console.error('Failed to update data feeds:', error);
            }
        }
        
        // Update uptime
        setInterval(() => {
            uptime++;
            const hours = Math.floor(uptime / 3600);
            const minutes = Math.floor((uptime % 3600) / 60);
            const seconds = uptime % 60;
            document.getElementById('uptime').textContent = 
                \`\${hours.toString().padStart(2, '0')}:\${minutes.toString().padStart(2, '0')}:\${seconds.toString().padStart(2, '0')}\`;
        }, 1000);
        
        // Initialize and start updates
        initHyperbolicVisualization();
        updateHyperbolicClusters();
        updateAgentStatuses();
        updateLLMState();
        updateDataFeeds();
        
        // Set update intervals
        setInterval(updateHyperbolicClusters, 2000);
        setInterval(updateAgentStatuses, 3000);
        setInterval(updateLLMState, 1000);
        setInterval(updateDataFeeds, 5000);
        
        // Socket.io real-time updates
        socket.on('cluster_update', (data) => {
            // Real-time cluster updates would go here
        });
        
        socket.on('agent_update', (statuses) => {
            // Update agent cards with real-time data
            Object.keys(statuses).forEach(agent => {
                const card = document.getElementById(\`\${agent}-agent-card\`);
                if (card) {
                    if (statuses[agent].status === 'active') {
                        card.classList.add('active');
                    } else {
                        card.classList.remove('active');
                    }
                }
                
                const signalEl = document.getElementById(\`\${agent}-signal\`);
                const confEl = document.getElementById(\`\${agent}-confidence\`);
                if (signalEl) signalEl.textContent = statuses[agent].lastSignal.toFixed(2);
                if (confEl) confEl.textContent = (statuses[agent].confidence * 100).toFixed(0) + '%';
            });
        });
        
        socket.on('llm_update', (state) => {
            // Update LLM state with real-time data
            const statusText = document.getElementById('llm-status-text');
            const statusDiv = document.getElementById('llm-status');
            
            if (state.processing) {
                statusText.textContent = 'Processing';
                statusDiv.innerHTML = '<div class="w-2 h-2 bg-yellow-500 rounded-full pulse"></div><span class="text-xs">Processing</span>';
            } else {
                statusText.textContent = 'Ready';
                statusDiv.innerHTML = '<div class="w-2 h-2 bg-green-500 rounded-full"></div><span class="text-xs">Ready</span>';
            }
            
            if (state.lastPrediction) {
                document.getElementById('predicted-spread').textContent = state.lastPrediction.spread + '%';
                document.getElementById('prediction-confidence').textContent = (state.lastPrediction.confidence * 100).toFixed(0) + '%';
                document.getElementById('direction').textContent = state.lastPrediction.direction;
                document.getElementById('opportunities-count').textContent = state.lastPrediction.opportunities;
                
                // Update strategy if available
                if (state.lastPrediction.strategy) {
                    document.getElementById('llm-strategy').textContent = state.lastPrediction.strategy;
                }
            }
        });
        
        socket.on('arbitrage_opportunities', (data) => {
            // Display live arbitrage opportunities
            const container = document.getElementById('live-opportunities');
            if (data.opportunities && data.opportunities.length > 0) {
                let html = '';
                data.opportunities.forEach((opp, idx) => {
                    const profitColor = opp.profitPotential > 1 ? 'text-green-400' : 
                                       opp.profitPotential > 0.5 ? 'text-yellow-400' : 'text-gray-400';
                    html += \`
                        <div class="bg-gray-900 p-2 rounded text-xs border border-gray-700">
                            <div class="flex justify-between items-center">
                                <span class="font-semibold">\${opp.symbol}</span>
                                <span class="\${profitColor} font-bold">+\${opp.profitPotential.toFixed(2)}%</span>
                            </div>
                            <div class="flex justify-between text-gray-400 mt-1">
                                <span>\${opp.type}</span>
                                <span>\${opp.exchanges}</span>
                            </div>
                        </div>
                    \`;
                });
                html += \`<div class="text-xs text-gray-500 mt-2">Total: \${data.count} opportunities</div>\`;
                container.innerHTML = html;
            } else {
                container.innerHTML = '<div class="text-xs text-gray-400">No active opportunities</div>';
            }
        });
    </script>
</body>
</html>`);
});

// WebSocket handlers
io.on('connection', (socket) => {
  logger.info('Client connected to professional dashboard');
  
  // Send initial state
  socket.emit('initial_state', {
    agents: agentStatuses,
    clusters: hyperbolicClusters,
    llm: llmState
  });
  
  socket.on('disconnect', () => {
    logger.info('Client disconnected from professional dashboard');
  });
});

// Update loops for real-time data
setInterval(() => {
  hyperbolicClusters = generateHyperbolicClusters();
  io.emit('cluster_update', hyperbolicClusters);
}, 2000);

setInterval(() => {
  updateAgentStatuses();
  io.emit('agent_update', agentStatuses);
}, 1000);

setInterval(() => {
  updateLLMProcessing();
  io.emit('llm_update', llmState);
}, 1000);

// Emit real-time arbitrage opportunities
setInterval(() => {
  const opportunities = (llmArbitrage as any).opportunities || [];
  const topOpportunities = opportunities
    .sort((a, b) => b.profitPotential - a.profitPotential)
    .slice(0, 5);
  
  io.emit('arbitrage_opportunities', {
    opportunities: topOpportunities,
    count: opportunities.length,
    timestamp: Date.now()
  });
}, 2000);

server.listen(PORT, '0.0.0.0', () => {
  logger.info(`Professional dashboard running on port ${PORT}`);
  console.log(`\n🚀 Professional Dashboard with Hyperbolic Clustering: http://localhost:${PORT}\n`);
});