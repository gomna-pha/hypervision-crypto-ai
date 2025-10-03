import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serveStatic } from 'hono/cloudflare-workers'

const app = new Hono()

// Enable CORS for API routes
app.use('/api/*', cors())

// Serve static files
app.use('/static/*', serveStatic({ root: './public' }))

// Market data simulation - In production, this would connect to real APIs
const generateMarketData = () => {
  const basePrice = {
    BTC: 67234.56,
    ETH: 3456.08,
    SOL: 123.45
  }
  
  return {
    BTC: {
      price: basePrice.BTC + (Math.random() - 0.5) * 100,
      change24h: (Math.random() - 0.5) * 10,
      volume: Math.floor(Math.random() * 50000) + 10000,
      trades: Math.floor(Math.random() * 1000000) + 500000
    },
    ETH: {
      price: basePrice.ETH + (Math.random() - 0.5) * 50,
      change24h: (Math.random() - 0.5) * 8,
      volume: Math.floor(Math.random() * 30000) + 15000,
      trades: Math.floor(Math.random() * 800000) + 400000
    },
    SOL: {
      price: basePrice.SOL + (Math.random() - 0.5) * 10,
      change24h: (Math.random() - 0.5) * 12,
      volume: Math.floor(Math.random() * 5000) + 2000,
      trades: Math.floor(Math.random() * 200000) + 100000
    }
  }
}

// Generate arbitrage opportunities
const generateArbitrageOpportunities = () => {
  return [
    {
      type: 'Cross-Exchange Arbitrage',
      pair: 'BTC: Binance → Coinbase',
      profit: (Math.random() * 0.5 + 0.1).toFixed(2),
      profitUSD: Math.floor(Math.random() * 200 + 50),
      executionTime: Math.floor(Math.random() * 60 + 30),
      buyPrice: 67230 + Math.random() * 20,
      sellPrice: 67357 + Math.random() * 20,
      volume: (Math.random() * 3 + 1).toFixed(1),
      confidence: Math.floor(Math.random() * 20 + 80)
    },
    {
      type: 'Triangular Arbitrage',
      pair: 'BTC → ETH → USDT → BTC',
      profit: (Math.random() * 0.4 + 0.15).toFixed(2),
      profitUSD: Math.floor(Math.random() * 300 + 100),
      executionTime: Math.floor(Math.random() * 40 + 20),
      pathLength: 3,
      slippageRisk: 'Low (0.02%)',
      capital: 50000,
      confidence: Math.floor(Math.random() * 15 + 85)
    },
    {
      type: 'Statistical Pairs Trading',
      pair: 'ETH/BTC Mean Reversion',
      profit: (Math.random() * 0.5 + 0.2).toFixed(2),
      profitUSD: Math.floor(Math.random() * 400 + 150),
      executionTime: '2-4 hours',
      zScore: (Math.random() * 2 + 1.5).toFixed(1),
      correlation: (Math.random() * 0.2 + 0.8).toFixed(3),
      finBERT: (Math.random() * 0.5 + 0.5).toFixed(2),
      confidence: Math.floor(Math.random() * 10 + 90)
    }
  ]
}

// Portfolio data
const getPortfolioData = () => {
  const totalValue = 2847563
  return {
    totalValue,
    monthlyChange: 12.4,
    assets: {
      BTC: {
        percentage: 45,
        value: Math.floor(totalValue * 0.45),
        quantity: 19.065,
        avgPrice: 65420,
        currentPrice: 67234,
        pnl: 34562,
        pnlPercent: 2.8
      },
      ETH: {
        percentage: 30,
        value: Math.floor(totalValue * 0.30),
        quantity: 247.2,
        avgPrice: 3380,
        currentPrice: 3456,
        pnl: 18787,
        pnlPercent: 2.2
      },
      STABLE: {
        percentage: 15,
        value: Math.floor(totalValue * 0.15),
        quantity: 427134,
        avgPrice: 1.0,
        currentPrice: 1.0,
        pnl: 0,
        pnlPercent: 0
      },
      OTHER: {
        percentage: 10,
        value: Math.floor(totalValue * 0.10),
        quantity: 1,
        avgPrice: 284757,
        currentPrice: 284757,
        pnl: 0,
        pnlPercent: 0
      }
    },
    metrics: {
      sharpeRatio: 2.34,
      maxDrawdown: -3.2,
      var95: 45231,
      beta: 0.73
    }
  }
}

// Global market indices
const getGlobalMarkets = () => {
  return {
    crypto: {
      BTC: { price: 67234, change: 2.34 },
      ETH: { price: 3456, change: 1.87 },
      SOL: { price: 123, change: 4.56 }
    },
    equity: {
      SP500: { price: 5234.56, change: 0.45 },
      NASDAQ: { price: 18234, change: -0.23 },
      DOW: { price: 42156, change: 0.67 }
    },
    international: {
      FTSE: { price: 8234, change: 0.34 },
      NIKKEI: { price: 38234, change: -0.45 },
      DAX: { price: 19234, change: 0.89 }
    },
    commodities: {
      GOLD: { price: 2034, change: 0.23 },
      SILVER: { price: 24.56, change: 0.89 },
      OIL: { price: 78.34, change: -1.45 }
    },
    forex: {
      EURUSD: { price: 1.0856, change: 0.12 },
      GBPUSD: { price: 1.2634, change: -0.08 },
      USDJPY: { price: 149.23, change: 0.34 }
    }
  }
}

// Order book data
const getOrderBook = () => {
  const basePrice = 67810
  const bids = []
  const asks = []
  
  for (let i = 0; i < 10; i++) {
    bids.push({
      price: (basePrice - i * 0.5 - Math.random() * 0.3).toFixed(2),
      volume: (Math.random() * 20 + 1).toFixed(2)
    })
    asks.push({
      price: (basePrice + i * 0.5 + Math.random() * 0.3).toFixed(2),
      volume: (Math.random() * 20 + 1).toFixed(2)
    })
  }
  
  return { bids, asks, spread: (asks[0].price - bids[0].price).toFixed(2) }
}

// API Routes
app.get('/api/market-data', (c) => {
  return c.json(generateMarketData())
})

app.get('/api/arbitrage-opportunities', (c) => {
  return c.json(generateArbitrageOpportunities())
})

app.get('/api/portfolio', (c) => {
  return c.json(getPortfolioData())
})

app.get('/api/global-markets', (c) => {
  return c.json(getGlobalMarkets())
})

app.get('/api/orderbook/:symbol', (c) => {
  return c.json(getOrderBook())
})

app.post('/api/execute-arbitrage', async (c) => {
  const body = await c.req.json()
  
  // Simulate arbitrage execution
  const success = Math.random() > 0.1 // 90% success rate
  const executionTime = Math.floor(Math.random() * 100 + 50)
  
  return c.json({
    success,
    executionTime,
    message: success ? 'Arbitrage executed successfully' : 'Execution failed - market conditions changed',
    transactionId: `arb_${Date.now()}`
  })
})

app.post('/api/ai-query', async (c) => {
  const { query } = await c.req.json()
  
  // Simulate AI responses
  const responses = {
    'market analysis': 'Current market shows strong bullish momentum with BTC breaking resistance at $67,000. Volume indicators suggest continued upward pressure.',
    'risk assessment': 'Portfolio risk is well-managed with 73% correlation to market beta. Current VaR of $45,231 represents 1.6% of total portfolio value.',
    'arbitrage': 'Current cross-exchange spreads offer profitable opportunities. Binance-Coinbase spread of +0.18% provides $127 profit potential.',
    'default': `Based on current market conditions and your query about "${query}", I recommend monitoring the BTC resistance levels and ETH correlation patterns for optimal trading opportunities.`
  }
  
  const responseKey = Object.keys(responses).find(key => 
    query.toLowerCase().includes(key)
  ) || 'default'
  
  return c.json({
    response: responses[responseKey],
    confidence: Math.floor(Math.random() * 20 + 80),
    timestamp: new Date().toISOString()
  })
})

// Main dashboard route
app.get('/', (c) => {
  return c.html(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GOMNA Trading Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
          tailwind.config = {
            theme: {
              extend: {
                colors: {
                  'dark-bg': '#0f1419',
                  'card-bg': '#1a1f29',
                  'accent': '#00d4aa',
                  'danger': '#ff4757',
                  'warning': '#ffa502',
                  'profit': '#2ed573',
                  'loss': '#ff4757'
                }
              }
            }
          }
        </script>
        <link href="/static/style.css" rel="stylesheet">
    </head>
    <body class="bg-dark-bg text-white font-mono">
        <!-- Navigation -->
        <nav class="bg-card-bg border-b border-gray-700 px-6 py-3">
            <div class="flex items-center justify-between">
                <div class="flex items-center space-x-8">
                    <h1 class="text-2xl font-bold text-accent">GOMNA</h1>
                    <div class="flex space-x-6">
                        <button class="nav-item active" data-section="dashboard">
                            <i class="fas fa-chart-line mr-2"></i>TRADING DASHBOARD
                        </button>
                        <button class="nav-item" data-section="portfolio">
                            <i class="fas fa-briefcase mr-2"></i>PORTFOLIO
                        </button>
                        <button class="nav-item" data-section="markets">
                            <i class="fas fa-globe mr-2"></i>GLOBAL MARKETS
                        </button>
                        <button class="nav-item" data-section="transparency">
                            <i class="fas fa-microscope mr-2"></i>MODEL TRANSPARENCY
                        </button>
                        <button class="nav-item" data-section="assistant">
                            <i class="fas fa-robot mr-2"></i>AI ASSISTANT
                        </button>
                    </div>
                </div>
                <div class="text-sm text-gray-400">
                    <span id="current-time"></span>
                </div>
            </div>
        </nav>

        <!-- Main Content -->
        <div class="flex">
            <!-- Main Dashboard -->
            <main class="flex-1 p-6">
                <!-- Dashboard Section -->
                <div id="dashboard" class="section active">
                    <div class="grid grid-cols-12 gap-6">
                        <!-- Live Market Feeds -->
                        <div class="col-span-4 bg-card-bg rounded-lg p-6">
                            <h3 class="text-lg font-semibold mb-4 flex items-center">
                                <i class="fas fa-broadcast-tower mr-2 text-accent"></i>
                                LIVE MARKET FEEDS
                            </h3>
                            <div id="market-feeds" class="space-y-4">
                                <!-- Market data will be populated here -->
                            </div>
                            
                            <h4 class="text-md font-semibold mt-6 mb-3 text-warning">CROSS-EXCHANGE SPREADS</h4>
                            <div id="spreads" class="space-y-2 text-sm">
                                <!-- Spreads will be populated here -->
                            </div>
                        </div>

                        <!-- Arbitrage Opportunities -->
                        <div class="col-span-8 bg-card-bg rounded-lg p-6">
                            <h3 class="text-lg font-semibold mb-4 flex items-center">
                                <i class="fas fa-bullseye mr-2 text-accent"></i>
                                🎯 Live Arbitrage Opportunities
                                <span id="active-count" class="ml-2 bg-profit text-dark-bg px-2 py-1 rounded text-sm">6 ACTIVE</span>
                                <span class="ml-auto text-sm text-gray-400">Last scan: <span id="last-scan"></span></span>
                            </h3>
                            <div id="arbitrage-opportunities" class="space-y-4">
                                <!-- Arbitrage opportunities will be populated here -->
                            </div>
                        </div>

                        <!-- Order Book -->
                        <div class="col-span-4 bg-card-bg rounded-lg p-6">
                            <h3 class="text-lg font-semibold mb-4 flex items-center">
                                <i class="fas fa-list mr-2 text-accent"></i>
                                ORDER BOOK DEPTH
                            </h3>
                            <div id="order-book">
                                <!-- Order book will be populated here -->
                            </div>
                        </div>

                        <!-- Strategy Performance -->
                        <div class="col-span-4 bg-card-bg rounded-lg p-6">
                            <h3 class="text-lg font-semibold mb-4 flex items-center">
                                <i class="fas fa-chart-bar mr-2 text-accent"></i>
                                📈 Strategy Performance Analysis
                            </h3>
                            <div class="grid grid-cols-2 gap-4">
                                <div class="text-center">
                                    <div class="text-2xl font-bold text-profit">+$4,260</div>
                                    <div class="text-sm text-gray-400">Total P&L Today</div>
                                </div>
                                <div class="text-center">
                                    <div class="text-2xl font-bold text-accent">82.7%</div>
                                    <div class="text-sm text-gray-400">Combined Win Rate</div>
                                </div>
                                <div class="text-center">
                                    <div class="text-2xl font-bold">50</div>
                                    <div class="text-sm text-gray-400">Total Executions</div>
                                </div>
                                <div class="text-center">
                                    <div class="text-2xl font-bold">47μs</div>
                                    <div class="text-sm text-gray-400">Avg Execution Time</div>
                                </div>
                            </div>
                        </div>

                        <!-- Hyperbolic Space Engine -->
                        <div class="col-span-4 bg-card-bg rounded-lg p-6">
                            <h3 class="text-lg font-semibold mb-4 flex items-center">
                                <i class="fas fa-atom mr-2 text-accent"></i>
                                HYPERBOLIC SPACE ENGINE
                            </h3>
                            <div class="text-center mb-4">
                                <div class="text-warning font-semibold">Poincaré Disk Model</div>
                            </div>
                            <div id="hyperbolic-canvas" class="mb-4">
                                <canvas id="poincare-disk" width="250" height="250" class="mx-auto bg-gray-900 rounded-full"></canvas>
                            </div>
                            <div class="space-y-2 text-sm">
                                <div class="flex justify-between">
                                    <span>Geodesic Paths:</span>
                                    <span class="text-accent">791</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Space Curvature:</span>
                                    <span class="text-accent">-1.0</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Path Efficiency:</span>
                                    <span class="text-profit">99.5%</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Portfolio Section -->
                <div id="portfolio" class="section">
                    <div class="grid grid-cols-12 gap-6">
                        <div class="col-span-8 bg-card-bg rounded-lg p-6">
                            <h3 class="text-lg font-semibold mb-4">📊 Portfolio Overview</h3>
                            <div id="portfolio-content">
                                <!-- Portfolio content will be populated here -->
                            </div>
                        </div>
                        <div class="col-span-4 bg-card-bg rounded-lg p-6">
                            <h3 class="text-lg font-semibold mb-4">Asset Allocation</h3>
                            <canvas id="portfolio-chart" width="300" height="300"></canvas>
                        </div>
                    </div>
                </div>

                <!-- Global Markets Section -->
                <div id="markets" class="section">
                    <div class="bg-card-bg rounded-lg p-6">
                        <h3 class="text-lg font-semibold mb-4">🌍 Global Market Indices</h3>
                        <div id="global-markets-content">
                            <!-- Global markets content will be populated here -->
                        </div>
                    </div>
                </div>

                <!-- Model Transparency Section -->
                <div id="transparency" class="section">
                    <div class="bg-card-bg rounded-lg p-6">
                        <h3 class="text-lg font-semibold mb-4">🔬 Algorithm Transparency</h3>
                        <div id="transparency-content">
                            <!-- Model transparency content will be populated here -->
                        </div>
                    </div>
                </div>

                <!-- AI Assistant Section -->
                <div id="assistant" class="section">
                    <div class="grid grid-cols-12 gap-6">
                        <div class="col-span-8 bg-card-bg rounded-lg p-6">
                            <h3 class="text-lg font-semibold mb-4 flex items-center">
                                <i class="fas fa-robot mr-2 text-accent"></i>
                                GOMNA AI Assistant
                            </h3>
                            <div id="chat-container" class="h-96 overflow-y-auto bg-gray-900 rounded p-4 mb-4">
                                <div class="chat-message ai-message mb-4">
                                    <div class="font-semibold text-accent mb-1">GOMNA AI</div>
                                    <div>Welcome to your advanced trading assistant! I can help you with real-time market analysis, arbitrage evaluation, risk assessment, and trading strategy recommendations. What would you like to analyze?</div>
                                </div>
                            </div>
                            <div class="flex space-x-2">
                                <input type="text" id="chat-input" placeholder="Ask me anything about trading..." 
                                       class="flex-1 bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white">
                                <button id="send-message" class="bg-accent text-dark-bg px-4 py-2 rounded hover:bg-opacity-80">
                                    <i class="fas fa-paper-plane"></i>
                                </button>
                            </div>
                            <div class="mt-4 flex space-x-2 text-sm">
                                <button class="quick-query bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded" data-query="Analyze current market opportunities">
                                    📊 Analyze Opportunities
                                </button>
                                <button class="quick-query bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded" data-query="Assess portfolio risk">
                                    ⚖️ Risk Assessment
                                </button>
                                <button class="quick-query bg-gray-700 hover:bg-gray-600 px-3 py-1 rounded" data-query="Explain arbitrage strategy">
                                    🎯 Arbitrage Strategy
                                </button>
                            </div>
                        </div>
                        <div class="col-span-4 bg-card-bg rounded-lg p-6">
                            <h4 class="font-semibold mb-4">AI Assistant Metrics</h4>
                            <div class="space-y-3">
                                <div class="flex justify-between">
                                    <span>Queries Today:</span>
                                    <span class="text-accent">47</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Accuracy Rate:</span>
                                    <span class="text-profit">94.2%</span>
                                </div>
                                <div class="flex justify-between">
                                    <span>Response Time:</span>
                                    <span class="text-accent">0.8s</span>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/axios@1.6.0/dist/axios.min.js"></script>
        <script src="/static/app.js"></script>
    </body>
    </html>
  `)
})

export default app