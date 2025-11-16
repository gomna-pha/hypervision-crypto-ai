import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serveStatic } from 'hono/cloudflare-workers'

const app = new Hono()

// Enable CORS for API routes
app.use('/api/*', cors())

// Serve static files from public directory
app.use('/static/*', serveStatic({ root: './public' }))

// API Routes for data simulation
app.get('/api/agents', (c) => {
  return c.json({
    economic: generateEconomicData(),
    sentiment: generateSentimentData(),
    crossExchange: generateCrossExchangeData(),
    onChain: generateOnChainData(),
    cnnPattern: generateCNNPatternData(),
    composite: generateCompositeSignal()
  })
})

app.get('/api/opportunities', (c) => {
  return c.json(generateOpportunities())
})

app.get('/api/backtest', (c) => {
  const withCNN = c.req.query('cnn') === 'true'
  return c.json(generateBacktestData(withCNN))
})

app.get('/api/patterns/timeline', (c) => {
  return c.json(generatePatternTimeline())
})

// Main dashboard route
app.get('/', (c) => {
  return c.html(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>ArbitrageAI - Production Crypto Arbitrage Platform</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
        <style>
          :root {
            --cream-bg: #FAF7F0;
            --cream-100: #F5F0E8;
            --cream-200: #F0EBE3;
            --cream-300: #E8DDD0;
            --navy: #1B365D;
            --navy-700: #1B365D;
            --navy-800: #142847;
            --forest: #2D5F3F;
            --burnt: #C07F39;
            --deep-red: #8B3A3A;
            --dark-brown: #2C2416;
            --warm-gray: #6B5D4F;
          }
          
          body {
            background-color: var(--cream-bg);
            color: var(--dark-brown);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
          }
          
          .card {
            background: white;
            border: 2px solid var(--cream-300);
            border-radius: 12px;
            padding: 1.5rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
            transition: all 0.3s ease;
          }
          
          .card:hover {
            box-shadow: 0 4px 12px rgba(0,0,0,0.08);
          }
          
          .metric-card {
            background: var(--cream-100);
            border-radius: 8px;
            padding: 1rem;
            text-align: center;
          }
          
          .btn-primary {
            background: var(--navy);
            color: var(--cream-bg);
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            font-weight: 600;
            border: none;
            cursor: pointer;
            transition: all 0.2s;
          }
          
          .btn-primary:hover {
            background: var(--navy-800);
            transform: translateY(-1px);
          }
          
          .pulse-dot {
            animation: pulse 2s infinite;
          }
          
          @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
          }
          
          .fade-in {
            animation: fadeIn 0.5s ease-in;
          }
          
          @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
          }
          
          .nav-tab {
            padding: 0.75rem 1.5rem;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
            color: var(--warm-gray);
          }
          
          .nav-tab.active {
            color: var(--navy);
            border-bottom-color: var(--navy);
            font-weight: 600;
          }
          
          .nav-tab:hover {
            color: var(--navy);
          }
          
          .progress-bar {
            height: 8px;
            background: var(--cream-300);
            border-radius: 4px;
            overflow: hidden;
          }
          
          .progress-fill {
            height: 100%;
            background: var(--navy);
            transition: width 0.5s ease;
          }
          
          .heatmap-cell {
            padding: 0.5rem;
            text-align: center;
            font-weight: 600;
            font-size: 0.75rem;
            border-radius: 4px;
          }
          
          .pattern-marker {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            position: absolute;
            transform: translateX(-50%);
          }
          
          .pattern-line {
            position: absolute;
            width: 2px;
            bottom: 20px;
            transform: translateX(-50%);
          }
        </style>
    </head>
    <body>
        <!-- First Visit Modal -->
        <div id="disclaimerModal" class="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 hidden">
          <div class="bg-white rounded-xl p-8 max-w-2xl mx-4 border-4 border-navy">
            <h2 class="text-2xl font-bold mb-4" style="color: var(--navy)">
              ⚠️ Important Legal Disclaimer
            </h2>
            <div class="space-y-3 text-sm" style="color: var(--dark-brown)">
              <p><strong>Educational Platform Only:</strong> This is a demonstration and educational tool showcasing advanced arbitrage trading concepts with CNN pattern recognition.</p>
              <p><strong>Simulated Data:</strong> All data displayed is simulated for demonstration purposes. Real-time API integration requires production deployment.</p>
              <p><strong>No Investment Advice:</strong> This platform does not provide investment advice, financial advice, trading advice, or any other sort of advice.</p>
              <p><strong>Risk Warning:</strong> Cryptocurrency trading carries substantial risk of loss. Past performance does not guarantee future results.</p>
              <p><strong>No Guarantees:</strong> Performance metrics shown are based on backtested simulations and do not represent actual trading results.</p>
            </div>
            <button onclick="acceptDisclaimer()" class="btn-primary w-full mt-6">
              I Understand - Continue to Platform
            </button>
          </div>
        </div>

        <!-- Header -->
        <header class="border-b-2" style="border-color: var(--cream-300); background: white;">
          <div class="container mx-auto px-6 py-4 flex justify-between items-center">
            <div class="flex items-center gap-3">
              <div class="w-10 h-10 rounded-lg flex items-center justify-center text-white text-xl" style="background: var(--navy)">
                🤖
              </div>
              <div>
                <h1 class="text-2xl font-bold" style="color: var(--navy)">ArbitrageAI</h1>
                <p class="text-xs" style="color: var(--warm-gray)">CNN-Enhanced Arbitrage Platform</p>
              </div>
            </div>
            <div class="flex items-center gap-6">
              <div class="text-right">
                <div class="text-xs" style="color: var(--warm-gray)">Portfolio Balance</div>
                <div class="text-xl font-bold" style="color: var(--navy)">$50,000</div>
              </div>
              <div class="text-right">
                <div class="text-xs" style="color: var(--warm-gray)">Active Strategies</div>
                <div class="text-xl font-bold" style="color: var(--forest)">4</div>
              </div>
              <div class="flex items-center gap-2">
                <div class="w-2 h-2 rounded-full pulse-dot" style="background: var(--forest)"></div>
                <span class="text-sm" style="color: var(--warm-gray)">Live</span>
              </div>
            </div>
          </div>
        </header>

        <!-- Navigation -->
        <nav class="border-b-2" style="border-color: var(--cream-300); background: white;">
          <div class="container mx-auto px-6 flex gap-1">
            <div class="nav-tab active" onclick="switchTab('dashboard')">
              <i class="fas fa-chart-line mr-2"></i>Dashboard
            </div>
            <div class="nav-tab" onclick="switchTab('opportunities')">
              <i class="fas fa-bolt mr-2"></i>Opportunities
            </div>
            <div class="nav-tab" onclick="switchTab('strategies')">
              <i class="fas fa-chess mr-2"></i>Strategies
            </div>
            <div class="nav-tab" onclick="switchTab('backtest')">
              <i class="fas fa-flask mr-2"></i>Backtest
            </div>
            <div class="nav-tab" onclick="switchTab('analytics')">
              <i class="fas fa-chart-bar mr-2"></i>Analytics
            </div>
          </div>
        </nav>

        <!-- Main Content -->
        <main class="container mx-auto px-6 py-8">
          <!-- Dashboard Tab -->
          <div id="dashboard-tab" class="tab-content">
            <!-- Agent Dashboard Grid (3x2) -->
            <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
              <!-- Economic Agent -->
              <div id="economic-agent" class="card fade-in"></div>
              
              <!-- Sentiment Agent -->
              <div id="sentiment-agent" class="card fade-in"></div>
              
              <!-- Cross-Exchange Agent -->
              <div id="cross-exchange-agent" class="card fade-in"></div>
              
              <!-- On-Chain Agent -->
              <div id="on-chain-agent" class="card fade-in"></div>
              
              <!-- CNN Pattern Agent -->
              <div id="cnn-pattern-agent" class="card fade-in"></div>
              
              <!-- Composite Signal -->
              <div id="composite-signal" class="card fade-in" style="border: 3px solid var(--navy)"></div>
            </div>

            <!-- Active Opportunities -->
            <div class="card mb-8">
              <h3 class="text-xl font-bold mb-4" style="color: var(--navy)">
                <i class="fas fa-bolt mr-2"></i>Top Arbitrage Opportunities
              </h3>
              <div id="opportunities-table" class="overflow-x-auto"></div>
            </div>

            <!-- Performance Overview -->
            <div class="card mb-8">
              <h3 class="text-xl font-bold mb-4" style="color: var(--navy)">
                <i class="fas fa-chart-area mr-2"></i>Performance Overview (Last 30 Days)
              </h3>
              <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div class="metric-card">
                  <div class="text-xs mb-1" style="color: var(--warm-gray)">Total Return</div>
                  <div class="text-2xl font-bold" style="color: var(--forest)">+14.8%</div>
                  <div class="text-xs" style="color: var(--forest)">↑ +2.4% with CNN</div>
                </div>
                <div class="metric-card">
                  <div class="text-xs mb-1" style="color: var(--warm-gray)">Sharpe Ratio</div>
                  <div class="text-2xl font-bold" style="color: var(--navy)">2.3</div>
                </div>
                <div class="metric-card">
                  <div class="text-xs mb-1" style="color: var(--warm-gray)">Win Rate</div>
                  <div class="text-2xl font-bold" style="color: var(--forest)">76%</div>
                  <div class="text-xs" style="color: var(--forest)">↑ +3% with CNN</div>
                </div>
                <div class="metric-card">
                  <div class="text-xs mb-1" style="color: var(--warm-gray)">Total Trades</div>
                  <div class="text-2xl font-bold" style="color: var(--dark-brown)">247</div>
                </div>
              </div>
              <div style="height: 300px; position: relative;">
                <canvas id="equity-curve-chart"></canvas>
              </div>
            </div>

            <!-- Signal Attribution -->
            <div class="card">
              <h3 class="text-xl font-bold mb-4" style="color: var(--navy)">
                <i class="fas fa-layer-group mr-2"></i>Ensemble Signal Attribution
              </h3>
              <div style="height: 200px; position: relative;">
                <canvas id="attribution-chart"></canvas>
              </div>
              <p class="text-xs mt-4" style="color: var(--warm-gray)">
                📚 Weighted ensemble: Cross-exchange (35%), CNN patterns (25%), Sentiment (20%), Economic (10%), On-chain (10%)
              </p>
            </div>
          </div>

          <!-- Opportunities Tab -->
          <div id="opportunities-tab" class="tab-content hidden">
            <div class="card mb-6">
              <div class="flex justify-between items-center mb-6">
                <h2 class="text-2xl font-bold" style="color: var(--navy)">Live Arbitrage Opportunities</h2>
                <div class="flex items-center gap-4">
                  <select class="px-4 py-2 border-2 rounded-lg" style="border-color: var(--cream-300)">
                    <option>All Strategies</option>
                    <option>Spatial Arbitrage</option>
                    <option>Triangular Arbitrage</option>
                    <option>Statistical Arbitrage</option>
                    <option>Funding Rate</option>
                  </select>
                  <button class="btn-primary">
                    <i class="fas fa-filter mr-2"></i>Filters
                  </button>
                </div>
              </div>
              <div id="all-opportunities-table"></div>
            </div>
          </div>

          <!-- Strategies Tab -->
          <div id="strategies-tab" class="tab-content hidden">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              <div class="card">
                <h3 class="text-xl font-bold mb-4" style="color: var(--navy)">
                  Multi-Strategy Performance Comparison
                </h3>
                <div style="height: 300px; position: relative;">
                  <canvas id="strategy-performance-chart"></canvas>
                </div>
              </div>
              <div class="card">
                <h3 class="text-xl font-bold mb-4" style="color: var(--navy)">
                  Risk-Return Analysis
                </h3>
                <div style="height: 300px; position: relative;">
                  <canvas id="risk-return-chart"></canvas>
                </div>
              </div>
            </div>
            
            <div class="card mb-8">
              <h3 class="text-xl font-bold mb-4" style="color: var(--navy)">
                Strategy Ranking Evolution
              </h3>
              <div style="height: 300px; position: relative;">
                <canvas id="ranking-chart"></canvas>
              </div>
            </div>
          </div>

          <!-- Backtest Tab -->
          <div id="backtest-tab" class="tab-content hidden">
            <div class="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
              <div class="card">
                <h3 class="text-lg font-bold mb-4" style="color: var(--navy)">
                  ⚙️ Backtest Configuration
                </h3>
                
                <div class="space-y-4">
                  <div>
                    <label class="block text-sm font-semibold mb-2">Strategy Selection</label>
                    <select id="backtest-strategy" class="w-full border-2 rounded-lg px-3 py-2" style="border-color: var(--cream-300)">
                      <option>All Strategies</option>
                      <option>Spatial Arbitrage</option>
                      <option>Triangular Arbitrage</option>
                      <option>Statistical Arbitrage</option>
                      <option>Funding Rate Arbitrage</option>
                    </select>
                  </div>
                  
                  <div>
                    <label class="block text-sm font-semibold mb-2">Date Range</label>
                    <select id="backtest-range" class="w-full border-2 rounded-lg px-3 py-2" style="border-color: var(--cream-300)">
                      <option>Last 30 Days</option>
                      <option>Last 90 Days</option>
                      <option>Last 6 Months</option>
                    </select>
                  </div>
                  
                  <div class="p-4 rounded-lg" style="background: var(--cream-100); border: 2px solid var(--navy)">
                    <h4 class="text-sm font-bold mb-3" style="color: var(--navy)">
                      🧠 CNN Pattern Recognition
                    </h4>
                    
                    <div class="space-y-3">
                      <div class="flex items-center justify-between">
                        <label class="text-sm">Enable CNN</label>
                        <input type="checkbox" id="enable-cnn" checked class="w-5 h-5">
                      </div>
                      
                      <div class="flex items-center justify-between">
                        <label class="text-sm">Sentiment Boost</label>
                        <input type="checkbox" id="enable-sentiment" checked class="w-5 h-5">
                      </div>
                      
                      <div>
                        <label class="block text-sm mb-1">Min Confidence</label>
                        <input type="range" id="min-confidence" min="50" max="95" value="75" class="w-full">
                        <div class="flex justify-between text-xs" style="color: var(--warm-gray)">
                          <span>50%</span>
                          <span id="confidence-value">75%</span>
                          <span>95%</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  <div class="p-4 rounded-lg" style="background: rgba(192, 127, 57, 0.1); border: 2px solid var(--burnt)">
                    <h4 class="text-sm font-bold mb-2" style="color: var(--burnt)">
                      🔬 A/B Testing Mode
                    </h4>
                    <p class="text-xs mb-3" style="color: var(--warm-gray)">
                      Compare performance with vs without CNN
                    </p>
                    <button onclick="runABTest()" class="w-full py-2 rounded-lg text-white font-semibold" style="background: var(--burnt)">
                      Run A/B Comparison
                    </button>
                  </div>
                  
                  <button onclick="runBacktest()" class="btn-primary w-full">
                    🚀 Run Backtest
                  </button>
                </div>
              </div>
              
              <div class="lg:col-span-2">
                <div id="backtest-results" class="card">
                  <h3 class="text-xl font-bold mb-4" style="color: var(--navy)">
                    📊 Backtest Results
                  </h3>
                  <div class="text-center py-12" style="color: var(--warm-gray)">
                    <i class="fas fa-chart-line text-6xl mb-4"></i>
                    <p>Configure parameters and run backtest to see results</p>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <!-- Analytics Tab -->
          <div id="analytics-tab" class="tab-content hidden">
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              <div class="card">
                <h3 class="text-xl font-bold mb-4" style="color: var(--navy)">
                  🎯 ML + CNN Prediction Accuracy
                </h3>
                <div style="height: 300px; position: relative;">
                  <canvas id="prediction-accuracy-chart"></canvas>
                </div>
              </div>
              
              <div class="card">
                <h3 class="text-xl font-bold mb-4" style="color: var(--navy)">
                  🔥 Pattern Success by Sentiment
                </h3>
                <div id="sentiment-pattern-heatmap"></div>
              </div>
            </div>
            
            <div class="card mb-8">
              <h3 class="text-xl font-bold mb-4" style="color: var(--navy)">
                📊 CNN Pattern Detection Timeline
              </h3>
              <div id="pattern-timeline" style="height: 250px; position: relative; border: 2px solid var(--cream-300); border-radius: 8px; padding: 20px;"></div>
              <div class="grid grid-cols-3 gap-4 mt-6">
                <div class="metric-card">
                  <div class="text-xs mb-1" style="color: var(--warm-gray)">Patterns Detected</div>
                  <div class="text-2xl font-bold" style="color: var(--navy)">87</div>
                </div>
                <div class="metric-card">
                  <div class="text-xs mb-1" style="color: var(--warm-gray)">Pattern Win Rate</div>
                  <div class="text-2xl font-bold" style="color: var(--forest)">78%</div>
                </div>
                <div class="metric-card">
                  <div class="text-xs mb-1" style="color: var(--warm-gray)">Avg Confidence</div>
                  <div class="text-2xl font-bold" style="color: var(--navy)">82%</div>
                </div>
              </div>
            </div>
            
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
              <div class="card">
                <h3 class="text-xl font-bold mb-4" style="color: var(--navy)">
                  Strategy Correlation Heatmap
                </h3>
                <div style="height: 300px; position: relative;">
                  <canvas id="correlation-heatmap"></canvas>
                </div>
              </div>
              
              <div class="card">
                <h3 class="text-xl font-bold mb-4" style="color: var(--navy)">
                  Drawdown Comparison
                </h3>
                <div style="height: 300px; position: relative;">
                  <canvas id="drawdown-chart"></canvas>
                </div>
              </div>
            </div>
            
            <div class="card">
              <h3 class="text-xl font-bold mb-4" style="color: var(--navy)">
                📚 Academic Research Foundations
              </h3>
              <p class="text-sm mb-4" style="color: var(--warm-gray)">
                All algorithms and weightings are backed by peer-reviewed academic research:
              </p>
              <div id="academic-references" class="space-y-4"></div>
            </div>
          </div>
        </main>

        <!-- Footer -->
        <footer class="border-t-2 mt-12 py-8" style="border-color: var(--cream-300); background: white;">
          <div class="container mx-auto px-6">
            <div class="grid grid-cols-1 md:grid-cols-3 gap-8">
              <div>
                <h4 class="font-bold mb-3" style="color: var(--navy)">⚠️ Risk Disclaimer</h4>
                <p class="text-xs" style="color: var(--warm-gray)">
                  This is an educational demonstration platform. Cryptocurrency trading involves substantial risk of loss. Past performance does not indicate future results. All data is simulated.
                </p>
              </div>
              <div>
                <h4 class="font-bold mb-3" style="color: var(--navy)">💰 Cost Analysis</h4>
                <p class="text-xs" style="color: var(--warm-gray)">
                  <strong>Total Cost:</strong> $468/mo (APIs: $278, GPU: $110, Infrastructure: $80)<br>
                  <strong>vs Traditional:</strong> $2,860/mo<br>
                  <strong>Savings:</strong> 83.6% ($2,392/mo)
                </p>
              </div>
              <div>
                <h4 class="font-bold mb-3" style="color: var(--navy)">📊 Platform Info</h4>
                <p class="text-xs" style="color: var(--warm-gray)">
                  <strong>Architecture:</strong> 3-Tier Hybrid (API + CNN + Ensemble)<br>
                  <strong>CNN Accuracy:</strong> 78% (vs 71% baseline)<br>
                  <strong>Update Frequency:</strong> Real-time (4s refresh)
                </p>
              </div>
            </div>
            <div class="text-center mt-8 pt-8 border-t-2" style="border-color: var(--cream-300)">
              <p class="text-xs" style="color: var(--warm-gray)">
                © 2024 ArbitrageAI | Educational Platform | Built with Hono + Cloudflare Pages
              </p>
            </div>
          </div>
        </footer>

        <script src="https://cdn.jsdelivr.net/npm/axios@1.6.0/dist/axios.min.js"></script>
        <script src="/static/app.js"></script>
    </body>
    </html>
  `)
})

// Data generation functions
function generateEconomicData() {
  return {
    score: Math.round(45 + Math.random() * 20),
    fedRate: 4.09,
    cpi: 3.2,
    gdp: 3.1,
    pmi: 48.5,
    unemployment: 3.7,
    policyStance: 'NEUTRAL',
    cryptoOutlook: 'NEUTRAL',
    lastUpdate: new Date().toISOString()
  }
}

function generateSentimentData() {
  const fearGreed = Math.round(Math.random() * 100)
  const googleTrends = Math.round(40 + Math.random() * 30)
  const vix = 18.45
  
  const score = Math.round(
    (googleTrends * 0.60) +
    (fearGreed * 0.25) +
    ((100 - vix * 2) * 0.15)
  )
  
  return {
    score,
    fearGreed,
    googleTrends,
    vix,
    signal: score < 40 ? 'BEARISH' : score > 60 ? 'BULLISH' : 'NEUTRAL',
    fearGreedLevel: fearGreed < 25 ? 'EXTREME FEAR' : 
                    fearGreed < 45 ? 'FEAR' :
                    fearGreed < 55 ? 'NEUTRAL' :
                    fearGreed < 75 ? 'GREED' : 'EXTREME GREED',
    lastUpdate: new Date().toISOString()
  }
}

function generateCrossExchangeData() {
  const basePrice = 94000 + (Math.random() - 0.5) * 1000
  const spread = 0.15 + Math.random() * 0.25
  
  return {
    vwap: Math.round(basePrice),
    bestBid: Math.round(basePrice - 50),
    bestAsk: Math.round(basePrice + 150),
    spread: spread.toFixed(3),
    buyExchange: 'Kraken',
    sellExchange: 'Coinbase',
    liquidityScore: Math.round(70 + Math.random() * 25),
    liquidityRating: 'good',
    marketEfficiency: 'Efficient',
    lastUpdate: new Date().toISOString()
  }
}

function generateOnChainData() {
  return {
    score: Math.round(55 + Math.random() * 15),
    exchangeNetflow: -5200,
    sopr: 0.97,
    mvrv: 1.8,
    activeAddresses: 920000,
    whaleActivity: 'HIGH',
    networkHealth: 'STRONG',
    signal: 'BULLISH',
    lastUpdate: new Date().toISOString()
  }
}

function generateCNNPatternData() {
  const patterns = ['Head & Shoulders', 'Double Top', 'Bull Flag', 'Bear Flag', 'Triangle Breakout']
  const pattern = patterns[Math.floor(Math.random() * patterns.length)]
  const isBearish = pattern.includes('Head') || pattern.includes('Double Top') || pattern.includes('Bear')
  const baseConfidence = 0.75 + Math.random() * 0.20
  const sentimentBoost = isBearish ? 1.30 : 1.20
  const reinforcedConfidence = Math.min(0.99, baseConfidence * sentimentBoost)
  
  return {
    pattern,
    direction: isBearish ? 'bearish' : 'bullish',
    baseConfidence: (baseConfidence * 100).toFixed(0),
    reinforcedConfidence: (reinforcedConfidence * 100).toFixed(0),
    sentimentMultiplier: sentimentBoost,
    targetPrice: isBearish ? 92340 : 96780,
    timeframe: '1h',
    chartImage: 'data:image/svg+xml;base64,...', // Would be actual chart image
    lastUpdate: new Date().toISOString()
  }
}

function generateCompositeSignal() {
  const crossExchangeContrib = 24.5
  const cnnContrib = 22.8
  const sentimentContrib = 13.6
  const economicContrib = 5.2
  const onChainContrib = 6.4
  
  const compositeScore = Math.round(
    crossExchangeContrib + cnnContrib + sentimentContrib + economicContrib + onChainContrib
  )
  
  return {
    compositeScore,
    signal: compositeScore > 70 ? 'STRONG_BUY' :
            compositeScore > 55 ? 'BUY' :
            compositeScore > 45 ? 'NEUTRAL' :
            compositeScore > 30 ? 'SELL' : 'STRONG_SELL',
    confidence: 85,
    contributions: {
      crossExchange: crossExchangeContrib,
      cnnPattern: cnnContrib,
      sentiment: sentimentContrib,
      economic: economicContrib,
      onChain: onChainContrib
    },
    riskVetos: [],
    executeRecommendation: compositeScore > 65,
    lastUpdate: new Date().toISOString()
  }
}

function generateOpportunities() {
  return [
    {
      id: 1,
      timestamp: new Date(Date.now() - 60000).toISOString(),
      strategy: 'Spatial',
      buyExchange: 'Kraken',
      sellExchange: 'Coinbase',
      spread: 0.31,
      netProfit: 0.18,
      mlConfidence: 78,
      cnnConfidence: 87,
      constraintsPassed: true
    },
    {
      id: 2,
      timestamp: new Date(Date.now() - 120000).toISOString(),
      strategy: 'Funding Rate',
      buyExchange: 'Binance',
      sellExchange: 'Binance Perp',
      spread: 0.25,
      netProfit: 0.19,
      mlConfidence: 82,
      cnnConfidence: null,
      constraintsPassed: true
    },
    {
      id: 3,
      timestamp: new Date(Date.now() - 180000).toISOString(),
      strategy: 'Statistical',
      buyExchange: 'BTC',
      sellExchange: 'ETH',
      spread: 0.42,
      netProfit: 0.28,
      mlConfidence: 71,
      cnnConfidence: 85,
      constraintsPassed: true
    }
  ]
}

function generateBacktestData(withCNN: boolean) {
  if (withCNN) {
    return {
      totalReturn: 14.8,
      sharpe: 2.3,
      winRate: 76,
      maxDrawdown: 3.2,
      totalTrades: 247,
      avgProfit: 0.059
    }
  } else {
    return {
      totalReturn: 12.4,
      sharpe: 2.1,
      winRate: 73,
      maxDrawdown: 4.1,
      totalTrades: 241,
      avgProfit: 0.051
    }
  }
}

function generatePatternTimeline() {
  const patterns = []
  const now = Date.now()
  
  for (let i = 0; i < 20; i++) {
    const timestamp = now - (i * 3600000 * 6) // Every 6 hours
    const patternNames = ['Head & Shoulders', 'Bull Flag', 'Double Top', 'Bear Flag']
    const pattern = patternNames[Math.floor(Math.random() * patternNames.length)]
    const isBearish = pattern.includes('Head') || pattern.includes('Double Top') || pattern.includes('Bear')
    
    patterns.push({
      timestamp,
      pattern,
      direction: isBearish ? 'bearish' : 'bullish',
      confidence: 75 + Math.random() * 20,
      tradeExecuted: Math.random() > 0.3,
      tradeProfit: Math.random() > 0.25 ? (Math.random() * 0.5) : -(Math.random() * 0.3)
    })
  }
  
  return patterns
}

export default app
