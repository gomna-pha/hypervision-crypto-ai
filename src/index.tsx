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

// LLM Strategic Insights API - Calls real LLM with all agent data
app.post('/api/llm/insights', async (c) => {
  try {
    const startTime = Date.now()
    
    // Gather all agent data
    const agentData = {
      economic: generateEconomicData(),
      sentiment: generateSentimentData(),
      crossExchange: generateCrossExchangeData(),
      onChain: generateOnChainData(),
      cnnPattern: generateCNNPatternData(),
      composite: generateCompositeSignal()
    }
    
    // Construct comprehensive prompt for LLM
    const prompt = `You are a senior quantitative analyst at a top-tier hedge fund. Analyze the following real-time cryptocurrency market data from multiple specialized agents and provide strategic trading insights.

**AGENT DATA:**

**Economic Agent (Macro Environment):**
- Score: ${agentData.economic.score}/100
- Fed Rate: ${agentData.economic.fedRate}%
- CPI Inflation: ${agentData.economic.cpi}%
- GDP Growth: ${agentData.economic.gdp}%
- PMI: ${agentData.economic.pmi}
- Policy Stance: ${agentData.economic.policyStance}
- Crypto Outlook: ${agentData.economic.cryptoOutlook}

**Sentiment Agent (Market Psychology):**
- Composite Score: ${agentData.sentiment.score}/100
- Fear & Greed Index: ${agentData.sentiment.fearGreed}/100 (${agentData.sentiment.fearGreedLevel})
- Google Trends: ${agentData.sentiment.googleTrends}/100
- VIX (Volatility): ${agentData.sentiment.vix}
- Signal: ${agentData.sentiment.signal}

**Cross-Exchange Agent (Price Arbitrage):**
- VWAP: $${agentData.crossExchange.vwap.toLocaleString()}
- Spread: ${agentData.crossExchange.spread}%
- Best Bid: $${agentData.crossExchange.bestBid.toLocaleString()}
- Best Ask: $${agentData.crossExchange.bestAsk.toLocaleString()}
- Buy Exchange: ${agentData.crossExchange.buyExchange}
- Sell Exchange: ${agentData.crossExchange.sellExchange}
- Liquidity Score: ${agentData.crossExchange.liquidityScore}/100

**On-Chain Agent (Blockchain Metrics):**
- Score: ${agentData.onChain.score}/100
- Exchange Netflow: ${agentData.onChain.exchangeNetflow.toLocaleString()} BTC
- SOPR: ${agentData.onChain.sopr}
- MVRV Ratio: ${agentData.onChain.mvrv}
- Active Addresses: ${agentData.onChain.activeAddresses.toLocaleString()}
- Whale Activity: ${agentData.onChain.whaleActivity}
- Network Health: ${agentData.onChain.networkHealth}
- Signal: ${agentData.onChain.signal}

**CNN Pattern Recognition Agent (Technical Analysis):**
- Detected Pattern: ${agentData.cnnPattern.pattern}
- Direction: ${agentData.cnnPattern.direction}
- Base Confidence: ${agentData.cnnPattern.baseConfidence}%
- Sentiment-Reinforced Confidence: ${agentData.cnnPattern.reinforcedConfidence}%
- Sentiment Multiplier: ${agentData.cnnPattern.sentimentMultiplier}x
- Target Price: $${agentData.cnnPattern.targetPrice.toLocaleString()}

**Composite Ensemble Signal:**
- Overall Score: ${agentData.composite.compositeScore}/100
- Signal: ${agentData.composite.signal}
- Confidence: ${agentData.composite.confidence}%
- Execute Recommendation: ${agentData.composite.executeRecommendation ? 'YES' : 'NO'}

**YOUR TASK:**
Provide a comprehensive strategic analysis in the following format:

1. **Market Context** (2-3 sentences): What's the current macro environment telling us?

2. **Key Insights** (3-4 bullet points): What are the most critical signals from the agents? Focus on agreement/disagreement between agents.

3. **Arbitrage Opportunity Assessment** (2-3 sentences): Given the cross-exchange spread and CNN pattern, is there a viable arbitrage opportunity?

4. **Risk Factors** (2-3 bullet points): What risks should traders be aware of? Consider sentiment extremes, liquidity, volatility.

5. **Strategic Recommendation** (2-3 sentences): Clear actionable advice - BUY/SELL/HOLD with reasoning. Include position sizing suggestion (conservative/moderate/aggressive).

6. **Timeframe** (1 sentence): What's the expected holding period for this recommendation?

Be concise, professional, and data-driven. Use financial terminology. This is real money at stake.`

    // Call OpenRouter API (supports multiple LLM providers)
    const response = await fetch('https://openrouter.ai/api/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${c.env?.OPENROUTER_API_KEY || 'sk-or-v1-demo-key'}`,
        'HTTP-Referer': 'https://arbitrageai.pages.dev',
        'X-Title': 'ArbitrageAI Platform',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: 'openai/gpt-4o-mini', // Fast and cost-effective
        messages: [
          {
            role: 'system',
            content: 'You are a senior quantitative analyst specializing in cryptocurrency arbitrage trading. Provide concise, actionable insights based on multi-agent data analysis.'
          },
          {
            role: 'user',
            content: prompt
          }
        ],
        temperature: 0.7,
        max_tokens: 800
      })
    })

    if (!response.ok) {
      throw new Error(`LLM API error: ${response.status}`)
    }

    const data = await response.json()
    const insights = data.choices[0].message.content
    const responseTime = Date.now() - startTime

    return c.json({
      success: true,
      insights,
      metadata: {
        model: 'gpt-4o-mini',
        responseTime: `${responseTime}ms`,
        timestamp: new Date().toISOString(),
        agentData // Include raw data for debugging
      }
    })
  } catch (error) {
    console.error('LLM Insights Error:', error)
    
    // Fallback to intelligent templated response if API fails
    const fallbackInsights = generateFallbackInsights()
    
    return c.json({
      success: false,
      insights: fallbackInsights,
      metadata: {
        model: 'fallback-template',
        responseTime: '50ms',
        timestamp: new Date().toISOString(),
        error: 'LLM API unavailable - using fallback analysis'
      }
    })
  }
})

// Execute Arbitrage Opportunity API
app.post('/api/execute/:id', async (c) => {
  const oppId = parseInt(c.req.param('id'))
  const startTime = Date.now()
  
  try {
    // Simulate execution time (real implementation would call exchange APIs)
    await new Promise(resolve => setTimeout(resolve, 1500))
    
    // Get opportunity details
    const opportunities = generateOpportunities()
    const opportunity = opportunities.find(o => o.id === oppId)
    
    if (!opportunity) {
      return c.json({
        success: false,
        error: 'Opportunity not found'
      }, 404)
    }
    
    // Calculate actual profit (with slippage simulation)
    const slippage = 0.05 + Math.random() * 0.10 // 0.05-0.15% slippage
    const actualNetProfit = Math.max(0.05, opportunity.netProfit - slippage)
    const positionSize = 10000 // $10k per trade
    const profit = (positionSize * actualNetProfit / 100)
    
    // Calculate execution time
    const executionTime = Date.now() - startTime
    
    // Log execution (in production, this would go to database)
    console.log(`[EXECUTION] Opportunity #${oppId} executed successfully`)
    console.log(`  Strategy: ${opportunity.strategy}`)
    console.log(`  Route: ${opportunity.buyExchange} → ${opportunity.sellExchange}`)
    console.log(`  Gross Spread: ${opportunity.spread}%`)
    console.log(`  Net Profit: ${actualNetProfit.toFixed(3)}%`)
    console.log(`  Position Size: $${positionSize}`)
    console.log(`  Realized Profit: $${profit.toFixed(2)}`)
    console.log(`  Execution Time: ${executionTime}ms`)
    console.log(`  ML Confidence: ${opportunity.mlConfidence}%`)
    console.log(`  CNN Confidence: ${opportunity.cnnConfidence || 'N/A'}%`)
    
    return c.json({
      success: true,
      opportunityId: oppId,
      strategy: opportunity.strategy,
      route: `${opportunity.buyExchange} → ${opportunity.sellExchange}`,
      grossSpread: opportunity.spread,
      slippage: slippage.toFixed(3),
      netProfit: actualNetProfit.toFixed(3),
      positionSize: positionSize,
      profit: profit.toFixed(2),
      executionTime: executionTime,
      timestamp: new Date().toISOString(),
      details: {
        buyExchange: opportunity.buyExchange,
        sellExchange: opportunity.sellExchange,
        buyPrice: 94000 - (Math.random() * 100),
        sellPrice: 94000 + (Math.random() * 100),
        volume: (positionSize / 94000).toFixed(6) + ' BTC',
        fees: {
          buy: (positionSize * 0.001).toFixed(2),
          sell: (positionSize * 0.001).toFixed(2),
          total: (positionSize * 0.002).toFixed(2)
        }
      }
    })
    
  } catch (error) {
    console.error('[EXECUTION ERROR]', error)
    
    return c.json({
      success: false,
      error: error.message || 'Execution failed',
      opportunityId: oppId,
      timestamp: new Date().toISOString()
    }, 500)
  }
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
          
          /* Prose styling for LLM insights */
          .prose {
            color: var(--dark-brown);
            line-height: 1.7;
          }
          
          .prose p {
            margin-bottom: 0.75rem;
            color: var(--warm-gray);
          }
          
          .prose h4 {
            margin-top: 1rem;
            margin-bottom: 0.5rem;
            font-weight: 700;
            color: var(--navy);
          }
          
          .prose strong {
            font-weight: 600;
            color: var(--navy);
          }
          
          .prose ul {
            list-style: none;
            padding-left: 0;
          }
          
          .prose li {
            margin-bottom: 0.5rem;
            padding-left: 1.5rem;
            position: relative;
          }
          
          .prose li:before {
            content: "•";
            position: absolute;
            left: 0.5rem;
            color: var(--burnt);
            font-weight: bold;
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
            <!-- Agent Dashboard Grid (3x2 + LLM Insights) -->
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

            <!-- LLM Strategic Insights (Full Width) -->
            <div class="card mb-8" style="border: 3px solid var(--burnt); background: linear-gradient(135deg, var(--cream-100) 0%, white 100%)">
              <div class="flex items-start justify-between mb-4">
                <div class="flex items-center gap-3">
                  <div class="w-12 h-12 rounded-lg flex items-center justify-center text-2xl" style="background: var(--burnt); color: white">
                    🧠
                  </div>
                  <div>
                    <h3 class="text-xl font-bold" style="color: var(--navy)">
                      LLM Strategic Analyst
                    </h3>
                    <p class="text-xs" style="color: var(--warm-gray)">
                      AI-powered holistic market analysis from all agent signals
                    </p>
                  </div>
                </div>
                <div class="flex items-center gap-2">
                  <button onclick="refreshLLMInsights()" class="px-4 py-2 rounded-lg text-sm font-semibold transition-all hover:opacity-80" style="background: var(--burnt); color: white">
                    <i class="fas fa-sync-alt mr-2"></i>Refresh Analysis
                  </button>
                  <div class="flex items-center gap-1">
                    <div class="w-2 h-2 rounded-full" id="llm-status-dot" style="background: var(--forest)"></div>
                    <span class="text-xs" style="color: var(--warm-gray)" id="llm-status-text">Ready</span>
                  </div>
                </div>
              </div>

              <div id="llm-insights-content" class="space-y-4">
                <div class="flex items-center justify-center py-8" style="color: var(--warm-gray)">
                  <i class="fas fa-spinner fa-spin text-3xl mr-3"></i>
                  <span>Initializing LLM analysis...</span>
                </div>
              </div>

              <div class="mt-4 pt-4 border-t-2 flex items-center justify-between text-xs" style="border-color: var(--cream-300); color: var(--warm-gray)">
                <div class="flex items-center gap-4">
                  <span><i class="fas fa-brain mr-1"></i>Model: <strong id="llm-model-name">GPT-4o-mini</strong></span>
                  <span><i class="fas fa-clock mr-1"></i>Last Updated: <strong id="llm-last-update">-</strong></span>
                  <span><i class="fas fa-bolt mr-1"></i>Response Time: <strong id="llm-response-time">-</strong></span>
                </div>
                <span class="text-xs">
                  <i class="fas fa-info-circle mr-1"></i>Non-hardcoded contextual insights
                </span>
              </div>
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

            <!-- Advanced Strategies Section -->
            <div class="mb-8">
              <h2 class="text-2xl font-bold mb-2" style="color: var(--navy)">
                🎯 Advanced Arbitrage Strategies
              </h2>
              <p class="text-sm mb-6" style="color: var(--warm-gray)">
                Multi-dimensional arbitrage detection including triangular, statistical, and funding rate opportunities
              </p>

              <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                <!-- Strategy 1: Advanced Arbitrage -->
                <div class="card">
                  <h3 class="text-lg font-bold mb-3" style="color: var(--navy)">
                    <i class="fas fa-exchange-alt mr-2"></i>Advanced Arbitrage
                  </h3>
                  <p class="text-xs mb-4" style="color: var(--warm-gray)">
                    Multi-dimensional arbitrage detection including triangular, statistical, and funding rate opportunities
                  </p>

                  <div class="space-y-2 mb-4">
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Spatial Arbitrage (Cross-Exchange)
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Triangular Arbitrage (BTC-ETH-USDT)
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Statistical Arbitrage (Mean Reversion)
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Funding Rate Arbitrage
                    </div>
                  </div>

                  <button onclick="detectArbitrageOpportunities()" class="btn-primary w-full mb-3">
                    <i class="fas fa-search mr-2"></i>Detect Opportunities
                  </button>

                  <div id="arbitrage-results" class="p-3 rounded-lg hidden" style="background: var(--cream-100)">
                    <div class="flex items-center justify-between mb-2">
                      <span class="text-xs font-semibold" style="color: var(--forest)">
                        <i class="fas fa-check-circle mr-1"></i>Found <span id="arb-count">0</span> Opportunities
                      </span>
                    </div>
                    <div class="text-xs space-y-1" style="color: var(--warm-gray)">
                      <div><strong>Spatial:</strong> <span id="spatial-count">0</span> opportunities</div>
                      <div><strong>Min profit threshold:</strong> 0.3% after fees</div>
                    </div>
                  </div>
                </div>

                <!-- Strategy 2: Statistical Pair Trading -->
                <div class="card">
                  <h3 class="text-lg font-bold mb-3" style="color: var(--navy)">
                    <i class="fas fa-chart-line mr-2"></i>Statistical Pair Trading
                  </h3>
                  <p class="text-xs mb-4" style="color: var(--warm-gray)">
                    Cointegration-based pairs trading with dynamic hedge ratios and mean reversion signals
                  </p>

                  <div class="space-y-2 mb-4">
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Cointegration Testing (ADF)
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Z-Score Signal Generation
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Kalman Filter Hedge Ratios
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Half-Life Estimation
                    </div>
                  </div>

                  <button onclick="analyzePairTrading()" class="btn-primary w-full mb-3">
                    <i class="fas fa-calculator mr-2"></i>Analyze BTC-ETH Pair
                  </button>

                  <div id="pair-trading-results" class="p-3 rounded-lg hidden" style="background: var(--cream-100)">
                    <div class="flex items-center justify-between mb-2">
                      <span class="text-xs font-semibold" style="color: var(--forest)">
                        <i class="fas fa-check-circle mr-1"></i>Signal: <span id="pair-signal">HOLD</span>
                      </span>
                    </div>
                    <div class="text-xs space-y-1" style="color: var(--warm-gray)">
                      <div><strong>Z-Score:</strong> <span id="z-score">0.50</span></div>
                      <div><strong>Cointegrated:</strong> <span id="cointegrated">Yes</span></div>
                      <div><strong>Half-Life:</strong> <span id="half-life">15</span> days</div>
                    </div>
                  </div>
                </div>

                <!-- Strategy 3: Multi-Factor Alpha -->
                <div class="card">
                  <h3 class="text-lg font-bold mb-3" style="color: var(--navy)">
                    <i class="fas fa-layer-group mr-2"></i>Multi-Factor Alpha
                  </h3>
                  <p class="text-xs mb-4" style="color: var(--warm-gray)">
                    Academic factor models including Fama-French 5-factor and Carhart 4-factor momentum
                  </p>

                  <div class="space-y-2 mb-4">
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Fama-French 5-Factor Model
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Carhart Momentum Factor
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Quality & Volatility Factors
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Composite Alpha Scoring
                    </div>
                  </div>

                  <button onclick="calculateAlphaScore()" class="btn-primary w-full mb-3">
                    <i class="fas fa-calculator mr-2"></i>Calculate Alpha Score
                  </button>

                  <div id="alpha-results" class="p-3 rounded-lg hidden" style="background: var(--cream-100)">
                    <div class="flex items-center justify-between mb-2">
                      <span class="text-xs font-semibold" style="color: var(--deep-red)">
                        <i class="fas fa-check-circle mr-1"></i>Signal: <span id="alpha-signal">SELL</span>
                      </span>
                    </div>
                    <div class="text-xs space-y-1" style="color: var(--warm-gray)">
                      <div><strong>Alpha Score:</strong> <span id="alpha-score">36</span>/100</div>
                      <div><strong>Dominant Factor:</strong> <span id="dominant-factor">market</span></div>
                      <div><strong>5-Factor + Momentum Analysis</strong></div>
                    </div>
                  </div>
                </div>

                <!-- Strategy 4: Machine Learning Ensemble -->
                <div class="card">
                  <h3 class="text-lg font-bold mb-3" style="color: var(--navy)">
                    <i class="fas fa-brain mr-2"></i>Machine Learning Ensemble
                  </h3>
                  <p class="text-xs mb-4" style="color: var(--warm-gray)">
                    Ensemble ML models with feature importance and SHAP value analysis
                  </p>

                  <div class="space-y-2 mb-4">
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Random Forest Classifier
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Gradient Boosting (XGBoost)
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Support Vector Machine
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Neural Network
                    </div>
                  </div>

                  <button onclick="generateMLPrediction()" class="btn-primary w-full mb-3">
                    <i class="fas fa-robot mr-2"></i>Generate ML Prediction
                  </button>

                  <div id="ml-results" class="p-3 rounded-lg hidden" style="background: var(--cream-100)">
                    <div class="flex items-center justify-between mb-2">
                      <span class="text-xs font-semibold" style="color: var(--deep-red)">
                        <i class="fas fa-check-circle mr-1"></i>Ensemble: <span id="ml-signal">SELL</span>
                      </span>
                    </div>
                    <div class="text-xs space-y-1" style="color: var(--warm-gray)">
                      <div><strong>Confidence:</strong> <span id="ml-confidence">40</span>%</div>
                      <div><strong>Model Agreement:</strong> <span id="ml-agreement">40</span>%</div>
                      <div><strong>5 models:</strong> RF, XGB, SVM, LR, NN</div>
                    </div>
                  </div>
                </div>

                <!-- Strategy 5: Deep Learning Models -->
                <div class="card">
                  <h3 class="text-lg font-bold mb-3" style="color: var(--navy)">
                    <i class="fas fa-network-wired mr-2"></i>Deep Learning Models
                  </h3>
                  <p class="text-xs mb-4" style="color: var(--warm-gray)">
                    Advanced neural networks including LSTM, Transformers, and GAN-based scenario generation
                  </p>

                  <div class="space-y-2 mb-4">
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      LSTM Time Series Forecasting
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Transformer Attention Models
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      GAN Scenario Generation
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      CNN Pattern Recognition
                    </div>
                  </div>

                  <button onclick="runDLAnalysis()" class="btn-primary w-full mb-3">
                    <i class="fas fa-microchip mr-2"></i>Run DL Analysis
                  </button>

                  <div id="dl-results" class="p-3 rounded-lg hidden" style="background: var(--cream-100)">
                    <div class="flex items-center justify-between mb-2">
                      <span class="text-xs font-semibold" style="color: var(--forest)">
                        <i class="fas fa-check-circle mr-1"></i>DL Signal: <span id="dl-signal">STRONG_BUY</span>
                      </span>
                    </div>
                    <div class="text-xs space-y-1" style="color: var(--warm-gray)">
                      <div><strong>Confidence:</strong> <span id="dl-confidence">78</span>%</div>
                      <div><strong>LSTM Trend:</strong> <span id="lstm-trend">upward</span></div>
                      <div><strong>LSTM + Transformer + GAN</strong></div>
                    </div>
                  </div>
                </div>

                <!-- Strategy 6: Strategy Comparison -->
                <div class="card">
                  <h3 class="text-lg font-bold mb-3" style="color: var(--navy)">
                    <i class="fas fa-balance-scale mr-2"></i>Strategy Comparison
                  </h3>
                  <p class="text-xs mb-4" style="color: var(--warm-gray)">
                    Compare all advanced strategies side-by-side with performance metrics
                  </p>

                  <div class="space-y-2 mb-4">
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Signal Consistency Analysis
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Risk-Adjusted Returns
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Correlation Matrix
                    </div>
                    <div class="flex items-center text-xs" style="color: var(--warm-gray)">
                      <i class="fas fa-check-circle mr-2" style="color: var(--forest)"></i>
                      Portfolio Optimization
                    </div>
                  </div>

                  <button onclick="compareAllStrategies()" class="btn-primary w-full mb-3">
                    <i class="fas fa-chart-bar mr-2"></i>Compare All Strategies
                  </button>

                  <div id="comparison-results" class="p-3 rounded-lg hidden" style="background: var(--cream-100)">
                    <div class="flex items-center justify-between mb-2">
                      <span class="text-xs font-semibold" style="color: var(--forest)">
                        <i class="fas fa-check-circle mr-1"></i>All Strategies Complete
                      </span>
                    </div>
                    <div class="text-xs" style="color: var(--warm-gray)">
                      Check results table below
                    </div>
                  </div>
                </div>
              </div>
            </div>

            <!-- Strategy Comparison Table -->
            <div class="card">
              <h3 class="text-xl font-bold mb-4" style="color: var(--navy)">
                📊 Strategy Signals & Performance Summary
              </h3>
              <div class="overflow-x-auto">
                <table class="w-full text-sm">
                  <thead>
                    <tr style="background: var(--cream-100)">
                      <th class="px-4 py-3 text-left font-semibold" style="color: var(--navy)">Strategy</th>
                      <th class="px-4 py-3 text-left font-semibold" style="color: var(--navy)">Signal</th>
                      <th class="px-4 py-3 text-left font-semibold" style="color: var(--navy)">Confidence</th>
                      <th class="px-4 py-3 text-left font-semibold" style="color: var(--navy)">30D Return</th>
                      <th class="px-4 py-3 text-left font-semibold" style="color: var(--navy)">Sharpe</th>
                      <th class="px-4 py-3 text-left font-semibold" style="color: var(--navy)">Win Rate</th>
                      <th class="px-4 py-3 text-left font-semibold" style="color: var(--navy)">Status</th>
                    </tr>
                  </thead>
                  <tbody id="strategy-comparison-table">
                    <tr style="border-bottom: 1px solid var(--cream-300)">
                      <td class="px-4 py-3 font-medium" style="color: var(--navy)">Advanced Arbitrage</td>
                      <td class="px-4 py-3"><span class="px-2 py-1 rounded text-xs font-semibold" style="background: rgba(45, 95, 63, 0.2); color: var(--forest)">BUY</span></td>
                      <td class="px-4 py-3">78%</td>
                      <td class="px-4 py-3" style="color: var(--forest)">+3.2%</td>
                      <td class="px-4 py-3">2.1</td>
                      <td class="px-4 py-3">72%</td>
                      <td class="px-4 py-3"><i class="fas fa-check-circle" style="color: var(--forest)"></i> Active</td>
                    </tr>
                    <tr style="border-bottom: 1px solid var(--cream-300)">
                      <td class="px-4 py-3 font-medium" style="color: var(--navy)">Statistical Pair Trading</td>
                      <td class="px-4 py-3"><span class="px-2 py-1 rounded text-xs font-semibold" style="background: rgba(107, 93, 79, 0.2); color: var(--warm-gray)">HOLD</span></td>
                      <td class="px-4 py-3">65%</td>
                      <td class="px-4 py-3" style="color: var(--forest)">+1.8%</td>
                      <td class="px-4 py-3">1.8</td>
                      <td class="px-4 py-3">68%</td>
                      <td class="px-4 py-3"><i class="fas fa-check-circle" style="color: var(--forest)"></i> Active</td>
                    </tr>
                    <tr style="border-bottom: 1px solid var(--cream-300)">
                      <td class="px-4 py-3 font-medium" style="color: var(--navy)">Multi-Factor Alpha</td>
                      <td class="px-4 py-3"><span class="px-2 py-1 rounded text-xs font-semibold" style="background: rgba(139, 58, 58, 0.2); color: var(--deep-red)">SELL</span></td>
                      <td class="px-4 py-3">52%</td>
                      <td class="px-4 py-3" style="color: var(--deep-red)">-0.8%</td>
                      <td class="px-4 py-3">1.2</td>
                      <td class="px-4 py-3">58%</td>
                      <td class="px-4 py-3"><i class="fas fa-pause-circle" style="color: var(--warm-gray)"></i> Monitoring</td>
                    </tr>
                    <tr style="border-bottom: 1px solid var(--cream-300)">
                      <td class="px-4 py-3 font-medium" style="color: var(--navy)">ML Ensemble</td>
                      <td class="px-4 py-3"><span class="px-2 py-1 rounded text-xs font-semibold" style="background: rgba(139, 58, 58, 0.2); color: var(--deep-red)">SELL</span></td>
                      <td class="px-4 py-3">40%</td>
                      <td class="px-4 py-3" style="color: var(--deep-red)">-1.2%</td>
                      <td class="px-4 py-3">0.9</td>
                      <td class="px-4 py-3">54%</td>
                      <td class="px-4 py-3"><i class="fas fa-pause-circle" style="color: var(--warm-gray)"></i> Monitoring</td>
                    </tr>
                    <tr style="border-bottom: 1px solid var(--cream-300)">
                      <td class="px-4 py-3 font-medium" style="color: var(--navy)">Deep Learning</td>
                      <td class="px-4 py-3"><span class="px-2 py-1 rounded text-xs font-semibold" style="background: rgba(45, 95, 63, 0.2); color: var(--forest)">STRONG_BUY</span></td>
                      <td class="px-4 py-3">78%</td>
                      <td class="px-4 py-3" style="color: var(--forest)">+4.5%</td>
                      <td class="px-4 py-3">2.6</td>
                      <td class="px-4 py-3">76%</td>
                      <td class="px-4 py-3"><i class="fas fa-check-circle" style="color: var(--forest)"></i> Active</td>
                    </tr>
                    <tr>
                      <td class="px-4 py-3 font-bold" style="color: var(--navy)">CNN-Enhanced Composite</td>
                      <td class="px-4 py-3"><span class="px-2 py-1 rounded text-xs font-semibold" style="background: rgba(45, 95, 63, 0.2); color: var(--forest)">STRONG_BUY</span></td>
                      <td class="px-4 py-3"><strong>85%</strong></td>
                      <td class="px-4 py-3 font-bold" style="color: var(--forest)">+5.8%</td>
                      <td class="px-4 py-3"><strong>2.9</strong></td>
                      <td class="px-4 py-3"><strong>79%</strong></td>
                      <td class="px-4 py-3"><i class="fas fa-star" style="color: var(--burnt)"></i> <strong>Primary</strong></td>
                    </tr>
                  </tbody>
                </table>
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

// Fallback LLM Insights Generator (used when API is unavailable)
function generateFallbackInsights() {
  const agentData = {
    economic: generateEconomicData(),
    sentiment: generateSentimentData(),
    crossExchange: generateCrossExchangeData(),
    onChain: generateOnChainData(),
    cnnPattern: generateCNNPatternData(),
    composite: generateCompositeSignal()
  }
  
  // Contextual analysis based on agent signals
  const sentiment = agentData.sentiment.signal
  const composite = agentData.composite.signal
  const spread = parseFloat(agentData.crossExchange.spread)
  const fearGreed = agentData.sentiment.fearGreedLevel
  const pattern = agentData.cnnPattern.pattern
  const direction = agentData.cnnPattern.direction
  
  // Market Context
  let marketContext = ''
  if (agentData.economic.score > 60) {
    marketContext = `The macro environment shows ${agentData.economic.policyStance.toLowerCase()} monetary policy with Fed rates at ${agentData.economic.fedRate}%. Economic indicators suggest a ${agentData.economic.cryptoOutlook.toLowerCase()} outlook for risk assets including cryptocurrencies.`
  } else {
    marketContext = `Current macro headwinds with ${agentData.economic.policyStance.toLowerCase()} policy stance. PMI at ${agentData.economic.pmi} indicates economic uncertainty, creating volatility in crypto markets.`
  }
  
  // Key Insights
  const insights = []
  
  if (Math.abs(agentData.sentiment.score - agentData.composite.compositeScore) < 15) {
    insights.push(`✅ **Strong Agent Consensus**: Sentiment (${agentData.sentiment.score}/100) and Composite Signal (${agentData.composite.compositeScore}/100) are aligned, suggesting reliable directional bias.`)
  } else {
    insights.push(`⚠️ **Agent Divergence**: Sentiment diverges from composite signal - suggests mixed market conditions requiring cautious positioning.`)
  }
  
  insights.push(`📊 **CNN Pattern Detection**: ${pattern} pattern identified with ${agentData.cnnPattern.reinforcedConfidence}% confidence (${direction}). Sentiment reinforcement ${agentData.cnnPattern.sentimentMultiplier > 1.2 ? 'boosting' : 'maintaining'} signal strength.`)
  
  if (agentData.onChain.exchangeNetflow < -3000) {
    insights.push(`🐋 **On-Chain Bullish**: Exchange outflows of ${Math.abs(agentData.onChain.exchangeNetflow).toLocaleString()} BTC indicate accumulation. Whale activity ${agentData.onChain.whaleActivity.toLowerCase()} with ${agentData.onChain.whaleActivity === 'HIGH' ? 'strong' : 'moderate'} conviction.`)
  } else {
    insights.push(`📉 **On-Chain Neutral**: Exchange flows showing distribution. MVRV at ${agentData.onChain.mvrv} suggests ${agentData.onChain.mvrv > 2 ? 'overvalued' : 'fair value'} territory.`)
  }
  
  if (spread > 0.25) {
    insights.push(`💰 **Arbitrage Window Open**: Cross-exchange spread at ${spread}% exceeds profit threshold. Buy ${agentData.crossExchange.buyExchange}, sell ${agentData.crossExchange.sellExchange}.`)
  }
  
  // Arbitrage Assessment
  let arbAssessment = ''
  if (spread > 0.25 && agentData.crossExchange.liquidityScore > 70) {
    arbAssessment = `Spatial arbitrage opportunity exists with ${spread}% spread between ${agentData.crossExchange.buyExchange} and ${agentData.crossExchange.sellExchange}. Liquidity score of ${agentData.crossExchange.liquidityScore}/100 supports execution. Combined with ${direction} CNN pattern, this creates a favorable entry point.`
  } else if (spread > 0.15) {
    arbAssessment = `Marginal arbitrage spread at ${spread}%. While technically profitable, execution risk and slippage may compress net returns. Consider funding rate arbitrage as alternative.`
  } else {
    arbAssessment = `Current spread of ${spread}% is below profitable threshold after fees. Market efficiency high. Focus on statistical arbitrage and pattern-based entries instead.`
  }
  
  // Risk Factors
  const risks = []
  
  if (fearGreed.includes('EXTREME')) {
    risks.push(`🔴 **Sentiment Extreme**: Fear & Greed at ${fearGreed} - historically precedes mean reversion. Reduce position sizes.`)
  }
  
  if (agentData.sentiment.vix > 25) {
    risks.push(`⚡ **High Volatility**: VIX at ${agentData.sentiment.vix} suggests elevated market stress. Widen stop-losses and consider delta-hedging.`)
  } else {
    risks.push(`✅ **Volatility Contained**: VIX at ${agentData.sentiment.vix} within normal range. Standard risk management applies.`)
  }
  
  if (agentData.crossExchange.liquidityScore < 60) {
    risks.push(`💧 **Liquidity Concerns**: Cross-exchange liquidity at ${agentData.crossExchange.liquidityScore}/100 may cause slippage. Scale entry/exit.`)
  }
  
  // Strategic Recommendation
  let recommendation = ''
  let positionSize = 'moderate'
  
  if (composite.includes('STRONG_BUY') && agentData.composite.confidence > 80) {
    recommendation = `**BUY Signal** - High-conviction setup with ${agentData.composite.confidence}% ensemble confidence. Multiple agents confirm bullish bias with ${pattern} pattern targeting $${agentData.cnnPattern.targetPrice.toLocaleString()}.`
    positionSize = 'moderate to aggressive'
  } else if (composite.includes('BUY')) {
    recommendation = `**BUY Signal** - Moderate conviction at ${agentData.composite.confidence}% confidence. Enter with ${positionSize} position sizing, targeting ${pattern} completion levels.`
  } else if (composite.includes('SELL')) {
    recommendation = `**SELL Signal** - Composite analysis suggests ${direction} pressure. Consider closing longs or initiating hedges with ${positionSize} position sizing.`
    positionSize = 'conservative'
  } else {
    recommendation = `**HOLD/NEUTRAL** - Mixed signals across agents. Maintain current positions but avoid new entries until clearer directional bias emerges.`
    positionSize = 'conservative'
  }
  
  recommendation += ` Position sizing: ${positionSize.toUpperCase()}.`
  
  // Timeframe
  let timeframe = ''
  if (pattern.includes('Flag') || pattern.includes('Triangle')) {
    timeframe = `**Expected Timeframe**: Short-term (1-3 days) - continuation patterns typically resolve quickly.`
  } else if (pattern.includes('Head') || pattern.includes('Double')) {
    timeframe = `**Expected Timeframe**: Medium-term (3-7 days) - reversal patterns require confirmation over multiple sessions.`
  } else {
    timeframe = `**Expected Timeframe**: Intraday to short-term (hours to 2 days) based on ${pattern} dynamics and current volatility regime.`
  }
  
  // Construct formatted output
  return `**1. Market Context**
${marketContext}

**2. Key Insights**
${insights.map(i => `${i}`).join('\n')}

**3. Arbitrage Opportunity Assessment**
${arbAssessment}

**4. Risk Factors**
${risks.map(r => `${r}`).join('\n')}

**5. Strategic Recommendation**
${recommendation}

**6. ${timeframe}**

---
*Note: This analysis uses template-based logic as LLM API is currently unavailable. For fully dynamic insights, configure OPENROUTER_API_KEY environment variable.*`
}

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
  const now = Date.now()
  
  return [
    // Core Spatial Arbitrage
    {
      id: 1,
      timestamp: new Date(now - 45000).toISOString(),
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
      timestamp: new Date(now - 90000).toISOString(),
      strategy: 'Spatial',
      buyExchange: 'Binance',
      sellExchange: 'Kraken',
      spread: 0.28,
      netProfit: 0.15,
      mlConfidence: 81,
      cnnConfidence: 89,
      constraintsPassed: true
    },
    
    // Triangular Arbitrage
    {
      id: 3,
      timestamp: new Date(now - 60000).toISOString(),
      strategy: 'Triangular',
      buyExchange: 'BTC-ETH-USDT',
      sellExchange: 'Binance',
      spread: 0.22,
      netProfit: 0.12,
      mlConfidence: 85,
      cnnConfidence: null,
      constraintsPassed: true
    },
    {
      id: 4,
      timestamp: new Date(now - 135000).toISOString(),
      strategy: 'Triangular',
      buyExchange: 'ETH-BTC-USDC',
      sellExchange: 'Coinbase',
      spread: 0.19,
      netProfit: 0.09,
      mlConfidence: 79,
      cnnConfidence: null,
      constraintsPassed: true
    },
    
    // Statistical Arbitrage (Pair Trading)
    {
      id: 5,
      timestamp: new Date(now - 120000).toISOString(),
      strategy: 'Statistical',
      buyExchange: 'BTC/ETH Pair',
      sellExchange: 'Mean Reversion',
      spread: 0.42,
      netProfit: 0.28,
      mlConfidence: 71,
      cnnConfidence: 85,
      constraintsPassed: true
    },
    {
      id: 6,
      timestamp: new Date(now - 180000).toISOString(),
      strategy: 'Statistical',
      buyExchange: 'SOL/AVAX Pair',
      sellExchange: 'Cointegration',
      spread: 0.38,
      netProfit: 0.24,
      mlConfidence: 68,
      cnnConfidence: 82,
      constraintsPassed: true
    },
    
    // Funding Rate Arbitrage
    {
      id: 7,
      timestamp: new Date(now - 75000).toISOString(),
      strategy: 'Funding Rate',
      buyExchange: 'Binance Spot',
      sellExchange: 'Binance Perp',
      spread: 0.25,
      netProfit: 0.19,
      mlConfidence: 82,
      cnnConfidence: null,
      constraintsPassed: true
    },
    {
      id: 8,
      timestamp: new Date(now - 150000).toISOString(),
      strategy: 'Funding Rate',
      buyExchange: 'OKX Spot',
      sellExchange: 'OKX Futures',
      spread: 0.21,
      netProfit: 0.16,
      mlConfidence: 77,
      cnnConfidence: null,
      constraintsPassed: true
    },
    
    // Multi-Factor Alpha Strategy
    {
      id: 9,
      timestamp: new Date(now - 105000).toISOString(),
      strategy: 'Multi-Factor Alpha',
      buyExchange: 'Fama-French 5F',
      sellExchange: 'Alpha Capture',
      spread: 0.35,
      netProfit: 0.21,
      mlConfidence: 73,
      cnnConfidence: 80,
      constraintsPassed: true
    },
    {
      id: 10,
      timestamp: new Date(now - 195000).toISOString(),
      strategy: 'Multi-Factor Alpha',
      buyExchange: 'Carhart 4F + Mom',
      sellExchange: 'Factor Portfolio',
      spread: 0.29,
      netProfit: 0.17,
      mlConfidence: 70,
      cnnConfidence: 78,
      constraintsPassed: true
    },
    
    // ML Ensemble Strategy
    {
      id: 11,
      timestamp: new Date(now - 165000).toISOString(),
      strategy: 'ML Ensemble',
      buyExchange: 'RF+XGB+SVM',
      sellExchange: 'Ensemble Signal',
      spread: 0.33,
      netProfit: 0.20,
      mlConfidence: 88,
      cnnConfidence: 91,
      constraintsPassed: true
    },
    {
      id: 12,
      timestamp: new Date(now - 210000).toISOString(),
      strategy: 'ML Ensemble',
      buyExchange: '5-Model Consensus',
      sellExchange: 'High Conviction',
      spread: 0.37,
      netProfit: 0.23,
      mlConfidence: 92,
      cnnConfidence: 94,
      constraintsPassed: true
    },
    
    // Deep Learning Strategy
    {
      id: 13,
      timestamp: new Date(now - 30000).toISOString(),
      strategy: 'Deep Learning',
      buyExchange: 'LSTM Forecast',
      sellExchange: 'Transformer',
      spread: 0.45,
      netProfit: 0.31,
      mlConfidence: 86,
      cnnConfidence: 93,
      constraintsPassed: true
    },
    {
      id: 14,
      timestamp: new Date(now - 225000).toISOString(),
      strategy: 'Deep Learning',
      buyExchange: 'GAN Scenario',
      sellExchange: 'CNN Pattern',
      spread: 0.41,
      netProfit: 0.27,
      mlConfidence: 84,
      cnnConfidence: 90,
      constraintsPassed: true
    },
    
    // Volatility Arbitrage
    {
      id: 15,
      timestamp: new Date(now - 195000).toISOString(),
      strategy: 'Volatility',
      buyExchange: 'Options Delta',
      sellExchange: 'Gamma Hedge',
      spread: 0.52,
      netProfit: 0.36,
      mlConfidence: 75,
      cnnConfidence: null,
      constraintsPassed: true
    },
    
    // Cross-Asset Arbitrage
    {
      id: 16,
      timestamp: new Date(now - 270000).toISOString(),
      strategy: 'Cross-Asset',
      buyExchange: 'BTC/Gold Ratio',
      sellExchange: 'Macro Hedge',
      spread: 0.48,
      netProfit: 0.33,
      mlConfidence: 69,
      cnnConfidence: 76,
      constraintsPassed: true
    },
    
    // High-Frequency Micro Arbitrage
    {
      id: 17,
      timestamp: new Date(now - 15000).toISOString(),
      strategy: 'HFT Micro',
      buyExchange: 'Latency Edge',
      sellExchange: 'Order Flow',
      spread: 0.15,
      netProfit: 0.08,
      mlConfidence: 94,
      cnnConfidence: null,
      constraintsPassed: true
    },
    
    // Market Making Arbitrage
    {
      id: 18,
      timestamp: new Date(now - 240000).toISOString(),
      strategy: 'Market Making',
      buyExchange: 'Bid-Ask Spread',
      sellExchange: 'Inventory Risk',
      spread: 0.26,
      netProfit: 0.14,
      mlConfidence: 80,
      cnnConfidence: null,
      constraintsPassed: true
    },
    
    // Seasonal/Calendar Arbitrage
    {
      id: 19,
      timestamp: new Date(now - 285000).toISOString(),
      strategy: 'Seasonal',
      buyExchange: 'Monthly Pattern',
      sellExchange: 'Calendar Effect',
      spread: 0.34,
      netProfit: 0.22,
      mlConfidence: 66,
      cnnConfidence: 74,
      constraintsPassed: true
    },
    
    // Sentiment-Driven Arbitrage
    {
      id: 20,
      timestamp: new Date(now - 330000).toISOString(),
      strategy: 'Sentiment',
      buyExchange: 'Fear & Greed',
      sellExchange: 'Contrarian',
      spread: 0.39,
      netProfit: 0.25,
      mlConfidence: 72,
      cnnConfidence: 88,
      constraintsPassed: true
    }
  ].sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime()) // Sort by most recent first
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
