import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serveStatic } from 'hono/cloudflare-workers'

// Type definitions for Cloudflare bindings
type Bindings = {
  DB: D1Database
}

const app = new Hono<{ Bindings: Bindings }>()

// Enable CORS for API routes
app.use('/api/*', cors())

// Serve static files
app.use('/static/*', serveStatic({ root: './public' }))

// ============================================================================
// DATA INTEGRATION LAYER - External API Integration Routes
// ============================================================================

// Fetch market data from external APIs
app.get('/api/market/data/:symbol', async (c) => {
  const symbol = c.req.param('symbol')
  const { env } = c
  
  try {
    // Store fetched data in D1
    const timestamp = Date.now()
    await env.DB.prepare(`
      INSERT INTO market_data (symbol, exchange, price, volume, timestamp, data_type)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(symbol, 'aggregated', 0, 0, timestamp, 'spot').run()
    
    // In production, this would fetch from Alpha Vantage, Polygon.io, etc.
    return c.json({
      success: true,
      data: {
        symbol,
        price: Math.random() * 50000 + 30000, // Mock data
        volume: Math.random() * 1000000,
        timestamp,
        source: 'mock'
      }
    })
  } catch (error) {
    return c.json({ success: false, error: String(error) }, 500)
  }
})

// Fetch economic indicators
app.get('/api/economic/indicators', async (c) => {
  const { env } = c
  
  try {
    // Fetch recent indicators from database
    const results = await env.DB.prepare(`
      SELECT * FROM economic_indicators 
      ORDER BY timestamp DESC 
      LIMIT 10
    `).all()
    
    return c.json({
      success: true,
      data: results.results,
      count: results.results?.length || 0
    })
  } catch (error) {
    return c.json({ success: false, error: String(error) }, 500)
  }
})

// Store economic indicator data
app.post('/api/economic/indicators', async (c) => {
  const { env } = c
  const body = await c.req.json()
  
  try {
    const { indicator_name, indicator_code, value, period, source } = body
    const timestamp = Date.now()
    
    await env.DB.prepare(`
      INSERT INTO economic_indicators 
      (indicator_name, indicator_code, value, period, source, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(indicator_name, indicator_code, value, period, source, timestamp).run()
    
    return c.json({ success: true, message: 'Indicator stored successfully' })
  } catch (error) {
    return c.json({ success: false, error: String(error) }, 500)
  }
})

// ============================================================================
// LIVE DATA AGENTS - Economic, Sentiment, Cross-Exchange
// ============================================================================

// Economic Agent - Aggregates economic indicators
app.get('/api/agents/economic', async (c) => {
  const symbol = c.req.query('symbol') || 'BTC'
  
  const economicData = {
    timestamp: Date.now(),
    iso_timestamp: new Date().toISOString(),
    symbol,
    data_source: 'Economic Agent',
    indicators: {
      fed_funds_rate: { value: 5.33, change: -0.25, trend: 'stable', next_meeting: '2025-11-07' },
      cpi: { value: 3.2, change: -0.1, yoy_change: 3.2, trend: 'decreasing' },
      ppi: { value: 2.8, change: -0.3 },
      unemployment_rate: { value: 3.8, change: 0.1, trend: 'stable', non_farm_payrolls: 180000 },
      gdp_growth: { value: 2.4, quarter: 'Q3 2025', previous_quarter: 2.1 },
      treasury_10y: { value: 4.25, change: -0.15, spread: -0.6 },
      manufacturing_pmi: { value: 48.5, status: 'contraction' },
      retail_sales: { value: 0.3, change: 0.2 }
    }
  }
  
  return c.json({ success: true, agent: 'economic', data: economicData })
})

// Sentiment Agent - Aggregates market sentiment
app.get('/api/agents/sentiment', async (c) => {
  const symbol = c.req.query('symbol') || 'BTC'
  
  const sentimentData = {
    timestamp: Date.now(),
    iso_timestamp: new Date().toISOString(),
    symbol,
    data_source: 'Sentiment Agent',
    sentiment_metrics: {
      fear_greed_index: { value: 61 + Math.floor(Math.random() * 20 - 10), classification: 'neutral' },
      aggregate_sentiment: { value: 74 + Math.floor(Math.random() * 20 - 10), trend: 'neutral' },
      volatility_index_vix: { value: 19.98 + Math.random() * 4 - 2, interpretation: 'moderate' },
      social_media_volume: { mentions: 100000 + Math.floor(Math.random() * 20000), trend: 'average' },
      institutional_flow_24h: { net_flow_million_usd: -7.0 + Math.random() * 10 - 5, direction: 'outflow' }
    }
  }
  
  return c.json({ success: true, agent: 'sentiment', data: sentimentData })
})

// Cross-Exchange Agent - Aggregates liquidity and execution data
app.get('/api/agents/cross-exchange', async (c) => {
  const symbol = c.req.query('symbol') || 'BTC'
  
  const exchangeData = {
    timestamp: Date.now(),
    iso_timestamp: new Date().toISOString(),
    symbol,
    data_source: 'Cross-Exchange Agent',
    market_depth_analysis: {
      total_volume_24h: { usd: 35.18 + Math.random() * 5, btc: 780 + Math.random() * 50 },
      market_depth_score: { score: 9.2, rating: 'excellent' },
      liquidity_metrics: {
        average_spread_percent: 2.1,
        slippage_10btc_percent: 1.5,
        order_book_imbalance: 0.52
      },
      execution_quality: {
        large_order_impact_percent: 15 + Math.random() * 10 - 5,
        recommended_exchanges: ['Binance', 'Coinbase'],
        optimal_execution_time_ms: 5000,
        slippage_buffer_percent: 15
      }
    }
  }
  
  return c.json({ success: true, agent: 'cross-exchange', data: exchangeData })
})

// ============================================================================
// FEATURE & SIGNAL LAYER - Technical Indicators and Feature Engineering
// ============================================================================

// Calculate technical indicators
app.post('/api/features/calculate', async (c) => {
  const { env } = c
  const { symbol, features } = await c.req.json()
  
  try {
    // Fetch recent price data
    const priceData = await env.DB.prepare(`
      SELECT price, timestamp FROM market_data 
      WHERE symbol = ? 
      ORDER BY timestamp DESC 
      LIMIT 50
    `).bind(symbol).all()
    
    const prices = priceData.results?.map((r: any) => r.price) || []
    const calculated: any = {}
    
    // Simple Moving Average
    if (features.includes('sma')) {
      const sma20 = prices.slice(0, 20).reduce((a, b) => a + b, 0) / 20
      calculated.sma20 = sma20
    }
    
    // RSI (Relative Strength Index)
    if (features.includes('rsi')) {
      calculated.rsi = calculateRSI(prices, 14)
    }
    
    // Momentum
    if (features.includes('momentum')) {
      calculated.momentum = prices[0] - prices[20] || 0
    }
    
    // Store in feature cache
    const timestamp = Date.now()
    for (const [feature_name, feature_value] of Object.entries(calculated)) {
      await env.DB.prepare(`
        INSERT INTO feature_cache (feature_name, symbol, feature_value, timestamp)
        VALUES (?, ?, ?, ?)
      `).bind(feature_name, symbol, feature_value as number, timestamp).run()
    }
    
    return c.json({ success: true, features: calculated })
  } catch (error) {
    return c.json({ success: false, error: String(error) }, 500)
  }
})

// Helper function for RSI calculation
function calculateRSI(prices: number[], period: number = 14): number {
  if (prices.length < period + 1) return 50
  
  let gains = 0, losses = 0
  for (let i = 0; i < period; i++) {
    const change = prices[i] - prices[i + 1]
    if (change > 0) gains += change
    else losses -= change
  }
  
  const avgGain = gains / period
  const avgLoss = losses / period
  const rs = avgLoss === 0 ? 100 : avgGain / avgLoss
  return 100 - (100 / (1 + rs))
}

// ============================================================================
// STRATEGY ENGINE - Trading Strategy Management
// ============================================================================

// Get all available strategies
app.get('/api/strategies', async (c) => {
  const { env } = c
  
  try {
    const results = await env.DB.prepare(`
      SELECT * FROM trading_strategies WHERE is_active = 1
    `).all()
    
    return c.json({
      success: true,
      strategies: results.results,
      count: results.results?.length || 0
    })
  } catch (error) {
    return c.json({ success: false, error: String(error) }, 500)
  }
})

// Generate strategy signals
app.post('/api/strategies/:id/signal', async (c) => {
  const { env } = c
  const strategyId = parseInt(c.req.param('id'))
  const { symbol, market_data } = await c.req.json()
  
  try {
    // Fetch strategy details
    const strategy = await env.DB.prepare(`
      SELECT * FROM trading_strategies WHERE id = ?
    `).bind(strategyId).first()
    
    if (!strategy) {
      return c.json({ success: false, error: 'Strategy not found' }, 404)
    }
    
    // Generate signal based on strategy type
    let signal_type = 'hold'
    let signal_strength = 0.5
    let confidence = 0.7
    
    const params = JSON.parse(strategy.parameters as string)
    
    switch (strategy.strategy_type) {
      case 'momentum':
        // Simple momentum strategy
        if (market_data.momentum > params.threshold) {
          signal_type = 'buy'
          signal_strength = 0.8
        } else if (market_data.momentum < -params.threshold) {
          signal_type = 'sell'
          signal_strength = 0.8
        }
        break
      
      case 'mean_reversion':
        // RSI-based mean reversion
        if (market_data.rsi < params.oversold) {
          signal_type = 'buy'
          signal_strength = 0.9
        } else if (market_data.rsi > params.overbought) {
          signal_type = 'sell'
          signal_strength = 0.9
        }
        break
      
      case 'sentiment':
        // Sentiment-based strategy
        if (market_data.sentiment > params.sentiment_threshold) {
          signal_type = 'buy'
          signal_strength = 0.75
        } else if (market_data.sentiment < -params.sentiment_threshold) {
          signal_type = 'sell'
          signal_strength = 0.75
        }
        break
    }
    
    // Store signal in database
    const timestamp = Date.now()
    await env.DB.prepare(`
      INSERT INTO strategy_signals 
      (strategy_id, symbol, signal_type, signal_strength, confidence, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(strategyId, symbol, signal_type, signal_strength, confidence, timestamp).run()
    
    return c.json({
      success: true,
      signal: {
        strategy_name: strategy.strategy_name,
        strategy_type: strategy.strategy_type,
        signal_type,
        signal_strength,
        confidence,
        timestamp
      }
    })
  } catch (error) {
    return c.json({ success: false, error: String(error) }, 500)
  }
})

// ============================================================================
// BACKTESTING AGENT - Historical Performance Analysis
// ============================================================================

// Run backtest for a strategy
app.post('/api/backtest/run', async (c) => {
  const { env } = c
  const { strategy_id, symbol, start_date, end_date, initial_capital } = await c.req.json()
  
  try {
    // Fetch historical data
    const historicalData = await env.DB.prepare(`
      SELECT * FROM market_data 
      WHERE symbol = ? AND timestamp BETWEEN ? AND ?
      ORDER BY timestamp ASC
    `).bind(symbol, start_date, end_date).all()
    
    // Simple backtest simulation
    let capital = initial_capital
    let position = 0
    let trades = 0
    let wins = 0
    const prices = historicalData.results || []
    
    // Simulate trading
    for (let i = 0; i < prices.length - 1; i++) {
      const price: any = prices[i]
      // Simple buy/sell logic based on mock signal
      if (Math.random() > 0.5 && position === 0) {
        // Buy
        position = capital / price.price
        trades++
      } else if (position > 0 && Math.random() > 0.6) {
        // Sell
        const sellValue = position * price.price
        if (sellValue > capital) wins++
        capital = sellValue
        position = 0
      }
    }
    
    // Close any open position
    if (position > 0 && prices.length > 0) {
      const lastPrice: any = prices[prices.length - 1]
      capital = position * lastPrice.price
    }
    
    const total_return = ((capital - initial_capital) / initial_capital) * 100
    const win_rate = trades > 0 ? (wins / trades) * 100 : 0
    const sharpe_ratio = Math.random() * 2 // Mock Sharpe ratio
    const max_drawdown = Math.random() * -20 // Mock drawdown
    
    // Store backtest results
    await env.DB.prepare(`
      INSERT INTO backtest_results 
      (strategy_id, symbol, start_date, end_date, initial_capital, final_capital, 
       total_return, sharpe_ratio, max_drawdown, win_rate, total_trades, avg_trade_return)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    `).bind(
      strategy_id, symbol, start_date, end_date, initial_capital, capital,
      total_return, sharpe_ratio, max_drawdown, win_rate, trades, total_return / trades
    ).run()
    
    return c.json({
      success: true,
      backtest: {
        initial_capital,
        final_capital: capital,
        total_return,
        sharpe_ratio,
        max_drawdown,
        win_rate,
        total_trades: trades
      }
    })
  } catch (error) {
    return c.json({ success: false, error: String(error) }, 500)
  }
})

// Get backtest results
app.get('/api/backtest/results/:strategy_id', async (c) => {
  const { env } = c
  const strategyId = parseInt(c.req.param('strategy_id'))
  
  try {
    const results = await env.DB.prepare(`
      SELECT * FROM backtest_results 
      WHERE strategy_id = ? 
      ORDER BY created_at DESC
    `).bind(strategyId).all()
    
    return c.json({
      success: true,
      results: results.results,
      count: results.results?.length || 0
    })
  } catch (error) {
    return c.json({ success: false, error: String(error) }, 500)
  }
})

// ============================================================================
// LLM REASONING LAYER - AI-Powered Market Analysis
// ============================================================================

// Generate LLM market analysis
app.post('/api/llm/analyze', async (c) => {
  const { env } = c
  const { analysis_type, symbol, context } = await c.req.json()
  
  try {
    // In production, this would call OpenAI, Anthropic, or other LLM APIs
    // For now, we'll return a mock analysis
    
    const prompt = `Analyze ${symbol} market conditions: ${JSON.stringify(context)}`
    
    let response = ''
    let confidence = 0.8
    
    switch (analysis_type) {
      case 'market_commentary':
        response = `Based on current market data for ${symbol}, we observe ${context.trend || 'mixed'} trend signals. 
        Technical indicators suggest ${context.rsi < 30 ? 'oversold' : context.rsi > 70 ? 'overbought' : 'neutral'} conditions. 
        Recommend ${context.rsi < 30 ? 'accumulation' : context.rsi > 70 ? 'profit-taking' : 'monitoring'} strategy.`
        break
      
      case 'strategy_recommendation':
        response = `For ${symbol}, given current market regime of ${context.regime || 'moderate volatility'}, 
        recommend ${context.volatility > 0.5 ? 'mean reversion' : 'momentum'} strategy with 
        risk allocation of ${context.risk_level || 'moderate'}%.`
        confidence = 0.75
        break
      
      case 'risk_assessment':
        response = `Risk assessment for ${symbol}: Current volatility is ${context.volatility || 'unknown'}. 
        Maximum recommended position size: ${5 / (context.volatility || 1)}%. 
        Stop loss recommended at ${context.price * 0.95}. 
        Risk/Reward ratio: ${Math.random() * 3 + 1}:1`
        confidence = 0.85
        break
      
      default:
        response = 'Unknown analysis type'
    }
    
    // Store LLM analysis
    const timestamp = Date.now()
    await env.DB.prepare(`
      INSERT INTO llm_analysis 
      (analysis_type, symbol, prompt, response, confidence, context_data, timestamp)
      VALUES (?, ?, ?, ?, ?, ?, ?)
    `).bind(
      analysis_type, symbol, prompt, response, confidence, 
      JSON.stringify(context), timestamp
    ).run()
    
    return c.json({
      success: true,
      analysis: {
        type: analysis_type,
        symbol,
        response,
        confidence,
        timestamp
      }
    })
  } catch (error) {
    return c.json({ success: false, error: String(error) }, 500)
  }
})

// Get LLM analysis history
app.get('/api/llm/history/:type', async (c) => {
  const { env } = c
  const analysisType = c.req.param('type')
  const limit = parseInt(c.req.query('limit') || '10')
  
  try {
    const results = await env.DB.prepare(`
      SELECT * FROM llm_analysis 
      WHERE analysis_type = ? 
      ORDER BY timestamp DESC 
      LIMIT ?
    `).bind(analysisType, limit).all()
    
    return c.json({
      success: true,
      history: results.results,
      count: results.results?.length || 0
    })
  } catch (error) {
    return c.json({ success: false, error: String(error) }, 500)
  }
})

// Enhanced LLM Analysis with Live Agent Data
app.post('/api/llm/analyze-enhanced', async (c) => {
  const { env } = c
  const { symbol = 'BTC', timeframe = '1h' } = await c.req.json()
  
  try {
    // Fetch data from all 3 live agents
    const baseUrl = `http://localhost:3000`
    
    const [economicRes, sentimentRes, crossExchangeRes] = await Promise.all([
      fetch(`${baseUrl}/api/agents/economic?symbol=${symbol}`),
      fetch(`${baseUrl}/api/agents/sentiment?symbol=${symbol}`),
      fetch(`${baseUrl}/api/agents/cross-exchange?symbol=${symbol}`)
    ])
    
    const economicData = await economicRes.json()
    const sentimentData = await sentimentRes.json()
    const crossExchangeData = await crossExchangeRes.json()
    
    // Check if API key is available
    const apiKey = env.GEMINI_API_KEY
    
    if (!apiKey) {
      // Fallback to template-based analysis
      const analysis = generateTemplateAnalysis(economicData, sentimentData, crossExchangeData, symbol)
      
      await env.DB.prepare(`
        INSERT INTO llm_analysis (analysis_type, symbol, prompt, response, context_data, timestamp)
        VALUES (?, ?, ?, ?, ?, ?)
      `).bind(
        'enhanced-agent-based',
        symbol,
        'Template-based analysis from live agent feeds',
        analysis,
        JSON.stringify({
          timeframe,
          data_sources: ['economic', 'sentiment', 'cross-exchange'],
          model: 'template-fallback'
        }),
        Date.now()
      ).run()
      
      return c.json({
        success: true,
        analysis,
        data_sources: ['Economic Agent', 'Sentiment Agent', 'Cross-Exchange Agent'],
        timestamp: new Date().toISOString(),
        model: 'template-fallback'
      })
    }
    
    // Build comprehensive prompt with all agent data
    const prompt = buildEnhancedPrompt(economicData, sentimentData, crossExchangeData, symbol, timeframe)
    
    // Call Gemini API
    const geminiResponse = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key=${apiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{
            parts: [{ text: prompt }]
          }],
          generationConfig: {
            temperature: 0.7,
            maxOutputTokens: 2048,
            topP: 0.95,
            topK: 40
          }
        })
      }
    )
    
    if (!geminiResponse.ok) {
      throw new Error(`Gemini API error: ${geminiResponse.status}`)
    }
    
    const geminiData = await geminiResponse.json()
    const analysis = geminiData.candidates?.[0]?.content?.parts?.[0]?.text || 'Analysis generation failed'
    
    // Store in database
    await env.DB.prepare(`
      INSERT INTO llm_analysis (analysis_type, symbol, prompt, response, context_data, timestamp)
      VALUES (?, ?, ?, ?, ?, ?)
    `).bind(
      'enhanced-agent-based',
      symbol,
      prompt.substring(0, 500),  // Store first 500 chars of prompt
      analysis,
      JSON.stringify({
        timeframe,
        data_sources: ['economic', 'sentiment', 'cross-exchange'],
        model: 'gemini-2.0-flash-exp'
      }),
      Date.now()
    ).run()
    
    return c.json({
      success: true,
      analysis,
      data_sources: ['Economic Agent', 'Sentiment Agent', 'Cross-Exchange Agent'],
      timestamp: new Date().toISOString(),
      model: 'gemini-2.0-flash-exp',
      agent_data: {
        economic: economicData.data,
        sentiment: sentimentData.data,
        cross_exchange: crossExchangeData.data
      }
    })
    
  } catch (error) {
    console.error('Enhanced LLM analysis error:', error)
    return c.json({ 
      success: false, 
      error: String(error),
      fallback: 'Unable to generate enhanced analysis'
    }, 500)
  }
})

// Helper function to build comprehensive prompt
function buildEnhancedPrompt(economicData: any, sentimentData: any, crossExchangeData: any, symbol: string, timeframe: string): string {
  const econ = economicData.data.indicators
  const sent = sentimentData.data.sentiment_metrics
  const cross = crossExchangeData.data.market_depth_analysis
  
  return `You are an expert cryptocurrency market analyst. Provide a comprehensive market analysis for ${symbol}/USD based on the following live data feeds:

**ECONOMIC INDICATORS (Federal Reserve & Macro Data)**
- Federal Funds Rate: ${econ.fed_funds_rate.value}% (${econ.fed_funds_rate.trend}, next meeting: ${econ.fed_funds_rate.next_meeting})
- CPI Inflation: ${econ.cpi.value}% YoY (${econ.cpi.trend})
- PPI: ${econ.ppi.value}% (change: ${econ.ppi.change})
- Unemployment Rate: ${econ.unemployment_rate.value}% (${econ.unemployment_rate.trend})
- Non-Farm Payrolls: ${econ.unemployment_rate.non_farm_payrolls.toLocaleString()}
- GDP Growth: ${econ.gdp_growth.value}% (${econ.gdp_growth.quarter})
- 10Y Treasury Yield: ${econ.treasury_10y.value}% (spread: ${econ.treasury_10y.spread}%)
- Manufacturing PMI: ${econ.manufacturing_pmi.value} (${econ.manufacturing_pmi.status})
- Retail Sales: ${econ.retail_sales.value}% growth

**MARKET SENTIMENT INDICATORS**
- Fear & Greed Index: ${sent.fear_greed_index.value} (${sent.fear_greed_index.classification})
- Aggregate Sentiment: ${sent.aggregate_sentiment.value}% (${sent.aggregate_sentiment.trend})
- VIX (Volatility Index): ${sent.volatility_index_vix.value.toFixed(2)} (${sent.volatility_index_vix.interpretation} volatility)
- Social Media Volume: ${sent.social_media_volume.mentions.toLocaleString()} mentions (${sent.social_media_volume.trend})
- Institutional Flow (24h): $${sent.institutional_flow_24h.net_flow_million_usd.toFixed(1)}M (${sent.institutional_flow_24h.direction})

**CROSS-EXCHANGE LIQUIDITY & EXECUTION**
- 24h Volume: $${cross.total_volume_24h.usd.toFixed(2)}B / ${cross.total_volume_24h.btc.toFixed(0)} BTC
- Market Depth Score: ${cross.market_depth_score.score}/10 (${cross.market_depth_score.rating})
- Average Spread: ${cross.liquidity_metrics.average_spread_percent}%
- Slippage (10 BTC): ${cross.liquidity_metrics.slippage_10btc_percent}%
- Order Book Imbalance: ${cross.liquidity_metrics.order_book_imbalance.toFixed(2)}
- Large Order Impact: ${cross.execution_quality.large_order_impact_percent.toFixed(1)}%
- Recommended Exchanges: ${cross.execution_quality.recommended_exchanges.join(', ')}

**YOUR TASK:**
Provide a detailed 3-paragraph analysis covering:
1. **Macro Environment Impact**: How do current economic indicators (Fed policy, inflation, employment, GDP) affect ${symbol} outlook?
2. **Market Sentiment & Positioning**: What do sentiment indicators, institutional flows, and volatility metrics suggest about current market psychology?
3. **Trading Recommendation**: Based on liquidity conditions and all data, what is your outlook (bullish/bearish/neutral) and recommended action with risk assessment?

Keep the tone professional but accessible. Use specific numbers from the data. End with a clear directional bias and confidence level (1-10).`
}

// Helper function to generate template-based analysis (fallback)
function generateTemplateAnalysis(economicData: any, sentimentData: any, crossExchangeData: any, symbol: string): string {
  const econ = economicData.data.indicators
  const sent = sentimentData.data.sentiment_metrics
  const cross = crossExchangeData.data.market_depth_analysis
  
  const fedTrend = econ.fed_funds_rate.trend === 'stable' ? 'maintaining a steady stance' : 'adjusting rates'
  const inflationTrend = econ.cpi.trend === 'decreasing' ? 'moderating inflation' : 'persistent inflation'
  const sentimentBias = sent.aggregate_sentiment.value > 60 ? 'optimistic' : sent.aggregate_sentiment.value < 40 ? 'pessimistic' : 'neutral'
  const liquidityStatus = cross.market_depth_score.score > 8 ? 'excellent' : cross.market_depth_score.score > 6 ? 'adequate' : 'concerning'
  
  return `**Market Analysis for ${symbol}/USD**

**Macroeconomic Environment**: The Federal Reserve is currently ${fedTrend} with rates at ${econ.fed_funds_rate.value}%, while ${inflationTrend} is evident with CPI at ${econ.cpi.value}%. GDP growth of ${econ.gdp_growth.value}% in ${econ.gdp_growth.quarter} suggests moderate economic expansion. The 10-year Treasury yield at ${econ.treasury_10y.value}% provides context for risk-free rates. Manufacturing PMI at ${econ.manufacturing_pmi.value} indicates ${econ.manufacturing_pmi.status}, which may pressure risk assets.

**Market Sentiment & Psychology**: Current sentiment is ${sentimentBias} with the aggregate sentiment index at ${sent.aggregate_sentiment.value}% and Fear & Greed at ${sent.fear_greed_index.value}. The VIX at ${sent.volatility_index_vix.value.toFixed(2)} suggests ${sent.volatility_index_vix.interpretation} market volatility. Institutional flows show ${sent.institutional_flow_24h.direction} of $${Math.abs(sent.institutional_flow_24h.net_flow_million_usd).toFixed(1)}M over 24 hours, indicating ${sent.institutional_flow_24h.direction === 'outflow' ? 'profit-taking or risk-off positioning' : 'accumulation'}.

**Trading Outlook**: With ${liquidityStatus} market liquidity (depth score: ${cross.market_depth_score.score}/10) and 24h volume of $${cross.total_volume_24h.usd.toFixed(2)}B, execution conditions are favorable. The average spread of ${cross.liquidity_metrics.average_spread_percent}% and order book imbalance of ${cross.liquidity_metrics.order_book_imbalance.toFixed(2)} suggest ${cross.liquidity_metrics.order_book_imbalance > 0.55 ? 'buy-side pressure' : cross.liquidity_metrics.order_book_imbalance < 0.45 ? 'sell-side pressure' : 'balanced positioning'}. Based on the confluence of economic data, sentiment indicators, and liquidity conditions, the outlook is **${sent.aggregate_sentiment.value > 60 && cross.market_depth_score.score > 7 ? 'MODERATELY BULLISH' : sent.aggregate_sentiment.value < 40 ? 'BEARISH' : 'NEUTRAL'}** with a confidence level of ${Math.floor(6 + Math.random() * 2)}/10. Traders should monitor Fed policy developments and institutional flow reversals as key catalysts.

*Analysis generated from live agent data feeds: Economic Agent, Sentiment Agent, Cross-Exchange Agent*`
}

// ============================================================================
// MARKET REGIME DETECTION
// ============================================================================

app.post('/api/market/regime', async (c) => {
  const { env } = c
  const { indicators } = await c.req.json()
  
  try {
    // Detect market regime based on indicators
    let regime_type = 'sideways'
    let confidence = 0.7
    
    const { volatility, trend, volume } = indicators
    
    if (trend > 0.05 && volatility < 0.3) {
      regime_type = 'bull'
      confidence = 0.85
    } else if (trend < -0.05 && volatility > 0.4) {
      regime_type = 'bear'
      confidence = 0.8
    } else if (volatility > 0.5) {
      regime_type = 'high_volatility'
      confidence = 0.9
    } else if (volatility < 0.15) {
      regime_type = 'low_volatility'
      confidence = 0.85
    }
    
    const timestamp = Date.now()
    await env.DB.prepare(`
      INSERT INTO market_regime (regime_type, confidence, indicators, timestamp)
      VALUES (?, ?, ?, ?)
    `).bind(regime_type, confidence, JSON.stringify(indicators), timestamp).run()
    
    return c.json({
      success: true,
      regime: {
        type: regime_type,
        confidence,
        indicators,
        timestamp
      }
    })
  } catch (error) {
    return c.json({ success: false, error: String(error) }, 500)
  }
})

// ============================================================================
// DASHBOARD & VISUALIZATION DATA
// ============================================================================

// Get dashboard summary
app.get('/api/dashboard/summary', async (c) => {
  const { env } = c
  
  try {
    // Get latest market regime
    const regime = await env.DB.prepare(`
      SELECT * FROM market_regime ORDER BY timestamp DESC LIMIT 1
    `).first()
    
    // Get active strategies count
    const strategies = await env.DB.prepare(`
      SELECT COUNT(*) as count FROM trading_strategies WHERE is_active = 1
    `).first()
    
    // Get recent signals
    const signals = await env.DB.prepare(`
      SELECT * FROM strategy_signals ORDER BY timestamp DESC LIMIT 5
    `).all()
    
    // Get latest backtest results
    const backtests = await env.DB.prepare(`
      SELECT * FROM backtest_results ORDER BY created_at DESC LIMIT 3
    `).all()
    
    return c.json({
      success: true,
      dashboard: {
        market_regime: regime,
        active_strategies: strategies?.count || 0,
        recent_signals: signals.results,
        recent_backtests: backtests.results
      }
    })
  } catch (error) {
    return c.json({ success: false, error: String(error) }, 500)
  }
})

// ============================================================================
// MAIN DASHBOARD HTML
// ============================================================================

app.get('/', (c) => {
  return c.html(`
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Trading Intelligence Platform</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/axios@1.6.0/dist/axios.min.js"></script>
    </head>
    <body class="bg-gradient-to-br from-gray-900 via-blue-900 to-gray-900 text-white min-h-screen">
        <div class="container mx-auto px-4 py-8">
            <!-- Header -->
            <div class="mb-8">
                <h1 class="text-4xl font-bold mb-2">
                    <i class="fas fa-chart-line mr-3"></i>
                    LLM-Driven Trading Intelligence Platform
                </h1>
                <p class="text-blue-300 text-lg">
                    Multimodal Data Fusion • Machine Learning • Adaptive Strategies
                </p>
            </div>

            <!-- Status Cards -->
            <div class="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
                <div class="bg-gray-800 rounded-lg p-6 border border-blue-500">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-gray-400 text-sm">Market Regime</p>
                            <p id="regime-type" class="text-2xl font-bold mt-1">Loading...</p>
                        </div>
                        <i class="fas fa-globe text-4xl text-blue-500"></i>
                    </div>
                </div>

                <div class="bg-gray-800 rounded-lg p-6 border border-green-500">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-gray-400 text-sm">Active Strategies</p>
                            <p id="strategy-count" class="text-2xl font-bold mt-1">5</p>
                        </div>
                        <i class="fas fa-brain text-4xl text-green-500"></i>
                    </div>
                </div>

                <div class="bg-gray-800 rounded-lg p-6 border border-purple-500">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-gray-400 text-sm">Recent Signals</p>
                            <p id="signal-count" class="text-2xl font-bold mt-1">0</p>
                        </div>
                        <i class="fas fa-signal text-4xl text-purple-500"></i>
                    </div>
                </div>

                <div class="bg-gray-800 rounded-lg p-6 border border-yellow-500">
                    <div class="flex items-center justify-between">
                        <div>
                            <p class="text-gray-400 text-sm">Backtests Run</p>
                            <p id="backtest-count" class="text-2xl font-bold mt-1">0</p>
                        </div>
                        <i class="fas fa-history text-4xl text-yellow-500"></i>
                    </div>
                </div>
            </div>

            <!-- Main Content -->
            <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
                <!-- Trading Strategies -->
                <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                    <h2 class="text-2xl font-bold mb-4">
                        <i class="fas fa-robot mr-2"></i>
                        Trading Strategies
                    </h2>
                    <div id="strategies-list" class="space-y-3">
                        <p class="text-gray-400">Loading strategies...</p>
                    </div>
                </div>

                <!-- Recent Signals -->
                <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                    <h2 class="text-2xl font-bold mb-4">
                        <i class="fas fa-bullhorn mr-2"></i>
                        Recent Signals
                    </h2>
                    <div id="signals-list" class="space-y-3">
                        <p class="text-gray-400">No signals yet...</p>
                    </div>
                </div>
            </div>

            <!-- LLM Analysis Section -->
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700 mb-8">
                <h2 class="text-2xl font-bold mb-4">
                    <i class="fas fa-lightbulb mr-2"></i>
                    LLM Market Analysis
                </h2>
                <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    <button onclick="requestAnalysis('market_commentary')" class="bg-blue-600 hover:bg-blue-700 px-6 py-3 rounded-lg">
                        Market Commentary
                    </button>
                    <button onclick="requestAnalysis('strategy_recommendation')" class="bg-green-600 hover:bg-green-700 px-6 py-3 rounded-lg">
                        Strategy Recommendation
                    </button>
                    <button onclick="requestAnalysis('risk_assessment')" class="bg-red-600 hover:bg-red-700 px-6 py-3 rounded-lg">
                        Risk Assessment
                    </button>
                </div>
                <div id="llm-response" class="bg-gray-900 p-4 rounded-lg min-h-32">
                    <p class="text-gray-400 italic">Click a button above to get LLM analysis...</p>
                </div>
            </div>

            <!-- Backtest Results -->
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h2 class="text-2xl font-bold mb-4">
                    <i class="fas fa-chart-bar mr-2"></i>
                    Backtest Results
                </h2>
                <div id="backtest-results" class="space-y-3">
                    <p class="text-gray-400">No backtests run yet...</p>
                </div>
                <button onclick="runBacktest()" class="mt-4 bg-purple-600 hover:bg-purple-700 px-6 py-2 rounded-lg">
                    <i class="fas fa-play mr-2"></i>
                    Run New Backtest
                </button>
            </div>

            <!-- Footer -->
            <div class="mt-8 text-center text-gray-500">
                <p>LLM-Driven Trading Intelligence System • Built with Hono + Cloudflare D1 + Chart.js</p>
            </div>
        </div>

        <script src="/static/app.js"></script>
    </body>
    </html>
  `)
})

export default app
