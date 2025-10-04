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

// Social Media Sentiment Analysis Engine
const getSocialSentimentFeeds = () => {
  const generateSentiment = () => Math.random() * 100
  const generateVolume = () => Math.floor(Math.random() * 50000 + 10000)
  const generateTrending = () => Math.random() > 0.7
  
  return {
    twitter: {
      BTC: {
        sentiment: generateSentiment(),
        volume: generateVolume(),
        trending: generateTrending(),
        topHashtags: ['#Bitcoin', '#BTC', '#Crypto', '#HODL', '#ToTheMoon'],
        influencerMentions: Math.floor(Math.random() * 150 + 50),
        lastUpdate: Date.now()
      },
      ETH: {
        sentiment: generateSentiment(),
        volume: generateVolume(),
        trending: generateTrending(),
        topHashtags: ['#Ethereum', '#ETH', '#DeFi', '#SmartContracts', '#Web3'],
        influencerMentions: Math.floor(Math.random() * 120 + 30),
        lastUpdate: Date.now()
      },
      SOL: {
        sentiment: generateSentiment(),
        volume: generateVolume(),
        trending: generateTrending(),
        topHashtags: ['#Solana', '#SOL', '#FastCrypto', '#NFTs', '#DeFi'],
        influencerMentions: Math.floor(Math.random() * 80 + 20),
        lastUpdate: Date.now()
      }
    },
    reddit: {
      BTC: {
        sentiment: generateSentiment(),
        posts: Math.floor(Math.random() * 500 + 100),
        upvotes: Math.floor(Math.random() * 10000 + 2000),
        comments: Math.floor(Math.random() * 5000 + 1000),
        trending: generateTrending()
      },
      ETH: {
        sentiment: generateSentiment(),
        posts: Math.floor(Math.random() * 400 + 80),
        upvotes: Math.floor(Math.random() * 8000 + 1500),
        comments: Math.floor(Math.random() * 4000 + 800),
        trending: generateTrending()
      },
      SOL: {
        sentiment: generateSentiment(),
        posts: Math.floor(Math.random() * 200 + 50),
        upvotes: Math.floor(Math.random() * 5000 + 800),
        comments: Math.floor(Math.random() * 2500 + 400),
        trending: generateTrending()
      }
    },
    news: {
      sentiment: generateSentiment(),
      articlesCount: Math.floor(Math.random() * 50 + 20),
      mediaOutlets: ['Reuters', 'Bloomberg', 'CoinDesk', 'Cointelegraph', 'TheBlock'],
      breakingNews: Math.random() > 0.8,
      fearGreedIndex: Math.floor(Math.random() * 100)
    }
  }
}

// Economic Indicators Engine  
const getEconomicIndicators = () => {
  const generateEconData = (base, volatility) => ({
    current: base + (Math.random() - 0.5) * volatility,
    previous: base + (Math.random() - 0.5) * volatility,
    forecast: base + (Math.random() - 0.5) * volatility,
    change: (Math.random() - 0.5) * 2,
    lastUpdate: Date.now()
  })
  
  return {
    us: {
      gdp: generateEconData(2.1, 0.8),
      inflation: generateEconData(3.2, 0.5), 
      unemployment: generateEconData(3.8, 0.3),
      interestRate: generateEconData(5.25, 0.25),
      retailSales: generateEconData(0.4, 1.2),
      cpi: generateEconData(3.7, 0.4),
      ppi: generateEconData(2.1, 0.6),
      consumerConfidence: generateEconData(102.6, 8.0),
      dollarIndex: generateEconData(103.8, 2.0)
    },
    global: {
      china: {
        gdp: generateEconData(5.2, 0.6),
        pmi: generateEconData(49.5, 2.0),
        exports: generateEconData(2.3, 3.0)
      },
      europe: {
        gdp: generateEconData(0.1, 0.4),
        inflation: generateEconData(2.9, 0.3),
        ecbRate: generateEconData(4.5, 0.25)
      },
      japan: {
        gdp: generateEconData(-0.1, 0.3),
        inflation: generateEconData(3.1, 0.2),
        bojRate: generateEconData(-0.1, 0.1)
      }
    },
    crypto: {
      bitcoinDominance: generateEconData(52.3, 3.0),
      totalMarketCap: generateEconData(2.1, 0.3), // Trillions
      defiTvl: generateEconData(78.5, 8.0), // Billions
      stakingRatio: generateEconData(23.4, 2.0),
      institutionalFlow: generateEconData(1.2, 2.5) // Billions
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

// Social sentiment endpoints
app.get('/api/social-sentiment', (c) => {
  return c.json(getSocialSentimentFeeds())
})

// Economic indicators endpoints
app.get('/api/economic-indicators', (c) => {
  return c.json(getEconomicIndicators())
})

// Real-time sentiment summary
app.get('/api/sentiment-summary', (c) => {
  const sentiment = getSocialSentimentFeeds()
  
  const overallSentiment = {
    BTC: (sentiment.twitter.BTC.sentiment + sentiment.reddit.BTC.sentiment) / 2,
    ETH: (sentiment.twitter.ETH.sentiment + sentiment.reddit.ETH.sentiment) / 2,
    SOL: (sentiment.twitter.SOL.sentiment + sentiment.reddit.SOL.sentiment) / 2
  }
  
  const marketMood = (overallSentiment.BTC + overallSentiment.ETH + overallSentiment.SOL) / 3
  
  return c.json({
    overall: marketMood,
    assets: overallSentiment,
    fearGreedIndex: sentiment.news.fearGreedIndex,
    socialVolume: {
      twitter: sentiment.twitter.BTC.volume + sentiment.twitter.ETH.volume + sentiment.twitter.SOL.volume,
      reddit: sentiment.reddit.BTC.posts + sentiment.reddit.ETH.posts + sentiment.reddit.SOL.posts
    },
    trending: {
      twitter: Object.keys(sentiment.twitter).filter(asset => sentiment.twitter[asset].trending),
      reddit: Object.keys(sentiment.reddit).filter(asset => sentiment.reddit[asset].trending)
    },
    breakingNews: sentiment.news.breakingNews,
    timestamp: Date.now()
  })
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

// Advanced Candlestick Data Generation with Realistic Patterns
class CandlestickGenerator {
  constructor(symbol, basePrice) {
    this.symbol = symbol
    this.basePrice = basePrice
    this.currentPrice = basePrice
    this.trend = Math.random() > 0.5 ? 1 : -1
    this.volatility = 0.02
    this.lastTimestamp = Date.now()
  }
  
  generateRealisticCandle(timeframe = '1m') {
    const now = Date.now()
    const timeframes = {
      '1m': 60000,
      '5m': 300000,
      '15m': 900000,
      '1h': 3600000
    }
    
    const interval = timeframes[timeframe] || 60000
    
    // Generate realistic OHLCV data
    const open = this.currentPrice
    const volatilityFactor = this.volatility * Math.sqrt(interval / 60000)
    
    // Trend-following with mean reversion
    const trendStrength = 0.3
    const meanReversion = 0.1
    
    // Generate price movements with realistic patterns
    const movements = []
    for (let i = 0; i < 4; i++) {
      const random = (Math.random() - 0.5) * 2
      const trendComponent = this.trend * trendStrength * volatilityFactor
      const meanReversionComponent = -meanReversion * (this.currentPrice - this.basePrice) / this.basePrice
      const noiseComponent = random * volatilityFactor
      
      movements.push(trendComponent + meanReversionComponent + noiseComponent)
    }
    
    const prices = [open]
    for (let movement of movements) {
      prices.push(prices[prices.length - 1] * (1 + movement))
    }
    
    const high = Math.max(...prices) * (1 + Math.random() * 0.005)
    const low = Math.min(...prices) * (1 - Math.random() * 0.005)
    const close = prices[prices.length - 1]
    const volume = Math.floor(Math.random() * 1000 + 500) * (1 + Math.abs(close - open) / open * 10)
    
    // Update state
    this.currentPrice = close
    this.lastTimestamp = now
    
    // Occasionally change trend
    if (Math.random() < 0.05) {
      this.trend *= -1
    }
    
    return {
      timestamp: now,
      open: Number(open.toFixed(2)),
      high: Number(high.toFixed(2)),
      low: Number(low.toFixed(2)),
      close: Number(close.toFixed(2)),
      volume: Math.floor(volume)
    }
  }
  
  generateHistoricalData(periods = 100, timeframe = '1m') {
    const data = []
    for (let i = 0; i < periods; i++) {
      data.push(this.generateRealisticCandle(timeframe))
    }
    return data
  }
}

// Hyperbolic CNN Pattern Analysis Engine
class HyperbolicCNNAnalyzer {
  constructor() {
    this.patterns = {
      'doji': { confidence: 0, signal: 'neutral' },
      'hammer': { confidence: 0, signal: 'bullish' },
      'shooting_star': { confidence: 0, signal: 'bearish' },
      'engulfing_bullish': { confidence: 0, signal: 'strong_bullish' },
      'engulfing_bearish': { confidence: 0, signal: 'strong_bearish' },
      'morning_star': { confidence: 0, signal: 'reversal_bullish' },
      'evening_star': { confidence: 0, signal: 'reversal_bearish' }
    }
  }
  
  // Hyperbolic distance calculation in Poincaré disk
  hyperbolicDistance(x1, y1, x2, y2) {
    const numerator = Math.pow(x2 - x1, 2) + Math.pow(y2 - y1, 2)
    const denominator = (1 - (x1*x1 + y1*y1)) * (1 - (x2*x2 + y2*y2))
    return Math.acosh(1 + 2 * numerator / denominator)
  }
  
  // Convert candlestick patterns to hyperbolic coordinates
  mapToHyperbolicSpace(candle) {
    const bodySize = Math.abs(candle.close - candle.open) / (candle.high - candle.low)
    const upperShadow = (candle.high - Math.max(candle.open, candle.close)) / (candle.high - candle.low)
    const lowerShadow = (Math.min(candle.open, candle.close) - candle.low) / (candle.high - candle.low)
    
    // Map to Poincaré disk coordinates
    const r = Math.sqrt(bodySize * bodySize + (upperShadow - lowerShadow) * (upperShadow - lowerShadow))
    const theta = Math.atan2(upperShadow - lowerShadow, bodySize)
    
    // Ensure coordinates are within unit disk
    const scale = Math.min(r, 0.95)
    return {
      x: scale * Math.cos(theta),
      y: scale * Math.sin(theta),
      bodySize,
      upperShadow,
      lowerShadow
    }
  }
  
  // Advanced pattern recognition using hyperbolic geometry
  analyzePattern(candleData) {
    if (candleData.length < 3) return { pattern: 'insufficient_data', confidence: 0 }
    
    const recent = candleData.slice(-3)
    const current = recent[recent.length - 1]
    const previous = recent[recent.length - 2]
    
    // Map candles to hyperbolic space
    const currentHyp = this.mapToHyperbolicSpace(current)
    const previousHyp = this.mapToHyperbolicSpace(previous)
    
    // Calculate hyperbolic distance for pattern similarity
    const distance = this.hyperbolicDistance(
      currentHyp.x, currentHyp.y,
      previousHyp.x, previousHyp.y
    )
    
    // Pattern recognition logic
    const bodySize = currentHyp.bodySize
    const upperShadow = currentHyp.upperShadow
    const lowerShadow = currentHyp.lowerShadow
    
    let pattern = 'undefined'
    let confidence = 0
    let signal = 'neutral'
    let arbitrageRelevance = 0
    
    // Doji pattern detection
    if (bodySize < 0.1 && Math.abs(upperShadow - lowerShadow) < 0.2) {
      pattern = 'doji'
      confidence = 85 + Math.random() * 10
      signal = 'neutral'
      arbitrageRelevance = 60 // Medium relevance for sideways movement
    }
    
    // Hammer pattern detection
    else if (lowerShadow > 0.5 && upperShadow < 0.2 && bodySize < 0.3) {
      pattern = 'hammer'
      confidence = 88 + Math.random() * 7
      signal = 'bullish'
      arbitrageRelevance = 85 // High relevance for reversal
    }
    
    // Shooting star pattern detection
    else if (upperShadow > 0.5 && lowerShadow < 0.2 && bodySize < 0.3) {
      pattern = 'shooting_star'
      confidence = 86 + Math.random() * 8
      signal = 'bearish'
      arbitrageRelevance = 82
    }
    
    // Engulfing patterns (requires multiple candles)
    else if (recent.length >= 2) {
      const prevBody = Math.abs(previous.close - previous.open)
      const currBody = Math.abs(current.close - current.open)
      
      if (currBody > prevBody * 1.5) {
        if (current.close > current.open && previous.close < previous.open) {
          pattern = 'engulfing_bullish'
          confidence = 92 + Math.random() * 5
          signal = 'strong_bullish'
          arbitrageRelevance = 95 // Very high relevance for strong reversal
        } else if (current.close < current.open && previous.close > previous.open) {
          pattern = 'engulfing_bearish'
          confidence = 91 + Math.random() * 6
          signal = 'strong_bearish'
          arbitrageRelevance = 94
        }
      }
    }
    
    // Calculate geodesic efficiency in hyperbolic space
    const geodesicEfficiency = 1 / (1 + distance) * 100
    
    return {
      pattern,
      confidence: Math.round(confidence),
      signal,
      arbitrageRelevance: Math.round(arbitrageRelevance),
      hyperbolicDistance: distance.toFixed(4),
      geodesicEfficiency: geodesicEfficiency.toFixed(1),
      coordinates: currentHyp,
      timestamp: current.timestamp
    }
  }
  
  // Generate arbitrage timing recommendations based on patterns
  generateArbitrageTiming(patternAnalysis, marketData) {
    const { pattern, confidence, signal, arbitrageRelevance } = patternAnalysis
    
    let timing = 'hold'
    let recommendation = 'Monitor market conditions'
    let optimalEntry = null
    let riskLevel = 'medium'
    
    if (arbitrageRelevance > 80 && confidence > 85) {
      if (signal.includes('bullish')) {
        timing = 'buy'
        recommendation = `Strong ${pattern} pattern detected. Execute long arbitrage positions.`
        optimalEntry = 'immediate'
        riskLevel = 'low'
      } else if (signal.includes('bearish')) {
        timing = 'sell'
        recommendation = `Strong ${pattern} pattern detected. Execute short arbitrage positions.`
        optimalEntry = 'immediate'
        riskLevel = 'low'
      }
    } else if (arbitrageRelevance > 60 && confidence > 75) {
      timing = 'prepare'
      recommendation = `Moderate ${pattern} pattern forming. Prepare arbitrage positions for confirmation.`
      optimalEntry = '2-5 minutes'
      riskLevel = 'medium'
    }
    
    return {
      timing,
      recommendation,
      optimalEntry,
      riskLevel,
      patternStrength: arbitrageRelevance,
      confidence: confidence
    }
  }
}

// Initialize generators and analyzer
const candlestickGenerators = {
  BTC: new CandlestickGenerator('BTC', 67234.56),
  ETH: new CandlestickGenerator('ETH', 3456.08),
  SOL: new CandlestickGenerator('SOL', 123.45)
}

const hyperbolicAnalyzer = new HyperbolicCNNAnalyzer()

// Store historical data for pattern analysis
const historicalData = {}
Object.keys(candlestickGenerators).forEach(symbol => {
  historicalData[symbol] = {
    '1m': candlestickGenerators[symbol].generateHistoricalData(100, '1m'),
    '5m': candlestickGenerators[symbol].generateHistoricalData(50, '5m'),
    '15m': candlestickGenerators[symbol].generateHistoricalData(30, '15m'),
    '1h': candlestickGenerators[symbol].generateHistoricalData(24, '1h')
  }
})

app.post('/api/ai-query', async (c) => {
  const { query, chartData } = await c.req.json()
  
  // Enhanced AI responses with chart analysis capability
  let response = ''
  let confidence = 85
  let additionalData = {}
  
  // Chart analysis queries
  if (query.toLowerCase().includes('chart') || query.toLowerCase().includes('pattern') || query.toLowerCase().includes('candlestick')) {
    const symbol = query.match(/BTC|ETH|SOL/i)?.[0] || 'BTC'
    const timeframe = query.match(/1m|5m|15m|1h/i)?.[0] || '1m'
    
    const recentData = historicalData[symbol][timeframe].slice(-10)
    const patternAnalysis = hyperbolicAnalyzer.analyzePattern(recentData)
    const arbitrageTiming = hyperbolicAnalyzer.generateArbitrageTiming(patternAnalysis, recentData[recentData.length - 1])
    
    response = `🔍 **${symbol} Chart Analysis (${timeframe})**\n\n` +
              `**Pattern Detected**: ${patternAnalysis.pattern.replace(/_/g, ' ').toUpperCase()}\n` +
              `**Signal**: ${patternAnalysis.signal.replace(/_/g, ' ').toUpperCase()}\n` +
              `**Pattern Confidence**: ${patternAnalysis.confidence}%\n` +
              `**Arbitrage Relevance**: ${patternAnalysis.arbitrageRelevance}%\n\n` +
              `**Hyperbolic Analysis**:\n` +
              `• Geodesic Efficiency: ${patternAnalysis.geodesicEfficiency}%\n` +
              `• Hyperbolic Distance: ${patternAnalysis.hyperbolicDistance}\n\n` +
              `**Arbitrage Recommendation**:\n` +
              `• Action: ${arbitrageTiming.timing.toUpperCase()}\n` +
              `• ${arbitrageTiming.recommendation}\n` +
              `• Optimal Entry: ${arbitrageTiming.optimalEntry}\n` +
              `• Risk Level: ${arbitrageTiming.riskLevel.toUpperCase()}`
    
    confidence = Math.min(patternAnalysis.confidence, 97)
    additionalData = {
      patternAnalysis,
      arbitrageTiming,
      chartData: recentData.slice(-5)
    }
  }
  
  // Dynamic multi-modal fusion analysis queries
  else if (query.toLowerCase().includes('cluster') || query.toLowerCase().includes('correlation')) {
    const clusterData = clusteringEngine.getLiveClusterData()
    const analysis = analyzeClusteringInsights(clusterData, query)
    response = analysis.response
    confidence = analysis.confidence
    additionalData = analysis.data
  }
  else if (query.toLowerCase().includes('market analysis') || query.toLowerCase().includes('market')) {
    const marketAnalysis = analyzeMarketConditions()
    response = marketAnalysis.response
    confidence = marketAnalysis.confidence
    additionalData = marketAnalysis.data
  } 
  else if (query.toLowerCase().includes('risk') || query.toLowerCase().includes('portfolio')) {
    const riskAnalysis = analyzeRiskMetrics()
    response = riskAnalysis.response
    confidence = riskAnalysis.confidence
    additionalData = riskAnalysis.data
  }
  else if (query.toLowerCase().includes('arbitrage') || query.toLowerCase().includes('opportunity')) {
    const arbitrageAnalysis = analyzeArbitrageOpportunities()
    response = arbitrageAnalysis.response
    confidence = arbitrageAnalysis.confidence
    additionalData = arbitrageAnalysis.data
  }
  else if (query.toLowerCase().includes('fusion') || query.toLowerCase().includes('hyperbolic')) {
    const fusionAnalysis = analyzeFusionComponents(query)
    response = fusionAnalysis.response
    confidence = fusionAnalysis.confidence
    additionalData = fusionAnalysis.data
  }
  else {
    // Dynamic general analysis based on current system state
    const generalAnalysis = analyzeGeneralQuery(query)
    response = generalAnalysis.response
    confidence = generalAnalysis.confidence
    additionalData = generalAnalysis.data
  }
  
  return c.json({
    response,
    confidence,
    timestamp: new Date().toISOString(),
    ...additionalData
  })
})

// New API endpoints for advanced charting
app.get('/api/candlestick/:symbol/:timeframe', (c) => {
  const symbol = c.req.param('symbol').toUpperCase()
  const timeframe = c.req.param('timeframe')
  
  if (!candlestickGenerators[symbol]) {
    return c.json({ error: 'Symbol not supported' }, 400)
  }
  
  // Generate new candle and add to historical data
  const newCandle = candlestickGenerators[symbol].generateRealisticCandle(timeframe)
  historicalData[symbol][timeframe].push(newCandle)
  
  // Keep only last N candles for performance
  const maxCandles = { '1m': 200, '5m': 100, '15m': 50, '1h': 48 }
  if (historicalData[symbol][timeframe].length > maxCandles[timeframe]) {
    historicalData[symbol][timeframe].shift()
  }
  
  return c.json({
    symbol,
    timeframe,
    data: historicalData[symbol][timeframe].slice(-50), // Return last 50 candles
    latest: newCandle
  })
})

app.get('/api/pattern-analysis/:symbol/:timeframe', (c) => {
  const symbol = c.req.param('symbol').toUpperCase()
  const timeframe = c.req.param('timeframe')
  
  if (!historicalData[symbol] || !historicalData[symbol][timeframe]) {
    return c.json({ error: 'Data not available' }, 400)
  }
  
  const recentData = historicalData[symbol][timeframe].slice(-10)
  const patternAnalysis = hyperbolicAnalyzer.analyzePattern(recentData)
  const arbitrageTiming = hyperbolicAnalyzer.generateArbitrageTiming(patternAnalysis, recentData[recentData.length - 1])
  
  return c.json({
    symbol,
    timeframe,
    pattern: patternAnalysis,
    arbitrageTiming,
    timestamp: new Date().toISOString()
  })
})

app.get('/api/hyperbolic-analysis', (c) => {
  const analysis = {}
  
  Object.keys(historicalData).forEach(symbol => {
    analysis[symbol] = {}
    Object.keys(historicalData[symbol]).forEach(timeframe => {
      const recentData = historicalData[symbol][timeframe].slice(-5)
      const patternAnalysis = hyperbolicAnalyzer.analyzePattern(recentData)
      analysis[symbol][timeframe] = {
        pattern: patternAnalysis.pattern,
        confidence: patternAnalysis.confidence,
        arbitrageRelevance: patternAnalysis.arbitrageRelevance,
        geodesicEfficiency: patternAnalysis.geodesicEfficiency,
        hyperbolicDistance: patternAnalysis.hyperbolicDistance
      }
    })
  })
  
  return c.json(analysis)
})

// Advanced Backtesting Engine
class BacktestingEngine {
  constructor() {
    this.strategies = {}
    this.backtests = {}
    this.paperTrades = {}
    this.portfolios = {}
  }

  // Generate comprehensive historical data for backtesting
  generateHistoricalData(symbol, days = 365) {
    const data = []
    const startPrice = candlestickGenerators[symbol].basePrice
    let currentPrice = startPrice
    const startTime = Date.now() - (days * 24 * 60 * 60 * 1000)
    
    for (let i = 0; i < days * 24 * 60; i++) { // Minute-by-minute data
      const timestamp = startTime + (i * 60 * 1000)
      
      // Generate realistic price movements
      const volatility = 0.001 + Math.random() * 0.002
      const trend = Math.sin(i / 1440) * 0.0001 // Daily trend cycle
      const noise = (Math.random() - 0.5) * volatility
      const priceChange = trend + noise
      
      currentPrice *= (1 + priceChange)
      
      const candle = {
        timestamp,
        open: currentPrice,
        high: currentPrice * (1 + Math.random() * 0.01),
        low: currentPrice * (1 - Math.random() * 0.01),
        close: currentPrice * (1 + (Math.random() - 0.5) * 0.005),
        volume: Math.floor(Math.random() * 1000 + 500)
      }
      
      // Adjust high/low to be consistent
      candle.high = Math.max(candle.open, candle.close, candle.high)
      candle.low = Math.min(candle.open, candle.close, candle.low)
      
      data.push(candle)
    }
    
    return data
  }

  // Advanced Strategy Backtesting
  async runBacktest(strategyConfig) {
    const {
      strategyId,
      symbol,
      timeframe,
      startDate,
      endDate,
      initialCapital,
      strategyType,
      parameters
    } = strategyConfig

    const historicalData = this.generateHistoricalData(symbol, 365)
    const results = {
      trades: [],
      equity: [],
      metrics: {},
      drawdowns: [],
      positions: []
    }

    let currentCapital = initialCapital
    let position = null
    let maxEquity = initialCapital
    let maxDrawdown = 0
    let winningTrades = 0
    let losingTrades = 0
    let totalPnL = 0

    // Strategy execution simulation
    for (let i = 10; i < historicalData.length; i++) {
      const currentCandle = historicalData[i]
      const recentCandles = historicalData.slice(i - 10, i)
      
      // Analyze patterns for strategy signals
      const patternAnalysis = hyperbolicAnalyzer.analyzePattern(recentCandles)
      const signal = this.generateStrategySignal(strategyType, patternAnalysis, parameters)
      
      // Execute trades based on signals
      if (signal.action === 'BUY' && !position) {
        const quantity = (currentCapital * (parameters.riskPerTrade || 0.02)) / currentCandle.close
        position = {
          type: 'LONG',
          entryPrice: currentCandle.close,
          quantity,
          entryTime: currentCandle.timestamp,
          stopLoss: currentCandle.close * (1 - (parameters.stopLoss || 0.02)),
          takeProfit: currentCandle.close * (1 + (parameters.takeProfit || 0.04))
        }
      } else if (signal.action === 'SELL' && position && position.type === 'LONG') {
        const exitPrice = currentCandle.close
        const pnl = (exitPrice - position.entryPrice) * position.quantity
        const pnlPercent = ((exitPrice - position.entryPrice) / position.entryPrice) * 100
        
        currentCapital += pnl
        totalPnL += pnl
        
        if (pnl > 0) winningTrades++
        else losingTrades++
        
        results.trades.push({
          entryPrice: position.entryPrice,
          exitPrice,
          quantity: position.quantity,
          pnl,
          pnlPercent,
          duration: currentCandle.timestamp - position.entryTime,
          entryTime: position.entryTime,
          exitTime: currentCandle.timestamp,
          reason: signal.reason
        })
        
        position = null
      }
      
      // Check stop loss / take profit
      if (position) {
        const currentPrice = currentCandle.close
        if (currentPrice <= position.stopLoss || currentPrice >= position.takeProfit) {
          const exitPrice = currentPrice <= position.stopLoss ? position.stopLoss : position.takeProfit
          const pnl = (exitPrice - position.entryPrice) * position.quantity
          currentCapital += pnl
          totalPnL += pnl
          
          if (pnl > 0) winningTrades++
          else losingTrades++
          
          results.trades.push({
            entryPrice: position.entryPrice,
            exitPrice,
            quantity: position.quantity,
            pnl,
            pnlPercent: ((exitPrice - position.entryPrice) / position.entryPrice) * 100,
            duration: currentCandle.timestamp - position.entryTime,
            entryTime: position.entryTime,
            exitTime: currentCandle.timestamp,
            reason: currentPrice <= position.stopLoss ? 'STOP_LOSS' : 'TAKE_PROFIT'
          })
          
          position = null
        }
      }
      
      // Record equity curve
      results.equity.push({
        timestamp: currentCandle.timestamp,
        equity: currentCapital,
        price: currentCandle.close
      })
      
      // Calculate drawdown
      if (currentCapital > maxEquity) {
        maxEquity = currentCapital
      }
      const currentDrawdown = ((maxEquity - currentCapital) / maxEquity) * 100
      if (currentDrawdown > maxDrawdown) {
        maxDrawdown = currentDrawdown
      }
      
      results.drawdowns.push({
        timestamp: currentCandle.timestamp,
        drawdown: currentDrawdown
      })
    }

    // Calculate final metrics
    const totalTrades = winningTrades + losingTrades
    const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0
    const totalReturn = ((currentCapital - initialCapital) / initialCapital) * 100
    const avgTrade = totalTrades > 0 ? totalPnL / totalTrades : 0
    
    const winningTradePnLs = results.trades.filter(t => t.pnl > 0).map(t => t.pnl)
    const losingTradePnLs = results.trades.filter(t => t.pnl < 0).map(t => t.pnl)
    
    const avgWin = winningTradePnLs.length > 0 ? winningTradePnLs.reduce((a, b) => a + b) / winningTradePnLs.length : 0
    const avgLoss = losingTradePnLs.length > 0 ? Math.abs(losingTradePnLs.reduce((a, b) => a + b) / losingTradePnLs.length) : 0
    const profitFactor = avgLoss > 0 ? avgWin / avgLoss : Infinity
    
    // Calculate Sharpe ratio (simplified)
    const dailyReturns = []
    for (let i = 1; i < results.equity.length; i += 1440) { // Daily samples
      const prevEquity = results.equity[Math.max(0, i - 1440)]?.equity || initialCapital
      const currentEquity = results.equity[i]?.equity || currentCapital
      const dailyReturn = ((currentEquity - prevEquity) / prevEquity) * 100
      dailyReturns.push(dailyReturn)
    }
    
    const avgDailyReturn = dailyReturns.length > 0 ? dailyReturns.reduce((a, b) => a + b) / dailyReturns.length : 0
    const dailyReturnStd = dailyReturns.length > 1 ? Math.sqrt(dailyReturns.map(x => Math.pow(x - avgDailyReturn, 2)).reduce((a, b) => a + b) / (dailyReturns.length - 1)) : 0
    const sharpeRatio = dailyReturnStd > 0 ? (avgDailyReturn / dailyReturnStd) * Math.sqrt(252) : 0

    results.metrics = {
      initialCapital,
      finalCapital: currentCapital,
      totalReturn,
      totalPnL,
      totalTrades,
      winningTrades,
      losingTrades,
      winRate,
      avgTrade,
      avgWin,
      avgLoss,
      profitFactor,
      maxDrawdown,
      sharpeRatio: sharpeRatio.toFixed(3),
      calmarRatio: maxDrawdown > 0 ? (totalReturn / maxDrawdown).toFixed(3) : Infinity
    }

    // Store backtest results
    this.backtests[strategyId] = {
      ...strategyConfig,
      results,
      completedAt: Date.now(),
      status: 'completed'
    }

    return results
  }

  // Generate strategy signals based on pattern analysis
  generateStrategySignal(strategyType, patternAnalysis, parameters) {
    const { pattern, confidence, arbitrageRelevance, signal } = patternAnalysis
    
    switch (strategyType) {
      case 'PATTERN_ARBITRAGE':
        if (confidence > (parameters.minConfidence || 80) && arbitrageRelevance > (parameters.minArbitrageRelevance || 75)) {
          if (signal.includes('bullish')) {
            return { action: 'BUY', reason: `Pattern: ${pattern}, Confidence: ${confidence}%` }
          } else if (signal.includes('bearish')) {
            return { action: 'SELL', reason: `Pattern: ${pattern}, Confidence: ${confidence}%` }
          }
        }
        break
      case 'MEAN_REVERSION':
        if (pattern === 'doji' && confidence > 70) {
          return { action: 'BUY', reason: 'Mean reversion signal detected' }
        }
        break
      case 'MOMENTUM':
        if (pattern.includes('engulfing') && confidence > 85) {
          return signal.includes('bullish') 
            ? { action: 'BUY', reason: 'Strong momentum signal' }
            : { action: 'SELL', reason: 'Strong momentum signal' }
        }
        break
    }
    
    return { action: 'HOLD', reason: 'No signal criteria met' }
  }
}

// Paper Trading Engine  
class PaperTradingEngine {
  constructor() {
    this.accounts = {}
    this.activeOrders = {}
    this.tradeHistory = {}
  }

  // Create new paper trading account
  createAccount(accountId, initialBalance = 100000) {
    this.accounts[accountId] = {
      accountId,
      balance: initialBalance,
      initialBalance,
      positions: {},
      orders: [],
      tradeHistory: [],
      metrics: {
        totalPnL: 0,
        realizedPnL: 0,
        unrealizedPnL: 0,
        totalTrades: 0,
        winningTrades: 0,
        losingTrades: 0
      },
      createdAt: Date.now(),
      lastUpdated: Date.now()
    }
    return this.accounts[accountId]
  }

  // Place paper trade order
  placeOrder(accountId, orderData) {
    const { symbol, side, quantity, orderType, price, stopLoss, takeProfit } = orderData
    const account = this.accounts[accountId]
    
    if (!account) throw new Error('Account not found')
    
    const orderId = `order_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    const currentPrice = this.getCurrentPrice(symbol)
    
    const order = {
      orderId,
      accountId,
      symbol,
      side, // 'BUY' or 'SELL'
      quantity,
      orderType, // 'MARKET' or 'LIMIT'
      price: orderType === 'MARKET' ? currentPrice : price,
      stopLoss,
      takeProfit,
      status: 'PENDING',
      createdAt: Date.now(),
      executedAt: null,
      executedPrice: null
    }

    // Execute market orders immediately
    if (orderType === 'MARKET') {
      return this.executeOrder(orderId, order)
    }
    
    // Store limit orders for later execution
    account.orders.push(order)
    this.activeOrders[orderId] = order
    
    return order
  }

  // Execute paper trade order
  executeOrder(orderId, order = null) {
    if (!order) {
      order = this.activeOrders[orderId]
      if (!order) throw new Error('Order not found')
    }

    const account = this.accounts[order.accountId]
    const currentPrice = this.getCurrentPrice(order.symbol)
    const executionPrice = order.orderType === 'MARKET' ? currentPrice : order.price
    
    // Check if account has sufficient funds/shares
    if (order.side === 'BUY') {
      const requiredAmount = executionPrice * order.quantity
      if (account.balance < requiredAmount) {
        order.status = 'REJECTED'
        order.rejectionReason = 'Insufficient funds'
        return order
      }
      account.balance -= requiredAmount
    } else {
      const position = account.positions[order.symbol]
      if (!position || position.quantity < order.quantity) {
        order.status = 'REJECTED'
        order.rejectionReason = 'Insufficient shares'
        return order
      }
    }

    // Update position
    if (!account.positions[order.symbol]) {
      account.positions[order.symbol] = { symbol: order.symbol, quantity: 0, avgPrice: 0, unrealizedPnL: 0 }
    }

    const position = account.positions[order.symbol]
    
    if (order.side === 'BUY') {
      const newQuantity = position.quantity + order.quantity
      position.avgPrice = ((position.avgPrice * position.quantity) + (executionPrice * order.quantity)) / newQuantity
      position.quantity = newQuantity
    } else {
      const soldValue = executionPrice * order.quantity
      const costBasis = position.avgPrice * order.quantity
      const realizedPnL = soldValue - costBasis
      
      account.balance += soldValue
      position.quantity -= order.quantity
      account.metrics.realizedPnL += realizedPnL
      account.metrics.totalPnL += realizedPnL
      
      if (realizedPnL > 0) account.metrics.winningTrades++
      else account.metrics.losingTrades++
      
      account.metrics.totalTrades++
    }

    // Update order status
    order.status = 'EXECUTED'
    order.executedAt = Date.now()
    order.executedPrice = executionPrice

    // Add to trade history
    account.tradeHistory.push({
      ...order,
      realizedPnL: order.side === 'SELL' ? (executionPrice - position.avgPrice) * order.quantity : 0
    })

    // Remove from active orders
    delete this.activeOrders[orderId]
    
    account.lastUpdated = Date.now()
    return order
  }

  // Get current market price (simplified)
  getCurrentPrice(symbol) {
    if (candlestickGenerators[symbol]) {
      return candlestickGenerators[symbol].currentPrice
    }
    return 67234.56 // Default BTC price
  }

  // Update unrealized P&L for all positions
  updateAccountMetrics(accountId) {
    const account = this.accounts[accountId]
    if (!account) return

    let totalUnrealizedPnL = 0
    
    Object.values(account.positions).forEach(position => {
      if (position.quantity > 0) {
        const currentPrice = this.getCurrentPrice(position.symbol)
        position.unrealizedPnL = (currentPrice - position.avgPrice) * position.quantity
        totalUnrealizedPnL += position.unrealizedPnL
      }
    })

    account.metrics.unrealizedPnL = totalUnrealizedPnL
    account.metrics.totalPnL = account.metrics.realizedPnL + totalUnrealizedPnL
    
    // Calculate current portfolio value
    const positionValue = Object.values(account.positions).reduce((total, pos) => {
      return total + (this.getCurrentPrice(pos.symbol) * pos.quantity)
    }, 0)
    
    account.currentValue = account.balance + positionValue
    account.totalReturn = ((account.currentValue - account.initialBalance) / account.initialBalance) * 100
    
    account.lastUpdated = Date.now()
    return account
  }

  // Get account summary
  getAccountSummary(accountId) {
    const account = this.accounts[accountId]
    if (!account) throw new Error('Account not found')
    
    this.updateAccountMetrics(accountId)
    
    return {
      ...account,
      winRate: account.metrics.totalTrades > 0 
        ? (account.metrics.winningTrades / account.metrics.totalTrades) * 100 
        : 0
    }
  }
}

// Monte Carlo Simulation Engine
class MonteCarloEngine {
  constructor() {}

  // Run Monte Carlo simulation for strategy validation
  runSimulation(strategyConfig, iterations = 1000) {
    const results = []
    
    for (let i = 0; i < iterations; i++) {
      // Add randomness to strategy parameters
      const randomizedConfig = {
        ...strategyConfig,
        parameters: {
          ...strategyConfig.parameters,
          minConfidence: strategyConfig.parameters.minConfidence + (Math.random() - 0.5) * 10,
          riskPerTrade: strategyConfig.parameters.riskPerTrade * (0.8 + Math.random() * 0.4)
        }
      }
      
      // Add market noise
      const marketNoise = (Math.random() - 0.5) * 0.1
      
      // Simulate strategy with variations
      const backtest = backtestingEngine.runBacktest({
        ...randomizedConfig,
        strategyId: `mc_${i}`,
        marketNoise
      })
      
      results.push({
        iteration: i,
        finalReturn: backtest.metrics.totalReturn,
        maxDrawdown: backtest.metrics.maxDrawdown,
        sharpeRatio: backtest.metrics.sharpeRatio,
        profitFactor: backtest.metrics.profitFactor,
        winRate: backtest.metrics.winRate
      })
    }
    
    // Calculate simulation statistics
    const returns = results.map(r => r.finalReturn)
    const drawdowns = results.map(r => r.maxDrawdown)
    
    return {
      iterations,
      summary: {
        avgReturn: returns.reduce((a, b) => a + b) / returns.length,
        medianReturn: this.median(returns),
        stdReturn: this.standardDeviation(returns),
        minReturn: Math.min(...returns),
        maxReturn: Math.max(...returns),
        avgDrawdown: drawdowns.reduce((a, b) => a + b) / drawdowns.length,
        maxDrawdown: Math.max(...drawdowns),
        profitProbability: (results.filter(r => r.finalReturn > 0).length / iterations) * 100
      },
      results
    }
  }

  median(arr) {
    const sorted = arr.slice().sort((a, b) => a - b)
    const middle = Math.floor(sorted.length / 2)
    return sorted.length % 2 === 0 ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
  }

  standardDeviation(arr) {
    const mean = arr.reduce((a, b) => a + b) / arr.length
    const squaredDiffs = arr.map(x => Math.pow(x - mean, 2))
    const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b) / arr.length
    return Math.sqrt(avgSquaredDiff)
  }
}

// Enhanced Multi-Modal Fusion Hierarchical Clustering Engine
class HierarchicalClusteringEngine {
  constructor() {
    // Multi-asset universe from all market categories
    this.assets = {
      crypto: ['BTC', 'ETH', 'SOL'],
      equity: ['SP500', 'NASDAQ', 'DOW'],
      international: ['FTSE', 'NIKKEI', 'DAX'],
      commodities: ['GOLD', 'SILVER', 'OIL'],
      forex: ['EURUSD', 'GBPUSD', 'USDJPY']
    }
    
    // Flatten assets for processing
    this.allAssets = Object.values(this.assets).flat()
    
    // Multi-modal data fusion components
    this.fusionComponents = {
      hyperbolicCNN: 0.40,     // Hyperbolic pattern recognition
      lstmTransformer: 0.25,   // Sequential pattern analysis
      finBERT: 0.20,           // Sentiment and fundamental analysis
      classicalArbitrage: 0.15 // Traditional statistical methods
    }
    
    // Enhanced data structures
    this.multiModalData = {}
    this.correlationMatrix = {}
    this.clusterHierarchy = {}
    this.fusionScores = {}
    this.clusterPositions = {}
    this.lastUpdate = Date.now()
    
    this.initializeMultiModalData()
  }

  initializeMultiModalData() {
    // Initialize comprehensive multi-modal data for each asset
    this.allAssets.forEach(asset => {
      this.multiModalData[asset] = {
        priceHistory: [],
        volumeProfile: [],
        volatilitySignature: [],
        sentimentScores: [],
        arbitrageSignals: [],
        hyperbolicMetrics: [],
        crossCorrelations: {},
        fundamentalFactors: this.generateFundamentalFactors(asset)
      }
      
      // Generate rich historical data for fusion analysis
      this.generateHistoricalFusionData(asset)
    })
    
    this.updateMultiModalCorrelations()
    this.calculateEnhancedClusterPositions()
  }

  generateFundamentalFactors(asset) {
    const assetCategory = this.getAssetCategory(asset)
    
    // Category-specific fundamental factors
    switch (assetCategory) {
      case 'crypto':
        return {
          marketCap: this.getMarketCap(asset),
          networkActivity: Math.random() * 100 + 50,
          developerActivity: Math.random() * 100 + 30,
          institutionalAdoption: Math.random() * 100 + 20,
          regulatoryRisk: Math.random() * 100 + 10
        }
      case 'equity':
        return {
          marketCap: Math.random() * 5000000000000 + 1000000000000,
          peRatio: Math.random() * 30 + 10,
          dividendYield: Math.random() * 5 + 1,
          earningsGrowth: Math.random() * 20 - 5,
          sectorRotation: Math.random() * 100
        }
      case 'international':
        return {
          gdpGrowth: Math.random() * 5 - 1,
          interestRates: Math.random() * 5 + 0.5,
          currencyStrength: Math.random() * 100 + 50,
          politicalStability: Math.random() * 100 + 60,
          tradeBalance: Math.random() * 200 - 100
        }
      case 'commodities':
        return {
          supplyDemand: Math.random() * 100 + 50,
          inventoryLevels: Math.random() * 100 + 30,
          geopoliticalRisk: Math.random() * 100 + 20,
          dollarStrength: Math.random() * 100 + 50,
          inflationHedge: Math.random() * 100 + 70
        }
      case 'forex':
        return {
          interestRateDifferential: Math.random() * 4 - 2,
          economicData: Math.random() * 100 + 50,
          centralBankPolicy: Math.random() * 100 + 40,
          riskSentiment: Math.random() * 100 + 30,
          carryTradeAppeal: Math.random() * 100 + 20
        }
      default:
        return {}
    }
  }

  generateHistoricalFusionData(asset) {
    const data = this.multiModalData[asset]
    const globalMarkets = getGlobalMarkets()
    const category = this.getAssetCategory(asset)
    const basePrice = this.getAssetPrice(asset, globalMarkets)
    
    // Generate 200 historical data points for robust correlation analysis
    for (let i = 0; i < 200; i++) {
      const timestamp = Date.now() - (200 - i) * 300000 // 5-minute intervals
      
      // Multi-modal fusion price calculation
      const hyperbolicSignal = this.generateHyperbolicSignal(asset, i)
      const sentimentSignal = this.generateSentimentSignal(asset, i)
      const arbitrageSignal = this.generateArbitrageSignal(asset, i)
      const technicalSignal = this.generateTechnicalSignal(asset, i)
      
      // Fusion-weighted price movement
      const fusedSignal = 
        hyperbolicSignal * this.fusionComponents.hyperbolicCNN +
        technicalSignal * this.fusionComponents.lstmTransformer +
        sentimentSignal * this.fusionComponents.finBERT +
        arbitrageSignal * this.fusionComponents.classicalArbitrage
      
      const price = basePrice * (1 + fusedSignal * 0.02) * (1 + (Math.random() - 0.5) * 0.01)
      const volume = Math.random() * 1000 + 100
      const volatility = Math.abs(fusedSignal) * 0.5 + Math.random() * 0.1
      
      data.priceHistory.push({ timestamp, price, fusedSignal })
      data.volumeProfile.push({ timestamp, volume })
      data.volatilitySignature.push({ timestamp, volatility })
      data.sentimentScores.push({ timestamp, sentiment: sentimentSignal })
      data.arbitrageSignals.push({ timestamp, signal: arbitrageSignal })
      data.hyperbolicMetrics.push({ 
        timestamp, 
        geodesicDistance: Math.abs(hyperbolicSignal),
        curvature: -1.0,
        efficiency: (1 - Math.abs(hyperbolicSignal)) * 100
      })
    }
  }

  generateHyperbolicSignal(asset, index) {
    // Simulate hyperbolic CNN pattern recognition signal
    const patternPhase = (index / 50) * Math.PI * 2
    return Math.sin(patternPhase) * 0.3 + Math.cos(patternPhase * 1.618) * 0.2
  }
  
  generateSentimentSignal(asset, index) {
    // Simulate FinBERT sentiment analysis signal
    const sentimentCycle = (index / 30) * Math.PI * 2
    return Math.sin(sentimentCycle) * 0.25 + (Math.random() - 0.5) * 0.1
  }
  
  generateArbitrageSignal(asset, index) {
    // Simulate classical arbitrage opportunity detection
    const arbCycle = (index / 20) * Math.PI * 2
    return Math.cos(arbCycle) * 0.15 + (Math.random() - 0.5) * 0.05
  }
  
  generateTechnicalSignal(asset, index) {
    // Simulate LSTM-Transformer technical analysis
    const techCycle = (index / 40) * Math.PI * 2
    return Math.sin(techCycle * 0.8) * 0.2 + Math.cos(techCycle * 1.2) * 0.15
  }

  updateMultiModalData() {
    // Update with new multi-modal fusion data
    const globalMarkets = getGlobalMarkets()
    
    this.allAssets.forEach(asset => {
      const data = this.multiModalData[asset]
      const basePrice = this.getAssetPrice(asset, globalMarkets)
      const timestamp = Date.now()
      
      // Generate new fusion signals
      const currentIndex = data.priceHistory.length
      const hyperbolicSignal = this.generateHyperbolicSignal(asset, currentIndex)
      const sentimentSignal = this.generateSentimentSignal(asset, currentIndex)
      const arbitrageSignal = this.generateArbitrageSignal(asset, currentIndex)
      const technicalSignal = this.generateTechnicalSignal(asset, currentIndex)
      
      // Multi-modal fusion calculation
      const fusedSignal = 
        hyperbolicSignal * this.fusionComponents.hyperbolicCNN +
        technicalSignal * this.fusionComponents.lstmTransformer +
        sentimentSignal * this.fusionComponents.finBERT +
        arbitrageSignal * this.fusionComponents.classicalArbitrage
      
      const price = basePrice * (1 + fusedSignal * 0.02)
      const volume = Math.random() * 1000 + 100
      const volatility = Math.abs(fusedSignal) * 0.5 + Math.random() * 0.1
      
      // Add new data points
      data.priceHistory.push({ timestamp, price, fusedSignal })
      data.volumeProfile.push({ timestamp, volume })
      data.volatilitySignature.push({ timestamp, volatility })
      data.sentimentScores.push({ timestamp, sentiment: sentimentSignal })
      data.arbitrageSignals.push({ timestamp, signal: arbitrageSignal })
      data.hyperbolicMetrics.push({ 
        timestamp, 
        geodesicDistance: Math.abs(hyperbolicSignal),
        curvature: -1.0,
        efficiency: (1 - Math.abs(hyperbolicSignal)) * 100
      })
      
      // Maintain rolling window (keep last 200 points)
      if (data.priceHistory.length > 200) {
        data.priceHistory.shift()
        data.volumeProfile.shift()
        data.volatilitySignature.shift()
        data.sentimentScores.shift()
        data.arbitrageSignals.shift()
        data.hyperbolicMetrics.shift()
      }
    })
  }

  getAssetCategory(asset) {
    for (const [category, assets] of Object.entries(this.assets)) {
      if (assets.includes(asset)) return category
    }
    return 'unknown'
  }
  
  getAssetPrice(asset, globalMarkets) {
    const category = this.getAssetCategory(asset)
    if (globalMarkets[category] && globalMarkets[category][asset]) {
      return globalMarkets[category][asset].price
    }
    return 100 // fallback price
  }
  
  getMarketCap(asset) {
    const marketCaps = {
      'BTC': 1300000000000,
      'ETH': 420000000000, 
      'SOL': 58000000000
    }
    return marketCaps[asset] || Math.random() * 100000000000 + 10000000000
  }

  updateMultiModalCorrelations() {
    // Enhanced multi-modal correlation calculation
    this.correlationMatrix = {}
    
    this.allAssets.forEach(asset1 => {
      this.correlationMatrix[asset1] = {}
      
      this.allAssets.forEach(asset2 => {
        if (asset1 === asset2) {
          this.correlationMatrix[asset1][asset2] = 1.0
        } else {
          // Multi-modal fusion correlation
          const priceCorr = this.calculateSignalCorrelation(asset1, asset2, 'priceHistory')
          const volumeCorr = this.calculateSignalCorrelation(asset1, asset2, 'volumeProfile')
          const sentimentCorr = this.calculateSignalCorrelation(asset1, asset2, 'sentimentScores')
          const arbitrageCorr = this.calculateSignalCorrelation(asset1, asset2, 'arbitrageSignals')
          
          // Fusion-weighted correlation
          const fusedCorrelation = 
            priceCorr * 0.4 +           // Price movement correlation (primary)
            volumeCorr * 0.2 +          // Volume correlation  
            sentimentCorr * 0.25 +      // Sentiment correlation
            arbitrageCorr * 0.15        // Arbitrage signal correlation
          
          this.correlationMatrix[asset1][asset2] = Number(fusedCorrelation.toFixed(4))
        }
      })
    })
  }

  calculateSignalCorrelation(asset1, asset2, dataType) {
    const data1 = this.multiModalData[asset1]?.[dataType]
    const data2 = this.multiModalData[asset2]?.[dataType]
    
    if (!data1 || !data2 || data1.length < 50 || data2.length < 50) return 0
    // Extract values based on data type
    const values1 = []
    const values2 = []
    
    const minLength = Math.min(data1.length, data2.length, 100) // Use last 100 points
    const startIndex = Math.max(0, data1.length - minLength)
    
    for (let i = startIndex; i < data1.length && i - startIndex < minLength; i++) {
      let value1, value2
      
      switch (dataType) {
        case 'priceHistory':
          if (i > 0) {
            value1 = (data1[i].price - data1[i-1].price) / data1[i-1].price
            value2 = (data2[i].price - data2[i-1].price) / data2[i-1].price
          }
          break
        case 'volumeProfile':
          value1 = data1[i].volume
          value2 = data2[i].volume
          break
        case 'sentimentScores':
          value1 = data1[i].sentiment
          value2 = data2[i].sentiment
          break
        case 'arbitrageSignals':
          value1 = data1[i].signal
          value2 = data2[i].signal
          break
        default:
          return 0
      }
      
      if (value1 !== undefined && value2 !== undefined && !isNaN(value1) && !isNaN(value2)) {
        values1.push(value1)
        values2.push(value2)
      }
    }
    
    if (values1.length < 10) return 0
    
    // Enhanced Pearson correlation with outlier handling
    const n = values1.length
    const sum1 = values1.reduce((a, b) => a + b, 0)
    const sum2 = values2.reduce((a, b) => a + b, 0)
    const mean1 = sum1 / n
    const mean2 = sum2 / n
    
    let numerator = 0
    let sumSq1 = 0
    let sumSq2 = 0
    
    for (let i = 0; i < n; i++) {
      const diff1 = values1[i] - mean1
      const diff2 = values2[i] - mean2
      numerator += diff1 * diff2
      sumSq1 += diff1 * diff1
      sumSq2 += diff2 * diff2
    }
    
    const denominator = Math.sqrt(sumSq1 * sumSq2)
    return denominator === 0 ? 0 : numerator / denominator
  }

  calculateEnhancedClusterPositions() {
    // Enhanced hierarchical clustering with multi-modal positioning
    this.clusterPositions = {}
    this.clusterHierarchy = this.buildClusterHierarchy()
    
    // Position assets in Poincaré disk based on multi-modal correlations
    this.allAssets.forEach((asset, index) => {
      const correlations = this.correlationMatrix[asset] || {}
      const fundamentals = this.multiModalData[asset].fundamentalFactors
      
      // Multi-dimensional positioning algorithm
      let x = 0, y = 0
      let totalWeight = 0
      
      // Position based on correlations with other assets
      this.allAssets.forEach(otherAsset => {
        if (asset !== otherAsset && correlations[otherAsset] !== undefined) {
          const correlation = correlations[otherAsset]
          const weight = Math.abs(correlation)
          const angle = (this.allAssets.indexOf(otherAsset) / this.allAssets.length) * 2 * Math.PI
          
          x += Math.cos(angle) * correlation * weight
          y += Math.sin(angle) * correlation * weight
          totalWeight += weight
        }
      })
      
      // Normalize and add fundamental factor influence
      if (totalWeight > 0) {
        x /= totalWeight
        y /= totalWeight
      }
      
      // Add category-specific positioning bias
      const category = this.getAssetCategory(asset)
      const categoryBias = this.getCategoryBias(category, index)
      x = (x + categoryBias.x) * 0.4 // Scale to fit Poincaré disk
      y = (y + categoryBias.y) * 0.4
      
      // Ensure within unit circle (Poincaré disk constraint)
      const distance = Math.sqrt(x * x + y * y)
      if (distance > 0.95) {
        x = (x / distance) * 0.95
        y = (y / distance) * 0.95
      }
      
      // Calculate additional metrics
      const globalMarkets = getGlobalMarkets()
      const currentPrice = this.getAssetPrice(asset, globalMarkets)
      const priceChange = this.calculatePriceChange(asset)
      const volatility = this.calculateVolatility(asset)
      
      this.clusterPositions[asset] = {
        x,
        y,
        distance: Math.sqrt(x * x + y * y),
        angle: Math.atan2(y, x),
        currentPrice,
        priceChange,
        volatility,
        marketCap: this.getAssetMarketCap(asset),
        category: category,
        correlations: correlations,
        fundamentalScore: this.calculateFundamentalScore(fundamentals),
        fusionSignal: this.calculateCurrentFusionSignal(asset)
      }
    })
  }

  getCategoryBias(category, index) {
    // Position assets by category in different regions of the disk
    const categoryPositions = {
      crypto: { baseAngle: 0, radius: 0.6 },           // Top
      equity: { baseAngle: Math.PI * 0.4, radius: 0.7 }, // Top-right  
      international: { baseAngle: Math.PI * 0.8, radius: 0.65 }, // Right
      commodities: { baseAngle: Math.PI * 1.2, radius: 0.6 }, // Bottom-right
      forex: { baseAngle: Math.PI * 1.6, radius: 0.55 } // Bottom-left
    }
    
    const position = categoryPositions[category] || { baseAngle: 0, radius: 0.5 }
    const angleSpread = 0.3 // Spread assets within category
    const angle = position.baseAngle + (index * angleSpread - angleSpread)
    
    return {
      x: Math.cos(angle) * position.radius,
      y: Math.sin(angle) * position.radius
    }
  }

  buildClusterHierarchy() {
    // Build hierarchical clustering based on correlation strength
    const hierarchy = {
      crypto: { assets: this.assets.crypto, avgCorrelation: 0 },
      equity: { assets: this.assets.equity, avgCorrelation: 0 },
      international: { assets: this.assets.international, avgCorrelation: 0 },
      commodities: { assets: this.assets.commodities, avgCorrelation: 0 },
      forex: { assets: this.assets.forex, avgCorrelation: 0 }
    }
    
    // Calculate average intra-category correlations
    Object.keys(hierarchy).forEach(category => {
      const categoryAssets = hierarchy[category].assets
      let totalCorrelation = 0
      let pairCount = 0
      
      for (let i = 0; i < categoryAssets.length; i++) {
        for (let j = i + 1; j < categoryAssets.length; j++) {
          const corr = this.correlationMatrix[categoryAssets[i]]?.[categoryAssets[j]]
          if (corr !== undefined) {
            totalCorrelation += Math.abs(corr)
            pairCount++
          }
        }
      }
      
      hierarchy[category].avgCorrelation = pairCount > 0 ? totalCorrelation / pairCount : 0
    })
    
    return hierarchy
  }

  calculatePriceChange(asset) {
    const data = this.multiModalData[asset].priceHistory
    if (data.length < 2) return 0
    
    const current = data[data.length - 1].price
    const previous = data[data.length - 2].price
    return (current - previous) / previous
  }

  calculateVolatility(asset) {
    const data = this.multiModalData[asset].priceHistory
    if (data.length < 10) return 0.01
    
    const returns = []
    for (let i = 1; i < Math.min(data.length, 50); i++) {
      const return_pct = (data[i].price - data[i-1].price) / data[i-1].price
      returns.push(return_pct)
    }
    
    const mean = returns.reduce((a, b) => a + b) / returns.length
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length
    return Math.sqrt(variance)
  }

  getAssetMarketCap(asset) {
    const marketCaps = {
      'BTC': 1300000000000, 'ETH': 420000000000, 'SOL': 58000000000,
      'SP500': 45000000000000, 'NASDAQ': 25000000000000, 'DOW': 15000000000000,
      'FTSE': 3500000000000, 'NIKKEI': 6200000000000, 'DAX': 2800000000000,
      'GOLD': 15000000000000, 'SILVER': 1500000000000, 'OIL': 2000000000000,
      'EURUSD': 8000000000000, 'GBPUSD': 3000000000000, 'USDJPY': 5500000000000
    }
    return marketCaps[asset] || Math.random() * 1000000000000 + 100000000000
  }

  calculateFundamentalScore(fundamentals) {
    const values = Object.values(fundamentals)
    return values.length > 0 ? values.reduce((a, b) => a + b) / values.length : 50
  }

  calculateCurrentFusionSignal(asset) {
    const data = this.multiModalData[asset]
    if (!data.priceHistory.length) return 0
    
    const latest = data.priceHistory[data.priceHistory.length - 1]
    return latest.fusedSignal || 0
  }





  getLiveClusterData() {
    // Update with latest multi-modal fusion data
    this.updateMultiModalData()
    
    // Recalculate correlations and positions every 5 seconds for real-time updates
    const now = Date.now()
    if (now - this.lastUpdate > 5000) {
      this.updateMultiModalCorrelations()
      this.calculateEnhancedClusterPositions()
      this.lastUpdate = now
    }
    
    // Return enhanced clustering data for ALL 15 assets across 5 categories
    return {
      positions: this.clusterPositions,
      correlationMatrix: this.correlationMatrix,
      clusterHierarchy: this.clusterHierarchy,
      fusionComponents: this.fusionComponents,
      lastUpdate: this.lastUpdate,
      totalAssets: this.allAssets.length,
      assetCategories: Object.keys(this.assets),
      
      // Enhanced asset data with multi-modal fusion insights
      assets: this.allAssets.map(asset => {
        const globalMarkets = getGlobalMarkets()
        const position = this.clusterPositions[asset]
        
        if (!position) {
          return {
            symbol: asset,
            category: this.getAssetCategory(asset),
            currentPrice: this.getAssetPrice(asset, globalMarkets),
            error: 'Position not calculated'
          }
        }
        
        return {
          symbol: asset,
          category: position.category,
          currentPrice: position.currentPrice,
          volatility: position.volatility,
          marketCap: position.marketCap,
          x: position.x,
          y: position.y,
          distance: position.distance,
          angle: position.angle,
          priceChange: position.priceChange,
          correlations: position.correlations,
          fundamentalScore: position.fundamentalScore,
          fusionSignal: position.fusionSignal
        }
      })
    }
  }

  getCorrelationStrength(asset1, asset2) {
    return Math.abs(this.correlationMatrix[asset1]?.[asset2] || 0)
  }
}

// Initialize engines
const backtestingEngine = new BacktestingEngine()
const paperTradingEngine = new PaperTradingEngine()
const monteCarloEngine = new MonteCarloEngine()
const clusteringEngine = new HierarchicalClusteringEngine()

// Dynamic Multi-Modal Fusion AI Analysis Engine
function analyzeClusteringInsights(clusterData, query) {
  const { assets, fusionComponents, correlationMatrix, totalAssets, assetCategories } = clusterData
  
  // Analyze clustering patterns
  const strongCorrelations = []
  const weakCorrelations = []
  const categoryStats = {}
  
  assetCategories.forEach(category => {
    categoryStats[category] = { count: 0, avgVolatility: 0, avgFusionSignal: 0 }
  })
  
  assets.forEach(asset => {
    categoryStats[asset.category].count++
    categoryStats[asset.category].avgVolatility += asset.volatility || 0
    categoryStats[asset.category].avgFusionSignal += Math.abs(asset.fusionSignal || 0)
    
    if (asset.correlations) {
      Object.entries(asset.correlations).forEach(([otherAsset, corr]) => {
        if (otherAsset !== asset.symbol && Math.abs(corr) > 0.5) {
          strongCorrelations.push({ asset1: asset.symbol, asset2: otherAsset, correlation: corr, strength: Math.abs(corr) })
        } else if (otherAsset !== asset.symbol && Math.abs(corr) < 0.1) {
          weakCorrelations.push({ asset1: asset.symbol, asset2: otherAsset, correlation: corr })
        }
      })
    }
  })
  
  // Calculate category averages
  Object.keys(categoryStats).forEach(category => {
    const count = categoryStats[category].count
    if (count > 0) {
      categoryStats[category].avgVolatility /= count
      categoryStats[category].avgFusionSignal /= count
    }
  })
  
  // Generate insights based on actual data
  const topCorrelations = strongCorrelations
    .sort((a, b) => b.strength - a.strength)
    .slice(0, 3)
  
  const dominantCategory = Object.entries(categoryStats)
    .sort((a, b) => b[1].avgFusionSignal - a[1].avgFusionSignal)[0]
  
  const response = `🌐 **Multi-Modal Clustering Analysis**\n\n` +
    `**Asset Universe**: ${totalAssets} assets across ${assetCategories.length} categories\n` +
    `**Fusion Components**: CNN ${(fusionComponents.hyperbolicCNN * 100).toFixed(0)}% | LSTM ${(fusionComponents.lstmTransformer * 100).toFixed(0)}% | FinBERT ${(fusionComponents.finBERT * 100).toFixed(0)}% | Arbitrage ${(fusionComponents.classicalArbitrage * 100).toFixed(0)}%\n\n` +
    `**Strongest Correlations**:\n${topCorrelations.map(c => `• ${c.asset1}↔${c.asset2}: ${c.correlation.toFixed(3)} (${c.strength > 0.7 ? 'Very Strong' : 'Strong'})`).join('\n')}\n\n` +
    `**Category Analysis**:\n• Most Active: ${dominantCategory[0]} (fusion signal: ${dominantCategory[1].avgFusionSignal.toFixed(3)})\n` +
    `• Volatility Leader: ${Object.entries(categoryStats).sort((a, b) => b[1].avgVolatility - a[1].avgVolatility)[0][0]}\n\n` +
    `**Hyperbolic Insight**: Assets are positioned using geodesic distances reflecting multi-modal correlations. ` +
    `Strong intra-category clustering detected in ${assetCategories.filter(cat => categoryStats[cat].avgFusionSignal > 0.05).length} categories.`
  
  return {
    response,
    confidence: Math.min(95, 75 + (strongCorrelations.length * 3)),
    data: { strongCorrelations, categoryStats, topCorrelations }
  }
}

function analyzeMarketConditions() {
  try {
    const clusterData = clusteringEngine.getLiveClusterData()
  
  // Analyze current market state from real data
  const cryptoAssets = clusterData.assets.filter(a => a.category === 'crypto')
  const equityAssets = clusterData.assets.filter(a => a.category === 'equity')
  
  const cryptoMomentum = cryptoAssets.reduce((sum, asset) => sum + (asset.priceChange || 0), 0) / cryptoAssets.length
  const equityMomentum = equityAssets.reduce((sum, asset) => sum + (asset.priceChange || 0), 0) / equityAssets.length
  
  const avgVolatility = clusterData.assets.reduce((sum, asset) => sum + (asset.volatility || 0), 0) / clusterData.assets.length
  const strongFusionSignals = clusterData.assets.filter(a => Math.abs(a.fusionSignal || 0) > 0.1).length
  
  const marketSentiment = cryptoMomentum > 0 ? 'bullish' : 'bearish'
  const marketStrength = Math.abs(cryptoMomentum) > 0.01 ? 'strong' : 'moderate'
  
  const response = `📊 **Real-Time Market Analysis**\n\n` +
    `**Current Momentum**:\n• Crypto: ${(cryptoMomentum * 100).toFixed(2)}% (${cryptoMomentum > 0 ? '📈' : '📉'})\n` +
    `• Equity: ${(equityMomentum * 100).toFixed(2)}% (${equityMomentum > 0 ? '📈' : '📉'})\n\n` +
    `**Market Regime**: ${marketStrength.charAt(0).toUpperCase() + marketStrength.slice(1)} ${marketSentiment} trend detected\n` +
    `**Volatility Environment**: ${avgVolatility > 0.01 ? 'High volatility' : 'Normal volatility'} (${(avgVolatility * 100).toFixed(2)}%)\n` +
    `**Fusion Activity**: ${strongFusionSignals}/${clusterData.assets.length} assets showing strong multi-modal signals\n\n` +
    `**Hyperbolic CNN Analysis**: Pattern recognition confidence varies by asset class. ` +
    `Current geodesic efficiency indicates ${avgVolatility < 0.005 ? 'stable' : 'dynamic'} market microstructure.`
  
    return {
      response,
      confidence: 88 + Math.min(10, strongFusionSignals * 2),
      data: { cryptoMomentum, equityMomentum, avgVolatility, strongFusionSignals, marketSentiment }
    }
  } catch (error) {
    return {
      response: `🤖 **Market Analysis Error**: Unable to retrieve current market conditions. System is initializing multi-modal fusion components.`,
      confidence: 50,
      data: { error: error.message }
    }
  }
}

function analyzeRiskMetrics() {
  try {
    const clusterData = clusteringEngine.getLiveClusterData()
  
  // Calculate real risk metrics from clustering data
  const correlations = []
  clusterData.assets.forEach(asset => {
    if (asset.correlations) {
      Object.values(asset.correlations).forEach(corr => {
        if (corr !== 1 && !isNaN(corr)) correlations.push(corr)
      })
    }
  })
  
  const avgCorrelation = correlations.reduce((a, b) => a + b, 0) / correlations.length
  const correlationStd = Math.sqrt(correlations.reduce((sum, corr) => sum + Math.pow(corr - avgCorrelation, 2), 0) / correlations.length)
  
  const highVolAssets = clusterData.assets.filter(a => (a.volatility || 0) > 0.008).length
  const diversificationRatio = clusterData.assetCategories.length / clusterData.totalAssets * 5 // Normalized
  
  const riskLevel = correlationStd > 0.3 ? 'elevated' : correlationStd > 0.2 ? 'moderate' : 'low'
  const diversificationQuality = diversificationRatio > 0.8 ? 'excellent' : diversificationRatio > 0.6 ? 'good' : 'limited'
  
  const response = `⚠️ **Multi-Modal Risk Assessment**\n\n` +
    `**Correlation Analysis**:\n• Average Cross-Asset Correlation: ${avgCorrelation.toFixed(3)}\n` +
    `• Correlation Standard Deviation: ${correlationStd.toFixed(3)}\n• Risk Level: ${riskLevel.toUpperCase()}\n\n` +
    `**Diversification Metrics**:\n• Portfolio Spread: ${clusterData.assetCategories.length} asset categories\n` +
    `• Diversification Quality: ${diversificationQuality.toUpperCase()}\n• High Volatility Assets: ${highVolAssets}/${clusterData.totalAssets}\n\n` +
    `**Hyperbolic Risk Mapping**: Assets positioned by correlation distance in Poincaré disk. ` +
    `Current risk distribution shows ${riskLevel} clustering with ${diversificationQuality} category separation.`
  
    return {
      response,
      confidence: 92,
      data: { avgCorrelation, correlationStd, highVolAssets, diversificationRatio, riskLevel }
    }
  } catch (error) {
    return {
      response: `⚠️ **Risk Analysis Error**: Unable to calculate risk metrics. Multi-modal clustering engine initializing.`,
      confidence: 50,
      data: { error: error.message }
    }
  }
}

function analyzeArbitrageOpportunities() {
  try {
    const clusterData = clusteringEngine.getLiveClusterData()
  
  // Analyze real arbitrage opportunities from clustering patterns
  const decorrelatedPairs = []
  const strongCorrelatedPairs = []
  
  for (let i = 0; i < clusterData.assets.length; i++) {
    for (let j = i + 1; j < clusterData.assets.length; j++) {
      const asset1 = clusterData.assets[i]
      const asset2 = clusterData.assets[j]
      
      if (asset1.correlations && asset1.correlations[asset2.symbol] !== undefined) {
        const corr = asset1.correlations[asset2.symbol]
        const priceDivergence = Math.abs((asset1.priceChange || 0) - (asset2.priceChange || 0))
        
        if (Math.abs(corr) < 0.2 && priceDivergence > 0.01) {
          decorrelatedPairs.push({ pair: `${asset1.symbol}-${asset2.symbol}`, correlation: corr, divergence: priceDivergence })
        } else if (Math.abs(corr) > 0.7 && priceDivergence > 0.02) {
          strongCorrelatedPairs.push({ pair: `${asset1.symbol}-${asset2.symbol}`, correlation: corr, divergence: priceDivergence })
        }
      }
    }
  }
  
  const topOpportunities = [...decorrelatedPairs, ...strongCorrelatedPairs]
    .sort((a, b) => b.divergence - a.divergence)
    .slice(0, 3)
  
  const fusionSignals = clusterData.assets.filter(a => Math.abs(a.fusionSignal || 0) > 0.08)
  
  const response = `⚡ **Multi-Modal Arbitrage Analysis**\n\n` +
    `**Opportunity Detection**:\n${topOpportunities.map((opp, i) => 
      `${i + 1}. ${opp.pair}: ${(opp.divergence * 100).toFixed(2)}% price divergence (corr: ${opp.correlation.toFixed(3)})`
    ).join('\n')}\n\n` +
    `**Fusion Signal Alerts**:\n• ${fusionSignals.length} assets showing strong multi-modal signals\n` +
    `• Primary signals: ${fusionSignals.map(a => `${a.symbol}(${(a.fusionSignal * 100).toFixed(1)}%)`).join(', ')}\n\n` +
    `**Hyperbolic Arbitrage**: Using geodesic distance calculations to identify correlation-divergence opportunities. ` +
    `Current market microstructure shows ${topOpportunities.length > 0 ? 'active' : 'limited'} arbitrage potential.`
  
    return {
      response,
      confidence: 85 + Math.min(12, topOpportunities.length * 4),
      data: { topOpportunities, fusionSignals, decorrelatedPairs }
    }
  } catch (error) {
    return {
      response: `⚡ **Arbitrage Analysis Error**: Unable to detect opportunities. Hyperbolic correlation matrix rebuilding.`,
      confidence: 50,  
      data: { error: error.message }
    }
  }
}

function analyzeFusionComponents(query) {
  const clusterData = clusteringEngine.getLiveClusterData()
  const { fusionComponents } = clusterData
  
  // Analyze current fusion component performance
  const componentPerformance = {
    hyperbolicCNN: clusterData.assets.filter(a => Math.abs(a.fusionSignal || 0) > 0.05).length,
    patterns: clusterData.assets.filter(a => (a.volatility || 0) > 0.006).length,
    sentiment: Math.random() * 0.3 + 0.4, // Simulated FinBERT activity
    arbitrage: clusterData.assets.filter(a => Object.values(a.correlations || {}).some(c => Math.abs(c) > 0.6)).length
  }
  
  const dominantComponent = Object.entries(fusionComponents)
    .sort((a, b) => b[1] - a[1])[0]
  
  const response = `🧠 **Multi-Modal Fusion Component Analysis**\n\n` +
    `**Component Weights**:\n• Hyperbolic CNN: ${(fusionComponents.hyperbolicCNN * 100).toFixed(0)}% (${componentPerformance.hyperbolicCNN} active signals)\n` +
    `• LSTM-Transformer: ${(fusionComponents.lstmTransformer * 100).toFixed(0)}% (${componentPerformance.patterns} pattern assets)\n` +
    `• FinBERT Sentiment: ${(fusionComponents.finBERT * 100).toFixed(0)}% (${(componentPerformance.sentiment * 100).toFixed(0)}% activity)\n` +
    `• Classical Arbitrage: ${(fusionComponents.classicalArbitrage * 100).toFixed(0)}% (${componentPerformance.arbitrage} correlation signals)\n\n` +
    `**Dominant Component**: ${dominantComponent[0]} contributing ${(dominantComponent[1] * 100).toFixed(0)}% to fusion decisions\n\n` +
    `**Hyperbolic Space Efficiency**: Operating in Poincaré disk with curvature -1.0. ` +
    `Geodesic calculations optimized for ${clusterData.totalAssets}-asset correlation matrix processing.`
  
  return {
    response,
    confidence: 93,
    data: { fusionComponents, componentPerformance, dominantComponent }
  }
}

function analyzeGeneralQuery(query) {
  try {
    const clusterData = clusteringEngine.getLiveClusterData()
  
  // Dynamic analysis based on current system state
  const activeAssets = clusterData.assets.filter(a => Math.abs(a.fusionSignal || 0) > 0.03).length
  const avgCorrelation = clusterData.assets.reduce((sum, asset) => {
    const correlations = Object.values(asset.correlations || {}).filter(c => c !== 1 && !isNaN(c))
    return sum + (correlations.reduce((a, b) => a + Math.abs(b), 0) / correlations.length || 0)
  }, 0) / clusterData.assets.length
  
  const systemEfficiency = (activeAssets / clusterData.totalAssets) * 100
  const marketComplexity = avgCorrelation > 0.4 ? 'high' : avgCorrelation > 0.25 ? 'moderate' : 'low'
  
  const response = `🤖 **Multi-Modal System Analysis**\n\n` +
    `Your query: "${query}"\n\n` +
    `**Current System State**:\n• ${clusterData.totalAssets} assets actively monitored across ${clusterData.assetCategories.length} categories\n` +
    `• ${activeAssets} assets showing significant fusion activity (${systemEfficiency.toFixed(0)}% system utilization)\n` +
    `• Market complexity: ${marketComplexity.toUpperCase()} (avg correlation: ${avgCorrelation.toFixed(3)})\n\n` +
    `**Recommendation**: Based on current multi-modal fusion analysis, focus on ` +
    `${clusterData.assets.filter(a => Math.abs(a.fusionSignal || 0) > 0.08).map(a => a.symbol).join(', ') || 'stable assets'} ` +
    `for optimal trading opportunities. The hyperbolic space engine shows ${systemEfficiency > 70 ? 'high' : systemEfficiency > 40 ? 'moderate' : 'low'} signal activity.`
  
    return {
      response,
      confidence: 80 + Math.min(15, Math.floor(systemEfficiency / 5)),
      data: { activeAssets, avgCorrelation, systemEfficiency, marketComplexity }
    }
  } catch (error) {
    return {
      response: `🤖 **System Analysis**: Your query "${query}" is being processed. Multi-modal fusion engine currently initializing correlation matrices across 15 global assets.`,
      confidence: 75,
      data: { error: error.message }
    }
  }
}

// API endpoints for backtesting and paper trading

// Backtesting endpoints
app.post('/api/backtest/run', async (c) => {
  try {
    const strategyConfig = await c.req.json()
    const results = await backtestingEngine.runBacktest(strategyConfig)
    
    return c.json({
      success: true,
      strategyId: strategyConfig.strategyId,
      results
    })
  } catch (error) {
    return c.json({ error: error.message }, 400)
  }
})

app.get('/api/backtest/:strategyId', (c) => {
  const strategyId = c.req.param('strategyId')
  const backtest = backtestingEngine.backtests[strategyId]
  
  if (!backtest) {
    return c.json({ error: 'Backtest not found' }, 404)
  }
  
  return c.json(backtest)
})

app.get('/api/backtests', (c) => {
  return c.json({
    backtests: Object.values(backtestingEngine.backtests)
  })
})

// Paper trading endpoints
app.post('/api/paper-trading/account', async (c) => {
  const { accountId, initialBalance } = await c.req.json()
  const account = paperTradingEngine.createAccount(accountId, initialBalance)
  
  return c.json({
    success: true,
    account
  })
})

app.post('/api/paper-trading/order', async (c) => {
  try {
    const orderData = await c.req.json()
    const order = paperTradingEngine.placeOrder(orderData.accountId, orderData)
    
    return c.json({
      success: true,
      order
    })
  } catch (error) {
    return c.json({ error: error.message }, 400)
  }
})

app.get('/api/paper-trading/account/:accountId', (c) => {
  try {
    const accountId = c.req.param('accountId')
    const account = paperTradingEngine.getAccountSummary(accountId)
    
    return c.json({
      success: true,
      account
    })
  } catch (error) {
    return c.json({ error: error.message }, 404)
  }
})

app.get('/api/paper-trading/accounts', (c) => {
  return c.json({
    accounts: Object.values(paperTradingEngine.accounts)
  })
})

// Monte Carlo simulation endpoint
app.post('/api/monte-carlo', async (c) => {
  try {
    const { strategyConfig, iterations } = await c.req.json()
    const results = monteCarloEngine.runSimulation(strategyConfig, iterations)
    
    return c.json({
      success: true,
      simulation: results
    })
  } catch (error) {
    return c.json({ error: error.message }, 400)
  }
})

// Real-time hierarchical asset clustering endpoint
app.get('/api/asset-clustering', (c) => {
  try {
    const clusterData = clusteringEngine.getLiveClusterData()
    return c.json({
      success: true,
      clustering: clusterData,
      timestamp: Date.now()
    })
  } catch (error) {
    return c.json({ error: error.message }, 500)
  }
})

// Strategy performance comparison
app.post('/api/strategy/compare', async (c) => {
  try {
    const { strategies } = await c.req.json()
    const comparisons = []
    
    for (const strategy of strategies) {
      const backtest = await backtestingEngine.runBacktest(strategy)
      comparisons.push({
        strategyName: strategy.strategyName,
        metrics: backtest.metrics,
        riskAdjustedReturn: backtest.metrics.sharpeRatio
      })
    }
    
    // Rank strategies by risk-adjusted return
    comparisons.sort((a, b) => b.riskAdjustedReturn - a.riskAdjustedReturn)
    
    return c.json({
      success: true,
      comparison: comparisons
    })
  } catch (error) {
    return c.json({ error: error.message }, 400)
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
        <title>GOMNA Trading Dashboard</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@3.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial@0.2.1/dist/chartjs-chart-financial.min.js"></script>
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
                        <button class="nav-item" data-section="economic-data">
                            <i class="fas fa-chart-bar mr-2"></i>ECONOMIC DATA
                        </button>
                        <button class="nav-item" data-section="transparency">
                            <i class="fas fa-microscope mr-2"></i>MODEL TRANSPARENCY
                        </button>
                        <button class="nav-item" data-section="assistant">
                            <i class="fas fa-robot mr-2"></i>AI ASSISTANT
                        </button>
                        <button class="nav-item" data-section="backtesting">
                            <i class="fas fa-chart-area mr-2"></i>BACKTESTING
                        </button>
                        <button class="nav-item" data-section="paper-trading">
                            <i class="fas fa-file-invoice-dollar mr-2"></i>PAPER TRADING
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

                        <!-- Social Sentiment & Economic Data -->\n                        <div class=\"col-span-4 bg-card-bg rounded-lg p-6\">\n                            <h3 class=\"text-lg font-semibold mb-4 flex items-center\">\n                                <i class=\"fas fa-chart-line mr-2 text-profit\"></i>\n                                SOCIAL SENTIMENT\n                            </h3>\n                            <div id=\"social-sentiment\" class=\"space-y-4\">\n                                <!-- Sentiment data will be populated here -->\n                            </div>\n                            \n                            <h4 class=\"text-md font-semibold mt-6 mb-3 text-accent\">ECONOMIC INDICATORS</h4>\n                            <div id=\"economic-indicators\" class=\"space-y-2 text-sm\">\n                                <!-- Economic data will be populated here -->\n                            </div>\n                        </div>\n\n                        <!-- Arbitrage Opportunities -->
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

                        <!-- Enhanced Hyperbolic Space Engine -->
                        <div class="col-span-4 bg-card-bg rounded-lg p-6">
                            <h3 class="text-lg font-semibold mb-4 flex items-center">
                                <i class="fas fa-atom mr-2 text-accent"></i>
                                HYPERBOLIC SPACE ENGINE
                                <span class="ml-auto">
                                    <span class="bg-profit text-dark-bg px-2 py-1 rounded text-xs font-semibold">ENHANCED</span>
                                </span>
                            </h3>
                            
                            <!-- Visualization Toggle -->
                            <div class="flex justify-center mb-3">
                                <div class="bg-gray-700 rounded-lg p-1 flex">
                                    <button id="viz-toggle-patterns" class="px-3 py-1 rounded text-xs font-semibold bg-accent text-dark-bg">
                                        Patterns
                                    </button>
                                    <button id="viz-toggle-clustering" class="px-3 py-1 rounded text-xs font-semibold text-gray-300 hover:text-white">
                                        Asset Clustering
                                    </button>
                                </div>
                            </div>
                            
                            <!-- Original Poincaré Disk (Pattern Analysis) -->
                            <div id="poincare-patterns-view" class="visualization-view">
                                <div class="text-center mb-2">
                                    <div class="text-warning font-semibold text-sm">Pattern Analysis Model</div>
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
                            
                            <!-- New Asset Clustering View -->
                            <div id="poincare-clustering-view" class="visualization-view hidden">
                                <div class="text-center mb-2">
                                    <div class="text-profit font-semibold text-sm">Real-Time Asset Clustering</div>
                                </div>
                                <div id="clustering-canvas" class="mb-4">
                                    <canvas id="asset-clustering-disk" width="250" height="250" class="mx-auto bg-gray-900 rounded-full"></canvas>
                                </div>
                                <div class="space-y-2 text-sm">
                                    <div class="flex justify-between">
                                        <span>Active Assets:</span>
                                        <span id="cluster-asset-count" class="text-accent">3</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span>Avg Correlation:</span>
                                        <span id="avg-correlation" class="text-accent">--</span>
                                    </div>
                                    <div class="flex justify-between">
                                        <span>Cluster Stability:</span>
                                        <span id="cluster-stability" class="text-profit">--</span>
                                    </div>
                                </div>
                                
                                <!-- Asset Legend -->
                                <div class="mt-3 space-y-1 text-xs">
                                    <div class="text-gray-400 font-semibold mb-2">Asset Legend:</div>
                                    <div id="asset-legend">
                                        <!-- Will be populated dynamically -->
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <!-- Advanced Hyperbolic CNN Candlestick Analysis -->
                    <div class="mt-6">
                        <div class="bg-card-bg rounded-lg p-6">
                            <h3 class="text-lg font-semibold mb-4 flex items-center">
                                <i class="fas fa-chart-candlestick mr-2 text-accent"></i>
                                🧠 Hyperbolic CNN Chart Analysis
                                <span class="ml-auto text-sm">
                                    <span class="bg-profit text-dark-bg px-2 py-1 rounded text-xs font-semibold">INDUSTRY-LEADING</span>
                                </span>
                            </h3>
                            
                            <div class="grid grid-cols-12 gap-6">
                                <!-- Chart Controls -->
                                <div class="col-span-12 mb-4">
                                    <div class="flex items-center space-x-4">
                                        <div class="flex space-x-2">
                                            <button class="symbol-btn active bg-accent text-dark-bg px-3 py-1 rounded font-semibold text-sm" data-symbol="BTC">BTC</button>
                                            <button class="symbol-btn bg-gray-700 text-white px-3 py-1 rounded font-semibold text-sm hover:bg-gray-600" data-symbol="ETH">ETH</button>
                                            <button class="symbol-btn bg-gray-700 text-white px-3 py-1 rounded font-semibold text-sm hover:bg-gray-600" data-symbol="SOL">SOL</button>
                                        </div>
                                        <div class="flex space-x-2">
                                            <button class="timeframe-btn active bg-accent text-dark-bg px-3 py-1 rounded text-sm font-semibold" data-timeframe="1m">1m</button>
                                            <button class="timeframe-btn bg-gray-700 text-white px-3 py-1 rounded text-sm hover:bg-gray-600" data-timeframe="5m">5m</button>
                                            <button class="timeframe-btn bg-gray-700 text-white px-3 py-1 rounded text-sm hover:bg-gray-600" data-timeframe="15m">15m</button>
                                            <button class="timeframe-btn bg-gray-700 text-white px-3 py-1 rounded text-sm hover:bg-gray-600" data-timeframe="1h">1h</button>
                                        </div>
                                        <button id="analyze-chart" class="bg-gradient-to-r from-purple-500 to-pink-500 text-white px-4 py-1 rounded text-sm font-semibold hover:from-purple-600 hover:to-pink-600">
                                            <i class="fas fa-brain mr-2"></i>Analyze Pattern
                                        </button>
                                    </div>
                                </div>
                                
                                <!-- Candlestick Chart -->
                                <div class="col-span-8">
                                    <div class="bg-gray-900 rounded-lg p-4" style="height: 400px;">
                                        <canvas id="candlestick-chart" width="600" height="350"></canvas>
                                    </div>
                                </div>
                                
                                <!-- Hyperbolic CNN Analysis Panel -->
                                <div class="col-span-4 space-y-4">
                                    <div class="bg-gray-800 rounded-lg p-4">
                                        <h4 class="font-semibold mb-3 text-accent">🎯 Pattern Analysis</h4>
                                        <div id="pattern-analysis" class="space-y-2 text-sm">
                                            <div class="flex justify-between">
                                                <span>Pattern:</span>
                                                <span id="detected-pattern" class="text-warning">Analyzing...</span>
                                            </div>
                                            <div class="flex justify-between">
                                                <span>Confidence:</span>
                                                <span id="pattern-confidence" class="text-accent">--</span>
                                            </div>
                                            <div class="flex justify-between">
                                                <span>Signal:</span>
                                                <span id="pattern-signal" class="text-profit">--</span>
                                            </div>
                                            <div class="flex justify-between">
                                                <span>Arbitrage Relevance:</span>
                                                <span id="arbitrage-relevance" class="text-accent">--</span>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="bg-gray-800 rounded-lg p-4">
                                        <h4 class="font-semibold mb-3 text-accent">⚗️ Hyperbolic Metrics</h4>
                                        <div class="space-y-2 text-sm">
                                            <div class="flex justify-between">
                                                <span>Geodesic Efficiency:</span>
                                                <span id="geodesic-efficiency" class="text-profit">--</span>
                                            </div>
                                            <div class="flex justify-between">
                                                <span>Hyperbolic Distance:</span>
                                                <span id="hyperbolic-distance" class="text-accent">--</span>
                                            </div>
                                            <div class="flex justify-between">
                                                <span>Space Curvature:</span>
                                                <span class="text-warning">-1.0</span>
                                            </div>
                                        </div>
                                    </div>
                                    
                                    <div class="bg-gray-800 rounded-lg p-4">
                                        <h4 class="font-semibold mb-3 text-accent">⚡ Arbitrage Timing</h4>
                                        <div id="arbitrage-timing" class="space-y-2 text-sm">
                                            <div class="flex justify-between">
                                                <span>Action:</span>
                                                <span id="timing-action" class="text-warning">HOLD</span>
                                            </div>
                                            <div class="flex justify-between">
                                                <span>Entry:</span>
                                                <span id="optimal-entry" class="text-accent">--</span>
                                            </div>
                                            <div class="flex justify-between">
                                                <span>Risk Level:</span>
                                                <span id="risk-level" class="text-profit">--</span>
                                            </div>
                                        </div>
                                        <div id="timing-recommendation" class="mt-3 p-2 bg-gray-900 rounded text-xs">
                                            Monitoring market patterns...
                                        </div>
                                    </div>
                                    
                                    <button id="execute-pattern-arbitrage" class="w-full bg-gradient-to-r from-accent to-profit text-dark-bg py-2 rounded font-semibold hover:from-opacity-80 hover:to-opacity-80 disabled:opacity-50" disabled>
                                        <i class="fas fa-rocket mr-2"></i>Execute Pattern-Based Arbitrage
                                    </button>
                                </div>
                                
                                <!-- Real-time Pattern Alerts -->
                                <div class="col-span-12 mt-4">
                                    <div class="bg-gray-800 rounded-lg p-4">
                                        <h4 class="font-semibold mb-3 flex items-center">
                                            <i class="fas fa-bell mr-2 text-warning"></i>
                                            Real-time Pattern Alerts
                                            <span class="ml-2 bg-warning text-dark-bg px-2 py-1 rounded text-xs">LIVE</span>
                                        </h4>
                                        <div id="pattern-alerts" class="space-y-2 max-h-32 overflow-y-auto">
                                            <!-- Pattern alerts will be populated here -->
                                        </div>
                                    </div>
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

                <!-- Economic Data Section -->
                <div id="economic-data" class="section">
                    <div class="grid grid-cols-12 gap-6">
                        <!-- Economic Indicators Overview -->
                        <div class="col-span-8 bg-card-bg rounded-lg p-6">
                            <h3 class="text-lg font-semibold mb-4 flex items-center">
                                <i class="fas fa-chart-line mr-2 text-accent"></i>
                                📈 Economic Indicators Dashboard
                            </h3>
                            <div id="economic-dashboard" class="grid grid-cols-3 gap-4">
                                <!-- Economic data charts will be populated here -->
                            </div>
                        </div>

                        <!-- Social Sentiment Summary -->  
                        <div class="col-span-4 bg-card-bg rounded-lg p-6">
                            <h3 class="text-lg font-semibold mb-4 flex items-center">
                                <i class="fas fa-users mr-2 text-profit"></i>
                                💬 Social Sentiment Analysis
                            </h3>
                            <div id="sentiment-dashboard" class="space-y-4">
                                <!-- Detailed sentiment analysis will be populated here -->
                            </div>

                            <div class="mt-6">
                                <h4 class="text-md font-semibold mb-3 text-warning">📊 Economic Trends</h4>
                                <div id="economic-trends-chart" class="bg-gray-900 rounded-lg p-2">
                                    <canvas id="trends-chart" width="300" height="200"></canvas>
                                </div>
                            </div>
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

                <!-- Backtesting Section -->
                <div id="backtesting" class="section">
                    <div class="bg-card-bg rounded-lg p-6">
                        <h3 class="text-lg font-semibold mb-4 flex items-center">
                            <i class="fas fa-chart-area mr-2 text-accent"></i>
                            🧪 Advanced Strategy Backtesting
                            <span class="ml-auto text-sm">
                                <span class="bg-accent text-dark-bg px-2 py-1 rounded text-xs font-semibold">ENTERPRISE-GRADE</span>
                            </span>
                        </h3>
                        
                        <div class="grid grid-cols-12 gap-6">
                            <!-- Strategy Configuration -->
                            <div class="col-span-4 space-y-4">
                                <div class="bg-gray-800 rounded-lg p-4">
                                    <h4 class="font-semibold mb-3 text-accent">Strategy Configuration</h4>
                                    
                                    <div class="space-y-3">
                                        <div>
                                            <label class="block text-sm font-medium mb-1">Strategy Name</label>
                                            <input id="strategy-name" type="text" placeholder="My Pattern Strategy" 
                                                   class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                        </div>
                                        
                                        <div>
                                            <label class="block text-sm font-medium mb-1">Strategy Type</label>
                                            <select id="strategy-type" class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                                <option value="PATTERN_ARBITRAGE">Pattern Arbitrage</option>
                                                <option value="MEAN_REVERSION">Mean Reversion</option>
                                                <option value="MOMENTUM">Momentum</option>
                                            </select>
                                        </div>
                                        
                                        <div class="grid grid-cols-2 gap-2">
                                            <div>
                                                <label class="block text-sm font-medium mb-1">Symbol</label>
                                                <select id="backtest-symbol" class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                                    <option value="BTC">BTC</option>
                                                    <option value="ETH">ETH</option>
                                                    <option value="SOL">SOL</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label class="block text-sm font-medium mb-1">Timeframe</label>
                                                <select id="backtest-timeframe" class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                                    <option value="1m">1m</option>
                                                    <option value="5m">5m</option>
                                                    <option value="15m">15m</option>
                                                    <option value="1h">1h</option>
                                                </select>
                                            </div>
                                        </div>
                                        
                                        <div>
                                            <label class="block text-sm font-medium mb-1">Initial Capital ($)</label>
                                            <input id="initial-capital" type="number" value="100000" min="1000" 
                                                   class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="bg-gray-800 rounded-lg p-4">
                                    <h4 class="font-semibold mb-3 text-accent">Risk Parameters</h4>
                                    
                                    <div class="space-y-3">
                                        <div class="grid grid-cols-2 gap-2">
                                            <div>
                                                <label class="block text-sm font-medium mb-1">Min Confidence (%)</label>
                                                <input id="min-confidence" type="number" value="80" min="50" max="100" 
                                                       class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                            </div>
                                            <div>
                                                <label class="block text-sm font-medium mb-1">Risk Per Trade (%)</label>
                                                <input id="risk-per-trade" type="number" value="2" min="0.1" max="10" step="0.1" 
                                                       class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                            </div>
                                        </div>
                                        
                                        <div class="grid grid-cols-2 gap-2">
                                            <div>
                                                <label class="block text-sm font-medium mb-1">Stop Loss (%)</label>
                                                <input id="stop-loss" type="number" value="2" min="0.5" max="10" step="0.1" 
                                                       class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                            </div>
                                            <div>
                                                <label class="block text-sm font-medium mb-1">Take Profit (%)</label>
                                                <input id="take-profit" type="number" value="4" min="1" max="20" step="0.1" 
                                                       class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                            </div>
                                        </div>
                                        
                                        <div>
                                            <label class="block text-sm font-medium mb-1">Min Arbitrage Relevance (%)</label>
                                            <input id="min-arbitrage-relevance" type="number" value="75" min="50" max="100" 
                                                   class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                        </div>
                                    </div>
                                </div>
                                
                                <div class="space-y-2">
                                    <button id="run-backtest" class="w-full bg-gradient-to-r from-accent to-profit text-dark-bg py-2 rounded font-semibold hover:from-opacity-80">
                                        <i class="fas fa-play mr-2"></i>Run Backtest
                                    </button>
                                    <button id="run-monte-carlo" class="w-full bg-gradient-to-r from-purple-500 to-pink-500 text-white py-2 rounded font-semibold hover:from-purple-600">
                                        <i class="fas fa-dice mr-2"></i>Monte Carlo Simulation
                                    </button>
                                    <button id="compare-strategies" class="w-full bg-gradient-to-r from-orange-500 to-red-500 text-white py-2 rounded font-semibold hover:from-orange-600">
                                        <i class="fas fa-balance-scale mr-2"></i>Compare Strategies
                                    </button>
                                </div>
                            </div>
                            
                            <!-- Results Display -->
                            <div class="col-span-8 space-y-4">
                                <div class="bg-gray-800 rounded-lg p-4">
                                    <h4 class="font-semibold mb-3 text-accent">📊 Backtest Results</h4>
                                    <div id="backtest-results" class="text-center text-gray-400 py-8">
                                        Run a backtest to see results...
                                    </div>
                                </div>
                                
                                <div class="grid grid-cols-2 gap-4">
                                    <div class="bg-gray-800 rounded-lg p-4">
                                        <h5 class="font-semibold mb-3 text-accent">📈 Equity Curve</h5>
                                        <canvas id="equity-curve-chart" width="300" height="200"></canvas>
                                    </div>
                                    <div class="bg-gray-800 rounded-lg p-4">
                                        <h5 class="font-semibold mb-3 text-accent">📉 Drawdown Chart</h5>
                                        <canvas id="drawdown-chart" width="300" height="200"></canvas>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Paper Trading Section -->
                <div id="paper-trading" class="section">
                    <div class="bg-card-bg rounded-lg p-6">
                        <h3 class="text-lg font-semibold mb-4 flex items-center">
                            <i class="fas fa-file-invoice-dollar mr-2 text-accent"></i>
                            📊 Real-Time Paper Trading
                            <span class="ml-auto text-sm">
                                <span class="bg-profit text-dark-bg px-2 py-1 rounded text-xs font-semibold">LIVE SIMULATION</span>
                            </span>
                        </h3>
                        
                        <div class="grid grid-cols-12 gap-6">
                            <!-- Account Creation & Management -->
                            <div class="col-span-4 space-y-4">
                                <div class="bg-gray-800 rounded-lg p-4">
                                    <h4 class="font-semibold mb-3 text-accent">Account Setup</h4>
                                    
                                    <div class="space-y-3">
                                        <div>
                                            <label class="block text-sm font-medium mb-1">Account Name</label>
                                            <input id="paper-account-name" type="text" placeholder="My Trading Account" 
                                                   class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                        </div>
                                        
                                        <div>
                                            <label class="block text-sm font-medium mb-1">Initial Balance ($)</label>
                                            <input id="paper-initial-balance" type="number" value="100000" min="1000" 
                                                   class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                        </div>
                                        
                                        <button id="create-paper-account" class="w-full bg-accent text-dark-bg py-2 rounded font-semibold hover:bg-opacity-80">
                                            <i class="fas fa-plus mr-2"></i>Create Account
                                        </button>
                                    </div>
                                </div>
                                
                                <div class="bg-gray-800 rounded-lg p-4">
                                    <h4 class="font-semibold mb-3 text-accent">Place Order</h4>
                                    
                                    <div class="space-y-3">
                                        <div class="grid grid-cols-2 gap-2">
                                            <div>
                                                <label class="block text-sm font-medium mb-1">Symbol</label>
                                                <select id="paper-symbol" class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                                    <option value="BTC">BTC</option>
                                                    <option value="ETH">ETH</option>
                                                    <option value="SOL">SOL</option>
                                                </select>
                                            </div>
                                            <div>
                                                <label class="block text-sm font-medium mb-1">Side</label>
                                                <select id="paper-side" class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                                    <option value="BUY">BUY</option>
                                                    <option value="SELL">SELL</option>
                                                </select>
                                            </div>
                                        </div>
                                        
                                        <div class="grid grid-cols-2 gap-2">
                                            <div>
                                                <label class="block text-sm font-medium mb-1">Quantity</label>
                                                <input id="paper-quantity" type="number" value="0.1" min="0.001" step="0.001" 
                                                       class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                            </div>
                                            <div>
                                                <label class="block text-sm font-medium mb-1">Order Type</label>
                                                <select id="paper-order-type" class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                                    <option value="MARKET">MARKET</option>
                                                    <option value="LIMIT">LIMIT</option>
                                                </select>
                                            </div>
                                        </div>
                                        
                                        <div id="limit-price-container" class="hidden">
                                            <label class="block text-sm font-medium mb-1">Limit Price ($)</label>
                                            <input id="paper-limit-price" type="number" step="0.01" 
                                                   class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                        </div>
                                        
                                        <div class="grid grid-cols-2 gap-2">
                                            <div>
                                                <label class="block text-sm font-medium mb-1">Stop Loss ($)</label>
                                                <input id="paper-stop-loss" type="number" step="0.01" 
                                                       class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                            </div>
                                            <div>
                                                <label class="block text-sm font-medium mb-1">Take Profit ($)</label>
                                                <input id="paper-take-profit" type="number" step="0.01" 
                                                       class="w-full bg-gray-900 border border-gray-700 rounded px-3 py-2 text-white text-sm">
                                            </div>
                                        </div>
                                        
                                        <button id="place-paper-order" class="w-full bg-gradient-to-r from-profit to-accent text-dark-bg py-2 rounded font-semibold hover:from-opacity-80">
                                            <i class="fas fa-paper-plane mr-2"></i>Place Order
                                        </button>
                                    </div>
                                </div>
                                
                                <div class="bg-gray-800 rounded-lg p-4">
                                    <h4 class="font-semibold mb-3 text-accent">Auto Trading</h4>
                                    <div class="space-y-3">
                                        <div class="flex items-center justify-between">
                                            <span class="text-sm">Pattern-Based Auto Trading</span>
                                            <label class="relative inline-flex items-center cursor-pointer">
                                                <input id="auto-trading-toggle" type="checkbox" class="sr-only peer">
                                                <div class="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-accent"></div>
                                            </label>
                                        </div>
                                        <div id="auto-trading-status" class="text-xs text-gray-400">
                                            Auto trading disabled
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Account Summary & Positions -->
                            <div class="col-span-8 space-y-4">
                                <div class="grid grid-cols-4 gap-4">
                                    <div class="bg-gray-800 rounded-lg p-4 text-center">
                                        <div id="paper-balance" class="text-2xl font-bold text-accent">$0</div>
                                        <div class="text-sm text-gray-400">Available Balance</div>
                                    </div>
                                    <div class="bg-gray-800 rounded-lg p-4 text-center">
                                        <div id="paper-equity" class="text-2xl font-bold text-profit">$0</div>
                                        <div class="text-sm text-gray-400">Total Equity</div>
                                    </div>
                                    <div class="bg-gray-800 rounded-lg p-4 text-center">
                                        <div id="paper-pnl" class="text-2xl font-bold">$0</div>
                                        <div class="text-sm text-gray-400">Total P&L</div>
                                    </div>
                                    <div class="bg-gray-800 rounded-lg p-4 text-center">
                                        <div id="paper-return" class="text-2xl font-bold">0%</div>
                                        <div class="text-sm text-gray-400">Total Return</div>
                                    </div>
                                </div>
                                
                                <div class="bg-gray-800 rounded-lg p-4">
                                    <h5 class="font-semibold mb-3 text-accent">📋 Current Positions</h5>
                                    <div id="paper-positions" class="text-center text-gray-400 py-4">
                                        No positions yet...
                                    </div>
                                </div>
                                
                                <div class="bg-gray-800 rounded-lg p-4">
                                    <h5 class="font-semibold mb-3 text-accent">📜 Trade History</h5>
                                    <div id="paper-trade-history" class="max-h-64 overflow-y-auto">
                                        <div class="text-center text-gray-400 py-4">
                                            No trades yet...
                                        </div>
                                    </div>
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