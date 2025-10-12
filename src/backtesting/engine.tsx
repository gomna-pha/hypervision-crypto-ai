/**
 * GOMNA Trading - Advanced Backtesting Engine
 * 
 * Production-ready backtesting engine with academic rigor and industry standards
 * Eliminates common backtesting pitfalls and implements proper statistical analysis
 */

import { 
  EnhancedMarketDataGenerator, 
  TransactionCostModel, 
  RiskManagementEngine,
  type MarketData,
  type Position,
  type Trade,
  type RiskMetrics,
  type BacktestConfig,
  type BacktestResult
} from './index'

// ============================================================================
// PERFORMANCE ANALYTICS ENGINE
// ============================================================================

class PerformanceAnalytics {
  
  static calculateRiskMetrics(
    equity: Array<{timestamp: number, equity: number}>,
    trades: Trade[],
    benchmarkReturns: number[],
    riskFreeRate: number,
    initialCapital: number
  ): RiskMetrics {
    
    const returns = this.calculateReturns(equity)
    const annualizedReturn = this.calculateAnnualizedReturn(returns)
    const volatility = this.calculateVolatility(returns)
    
    return {
      totalReturn: ((equity[equity.length - 1]?.equity || initialCapital) - initialCapital) / initialCapital * 100,
      annualizedReturn,
      volatility: volatility * Math.sqrt(252) * 100, // Annualized volatility
      sharpeRatio: this.calculateSharpeRatio(returns, riskFreeRate),
      sortinoRatio: this.calculateSortinoRatio(returns, riskFreeRate),
      calmarRatio: this.calculateCalmarRatio(returns, equity),
      maxDrawdown: this.calculateMaxDrawdown(equity),
      maxDrawdownDuration: this.calculateMaxDrawdownDuration(equity),
      var95: this.calculateVaR(returns, 0.95),
      var99: this.calculateVaR(returns, 0.99),
      cvar95: this.calculateCVaR(returns, 0.95),
      beta: this.calculateBeta(returns, benchmarkReturns),
      alpha: this.calculateAlpha(returns, benchmarkReturns, riskFreeRate),
      informationRatio: this.calculateInformationRatio(returns, benchmarkReturns),
      treynorRatio: this.calculateTreynorRatio(returns, benchmarkReturns, riskFreeRate),
      trackingError: this.calculateTrackingError(returns, benchmarkReturns),
      winRate: this.calculateWinRate(trades),
      profitFactor: this.calculateProfitFactor(trades),
      payoffRatio: this.calculatePayoffRatio(trades)
    }
  }

  private static calculateReturns(equity: Array<{timestamp: number, equity: number}>): number[] {
    const returns: number[] = []
    for (let i = 1; i < equity.length; i++) {
      const ret = (equity[i].equity - equity[i-1].equity) / equity[i-1].equity
      returns.push(ret)
    }
    return returns
  }

  private static calculateAnnualizedReturn(returns: number[]): number {
    if (returns.length === 0) return 0
    const totalReturn = returns.reduce((acc, ret) => acc * (1 + ret), 1) - 1
    const periodsPerYear = 252 // Trading days
    const annualizationFactor = periodsPerYear / returns.length
    return (Math.pow(1 + totalReturn, annualizationFactor) - 1) * 100
  }

  private static calculateVolatility(returns: number[]): number {
    if (returns.length <= 1) return 0
    const mean = returns.reduce((a, b) => a + b) / returns.length
    const variance = returns.reduce((acc, ret) => acc + Math.pow(ret - mean, 2), 0) / (returns.length - 1)
    return Math.sqrt(variance)
  }

  private static calculateSharpeRatio(returns: number[], riskFreeRate: number): number {
    const dailyRiskFreeRate = riskFreeRate / 252
    const excessReturns = returns.map(ret => ret - dailyRiskFreeRate)
    const meanExcessReturn = excessReturns.reduce((a, b) => a + b) / excessReturns.length
    const volatility = this.calculateVolatility(excessReturns)
    return volatility > 0 ? (meanExcessReturn / volatility) * Math.sqrt(252) : 0
  }

  private static calculateSortinoRatio(returns: number[], riskFreeRate: number): number {
    const dailyRiskFreeRate = riskFreeRate / 252
    const excessReturns = returns.map(ret => ret - dailyRiskFreeRate)
    const meanExcessReturn = excessReturns.reduce((a, b) => a + b) / excessReturns.length
    
    // Downside deviation (only negative returns)
    const negativeReturns = excessReturns.filter(ret => ret < 0)
    if (negativeReturns.length === 0) return Infinity
    
    const downsideVariance = negativeReturns.reduce((acc, ret) => acc + ret * ret, 0) / negativeReturns.length
    const downsideDeviation = Math.sqrt(downsideVariance)
    
    return downsideDeviation > 0 ? (meanExcessReturn / downsideDeviation) * Math.sqrt(252) : 0
  }

  private static calculateCalmarRatio(returns: number[], equity: Array<{timestamp: number, equity: number}>): number {
    const annualizedReturn = this.calculateAnnualizedReturn(returns)
    const maxDrawdown = this.calculateMaxDrawdown(equity)
    return maxDrawdown > 0 ? annualizedReturn / maxDrawdown : 0
  }

  private static calculateMaxDrawdown(equity: Array<{timestamp: number, equity: number}>): number {
    let maxEquity = equity[0]?.equity || 0
    let maxDrawdown = 0
    
    for (const point of equity) {
      maxEquity = Math.max(maxEquity, point.equity)
      const drawdown = (maxEquity - point.equity) / maxEquity * 100
      maxDrawdown = Math.max(maxDrawdown, drawdown)
    }
    
    return maxDrawdown
  }

  private static calculateMaxDrawdownDuration(equity: Array<{timestamp: number, equity: number}>): number {
    let maxEquity = equity[0]?.equity || 0
    let maxDuration = 0
    let currentDuration = 0
    let inDrawdown = false
    
    for (const point of equity) {
      if (point.equity >= maxEquity) {
        maxEquity = point.equity
        if (inDrawdown) {
          maxDuration = Math.max(maxDuration, currentDuration)
          currentDuration = 0
          inDrawdown = false
        }
      } else {
        if (!inDrawdown) {
          inDrawdown = true
          currentDuration = 0
        }
        currentDuration++
      }
    }
    
    return Math.max(maxDuration, currentDuration)
  }

  private static calculateVaR(returns: number[], confidence: number): number {
    const sorted = [...returns].sort((a, b) => a - b)
    const index = Math.floor((1 - confidence) * sorted.length)
    return sorted[index] * 100 || 0
  }

  private static calculateCVaR(returns: number[], confidence: number): number {
    const sorted = [...returns].sort((a, b) => a - b)
    const cutoff = Math.floor((1 - confidence) * sorted.length)
    const tailReturns = sorted.slice(0, cutoff)
    const avgTailReturn = tailReturns.reduce((a, b) => a + b, 0) / tailReturns.length
    return (avgTailReturn || 0) * 100
  }

  private static calculateBeta(returns: number[], benchmarkReturns: number[]): number {
    if (returns.length !== benchmarkReturns.length || returns.length < 2) return 1
    
    const meanReturn = returns.reduce((a, b) => a + b) / returns.length
    const meanBenchmark = benchmarkReturns.reduce((a, b) => a + b) / benchmarkReturns.length
    
    let covariance = 0
    let benchmarkVariance = 0
    
    for (let i = 0; i < returns.length; i++) {
      const returnDiff = returns[i] - meanReturn
      const benchmarkDiff = benchmarkReturns[i] - meanBenchmark
      covariance += returnDiff * benchmarkDiff
      benchmarkVariance += benchmarkDiff * benchmarkDiff
    }
    
    covariance /= (returns.length - 1)
    benchmarkVariance /= (returns.length - 1)
    
    return benchmarkVariance > 0 ? covariance / benchmarkVariance : 1
  }

  private static calculateAlpha(returns: number[], benchmarkReturns: number[], riskFreeRate: number): number {
    const beta = this.calculateBeta(returns, benchmarkReturns)
    const avgReturn = returns.reduce((a, b) => a + b) / returns.length
    const avgBenchmarkReturn = benchmarkReturns.reduce((a, b) => a + b) / benchmarkReturns.length
    const dailyRiskFreeRate = riskFreeRate / 252
    
    const alpha = avgReturn - (dailyRiskFreeRate + beta * (avgBenchmarkReturn - dailyRiskFreeRate))
    return alpha * 252 * 100 // Annualized alpha in percentage
  }

  private static calculateInformationRatio(returns: number[], benchmarkReturns: number[]): number {
    if (returns.length !== benchmarkReturns.length) return 0
    
    const activeReturns = returns.map((ret, i) => ret - benchmarkReturns[i])
    const meanActiveReturn = activeReturns.reduce((a, b) => a + b) / activeReturns.length
    const trackingError = this.calculateVolatility(activeReturns)
    
    return trackingError > 0 ? (meanActiveReturn / trackingError) * Math.sqrt(252) : 0
  }

  private static calculateTreynorRatio(returns: number[], benchmarkReturns: number[], riskFreeRate: number): number {
    const beta = this.calculateBeta(returns, benchmarkReturns)
    const avgReturn = returns.reduce((a, b) => a + b) / returns.length
    const dailyRiskFreeRate = riskFreeRate / 252
    const excessReturn = avgReturn - dailyRiskFreeRate
    
    return beta > 0 ? (excessReturn * 252) / beta : 0
  }

  private static calculateTrackingError(returns: number[], benchmarkReturns: number[]): number {
    if (returns.length !== benchmarkReturns.length) return 0
    
    const activeReturns = returns.map((ret, i) => ret - benchmarkReturns[i])
    return this.calculateVolatility(activeReturns) * Math.sqrt(252) * 100
  }

  private static calculateWinRate(trades: Trade[]): number {
    if (trades.length === 0) return 0
    const winningTrades = trades.filter(trade => trade.realizedPnL > 0).length
    return (winningTrades / trades.length) * 100
  }

  private static calculateProfitFactor(trades: Trade[]): number {
    const wins = trades.filter(trade => trade.realizedPnL > 0)
    const losses = trades.filter(trade => trade.realizedPnL < 0)
    
    const totalWins = wins.reduce((sum, trade) => sum + trade.realizedPnL, 0)
    const totalLosses = Math.abs(losses.reduce((sum, trade) => sum + trade.realizedPnL, 0))
    
    return totalLosses > 0 ? totalWins / totalLosses : Infinity
  }

  private static calculatePayoffRatio(trades: Trade[]): number {
    const wins = trades.filter(trade => trade.realizedPnL > 0)
    const losses = trades.filter(trade => trade.realizedPnL < 0)
    
    if (wins.length === 0 || losses.length === 0) return 0
    
    const avgWin = wins.reduce((sum, trade) => sum + trade.realizedPnL, 0) / wins.length
    const avgLoss = Math.abs(losses.reduce((sum, trade) => sum + trade.realizedPnL, 0) / losses.length)
    
    return avgLoss > 0 ? avgWin / avgLoss : 0
  }
}

// ============================================================================
// STRATEGY SIGNAL GENERATORS
// ============================================================================

interface StrategySignal {
  action: 'BUY' | 'SELL' | 'HOLD'
  confidence: number
  reason: string
  stopLoss?: number
  takeProfit?: number
  positionSize?: number
}

class StrategyLibrary {
  
  static generateSignal(
    strategyType: string, 
    marketData: MarketData[], 
    parameters: Record<string, any>,
    currentPrice: number
  ): StrategySignal {
    
    switch (strategyType) {
      case 'MEAN_REVERSION':
        return this.meanReversionStrategy(marketData, parameters, currentPrice)
      case 'MOMENTUM_BREAKOUT':
        return this.momentumBreakoutStrategy(marketData, parameters, currentPrice)
      case 'RSI_DIVERGENCE':
        return this.rsiDivergenceStrategy(marketData, parameters, currentPrice)
      case 'BOLLINGER_MEAN_REVERSION':
        return this.bollingerMeanReversionStrategy(marketData, parameters, currentPrice)
      case 'MACD_CROSSOVER':
        return this.macdCrossoverStrategy(marketData, parameters, currentPrice)
      case 'PATTERN_RECOGNITION':
        return this.patternRecognitionStrategy(marketData, parameters, currentPrice)
      default:
        return { action: 'HOLD', confidence: 0, reason: 'Unknown strategy' }
    }
  }

  private static meanReversionStrategy(data: MarketData[], params: any, currentPrice: number): StrategySignal {
    if (data.length < (params.lookback || 20)) {
      return { action: 'HOLD', confidence: 0, reason: 'Insufficient data' }
    }

    const prices = data.slice(-(params.lookback || 20)).map(d => d.close)
    const sma = prices.reduce((a, b) => a + b) / prices.length
    const std = Math.sqrt(prices.reduce((acc, p) => acc + Math.pow(p - sma, 2), 0) / prices.length)
    
    const zScore = (currentPrice - sma) / std
    const threshold = params.zScoreThreshold || 2.0

    if (Math.abs(zScore) > threshold) {
      const action = zScore > 0 ? 'SELL' : 'BUY'
      const confidence = Math.min(95, Math.abs(zScore) * 30)
      
      return {
        action,
        confidence,
        reason: `Mean reversion signal: Z-score = ${zScore.toFixed(2)}`,
        stopLoss: action === 'BUY' ? currentPrice * 0.98 : currentPrice * 1.02,
        takeProfit: action === 'BUY' ? sma : sma
      }
    }

    return { action: 'HOLD', confidence: 0, reason: 'No mean reversion signal' }
  }

  private static momentumBreakoutStrategy(data: MarketData[], params: any, currentPrice: number): StrategySignal {
    if (data.length < (params.lookback || 20)) {
      return { action: 'HOLD', confidence: 0, reason: 'Insufficient data' }
    }

    const recent = data.slice(-(params.lookback || 20))
    const highs = recent.map(d => d.high)
    const lows = recent.map(d => d.low)
    const volumes = recent.map(d => d.volume)
    
    const highestHigh = Math.max(...highs)
    const lowestLow = Math.min(...lows)
    const avgVolume = volumes.reduce((a, b) => a + b) / volumes.length
    const currentVolume = data[data.length - 1].volume

    // Breakout conditions
    const breakoutUp = currentPrice > highestHigh * (1 + (params.breakoutThreshold || 0.01))
    const breakoutDown = currentPrice < lowestLow * (1 - (params.breakoutThreshold || 0.01))
    const volumeConfirmation = currentVolume > avgVolume * (params.volumeMultiplier || 1.5)

    if ((breakoutUp || breakoutDown) && volumeConfirmation) {
      const action = breakoutUp ? 'BUY' : 'SELL'
      const confidence = Math.min(90, 60 + (currentVolume / avgVolume) * 10)
      
      return {
        action,
        confidence,
        reason: `Momentum breakout: Price ${breakoutUp ? 'above' : 'below'} range, Volume: ${(currentVolume/avgVolume).toFixed(1)}x`,
        stopLoss: action === 'BUY' ? lowestLow : highestHigh,
        takeProfit: action === 'BUY' ? currentPrice * 1.06 : currentPrice * 0.94
      }
    }

    return { action: 'HOLD', confidence: 0, reason: 'No momentum breakout' }
  }

  private static rsiDivergenceStrategy(data: MarketData[], params: any, currentPrice: number): StrategySignal {
    if (data.length < (params.rsiPeriod || 14) + 10) {
      return { action: 'HOLD', confidence: 0, reason: 'Insufficient data for RSI' }
    }

    const period = params.rsiPeriod || 14
    const rsiValues = this.calculateRSI(data, period)
    const prices = data.map(d => d.close)
    
    if (rsiValues.length < 10) {
      return { action: 'HOLD', confidence: 0, reason: 'Insufficient RSI data' }
    }

    const recentRSI = rsiValues.slice(-5)
    const recentPrices = prices.slice(-5)
    
    // Check for divergence
    const priceSlope = this.calculateSlope(recentPrices)
    const rsiSlope = this.calculateSlope(recentRSI)
    
    const currentRSI = recentRSI[recentRSI.length - 1]
    const divergenceStrength = Math.abs(priceSlope - rsiSlope)

    // Oversold/Overbought with divergence
    if (currentRSI < (params.oversoldLevel || 30) && priceSlope < 0 && rsiSlope > 0 && divergenceStrength > 0.5) {
      return {
        action: 'BUY',
        confidence: Math.min(85, 70 + divergenceStrength * 10),
        reason: `RSI bullish divergence: RSI=${currentRSI.toFixed(1)}, Price falling but RSI rising`,
        stopLoss: currentPrice * 0.97,
        takeProfit: currentPrice * 1.05
      }
    }

    if (currentRSI > (params.overboughtLevel || 70) && priceSlope > 0 && rsiSlope < 0 && divergenceStrength > 0.5) {
      return {
        action: 'SELL',
        confidence: Math.min(85, 70 + divergenceStrength * 10),
        reason: `RSI bearish divergence: RSI=${currentRSI.toFixed(1)}, Price rising but RSI falling`,
        stopLoss: currentPrice * 1.03,
        takeProfit: currentPrice * 0.95
      }
    }

    return { action: 'HOLD', confidence: 0, reason: 'No RSI divergence signal' }
  }

  private static bollingerMeanReversionStrategy(data: MarketData[], params: any, currentPrice: number): StrategySignal {
    if (data.length < (params.period || 20)) {
      return { action: 'HOLD', confidence: 0, reason: 'Insufficient data for Bollinger Bands' }
    }

    const period = params.period || 20
    const stdMultiplier = params.stdMultiplier || 2.0
    const prices = data.slice(-period).map(d => d.close)
    
    const sma = prices.reduce((a, b) => a + b) / prices.length
    const variance = prices.reduce((acc, p) => acc + Math.pow(p - sma, 2), 0) / prices.length
    const std = Math.sqrt(variance)
    
    const upperBand = sma + (std * stdMultiplier)
    const lowerBand = sma - (std * stdMultiplier)
    
    const bandPosition = (currentPrice - lowerBand) / (upperBand - lowerBand)
    
    // Mean reversion signals at band extremes
    if (bandPosition > 0.95) { // Near upper band
      return {
        action: 'SELL',
        confidence: Math.min(80, 60 + (bandPosition - 0.95) * 200),
        reason: `Price at upper Bollinger Band: ${(bandPosition * 100).toFixed(1)}% of band width`,
        stopLoss: upperBand * 1.01,
        takeProfit: sma
      }
    }

    if (bandPosition < 0.05) { // Near lower band
      return {
        action: 'BUY',
        confidence: Math.min(80, 60 + (0.05 - bandPosition) * 200),
        reason: `Price at lower Bollinger Band: ${(bandPosition * 100).toFixed(1)}% of band width`,
        stopLoss: lowerBand * 0.99,
        takeProfit: sma
      }
    }

    return { action: 'HOLD', confidence: 0, reason: 'Price within Bollinger Bands' }
  }

  private static macdCrossoverStrategy(data: MarketData[], params: any, currentPrice: number): StrategySignal {
    if (data.length < (params.slowPeriod || 26) + 10) {
      return { action: 'HOLD', confidence: 0, reason: 'Insufficient data for MACD' }
    }

    const fastPeriod = params.fastPeriod || 12
    const slowPeriod = params.slowPeriod || 26
    const signalPeriod = params.signalPeriod || 9
    
    const macd = this.calculateMACD(data, fastPeriod, slowPeriod, signalPeriod)
    
    if (macd.length < 2) {
      return { action: 'HOLD', confidence: 0, reason: 'Insufficient MACD data' }
    }

    const current = macd[macd.length - 1]
    const previous = macd[macd.length - 2]
    
    // MACD line crossing signal line
    const bullishCross = previous.macdLine <= previous.signalLine && current.macdLine > current.signalLine
    const bearishCross = previous.macdLine >= previous.signalLine && current.macdLine < current.signalLine
    
    // Histogram momentum
    const histogramMomentum = Math.abs(current.histogram) > Math.abs(previous.histogram)
    
    if (bullishCross && current.macdLine < 0 && histogramMomentum) {
      return {
        action: 'BUY',
        confidence: Math.min(75, 50 + Math.abs(current.histogram) * 100),
        reason: `MACD bullish crossover: MACD=${current.macdLine.toFixed(4)}, Signal=${current.signalLine.toFixed(4)}`,
        stopLoss: currentPrice * 0.97,
        takeProfit: currentPrice * 1.04
      }
    }

    if (bearishCross && current.macdLine > 0 && histogramMomentum) {
      return {
        action: 'SELL',
        confidence: Math.min(75, 50 + Math.abs(current.histogram) * 100),
        reason: `MACD bearish crossover: MACD=${current.macdLine.toFixed(4)}, Signal=${current.signalLine.toFixed(4)}`,
        stopLoss: currentPrice * 1.03,
        takeProfit: currentPrice * 0.96
      }
    }

    return { action: 'HOLD', confidence: 0, reason: 'No MACD crossover signal' }
  }

  private static patternRecognitionStrategy(data: MarketData[], params: any, currentPrice: number): StrategySignal {
    if (data.length < 5) {
      return { action: 'HOLD', confidence: 0, reason: 'Insufficient data for pattern recognition' }
    }

    const recent = data.slice(-5)
    
    // Doji pattern detection
    const isDoji = this.detectDoji(recent[recent.length - 1])
    if (isDoji.detected) {
      return {
        action: 'HOLD', // Doji suggests indecision
        confidence: isDoji.strength,
        reason: `Doji pattern detected: ${isDoji.type}`,
      }
    }

    // Hammer/Shooting star detection
    const candlestickPattern = this.detectCandlestickPatterns(recent)
    if (candlestickPattern.detected) {
      const action = candlestickPattern.signal === 'bullish' ? 'BUY' : 'SELL'
      return {
        action,
        confidence: candlestickPattern.strength,
        reason: `${candlestickPattern.pattern} pattern detected`,
        stopLoss: action === 'BUY' ? currentPrice * 0.98 : currentPrice * 1.02,
        takeProfit: action === 'BUY' ? currentPrice * 1.04 : currentPrice * 0.96
      }
    }

    return { action: 'HOLD', confidence: 0, reason: 'No patterns detected' }
  }

  // Helper methods for technical indicators
  private static calculateRSI(data: MarketData[], period: number): number[] {
    if (data.length < period + 1) return []
    
    const rsi: number[] = []
    let avgGain = 0
    let avgLoss = 0
    
    // Initial calculation
    for (let i = 1; i <= period; i++) {
      const change = data[i].close - data[i-1].close
      if (change > 0) avgGain += change
      else avgLoss += Math.abs(change)
    }
    
    avgGain /= period
    avgLoss /= period
    
    let rs = avgGain / avgLoss
    rsi.push(100 - (100 / (1 + rs)))
    
    // Subsequent calculations
    for (let i = period + 1; i < data.length; i++) {
      const change = data[i].close - data[i-1].close
      const gain = change > 0 ? change : 0
      const loss = change < 0 ? Math.abs(change) : 0
      
      avgGain = ((avgGain * (period - 1)) + gain) / period
      avgLoss = ((avgLoss * (period - 1)) + loss) / period
      
      rs = avgGain / avgLoss
      rsi.push(100 - (100 / (1 + rs)))
    }
    
    return rsi
  }

  private static calculateMACD(data: MarketData[], fastPeriod: number, slowPeriod: number, signalPeriod: number) {
    if (data.length < slowPeriod + signalPeriod) return []
    
    const prices = data.map(d => d.close)
    const emaFast = this.calculateEMA(prices, fastPeriod)
    const emaSlow = this.calculateEMA(prices, slowPeriod)
    
    const macdLine: number[] = []
    for (let i = slowPeriod - 1; i < emaFast.length; i++) {
      macdLine.push(emaFast[i] - emaSlow[i - slowPeriod + 1])
    }
    
    const signalLine = this.calculateEMA(macdLine, signalPeriod)
    
    const result = []
    for (let i = signalPeriod - 1; i < macdLine.length; i++) {
      result.push({
        macdLine: macdLine[i],
        signalLine: signalLine[i - signalPeriod + 1],
        histogram: macdLine[i] - signalLine[i - signalPeriod + 1]
      })
    }
    
    return result
  }

  private static calculateEMA(prices: number[], period: number): number[] {
    if (prices.length < period) return []
    
    const k = 2 / (period + 1)
    const ema: number[] = []
    
    // First EMA is SMA
    let sum = 0
    for (let i = 0; i < period; i++) {
      sum += prices[i]
    }
    ema.push(sum / period)
    
    // Subsequent EMAs
    for (let i = period; i < prices.length; i++) {
      ema.push(prices[i] * k + ema[ema.length - 1] * (1 - k))
    }
    
    return ema
  }

  private static calculateSlope(values: number[]): number {
    if (values.length < 2) return 0
    
    const n = values.length
    const sumX = (n * (n - 1)) / 2  // Sum of indices
    const sumY = values.reduce((a, b) => a + b)
    const sumXY = values.reduce((sum, val, idx) => sum + val * idx, 0)
    const sumX2 = (n * (n - 1) * (2 * n - 1)) / 6  // Sum of squared indices
    
    return (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX)
  }

  private static detectDoji(candle: MarketData): {detected: boolean, strength: number, type: string} {
    const bodySize = Math.abs(candle.close - candle.open)
    const totalRange = candle.high - candle.low
    const bodyRatio = bodySize / totalRange
    
    if (bodyRatio < 0.1) {
      return {
        detected: true,
        strength: Math.round(70 + (0.1 - bodyRatio) * 200),
        type: 'Standard Doji'
      }
    }
    
    return { detected: false, strength: 0, type: '' }
  }

  private static detectCandlestickPatterns(candles: MarketData[]): {detected: boolean, pattern: string, signal: string, strength: number} {
    if (candles.length < 2) return { detected: false, pattern: '', signal: '', strength: 0 }
    
    const current = candles[candles.length - 1]
    const bodySize = Math.abs(current.close - current.open)
    const upperShadow = current.high - Math.max(current.open, current.close)
    const lowerShadow = Math.min(current.open, current.close) - current.low
    const totalRange = current.high - current.low
    
    // Hammer detection
    if (lowerShadow > bodySize * 2 && upperShadow < bodySize && current.close > current.open) {
      return {
        detected: true,
        pattern: 'Hammer',
        signal: 'bullish',
        strength: Math.round(60 + (lowerShadow / totalRange) * 30)
      }
    }
    
    // Shooting Star detection
    if (upperShadow > bodySize * 2 && lowerShadow < bodySize && current.close < current.open) {
      return {
        detected: true,
        pattern: 'Shooting Star',
        signal: 'bearish',
        strength: Math.round(60 + (upperShadow / totalRange) * 30)
      }
    }
    
    return { detected: false, pattern: '', signal: '', strength: 0 }
  }
}

export { PerformanceAnalytics, StrategyLibrary, type StrategySignal }