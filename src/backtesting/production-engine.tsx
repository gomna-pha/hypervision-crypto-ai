/**
 * GOMNA Trading - Production Backtesting Engine
 * 
 * Complete backtesting system with academic rigor and industry standards
 * Implements proper bias elimination, risk management, and performance analytics
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
import { PerformanceAnalytics, StrategyLibrary, type StrategySignal } from './engine'

// ============================================================================
// PRODUCTION BACKTESTING ENGINE
// ============================================================================

export class ProductionBacktestingEngine {
  private marketDataGenerator: EnhancedMarketDataGenerator
  private transactionCostModel: TransactionCostModel
  private riskManager: RiskManagementEngine
  private runningBacktests: Map<string, any>
  private completedBacktests: Map<string, BacktestResult>

  constructor() {
    this.runningBacktests = new Map()
    this.completedBacktests = new Map()
  }

  /**
   * Run comprehensive backtest with academic rigor
   */
  async runBacktest(config: BacktestConfig): Promise<BacktestResult> {
    // Validate configuration
    this.validateConfig(config)
    
    // Initialize components with configuration
    this.initializeComponents(config)
    
    // Mark backtest as running
    this.runningBacktests.set(config.strategyId, {
      ...config,
      status: 'RUNNING',
      startTime: Date.now(),
      progress: 0
    })

    try {
      // Generate historical data for all symbols
      const historicalData = await this.generateHistoricalData(config)
      
      // Generate benchmark data
      const benchmarkData = await this.generateBenchmarkData(config)
      
      // Run the actual backtest simulation
      const result = await this.runBacktestSimulation(config, historicalData, benchmarkData)
      
      // Store completed backtest
      this.completedBacktests.set(config.strategyId, result)
      this.runningBacktests.delete(config.strategyId)
      
      return result
      
    } catch (error) {
      // Mark backtest as failed
      this.runningBacktests.set(config.strategyId, {
        ...config,
        status: 'FAILED',
        error: error.message
      })
      throw error
    }
  }

  /**
   * Run walk-forward optimization
   */
  async runWalkForwardOptimization(
    config: BacktestConfig,
    parameterRanges: Record<string, any[]>,
    walkForwardSteps: number = 6
  ): Promise<{optimizedParams: any, results: BacktestResult[]}> {
    
    const startDate = new Date(config.startDate)
    const endDate = new Date(config.endDate)
    const totalDays = (endDate.getTime() - startDate.getTime()) / (24 * 60 * 60 * 1000)
    const stepDays = Math.floor(totalDays / walkForwardSteps)
    
    const results: BacktestResult[] = []
    let bestParams = config.strategyParameters
    let bestSharpe = -Infinity
    
    for (let step = 0; step < walkForwardSteps; step++) {
      const stepStartDate = new Date(startDate.getTime() + step * stepDays * 24 * 60 * 60 * 1000)
      const stepEndDate = new Date(stepStartDate.getTime() + stepDays * 24 * 60 * 60 * 1000)
      
      // Optimization phase (use 80% of step data)
      const optimizationEndDate = new Date(stepStartDate.getTime() + stepDays * 0.8 * 24 * 60 * 60 * 1000)
      
      console.log(`Walk-forward step ${step + 1}/${walkForwardSteps}: Optimizing ${stepStartDate.toDateString()} to ${optimizationEndDate.toDateString()}`)
      
      // Find optimal parameters for this period
      const optimalParams = await this.optimizeParameters(
        {
          ...config,
          startDate: stepStartDate.toISOString(),
          endDate: optimizationEndDate.toISOString()
        },
        parameterRanges
      )
      
      // Test on out-of-sample data (remaining 20%)
      const testStartDate = optimizationEndDate
      const testResult = await this.runBacktest({
        ...config,
        strategyId: `${config.strategyId}_wf_${step}`,
        startDate: testStartDate.toISOString(),
        endDate: stepEndDate.toISOString(),
        strategyParameters: optimalParams
      })
      
      results.push(testResult)
      
      // Track best parameters
      if (testResult.performance.sharpeRatio > bestSharpe) {
        bestSharpe = testResult.performance.sharpeRatio
        bestParams = optimalParams
      }
    }
    
    return { optimizedParams: bestParams, results }
  }

  /**
   * Run Monte Carlo simulation for robustness testing
   */
  async runMonteCarloSimulation(
    config: BacktestConfig,
    iterations: number = 1000,
    perturbationLevel: number = 0.1
  ): Promise<{
    meanResult: BacktestResult,
    confidenceIntervals: Record<string, {lower: number, upper: number}>,
    worstCase: BacktestResult,
    bestCase: BacktestResult
  }> {
    
    const results: BacktestResult[] = []
    
    for (let i = 0; i < iterations; i++) {
      // Perturb strategy parameters
      const perturbedParams = this.perturbParameters(config.strategyParameters, perturbationLevel)
      
      // Run backtest with perturbed parameters
      const result = await this.runBacktest({
        ...config,
        strategyId: `${config.strategyId}_mc_${i}`,
        strategyParameters: perturbedParams
      })
      
      results.push(result)
      
      if (i % 100 === 0) {
        console.log(`Monte Carlo progress: ${i}/${iterations}`)
      }
    }
    
    // Calculate statistics
    const returns = results.map(r => r.performance.totalReturn)
    const sharpeRatios = results.map(r => r.performance.sharpeRatio)
    const maxDrawdowns = results.map(r => r.performance.maxDrawdown)
    
    returns.sort((a, b) => a - b)
    sharpeRatios.sort((a, b) => a - b)
    maxDrawdowns.sort((a, b) => a - b)
    
    const meanResult = results[Math.floor(results.length / 2)]
    const worstCase = results.reduce((worst, current) => 
      current.performance.sharpeRatio < worst.performance.sharpeRatio ? current : worst
    )
    const bestCase = results.reduce((best, current) => 
      current.performance.sharpeRatio > best.performance.sharpeRatio ? current : best
    )
    
    return {
      meanResult,
      worstCase,
      bestCase,
      confidenceIntervals: {
        totalReturn: {
          lower: returns[Math.floor(returns.length * 0.05)],
          upper: returns[Math.floor(returns.length * 0.95)]
        },
        sharpeRatio: {
          lower: sharpeRatios[Math.floor(sharpeRatios.length * 0.05)],
          upper: sharpeRatios[Math.floor(sharpeRatios.length * 0.95)]
        },
        maxDrawdown: {
          lower: maxDrawdowns[Math.floor(maxDrawdowns.length * 0.05)],
          upper: maxDrawdowns[Math.floor(maxDrawdowns.length * 0.95)]
        }
      }
    }
  }

  /**
   * Get backtest status
   */
  getBacktestStatus(strategyId: string): any {
    if (this.runningBacktests.has(strategyId)) {
      return this.runningBacktests.get(strategyId)
    }
    if (this.completedBacktests.has(strategyId)) {
      return { status: 'COMPLETED', result: this.completedBacktests.get(strategyId) }
    }
    return { status: 'NOT_FOUND' }
  }

  /**
   * Get all completed backtests
   */
  getAllBacktests(): BacktestResult[] {
    return Array.from(this.completedBacktests.values())
  }

  /**
   * Compare multiple strategies with statistical significance testing
   */
  compareStrategies(strategyIds: string[]): {
    comparison: Record<string, any>,
    rankings: Array<{strategyId: string, score: number, rank: number}>,
    statisticalTests: Record<string, any>,
    performanceMatrix: Array<Array<number>>,
    riskAdjustedRankings: Array<{strategyId: string, riskAdjustedScore: number}>
  } {
    const results = strategyIds
      .map(id => this.completedBacktests.get(id))
      .filter(result => result !== undefined) as BacktestResult[]

    if (results.length === 0) {
      throw new Error('No completed backtests found for comparison')
    }

    if (results.length < 2) {
      throw new Error('At least 2 strategies required for comparison')
    }

    const comparison: Record<string, any> = {}
    const metrics = ['totalReturn', 'sharpeRatio', 'maxDrawdown', 'winRate', 'profitFactor', 'sortinoRatio', 'calmarRatio', 'var95', 'cvar95']
    
    metrics.forEach(metric => {
      comparison[metric] = results.map(result => ({
        strategyId: result.config.strategyId,
        value: result.performance[metric as keyof RiskMetrics]
      }))
    })

    // Statistical significance testing
    const statisticalTests = this.performStatisticalTests(results)
    
    // Performance matrix for heat map visualization
    const performanceMatrix = this.createPerformanceMatrix(results)
    
    // Enhanced composite scoring with risk adjustment
    const rankings = results.map((result, index) => {
      const perf = result.performance
      
      // Multi-factor scoring model
      const returnScore = Math.min(perf.totalReturn / 50, 2) // Cap at 50% return = score 2
      const riskScore = Math.max(0, 2 - perf.maxDrawdown / 20) // Lower drawdown = higher score
      const consistencyScore = Math.min(perf.winRate / 50, 2) // Cap at 50% win rate = score 2
      const efficiencyScore = Math.min(perf.sharpeRatio / 1.5, 2) // Cap at 1.5 Sharpe = score 2
      const robustnessScore = Math.min(perf.sortinoRatio / 2, 2) // Cap at 2 Sortino = score 2
      
      const compositeScore = (
        returnScore * 0.25 +
        riskScore * 0.25 +
        consistencyScore * 0.2 +
        efficiencyScore * 0.2 +
        robustnessScore * 0.1
      )

      return {
        strategyId: result.config.strategyId,
        score: Math.round(compositeScore * 100) / 100,
        rank: index + 1
      }
    }).sort((a, b) => b.score - a.score)
    
    // Assign final ranks
    rankings.forEach((ranking, index) => {
      ranking.rank = index + 1
    })

    // Risk-adjusted rankings using multiple criteria
    const riskAdjustedRankings = results.map(result => {
      const perf = result.performance
      
      // Risk-adjusted score using Information Ratio, Calmar Ratio, and Omega Ratio
      const riskAdjustedScore = (
        perf.sharpeRatio * 0.3 +
        perf.sortinoRatio * 0.25 +
        perf.calmarRatio * 0.2 +
        (perf.informationRatio || 0) * 0.15 +
        Math.min(perf.profitFactor / 2, 2) * 0.1 // Cap profit factor impact
      )
      
      return {
        strategyId: result.config.strategyId,
        riskAdjustedScore: Math.round(riskAdjustedScore * 100) / 100
      }
    }).sort((a, b) => b.riskAdjustedScore - a.riskAdjustedScore)

    return { 
      comparison, 
      rankings, 
      statisticalTests,
      performanceMatrix,
      riskAdjustedRankings
    }
  }

  /**
   * Perform statistical significance tests between strategies
   */
  private performStatisticalTests(results: BacktestResult[]): Record<string, any> {
    const tests: Record<string, any> = {
      tTests: [],
      correlationMatrix: [],
      significanceMatrix: []
    }

    // Extract daily returns for each strategy
    const strategyReturns: Record<string, number[]> = {}
    
    results.forEach(result => {
      const dailyReturns: number[] = []
      for (let i = 1; i < result.equity.length; i++) {
        const ret = (result.equity[i].equity - result.equity[i-1].equity) / result.equity[i-1].equity
        dailyReturns.push(ret)
      }
      strategyReturns[result.config.strategyId] = dailyReturns
    })

    // Pairwise t-tests for statistical significance
    for (let i = 0; i < results.length; i++) {
      for (let j = i + 1; j < results.length; j++) {
        const strategy1 = results[i].config.strategyId
        const strategy2 = results[j].config.strategyId
        
        const returns1 = strategyReturns[strategy1]
        const returns2 = strategyReturns[strategy2]
        
        const tTest = this.performTTest(returns1, returns2)
        
        tests.tTests.push({
          strategy1,
          strategy2,
          tStatistic: tTest.tStatistic,
          pValue: tTest.pValue,
          significant: tTest.pValue < 0.05,
          confidenceLevel: tTest.pValue < 0.01 ? '99%' : tTest.pValue < 0.05 ? '95%' : 'Not Significant'
        })
      }
    }

    // Correlation matrix
    const strategyIds = results.map(r => r.config.strategyId)
    tests.correlationMatrix = this.calculateCorrelationMatrix(strategyReturns, strategyIds)
    
    // Significance matrix (p-values)
    tests.significanceMatrix = this.createSignificanceMatrix(tests.tTests, strategyIds)

    return tests
  }

  /**
   * Perform Welch's t-test for unequal variances
   */
  private performTTest(sample1: number[], sample2: number[]): {tStatistic: number, pValue: number} {
    const n1 = sample1.length
    const n2 = sample2.length
    
    const mean1 = sample1.reduce((a, b) => a + b) / n1
    const mean2 = sample2.reduce((a, b) => a + b) / n2
    
    const var1 = sample1.reduce((acc, x) => acc + Math.pow(x - mean1, 2), 0) / (n1 - 1)
    const var2 = sample2.reduce((acc, x) => acc + Math.pow(x - mean2, 2), 0) / (n2 - 1)
    
    const pooledStdError = Math.sqrt(var1/n1 + var2/n2)
    const tStatistic = (mean1 - mean2) / pooledStdError
    
    // Degrees of freedom (Welch's formula)
    const df = Math.pow(var1/n1 + var2/n2, 2) / (Math.pow(var1/n1, 2)/(n1-1) + Math.pow(var2/n2, 2)/(n2-1))
    
    // Approximate p-value using t-distribution approximation
    const pValue = this.calculateTTestPValue(Math.abs(tStatistic), df)
    
    return { tStatistic, pValue }
  }

  /**
   * Approximate p-value calculation for t-test
   */
  private calculateTTestPValue(tStat: number, df: number): number {
    // Simplified approximation - in production, use proper t-distribution
    if (tStat < 1.96) return 0.05 + (1.96 - tStat) * 0.45 / 1.96
    if (tStat < 2.58) return 0.01 + (2.58 - tStat) * 0.04 / 0.62
    return 0.01 * Math.exp(-(tStat - 2.58))
  }

  /**
   * Calculate correlation matrix between strategies
   */
  private calculateCorrelationMatrix(strategyReturns: Record<string, number[]>, strategyIds: string[]): number[][] {
    const n = strategyIds.length
    const matrix: number[][] = Array(n).fill(null).map(() => Array(n).fill(0))
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) {
          matrix[i][j] = 1.0
        } else {
          const returns1 = strategyReturns[strategyIds[i]]
          const returns2 = strategyReturns[strategyIds[j]]
          matrix[i][j] = this.calculateCorrelation(returns1, returns2)
        }
      }
    }
    
    return matrix
  }

  /**
   * Calculate Pearson correlation coefficient
   */
  private calculateCorrelation(x: number[], y: number[]): number {
    const n = Math.min(x.length, y.length)
    if (n < 2) return 0
    
    const meanX = x.slice(0, n).reduce((a, b) => a + b) / n
    const meanY = y.slice(0, n).reduce((a, b) => a + b) / n
    
    let numerator = 0
    let sumXSquared = 0
    let sumYSquared = 0
    
    for (let i = 0; i < n; i++) {
      const deltaX = x[i] - meanX
      const deltaY = y[i] - meanY
      numerator += deltaX * deltaY
      sumXSquared += deltaX * deltaX
      sumYSquared += deltaY * deltaY
    }
    
    const denominator = Math.sqrt(sumXSquared * sumYSquared)
    return denominator === 0 ? 0 : numerator / denominator
  }

  /**
   * Create performance matrix for visualization
   */
  private createPerformanceMatrix(results: BacktestResult[]): number[][] {
    const metrics = ['totalReturn', 'sharpeRatio', 'maxDrawdown', 'winRate', 'sortinoRatio']
    const matrix: number[][] = []
    
    results.forEach(result => {
      const row: number[] = []
      metrics.forEach(metric => {
        let value = result.performance[metric as keyof RiskMetrics] as number
        
        // Normalize metrics to 0-1 scale for visualization
        switch (metric) {
          case 'totalReturn':
            value = Math.max(0, Math.min(1, (value + 50) / 100)) // -50% to +50% mapped to 0-1
            break
          case 'sharpeRatio':
            value = Math.max(0, Math.min(1, (value + 2) / 4)) // -2 to +2 mapped to 0-1
            break
          case 'maxDrawdown':
            value = Math.max(0, Math.min(1, 1 - value / 50)) // 0% to 50% drawdown mapped to 1-0
            break
          case 'winRate':
            value = Math.max(0, Math.min(1, value / 100)) // 0% to 100% mapped to 0-1
            break
          case 'sortinoRatio':
            value = Math.max(0, Math.min(1, (value + 2) / 4)) // -2 to +2 mapped to 0-1
            break
        }
        row.push(value)
      })
      matrix.push(row)
    })
    
    return matrix
  }

  /**
   * Create significance matrix from t-test results
   */
  private createSignificanceMatrix(tTests: any[], strategyIds: string[]): number[][] {
    const n = strategyIds.length
    const matrix: number[][] = Array(n).fill(null).map(() => Array(n).fill(1.0))
    
    // Fill diagonal with 0 (perfect significance with self)
    for (let i = 0; i < n; i++) {
      matrix[i][i] = 0.0
    }
    
    tTests.forEach(test => {
      const i = strategyIds.indexOf(test.strategy1)
      const j = strategyIds.indexOf(test.strategy2)
      
      if (i !== -1 && j !== -1) {
        matrix[i][j] = test.pValue
        matrix[j][i] = test.pValue
      }
    })
    
    return matrix
  }

  // ============================================================================
  // PRIVATE METHODS
  // ============================================================================

  private validateConfig(config: BacktestConfig): void {
    if (!config.strategyId) throw new Error('Strategy ID is required')
    if (!config.symbols || config.symbols.length === 0) throw new Error('At least one symbol is required')
    if (!config.startDate || !config.endDate) throw new Error('Start and end dates are required')
    if (new Date(config.startDate) >= new Date(config.endDate)) throw new Error('End date must be after start date')
    if (config.initialCapital <= 0) throw new Error('Initial capital must be positive')
    
    // Validate risk management parameters
    const rm = config.riskManagement
    if (rm.maxPositionSize <= 0 || rm.maxPositionSize > 1) {
      throw new Error('Max position size must be between 0 and 1')
    }
    if (rm.maxPortfolioRisk <= 0 || rm.maxPortfolioRisk > 1) {
      throw new Error('Max portfolio risk must be between 0 and 1')
    }
  }

  private initializeComponents(config: BacktestConfig): void {
    this.marketDataGenerator = new EnhancedMarketDataGenerator(config.symbols)
    this.transactionCostModel = new TransactionCostModel(config.transactionCosts)
    this.riskManager = new RiskManagementEngine(config.riskManagement)
  }

  private async generateHistoricalData(config: BacktestConfig): Promise<Record<string, MarketData[]>> {
    const data: Record<string, MarketData[]> = {}
    const startDate = new Date(config.startDate)
    const endDate = new Date(config.endDate)
    
    for (const symbol of config.symbols) {
      data[symbol] = this.marketDataGenerator.generateHistoricalData(
        symbol, 
        startDate, 
        endDate, 
        config.dataSettings.timeframe as any
      )
    }
    
    return data
  }

  private async generateBenchmarkData(config: BacktestConfig): Promise<MarketData[]> {
    const startDate = new Date(config.startDate)
    const endDate = new Date(config.endDate)
    
    return this.marketDataGenerator.generateHistoricalData(
      config.benchmark, 
      startDate, 
      endDate, 
      config.dataSettings.timeframe as any
    )
  }

  private async runBacktestSimulation(
    config: BacktestConfig, 
    historicalData: Record<string, MarketData[]>, 
    benchmarkData: MarketData[]
  ): Promise<BacktestResult> {
    
    const startTime = Date.now()
    let currentCapital = config.initialCapital
    const positions: Map<string, Position> = new Map()
    const trades: Trade[] = []
    const equityCurve: Array<{timestamp: number, equity: number, drawdown: number}> = []
    const positionHistory: Array<{timestamp: number, positions: Position[]}> = []
    
    // Get the longest data series to determine simulation length
    const maxLength = Math.max(...Object.values(historicalData).map(data => data.length))
    let maxEquity = currentCapital
    
    // Initialize risk manager
    this.riskManager.updateEquity(currentCapital)
    
    // Main simulation loop - CRITICAL: Prevents look-ahead bias
    for (let i = 50; i < maxLength; i++) { // Start at 50 to have sufficient history
      
      // Update progress for running backtests
      if (this.runningBacktests.has(config.strategyId)) {
        const progress = Math.round((i / maxLength) * 100)
        this.runningBacktests.get(config.strategyId)!.progress = progress
      }
      
      // Get current market state for all symbols (point-in-time)
      const currentMarketState: Record<string, MarketData> = {}
      let currentTimestamp = 0
      
      for (const symbol of config.symbols) {
        if (i < historicalData[symbol].length) {
          currentMarketState[symbol] = historicalData[symbol][i]
          currentTimestamp = historicalData[symbol][i].timestamp
        }
      }
      
      // Skip if we don't have data for all symbols at this timestamp
      if (Object.keys(currentMarketState).length !== config.symbols.length) {
        continue
      }
      
      // Update position mark-to-market values
      let totalPositionValue = 0
      positions.forEach((position, symbol) => {
        if (currentMarketState[symbol]) {
          const currentPrice = currentMarketState[symbol].close
          position.marketValue = position.quantity * currentPrice
          position.unrealizedPnL = (currentPrice - position.avgPrice) * position.quantity
          totalPositionValue += Math.abs(position.marketValue)
        }
      })
      
      // Update current equity
      currentCapital = config.initialCapital + Array.from(positions.values())
        .reduce((sum, pos) => sum + pos.realizedPnL + pos.unrealizedPnL, 0)
      
      // Update risk manager
      this.riskManager.updateEquity(currentCapital)
      maxEquity = Math.max(maxEquity, currentCapital)
      
      // Calculate current drawdown
      const currentDrawdown = ((maxEquity - currentCapital) / maxEquity) * 100
      
      // Record equity curve
      equityCurve.push({
        timestamp: currentTimestamp,
        equity: currentCapital,
        drawdown: currentDrawdown
      })
      
      // Record position snapshot
      positionHistory.push({
        timestamp: currentTimestamp,
        positions: Array.from(positions.values()).map(pos => ({...pos}))
      })
      
      // Generate signals for each symbol
      for (const symbol of config.symbols) {
        const symbolData = historicalData[symbol].slice(0, i + 1) // Only use past data
        const currentPrice = currentMarketState[symbol].close
        
        // Generate strategy signal
        const signal = StrategyLibrary.generateSignal(
          config.name, // Use strategy name as type
          symbolData,
          config.strategyParameters,
          currentPrice
        )
        
        // Execute trades based on signals
        if (signal.action !== 'HOLD') {
          await this.executeSignal(
            symbol,
            signal,
            currentMarketState[symbol],
            positions,
            trades,
            config
          )
        }
      }
      
      // Update risk manager positions
      positions.forEach((position, symbol) => {
        this.riskManager.updatePosition(symbol, position)
      })
    }
    
    const executionTime = Date.now() - startTime
    
    // Calculate performance metrics
    const benchmarkReturns = this.calculateBenchmarkReturns(benchmarkData)
    const performance = PerformanceAnalytics.calculateRiskMetrics(
      equityCurve,
      trades,
      benchmarkReturns,
      config.riskFreeRate,
      config.initialCapital
    )
    
    // Calculate attribution
    const attribution = this.calculateAttribution(trades, config.symbols)
    
    // Calculate statistics
    const statistics = this.calculateStatistics(equityCurve, trades)
    
    const result: BacktestResult = {
      config,
      summary: {
        totalTrades: trades.length,
        winningTrades: trades.filter(t => t.realizedPnL > 0).length,
        losingTrades: trades.filter(t => t.realizedPnL < 0).length,
        totalCommissions: trades.reduce((sum, t) => sum + t.commission, 0),
        totalSlippage: trades.reduce((sum, t) => sum + Math.abs(t.slippage), 0),
        executionTime
      },
      performance,
      trades,
      equity: equityCurve,
      positions: positionHistory,
      attribution,
      statistics
    }
    
    return result
  }

  private async executeSignal(
    symbol: string,
    signal: StrategySignal,
    marketData: MarketData,
    positions: Map<string, Position>,
    trades: Trade[],
    config: BacktestConfig
  ): Promise<void> {
    
    const currentPrice = marketData.close
    const currentPosition = positions.get(symbol)
    
    // Determine trade quantity
    let quantity = 0
    
    if (signal.action === 'BUY') {
      if (!currentPosition || currentPosition.quantity <= 0) {
        // New long position
        const riskAmount = this.riskManager.updateEquity(config.initialCapital) * config.riskManagement.riskPerTrade
        quantity = Math.floor(riskAmount / currentPrice)
      }
    } else if (signal.action === 'SELL') {
      if (currentPosition && currentPosition.quantity > 0) {
        // Close long position
        quantity = -currentPosition.quantity
      } else if (!currentPosition || currentPosition.quantity === 0) {
        // New short position
        const riskAmount = this.riskManager.updateEquity(config.initialCapital) * config.riskManagement.riskPerTrade
        quantity = -Math.floor(riskAmount / currentPrice)
      }
    }
    
    if (quantity === 0) return
    
    // Risk management validation
    const riskValidation = this.riskManager.validateTradeRisk(
      symbol,
      quantity > 0 ? 'BUY' : 'SELL',
      Math.abs(quantity),
      currentPrice
    )
    
    if (!riskValidation.allowed) {
      console.warn(`Trade rejected for ${symbol}: ${riskValidation.reason}`)
      return
    }
    
    // Adjust quantity based on risk management
    if (riskValidation.adjustedQuantity !== Math.abs(quantity)) {
      quantity = quantity > 0 ? riskValidation.adjustedQuantity : -riskValidation.adjustedQuantity
    }
    
    // Calculate transaction costs
    const costs = this.transactionCostModel.calculateTransactionCosts(
      symbol,
      quantity > 0 ? 'BUY' : 'SELL',
      Math.abs(quantity),
      currentPrice,
      marketData.volume,
      marketData.spread || 0.01
    )
    
    // Execute the trade
    const trade: Trade = {
      tradeId: `${symbol}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      symbol,
      side: quantity > 0 ? 'BUY' : 'SELL',
      quantity: Math.abs(quantity),
      price: currentPrice + (quantity > 0 ? costs.slippage : -costs.slippage),
      executionTime: marketData.timestamp,
      commission: costs.commission,
      slippage: costs.slippage,
      marketImpact: costs.marketImpact,
      totalCost: costs.totalCost,
      realizedPnL: 0,
      strategyId: config.strategyId
    }
    
    // Update position
    if (!currentPosition) {
      positions.set(symbol, {
        symbol,
        quantity: quantity,
        avgPrice: trade.price,
        entryTime: marketData.timestamp,
        unrealizedPnL: 0,
        realizedPnL: 0,
        marketValue: quantity * trade.price
      })
    } else {
      if (Math.sign(quantity) === Math.sign(currentPosition.quantity)) {
        // Adding to position
        const newQuantity = currentPosition.quantity + quantity
        const newAvgPrice = ((currentPosition.avgPrice * currentPosition.quantity) + (trade.price * quantity)) / newQuantity
        currentPosition.quantity = newQuantity
        currentPosition.avgPrice = newAvgPrice
      } else {
        // Reducing or closing position
        const tradedQuantity = Math.min(Math.abs(quantity), Math.abs(currentPosition.quantity))
        const realizedPnL = tradedQuantity * (trade.price - currentPosition.avgPrice) * Math.sign(currentPosition.quantity)
        
        trade.realizedPnL = realizedPnL - costs.totalCost
        currentPosition.realizedPnL += trade.realizedPnL
        currentPosition.quantity += quantity
        
        if (Math.abs(currentPosition.quantity) < 0.0001) {
          positions.delete(symbol)
        }
      }
    }
    
    trades.push(trade)
  }

  private calculateBenchmarkReturns(benchmarkData: MarketData[]): number[] {
    const returns: number[] = []
    for (let i = 1; i < benchmarkData.length; i++) {
      const ret = (benchmarkData[i].close - benchmarkData[i-1].close) / benchmarkData[i-1].close
      returns.push(ret)
    }
    return returns
  }

  private calculateAttribution(trades: Trade[], symbols: string[]): any {
    const byAsset: Record<string, number> = {}
    const byTimeframe: Record<string, number> = {}
    const byStrategy: Record<string, number> = {}
    
    // Initialize
    symbols.forEach(symbol => { byAsset[symbol] = 0 })
    
    // Calculate attribution
    trades.forEach(trade => {
      byAsset[trade.symbol] = (byAsset[trade.symbol] || 0) + trade.realizedPnL
      
      const month = new Date(trade.executionTime).toISOString().substring(0, 7)
      byTimeframe[month] = (byTimeframe[month] || 0) + trade.realizedPnL
      
      byStrategy[trade.strategyId] = (byStrategy[trade.strategyId] || 0) + trade.realizedPnL
    })
    
    return { byAsset, byTimeframe, byStrategy }
  }

  private calculateStatistics(
    equity: Array<{timestamp: number, equity: number}>, 
    trades: Trade[]
  ): any {
    const monthlyReturns: Array<{month: string, return: number}> = []
    const yearlyReturns: Array<{year: number, return: number}> = []
    const rollingMetrics: Array<{date: string, sharpe: number, volatility: number}> = []
    
    // Calculate monthly returns
    const monthlyData = new Map<string, {start: number, end: number}>()
    equity.forEach(point => {
      const month = new Date(point.timestamp).toISOString().substring(0, 7)
      if (!monthlyData.has(month)) {
        monthlyData.set(month, {start: point.equity, end: point.equity})
      } else {
        monthlyData.get(month)!.end = point.equity
      }
    })
    
    monthlyData.forEach((data, month) => {
      const monthReturn = ((data.end - data.start) / data.start) * 100
      monthlyReturns.push({month, return: monthReturn})
    })
    
    // Calculate yearly returns
    const yearlyData = new Map<number, {start: number, end: number}>()
    equity.forEach(point => {
      const year = new Date(point.timestamp).getFullYear()
      if (!yearlyData.has(year)) {
        yearlyData.set(year, {start: point.equity, end: point.equity})
      } else {
        yearlyData.get(year)!.end = point.equity
      }
    })
    
    yearlyData.forEach((data, year) => {
      const yearReturn = ((data.end - data.start) / data.start) * 100
      yearlyReturns.push({year, return: yearReturn})
    })
    
    return { monthlyReturns, yearlyReturns, rollingMetrics }
  }

  private async optimizeParameters(
    config: BacktestConfig,
    parameterRanges: Record<string, any[]>
  ): Promise<any> {
    
    const parameterKeys = Object.keys(parameterRanges)
    let bestParams = config.strategyParameters
    let bestScore = -Infinity
    
    // Generate all parameter combinations (for small parameter spaces)
    const combinations = this.generateParameterCombinations(parameterRanges)
    
    for (const params of combinations.slice(0, 100)) { // Increased to 100 combinations for better optimization
      const testConfig = {
        ...config,
        strategyId: `${config.strategyId}_opt_${Date.now()}`,
        strategyParameters: { ...config.strategyParameters, ...params }
      }
      
      const result = await this.runBacktest(testConfig)
      
      // Use composite score: Sharpe ratio with penalty for high drawdown
      const score = result.performance.sharpeRatio - (result.performance.maxDrawdown / 100)
      
      if (score > bestScore) {
        bestScore = score
        bestParams = testConfig.strategyParameters
      }
    }
    
    return bestParams
  }

  private generateParameterCombinations(ranges: Record<string, any[]>): any[] {
    const keys = Object.keys(ranges)
    if (keys.length === 0) return [{}]
    
    const [firstKey, ...restKeys] = keys
    const restRanges = Object.fromEntries(restKeys.map(key => [key, ranges[key]]))
    const restCombinations = this.generateParameterCombinations(restRanges)
    
    const combinations: any[] = []
    for (const value of ranges[firstKey]) {
      for (const restCombination of restCombinations) {
        combinations.push({ [firstKey]: value, ...restCombination })
      }
    }
    
    return combinations
  }

  private perturbParameters(params: any, perturbationLevel: number): any {
    // Handle null/undefined params
    if (!params || typeof params !== 'object') {
      console.warn('perturbParameters: params is null/undefined, returning default parameters')
      return {
        entryThreshold: 0.02 * (1 + (Math.random() - 0.5) * 2 * perturbationLevel),
        exitThreshold: 0.01 * (1 + (Math.random() - 0.5) * 2 * perturbationLevel),
        stopLoss: 0.05 * (1 + (Math.random() - 0.5) * 2 * perturbationLevel),
        takeProfit: 0.03 * (1 + (Math.random() - 0.5) * 2 * perturbationLevel),
        maxPositionSize: 0.1 * (1 + (Math.random() - 0.5) * 2 * perturbationLevel)
      }
    }
    
    const perturbed = { ...params }
    
    for (const [key, value] of Object.entries(params)) {
      if (typeof value === 'number') {
        const perturbation = (Math.random() - 0.5) * 2 * perturbationLevel
        perturbed[key] = value * (1 + perturbation)
      }
    }
    
    return perturbed
  }
}

export default ProductionBacktestingEngine