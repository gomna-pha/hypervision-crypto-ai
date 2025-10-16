import { EventEmitter } from 'events';
import { KafkaManager, RedisManager, DatabaseManager } from '../core/infrastructure';
import { ArbitragePrediction } from '../core/fusion/fusion-brain';
import { ExecutionPlan } from '../decision/decision-engine';
import { ArbitrageExecution } from '../execution/execution-agent';

export interface HistoricalDataPoint {
  timestamp: number;
  symbol: string;
  exchange: string;
  price: number;
  volume: number;
  bid: number;
  ask: number;
  spread: number;
}

export interface BacktestPosition {
  id: string;
  symbol: string;
  entryPrice: number;
  exitPrice?: number;
  quantity: number;
  side: 'long' | 'short';
  entryTime: number;
  exitTime?: number;
  pnl?: number;
  fees: number;
  status: 'open' | 'closed' | 'cancelled';
}

export interface BacktestMetrics {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  totalPnL: number;
  totalFees: number;
  netPnL: number;
  maxDrawdown: number;
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  maxConsecutiveWins: number;
  maxConsecutiveLosses: number;
  averageWin: number;
  averageLoss: number;
  profitFactor: number;
  recoveryFactor: number;
  volatility: number;
  beta: number;
  alpha: number;
}

export interface BacktestConfig {
  startDate: number;
  endDate: number;
  initialCapital: number;
  maxPositionSize: number;
  maxConcurrentPositions: number;
  commission: number; // Per trade commission in basis points
  slippage: number; // Slippage in basis points
  riskFreeRate: number; // Annual risk-free rate for Sharpe calculation
  benchmarkSymbol?: string; // For beta/alpha calculation
  timeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
  symbols: string[];
  exchanges: string[];
}

export interface BacktestResult {
  id: string;
  config: BacktestConfig;
  metrics: BacktestMetrics;
  positions: BacktestPosition[];
  equityCurve: { timestamp: number; equity: number; drawdown: number }[];
  predictions: ArbitragePrediction[];
  executionPlans: ExecutionPlan[];
  performanceBySymbol: Map<string, BacktestMetrics>;
  performanceByExchange: Map<string, BacktestMetrics>;
  startTime: number;
  endTime: number;
  duration: number;
}

export interface LiveValidationResult {
  predictionId: string;
  actualOutcome: {
    realized: boolean;
    actualPnL: number;
    timeToRealization: number;
    accuracyScore: number; // 0-1 score of prediction accuracy
  };
  timestamp: number;
}

export class BacktestingEngine extends EventEmitter {
  private kafka: KafkaManager;
  private redis: RedisManager;
  private db: DatabaseManager;
  private isRunning: boolean = false;
  private currentBacktest?: BacktestResult;
  private historicalData: Map<string, HistoricalDataPoint[]> = new Map();
  private liveValidations: Map<string, LiveValidationResult> = new Map();

  constructor() {
    super();
    this.kafka = new KafkaManager();
    this.redis = new RedisManager();
    this.db = new DatabaseManager();
    this.setupEventListeners();
  }

  private setupEventListeners(): void {
    // Listen for live predictions to validate
    this.kafka.subscribe('fusion-predictions', (prediction: ArbitragePrediction) => {
      this.startLiveValidation(prediction);
    });

    // Listen for execution results for validation
    this.kafka.subscribe('execution-results', (execution: ArbitrageExecution) => {
      this.updateLiveValidation(execution);
    });
  }

  async initialize(): Promise<void> {
    await this.kafka.connect();
    await this.redis.connect();
    await this.db.connect();
    
    // Create backtesting tables
    await this.createBacktestingTables();
    
    console.log('BacktestingEngine initialized');
  }

  private async createBacktestingTables(): Promise<void> {
    const queries = [
      `CREATE TABLE IF NOT EXISTS backtests (
        id TEXT PRIMARY KEY,
        config TEXT NOT NULL,
        metrics TEXT NOT NULL,
        start_time INTEGER NOT NULL,
        end_time INTEGER NOT NULL,
        duration INTEGER NOT NULL,
        created_at INTEGER DEFAULT (strftime('%s', 'now'))
      )`,
      `CREATE TABLE IF NOT EXISTS backtest_positions (
        id TEXT PRIMARY KEY,
        backtest_id TEXT NOT NULL,
        symbol TEXT NOT NULL,
        entry_price REAL NOT NULL,
        exit_price REAL,
        quantity REAL NOT NULL,
        side TEXT NOT NULL,
        entry_time INTEGER NOT NULL,
        exit_time INTEGER,
        pnl REAL,
        fees REAL NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (backtest_id) REFERENCES backtests (id)
      )`,
      `CREATE TABLE IF NOT EXISTS live_validations (
        prediction_id TEXT PRIMARY KEY,
        actual_outcome TEXT NOT NULL,
        timestamp INTEGER NOT NULL,
        accuracy_score REAL NOT NULL
      )`,
      `CREATE TABLE IF NOT EXISTS historical_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        symbol TEXT NOT NULL,
        exchange TEXT NOT NULL,
        price REAL NOT NULL,
        volume REAL NOT NULL,
        bid REAL NOT NULL,
        ask REAL NOT NULL,
        spread REAL NOT NULL,
        UNIQUE(timestamp, symbol, exchange)
      )`
    ];

    for (const query of queries) {
      await this.db.run(query);
    }
  }

  async loadHistoricalData(config: BacktestConfig): Promise<void> {
    console.log('Loading historical data for backtesting...');
    
    for (const symbol of config.symbols) {
      for (const exchange of config.exchanges) {
        const data = await this.fetchHistoricalData(
          symbol,
          exchange,
          config.startDate,
          config.endDate,
          config.timeframe
        );
        
        const key = `${symbol}-${exchange}`;
        this.historicalData.set(key, data);
      }
    }
    
    console.log(`Loaded historical data for ${this.historicalData.size} symbol-exchange pairs`);
  }

  private async fetchHistoricalData(
    symbol: string,
    exchange: string,
    startDate: number,
    endDate: number,
    timeframe: string
  ): Promise<HistoricalDataPoint[]> {
    // First try to get from database
    const query = `
      SELECT * FROM historical_data 
      WHERE symbol = ? AND exchange = ? 
      AND timestamp BETWEEN ? AND ?
      ORDER BY timestamp ASC
    `;
    
    const rows = await this.db.all(query, [symbol, exchange, startDate, endDate]);
    
    if (rows.length > 0) {
      return rows.map(row => ({
        timestamp: row.timestamp,
        symbol: row.symbol,
        exchange: row.exchange,
        price: row.price,
        volume: row.volume,
        bid: row.bid,
        ask: row.ask,
        spread: row.spread
      }));
    }

    // If not in database, fetch from exchange APIs (simplified simulation)
    console.log(`Fetching historical data for ${symbol} on ${exchange}`);
    const data: HistoricalDataPoint[] = [];
    
    // Generate simulated historical data for demonstration
    // In production, this would call real exchange APIs
    const intervalMs = this.getIntervalMs(timeframe);
    let currentTime = startDate;
    let basePrice = 50000; // Starting price
    
    while (currentTime <= endDate) {
      const price = basePrice * (1 + (Math.random() - 0.5) * 0.02); // 2% max variation
      const spread = price * 0.001; // 0.1% spread
      const volume = Math.random() * 1000;
      
      const dataPoint: HistoricalDataPoint = {
        timestamp: currentTime,
        symbol,
        exchange,
        price,
        volume,
        bid: price - spread / 2,
        ask: price + spread / 2,
        spread: spread / price * 10000 // In basis points
      };
      
      data.push(dataPoint);
      
      // Store in database for future use
      await this.db.run(
        `INSERT OR REPLACE INTO historical_data 
         (timestamp, symbol, exchange, price, volume, bid, ask, spread) 
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        [dataPoint.timestamp, dataPoint.symbol, dataPoint.exchange, 
         dataPoint.price, dataPoint.volume, dataPoint.bid, dataPoint.ask, dataPoint.spread]
      );
      
      basePrice = price; // Walk the price
      currentTime += intervalMs;
    }
    
    return data;
  }

  private getIntervalMs(timeframe: string): number {
    const intervals = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      '4h': 4 * 60 * 60 * 1000,
      '1d': 24 * 60 * 60 * 1000
    };
    return intervals[timeframe] || intervals['1h'];
  }

  async runBacktest(config: BacktestConfig): Promise<BacktestResult> {
    console.log('Starting backtest with config:', config);
    this.isRunning = true;
    const startTime = Date.now();
    
    const backtestId = `backtest_${startTime}`;
    const positions: BacktestPosition[] = [];
    const equityCurve: { timestamp: number; equity: number; drawdown: number }[] = [];
    const predictions: ArbitragePrediction[] = [];
    const executionPlans: ExecutionPlan[] = [];
    
    let currentEquity = config.initialCapital;
    let peakEquity = config.initialCapital;
    let maxDrawdown = 0;
    
    // Load historical data
    await this.loadHistoricalData(config);
    
    // Get all timestamps for the backtest period
    const allTimestamps = this.getAllTimestamps(config);
    
    // Simulate the strategy over historical data
    for (let i = 0; i < allTimestamps.length; i++) {
      const timestamp = allTimestamps[i];
      
      // Get market data at this timestamp
      const marketData = this.getMarketDataAtTimestamp(timestamp);
      
      // Skip if insufficient data
      if (marketData.length < 2) continue;
      
      // Simulate agent data collection (simplified for backtesting)
      const agentData = this.simulateAgentData(marketData, timestamp);
      
      // Generate prediction using simplified logic
      const prediction = this.simulatePrediction(agentData, timestamp);
      if (prediction) {
        predictions.push(prediction);
        
        // Generate execution plan
        const executionPlan = this.simulateExecutionPlan(prediction, marketData);
        if (executionPlan) {
          executionPlans.push(executionPlan);
          
          // Execute the plan and track position
          const newPositions = await this.simulateExecution(
            executionPlan, 
            marketData, 
            config,
            timestamp
          );
          positions.push(...newPositions);
        }
      }
      
      // Close positions that have reached targets or stops
      await this.updatePositions(positions, marketData, timestamp);
      
      // Calculate current equity
      currentEquity = this.calculateCurrentEquity(positions, marketData, config.initialCapital);
      
      // Update peak and drawdown
      if (currentEquity > peakEquity) {
        peakEquity = currentEquity;
      }
      const currentDrawdown = (peakEquity - currentEquity) / peakEquity;
      if (currentDrawdown > maxDrawdown) {
        maxDrawdown = currentDrawdown;
      }
      
      // Record equity curve
      equityCurve.push({
        timestamp,
        equity: currentEquity,
        drawdown: currentDrawdown
      });
      
      // Emit progress updates
      if (i % 100 === 0) {
        this.emit('backtest-progress', {
          backtestId,
          progress: (i / allTimestamps.length) * 100,
          currentEquity,
          maxDrawdown,
          totalPositions: positions.length
        });
      }
    }
    
    const endTime = Date.now();
    const duration = endTime - startTime;
    
    // Calculate final metrics
    const metrics = this.calculateMetrics(positions, equityCurve, config);
    
    // Create result
    const result: BacktestResult = {
      id: backtestId,
      config,
      metrics,
      positions,
      equityCurve,
      predictions,
      executionPlans,
      performanceBySymbol: this.calculatePerformanceBySymbol(positions),
      performanceByExchange: this.calculatePerformanceByExchange(positions),
      startTime,
      endTime,
      duration
    };
    
    // Store in database
    await this.storeBacktestResult(result);
    
    this.currentBacktest = result;
    this.isRunning = false;
    
    console.log(`Backtest completed in ${duration}ms`);
    this.emit('backtest-complete', result);
    
    return result;
  }

  private getAllTimestamps(config: BacktestConfig): number[] {
    const timestamps = new Set<number>();
    
    for (const [key, data] of this.historicalData) {
      for (const point of data) {
        if (point.timestamp >= config.startDate && point.timestamp <= config.endDate) {
          timestamps.add(point.timestamp);
        }
      }
    }
    
    return Array.from(timestamps).sort((a, b) => a - b);
  }

  private getMarketDataAtTimestamp(timestamp: number): HistoricalDataPoint[] {
    const data: HistoricalDataPoint[] = [];
    
    for (const [key, points] of this.historicalData) {
      const point = points.find(p => p.timestamp === timestamp);
      if (point) {
        data.push(point);
      }
    }
    
    return data;
  }

  private simulateAgentData(marketData: HistoricalDataPoint[], timestamp: number): any {
    // Simplified agent data simulation for backtesting
    const prices = marketData.map(d => d.price);
    const volumes = marketData.map(d => d.volume);
    const spreads = marketData.map(d => d.spread);
    
    return {
      economic: {
        timestamp,
        cpi: 3.2 + Math.random() * 0.5,
        fedFundsRate: 5.25 + Math.random() * 0.25,
        unemployment: 3.8 + Math.random() * 0.3
      },
      sentiment: {
        timestamp,
        twitterSentiment: Math.random() * 2 - 1, // -1 to 1
        redditSentiment: Math.random() * 2 - 1,
        newsSentiment: Math.random() * 2 - 1
      },
      price: {
        timestamp,
        prices,
        spreads,
        volatility: this.calculateVolatility(prices)
      },
      volume: {
        timestamp,
        volumes,
        volumeProfile: this.calculateVolumeProfile(volumes)
      }
    };
  }

  private calculateVolatility(prices: number[]): number {
    if (prices.length < 2) return 0;
    
    const returns = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push((prices[i] - prices[i-1]) / prices[i-1]);
    }
    
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
    return Math.sqrt(variance);
  }

  private calculateVolumeProfile(volumes: number[]): any {
    const totalVolume = volumes.reduce((a, b) => a + b, 0);
    return {
      total: totalVolume,
      average: totalVolume / volumes.length,
      max: Math.max(...volumes),
      min: Math.min(...volumes)
    };
  }

  private simulatePrediction(agentData: any, timestamp: number): ArbitragePrediction | null {
    // Simplified prediction logic for backtesting
    const volatility = agentData.price.volatility;
    const sentiment = (agentData.sentiment.twitterSentiment + 
                      agentData.sentiment.redditSentiment + 
                      agentData.sentiment.newsSentiment) / 3;
    
    // Only generate predictions when conditions are favorable
    if (volatility > 0.02 && Math.abs(sentiment) > 0.3) {
      return {
        id: `pred_${timestamp}`,
        timestamp,
        opportunities: [{
          id: `opp_${timestamp}`,
          type: 'price_arbitrage',
          buyExchange: 'binance',
          sellExchange: 'coinbase',
          symbol: 'BTC/USDT',
          expectedProfit: volatility * 1000, // Simplified profit calculation
          confidence: Math.min(Math.abs(sentiment) + volatility, 1.0),
          timeHorizon: 300, // 5 minutes
          riskLevel: volatility < 0.03 ? 'low' : 'medium'
        }],
        confidence: Math.min(Math.abs(sentiment) + volatility, 1.0),
        riskAssessment: {
          overall: volatility < 0.03 ? 'low' : 'medium',
          factors: ['market_volatility', 'sentiment_strength']
        },
        reasoning: `High volatility (${volatility.toFixed(4)}) and strong sentiment (${sentiment.toFixed(2)}) suggest arbitrage opportunity`,
        hyperbolicContext: {
          embedding: [Math.random(), Math.random(), Math.random()],
          neighbors: [],
          confidence: 0.8
        }
      };
    }
    
    return null;
  }

  private simulateExecutionPlan(prediction: ArbitragePrediction, marketData: HistoricalDataPoint[]): ExecutionPlan | null {
    if (prediction.opportunities.length === 0) return null;
    
    const opportunity = prediction.opportunities[0];
    const buyData = marketData.find(d => d.exchange === opportunity.buyExchange);
    const sellData = marketData.find(d => d.exchange === opportunity.sellExchange);
    
    if (!buyData || !sellData) return null;
    
    return {
      id: `plan_${prediction.id}`,
      predictionId: prediction.id,
      trades: [
        {
          id: `trade_buy_${prediction.id}`,
          exchange: opportunity.buyExchange,
          symbol: opportunity.symbol,
          side: 'buy',
          type: 'market',
          quantity: 0.1,
          expectedPrice: buyData.ask,
          expectedFees: buyData.ask * 0.1 * 0.001
        },
        {
          id: `trade_sell_${prediction.id}`,
          exchange: opportunity.sellExchange,
          symbol: opportunity.symbol,
          side: 'sell',
          type: 'market',
          quantity: 0.1,
          expectedPrice: sellData.bid,
          expectedFees: sellData.bid * 0.1 * 0.001
        }
      ],
      expectedProfit: opportunity.expectedProfit,
      maxRisk: opportunity.expectedProfit * 2,
      timeLimit: Date.now() + (opportunity.timeHorizon * 1000),
      constraints: {
        maxSlippage: 0.001,
        maxLatency: 1000,
        minLiquidity: 1000
      }
    };
  }

  private async simulateExecution(
    plan: ExecutionPlan, 
    marketData: HistoricalDataPoint[], 
    config: BacktestConfig,
    timestamp: number
  ): Promise<BacktestPosition[]> {
    const positions: BacktestPosition[] = [];
    
    for (const trade of plan.trades) {
      const exchangeData = marketData.find(d => d.exchange === trade.exchange);
      if (!exchangeData) continue;
      
      // Apply slippage
      const slippageMultiplier = 1 + (trade.side === 'buy' ? config.slippage : -config.slippage) / 10000;
      const executionPrice = trade.expectedPrice * slippageMultiplier;
      
      // Calculate fees
      const fees = executionPrice * trade.quantity * config.commission / 10000;
      
      const position: BacktestPosition = {
        id: `pos_${trade.id}`,
        symbol: trade.symbol,
        entryPrice: executionPrice,
        quantity: trade.side === 'buy' ? trade.quantity : -trade.quantity,
        side: trade.side === 'buy' ? 'long' : 'short',
        entryTime: timestamp,
        fees,
        status: 'open'
      };
      
      positions.push(position);
    }
    
    return positions;
  }

  private async updatePositions(
    positions: BacktestPosition[], 
    marketData: HistoricalDataPoint[], 
    timestamp: number
  ): Promise<void> {
    for (const position of positions) {
      if (position.status !== 'open') continue;
      
      // Find current market price
      const currentData = marketData.find(d => 
        d.symbol === position.symbol.replace('/', '')
      );
      if (!currentData) continue;
      
      const currentPrice = position.side === 'long' ? currentData.bid : currentData.ask;
      const unrealizedPnL = position.side === 'long' 
        ? (currentPrice - position.entryPrice) * Math.abs(position.quantity)
        : (position.entryPrice - currentPrice) * Math.abs(position.quantity);
      
      // Simple exit logic: close position after 5 minutes or if profit/loss exceeds thresholds
      const timeElapsed = timestamp - position.entryTime;
      const profitThreshold = position.entryPrice * 0.01; // 1% profit target
      const lossThreshold = position.entryPrice * 0.005; // 0.5% stop loss
      
      if (timeElapsed > 5 * 60 * 1000 || // 5 minutes
          unrealizedPnL > profitThreshold || 
          unrealizedPnL < -lossThreshold) {
        
        position.exitPrice = currentPrice;
        position.exitTime = timestamp;
        position.pnl = unrealizedPnL - position.fees;
        position.status = 'closed';
      }
    }
  }

  private calculateCurrentEquity(
    positions: BacktestPosition[], 
    marketData: HistoricalDataPoint[], 
    initialCapital: number
  ): number {
    let equity = initialCapital;
    
    for (const position of positions) {
      if (position.status === 'closed' && position.pnl !== undefined) {
        equity += position.pnl;
      } else if (position.status === 'open') {
        // Calculate unrealized PnL
        const currentData = marketData.find(d => 
          d.symbol === position.symbol.replace('/', '')
        );
        if (currentData) {
          const currentPrice = position.side === 'long' ? currentData.bid : currentData.ask;
          const unrealizedPnL = position.side === 'long' 
            ? (currentPrice - position.entryPrice) * Math.abs(position.quantity)
            : (position.entryPrice - currentPrice) * Math.abs(position.quantity);
          equity += unrealizedPnL - position.fees;
        }
      }
    }
    
    return equity;
  }

  private calculateMetrics(
    positions: BacktestPosition[], 
    equityCurve: { timestamp: number; equity: number; drawdown: number }[], 
    config: BacktestConfig
  ): BacktestMetrics {
    const closedPositions = positions.filter(p => p.status === 'closed' && p.pnl !== undefined);
    const winningTrades = closedPositions.filter(p => p.pnl! > 0);
    const losingTrades = closedPositions.filter(p => p.pnl! <= 0);
    
    const totalPnL = closedPositions.reduce((sum, p) => sum + p.pnl!, 0);
    const totalFees = positions.reduce((sum, p) => sum + p.fees, 0);
    const netPnL = totalPnL - totalFees;
    
    const winRate = closedPositions.length > 0 ? winningTrades.length / closedPositions.length : 0;
    const averageWin = winningTrades.length > 0 ? 
      winningTrades.reduce((sum, p) => sum + p.pnl!, 0) / winningTrades.length : 0;
    const averageLoss = losingTrades.length > 0 ? 
      Math.abs(losingTrades.reduce((sum, p) => sum + p.pnl!, 0)) / losingTrades.length : 0;
    
    const profitFactor = averageLoss > 0 ? (averageWin * winningTrades.length) / (averageLoss * losingTrades.length) : 0;
    
    // Calculate Sharpe ratio
    const returns = equityCurve.slice(1).map((point, i) => 
      (point.equity - equityCurve[i].equity) / equityCurve[i].equity
    );
    const averageReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const returnVolatility = Math.sqrt(
      returns.reduce((sum, r) => sum + Math.pow(r - averageReturn, 2), 0) / returns.length
    );
    const sharpeRatio = returnVolatility > 0 ? 
      (averageReturn - config.riskFreeRate / 252) / returnVolatility * Math.sqrt(252) : 0;
    
    // Calculate max drawdown
    const maxDrawdown = Math.max(...equityCurve.map(point => point.drawdown));
    
    // Calculate Sortino ratio (downside deviation)
    const negativeReturns = returns.filter(r => r < 0);
    const downsideDeviation = negativeReturns.length > 0 ? 
      Math.sqrt(negativeReturns.reduce((sum, r) => sum + r * r, 0) / negativeReturns.length) : 0;
    const sortinoRatio = downsideDeviation > 0 ? 
      (averageReturn - config.riskFreeRate / 252) / downsideDeviation * Math.sqrt(252) : 0;
    
    // Calculate Calmar ratio
    const annualizedReturn = averageReturn * 252;
    const calmarRatio = maxDrawdown > 0 ? annualizedReturn / maxDrawdown : 0;
    
    return {
      totalTrades: closedPositions.length,
      winningTrades: winningTrades.length,
      losingTrades: losingTrades.length,
      winRate,
      totalPnL,
      totalFees,
      netPnL,
      maxDrawdown,
      sharpeRatio,
      sortinoRatio,
      calmarRatio,
      maxConsecutiveWins: this.calculateMaxConsecutive(closedPositions, true),
      maxConsecutiveLosses: this.calculateMaxConsecutive(closedPositions, false),
      averageWin,
      averageLoss,
      profitFactor,
      recoveryFactor: maxDrawdown > 0 ? netPnL / (config.initialCapital * maxDrawdown) : 0,
      volatility: returnVolatility * Math.sqrt(252),
      beta: 1.0, // Simplified - would need benchmark data
      alpha: annualizedReturn - config.riskFreeRate // Simplified alpha calculation
    };
  }

  private calculateMaxConsecutive(positions: BacktestPosition[], wins: boolean): number {
    let maxStreak = 0;
    let currentStreak = 0;
    
    for (const position of positions) {
      if (!position.pnl) continue;
      
      const isWin = position.pnl > 0;
      if (isWin === wins) {
        currentStreak++;
        maxStreak = Math.max(maxStreak, currentStreak);
      } else {
        currentStreak = 0;
      }
    }
    
    return maxStreak;
  }

  private calculatePerformanceBySymbol(positions: BacktestPosition[]): Map<string, BacktestMetrics> {
    const symbolGroups = new Map<string, BacktestPosition[]>();
    
    for (const position of positions) {
      if (!symbolGroups.has(position.symbol)) {
        symbolGroups.set(position.symbol, []);
      }
      symbolGroups.get(position.symbol)!.push(position);
    }
    
    const result = new Map<string, BacktestMetrics>();
    for (const [symbol, symbolPositions] of symbolGroups) {
      // Create a simplified config for symbol-specific metrics
      const symbolConfig: BacktestConfig = {
        startDate: 0,
        endDate: 0,
        initialCapital: 10000,
        maxPositionSize: 1000,
        maxConcurrentPositions: 5,
        commission: 10,
        slippage: 5,
        riskFreeRate: 0.02,
        timeframe: '1h',
        symbols: [symbol],
        exchanges: ['binance']
      };
      
      const symbolEquityCurve = this.buildEquityCurveFromPositions(symbolPositions);
      const metrics = this.calculateMetrics(symbolPositions, symbolEquityCurve, symbolConfig);
      result.set(symbol, metrics);
    }
    
    return result;
  }

  private calculatePerformanceByExchange(positions: BacktestPosition[]): Map<string, BacktestMetrics> {
    // Similar to calculatePerformanceBySymbol but grouped by exchange
    // Implementation would analyze positions by their originating exchange
    return new Map();
  }

  private buildEquityCurveFromPositions(positions: BacktestPosition[]): { timestamp: number; equity: number; drawdown: number }[] {
    // Simplified equity curve builder for symbol-specific analysis
    const curve: { timestamp: number; equity: number; drawdown: number }[] = [];
    let equity = 10000;
    let peak = equity;
    
    for (const position of positions.filter(p => p.status === 'closed')) {
      if (position.pnl) {
        equity += position.pnl;
        if (equity > peak) peak = equity;
        
        curve.push({
          timestamp: position.exitTime || position.entryTime,
          equity,
          drawdown: (peak - equity) / peak
        });
      }
    }
    
    return curve;
  }

  private async storeBacktestResult(result: BacktestResult): Promise<void> {
    await this.db.run(
      `INSERT INTO backtests (id, config, metrics, start_time, end_time, duration) 
       VALUES (?, ?, ?, ?, ?, ?)`,
      [
        result.id,
        JSON.stringify(result.config),
        JSON.stringify(result.metrics),
        result.startTime,
        result.endTime,
        result.duration
      ]
    );
    
    for (const position of result.positions) {
      await this.db.run(
        `INSERT INTO backtest_positions 
         (id, backtest_id, symbol, entry_price, exit_price, quantity, side, 
          entry_time, exit_time, pnl, fees, status) 
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          position.id,
          result.id,
          position.symbol,
          position.entryPrice,
          position.exitPrice,
          position.quantity,
          position.side,
          position.entryTime,
          position.exitTime,
          position.pnl,
          position.fees,
          position.status
        ]
      );
    }
  }

  // Live Validation Methods
  private startLiveValidation(prediction: ArbitragePrediction): void {
    console.log(`Starting live validation for prediction ${prediction.id}`);
    
    // Set up tracking for this prediction
    const validationEndTime = prediction.timestamp + (30 * 60 * 1000); // 30 minutes
    
    setTimeout(() => {
      this.completeLiveValidation(prediction.id);
    }, validationEndTime - Date.now());
  }

  private updateLiveValidation(execution: ArbitrageExecution): void {
    if (this.liveValidations.has(execution.predictionId)) {
      const validation = this.liveValidations.get(execution.predictionId)!;
      
      // Update with actual execution results
      validation.actualOutcome.realized = execution.status === 'completed';
      validation.actualOutcome.actualPnL = execution.actualProfit || 0;
      validation.actualOutcome.timeToRealization = Date.now() - validation.timestamp;
      
      // Calculate accuracy score based on prediction vs reality
      const prediction = execution.executionPlan?.expectedProfit || 0;
      const actual = execution.actualProfit || 0;
      validation.actualOutcome.accuracyScore = this.calculateAccuracyScore(prediction, actual);
      
      this.liveValidations.set(execution.predictionId, validation);
      
      this.emit('live-validation-update', validation);
    }
  }

  private completeLiveValidation(predictionId: string): void {
    const validation = this.liveValidations.get(predictionId);
    if (validation) {
      // Store in database
      this.db.run(
        `INSERT INTO live_validations (prediction_id, actual_outcome, timestamp, accuracy_score) 
         VALUES (?, ?, ?, ?)`,
        [
          predictionId,
          JSON.stringify(validation.actualOutcome),
          validation.timestamp,
          validation.actualOutcome.accuracyScore
        ]
      );
      
      this.emit('live-validation-complete', validation);
    }
  }

  private calculateAccuracyScore(predicted: number, actual: number): number {
    if (predicted === 0) return actual === 0 ? 1.0 : 0.0;
    
    const error = Math.abs(predicted - actual) / Math.abs(predicted);
    return Math.max(0, 1 - error);
  }

  // Public methods for dashboard integration
  async getBacktestResults(limit: number = 10): Promise<BacktestResult[]> {
    const rows = await this.db.all(
      `SELECT * FROM backtests ORDER BY created_at DESC LIMIT ?`,
      [limit]
    );
    
    const results: BacktestResult[] = [];
    for (const row of rows) {
      const positions = await this.db.all(
        `SELECT * FROM backtest_positions WHERE backtest_id = ?`,
        [row.id]
      );
      
      results.push({
        id: row.id,
        config: JSON.parse(row.config),
        metrics: JSON.parse(row.metrics),
        positions: positions.map(p => ({
          id: p.id,
          symbol: p.symbol,
          entryPrice: p.entry_price,
          exitPrice: p.exit_price,
          quantity: p.quantity,
          side: p.side,
          entryTime: p.entry_time,
          exitTime: p.exit_time,
          pnl: p.pnl,
          fees: p.fees,
          status: p.status
        })),
        equityCurve: [], // Would be reconstructed from positions
        predictions: [],
        executionPlans: [],
        performanceBySymbol: new Map(),
        performanceByExchange: new Map(),
        startTime: row.start_time,
        endTime: row.end_time,
        duration: row.duration
      });
    }
    
    return results;
  }

  async getLiveValidations(limit: number = 50): Promise<LiveValidationResult[]> {
    const rows = await this.db.all(
      `SELECT * FROM live_validations ORDER BY timestamp DESC LIMIT ?`,
      [limit]
    );
    
    return rows.map(row => ({
      predictionId: row.prediction_id,
      actualOutcome: JSON.parse(row.actual_outcome),
      timestamp: row.timestamp
    }));
  }

  getCurrentBacktest(): BacktestResult | undefined {
    return this.currentBacktest;
  }

  isBacktesting(): boolean {
    return this.isRunning;
  }

  async stop(): Promise<void> {
    this.isRunning = false;
    await this.kafka.disconnect();
    await this.redis.disconnect();
    await this.db.close();
  }
}