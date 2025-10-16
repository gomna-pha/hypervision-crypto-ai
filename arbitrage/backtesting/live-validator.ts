import { EventEmitter } from 'events';
import { KafkaManager, RedisManager, DatabaseManager } from '../core/infrastructure';
import { ArbitragePrediction } from '../core/fusion/fusion-brain';
import { ExecutionPlan } from '../decision/decision-engine';
import { ArbitrageExecution } from '../execution/execution-agent';
import { BacktestingEngine, BacktestConfig } from './backtesting-engine';

export interface WalkForwardResult {
  id: string;
  period: string;
  inSamplePeriod: { start: number; end: number };
  outOfSamplePeriod: { start: number; end: number };
  inSampleMetrics: any;
  outOfSampleMetrics: any;
  degradation: number; // Performance degradation from in-sample to out-of-sample
  timestamp: number;
}

export interface PredictionValidation {
  predictionId: string;
  prediction: ArbitragePrediction;
  actualMarketMovement: {
    symbol: string;
    priceChange: number;
    volumeChange: number;
    timeToMove: number;
    realized: boolean;
  };
  accuracyMetrics: {
    directionAccuracy: number; // 0-1 score for direction prediction
    magnitudeAccuracy: number; // 0-1 score for magnitude prediction
    timingAccuracy: number; // 0-1 score for timing prediction
    overallScore: number; // Weighted average
  };
  timestamp: number;
}

export interface LiveMetrics {
  timestamp: number;
  totalPredictions: number;
  validatedPredictions: number;
  averageAccuracy: number;
  recentAccuracy: number; // Last 24 hours
  confidenceCalibration: number; // How well confidence matches actual accuracy
  profitabilityScore: number; // Actual profit vs predicted profit
  riskScore: number; // Actual risk vs predicted risk
  modelDrift: number; // Measure of model performance degradation
}

export interface ComparisonResult {
  backtestId: string;
  livePerformance: {
    period: { start: number; end: number };
    actualPnL: number;
    actualWinRate: number;
    actualSharpe: number;
    actualMaxDrawdown: number;
  };
  backtestPrediction: {
    expectedPnL: number;
    expectedWinRate: number;
    expectedSharpe: number;
    expectedMaxDrawdown: number;
  };
  deviation: {
    pnlDeviation: number;
    winRateDeviation: number;
    sharpeDeviation: number;
    drawdownDeviation: number;
    overallDeviation: number;
  };
  reliability: number; // 0-1 score of backtest reliability
}

export class LiveValidator extends EventEmitter {
  private kafka: KafkaManager;
  private redis: RedisManager;
  private db: DatabaseManager;
  private backtestEngine: BacktestingEngine;
  private isRunning: boolean = false;
  private activePredictions: Map<string, PredictionValidation> = new Map();
  private liveMetrics: LiveMetrics;
  private walkForwardResults: WalkForwardResult[] = [];

  constructor() {
    super();
    this.kafka = new KafkaManager();
    this.redis = new RedisManager();
    this.db = new DatabaseManager();
    this.backtestEngine = new BacktestingEngine();
    
    this.liveMetrics = {
      timestamp: Date.now(),
      totalPredictions: 0,
      validatedPredictions: 0,
      averageAccuracy: 0,
      recentAccuracy: 0,
      confidenceCalibration: 0,
      profitabilityScore: 0,
      riskScore: 0,
      modelDrift: 0
    };
    
    this.setupEventListeners();
  }

  private setupEventListeners(): void {
    // Listen for new predictions from Fusion Brain
    this.kafka.subscribe('fusion-predictions', (prediction: ArbitragePrediction) => {
      this.startPredictionValidation(prediction);
    });

    // Listen for execution results
    this.kafka.subscribe('execution-results', (execution: ArbitrageExecution) => {
      this.updatePredictionValidation(execution);
    });

    // Listen for market data updates
    this.kafka.subscribe('market-data', (data: any) => {
      this.updateMarketValidations(data);
    });

    // Periodic metrics updates
    setInterval(() => {
      this.updateLiveMetrics();
    }, 60000); // Every minute

    // Daily walk-forward analysis
    setInterval(() => {
      this.runWalkForwardAnalysis();
    }, 24 * 60 * 60 * 1000); // Daily
  }

  async initialize(): Promise<void> {
    await this.kafka.connect();
    await this.redis.connect();
    await this.db.connect();
    await this.backtestEngine.initialize();
    
    await this.createValidationTables();
    
    this.isRunning = true;
    console.log('LiveValidator initialized');
  }

  private async createValidationTables(): Promise<void> {
    const queries = [
      `CREATE TABLE IF NOT EXISTS prediction_validations (
        prediction_id TEXT PRIMARY KEY,
        prediction_data TEXT NOT NULL,
        actual_market_movement TEXT,
        accuracy_metrics TEXT,
        timestamp INTEGER NOT NULL,
        validation_complete BOOLEAN DEFAULT FALSE
      )`,
      `CREATE TABLE IF NOT EXISTS walk_forward_results (
        id TEXT PRIMARY KEY,
        period TEXT NOT NULL,
        in_sample_start INTEGER NOT NULL,
        in_sample_end INTEGER NOT NULL,
        out_sample_start INTEGER NOT NULL,
        out_sample_end INTEGER NOT NULL,
        in_sample_metrics TEXT NOT NULL,
        out_sample_metrics TEXT NOT NULL,
        degradation REAL NOT NULL,
        timestamp INTEGER NOT NULL
      )`,
      `CREATE TABLE IF NOT EXISTS live_metrics_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        metrics TEXT NOT NULL
      )`,
      `CREATE TABLE IF NOT EXISTS backtest_comparisons (
        id TEXT PRIMARY KEY,
        backtest_id TEXT NOT NULL,
        live_performance TEXT NOT NULL,
        backtest_prediction TEXT NOT NULL,
        deviation TEXT NOT NULL,
        reliability REAL NOT NULL,
        timestamp INTEGER NOT NULL
      )`
    ];

    for (const query of queries) {
      await this.db.run(query);
    }
  }

  private startPredictionValidation(prediction: ArbitragePrediction): void {
    console.log(`Starting validation for prediction ${prediction.id}`);
    
    const validation: PredictionValidation = {
      predictionId: prediction.id,
      prediction,
      actualMarketMovement: {
        symbol: prediction.opportunities[0]?.symbol || 'BTC/USDT',
        priceChange: 0,
        volumeChange: 0,
        timeToMove: 0,
        realized: false
      },
      accuracyMetrics: {
        directionAccuracy: 0,
        magnitudeAccuracy: 0,
        timingAccuracy: 0,
        overallScore: 0
      },
      timestamp: Date.now()
    };
    
    this.activePredictions.set(prediction.id, validation);
    
    // Store in database
    this.db.run(
      `INSERT INTO prediction_validations 
       (prediction_id, prediction_data, timestamp) 
       VALUES (?, ?, ?)`,
      [prediction.id, JSON.stringify(prediction), validation.timestamp]
    );
    
    // Set up validation timeout
    const timeHorizon = prediction.opportunities[0]?.timeHorizon || 300; // Default 5 minutes
    setTimeout(() => {
      this.completePredictionValidation(prediction.id);
    }, timeHorizon * 1000);
    
    this.emit('validation-started', validation);
  }

  private updatePredictionValidation(execution: ArbitrageExecution): void {
    const validation = this.activePredictions.get(execution.predictionId);
    if (!validation) return;
    
    // Update with execution results
    if (execution.trades && execution.trades.length > 0) {
      const totalPnL = execution.trades.reduce((sum, trade) => sum + (trade.pnl || 0), 0);
      const avgPrice = execution.trades.reduce((sum, trade) => sum + trade.executedPrice, 0) / execution.trades.length;
      
      validation.actualMarketMovement.priceChange = avgPrice;
      validation.actualMarketMovement.realized = execution.status === 'completed';
      validation.actualMarketMovement.timeToMove = Date.now() - validation.timestamp;
      
      // Calculate accuracy metrics
      this.calculateAccuracyMetrics(validation);
      
      this.activePredictions.set(execution.predictionId, validation);
      this.emit('validation-updated', validation);
    }
  }

  private updateMarketValidations(marketData: any): void {
    // Update all active validations with new market data
    for (const [predictionId, validation] of this.activePredictions) {
      if (marketData.symbol === validation.actualMarketMovement.symbol) {
        // Update price change from prediction time
        const initialPrice = this.getInitialPrice(validation);
        if (initialPrice > 0) {
          validation.actualMarketMovement.priceChange = 
            (marketData.price - initialPrice) / initialPrice;
        }
        
        // Update volume change
        const initialVolume = this.getInitialVolume(validation);
        if (initialVolume > 0) {
          validation.actualMarketMovement.volumeChange = 
            (marketData.volume - initialVolume) / initialVolume;
        }
        
        // Recalculate accuracy metrics
        this.calculateAccuracyMetrics(validation);
        
        this.activePredictions.set(predictionId, validation);
      }
    }
  }

  private getInitialPrice(validation: PredictionValidation): number {
    // Extract initial price from prediction context
    return validation.prediction.opportunities[0]?.expectedProfit || 0;
  }

  private getInitialVolume(validation: PredictionValidation): number {
    // Extract initial volume from prediction context
    return 1000; // Simplified - would extract from actual prediction data
  }

  private calculateAccuracyMetrics(validation: PredictionValidation): void {
    const prediction = validation.prediction;
    const actual = validation.actualMarketMovement;
    
    // Direction accuracy
    const predictedDirection = prediction.opportunities[0]?.expectedProfit || 0;
    const actualDirection = actual.priceChange;
    validation.accuracyMetrics.directionAccuracy = 
      Math.sign(predictedDirection) === Math.sign(actualDirection) ? 1.0 : 0.0;
    
    // Magnitude accuracy (inverse of relative error)
    if (predictedDirection !== 0) {
      const magnitudeError = Math.abs(actualDirection - predictedDirection) / Math.abs(predictedDirection);
      validation.accuracyMetrics.magnitudeAccuracy = Math.max(0, 1 - magnitudeError);
    }
    
    // Timing accuracy (based on time horizon)
    const predictedTime = prediction.opportunities[0]?.timeHorizon || 300;
    const actualTime = actual.timeToMove / 1000; // Convert to seconds
    const timingError = Math.abs(actualTime - predictedTime) / predictedTime;
    validation.accuracyMetrics.timingAccuracy = Math.max(0, 1 - timingError);
    
    // Overall score (weighted average)
    validation.accuracyMetrics.overallScore = 
      validation.accuracyMetrics.directionAccuracy * 0.4 +
      validation.accuracyMetrics.magnitudeAccuracy * 0.4 +
      validation.accuracyMetrics.timingAccuracy * 0.2;
  }

  private completePredictionValidation(predictionId: string): void {
    const validation = this.activePredictions.get(predictionId);
    if (!validation) return;
    
    // Final accuracy calculation
    this.calculateAccuracyMetrics(validation);
    
    // Update database
    this.db.run(
      `UPDATE prediction_validations 
       SET actual_market_movement = ?, accuracy_metrics = ?, validation_complete = TRUE 
       WHERE prediction_id = ?`,
      [
        JSON.stringify(validation.actualMarketMovement),
        JSON.stringify(validation.accuracyMetrics),
        predictionId
      ]
    );
    
    // Remove from active validations
    this.activePredictions.delete(predictionId);
    
    // Update live metrics
    this.liveMetrics.totalPredictions++;
    this.liveMetrics.validatedPredictions++;
    
    this.emit('validation-completed', validation);
    console.log(`Completed validation for prediction ${predictionId}`);
  }

  private updateLiveMetrics(): void {
    // Calculate updated metrics based on recent validations
    this.calculateAverageAccuracy();
    this.calculateRecentAccuracy();
    this.calculateConfidenceCalibration();
    this.calculateProfitabilityScore();
    this.calculateRiskScore();
    this.calculateModelDrift();
    
    this.liveMetrics.timestamp = Date.now();
    
    // Store metrics in database
    this.db.run(
      `INSERT INTO live_metrics_history (timestamp, metrics) VALUES (?, ?)`,
      [this.liveMetrics.timestamp, JSON.stringify(this.liveMetrics)]
    );
    
    // Cache in Redis for fast access
    this.redis.set('live_metrics', JSON.stringify(this.liveMetrics), 60); // 1 minute TTL
    
    this.emit('metrics-updated', this.liveMetrics);
  }

  private async calculateAverageAccuracy(): Promise<void> {
    const rows = await this.db.all(
      `SELECT accuracy_metrics FROM prediction_validations 
       WHERE validation_complete = TRUE AND accuracy_metrics IS NOT NULL`
    );
    
    if (rows.length === 0) {
      this.liveMetrics.averageAccuracy = 0;
      return;
    }
    
    const totalScore = rows.reduce((sum, row) => {
      const metrics = JSON.parse(row.accuracy_metrics);
      return sum + metrics.overallScore;
    }, 0);
    
    this.liveMetrics.averageAccuracy = totalScore / rows.length;
  }

  private async calculateRecentAccuracy(): Promise<void> {
    const oneDayAgo = Date.now() - 24 * 60 * 60 * 1000;
    
    const rows = await this.db.all(
      `SELECT accuracy_metrics FROM prediction_validations 
       WHERE validation_complete = TRUE 
       AND accuracy_metrics IS NOT NULL 
       AND timestamp > ?`,
      [oneDayAgo]
    );
    
    if (rows.length === 0) {
      this.liveMetrics.recentAccuracy = this.liveMetrics.averageAccuracy;
      return;
    }
    
    const totalScore = rows.reduce((sum, row) => {
      const metrics = JSON.parse(row.accuracy_metrics);
      return sum + metrics.overallScore;
    }, 0);
    
    this.liveMetrics.recentAccuracy = totalScore / rows.length;
  }

  private async calculateConfidenceCalibration(): Promise<void> {
    const rows = await this.db.all(
      `SELECT prediction_data, accuracy_metrics FROM prediction_validations 
       WHERE validation_complete = TRUE 
       AND accuracy_metrics IS NOT NULL`
    );
    
    if (rows.length === 0) {
      this.liveMetrics.confidenceCalibration = 0;
      return;
    }
    
    let totalCalibrationError = 0;
    
    for (const row of rows) {
      const prediction = JSON.parse(row.prediction_data);
      const metrics = JSON.parse(row.accuracy_metrics);
      
      const predictedConfidence = prediction.confidence || 0.5;
      const actualAccuracy = metrics.overallScore;
      
      totalCalibrationError += Math.abs(predictedConfidence - actualAccuracy);
    }
    
    // Calibration score: 1 means perfect calibration, 0 means completely miscalibrated
    this.liveMetrics.confidenceCalibration = Math.max(0, 1 - (totalCalibrationError / rows.length));
  }

  private calculateProfitabilityScore(): void {
    // Simplified profitability calculation
    // Would compare predicted profits vs actual execution results
    this.liveMetrics.profitabilityScore = 0.75; // Placeholder
  }

  private calculateRiskScore(): void {
    // Simplified risk assessment
    // Would compare predicted risk vs actual risk experienced
    this.liveMetrics.riskScore = 0.8; // Placeholder
  }

  private calculateModelDrift(): void {
    // Compare recent performance to historical baseline
    const performanceChange = this.liveMetrics.recentAccuracy - this.liveMetrics.averageAccuracy;
    this.liveMetrics.modelDrift = Math.abs(performanceChange);
  }

  async runWalkForwardAnalysis(): Promise<void> {
    console.log('Starting walk-forward analysis...');
    
    const now = Date.now();
    const thirtyDaysAgo = now - 30 * 24 * 60 * 60 * 1000;
    const sixtyDaysAgo = now - 60 * 24 * 60 * 60 * 1000;
    
    // In-sample period: 60-30 days ago
    // Out-of-sample period: 30 days ago to now
    
    const inSampleConfig: BacktestConfig = {
      startDate: sixtyDaysAgo,
      endDate: thirtyDaysAgo,
      initialCapital: 10000,
      maxPositionSize: 1000,
      maxConcurrentPositions: 5,
      commission: 10,
      slippage: 5,
      riskFreeRate: 0.02,
      timeframe: '1h',
      symbols: ['BTC/USDT', 'ETH/USDT'],
      exchanges: ['binance', 'coinbase']
    };
    
    const outOfSampleConfig: BacktestConfig = {
      ...inSampleConfig,
      startDate: thirtyDaysAgo,
      endDate: now
    };
    
    try {
      // Run in-sample backtest
      const inSampleResult = await this.backtestEngine.runBacktest(inSampleConfig);
      
      // Run out-of-sample backtest
      const outOfSampleResult = await this.backtestEngine.runBacktest(outOfSampleConfig);
      
      // Calculate performance degradation
      const degradation = this.calculatePerformanceDegradation(
        inSampleResult.metrics,
        outOfSampleResult.metrics
      );
      
      const walkForwardResult: WalkForwardResult = {
        id: `wf_${now}`,
        period: `${new Date(sixtyDaysAgo).toISOString().split('T')[0]}_to_${new Date(now).toISOString().split('T')[0]}`,
        inSamplePeriod: { start: sixtyDaysAgo, end: thirtyDaysAgo },
        outOfSamplePeriod: { start: thirtyDaysAgo, end: now },
        inSampleMetrics: inSampleResult.metrics,
        outOfSampleMetrics: outOfSampleResult.metrics,
        degradation,
        timestamp: now
      };
      
      this.walkForwardResults.push(walkForwardResult);
      
      // Store in database
      await this.db.run(
        `INSERT INTO walk_forward_results 
         (id, period, in_sample_start, in_sample_end, out_sample_start, out_sample_end,
          in_sample_metrics, out_sample_metrics, degradation, timestamp)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        [
          walkForwardResult.id,
          walkForwardResult.period,
          walkForwardResult.inSamplePeriod.start,
          walkForwardResult.inSamplePeriod.end,
          walkForwardResult.outOfSamplePeriod.start,
          walkForwardResult.outOfSamplePeriod.end,
          JSON.stringify(walkForwardResult.inSampleMetrics),
          JSON.stringify(walkForwardResult.outOfSampleMetrics),
          walkForwardResult.degradation,
          walkForwardResult.timestamp
        ]
      );
      
      this.emit('walk-forward-complete', walkForwardResult);
      console.log(`Walk-forward analysis completed with ${degradation.toFixed(2)} degradation`);
      
    } catch (error) {
      console.error('Walk-forward analysis failed:', error);
    }
  }

  private calculatePerformanceDegradation(inSampleMetrics: any, outOfSampleMetrics: any): number {
    // Calculate performance degradation as percentage change in key metrics
    const sharpeChange = (inSampleMetrics.sharpeRatio - outOfSampleMetrics.sharpeRatio) / 
                        Math.abs(inSampleMetrics.sharpeRatio || 1);
    const winRateChange = (inSampleMetrics.winRate - outOfSampleMetrics.winRate) / 
                         Math.abs(inSampleMetrics.winRate || 1);
    const profitChange = (inSampleMetrics.netPnL - outOfSampleMetrics.netPnL) / 
                        Math.abs(inSampleMetrics.netPnL || 1);
    
    // Average degradation (higher means worse out-of-sample performance)
    return (sharpeChange + winRateChange + profitChange) / 3;
  }

  async compareBacktestToLive(backtestId: string, livePeriod: { start: number; end: number }): Promise<ComparisonResult> {
    // Get backtest results
    const backtestRow = await this.db.get(
      `SELECT * FROM backtests WHERE id = ?`,
      [backtestId]
    );
    
    if (!backtestRow) {
      throw new Error(`Backtest ${backtestId} not found`);
    }
    
    const backtestMetrics = JSON.parse(backtestRow.metrics);
    
    // Calculate live performance for the same period
    const livePerformance = await this.calculateLivePerformance(livePeriod);
    
    // Calculate deviations
    const deviation = {
      pnlDeviation: Math.abs(livePerformance.actualPnL - backtestMetrics.netPnL) / 
                   Math.abs(backtestMetrics.netPnL || 1),
      winRateDeviation: Math.abs(livePerformance.actualWinRate - backtestMetrics.winRate),
      sharpeDeviation: Math.abs(livePerformance.actualSharpe - backtestMetrics.sharpeRatio),
      drawdownDeviation: Math.abs(livePerformance.actualMaxDrawdown - backtestMetrics.maxDrawdown),
      overallDeviation: 0
    };
    
    deviation.overallDeviation = (deviation.pnlDeviation + deviation.winRateDeviation + 
                                 deviation.sharpeDeviation + deviation.drawdownDeviation) / 4;
    
    // Calculate reliability score (inverse of deviation)
    const reliability = Math.max(0, 1 - deviation.overallDeviation);
    
    const comparison: ComparisonResult = {
      backtestId,
      livePerformance,
      backtestPrediction: {
        expectedPnL: backtestMetrics.netPnL,
        expectedWinRate: backtestMetrics.winRate,
        expectedSharpe: backtestMetrics.sharpeRatio,
        expectedMaxDrawdown: backtestMetrics.maxDrawdown
      },
      deviation,
      reliability
    };
    
    // Store comparison result
    await this.db.run(
      `INSERT INTO backtest_comparisons 
       (id, backtest_id, live_performance, backtest_prediction, deviation, reliability, timestamp)
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [
        `comp_${Date.now()}`,
        backtestId,
        JSON.stringify(livePerformance),
        JSON.stringify(comparison.backtestPrediction),
        JSON.stringify(deviation),
        reliability,
        Date.now()
      ]
    );
    
    return comparison;
  }

  private async calculateLivePerformance(period: { start: number; end: number }): Promise<any> {
    // Get all validated predictions in the period
    const rows = await this.db.all(
      `SELECT * FROM prediction_validations 
       WHERE validation_complete = TRUE 
       AND timestamp BETWEEN ? AND ?`,
      [period.start, period.end]
    );
    
    if (rows.length === 0) {
      return {
        period,
        actualPnL: 0,
        actualWinRate: 0,
        actualSharpe: 0,
        actualMaxDrawdown: 0
      };
    }
    
    let totalPnL = 0;
    let wins = 0;
    const returns: number[] = [];
    
    for (const row of rows) {
      const prediction = JSON.parse(row.prediction_data);
      const actualMovement = JSON.parse(row.actual_market_movement || '{}');
      const metrics = JSON.parse(row.accuracy_metrics);
      
      // Estimate PnL based on accuracy and predicted profit
      const predictedProfit = prediction.opportunities[0]?.expectedProfit || 0;
      const actualPnL = predictedProfit * metrics.overallScore;
      
      totalPnL += actualPnL;
      if (actualPnL > 0) wins++;
      
      returns.push(actualPnL / 10000); // Normalize to portfolio percentage
    }
    
    const winRate = wins / rows.length;
    
    // Calculate Sharpe ratio
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const returnStd = Math.sqrt(
      returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
    );
    const sharpeRatio = returnStd > 0 ? avgReturn / returnStd : 0;
    
    // Calculate max drawdown (simplified)
    let equity = 10000;
    let peak = equity;
    let maxDrawdown = 0;
    
    for (const returnPct of returns) {
      equity += returnPct * 10000;
      if (equity > peak) peak = equity;
      const drawdown = (peak - equity) / peak;
      if (drawdown > maxDrawdown) maxDrawdown = drawdown;
    }
    
    return {
      period,
      actualPnL: totalPnL,
      actualWinRate: winRate,
      actualSharpe: sharpeRatio,
      actualMaxDrawdown: maxDrawdown
    };
  }

  // Public methods for dashboard integration
  getLiveMetrics(): LiveMetrics {
    return this.liveMetrics;
  }

  getActivePredictions(): PredictionValidation[] {
    return Array.from(this.activePredictions.values());
  }

  async getValidationHistory(limit: number = 100): Promise<PredictionValidation[]> {
    const rows = await this.db.all(
      `SELECT * FROM prediction_validations 
       WHERE validation_complete = TRUE 
       ORDER BY timestamp DESC LIMIT ?`,
      [limit]
    );
    
    return rows.map(row => ({
      predictionId: row.prediction_id,
      prediction: JSON.parse(row.prediction_data),
      actualMarketMovement: JSON.parse(row.actual_market_movement || '{}'),
      accuracyMetrics: JSON.parse(row.accuracy_metrics || '{}'),
      timestamp: row.timestamp
    }));
  }

  async getWalkForwardResults(limit: number = 10): Promise<WalkForwardResult[]> {
    const rows = await this.db.all(
      `SELECT * FROM walk_forward_results 
       ORDER BY timestamp DESC LIMIT ?`,
      [limit]
    );
    
    return rows.map(row => ({
      id: row.id,
      period: row.period,
      inSamplePeriod: { start: row.in_sample_start, end: row.in_sample_end },
      outOfSamplePeriod: { start: row.out_sample_start, end: row.out_sample_end },
      inSampleMetrics: JSON.parse(row.in_sample_metrics),
      outOfSampleMetrics: JSON.parse(row.out_sample_metrics),
      degradation: row.degradation,
      timestamp: row.timestamp
    }));
  }

  async getBacktestComparisons(limit: number = 10): Promise<ComparisonResult[]> {
    const rows = await this.db.all(
      `SELECT * FROM backtest_comparisons 
       ORDER BY timestamp DESC LIMIT ?`,
      [limit]
    );
    
    return rows.map(row => ({
      backtestId: row.backtest_id,
      livePerformance: JSON.parse(row.live_performance),
      backtestPrediction: JSON.parse(row.backtest_prediction),
      deviation: JSON.parse(row.deviation),
      reliability: row.reliability
    }));
  }

  async stop(): Promise<void> {
    this.isRunning = false;
    await this.backtestEngine.stop();
    await this.kafka.disconnect();
    await this.redis.disconnect();
    await this.db.close();
  }
}