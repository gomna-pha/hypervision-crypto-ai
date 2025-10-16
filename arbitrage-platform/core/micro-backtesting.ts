import EventEmitter from 'events';
import { ArbitragePrediction } from './fusion-brain';
import { ExecutionDecision } from './decision-engine';

export interface BacktestResult {
  backtest_id: string;
  prediction_id: string;
  decision_id?: string;
  timestamp: string;
  test_duration_ms: number;
  
  // Prediction accuracy metrics
  prediction_accuracy: {
    spread_prediction_error_pct: number;        // Actual vs predicted spread
    direction_correct: boolean;                 // Did spread converge/diverge as predicted
    timing_accuracy_pct: number;                // How close was the timing
    confidence_calibration: number;             // How well calibrated was confidence
  };
  
  // Simulated execution results
  execution_simulation: {
    would_have_executed: boolean;
    simulated_pnl_usd: number;                 // P&L if executed
    simulated_pnl_pct: number;                 // P&L as % of notional
    slippage_impact_bps: number;               // Estimated slippage cost
    execution_time_actual_s: number;           // How long execution would take
    market_impact_bps: number;                 // Estimated market impact
  };
  
  // Market condition analysis
  market_conditions: {
    volatility_during_window: number;           // Market volatility during prediction window
    volume_profile: 'low' | 'medium' | 'high'; // Trading volume during window
    spread_stability: number;                   // How stable was the spread
    external_events: string[];                  // News/events during window
  };
  
  // Performance attribution
  performance_attribution: {
    alpha_component_bps: number;                // Pure arbitrage alpha
    timing_component_bps: number;               // Timing skill component
    execution_component_bps: number;            // Execution efficiency component
    risk_component_bps: number;                 // Risk management component
  };
  
  // Learning feedback
  learning_feedback: {
    prediction_quality_score: number;          // 0-100 overall quality
    model_improvement_suggestions: string[];   // Specific improvement areas
    feature_importance_insights: Record<string, number>; // Which features helped/hurt
  };
  
  raw_data: {
    prediction_snapshot: ArbitragePrediction;
    market_data_history: any[];
    decision_snapshot?: ExecutionDecision;
  };
}

export interface MicroBacktesterConfig {
  backtest_window_minutes: number;
  sampling_interval_ms: number;
  slippage_model: 'linear' | 'sqrt' | 'exponential';
  market_impact_model: 'temporary' | 'permanent' | 'hybrid';
  execution_delay_ms: number;
  historical_data_retention_hours: number;
}

/**
 * MicroBacktester - Runs immediate backtests after each LLM prediction
 * 
 * This system provides instant feedback on prediction quality by:
 * 1. Capturing market state at prediction time
 * 2. Monitoring actual market movements during predicted timeframe  
 * 3. Simulating execution with realistic costs and slippage
 * 4. Calculating what P&L would have been achieved
 * 5. Providing learning feedback for model improvement
 * 
 * The backtester operates in real-time, starting immediately when a prediction
 * is made and completing when the predicted time window expires.
 */
export class MicroBacktester extends EventEmitter {
  private config: MicroBacktesterConfig;
  private activeBacktests: Map<string, {
    prediction: ArbitragePrediction;
    decision?: ExecutionDecision;
    startTime: number;
    marketDataHistory: any[];
    intervalId: NodeJS.Timeout;
  }> = new Map();
  
  private completedBacktests: BacktestResult[] = [];
  private marketDataCache: Map<string, any[]> = new Map();
  
  // Visible Parameters for Investors
  public readonly parameters = {
    backtest_window_extend_pct: 0.2,     // Extend window 20% beyond prediction
    market_data_frequency_ms: 1000,      // Sample market data every 1 second
    execution_simulation_enabled: true,   // Simulate actual execution
    slippage_base_bps: 5,                // Base slippage cost (5 bps)
    market_impact_threshold: 0.01,       // 1% of average volume threshold
    confidence_calibration_window: 100,  // Use last 100 predictions for calibration
    learning_feedback_enabled: true,     // Generate model improvement feedback
    performance_attribution_enabled: true, // Break down P&L sources
  };

  // Visible Constraints for Investors  
  public readonly constraints = {
    max_concurrent_backtests: 10,        // Maximum simultaneous backtests
    max_backtest_duration_minutes: 60,   // Hard cap on backtest duration
    min_market_data_points: 10,          // Minimum data points for valid backtest
    max_memory_usage_mb: 500,            // Memory limit for market data storage
    sampling_rate_limit_ms: 500,         // Minimum sampling interval
  };

  // Visible Bounds for Investors
  public readonly bounds = {
    pnl_calculation_range: { min: -0.1, max: 0.1 },      // ±10% max P&L
    slippage_range_bps: { min: 1, max: 100 },            // 1-100 bps slippage
    market_impact_range_bps: { min: 0, max: 50 },        // 0-50 bps market impact
    execution_time_range_s: { min: 1, max: 3600 },       // 1 sec to 1 hour
    volatility_range: { min: 0.0001, max: 0.05 },        // 0.01% to 5% volatility
  };

  constructor(config?: Partial<MicroBacktesterConfig>) {
    super();
    
    this.config = {
      backtest_window_minutes: config?.backtest_window_minutes || 30,
      sampling_interval_ms: config?.sampling_interval_ms || 1000,
      slippage_model: config?.slippage_model || 'sqrt',
      market_impact_model: config?.market_impact_model || 'hybrid',
      execution_delay_ms: config?.execution_delay_ms || 2000,
      historical_data_retention_hours: config?.historical_data_retention_hours || 24
    };
    
    console.log('✅ MicroBacktester initialized with real-time backtesting');
    
    // Cleanup completed backtests periodically
    setInterval(() => this.cleanupOldBacktests(), 60000); // Every minute
  }

  /**
   * Start immediate backtest for a new prediction
   * This is called right after the LLM makes a prediction
   */
  async startBacktest(
    prediction: ArbitragePrediction, 
    decision?: ExecutionDecision
  ): Promise<string> {
    
    const backtestId = `bt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    try {
      // Check capacity constraints
      if (this.activeBacktests.size >= this.constraints.max_concurrent_backtests) {
        throw new Error('Maximum concurrent backtests reached');
      }
      
      console.log(`🧪 Starting micro-backtest: ${backtestId} for prediction: ${prediction.id}`);
      
      // Calculate backtest duration with buffer
      const baseDuration = prediction.expected_time_s * 1000;
      const extendedDuration = baseDuration * (1 + this.parameters.backtest_window_extend_pct);
      const backtestDurationMs = Math.min(
        extendedDuration,
        this.constraints.max_backtest_duration_minutes * 60 * 1000
      );
      
      // Capture initial market state
      const initialMarketData = await this.captureMarketSnapshot(prediction);
      
      // Setup market data monitoring
      const marketDataHistory: any[] = [initialMarketData];
      
      const intervalId = setInterval(async () => {
        try {
          const marketData = await this.captureMarketSnapshot(prediction);
          marketDataHistory.push(marketData);
          
          // Limit memory usage
          if (marketDataHistory.length > 1000) {
            marketDataHistory.splice(0, 100); // Remove oldest 100 entries
          }
        } catch (error) {
          console.error('❌ Market data capture error:', error);
        }
      }, Math.max(this.config.sampling_interval_ms, this.constraints.sampling_rate_limit_ms));
      
      // Store active backtest
      this.activeBacktests.set(backtestId, {
        prediction,
        decision,
        startTime: Date.now(),
        marketDataHistory,
        intervalId
      });
      
      // Schedule backtest completion
      setTimeout(() => {
        this.completeBacktest(backtestId);
      }, backtestDurationMs);
      
      this.emit('backtest_started', { backtestId, predictionId: prediction.id });
      
      return backtestId;
      
    } catch (error) {
      console.error('❌ Failed to start backtest:', error);
      throw error;
    }
  }

  /**
   * Complete and analyze backtest results
   */
  private async completeBacktest(backtestId: string): Promise<void> {
    const activeBacktest = this.activeBacktests.get(backtestId);
    if (!activeBacktest) {
      console.warn(`⚠️ Active backtest not found: ${backtestId}`);
      return;
    }
    
    try {
      console.log(`📊 Completing micro-backtest: ${backtestId}`);
      
      // Stop market data collection
      clearInterval(activeBacktest.intervalId);
      
      const testDurationMs = Date.now() - activeBacktest.startTime;
      
      // Validate minimum data requirements
      if (activeBacktest.marketDataHistory.length < this.constraints.min_market_data_points) {
        console.warn(`⚠️ Insufficient market data for backtest: ${activeBacktest.marketDataHistory.length} points`);
        this.activeBacktests.delete(backtestId);
        return;
      }
      
      // Analyze prediction accuracy
      const predictionAccuracy = this.analyzePredictionAccuracy(
        activeBacktest.prediction,
        activeBacktest.marketDataHistory
      );
      
      // Simulate execution
      const executionSimulation = this.simulateExecution(
        activeBacktest.prediction,
        activeBacktest.marketDataHistory
      );
      
      // Analyze market conditions
      const marketConditions = this.analyzeMarketConditions(
        activeBacktest.marketDataHistory
      );
      
      // Attribute performance
      const performanceAttribution = this.attributePerformance(
        predictionAccuracy,
        executionSimulation,
        marketConditions
      );
      
      // Generate learning feedback
      const learningFeedback = this.generateLearningFeedback(
        activeBacktest.prediction,
        predictionAccuracy,
        executionSimulation,
        marketConditions
      );
      
      // Create backtest result
      const result: BacktestResult = {
        backtest_id: backtestId,
        prediction_id: activeBacktest.prediction.id,
        decision_id: activeBacktest.decision?.decision_id,
        timestamp: new Date().toISOString(),
        test_duration_ms: testDurationMs,
        prediction_accuracy: predictionAccuracy,
        execution_simulation: executionSimulation,
        market_conditions: marketConditions,
        performance_attribution: performanceAttribution,
        learning_feedback: learningFeedback,
        raw_data: {
          prediction_snapshot: activeBacktest.prediction,
          market_data_history: activeBacktest.marketDataHistory,
          decision_snapshot: activeBacktest.decision
        }
      };
      
      // Store result
      this.completedBacktests.push(result);
      
      // Clean up active backtest
      this.activeBacktests.delete(backtestId);
      
      // Emit completion event
      this.emit('backtest_completed', result);
      
      console.log(`✅ Backtest completed: ${backtestId} | P&L: ${executionSimulation.simulated_pnl_pct.toFixed(3)}% | Accuracy: ${(predictionAccuracy.confidence_calibration * 100).toFixed(1)}%`);
      
    } catch (error) {
      console.error('❌ Failed to complete backtest:', error);
      
      // Clean up on error
      clearInterval(activeBacktest.intervalId);
      this.activeBacktests.delete(backtestId);
      
      this.emit('backtest_error', { backtestId, error: (error as Error).message });
    }
  }

  private async captureMarketSnapshot(prediction: ArbitragePrediction): Promise<any> {
    // In a real implementation, this would fetch live market data
    // For now, generate realistic simulated market data
    
    const timestamp = new Date().toISOString();
    const basePrice = 50000 + Math.sin(Date.now() / 60000) * 1000; // Oscillating around $50k
    
    // Simulate bid-ask spreads for different exchanges
    const exchanges = [prediction.arbitrage_plan.buy, prediction.arbitrage_plan.sell];
    const marketData: any = {
      timestamp,
      exchanges: {}
    };
    
    for (const exchange of exchanges) {
      const spread_bps = 5 + Math.random() * 15; // 5-20 bps spread
      const mid_price = basePrice + (Math.random() - 0.5) * 100; // ±$50 price difference
      
      marketData.exchanges[exchange] = {
        bid: mid_price - (spread_bps / 10000) * mid_price / 2,
        ask: mid_price + (spread_bps / 10000) * mid_price / 2,
        mid: mid_price,
        volume_24h: 1000000 + Math.random() * 5000000, // $1M-$6M daily volume
        last_trade_size: 1000 + Math.random() * 50000,
        orderbook_depth_usd: 100000 + Math.random() * 1000000
      };
    }
    
    // Calculate cross-exchange metrics
    const exchange1 = marketData.exchanges[prediction.arbitrage_plan.buy];
    const exchange2 = marketData.exchanges[prediction.arbitrage_plan.sell];
    
    marketData.cross_exchange = {
      price_difference_bps: Math.abs(exchange1.mid - exchange2.mid) / exchange1.mid * 10000,
      spread_difference_bps: Math.abs(
        (exchange1.ask - exchange1.bid) - (exchange2.ask - exchange2.bid)
      ) / exchange1.mid * 10000,
      volume_imbalance: Math.abs(exchange1.volume_24h - exchange2.volume_24h) / 
                       Math.max(exchange1.volume_24h, exchange2.volume_24h)
    };
    
    return marketData;
  }

  private analyzePredictionAccuracy(
    prediction: ArbitragePrediction,
    marketHistory: any[]
  ): any {
    
    const initialData = marketHistory[0];
    const finalData = marketHistory[marketHistory.length - 1];
    
    // Calculate actual spread movement
    const initialSpread = initialData.cross_exchange.price_difference_bps / 10000;
    const finalSpread = finalData.cross_exchange.price_difference_bps / 10000;
    
    const actualSpreadChange = finalSpread - initialSpread;
    const predictedSpreadChange = prediction.direction === 'converge' 
      ? -Math.abs(prediction.predicted_spread_pct)
      : Math.abs(prediction.predicted_spread_pct);
    
    // Spread prediction error
    const spreadPredictionError = Math.abs(actualSpreadChange - predictedSpreadChange) / 
                                 Math.abs(predictedSpreadChange) * 100;
    
    // Direction accuracy
    const directionCorrect = (
      (prediction.direction === 'converge' && actualSpreadChange < 0) ||
      (prediction.direction === 'diverge' && actualSpreadChange > 0)
    );
    
    // Timing accuracy (how much of the movement happened in predicted timeframe)
    const timingAccuracy = this.calculateTimingAccuracy(prediction, marketHistory);
    
    // Confidence calibration (how well calibrated was the confidence score)
    const confidenceCalibration = this.calculateConfidenceCalibration(
      prediction.confidence,
      directionCorrect,
      spreadPredictionError
    );
    
    return {
      spread_prediction_error_pct: Math.min(100, spreadPredictionError),
      direction_correct: directionCorrect,
      timing_accuracy_pct: timingAccuracy,
      confidence_calibration: confidenceCalibration
    };
  }

  private calculateTimingAccuracy(
    prediction: ArbitragePrediction,
    marketHistory: any[]
  ): number {
    
    if (marketHistory.length < 2) return 0;
    
    // Calculate when most of the spread movement occurred
    const expectedDurationMs = prediction.expected_time_s * 1000;
    const actualDurationMs = new Date(marketHistory[marketHistory.length - 1].timestamp).getTime() - 
                           new Date(marketHistory[0].timestamp).getTime();
    
    const timingError = Math.abs(actualDurationMs - expectedDurationMs) / expectedDurationMs;
    
    return Math.max(0, 100 * (1 - timingError));
  }

  private calculateConfidenceCalibration(
    predictedConfidence: number,
    wasCorrect: boolean,
    errorPct: number
  ): number {
    
    // Simple calibration metric: confidence should match actual accuracy
    // Perfect calibration = 1.0, random = 0.5, anti-calibrated = 0.0
    
    if (wasCorrect) {
      // If prediction was correct, confidence calibration is good if confidence was high
      return predictedConfidence;
    } else {
      // If prediction was wrong, confidence calibration is good if confidence was low
      return 1 - predictedConfidence;
    }
  }

  private simulateExecution(
    prediction: ArbitragePrediction,
    marketHistory: any[]
  ): any {
    
    const initialData = marketHistory[0];
    const executionData = marketHistory[Math.min(5, marketHistory.length - 1)]; // Simulate 5-second execution delay
    
    const notionalUsd = prediction.arbitrage_plan.notional_usd;
    
    // Calculate execution prices with slippage
    const buyExchange = prediction.arbitrage_plan.buy;
    const sellExchange = prediction.arbitrage_plan.sell;
    
    const buyPrice = executionData.exchanges[buyExchange].ask;
    const sellPrice = executionData.exchanges[sellExchange].bid;
    
    // Calculate slippage based on position size and market depth
    const slippageBps = this.calculateSlippage(
      notionalUsd,
      executionData.exchanges[buyExchange].orderbook_depth_usd,
      executionData.exchanges[sellExchange].orderbook_depth_usd
    );
    
    const slippageAdjustedBuyPrice = buyPrice * (1 + slippageBps / 10000);
    const slippageAdjustedSellPrice = sellPrice * (1 - slippageBps / 10000);
    
    // Calculate P&L
    const grossPnlUsd = (sellPrice - buyPrice) * (notionalUsd / buyPrice);
    const slippageCostUsd = notionalUsd * slippageBps / 10000;
    const netPnlUsd = grossPnlUsd - slippageCostUsd;
    const pnlPct = netPnlUsd / notionalUsd;
    
    // Calculate market impact
    const marketImpactBps = this.calculateMarketImpact(notionalUsd, executionData);
    
    // Determine if execution would have actually happened
    const wouldHaveExecuted = this.determineExecutionFeasibility(
      prediction,
      executionData,
      slippageBps,
      marketImpactBps
    );
    
    return {
      would_have_executed: wouldHaveExecuted,
      simulated_pnl_usd: this.clampToRange(netPnlUsd, { min: -notionalUsd * 0.1, max: notionalUsd * 0.1 }),
      simulated_pnl_pct: this.clampToRange(pnlPct, this.bounds.pnl_calculation_range),
      slippage_impact_bps: this.clampToRange(slippageBps, this.bounds.slippage_range_bps),
      execution_time_actual_s: this.config.execution_delay_ms / 1000,
      market_impact_bps: this.clampToRange(marketImpactBps, this.bounds.market_impact_range_bps)
    };
  }

  private calculateSlippage(
    notionalUsd: number,
    buyDepthUsd: number,
    sellDepthUsd: number
  ): number {
    
    const avgDepth = (buyDepthUsd + sellDepthUsd) / 2;
    const sizeRatio = notionalUsd / avgDepth;
    
    // Different slippage models
    switch (this.config.slippage_model) {
      case 'linear':
        return this.parameters.slippage_base_bps + sizeRatio * 100;
        
      case 'sqrt':
        return this.parameters.slippage_base_bps + Math.sqrt(sizeRatio) * 20;
        
      case 'exponential':
        return this.parameters.slippage_base_bps * Math.exp(sizeRatio * 2);
        
      default:
        return this.parameters.slippage_base_bps;
    }
  }

  private calculateMarketImpact(notionalUsd: number, marketData: any): number {
    // Simplified market impact calculation
    const avgVolume = Object.values(marketData.exchanges).reduce((sum: number, ex: any) => 
      sum + ex.volume_24h, 0) / Object.keys(marketData.exchanges).length;
    
    const impactRatio = notionalUsd / (avgVolume * 0.01); // Compare to 1% of daily volume
    
    return Math.min(50, impactRatio * 10); // Cap at 50 bps
  }

  private determineExecutionFeasibility(
    prediction: ArbitragePrediction,
    marketData: any,
    slippageBps: number,
    marketImpactBps: number
  ): boolean {
    
    // Would execution have been feasible given market conditions?
    
    // Check if slippage doesn't eat all profits
    const totalCostBps = slippageBps + marketImpactBps;
    const expectedProfitBps = prediction.predicted_spread_pct * 10000;
    
    if (totalCostBps >= expectedProfitBps * 0.8) { // Costs > 80% of expected profit
      return false;
    }
    
    // Check liquidity availability
    const minDepth = Math.min(
      ...Object.values(marketData.exchanges).map((ex: any) => ex.orderbook_depth_usd)
    );
    
    if (prediction.arbitrage_plan.notional_usd > minDepth * 0.1) { // Position > 10% of depth
      return false;
    }
    
    return true;
  }

  private analyzeMarketConditions(marketHistory: any[]): any {
    if (marketHistory.length < 2) {
      return {
        volatility_during_window: 0,
        volume_profile: 'low',
        spread_stability: 0,
        external_events: []
      };
    }
    
    // Calculate price volatility during window
    const prices = marketHistory.map(data => {
      const exchanges = Object.values(data.exchanges);
      return exchanges.reduce((sum: number, ex: any) => sum + ex.mid, 0) / exchanges.length;
    });
    
    const returns = prices.slice(1).map((price, i) => 
      Math.log(price / prices[i])
    );
    
    const volatility = Math.sqrt(
      returns.reduce((sum, ret) => sum + ret * ret, 0) / returns.length
    ) * Math.sqrt(252 * 24 * 60 * 60); // Annualized volatility
    
    // Analyze volume profile
    const volumes = marketHistory.map(data => {
      const exchanges = Object.values(data.exchanges);
      return exchanges.reduce((sum: number, ex: any) => sum + ex.volume_24h, 0) / exchanges.length;
    });
    
    const avgVolume = volumes.reduce((sum, vol) => sum + vol, 0) / volumes.length;
    const volumeProfile = avgVolume < 2000000 ? 'low' : avgVolume < 10000000 ? 'medium' : 'high';
    
    // Analyze spread stability
    const spreads = marketHistory.map(data => data.cross_exchange.price_difference_bps);
    const spreadStd = Math.sqrt(
      spreads.reduce((sum, spread) => {
        const mean = spreads.reduce((s, sp) => s + sp, 0) / spreads.length;
        return sum + Math.pow(spread - mean, 2);
      }, 0) / spreads.length
    );
    const spreadStability = Math.max(0, 100 - spreadStd); // Higher std = lower stability
    
    return {
      volatility_during_window: this.clampToRange(volatility, this.bounds.volatility_range),
      volume_profile: volumeProfile as 'low' | 'medium' | 'high',
      spread_stability: Math.max(0, Math.min(100, spreadStability)),
      external_events: [] // Would be populated with actual event detection
    };
  }

  private attributePerformance(
    predictionAccuracy: any,
    executionSimulation: any,
    marketConditions: any
  ): any {
    
    const totalPnlBps = executionSimulation.simulated_pnl_pct * 10000;
    
    // Attribute P&L to different sources
    const alphaComponent = totalPnlBps * (predictionAccuracy.direction_correct ? 0.6 : 0);
    const timingComponent = totalPnlBps * (predictionAccuracy.timing_accuracy_pct / 100) * 0.2;
    const executionComponent = -executionSimulation.slippage_impact_bps; // Execution costs
    const riskComponent = totalPnlBps * 0.1; // Risk management contribution
    
    return {
      alpha_component_bps: alphaComponent,
      timing_component_bps: timingComponent,
      execution_component_bps: executionComponent,
      risk_component_bps: riskComponent
    };
  }

  private generateLearningFeedback(
    prediction: ArbitragePrediction,
    predictionAccuracy: any,
    executionSimulation: any,
    marketConditions: any
  ): any {
    
    const suggestions: string[] = [];
    const featureImportance: Record<string, number> = {};
    
    // Generate specific improvement suggestions based on results
    if (predictionAccuracy.spread_prediction_error_pct > 50) {
      suggestions.push('Improve spread magnitude prediction accuracy');
    }
    
    if (!predictionAccuracy.direction_correct) {
      suggestions.push('Enhance directional prediction model');
    }
    
    if (predictionAccuracy.timing_accuracy_pct < 70) {
      suggestions.push('Refine timing prediction algorithms');
    }
    
    if (executionSimulation.slippage_impact_bps > 20) {
      suggestions.push('Account for higher slippage in illiquid conditions');
    }
    
    if (marketConditions.volatility_during_window > 0.02) {
      suggestions.push('Improve volatility regime detection');
    }
    
    // Analyze feature importance from prediction inputs
    if (prediction.fusion_input) {
      // This would analyze which agent signals contributed to accuracy
      featureImportance['economic_signal'] = predictionAccuracy.direction_correct ? 0.3 : -0.2;
      featureImportance['sentiment_signal'] = predictionAccuracy.timing_accuracy_pct > 80 ? 0.4 : -0.1;
      featureImportance['microstructure_signal'] = executionSimulation.would_have_executed ? 0.5 : -0.3;
    }
    
    // Calculate overall prediction quality score
    const qualityScore = (
      (predictionAccuracy.direction_correct ? 30 : 0) +
      Math.max(0, 30 - predictionAccuracy.spread_prediction_error_pct / 2) +
      (predictionAccuracy.timing_accuracy_pct * 0.2) +
      (predictionAccuracy.confidence_calibration * 20)
    );
    
    return {
      prediction_quality_score: Math.max(0, Math.min(100, qualityScore)),
      model_improvement_suggestions: suggestions,
      feature_importance_insights: featureImportance
    };
  }

  private clampToRange(value: number, range: { min: number; max: number }): number {
    return Math.max(range.min, Math.min(range.max, value));
  }

  private cleanupOldBacktests(): void {
    const cutoff = Date.now() - (this.config.historical_data_retention_hours * 60 * 60 * 1000);
    
    this.completedBacktests = this.completedBacktests.filter(result => {
      return new Date(result.timestamp).getTime() > cutoff;
    });
    
    // Also cleanup market data cache
    for (const [key, data] of this.marketDataCache.entries()) {
      const filteredData = data.filter((entry: any) => {
        return new Date(entry.timestamp).getTime() > cutoff;
      });
      
      if (filteredData.length === 0) {
        this.marketDataCache.delete(key);
      } else {
        this.marketDataCache.set(key, filteredData);
      }
    }
  }

  // Public methods for transparency
  getVisibleParameters(): any {
    return {
      parameters: this.parameters,
      constraints: this.constraints,
      bounds: this.bounds,
      config: this.config,
      active_backtests: this.activeBacktests.size,
      completed_backtests: this.completedBacktests.length,
      memory_usage_estimate_mb: this.estimateMemoryUsage()
    };
  }

  getActiveBacktests(): Array<{id: string, predictionId: string, startTime: string, duration_s: number}> {
    const now = Date.now();
    return Array.from(this.activeBacktests.entries()).map(([id, backtest]) => ({
      id,
      predictionId: backtest.prediction.id,
      startTime: new Date(backtest.startTime).toISOString(),
      duration_s: (now - backtest.startTime) / 1000
    }));
  }

  getCompletedBacktests(limit: number = 50): BacktestResult[] {
    return this.completedBacktests
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit);
  }

  getBacktestPerformanceMetrics(): any {
    if (this.completedBacktests.length === 0) {
      return { status: 'no_data' };
    }
    
    const recentResults = this.completedBacktests.slice(-100); // Last 100 backtests
    
    const avgPnlPct = recentResults.reduce((sum, result) => 
      sum + result.execution_simulation.simulated_pnl_pct, 0) / recentResults.length;
    
    const winRate = recentResults.filter(result => 
      result.execution_simulation.simulated_pnl_pct > 0).length / recentResults.length;
    
    const avgConfidenceCalibration = recentResults.reduce((sum, result) =>
      sum + result.prediction_accuracy.confidence_calibration, 0) / recentResults.length;
    
    const directionAccuracy = recentResults.filter(result =>
      result.prediction_accuracy.direction_correct).length / recentResults.length;
    
    return {
      status: 'active',
      sample_size: recentResults.length,
      avg_pnl_pct: avgPnlPct,
      win_rate: winRate,
      direction_accuracy: directionAccuracy,
      confidence_calibration: avgConfidenceCalibration,
      avg_quality_score: recentResults.reduce((sum, r) => 
        sum + r.learning_feedback.prediction_quality_score, 0) / recentResults.length
    };
  }

  private estimateMemoryUsage(): number {
    // Rough estimate of memory usage in MB
    const activeBacktestMemory = this.activeBacktests.size * 1; // ~1MB per active backtest
    const completedBacktestMemory = this.completedBacktests.length * 0.1; // ~0.1MB per completed
    const marketDataMemory = Array.from(this.marketDataCache.values())
      .reduce((sum, data) => sum + data.length * 0.001, 0); // ~1KB per data point
    
    return activeBacktestMemory + completedBacktestMemory + marketDataMemory;
  }

  /**
   * Force complete all active backtests (useful for shutdown)
   */
  async forceCompleteAllBacktests(): Promise<void> {
    const activeIds = Array.from(this.activeBacktests.keys());
    
    console.log(`🏁 Force completing ${activeIds.length} active backtests...`);
    
    for (const backtestId of activeIds) {
      try {
        await this.completeBacktest(backtestId);
      } catch (error) {
        console.error(`❌ Failed to force complete backtest ${backtestId}:`, error);
      }
    }
  }
}