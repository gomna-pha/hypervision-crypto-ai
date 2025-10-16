import EventEmitter from 'events';
import { ArbitragePrediction } from './fusion-brain';
import * as yaml from 'js-yaml';
import * as fs from 'fs';
import * as path from 'path';

export interface ArbitrageOpportunityScore {
  overall_score: number;        // 0 to 100 (AOS)
  component_scores: {
    spread_magnitude: number;   // 0 to 100
    confidence_level: number;   // 0 to 100  
    time_window: number;        // 0 to 100
    liquidity_factor: number;   // 0 to 100
    risk_adjustment: number;    // -50 to 0 (penalty)
  };
  score_weights: {
    spread_magnitude: number;
    confidence_level: number;
    time_window: number; 
    liquidity_factor: number;
    risk_adjustment: number;
  };
  risk_flags_impact: Record<string, number>;
  execution_recommendation: 'execute' | 'monitor' | 'reject';
  rationale: string;
}

export interface ExecutionDecision {
  decision_id: string;
  prediction_id: string;
  timestamp: string;
  aos_score: ArbitrageOpportunityScore;
  execution_action: 'execute' | 'monitor' | 'reject';
  constraints_passed: boolean;
  bounds_compliant: boolean;
  execution_plan?: {
    buy_exchange: string;
    sell_exchange: string;
    position_size_usd: number;
    max_slippage_bps: number;
    timeout_seconds: number;
    stop_loss_pct?: number;
  };
  rejection_reasons?: string[];
  monitoring_conditions?: string[];
}

export interface ConstraintCheckResult {
  passed: boolean;
  failed_constraints: string[];
  warnings: string[];
  details: Record<string, any>;
}

/**
 * Decision Engine - Evaluates LLM predictions and makes execution decisions
 * 
 * Core responsibilities:
 * 1. Calculate Arbitrage Opportunity Score (AOS) with transparent weightings
 * 2. Enforce risk constraints and position bounds 
 * 3. Generate execution decisions with full audit trail
 * 4. Provide investor-visible parameters and reasoning
 * 
 * AOS Scoring System:
 * - Spread Magnitude (25%): Higher spreads = higher opportunity
 * - Confidence Level (30%): LLM prediction confidence 
 * - Time Window (20%): Optimal execution timeframe
 * - Liquidity Factor (15%): Market depth and volume
 * - Risk Adjustment (10%): Penalty for risk flags
 */
export class DecisionEngine extends EventEmitter {
  private config: any;
  private executionHistory: ExecutionDecision[] = [];
  private activeMonitoredPositions: Map<string, ExecutionDecision> = new Map();
  
  // Visible Parameters for Investors
  public readonly parameters = {
    aos_execution_threshold: 75,      // Execute if AOS >= 75
    aos_monitor_threshold: 50,        // Monitor if AOS >= 50  
    max_concurrent_positions: 3,      // Maximum simultaneous arbitrage positions
    position_sizing_method: 'fixed_fraction', // 'fixed_fraction' | 'kelly' | 'var_based'
    default_position_fraction: 0.02,  // 2% of available capital per position
    max_position_size_usd: 50000,     // Hard cap on position size
    slippage_buffer_bps: 10,          // Additional slippage buffer (10 bps)
    execution_timeout_sec: 300,       // 5 minute execution timeout
    stop_loss_enabled: true,
    default_stop_loss_pct: 0.5,       // 0.5% stop loss
  };

  // Visible Constraints for Investors
  public readonly constraints = {
    min_spread_threshold_pct: 0.001,  // 0.1% minimum spread
    max_spread_threshold_pct: 0.02,   // 2% maximum spread (avoid manipulation)
    min_prediction_confidence: 0.6,   // 60% minimum LLM confidence
    min_time_window_sec: 60,          // 1 minute minimum execution window
    max_time_window_sec: 1800,        // 30 minute maximum execution window
    max_notional_usd: 100000,         // $100k maximum notional per trade
    min_notional_usd: 1000,           // $1k minimum notional per trade
    blacklisted_exchanges: [] as string[],        // No exchange blacklist by default
    max_daily_trades: 50,             // Maximum trades per day
    max_risk_flags: 3,                // Reject if more than 3 risk flags
    require_both_exchanges_liquid: true, // Both exchanges must show liquidity
  };

  // Visible Bounds for Investors  
  public readonly bounds = {
    aos_score_range: { min: 0, max: 100 },
    position_size_range: { min: 0.001, max: 0.1 }, // 0.1% to 10% of capital
    slippage_tolerance_range: { min: 5, max: 50 },  // 5 to 50 bps
    execution_time_range: { min: 30, max: 600 },    // 30 sec to 10 min
    correlation_threshold: 0.8,        // Reject highly correlated positions
    volatility_multiplier_max: 3.0,    // Max 3x normal volatility
  };

  // AOS Component Weights (Visible to Investors)
  public readonly aosWeights = {
    spread_magnitude: 0.25,   // 25% - Size of arbitrage opportunity
    confidence_level: 0.30,   // 30% - LLM prediction confidence
    time_window: 0.20,        // 20% - Execution time favorability  
    liquidity_factor: 0.15,   // 15% - Market depth and liquidity
    risk_adjustment: 0.10,    // 10% - Risk flag penalties
  };

  // Risk Flag Impact Scores (Visible to Investors)
  public readonly riskFlagPenalties = {
    'low_liquidity_warning': -15,
    'high_volatility_warning': -10,
    'exchange_connectivity_issue': -25,
    'market_manipulation_detected': -50,
    'unusual_spread_behavior': -20,
    'order_book_anomaly': -15,
    'cross_exchange_latency_high': -10,
    'regulatory_concern': -30,
    'counterparty_risk_elevated': -20
  };

  constructor() {
    super();
    this.loadConfiguration();
    console.log('✅ DecisionEngine initialized with AOS scoring system');
  }

  private loadConfiguration(): void {
    try {
      const configPath = path.join(__dirname, '..', 'config.yaml');
      const configFile = fs.readFileSync(configPath, 'utf8');
      this.config = yaml.load(configFile) as any;
    } catch (error) {
      console.warn('⚠️ Could not load config.yaml, using defaults');
      this.config = { decision_engine: {} };
    }
  }

  /**
   * Main decision evaluation method
   * Takes LLM prediction and returns execution decision with full AOS breakdown
   */
  async evaluateArbitrageOpportunity(prediction: ArbitragePrediction): Promise<ExecutionDecision> {
    const decisionId = `decision_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    try {
      console.log(`🎯 Evaluating arbitrage opportunity: ${prediction.id}`);
      
      // Step 1: Check hard constraints
      const constraintCheck = this.checkConstraints(prediction);
      
      // Step 2: Check bounds compliance  
      const boundsCheck = this.checkBounds(prediction);
      
      // Step 3: Calculate Arbitrage Opportunity Score (AOS)
      const aosScore = this.calculateAOS(prediction);
      
      // Step 4: Make execution decision based on AOS and constraints
      const executionAction = this.determineExecutionAction(
        aosScore, 
        constraintCheck, 
        boundsCheck
      );
      
      // Step 5: Create execution plan if approved
      const executionPlan = executionAction === 'execute' 
        ? this.createExecutionPlan(prediction, aosScore)
        : undefined;
      
      // Step 6: Build decision object
      const decision: ExecutionDecision = {
        decision_id: decisionId,
        prediction_id: prediction.id,
        timestamp: new Date().toISOString(),
        aos_score: aosScore,
        execution_action: executionAction,
        constraints_passed: constraintCheck.passed,
        bounds_compliant: boundsCheck.passed,
        execution_plan: executionPlan,
        rejection_reasons: this.buildRejectionReasons(constraintCheck, boundsCheck, aosScore),
        monitoring_conditions: executionAction === 'monitor' 
          ? this.buildMonitoringConditions(prediction, aosScore)
          : undefined
      };
      
      // Step 7: Store decision and emit event
      this.executionHistory.push(decision);
      
      if (executionAction === 'monitor') {
        this.activeMonitoredPositions.set(prediction.id, decision);
      }
      
      this.emit('decision', decision);
      
      console.log(`📊 Decision: ${executionAction.toUpperCase()} | AOS: ${aosScore.overall_score.toFixed(1)} | ID: ${decisionId}`);
      
      return decision;
      
    } catch (error) {
      console.error('❌ Decision evaluation error:', error);
      
      // Return rejection decision on error
      const errorDecision: ExecutionDecision = {
        decision_id: decisionId,
        prediction_id: prediction.id,
        timestamp: new Date().toISOString(),
        aos_score: this.createDefaultAOS(),
        execution_action: 'reject',
        constraints_passed: false,
        bounds_compliant: false,
        rejection_reasons: [`System error: ${(error as Error).message}`]
      };
      
      this.executionHistory.push(errorDecision);
      this.emit('decision', errorDecision);
      
      return errorDecision;
    }
  }

  /**
   * Calculate Arbitrage Opportunity Score (AOS) with full transparency
   */
  private calculateAOS(prediction: ArbitragePrediction): ArbitrageOpportunityScore {
    // Component 1: Spread Magnitude Score (0-100)
    const spreadMagnitude = this.scoreSpreadMagnitude(prediction.predicted_spread_pct);
    
    // Component 2: Confidence Level Score (0-100)
    const confidenceLevel = this.scoreConfidenceLevel(prediction.confidence);
    
    // Component 3: Time Window Score (0-100)
    const timeWindow = this.scoreTimeWindow(prediction.expected_time_s);
    
    // Component 4: Liquidity Factor Score (0-100) 
    const liquidityFactor = this.scoreLiquidityFactor(prediction);
    
    // Component 5: Risk Adjustment Score (-50 to 0)
    const riskAdjustment = this.scoreRiskAdjustment(prediction.risk_flags);
    
    // Calculate weighted overall score
    const overallScore = Math.max(0, Math.min(100,
      spreadMagnitude * this.aosWeights.spread_magnitude +
      confidenceLevel * this.aosWeights.confidence_level +
      timeWindow * this.aosWeights.time_window +
      liquidityFactor * this.aosWeights.liquidity_factor +
      riskAdjustment * this.aosWeights.risk_adjustment
    ));
    
    // Determine execution recommendation
    let executionRecommendation: 'execute' | 'monitor' | 'reject';
    if (overallScore >= this.parameters.aos_execution_threshold) {
      executionRecommendation = 'execute';
    } else if (overallScore >= this.parameters.aos_monitor_threshold) {
      executionRecommendation = 'monitor';
    } else {
      executionRecommendation = 'reject';
    }
    
    const rationale = this.buildAOSRationale(
      overallScore, 
      { spreadMagnitude, confidenceLevel, timeWindow, liquidityFactor, riskAdjustment },
      prediction.risk_flags
    );
    
    return {
      overall_score: overallScore,
      component_scores: {
        spread_magnitude: spreadMagnitude,
        confidence_level: confidenceLevel,
        time_window: timeWindow,
        liquidity_factor: liquidityFactor,
        risk_adjustment: riskAdjustment
      },
      score_weights: { ...this.aosWeights },
      risk_flags_impact: this.calculateRiskFlagImpacts(prediction.risk_flags),
      execution_recommendation: executionRecommendation,
      rationale: rationale
    };
  }

  private scoreSpreadMagnitude(spreadPct: number): number {
    // Score spread magnitude on a curve - higher spreads get higher scores
    // 0.1% spread = 20 points, 0.5% spread = 70 points, 1%+ spread = 100 points
    
    const spreadBps = spreadPct * 10000; // Convert to basis points
    
    if (spreadBps <= 10) return 20;  // 0.1% = 20/100
    if (spreadBps <= 25) return 35;  // 0.25% = 35/100  
    if (spreadBps <= 50) return 70;  // 0.5% = 70/100
    if (spreadBps <= 75) return 85;  // 0.75% = 85/100
    return 100; // 1%+ = 100/100
  }

  private scoreConfidenceLevel(confidence: number): number {
    // Linear scaling of confidence to 0-100 score
    return Math.max(0, Math.min(100, confidence * 100));
  }

  private scoreTimeWindow(expectedTimeS: number): number {
    // Optimal time window is 3-10 minutes for arbitrage execution
    // Too fast = execution risk, too slow = opportunity decay
    
    if (expectedTimeS < 180) return 60;        // < 3 min = moderate score
    if (expectedTimeS <= 600) return 100;      // 3-10 min = optimal
    if (expectedTimeS <= 900) return 80;       // 10-15 min = good  
    if (expectedTimeS <= 1800) return 50;      // 15-30 min = fair
    return 20;                                 // > 30 min = poor
  }

  private scoreLiquidityFactor(prediction: ArbitragePrediction): number {
    // Estimate liquidity score based on notional size and market conditions
    // This would ideally use real-time order book depth data
    
    const notionalUSD = prediction.arbitrage_plan.notional_usd;
    
    // Penalty for very large positions that might impact market
    if (notionalUSD > 50000) return 60;        // Large size = liquidity concern
    if (notionalUSD > 25000) return 80;        // Medium-large size
    if (notionalUSD >= 5000) return 100;       // Optimal size range
    if (notionalUSD >= 1000) return 90;        // Small but viable
    return 70;                                 // Very small size
  }

  private scoreRiskAdjustment(riskFlags: string[]): number {
    // Apply penalties for each risk flag
    let totalPenalty = 0;
    
    for (const flag of riskFlags) {
      const penalty = this.riskFlagPenalties[flag] || -5; // Default -5 penalty
      totalPenalty += penalty;
    }
    
    // Cap total penalty at -50
    return Math.max(-50, totalPenalty);
  }

  private calculateRiskFlagImpacts(riskFlags: string[]): Record<string, number> {
    const impacts: Record<string, number> = {};
    
    for (const flag of riskFlags) {
      impacts[flag] = this.riskFlagPenalties[flag] || -5;
    }
    
    return impacts;
  }

  private buildAOSRationale(
    overallScore: number,
    componentScores: any,
    riskFlags: string[]
  ): string {
    const parts: string[] = [];
    
    parts.push(`Overall AOS: ${overallScore.toFixed(1)}/100`);
    
    // Highlight strongest components
    const strongest = Object.entries(componentScores)
      .sort(([,a], [,b]) => (b as number) - (a as number))[0];
    parts.push(`Strongest factor: ${strongest[0]} (${(strongest[1] as number).toFixed(1)})`);
    
    // Highlight risk concerns
    if (riskFlags.length > 0) {
      parts.push(`Risk flags: ${riskFlags.length} (${riskFlags.join(', ')})`);
    }
    
    // Add execution guidance
    if (overallScore >= this.parameters.aos_execution_threshold) {
      parts.push('Recommended for immediate execution');
    } else if (overallScore >= this.parameters.aos_monitor_threshold) {
      parts.push('Recommended for monitoring - may improve');
    } else {
      parts.push('Below execution threshold - opportunity insufficient');
    }
    
    return parts.join('; ');
  }

  private checkConstraints(prediction: ArbitragePrediction): ConstraintCheckResult {
    const failed: string[] = [];
    const warnings: string[] = [];
    const details: Record<string, any> = {};
    
    // Check spread thresholds
    if (prediction.predicted_spread_pct < this.constraints.min_spread_threshold_pct) {
      failed.push(`Spread too small: ${(prediction.predicted_spread_pct * 100).toFixed(3)}% < ${(this.constraints.min_spread_threshold_pct * 100).toFixed(3)}%`);
    }
    
    if (prediction.predicted_spread_pct > this.constraints.max_spread_threshold_pct) {
      failed.push(`Spread too large: ${(prediction.predicted_spread_pct * 100).toFixed(3)}% > ${(this.constraints.max_spread_threshold_pct * 100).toFixed(3)}%`);
    }
    
    // Check confidence threshold
    if (prediction.confidence < this.constraints.min_prediction_confidence) {
      failed.push(`Confidence too low: ${(prediction.confidence * 100).toFixed(1)}% < ${(this.constraints.min_prediction_confidence * 100).toFixed(1)}%`);
    }
    
    // Check time window
    if (prediction.expected_time_s < this.constraints.min_time_window_sec) {
      failed.push(`Time window too short: ${prediction.expected_time_s}s < ${this.constraints.min_time_window_sec}s`);
    }
    
    if (prediction.expected_time_s > this.constraints.max_time_window_sec) {
      failed.push(`Time window too long: ${prediction.expected_time_s}s > ${this.constraints.max_time_window_sec}s`);
    }
    
    // Check notional limits
    if (prediction.arbitrage_plan.notional_usd < this.constraints.min_notional_usd) {
      failed.push(`Notional too small: $${prediction.arbitrage_plan.notional_usd} < $${this.constraints.min_notional_usd}`);
    }
    
    if (prediction.arbitrage_plan.notional_usd > this.constraints.max_notional_usd) {
      failed.push(`Notional too large: $${prediction.arbitrage_plan.notional_usd} > $${this.constraints.max_notional_usd}`);
    }
    
    // Check risk flags
    if (prediction.risk_flags.length > this.constraints.max_risk_flags) {
      failed.push(`Too many risk flags: ${prediction.risk_flags.length} > ${this.constraints.max_risk_flags}`);
    }
    
    // Check exchange blacklist
    const buyExchange = prediction.arbitrage_plan.buy;
    const sellExchange = prediction.arbitrage_plan.sell;
    
    if (this.constraints.blacklisted_exchanges.includes(buyExchange)) {
      failed.push(`Buy exchange blacklisted: ${buyExchange}`);
    }
    
    if (this.constraints.blacklisted_exchanges.includes(sellExchange)) {
      failed.push(`Sell exchange blacklisted: ${sellExchange}`);
    }
    
    // Check daily trade limit
    const todayTrades = this.getTodayTradeCount();
    if (todayTrades >= this.constraints.max_daily_trades) {
      failed.push(`Daily trade limit reached: ${todayTrades}/${this.constraints.max_daily_trades}`);
    }
    
    details.today_trade_count = todayTrades;
    details.spread_pct = prediction.predicted_spread_pct;
    details.confidence_pct = prediction.confidence * 100;
    details.time_window_sec = prediction.expected_time_s;
    details.notional_usd = prediction.arbitrage_plan.notional_usd;
    details.risk_flag_count = prediction.risk_flags.length;
    
    return {
      passed: failed.length === 0,
      failed_constraints: failed,
      warnings: warnings,
      details: details
    };
  }

  private checkBounds(prediction: ArbitragePrediction): ConstraintCheckResult {
    const failed: string[] = [];
    const warnings: string[] = [];
    const details: Record<string, any> = {};
    
    // These are typically soft bounds that get enforced via clamping
    // rather than hard rejections, but we can warn about them
    
    const spreadPct = prediction.predicted_spread_pct;
    const confidence = prediction.confidence;
    const timeS = prediction.expected_time_s;
    
    // Check if values are within reasonable bounds for clamping
    if (spreadPct > 0.05) {  // 5%
      warnings.push(`Very high spread: ${(spreadPct * 100).toFixed(2)}% - will be clamped`);
    }
    
    if (confidence < 0.3) {  // 30%
      warnings.push(`Very low confidence: ${(confidence * 100).toFixed(1)}% - concerning`);
    }
    
    if (timeS > 3600) {  // 1 hour
      warnings.push(`Very long time horizon: ${timeS}s - may be stale`);
    }
    
    details.bounds_checked = true;
    details.clamping_required = warnings.length > 0;
    
    return {
      passed: true, // Bounds are soft limits
      failed_constraints: failed,
      warnings: warnings,
      details: details
    };
  }

  private determineExecutionAction(
    aosScore: ArbitrageOpportunityScore,
    constraintCheck: ConstraintCheckResult,
    boundsCheck: ConstraintCheckResult
  ): 'execute' | 'monitor' | 'reject' {
    
    // Hard rejection for constraint failures
    if (!constraintCheck.passed) {
      return 'reject';
    }
    
    // Check concurrent position limit
    if (this.activeMonitoredPositions.size >= this.parameters.max_concurrent_positions) {
      return 'reject';
    }
    
    // Use AOS recommendation if constraints pass
    return aosScore.execution_recommendation;
  }

  private createExecutionPlan(
    prediction: ArbitragePrediction, 
    aosScore: ArbitrageOpportunityScore
  ): any {
    
    const baseSlippage = this.parameters.slippage_buffer_bps;
    const executionTimeout = Math.min(
      prediction.expected_time_s,
      this.parameters.execution_timeout_sec
    );
    
    // Adjust position size based on AOS score and risk
    let positionSizeUsd = prediction.arbitrage_plan.notional_usd;
    
    // Scale down position size for lower AOS scores
    if (aosScore.overall_score < 90) {
      const scaleFactor = aosScore.overall_score / 100;
      positionSizeUsd *= scaleFactor;
    }
    
    // Apply maximum position size limit
    positionSizeUsd = Math.min(positionSizeUsd, this.parameters.max_position_size_usd);
    
    return {
      buy_exchange: prediction.arbitrage_plan.buy,
      sell_exchange: prediction.arbitrage_plan.sell,  
      position_size_usd: Math.round(positionSizeUsd),
      max_slippage_bps: baseSlippage,
      timeout_seconds: executionTimeout,
      stop_loss_pct: this.parameters.stop_loss_enabled ? this.parameters.default_stop_loss_pct : undefined
    };
  }

  private buildRejectionReasons(
    constraintCheck: ConstraintCheckResult,
    boundsCheck: ConstraintCheckResult,
    aosScore: ArbitrageOpportunityScore
  ): string[] | undefined {
    
    const reasons: string[] = [];
    
    // Add constraint failures
    reasons.push(...constraintCheck.failed_constraints);
    
    // Add AOS-based rejection
    if (aosScore.overall_score < this.parameters.aos_monitor_threshold) {
      reasons.push(`AOS score too low: ${aosScore.overall_score.toFixed(1)} < ${this.parameters.aos_monitor_threshold}`);
    }
    
    // Add capacity concerns
    if (this.activeMonitoredPositions.size >= this.parameters.max_concurrent_positions) {
      reasons.push(`Maximum concurrent positions reached: ${this.activeMonitoredPositions.size}`);
    }
    
    return reasons.length > 0 ? reasons : undefined;
  }

  private buildMonitoringConditions(
    prediction: ArbitragePrediction,
    aosScore: ArbitrageOpportunityScore
  ): string[] {
    
    const conditions: string[] = [];
    
    // Conditions for moving from monitor to execute
    if (aosScore.component_scores.confidence_level < 80) {
      conditions.push(`Wait for higher confidence (current: ${aosScore.component_scores.confidence_level.toFixed(1)})`);
    }
    
    if (aosScore.component_scores.spread_magnitude < 70) {
      conditions.push(`Wait for larger spread (current: ${(prediction.predicted_spread_pct * 100).toFixed(3)}%)`);
    }
    
    if (prediction.risk_flags.length > 1) {
      conditions.push(`Wait for risk flags to clear (current: ${prediction.risk_flags.length})`);
    }
    
    conditions.push(`Re-evaluate when AOS >= ${this.parameters.aos_execution_threshold}`);
    
    return conditions;
  }

  private getTodayTradeCount(): number {
    const today = new Date().toDateString();
    return this.executionHistory.filter(decision => {
      return new Date(decision.timestamp).toDateString() === today &&
             decision.execution_action === 'execute';
    }).length;
  }

  private createDefaultAOS(): ArbitrageOpportunityScore {
    return {
      overall_score: 0,
      component_scores: {
        spread_magnitude: 0,
        confidence_level: 0,
        time_window: 0,
        liquidity_factor: 0,
        risk_adjustment: 0
      },
      score_weights: { ...this.aosWeights },
      risk_flags_impact: {},
      execution_recommendation: 'reject',
      rationale: 'Default score due to evaluation error'
    };
  }

  // Public methods for transparency
  getVisibleParameters(): any {
    return {
      parameters: this.parameters,
      constraints: this.constraints,
      bounds: this.bounds,
      aos_weights: this.aosWeights,
      risk_flag_penalties: this.riskFlagPenalties,
      active_monitored_positions: this.activeMonitoredPositions.size,
      total_decisions_today: this.getTodayTradeCount(),
      execution_history_count: this.executionHistory.length
    };
  }

  getExecutionHistory(limit: number = 50): ExecutionDecision[] {
    return this.executionHistory
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit);
  }

  getActiveMonitoredPositions(): ExecutionDecision[] {
    return Array.from(this.activeMonitoredPositions.values());
  }

  getPerformanceMetrics(): any {
    const totalDecisions = this.executionHistory.length;
    const executeDecisions = this.executionHistory.filter(d => d.execution_action === 'execute').length;
    const monitorDecisions = this.executionHistory.filter(d => d.execution_action === 'monitor').length;
    const rejectDecisions = this.executionHistory.filter(d => d.execution_action === 'reject').length;
    
    const avgAOS = this.executionHistory.reduce((sum, d) => sum + d.aos_score.overall_score, 0) / totalDecisions || 0;
    
    return {
      total_decisions: totalDecisions,
      execution_rate: totalDecisions > 0 ? (executeDecisions / totalDecisions) : 0,
      monitor_rate: totalDecisions > 0 ? (monitorDecisions / totalDecisions) : 0,
      rejection_rate: totalDecisions > 0 ? (rejectDecisions / totalDecisions) : 0,
      average_aos_score: avgAOS,
      decisions_today: this.getTodayTradeCount(),
      active_monitors: this.activeMonitoredPositions.size
    };
  }

  /**
   * Remove stale monitored positions
   */
  cleanupStaleMonitors(maxAgeMinutes: number = 30): void {
    const cutoff = Date.now() - (maxAgeMinutes * 60 * 1000);
    
    for (const [predictionId, decision] of this.activeMonitoredPositions.entries()) {
      const decisionTime = new Date(decision.timestamp).getTime();
      if (decisionTime < cutoff) {
        this.activeMonitoredPositions.delete(predictionId);
        console.log(`🧹 Cleaned up stale monitor: ${predictionId}`);
      }
    }
  }
}