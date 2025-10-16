/**
 * Decision Engine - Deterministic Constraint & Bounds Checking
 * Enforces hard constraints, risk bounds, and safety limits before execution
 * Provides auditable, deterministic decision-making for arbitrage opportunities
 */

import { FusionPrediction } from '../core/fusion/fusion-brain.js';
import { AgentOutput } from '../core/base-agent.js';
import yaml from 'yaml';
import { readFileSync } from 'fs';

export interface GlobalConstraints {
  max_open_exposure_pct_of_NAV: number;
  api_health_pause_threshold: number;
  event_blackout_sec: number;
  data_freshness_max_sec: number;
  authorized_exchanges_only: boolean;
  max_concurrent_agents: number;
  circuit_breaker_drawdown_pct: number;
}

export interface AgentConstraints {
  economic: {
    data_age_max_hours: number;
    confidence_min: number;
  };
  sentiment: {
    mention_volume_min: number;
    confidence_min: number;
  };
  price: {
    min_24h_volume_usd: number;
    orderbook_depth_min_usd: number;
    latency_max_ms: number;
  };
  volume: {
    liquidity_index_min: number;
  };
  trade: {
    slippage_max_pct: number;
  };
  image: {
    visual_confidence_min: number;
    image_age_max_sec: number;
  };
}

export interface DecisionBounds {
  min_spread_pct_for_execution: number;
  llm_confidence_threshold: number;
  min_expected_net_profit_usd: number;
  max_hold_time_sec: number;
  max_simultaneous_trades: number;
  max_slippage_pct_estimate: number;
  z_threshold_k: number;
  max_position_size_pct: number;
  stop_loss_pct: number;
  take_profit_pct: number;
}

export interface ExecutionPlan {
  approved: boolean;
  prediction: FusionPrediction;
  execution_params: {
    buy_exchange: string;
    sell_exchange: string;
    pair: string;
    notional_usd: number;
    max_slippage_bps: number;
    time_limit_sec: number;
    stop_loss_price?: number;
    take_profit_price?: number;
  };
  risk_assessment: {
    risk_score: number;           // 0-1, higher = riskier
    confidence_adjusted: number;  // LLM confidence adjusted for constraints
    expected_sharpe: number;      // Expected Sharpe ratio for this trade
    max_drawdown_estimate: number; // Estimated maximum drawdown
  };
  constraint_results: {
    global_constraints_passed: boolean;
    agent_constraints_passed: boolean;
    bounds_checks_passed: boolean;
    failed_constraints: string[];
    warnings: string[];
  };
  timestamp: string;
  decision_id: string;
}

export interface SystemStatus {
  circuit_breaker_active: boolean;
  active_trades_count: number;
  current_exposure_pct: number;
  unhealthy_apis: string[];
  event_blackout_active: boolean;
  last_decision_timestamp: string;
}

/**
 * Decision Engine Class - Deterministic Arbitrage Decision Making
 */
export class DecisionEngine {
  private config: {
    constraints: GlobalConstraints;
    agent_constraints: AgentConstraints;
    bounds: DecisionBounds;
  };
  
  private systemStatus: SystemStatus;
  private executionHistory: ExecutionPlan[] = [];
  private circuitBreakerTriggered: number = 0;
  private currentDrawdown: number = 0;
  private portfolioValue: number = 1000000; // $1M default NAV
  
  // Event blackout periods (economic announcements, etc.)
  private eventBlackouts: Array<{ start: Date; end: Date; reason: string }> = [];
  
  // Exchange health tracking
  private exchangeHealth: Map<string, { healthy: boolean; last_check: number }> = new Map();
  
  constructor(configPath: string = './arbitrage/config/platform.yaml') {
    this.loadConfiguration(configPath);
    this.initializeSystemStatus();
  }

  /**
   * Main decision function - evaluate prediction and generate execution plan
   */
  async makeDecision(
    prediction: FusionPrediction,
    agents: Record<string, AgentOutput>
  ): Promise<ExecutionPlan> {
    const timestamp = new Date().toISOString();
    const decisionId = this.generateDecisionId();

    const executionPlan: ExecutionPlan = {
      approved: false,
      prediction,
      execution_params: {
        buy_exchange: prediction.arbitrage_plan.buy_exchange,
        sell_exchange: prediction.arbitrage_plan.sell_exchange,
        pair: prediction.arbitrage_plan.pair,
        notional_usd: prediction.arbitrage_plan.notional_usd,
        max_slippage_bps: 20, // 20 basis points default
        time_limit_sec: prediction.expected_time_s
      },
      risk_assessment: {
        risk_score: 0,
        confidence_adjusted: prediction.confidence,
        expected_sharpe: 0,
        max_drawdown_estimate: 0
      },
      constraint_results: {
        global_constraints_passed: false,
        agent_constraints_passed: false,
        bounds_checks_passed: false,
        failed_constraints: [],
        warnings: []
      },
      timestamp,
      decision_id: decisionId
    };

    try {
      // Step 1: Check global constraints
      const globalResult = this.checkGlobalConstraints(prediction, agents);
      executionPlan.constraint_results.global_constraints_passed = globalResult.passed;
      executionPlan.constraint_results.failed_constraints.push(...globalResult.failures);
      executionPlan.constraint_results.warnings.push(...globalResult.warnings);

      // Step 2: Check agent-specific constraints
      const agentResult = this.checkAgentConstraints(agents);
      executionPlan.constraint_results.agent_constraints_passed = agentResult.passed;
      executionPlan.constraint_results.failed_constraints.push(...agentResult.failures);
      executionPlan.constraint_results.warnings.push(...agentResult.warnings);

      // Step 3: Check decision bounds
      const boundsResult = this.checkDecisionBounds(prediction);
      executionPlan.constraint_results.bounds_checks_passed = boundsResult.passed;
      executionPlan.constraint_results.failed_constraints.push(...boundsResult.failures);
      executionPlan.constraint_results.warnings.push(...boundsResult.warnings);

      // Step 4: Calculate risk assessment
      executionPlan.risk_assessment = this.calculateRiskAssessment(prediction, agents);

      // Step 5: Final approval decision
      const allChecksPassed = globalResult.passed && agentResult.passed && boundsResult.passed;
      const riskAcceptable = executionPlan.risk_assessment.risk_score <= 0.7;
      const confidenceAcceptable = executionPlan.risk_assessment.confidence_adjusted >= this.config.bounds.llm_confidence_threshold;

      executionPlan.approved = allChecksPassed && riskAcceptable && confidenceAcceptable;

      // Step 6: Enhance execution parameters if approved
      if (executionPlan.approved) {
        this.enhanceExecutionParameters(executionPlan);
        this.updateSystemStatus(executionPlan);
      }

      // Step 7: Log decision
      this.logDecision(executionPlan);
      this.executionHistory.push(executionPlan);
      this.trimExecutionHistory();

      console.log(`Decision ${decisionId}: ${executionPlan.approved ? 'APPROVED' : 'REJECTED'} - Risk: ${executionPlan.risk_assessment.risk_score.toFixed(3)}, Confidence: ${executionPlan.risk_assessment.confidence_adjusted.toFixed(3)}`);

      return executionPlan;

    } catch (error) {
      console.error('Decision Engine error:', error);
      executionPlan.constraint_results.failed_constraints.push(`Decision engine error: ${error.message}`);
      return executionPlan;
    }
  }

  /**
   * Check global constraints (hard limits)
   */
  private checkGlobalConstraints(prediction: FusionPrediction, agents: Record<string, AgentOutput>): {
    passed: boolean;
    failures: string[];
    warnings: string[];
  } {
    const failures: string[] = [];
    const warnings: string[] = [];

    // Check exposure limits
    const newExposure = (prediction.arbitrage_plan.notional_usd / this.portfolioValue) * 100;
    if (this.systemStatus.current_exposure_pct + newExposure > this.config.constraints.max_open_exposure_pct_of_NAV * 100) {
      failures.push(`Exposure limit exceeded: ${(this.systemStatus.current_exposure_pct + newExposure).toFixed(2)}% > ${(this.config.constraints.max_open_exposure_pct_of_NAV * 100).toFixed(2)}%`);
    }

    // Check API health
    if (this.systemStatus.unhealthy_apis.length >= this.config.constraints.api_health_pause_threshold) {
      failures.push(`Too many unhealthy APIs: ${this.systemStatus.unhealthy_apis.length} >= ${this.config.constraints.api_health_pause_threshold}`);
    }

    // Check event blackout
    if (this.systemStatus.event_blackout_active) {
      failures.push('Trading paused due to economic event blackout period');
    }

    // Check circuit breaker
    if (this.systemStatus.circuit_breaker_active) {
      failures.push('Circuit breaker is active due to excessive drawdown');
    }

    // Check concurrent trades limit
    if (this.systemStatus.active_trades_count >= this.config.bounds.max_simultaneous_trades) {
      failures.push(`Maximum concurrent trades reached: ${this.systemStatus.active_trades_count} >= ${this.config.bounds.max_simultaneous_trades}`);
    }

    // Check authorized exchanges
    if (this.config.constraints.authorized_exchanges_only) {
      const authorizedExchanges = ['binance', 'coinbase', 'kraken'];
      const plan = prediction.arbitrage_plan;
      
      if (!authorizedExchanges.includes(plan.buy_exchange)) {
        failures.push(`Unauthorized buy exchange: ${plan.buy_exchange}`);
      }
      
      if (!authorizedExchanges.includes(plan.sell_exchange)) {
        failures.push(`Unauthorized sell exchange: ${plan.sell_exchange}`);
      }
    }

    // Check data freshness
    const now = Date.now();
    const predictionAge = now - new Date(prediction.timestamp).getTime();
    
    if (predictionAge > this.config.constraints.data_freshness_max_sec * 1000) {
      failures.push(`Prediction is stale: ${predictionAge}ms > ${this.config.constraints.data_freshness_max_sec * 1000}ms`);
    }

    return {
      passed: failures.length === 0,
      failures,
      warnings
    };
  }

  /**
   * Check agent-specific constraints
   */
  private checkAgentConstraints(agents: Record<string, AgentOutput>): {
    passed: boolean;
    failures: string[];
    warnings: string[];
  } {
    const failures: string[] = [];
    const warnings: string[] = [];
    const now = Date.now();

    // Economic agent constraints
    if (agents['economic']) {
      const agent = agents['economic'];
      const ageHours = (now - new Date(agent.timestamp).getTime()) / (1000 * 60 * 60);
      
      if (ageHours > this.config.agent_constraints.economic.data_age_max_hours) {
        failures.push(`Economic data too old: ${ageHours.toFixed(1)}h > ${this.config.agent_constraints.economic.data_age_max_hours}h`);
      }
      
      if (agent.confidence < this.config.agent_constraints.economic.confidence_min) {
        failures.push(`Economic confidence too low: ${agent.confidence.toFixed(3)} < ${this.config.agent_constraints.economic.confidence_min}`);
      }
    } else {
      warnings.push('Economic agent data not available');
    }

    // Sentiment agent constraints
    if (agents['sentiment']) {
      const agent = agents['sentiment'];
      const mentionVolume = agent.features?.twitter_mention_volume || 0;
      
      if (mentionVolume < this.config.agent_constraints.sentiment.mention_volume_min) {
        warnings.push(`Low sentiment mention volume: ${mentionVolume} < ${this.config.agent_constraints.sentiment.mention_volume_min}`);
      }
      
      if (agent.confidence < this.config.agent_constraints.sentiment.confidence_min) {
        failures.push(`Sentiment confidence too low: ${agent.confidence.toFixed(3)} < ${this.config.agent_constraints.sentiment.confidence_min}`);
      }
    } else {
      warnings.push('Sentiment agent data not available');
    }

    // Price agent constraints
    if (agents['price']) {
      const agent = agents['price'];
      const volume24h = agent.features?.exchange_data?.['binance_BTC-USDT']?.volume_1m * 60 * 24 || 0;
      const orderbookDepth = agent.features?.exchange_data?.['binance_BTC-USDT']?.orderbook_depth_usd || 0;
      
      if (volume24h < this.config.agent_constraints.price.min_24h_volume_usd) {
        failures.push(`Insufficient 24h volume: $${volume24h.toFixed(0)} < $${this.config.agent_constraints.price.min_24h_volume_usd}`);
      }
      
      if (orderbookDepth < this.config.agent_constraints.price.orderbook_depth_min_usd) {
        failures.push(`Insufficient orderbook depth: $${orderbookDepth.toFixed(0)} < $${this.config.agent_constraints.price.orderbook_depth_min_usd}`);
      }
    } else {
      failures.push('Price agent data is required but not available');
    }

    // Volume agent constraints
    if (agents['volume']) {
      const agent = agents['volume'];
      const liquidityIndex = agent.features?.liquidity_index || 0;
      
      if (liquidityIndex < this.config.agent_constraints.volume.liquidity_index_min) {
        failures.push(`Liquidity index too low: ${liquidityIndex.toFixed(3)} < ${this.config.agent_constraints.volume.liquidity_index_min}`);
      }
    } else {
      warnings.push('Volume agent data not available');
    }

    // Trade agent constraints
    if (agents['trade']) {
      const agent = agents['trade'];
      const slippage = agent.features?.btc_trade_data?.slippage_estimate_bps || 0;
      
      if (slippage > this.config.agent_constraints.trade.slippage_max_pct * 10000) {
        failures.push(`Slippage estimate too high: ${slippage}bps > ${this.config.agent_constraints.trade.slippage_max_pct * 10000}bps`);
      }
    } else {
      warnings.push('Trade agent data not available');
    }

    return {
      passed: failures.length === 0,
      failures,
      warnings
    };
  }

  /**
   * Check decision bounds (thresholds)
   */
  private checkDecisionBounds(prediction: FusionPrediction): {
    passed: boolean;
    failures: string[];
    warnings: string[];
  } {
    const failures: string[] = [];
    const warnings: string[] = [];

    // Minimum spread threshold
    if (prediction.predicted_spread_pct < this.config.bounds.min_spread_pct_for_execution) {
      failures.push(`Spread too small: ${(prediction.predicted_spread_pct * 100).toFixed(4)}% < ${(this.config.bounds.min_spread_pct_for_execution * 100).toFixed(4)}%`);
    }

    // LLM confidence threshold
    if (prediction.confidence < this.config.bounds.llm_confidence_threshold) {
      failures.push(`LLM confidence too low: ${prediction.confidence.toFixed(3)} < ${this.config.bounds.llm_confidence_threshold}`);
    }

    // Expected profit threshold
    const estimatedProfit = prediction.arbitrage_plan.estimated_profit_usd || 0;
    if (estimatedProfit < this.config.bounds.min_expected_net_profit_usd) {
      failures.push(`Expected profit too low: $${estimatedProfit.toFixed(2)} < $${this.config.bounds.min_expected_net_profit_usd}`);
    }

    // Hold time bounds
    if (prediction.expected_time_s > this.config.bounds.max_hold_time_sec) {
      failures.push(`Hold time too long: ${prediction.expected_time_s}s > ${this.config.bounds.max_hold_time_sec}s`);
    }

    // Position size bounds
    const positionSizePct = (prediction.arbitrage_plan.notional_usd / this.portfolioValue) * 100;
    if (positionSizePct > this.config.bounds.max_position_size_pct * 100) {
      failures.push(`Position size too large: ${positionSizePct.toFixed(2)}% > ${(this.config.bounds.max_position_size_pct * 100).toFixed(2)}%`);
    }

    // Risk flags check
    const highRiskFlags = ['low_liquidity_on_sell_exchange', 'high_volatility', 'api_degradation'];
    const hasHighRisk = prediction.risk_flags.some(flag => highRiskFlags.includes(flag));
    
    if (hasHighRisk) {
      warnings.push(`High risk flags detected: ${prediction.risk_flags.join(', ')}`);
    }

    return {
      passed: failures.length === 0,
      failures,
      warnings
    };
  }

  /**
   * Calculate comprehensive risk assessment
   */
  private calculateRiskAssessment(prediction: FusionPrediction, agents: Record<string, AgentOutput>): {
    risk_score: number;
    confidence_adjusted: number;
    expected_sharpe: number;
    max_drawdown_estimate: number;
  } {
    let riskFactors: number[] = [];

    // Market volatility risk
    const volatility = agents['price']?.features?.volatility_1m || 0.5;
    riskFactors.push(volatility);

    // Liquidity risk
    const liquidityIndex = agents['volume']?.features?.liquidity_index || 0.5;
    riskFactors.push(1 - liquidityIndex);

    // Execution risk
    const executionQuality = agents['trade']?.features?.execution_quality || 0.5;
    riskFactors.push(1 - executionQuality);

    // Spread risk (tighter spreads are riskier to capture)
    const spreadRisk = Math.max(0, 1 - (prediction.predicted_spread_pct / 0.01)); // Normalize to 1% spread
    riskFactors.push(spreadRisk);

    // Time horizon risk (longer holds are riskier)
    const timeRisk = Math.min(1, prediction.expected_time_s / this.config.bounds.max_hold_time_sec);
    riskFactors.push(timeRisk);

    // Position size risk
    const positionSizePct = (prediction.arbitrage_plan.notional_usd / this.portfolioValue);
    const sizeRisk = Math.min(1, positionSizePct / this.config.bounds.max_position_size_pct);
    riskFactors.push(sizeRisk);

    // Calculate composite risk score
    const riskScore = riskFactors.reduce((sum, r) => sum + r, 0) / riskFactors.length;

    // Adjust confidence based on risk factors
    const confidenceAdjusted = prediction.confidence * (1 - (riskScore * 0.3));

    // Estimate Sharpe ratio (return/risk)
    const expectedReturn = prediction.predicted_spread_pct;
    const expectedSharpe = riskScore > 0 ? (expectedReturn / riskScore) * Math.sqrt(252) : 0;

    // Estimate maximum drawdown
    const maxDrawdownEstimate = Math.min(0.5, riskScore * positionSizePct * 2);

    return {
      risk_score: Math.max(0, Math.min(1, riskScore)),
      confidence_adjusted: Math.max(0, Math.min(1, confidenceAdjusted)),
      expected_sharpe: Math.max(0, expectedSharpe),
      max_drawdown_estimate: Math.max(0, maxDrawdownEstimate)
    };
  }

  /**
   * Enhance execution parameters for approved trades
   */
  private enhanceExecutionParameters(executionPlan: ExecutionPlan): void {
    const prediction = executionPlan.prediction;
    const params = executionPlan.execution_params;

    // Calculate stop loss and take profit levels
    const estimatedPrice = 45000; // TODO: Get from price agent
    const stopLossDistance = estimatedPrice * this.config.bounds.stop_loss_pct;
    const takeProfitDistance = estimatedPrice * this.config.bounds.take_profit_pct;

    if (prediction.direction === 'converge') {
      params.stop_loss_price = estimatedPrice - stopLossDistance;
      params.take_profit_price = estimatedPrice + takeProfitDistance;
    } else {
      params.stop_loss_price = estimatedPrice + stopLossDistance;
      params.take_profit_price = estimatedPrice - takeProfitDistance;
    }

    // Adjust slippage tolerance based on risk
    const baseSlippage = 20; // 20 basis points
    const riskMultiplier = 1 + executionPlan.risk_assessment.risk_score;
    params.max_slippage_bps = Math.ceil(baseSlippage * riskMultiplier);

    // Adjust time limit based on volatility
    const volatilityFactor = 1; // TODO: Get from agents
    params.time_limit_sec = Math.floor(prediction.expected_time_s / volatilityFactor);
  }

  /**
   * Update system status after decision
   */
  private updateSystemStatus(executionPlan: ExecutionPlan): void {
    if (executionPlan.approved) {
      this.systemStatus.active_trades_count++;
      
      const exposureIncrease = (executionPlan.execution_params.notional_usd / this.portfolioValue) * 100;
      this.systemStatus.current_exposure_pct += exposureIncrease;
    }

    this.systemStatus.last_decision_timestamp = executionPlan.timestamp;

    // Check circuit breaker conditions
    if (this.currentDrawdown > this.config.constraints.circuit_breaker_drawdown_pct) {
      this.systemStatus.circuit_breaker_active = true;
      this.circuitBreakerTriggered = Date.now();
      console.warn(`Circuit breaker activated: Drawdown ${(this.currentDrawdown * 100).toFixed(2)}% > ${(this.config.constraints.circuit_breaker_drawdown_pct * 100).toFixed(2)}%`);
    }
  }

  /**
   * Load configuration from YAML file
   */
  private loadConfiguration(configPath: string): void {
    try {
      const configFile = readFileSync(configPath, 'utf8');
      const fullConfig = yaml.parse(configFile);

      this.config = {
        constraints: fullConfig.constraints,
        agent_constraints: fullConfig.agents,
        bounds: fullConfig.bounds
      };

      console.log('Decision Engine configuration loaded successfully');
    } catch (error) {
      console.error('Failed to load configuration, using defaults:', error.message);
      this.setDefaultConfiguration();
    }
  }

  /**
   * Set default configuration if YAML loading fails
   */
  private setDefaultConfiguration(): void {
    this.config = {
      constraints: {
        max_open_exposure_pct_of_NAV: 0.03,
        api_health_pause_threshold: 2,
        event_blackout_sec: 300,
        data_freshness_max_sec: 5,
        authorized_exchanges_only: true,
        max_concurrent_agents: 10,
        circuit_breaker_drawdown_pct: 0.05
      },
      agent_constraints: {
        economic: { data_age_max_hours: 6, confidence_min: 0.5 },
        sentiment: { mention_volume_min: 50, confidence_min: 0.4 },
        price: { min_24h_volume_usd: 5000000, orderbook_depth_min_usd: 100000, latency_max_ms: 200 },
        volume: { liquidity_index_min: 0.2 },
        trade: { slippage_max_pct: 0.02 },
        image: { visual_confidence_min: 0.6, image_age_max_sec: 60 }
      },
      bounds: {
        min_spread_pct_for_execution: 0.005,
        llm_confidence_threshold: 0.8,
        min_expected_net_profit_usd: 10,
        max_hold_time_sec: 3600,
        max_simultaneous_trades: 5,
        max_slippage_pct_estimate: 0.002,
        z_threshold_k: 2.5,
        max_position_size_pct: 0.02,
        stop_loss_pct: 0.01,
        take_profit_pct: 0.005
      }
    };
  }

  /**
   * Initialize system status
   */
  private initializeSystemStatus(): void {
    this.systemStatus = {
      circuit_breaker_active: false,
      active_trades_count: 0,
      current_exposure_pct: 0,
      unhealthy_apis: [],
      event_blackout_active: false,
      last_decision_timestamp: new Date().toISOString()
    };
  }

  /**
   * Generate unique decision ID
   */
  private generateDecisionId(): string {
    const timestamp = Date.now();
    const random = Math.random().toString(36).substring(2, 8);
    return `DEC_${timestamp}_${random}`;
  }

  /**
   * Log decision for audit trail
   */
  private logDecision(executionPlan: ExecutionPlan): void {
    const logEntry = {
      decision_id: executionPlan.decision_id,
      timestamp: executionPlan.timestamp,
      approved: executionPlan.approved,
      prediction_confidence: executionPlan.prediction.confidence,
      risk_score: executionPlan.risk_assessment.risk_score,
      notional_usd: executionPlan.execution_params.notional_usd,
      failed_constraints: executionPlan.constraint_results.failed_constraints,
      warnings: executionPlan.constraint_results.warnings
    };

    // In production, log to persistent storage
    console.log(`[AUDIT] Decision logged:`, JSON.stringify(logEntry));
  }

  /**
   * Trim execution history to prevent memory bloat
   */
  private trimExecutionHistory(): void {
    if (this.executionHistory.length > 500) {
      this.executionHistory = this.executionHistory.slice(-250);
    }
  }

  /**
   * Get system status for monitoring
   */
  getSystemStatus(): SystemStatus {
    return { ...this.systemStatus };
  }

  /**
   * Get recent execution history
   */
  getExecutionHistory(limit: number = 50): ExecutionPlan[] {
    return this.executionHistory.slice(-limit);
  }

  /**
   * Add event blackout period
   */
  addEventBlackout(start: Date, end: Date, reason: string): void {
    this.eventBlackouts.push({ start, end, reason });
    this.updateEventBlackoutStatus();
  }

  /**
   * Update event blackout status
   */
  private updateEventBlackoutStatus(): void {
    const now = new Date();
    this.systemStatus.event_blackout_active = this.eventBlackouts.some(
      blackout => now >= blackout.start && now <= blackout.end
    );
  }

  /**
   * Update exchange health status
   */
  updateExchangeHealth(exchange: string, healthy: boolean): void {
    this.exchangeHealth.set(exchange, {
      healthy,
      last_check: Date.now()
    });

    // Update unhealthy APIs list
    this.systemStatus.unhealthy_apis = Array.from(this.exchangeHealth.entries())
      .filter(([, health]) => !health.healthy)
      .map(([exchange]) => exchange);
  }

  /**
   * Reset circuit breaker (manual override)
   */
  resetCircuitBreaker(): void {
    this.systemStatus.circuit_breaker_active = false;
    this.circuitBreakerTriggered = 0;
    this.currentDrawdown = 0;
    console.log('Circuit breaker manually reset');
  }

  /**
   * Update portfolio metrics
   */
  updatePortfolioMetrics(newValue: number, drawdown: number): void {
    this.portfolioValue = newValue;
    this.currentDrawdown = drawdown;
  }

  /**
   * Get decision statistics
   */
  getDecisionStatistics(): {
    total_decisions: number;
    approval_rate: number;
    avg_risk_score: number;
    avg_notional_usd: number;
    common_rejection_reasons: string[];
  } {
    if (this.executionHistory.length === 0) {
      return {
        total_decisions: 0,
        approval_rate: 0,
        avg_risk_score: 0,
        avg_notional_usd: 0,
        common_rejection_reasons: []
      };
    }

    const approved = this.executionHistory.filter(p => p.approved).length;
    const approvalRate = approved / this.executionHistory.length;
    const avgRiskScore = this.executionHistory.reduce((sum, p) => sum + p.risk_assessment.risk_score, 0) / this.executionHistory.length;
    const avgNotional = this.executionHistory.reduce((sum, p) => sum + p.execution_params.notional_usd, 0) / this.executionHistory.length;

    // Count rejection reasons
    const rejectionReasons: Record<string, number> = {};
    this.executionHistory.filter(p => !p.approved).forEach(p => {
      p.constraint_results.failed_constraints.forEach(reason => {
        rejectionReasons[reason] = (rejectionReasons[reason] || 0) + 1;
      });
    });

    const commonRejections = Object.entries(rejectionReasons)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([reason]) => reason);

    return {
      total_decisions: this.executionHistory.length,
      approval_rate: approvalRate,
      avg_risk_score: avgRiskScore,
      avg_notional_usd: avgNotional,
      common_rejection_reasons: commonRejections
    };
  }
}

/**
 * Factory function to create DecisionEngine
 */
export function createDecisionEngine(configPath?: string): DecisionEngine {
  return new DecisionEngine(configPath);
}

// Export for testing
export { DecisionEngine as default };