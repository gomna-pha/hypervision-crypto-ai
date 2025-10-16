/**
 * Decision Engine - Deterministic Constraints & Bounds Management
 * Applies hard constraints, bounds checking, and risk management rules
 * Ensures all trading decisions meet safety and regulatory requirements
 */

import { FusionPrediction, ArbitragePlan } from '../core/fusion/fusion-brain.js';
import { AgentOutput } from '../core/base-agent.js';
import yaml from 'yaml';
import { readFileSync } from 'fs';

export interface DecisionConstraints {
  // Global Constraints (Hard Limits)
  max_open_exposure_pct_of_NAV: number;
  api_health_pause_threshold: number;
  event_blackout_sec: number;
  data_freshness_max_sec: number;
  authorized_exchanges_only: boolean;
  max_concurrent_agents: number;
  circuit_breaker_drawdown_pct: number;
}

export interface DecisionBounds {
  // Decision Engine Bounds
  min_spread_pct_for_execution: number;
  llm_confidence_threshold: number;
  min_expected_net_profit_usd: number;
  max_hold_time_sec: number;
  max_simultaneous_trades: number;
  max_slippage_pct_estimate: number;
  z_threshold_k: number;
}

export interface AgentConstraints {
  economic: {
    data_age_max_hours: number;
    confidence_min: number;
  };
  sentiment: {
    min_mention_volume: number;
    confidence_min: number;
  };
  price: {
    min_24h_volume_usd: number;
    orderbook_depth_usd_min: number;
    latency_ms_max: number;
  };
  volume: {
    liquidity_index_min: number;
  };
  trade: {
    slippage_estimate_max_pct: number;
  };
  image: {
    visual_confidence_min: number;
    image_age_max_sec: number;
  };
}

export interface DecisionResult {
  approved: boolean;
  execution_plan?: ExecutionPlan;
  rejection_reasons?: string[];
  risk_assessment: RiskAssessment;
  decision_metadata: {
    aos_score: number;
    constraint_checks: Record<string, boolean>;
    bound_checks: Record<string, boolean>;
    agent_health: Record<string, boolean>;
    timestamp: string;
  };
}

export interface ExecutionPlan {
  trade_id: string;
  arbitrage_plan: ArbitragePlan;
  risk_limits: {
    max_position_size_usd: number;
    stop_loss_pct: number;
    take_profit_pct: number;
    max_hold_time_sec: number;
  };
  execution_params: {
    order_type: 'market' | 'limit' | 'post_only';
    slippage_tolerance_pct: number;
    retry_attempts: number;
    timeout_sec: number;
  };
  monitoring: {
    price_alerts: number[];
    time_alerts: number[];
    risk_alerts: string[];
  };
}

export interface RiskAssessment {
  overall_risk_score: number;      // 0-1 scale (1 = highest risk)
  liquidity_risk: number;          // Risk of insufficient liquidity
  execution_risk: number;          // Risk of poor execution
  market_risk: number;             // Market movement risk
  operational_risk: number;        // System/API risk
  concentration_risk: number;      // Portfolio concentration risk
  risk_factors: string[];          // List of identified risk factors
}

export interface PortfolioState {
  current_exposure_usd: number;
  open_positions: number;
  nav_estimate_usd: number;
  recent_pnl_usd: number;
  drawdown_from_peak_pct: number;
}

export class DecisionEngine {
  private constraints: DecisionConstraints;
  private bounds: DecisionBounds;
  private agentConstraints: AgentConstraints;
  private portfolioState: PortfolioState;
  private authorizedExchanges: Set<string>;
  private eventBlacklist: Array<{ start: number; end: number; event: string }> = [];
  private activePositions: Map<string, ExecutionPlan> = new Map();
  private decisionHistory: DecisionResult[] = [];

  constructor(configPath: string = './arbitrage/config/platform.yaml') {
    const config = this.loadConfig(configPath);
    this.constraints = config.constraints;
    this.bounds = config.bounds;
    this.agentConstraints = config.agent_constraints;
    this.authorizedExchanges = new Set(config.authorized_exchanges || ['binance', 'coinbase', 'kraken']);
    
    // Initialize portfolio state
    this.portfolioState = {
      current_exposure_usd: 0,
      open_positions: 0,
      nav_estimate_usd: 1000000, // $1M default NAV for demo
      recent_pnl_usd: 0,
      drawdown_from_peak_pct: 0
    };
  }

  /**
   * Main decision process - evaluate fusion prediction
   */
  async evaluateDecision(
    prediction: FusionPrediction,
    agents: Record<string, AgentOutput>
  ): Promise<DecisionResult> {
    const timestamp = new Date().toISOString();

    try {
      // Step 1: Check global constraints
      const constraintChecks = this.checkGlobalConstraints(agents);
      
      // Step 2: Check agent-specific constraints
      const agentHealthChecks = this.checkAgentConstraints(agents);
      
      // Step 3: Check decision bounds
      const boundChecks = this.checkDecisionBounds(prediction);
      
      // Step 4: Calculate risk assessment
      const riskAssessment = this.calculateRiskAssessment(prediction, agents);
      
      // Step 5: Make final decision
      const approved = this.makeFinalDecision(
        constraintChecks,
        agentHealthChecks,
        boundChecks,
        riskAssessment
      );
      
      // Step 6: Generate execution plan if approved
      let executionPlan: ExecutionPlan | undefined;
      let rejectionReasons: string[] | undefined;
      
      if (approved) {
        executionPlan = this.generateExecutionPlan(prediction, riskAssessment);
        
        // Register the active position
        this.activePositions.set(executionPlan.trade_id, executionPlan);
        this.updatePortfolioState(executionPlan);
      } else {
        rejectionReasons = this.generateRejectionReasons(
          constraintChecks,
          agentHealthChecks,
          boundChecks,
          riskAssessment
        );
      }
      
      const result: DecisionResult = {
        approved,
        execution_plan: executionPlan,
        rejection_reasons: rejectionReasons,
        risk_assessment: riskAssessment,
        decision_metadata: {
          aos_score: prediction.aos_score,
          constraint_checks: constraintChecks,
          bound_checks: boundChecks,
          agent_health: agentHealthChecks,
          timestamp
        }
      };
      
      // Store decision for audit
      this.decisionHistory.push(result);
      this.trimDecisionHistory();
      
      console.log(`Decision: ${approved ? 'APPROVED' : 'REJECTED'} - Risk: ${riskAssessment.overall_risk_score.toFixed(3)}`);
      
      return result;
      
    } catch (error) {
      console.error('Decision engine evaluation failed:', error);
      throw error;
    }
  }

  /**
   * Check global constraints (hard limits)
   */
  private checkGlobalConstraints(agents: Record<string, AgentOutput>): Record<string, boolean> {
    const checks: Record<string, boolean> = {};
    
    // Check exposure limits
    const exposurePct = this.portfolioState.current_exposure_usd / this.portfolioState.nav_estimate_usd;
    checks['max_exposure'] = exposurePct <= this.constraints.max_open_exposure_pct_of_NAV;
    
    // Check API health (count unhealthy agents)
    const unhealthyAgents = Object.values(agents).filter(agent => 
      !agent || agent.confidence < 0.3
    ).length;
    checks['api_health'] = unhealthyAgents < this.constraints.api_health_pause_threshold;
    
    // Check event blackout periods
    checks['event_blackout'] = !this.isInEventBlackout();
    
    // Check data freshness
    const now = Date.now();
    const staleAgents = Object.values(agents).filter(agent => 
      !agent || (now - new Date(agent.timestamp).getTime()) > (this.constraints.data_freshness_max_sec * 1000)
    ).length;
    checks['data_freshness'] = staleAgents === 0;
    
    // Check circuit breaker
    checks['circuit_breaker'] = this.portfolioState.drawdown_from_peak_pct <= this.constraints.circuit_breaker_drawdown_pct;
    
    // Check concurrent agents
    checks['concurrent_agents'] = Object.keys(agents).length <= this.constraints.max_concurrent_agents;
    
    return checks;
  }

  /**
   * Check agent-specific constraints
   */
  private checkAgentConstraints(agents: Record<string, AgentOutput>): Record<string, boolean> {
    const checks: Record<string, boolean> = {};
    const now = Date.now();
    
    // Economic agent checks
    if (agents['economic']) {
      const age = (now - new Date(agents['economic'].timestamp).getTime()) / (1000 * 60 * 60);
      checks['economic_age'] = age <= this.agentConstraints.economic.data_age_max_hours;
      checks['economic_confidence'] = agents['economic'].confidence >= this.agentConstraints.economic.confidence_min;
    } else {
      checks['economic_age'] = false;
      checks['economic_confidence'] = false;
    }
    
    // Sentiment agent checks
    if (agents['sentiment']) {
      const mentionVolume = agents['sentiment'].features?.mention_volume || 0;
      checks['sentiment_volume'] = mentionVolume >= this.agentConstraints.sentiment.min_mention_volume;
      checks['sentiment_confidence'] = agents['sentiment'].confidence >= this.agentConstraints.sentiment.confidence_min;
    } else {
      checks['sentiment_volume'] = false;
      checks['sentiment_confidence'] = false;
    }
    
    // Price agent checks
    if (agents['price']) {
      const volume24h = agents['price'].features?.volume_24h_usd || 0;
      const orderbookDepth = agents['price'].features?.orderbook_depth_usd || 0;
      const latency = agents['price'].features?.api_latency_ms || Infinity;
      
      checks['price_volume'] = volume24h >= this.agentConstraints.price.min_24h_volume_usd;
      checks['price_depth'] = orderbookDepth >= this.agentConstraints.price.orderbook_depth_usd_min;
      checks['price_latency'] = latency <= this.agentConstraints.price.latency_ms_max;
    } else {
      checks['price_volume'] = false;
      checks['price_depth'] = false;
      checks['price_latency'] = false;
    }
    
    // Volume agent checks
    if (agents['volume']) {
      const liquidityIndex = agents['volume'].features?.liquidity_index || 0;
      checks['volume_liquidity'] = liquidityIndex >= this.agentConstraints.volume.liquidity_index_min;
    } else {
      checks['volume_liquidity'] = false;
    }
    
    // Trade agent checks
    if (agents['trade']) {
      const slippageEstimate = agents['trade'].features?.slippage_estimate || 1;
      checks['trade_slippage'] = slippageEstimate <= this.agentConstraints.trade.slippage_estimate_max_pct;
    } else {
      checks['trade_slippage'] = false;
    }
    
    // Image agent checks
    if (agents['image']) {
      const visualConfidence = agents['image'].features?.visual_confidence || 0;
      const imageAge = (now - new Date(agents['image'].timestamp).getTime()) / 1000;
      
      checks['image_confidence'] = visualConfidence >= this.agentConstraints.image.visual_confidence_min;
      checks['image_age'] = imageAge <= this.agentConstraints.image.image_age_max_sec;
    } else {
      checks['image_confidence'] = true; // Image agent is optional
      checks['image_age'] = true;
    }
    
    return checks;
  }

  /**
   * Check decision bounds (thresholds)
   */
  private checkDecisionBounds(prediction: FusionPrediction): Record<string, boolean> {
    const checks: Record<string, boolean> = {};
    
    // Minimum spread check
    checks['min_spread'] = prediction.predicted_spread_pct >= this.bounds.min_spread_pct_for_execution;
    
    // LLM confidence check
    checks['llm_confidence'] = prediction.confidence >= this.bounds.llm_confidence_threshold;
    
    // Expected profit check
    const estimatedProfit = prediction.arbitrage_plan.estimated_profit_usd || 0;
    checks['min_profit'] = estimatedProfit >= this.bounds.min_expected_net_profit_usd;
    
    // Hold time check
    checks['max_hold_time'] = prediction.expected_time_s <= this.bounds.max_hold_time_sec;
    
    // Simultaneous trades check
    checks['max_trades'] = this.activePositions.size < this.bounds.max_simultaneous_trades;
    
    // Exchange authorization check
    const buyExchange = prediction.arbitrage_plan.buy_exchange;
    const sellExchange = prediction.arbitrage_plan.sell_exchange;
    checks['authorized_exchanges'] = this.authorizedExchanges.has(buyExchange) && 
                                     this.authorizedExchanges.has(sellExchange);
    
    return checks;
  }

  /**
   * Calculate comprehensive risk assessment
   */
  private calculateRiskAssessment(
    prediction: FusionPrediction, 
    agents: Record<string, AgentOutput>
  ): RiskAssessment {
    const riskFactors: string[] = [];
    
    // Liquidity risk
    const liquidityIndex = agents['volume']?.features?.liquidity_index || 0.5;
    const liquidityRisk = 1 - liquidityIndex;
    if (liquidityRisk > 0.6) riskFactors.push('low_liquidity');
    
    // Execution risk (slippage and latency)
    const slippageEstimate = agents['trade']?.features?.slippage_estimate || 0.01;
    const apiLatency = agents['price']?.features?.api_latency_ms || 100;
    const executionRisk = (slippageEstimate / 0.02) * 0.7 + (apiLatency / 500) * 0.3;
    if (executionRisk > 0.6) riskFactors.push('high_execution_risk');
    
    // Market risk (volatility and momentum)
    const volatility = agents['price']?.features?.volatility_1m || 0.5;
    const momentum = Math.abs(agents['price']?.features?.price_momentum || 0);
    const marketRisk = (volatility * 0.6) + (momentum * 0.4);
    if (marketRisk > 0.7) riskFactors.push('high_market_volatility');
    
    // Operational risk (agent health and data quality)
    const avgConfidence = Object.values(agents).reduce((sum, agent) => 
      sum + (agent?.confidence || 0), 0
    ) / Object.keys(agents).length;
    const operationalRisk = 1 - avgConfidence;
    if (operationalRisk > 0.4) riskFactors.push('low_data_quality');
    
    // Concentration risk (position sizing)
    const notionalUsd = prediction.arbitrage_plan.notional_usd;
    const concentrationRisk = notionalUsd / this.portfolioState.nav_estimate_usd;
    if (concentrationRisk > 0.1) riskFactors.push('large_position_size');
    
    // Overall risk score (weighted average)
    const overallRisk = (
      liquidityRisk * 0.25 +
      executionRisk * 0.25 +
      marketRisk * 0.25 +
      operationalRisk * 0.15 +
      concentrationRisk * 0.1
    );
    
    // Add prediction-specific risks
    if (prediction.risk_flags && prediction.risk_flags.length > 0) {
      riskFactors.push(...prediction.risk_flags);
    }
    
    return {
      overall_risk_score: Math.max(0, Math.min(1, overallRisk)),
      liquidity_risk: liquidityRisk,
      execution_risk: executionRisk,
      market_risk: marketRisk,
      operational_risk: operationalRisk,
      concentration_risk: concentrationRisk,
      risk_factors: riskFactors
    };
  }

  /**
   * Make final approve/reject decision
   */
  private makeFinalDecision(
    constraintChecks: Record<string, boolean>,
    agentHealthChecks: Record<string, boolean>,
    boundChecks: Record<string, boolean>,
    riskAssessment: RiskAssessment
  ): boolean {
    // All global constraints must pass
    const constraintsPassed = Object.values(constraintChecks).every(check => check);
    
    // Critical agent checks must pass (allow some flexibility for optional agents)
    const criticalAgentChecks = [
      'price_volume', 'price_depth', 'price_latency',
      'volume_liquidity', 'trade_slippage'
    ];
    const criticalAgentsPassed = criticalAgentChecks.every(check => 
      agentHealthChecks[check] !== false
    );
    
    // All decision bounds must pass
    const boundsPassed = Object.values(boundChecks).every(check => check);
    
    // Overall risk must be acceptable (configurable threshold)
    const riskAcceptable = riskAssessment.overall_risk_score <= 0.7; // 70% max risk
    
    return constraintsPassed && criticalAgentsPassed && boundsPassed && riskAcceptable;
  }

  /**
   * Generate execution plan for approved trade
   */
  private generateExecutionPlan(
    prediction: FusionPrediction, 
    riskAssessment: RiskAssessment
  ): ExecutionPlan {
    const tradeId = this.generateTradeId();
    const notionalUsd = prediction.arbitrage_plan.notional_usd;
    
    // Risk-adjusted position sizing
    const riskMultiplier = Math.max(0.1, 1 - riskAssessment.overall_risk_score);
    const adjustedNotional = notionalUsd * riskMultiplier;
    
    // Dynamic risk limits based on prediction and market conditions
    const stopLossPercentage = Math.max(0.005, riskAssessment.market_risk * 0.02); // 0.5% - 2%
    const takeProfitPercentage = prediction.predicted_spread_pct * 0.8; // Take 80% of predicted spread
    
    return {
      trade_id: tradeId,
      arbitrage_plan: {
        ...prediction.arbitrage_plan,
        notional_usd: adjustedNotional,
        estimated_profit_usd: adjustedNotional * prediction.predicted_spread_pct * 0.8
      },
      risk_limits: {
        max_position_size_usd: adjustedNotional,
        stop_loss_pct: stopLossPercentage,
        take_profit_pct: takeProfitPercentage,
        max_hold_time_sec: prediction.expected_time_s
      },
      execution_params: {
        order_type: riskAssessment.liquidity_risk > 0.5 ? 'limit' : 'market',
        slippage_tolerance_pct: Math.min(0.002, riskAssessment.execution_risk * 0.005),
        retry_attempts: 3,
        timeout_sec: 30
      },
      monitoring: {
        price_alerts: [
          prediction.arbitrage_plan.notional_usd * (1 - stopLossPercentage),
          prediction.arbitrage_plan.notional_usd * (1 + takeProfitPercentage)
        ],
        time_alerts: [
          Math.floor(prediction.expected_time_s * 0.5),
          Math.floor(prediction.expected_time_s * 0.8),
          prediction.expected_time_s
        ],
        risk_alerts: riskAssessment.risk_factors
      }
    };
  }

  /**
   * Generate rejection reasons
   */
  private generateRejectionReasons(
    constraintChecks: Record<string, boolean>,
    agentHealthChecks: Record<string, boolean>,
    boundChecks: Record<string, boolean>,
    riskAssessment: RiskAssessment
  ): string[] {
    const reasons: string[] = [];
    
    // Constraint violations
    for (const [check, passed] of Object.entries(constraintChecks)) {
      if (!passed) {
        reasons.push(`Global constraint violation: ${check}`);
      }
    }
    
    // Agent health violations
    for (const [check, passed] of Object.entries(agentHealthChecks)) {
      if (!passed) {
        reasons.push(`Agent constraint violation: ${check}`);
      }
    }
    
    // Bound violations
    for (const [check, passed] of Object.entries(boundChecks)) {
      if (!passed) {
        reasons.push(`Decision bound violation: ${check}`);
      }
    }
    
    // Risk violations
    if (riskAssessment.overall_risk_score > 0.7) {
      reasons.push(`Excessive risk: ${(riskAssessment.overall_risk_score * 100).toFixed(1)}%`);
    }
    
    // Specific risk factors
    for (const factor of riskAssessment.risk_factors) {
      reasons.push(`Risk factor: ${factor}`);
    }
    
    return reasons;
  }

  /**
   * Generate unique trade ID
   */
  private generateTradeId(): string {
    const timestamp = Date.now().toString(36);
    const random = Math.random().toString(36).substring(2, 8);
    return `ARB_${timestamp}_${random}`.toUpperCase();
  }

  /**
   * Update portfolio state after trade approval
   */
  private updatePortfolioState(executionPlan: ExecutionPlan): void {
    this.portfolioState.current_exposure_usd += executionPlan.arbitrage_plan.notional_usd;
    this.portfolioState.open_positions = this.activePositions.size;
  }

  /**
   * Check if current time is in event blackout period
   */
  private isInEventBlackout(): boolean {
    const now = Date.now();
    
    return this.eventBlacklist.some(event => 
      now >= event.start && now <= event.end
    );
  }

  /**
   * Add event blackout period
   */
  addEventBlackout(startTime: Date, endTime: Date, eventName: string): void {
    this.eventBlacklist.push({
      start: startTime.getTime(),
      end: endTime.getTime(),
      event: eventName
    });
    
    // Clean up old events
    const now = Date.now();
    this.eventBlacklist = this.eventBlacklist.filter(event => event.end > now);
  }

  /**
   * Get active positions
   */
  getActivePositions(): ExecutionPlan[] {
    return Array.from(this.activePositions.values());
  }

  /**
   * Close position (called by execution engine)
   */
  closePosition(tradeId: string, pnl: number): void {
    const position = this.activePositions.get(tradeId);
    if (position) {
      this.portfolioState.current_exposure_usd -= position.arbitrage_plan.notional_usd;
      this.portfolioState.recent_pnl_usd += pnl;
      this.portfolioState.open_positions = this.activePositions.size - 1;
      
      this.activePositions.delete(tradeId);
    }
  }

  /**
   * Get decision statistics
   */
  getDecisionStatistics(): {
    total_decisions: number;
    approval_rate: number;
    avg_risk_score: number;
    common_rejection_reasons: string[];
    portfolio_state: PortfolioState;
  } {
    if (this.decisionHistory.length === 0) {
      return {
        total_decisions: 0,
        approval_rate: 0,
        avg_risk_score: 0,
        common_rejection_reasons: [],
        portfolio_state: this.portfolioState
      };
    }
    
    const totalDecisions = this.decisionHistory.length;
    const approvedDecisions = this.decisionHistory.filter(d => d.approved).length;
    const approvalRate = approvedDecisions / totalDecisions;
    
    const avgRiskScore = this.decisionHistory.reduce((sum, d) => 
      sum + d.risk_assessment.overall_risk_score, 0
    ) / totalDecisions;
    
    // Count rejection reasons
    const rejectionReasons: Record<string, number> = {};
    for (const decision of this.decisionHistory) {
      if (!decision.approved && decision.rejection_reasons) {
        for (const reason of decision.rejection_reasons) {
          rejectionReasons[reason] = (rejectionReasons[reason] || 0) + 1;
        }
      }
    }
    
    // Get top 5 most common rejection reasons
    const commonRejections = Object.entries(rejectionReasons)
      .sort(([, a], [, b]) => b - a)
      .slice(0, 5)
      .map(([reason]) => reason);
    
    return {
      total_decisions: totalDecisions,
      approval_rate: approvalRate,
      avg_risk_score: avgRiskScore,
      common_rejection_reasons: commonRejections,
      portfolio_state: this.portfolioState
    };
  }

  /**
   * Load configuration from YAML file
   */
  private loadConfig(configPath: string): any {
    try {
      const configFile = readFileSync(configPath, 'utf8');
      const config = yaml.parse(configFile);
      
      return {
        constraints: config.constraints || {},
        bounds: config.bounds || {},
        agent_constraints: {
          economic: config.agents?.economic || {},
          sentiment: config.agents?.sentiment || {},
          price: config.agents?.price || {},
          volume: config.agents?.volume || {},
          trade: config.agents?.trade || {},
          image: config.agents?.image || {}
        },
        authorized_exchanges: ['binance', 'coinbase', 'kraken']
      };
    } catch (error) {
      console.warn('Failed to load decision config, using defaults:', error.message);
      
      // Return safe defaults
      return {
        constraints: {
          max_open_exposure_pct_of_NAV: 0.05,
          api_health_pause_threshold: 2,
          event_blackout_sec: 300,
          data_freshness_max_sec: 10,
          authorized_exchanges_only: true,
          max_concurrent_agents: 10,
          circuit_breaker_drawdown_pct: 0.1
        },
        bounds: {
          min_spread_pct_for_execution: 0.001,
          llm_confidence_threshold: 0.7,
          min_expected_net_profit_usd: 10,
          max_hold_time_sec: 3600,
          max_simultaneous_trades: 3,
          max_slippage_pct_estimate: 0.005,
          z_threshold_k: 2.5
        },
        agent_constraints: {
          economic: { data_age_max_hours: 12, confidence_min: 0.3 },
          sentiment: { min_mention_volume: 10, confidence_min: 0.3 },
          price: { min_24h_volume_usd: 1000000, orderbook_depth_usd_min: 50000, latency_ms_max: 1000 },
          volume: { liquidity_index_min: 0.1 },
          trade: { slippage_estimate_max_pct: 0.01 },
          image: { visual_confidence_min: 0.5, image_age_max_sec: 120 }
        },
        authorized_exchanges: ['binance', 'coinbase', 'kraken']
      };
    }
  }

  /**
   * Trim decision history to prevent memory bloat
   */
  private trimDecisionHistory(): void {
    if (this.decisionHistory.length > 1000) {
      this.decisionHistory = this.decisionHistory.slice(-500);
    }
  }

  /**
   * Update NAV estimate (should be called periodically)
   */
  updateNAV(newNAV: number): void {
    this.portfolioState.nav_estimate_usd = newNAV;
  }

  /**
   * Update drawdown tracking
   */
  updateDrawdown(peakNAV: number, currentNAV: number): void {
    this.portfolioState.drawdown_from_peak_pct = Math.max(0, (peakNAV - currentNAV) / peakNAV);
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