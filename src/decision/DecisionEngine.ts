import { EventEmitter } from 'events';
import Logger from '../utils/logger';
import config from '../utils/ConfigLoader';
import {
  FusionPrediction,
  DecisionConstraints,
  DecisionBounds,
  ExecutionPlan,
  AOSWeights,
  AgentOutput
} from '../types';

interface ConstraintCheck {
  name: string;
  passed: boolean;
  value: number;
  threshold: number;
  reason?: string;
}

interface DecisionAudit {
  timestamp: string;
  prediction: FusionPrediction;
  constraintChecks: ConstraintCheck[];
  boundChecks: ConstraintCheck[];
  aosScore: number;
  aosComponents: Record<string, number>;
  decision: 'approved' | 'rejected';
  rejectionReasons: string[];
  executionPlan?: ExecutionPlan;
}

export class DecisionEngine extends EventEmitter {
  private logger: Logger;
  private constraints: DecisionConstraints;
  private bounds: DecisionBounds;
  private aosWeights: AOSWeights;
  private auditLog: DecisionAudit[] = [];
  private activePositions: Map<string, ExecutionPlan> = new Map();
  private apiHealth: Map<string, boolean> = new Map();
  private eventBlackoutActive: boolean = false;
  private nav: number = 1000000; // Net Asset Value in USD

  constructor() {
    super();
    this.logger = Logger.getInstance('DecisionEngine');
    
    // Load configuration
    this.constraints = config.get('constraints');
    this.bounds = config.get('bounds');
    this.aosWeights = config.get('aos_weights');
    
    this.initializeHealthMonitoring();
  }

  /**
   * Initialize API health monitoring
   */
  private initializeHealthMonitoring(): void {
    // Monitor authorized exchanges
    for (const exchange of this.constraints.authorized_exchanges || []) {
      this.apiHealth.set(exchange, true);
    }

    // Start health check interval
    setInterval(() => {
      this.checkApiHealth();
    }, 30000); // Check every 30 seconds
  }

  /**
   * Check API health for all exchanges
   */
  private async checkApiHealth(): Promise<void> {
    // In production, would ping each exchange API
    // For now, simulate with random health
    for (const [exchange] of this.apiHealth) {
      const isHealthy = Math.random() > 0.1; // 90% healthy
      this.apiHealth.set(exchange, isHealthy);
    }
  }

  /**
   * Process fusion prediction through decision pipeline
   */
  async processDecision(
    prediction: FusionPrediction,
    agentData: Map<string, AgentOutput>
  ): Promise<DecisionAudit> {
    const audit: DecisionAudit = {
      timestamp: new Date().toISOString(),
      prediction,
      constraintChecks: [],
      boundChecks: [],
      aosScore: 0,
      aosComponents: {},
      decision: 'rejected',
      rejectionReasons: []
    };

    try {
      // 1. Check global constraints
      const globalConstraints = this.checkGlobalConstraints(prediction);
      audit.constraintChecks.push(...globalConstraints);

      if (!this.allChecksPassed(globalConstraints)) {
        audit.rejectionReasons.push('Failed global constraints');
        this.logDecision(audit);
        return audit;
      }

      // 2. Check agent-specific constraints
      const agentConstraints = this.checkAgentConstraints(agentData);
      audit.constraintChecks.push(...agentConstraints);

      if (!this.allChecksPassed(agentConstraints)) {
        audit.rejectionReasons.push('Failed agent constraints');
        this.logDecision(audit);
        return audit;
      }

      // 3. Check decision bounds
      const boundChecks = this.checkBounds(prediction);
      audit.boundChecks.push(...boundChecks);

      if (!this.allChecksPassed(boundChecks)) {
        audit.rejectionReasons.push('Failed decision bounds');
        this.logDecision(audit);
        return audit;
      }

      // 4. Calculate AOS (Arbitrage Opportunity Score)
      const { score, components } = this.calculateAOS(prediction, agentData);
      audit.aosScore = score;
      audit.aosComponents = components;

      // 5. Check AOS threshold
      const aosThreshold = 0.6;
      if (score < aosThreshold) {
        audit.rejectionReasons.push(`AOS score ${score.toFixed(3)} below threshold ${aosThreshold}`);
        this.logDecision(audit);
        return audit;
      }

      // 6. Create execution plan
      const executionPlan = this.createExecutionPlan(prediction, score);
      audit.executionPlan = executionPlan;
      audit.decision = 'approved';

      // 7. Store and emit decision
      this.activePositions.set(executionPlan.id, executionPlan);
      this.logDecision(audit);
      this.emit('decision', audit);

      this.logger.info('Decision approved', {
        planId: executionPlan.id,
        aosScore: score,
        spread: prediction.predicted_spread_pct
      });

    } catch (error) {
      this.logger.error('Decision processing error', error);
      audit.rejectionReasons.push('Processing error');
    }

    return audit;
  }

  /**
   * Check global constraints
   */
  private checkGlobalConstraints(prediction: FusionPrediction): ConstraintCheck[] {
    const checks: ConstraintCheck[] = [];

    // 1. Max open exposure check
    const currentExposure = this.calculateCurrentExposure();
    const maxExposure = this.nav * this.constraints.max_open_exposure_pct_of_NAV;
    checks.push({
      name: 'max_open_exposure',
      passed: currentExposure + prediction.arbitrage_plan.notional_usd <= maxExposure,
      value: currentExposure + prediction.arbitrage_plan.notional_usd,
      threshold: maxExposure,
      reason: `Current exposure: $${currentExposure.toFixed(2)}`
    });

    // 2. API health check
    const unhealthyAPIs = Array.from(this.apiHealth.entries())
      .filter(([_, healthy]) => !healthy).length;
    checks.push({
      name: 'api_health',
      passed: unhealthyAPIs < this.constraints.api_health_pause_threshold,
      value: unhealthyAPIs,
      threshold: this.constraints.api_health_pause_threshold,
      reason: `${unhealthyAPIs} APIs unhealthy`
    });

    // 3. Event blackout check
    checks.push({
      name: 'event_blackout',
      passed: !this.eventBlackoutActive,
      value: this.eventBlackoutActive ? 1 : 0,
      threshold: 0,
      reason: this.eventBlackoutActive ? 'Economic event blackout active' : 'No blackout'
    });

    // 4. Authorized exchanges check
    const buyExchange = prediction.arbitrage_plan.buy;
    const sellExchange = prediction.arbitrage_plan.sell;
    const authorizedExchanges = this.constraints.authorized_exchanges || [];
    checks.push({
      name: 'authorized_exchanges',
      passed: authorizedExchanges.includes(buyExchange) && authorizedExchanges.includes(sellExchange),
      value: 1,
      threshold: 1,
      reason: `Buy: ${buyExchange}, Sell: ${sellExchange}`
    });

    // 5. Maximum simultaneous trades
    checks.push({
      name: 'max_simultaneous_trades',
      passed: this.activePositions.size < this.bounds.max_simultaneous_trades,
      value: this.activePositions.size,
      threshold: this.bounds.max_simultaneous_trades,
      reason: `Active positions: ${this.activePositions.size}`
    });

    return checks;
  }

  /**
   * Check agent-specific constraints
   */
  private checkAgentConstraints(agentData: Map<string, AgentOutput>): ConstraintCheck[] {
    const checks: ConstraintCheck[] = [];

    // Check data freshness for each agent
    const now = Date.now();
    for (const [agentName, data] of agentData) {
      const timestamp = new Date(data.timestamp).getTime();
      const age = (now - timestamp) / 1000; // Age in seconds
      
      checks.push({
        name: `${agentName}_freshness`,
        passed: age <= this.constraints.data_freshness_max_sec,
        value: age,
        threshold: this.constraints.data_freshness_max_sec,
        reason: `Data age: ${age.toFixed(1)}s`
      });

      // Check confidence threshold
      const minConfidence = this.getAgentMinConfidence(agentName);
      checks.push({
        name: `${agentName}_confidence`,
        passed: data.confidence >= minConfidence,
        value: data.confidence,
        threshold: minConfidence,
        reason: `Confidence: ${(data.confidence * 100).toFixed(1)}%`
      });
    }

    return checks;
  }

  /**
   * Check decision bounds
   */
  private checkBounds(prediction: FusionPrediction): ConstraintCheck[] {
    const checks: ConstraintCheck[] = [];

    // 1. Minimum spread check
    checks.push({
      name: 'min_spread_pct',
      passed: prediction.predicted_spread_pct >= this.bounds.min_spread_pct_for_execution,
      value: prediction.predicted_spread_pct,
      threshold: this.bounds.min_spread_pct_for_execution,
      reason: `Predicted spread: ${prediction.predicted_spread_pct.toFixed(3)}%`
    });

    // 2. LLM confidence threshold
    checks.push({
      name: 'llm_confidence',
      passed: prediction.confidence >= this.bounds.llm_confidence_threshold,
      value: prediction.confidence,
      threshold: this.bounds.llm_confidence_threshold,
      reason: `LLM confidence: ${(prediction.confidence * 100).toFixed(1)}%`
    });

    // 3. Expected profit check
    const expectedProfit = (prediction.arbitrage_plan.notional_usd * prediction.predicted_spread_pct / 100) -
                          this.estimateFees(prediction.arbitrage_plan.notional_usd);
    checks.push({
      name: 'min_expected_profit',
      passed: expectedProfit >= this.bounds.min_expected_net_profit_usd,
      value: expectedProfit,
      threshold: this.bounds.min_expected_net_profit_usd,
      reason: `Expected profit: $${expectedProfit.toFixed(2)}`
    });

    // 4. Max hold time check
    checks.push({
      name: 'max_hold_time',
      passed: prediction.expected_time_s <= this.bounds.max_hold_time_sec,
      value: prediction.expected_time_s,
      threshold: this.bounds.max_hold_time_sec,
      reason: `Hold time: ${prediction.expected_time_s}s`
    });

    // 5. Slippage estimate check
    const slippageEstimate = this.estimateSlippage(prediction);
    checks.push({
      name: 'max_slippage',
      passed: slippageEstimate <= this.bounds.max_slippage_pct_estimate,
      value: slippageEstimate,
      threshold: this.bounds.max_slippage_pct_estimate,
      reason: `Estimated slippage: ${(slippageEstimate * 100).toFixed(3)}%`
    });

    return checks;
  }

  /**
   * Calculate Arbitrage Opportunity Score (AOS)
   */
  private calculateAOS(
    prediction: FusionPrediction,
    agentData: Map<string, AgentOutput>
  ): { score: number; components: Record<string, number> } {
    const components: Record<string, number> = {};
    
    // 1. Price component (normalized spread)
    const normalizedSpread = Math.min(prediction.predicted_spread_pct / 2, 1); // Cap at 2%
    components.price = normalizedSpread * this.aosWeights.price;

    // 2. Sentiment component
    const sentimentAgent = Array.from(agentData.values()).find(a => a.agent_name === 'SentimentAgent');
    const sentimentScore = sentimentAgent ? sentimentAgent.key_signal * sentimentAgent.confidence : 0.5;
    components.sentiment = sentimentScore * this.aosWeights.sentiment;

    // 3. Volume/Liquidity component
    const volumeAgent = Array.from(agentData.values()).find(a => a.agent_name === 'VolumeAgent');
    const liquidityScore = volumeAgent ? volumeAgent.key_signal : 0.5;
    components.volume = liquidityScore * this.aosWeights.volume;

    // 4. Image/Visual component
    const imageAgent = Array.from(agentData.values()).find(a => a.agent_name === 'ImageAgent');
    const imageScore = imageAgent ? imageAgent.key_signal : 0.5;
    components.image = imageScore * this.aosWeights.image;

    // 5. Risk component (inverse - lower risk = higher score)
    const riskScore = 1 - (prediction.risk_flags.length / 10); // Assume max 10 risk flags
    components.risk = riskScore * this.aosWeights.risk;

    // Calculate total score
    const score = Object.values(components).reduce((sum, val) => sum + val, 0);

    return { score, components };
  }

  /**
   * Create execution plan
   */
  private createExecutionPlan(prediction: FusionPrediction, aosScore: number): ExecutionPlan {
    const planId = `exec-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    return {
      id: planId,
      timestamp: new Date().toISOString(),
      buy_exchange: prediction.arbitrage_plan.buy,
      sell_exchange: prediction.arbitrage_plan.sell,
      pair: 'BTC-USDT', // Default for now
      notional_usd: prediction.arbitrage_plan.notional_usd,
      predicted_spread_pct: prediction.predicted_spread_pct,
      expected_profit_usd: (prediction.arbitrage_plan.notional_usd * prediction.predicted_spread_pct / 100) -
                          this.estimateFees(prediction.arbitrage_plan.notional_usd),
      max_hold_time_sec: prediction.expected_time_s,
      slippage_estimate_pct: this.estimateSlippage(prediction),
      aos_score: aosScore,
      risk_flags: prediction.risk_flags,
      status: 'approved'
    };
  }

  /**
   * Helper functions
   */
  private allChecksPassed(checks: ConstraintCheck[]): boolean {
    return checks.every(check => check.passed);
  }

  private calculateCurrentExposure(): number {
    return Array.from(this.activePositions.values())
      .reduce((sum, plan) => sum + plan.notional_usd, 0);
  }

  private getAgentMinConfidence(agentName: string): number {
    const agentType = agentName.replace('Agent', '').toLowerCase();
    const agentConfig = config.get(`agents.${agentType}`);
    return agentConfig?.confidence_min || 0.5;
  }

  private estimateFees(notional: number): number {
    // Simple fee estimation: 0.1% maker + 0.1% taker
    return notional * 0.002;
  }

  private estimateSlippage(prediction: FusionPrediction): number {
    // Simple slippage model based on notional size
    const baseSlippage = 0.0005; // 0.05%
    const sizeImpact = prediction.arbitrage_plan.notional_usd / 1000000; // Impact per million
    return baseSlippage + sizeImpact * 0.0001;
  }

  private logDecision(audit: DecisionAudit): void {
    this.auditLog.push(audit);
    
    // Keep only last 1000 decisions
    if (this.auditLog.length > 1000) {
      this.auditLog = this.auditLog.slice(-1000);
    }

    this.logger.info('Decision logged', {
      decision: audit.decision,
      aosScore: audit.aosScore,
      constraintsPassed: audit.constraintChecks.filter(c => c.passed).length,
      constraintsTotal: audit.constraintChecks.length,
      boundsPassed: audit.boundChecks.filter(c => c.passed).length,
      boundsTotal: audit.boundChecks.length
    });
  }

  /**
   * Get current constraints and bounds
   */
  getParameters(): {
    constraints: DecisionConstraints;
    bounds: DecisionBounds;
    aosWeights: AOSWeights;
  } {
    return {
      constraints: this.constraints,
      bounds: this.bounds,
      aosWeights: this.aosWeights
    };
  }

  /**
   * Get audit log
   */
  getAuditLog(limit: number = 100): DecisionAudit[] {
    return this.auditLog.slice(-limit);
  }

  /**
   * Update NAV (Net Asset Value)
   */
  updateNAV(value: number): void {
    this.nav = value;
    this.logger.info(`NAV updated to $${value.toFixed(2)}`);
  }

  /**
   * Set event blackout
   */
  setEventBlackout(active: boolean): void {
    this.eventBlackoutActive = active;
    this.logger.info(`Event blackout ${active ? 'activated' : 'deactivated'}`);
  }
}

export default DecisionEngine;