/**
 * Horizon-Extended Hierarchical Graph
 * 
 * Extends the base Hierarchical Graph with horizon-aware features:
 * - Horizon nodes (hourly, weekly, monthly)
 * - Sharpe decay tracking per signal
 * - Regime fragility (transition likelihood)
 * - 4-level hierarchy: Signal → Horizon → Regime → Strategy
 */

import { HorizonIndexedSignal } from './multi-horizon-signal-pool';
import { TimeHorizon } from './time-scale-feature-store';
import { MarketRegime } from './market-regime-detection';

// ============================================================================
// Extended Graph Node Types
// ============================================================================

export interface HorizonGraphNode {
  id: string;
  type: 'signal' | 'horizon' | 'regime' | 'strategy';
  label: string;
  metadata: any;
}

export interface HorizonSignalNode extends HorizonGraphNode {
  type: 'signal';
  agentType: string;
  horizon: TimeHorizon;
  direction: -1 | 0 | 1;
  strength: number;
  confidence: number;
  sharpeDecay: number;  // Rate at which signal Sharpe degrades over time
  timestamp: Date;
}

export interface HorizonNode extends HorizonGraphNode {
  type: 'horizon';
  horizon: TimeHorizon;
  decayHours: number;
  signalCount: number;
  avgStability: number;
}

export interface RegimeNode extends HorizonGraphNode {
  type: 'regime';
  regime: MarketRegime;
  confidence: number;
  volatility: number;
  fragility: number;  // Likelihood of regime transition [0, 1]
  horizonAgreement: Map<TimeHorizon, number>;  // Agreement score per horizon
}

export interface StrategyNode extends HorizonGraphNode {
  type: 'strategy';
  strategyType: string;
  suitableRegimes: MarketRegime[];
  suitableHorizons: TimeHorizon[];
  expectedSharpe: Map<MarketRegime, number>;
}

// ============================================================================
// Extended Graph Edge Types
// ============================================================================

export interface HorizonGraphEdge {
  from: string;  // Node ID
  to: string;    // Node ID
  weight: number;
  type: 
    | 'signal_to_horizon'
    | 'horizon_to_regime'
    | 'regime_to_strategy'
    | 'conditional_dependence'
    | 'survival_prob'
    | 'regime_transition'
    | 'strategy_compat';
  metadata?: any;
}

// ============================================================================
// Horizon Hierarchical Graph
// ============================================================================

export class HorizonHierarchicalGraph {
  private nodes: Map<string, HorizonGraphNode> = new Map();
  private edges: HorizonGraphEdge[] = [];
  private adjacencyList: Map<string, string[]> = new Map();
  
  // Regime transition tracking
  private regimeTransitionMatrix: Map<MarketRegime, Map<MarketRegime, number>> = new Map();
  private regimeHistory: { regime: MarketRegime; timestamp: Date }[] = [];
  private maxHistoryLength = 1000;
  
  // Sharpe decay tracking per signal
  private sharpeDecayHistory: Map<string, { timestamp: Date; sharpe: number }[]> = new Map();
  
  constructor() {
    this.initializeRegimeTransitionMatrix();
  }
  
  // ============================================================================
  // Graph Building (4-Level Hierarchy)
  // ============================================================================
  
  /**
   * Build horizon-aware graph
   * 
   * Hierarchy: Signal → Horizon → Regime → Strategy
   */
  buildHorizonGraph(
    signals: {
      hourly: HorizonIndexedSignal[];
      weekly: HorizonIndexedSignal[];
      monthly: HorizonIndexedSignal[];
    },
    currentRegime: MarketRegime,
    regimeConfidence: number,
    regimeVolatility: number,
    strategies: {
      type: string;
      suitableRegimes: MarketRegime[];
      suitableHorizons: TimeHorizon[];
      expectedSharpe: Map<MarketRegime, number>;
    }[]
  ): void {
    // Clear previous graph
    this.clear();
    
    // 1. Add Signal Nodes (all 15 signals)
    this.addSignalNodes(signals);
    
    // 2. Add Horizon Nodes (3 horizons)
    this.addHorizonNodes(signals);
    
    // 3. Add Regime Node (current)
    this.addRegimeNode(currentRegime, regimeConfidence, regimeVolatility, signals);
    
    // 4. Add Strategy Nodes
    this.addStrategyNodes(strategies);
    
    // 5. Add Edges: Signal → Horizon
    this.addSignalToHorizonEdges(signals);
    
    // 6. Add Edges: Horizon → Regime
    this.addHorizonToRegimeEdges(currentRegime);
    
    // 7. Add Edges: Regime → Strategy
    this.addRegimeToStrategyEdges(currentRegime, strategies);
    
    // 8. Add Cross-Signal Edges (Survival Probability)
    this.addCrossSignalEdges(signals);
    
    // 9. Add Regime Transition Edges
    this.addRegimeTransitionEdges(currentRegime);
  }
  
  // ============================================================================
  // Node Creation
  // ============================================================================
  
  private addSignalNodes(signals: {
    hourly: HorizonIndexedSignal[];
    weekly: HorizonIndexedSignal[];
    monthly: HorizonIndexedSignal[];
  }): void {
    const allSignals = [...signals.hourly, ...signals.weekly, ...signals.monthly];
    
    for (const signal of allSignals) {
      const sharpeDecay = this.calculateSharpeDecay(signal);
      
      const node: HorizonSignalNode = {
        id: `signal_${signal.agentType}_${signal.horizon}`,
        type: 'signal',
        label: `${signal.agentType} (${signal.horizon})`,
        agentType: signal.agentType,
        horizon: signal.horizon,
        direction: signal.direction,
        strength: signal.strength,
        confidence: signal.confidence,
        sharpeDecay,
        timestamp: signal.timestamp,
        metadata: {
          decayRate: signal.decayRate,
          stabilityScore: signal.stabilityScore,
          riskScore: signal.riskScore,
        },
      };
      
      this.addNode(node);
    }
  }
  
  private addHorizonNodes(signals: {
    hourly: HorizonIndexedSignal[];
    weekly: HorizonIndexedSignal[];
    monthly: HorizonIndexedSignal[];
  }): void {
    const horizons: { horizon: TimeHorizon; decayHours: number }[] = [
      { horizon: 'hourly', decayHours: 6 },
      { horizon: 'weekly', decayHours: 48 },
      { horizon: 'monthly', decayHours: 168 },
    ];
    
    for (const { horizon, decayHours } of horizons) {
      const horizonSignals = signals[horizon];
      const avgStability = horizonSignals.length > 0
        ? horizonSignals.reduce((sum, s) => sum + s.stabilityScore, 0) / horizonSignals.length
        : 0.5;
      
      const node: HorizonNode = {
        id: `horizon_${horizon}`,
        type: 'horizon',
        label: `${horizon.charAt(0).toUpperCase() + horizon.slice(1)} Horizon`,
        horizon,
        decayHours,
        signalCount: horizonSignals.length,
        avgStability,
        metadata: {},
      };
      
      this.addNode(node);
    }
  }
  
  private addRegimeNode(
    currentRegime: MarketRegime,
    regimeConfidence: number,
    regimeVolatility: number,
    signals: {
      hourly: HorizonIndexedSignal[];
      weekly: HorizonIndexedSignal[];
      monthly: HorizonIndexedSignal[];
    }
  ): void {
    // Calculate regime fragility (transition likelihood)
    const fragility = this.assessRegimeFragility(currentRegime, regimeVolatility);
    
    // Calculate horizon agreement
    const horizonAgreement = new Map<TimeHorizon, number>();
    horizonAgreement.set('hourly', this.calculateHorizonRegimeAgreement(signals.hourly, currentRegime));
    horizonAgreement.set('weekly', this.calculateHorizonRegimeAgreement(signals.weekly, currentRegime));
    horizonAgreement.set('monthly', this.calculateHorizonRegimeAgreement(signals.monthly, currentRegime));
    
    const node: RegimeNode = {
      id: `regime_${currentRegime}`,
      type: 'regime',
      label: currentRegime,
      regime: currentRegime,
      confidence: regimeConfidence,
      volatility: regimeVolatility,
      fragility,
      horizonAgreement,
      metadata: {},
    };
    
    this.addNode(node);
    
    // Update regime history
    this.regimeHistory.push({ regime: currentRegime, timestamp: new Date() });
    if (this.regimeHistory.length > this.maxHistoryLength) {
      this.regimeHistory.shift();
    }
  }
  
  private addStrategyNodes(strategies: {
    type: string;
    suitableRegimes: MarketRegime[];
    suitableHorizons: TimeHorizon[];
    expectedSharpe: Map<MarketRegime, number>;
  }[]): void {
    for (const strategy of strategies) {
      const node: StrategyNode = {
        id: `strategy_${strategy.type}`,
        type: 'strategy',
        label: strategy.type,
        strategyType: strategy.type,
        suitableRegimes: strategy.suitableRegimes,
        suitableHorizons: strategy.suitableHorizons,
        expectedSharpe: strategy.expectedSharpe,
        metadata: {},
      };
      
      this.addNode(node);
    }
  }
  
  // ============================================================================
  // Edge Creation
  // ============================================================================
  
  private addSignalToHorizonEdges(signals: {
    hourly: HorizonIndexedSignal[];
    weekly: HorizonIndexedSignal[];
    monthly: HorizonIndexedSignal[];
  }): void {
    const allSignals = [...signals.hourly, ...signals.weekly, ...signals.monthly];
    
    for (const signal of allSignals) {
      this.addEdge({
        from: `signal_${signal.agentType}_${signal.horizon}`,
        to: `horizon_${signal.horizon}`,
        weight: signal.strength * signal.confidence,
        type: 'signal_to_horizon',
        metadata: { stabilityScore: signal.stabilityScore },
      });
    }
  }
  
  private addHorizonToRegimeEdges(currentRegime: MarketRegime): void {
    const horizons: TimeHorizon[] = ['hourly', 'weekly', 'monthly'];
    
    for (const horizon of horizons) {
      const horizonNode = this.nodes.get(`horizon_${horizon}`) as HorizonNode;
      if (!horizonNode) continue;
      
      this.addEdge({
        from: `horizon_${horizon}`,
        to: `regime_${currentRegime}`,
        weight: horizonNode.avgStability,
        type: 'horizon_to_regime',
        metadata: { signalCount: horizonNode.signalCount },
      });
    }
  }
  
  private addRegimeToStrategyEdges(
    currentRegime: MarketRegime,
    strategies: {
      type: string;
      suitableRegimes: MarketRegime[];
      suitableHorizons: TimeHorizon[];
      expectedSharpe: Map<MarketRegime, number>;
    }[]
  ): void {
    for (const strategy of strategies) {
      // Only connect if strategy is suitable for this regime
      if (strategy.suitableRegimes.includes(currentRegime)) {
        const expectedSharpe = strategy.expectedSharpe.get(currentRegime) || 0;
        const weight = Math.max(0, Math.min(1, expectedSharpe / 2.0));  // Normalize
        
        this.addEdge({
          from: `regime_${currentRegime}`,
          to: `strategy_${strategy.type}`,
          weight,
          type: 'regime_to_strategy',
          metadata: { expectedSharpe },
        });
      }
    }
  }
  
  private addCrossSignalEdges(signals: {
    hourly: HorizonIndexedSignal[];
    weekly: HorizonIndexedSignal[];
    monthly: HorizonIndexedSignal[];
  }): void {
    const allSignals = [...signals.hourly, ...signals.weekly, ...signals.monthly];
    
    for (let i = 0; i < allSignals.length; i++) {
      for (let j = i + 1; j < allSignals.length; j++) {
        const s1 = allSignals[i];
        const s2 = allSignals[j];
        
        // Calculate survival probability (how well signals reinforce each other)
        const survivalProb = this.calculateSurvivalProbability(s1, s2);
        
        if (survivalProb > 0.3) {  // Only add significant edges
          this.addEdge({
            from: `signal_${s1.agentType}_${s1.horizon}`,
            to: `signal_${s2.agentType}_${s2.horizon}`,
            weight: survivalProb,
            type: 'survival_prob',
            metadata: { 
              correlation: s1.direction * s2.direction,
              horizonDiff: s1.horizon === s2.horizon ? 0 : 1,
            },
          });
        }
      }
    }
  }
  
  private addRegimeTransitionEdges(currentRegime: MarketRegime): void {
    const allRegimes: MarketRegime[] = ['CRISIS', 'DEFENSIVE', 'NEUTRAL', 'RISK_ON', 'HIGH_CONVICTION'];
    
    for (const targetRegime of allRegimes) {
      if (targetRegime === currentRegime) continue;
      
      const transitionProb = this.getRegimeTransitionProbability(currentRegime, targetRegime);
      
      if (transitionProb > 0.05) {  // Only add likely transitions
        this.addEdge({
          from: `regime_${currentRegime}`,
          to: `regime_${targetRegime}`,
          weight: transitionProb,
          type: 'regime_transition',
          metadata: { historicalFrequency: transitionProb },
        });
      }
    }
  }
  
  // ============================================================================
  // Sharpe Decay Calculation
  // ============================================================================
  
  /**
   * Calculate Sharpe decay rate for a signal
   * 
   * Measures how quickly signal's Sharpe ratio degrades over time
   */
  calculateSharpeDecay(signal: HorizonIndexedSignal): number {
    const key = `${signal.agentType}_${signal.horizon}`;
    
    // Get historical Sharpe values for this signal
    const history = this.sharpeDecayHistory.get(key) || [];
    
    if (history.length < 2) {
      // Default decay based on horizon
      const defaultDecay = {
        hourly: 0.15,   // Fast decay
        weekly: 0.08,   // Moderate decay
        monthly: 0.03,  // Slow decay
      };
      return defaultDecay[signal.horizon];
    }
    
    // Calculate decay from historical data
    // Sharpe(t) = Sharpe(0) * e^(-decay * t)
    // decay = -ln(Sharpe(t) / Sharpe(0)) / t
    
    const oldest = history[0];
    const newest = history[history.length - 1];
    
    const timeElapsedHours = (newest.timestamp.getTime() - oldest.timestamp.getTime()) / (1000 * 60 * 60);
    
    if (timeElapsedHours < 1 || oldest.sharpe === 0) {
      return 0.10;  // Default
    }
    
    const decayRate = -Math.log(Math.max(0.01, newest.sharpe) / oldest.sharpe) / timeElapsedHours;
    
    return Math.max(0, Math.min(1, decayRate));
  }
  
  /**
   * Update Sharpe decay history for a signal
   */
  updateSharpeHistory(agentType: string, horizon: TimeHorizon, sharpe: number): void {
    const key = `${agentType}_${horizon}`;
    const history = this.sharpeDecayHistory.get(key) || [];
    
    history.push({ timestamp: new Date(), sharpe });
    
    // Keep last 50 data points
    if (history.length > 50) {
      history.shift();
    }
    
    this.sharpeDecayHistory.set(key, history);
  }
  
  // ============================================================================
  // Regime Fragility Assessment
  // ============================================================================
  
  /**
   * Assess regime fragility (likelihood of transition)
   * 
   * High fragility = regime is unstable, likely to change soon
   * Low fragility = regime is stable
   */
  assessRegimeFragility(currentRegime: MarketRegime, volatility: number): number {
    // Base fragility from volatility
    let fragility = 0;
    
    // High volatility = high fragility
    if (volatility > 0.035) {
      fragility += 0.4;
    } else if (volatility > 0.025) {
      fragility += 0.25;
    } else if (volatility > 0.015) {
      fragility += 0.15;
    }
    
    // Regime-specific fragility
    const regimeFragilityMap: Record<MarketRegime, number> = {
      CRISIS: 0.35,          // Very unstable, likely to transition
      DEFENSIVE: 0.20,       // Moderately stable
      NEUTRAL: 0.25,         // Moderately unstable
      RISK_ON: 0.30,         // Can reverse quickly
      HIGH_CONVICTION: 0.15, // Most stable
    };
    
    fragility += regimeFragilityMap[currentRegime] || 0.25;
    
    // Historical transition frequency
    const recentTransitions = this.countRecentTransitions(10);
    fragility += Math.min(0.3, recentTransitions * 0.05);
    
    return Math.max(0, Math.min(1, fragility));
  }
  
  private countRecentTransitions(lookbackCount: number): number {
    if (this.regimeHistory.length < 2) return 0;
    
    const recentHistory = this.regimeHistory.slice(-lookbackCount);
    let transitions = 0;
    
    for (let i = 1; i < recentHistory.length; i++) {
      if (recentHistory[i].regime !== recentHistory[i - 1].regime) {
        transitions++;
      }
    }
    
    return transitions;
  }
  
  // ============================================================================
  // Horizon-Regime Agreement
  // ============================================================================
  
  /**
   * Calculate how well signals from a horizon agree with current regime
   */
  private calculateHorizonRegimeAgreement(
    signals: HorizonIndexedSignal[],
    regime: MarketRegime
  ): number {
    if (signals.length === 0) return 0.5;
    
    // Expected signal direction for each regime
    const regimeExpectedDirection: Record<MarketRegime, number> = {
      CRISIS: -1,
      DEFENSIVE: -0.5,
      NEUTRAL: 0,
      RISK_ON: 0.5,
      HIGH_CONVICTION: 1,
    };
    
    const expectedDirection = regimeExpectedDirection[regime];
    
    // Calculate average signal direction weighted by confidence
    let weightedSum = 0;
    let weightSum = 0;
    
    for (const signal of signals) {
      const weight = signal.confidence * signal.stabilityScore;
      weightedSum += signal.direction * weight;
      weightSum += weight;
    }
    
    const avgDirection = weightSum > 0 ? weightedSum / weightSum : 0;
    
    // Agreement score (1 = perfect agreement, 0 = complete disagreement)
    const agreement = 1 - Math.abs(avgDirection - expectedDirection) / 2;
    
    return Math.max(0, Math.min(1, agreement));
  }
  
  // ============================================================================
  // Survival Probability
  // ============================================================================
  
  private calculateSurvivalProbability(
    signal1: HorizonIndexedSignal,
    signal2: HorizonIndexedSignal
  ): number {
    // Signals reinforce each other if:
    // 1. Same direction
    // 2. High confidence
    // 3. Similar stability scores
    
    const directionAlignment = signal1.direction === signal2.direction ? 1.0 : 0.0;
    const avgConfidence = (signal1.confidence + signal2.confidence) / 2;
    const stabilityAlignment = 1 - Math.abs(signal1.stabilityScore - signal2.stabilityScore);
    
    // Horizon diversity bonus (cross-horizon signals are valuable)
    const horizonBonus = signal1.horizon !== signal2.horizon ? 0.2 : 0;
    
    const survivalProb = 
      0.4 * directionAlignment +
      0.3 * avgConfidence +
      0.2 * stabilityAlignment +
      0.1 + horizonBonus;
    
    return Math.max(0, Math.min(1, survivalProb));
  }
  
  // ============================================================================
  // Regime Transition Probability
  // ============================================================================
  
  private initializeRegimeTransitionMatrix(): void {
    const regimes: MarketRegime[] = ['CRISIS', 'DEFENSIVE', 'NEUTRAL', 'RISK_ON', 'HIGH_CONVICTION'];
    
    for (const from of regimes) {
      const transitions = new Map<MarketRegime, number>();
      for (const to of regimes) {
        transitions.set(to, 0.0);
      }
      this.regimeTransitionMatrix.set(from, transitions);
    }
  }
  
  private getRegimeTransitionProbability(from: MarketRegime, to: MarketRegime): number {
    // Get historical transition probability
    const fromMap = this.regimeTransitionMatrix.get(from);
    if (!fromMap) return 0.05;
    
    const historicalProb = fromMap.get(to) || 0.05;
    return historicalProb;
  }
  
  /**
   * Update transition matrix based on observed transition
   */
  updateRegimeTransition(from: MarketRegime, to: MarketRegime): void {
    const fromMap = this.regimeTransitionMatrix.get(from);
    if (!fromMap) return;
    
    const currentProb = fromMap.get(to) || 0.0;
    const updatedProb = currentProb * 0.95 + 0.05;  // Exponential moving average
    fromMap.set(to, updatedProb);
  }
  
  // ============================================================================
  // Strategy Compatibility
  // ============================================================================
  
  /**
   * Get best strategies for a given (regime, horizon) pair
   */
  getBestStrategies(regime: MarketRegime, horizon: TimeHorizon): StrategyNode[] {
    const strategies: StrategyNode[] = [];
    
    for (const node of this.nodes.values()) {
      if (node.type === 'strategy') {
        const strategyNode = node as StrategyNode;
        
        // Check if strategy is suitable for this regime and horizon
        if (
          strategyNode.suitableRegimes.includes(regime) &&
          strategyNode.suitableHorizons.includes(horizon)
        ) {
          strategies.push(strategyNode);
        }
      }
    }
    
    // Sort by expected Sharpe for this regime
    strategies.sort((a, b) => {
      const sharpeA = a.expectedSharpe.get(regime) || 0;
      const sharpeB = b.expectedSharpe.get(regime) || 0;
      return sharpeB - sharpeA;
    });
    
    return strategies;
  }
  
  /**
   * Calculate strategy compatibility score for current state
   */
  calculateStrategyCompatibility(
    strategyType: string,
    currentRegime: MarketRegime,
    horizonWeights: { hourly: number; weekly: number; monthly: number }
  ): number {
    const strategyNode = this.nodes.get(`strategy_${strategyType}`) as StrategyNode;
    if (!strategyNode) return 0;
    
    // Regime compatibility
    const regimeCompat = strategyNode.suitableRegimes.includes(currentRegime) ? 1.0 : 0.3;
    
    // Horizon compatibility (weighted by current horizon allocation)
    let horizonCompat = 0;
    for (const horizon of strategyNode.suitableHorizons) {
      horizonCompat += horizonWeights[horizon];
    }
    horizonCompat /= strategyNode.suitableHorizons.length;
    
    // Expected Sharpe
    const expectedSharpe = strategyNode.expectedSharpe.get(currentRegime) || 0;
    const sharpeScore = Math.max(0, Math.min(1, expectedSharpe / 2.0));
    
    return 0.4 * regimeCompat + 0.3 * horizonCompat + 0.3 * sharpeScore;
  }
  
  // ============================================================================
  // Graph Utilities
  // ============================================================================
  
  private addNode(node: HorizonGraphNode): void {
    this.nodes.set(node.id, node);
    this.adjacencyList.set(node.id, []);
  }
  
  private addEdge(edge: HorizonGraphEdge): void {
    this.edges.push(edge);
    
    // Update adjacency list
    const neighbors = this.adjacencyList.get(edge.from) || [];
    neighbors.push(edge.to);
    this.adjacencyList.set(edge.from, neighbors);
  }
  
  private clear(): void {
    this.nodes.clear();
    this.edges = [];
    this.adjacencyList.clear();
  }
  
  // Getters
  getNodes(): Map<string, HorizonGraphNode> {
    return this.nodes;
  }
  
  getEdges(): HorizonGraphEdge[] {
    return this.edges;
  }
  
  getAdjacencyList(): Map<string, string[]> {
    return this.adjacencyList;
  }
  
  getRegimeNode(regime: MarketRegime): RegimeNode | undefined {
    return this.nodes.get(`regime_${regime}`) as RegimeNode;
  }
  
  getSignalNode(agentType: string, horizon: TimeHorizon): HorizonSignalNode | undefined {
    return this.nodes.get(`signal_${agentType}_${horizon}`) as HorizonSignalNode;
  }
}
