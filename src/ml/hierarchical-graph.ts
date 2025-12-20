/**
 * Hierarchical Signal-Regime Graph
 * 
 * Constructs a hierarchical graph with:
 * - Nodes: Signals, Regimes, Strategies
 * - Edges: Conditional dependence, Survival probability, 
 *          Regime transitions, Strategy compatibility
 */

import { NormalizedSignal } from './signal-pool';
import { MarketRegime } from './market-regime-detection';

export interface GraphNode {
  id: string;
  type: 'signal' | 'regime' | 'strategy';
  label: string;
  metadata: any;
}

export interface SignalNode extends GraphNode {
  type: 'signal';
  agentId: string;
  direction: -1 | 0 | 1;
  strength: number;
  timestamp: Date;
}

export interface RegimeNode extends GraphNode {
  type: 'regime';
  regime: MarketRegime;
  confidence: number;
  volatility: number;
}

export interface StrategyNode extends GraphNode {
  type: 'strategy';
  strategyType: string;
  expectedReturn: number;
  riskLevel: number;
}

export interface GraphEdge {
  from: string;  // Node ID
  to: string;    // Node ID
  weight: number;
  type: 'conditional_dependence' | 'survival_prob' | 'regime_transition' | 'strategy_compat';
  metadata?: any;
}

export interface HierarchicalGraphStructure {
  nodes: Map<string, GraphNode>;
  edges: GraphEdge[];
  adjacencyList: Map<string, string[]>;
  regimeTransitionMatrix: Map<MarketRegime, Map<MarketRegime, number>>;
}

export class HierarchicalGraph {
  private nodes: Map<string, GraphNode> = new Map();
  private edges: GraphEdge[] = [];
  private adjacencyList: Map<string, string[]> = new Map();
  private regimeTransitionMatrix: Map<MarketRegime, Map<MarketRegime, number>> = new Map();
  
  // Historical data for transition probabilities
  private regimeHistory: { regime: MarketRegime; timestamp: Date }[] = [];
  private maxHistoryLength = 1000;

  constructor() {
    this.initializeRegimeTransitionMatrix();
  }

  /**
   * Build graph from current system state
   */
  buildGraph(
    signals: NormalizedSignal[],
    currentRegime: MarketRegime,
    regimeConfidence: number,
    regimeVolatility: number,
    strategies: { type: string; expectedReturn: number; riskLevel: number }[]
  ): void {
    // Clear previous graph
    this.clear();

    // 1. Add Signal Nodes
    signals.forEach(signal => {
      const node: SignalNode = {
        id: `signal_${signal.agentId}`,
        type: 'signal',
        label: `${signal.agentId} Signal`,
        agentId: signal.agentId,
        direction: signal.direction,
        strength: signal.strength,
        timestamp: signal.timestamp,
        metadata: {
          rawSignal: signal.rawSignal,
          explanation: signal.explanation
        }
      };
      this.addNode(node);
    });

    // 2. Add Regime Node (current)
    const regimeNode: RegimeNode = {
      id: `regime_${currentRegime}`,
      type: 'regime',
      label: currentRegime,
      regime: currentRegime,
      confidence: regimeConfidence,
      volatility: regimeVolatility,
      metadata: {}
    };
    this.addNode(regimeNode);

    // 3. Add Strategy Nodes
    strategies.forEach((strategy, idx) => {
      const node: StrategyNode = {
        id: `strategy_${idx}_${strategy.type}`,
        type: 'strategy',
        label: strategy.type,
        strategyType: strategy.type,
        expectedReturn: strategy.expectedReturn,
        riskLevel: strategy.riskLevel,
        metadata: {}
      };
      this.addNode(node);
    });

    // 4. Add Signal → Regime Edges (Conditional Dependence)
    signals.forEach(signal => {
      const strength = this.calculateSignalRegimeDependence(signal, currentRegime);
      this.addEdge({
        from: `signal_${signal.agentId}`,
        to: `regime_${currentRegime}`,
        weight: strength,
        type: 'conditional_dependence',
        metadata: { signalStrength: signal.strength }
      });
    });

    // 5. Add Signal → Signal Edges (Survival Probability)
    for (let i = 0; i < signals.length; i++) {
      for (let j = i + 1; j < signals.length; j++) {
        const survivalProb = this.calculateSurvivalProbability(signals[i], signals[j]);
        if (survivalProb > 0.3) {  // Only add significant edges
          this.addEdge({
            from: `signal_${signals[i].agentId}`,
            to: `signal_${signals[j].agentId}`,
            weight: survivalProb,
            type: 'survival_prob',
            metadata: { 
              agreement: signals[i].direction === signals[j].direction
            }
          });
        }
      }
    }

    // 6. Add Regime → Regime Edges (Regime Transitions)
    this.addRegimeTransitionEdges(currentRegime);

    // 7. Add Regime → Strategy Edges (Strategy Compatibility)
    strategies.forEach((strategy, idx) => {
      const compatibility = this.calculateStrategyCompatibility(strategy, currentRegime);
      this.addEdge({
        from: `regime_${currentRegime}`,
        to: `strategy_${idx}_${strategy.type}`,
        weight: compatibility,
        type: 'strategy_compat',
        metadata: { expectedReturn: strategy.expectedReturn }
      });
    });

    // Update regime history
    this.updateRegimeHistory(currentRegime);
  }

  /**
   * Calculate conditional dependence between signal and regime
   */
  private calculateSignalRegimeDependence(signal: NormalizedSignal, regime: MarketRegime): number {
    // Dependence based on signal strength and regime alignment
    const regimeMapping = {
      [MarketRegime.CRISIS_STRESS]: -1,
      [MarketRegime.DEFENSIVE]: -0.5,
      [MarketRegime.NEUTRAL]: 0,
      [MarketRegime.RISK_ON]: 0.5,
      [MarketRegime.HIGH_CONVICTION]: 1
    };

    const regimeValue = regimeMapping[regime];
    const signalRegimeAlignment = 1 - Math.abs(signal.direction - regimeValue);
    
    return signal.strength * signalRegimeAlignment;
  }

  /**
   * Calculate survival probability between two signals
   * (likelihood both signals persist together)
   */
  private calculateSurvivalProbability(signal1: NormalizedSignal, signal2: NormalizedSignal): number {
    // Signals with same direction and high strength have high survival probability
    if (signal1.direction === signal2.direction) {
      return (signal1.strength + signal2.strength) / 2;
    }
    
    // Conflicting signals have low survival probability
    if (signal1.direction === -signal2.direction) {
      return 0.1;
    }
    
    // Neutral signals have moderate survival probability
    return 0.5;
  }

  /**
   * Calculate strategy compatibility with regime
   */
  private calculateStrategyCompatibility(
    strategy: { type: string; expectedReturn: number; riskLevel: number },
    regime: MarketRegime
  ): number {
    // Define strategy-regime compatibility matrix
    const compatibility: Record<string, Partial<Record<MarketRegime, number>>> = {
      'cross_exchange_spread': {
        [MarketRegime.CRISIS_STRESS]: 0.9,  // High vol = wide spreads
        [MarketRegime.NEUTRAL]: 0.7,
        [MarketRegime.RISK_ON]: 0.5
      },
      'funding_rate_carry': {
        [MarketRegime.RISK_ON]: 0.9,
        [MarketRegime.HIGH_CONVICTION]: 0.8,
        [MarketRegime.NEUTRAL]: 0.6
      },
      'volatility_basis': {
        [MarketRegime.CRISIS_STRESS]: 0.95,
        [MarketRegime.DEFENSIVE]: 0.7,
        [MarketRegime.RISK_ON]: 0.4
      },
      'stat_arb': {
        [MarketRegime.NEUTRAL]: 0.9,
        [MarketRegime.DEFENSIVE]: 0.7,
        [MarketRegime.RISK_ON]: 0.6
      }
    };

    return compatibility[strategy.type]?.[regime] ?? 0.5;
  }

  /**
   * Add regime transition edges based on historical transitions
   */
  private addRegimeTransitionEdges(currentRegime: MarketRegime): void {
    const allRegimes = [
      MarketRegime.CRISIS_STRESS,
      MarketRegime.DEFENSIVE,
      MarketRegime.NEUTRAL,
      MarketRegime.RISK_ON,
      MarketRegime.HIGH_CONVICTION
    ];

    allRegimes.forEach(targetRegime => {
      if (targetRegime === currentRegime) return;

      const transitionProb = this.getTransitionProbability(currentRegime, targetRegime);
      
      if (transitionProb > 0.05) {  // Only add significant transitions
        // Add target regime node if not exists
        if (!this.nodes.has(`regime_${targetRegime}`)) {
          const regimeNode: RegimeNode = {
            id: `regime_${targetRegime}`,
            type: 'regime',
            label: targetRegime,
            regime: targetRegime,
            confidence: 0,
            volatility: 0,
            metadata: { isPotentialTransition: true }
          };
          this.addNode(regimeNode);
        }

        this.addEdge({
          from: `regime_${currentRegime}`,
          to: `regime_${targetRegime}`,
          weight: transitionProb,
          type: 'regime_transition',
          metadata: { transitionProbability: transitionProb }
        });
      }
    });
  }

  /**
   * Initialize regime transition matrix with priors
   */
  private initializeRegimeTransitionMatrix(): void {
    const regimes = [
      MarketRegime.CRISIS_STRESS,
      MarketRegime.DEFENSIVE,
      MarketRegime.NEUTRAL,
      MarketRegime.RISK_ON,
      MarketRegime.HIGH_CONVICTION
    ];

    // Initialize with uniform priors (will be updated with historical data)
    regimes.forEach(from => {
      const transitions = new Map<MarketRegime, number>();
      regimes.forEach(to => {
        if (from === to) {
          transitions.set(to, 0.6); // Persistence probability
        } else {
          transitions.set(to, 0.1); // Equal transition probability to others
        }
      });
      this.regimeTransitionMatrix.set(from, transitions);
    });
  }

  /**
   * Update regime transition matrix with new observation
   */
  private updateRegimeHistory(newRegime: MarketRegime): void {
    this.regimeHistory.push({ regime: newRegime, timestamp: new Date() });
    
    if (this.regimeHistory.length > this.maxHistoryLength) {
      this.regimeHistory.shift();
    }

    // Recompute transition matrix from history
    if (this.regimeHistory.length > 10) {
      this.computeTransitionMatrixFromHistory();
    }
  }

  /**
   * Compute transition matrix from historical regime sequence
   */
  private computeTransitionMatrixFromHistory(): void {
    const transitionCounts = new Map<MarketRegime, Map<MarketRegime, number>>();
    
    // Initialize counts
    const regimes = [
      MarketRegime.CRISIS_STRESS,
      MarketRegime.DEFENSIVE,
      MarketRegime.NEUTRAL,
      MarketRegime.RISK_ON,
      MarketRegime.HIGH_CONVICTION
    ];
    
    regimes.forEach(from => {
      const counts = new Map<MarketRegime, number>();
      regimes.forEach(to => counts.set(to, 0));
      transitionCounts.set(from, counts);
    });

    // Count transitions
    for (let i = 0; i < this.regimeHistory.length - 1; i++) {
      const from = this.regimeHistory[i].regime;
      const to = this.regimeHistory[i + 1].regime;
      
      const currentCount = transitionCounts.get(from)!.get(to)!;
      transitionCounts.get(from)!.set(to, currentCount + 1);
    }

    // Convert counts to probabilities
    regimes.forEach(from => {
      const counts = transitionCounts.get(from)!;
      const total = Array.from(counts.values()).reduce((a, b) => a + b, 0);
      
      if (total > 0) {
        const probs = new Map<MarketRegime, number>();
        counts.forEach((count, to) => {
          probs.set(to, count / total);
        });
        this.regimeTransitionMatrix.set(from, probs);
      }
    });
  }

  /**
   * Get transition probability from one regime to another
   */
  getTransitionProbability(from: MarketRegime, to: MarketRegime): number {
    return this.regimeTransitionMatrix.get(from)?.get(to) ?? 0.1;
  }

  /**
   * Get all transition probabilities from current regime
   */
  getTransitionProbabilities(currentRegime: MarketRegime): Map<MarketRegime, number> {
    return this.regimeTransitionMatrix.get(currentRegime) ?? new Map();
  }

  /**
   * Add node to graph
   */
  private addNode(node: GraphNode): void {
    this.nodes.set(node.id, node);
    this.adjacencyList.set(node.id, []);
  }

  /**
   * Add edge to graph
   */
  private addEdge(edge: GraphEdge): void {
    this.edges.push(edge);
    this.adjacencyList.get(edge.from)?.push(edge.to);
  }

  /**
   * Clear graph
   */
  private clear(): void {
    this.nodes.clear();
    this.edges.clear();
    this.adjacencyList.clear();
  }

  /**
   * Get graph structure
   */
  getGraphStructure(): HierarchicalGraphStructure {
    return {
      nodes: this.nodes,
      edges: this.edges,
      adjacencyList: this.adjacencyList,
      regimeTransitionMatrix: this.regimeTransitionMatrix
    };
  }

  /**
   * Get nodes of specific type
   */
  getNodesByType(type: 'signal' | 'regime' | 'strategy'): GraphNode[] {
    return Array.from(this.nodes.values()).filter(n => n.type === type);
  }

  /**
   * Get edges of specific type
   */
  getEdgesByType(type: GraphEdge['type']): GraphEdge[] {
    return this.edges.filter(e => e.type === type);
  }

  /**
   * Get neighbors of a node
   */
  getNeighbors(nodeId: string): GraphNode[] {
    const neighborIds = this.adjacencyList.get(nodeId) ?? [];
    return neighborIds.map(id => this.nodes.get(id)!).filter(Boolean);
  }

  /**
   * Get compatible strategies for regime
   */
  getCompatibleStrategies(regime: MarketRegime): StrategyNode[] {
    const regimeNodeId = `regime_${regime}`;
    const strategyEdges = this.edges.filter(e => 
      e.from === regimeNodeId && e.type === 'strategy_compat' && e.weight > 0.5
    );
    
    return strategyEdges
      .map(e => this.nodes.get(e.to) as StrategyNode)
      .filter(Boolean)
      .sort((a, b) => b.expectedReturn - a.expectedReturn);
  }

  /**
   * Get regime for signal (based on strongest conditional dependence)
   */
  getRegimeForSignal(signalId: string): MarketRegime | null {
    const regimeEdges = this.edges.filter(e => 
      e.from === signalId && e.type === 'conditional_dependence'
    );
    
    if (regimeEdges.length === 0) return null;
    
    const strongestEdge = regimeEdges.reduce((max, edge) => 
      edge.weight > max.weight ? edge : max
    );
    
    const regimeNode = this.nodes.get(strongestEdge.to) as RegimeNode;
    return regimeNode?.regime ?? null;
  }
}
