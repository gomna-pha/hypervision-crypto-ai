/**
 * Multi-Horizon Signal Aggregation
 * 
 * Aggregates signals from all 15 agents (5 types × 3 horizons) with horizon indexing.
 * 
 * Signal Structure: sᵢ(t, h) ∈ {-1, 0, +1}
 * Indexed by: [Agent Type] × [Horizon] × [Time]
 * 
 * Key Features:
 * - Horizon-indexed signal pool
 * - Cross-horizon consistency checking
 * - Signal diversity calculation
 * - Correlation management
 * - Disagreement detection
 */

import { AgentSignal } from './agent-signal';
import { TimeHorizon } from './time-scale-feature-store';

// ============================================================================
// Horizon-Indexed Signal Interface
// ============================================================================

export interface HorizonIndexedSignal extends AgentSignal {
  horizon: TimeHorizon;
  stabilityScore: number;  // Cross-horizon consistency [0, 1]
  horizonWeight: number;   // Dynamic weight based on conditions [0, 1]
}

// ============================================================================
// Signal Aggregation Metrics
// ============================================================================

export interface SignalAggregationMetrics {
  // Signal diversity
  diversity: number;  // [0, 1]: 0 = all same, 1 = maximum diversity
  
  // Consensus metrics
  consensus: {
    direction: -1 | 0 | 1;  // Overall consensus direction
    strength: number;       // [0, 1]: How strong is the consensus
    confidence: number;     // [0, 1]: Average confidence
  };
  
  // Correlation metrics
  correlation: {
    overall: number;              // [-1, 1]: Average correlation
    byHorizon: Map<TimeHorizon, number>;
    byAgentType: Map<string, number>;
  };
  
  // Disagreement detection
  disagreements: {
    count: number;
    severity: 'low' | 'medium' | 'high';
    details: {
      signal1: HorizonIndexedSignal;
      signal2: HorizonIndexedSignal;
      divergence: number;
    }[];
  };
  
  // Cross-horizon metrics
  horizonAlignment: {
    hourly_weekly: number;   // [-1, 1]
    weekly_monthly: number;  // [-1, 1]
    hourly_monthly: number;  // [-1, 1]
    overall: number;         // [-1, 1]
  };
}

// ============================================================================
// Multi-Horizon Signal Pool
// ============================================================================

export class MultiHorizonSignalPool {
  // Signal storage: Map<horizon, Map<agentType, signal>>
  private signals: Map<TimeHorizon, Map<string, HorizonIndexedSignal>> = new Map();
  
  // Historical signals for decay tracking
  private signalHistory: HorizonIndexedSignal[] = [];
  private maxHistoryLength = 100;
  
  constructor() {
    // Initialize signal maps for each horizon
    this.signals.set('hourly', new Map());
    this.signals.set('weekly', new Map());
    this.signals.set('monthly', new Map());
  }
  
  // ============================================================================
  // Signal Ingestion
  // ============================================================================
  
  /**
   * Add signals from all horizons
   */
  addSignals(
    hourlySignals: AgentSignal[],
    weeklySignals: AgentSignal[],
    monthlySignals: AgentSignal[],
    horizonWeights: { hourly: number; weekly: number; monthly: number } = { hourly: 0.4, weekly: 0.4, monthly: 0.2 }
  ): void {
    // Process hourly signals
    this.processSignalsForHorizon(hourlySignals, 'hourly', horizonWeights.hourly);
    
    // Process weekly signals
    this.processSignalsForHorizon(weeklySignals, 'weekly', horizonWeights.weekly);
    
    // Process monthly signals
    this.processSignalsForHorizon(monthlySignals, 'monthly', horizonWeights.monthly);
    
    // Calculate stability scores (cross-horizon consistency)
    this.updateStabilityScores();
  }
  
  private processSignalsForHorizon(
    signals: AgentSignal[],
    horizon: TimeHorizon,
    horizonWeight: number
  ): void {
    const horizonMap = this.signals.get(horizon)!;
    
    for (const signal of signals) {
      const horizonSignal: HorizonIndexedSignal = {
        ...signal,
        horizon,
        stabilityScore: 0.5, // Will be updated
        horizonWeight,
      };
      
      horizonMap.set(signal.agentType, horizonSignal);
      
      // Add to history
      this.signalHistory.push(horizonSignal);
      
      // Trim history if too long
      if (this.signalHistory.length > this.maxHistoryLength) {
        this.signalHistory.shift();
      }
    }
  }
  
  /**
   * Update stability scores based on cross-horizon consistency
   */
  private updateStabilityScores(): void {
    const agentTypes: Array<'economic' | 'sentiment' | 'crossExchange' | 'onChain' | 'cnnPattern'> = [
      'economic', 'sentiment', 'crossExchange', 'onChain', 'cnnPattern'
    ];
    
    for (const agentType of agentTypes) {
      const hourlySignal = this.signals.get('hourly')?.get(agentType);
      const weeklySignal = this.signals.get('weekly')?.get(agentType);
      const monthlySignal = this.signals.get('monthly')?.get(agentType);
      
      if (hourlySignal && weeklySignal && monthlySignal) {
        // Calculate 3-way alignment
        const alignment = this.calculateThreeWayAlignment(
          hourlySignal.direction,
          weeklySignal.direction,
          monthlySignal.direction
        );
        
        // Update stability scores (higher = more aligned)
        hourlySignal.stabilityScore = alignment;
        weeklySignal.stabilityScore = alignment;
        monthlySignal.stabilityScore = alignment;
      }
    }
  }
  
  private calculateThreeWayAlignment(s1: number, s2: number, s3: number): number {
    // Perfect alignment: all same sign
    if ((s1 === s2 && s2 === s3)) return 1.0;
    
    // Two aligned, one different
    if (s1 === s2 || s2 === s3 || s1 === s3) return 0.6;
    
    // All different
    return 0.2;
  }
  
  // ============================================================================
  // Signal Retrieval
  // ============================================================================
  
  /**
   * Get all signals for a specific horizon
   */
  getSignalsByHorizon(horizon: TimeHorizon): HorizonIndexedSignal[] {
    const horizonMap = this.signals.get(horizon);
    if (!horizonMap) return [];
    return Array.from(horizonMap.values());
  }
  
  /**
   * Get signals for a specific agent type across all horizons
   */
  getSignalsByAgentType(agentType: string): HorizonIndexedSignal[] {
    const signals: HorizonIndexedSignal[] = [];
    
    for (const [horizon, horizonMap] of this.signals.entries()) {
      const signal = horizonMap.get(agentType);
      if (signal) signals.push(signal);
    }
    
    return signals;
  }
  
  /**
   * Get all signals (flattened)
   */
  getAllSignals(): HorizonIndexedSignal[] {
    const allSignals: HorizonIndexedSignal[] = [];
    
    for (const horizonMap of this.signals.values()) {
      allSignals.push(...Array.from(horizonMap.values()));
    }
    
    return allSignals;
  }
  
  /**
   * Get signals filtered by direction
   */
  getSignalsByDirection(direction: -1 | 0 | 1): HorizonIndexedSignal[] {
    return this.getAllSignals().filter(s => s.direction === direction);
  }
  
  // ============================================================================
  // Signal Aggregation Metrics
  // ============================================================================
  
  /**
   * Calculate comprehensive aggregation metrics
   */
  calculateMetrics(): SignalAggregationMetrics {
    const allSignals = this.getAllSignals();
    
    return {
      diversity: this.calculateDiversity(allSignals),
      consensus: this.calculateConsensus(allSignals),
      correlation: this.calculateCorrelation(),
      disagreements: this.detectDisagreements(allSignals),
      horizonAlignment: this.calculateHorizonAlignment(),
    };
  }
  
  /**
   * Calculate signal diversity (0 = all same, 1 = maximum diversity)
   */
  private calculateDiversity(signals: HorizonIndexedSignal[]): number {
    if (signals.length === 0) return 0;
    
    const bullish = signals.filter(s => s.direction === 1).length;
    const bearish = signals.filter(s => s.direction === -1).length;
    const neutral = signals.filter(s => s.direction === 0).length;
    
    const total = signals.length;
    
    // Shannon entropy
    const pBullish = bullish / total;
    const pBearish = bearish / total;
    const pNeutral = neutral / total;
    
    let entropy = 0;
    if (pBullish > 0) entropy -= pBullish * Math.log2(pBullish);
    if (pBearish > 0) entropy -= pBearish * Math.log2(pBearish);
    if (pNeutral > 0) entropy -= pNeutral * Math.log2(pNeutral);
    
    // Normalize to [0, 1] (max entropy for 3 categories = log2(3) ≈ 1.585)
    return entropy / 1.585;
  }
  
  /**
   * Calculate consensus direction and strength
   */
  private calculateConsensus(signals: HorizonIndexedSignal[]): {
    direction: -1 | 0 | 1;
    strength: number;
    confidence: number;
  } {
    if (signals.length === 0) {
      return { direction: 0, strength: 0, confidence: 0 };
    }
    
    // Weighted average by confidence and horizon weight
    let weightedSum = 0;
    let weightSum = 0;
    let confidenceSum = 0;
    
    for (const signal of signals) {
      const weight = signal.confidence * signal.horizonWeight * signal.stabilityScore;
      weightedSum += signal.direction * weight;
      weightSum += weight;
      confidenceSum += signal.confidence;
    }
    
    const avgDirection = weightSum > 0 ? weightedSum / weightSum : 0;
    const avgConfidence = confidenceSum / signals.length;
    
    // Determine consensus direction
    let direction: -1 | 0 | 1;
    if (avgDirection > 0.3) direction = 1;
    else if (avgDirection < -0.3) direction = -1;
    else direction = 0;
    
    // Consensus strength (how aligned are the signals)
    const strength = Math.abs(avgDirection);
    
    return {
      direction,
      strength,
      confidence: avgConfidence,
    };
  }
  
  /**
   * Calculate correlation metrics
   */
  private calculateCorrelation(): {
    overall: number;
    byHorizon: Map<TimeHorizon, number>;
    byAgentType: Map<string, number>;
  } {
    const allSignals = this.getAllSignals();
    
    // Overall correlation (average pairwise)
    let correlationSum = 0;
    let pairCount = 0;
    
    for (let i = 0; i < allSignals.length; i++) {
      for (let j = i + 1; j < allSignals.length; j++) {
        const corr = allSignals[i].direction * allSignals[j].direction;
        correlationSum += corr;
        pairCount++;
      }
    }
    
    const overall = pairCount > 0 ? correlationSum / pairCount : 0;
    
    // Correlation by horizon
    const byHorizon = new Map<TimeHorizon, number>();
    for (const horizon of ['hourly', 'weekly', 'monthly'] as TimeHorizon[]) {
      const horizonSignals = this.getSignalsByHorizon(horizon);
      if (horizonSignals.length >= 2) {
        let horizonCorrSum = 0;
        let horizonPairCount = 0;
        
        for (let i = 0; i < horizonSignals.length; i++) {
          for (let j = i + 1; j < horizonSignals.length; j++) {
            horizonCorrSum += horizonSignals[i].direction * horizonSignals[j].direction;
            horizonPairCount++;
          }
        }
        
        byHorizon.set(horizon, horizonPairCount > 0 ? horizonCorrSum / horizonPairCount : 0);
      }
    }
    
    // Correlation by agent type
    const byAgentType = new Map<string, number>();
    const agentTypes = ['economic', 'sentiment', 'crossExchange', 'onChain', 'cnnPattern'];
    
    for (const agentType of agentTypes) {
      const agentSignals = this.getSignalsByAgentType(agentType);
      if (agentSignals.length >= 2) {
        let agentCorrSum = 0;
        let agentPairCount = 0;
        
        for (let i = 0; i < agentSignals.length; i++) {
          for (let j = i + 1; j < agentSignals.length; j++) {
            agentCorrSum += agentSignals[i].direction * agentSignals[j].direction;
            agentPairCount++;
          }
        }
        
        byAgentType.set(agentType, agentPairCount > 0 ? agentCorrSum / agentPairCount : 0);
      }
    }
    
    return {
      overall,
      byHorizon,
      byAgentType,
    };
  }
  
  /**
   * Detect disagreements between signals
   */
  private detectDisagreements(signals: HorizonIndexedSignal[]): {
    count: number;
    severity: 'low' | 'medium' | 'high';
    details: {
      signal1: HorizonIndexedSignal;
      signal2: HorizonIndexedSignal;
      divergence: number;
    }[];
  } {
    const disagreements: {
      signal1: HorizonIndexedSignal;
      signal2: HorizonIndexedSignal;
      divergence: number;
    }[] = [];
    
    // Find opposing signals
    for (let i = 0; i < signals.length; i++) {
      for (let j = i + 1; j < signals.length; j++) {
        const s1 = signals[i];
        const s2 = signals[j];
        
        // Opposite directions with high confidence = disagreement
        if (s1.direction * s2.direction < 0 && s1.confidence > 0.6 && s2.confidence > 0.6) {
          const divergence = Math.abs(s1.direction - s2.direction) * Math.min(s1.confidence, s2.confidence);
          disagreements.push({
            signal1: s1,
            signal2: s2,
            divergence,
          });
        }
      }
    }
    
    // Determine severity
    let severity: 'low' | 'medium' | 'high' = 'low';
    if (disagreements.length >= 5) severity = 'high';
    else if (disagreements.length >= 3) severity = 'medium';
    
    return {
      count: disagreements.length,
      severity,
      details: disagreements.slice(0, 5), // Top 5 disagreements
    };
  }
  
  /**
   * Calculate cross-horizon alignment
   */
  private calculateHorizonAlignment(): {
    hourly_weekly: number;
    weekly_monthly: number;
    hourly_monthly: number;
    overall: number;
  } {
    const hourlySignals = this.getSignalsByHorizon('hourly');
    const weeklySignals = this.getSignalsByHorizon('weekly');
    const monthlySignals = this.getSignalsByHorizon('monthly');
    
    const hourlyAvg = this.averageSignalDirection(hourlySignals);
    const weeklyAvg = this.averageSignalDirection(weeklySignals);
    const monthlyAvg = this.averageSignalDirection(monthlySignals);
    
    // Calculate pairwise alignment
    const hourly_weekly = this.signalSimilarity(hourlyAvg, weeklyAvg);
    const weekly_monthly = this.signalSimilarity(weeklyAvg, monthlyAvg);
    const hourly_monthly = this.signalSimilarity(hourlyAvg, monthlyAvg);
    
    const overall = (hourly_weekly + weekly_monthly + hourly_monthly) / 3;
    
    return {
      hourly_weekly,
      weekly_monthly,
      hourly_monthly,
      overall,
    };
  }
  
  private averageSignalDirection(signals: HorizonIndexedSignal[]): number {
    if (signals.length === 0) return 0;
    const sum = signals.reduce((acc, s) => acc + s.direction * s.confidence, 0);
    return sum / signals.length;
  }
  
  private signalSimilarity(s1: number, s2: number): number {
    // Cosine similarity
    if (s1 === 0 || s2 === 0) return 0;
    return (s1 * s2) / (Math.abs(s1) * Math.abs(s2));
  }
  
  // ============================================================================
  // Signal Decay Management
  // ============================================================================
  
  /**
   * Apply signal decay based on time elapsed
   */
  applyDecay(hoursElapsed: number): void {
    for (const horizonMap of this.signals.values()) {
      for (const signal of horizonMap.values()) {
        // Exponential decay: strength(t) = strength(0) * e^(-decayRate * t)
        const decayFactor = Math.exp(-signal.decayRate * hoursElapsed);
        signal.strength *= decayFactor;
        
        // If strength drops below threshold, mark as neutral
        if (signal.strength < 0.1) {
          signal.direction = 0;
          signal.strength = 0;
        }
      }
    }
  }
  
  /**
   * Get signals that have decayed significantly
   */
  getDecayedSignals(threshold: number = 0.3): HorizonIndexedSignal[] {
    return this.getAllSignals().filter(s => s.strength < threshold && s.strength > 0);
  }
  
  // ============================================================================
  // Utilities
  // ============================================================================
  
  /**
   * Clear all signals
   */
  clear(): void {
    this.signals.get('hourly')?.clear();
    this.signals.get('weekly')?.clear();
    this.signals.get('monthly')?.clear();
    this.signalHistory = [];
  }
  
  /**
   * Get signal count
   */
  getSignalCount(): { total: number; byHorizon: Map<TimeHorizon, number> } {
    const byHorizon = new Map<TimeHorizon, number>();
    let total = 0;
    
    for (const [horizon, horizonMap] of this.signals.entries()) {
      const count = horizonMap.size;
      byHorizon.set(horizon, count);
      total += count;
    }
    
    return { total, byHorizon };
  }
  
  /**
   * Get signal summary
   */
  getSummary(): {
    totalSignals: number;
    bullishCount: number;
    bearishCount: number;
    neutralCount: number;
    avgConfidence: number;
    avgStability: number;
  } {
    const allSignals = this.getAllSignals();
    
    const bullishCount = allSignals.filter(s => s.direction === 1).length;
    const bearishCount = allSignals.filter(s => s.direction === -1).length;
    const neutralCount = allSignals.filter(s => s.direction === 0).length;
    
    const avgConfidence = allSignals.length > 0
      ? allSignals.reduce((sum, s) => sum + s.confidence, 0) / allSignals.length
      : 0;
    
    const avgStability = allSignals.length > 0
      ? allSignals.reduce((sum, s) => sum + s.stabilityScore, 0) / allSignals.length
      : 0;
    
    return {
      totalSignals: allSignals.length,
      bullishCount,
      bearishCount,
      neutralCount,
      avgConfidence,
      avgStability,
    };
  }
}
