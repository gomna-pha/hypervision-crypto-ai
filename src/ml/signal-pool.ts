/**
 * Signal Pool - Multi-Agent Signal Aggregation
 * 
 * Aggregates signals from all 5 agents into a normalized pool
 * with {-1, 0, +1} direction encoding and diversity tracking.
 */

import { AgentSignal } from './agent-signal';

export interface NormalizedSignal {
  agentId: string;
  direction: -1 | 0 | 1;  // Short | Neutral | Long
  strength: number;        // [0, 1] confidence
  timestamp: Date;
  rawSignal: number;       // Original signal value
  explanation: string;
}

export interface SignalPoolMetrics {
  diversity: number;           // Shannon entropy of signal directions
  consensus: number;           // Agreement level [0, 1]
  correlationMatrix: number[][]; // Pairwise signal correlations
  dominantDirection: -1 | 0 | 1;
  signalCount: number;
}

export class SignalPool {
  private signals: NormalizedSignal[] = [];
  private correlationWindow: NormalizedSignal[][] = [];
  private windowSize: number = 100; // Rolling window for correlation

  /**
   * Aggregate signals from all agents into normalized pool
   */
  aggregateSignals(agentSignals: AgentSignal[]): NormalizedSignal[] {
    this.signals = agentSignals.map(signal => this.normalizeSignal(signal));
    
    // Update correlation window
    this.correlationWindow.push([...this.signals]);
    if (this.correlationWindow.length > this.windowSize) {
      this.correlationWindow.shift();
    }
    
    return this.signals;
  }

  /**
   * Normalize agent signal to {-1, 0, +1} with strength
   */
  normalizeSignal(signal: AgentSignal): NormalizedSignal {
    // Determine direction based on signal value
    let direction: -1 | 0 | 1;
    if (signal.signal < -0.3) {
      direction = -1; // Short
    } else if (signal.signal > 0.3) {
      direction = 1;  // Long
    } else {
      direction = 0;  // Neutral
    }

    // Strength is the absolute signal value capped at 1.0
    const strength = Math.min(Math.abs(signal.signal), 1.0);

    return {
      agentId: signal.agentId,
      direction,
      strength,
      timestamp: new Date(),
      rawSignal: signal.signal,
      explanation: signal.explanation
    };
  }

  /**
   * Calculate signal diversity using Shannon entropy
   */
  calculateDiversity(): number {
    if (this.signals.length === 0) return 0;

    // Count direction frequencies
    const counts = { '-1': 0, '0': 0, '1': 0 };
    this.signals.forEach(s => {
      counts[s.direction.toString() as keyof typeof counts]++;
    });

    // Calculate Shannon entropy
    const total = this.signals.length;
    let entropy = 0;
    
    Object.values(counts).forEach(count => {
      if (count > 0) {
        const p = count / total;
        entropy -= p * Math.log2(p);
      }
    });

    // Normalize to [0, 1] (max entropy for 3 directions is log2(3) ≈ 1.585)
    return entropy / Math.log2(3);
  }

  /**
   * Calculate signal consensus (agreement level)
   */
  calculateConsensus(): number {
    if (this.signals.length === 0) return 0;

    // Weight by strength
    let weightedSum = 0;
    let totalWeight = 0;

    this.signals.forEach(s => {
      weightedSum += s.direction * s.strength;
      totalWeight += s.strength;
    });

    if (totalWeight === 0) return 0;

    // Consensus is absolute weighted average normalized to [0, 1]
    return Math.abs(weightedSum / totalWeight);
  }

  /**
   * Get correlation matrix between signals
   */
  getCorrelationMatrix(): number[][] {
    if (this.correlationWindow.length < 10) {
      // Not enough data, return identity matrix
      const n = this.signals.length;
      return Array(n).fill(0).map((_, i) => 
        Array(n).fill(0).map((_, j) => i === j ? 1 : 0)
      );
    }

    const n = this.signals.length;
    const matrix: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));

    // Compute pairwise Pearson correlation
    for (let i = 0; i < n; i++) {
      for (let j = i; j < n; j++) {
        const corr = this.computeCorrelation(i, j);
        matrix[i][j] = corr;
        matrix[j][i] = corr;
      }
    }

    return matrix;
  }

  /**
   * Compute Pearson correlation between two signal series
   */
  private computeCorrelation(agentIdx1: number, agentIdx2: number): number {
    const series1 = this.correlationWindow.map(w => w[agentIdx1]?.rawSignal ?? 0);
    const series2 = this.correlationWindow.map(w => w[agentIdx2]?.rawSignal ?? 0);

    const n = series1.length;
    const mean1 = series1.reduce((a, b) => a + b, 0) / n;
    const mean2 = series2.reduce((a, b) => a + b, 0) / n;

    let numerator = 0;
    let denom1 = 0;
    let denom2 = 0;

    for (let i = 0; i < n; i++) {
      const diff1 = series1[i] - mean1;
      const diff2 = series2[i] - mean2;
      numerator += diff1 * diff2;
      denom1 += diff1 * diff1;
      denom2 += diff2 * diff2;
    }

    if (denom1 === 0 || denom2 === 0) return 0;
    return numerator / Math.sqrt(denom1 * denom2);
  }

  /**
   * Get dominant signal direction
   */
  getDominantDirection(): -1 | 0 | 1 {
    if (this.signals.length === 0) return 0;

    // Weight by strength
    let weightedSum = 0;
    let totalWeight = 0;

    this.signals.forEach(s => {
      weightedSum += s.direction * s.strength;
      totalWeight += s.strength;
    });

    if (totalWeight === 0) return 0;

    const avgDirection = weightedSum / totalWeight;
    
    if (avgDirection < -0.3) return -1;
    if (avgDirection > 0.3) return 1;
    return 0;
  }

  /**
   * Get pool metrics summary
   */
  getMetrics(): SignalPoolMetrics {
    return {
      diversity: this.calculateDiversity(),
      consensus: this.calculateConsensus(),
      correlationMatrix: this.getCorrelationMatrix(),
      dominantDirection: this.getDominantDirection(),
      signalCount: this.signals.length
    };
  }

  /**
   * Get current normalized signals
   */
  getSignals(): NormalizedSignal[] {
    return this.signals;
  }

  /**
   * Filter signals by minimum strength threshold
   */
  filterByStrength(minStrength: number): NormalizedSignal[] {
    return this.signals.filter(s => s.strength >= minStrength);
  }

  /**
   * Get signals by direction
   */
  getSignalsByDirection(direction: -1 | 0 | 1): NormalizedSignal[] {
    return this.signals.filter(s => s.direction === direction);
  }

  /**
   * Check if pool has disagreement (conflicting strong signals)
   */
  hasDisagreement(strengthThreshold: number = 0.6): boolean {
    const strongLong = this.signals.filter(s => s.direction === 1 && s.strength >= strengthThreshold);
    const strongShort = this.signals.filter(s => s.direction === -1 && s.strength >= strengthThreshold);
    
    return strongLong.length > 0 && strongShort.length > 0;
  }

  /**
   * Get conflicting signal pairs
   */
  getConflictingSignals(strengthThreshold: number = 0.6): [NormalizedSignal, NormalizedSignal][] {
    const strongLong = this.signals.filter(s => s.direction === 1 && s.strength >= strengthThreshold);
    const strongShort = this.signals.filter(s => s.direction === -1 && s.strength >= strengthThreshold);
    
    const conflicts: [NormalizedSignal, NormalizedSignal][] = [];
    
    strongLong.forEach(long => {
      strongShort.forEach(short => {
        conflicts.push([long, short]);
      });
    });
    
    return conflicts;
  }
}
