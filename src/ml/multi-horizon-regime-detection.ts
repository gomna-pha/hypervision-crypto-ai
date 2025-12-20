/**
 * Multi-Horizon Regime Detection with Transition States
 * 
 * Extends regime detection with:
 * - Transition state detection (Stabilizing, Deteriorating, Shifting)
 * - Horizon agreement scoring (consensus across hourly/weekly/monthly)
 * - Multi-horizon confidence weighting
 * - Regime persistence tracking
 * - Transition likelihood prediction
 */

import { HorizonIndexedSignal } from './multi-horizon-signal-pool';
import { TimeHorizon } from './time-scale-feature-store';

// ============================================================================
// Market Regime Types
// ============================================================================

export type MarketRegime = 'CRISIS' | 'DEFENSIVE' | 'NEUTRAL' | 'RISK_ON' | 'HIGH_CONVICTION';

export type TransitionState = 
  | 'STABLE'          // Regime is stable, no transition imminent
  | 'STABILIZING'     // Moving towards stable regime
  | 'DETERIORATING'   // Regime weakening, may transition
  | 'SHIFTING'        // Active transition in progress
  | 'VOLATILE';       // Rapid regime changes

// ============================================================================
// Regime Features
// ============================================================================

export interface RegimeFeatures {
  volatility: number;       // [0, 1] normalized volatility
  returns: number;          // [-1, 1] normalized returns
  sentiment: number;        // [0, 1] Fear & Greed normalized
  volume: number;           // [0, 1] normalized volume
  liquidity: number;        // [0, 1] liquidity score
  spread: number;           // [0, 1] normalized spread
  flowImbalance: number;    // [-1, 1] buy/sell pressure
}

// ============================================================================
// Multi-Horizon Regime State
// ============================================================================

export interface MultiHorizonRegimeState {
  // Current regime
  currentRegime: MarketRegime;
  confidence: number;  // [0, 1]
  
  // Transition state
  transitionState: TransitionState;
  transitionLikelihood: number;  // [0, 1] probability of transition
  
  // Per-horizon regimes
  horizonRegimes: {
    hourly: MarketRegime;
    weekly: MarketRegime;
    monthly: MarketRegime;
  };
  
  // Horizon agreement (consensus across horizons)
  horizonAgreement: {
    hourly_weekly: number;   // [0, 1]
    weekly_monthly: number;  // [0, 1]
    hourly_monthly: number;  // [0, 1]
    overall: number;         // [0, 1] average agreement
  };
  
  // Horizon-specific confidence
  horizonConfidence: {
    hourly: number;
    weekly: number;
    monthly: number;
  };
  
  // Transition probabilities
  transitionProb: Record<MarketRegime, number>;
  
  // Most likely next regime (if transitioning)
  nextRegime: MarketRegime | null;
  
  // Regime persistence
  regimeDurationHours: number;
  recentTransitions: number;  // Count in last 24 hours
  
  // Features used
  features: RegimeFeatures;
  timestamp: Date;
}

// ============================================================================
// Regime History Entry
// ============================================================================

interface RegimeHistoryEntry {
  regime: MarketRegime;
  confidence: number;
  timestamp: Date;
  durationHours: number;
}

// ============================================================================
// Multi-Horizon Regime Detector
// ============================================================================

export class MultiHorizonRegimeDetector {
  private currentRegime: MarketRegime = 'NEUTRAL';
  private regimeHistory: RegimeHistoryEntry[] = [];
  private maxHistoryLength = 1000;
  
  // Transition matrix (5x5)
  private transitionMatrix: Map<MarketRegime, Map<MarketRegime, number>>;
  
  // Regime started timestamp
  private regimeStartTime: Date = new Date();
  
  constructor() {
    this.transitionMatrix = this.initializeTransitionMatrix();
  }
  
  // ============================================================================
  // Main Detection Method
  // ============================================================================
  
  /**
   * Detect regime using multi-horizon signals
   */
  detectRegime(
    signals: {
      hourly: HorizonIndexedSignal[];
      weekly: HorizonIndexedSignal[];
      monthly: HorizonIndexedSignal[];
    },
    features: RegimeFeatures,
    hyperbolicDistances: Map<TimeHorizon, number>
  ): MultiHorizonRegimeState {
    // 1. Detect regime per horizon
    const hourlyRegime = this.detectHorizonRegime(signals.hourly, features, 'hourly');
    const weeklyRegime = this.detectHorizonRegime(signals.weekly, features, 'weekly');
    const monthlyRegime = this.detectHorizonRegime(signals.monthly, features, 'monthly');
    
    // 2. Calculate horizon agreement
    const horizonAgreement = this.calculateHorizonAgreement(
      hourlyRegime.regime,
      weeklyRegime.regime,
      monthlyRegime.regime
    );
    
    // 3. Determine final regime (weighted by horizon & agreement)
    const finalRegime = this.determineConsensusRegime(
      { hourly: hourlyRegime, weekly: weeklyRegime, monthly: monthlyRegime },
      horizonAgreement
    );
    
    // 4. Calculate overall confidence
    const confidence = this.calculateOverallConfidence(
      { hourly: hourlyRegime, weekly: weeklyRegime, monthly: monthlyRegime },
      horizonAgreement
    );
    
    // 5. Detect transition state
    const transitionState = this.detectTransitionState(
      finalRegime.regime,
      this.currentRegime,
      horizonAgreement,
      features
    );
    
    // 6. Calculate transition likelihood
    const transitionLikelihood = this.calculateTransitionLikelihood(
      finalRegime.regime,
      transitionState,
      features,
      horizonAgreement.overall
    );
    
    // 7. Get transition probabilities
    const transitionProb = this.getTransitionProbabilities(finalRegime.regime);
    
    // 8. Predict next regime (if transitioning)
    const nextRegime = transitionState === 'SHIFTING' || transitionState === 'DETERIORATING'
      ? this.predictNextRegime(finalRegime.regime, transitionProb)
      : null;
    
    // 9. Update regime history
    const regimeDurationHours = this.updateRegimeHistory(finalRegime.regime, confidence);
    const recentTransitions = this.countRecentTransitions(24);
    
    return {
      currentRegime: finalRegime.regime,
      confidence,
      transitionState,
      transitionLikelihood,
      horizonRegimes: {
        hourly: hourlyRegime.regime,
        weekly: weeklyRegime.regime,
        monthly: monthlyRegime.regime,
      },
      horizonAgreement,
      horizonConfidence: {
        hourly: hourlyRegime.confidence,
        weekly: weeklyRegime.confidence,
        monthly: monthlyRegime.confidence,
      },
      transitionProb,
      nextRegime,
      regimeDurationHours,
      recentTransitions,
      features,
      timestamp: new Date(),
    };
  }
  
  // ============================================================================
  // Per-Horizon Regime Detection
  // ============================================================================
  
  /**
   * Detect regime for a single horizon
   */
  private detectHorizonRegime(
    signals: HorizonIndexedSignal[],
    features: RegimeFeatures,
    horizon: TimeHorizon
  ): { regime: MarketRegime; confidence: number } {
    // Calculate weighted signal direction
    let weightedSignal = 0;
    let totalWeight = 0;
    
    for (const signal of signals) {
      const weight = signal.confidence * signal.stabilityScore;
      weightedSignal += signal.direction * weight;
      totalWeight += weight;
    }
    
    const avgSignal = totalWeight > 0 ? weightedSignal / totalWeight : 0;
    
    // Classify regime based on features and signals
    const regime = this.classifyRegimeFromFeatures(features, avgSignal, horizon);
    
    // Calculate confidence
    const confidence = this.calculateRegimeConfidence(features, regime, signals);
    
    return { regime, confidence };
  }
  
  /**
   * Classify regime from features and signal direction
   */
  private classifyRegimeFromFeatures(
    features: RegimeFeatures,
    avgSignal: number,
    horizon: TimeHorizon
  ): MarketRegime {
    const { volatility, returns, sentiment, liquidity, spread } = features;
    
    // CRISIS: High volatility + negative returns + extreme fear
    if (volatility > 0.7 && returns < -0.3 && sentiment < 0.25) {
      return 'CRISIS';
    }
    
    // DEFENSIVE: Moderate-high volatility + cautious
    if (volatility > 0.5 && sentiment < 0.4 && avgSignal < 0) {
      return 'DEFENSIVE';
    }
    
    // HIGH_CONVICTION: Low volatility + strong trend + high confidence
    if (volatility < 0.3 && Math.abs(returns) > 0.3 && Math.abs(avgSignal) > 0.6) {
      return 'HIGH_CONVICTION';
    }
    
    // RISK_ON: Low volatility + positive returns + bullish sentiment
    if (volatility < 0.4 && returns > 0.1 && sentiment > 0.6 && avgSignal > 0) {
      return 'RISK_ON';
    }
    
    // Default: NEUTRAL
    return 'NEUTRAL';
  }
  
  /**
   * Calculate regime confidence
   */
  private calculateRegimeConfidence(
    features: RegimeFeatures,
    regime: MarketRegime,
    signals: HorizonIndexedSignal[]
  ): number {
    // Base confidence from signal consensus
    const signalDirections = signals.map(s => s.direction);
    const consensus = this.calculateConsensus(signalDirections);
    
    // Feature strength
    const featureStrength = this.getFeatureStrengthForRegime(features, regime);
    
    // Combined confidence
    const confidence = 0.6 * consensus + 0.4 * featureStrength;
    
    return Math.max(0, Math.min(1, confidence));
  }
  
  private calculateConsensus(directions: number[]): number {
    if (directions.length === 0) return 0.5;
    
    const bullish = directions.filter(d => d === 1).length;
    const bearish = directions.filter(d => d === -1).length;
    const total = directions.length;
    
    const consensus = Math.abs(bullish - bearish) / total;
    return consensus;
  }
  
  private getFeatureStrengthForRegime(features: RegimeFeatures, regime: MarketRegime): number {
    const { volatility, returns, sentiment } = features;
    
    switch (regime) {
      case 'CRISIS':
        return (volatility + (1 - sentiment) + Math.max(0, -returns)) / 3;
      case 'DEFENSIVE':
        return (volatility * 0.7 + (1 - sentiment) * 0.3);
      case 'NEUTRAL':
        return 1 - Math.abs(returns) - Math.abs(volatility - 0.5);
      case 'RISK_ON':
        return (sentiment + Math.max(0, returns) + (1 - volatility)) / 3;
      case 'HIGH_CONVICTION':
        return (Math.abs(returns) + (1 - volatility) + sentiment) / 3;
      default:
        return 0.5;
    }
  }
  
  // ============================================================================
  // Horizon Agreement
  // ============================================================================
  
  /**
   * Calculate agreement between horizons
   */
  private calculateHorizonAgreement(
    hourly: MarketRegime,
    weekly: MarketRegime,
    monthly: MarketRegime
  ): {
    hourly_weekly: number;
    weekly_monthly: number;
    hourly_monthly: number;
    overall: number;
  } {
    const hourly_weekly = hourly === weekly ? 1.0 : this.getRegimeSimilarity(hourly, weekly);
    const weekly_monthly = weekly === monthly ? 1.0 : this.getRegimeSimilarity(weekly, monthly);
    const hourly_monthly = hourly === monthly ? 1.0 : this.getRegimeSimilarity(hourly, monthly);
    
    const overall = (hourly_weekly + weekly_monthly + hourly_monthly) / 3;
    
    return { hourly_weekly, weekly_monthly, hourly_monthly, overall };
  }
  
  /**
   * Get similarity between two regimes [0, 1]
   */
  private getRegimeSimilarity(regime1: MarketRegime, regime2: MarketRegime): number {
    // Regime ordering: CRISIS < DEFENSIVE < NEUTRAL < RISK_ON < HIGH_CONVICTION
    const order: Record<MarketRegime, number> = {
      CRISIS: 0,
      DEFENSIVE: 1,
      NEUTRAL: 2,
      RISK_ON: 3,
      HIGH_CONVICTION: 4,
    };
    
    const distance = Math.abs(order[regime1] - order[regime2]);
    const maxDistance = 4;
    
    return 1 - distance / maxDistance;
  }
  
  // ============================================================================
  // Consensus Regime Determination
  // ============================================================================
  
  /**
   * Determine consensus regime from multi-horizon regimes
   */
  private determineConsensusRegime(
    horizonRegimes: {
      hourly: { regime: MarketRegime; confidence: number };
      weekly: { regime: MarketRegime; confidence: number };
      monthly: { regime: MarketRegime; confidence: number };
    },
    agreement: { overall: number }
  ): { regime: MarketRegime } {
    // If high agreement, use majority vote
    if (agreement.overall > 0.7) {
      return { regime: this.getMajorityRegime(horizonRegimes) };
    }
    
    // Otherwise, weight by confidence
    const regimes = [horizonRegimes.hourly, horizonRegimes.weekly, horizonRegimes.monthly];
    regimes.sort((a, b) => b.confidence - a.confidence);
    
    // Return highest confidence regime
    return { regime: regimes[0].regime };
  }
  
  private getMajorityRegime(horizonRegimes: {
    hourly: { regime: MarketRegime };
    weekly: { regime: MarketRegime };
    monthly: { regime: MarketRegime };
  }): MarketRegime {
    const regimes = [
      horizonRegimes.hourly.regime,
      horizonRegimes.weekly.regime,
      horizonRegimes.monthly.regime,
    ];
    
    // Count occurrences
    const counts = new Map<MarketRegime, number>();
    for (const regime of regimes) {
      counts.set(regime, (counts.get(regime) || 0) + 1);
    }
    
    // Find majority (or most common)
    let maxCount = 0;
    let majorityRegime: MarketRegime = 'NEUTRAL';
    for (const [regime, count] of counts.entries()) {
      if (count > maxCount) {
        maxCount = count;
        majorityRegime = regime;
      }
    }
    
    return majorityRegime;
  }
  
  private calculateOverallConfidence(
    horizonRegimes: {
      hourly: { confidence: number };
      weekly: { confidence: number };
      monthly: { confidence: number };
    },
    agreement: { overall: number }
  ): number {
    const avgConfidence = (
      horizonRegimes.hourly.confidence +
      horizonRegimes.weekly.confidence +
      horizonRegimes.monthly.confidence
    ) / 3;
    
    // Weight by agreement
    return 0.7 * avgConfidence + 0.3 * agreement.overall;
  }
  
  // ============================================================================
  // Transition State Detection
  // ============================================================================
  
  /**
   * Detect current transition state
   */
  private detectTransitionState(
    newRegime: MarketRegime,
    currentRegime: MarketRegime,
    agreement: { overall: number },
    features: RegimeFeatures
  ): TransitionState {
    // Check if regime changed
    const regimeChanged = newRegime !== currentRegime;
    
    // Check volatility
    const { volatility } = features;
    
    // Check recent transition frequency
    const recentTransitions = this.countRecentTransitions(6);  // Last 6 hours
    
    // VOLATILE: Many recent transitions
    if (recentTransitions >= 3) {
      return 'VOLATILE';
    }
    
    // SHIFTING: Active regime change
    if (regimeChanged) {
      return 'SHIFTING';
    }
    
    // DETERIORATING: Low agreement + high volatility
    if (agreement.overall < 0.5 && volatility > 0.6) {
      return 'DETERIORATING';
    }
    
    // STABILIZING: Improving agreement after transition
    if (agreement.overall > 0.6 && this.regimeHistory.length > 0) {
      const lastEntry = this.regimeHistory[this.regimeHistory.length - 1];
      if (lastEntry.durationHours < 6) {
        return 'STABILIZING';
      }
    }
    
    // STABLE: High agreement, no recent changes
    if (agreement.overall > 0.7 && recentTransitions === 0) {
      return 'STABLE';
    }
    
    return 'STABLE';
  }
  
  /**
   * Calculate transition likelihood
   */
  private calculateTransitionLikelihood(
    currentRegime: MarketRegime,
    transitionState: TransitionState,
    features: RegimeFeatures,
    agreementScore: number
  ): number {
    // Base likelihood from transition state
    const baseMap: Record<TransitionState, number> = {
      STABLE: 0.1,
      STABILIZING: 0.2,
      DETERIORATING: 0.6,
      SHIFTING: 0.9,
      VOLATILE: 0.8,
    };
    
    let likelihood = baseMap[transitionState];
    
    // Adjust by agreement (low agreement = higher transition risk)
    likelihood += (1 - agreementScore) * 0.2;
    
    // Adjust by volatility
    likelihood += features.volatility * 0.2;
    
    return Math.max(0, Math.min(1, likelihood));
  }
  
  // ============================================================================
  // Transition Matrix & Prediction
  // ============================================================================
  
  private initializeTransitionMatrix(): Map<MarketRegime, Map<MarketRegime, number>> {
    const regimes: MarketRegime[] = ['CRISIS', 'DEFENSIVE', 'NEUTRAL', 'RISK_ON', 'HIGH_CONVICTION'];
    const matrix = new Map<MarketRegime, Map<MarketRegime, number>>();
    
    // Initialize with empirical probabilities
    const transitions = {
      CRISIS: { CRISIS: 0.50, DEFENSIVE: 0.35, NEUTRAL: 0.10, RISK_ON: 0.03, HIGH_CONVICTION: 0.02 },
      DEFENSIVE: { CRISIS: 0.15, DEFENSIVE: 0.45, NEUTRAL: 0.30, RISK_ON: 0.08, HIGH_CONVICTION: 0.02 },
      NEUTRAL: { CRISIS: 0.05, DEFENSIVE: 0.20, NEUTRAL: 0.50, RISK_ON: 0.20, HIGH_CONVICTION: 0.05 },
      RISK_ON: { CRISIS: 0.02, DEFENSIVE: 0.08, NEUTRAL: 0.30, RISK_ON: 0.48, HIGH_CONVICTION: 0.12 },
      HIGH_CONVICTION: { CRISIS: 0.08, DEFENSIVE: 0.12, NEUTRAL: 0.25, RISK_ON: 0.35, HIGH_CONVICTION: 0.20 },
    };
    
    for (const from of regimes) {
      const toMap = new Map<MarketRegime, number>();
      for (const to of regimes) {
        toMap.set(to, transitions[from][to]);
      }
      matrix.set(from, toMap);
    }
    
    return matrix;
  }
  
  private getTransitionProbabilities(currentRegime: MarketRegime): Record<MarketRegime, number> {
    const fromMap = this.transitionMatrix.get(currentRegime);
    if (!fromMap) {
      return {
        CRISIS: 0.2,
        DEFENSIVE: 0.2,
        NEUTRAL: 0.2,
        RISK_ON: 0.2,
        HIGH_CONVICTION: 0.2,
      };
    }
    
    return {
      CRISIS: fromMap.get('CRISIS') || 0.2,
      DEFENSIVE: fromMap.get('DEFENSIVE') || 0.2,
      NEUTRAL: fromMap.get('NEUTRAL') || 0.2,
      RISK_ON: fromMap.get('RISK_ON') || 0.2,
      HIGH_CONVICTION: fromMap.get('HIGH_CONVICTION') || 0.2,
    };
  }
  
  private predictNextRegime(
    currentRegime: MarketRegime,
    transitionProb: Record<MarketRegime, number>
  ): MarketRegime {
    // Find most likely next regime (excluding current)
    let maxProb = 0;
    let nextRegime: MarketRegime = 'NEUTRAL';
    
    for (const [regime, prob] of Object.entries(transitionProb)) {
      if (regime !== currentRegime && prob > maxProb) {
        maxProb = prob;
        nextRegime = regime as MarketRegime;
      }
    }
    
    return nextRegime;
  }
  
  // ============================================================================
  // Regime History Management
  // ============================================================================
  
  private updateRegimeHistory(regime: MarketRegime, confidence: number): number {
    const now = new Date();
    
    // Check if regime changed
    if (regime !== this.currentRegime) {
      // Add previous regime to history
      if (this.currentRegime) {
        const duration = (now.getTime() - this.regimeStartTime.getTime()) / (1000 * 60 * 60);
        this.regimeHistory.push({
          regime: this.currentRegime,
          confidence,
          timestamp: this.regimeStartTime,
          durationHours: duration,
        });
      }
      
      // Update current regime
      this.currentRegime = regime;
      this.regimeStartTime = now;
      
      // Trim history
      if (this.regimeHistory.length > this.maxHistoryLength) {
        this.regimeHistory.shift();
      }
    }
    
    // Return current regime duration
    return (now.getTime() - this.regimeStartTime.getTime()) / (1000 * 60 * 60);
  }
  
  private countRecentTransitions(hoursLookback: number): number {
    const now = new Date();
    const cutoff = new Date(now.getTime() - hoursLookback * 60 * 60 * 1000);
    
    let transitions = 0;
    let lastRegime: MarketRegime | null = null;
    
    for (const entry of this.regimeHistory) {
      if (entry.timestamp >= cutoff) {
        if (lastRegime && lastRegime !== entry.regime) {
          transitions++;
        }
        lastRegime = entry.regime;
      }
    }
    
    return transitions;
  }
  
  // ============================================================================
  // Getters
  // ============================================================================
  
  getCurrentRegime(): MarketRegime {
    return this.currentRegime;
  }
  
  getRegimeHistory(): RegimeHistoryEntry[] {
    return [...this.regimeHistory];
  }
  
  getRegimeDuration(): number {
    const now = new Date();
    return (now.getTime() - this.regimeStartTime.getTime()) / (1000 * 60 * 60);
  }
}
