/**
 * Meta-Strategy Controller (The "Brain")
 * 
 * Integrates all system layers to produce dynamic horizon weights:
 * - Input: Volatility regime, regime distance, horizon stability, signal alignment
 * - Output: w_hourly + w_weekly + w_monthly = 1.0
 * 
 * Uses simplified XGBoost-like decision tree ensemble for real-time decisions.
 * 
 * Key Features:
 * - Dynamic horizon weight allocation
 * - Exposure scaling based on confidence
 * - Risk aversion adjustment
 * - Real-time adaptation to market conditions
 */

import { MarketRegime, TransitionState } from './multi-horizon-regime-detection';
import { TimeHorizon } from './time-scale-feature-store';
import { HorizonIndexedSignal } from './multi-horizon-signal-pool';

// ============================================================================
// Meta-Controller Input Features
// ============================================================================

export interface MetaControllerInput {
  // Volatility metrics
  volatility: {
    current: number;           // [0, 1] Current volatility
    regime: 'low' | 'normal' | 'high' | 'extreme';
    trend: 'increasing' | 'stable' | 'decreasing';
  };
  
  // Regime information
  regime: {
    current: MarketRegime;
    confidence: number;        // [0, 1]
    fragility: number;         // [0, 1] Transition likelihood
    transitionState: TransitionState;
    durationHours: number;
  };
  
  // Regime distances (from hyperbolic embedding)
  regimeDistance: {
    toCrisis: number;
    toDefensive: number;
    toNeutral: number;
    toRiskOn: number;
    toHighConviction: number;
  };
  
  // Horizon stability
  horizonStability: {
    hourly: number;            // [0, 1]
    weekly: number;            // [0, 1]
    monthly: number;           // [0, 1]
  };
  
  // Signal metrics
  signalAlignment: {
    hourly_weekly: number;     // [0, 1]
    weekly_monthly: number;    // [0, 1]
    hourly_monthly: number;    // [0, 1]
    overall: number;           // [0, 1]
  };
  
  // Historical performance (if available)
  recentPerformance: {
    hourly_sharpe: number;
    weekly_sharpe: number;
    monthly_sharpe: number;
  } | null;
}

// ============================================================================
// Meta-Controller Output
// ============================================================================

export interface MetaControllerOutput {
  // Horizon weights (sum = 1.0)
  horizonWeights: {
    w_hourly: number;   // ∈ [0, 1]
    w_weekly: number;   // ∈ [0, 1]
    w_monthly: number;  // ∈ [0, 1]
  };
  // Constraint: w_hourly + w_weekly + w_monthly = 1.0
  
  // Exposure scaling
  exposureScaling: number;  // ∈ [0, 2] - multiplier for position sizing
  
  // Risk aversion level
  riskAversion: number;     // ∈ [1, 10] - higher = more conservative
  
  // Confidence in decision
  decisionConfidence: number;  // [0, 1]
  
  // Reasoning
  reasoning: {
    primaryFactor: string;
    secondaryFactors: string[];
    warnings: string[];
  };
  
  timestamp: Date;
}

// ============================================================================
// Decision Tree Node (Simplified XGBoost-like)
// ============================================================================

interface DecisionNode {
  feature: string;
  threshold: number;
  leftChild: DecisionNode | number[];  // Node or leaf weights
  rightChild: DecisionNode | number[]; // Node or leaf weights
}

// ============================================================================
// Meta-Strategy Controller
// ============================================================================

export class MetaStrategyController {
  private decisionTrees: DecisionNode[];
  private learningRate: number;
  
  constructor(config: { learningRate?: number } = {}) {
    this.learningRate = config.learningRate || 0.1;
    this.decisionTrees = this.initializeDecisionTrees();
  }
  
  // ============================================================================
  // Main Prediction Method
  // ============================================================================
  
  /**
   * Compute optimal horizon weights and risk parameters
   */
  predictHorizonWeights(input: MetaControllerInput): MetaControllerOutput {
    // 1. Extract features
    const features = this.extractFeatures(input);
    
    // 2. Get base horizon weights from ensemble
    const baseWeights = this.ensemblePredict(features);
    
    // 3. Apply volatility adaptation
    const volatilityAdjusted = this.applyVolatilityAdaptation(
      baseWeights,
      input.volatility
    );
    
    // 4. Apply regime-based adjustment
    const regimeAdjusted = this.applyRegimeAdjustment(
      volatilityAdjusted,
      input.regime
    );
    
    // 5. Apply stability adjustment
    const finalWeights = this.applyStabilityAdjustment(
      regimeAdjusted,
      input.horizonStability,
      input.signalAlignment
    );
    
    // 6. Normalize to sum = 1.0
    const normalized = this.normalizeWeights(finalWeights);
    
    // 7. Calculate exposure scaling
    const exposureScaling = this.calculateExposureScaling(input);
    
    // 8. Calculate risk aversion
    const riskAversion = this.calculateRiskAversion(input);
    
    // 9. Calculate decision confidence
    const decisionConfidence = this.calculateDecisionConfidence(input);
    
    // 10. Generate reasoning
    const reasoning = this.generateReasoning(input, normalized, exposureScaling);
    
    return {
      horizonWeights: normalized,
      exposureScaling,
      riskAversion,
      decisionConfidence,
      reasoning,
      timestamp: new Date(),
    };
  }
  
  // ============================================================================
  // Feature Engineering for Meta-Model
  // ============================================================================
  
  private extractFeatures(input: MetaControllerInput): number[] {
    const features: number[] = [];
    
    // Volatility features (3)
    features.push(input.volatility.current);
    features.push(this.encodeVolatilityRegime(input.volatility.regime));
    features.push(this.encodeVolatilityTrend(input.volatility.trend));
    
    // Regime features (5)
    features.push(this.encodeRegime(input.regime.current));
    features.push(input.regime.confidence);
    features.push(input.regime.fragility);
    features.push(this.encodeTransitionState(input.regime.transitionState));
    features.push(Math.min(1.0, input.regime.durationHours / 24)); // Normalize
    
    // Regime distances (5)
    features.push(input.regimeDistance.toCrisis);
    features.push(input.regimeDistance.toDefensive);
    features.push(input.regimeDistance.toNeutral);
    features.push(input.regimeDistance.toRiskOn);
    features.push(input.regimeDistance.toHighConviction);
    
    // Horizon stability (3)
    features.push(input.horizonStability.hourly);
    features.push(input.horizonStability.weekly);
    features.push(input.horizonStability.monthly);
    
    // Signal alignment (4)
    features.push(input.signalAlignment.hourly_weekly);
    features.push(input.signalAlignment.weekly_monthly);
    features.push(input.signalAlignment.hourly_monthly);
    features.push(input.signalAlignment.overall);
    
    // Historical performance (3) - if available
    if (input.recentPerformance) {
      features.push(Math.max(-1, Math.min(3, input.recentPerformance.hourly_sharpe)) / 3);
      features.push(Math.max(-1, Math.min(3, input.recentPerformance.weekly_sharpe)) / 3);
      features.push(Math.max(-1, Math.min(3, input.recentPerformance.monthly_sharpe)) / 3);
    } else {
      features.push(0, 0, 0);
    }
    
    return features;
  }
  
  // Encoding helpers
  private encodeVolatilityRegime(regime: string): number {
    const map = { low: 0.25, normal: 0.50, high: 0.75, extreme: 1.0 };
    return map[regime as keyof typeof map] || 0.5;
  }
  
  private encodeVolatilityTrend(trend: string): number {
    const map = { decreasing: 0, stable: 0.5, increasing: 1.0 };
    return map[trend as keyof typeof map] || 0.5;
  }
  
  private encodeRegime(regime: MarketRegime): number {
    const map = { CRISIS: 0, DEFENSIVE: 0.25, NEUTRAL: 0.5, RISK_ON: 0.75, HIGH_CONVICTION: 1.0 };
    return map[regime] || 0.5;
  }
  
  private encodeTransitionState(state: TransitionState): number {
    const map = { STABLE: 0, STABILIZING: 0.25, DETERIORATING: 0.75, SHIFTING: 1.0, VOLATILE: 0.9 };
    return map[state] || 0.5;
  }
  
  // ============================================================================
  // Ensemble Prediction (Simplified XGBoost)
  // ============================================================================
  
  private ensemblePredict(features: number[]): { hourly: number; weekly: number; monthly: number } {
    // Aggregate predictions from all trees
    let hourly = 0;
    let weekly = 0;
    let monthly = 0;
    
    for (const tree of this.decisionTrees) {
      const prediction = this.traverseTree(tree, features);
      hourly += prediction[0];
      weekly += prediction[1];
      monthly += prediction[2];
    }
    
    // Average (with learning rate)
    const numTrees = this.decisionTrees.length;
    return {
      hourly: hourly / numTrees,
      weekly: weekly / numTrees,
      monthly: monthly / numTrees,
    };
  }
  
  private traverseTree(node: DecisionNode | number[], features: number[]): number[] {
    // Leaf node
    if (Array.isArray(node)) {
      return node;
    }
    
    // Internal node
    const featureIndex = this.getFeatureIndex(node.feature);
    const featureValue = features[featureIndex];
    
    if (featureValue <= node.threshold) {
      return this.traverseTree(node.leftChild, features);
    } else {
      return this.traverseTree(node.rightChild, features);
    }
  }
  
  private getFeatureIndex(featureName: string): number {
    // Map feature names to indices
    const featureMap: Record<string, number> = {
      'volatility_current': 0,
      'volatility_regime': 1,
      'volatility_trend': 2,
      'regime_current': 3,
      'regime_confidence': 4,
      'regime_fragility': 5,
      'transition_state': 6,
      'regime_duration': 7,
      // Add more mappings as needed
    };
    
    return featureMap[featureName] || 0;
  }
  
  // ============================================================================
  // Volatility Adaptation
  // ============================================================================
  
  private applyVolatilityAdaptation(
    weights: { hourly: number; weekly: number; monthly: number },
    volatility: MetaControllerInput['volatility']
  ): { hourly: number; weekly: number; monthly: number } {
    // Target weights by volatility regime
    const targets = {
      low: { hourly: 0.60, weekly: 0.30, monthly: 0.10 },
      normal: { hourly: 0.40, weekly: 0.40, monthly: 0.20 },
      high: { hourly: 0.30, weekly: 0.40, monthly: 0.30 },
      extreme: { hourly: 0.10, weekly: 0.30, monthly: 0.60 },
    };
    
    const target = targets[volatility.regime];
    
    // Blend model weights with volatility targets (70% target, 30% model)
    return {
      hourly: 0.7 * target.hourly + 0.3 * weights.hourly,
      weekly: 0.7 * target.weekly + 0.3 * weights.weekly,
      monthly: 0.7 * target.monthly + 0.3 * weights.monthly,
    };
  }
  
  // ============================================================================
  // Regime Adjustment
  // ============================================================================
  
  private applyRegimeAdjustment(
    weights: { hourly: number; weekly: number; monthly: number },
    regime: MetaControllerInput['regime']
  ): { hourly: number; weekly: number; monthly: number } {
    let adjusted = { ...weights };
    
    // CRISIS: Shift heavily to monthly (structural signals)
    if (regime.current === 'CRISIS') {
      adjusted.monthly *= 1.5;
      adjusted.hourly *= 0.5;
    }
    
    // DETERIORATING or SHIFTING: Reduce hourly, increase monthly
    if (regime.transitionState === 'DETERIORATING' || regime.transitionState === 'SHIFTING') {
      adjusted.hourly *= 0.8;
      adjusted.monthly *= 1.3;
    }
    
    // VOLATILE: Balance across all horizons
    if (regime.transitionState === 'VOLATILE') {
      adjusted = {
        hourly: 0.33,
        weekly: 0.33,
        monthly: 0.34,
      };
    }
    
    // HIGH_CONVICTION with high confidence: Allow more hourly
    if (regime.current === 'HIGH_CONVICTION' && regime.confidence > 0.8) {
      adjusted.hourly *= 1.2;
      adjusted.monthly *= 0.9;
    }
    
    return adjusted;
  }
  
  // ============================================================================
  // Stability Adjustment
  // ============================================================================
  
  private applyStabilityAdjustment(
    weights: { hourly: number; weekly: number; monthly: number },
    stability: MetaControllerInput['horizonStability'],
    alignment: MetaControllerInput['signalAlignment']
  ): { hourly: number; weekly: number; monthly: number } {
    // Weight by stability scores
    const adjusted = {
      hourly: weights.hourly * (0.5 + 0.5 * stability.hourly),
      weekly: weights.weekly * (0.5 + 0.5 * stability.weekly),
      monthly: weights.monthly * (0.5 + 0.5 * stability.monthly),
    };
    
    // Boost if high alignment
    if (alignment.overall > 0.7) {
      // High agreement: boost all equally
      adjusted.hourly *= 1.1;
      adjusted.weekly *= 1.1;
      adjusted.monthly *= 1.1;
    }
    
    // Penalize if low alignment
    if (alignment.overall < 0.4) {
      // Low agreement: shift to monthly (most stable)
      adjusted.hourly *= 0.8;
      adjusted.monthly *= 1.2;
    }
    
    return adjusted;
  }
  
  // ============================================================================
  // Normalization
  // ============================================================================
  
  private normalizeWeights(weights: {
    hourly: number;
    weekly: number;
    monthly: number;
  }): { w_hourly: number; w_weekly: number; w_monthly: number } {
    const sum = weights.hourly + weights.weekly + weights.monthly;
    
    if (sum === 0) {
      return { w_hourly: 0.33, w_weekly: 0.33, w_monthly: 0.34 };
    }
    
    return {
      w_hourly: weights.hourly / sum,
      w_weekly: weights.weekly / sum,
      w_monthly: weights.monthly / sum,
    };
  }
  
  // ============================================================================
  // Exposure Scaling
  // ============================================================================
  
  private calculateExposureScaling(input: MetaControllerInput): number {
    let scaling = 1.0;
    
    // Scale down in high volatility
    if (input.volatility.regime === 'extreme') {
      scaling *= 0.5;
    } else if (input.volatility.regime === 'high') {
      scaling *= 0.75;
    }
    
    // Scale down if regime is fragile
    if (input.regime.fragility > 0.7) {
      scaling *= 0.7;
    }
    
    // Scale down if low confidence
    if (input.regime.confidence < 0.5) {
      scaling *= 0.8;
    }
    
    // Scale down if transitioning
    if (input.regime.transitionState === 'SHIFTING' || input.regime.transitionState === 'VOLATILE') {
      scaling *= 0.6;
    }
    
    // Scale up if high conviction and stable
    if (input.regime.current === 'HIGH_CONVICTION' && 
        input.regime.transitionState === 'STABLE' &&
        input.regime.confidence > 0.8) {
      scaling *= 1.5;
    }
    
    // Clamp to [0, 2]
    return Math.max(0, Math.min(2, scaling));
  }
  
  // ============================================================================
  // Risk Aversion
  // ============================================================================
  
  private calculateRiskAversion(input: MetaControllerInput): number {
    let riskAversion = 5.0;  // Neutral
    
    // Increase in crisis
    if (input.regime.current === 'CRISIS') {
      riskAversion += 3.0;
    } else if (input.regime.current === 'DEFENSIVE') {
      riskAversion += 1.5;
    }
    
    // Increase in high volatility
    if (input.volatility.regime === 'extreme') {
      riskAversion += 2.0;
    } else if (input.volatility.regime === 'high') {
      riskAversion += 1.0;
    }
    
    // Increase if fragile
    if (input.regime.fragility > 0.7) {
      riskAversion += 1.5;
    }
    
    // Decrease if high conviction
    if (input.regime.current === 'HIGH_CONVICTION' && input.regime.confidence > 0.8) {
      riskAversion -= 2.0;
    }
    
    // Clamp to [1, 10]
    return Math.max(1, Math.min(10, riskAversion));
  }
  
  // ============================================================================
  // Decision Confidence
  // ============================================================================
  
  private calculateDecisionConfidence(input: MetaControllerInput): number {
    let confidence = 0.5;
    
    // High if stable regime
    if (input.regime.transitionState === 'STABLE') {
      confidence += 0.2;
    }
    
    // High if good signal alignment
    if (input.signalAlignment.overall > 0.7) {
      confidence += 0.2;
    }
    
    // High if high regime confidence
    confidence += input.regime.confidence * 0.3;
    
    // Low if volatile or shifting
    if (input.regime.transitionState === 'VOLATILE' || input.regime.transitionState === 'SHIFTING') {
      confidence -= 0.3;
    }
    
    return Math.max(0, Math.min(1, confidence));
  }
  
  // ============================================================================
  // Reasoning Generation
  // ============================================================================
  
  private generateReasoning(
    input: MetaControllerInput,
    weights: { w_hourly: number; w_weekly: number; w_monthly: number },
    exposure: number
  ): { primaryFactor: string; secondaryFactors: string[]; warnings: string[] } {
    const factors: string[] = [];
    const warnings: string[] = [];
    
    // Determine primary factor
    let primaryFactor = '';
    
    if (input.volatility.regime === 'extreme') {
      primaryFactor = 'Extreme volatility detected - shifted to monthly (60%) for stability';
      factors.push('High market stress requires structural signals');
    } else if (input.regime.current === 'CRISIS') {
      primaryFactor = 'Crisis regime - prioritizing monthly signals for safety';
      factors.push('Defensive positioning with long-term perspective');
    } else if (input.regime.transitionState === 'VOLATILE') {
      primaryFactor = 'Volatile regime transitions - balanced allocation across horizons';
      factors.push('Multiple recent transitions detected');
    } else if (input.signalAlignment.overall < 0.4) {
      primaryFactor = 'Low signal alignment - shifting to monthly for stability';
      factors.push('Horizons disagree on market direction');
    } else if (weights.w_hourly > 0.5) {
      primaryFactor = 'Low volatility + high conviction - exploiting short-term opportunities';
      factors.push('Favorable conditions for intraday trading');
    } else {
      primaryFactor = 'Balanced allocation based on current market conditions';
    }
    
    // Add secondary factors
    if (input.regime.fragility > 0.7) {
      factors.push(`High regime fragility (${(input.regime.fragility * 100).toFixed(0)}%)`);
      warnings.push('Regime transition likely - monitor closely');
    }
    
    if (input.regime.confidence < 0.5) {
      warnings.push('Low regime confidence - exercise caution');
    }
    
    if (exposure < 0.7) {
      warnings.push(`Reduced exposure (${(exposure * 100).toFixed(0)}%) due to elevated risk`);
    }
    
    return {
      primaryFactor,
      secondaryFactors: factors.slice(0, 3),
      warnings,
    };
  }
  
  // ============================================================================
  // Decision Tree Initialization (Simplified)
  // ============================================================================
  
  private initializeDecisionTrees(): DecisionNode[] {
    // Initialize with 5 simple decision trees
    // In production, these would be trained on historical data
    
    const trees: DecisionNode[] = [];
    
    // Tree 1: Volatility-based
    trees.push({
      feature: 'volatility_current',
      threshold: 0.5,
      leftChild: [0.5, 0.3, 0.2],   // Low vol: more hourly
      rightChild: [0.2, 0.3, 0.5],  // High vol: more monthly
    });
    
    // Tree 2: Regime-based
    trees.push({
      feature: 'regime_current',
      threshold: 0.5,
      leftChild: [0.2, 0.3, 0.5],   // Crisis/Defensive: more monthly
      rightChild: [0.5, 0.3, 0.2],  // Risk-on/High conviction: more hourly
    });
    
    // Tree 3: Stability-based
    trees.push({
      feature: 'transition_state',
      threshold: 0.5,
      leftChild: [0.4, 0.4, 0.2],   // Stable: balanced
      rightChild: [0.2, 0.3, 0.5],  // Unstable: more monthly
    });
    
    // Tree 4: Alignment-based
    trees.push({
      feature: 'regime_confidence',
      threshold: 0.7,
      leftChild: [0.3, 0.3, 0.4],   // Low confidence: cautious
      rightChild: [0.5, 0.3, 0.2],  // High confidence: aggressive
    });
    
    // Tree 5: Fragility-based
    trees.push({
      feature: 'regime_fragility',
      threshold: 0.6,
      leftChild: [0.4, 0.4, 0.2],   // Low fragility: balanced
      rightChild: [0.2, 0.3, 0.5],  // High fragility: defensive
    });
    
    return trees;
  }
  
  // ============================================================================
  // Model Update (Online Learning - Placeholder)
  // ============================================================================
  
  /**
   * Update model based on realized performance
   * (Placeholder for future implementation)
   */
  updateModel(
    input: MetaControllerInput,
    actualWeights: { w_hourly: number; w_weekly: number; w_monthly: number },
    realizedSharpe: { hourly: number; weekly: number; monthly: number }
  ): void {
    // TODO: Implement online learning
    // - Calculate prediction error
    // - Update tree parameters using gradient boosting
    // - Adjust feature importance
    
    console.log('[MetaController] Online learning not yet implemented');
  }
}
