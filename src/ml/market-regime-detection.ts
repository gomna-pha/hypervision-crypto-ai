/**
 * Market Regime Identification Layer
 * 
 * Identifies current market regime using Hidden Markov Model (HMM)
 * and transition probabilities.
 * 
 * Regimes:
 * - CRISIS_STRESS: High volatility, extreme fear, negative returns
 * - DEFENSIVE: Moderate volatility, cautious sentiment
 * - NEUTRAL: Normal market conditions
 * - RISK_ON: Low volatility, positive sentiment
 * - HIGH_CONVICTION: Strong trends, high confidence
 */

export enum MarketRegime {
  CRISIS_STRESS = 'crisis_stress',
  DEFENSIVE = 'defensive',
  NEUTRAL = 'neutral',
  RISK_ON = 'risk_on',
  HIGH_CONVICTION = 'high_conviction',
}

export interface RegimeFeatures {
  volatility: number;       // 0-100 (VIX-like)
  returns: number;          // Daily returns (%)
  sentiment: number;        // 0-100 (Fear & Greed)
  volume: number;           // Normalized volume ratio
  liquidity: number;        // 0-100 liquidity score
  spread: number;           // Cross-exchange spread (bps)
}

export interface RegimeState {
  regime: MarketRegime;
  confidence: number;       // 0-1
  transitionProb: Record<MarketRegime, number>; // Probability of transitioning to each regime
  features: RegimeFeatures;
  timestamp: Date;
}

export interface HMMConfig {
  numStates: number;        // Number of hidden states (regimes)
  numIterations: number;    // Max iterations for Baum-Welch
  convergenceThreshold: number;
}

/**
 * Simple HMM implementation for regime detection
 */
export class MarketRegimeDetector {
  private config: HMMConfig;
  private transitionMatrix: number[][];   // A[i][j] = P(state_j | state_i)
  private emissionMatrix: number[][];      // B[i][k] = P(obs_k | state_i)
  private initialProb: number[];           // π[i] = P(start in state_i)
  private currentState: MarketRegime;
  private stateHistory: MarketRegime[];

  constructor(config: Partial<HMMConfig> = {}) {
    this.config = {
      numStates: 5,
      numIterations: config.numIterations || 100,
      convergenceThreshold: config.convergenceThreshold || 0.001,
    };

    // Initialize matrices with uniform distribution
    this.transitionMatrix = this.initializeTransitionMatrix();
    this.emissionMatrix = this.initializeEmissionMatrix();
    this.initialProb = this.initializeInitialProb();
    this.currentState = MarketRegime.NEUTRAL;
    this.stateHistory = [];
  }

  /**
   * Initialize transition matrix with domain knowledge
   */
  private initializeTransitionMatrix(): number[][] {
    // Transition probabilities based on market behavior
    // Rows: current state, Columns: next state
    const matrix = [
      // CRISIS_STRESS -> [CRISIS, DEFENSIVE, NEUTRAL, RISK_ON, HIGH_CONV]
      [0.6, 0.3, 0.08, 0.01, 0.01],  // Crisis tends to persist or move to defensive
      // DEFENSIVE -> 
      [0.1, 0.5, 0.3, 0.08, 0.02],   // Defensive can move to neutral or stay
      // NEUTRAL ->
      [0.05, 0.2, 0.5, 0.2, 0.05],   // Neutral is most stable
      // RISK_ON ->
      [0.01, 0.08, 0.3, 0.5, 0.11],  // Risk-on tends to persist or high conviction
      // HIGH_CONVICTION ->
      [0.05, 0.1, 0.25, 0.4, 0.2],   // High conviction can reverse quickly
    ];

    return matrix;
  }

  /**
   * Initialize emission matrix (simplified)
   */
  private initializeEmissionMatrix(): number[][] {
    // Simplified: discretize observations into 10 bins
    const numBins = 10;
    const matrix: number[][] = [];

    for (let i = 0; i < this.config.numStates; i++) {
      const row: number[] = [];
      for (let j = 0; j < numBins; j++) {
        // Gaussian-like distribution around expected observation
        row.push(1 / numBins + Math.random() * 0.01);
      }
      // Normalize
      const sum = row.reduce((s, v) => s + v, 0);
      matrix.push(row.map(v => v / sum));
    }

    return matrix;
  }

  /**
   * Initialize initial state probabilities
   */
  private initializeInitialProb(): number[] {
    // Start with neutral as most likely
    return [0.1, 0.15, 0.5, 0.15, 0.1];
  }

  /**
   * Detect current market regime using features
   */
  detectRegime(features: RegimeFeatures): RegimeState {
    // Rule-based classification (simplified HMM)
    const regime = this.classifyRegime(features);
    const confidence = this.calculateConfidence(features, regime);
    const transitionProb = this.getTransitionProbabilities(regime);

    this.currentState = regime;
    this.stateHistory.push(regime);

    // Keep last 100 states
    if (this.stateHistory.length > 100) {
      this.stateHistory.shift();
    }

    return {
      regime,
      confidence,
      transitionProb,
      features,
      timestamp: new Date(),
    };
  }

  /**
   * Rule-based regime classification
   */
  private classifyRegime(features: RegimeFeatures): MarketRegime {
    const { volatility, returns, sentiment, volume, liquidity, spread } = features;

    // Crisis/Stress: High vol (>40), negative returns, extreme fear (<25)
    if (volatility > 40 && returns < -2 && sentiment < 25) {
      return MarketRegime.CRISIS_STRESS;
    }

    // High Conviction: Moderate vol, strong returns, high sentiment
    if (volatility > 20 && volatility < 35 && Math.abs(returns) > 3 && sentiment > 60) {
      return MarketRegime.HIGH_CONVICTION;
    }

    // Risk-On: Low vol (<20), positive returns, positive sentiment (>55)
    if (volatility < 20 && returns > 0 && sentiment > 55) {
      return MarketRegime.RISK_ON;
    }

    // Defensive: Moderate vol (25-40), cautious sentiment (40-55)
    if (volatility >= 25 && volatility <= 40 && sentiment >= 40 && sentiment <= 55) {
      return MarketRegime.DEFENSIVE;
    }

    // Default: Neutral
    return MarketRegime.NEUTRAL;
  }

  /**
   * Calculate confidence in regime classification
   */
  private calculateConfidence(features: RegimeFeatures, regime: MarketRegime): number {
    const { volatility, returns, sentiment, volume, liquidity } = features;

    // Confidence based on how clearly features match regime characteristics
    let confidence = 0.5; // Base confidence

    switch (regime) {
      case MarketRegime.CRISIS_STRESS:
        // Strong negative signals increase confidence
        if (volatility > 50) confidence += 0.2;
        if (returns < -5) confidence += 0.15;
        if (sentiment < 20) confidence += 0.15;
        break;

      case MarketRegime.HIGH_CONVICTION:
        // Strong directional signals
        if (Math.abs(returns) > 5) confidence += 0.2;
        if (sentiment > 70 || sentiment < 30) confidence += 0.15;
        if (volume > 1.5) confidence += 0.15;
        break;

      case MarketRegime.RISK_ON:
        // Positive signals
        if (volatility < 15) confidence += 0.2;
        if (returns > 3) confidence += 0.15;
        if (sentiment > 65) confidence += 0.15;
        break;

      case MarketRegime.DEFENSIVE:
        // Moderate signals
        if (volatility >= 25 && volatility <= 35) confidence += 0.15;
        if (sentiment >= 45 && sentiment <= 55) confidence += 0.15;
        break;

      case MarketRegime.NEUTRAL:
        // Low volatility, neutral sentiment
        if (volatility >= 15 && volatility <= 25) confidence += 0.15;
        if (sentiment >= 45 && sentiment <= 55) confidence += 0.15;
        break;
    }

    return Math.min(1, Math.max(0, confidence));
  }

  /**
   * Get transition probabilities from current regime
   */
  private getTransitionProbabilities(regime: MarketRegime): Record<MarketRegime, number> {
    const regimeIndex = this.getRegimeIndex(regime);
    const probs = this.transitionMatrix[regimeIndex];

    return {
      [MarketRegime.CRISIS_STRESS]: probs[0],
      [MarketRegime.DEFENSIVE]: probs[1],
      [MarketRegime.NEUTRAL]: probs[2],
      [MarketRegime.RISK_ON]: probs[3],
      [MarketRegime.HIGH_CONVICTION]: probs[4],
    };
  }

  /**
   * Get regime index for matrix access
   */
  private getRegimeIndex(regime: MarketRegime): number {
    const mapping: Record<MarketRegime, number> = {
      [MarketRegime.CRISIS_STRESS]: 0,
      [MarketRegime.DEFENSIVE]: 1,
      [MarketRegime.NEUTRAL]: 2,
      [MarketRegime.RISK_ON]: 3,
      [MarketRegime.HIGH_CONVICTION]: 4,
    };
    return mapping[regime];
  }

  /**
   * Viterbi algorithm: find most likely sequence of states
   */
  viterbi(observations: number[]): MarketRegime[] {
    const T = observations.length;
    const N = this.config.numStates;

    // Delta[t][i] = max probability of path ending in state i at time t
    const delta: number[][] = Array(T).fill(0).map(() => Array(N).fill(0));
    // Psi[t][i] = most likely previous state leading to state i at time t
    const psi: number[][] = Array(T).fill(0).map(() => Array(N).fill(0));

    // Initialization
    for (let i = 0; i < N; i++) {
      delta[0][i] = this.initialProb[i] * this.emissionMatrix[i][observations[0]];
      psi[0][i] = 0;
    }

    // Recursion
    for (let t = 1; t < T; t++) {
      for (let j = 0; j < N; j++) {
        let maxProb = 0;
        let maxState = 0;

        for (let i = 0; i < N; i++) {
          const prob = delta[t - 1][i] * this.transitionMatrix[i][j];
          if (prob > maxProb) {
            maxProb = prob;
            maxState = i;
          }
        }

        delta[t][j] = maxProb * this.emissionMatrix[j][observations[t]];
        psi[t][j] = maxState;
      }
    }

    // Termination: find best final state
    let maxProb = 0;
    let maxState = 0;
    for (let i = 0; i < N; i++) {
      if (delta[T - 1][i] > maxProb) {
        maxProb = delta[T - 1][i];
        maxState = i;
      }
    }

    // Backtrack to find path
    const path: number[] = Array(T);
    path[T - 1] = maxState;
    for (let t = T - 2; t >= 0; t--) {
      path[t] = psi[t + 1][path[t + 1]];
    }

    // Convert indices to regimes
    const regimes = path.map(idx => this.indexToRegime(idx));
    return regimes;
  }

  /**
   * Convert state index to regime
   */
  private indexToRegime(index: number): MarketRegime {
    const mapping: MarketRegime[] = [
      MarketRegime.CRISIS_STRESS,
      MarketRegime.DEFENSIVE,
      MarketRegime.NEUTRAL,
      MarketRegime.RISK_ON,
      MarketRegime.HIGH_CONVICTION,
    ];
    return mapping[index];
  }

  /**
   * Get current regime
   */
  getCurrentRegime(): MarketRegime {
    return this.currentState;
  }

  /**
   * Get regime history
   */
  getRegimeHistory(): MarketRegime[] {
    return this.stateHistory;
  }

  /**
   * Get regime persistence (how long in current regime)
   */
  getRegimePersistence(): number {
    if (this.stateHistory.length === 0) return 0;

    let count = 0;
    for (let i = this.stateHistory.length - 1; i >= 0; i--) {
      if (this.stateHistory[i] === this.currentState) {
        count++;
      } else {
        break;
      }
    }

    return count;
  }
}

/**
 * Example usage:
 * 
 * const detector = new MarketRegimeDetector();
 * 
 * const features: RegimeFeatures = {
 *   volatility: 45,
 *   returns: -3.5,
 *   sentiment: 18,
 *   volume: 1.8,
 *   liquidity: 65,
 *   spread: 25,
 * };
 * 
 * const state = detector.detectRegime(features);
 * console.log('Current regime:', state.regime);
 * console.log('Confidence:', state.confidence);
 * console.log('Transition probabilities:', state.transitionProb);
 */
