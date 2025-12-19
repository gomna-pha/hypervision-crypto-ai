/**
 * XGBoost Meta-Model — Arbitrage Confidence Layer
 * 
 * Inputs:
 * - GA-selected signals (weights & active flags)
 * - Hyperbolic distances (signal-regime proximity)
 * - Regime label & transition probability
 * - Volatility & liquidity state
 * - CNN pattern confidence
 * 
 * Outputs:
 * - Arbitrage confidence score (0-100)
 * - Signal disagreement flags
 * - Dynamic exposure / leverage scaler
 * 
 * Note: This is a lightweight implementation without external XGBoost library.
 * For production, integrate actual XGBoost via WASM or API.
 */

import { SignalGenome } from './genetic-algorithm';
import { HyperbolicPoint } from './hyperbolic-embedding';
import { MarketRegime, RegimeState } from './market-regime-detection';
import { AgentSignal } from './agent-signal';

export interface MetaModelInput {
  // GA-selected signals
  gaGenome: SignalGenome;
  agentSignals: AgentSignal[];

  // Hyperbolic embeddings
  signalEmbeddings: Map<string, HyperbolicPoint>;
  regimeEmbedding: HyperbolicPoint;

  // Market regime
  regimeState: RegimeState;

  // Market state
  volatility: number;        // 0-100
  liquidity: number;         // 0-100
  spread: number;            // bps
}

export interface MetaModelOutput {
  confidenceScore: number;   // 0-100 (main output)
  action: 'EXECUTE' | 'WAIT' | 'REDUCE'; // Trading action
  
  // Signal analysis
  signalAgreement: number;   // 0-1 (how much signals agree)
  signalDivergence: boolean; // True if major disagreement
  
  // Dynamic scaling
  exposureScaler: number;    // 0-2 (multiply position size)
  leverageScaler: number;    // 0-2 (multiply leverage)
  
  // Feature contributions (for explainability)
  featureImportance: Record<string, number>;
  
  // Metadata
  timestamp: Date;
  latencyMs: number;
}

export interface DecisionTreeNode {
  feature: string;
  threshold: number;
  leftChild: DecisionTreeNode | number; // Number = leaf value
  rightChild: DecisionTreeNode | number;
}

export class XGBoostMetaModel {
  private trees: DecisionTreeNode[];
  private learningRate: number;
  private numTrees: number;
  private maxDepth: number;
  private featureNames: string[];

  constructor(config: {
    learningRate?: number;
    numTrees?: number;
    maxDepth?: number;
  } = {}) {
    this.learningRate = config.learningRate || 0.1;
    this.numTrees = config.numTrees || 10;
    this.maxDepth = config.maxDepth || 6;
    this.trees = [];
    this.featureNames = [
      'gaFitness',
      'gaActiveCount',
      'gaWeightEntropy',
      'signalMean',
      'signalStd',
      'signalAgreement',
      'hyperbolicDistance',
      'regimeConfidence',
      'regimeTransitionRisk',
      'volatility',
      'liquidity',
      'spread',
      'cnnConfidence',
    ];

    // Initialize with pre-trained decision trees (simplified)
    this.initializePretrainedTrees();
  }

  /**
   * Initialize pre-trained decision trees (simplified for demo)
   * In production, load actual trained XGBoost model
   */
  private initializePretrainedTrees(): void {
    // Tree 1: Focus on signal agreement
    this.trees.push({
      feature: 'signalAgreement',
      threshold: 0.7,
      leftChild: {
        feature: 'volatility',
        threshold: 30,
        leftChild: 0.8,  // High agreement + low vol = high confidence
        rightChild: 0.5, // High agreement + high vol = moderate
      },
      rightChild: {
        feature: 'liquidity',
        threshold: 60,
        leftChild: 0.3,  // Low agreement + low liquidity = low confidence
        rightChild: 0.4,
      },
    });

    // Tree 2: Focus on regime
    this.trees.push({
      feature: 'regimeConfidence',
      threshold: 0.6,
      leftChild: {
        feature: 'regimeTransitionRisk',
        threshold: 0.4,
        leftChild: 0.6,
        rightChild: 0.3,
      },
      rightChild: {
        feature: 'signalMean',
        threshold: 0.5,
        leftChild: 0.4,
        rightChild: 0.7,
      },
    });

    // Tree 3: Focus on hyperbolic distance
    this.trees.push({
      feature: 'hyperbolicDistance',
      threshold: 1.0,
      leftChild: {
        feature: 'cnnConfidence',
        threshold: 0.75,
        leftChild: 0.7,
        rightChild: 0.5,
      },
      rightChild: {
        feature: 'spread',
        threshold: 20,
        leftChild: 0.4,
        rightChild: 0.6,
      },
    });

    // Add more trees for robustness (simplified)
    for (let i = 0; i < this.numTrees - 3; i++) {
      this.trees.push({
        feature: 'signalAgreement',
        threshold: 0.5,
        leftChild: 0.4 + Math.random() * 0.2,
        rightChild: 0.6 + Math.random() * 0.2,
      });
    }
  }

  /**
   * Predict arbitrage confidence score
   */
  predict(input: MetaModelInput): MetaModelOutput {
    const startTime = Date.now();

    // Extract features
    const features = this.extractFeatures(input);

    // Ensemble prediction (sum of all tree predictions)
    let prediction = 0;
    const treeContributions: number[] = [];

    for (const tree of this.trees) {
      const treeOutput = this.evaluateTree(tree, features);
      prediction += this.learningRate * treeOutput;
      treeContributions.push(treeOutput);
    }

    // Sigmoid transformation to [0, 1]
    const confidenceProbability = 1 / (1 + Math.exp(-prediction));
    const confidenceScore = confidenceProbability * 100;

    // Calculate signal agreement
    const signalAgreement = features.signalAgreement;
    const signalDivergence = signalAgreement < 0.5;

    // Dynamic scalers based on confidence
    const exposureScaler = this.calculateExposureScaler(confidenceScore, input);
    const leverageScaler = this.calculateLeverageScaler(confidenceScore, input);

    // Determine action
    const action = this.determineAction(confidenceScore, input);

    // Feature importance (simplified: variance-based)
    const featureImportance = this.calculateFeatureImportance(features);

    return {
      confidenceScore,
      action,
      signalAgreement,
      signalDivergence,
      exposureScaler,
      leverageScaler,
      featureImportance,
      timestamp: new Date(),
      latencyMs: Date.now() - startTime,
    };
  }

  /**
   * Extract features from input
   */
  private extractFeatures(input: MetaModelInput): Record<string, number> {
    const { gaGenome, agentSignals, signalEmbeddings, regimeEmbedding, regimeState, volatility, liquidity, spread } = input;

    // GA features
    const gaFitness = gaGenome.fitness;
    const gaActiveCount = gaGenome.activeSignals.reduce((sum, val) => sum + val, 0);
    const gaWeightEntropy = this.calculateEntropy(gaGenome.weights);

    // Signal features
    const signalValues = agentSignals.map(s => s.signal);
    const signalConfidences = agentSignals.map(s => s.confidence);
    const signalMean = signalValues.reduce((sum, val) => sum + val, 0) / signalValues.length;
    const signalStd = this.calculateStdDev(signalValues);
    const signalAgreement = this.calculateSignalAgreement(signalValues, signalConfidences);

    // Hyperbolic features
    const hyperbolicDistance = this.calculateAverageHyperbolicDistance(signalEmbeddings, regimeEmbedding);

    // Regime features
    const regimeConfidence = regimeState.confidence;
    const regimeTransitionRisk = this.calculateRegimeTransitionRisk(regimeState);

    // CNN confidence (from CNN Pattern Agent)
    const cnnAgent = agentSignals.find(s => s.agentId === 'cnn_pattern_agent');
    const cnnConfidence = cnnAgent ? cnnAgent.confidence : 0.5;

    return {
      gaFitness,
      gaActiveCount,
      gaWeightEntropy,
      signalMean,
      signalStd,
      signalAgreement,
      hyperbolicDistance,
      regimeConfidence,
      regimeTransitionRisk,
      volatility,
      liquidity,
      spread,
      cnnConfidence,
    };
  }

  /**
   * Evaluate a single decision tree
   */
  private evaluateTree(node: DecisionTreeNode | number, features: Record<string, number>): number {
    // Leaf node
    if (typeof node === 'number') {
      return node;
    }

    // Internal node
    const featureValue = features[node.feature] || 0;
    
    if (featureValue <= node.threshold) {
      return this.evaluateTree(node.leftChild, features);
    } else {
      return this.evaluateTree(node.rightChild, features);
    }
  }

  /**
   * Calculate entropy of weight distribution
   */
  private calculateEntropy(weights: number[]): number {
    let entropy = 0;
    for (const w of weights) {
      if (w > 0) {
        entropy -= w * Math.log2(w);
      }
    }
    return entropy;
  }

  /**
   * Calculate standard deviation
   */
  private calculateStdDev(values: number[]): number {
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const variance = values.reduce((sum, val) => sum + Math.pow(val - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  /**
   * Calculate signal agreement (weighted by confidence)
   */
  private calculateSignalAgreement(signals: number[], confidences: number[]): number {
    if (signals.length === 0) return 0;

    // Weighted average of signal directions
    const weightedSum = signals.reduce((sum, sig, i) => sum + sig * confidences[i], 0);
    const weightSum = confidences.reduce((sum, conf) => sum + conf, 0);
    const avgSignal = weightSum > 0 ? weightedSum / weightSum : 0;

    // Agreement = how close all signals are to average
    const deviations = signals.map((sig, i) => Math.abs(sig - avgSignal) * confidences[i]);
    const avgDeviation = deviations.reduce((sum, dev) => sum + dev, 0) / weightSum;

    return 1 - Math.min(1, avgDeviation);
  }

  /**
   * Calculate average hyperbolic distance from signals to regime
   */
  private calculateAverageHyperbolicDistance(
    signalEmbeddings: Map<string, HyperbolicPoint>,
    regimeEmbedding: HyperbolicPoint
  ): number {
    const distances: number[] = [];

    for (const [_, signalPoint] of signalEmbeddings) {
      const dist = this.poincareDistance(signalPoint, regimeEmbedding);
      distances.push(dist);
    }

    return distances.length > 0
      ? distances.reduce((sum, d) => sum + d, 0) / distances.length
      : 1.0;
  }

  /**
   * Poincaré distance (simplified)
   */
  private poincareDistance(p1: HyperbolicPoint, p2: HyperbolicPoint): number {
    const diff = p1.coords.map((c, i) => c - p2.coords[i]);
    const normDiffSq = diff.reduce((sum, d) => sum + d * d, 0);

    const norm1Sq = p1.coords.reduce((sum, c) => sum + c * c, 0);
    const norm2Sq = p2.coords.reduce((sum, c) => sum + c * c, 0);

    const numerator = 2 * normDiffSq;
    const denominator = (1 - norm1Sq) * (1 - norm2Sq);

    const ratio = 1 + numerator / (denominator + 1e-10);
    return Math.acosh(Math.max(1.0, ratio));
  }

  /**
   * Calculate regime transition risk (high if likely to transition soon)
   */
  private calculateRegimeTransitionRisk(regimeState: RegimeState): number {
    const { regime, transitionProb } = regimeState;

    // Probability of staying in current regime
    const stayProb = transitionProb[regime];

    // Risk = 1 - stayProb
    return 1 - stayProb;
  }

  /**
   * Calculate exposure scaler (position size multiplier)
   */
  private calculateExposureScaler(confidenceScore: number, input: MetaModelInput): number {
    const { regimeState, volatility, liquidity } = input;

    // Base scaler from confidence
    let scaler = confidenceScore / 100;

    // Adjust for regime
    if (regimeState.regime === MarketRegime.CRISIS_STRESS) {
      scaler *= 0.3; // Reduce exposure in crisis
    } else if (regimeState.regime === MarketRegime.HIGH_CONVICTION) {
      scaler *= 1.5; // Increase exposure in high conviction
    }

    // Adjust for volatility
    if (volatility > 40) {
      scaler *= 0.7; // Reduce exposure in high volatility
    }

    // Adjust for liquidity
    if (liquidity < 50) {
      scaler *= 0.5; // Reduce exposure in low liquidity
    }

    return Math.max(0, Math.min(2, scaler));
  }

  /**
   * Calculate leverage scaler
   */
  private calculateLeverageScaler(confidenceScore: number, input: MetaModelInput): number {
    const { volatility, liquidity } = input;

    // Base leverage from confidence
    let scaler = confidenceScore / 100;

    // Conservative leverage adjustment
    if (volatility > 35) {
      scaler *= 0.5; // Reduce leverage in high volatility
    }

    if (liquidity < 60) {
      scaler *= 0.6; // Reduce leverage in low liquidity
    }

    return Math.max(0, Math.min(2, scaler));
  }

  /**
   * Determine trading action
   */
  private determineAction(confidenceScore: number, input: MetaModelInput): 'EXECUTE' | 'WAIT' | 'REDUCE' {
    const { regimeState, volatility } = input;

    // High confidence + favorable regime = EXECUTE
    if (confidenceScore > 75 && regimeState.regime !== MarketRegime.CRISIS_STRESS) {
      return 'EXECUTE';
    }

    // Low confidence or crisis = REDUCE
    if (confidenceScore < 50 || regimeState.regime === MarketRegime.CRISIS_STRESS) {
      return 'REDUCE';
    }

    // Moderate conditions = WAIT
    return 'WAIT';
  }

  /**
   * Calculate feature importance (simplified)
   */
  private calculateFeatureImportance(features: Record<string, number>): Record<string, number> {
    const importance: Record<string, number> = {};

    // Normalized feature values as proxy for importance
    const values = Object.values(features);
    const maxVal = Math.max(...values.map(Math.abs));

    for (const [key, value] of Object.entries(features)) {
      importance[key] = maxVal > 0 ? Math.abs(value) / maxVal : 0;
    }

    return importance;
  }

  /**
   * Train model (placeholder for future implementation)
   */
  async train(
    trainingData: Array<{ input: MetaModelInput; target: number }>
  ): Promise<void> {
    // TODO: Implement actual XGBoost training
    console.log('Training XGBoost model with', trainingData.length, 'samples');
    console.log('Note: Using pre-trained trees for now');
  }

  /**
   * Get feature names
   */
  getFeatureNames(): string[] {
    return this.featureNames;
  }

  /**
   * Get number of trees
   */
  getNumTrees(): number {
    return this.trees.length;
  }
}

/**
 * Example usage:
 * 
 * const metaModel = new XGBoostMetaModel({
 *   learningRate: 0.1,
 *   numTrees: 10,
 *   maxDepth: 6,
 * });
 * 
 * const input: MetaModelInput = {
 *   gaGenome: bestGenome,
 *   agentSignals: [...],
 *   signalEmbeddings: new Map(),
 *   regimeEmbedding: {...},
 *   regimeState: {...},
 *   volatility: 25,
 *   liquidity: 75,
 *   spread: 15,
 * };
 * 
 * const output = metaModel.predict(input);
 * console.log('Confidence:', output.confidenceScore);
 * console.log('Action:', output.action);
 * console.log('Exposure scaler:', output.exposureScaler);
 */
