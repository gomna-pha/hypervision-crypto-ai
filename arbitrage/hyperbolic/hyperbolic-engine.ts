/**
 * Hyperbolic Embedding Layer - Poincaré Ball Model Implementation
 * Implements hyperbolic geometry for better encoding of hierarchical relationships
 * in market data, sentiment cascades, and multi-modal agent features
 */

import { AgentOutput } from '../core/base-agent.js';

export interface HyperbolicConfig {
  model: 'poincare' | 'lorentz' | 'klein';
  curvature: number;           // Negative curvature parameter (default: -1.0)
  embedding_dim: number;       // Embedding dimension (default: 128)
  learning_rate: number;       // Learning rate for updates (default: 0.01)
  max_norm: number;           // Maximum norm in Poincaré ball (default: 0.99)
}

export interface HyperbolicPoint {
  coordinates: number[];       // Point coordinates in hyperbolic space
  norm: number;               // Euclidean norm of coordinates
  timestamp: number;          // When this point was created
  metadata: {
    agent_name: string;
    confidence: number;
    features_hash: string;
  };
}

export interface HyperbolicDistance {
  distance: number;           // Hyperbolic distance between points
  similarity: number;         // Normalized similarity score [0, 1]
  geodesic_path?: number[][];  // Optional geodesic path between points
}

export interface NearestNeighbors {
  query_point: HyperbolicPoint;
  neighbors: Array<{
    point: HyperbolicPoint;
    distance: number;
    similarity: number;
    agent_name: string;
  }>;
  k: number;
}

/**
 * Hyperbolic Embedding Engine using Poincaré Ball Model
 */
export class HyperbolicEngine {
  private config: HyperbolicConfig;
  private embeddings: Map<string, HyperbolicPoint> = new Map();
  private agentHistory: Map<string, HyperbolicPoint[]> = new Map();
  private featureMapping: Map<string, number[]> = new Map();
  
  constructor(config: Partial<HyperbolicConfig> = {}) {
    this.config = {
      model: 'poincare',
      curvature: -1.0,
      embedding_dim: 128,
      learning_rate: 0.01,
      max_norm: 0.99,
      ...config
    };
  }

  /**
   * Map agent output to hyperbolic space
   */
  mapAgentToHyperbolic(agentOutput: AgentOutput): HyperbolicPoint {
    // Extract numerical features from agent output
    const features = this.extractNumericalFeatures(agentOutput);
    
    // Create or update feature embedding using MLP-like transformation
    const euclideanVector = this.createEuclideanEmbedding(features, agentOutput.agent_name);
    
    // Map to Poincaré ball using exponential map
    const hyperbolicPoint = this.exponentialMap(euclideanVector);
    
    // Create hyperbolic point with metadata
    const point: HyperbolicPoint = {
      coordinates: hyperbolicPoint,
      norm: this.euclideanNorm(hyperbolicPoint),
      timestamp: new Date(agentOutput.timestamp).getTime(),
      metadata: {
        agent_name: agentOutput.agent_name,
        confidence: agentOutput.confidence,
        features_hash: this.hashFeatures(features)
      }
    };

    // Store the embedding
    const key = `${agentOutput.agent_name}_${point.timestamp}`;
    this.embeddings.set(key, point);
    
    // Update agent history
    this.updateAgentHistory(agentOutput.agent_name, point);
    
    return point;
  }

  /**
   * Extract numerical features from agent output
   */
  private extractNumericalFeatures(agentOutput: AgentOutput): number[] {
    const features: number[] = [];
    
    // Add key signal and confidence
    features.push(agentOutput.key_signal);
    features.push(agentOutput.confidence);
    
    // Recursively extract numerical values from features object
    this.extractFromObject(agentOutput.features, features);
    
    // Pad or truncate to desired feature size
    const targetSize = 64; // Base feature size before embedding transformation
    
    if (features.length < targetSize) {
      // Pad with zeros
      while (features.length < targetSize) {
        features.push(0);
      }
    } else if (features.length > targetSize) {
      // Truncate to target size
      features.splice(targetSize);
    }
    
    return features;
  }

  /**
   * Recursively extract numerical values from nested objects
   */
  private extractFromObject(obj: any, features: number[]): void {
    if (typeof obj === 'number' && !isNaN(obj) && isFinite(obj)) {
      features.push(obj);
    } else if (typeof obj === 'object' && obj !== null) {
      for (const value of Object.values(obj)) {
        this.extractFromObject(value, features);
      }
    }
  }

  /**
   * Create Euclidean embedding using MLP-like transformation
   */
  private createEuclideanEmbedding(features: number[], agentName: string): number[] {
    const inputDim = features.length;
    const outputDim = this.config.embedding_dim;
    
    // Get or create weight matrix for this agent type
    const weightKey = `weights_${agentName}`;
    let weights = this.featureMapping.get(weightKey);
    
    if (!weights) {
      // Initialize random weights (Xavier/Glorot initialization)
      weights = this.initializeWeights(inputDim, outputDim);
      this.featureMapping.set(weightKey, weights);
    }
    
    // Simple linear transformation: output = features * W + bias
    const output = new Array(outputDim).fill(0);
    
    for (let i = 0; i < outputDim; i++) {
      for (let j = 0; j < inputDim; j++) {
        const weightIdx = i * inputDim + j;
        output[i] += features[j] * weights[weightIdx];
      }
      
      // Add bias and apply tanh activation
      const bias = weights[inputDim * outputDim + i] || 0;
      output[i] = Math.tanh(output[i] + bias);
    }
    
    return output;
  }

  /**
   * Initialize random weights using Xavier initialization
   */
  private initializeWeights(inputDim: number, outputDim: number): number[] {
    const totalWeights = inputDim * outputDim + outputDim; // Weights + biases
    const weights = new Array(totalWeights);
    const limit = Math.sqrt(6 / (inputDim + outputDim));
    
    for (let i = 0; i < totalWeights; i++) {
      weights[i] = (Math.random() * 2 - 1) * limit;
    }
    
    return weights;
  }

  /**
   * Exponential map from tangent space to Poincaré ball
   */
  private exponentialMap(euclideanVector: number[]): number[] {
    const c = Math.abs(this.config.curvature);
    const norm = this.euclideanNorm(euclideanVector);
    
    if (norm === 0) {
      return new Array(euclideanVector.length).fill(0);
    }
    
    // Exponential map formula: exp_0(v) = tanh(sqrt(c)*||v||) * v / (sqrt(c)*||v||)
    const sqrtC = Math.sqrt(c);
    const tanhFactor = Math.tanh(sqrtC * norm);
    const scalingFactor = tanhFactor / (sqrtC * norm);
    
    const result = euclideanVector.map(x => x * scalingFactor);
    
    // Ensure the point stays within the Poincaré ball
    const resultNorm = this.euclideanNorm(result);
    if (resultNorm >= this.config.max_norm) {
      const rescale = this.config.max_norm / resultNorm;
      return result.map(x => x * rescale);
    }
    
    return result;
  }

  /**
   * Calculate Euclidean norm of a vector
   */
  private euclideanNorm(vector: number[]): number {
    const sumSquares = vector.reduce((sum, x) => sum + x * x, 0);
    return Math.sqrt(sumSquares);
  }

  /**
   * Calculate hyperbolic distance between two points in Poincaré ball
   */
  calculateHyperbolicDistance(point1: HyperbolicPoint, point2: HyperbolicPoint): HyperbolicDistance {
    const c = Math.abs(this.config.curvature);
    
    // Poincaré distance formula: d(u,v) = (2/sqrt(c)) * artanh(sqrt(c) * ||u-v||_M)
    // where ||u-v||_M = ||u-v|| / ((1-c||u||²)(1-c||v||²))^(1/2)
    
    const u = point1.coordinates;
    const v = point2.coordinates;
    
    // Calculate u - v
    const diff = u.map((x, i) => x - v[i]);
    const diffNorm = this.euclideanNorm(diff);
    
    if (diffNorm === 0) {
      return { distance: 0, similarity: 1 };
    }
    
    // Calculate denominators
    const uNormSq = u.reduce((sum, x) => sum + x * x, 0);
    const vNormSq = v.reduce((sum, x) => sum + x * x, 0);
    
    const denomU = 1 - c * uNormSq;
    const denomV = 1 - c * vNormSq;
    const denominator = Math.sqrt(Math.max(0.001, denomU * denomV)); // Avoid division by zero
    
    // Möbius metric
    const mobiusNorm = diffNorm / denominator;
    
    // Hyperbolic distance
    const sqrtC = Math.sqrt(c);
    const distance = (2 / sqrtC) * Math.atanh(Math.min(0.999, sqrtC * mobiusNorm));
    
    // Convert to similarity score (0 = dissimilar, 1 = very similar)
    const maxDistance = 10; // Reasonable maximum distance for normalization
    const normalizedDistance = Math.min(distance / maxDistance, 1);
    const similarity = 1 - normalizedDistance;
    
    return {
      distance: isNaN(distance) ? Infinity : distance,
      similarity: isNaN(similarity) ? 0 : Math.max(0, Math.min(1, similarity))
    };
  }

  /**
   * Find k-nearest neighbors in hyperbolic space
   */
  findNearestNeighbors(queryPoint: HyperbolicPoint, k: number = 5): NearestNeighbors {
    const neighbors: Array<{
      point: HyperbolicPoint;
      distance: number;
      similarity: number;
      agent_name: string;
    }> = [];

    // Calculate distances to all other points
    for (const [key, point] of this.embeddings) {
      if (point.timestamp === queryPoint.timestamp && 
          point.metadata.agent_name === queryPoint.metadata.agent_name) {
        continue; // Skip same point
      }

      const distanceInfo = this.calculateHyperbolicDistance(queryPoint, point);
      
      neighbors.push({
        point,
        distance: distanceInfo.distance,
        similarity: distanceInfo.similarity,
        agent_name: point.metadata.agent_name
      });
    }

    // Sort by distance (ascending) and take top k
    neighbors.sort((a, b) => a.distance - b.distance);
    const topNeighbors = neighbors.slice(0, k);

    return {
      query_point: queryPoint,
      neighbors: topNeighbors,
      k: Math.min(k, topNeighbors.length)
    };
  }

  /**
   * Get contextual neighbors for fusion brain analysis
   */
  getContextualNeighbors(agentOutputs: AgentOutput[], k: number = 10): Map<string, NearestNeighbors> {
    const contextMap = new Map<string, NearestNeighbors>();

    for (const output of agentOutputs) {
      // Map to hyperbolic space
      const point = this.mapAgentToHyperbolic(output);
      
      // Find neighbors
      const neighbors = this.findNearestNeighbors(point, k);
      
      contextMap.set(output.agent_name, neighbors);
    }

    return contextMap;
  }

  /**
   * Update agent history
   */
  private updateAgentHistory(agentName: string, point: HyperbolicPoint): void {
    if (!this.agentHistory.has(agentName)) {
      this.agentHistory.set(agentName, []);
    }

    const history = this.agentHistory.get(agentName)!;
    history.push(point);

    // Keep only last 100 points per agent
    if (history.length > 100) {
      history.shift();
    }
  }

  /**
   * Calculate hash of features for change detection
   */
  private hashFeatures(features: number[]): string {
    const rounded = features.map(f => Math.round(f * 1000) / 1000);
    return rounded.join(',');
  }

  /**
   * Get agent trajectory in hyperbolic space
   */
  getAgentTrajectory(agentName: string, windowSize: number = 20): HyperbolicPoint[] {
    const history = this.agentHistory.get(agentName) || [];
    return history.slice(-windowSize);
  }

  /**
   * Calculate trajectory convergence/divergence
   */
  calculateTrajectoryDivergence(agentName1: string, agentName2: string, windowSize: number = 10): number {
    const traj1 = this.getAgentTrajectory(agentName1, windowSize);
    const traj2 = this.getAgentTrajectory(agentName2, windowSize);
    
    if (traj1.length < 2 || traj2.length < 2) return 0;

    let totalDivergence = 0;
    const minLength = Math.min(traj1.length, traj2.length);

    for (let i = 0; i < minLength; i++) {
      const distanceInfo = this.calculateHyperbolicDistance(traj1[i], traj2[i]);
      totalDivergence += distanceInfo.distance;
    }

    return totalDivergence / minLength;
  }

  /**
   * Detect anomalies in hyperbolic space
   */
  detectAnomalies(agentName: string, threshold: number = 2.0): HyperbolicPoint[] {
    const trajectory = this.getAgentTrajectory(agentName);
    const anomalies: HyperbolicPoint[] = [];

    if (trajectory.length < 3) return anomalies;

    for (let i = 1; i < trajectory.length - 1; i++) {
      const current = trajectory[i];
      const neighbors = this.findNearestNeighbors(current, 5);
      
      if (neighbors.neighbors.length > 0) {
        const avgDistance = neighbors.neighbors.reduce((sum, n) => sum + n.distance, 0) / neighbors.neighbors.length;
        
        if (avgDistance > threshold) {
          anomalies.push(current);
        }
      }
    }

    return anomalies;
  }

  /**
   * Export embeddings for analysis
   */
  exportEmbeddings(): Array<{
    agent_name: string;
    timestamp: number;
    coordinates: number[];
    confidence: number;
  }> {
    return Array.from(this.embeddings.values()).map(point => ({
      agent_name: point.metadata.agent_name,
      timestamp: point.timestamp,
      coordinates: point.coordinates,
      confidence: point.metadata.confidence
    }));
  }

  /**
   * Get engine statistics
   */
  getStatistics(): {
    total_points: number;
    agents_tracked: number;
    embedding_dimension: number;
    avg_confidence: number;
    coverage_by_agent: Record<string, number>;
  } {
    const agentCounts = new Map<string, number>();
    let totalConfidence = 0;

    for (const point of this.embeddings.values()) {
      const agentName = point.metadata.agent_name;
      agentCounts.set(agentName, (agentCounts.get(agentName) || 0) + 1);
      totalConfidence += point.metadata.confidence;
    }

    return {
      total_points: this.embeddings.size,
      agents_tracked: agentCounts.size,
      embedding_dimension: this.config.embedding_dim,
      avg_confidence: this.embeddings.size > 0 ? totalConfidence / this.embeddings.size : 0,
      coverage_by_agent: Object.fromEntries(agentCounts)
    };
  }

  /**
   * Clear old embeddings to manage memory
   */
  clearOldEmbeddings(maxAgeMs: number = 24 * 60 * 60 * 1000): number {
    const now = Date.now();
    let cleared = 0;

    for (const [key, point] of this.embeddings) {
      if (now - point.timestamp > maxAgeMs) {
        this.embeddings.delete(key);
        cleared++;
      }
    }

    return cleared;
  }
}

/**
 * Factory function to create HyperbolicEngine with default config
 */
export function createHyperbolicEngine(config?: Partial<HyperbolicConfig>): HyperbolicEngine {
  return new HyperbolicEngine(config);
}

// Export for testing
export { HyperbolicEngine as default };