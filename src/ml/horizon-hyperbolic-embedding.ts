/**
 * Horizon-Aware Hyperbolic Embedding
 * 
 * Extends hyperbolic embedding with horizon-specific features:
 * - Horizon distance metrics
 * - Signal robustness (radial distance from origin)
 * - Regime similarity (angular distance in Poincaré disk)
 * - Cross-horizon signal comparison
 * 
 * Poincaré Disk Interpretation:
 * - Radial distance (0 to 1): Signal robustness/strength
 *   - Near origin (< 0.3): Weak/uncertain signals
 *   - Mid-range (0.3-0.7): Moderate signals
 *   - Edge (> 0.7): Strong/confident signals
 * 
 * - Angular distance: Regime/signal similarity
 *   - Small angle: Similar regimes/signals
 *   - Orthogonal (90°): Independent signals
 *   - Opposite (180°): Contradictory signals
 */

import { HorizonIndexedSignal } from './multi-horizon-signal-pool';
import { TimeHorizon } from './time-scale-feature-store';
import { MarketRegime } from './market-regime-detection';
import { HorizonGraphNode, HorizonSignalNode, RegimeNode } from './horizon-hierarchical-graph';

// ============================================================================
// Hyperbolic Point with Horizon Context
// ============================================================================

export interface HorizonHyperbolicPoint {
  coords: number[];      // Coordinates in Poincaré disk (5D)
  norm: number;          // Radial distance [0, 1)
  angle: number;         // Angular position [0, 2π)
  id: string;            // Node ID
  type: 'signal' | 'horizon' | 'regime' | 'strategy';
  horizon?: TimeHorizon; // For signal nodes
  
  // Hyperbolic properties
  signalRobustness: number;    // Radial interpretation [0, 1]
  regimeSimilarity: number;    // To current regime [-1, 1]
}

// ============================================================================
// Horizon Distance Metrics
// ============================================================================

export interface HorizonDistanceMetrics {
  // Poincaré distances
  poincareDistance: number;           // Hyperbolic distance
  euclideanDistance: number;          // Euclidean distance in embedding space
  
  // Geometric metrics
  radialDistance: number;             // Difference in norms
  angularDistance: number;            // Angular separation [0, π]
  
  // Horizon-specific
  horizonPenalty: number;             // Penalty for different horizons
  decayAdjustedDistance: number;      // Distance adjusted for signal decay
}

// ============================================================================
// Horizon Hyperbolic Embedding
// ============================================================================

export class HorizonHyperbolicEmbedding {
  private dimension: number;
  private curvature: number;
  private learningRate: number;
  private maxIterations: number;
  private embeddings: Map<string, HorizonHyperbolicPoint>;
  
  // Current regime embedding (reference point)
  private currentRegimeEmbedding: HorizonHyperbolicPoint | null = null;
  
  constructor(config: {
    dimension?: number;
    curvature?: number;
    learningRate?: number;
    maxIterations?: number;
  } = {}) {
    this.dimension = config.dimension || 5;
    this.curvature = config.curvature || 1.0;
    this.learningRate = config.learningRate || 0.1;
    this.maxIterations = config.maxIterations || 500;
    this.embeddings = new Map();
  }
  
  // ============================================================================
  // Embedding Generation
  // ============================================================================
  
  /**
   * Embed horizon graph into hyperbolic space
   */
  embedHorizonGraph(
    nodes: Map<string, HorizonGraphNode>,
    edges: { from: string; to: string; weight: number }[],
    currentRegime: MarketRegime
  ): Map<string, HorizonHyperbolicPoint> {
    // Initialize embeddings
    this.initializeEmbeddings(nodes);
    
    // Optimize embeddings using gradient descent
    this.optimizeEmbeddings(edges);
    
    // Compute signal robustness and regime similarity
    this.computeHyperbolicProperties(currentRegime);
    
    return this.embeddings;
  }
  
  /**
   * Initialize embeddings in Poincaré disk
   */
  private initializeEmbeddings(nodes: Map<string, HorizonGraphNode>): void {
    this.embeddings.clear();
    
    for (const [nodeId, node] of nodes.entries()) {
      const coords = this.generateInitialCoords(node);
      const norm = this.euclideanNorm(coords);
      const angle = this.computeAngle(coords);
      
      const point: HorizonHyperbolicPoint = {
        coords,
        norm,
        angle,
        id: nodeId,
        type: node.type,
        horizon: (node as any).horizon,
        signalRobustness: 0,  // Will be computed
        regimeSimilarity: 0,  // Will be computed
      };
      
      this.embeddings.set(nodeId, point);
      
      // Store regime embedding as reference
      if (node.type === 'regime') {
        this.currentRegimeEmbedding = point;
      }
    }
  }
  
  /**
   * Generate initial coordinates based on node type and properties
   */
  private generateInitialCoords(node: HorizonGraphNode): number[] {
    const coords: number[] = [];
    
    // Initialize based on node type
    if (node.type === 'signal') {
      const signalNode = node as HorizonSignalNode;
      
      // Radial position based on signal strength
      const targetNorm = signalNode.strength * signalNode.confidence * 0.7;
      
      // Angular position based on direction and horizon
      const horizonAngle = this.getHorizonBaseAngle(signalNode.horizon);
      const directionOffset = signalNode.direction * 0.3;
      
      for (let i = 0; i < this.dimension; i++) {
        if (i === 0) {
          coords.push(targetNorm * Math.cos(horizonAngle + directionOffset));
        } else if (i === 1) {
          coords.push(targetNorm * Math.sin(horizonAngle + directionOffset));
        } else {
          coords.push((Math.random() - 0.5) * 0.1);
        }
      }
    } else if (node.type === 'regime') {
      // Regime near center (reference point)
      for (let i = 0; i < this.dimension; i++) {
        coords.push((Math.random() - 0.5) * 0.2);
      }
    } else if (node.type === 'horizon') {
      // Horizons positioned at specific angles
      const horizon = (node as any).horizon as TimeHorizon;
      const angle = this.getHorizonBaseAngle(horizon);
      const radius = 0.5;
      
      for (let i = 0; i < this.dimension; i++) {
        if (i === 0) {
          coords.push(radius * Math.cos(angle));
        } else if (i === 1) {
          coords.push(radius * Math.sin(angle));
        } else {
          coords.push((Math.random() - 0.5) * 0.05);
        }
      }
    } else {
      // Strategy nodes
      for (let i = 0; i < this.dimension; i++) {
        coords.push((Math.random() - 0.5) * 0.3);
      }
    }
    
    // Ensure inside Poincaré disk
    return this.projectToPoincareDisc(coords);
  }
  
  /**
   * Get base angle for each horizon (for spatial separation)
   */
  private getHorizonBaseAngle(horizon: TimeHorizon): number {
    const angles = {
      hourly: 0,           // 0°
      weekly: (2 * Math.PI) / 3,  // 120°
      monthly: (4 * Math.PI) / 3, // 240°
    };
    return angles[horizon];
  }
  
  // ============================================================================
  // Optimization
  // ============================================================================
  
  /**
   * Optimize embeddings using Riemannian gradient descent
   */
  private optimizeEmbeddings(
    edges: { from: string; to: string; weight: number }[]
  ): void {
    for (let iter = 0; iter < this.maxIterations; iter++) {
      // Calculate gradients
      const gradients = new Map<string, number[]>();
      
      for (const edge of edges) {
        const fromPoint = this.embeddings.get(edge.from);
        const toPoint = this.embeddings.get(edge.to);
        
        if (!fromPoint || !toPoint) continue;
        
        // Compute gradient of distance loss
        const distance = this.poincareDistance(fromPoint, toPoint);
        const targetDistance = 1.0 - edge.weight;  // Higher weight = closer
        const loss = (distance - targetDistance) ** 2;
        
        // Gradient: ∂loss/∂coords
        const gradient = this.computeDistanceGradient(fromPoint, toPoint, loss);
        
        // Accumulate gradients
        if (!gradients.has(edge.from)) {
          gradients.set(edge.from, Array(this.dimension).fill(0));
        }
        const fromGrad = gradients.get(edge.from)!;
        for (let i = 0; i < this.dimension; i++) {
          fromGrad[i] += gradient[i];
        }
      }
      
      // Update embeddings
      for (const [nodeId, gradient] of gradients.entries()) {
        const point = this.embeddings.get(nodeId);
        if (!point) continue;
        
        // Riemannian gradient descent update
        const newCoords = point.coords.map((c, i) => 
          c - this.learningRate * gradient[i]
        );
        
        // Project back to Poincaré disc
        const projectedCoords = this.projectToPoincareDisc(newCoords);
        
        point.coords = projectedCoords;
        point.norm = this.euclideanNorm(projectedCoords);
        point.angle = this.computeAngle(projectedCoords);
      }
      
      // Decay learning rate
      if (iter % 100 === 0) {
        this.learningRate *= 0.95;
      }
    }
  }
  
  /**
   * Compute gradient of distance loss
   */
  private computeDistanceGradient(
    p1: HorizonHyperbolicPoint,
    p2: HorizonHyperbolicPoint,
    loss: number
  ): number[] {
    const gradient: number[] = [];
    const epsilon = 1e-6;
    
    for (let i = 0; i < this.dimension; i++) {
      // Numerical gradient: (f(x + ε) - f(x)) / ε
      const perturbedCoords = [...p1.coords];
      perturbedCoords[i] += epsilon;
      
      const perturbedPoint = { ...p1, coords: perturbedCoords };
      const perturbedDistance = this.poincareDistance(perturbedPoint, p2);
      const perturbedLoss = perturbedDistance ** 2;
      
      gradient.push((perturbedLoss - loss) / epsilon);
    }
    
    return gradient;
  }
  
  // ============================================================================
  // Hyperbolic Properties
  // ============================================================================
  
  /**
   * Compute signal robustness and regime similarity for all embeddings
   */
  private computeHyperbolicProperties(currentRegime: MarketRegime): void {
    if (!this.currentRegimeEmbedding) return;
    
    for (const point of this.embeddings.values()) {
      // Signal robustness: radial distance from origin
      point.signalRobustness = this.getSignalRobustness(point);
      
      // Regime similarity: angular distance to current regime
      point.regimeSimilarity = this.getRegimeSimilarity(point, this.currentRegimeEmbedding);
    }
  }
  
  /**
   * Get signal robustness (0 = weak, 1 = strong)
   * 
   * Interpretation:
   * - Near origin (norm < 0.3): Weak/uncertain signals
   * - Mid-range (0.3 - 0.7): Moderate signals
   * - Near edge (norm > 0.7): Strong/confident signals
   */
  getSignalRobustness(point: HorizonHyperbolicPoint): number {
    // Radial distance in Poincaré disk
    // Apply sigmoid-like transformation to emphasize differences
    const rawRobustness = point.norm;
    
    // Sigmoidal transformation: maps [0, 1) to [0, 1]
    // Emphasizes middle range
    const robustness = 1 / (1 + Math.exp(-10 * (rawRobustness - 0.5)));
    
    return robustness;
  }
  
  /**
   * Get regime similarity (-1 = opposite, 0 = orthogonal, 1 = aligned)
   */
  getRegimeSimilarity(
    point: HorizonHyperbolicPoint,
    regimePoint: HorizonHyperbolicPoint
  ): number {
    // Angular distance in Poincaré disk
    const angularDiff = this.angularDistance(point, regimePoint);
    
    // Convert to similarity: 0° = 1 (same), 180° = -1 (opposite)
    const similarity = Math.cos(angularDiff);
    
    return similarity;
  }
  
  // ============================================================================
  // Distance Metrics
  // ============================================================================
  
  /**
   * Compute comprehensive distance metrics between two points
   */
  horizonDistance(
    p1: HorizonHyperbolicPoint,
    p2: HorizonHyperbolicPoint
  ): HorizonDistanceMetrics {
    // Basic distances
    const poincareDistance = this.poincareDistance(p1, p2);
    const euclideanDistance = this.euclideanDistance(p1, p2);
    const radialDistance = Math.abs(p1.norm - p2.norm);
    const angularDistance = this.angularDistance(p1, p2);
    
    // Horizon penalty (if different horizons)
    const horizonPenalty = (p1.horizon && p2.horizon && p1.horizon !== p2.horizon) ? 0.2 : 0.0;
    
    // Decay-adjusted distance (account for signal decay rates)
    const decayAdjustment = this.computeDecayAdjustment(p1, p2);
    const decayAdjustedDistance = poincareDistance * (1 + decayAdjustment);
    
    return {
      poincareDistance,
      euclideanDistance,
      radialDistance,
      angularDistance,
      horizonPenalty,
      decayAdjustedDistance,
    };
  }
  
  /**
   * Poincaré distance (hyperbolic distance in Poincaré disk)
   * 
   * Formula: d(u, v) = arcosh(1 + 2||u-v||² / ((1-||u||²)(1-||v||²)))
   */
  poincareDistance(p1: HorizonHyperbolicPoint, p2: HorizonHyperbolicPoint): number {
    const diff = p1.coords.map((c, i) => c - p2.coords[i]);
    const normDiffSq = diff.reduce((sum, d) => sum + d * d, 0);
    
    const norm1Sq = p1.norm ** 2;
    const norm2Sq = p2.norm ** 2;
    
    const numerator = 2 * normDiffSq;
    const denominator = (1 - norm1Sq) * (1 - norm2Sq);
    
    const ratio = 1 + numerator / (denominator + 1e-10);
    
    return Math.acosh(Math.max(1.0, ratio));
  }
  
  /**
   * Euclidean distance in embedding space
   */
  euclideanDistance(p1: HorizonHyperbolicPoint, p2: HorizonHyperbolicPoint): number {
    const diff = p1.coords.map((c, i) => c - p2.coords[i]);
    return Math.sqrt(diff.reduce((sum, d) => sum + d * d, 0));
  }
  
  /**
   * Angular distance between two points
   */
  angularDistance(p1: HorizonHyperbolicPoint, p2: HorizonHyperbolicPoint): number {
    // Use first 2 dimensions for angle (primary plane)
    const angle1 = Math.atan2(p1.coords[1], p1.coords[0]);
    const angle2 = Math.atan2(p2.coords[1], p2.coords[0]);
    
    let diff = Math.abs(angle1 - angle2);
    
    // Normalize to [0, π]
    if (diff > Math.PI) {
      diff = 2 * Math.PI - diff;
    }
    
    return diff;
  }
  
  /**
   * Compute decay adjustment based on signal horizons
   */
  private computeDecayAdjustment(
    p1: HorizonHyperbolicPoint,
    p2: HorizonHyperbolicPoint
  ): number {
    if (!p1.horizon || !p2.horizon) return 0;
    
    // Decay rates by horizon
    const decayRates = {
      hourly: 0.15,
      weekly: 0.08,
      monthly: 0.03,
    };
    
    const decay1 = decayRates[p1.horizon];
    const decay2 = decayRates[p2.horizon];
    
    // Higher decay = less stable = higher adjustment
    return (decay1 + decay2) / 2;
  }
  
  // ============================================================================
  // Utilities
  // ============================================================================
  
  /**
   * Project coordinates to Poincaré disc (ensure norm < 1)
   */
  private projectToPoincareDisc(coords: number[]): number[] {
    const norm = this.euclideanNorm(coords);
    
    if (norm >= 0.95) {
      // Scale to 0.9 of unit disc
      return coords.map(c => c / norm * 0.9);
    }
    
    return coords;
  }
  
  /**
   * Euclidean norm
   */
  private euclideanNorm(coords: number[]): number {
    return Math.sqrt(coords.reduce((sum, c) => sum + c * c, 0));
  }
  
  /**
   * Compute angle in 2D plane (using first 2 dimensions)
   */
  private computeAngle(coords: number[]): number {
    return Math.atan2(coords[1], coords[0]);
  }
  
  // ============================================================================
  // Getters
  // ============================================================================
  
  getEmbedding(nodeId: string): HorizonHyperbolicPoint | undefined {
    return this.embeddings.get(nodeId);
  }
  
  getAllEmbeddings(): Map<string, HorizonHyperbolicPoint> {
    return this.embeddings;
  }
  
  getSignalEmbeddings(): HorizonHyperbolicPoint[] {
    return Array.from(this.embeddings.values()).filter(p => p.type === 'signal');
  }
  
  getRegimeEmbedding(): HorizonHyperbolicPoint | null {
    return this.currentRegimeEmbedding;
  }
  
  /**
   * Get signals by robustness threshold
   */
  getRobustSignals(minRobustness: number): HorizonHyperbolicPoint[] {
    return this.getSignalEmbeddings().filter(p => p.signalRobustness >= minRobustness);
  }
  
  /**
   * Get signals by regime similarity threshold
   */
  getSimilarSignals(minSimilarity: number): HorizonHyperbolicPoint[] {
    return this.getSignalEmbeddings().filter(p => p.regimeSimilarity >= minSimilarity);
  }
  
  /**
   * Get closest signals to a reference point
   */
  getClosestSignals(referenceId: string, count: number): HorizonHyperbolicPoint[] {
    const reference = this.embeddings.get(referenceId);
    if (!reference) return [];
    
    const signals = this.getSignalEmbeddings();
    
    // Sort by Poincaré distance
    signals.sort((a, b) => {
      const distA = this.poincareDistance(a, reference);
      const distB = this.poincareDistance(b, reference);
      return distA - distB;
    });
    
    return signals.slice(0, count);
  }
}
