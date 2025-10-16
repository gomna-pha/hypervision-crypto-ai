import { EventEmitter } from 'events';

export interface HyperbolicNeighbor {
  embedding: number[];
  distance: number;
  metadata?: any;
  timestamp: string;
}

export interface HyperbolicEmbedderConfig {
  dimension: number;
  curvature: number;
  learning_rate: number;
  max_history_size: number;
  similarity_threshold: number;
}

/**
 * HyperbolicEmbedder - Maps financial features to hyperbolic space using Poincaré ball model
 * 
 * The Poincaré ball model represents hyperbolic geometry in a unit disk where:
 * - Points closer to the center have lower "energy" (stable market conditions)  
 * - Points near the boundary represent high volatility/arbitrage opportunities
 * - Hyperbolic distance captures hierarchical relationships between market states
 * 
 * Key advantages for arbitrage:
 * 1. Natural representation of market hierarchies (macro -> micro factors)
 * 2. Distance metric that emphasizes rare arbitrage opportunities
 * 3. Continuous space for gradient-based optimization
 * 4. Interpretable embeddings (center = stability, boundary = opportunity)
 */
export class HyperbolicEmbedder extends EventEmitter {
  private config: HyperbolicEmbedderConfig;
  private embeddingHistory: Array<{
    embedding: number[];
    features: Record<string, number>;
    timestamp: string;
    metadata?: any;
  }> = [];
  
  private featureWeights: Map<string, number> = new Map();
  private featureStats: Map<string, { mean: number; std: number; min: number; max: number }> = new Map();
  
  // Visible Parameters for Investors
  public readonly parameters = {
    poincare_ball_radius: 1.0,
    embedding_dimension: 128,
    curvature_constant: 1.0,
    max_norm_threshold: 0.95, // Keep embeddings away from boundary
    feature_normalization: 'z_score',
    distance_metric: 'poincare',
    neighbor_search_method: 'euclidean_approximation',
    history_retention_hours: 24
  };

  // Visible Constraints for Investors  
  public readonly constraints = {
    max_feature_count: 50,
    min_feature_count: 3,
    max_embedding_norm: 0.99, // Prevent boundary singularities
    min_embedding_norm: 0.01,
    max_distance_computation_ms: 100,
    max_neighbor_search_size: 10000
  };

  constructor(config?: Partial<HyperbolicEmbedderConfig>) {
    super();
    
    this.config = {
      dimension: config?.dimension || this.parameters.embedding_dimension,
      curvature: config?.curvature || this.parameters.curvature_constant,
      learning_rate: config?.learning_rate || 0.01,
      max_history_size: config?.max_history_size || 10000,
      similarity_threshold: config?.similarity_threshold || 0.85
    };
    
    this.initializeFeatureWeights();
    console.log('✅ HyperbolicEmbedder initialized with Poincaré ball model');
  }

  private initializeFeatureWeights(): void {
    // Initialize feature importance weights based on arbitrage relevance
    const defaultWeights: Record<string, number> = {
      // Microstructure features (highest weight - direct arbitrage signals)
      'micro_signal': 1.0,
      'spread_bps': 0.9,
      'orderbook_imbalance': 0.85,
      'volume_surge': 0.8,
      'price_momentum': 0.75,
      
      // Sentiment features (medium-high weight - market direction)
      'sentiment_signal': 0.7,
      'news_sentiment': 0.65,
      'fear_greed': 0.6,
      'mention_volume': 0.55,
      
      // Economic features (medium weight - macro context)
      'economic_signal': 0.6,
      'cpi_yoy': 0.5,
      'fed_rate': 0.55,
      'real_rate': 0.5,
      'liquidity_conditions': 0.45
    };
    
    for (const [feature, weight] of Object.entries(defaultWeights)) {
      this.featureWeights.set(feature, weight);
    }
  }

  /**
   * Embed financial features into hyperbolic space
   * Maps feature vector to Poincaré ball coordinates
   */
  async embed(features: Record<string, number>): Promise<number[]> {
    try {
      // Step 1: Validate and normalize features
      const normalizedFeatures = this.normalizeFeatures(features);
      
      // Step 2: Apply feature weights 
      const weightedFeatures = this.applyFeatureWeights(normalizedFeatures);
      
      // Step 3: Map to hyperbolic space using Poincaré ball model
      const embedding = this.mapToPoincareBall(weightedFeatures);
      
      // Step 4: Store in history for neighbor searches
      this.addToHistory(embedding, features);
      
      // Step 5: Update feature statistics
      this.updateFeatureStats(features);
      
      return embedding;
      
    } catch (error) {
      console.error('❌ Hyperbolic embedding error:', error);
      // Return zero embedding as fallback
      return new Array(this.config.dimension).fill(0);
    }
  }

  /**
   * Find k-nearest neighbors in hyperbolic space
   * Uses Poincaré distance metric for similarity
   */
  async knnQuery(queryEmbedding: number[], k: number = 5): Promise<HyperbolicNeighbor[]> {
    if (this.embeddingHistory.length === 0) {
      return [];
    }
    
    try {
      const startTime = performance.now();
      
      // Calculate Poincaré distances to all historical embeddings
      const distances: Array<{ index: number; distance: number }> = [];
      
      for (let i = 0; i < this.embeddingHistory.length; i++) {
        const historicalEmbedding = this.embeddingHistory[i].embedding;
        const distance = this.poincareDistance(queryEmbedding, historicalEmbedding);
        distances.push({ index: i, distance });
      }
      
      // Sort by distance and take k nearest
      distances.sort((a, b) => a.distance - b.distance);
      const nearestIndices = distances.slice(0, k);
      
      const neighbors: HyperbolicNeighbor[] = nearestIndices.map(({ index, distance }) => {
        const historical = this.embeddingHistory[index];
        return {
          embedding: historical.embedding,
          distance,
          metadata: historical.metadata,
          timestamp: historical.timestamp
        };
      });
      
      const elapsed = performance.now() - startTime;
      if (elapsed > this.constraints.max_distance_computation_ms) {
        console.warn(`⚠️ Slow neighbor search: ${elapsed.toFixed(1)}ms`);
      }
      
      return neighbors;
      
    } catch (error) {
      console.error('❌ KNN query error:', error);
      return [];
    }
  }

  private normalizeFeatures(features: Record<string, number>): Record<string, number> {
    const normalized: Record<string, number> = {};
    
    for (const [key, value] of Object.entries(features)) {
      if (typeof value !== 'number' || !isFinite(value)) {
        normalized[key] = 0;
        continue;
      }
      
      // Get or create feature statistics
      let stats = this.featureStats.get(key);
      if (!stats) {
        stats = { mean: value, std: 1, min: value, max: value };
        this.featureStats.set(key, stats);
      }
      
      // Z-score normalization with clipping
      const zScore = stats.std > 0 ? (value - stats.mean) / stats.std : 0;
      normalized[key] = Math.max(-3, Math.min(3, zScore)); // Clip to [-3, 3]
    }
    
    return normalized;
  }

  private applyFeatureWeights(features: Record<string, number>): Record<string, number> {
    const weighted: Record<string, number> = {};
    
    for (const [key, value] of Object.entries(features)) {
      const weight = this.featureWeights.get(key) || 0.5; // Default weight
      weighted[key] = value * weight;
    }
    
    return weighted;
  }

  private mapToPoincareBall(features: Record<string, number>): number[] {
    const featureArray = Object.values(features);
    const embedding = new Array(this.config.dimension).fill(0);
    
    // Step 1: Create initial embedding using feature hashing and linear combination
    let featureIndex = 0;
    for (const [key, value] of Object.entries(features)) {
      const hash = this.hashFeatureName(key);
      
      // Distribute each feature across multiple embedding dimensions
      for (let i = 0; i < 4; i++) { // Each feature affects 4 dimensions
        const dimIndex = (hash + i) % this.config.dimension;
        const phase = (hash * 0.618 + i * 0.382) % (2 * Math.PI); // Golden ratio phases
        
        // Use sine and cosine to create smooth feature mappings
        embedding[dimIndex] += value * Math.cos(phase) * 0.1;
        if (dimIndex + 1 < this.config.dimension) {
          embedding[dimIndex + 1] += value * Math.sin(phase) * 0.1;
        }
      }
      
      featureIndex++;
    }
    
    // Step 2: Apply non-linear transformation to emphasize arbitrage opportunities
    for (let i = 0; i < embedding.length; i++) {
      // Hyperbolic tangent for bounded output
      embedding[i] = Math.tanh(embedding[i] * 2) * 0.8; // Scale to prevent boundary issues
    }
    
    // Step 3: Normalize to Poincaré ball (ensure ||x|| < 1)
    const norm = Math.sqrt(embedding.reduce((sum, x) => sum + x * x, 0));
    if (norm >= this.parameters.max_norm_threshold) {
      const scale = this.parameters.max_norm_threshold / norm;
      for (let i = 0; i < embedding.length; i++) {
        embedding[i] *= scale;
      }
    }
    
    // Step 4: Add small random noise to prevent identical embeddings
    for (let i = 0; i < embedding.length; i++) {
      embedding[i] += (Math.random() - 0.5) * 0.001;
    }
    
    return embedding;
  }

  private hashFeatureName(featureName: string): number {
    let hash = 0;
    for (let i = 0; i < featureName.length; i++) {
      hash = ((hash << 5) - hash + featureName.charCodeAt(i)) | 0;
    }
    return Math.abs(hash);
  }

  /**
   * Calculate Poincaré distance in hyperbolic space
   * Formula: d(x,y) = acosh(1 + 2 * ||x-y||^2 / ((1-||x||^2)(1-||y||^2)))
   */
  private poincareDistance(x: number[], y: number[]): number {
    if (x.length !== y.length) {
      console.warn('⚠️ Embedding dimension mismatch in distance calculation');
      return Infinity;
    }
    
    try {
      // Calculate squared norms
      const normXSq = x.reduce((sum, xi) => sum + xi * xi, 0);
      const normYSq = y.reduce((sum, yi) => sum + yi * yi, 0);
      
      // Calculate squared difference norm
      const diffNormSq = x.reduce((sum, xi, i) => {
        const diff = xi - y[i];
        return sum + diff * diff;
      }, 0);
      
      // Prevent division by zero and numerical instability
      const eps = 1e-8;
      const denomX = Math.max(eps, 1 - normXSq);
      const denomY = Math.max(eps, 1 - normYSq);
      
      // Poincaré distance formula
      const ratio = (2 * diffNormSq) / (denomX * denomY);
      const distance = Math.acosh(1 + ratio);
      
      // Return finite distance or large value for numerical issues
      return isFinite(distance) ? distance : 10.0;
      
    } catch (error) {
      console.warn('⚠️ Poincaré distance calculation error:', error);
      return 10.0; // Large distance for errors
    }
  }

  private addToHistory(embedding: number[], features: Record<string, number>): void {
    const entry = {
      embedding: [...embedding], // Copy array
      features: { ...features }, // Copy object
      timestamp: new Date().toISOString(),
      metadata: {
        norm: Math.sqrt(embedding.reduce((sum, x) => sum + x * x, 0)),
        feature_count: Object.keys(features).length
      }
    };
    
    this.embeddingHistory.push(entry);
    
    // Limit history size
    if (this.embeddingHistory.length > this.config.max_history_size) {
      // Remove oldest 10% when limit exceeded
      const removeCount = Math.floor(this.config.max_history_size * 0.1);
      this.embeddingHistory.splice(0, removeCount);
    }
    
    // Cleanup old entries (older than retention period)
    const retentionMs = this.parameters.history_retention_hours * 60 * 60 * 1000;
    const cutoffTime = Date.now() - retentionMs;
    
    this.embeddingHistory = this.embeddingHistory.filter(entry => {
      return new Date(entry.timestamp).getTime() > cutoffTime;
    });
  }

  private updateFeatureStats(features: Record<string, number>): void {
    for (const [key, value] of Object.entries(features)) {
      if (typeof value !== 'number' || !isFinite(value)) continue;
      
      let stats = this.featureStats.get(key);
      if (!stats) {
        stats = { mean: value, std: 0, min: value, max: value };
        this.featureStats.set(key, stats);
      } else {
        // Update running statistics
        const alpha = 0.01; // Exponential moving average factor
        stats.mean = (1 - alpha) * stats.mean + alpha * value;
        stats.min = Math.min(stats.min, value);
        stats.max = Math.max(stats.max, value);
        
        // Simple std approximation
        const variance = alpha * Math.pow(value - stats.mean, 2) + (1 - alpha) * Math.pow(stats.std, 2);
        stats.std = Math.sqrt(variance);
      }
    }
  }

  // Public methods for transparency and debugging
  getVisibleParameters(): any {
    return {
      parameters: this.parameters,
      constraints: this.constraints,
      config: this.config,
      history_size: this.embeddingHistory.length,
      feature_weights: Object.fromEntries(this.featureWeights),
      feature_statistics: Object.fromEntries(this.featureStats)
    };
  }

  getEmbeddingHistory(): Array<{ timestamp: string; norm: number; feature_count: number }> {
    return this.embeddingHistory.map(entry => ({
      timestamp: entry.timestamp,
      norm: entry.metadata?.norm || 0,
      feature_count: entry.metadata?.feature_count || 0
    }));
  }

  /**
   * Compute embedding quality metrics for monitoring
   */
  getQualityMetrics(): any {
    if (this.embeddingHistory.length < 2) {
      return { status: 'insufficient_data' };
    }
    
    const recentEmbeddings = this.embeddingHistory.slice(-100); // Last 100 embeddings
    
    // Calculate average pairwise distances
    let totalDistance = 0;
    let pairCount = 0;
    
    for (let i = 0; i < recentEmbeddings.length - 1; i++) {
      for (let j = i + 1; j < Math.min(i + 10, recentEmbeddings.length); j++) {
        const distance = this.poincareDistance(
          recentEmbeddings[i].embedding,
          recentEmbeddings[j].embedding
        );
        totalDistance += distance;
        pairCount++;
      }
    }
    
    const avgDistance = pairCount > 0 ? totalDistance / pairCount : 0;
    
    // Calculate embedding norms distribution
    const norms = recentEmbeddings.map(e => 
      Math.sqrt(e.embedding.reduce((sum, x) => sum + x * x, 0))
    );
    const avgNorm = norms.reduce((sum, n) => sum + n, 0) / norms.length;
    const maxNorm = Math.max(...norms);
    
    return {
      status: 'active',
      avg_pairwise_distance: avgDistance,
      avg_embedding_norm: avgNorm,
      max_embedding_norm: maxNorm,
      boundary_risk: maxNorm > 0.9 ? 'high' : maxNorm > 0.7 ? 'medium' : 'low',
      embedding_diversity: avgDistance > 0.5 ? 'good' : 'low',
      history_coverage_hours: this.getHistoryCoverageHours()
    };
  }

  private getHistoryCoverageHours(): number {
    if (this.embeddingHistory.length === 0) return 0;
    
    const oldest = new Date(this.embeddingHistory[0].timestamp);
    const newest = new Date(this.embeddingHistory[this.embeddingHistory.length - 1].timestamp);
    
    return (newest.getTime() - oldest.getTime()) / (1000 * 60 * 60);
  }

  /**
   * Clear history and reset statistics (useful for testing)
   */
  reset(): void {
    this.embeddingHistory = [];
    this.featureStats.clear();
    this.initializeFeatureWeights();
    console.log('✅ HyperbolicEmbedder reset completed');
  }
}