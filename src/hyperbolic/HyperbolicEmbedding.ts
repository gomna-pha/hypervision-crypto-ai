import * as tf from '@tensorflow/tfjs-node';
import { HyperbolicConfig, HyperbolicNeighbor } from '../types';
import Logger from '../utils/logger';
import config from '../utils/ConfigLoader';

/**
 * Hyperbolic Embedding Layer using Poincaré Ball Model
 * Maps agent features to hyperbolic space for better hierarchical representation
 */
export class HyperbolicEmbedding {
  private logger: Logger;
  private config: HyperbolicConfig;
  private embeddings: Map<string, tf.Tensor>;
  private featureEncoder: tf.Sequential | null = null;
  private initialized: boolean = false;

  constructor() {
    this.logger = Logger.getInstance('HyperbolicEmbedding');
    this.embeddings = new Map();
    
    this.config = config.get<HyperbolicConfig>('hyperbolic') || {
      model: 'poincare',
      curvature: 1.0,
      embedding_dim: 128,
      neighbor_k: 10,
      update_interval_sec: 300,
    };
  }

  async initialize(): Promise<void> {
    if (this.initialized) return;

    try {
      // Build feature encoder neural network
      this.featureEncoder = tf.sequential({
        layers: [
          tf.layers.dense({
            inputShape: [256], // Max input features
            units: 256,
            activation: 'relu',
            kernelInitializer: 'glorotUniform',
          }),
          tf.layers.dropout({ rate: 0.2 }),
          tf.layers.dense({
            units: 128,
            activation: 'relu',
          }),
          tf.layers.dense({
            units: this.config.embedding_dim,
            activation: 'tanh', // Output in [-1, 1]
          }),
        ],
      });

      this.featureEncoder.compile({
        optimizer: tf.train.adam(0.001),
        loss: 'meanSquaredError',
      });

      this.initialized = true;
      this.logger.info('Hyperbolic embedding layer initialized', {
        model: this.config.model,
        dim: this.config.embedding_dim,
        curvature: this.config.curvature,
      });
    } catch (error) {
      this.logger.error('Failed to initialize hyperbolic embedding', error);
      throw error;
    }
  }

  /**
   * Exponential map at origin (Poincaré ball model)
   * Maps Euclidean vector to hyperbolic space
   */
  private expMap0(u: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const c = this.config.curvature;
      const sqrtC = Math.sqrt(c);
      
      // ||u|| (Euclidean norm)
      const norm = tf.norm(u, 'euclidean', -1, true);
      
      // Avoid division by zero
      const normSafe = tf.maximum(norm, tf.scalar(1e-10));
      
      // tanh(sqrt(c) * ||u||) * u / (sqrt(c) * ||u||)
      const coefficient = tf.tanh(tf.mul(sqrtC, normSafe))
        .div(tf.mul(sqrtC, normSafe));
      
      return tf.mul(coefficient, u);
    });
  }

  /**
   * Poincaré distance between two points
   */
  private poincareDistance(x: tf.Tensor, y: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const c = this.config.curvature;
      
      // ||x - y||^2
      const diff = tf.sub(x, y);
      const diffNormSq = tf.sum(tf.square(diff), -1, true);
      
      // ||x||^2
      const xNormSq = tf.sum(tf.square(x), -1, true);
      
      // ||y||^2
      const yNormSq = tf.sum(tf.square(y), -1, true);
      
      // Möbius distance formula
      const numerator = tf.mul(2, diffNormSq);
      const denomX = tf.sub(1, tf.mul(c, xNormSq));
      const denomY = tf.sub(1, tf.mul(c, yNormSq));
      const denominator = tf.mul(denomX, denomY);
      
      const fraction = tf.div(numerator, tf.maximum(denominator, tf.scalar(1e-10)));
      
      // arccosh(1 + fraction) / sqrt(c)
      const distance = tf.div(
        tf.log(tf.add(1, tf.add(fraction, tf.sqrt(tf.mul(fraction, tf.add(fraction, 2)))))),
        Math.sqrt(c)
      );
      
      return distance;
    });
  }

  /**
   * Embed agent features into hyperbolic space
   */
  async embed(agentId: string, features: Record<string, number>): Promise<tf.Tensor> {
    if (!this.initialized) {
      await this.initialize();
    }

    return tf.tidy(() => {
      // Convert features to tensor
      const featureVector = this.featuresToVector(features);
      
      // Encode features using neural network
      const euclideanEmbedding = this.featureEncoder!.predict(featureVector) as tf.Tensor;
      
      // Map to hyperbolic space using exponential map
      const hyperbolicPoint = this.expMap0(euclideanEmbedding);
      
      // Store embedding
      this.storeEmbedding(agentId, hyperbolicPoint);
      
      return hyperbolicPoint;
    });
  }

  /**
   * Convert feature dictionary to fixed-size vector
   */
  private featuresToVector(features: Record<string, number>): tf.Tensor {
    const vectorSize = 256;
    const vector = new Float32Array(vectorSize);
    
    // Simple feature hashing for consistent mapping
    let index = 0;
    for (const [key, value] of Object.entries(features)) {
      if (typeof value === 'number') {
        // Hash key to index
        const hash = this.hashString(key);
        const idx = Math.abs(hash) % vectorSize;
        vector[idx] = value;
        index++;
      }
    }
    
    // Normalize
    const tensor = tf.tensor2d([vector], [1, vectorSize]);
    return tf.div(tensor, tf.maximum(tf.norm(tensor), tf.scalar(1)));
  }

  /**
   * Simple string hash function
   */
  private hashString(str: string): number {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return hash;
  }

  /**
   * Store embedding with automatic cleanup of old embeddings
   */
  private storeEmbedding(agentId: string, embedding: tf.Tensor): void {
    // Clean up old embedding if exists
    const oldEmbedding = this.embeddings.get(agentId);
    if (oldEmbedding) {
      oldEmbedding.dispose();
    }
    
    // Store new embedding (clone to prevent disposal)
    this.embeddings.set(agentId, embedding.clone());
    
    // Cleanup if too many embeddings (keep last 1000)
    if (this.embeddings.size > 1000) {
      const toDelete = Array.from(this.embeddings.keys()).slice(0, 100);
      for (const key of toDelete) {
        const tensor = this.embeddings.get(key);
        if (tensor) tensor.dispose();
        this.embeddings.delete(key);
      }
    }
  }

  /**
   * Find k-nearest neighbors in hyperbolic space
   */
  async findNeighbors(
    queryPoint: tf.Tensor, 
    k: number = 10, 
    excludeIds?: Set<string>
  ): Promise<HyperbolicNeighbor[]> {
    const neighbors: HyperbolicNeighbor[] = [];
    
    return tf.tidy(() => {
      for (const [agentId, embedding] of this.embeddings) {
        if (excludeIds?.has(agentId)) continue;
        
        // Calculate hyperbolic distance
        const distance = this.poincareDistance(queryPoint, embedding);
        const distanceValue = distance.arraySync() as number;
        
        neighbors.push({
          agent_id: agentId,
          distance: distanceValue,
          timestamp: new Date().toISOString(),
          features: {}, // Would be populated from storage in production
        });
      }
      
      // Sort by distance and return top k
      neighbors.sort((a, b) => a.distance - b.distance);
      return neighbors.slice(0, k);
    });
  }

  /**
   * Compute centroid in hyperbolic space (Fréchet mean)
   */
  async computeCentroid(points: tf.Tensor[]): Promise<tf.Tensor> {
    if (points.length === 0) {
      return tf.zeros([this.config.embedding_dim]);
    }
    
    return tf.tidy(() => {
      // Simple approximation: Project to tangent space, average, project back
      // For exact Fréchet mean, would need iterative optimization
      
      // Average in Euclidean space (approximation)
      const stacked = tf.stack(points);
      const mean = tf.mean(stacked, 0);
      
      // Project back to Poincaré ball
      return this.projectToPoincareBall(mean);
    });
  }

  /**
   * Project point to Poincaré ball (ensure ||x|| < 1/sqrt(c))
   */
  private projectToPoincareBall(x: tf.Tensor): tf.Tensor {
    return tf.tidy(() => {
      const c = this.config.curvature;
      const maxNorm = 1 / Math.sqrt(c) - 1e-5; // Small epsilon for numerical stability
      
      const norm = tf.norm(x, 'euclidean', -1, true);
      const scale = tf.minimum(1, tf.div(maxNorm, tf.maximum(norm, tf.scalar(1e-10))));
      
      return tf.mul(x, scale);
    });
  }

  /**
   * Update embeddings based on new agent data
   */
  async updateEmbeddings(agentOutputs: Map<string, any>): Promise<void> {
    for (const [agentId, output] of agentOutputs) {
      if (output.features) {
        await this.embed(agentId, output.features);
      }
    }
  }

  /**
   * Get embedding for a specific agent
   */
  getEmbedding(agentId: string): tf.Tensor | null {
    return this.embeddings.get(agentId) || null;
  }

  /**
   * Calculate embedding quality metrics
   */
  async calculateMetrics(): Promise<{
    avgDistance: number;
    clustering: number;
    coverage: number;
  }> {
    const allDistances: number[] = [];
    const embedArray = Array.from(this.embeddings.values());
    
    return tf.tidy(() => {
      // Calculate pairwise distances
      for (let i = 0; i < embedArray.length; i++) {
        for (let j = i + 1; j < embedArray.length; j++) {
          const dist = this.poincareDistance(embedArray[i], embedArray[j]);
          allDistances.push(dist.arraySync() as number);
        }
      }
      
      const avgDistance = allDistances.length > 0 
        ? allDistances.reduce((a, b) => a + b, 0) / allDistances.length 
        : 0;
      
      // Simple clustering coefficient (ratio of close pairs)
      const threshold = avgDistance * 0.5;
      const closePairs = allDistances.filter(d => d < threshold).length;
      const clustering = allDistances.length > 0 
        ? closePairs / allDistances.length 
        : 0;
      
      // Coverage (spread in space)
      const maxDist = Math.max(...allDistances, 0);
      const minDist = Math.min(...allDistances, 0);
      const coverage = maxDist > 0 ? (maxDist - minDist) / maxDist : 0;
      
      return {
        avgDistance,
        clustering,
        coverage,
      };
    });
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    // Dispose all tensors
    for (const tensor of this.embeddings.values()) {
      tensor.dispose();
    }
    this.embeddings.clear();
    
    // Dispose model
    if (this.featureEncoder) {
      this.featureEncoder.dispose();
    }
    
    this.logger.info('Hyperbolic embedding cleanup completed');
  }
}

export default HyperbolicEmbedding;