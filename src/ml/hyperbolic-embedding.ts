/**
 * Hyperbolic Embedding Layer (Poincaré Ball)
 * 
 * Embeds hierarchical signal-regime graph into hyperbolic space:
 * - Radial distance → Signal robustness
 * - Angular distance → Regime similarity
 * - Curvature preserves tree-like structure
 * 
 * Output: Low-dim hyperbolic coordinates & distances
 */

export interface HyperbolicPoint {
  coords: number[];      // Coordinates in Poincaré ball (n-dimensional)
  norm: number;          // Euclidean norm (must be < 1 for valid point)
  id: string;            // Node ID (signal, regime, strategy)
  type: 'signal' | 'regime' | 'strategy';
}

export interface HierarchicalNode {
  id: string;
  type: 'signal' | 'regime' | 'strategy';
  children: string[];    // Child node IDs
  parent: string | null; // Parent node ID
  features: number[];    // Feature vector for embedding
}

export interface HierarchicalGraph {
  nodes: Map<string, HierarchicalNode>;
  edges: Map<string, Set<string>>; // Adjacency list
}

export class HyperbolicEmbedding {
  private dimension: number;
  private curvature: number;
  private learningRate: number;
  private maxIterations: number;
  private embeddings: Map<string, HyperbolicPoint>;

  constructor(config: {
    dimension?: number;
    curvature?: number;
    learningRate?: number;
    maxIterations?: number;
  } = {}) {
    this.dimension = config.dimension || 5; // Low-dimensional embedding
    this.curvature = config.curvature || 1.0; // Curvature of hyperbolic space
    this.learningRate = config.learningRate || 0.1;
    this.maxIterations = config.maxIterations || 1000;
    this.embeddings = new Map();
  }

  /**
   * Initialize embeddings randomly in Poincaré ball
   */
  private initializeEmbeddings(nodes: HierarchicalNode[]): void {
    this.embeddings.clear();

    for (const node of nodes) {
      // Random initialization with small norm (near origin)
      const coords = Array.from(
        { length: this.dimension },
        () => (Math.random() - 0.5) * 0.1 // Small initial values
      );

      // Ensure point is inside Poincaré ball (norm < 1)
      const norm = this.euclideanNorm(coords);
      const scaledCoords = coords.map(c => c / (norm + 1e-5) * 0.5); // Scale to 0.5 * radius

      this.embeddings.set(node.id, {
        coords: scaledCoords,
        norm: this.euclideanNorm(scaledCoords),
        id: node.id,
        type: node.type,
      });
    }
  }

  /**
   * Euclidean norm of a vector
   */
  private euclideanNorm(coords: number[]): number {
    return Math.sqrt(coords.reduce((sum, c) => sum + c * c, 0));
  }

  /**
   * Poincaré distance between two points in hyperbolic space
   * 
   * Formula: d(u, v) = arcosh(1 + 2 * ||u - v||² / ((1 - ||u||²)(1 - ||v||²)))
   */
  poincareDistance(p1: HyperbolicPoint, p2: HyperbolicPoint): number {
    const diff = p1.coords.map((c, i) => c - p2.coords[i]);
    const normDiffSq = diff.reduce((sum, d) => sum + d * d, 0);

    const norm1Sq = p1.coords.reduce((sum, c) => sum + c * c, 0);
    const norm2Sq = p2.coords.reduce((sum, c) => sum + c * c, 0);

    const numerator = 2 * normDiffSq;
    const denominator = (1 - norm1Sq) * (1 - norm2Sq);

    const ratio = 1 + numerator / (denominator + 1e-10);
    const distance = Math.acosh(Math.max(1.0, ratio)); // Ensure arg >= 1 for acosh

    return distance * this.curvature;
  }

  /**
   * Radial distance from origin (measures signal robustness)
   */
  radialDistance(p: HyperbolicPoint): number {
    const normSq = p.coords.reduce((sum, c) => sum + c * c, 0);
    return Math.atanh(Math.sqrt(normSq)) * this.curvature;
  }

  /**
   * Angular distance between two points (measures regime similarity)
   */
  angularDistance(p1: HyperbolicPoint, p2: HyperbolicPoint): number {
    // Cosine similarity in Euclidean space
    const dotProduct = p1.coords.reduce((sum, c, i) => sum + c * p2.coords[i], 0);
    const norm1 = Math.sqrt(p1.coords.reduce((sum, c) => sum + c * c, 0));
    const norm2 = Math.sqrt(p2.coords.reduce((sum, c) => sum + c * c, 0));

    const cosSim = dotProduct / (norm1 * norm2 + 1e-10);
    return Math.acos(Math.max(-1, Math.min(1, cosSim))); // Clamp to [-1, 1]
  }

  /**
   * Exponential map: maps tangent vector at origin to point on manifold
   */
  private exponentialMap(tangent: number[]): number[] {
    const norm = this.euclideanNorm(tangent);
    if (norm < 1e-10) {
      return tangent;
    }

    const factor = Math.tanh(this.curvature * norm) / norm;
    return tangent.map(t => t * factor);
  }

  /**
   * Logarithmic map: maps point on manifold to tangent vector at origin
   */
  private logarithmicMap(point: number[]): number[] {
    const norm = this.euclideanNorm(point);
    if (norm < 1e-10) {
      return point;
    }

    const factor = Math.atanh(norm) / (norm * this.curvature);
    return point.map(p => p * factor);
  }

  /**
   * Riemannian gradient in Poincaré ball
   */
  private riemannianGradient(point: number[], euclideanGrad: number[]): number[] {
    const normSq = point.reduce((sum, p) => sum + p * p, 0);
    const conformalFactor = Math.pow((1 - normSq) / 2, 2);
    return euclideanGrad.map(g => g * conformalFactor);
  }

  /**
   * Project point back into Poincaré ball (ensure norm < 1)
   */
  private projectToBall(coords: number[]): number[] {
    const norm = this.euclideanNorm(coords);
    if (norm >= 1) {
      // Scale down to 0.99 to stay inside ball
      const scale = 0.99 / norm;
      return coords.map(c => c * scale);
    }
    return coords;
  }

  /**
   * Loss function: sum of squared distance errors
   * 
   * L = Σ (d_hyperbolic(u, v) - d_target(u, v))²
   */
  private computeLoss(
    graph: HierarchicalGraph,
    targetDistances: Map<string, Map<string, number>>
  ): number {
    let totalLoss = 0;
    let count = 0;

    for (const [nodeId, neighbors] of graph.edges.entries()) {
      const p1 = this.embeddings.get(nodeId);
      if (!p1) continue;

      for (const neighborId of neighbors) {
        const p2 = this.embeddings.get(neighborId);
        if (!p2) continue;

        const actualDistance = this.poincareDistance(p1, p2);
        const targetDistance = targetDistances.get(nodeId)?.get(neighborId) || 1.0;

        const error = actualDistance - targetDistance;
        totalLoss += error * error;
        count++;
      }
    }

    return count > 0 ? totalLoss / count : 0;
  }

  /**
   * Compute gradient for a single node
   */
  private computeGradient(
    nodeId: string,
    graph: HierarchicalGraph,
    targetDistances: Map<string, Map<string, number>>
  ): number[] {
    const p1 = this.embeddings.get(nodeId);
    if (!p1) return Array(this.dimension).fill(0);

    const gradient = Array(this.dimension).fill(0);
    const neighbors = graph.edges.get(nodeId) || new Set();

    for (const neighborId of neighbors) {
      const p2 = this.embeddings.get(neighborId);
      if (!p2) continue;

      const actualDistance = this.poincareDistance(p1, p2);
      const targetDistance = targetDistances.get(nodeId)?.get(neighborId) || 1.0;

      const error = actualDistance - targetDistance;

      // Gradient of Poincaré distance (approximation)
      const diff = p1.coords.map((c, i) => c - p2.coords[i]);
      const norm1Sq = p1.coords.reduce((sum, c) => sum + c * c, 0);
      const norm2Sq = p2.coords.reduce((sum, c) => sum + c * c, 0);

      const factor = (4 * error) / ((1 - norm1Sq) * (1 - norm2Sq) + 1e-10);

      for (let i = 0; i < this.dimension; i++) {
        gradient[i] += factor * diff[i];
      }
    }

    return gradient;
  }

  /**
   * Embed hierarchical graph into Poincaré ball using gradient descent
   */
  embed(
    graph: HierarchicalGraph,
    targetDistances?: Map<string, Map<string, number>>
  ): Map<string, HyperbolicPoint> {
    const nodes = Array.from(graph.nodes.values());
    this.initializeEmbeddings(nodes);

    // If no target distances provided, use hierarchy-based distances
    if (!targetDistances) {
      targetDistances = this.computeHierarchyDistances(graph);
    }

    for (let iter = 0; iter < this.maxIterations; iter++) {
      const loss = this.computeLoss(graph, targetDistances);

      // Update embeddings for each node
      for (const [nodeId, point] of this.embeddings.entries()) {
        const euclideanGrad = this.computeGradient(nodeId, graph, targetDistances);
        const riemannianGrad = this.riemannianGradient(point.coords, euclideanGrad);

        // Gradient descent update
        const updatedCoords = point.coords.map(
          (c, i) => c - this.learningRate * riemannianGrad[i]
        );

        // Project back to ball
        const projectedCoords = this.projectToBall(updatedCoords);

        point.coords = projectedCoords;
        point.norm = this.euclideanNorm(projectedCoords);
      }

      if (iter % 100 === 0) {
        console.log(`Iteration ${iter}/${this.maxIterations}: Loss = ${loss.toFixed(6)}`);
      }

      // Early stopping if converged
      if (loss < 1e-6) {
        console.log(`Converged at iteration ${iter}`);
        break;
      }
    }

    return this.embeddings;
  }

  /**
   * Compute target distances based on graph hierarchy
   * Parent-child: distance = 0.5
   * Siblings: distance = 1.0
   * Distant relatives: distance = 1.5+
   */
  private computeHierarchyDistances(
    graph: HierarchicalGraph
  ): Map<string, Map<string, number>> {
    const distances = new Map<string, Map<string, number>>();

    for (const [nodeId, node] of graph.nodes.entries()) {
      distances.set(nodeId, new Map());

      // Parent-child relationships
      if (node.parent) {
        distances.get(nodeId)!.set(node.parent, 0.5);
      }

      for (const childId of node.children) {
        distances.get(nodeId)!.set(childId, 0.5);
      }

      // Sibling relationships
      if (node.parent) {
        const parent = graph.nodes.get(node.parent);
        if (parent) {
          for (const siblingId of parent.children) {
            if (siblingId !== nodeId) {
              distances.get(nodeId)!.set(siblingId, 1.0);
            }
          }
        }
      }

      // Default distance for all other nodes
      for (const [otherId, otherNode] of graph.nodes.entries()) {
        if (otherId !== nodeId && !distances.get(nodeId)!.has(otherId)) {
          distances.get(nodeId)!.set(otherId, 2.0);
        }
      }
    }

    return distances;
  }

  /**
   * Get embedding for a specific node
   */
  getEmbedding(nodeId: string): HyperbolicPoint | undefined {
    return this.embeddings.get(nodeId);
  }

  /**
   * Get all embeddings
   */
  getAllEmbeddings(): Map<string, HyperbolicPoint> {
    return this.embeddings;
  }

  /**
   * Find nearest neighbors in hyperbolic space
   */
  findNearestNeighbors(
    queryPoint: HyperbolicPoint,
    k: number = 5
  ): Array<{ point: HyperbolicPoint; distance: number }> {
    const distances: Array<{ point: HyperbolicPoint; distance: number }> = [];

    for (const [nodeId, point] of this.embeddings.entries()) {
      if (point.id !== queryPoint.id) {
        const distance = this.poincareDistance(queryPoint, point);
        distances.push({ point, distance });
      }
    }

    // Sort by distance (ascending)
    distances.sort((a, b) => a.distance - b.distance);

    return distances.slice(0, k);
  }

  /**
   * Get regime centroid (average position of regime nodes)
   */
  getRegimeCentroid(regimeType: string): HyperbolicPoint | null {
    const regimePoints = Array.from(this.embeddings.values()).filter(
      p => p.type === 'regime' && p.id.includes(regimeType)
    );

    if (regimePoints.length === 0) return null;

    // Average coordinates (approximation, not exact centroid in hyperbolic space)
    const avgCoords = Array(this.dimension).fill(0);
    for (const point of regimePoints) {
      for (let i = 0; i < this.dimension; i++) {
        avgCoords[i] += point.coords[i];
      }
    }

    for (let i = 0; i < this.dimension; i++) {
      avgCoords[i] /= regimePoints.length;
    }

    const projectedCoords = this.projectToBall(avgCoords);

    return {
      coords: projectedCoords,
      norm: this.euclideanNorm(projectedCoords),
      id: `centroid_${regimeType}`,
      type: 'regime',
    };
  }
}

/**
 * Example usage:
 * 
 * const graph: HierarchicalGraph = {
 *   nodes: new Map([
 *     ['signal_economic', { id: 'signal_economic', type: 'signal', children: [], parent: 'regime_stress', features: [] }],
 *     ['signal_sentiment', { id: 'signal_sentiment', type: 'signal', children: [], parent: 'regime_risk_on', features: [] }],
 *     ['regime_stress', { id: 'regime_stress', type: 'regime', children: ['signal_economic'], parent: null, features: [] }],
 *     ['regime_risk_on', { id: 'regime_risk_on', type: 'regime', children: ['signal_sentiment'], parent: null, features: [] }],
 *   ]),
 *   edges: new Map([
 *     ['signal_economic', new Set(['regime_stress'])],
 *     ['signal_sentiment', new Set(['regime_risk_on'])],
 *   ])
 * };
 * 
 * const embedding = new HyperbolicEmbedding({ dimension: 5, maxIterations: 1000 });
 * const embeddings = embedding.embed(graph);
 * 
 * console.log('Embeddings:', embeddings);
 */
