/**
 * Genetic Algorithm - Signal Selection Core
 * 
 * Natural selection of arbitrage signals with:
 * - Signal survival/extinction
 * - Weight evolution
 * - Correlation & redundancy penalty
 * - Turnover & drawdown constraints
 * 
 * Low-frequency re-optimization (daily/weekly)
 */

export interface SignalGenome {
  id: string;
  activeSignals: number[]; // Binary mask [0, 1, 1, 0, 1, ...] for 8 agents
  weights: number[];       // Weight vector [0.3, 0.0, 0.2, 0.0, 0.25, ...]
  fitness: number;         // Fitness score (higher is better)
  generation: number;      // Generation number
  age: number;             // Survival count (generations)
}

export interface GAConfig {
  populationSize: number;
  mutationRate: number;
  crossoverRate: number;
  eliteRatio: number;
  maxGenerations: number;
  fitnessWeights: {
    sharpe: number;
    correlation: number;
    turnover: number;
    drawdown: number;
  };
}

export interface BacktestResult {
  returns: number[];
  sharpe: number;
  maxDrawdown: number;
  turnover: number;
  totalTrades: number;
}

export class GeneticAlgorithmSignalSelector {
  private config: GAConfig;
  private population: SignalGenome[];
  private generation: number;
  private bestGenome: SignalGenome | null;

  constructor(config: Partial<GAConfig> = {}) {
    this.config = {
      populationSize: config.populationSize || 100,
      mutationRate: config.mutationRate || 0.05,
      crossoverRate: config.crossoverRate || 0.8,
      eliteRatio: config.eliteRatio || 0.1,
      maxGenerations: config.maxGenerations || 50,
      fitnessWeights: config.fitnessWeights || {
        sharpe: 0.5,
        correlation: 0.2,
        turnover: 0.15,
        drawdown: 0.15,
      },
    };
    this.population = [];
    this.generation = 0;
    this.bestGenome = null;
  }

  /**
   * Initialize population with random genomes
   */
  initializePopulation(numAgents: number): void {
    this.population = [];
    
    for (let i = 0; i < this.config.populationSize; i++) {
      // Random binary mask (70% chance of signal being active)
      const activeSignals = Array.from({ length: numAgents }, () =>
        Math.random() > 0.3 ? 1 : 0
      );

      // Ensure at least 2 signals are active
      const activeCount = activeSignals.reduce((sum, val) => sum + val, 0);
      if (activeCount < 2) {
        const indices = activeSignals.map((_, idx) => idx);
        const randomIdx1 = indices[Math.floor(Math.random() * indices.length)];
        let randomIdx2 = randomIdx1;
        while (randomIdx2 === randomIdx1) {
          randomIdx2 = indices[Math.floor(Math.random() * indices.length)];
        }
        activeSignals[randomIdx1] = 1;
        activeSignals[randomIdx2] = 1;
      }

      // Generate weights using Dirichlet distribution (sum to 1)
      const weights = this.generateDirichletWeights(numAgents);

      // Zero out weights for inactive signals
      for (let j = 0; j < numAgents; j++) {
        if (activeSignals[j] === 0) {
          weights[j] = 0;
        }
      }

      // Renormalize weights
      const sumWeights = weights.reduce((sum, w) => sum + w, 0);
      const normalizedWeights = weights.map(w => sumWeights > 0 ? w / sumWeights : 0);

      const genome: SignalGenome = {
        id: `genome_${i}_gen_0`,
        activeSignals,
        weights: normalizedWeights,
        fitness: 0,
        generation: 0,
        age: 0,
      };

      this.population.push(genome);
    }
  }

  /**
   * Generate weights from Dirichlet distribution (sum to 1)
   */
  private generateDirichletWeights(n: number): number[] {
    // Simplified Dirichlet: sample from Gamma, then normalize
    const alpha = 1; // Concentration parameter
    const gammas = Array.from({ length: n }, () => this.sampleGamma(alpha, 1));
    const sum = gammas.reduce((s, g) => s + g, 0);
    return gammas.map(g => g / sum);
  }

  /**
   * Sample from Gamma distribution (shape, scale)
   * Using Marsaglia and Tsang's method
   */
  private sampleGamma(shape: number, scale: number): number {
    if (shape < 1) {
      return this.sampleGamma(shape + 1, scale) * Math.pow(Math.random(), 1 / shape);
    }

    const d = shape - 1 / 3;
    const c = 1 / Math.sqrt(9 * d);

    while (true) {
      let x, v;
      do {
        x = this.sampleNormal(0, 1);
        v = 1 + c * x;
      } while (v <= 0);

      v = v * v * v;
      const u = Math.random();
      if (u < 1 - 0.0331 * x * x * x * x) {
        return d * v * scale;
      }
      if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) {
        return d * v * scale;
      }
    }
  }

  /**
   * Sample from normal distribution (Box-Muller transform)
   */
  private sampleNormal(mean: number, stdDev: number): number {
    const u1 = Math.random();
    const u2 = Math.random();
    const z0 = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    return z0 * stdDev + mean;
  }

  /**
   * Calculate fitness for a genome using backtest results
   */
  calculateFitness(
    genome: SignalGenome,
    backtestResult: BacktestResult,
    correlationMatrix: number[][]
  ): number {
    const { sharpe, maxDrawdown, turnover } = backtestResult;
    const { sharpe: sharpeWeight, correlation: corrWeight, turnover: turnoverWeight, drawdown: drawdownWeight } = this.config.fitnessWeights;

    // Sharpe ratio component (positive contribution)
    const sharpeComponent = Math.max(0, sharpe) * sharpeWeight;

    // Correlation penalty (penalize redundant signals)
    const correlationPenalty = this.calculateCorrelationPenalty(genome, correlationMatrix);
    const corrComponent = -correlationPenalty * corrWeight;

    // Turnover penalty (penalize high trading frequency)
    const turnoverPenalty = Math.max(0, turnover - 50) / 50; // Penalize if >50 trades
    const turnoverComponent = -turnoverPenalty * turnoverWeight;

    // Drawdown penalty (penalize large losses)
    const drawdownPenalty = Math.abs(maxDrawdown) / 0.1; // Normalize by 10% drawdown
    const drawdownComponent = -drawdownPenalty * drawdownWeight;

    // Total fitness
    const fitness =
      sharpeComponent +
      corrComponent +
      turnoverComponent +
      drawdownComponent;

    return fitness;
  }

  /**
   * Calculate correlation penalty for genome
   * Higher penalty if signals are highly correlated (redundant)
   */
  private calculateCorrelationPenalty(genome: SignalGenome, correlationMatrix: number[][]): number {
    const activeIndices = genome.activeSignals
      .map((val, idx) => (val === 1 ? idx : -1))
      .filter(idx => idx !== -1);

    if (activeIndices.length <= 1) {
      return 0; // No penalty for 1 or fewer signals
    }

    let totalCorrelation = 0;
    let count = 0;

    for (let i = 0; i < activeIndices.length; i++) {
      for (let j = i + 1; j < activeIndices.length; j++) {
        const idx1 = activeIndices[i];
        const idx2 = activeIndices[j];
        const corr = Math.abs(correlationMatrix[idx1][idx2]);
        totalCorrelation += corr;
        count++;
      }
    }

    return count > 0 ? totalCorrelation / count : 0;
  }

  /**
   * Tournament selection for parent selection
   */
  private tournamentSelection(tournamentSize: number = 5): SignalGenome {
    const tournament: SignalGenome[] = [];
    for (let i = 0; i < tournamentSize; i++) {
      const randomIdx = Math.floor(Math.random() * this.population.length);
      tournament.push(this.population[randomIdx]);
    }

    // Return genome with highest fitness
    return tournament.reduce((best, genome) =>
      genome.fitness > best.fitness ? genome : best
    );
  }

  /**
   * Crossover (breed) two parent genomes
   */
  private crossover(parent1: SignalGenome, parent2: SignalGenome): [SignalGenome, SignalGenome] {
    if (Math.random() > this.config.crossoverRate) {
      // No crossover, return copies of parents
      return [{ ...parent1 }, { ...parent2 }];
    }

    const numAgents = parent1.activeSignals.length;
    const crossoverPoint = Math.floor(Math.random() * numAgents);

    // Single-point crossover
    const child1ActiveSignals = [
      ...parent1.activeSignals.slice(0, crossoverPoint),
      ...parent2.activeSignals.slice(crossoverPoint),
    ];
    const child2ActiveSignals = [
      ...parent2.activeSignals.slice(0, crossoverPoint),
      ...parent1.activeSignals.slice(crossoverPoint),
    ];

    const child1Weights = [
      ...parent1.weights.slice(0, crossoverPoint),
      ...parent2.weights.slice(crossoverPoint),
    ];
    const child2Weights = [
      ...parent2.weights.slice(0, crossoverPoint),
      ...parent1.weights.slice(crossoverPoint),
    ];

    // Renormalize weights
    const child1 = this.createGenome(child1ActiveSignals, child1Weights);
    const child2 = this.createGenome(child2ActiveSignals, child2Weights);

    return [child1, child2];
  }

  /**
   * Mutate a genome
   */
  private mutate(genome: SignalGenome): SignalGenome {
    const mutated = { ...genome };
    mutated.activeSignals = [...genome.activeSignals];
    mutated.weights = [...genome.weights];

    const numAgents = mutated.activeSignals.length;

    for (let i = 0; i < numAgents; i++) {
      // Mutate active signals
      if (Math.random() < this.config.mutationRate) {
        mutated.activeSignals[i] = mutated.activeSignals[i] === 1 ? 0 : 1;
      }

      // Mutate weights (small Gaussian noise)
      if (Math.random() < this.config.mutationRate) {
        const noise = this.sampleNormal(0, 0.1);
        mutated.weights[i] = Math.max(0, mutated.weights[i] + noise);
      }
    }

    // Ensure at least 2 signals active
    const activeCount = mutated.activeSignals.reduce((sum, val) => sum + val, 0);
    if (activeCount < 2) {
      const inactiveIndices = mutated.activeSignals
        .map((val, idx) => (val === 0 ? idx : -1))
        .filter(idx => idx !== -1);
      if (inactiveIndices.length >= 2) {
        const randomIdx1 = inactiveIndices[Math.floor(Math.random() * inactiveIndices.length)];
        mutated.activeSignals[randomIdx1] = 1;
      }
    }

    // Renormalize
    return this.createGenome(mutated.activeSignals, mutated.weights);
  }

  /**
   * Create genome with normalized weights
   */
  private createGenome(activeSignals: number[], weights: number[]): SignalGenome {
    // Zero out weights for inactive signals
    const adjustedWeights = weights.map((w, idx) => (activeSignals[idx] === 1 ? w : 0));

    // Renormalize
    const sumWeights = adjustedWeights.reduce((sum, w) => sum + w, 0);
    const normalizedWeights = adjustedWeights.map(w => (sumWeights > 0 ? w / sumWeights : 0));

    return {
      id: `genome_${Math.random().toString(36).substr(2, 9)}`,
      activeSignals,
      weights: normalizedWeights,
      fitness: 0,
      generation: this.generation,
      age: 0,
    };
  }

  /**
   * Evolve population for one generation
   */
  evolve(
    fitnessEvaluator: (genome: SignalGenome) => BacktestResult,
    correlationMatrix: number[][]
  ): void {
    // Evaluate fitness for all genomes
    for (const genome of this.population) {
      const backtestResult = fitnessEvaluator(genome);
      genome.fitness = this.calculateFitness(genome, backtestResult, correlationMatrix);
    }

    // Sort by fitness (descending)
    this.population.sort((a, b) => b.fitness - a.fitness);

    // Elitism: keep top performers
    const eliteCount = Math.floor(this.config.populationSize * this.config.eliteRatio);
    const elite = this.population.slice(0, eliteCount);
    elite.forEach(g => g.age++);

    // Store best genome
    if (!this.bestGenome || elite[0].fitness > this.bestGenome.fitness) {
      this.bestGenome = { ...elite[0] };
    }

    // Generate offspring
    const offspring: SignalGenome[] = [];
    while (offspring.length < this.config.populationSize - eliteCount) {
      const parent1 = this.tournamentSelection();
      const parent2 = this.tournamentSelection();
      const [child1, child2] = this.crossover(parent1, parent2);

      offspring.push(this.mutate(child1));
      if (offspring.length < this.config.populationSize - eliteCount) {
        offspring.push(this.mutate(child2));
      }
    }

    // New population: elite + offspring
    this.population = [...elite, ...offspring];
    this.generation++;
  }

  /**
   * Run full GA evolution
   */
  run(
    fitnessEvaluator: (genome: SignalGenome) => BacktestResult,
    correlationMatrix: number[][],
    numAgents: number
  ): SignalGenome {
    this.initializePopulation(numAgents);

    for (let gen = 0; gen < this.config.maxGenerations; gen++) {
      this.evolve(fitnessEvaluator, correlationMatrix);

      const bestFitness = this.population[0].fitness;
      const avgFitness = this.population.reduce((sum, g) => sum + g.fitness, 0) / this.population.length;

      console.log(
        `Generation ${gen + 1}/${this.config.maxGenerations}: ` +
        `Best Fitness = ${bestFitness.toFixed(4)}, ` +
        `Avg Fitness = ${avgFitness.toFixed(4)}`
      );
    }

    return this.bestGenome!;
  }

  /**
   * Get best genome
   */
  getBestGenome(): SignalGenome | null {
    return this.bestGenome;
  }

  /**
   * Get current population
   */
  getPopulation(): SignalGenome[] {
    return this.population;
  }

  /**
   * Get generation number
   */
  getGeneration(): number {
    return this.generation;
  }
}

/**
 * Example usage:
 * 
 * const ga = new GeneticAlgorithmSignalSelector({
 *   populationSize: 100,
 *   mutationRate: 0.05,
 *   crossoverRate: 0.8,
 *   eliteRatio: 0.1,
 *   maxGenerations: 50
 * });
 * 
 * const correlationMatrix = [
 *   [1.0, 0.3, 0.2, 0.1, 0.4],
 *   [0.3, 1.0, 0.5, 0.2, 0.3],
 *   ...
 * ];
 * 
 * const fitnessEvaluator = (genome: SignalGenome) => {
 *   // Run backtest with genome's signal selection
 *   return backtestStrategy(genome);
 * };
 * 
 * const bestGenome = ga.run(fitnessEvaluator, correlationMatrix, 5);
 * console.log('Best genome:', bestGenome);
 */
