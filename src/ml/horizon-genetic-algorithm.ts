/**
 * Horizon-Aware Genetic Algorithm
 * 
 * Extends the base Genetic Algorithm with:
 * - Volatility-adaptive horizon weights
 * - Multi-horizon signal selection
 * - Cross-horizon correlation penalty
 * - Horizon stability penalty
 * - Dynamic weight rebalancing based on market conditions
 * 
 * Volatility Adaptation Rules:
 * - Low Vol (< 1.5%):    60% Hourly, 30% Weekly, 10% Monthly (exploit short-term)
 * - Normal Vol (1.5-2.5%): 40% Hourly, 40% Weekly, 20% Monthly (balanced)
 * - High Vol (2.5-3.5%):   30% Hourly, 40% Weekly, 30% Monthly (more stable)
 * - Extreme Vol (> 3.5%):  10% Hourly, 30% Weekly, 60% Monthly (structural safety)
 */

import { HorizonIndexedSignal } from './multi-horizon-signal-pool';
import { TimeHorizon } from './time-scale-feature-store';

// ============================================================================
// Horizon Genome Interface
// ============================================================================

export interface HorizonGenome {
  id: string;
  generation: number;
  age: number;
  
  // Signal selection per horizon (binary masks)
  activeSignals: {
    hourly: number[];   // [0, 1, 1, 0, 1] for 5 agents
    weekly: number[];   // [0, 1, 1, 0, 1] for 5 agents
    monthly: number[];  // [0, 1, 1, 0, 1] for 5 agents
  };
  
  // Signal weights per horizon (sum to 1 within each horizon)
  signalWeights: {
    hourly: number[];   // [0.0, 0.4, 0.3, 0.0, 0.3]
    weekly: number[];   // [0.0, 0.2, 0.5, 0.0, 0.3]
    monthly: number[];  // [0.0, 0.3, 0.4, 0.0, 0.3]
  };
  
  // Horizon allocation weights (sum to 1)
  horizonWeights: {
    hourly: number;   // ∈ [0, 1]
    weekly: number;   // ∈ [0, 1]
    monthly: number;  // ∈ [0, 1]
  };
  // Constraint: hourly + weekly + monthly = 1.0
  
  // Fitness metrics
  fitness: number;
  sharpe: number;
  maxDrawdown: number;
  correlationPenalty: number;
  stabilityPenalty: number;
  
  // Volatility adaptation
  adaptedForVolatility: 'low' | 'normal' | 'high' | 'extreme';
}

// ============================================================================
// Volatility Regime
// ============================================================================

export type VolatilityRegime = 'low' | 'normal' | 'high' | 'extreme';

export interface VolatilityRegimeParams {
  threshold: number;  // Vol threshold
  horizonWeights: {
    hourly: number;
    weekly: number;
    monthly: number;
  };
}

export const VOLATILITY_REGIMES: Record<VolatilityRegime, VolatilityRegimeParams> = {
  low: {
    threshold: 0.015,  // < 1.5% volatility
    horizonWeights: {
      hourly: 0.60,   // Exploit short-term opportunities
      weekly: 0.30,
      monthly: 0.10,
    },
  },
  normal: {
    threshold: 0.025,  // 1.5% - 2.5% volatility
    horizonWeights: {
      hourly: 0.40,   // Balanced approach
      weekly: 0.40,
      monthly: 0.20,
    },
  },
  high: {
    threshold: 0.035,  // 2.5% - 3.5% volatility
    horizonWeights: {
      hourly: 0.30,   // Shift to more stable signals
      weekly: 0.40,
      monthly: 0.30,
    },
  },
  extreme: {
    threshold: Infinity,  // > 3.5% volatility
    horizonWeights: {
      hourly: 0.10,   // Focus on structural signals
      weekly: 0.30,
      monthly: 0.60,  // Safe haven
    },
  },
};

// ============================================================================
// Horizon GA Configuration
// ============================================================================

export interface HorizonGAConfig {
  populationSize: number;
  mutationRate: number;
  crossoverRate: number;
  eliteRatio: number;
  maxGenerations: number;
  
  // Fitness weights
  fitnessWeights: {
    sharpe: number;
    correlation: number;
    stability: number;
    drawdown: number;
  };
  
  // Penalties
  correlationPenaltyWeight: number;  // Penalty for high cross-horizon correlation
  stabilityPenaltyWeight: number;    // Penalty for frequent horizon switches
  
  // Volatility adaptation
  volatilityAdaptive: boolean;
  volatilityLookbackHours: number;  // Hours to look back for volatility calculation
}

// ============================================================================
// Horizon-Aware Genetic Algorithm
// ============================================================================

export class HorizonAwareGeneticAlgorithm {
  private config: HorizonGAConfig;
  private population: HorizonGenome[];
  private generation: number;
  private bestGenome: HorizonGenome | null;
  private currentVolatilityRegime: VolatilityRegime;
  
  constructor(config: Partial<HorizonGAConfig> = {}) {
    this.config = {
      populationSize: config.populationSize || 100,
      mutationRate: config.mutationRate || 0.05,
      crossoverRate: config.crossoverRate || 0.8,
      eliteRatio: config.eliteRatio || 0.1,
      maxGenerations: config.maxGenerations || 50,
      fitnessWeights: config.fitnessWeights || {
        sharpe: 0.50,
        correlation: 0.20,
        stability: 0.15,
        drawdown: 0.15,
      },
      correlationPenaltyWeight: config.correlationPenaltyWeight || 0.5,
      stabilityPenaltyWeight: config.stabilityPenaltyWeight || 0.3,
      volatilityAdaptive: config.volatilityAdaptive ?? true,
      volatilityLookbackHours: config.volatilityLookbackHours || 24,
    };
    
    this.population = [];
    this.generation = 0;
    this.bestGenome = null;
    this.currentVolatilityRegime = 'normal';
  }
  
  // ============================================================================
  // Volatility Regime Detection
  // ============================================================================
  
  /**
   * Classify current volatility regime
   */
  classifyVolatilityRegime(volatility: number): VolatilityRegime {
    if (volatility < VOLATILITY_REGIMES.low.threshold) {
      return 'low';
    } else if (volatility < VOLATILITY_REGIMES.normal.threshold) {
      return 'normal';
    } else if (volatility < VOLATILITY_REGIMES.high.threshold) {
      return 'high';
    } else {
      return 'extreme';
    }
  }
  
  /**
   * Adapt horizon weights based on volatility regime
   */
  adaptToVolatilityRegime(volatility: number): void {
    const newRegime = this.classifyVolatilityRegime(volatility);
    
    if (newRegime !== this.currentVolatilityRegime) {
      console.log(`[HorizonGA] Volatility regime change: ${this.currentVolatilityRegime} → ${newRegime}`);
      console.log(`[HorizonGA] New horizon weights: ${JSON.stringify(VOLATILITY_REGIMES[newRegime].horizonWeights)}`);
      this.currentVolatilityRegime = newRegime;
    }
  }
  
  /**
   * Get current horizon weights based on volatility regime
   */
  getCurrentHorizonWeights(): { hourly: number; weekly: number; monthly: number } {
    return VOLATILITY_REGIMES[this.currentVolatilityRegime].horizonWeights;
  }
  
  // ============================================================================
  // Population Initialization
  // ============================================================================
  
  /**
   * Initialize population with random genomes
   */
  initializePopulation(numAgentsPerHorizon: number = 5): void {
    this.population = [];
    
    const targetWeights = this.getCurrentHorizonWeights();
    
    for (let i = 0; i < this.config.populationSize; i++) {
      const genome = this.createRandomGenome(numAgentsPerHorizon, targetWeights);
      this.population.push(genome);
    }
    
    this.generation = 0;
  }
  
  /**
   * Create a random genome
   */
  private createRandomGenome(
    numAgentsPerHorizon: number,
    targetHorizonWeights: { hourly: number; weekly: number; monthly: number }
  ): HorizonGenome {
    // Random active signals (70% chance per agent)
    const hourlyActive = this.generateRandomActiveMask(numAgentsPerHorizon);
    const weeklyActive = this.generateRandomActiveMask(numAgentsPerHorizon);
    const monthlyActive = this.generateRandomActiveMask(numAgentsPerHorizon);
    
    // Generate signal weights (Dirichlet-like)
    const hourlyWeights = this.generateSignalWeights(hourlyActive);
    const weeklyWeights = this.generateSignalWeights(weeklyActive);
    const monthlyWeights = this.generateSignalWeights(monthlyActive);
    
    // Horizon weights with small random perturbation (±10%)
    const horizonWeights = {
      hourly: this.perturbWeight(targetHorizonWeights.hourly, 0.1),
      weekly: this.perturbWeight(targetHorizonWeights.weekly, 0.1),
      monthly: this.perturbWeight(targetHorizonWeights.monthly, 0.1),
    };
    
    // Normalize to sum to 1.0
    const sum = horizonWeights.hourly + horizonWeights.weekly + horizonWeights.monthly;
    horizonWeights.hourly /= sum;
    horizonWeights.weekly /= sum;
    horizonWeights.monthly /= sum;
    
    return {
      id: this.generateGenomeId(),
      generation: this.generation,
      age: 0,
      activeSignals: {
        hourly: hourlyActive,
        weekly: weeklyActive,
        monthly: monthlyActive,
      },
      signalWeights: {
        hourly: hourlyWeights,
        weekly: weeklyWeights,
        monthly: monthlyWeights,
      },
      horizonWeights,
      fitness: 0,
      sharpe: 0,
      maxDrawdown: 0,
      correlationPenalty: 0,
      stabilityPenalty: 0,
      adaptedForVolatility: this.currentVolatilityRegime,
    };
  }
  
  private generateRandomActiveMask(length: number): number[] {
    const mask = Array.from({ length }, () => Math.random() > 0.3 ? 1 : 0);
    
    // Ensure at least 2 signals are active
    const activeCount = mask.reduce((sum, val) => sum + val, 0);
    if (activeCount < 2) {
      mask[0] = 1;
      mask[1] = 1;
    }
    
    return mask;
  }
  
  private generateSignalWeights(activeMask: number[]): number[] {
    const weights = activeMask.map(active => active === 1 ? Math.random() : 0);
    const sum = weights.reduce((s, w) => s + w, 0);
    return weights.map(w => sum > 0 ? w / sum : 0);
  }
  
  private perturbWeight(weight: number, maxPerturbation: number): number {
    const perturbation = (Math.random() - 0.5) * 2 * maxPerturbation * weight;
    return Math.max(0, Math.min(1, weight + perturbation));
  }
  
  private generateGenomeId(): string {
    return `genome_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  // ============================================================================
  // Fitness Evaluation
  // ============================================================================
  
  /**
   * Evaluate fitness for a genome
   */
  evaluateFitness(
    genome: HorizonGenome,
    signals: {
      hourly: HorizonIndexedSignal[];
      weekly: HorizonIndexedSignal[];
      monthly: HorizonIndexedSignal[];
    },
    backtestResults: {
      sharpe: number;
      maxDrawdown: number;
      returns: number[];
    }
  ): number {
    // Base fitness from backtest
    const sharpeFitness = Math.max(0, backtestResults.sharpe) / 3.0;  // Normalize to ~[0, 1]
    const drawdownFitness = Math.max(0, 1 - Math.abs(backtestResults.maxDrawdown));
    
    // Calculate correlation penalty
    const correlationPenalty = this.calculateCorrelationPenalty(genome, signals);
    
    // Calculate stability penalty (how often horizon weights change)
    const stabilityPenalty = this.calculateStabilityPenalty(genome);
    
    // Weighted fitness
    const fitness =
      this.config.fitnessWeights.sharpe * sharpeFitness +
      this.config.fitnessWeights.drawdown * drawdownFitness -
      this.config.fitnessWeights.correlation * correlationPenalty -
      this.config.fitnessWeights.stability * stabilityPenalty;
    
    // Update genome metrics
    genome.fitness = Math.max(0, fitness);
    genome.sharpe = backtestResults.sharpe;
    genome.maxDrawdown = backtestResults.maxDrawdown;
    genome.correlationPenalty = correlationPenalty;
    genome.stabilityPenalty = stabilityPenalty;
    
    return genome.fitness;
  }
  
  /**
   * Calculate correlation penalty across horizons
   */
  private calculateCorrelationPenalty(
    genome: HorizonGenome,
    signals: {
      hourly: HorizonIndexedSignal[];
      weekly: HorizonIndexedSignal[];
      monthly: HorizonIndexedSignal[];
    }
  ): number {
    // Get active signals from each horizon
    const hourlyActiveSignals = genome.activeSignals.hourly
      .map((active, idx) => active === 1 ? signals.hourly[idx] : null)
      .filter(s => s !== null) as HorizonIndexedSignal[];
    
    const weeklyActiveSignals = genome.activeSignals.weekly
      .map((active, idx) => active === 1 ? signals.weekly[idx] : null)
      .filter(s => s !== null) as HorizonIndexedSignal[];
    
    const monthlyActiveSignals = genome.activeSignals.monthly
      .map((active, idx) => active === 1 ? signals.monthly[idx] : null)
      .filter(s => s !== null) as HorizonIndexedSignal[];
    
    // Calculate pairwise correlation across horizons
    let totalCorrelation = 0;
    let pairCount = 0;
    
    // Hourly-Weekly correlation
    for (const h of hourlyActiveSignals) {
      for (const w of weeklyActiveSignals) {
        if (h.agentType === w.agentType) {
          totalCorrelation += h.direction * w.direction;
          pairCount++;
        }
      }
    }
    
    // Weekly-Monthly correlation
    for (const w of weeklyActiveSignals) {
      for (const m of monthlyActiveSignals) {
        if (w.agentType === m.agentType) {
          totalCorrelation += w.direction * m.direction;
          pairCount++;
        }
      }
    }
    
    // Hourly-Monthly correlation
    for (const h of hourlyActiveSignals) {
      for (const m of monthlyActiveSignals) {
        if (h.agentType === m.agentType) {
          totalCorrelation += h.direction * m.direction;
          pairCount++;
        }
      }
    }
    
    const avgCorrelation = pairCount > 0 ? totalCorrelation / pairCount : 0;
    
    // Penalize high correlation (we want diversification across horizons)
    return Math.max(0, avgCorrelation) * this.config.correlationPenaltyWeight;
  }
  
  /**
   * Calculate stability penalty (penalize frequent horizon weight changes)
   */
  private calculateStabilityPenalty(genome: HorizonGenome): number {
    const targetWeights = this.getCurrentHorizonWeights();
    
    // Measure deviation from target weights
    const deviation =
      Math.abs(genome.horizonWeights.hourly - targetWeights.hourly) +
      Math.abs(genome.horizonWeights.weekly - targetWeights.weekly) +
      Math.abs(genome.horizonWeights.monthly - targetWeights.monthly);
    
    return deviation * this.config.stabilityPenaltyWeight;
  }
  
  // ============================================================================
  // Evolution Operations
  // ============================================================================
  
  /**
   * Run one generation of evolution
   */
  evolve(
    signals: {
      hourly: HorizonIndexedSignal[];
      weekly: HorizonIndexedSignal[];
      monthly: HorizonIndexedSignal[];
    },
    currentVolatility: number
  ): HorizonGenome {
    // Adapt to current volatility regime
    if (this.config.volatilityAdaptive) {
      this.adaptToVolatilityRegime(currentVolatility);
    }
    
    // Sort population by fitness
    this.population.sort((a, b) => b.fitness - a.fitness);
    
    // Select elite genomes
    const eliteCount = Math.floor(this.config.populationSize * this.config.eliteRatio);
    const elites = this.population.slice(0, eliteCount);
    
    // Increment age for elites
    elites.forEach(e => e.age++);
    
    // Create new population
    const newPopulation: HorizonGenome[] = [...elites];
    
    // Fill rest with offspring
    while (newPopulation.length < this.config.populationSize) {
      // Tournament selection
      const parent1 = this.tournamentSelect(this.population);
      const parent2 = this.tournamentSelect(this.population);
      
      // Crossover
      let offspring: HorizonGenome;
      if (Math.random() < this.config.crossoverRate) {
        offspring = this.crossover(parent1, parent2);
      } else {
        offspring = { ...parent1, id: this.generateGenomeId(), generation: this.generation + 1, age: 0 };
      }
      
      // Mutation
      if (Math.random() < this.config.mutationRate) {
        this.mutate(offspring);
      }
      
      newPopulation.push(offspring);
    }
    
    this.population = newPopulation;
    this.generation++;
    
    // Update best genome
    this.bestGenome = this.population[0];
    
    return this.bestGenome;
  }
  
  /**
   * Tournament selection
   */
  private tournamentSelect(population: HorizonGenome[], tournamentSize: number = 3): HorizonGenome {
    const tournament: HorizonGenome[] = [];
    
    for (let i = 0; i < tournamentSize; i++) {
      const randomIdx = Math.floor(Math.random() * population.length);
      tournament.push(population[randomIdx]);
    }
    
    tournament.sort((a, b) => b.fitness - a.fitness);
    return tournament[0];
  }
  
  /**
   * Crossover two genomes
   */
  private crossover(parent1: HorizonGenome, parent2: HorizonGenome): HorizonGenome {
    const offspring: HorizonGenome = {
      id: this.generateGenomeId(),
      generation: this.generation + 1,
      age: 0,
      activeSignals: {
        hourly: [],
        weekly: [],
        monthly: [],
      },
      signalWeights: {
        hourly: [],
        weekly: [],
        monthly: [],
      },
      horizonWeights: { hourly: 0, weekly: 0, monthly: 0 },
      fitness: 0,
      sharpe: 0,
      maxDrawdown: 0,
      correlationPenalty: 0,
      stabilityPenalty: 0,
      adaptedForVolatility: this.currentVolatilityRegime,
    };
    
    // Single-point crossover for each horizon
    const crossoverPoint = Math.floor(parent1.activeSignals.hourly.length / 2);
    
    for (const horizon of ['hourly', 'weekly', 'monthly'] as TimeHorizon[]) {
      offspring.activeSignals[horizon] = [
        ...parent1.activeSignals[horizon].slice(0, crossoverPoint),
        ...parent2.activeSignals[horizon].slice(crossoverPoint),
      ];
      
      offspring.signalWeights[horizon] = [
        ...parent1.signalWeights[horizon].slice(0, crossoverPoint),
        ...parent2.signalWeights[horizon].slice(crossoverPoint),
      ];
      
      // Renormalize weights
      const sum = offspring.signalWeights[horizon].reduce((s, w) => s + w, 0);
      if (sum > 0) {
        offspring.signalWeights[horizon] = offspring.signalWeights[horizon].map(w => w / sum);
      }
    }
    
    // Average horizon weights
    offspring.horizonWeights.hourly = (parent1.horizonWeights.hourly + parent2.horizonWeights.hourly) / 2;
    offspring.horizonWeights.weekly = (parent1.horizonWeights.weekly + parent2.horizonWeights.weekly) / 2;
    offspring.horizonWeights.monthly = (parent1.horizonWeights.monthly + parent2.horizonWeights.monthly) / 2;
    
    // Normalize
    const hSum = offspring.horizonWeights.hourly + offspring.horizonWeights.weekly + offspring.horizonWeights.monthly;
    offspring.horizonWeights.hourly /= hSum;
    offspring.horizonWeights.weekly /= hSum;
    offspring.horizonWeights.monthly /= hSum;
    
    return offspring;
  }
  
  /**
   * Mutate a genome
   */
  private mutate(genome: HorizonGenome): void {
    // Mutate active signals (flip random bit)
    for (const horizon of ['hourly', 'weekly', 'monthly'] as TimeHorizon[]) {
      const idx = Math.floor(Math.random() * genome.activeSignals[horizon].length);
      genome.activeSignals[horizon][idx] = genome.activeSignals[horizon][idx] === 1 ? 0 : 1;
      
      // If deactivated, zero out weight
      if (genome.activeSignals[horizon][idx] === 0) {
        genome.signalWeights[horizon][idx] = 0;
      } else {
        genome.signalWeights[horizon][idx] = Math.random();
      }
      
      // Renormalize weights
      const sum = genome.signalWeights[horizon].reduce((s, w) => s + w, 0);
      if (sum > 0) {
        genome.signalWeights[horizon] = genome.signalWeights[horizon].map(w => w / sum);
      }
    }
    
    // Mutate horizon weights (small perturbation)
    const horizonToMutate = ['hourly', 'weekly', 'monthly'][Math.floor(Math.random() * 3)] as keyof typeof genome.horizonWeights;
    genome.horizonWeights[horizonToMutate] += (Math.random() - 0.5) * 0.2;
    genome.horizonWeights[horizonToMutate] = Math.max(0.05, Math.min(0.8, genome.horizonWeights[horizonToMutate]));
    
    // Renormalize horizon weights
    const hSum = genome.horizonWeights.hourly + genome.horizonWeights.weekly + genome.horizonWeights.monthly;
    genome.horizonWeights.hourly /= hSum;
    genome.horizonWeights.weekly /= hSum;
    genome.horizonWeights.monthly /= hSum;
  }
  
  // ============================================================================
  // Getters
  // ============================================================================
  
  getBestGenome(): HorizonGenome | null {
    return this.bestGenome;
  }
  
  getPopulation(): HorizonGenome[] {
    return this.population;
  }
  
  getGeneration(): number {
    return this.generation;
  }
  
  getVolatilityRegime(): VolatilityRegime {
    return this.currentVolatilityRegime;
  }
  
  getTopGenomes(count: number): HorizonGenome[] {
    return [...this.population].sort((a, b) => b.fitness - a.fitness).slice(0, count);
  }
}
