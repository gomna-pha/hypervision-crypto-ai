/**
 * Layer 10 Part 1: Complete ML Orchestrator with Multi-Horizon Integration
 * 
 * Integrates all 9 layers into a unified pipeline:
 * 1. Data Ingestion
 * 2. Feature Engineering (TimeScaleFeatureStore)
 * 3. Horizon-Based Agent Pool (15 agents across 3 horizons)
 * 4. Multi-Horizon Signal Aggregation
 * 5. Volatility-Adaptive GA (horizon weights)
 * 6. Hierarchical Graph + Hyperbolic Embedding
 * 7. Multi-Horizon Regime Detection
 * 8. Meta-Strategy Controller (the "brain")
 * 9. Horizon-Matched Execution Engine
 */

import { TimeScaleFeatureStore, TimeHorizon, HorizonFeatureSet } from './time-scale-feature-store';
import { MultiHorizonAgentPool, HorizonAgentType, MultiHorizonSignalOutput } from './multi-horizon-agents';
import { MultiHorizonSignalPool, HorizonIndexedSignal, SignalPoolMetrics } from './multi-horizon-signal-pool';
import { HorizonGeneticAlgorithm, VolatilityRegime, HorizonGenome } from './horizon-genetic-algorithm';
import { HorizonHierarchicalGraph } from './horizon-hierarchical-graph';
import { HorizonHyperbolicEmbedding } from './horizon-hyperbolic-embedding';
import { MultiHorizonRegimeDetector, MultiHorizonRegimeState, TransitionState } from './multi-horizon-regime-detection';
import { MetaStrategyController, MetaControllerInput, MetaControllerDecision } from './meta-strategy-controller';
import { HorizonExecutionEngine, ExecutionConfig, ExecutionOrder, ExecutionMethod } from './horizon-execution-engine';
import { RegimeConditionalStrategies, StrategySignal } from './regime-conditional-strategies';

export interface CompleteMLConfig {
  totalCapital: number;              // Total capital (USD)
  maxLeverage: number;               // Max leverage
  gaGenerations: number;             // GA generations per run
  gaOptimizationInterval: number;    // How often to run GA (ms)
  enableExecution: boolean;          // Enable actual execution
  enableBacktest: boolean;           // Enable backtesting mode
}

export interface CompletePipelineOutput {
  // Layer 1 & 2: Data & Features
  timestamp: Date;
  marketData: any;
  horizonFeatures: {
    hourly: HorizonFeatureSet;
    weekly: HorizonFeatureSet;
    monthly: HorizonFeatureSet;
  };

  // Layer 3: Agent Signals
  agentSignals: MultiHorizonSignalOutput;
  crossHorizonSync: {
    alignment: number;            // 0-1, how aligned are horizons
    conflicts: number;            // Number of conflicts detected
    avgCorrelation: number;       // Average cross-horizon correlation
  };

  // Layer 4: Signal Pool
  signalPoolMetrics: SignalPoolMetrics;
  topSignals: HorizonIndexedSignal[];

  // Layer 5: GA Optimization
  volatilityRegime: VolatilityRegime;
  horizonGenome: HorizonGenome;
  gaOptimizedWeights: {
    hourly: number;
    weekly: number;
    monthly: number;
  };

  // Layer 6: Graph & Embedding
  graphMetrics: {
    signalDecay: Map<string, number>;
    regimeFragility: number;
    horizonAgreement: number;
  };
  embeddingMetrics: {
    signalRobustness: Map<string, number>;
    regimeSimilarity: number;
  };

  // Layer 7: Regime Detection
  regimeState: MultiHorizonRegimeState;

  // Layer 8: Meta-Controller Decision
  metaDecision: MetaControllerDecision;

  // Layer 9: Execution
  executionConfig?: ExecutionConfig;
  activeOrders: ExecutionOrder[];
  executionSummary: {
    active: number;
    completed: number;
    totalVolume: number;
    avgSlippage: number;
  };

  // Strategy Signals
  strategySignals: StrategySignal[];
  activeTrades: number;

  // Performance Metrics
  performance: {
    latencyMs: number;
    componentTimings: Map<string, number>;
    throughput: number;            // Signals processed per second
  };
}

export class CompleteMLOrchestrator {
  private config: CompleteMLConfig;

  // All 9 layers
  private featureStore: TimeScaleFeatureStore;
  private agentPool: MultiHorizonAgentPool;
  private signalPool: MultiHorizonSignalPool;
  private horizonGA: HorizonGeneticAlgorithm;
  private hierarchicalGraph: HorizonHierarchicalGraph;
  private hyperbolicEmbedding: HorizonHyperbolicEmbedding;
  private regimeDetector: MultiHorizonRegimeDetector;
  private metaController: MetaStrategyController;
  private executionEngine: HorizonExecutionEngine;
  private strategies: RegimeConditionalStrategies;

  // State
  private lastGAOptimization: Date | null;
  private pipelineCount: number;

  constructor(config: Partial<CompleteMLConfig> = {}) {
    this.config = {
      totalCapital: config.totalCapital || 100000,
      maxLeverage: config.maxLeverage || 3,
      gaGenerations: config.gaGenerations || 30,
      gaOptimizationInterval: config.gaOptimizationInterval || 3600000, // 1 hour
      enableExecution: config.enableExecution !== false,
      enableBacktest: config.enableBacktest || false,
    };

    // Initialize all layers
    this.featureStore = new TimeScaleFeatureStore();
    this.agentPool = new MultiHorizonAgentPool();
    this.signalPool = new MultiHorizonSignalPool();
    this.horizonGA = new HorizonGeneticAlgorithm({
      populationSize: 50,
      maxGenerations: this.config.gaGenerations,
      fitnessWeights: {
        sharpe: 0.50,
        correlation: 0.15,
        drawdown: 0.20,
        stability: 0.15,
      },
    });
    this.hierarchicalGraph = new HorizonHierarchicalGraph();
    this.hyperbolicEmbedding = new HorizonHyperbolicEmbedding({ dimension: 8 });
    this.regimeDetector = new MultiHorizonRegimeDetector();
    this.metaController = new MetaStrategyController();
    this.executionEngine = new HorizonExecutionEngine();
    this.strategies = new RegimeConditionalStrategies();

    this.lastGAOptimization = null;
    this.pipelineCount = 0;
  }

  /**
   * Run complete ML pipeline (all 9 layers)
   */
  async runCompletePipeline(marketData: any): Promise<CompletePipelineOutput> {
    const startTime = Date.now();
    const componentTimings = new Map<string, number>();

    // ========== LAYER 1 & 2: DATA INGESTION + FEATURE ENGINEERING ==========
    let t0 = Date.now();
    
    this.featureStore.ingestMarketData(marketData);
    const hourlyFeatures = this.featureStore.getFeatures(TimeHorizon.HOURLY);
    const weeklyFeatures = this.featureStore.getFeatures(TimeHorizon.WEEKLY);
    const monthlyFeatures = this.featureStore.getFeatures(TimeHorizon.MONTHLY);

    componentTimings.set('FeatureEngineering', Date.now() - t0);

    // ========== LAYER 3: HORIZON-BASED AGENT POOL (15 AGENTS) ==========
    t0 = Date.now();

    const agentSignals = this.agentPool.generateMultiHorizonSignals(
      hourlyFeatures,
      weeklyFeatures,
      monthlyFeatures,
      marketData
    );

    const crossHorizonSync = {
      alignment: agentSignals.crossHorizonSync.alignment,
      conflicts: agentSignals.crossHorizonSync.conflicts.length,
      avgCorrelation: agentSignals.crossHorizonSync.avgCorrelation,
    };

    componentTimings.set('AgentPool', Date.now() - t0);

    // ========== LAYER 4: MULTI-HORIZON SIGNAL AGGREGATION ==========
    t0 = Date.now();

    // Add all signals to pool
    for (const signal of agentSignals.signals) {
      this.signalPool.addSignal(signal);
    }

    const signalPoolMetrics = this.signalPool.getPoolMetrics();
    const topSignals = this.signalPool.getTopSignals(10);

    componentTimings.set('SignalAggregation', Date.now() - t0);

    // ========== LAYER 5: VOLATILITY-ADAPTIVE GA ==========
    t0 = Date.now();

    // Determine if we need to run GA optimization
    const shouldRunGA = !this.lastGAOptimization || 
      (Date.now() - this.lastGAOptimization.getTime() > this.config.gaOptimizationInterval);

    let horizonGenome: HorizonGenome;
    let volatilityRegime: VolatilityRegime;

    if (shouldRunGA) {
      // Classify volatility regime
      volatilityRegime = this.horizonGA.classifyVolatilityRegime(
        hourlyFeatures.volatility.realized24h
      );

      // Prepare backtest data (simplified)
      const backtestData = this.prepareBacktestData(agentSignals);

      // Optimize horizon weights
      horizonGenome = this.horizonGA.optimizeHorizonWeights(
        backtestData,
        volatilityRegime
      );

      this.lastGAOptimization = new Date();
      console.log(`[GA] Optimized horizon weights: H=${horizonGenome.hourlyWeight.toFixed(2)}, W=${horizonGenome.weeklyWeight.toFixed(2)}, M=${horizonGenome.monthlyWeight.toFixed(2)}`);
    } else {
      // Use cached GA results
      volatilityRegime = this.horizonGA.classifyVolatilityRegime(
        hourlyFeatures.volatility.realized24h
      );
      horizonGenome = this.horizonGA.getCurrentBestGenome() || this.horizonGA.createDefaultGenome(volatilityRegime);
    }

    const gaOptimizedWeights = {
      hourly: horizonGenome.hourlyWeight,
      weekly: horizonGenome.weeklyWeight,
      monthly: horizonGenome.monthlyWeight,
    };

    componentTimings.set('GeneticAlgorithm', Date.now() - t0);

    // ========== LAYER 6: HIERARCHICAL GRAPH + HYPERBOLIC EMBEDDING ==========
    t0 = Date.now();

    // Build hierarchical graph
    const graph = this.hierarchicalGraph.buildGraph(
      agentSignals.signals,
      null, // Will be filled after regime detection
      []
    );

    // Calculate graph metrics
    const signalDecay = new Map<string, number>();
    for (const signal of agentSignals.signals) {
      const decay = this.hierarchicalGraph.calculateSharpeDecay(signal.horizon, 30); // 30 days
      signalDecay.set(`${signal.agentId}_${signal.horizon}`, decay);
    }

    // Embed into hyperbolic space
    this.hyperbolicEmbedding.embedHorizonGraph(graph);

    // Calculate embedding metrics
    const signalRobustness = new Map<string, number>();
    for (const signal of agentSignals.signals) {
      const nodeId = `signal_${signal.agentId}_${signal.horizon}`;
      const robustness = this.hyperbolicEmbedding.getSignalRobustness(nodeId);
      if (robustness !== undefined) {
        signalRobustness.set(nodeId, robustness);
      }
    }

    componentTimings.set('GraphEmbedding', Date.now() - t0);

    // ========== LAYER 7: MULTI-HORIZON REGIME DETECTION ==========
    t0 = Date.now();

    const regimeState = this.regimeDetector.detectMultiHorizonRegime(
      hourlyFeatures,
      weeklyFeatures,
      monthlyFeatures,
      marketData
    );

    componentTimings.set('RegimeDetection', Date.now() - t0);

    // Calculate additional graph metrics now that we have regime
    const regimeFragility = this.hierarchicalGraph.assessRegimeFragility(regimeState.consensusRegime);
    const horizonAgreement = this.hierarchicalGraph.calculateHorizonRegimeAgreement(
      regimeState.hourlyRegime,
      regimeState.weeklyRegime,
      regimeState.monthlyRegime
    );

    const graphMetrics = { signalDecay, regimeFragility, horizonAgreement };
    const embeddingMetrics = { 
      signalRobustness, 
      regimeSimilarity: 0.75, // Simplified for now
    };

    // ========== LAYER 8: META-STRATEGY CONTROLLER (THE BRAIN) ==========
    t0 = Date.now();

    const metaInput: MetaControllerInput = {
      volatility: hourlyFeatures.volatility.realized24h,
      regime: regimeState.consensusRegime,
      regimeConfidence: regimeState.confidence,
      transitionState: regimeState.transitionState,
      hourlyRegime: regimeState.hourlyRegime,
      weeklyRegime: regimeState.weeklyRegime,
      monthlyRegime: regimeState.monthlyRegime,
      horizonAlignment: agentSignals.crossHorizonSync.alignment,
      hourlyStability: this.calculateStability(hourlyFeatures),
      weeklyStability: this.calculateStability(weeklyFeatures),
      monthlyStability: this.calculateStability(monthlyFeatures),
      signalDiversity: signalPoolMetrics.diversity,
      signalConsensus: signalPoolMetrics.consensus,
      avgSignalDecay: Array.from(signalDecay.values()).reduce((a, b) => a + b, 0) / signalDecay.size,
      regimeFragility,
      topSignalStrength: topSignals[0]?.confidence || 0.5,
    };

    const metaDecision = this.metaController.generateDecision(metaInput);

    componentTimings.set('MetaController', Date.now() - t0);

    // ========== LAYER 9: REGIME-CONDITIONAL EXECUTION ==========
    t0 = Date.now();

    // Evaluate strategies
    const dummyMarketData = {
      spotPrice: marketData.spotPrice || 96500,
      perpPrice: marketData.perpPrice || 96530,
      fundingRate: marketData.fundingRate || 0.01,
      volatility: hourlyFeatures.volatility.realized24h,
      sma20: marketData.sma20 || 96500,
      priceVsSMA: 0,
    };

    const strategySignals = this.strategies.evaluateStrategies(
      { regime: regimeState.consensusRegime, confidence: regimeState.confidence } as any,
      { 
        confidenceScore: metaDecision.confidence * 100, 
        exposureScaler: metaDecision.exposureScaling,
        leverageScaler: 1.5,
      } as any,
      null as any,
      [], // Agent signals (simplified)
      dummyMarketData
    );

    // Generate execution orders if execution is enabled
    let executionConfig: ExecutionConfig | undefined;
    if (this.config.enableExecution && strategySignals.length > 0) {
      // Determine dominant horizon based on meta-decision weights
      const dominantHorizon = this.getDominantHorizon(metaDecision.horizonWeights);

      executionConfig = this.executionEngine.generateExecutionConfig(
        dominantHorizon,
        regimeState.consensusRegime,
        metaDecision
      );

      // Create execution orders for strategy signals
      for (const signal of strategySignals.slice(0, 3)) { // Limit to top 3 strategies
        if (signal.action === 'ENTER' && signal.trade) {
          const order = this.executionEngine.createExecutionOrder(
            signal,
            executionConfig,
            dummyMarketData
          );

          console.log(`[EXECUTION] Created order ${order.orderId}: ${order.method} for ${order.symbol}`);

          // Note: In production, we would execute the order here
          // For now, we just create it
        }
      }
    }

    const activeOrders = this.executionEngine.getActiveOrders();
    const executionSummary = this.executionEngine.getExecutionSummary();

    componentTimings.set('Execution', Date.now() - t0);

    // ========== FINALIZE OUTPUT ==========
    this.pipelineCount++;

    const totalLatency = Date.now() - startTime;
    const throughput = agentSignals.signals.length / (totalLatency / 1000);

    return {
      timestamp: new Date(),
      marketData,
      horizonFeatures: {
        hourly: hourlyFeatures,
        weekly: weeklyFeatures,
        monthly: monthlyFeatures,
      },
      agentSignals,
      crossHorizonSync,
      signalPoolMetrics,
      topSignals,
      volatilityRegime,
      horizonGenome,
      gaOptimizedWeights,
      graphMetrics,
      embeddingMetrics,
      regimeState,
      metaDecision,
      executionConfig,
      activeOrders,
      executionSummary,
      strategySignals,
      activeTrades: this.strategies.getActiveTrades().length,
      performance: {
        latencyMs: totalLatency,
        componentTimings,
        throughput,
      },
    };
  }

  /**
   * Get dominant horizon based on weights
   */
  private getDominantHorizon(weights: { hourly: number; weekly: number; monthly: number }): TimeHorizon {
    if (weights.hourly >= weights.weekly && weights.hourly >= weights.monthly) {
      return TimeHorizon.HOURLY;
    } else if (weights.weekly >= weights.monthly) {
      return TimeHorizon.WEEKLY;
    } else {
      return TimeHorizon.MONTHLY;
    }
  }

  /**
   * Calculate stability metric from features
   */
  private calculateStability(features: HorizonFeatureSet): number {
    // Simplified stability calculation based on volatility consistency
    const vol = features.volatility.realized24h;
    const expectedVol = 25; // Expected volatility
    const deviation = Math.abs(vol - expectedVol) / expectedVol;
    return Math.max(0, 1 - deviation);
  }

  /**
   * Prepare backtest data for GA optimization
   */
  private prepareBacktestData(agentSignals: MultiHorizonSignalOutput): any[] {
    // Simplified backtest data preparation
    // In production, this would use historical data
    return agentSignals.signals.map(signal => ({
      horizon: signal.horizon,
      agentId: signal.agentId,
      signal: signal.signal,
      confidence: signal.confidence,
      returns: Math.random() * 0.02 - 0.01, // Simulated returns
    }));
  }

  /**
   * Get system status for dashboard
   */
  getSystemStatus(): {
    status: string;
    pipelineRuns: number;
    lastUpdate: Date | null;
    activeComponents: string[];
  } {
    return {
      status: 'OPERATIONAL',
      pipelineRuns: this.pipelineCount,
      lastUpdate: this.lastGAOptimization,
      activeComponents: [
        'FeatureStore',
        'AgentPool (15 agents)',
        'SignalAggregation',
        'VolatilityGA',
        'HierarchicalGraph',
        'HyperbolicEmbedding',
        'RegimeDetection',
        'MetaController',
        'ExecutionEngine',
      ],
    };
  }

  /**
   * Get performance metrics
   */
  getPerformanceMetrics(): any {
    return {
      executionSummary: this.executionEngine.getExecutionSummary(),
      signalPoolMetrics: this.signalPool.getPoolMetrics(),
      activeTrades: this.strategies.getActiveTrades().length,
      totalCapital: this.config.totalCapital,
    };
  }
}

/**
 * Example Usage:
 * 
 * const orchestrator = new CompleteMLOrchestrator({
 *   totalCapital: 100000,
 *   maxLeverage: 3,
 *   gaGenerations: 30,
 *   enableExecution: true,
 * });
 * 
 * // Run complete pipeline
 * const output = await orchestrator.runCompletePipeline(marketData);
 * 
 * // Access all layer outputs
 * console.log('Horizon Weights:', output.metaDecision.horizonWeights);
 * console.log('Regime:', output.regimeState.consensusRegime);
 * console.log('Execution Orders:', output.activeOrders.length);
 */
