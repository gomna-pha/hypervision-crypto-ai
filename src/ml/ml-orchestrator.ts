/**
 * ML Orchestrator - Central Integration Hub
 * 
 * Coordinates all ML components:
 * 1. Feature Engineering
 * 2. Agent Signal Generation
 * 3. Genetic Algorithm Signal Selection
 * 4. Hyperbolic Embedding
 * 5. Market Regime Detection
 * 6. XGBoost Meta-Model
 * 7. Regime-Conditional Strategies
 * 8. Portfolio & Risk Management
 */

import { FeatureEngineer, RawMarketData, EngineeredFeatures } from './feature-engineering';
import {
  EconomicAgent,
  SentimentAgent,
  CrossExchangeAgent,
  OnChainAgent,
  CNNPatternAgent,
  AgentSignal,
} from './agent-signal';
import { GeneticAlgorithmSignalSelector, SignalGenome, BacktestResult } from './genetic-algorithm';
import { HyperbolicEmbedding, HierarchicalGraph, HierarchicalNode, HyperbolicPoint } from './hyperbolic-embedding';
import { MarketRegimeDetector, MarketRegime, RegimeState, RegimeFeatures } from './market-regime-detection';
import { XGBoostMetaModel, MetaModelInput, MetaModelOutput } from './xgboost-meta-model';
import { RegimeConditionalStrategies, StrategySignal, Trade } from './regime-conditional-strategies';
import { PortfolioRiskManager, PortfolioMetrics, RiskConstraint } from './portfolio-risk-manager';

export interface MLOrchestratorConfig {
  enableGA: boolean;
  enableHyperbolic: boolean;
  enableXGBoost: boolean;
  enableStrategies: boolean;
  enableRiskManager: boolean;
  gaGenerations?: number;
  hyperbolicDimension?: number;
  totalCapital?: number;
}

export interface MLPipelineOutput {
  // Raw & engineered data
  rawData: RawMarketData;
  features: EngineeredFeatures;
  
  // Agent signals
  agentSignals: AgentSignal[];
  
  // GA optimization
  gaGenome: SignalGenome | null;
  
  // Hyperbolic embeddings
  signalEmbeddings: Map<string, HyperbolicPoint> | null;
  regimeEmbedding: HyperbolicPoint | null;
  
  // Market regime
  regimeState: RegimeState;
  
  // Meta-model prediction
  metaModelOutput: MetaModelOutput | null;
  
  // Strategy signals
  strategySignals: StrategySignal[];
  
  // Portfolio metrics
  portfolioMetrics: PortfolioMetrics | null;
  riskConstraints: RiskConstraint[];
  
  // Metadata
  timestamp: Date;
  latencyMs: number;
}

export class MLOrchestrator {
  private config: MLOrchestratorConfig;
  
  // Components
  private featureEngineer: FeatureEngineer;
  private agents: {
    economic: EconomicAgent;
    sentiment: SentimentAgent;
    crossExchange: CrossExchangeAgent;
    onChain: OnChainAgent;
    cnnPattern: CNNPatternAgent;
  };
  private gaSelector: GeneticAlgorithmSignalSelector | null;
  private hyperbolicEmbedding: HyperbolicEmbedding | null;
  private regimeDetector: MarketRegimeDetector;
  private metaModel: XGBoostMetaModel | null;
  private strategies: RegimeConditionalStrategies | null;
  private riskManager: PortfolioRiskManager | null;
  
  // State
  private lastGAOptimization: Date | null;
  private gaOptimizationInterval: number; // ms
  
  constructor(config: Partial<MLOrchestratorConfig> = {}) {
    this.config = {
      enableGA: config.enableGA !== false,
      enableHyperbolic: config.enableHyperbolic !== false,
      enableXGBoost: config.enableXGBoost !== false,
      enableStrategies: config.enableStrategies !== false,
      enableRiskManager: config.enableRiskManager !== false,
      gaGenerations: config.gaGenerations || 50,
      hyperbolicDimension: config.hyperbolicDimension || 5,
      totalCapital: config.totalCapital || 100000,
    };
    
    // Initialize components
    this.featureEngineer = new FeatureEngineer();
    
    this.agents = {
      economic: new EconomicAgent(),
      sentiment: new SentimentAgent(),
      crossExchange: new CrossExchangeAgent(),
      onChain: new OnChainAgent(),
      cnnPattern: new CNNPatternAgent(),
    };
    
    this.gaSelector = this.config.enableGA
      ? new GeneticAlgorithmSignalSelector({ maxGenerations: this.config.gaGenerations })
      : null;
    
    this.hyperbolicEmbedding = this.config.enableHyperbolic
      ? new HyperbolicEmbedding({ dimension: this.config.hyperbolicDimension })
      : null;
    
    this.regimeDetector = new MarketRegimeDetector();
    
    this.metaModel = this.config.enableXGBoost
      ? new XGBoostMetaModel()
      : null;
    
    this.strategies = this.config.enableStrategies
      ? new RegimeConditionalStrategies()
      : null;
    
    this.riskManager = this.config.enableRiskManager
      ? new PortfolioRiskManager({ totalCapital: this.config.totalCapital })
      : null;
    
    this.lastGAOptimization = null;
    this.gaOptimizationInterval = 3600000; // 1 hour
  }
  
  /**
   * Run full ML pipeline
   */
  async runPipeline(rawData: RawMarketData): Promise<MLPipelineOutput> {
    const startTime = Date.now();
    
    // Step 1: Feature Engineering
    const features = this.featureEngineer.engineer(rawData);
    
    // Step 2: Generate agent signals
    const agentSignals = await this.generateAgentSignals(rawData, features);
    
    // Step 3: Genetic Algorithm (run periodically)
    const gaGenome = await this.runGAOptimization(agentSignals);
    
    // Step 4: Hyperbolic Embedding
    const { signalEmbeddings, regimeEmbedding } = await this.computeHyperbolicEmbeddings(
      agentSignals,
      features
    );
    
    // Step 5: Market Regime Detection
    const regimeState = this.detectMarketRegime(features, rawData);
    
    // Step 6: XGBoost Meta-Model
    const metaModelOutput = this.metaModel && gaGenome && signalEmbeddings && regimeEmbedding
      ? this.metaModel.predict({
          gaGenome,
          agentSignals,
          signalEmbeddings,
          regimeEmbedding,
          regimeState,
          volatility: features.volatility.realized24h,
          liquidity: rawData.liquidity / 10000000 * 100, // Normalize
          spread: features.spreads.bidAsk,
        })
      : null;
    
    // Step 7: Regime-Conditional Strategies
    const strategySignals = this.strategies && metaModelOutput && gaGenome
      ? this.strategies.evaluateStrategies(
          regimeState,
          metaModelOutput,
          gaGenome,
          agentSignals,
          {
            spotPrice: rawData.spotPrice,
            perpPrice: rawData.perpPrice,
            fundingRate: rawData.fundingRate,
            volatility: features.volatility.realized24h,
            priceVsSMA: features.relative.priceVsSMA,
            sma20: features.rolling.sma20,
          }
        )
      : [];
    
    // Step 8: Portfolio & Risk Management
    let portfolioMetrics: PortfolioMetrics | null = null;
    let riskConstraints: RiskConstraint[] = [];
    
    if (this.riskManager) {
      // Update positions from strategy signals
      for (const signal of strategySignals) {
        if (signal.action === 'ENTER' && signal.trade) {
          const existingTrades = this.strategies?.getTradesByStrategy(signal.strategy) || [];
          this.riskManager.updatePosition(signal.strategy, [...existingTrades, signal.trade]);
        }
      }
      
      portfolioMetrics = this.riskManager.calculateMetrics();
      riskConstraints = this.riskManager.checkRiskConstraints();
      this.riskManager.updateCapitalHistory();
    }
    
    return {
      rawData,
      features,
      agentSignals,
      gaGenome,
      signalEmbeddings,
      regimeEmbedding,
      regimeState,
      metaModelOutput,
      strategySignals,
      portfolioMetrics,
      riskConstraints,
      timestamp: new Date(),
      latencyMs: Date.now() - startTime,
    };
  }
  
  /**
   * Generate signals from all agents
   */
  private async generateAgentSignals(
    rawData: RawMarketData,
    features: EngineeredFeatures
  ): Promise<AgentSignal[]> {
    const marketData = {
      // Economic data
      fedRate: 4.25 + Math.random() * 0.5,
      cpi: 3.2 + Math.random() * 0.3,
      gdp: 2.8 + Math.random() * 0.4,
      vix: features.volatility.realized24h * 0.8,
      liquidityScore: Math.min(100, rawData.liquidity / 100000),
      
      // Sentiment data
      fearGreed: 50 + (features.returns.log1h * 10),
      googleTrends: 50 + Math.random() * 20,
      socialSentiment: 50 + (features.returns.log24h || 0) * 5,
      volumeRatio: features.relative.volumeVsAvg,
      
      // Cross-exchange data
      binancePrice: rawData.spotPrice,
      coinbasePrice: rawData.spotPrice * (1 + Math.random() * 0.001),
      krakenPrice: rawData.spotPrice * (1 + Math.random() * 0.001),
      binanceLiquidity: rawData.liquidity * 0.6,
      coinbaseLiquidity: rawData.liquidity * 0.4,
      binanceCoinbaseSpread: features.spreads.crossExchange[0] || 15,
      spreadZScore: features.zScores.spreadZ,
      
      // On-chain data
      exchangeNetflow: -5000 + Math.random() * 10000,
      whaleTransactions: 50 + Math.random() * 30,
      sopr: 1.02 + Math.random() * 0.1 - 0.05,
      mvrv: 1.8 + Math.random() * 0.4 - 0.2,
      
      // CNN pattern data
      pattern: this.selectRandomPattern(),
      patternConfidence: 0.65 + Math.random() * 0.25,
    };
    
    const signals = await Promise.all([
      this.agents.economic.generateSignal(marketData),
      this.agents.sentiment.generateSignal(marketData),
      this.agents.crossExchange.generateSignal(marketData),
      this.agents.onChain.generateSignal(marketData),
      this.agents.cnnPattern.generateSignal(marketData),
    ]);
    
    return signals;
  }
  
  /**
   * Run GA optimization (periodically)
   */
  private async runGAOptimization(agentSignals: AgentSignal[]): Promise<SignalGenome | null> {
    if (!this.gaSelector) return null;
    
    const now = new Date();
    const shouldOptimize = !this.lastGAOptimization || 
      (now.getTime() - this.lastGAOptimization.getTime()) > this.gaOptimizationInterval;
    
    if (!shouldOptimize) {
      return this.gaSelector.getBestGenome();
    }
    
    // Simplified fitness evaluator (use agent performance as proxy)
    const fitnessEvaluator = (genome: SignalGenome): BacktestResult => {
      // Weighted signal performance
      let returns = 0;
      for (let i = 0; i < genome.weights.length && i < agentSignals.length; i++) {
        returns += genome.weights[i] * agentSignals[i].signal * agentSignals[i].confidence;
      }
      
      return {
        returns: [returns],
        sharpe: Math.abs(returns) * 2,
        maxDrawdown: Math.abs(returns) * 0.3,
        turnover: 10,
        totalTrades: 5,
      };
    };
    
    // Correlation matrix (simplified)
    const correlationMatrix = this.computeSignalCorrelations(agentSignals);
    
    // Run GA (this is computationally expensive, so we cache it)
    console.log('[ML Orchestrator] Running GA optimization...');
    const bestGenome = this.gaSelector.run(fitnessEvaluator, correlationMatrix, agentSignals.length);
    
    this.lastGAOptimization = now;
    return bestGenome;
  }
  
  /**
   * Compute hyperbolic embeddings
   */
  private async computeHyperbolicEmbeddings(
    agentSignals: AgentSignal[],
    features: EngineeredFeatures
  ): Promise<{ signalEmbeddings: Map<string, HyperbolicPoint> | null; regimeEmbedding: HyperbolicPoint | null }> {
    if (!this.hyperbolicEmbedding) {
      return { signalEmbeddings: null, regimeEmbedding: null };
    }
    
    // Build hierarchical graph
    const graph = this.buildHierarchicalGraph(agentSignals);
    
    // Embed graph
    const embeddings = this.hyperbolicEmbedding.embed(graph);
    
    // Get regime centroid
    const regimeEmbedding = this.hyperbolicEmbedding.getRegimeCentroid('neutral') || {
      coords: [0, 0, 0, 0, 0],
      norm: 0,
      id: 'regime_neutral',
      type: 'regime' as const,
    };
    
    return { signalEmbeddings: embeddings, regimeEmbedding };
  }
  
  /**
   * Detect market regime
   */
  private detectMarketRegime(features: EngineeredFeatures, rawData: RawMarketData): RegimeState {
    const regimeFeatures: RegimeFeatures = {
      volatility: features.volatility.realized24h,
      returns: features.returns.log1h,
      sentiment: 50 + (features.returns.log1h * 10), // Approximation
      volume: features.relative.volumeVsAvg,
      liquidity: Math.min(100, rawData.liquidity / 100000),
      spread: features.spreads.bidAsk,
    };
    
    return this.regimeDetector.detectRegime(regimeFeatures);
  }
  
  // Helper functions
  
  private selectRandomPattern(): string {
    const patterns = ['Bull Flag', 'Bear Flag', 'Cup & Handle', 'Double Top', 'Double Bottom', 'Head & Shoulders'];
    return patterns[Math.floor(Math.random() * patterns.length)];
  }
  
  private computeSignalCorrelations(signals: AgentSignal[]): number[][] {
    const n = signals.length;
    const matrix: number[][] = Array(n).fill(0).map(() => Array(n).fill(0));
    
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) {
          matrix[i][j] = 1.0;
        } else {
          // Simplified: correlation based on signal similarity
          const corr = 1 - Math.abs(signals[i].signal - signals[j].signal) / 2;
          matrix[i][j] = corr;
        }
      }
    }
    
    return matrix;
  }
  
  private buildHierarchicalGraph(signals: AgentSignal[]): HierarchicalGraph {
    const nodes = new Map<string, HierarchicalNode>();
    const edges = new Map<string, Set<string>>();
    
    // Add regime nodes
    const regimes = ['crisis_stress', 'defensive', 'neutral', 'risk_on', 'high_conviction'];
    for (const regime of regimes) {
      nodes.set(regime, {
        id: regime,
        type: 'regime',
        children: [],
        parent: null,
        features: [],
      });
      edges.set(regime, new Set());
    }
    
    // Add signal nodes
    for (const signal of signals) {
      const signalId = `signal_${signal.agentId}`;
      const parentRegime = 'neutral'; // Simplified: all signals belong to neutral regime
      
      nodes.set(signalId, {
        id: signalId,
        type: 'signal',
        children: [],
        parent: parentRegime,
        features: Object.values(signal.features),
      });
      
      edges.set(signalId, new Set([parentRegime]));
      
      // Update parent
      const parent = nodes.get(parentRegime);
      if (parent) {
        parent.children.push(signalId);
      }
    }
    
    return { nodes, edges };
  }
  
  /**
   * Get components (for external access)
   */
  getComponents() {
    return {
      featureEngineer: this.featureEngineer,
      agents: this.agents,
      gaSelector: this.gaSelector,
      hyperbolicEmbedding: this.hyperbolicEmbedding,
      regimeDetector: this.regimeDetector,
      metaModel: this.metaModel,
      strategies: this.strategies,
      riskManager: this.riskManager,
    };
  }
}

/**
 * Example usage:
 * 
 * const orchestrator = new MLOrchestrator({
 *   enableGA: true,
 *   enableHyperbolic: true,
 *   enableXGBoost: true,
 *   enableStrategies: true,
 *   enableRiskManager: true,
 *   totalCapital: 100000,
 * });
 * 
 * const rawData: RawMarketData = {
 *   timestamp: new Date(),
 *   symbol: 'BTC-USD',
 *   spotPrice: 96500,
 *   perpPrice: 96530,
 *   bidPrice: 96495,
 *   askPrice: 96505,
 *   exchangePrices: { binance: 96500, coinbase: 96530 },
 *   volume24h: 1000000,
 *   bidVolume: 500000,
 *   askVolume: 500000,
 *   liquidity: 5000000,
 * };
 * 
 * const output = await orchestrator.runPipeline(rawData);
 * console.log('ML Pipeline Output:', output);
 */
