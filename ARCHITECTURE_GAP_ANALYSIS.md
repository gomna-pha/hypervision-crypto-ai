# 🏗️ Architecture Implementation Gap Analysis

## Current vs. Required Architecture

### ✅ **What We Have (Correct)**

1. **Real-Time Market Data Feeds** ✅
   - `src/data/realtime-data-feeds-node.ts`
   - WebSocket connections: Binance, Coinbase, Kraken
   - Spot/Perp prices, funding rates, order book data

2. **Feature Engineering** ✅
   - `src/ml/feature-engineering.ts`
   - Returns, spreads, volatility, flow, z-scores
   - Rolling windows implemented

3. **5 AI Agents** ✅
   - `src/ml/agent-signal.ts`
   - Economic, Sentiment, Cross-Exchange, On-Chain, CNN Pattern

4. **Genetic Algorithm** ✅
   - `src/ml/genetic-algorithm.ts`
   - Signal selection and weight evolution

5. **Hyperbolic Embedding** ✅
   - `src/ml/hyperbolic-embedding.ts`
   - Poincaré ball implementation

6. **Market Regime Detection** ✅
   - `src/ml/market-regime-detection.ts`
   - 5 regimes: Crisis/Stress, Defensive, Neutral, Risk-On, High Conviction

7. **XGBoost Meta-Model** ✅
   - `src/ml/xgboost-meta-model.ts`
   - Confidence scoring, exposure scaling

8. **Regime-Conditional Strategies** ✅
   - `src/ml/regime-conditional-strategies.ts`
   - Cross-exchange spread, funding carry, basis trades

9. **Portfolio Risk Manager** ✅
   - `src/ml/portfolio-risk-manager.ts`
   - Volatility targeting, drawdown limits

10. **ML Orchestrator** ✅
    - `src/ml/ml-orchestrator.ts`
    - Central integration hub

---

### ❌ **What's Missing/Incorrect**

#### 1. **Feature Store (Versioned, Time-Stamped)** ❌
**Architecture Says:**
> "Versioned Feature Store (Time-Stamped, Drift-Aware)"

**Current Issue:**
- Features computed on-the-fly, no persistence
- No versioning or drift detection
- No time-series storage

**Needs:**
```typescript
interface FeatureStore {
  // Store versioned features
  storeFeatures(timestamp: Date, features: EngineeredFeatures, version: string): Promise<void>;
  
  // Retrieve historical features
  getFeatures(timestamp: Date, version?: string): Promise<EngineeredFeatures>;
  
  // Detect feature drift
  detectDrift(current: EngineeredFeatures, baseline: EngineeredFeatures): DriftReport;
  
  // Feature stability metrics
  getStabilityMetrics(featureName: string, window: number): StabilityMetrics;
}
```

**Implementation:**
- InfluxDB for time-series storage
- Redis for real-time caching
- Version tagging for features

---

#### 2. **Multi-Agent Signal Pool** ❌
**Architecture Says:**
> "Weak, Diverse Signals s₁…s₅(t) ∈ {−1, 0, +1}"

**Current Issue:**
- Agents produce signals but no "pool" aggregation
- Missing signal normalization to {-1, 0, +1}
- No weak signal handling

**Needs:**
```typescript
interface SignalPool {
  // Aggregate signals from all agents
  aggregateSignals(agentSignals: AgentSignal[]): NormalizedSignal[];
  
  // Normalize to {-1, 0, +1}
  normalizeSignal(signal: AgentSignal): NormalizedSignal;
  
  // Track signal diversity
  calculateDiversity(signals: NormalizedSignal[]): number;
  
  // Signal correlation matrix
  getCorrelationMatrix(): number[][];
}

interface NormalizedSignal {
  agentId: string;
  direction: -1 | 0 | 1;  // Short | Neutral | Long
  strength: number;        // [0, 1]
  timestamp: Date;
}
```

---

#### 3. **Hierarchical Signal-Regime Graph** ❌
**Architecture Says:**
> "Nodes: Signals • Regimes • Arbitrage Strategies  
> Edges: Conditional Dependence • Survival Probability • Regime Transitions • Strategy Compatibility"

**Current Issue:**
- Hyperbolic embedding exists but no graph structure
- Missing signal-to-regime edges
- Missing regime transition edges
- Missing strategy compatibility edges

**Needs:**
```typescript
interface HierarchicalGraph {
  // Nodes
  signalNodes: SignalNode[];
  regimeNodes: RegimeNode[];
  strategyNodes: StrategyNode[];
  
  // Edges
  signalRegimeEdges: Edge[];      // Conditional dependence
  regimeTransitionEdges: Edge[];  // Regime transitions
  strategyCompatEdges: Edge[];    // Strategy compatibility
  
  // Build graph from current state
  buildGraph(
    signals: NormalizedSignal[],
    currentRegime: MarketRegime,
    strategies: Strategy[]
  ): void;
  
  // Query graph
  getRegimeForSignal(signalId: string): MarketRegime;
  getCompatibleStrategies(regime: MarketRegime): Strategy[];
  getTransitionProbability(from: MarketRegime, to: MarketRegime): number;
}

interface Edge {
  from: string;  // Node ID
  to: string;    // Node ID
  weight: number;
  type: 'conditional_dependence' | 'survival_prob' | 'regime_transition' | 'strategy_compat';
}
```

---

#### 4. **Hyperbolic Embedding FROM Graph** ❌
**Architecture Says:**
> "Embeds Hierarchical Graph  
> Radial Distance → Signal Robustness  
> Angular Distance → Regime Similarity  
> Curvature Preserves Tree-Like Structure"

**Current Issue:**
- Hyperbolic embedding computes distances but doesn't embed a graph
- No preservation of graph hierarchical structure
- No tree-like structure enforcement

**Needs:**
```typescript
class HyperbolicEmbedding {
  // Embed the hierarchical graph
  embedGraph(graph: HierarchicalGraph): Map<string, HyperbolicPoint>;
  
  // Preserve hierarchical structure
  preserveHierarchy(graph: HierarchicalGraph, embeddings: Map<string, HyperbolicPoint>): void;
  
  // Compute radial distance (signal robustness)
  getRadialDistance(point: HyperbolicPoint): number;
  
  // Compute angular distance (regime similarity)
  getAngularDistance(point1: HyperbolicPoint, point2: HyperbolicPoint): number;
  
  // Ensure tree-like curvature
  adjustCurvature(embeddings: Map<string, HyperbolicPoint>): void;
}
```

---

#### 5. **Market Regime FROM Hyperbolic Distances** ❌
**Architecture Says:**
> "Crisis / Stress | Defensive | Neutral | Risk-On | High Conviction  
> (Hyperbolic Distances + CNN Confirmation)"

**Current Issue:**
- Regime detection uses traditional features
- Not using hyperbolic distances
- CNN not used for confirmation

**Needs:**
```typescript
class MarketRegimeDetector {
  // Identify regime using hyperbolic distances
  identifyRegime(
    signalEmbeddings: Map<string, HyperbolicPoint>,
    regimeEmbedding: HyperbolicPoint,
    cnnConfidence: number
  ): MarketRegime;
  
  // Compute regime transition probability
  getTransitionProbability(
    currentRegime: MarketRegime,
    distances: Map<string, number>
  ): Map<MarketRegime, number>;
  
  // CNN confirmation layer
  confirmWithCNN(
    regimeCandidate: MarketRegime,
    cnnPattern: CNNPatternSignal
  ): { confirmed: boolean; confidence: number };
}
```

---

#### 6. **XGBoost Meta-Model Integration** ⚠️ **Partial**
**Architecture Says:**
> "Inputs: GA-selected signals, Hyperbolic distances, Regime transitions, Volatility & liquidity state  
> Outputs: Arbitrage confidence score, Signal disagreement flags, Dynamic exposure / leverage scaler  
> (No Direct Trade Generation)"

**Current Issue:**
- Meta-model exists but inputs incomplete
- Missing hyperbolic distances input
- Missing regime transition probabilities
- Outputs confidence but generates trades (should only scale exposure)

**Needs:**
```typescript
interface MetaModelInput {
  // GA-selected signals
  selectedSignals: SignalGenome;
  
  // Hyperbolic distances (NEW)
  signalDistances: Map<string, number>;
  regimeDistance: number;
  
  // Regime transition probabilities (NEW)
  regimeTransitions: Map<MarketRegime, number>;
  
  // Volatility & liquidity state
  volatility: {
    realized: number;
    implied: number;
    forecast: number;
  };
  liquidity: {
    bidAskSpread: number;
    depth: number;
    turnover: number;
  };
  
  // Market regime
  currentRegime: MarketRegime;
  
  // Features
  features: EngineeredFeatures;
}

interface MetaModelOutput {
  // Arbitrage confidence score [0, 1]
  confidenceScore: number;
  
  // Signal disagreement flags (NEW)
  disagreementFlags: {
    hasDisagreement: boolean;
    conflictingSignals: string[];
    agreementScore: number;
  };
  
  // Dynamic exposure scaler [0, 2]
  exposureScaler: number;
  
  // Leverage scaler [0, 3]
  leverageScaler: number;
  
  // Risk flags
  riskFlags: string[];
  
  // NO TRADES - only scaling factors
}
```

---

#### 7. **Portfolio Construction with Risk Aversion γ** ⚠️ **Partial**
**Architecture Says:**
> "Risk-Aversion Parameter γ (Conservative → Aggressive)  
> Dynamic Strategy Weighting"

**Current Issue:**
- Risk manager exists but no explicit γ parameter
- No sensitivity analysis to risk aversion
- Strategy weighting not dynamic based on γ

**Needs:**
```typescript
interface PortfolioConstructor {
  // Risk aversion parameter [1, 10]
  riskAversion: number;  // γ
  
  // Construct portfolio based on γ
  constructPortfolio(
    strategySignals: StrategySignal[],
    metaModelOutput: MetaModelOutput,
    riskAversion: number
  ): Portfolio;
  
  // Dynamic strategy weighting
  weightStrategies(
    strategies: Strategy[],
    regime: MarketRegime,
    riskAversion: number
  ): Map<string, number>;
  
  // Volatility targeting based on γ
  targetVolatility(riskAversion: number): number;
  
  // Position sizing based on γ
  sizePosition(
    signal: StrategySignal,
    confidence: number,
    riskAversion: number
  ): number;
}
```

---

#### 8. **Backtesting with Regime-Specific Analysis** ❌
**Architecture Says:**
> "Regime-Specific Arbitrage Backtests  
> Euclidean vs Hyperbolic Ablation Tests  
> Meta-Model On/Off Evaluation  
> Transaction Cost Sensitivity  
> Risk-Aversion Sensitivity (γ Sweeps)"

**Current Issue:**
- No backtesting framework
- No regime-specific performance tracking
- No γ sensitivity analysis

**Needs:**
```typescript
interface BacktestEngine {
  // Run backtest
  runBacktest(
    historicalData: MarketData[],
    config: BacktestConfig
  ): BacktestResult;
  
  // Regime-specific Sharpe
  calculateRegimeSharpebySharpe(
    trades: Trade[],
    regime: MarketRegime
  ): Map<MarketRegime, number>;
  
  // γ sensitivity sweep
  runRiskAversionSensitivity(
    gammaRange: number[],
    historicalData: MarketData[]
  ): Map<number, BacktestResult>;
  
  // Hyperbolic vs Euclidean ablation
  compareEmbeddingMethods(
    historicalData: MarketData[]
  ): {
    hyperbolic: BacktestResult;
    euclidean: BacktestResult;
    improvement: number;
  };
  
  // Meta-model on/off comparison
  evaluateMetaModel(
    historicalData: MarketData[]
  ): {
    withMetaModel: BacktestResult;
    withoutMetaModel: BacktestResult;
    improvement: number;
  };
}
```

---

#### 9. **Weekly Observations Dashboard** ⚠️ **Partial**
**Architecture Says:**
> "Live PnL | Sharpe by Regime | Volatility Attribution  
> Hyperbolic Maps | Feature Drift | Strategy Decomposition"

**Current Issue:**
- Dashboard created but:
  - Hyperbolic map not connected to real data
  - No feature drift visualization
  - No strategy decomposition
  - Sharpe by regime not computed from real trades

**Needs:**
- Real-time updates from live pipeline
- Historical data storage for weekly aggregation
- Feature drift detection integration
- Strategy attribution breakdown

---

## 🎯 **Priority Fix Order**

### **Phase 1: Core Data Flow (Critical)**
1. ✅ Feature Store implementation (InfluxDB + versioning)
2. ✅ Multi-Agent Signal Pool with normalization
3. ✅ Hierarchical Signal-Regime Graph construction

### **Phase 2: Embeddings & Regime (High Priority)**
4. ✅ Graph-based Hyperbolic Embedding
5. ✅ Regime Detection using Hyperbolic Distances
6. ✅ XGBoost Meta-Model full integration

### **Phase 3: Portfolio & Risk (High Priority)**
7. ✅ Risk Aversion γ parameter implementation
8. ✅ Dynamic strategy weighting based on γ
9. ✅ Portfolio construction with γ sensitivity

### **Phase 4: Analysis & Visualization (Medium Priority)**
10. ✅ Backtesting framework with regime-specific Sharpe
11. ✅ γ sensitivity sweep analysis
12. ✅ Dashboard integration with real pipeline data

---

## 🔧 **Immediate Action Items**

### **1. Fix ML Orchestrator Flow**
Current: `Market Data → Features → Agents → GA → Hyperbolic → Regime → XGBoost → Strategies → Portfolio`

Should be: `Market Data → Feature Store → Agents → Signal Pool → GA → Graph → Hyperbolic Embedding → Regime (using distances) → XGBoost (using distances) → Strategies → Portfolio (with γ)`

### **2. Create Missing Components**
- `src/ml/feature-store.ts`
- `src/ml/signal-pool.ts`
- `src/ml/hierarchical-graph.ts`
- `src/ml/backtest-engine.ts`

### **3. Update Existing Components**
- `src/ml/hyperbolic-embedding.ts` - Add graph embedding
- `src/ml/market-regime-detection.ts` - Use hyperbolic distances
- `src/ml/xgboost-meta-model.ts` - Add distance inputs
- `src/ml/portfolio-risk-manager.ts` - Add γ parameter

### **4. Connect Dashboard**
- Real hyperbolic visualization from embeddings
- Feature drift alerts
- Regime-specific performance
- γ sensitivity charts

---

## 📊 **Architecture Compliance Checklist**

| Component | Implemented | Correct Flow | Complete Integration |
|-----------|------------|--------------|---------------------|
| Market Data Feeds | ✅ | ✅ | ✅ |
| Feature Engineering | ✅ | ✅ | ⚠️ (no store) |
| Feature Store | ❌ | ❌ | ❌ |
| 5 AI Agents | ✅ | ⚠️ (should use feature store) | ⚠️ |
| Signal Pool | ❌ | ❌ | ❌ |
| Genetic Algorithm | ✅ | ⚠️ (should use signal pool) | ⚠️ |
| Hierarchical Graph | ❌ | ❌ | ❌ |
| Hyperbolic Embedding | ✅ | ❌ (not from graph) | ❌ |
| Regime Detection | ✅ | ❌ (not using distances) | ⚠️ |
| XGBoost Meta-Model | ✅ | ⚠️ (missing distance inputs) | ⚠️ |
| Strategies | ✅ | ✅ | ⚠️ |
| Portfolio (with γ) | ⚠️ | ⚠️ (no explicit γ) | ❌ |
| Execution | ❌ | ❌ | ❌ |
| Monitoring | ⚠️ | ⚠️ | ⚠️ |
| Backtesting | ❌ | ❌ | ❌ |

**Current Compliance: ~45%**  
**Target: 100%**

---

## 🚀 **Estimated Work**

- **Phase 1:** 16-20 hours
- **Phase 2:** 12-16 hours
- **Phase 3:** 12-16 hours
- **Phase 4:** 16-20 hours

**Total: 56-72 hours (7-9 full days)**

---

**Created:** December 20, 2025, 3:45 AM UTC  
**Status:** Gap analysis complete, ready to begin implementation
