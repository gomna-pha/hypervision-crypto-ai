# 🔧 Integration Guide - ML Modules into Existing Platform

**Date**: 2025-12-19  
**Target Platform**: ArbitrageAI v5.3.0 → v6.0.0  
**Integration Difficulty**: Medium (requires TypeScript knowledge)

---

## 📋 Overview

This guide shows you how to integrate the new ML modules (Genetic Algorithm, Hyperbolic Embedding, Enhanced Agents) into your existing Hono backend (`src/index.tsx`).

---

## 🎯 Integration Steps

### **Step 1: Install Dependencies**

First, install required npm packages:

```bash
cd /home/user/webapp

# For numerical computations
npm install --save mathjs

# For machine learning (if using XGBoost in future)
npm install --save @tensorflow/tfjs-node

# For WebSocket (real-time data feeds)
npm install --save ws

# For database (feature store)
npm install --save @influxdata/influxdb-client

# Development dependencies
npm install --save-dev @types/ws
```

---

### **Step 2: Import ML Modules into Backend**

Update `src/index.tsx` to import the new modules:

```typescript
// Add these imports at the top of src/index.tsx
import { 
  GeneticAlgorithmSignalSelector,
  type SignalGenome,
  type BacktestResult 
} from './ml/genetic-algorithm';

import { 
  HyperbolicEmbedding,
  type HyperbolicPoint,
  type HierarchicalGraph,
  type HierarchicalNode 
} from './ml/hyperbolic-embedding';

import {
  type AgentSignal,
  EconomicAgent,
  SentimentAgent,
  CrossExchangeAgent,
  OnChainAgent,
  CNNPatternAgent
} from './ml/agent-signal';
```

---

### **Step 3: Initialize Agents in Backend**

Replace your current agent generation with the new standardized agents:

```typescript
// In src/index.tsx, add global instances
const agents = {
  economic: new EconomicAgent(),
  sentiment: new SentimentAgent(),
  crossExchange: new CrossExchangeAgent(),
  onChain: new OnChainAgent(),
  cnnPattern: new CNNPatternAgent()
};

// Initialize GA (for weekly signal optimization)
const gaOptimizer = new GeneticAlgorithmSignalSelector({
  populationSize: 100,
  maxGenerations: 50,
  mutationRate: 0.05,
  crossoverRate: 0.8,
  eliteRatio: 0.1
});

// Initialize Hyperbolic Embedding (for signal-regime graph)
const hyperbolicEmbedding = new HyperbolicEmbedding({
  dimension: 5,
  curvature: 1.0,
  maxIterations: 1000
});
```

---

### **Step 4: Update `/api/agents` Endpoint**

Replace the current agent data generation with real agent signals:

```typescript
// OLD CODE (remove this)
app.get('/api/agents', async (c) => {
  const agents = generateAgentData(); // Simulated data
  return c.json(agents);
});

// NEW CODE (use this)
app.get('/api/agents', async (c) => {
  try {
    // Fetch market data (currently simulated, will be real in Phase 1)
    const marketData = await fetchMarketData();
    
    // Generate signals from all agents
    const economicSignal = await agents.economic.generateSignal({
      fedRate: marketData.fedRate,
      cpi: marketData.cpi,
      gdp: marketData.gdp,
      vix: marketData.vix,
      liquidityScore: marketData.liquidityScore
    });

    const sentimentSignal = await agents.sentiment.generateSignal({
      fearGreed: marketData.fearGreed,
      googleTrends: marketData.googleTrends,
      socialSentiment: marketData.socialSentiment,
      volumeRatio: marketData.volumeRatio
    });

    const crossExchangeSignal = await agents.crossExchange.generateSignal({
      binancePrice: marketData.binancePrice,
      coinbasePrice: marketData.coinbasePrice,
      krakenPrice: marketData.krakenPrice,
      binanceLiquidity: marketData.binanceLiquidity,
      coinbaseLiquidity: marketData.coinbaseLiquidity
    });

    const onChainSignal = await agents.onChain.generateSignal({
      exchangeNetflow: marketData.exchangeNetflow,
      whaleTransactions: marketData.whaleTransactions,
      sopr: marketData.sopr,
      mvrv: marketData.mvrv
    });

    const cnnPatternSignal = await agents.cnnPattern.generateSignal({
      pattern: marketData.pattern,
      patternConfidence: marketData.patternConfidence,
      fearGreed: marketData.fearGreed
    });

    // Calculate composite signal (weighted average)
    const compositeSignal = calculateCompositeSignal([
      { signal: economicSignal, weight: 0.10 },
      { signal: sentimentSignal, weight: 0.20 },
      { signal: crossExchangeSignal, weight: 0.35 },
      { signal: onChainSignal, weight: 0.05 },
      { signal: cnnPatternSignal, weight: 0.30 }
    ]);

    // Return agent data in expected format
    return c.json({
      economic: {
        score: (economicSignal.signal + 1) * 50, // Convert [-1,1] to [0,100]
        signal: economicSignal.signal,
        confidence: economicSignal.confidence,
        features: economicSignal.features,
        explanation: economicSignal.explanation,
        expectedAlpha: economicSignal.expectedAlpha,
        riskScore: economicSignal.riskScore,
        latencyMs: economicSignal.latencyMs
      },
      sentiment: {
        score: (sentimentSignal.signal + 1) * 50,
        signal: sentimentSignal.signal,
        confidence: sentimentSignal.confidence,
        features: sentimentSignal.features,
        explanation: sentimentSignal.explanation,
        expectedAlpha: sentimentSignal.expectedAlpha,
        riskScore: sentimentSignal.riskScore,
        latencyMs: sentimentSignal.latencyMs
      },
      crossExchange: {
        score: (crossExchangeSignal.signal + 1) * 50,
        signal: crossExchangeSignal.signal,
        confidence: crossExchangeSignal.confidence,
        features: crossExchangeSignal.features,
        explanation: crossExchangeSignal.explanation,
        expectedAlpha: crossExchangeSignal.expectedAlpha,
        riskScore: crossExchangeSignal.riskScore,
        latencyMs: crossExchangeSignal.latencyMs
      },
      onChain: {
        score: (onChainSignal.signal + 1) * 50,
        signal: onChainSignal.signal,
        confidence: onChainSignal.confidence,
        features: onChainSignal.features,
        explanation: onChainSignal.explanation,
        expectedAlpha: onChainSignal.expectedAlpha,
        riskScore: onChainSignal.riskScore,
        latencyMs: onChainSignal.latencyMs
      },
      cnnPattern: {
        score: (cnnPatternSignal.signal + 1) * 50,
        signal: cnnPatternSignal.signal,
        confidence: cnnPatternSignal.confidence,
        features: cnnPatternSignal.features,
        explanation: cnnPatternSignal.explanation,
        expectedAlpha: cnnPatternSignal.expectedAlpha,
        riskScore: cnnPatternSignal.riskScore,
        latencyMs: cnnPatternSignal.latencyMs
      },
      composite: {
        score: compositeSignal.score,
        signal: compositeSignal.signal,
        confidence: compositeSignal.confidence,
        explanation: compositeSignal.explanation
      }
    });
  } catch (error) {
    console.error('Error generating agent signals:', error);
    return c.json({ error: 'Failed to generate agent signals' }, 500);
  }
});

// Helper function: Calculate composite signal
function calculateCompositeSignal(weightedSignals: Array<{ signal: AgentSignal; weight: number }>) {
  const totalWeight = weightedSignals.reduce((sum, ws) => sum + ws.weight, 0);
  
  // Weighted average of signals
  const compositeSignal = weightedSignals.reduce(
    (sum, ws) => sum + ws.signal.signal * ws.weight,
    0
  ) / totalWeight;

  // Weighted average of confidence
  const compositeConfidence = weightedSignals.reduce(
    (sum, ws) => sum + ws.signal.confidence * ws.weight,
    0
  ) / totalWeight;

  // Calculate signal strength
  let signalLabel: string;
  const score = (compositeSignal + 1) * 50;
  if (score > 70) signalLabel = 'STRONG_BUY';
  else if (score > 56) signalLabel = 'BUY';
  else if (score > 44) signalLabel = 'NEUTRAL';
  else if (score > 30) signalLabel = 'SELL';
  else signalLabel = 'STRONG_SELL';

  return {
    score,
    signal: compositeSignal,
    confidence: compositeConfidence,
    label: signalLabel,
    explanation: `Composite signal from ${weightedSignals.length} agents: ${signalLabel}`
  };
}
```

---

### **Step 5: Add GA Optimization Endpoint**

Create a new endpoint for weekly signal optimization:

```typescript
// Add new endpoint for GA signal optimization
app.post('/api/signals/optimize', async (c) => {
  try {
    // Load historical signals (last 90 days)
    const historicalSignals = await loadHistoricalSignals(90);
    
    // Compute correlation matrix
    const correlationMatrix = computeCorrelationMatrix(historicalSignals);
    
    // Fitness evaluator function
    const fitnessEvaluator = (genome: SignalGenome): BacktestResult => {
      // Backtest genome on historical data
      return backtestGenome(genome, historicalSignals);
    };
    
    // Run GA optimization
    const startTime = Date.now();
    const bestGenome = gaOptimizer.run(fitnessEvaluator, correlationMatrix, 5);
    const optimizationTime = Date.now() - startTime;
    
    // Store optimal genome for future use
    await storeOptimalGenome(bestGenome);
    
    return c.json({
      success: true,
      bestGenome: {
        id: bestGenome.id,
        activeSignals: bestGenome.activeSignals,
        weights: bestGenome.weights,
        fitness: bestGenome.fitness,
        generation: bestGenome.generation
      },
      optimizationTimeMs: optimizationTime,
      message: 'Signal weights optimized successfully'
    });
  } catch (error) {
    console.error('GA optimization error:', error);
    return c.json({ error: 'Optimization failed' }, 500);
  }
});

// Helper: Backtest genome
function backtestGenome(genome: SignalGenome, historicalData: any): BacktestResult {
  // Simulate strategy with genome's signal selection
  const trades = [];
  let equity = 100000; // $100k starting capital
  
  for (const day of historicalData) {
    // Calculate weighted signal using genome
    const weightedSignal = genome.activeSignals.reduce((sum, active, idx) => {
      return sum + (active ? genome.weights[idx] * day.agentSignals[idx] : 0);
    }, 0);
    
    // Trade if signal strong enough
    if (Math.abs(weightedSignal) > 0.5) {
      const profitPct = weightedSignal * day.marketReturn; // Simplified
      equity *= (1 + profitPct);
      trades.push(profitPct);
    }
  }
  
  // Calculate metrics
  const returns = trades;
  const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
  const stdReturn = Math.sqrt(
    returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length
  );
  const sharpe = (avgReturn / stdReturn) * Math.sqrt(252); // Annualized
  
  const maxDrawdown = calculateMaxDrawdown(returns);
  const turnover = trades.length;
  
  return {
    returns,
    sharpe,
    maxDrawdown,
    turnover,
    totalTrades: trades.length
  };
}

// Helper: Calculate max drawdown
function calculateMaxDrawdown(returns: number[]): number {
  let peak = 0;
  let maxDD = 0;
  let cumReturn = 0;
  
  for (const ret of returns) {
    cumReturn += ret;
    peak = Math.max(peak, cumReturn);
    const drawdown = peak - cumReturn;
    maxDD = Math.max(maxDD, drawdown);
  }
  
  return maxDD;
}

// Helper: Compute correlation matrix
function computeCorrelationMatrix(historicalSignals: any): number[][] {
  const numAgents = 5;
  const matrix: number[][] = Array(numAgents).fill(null).map(() => Array(numAgents).fill(0));
  
  // Extract agent signal time series
  const agentTimeSeries = Array(numAgents).fill(null).map(() => []);
  for (const day of historicalSignals) {
    day.agentSignals.forEach((signal: number, idx: number) => {
      agentTimeSeries[idx].push(signal);
    });
  }
  
  // Compute pairwise correlations
  for (let i = 0; i < numAgents; i++) {
    for (let j = 0; j < numAgents; j++) {
      if (i === j) {
        matrix[i][j] = 1.0;
      } else {
        matrix[i][j] = pearsonCorrelation(agentTimeSeries[i], agentTimeSeries[j]);
      }
    }
  }
  
  return matrix;
}

// Helper: Pearson correlation
function pearsonCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  const sumX = x.reduce((sum, val) => sum + val, 0);
  const sumY = y.reduce((sum, val) => sum + val, 0);
  const sumXY = x.reduce((sum, val, idx) => sum + val * y[idx], 0);
  const sumX2 = x.reduce((sum, val) => sum + val * val, 0);
  const sumY2 = y.reduce((sum, val) => sum + val * val, 0);
  
  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));
  
  return numerator / denominator;
}
```

---

### **Step 6: Add Hyperbolic Embedding Endpoint**

Create endpoint for computing hyperbolic embeddings:

```typescript
// Add endpoint for hyperbolic embedding
app.post('/api/signals/embed', async (c) => {
  try {
    // Build signal-regime graph
    const graph = buildSignalRegimeGraph();
    
    // Embed into hyperbolic space
    const embeddings = hyperbolicEmbedding.embed(graph);
    
    // Extract insights
    const insights = {
      signalRobustness: {},
      regimeSimilarity: {},
      nearestNeighbors: {}
    };
    
    // Calculate signal robustness (radial distance)
    for (const [nodeId, point] of embeddings.entries()) {
      if (point.type === 'signal') {
        const robustness = hyperbolicEmbedding.radialDistance(point);
        insights.signalRobustness[nodeId] = {
          robustness,
          interpretation: robustness < 0.5 ? 'Robust (works in all regimes)' : 'Regime-specific'
        };
      }
    }
    
    // Calculate regime similarity (angular distance)
    const regimeNodes = Array.from(embeddings.values()).filter(p => p.type === 'regime');
    for (let i = 0; i < regimeNodes.length; i++) {
      for (let j = i + 1; j < regimeNodes.length; j++) {
        const similarity = hyperbolicEmbedding.angularDistance(regimeNodes[i], regimeNodes[j]);
        insights.regimeSimilarity[`${regimeNodes[i].id}_${regimeNodes[j].id}`] = {
          distance: similarity,
          interpretation: similarity < 1.0 ? 'Similar regimes' : 'Different regimes'
        };
      }
    }
    
    // Find nearest neighbors for each signal
    for (const [nodeId, point] of embeddings.entries()) {
      if (point.type === 'signal') {
        const neighbors = hyperbolicEmbedding.findNearestNeighbors(point, 3);
        insights.nearestNeighbors[nodeId] = neighbors.map(n => ({
          id: n.point.id,
          distance: n.distance,
          type: n.point.type
        }));
      }
    }
    
    return c.json({
      success: true,
      embeddings: Array.from(embeddings.entries()).map(([id, point]) => ({
        id,
        type: point.type,
        coords: point.coords,
        norm: point.norm
      })),
      insights
    });
  } catch (error) {
    console.error('Hyperbolic embedding error:', error);
    return c.json({ error: 'Embedding failed' }, 500);
  }
});

// Helper: Build signal-regime graph
function buildSignalRegimeGraph(): HierarchicalGraph {
  const nodes = new Map<string, HierarchicalNode>();
  const edges = new Map<string, Set<string>>();
  
  // Define regime nodes
  const regimes = ['crisis', 'stress', 'neutral', 'risk_on', 'high_conviction'];
  for (const regime of regimes) {
    nodes.set(`regime_${regime}`, {
      id: `regime_${regime}`,
      type: 'regime',
      children: [],
      parent: null,
      features: []
    });
    edges.set(`regime_${regime}`, new Set());
  }
  
  // Define signal nodes and their regime associations
  const signalRegimeMap = {
    'signal_economic': ['regime_stress', 'regime_neutral'],
    'signal_sentiment': ['regime_risk_on', 'regime_neutral'],
    'signal_cross_exchange': ['regime_crisis', 'regime_stress', 'regime_risk_on'],
    'signal_on_chain': ['regime_neutral', 'regime_high_conviction'],
    'signal_cnn_pattern': ['regime_high_conviction', 'regime_risk_on']
  };
  
  for (const [signalId, associatedRegimes] of Object.entries(signalRegimeMap)) {
    nodes.set(signalId, {
      id: signalId,
      type: 'signal',
      children: [],
      parent: associatedRegimes[0], // Primary regime
      features: []
    });
    
    edges.set(signalId, new Set(associatedRegimes));
    
    // Add signal as child of regimes
    for (const regimeId of associatedRegimes) {
      const regimeNode = nodes.get(regimeId);
      if (regimeNode) {
        regimeNode.children.push(signalId);
      }
    }
  }
  
  return { nodes, edges };
}
```

---

### **Step 7: Update Frontend to Display New Features**

Update `public/static/app.js` to show enhanced agent data:

```javascript
// In app.js, update fetchAgentData function
async function fetchAgentData() {
  try {
    const response = await axios.get('/api/agents');
    const data = response.data;
    
    // Update agent cards with new fields
    updateAgentCard('economic', {
      score: data.economic.score,
      signal: data.economic.signal,
      confidence: data.economic.confidence,
      expectedAlpha: data.economic.expectedAlpha,
      riskScore: data.economic.riskScore,
      latency: data.economic.latencyMs,
      explanation: data.economic.explanation
    });
    
    // Similar updates for other agents...
    
    // Display composite signal
    updateCompositeSignal({
      score: data.composite.score,
      signal: data.composite.signal,
      confidence: data.composite.confidence,
      label: data.composite.label,
      explanation: data.composite.explanation
    });
  } catch (error) {
    console.error('Error fetching agent data:', error);
  }
}

// Add new function to display agent details
function updateAgentCard(agentId, agentData) {
  const cardElement = document.getElementById(`agent-${agentId}`);
  if (!cardElement) return;
  
  cardElement.innerHTML = `
    <div class="agent-card">
      <h3>${agentId.toUpperCase()} AGENT</h3>
      <div class="agent-score">Score: ${agentData.score.toFixed(1)}</div>
      <div class="agent-signal">Signal: ${(agentData.signal * 100).toFixed(1)}%</div>
      <div class="agent-confidence">Confidence: ${(agentData.confidence * 100).toFixed(0)}%</div>
      <div class="agent-alpha">Expected Alpha: ${agentData.expectedAlpha.toFixed(1)} bps</div>
      <div class="agent-risk">Risk Score: ${(agentData.riskScore * 100).toFixed(0)}%</div>
      <div class="agent-latency">Latency: ${agentData.latency}ms</div>
      <div class="agent-explanation">${agentData.explanation}</div>
    </div>
  `;
}
```

---

## 🧪 Testing the Integration

### **Step 1: Test Agent Signals**

```bash
# Start development server
cd /home/user/webapp
npm run dev

# In another terminal, test agent endpoint
curl http://localhost:3000/api/agents | jq
```

Expected output:
```json
{
  "economic": {
    "score": 48.5,
    "signal": -0.03,
    "confidence": 0.75,
    "expectedAlpha": 8.2,
    "riskScore": 0.35,
    "latencyMs": 12
  },
  ...
}
```

### **Step 2: Test GA Optimization**

```bash
# Trigger GA optimization
curl -X POST http://localhost:3000/api/signals/optimize | jq
```

Expected output:
```json
{
  "success": true,
  "bestGenome": {
    "activeSignals": [1, 0, 1, 1, 0],
    "weights": [0.35, 0.0, 0.30, 0.25, 0.0],
    "fitness": 2.34
  },
  "optimizationTimeMs": 48523
}
```

### **Step 3: Test Hyperbolic Embedding**

```bash
# Compute embeddings
curl -X POST http://localhost:3000/api/signals/embed | jq
```

Expected output:
```json
{
  "success": true,
  "embeddings": [
    {
      "id": "signal_economic",
      "type": "signal",
      "coords": [0.12, -0.08, 0.15, -0.05, 0.20],
      "norm": 0.28
    }
  ],
  "insights": {
    "signalRobustness": {
      "signal_economic": {
        "robustness": 0.28,
        "interpretation": "Robust (works in all regimes)"
      }
    }
  }
}
```

---

## ⚠️ Important Notes

### **Data Dependencies**

The current implementation assumes you have a `fetchMarketData()` function. You'll need to implement this based on your data sources:

```typescript
async function fetchMarketData() {
  // TODO: Replace with real data feeds (Phase 1)
  return {
    // Economic data
    fedRate: 4.25,
    cpi: 3.2,
    gdp: 2.8,
    vix: 18,
    liquidityScore: 75,
    
    // Sentiment data
    fearGreed: 54,
    googleTrends: 65,
    socialSentiment: 58,
    volumeRatio: 1.2,
    
    // Cross-exchange data
    binancePrice: 96500,
    coinbasePrice: 96535,
    krakenPrice: 96520,
    binanceLiquidity: 1000000,
    coinbaseLiquidity: 800000,
    
    // On-chain data
    exchangeNetflow: -5521,
    whaleTransactions: 72,
    sopr: 1.02,
    mvrv: 1.8,
    
    // Pattern data
    pattern: 'Bull Flag',
    patternConfidence: 0.75
  };
}
```

### **Performance Considerations**

- **GA Optimization**: Run weekly, not on every request (takes ~50 seconds)
- **Hyperbolic Embedding**: Run daily or when graph structure changes
- **Agent Signals**: Can be generated on every request (< 50ms total)

---

## 📊 Next Steps

After integration:

1. **Test thoroughly** - Verify all endpoints work
2. **Add error handling** - Wrap everything in try-catch
3. **Implement caching** - Cache GA results for 7 days
4. **Add monitoring** - Track latency, errors, signal quality
5. **Phase 1 Week 3-4**: Implement real data feeds (WebSocket)

---

## 🆘 Troubleshooting

### **Issue**: TypeScript compilation errors

**Solution**: Ensure all imports are correct and types match:
```bash
npm run build
# Check for type errors
```

### **Issue**: GA optimization takes too long

**Solution**: Reduce population size or generations:
```typescript
const gaOptimizer = new GeneticAlgorithmSignalSelector({
  populationSize: 50,  // Reduced from 100
  maxGenerations: 25   // Reduced from 50
});
```

### **Issue**: Hyperbolic embedding doesn't converge

**Solution**: Increase iterations or adjust learning rate:
```typescript
const hyperbolicEmbedding = new HyperbolicEmbedding({
  maxIterations: 2000,  // Increased from 1000
  learningRate: 0.05    // Reduced from 0.1
});
```

---

**Ready to integrate?** Follow the steps above and your platform will be upgraded with advanced ML capabilities!

