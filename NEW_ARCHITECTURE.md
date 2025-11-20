# New Multi-Agent Portfolio Optimization Architecture

## Core Principle
**Agents are data sources → Strategies consume agent data → Portfolio optimization balances return vs risk**

---

## 1. AGENTS (4 Core Data Sources)

### Agent Structure
```typescript
interface Agent {
  name: string;
  score: number;        // 0-100% normalized score
  metrics: {
    [key: string]: number;  // Raw metrics (Fed rate, CPI, etc.)
  };
  lastUpdated: Date;
}
```

### Four Agents:
1. **Economic Agent**: Fed rate, CPI, GDP → Score (0-100%)
2. **Sentiment Agent**: Fear & Greed, Google Trends, VIX → Score (0-100%)
3. **Cross-Exchange Agent**: Price spreads, liquidity, volume → Score (0-100%)
4. **On-Chain Agent**: Exchange netflows, SOPR, MVRV → Score (0-100%)

---

## 2. STRATEGIES (7 Trading Algorithms)

### Linear Strategies (Normal Distribution)
```typescript
interface LinearStrategy {
  name: string;
  type: 'linear';
  agents: Agent[];           // Selected agents feeding this strategy
  historicalReturns: number[];  // Generated from agent scores
  expectedReturn: number;    // μ
  volatility: number;        // σ
  correlation: Record<string, number>;  // ρ with other strategies
}
```

**3 Linear Strategies:**
1. **Spatial Arbitrage** (Cross-Exchange driven)
2. **Triangular Arbitrage** (Cross-Exchange + Sentiment driven)
3. **Statistical Arbitrage** (Cross-Exchange + On-Chain driven)

### Non-Linear Strategies (Fat-Tailed Distribution)
```typescript
interface NonLinearStrategy {
  name: string;
  type: 'non-linear';
  agents: Agent[];
  scenarios: {              // 1000 simulated scenarios
    returns: number[];
    probabilities: number[];
  };
  expectedReturn: number;
  volatility: number;
  cvar95: number;           // Tail risk at 95%
  maxDrawdown: number;
}
```

**4 Non-Linear Strategies:**
1. **ML Ensemble** (All 4 agents)
2. **Deep Learning** (Sentiment + On-Chain)
3. **CNN Pattern** (Cross-Exchange + Sentiment)
4. **Sentiment Arbitrage** (Sentiment only)

---

## 3. AGENT-STRATEGY CONFIGURATION MATRIX

### UI Component (7×4 = 28 checkboxes)
```
                Economic  Sentiment  Cross-Exch  On-Chain
Spatial            ☐         ☐          ☑          ☐
Triangular         ☐         ☑          ☑          ☐
Statistical        ☐         ☐          ☑          ☑
ML Ensemble        ☑         ☑          ☑          ☑
Deep Learning      ☐         ☑          ☐          ☑
CNN Pattern        ☐         ☑          ☑          ☐
Sentiment Arb      ☐         ☑          ☐          ☐
```

### Backend Logic
```typescript
function calculateStrategyPerformance(
  strategy: string,
  selectedAgents: Agent[],
  historicalAgentScores: Record<string, number[]>  // 252 days of scores
): {
  expectedReturn: number;
  volatility: number;
  historicalReturns: number[];
} {
  // Step 1: Calculate daily strategy returns from agent scores
  const dailyReturns: number[] = [];
  
  for (let day = 0; day < 252; day++) {
    // Get agent scores for this day
    const agentScores = selectedAgents.map(agent => 
      historicalAgentScores[agent.name][day]
    );
    
    // Strategy return = f(agent scores) with strategy-specific formula
    const strategyReturn = calculateStrategyReturn(strategy, agentScores, day);
    dailyReturns.push(strategyReturn);
  }
  
  // Step 2: Calculate statistics
  const expectedReturn = mean(dailyReturns) * 252 * 100;  // Annualized %
  const volatility = std(dailyReturns) * Math.sqrt(252) * 100;  // Annualized %
  
  return { expectedReturn, volatility, historicalReturns: dailyReturns };
}
```

### Strategy Return Formulas (Examples)

**Spatial Arbitrage** (Cross-Exchange driven):
```typescript
// Return proportional to cross-exchange score
return = (crossExchangeScore - 50) * 0.002;  // -10% to +10% daily
```

**ML Ensemble** (All 4 agents):
```typescript
// Weighted combination of all agents
return = (
  0.3 * (crossExchangeScore - 50) +
  0.3 * (sentimentScore - 50) +
  0.2 * (economicScore - 50) +
  0.2 * (onChainScore - 50)
) * 0.001;
```

---

## 4. PORTFOLIO OPTIMIZATION METHODS

### A. For Linear Strategies (Covariance-Based)

#### Mean-Variance Optimization (Markowitz)
```typescript
function meanVarianceOptimization(
  strategies: LinearStrategy[],
  lambda: number  // Risk aversion 0-10
): {
  weights: number[];
  expectedReturn: number;
  volatility: number;
  sharpeRatio: number;
} {
  // Step 1: Build inputs
  const mu = strategies.map(s => s.expectedReturn / 100);  // Expected returns vector
  const Sigma = calculateCovarianceMatrix(strategies);      // Covariance matrix
  
  // Step 2: Solve optimization
  // max: μ'w - (λ/2) * w'Σw
  // subject to: Σw = 1, w >= 0
  
  const weights = solveQuadraticProgram(mu, Sigma, lambda);
  
  // Step 3: Calculate portfolio metrics
  const expectedReturn = dotProduct(weights, mu) * 100;
  const variance = quadraticForm(weights, Sigma);
  const volatility = Math.sqrt(variance) * 100;
  const sharpeRatio = (expectedReturn - 2) / volatility;  // 2% risk-free rate
  
  return { weights, expectedReturn, volatility, sharpeRatio };
}
```

#### Risk Parity
```typescript
function riskParityOptimization(strategies: LinearStrategy[]): Portfolio {
  // Objective: Each strategy contributes equal risk
  // wi * σi * ∂σp/∂wi = constant for all i
  
  const Sigma = calculateCovarianceMatrix(strategies);
  const weights = solveRiskParity(Sigma);
  
  return calculatePortfolioMetrics(weights, strategies);
}
```

#### Minimum Variance
```typescript
function minimumVarianceOptimization(strategies: LinearStrategy[]): Portfolio {
  // Objective: min w'Σw
  // subject to: Σw = 1, w >= 0
  
  const Sigma = calculateCovarianceMatrix(strategies);
  const weights = solveMinimumVariance(Sigma);
  
  return calculatePortfolioMetrics(weights, strategies);
}
```

### B. For Non-Linear Strategies (Scenario-Based)

#### CVaR Optimization
```typescript
function cvarOptimization(
  strategies: NonLinearStrategy[],
  alpha: number = 0.95  // Confidence level
): {
  weights: number[];
  expectedReturn: number;
  volatility: number;
  cvar: number;  // Tail risk
} {
  // Step 1: Generate scenarios (1000 simulations)
  const scenarios: number[][] = [];  // [scenario][strategy]
  
  for (let s = 0; s < 1000; s++) {
    const scenarioReturns = strategies.map(strategy => 
      sampleFromDistribution(strategy.scenarios)
    );
    scenarios.push(scenarioReturns);
  }
  
  // Step 2: Solve CVaR optimization
  // min: CVaR_α(L) = E[L | L >= VaR_α]
  // subject to: E[Rp] >= targetReturn, Σw = 1, w >= 0
  
  const weights = solveCVaROptimization(scenarios, alpha);
  
  // Step 3: Calculate portfolio metrics
  const portfolioReturns = scenarios.map(s => 
    dotProduct(weights, s)
  );
  
  const expectedReturn = mean(portfolioReturns) * 100;
  const volatility = std(portfolioReturns) * 100;
  const cvar = calculateCVaR(portfolioReturns, alpha) * 100;
  
  return { weights, expectedReturn, volatility, cvar };
}
```

#### Kelly Criterion
```typescript
function kellyOptimization(strategy: NonLinearStrategy): {
  kellyFraction: number;
  fractionalKelly: number;
} {
  // For discrete binary outcomes
  const winRate = strategy.scenarios.probabilities.filter(p => p > 0).length / 1000;
  const avgWin = mean(strategy.scenarios.returns.filter(r => r > 0));
  const avgLoss = mean(strategy.scenarios.returns.filter(r => r < 0));
  
  const b = Math.abs(avgWin / avgLoss);  // Win/loss ratio
  const p = winRate;
  const q = 1 - p;
  
  // Kelly fraction: f* = (p*b - q) / b
  const kellyFraction = (p * b - q) / b;
  
  // Fractional Kelly (25% of full Kelly for risk control)
  const fractionalKelly = kellyFraction * 0.25;
  
  return { kellyFraction, fractionalKelly };
}
```

### C. Hybrid Meta-Optimization

```typescript
function hybridOptimization(
  linearStrategies: LinearStrategy[],
  nonLinearStrategies: NonLinearStrategy[],
  riskConstraint: number  // Max portfolio volatility (e.g., 14%)
): {
  linearWeights: number[];
  nonLinearWeights: number[];
  allocationSplit: { linear: number; nonLinear: number };
  portfolioMetrics: {
    expectedReturn: number;
    volatility: number;
    sharpeRatio: number;
    cvar95: number;
  };
} {
  // Step 1: Optimize linear sub-portfolio
  const linearPortfolio = meanVarianceOptimization(linearStrategies, 5);
  
  // Step 2: Optimize non-linear sub-portfolio
  const nonLinearPortfolio = cvarOptimization(nonLinearStrategies, 0.95);
  
  // Step 3: Find optimal split between sub-portfolios
  // Subject to: α * σ_linear² + (1-α) * σ_nonlinear² + 2α(1-α)ρσ_linear*σ_nonlinear <= riskConstraint²
  
  const optimalSplit = findOptimalSplit(
    linearPortfolio,
    nonLinearPortfolio,
    riskConstraint
  );
  
  return {
    linearWeights: linearPortfolio.weights,
    nonLinearWeights: nonLinearPortfolio.weights,
    allocationSplit: optimalSplit,
    portfolioMetrics: calculateCombinedMetrics(
      linearPortfolio,
      nonLinearPortfolio,
      optimalSplit
    )
  };
}
```

---

## 5. OPTIMIZATION COMPARISON TABLE

### Data Structure
```typescript
interface OptimizationResult {
  method: string;
  expectedReturn: number;  // %
  volatility: number;      // %
  sharpeRatio: number;
  maxDrawdown: number;     // %
  cvar95: number;          // %
  weights: Record<string, number>;
  interpretation: string;
}

const comparisonTable: OptimizationResult[] = [
  {
    method: "Equal Weight",
    expectedReturn: 12.3,
    volatility: 18.5,
    sharpeRatio: 0.66,
    maxDrawdown: -15.2,
    cvar95: -18.1,
    weights: { Spatial: 0.14, Triangular: 0.14, ... },
    interpretation: "Naive diversification without optimization"
  },
  {
    method: "Mean-Variance (λ=3)",
    expectedReturn: 15.1,
    volatility: 16.2,
    sharpeRatio: 0.93,
    maxDrawdown: -12.1,
    cvar95: -14.5,
    weights: { Spatial: 0.15, Triangular: 0.25, ... },
    interpretation: "Prioritizes return (15.1%) over risk reduction"
  },
  // ... all other methods
];
```

---

## 6. API ENDPOINTS

### GET /api/agents
```json
{
  "economic": { "score": 45, "fedRate": 4.5, "cpi": 3.2, ... },
  "sentiment": { "score": 62, "fearGreed": 58, "vix": 18.5, ... },
  "crossExchange": { "score": 71, "spread": 0.18, "liquidity": 85, ... },
  "onChain": { "score": 54, "netflow": -4200, "sopr": 1.02, ... }
}
```

### POST /api/portfolio/configure
```json
{
  "agentStrategyMatrix": {
    "Spatial": ["Cross-Exchange"],
    "Triangular": ["Cross-Exchange", "Sentiment"],
    "ML Ensemble": ["Economic", "Sentiment", "Cross-Exchange", "On-Chain"]
  }
}
```

Response:
```json
{
  "strategies": [
    {
      "name": "Spatial",
      "selectedAgents": ["Cross-Exchange"],
      "performance": {
        "expectedReturn": 15.2,
        "volatility": 8.5,
        "sharpeRatio": 1.79
      },
      "comparisonToAllAgents": {
        "returnImprovement": "+3.1%",
        "riskReduction": "-2.7%",
        "interpretation": "Using only Cross-Exchange improves return-risk tradeoff"
      }
    }
  ]
}
```

### POST /api/portfolio/optimize
```json
{
  "strategies": ["Spatial", "Triangular", "Statistical"],
  "method": "mean-variance",  // or "risk-parity", "cvar", "hybrid"
  "riskPreference": 5,         // 0-10
  "constraints": {
    "maxRisk": 14,              // Max portfolio volatility %
    "targetReturn": 16          // Target return %
  }
}
```

Response:
```json
{
  "success": true,
  "method": "mean-variance",
  "weights": {
    "Spatial": 0.22,
    "Triangular": 0.35,
    "Statistical": 0.43
  },
  "metrics": {
    "expectedReturn": 16.8,
    "volatility": 13.5,
    "sharpeRatio": 1.09,
    "maxDrawdown": -10.2,
    "cvar95": -12.1
  },
  "efficient Frontier": [
    { "risk": 10, "return": 12.5 },
    { "risk": 12, "return": 14.8 },
    { "risk": 14, "return": 16.8 },
    { "risk": 16, "return": 18.2 }
  ]
}
```

### POST /api/portfolio/compare
```json
{
  "strategies": ["Spatial", "Triangular", "Statistical", "ML Ensemble"]
}
```

Response:
```json
{
  "comparisonTable": [
    {
      "method": "Equal Weight",
      "expectedReturn": 12.3,
      "volatility": 18.5,
      "sharpeRatio": 0.66,
      "weights": [0.25, 0.25, 0.25, 0.25]
    },
    {
      "method": "Mean-Variance (λ=3)",
      "expectedReturn": 15.1,
      "volatility": 16.2,
      "sharpeRatio": 0.93,
      "weights": [0.15, 0.25, 0.10, 0.50]
    },
    {
      "method": "Hybrid (65/35)",
      "expectedReturn": 15.9,
      "volatility": 14.7,
      "sharpeRatio": 1.08,
      "weights": [0.12, 0.28, 0.08, 0.52]
    }
  ]
}
```

---

## 7. FRONTEND SECTIONS

### Section 1: Agent Dashboard (4 Cards)
- Economic Agent (score 0-100%)
- Sentiment Agent (score 0-100%)
- Cross-Exchange Agent (score 0-100%)
- On-Chain Agent (score 0-100%)

### Section 2: Agent-Strategy Configuration Matrix
- 7 rows (strategies) × 4 columns (agents) = 28 checkboxes
- Live performance update as you check/uncheck agents
- Comparison to "all agents" baseline

### Section 3: Risk Preference Controls
- Risk slider: [Low Risk ←━━●━━→ High Return]
- Max acceptable risk: [Slider: 10% to 25%]
- Target return: [Slider: 8% to 20%]
- Constraint validation

### Section 4: Optimization Method Selector
- Radio buttons: Mean-Variance, Risk Parity, Min Variance, CVaR, Hybrid
- Lambda slider (for Mean-Variance): 0-10
- Alpha slider (for CVaR): 90%-99%

### Section 5: Optimization Comparison Table
- Shows all methods side-by-side
- Highlights best Sharpe ratio, lowest risk, highest return
- Interpretation text for each method

### Section 6: Efficient Frontier Visualization
- Chart: Risk (x-axis) vs Return (y-axis)
- Clickable curve
- Current portfolio marked as dot

### Section 7: Strategy Performance Details
- Individual strategy metrics
- Correlation matrix
- Return distribution histograms

---

## 8. IMPLEMENTATION PRIORITY

### Phase 1 (Core): ✅ Must Have
1. 4 Agents API with real scores
2. 7 Strategies with agent dependencies
3. Agent-Strategy Configuration Matrix UI
4. Mean-Variance Optimization
5. Optimization Comparison Table

### Phase 2 (Enhanced): 🔶 Should Have
6. Risk Parity and Minimum Variance
7. CVaR Optimization for non-linear strategies
8. Hybrid meta-optimization
9. Efficient Frontier visualization
10. Dynamic rebalancing

### Phase 3 (Advanced): 🔵 Nice to Have
11. Black-Litterman Model
12. Robust Optimization
13. Kelly Criterion calculator
14. Rolling 30-day reoptimization

---

## 9. KEY DIFFERENCES FROM OLD SYSTEM

### OLD (Wrong):
- ❌ Agents and strategies disconnected
- ❌ Portfolio optimization used fake historical prices
- ❌ No agent-strategy selection matrix
- ❌ Only Mean-Variance optimization
- ❌ Hardcoded weights and returns

### NEW (Correct):
- ✅ Agents feed strategies (configuration matrix)
- ✅ Strategy returns calculated from agent scores
- ✅ Investors select which agents power which strategies
- ✅ Multiple optimization methods (Mean-Variance, CVaR, Risk Parity, Hybrid)
- ✅ Everything driven by real agent data
- ✅ Comparison table shows tradeoffs
- ✅ Efficient frontier for risk-return visualization

---

## 10. MATH TRANSPARENCY

Every optimization shows:
1. **Objective function**: What we're maximizing/minimizing
2. **Input data**: Agent scores → Strategy returns
3. **Optimization method**: Mean-Variance, CVaR, etc.
4. **Output weights**: Allocation percentages
5. **Resulting metrics**: Return, risk, Sharpe, CVaR
6. **Interpretation**: Why this allocation makes sense

**Example Display:**
```
Strategy: Spatial Arbitrage
Selected Agents: Cross-Exchange only
Historical Performance (252 days):
  → Agent Score avg: 68/100
  → Daily returns: [-0.5%, +1.2%, -0.3%, ...]
  → Expected Return: 15.2% (annualized)
  → Volatility: 8.5% (annualized)
  → Sharpe Ratio: 1.79

Comparison (Cross-Exchange only vs All Agents):
  → Return: 15.2% vs 12.1% (+3.1%)
  → Risk: 8.5% vs 11.2% (-2.7%)
  → Interpretation: Using only Cross-Exchange agent improves return-risk tradeoff
```

---

This architecture puts **agents at the center** as data sources, allows **investor configuration** via the matrix, and drives all **optimization from agent-informed strategy returns**. No hardcoding, all data-driven.
