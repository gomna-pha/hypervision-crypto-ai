import EventEmitter from 'events';
import Logger from '../utils/logger';

const logger = Logger.getInstance('AdvancedTradingStrategies');

// Barra Risk Factors
interface BarraFactors {
  momentum: number;
  value: number;
  growth: number;
  profitability: number;
  investment: number;
  volatility: number;
  size: number;
  leverage: number;
  liquidity: number;
  beta: number;
}

// Statistical Arbitrage
interface PairTrade {
  pair: [string, string];
  spread: number;
  zScore: number;
  halfLife: number;
  cointegrationPValue: number;
  hedgeRatio: number;
  signal: 'LONG_PAIR' | 'SHORT_PAIR' | 'NEUTRAL';
  confidence: number;
  expectedReturn: number;
  sharpeRatio: number;
}

// Machine Learning Models
interface MLPrediction {
  model: 'RandomForest' | 'AdaBoost' | 'GradientBoosting' | 'XGBoost' | 'LightGBM';
  prediction: number;
  probability: number;
  features: number[];
  importance: Map<string, number>;
  confidence: number;
}

// Portfolio Optimization
interface OptimalPortfolio {
  weights: Map<string, number>;
  expectedReturn: number;
  risk: number;
  sharpeRatio: number;
  maxDrawdown: number;
  kellyFraction: number;
  var95: number;
  cvar95: number;
  metrics?: {
    expectedReturn: number;
    volatility: number;
    sharpeRatio: number;
    cvar: number;
  };
}

export class AdvancedTradingStrategies extends EventEmitter {
  private isRunning: boolean = false;
  private prices: Map<string, number[]> = new Map();
  private returns: Map<string, number[]> = new Map();
  private barraFactors: Map<string, BarraFactors> = new Map();
  private pairTrades: PairTrade[] = [];
  private mlModels: Map<string, any> = new Map();
  private mlPredictions: any = null;
  private portfolio: OptimalPortfolio | null = null;
  
  // Live market data
  private currentPrices: Map<string, number> = new Map();
  private orderBook: Map<string, { bids: any[], asks: any[] }> = new Map();
  private volume: Map<string, number> = new Map();

  constructor() {
    super();
    this.initializeModels();
    this.initializeMarketData();
  }

  private initializeModels(): void {
    // Initialize ML models
    this.mlModels.set('RandomForest', this.createRandomForestModel());
    this.mlModels.set('AdaBoost', this.createAdaBoostModel());
    this.mlModels.set('GradientBoosting', this.createGradientBoostingModel());
    this.mlModels.set('XGBoost', this.createXGBoostModel());
    this.mlModels.set('LightGBM', this.createLightGBMModel());
  }

  private initializeMarketData(): void {
    const symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC', 'LINK', 'DOT', 'UNI', 'AAVE', 'CRV'];
    
    symbols.forEach(symbol => {
      // Initialize with synthetic historical data
      const prices: number[] = [];
      let price = this.getInitialPrice(symbol);
      
      for (let i = 0; i < 500; i++) {
        const return_ = (Math.random() - 0.5) * 0.02 + 0.0001; // Slight positive drift
        price *= (1 + return_);
        prices.push(price);
      }
      
      this.prices.set(symbol, prices);
      this.currentPrices.set(symbol, price);
      this.volume.set(symbol, Math.random() * 10000000);
      
      // Calculate returns
      const returns = prices.slice(1).map((p, i) => (p - prices[i]) / prices[i]);
      this.returns.set(symbol, returns);
    });
  }

  private getInitialPrice(symbol: string): number {
    const prices: Record<string, number> = {
      'BTC': 45000,
      'ETH': 2800,
      'SOL': 120,
      'AVAX': 35,
      'MATIC': 0.85,
      'LINK': 15,
      'DOT': 7,
      'UNI': 6,
      'AAVE': 95,
      'CRV': 0.6
    };
    return prices[symbol] || 10;
  }

  async start(): Promise<void> {
    if (this.isRunning) return;
    this.isRunning = true;
    
    logger.info('Starting Advanced Trading Strategies Engine');
    
    // Start real-time calculations
    this.startBarraFactorCalculation();
    this.startStatisticalArbitrage();
    this.startMachineLearningPredictions();
    this.startPortfolioOptimization();
    this.startMarketDataSimulation();
    
    // Broadcast updates
    setInterval(() => this.broadcastStrategies(), 1000);
  }

  private startBarraFactorCalculation(): void {
    setInterval(() => {
      this.currentPrices.forEach((price, symbol) => {
        const prices = this.prices.get(symbol) || [];
        const returns = this.returns.get(symbol) || [];
        
        if (prices.length < 100) return;
        
        // Calculate Barra factors
        const factors: BarraFactors = {
          momentum: this.calculateMomentum(prices),
          value: this.calculateValue(price, prices),
          growth: this.calculateGrowth(returns),
          profitability: this.calculateProfitability(returns),
          investment: this.calculateInvestment(returns),
          volatility: this.calculateVolatility(returns),
          size: Math.log(price * (this.volume.get(symbol) || 1)),
          leverage: this.calculateLeverage(returns),
          liquidity: this.calculateLiquidity(symbol),
          beta: this.calculateBeta(returns)
        };
        
        this.barraFactors.set(symbol, factors);
      });
    }, 500); // Update every 500ms
  }

  private calculateMomentum(prices: number[]): number {
    // 12-month momentum with 1-month reversal
    const momentum12m = (prices[prices.length - 1] - prices[prices.length - 252]) / prices[prices.length - 252];
    const reversal1m = (prices[prices.length - 1] - prices[prices.length - 21]) / prices[prices.length - 21];
    return momentum12m - 0.5 * reversal1m;
  }

  private calculateValue(currentPrice: number, prices: number[]): number {
    // Book-to-Price ratio proxy
    const ma200 = prices.slice(-200).reduce((a, b) => a + b, 0) / 200;
    return ma200 / currentPrice - 1;
  }

  private calculateGrowth(returns: number[]): number {
    // Return growth rate
    const recentReturns = returns.slice(-20);
    const olderReturns = returns.slice(-40, -20);
    const recentAvg = recentReturns.reduce((a, b) => a + b, 0) / recentReturns.length;
    const olderAvg = olderReturns.reduce((a, b) => a + b, 0) / olderReturns.length;
    return (recentAvg - olderAvg) / Math.abs(olderAvg);
  }

  private calculateProfitability(returns: number[]): number {
    // ROE proxy using Sharpe ratio
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const std = Math.sqrt(returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length);
    return mean / (std || 0.001);
  }

  private calculateInvestment(returns: number[]): number {
    // Asset growth proxy
    const cumReturn = returns.reduce((a, b) => a * (1 + b), 1) - 1;
    return cumReturn / returns.length;
  }

  private calculateVolatility(returns: number[]): number {
    // Realized volatility
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    return Math.sqrt(returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length) * Math.sqrt(252);
  }

  private calculateLeverage(returns: number[]): number {
    // Debt-to-Equity proxy using volatility clustering
    const recentVol = this.calculateVolatility(returns.slice(-20));
    const longTermVol = this.calculateVolatility(returns.slice(-100));
    return recentVol / longTermVol - 1;
  }

  private calculateLiquidity(symbol: string): number {
    // Amihud illiquidity measure
    const volume = this.volume.get(symbol) || 1;
    const price = this.currentPrices.get(symbol) || 1;
    const returns = this.returns.get(symbol) || [];
    const lastReturn = returns[returns.length - 1] || 0.001;
    return Math.abs(lastReturn) / (price * volume) * 1e9;
  }

  private calculateBeta(returns: number[]): number {
    // Market beta (simplified - using BTC as market)
    const btcReturns = this.returns.get('BTC') || returns;
    const minLength = Math.min(returns.length, btcReturns.length);
    
    const assetReturns = returns.slice(-minLength);
    const marketReturns = btcReturns.slice(-minLength);
    
    const covariance = this.calculateCovariance(assetReturns, marketReturns);
    const marketVariance = this.calculateVariance(marketReturns);
    
    return covariance / (marketVariance || 0.001);
  }

  private calculateCovariance(x: number[], y: number[]): number {
    const meanX = x.reduce((a, b) => a + b, 0) / x.length;
    const meanY = y.reduce((a, b) => a + b, 0) / y.length;
    
    return x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0) / x.length;
  }

  private calculateVariance(x: number[]): number {
    const mean = x.reduce((a, b) => a + b, 0) / x.length;
    return x.reduce((sum, xi) => sum + Math.pow(xi - mean, 2), 0) / x.length;
  }

  private startStatisticalArbitrage(): void {
    setInterval(() => {
      const symbols = Array.from(this.currentPrices.keys());
      this.pairTrades = [];
      
      // Find cointegrated pairs
      for (let i = 0; i < symbols.length; i++) {
        for (let j = i + 1; j < symbols.length; j++) {
          const pair = this.analyzePair(symbols[i], symbols[j]);
          if (pair && pair.cointegrationPValue < 0.05) {
            this.pairTrades.push(pair);
          }
        }
      }
      
      // Sort by expected return
      this.pairTrades.sort((a, b) => b.expectedReturn - a.expectedReturn);
      
      // Keep top 10 pairs
      this.pairTrades = this.pairTrades.slice(0, 10);
    }, 2000); // Update every 2 seconds
  }

  private analyzePair(symbol1: string, symbol2: string): PairTrade | null {
    const prices1 = this.prices.get(symbol1);
    const prices2 = this.prices.get(symbol2);
    
    if (!prices1 || !prices2) return null;
    
    // Calculate spread
    const hedgeRatio = this.calculateHedgeRatio(prices1, prices2);
    const spread = prices1.map((p, i) => p - hedgeRatio * prices2[i]);
    
    // Calculate z-score
    const spreadMean = spread.reduce((a, b) => a + b, 0) / spread.length;
    const spreadStd = Math.sqrt(spread.reduce((a, b) => a + Math.pow(b - spreadMean, 2), 0) / spread.length);
    const currentSpread = spread[spread.length - 1];
    const zScore = (currentSpread - spreadMean) / spreadStd;
    
    // Calculate half-life (Ornstein-Uhlenbeck)
    const halfLife = this.calculateHalfLife(spread);
    
    // Cointegration test (simplified ADF test p-value)
    const cointegrationPValue = this.testCointegration(spread);
    
    // Generate signal
    let signal: 'LONG_PAIR' | 'SHORT_PAIR' | 'NEUTRAL' = 'NEUTRAL';
    if (zScore < -2) signal = 'LONG_PAIR';
    else if (zScore > 2) signal = 'SHORT_PAIR';
    
    // Expected return based on mean reversion
    const expectedReturn = -zScore * spreadStd / Math.abs(currentSpread) * Math.exp(-1 / halfLife);
    
    // Sharpe ratio of the pair trade
    const pairReturns = spread.slice(1).map((s, i) => (s - spread[i]) / spread[i]);
    const sharpeRatio = this.calculateSharpeRatio(pairReturns);
    
    return {
      pair: [symbol1, symbol2],
      spread: currentSpread,
      zScore,
      halfLife,
      cointegrationPValue,
      hedgeRatio,
      signal,
      confidence: Math.min(0.99, 1 / (1 + Math.exp(-Math.abs(zScore) + 2))),
      expectedReturn,
      sharpeRatio
    };
  }

  private calculateHedgeRatio(prices1: number[], prices2: number[]): number {
    // OLS regression to find hedge ratio
    const n = Math.min(prices1.length, prices2.length);
    const x = prices2.slice(-n);
    const y = prices1.slice(-n);
    
    const meanX = x.reduce((a, b) => a + b, 0) / n;
    const meanY = y.reduce((a, b) => a + b, 0) / n;
    
    const num = x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0);
    const den = x.reduce((sum, xi) => sum + Math.pow(xi - meanX, 2), 0);
    
    return num / (den || 1);
  }

  private calculateHalfLife(spread: number[]): number {
    // AR(1) model to estimate half-life
    const y = spread.slice(1);
    const x = spread.slice(0, -1);
    
    const n = y.length;
    const meanX = x.reduce((a, b) => a + b, 0) / n;
    const meanY = y.reduce((a, b) => a + b, 0) / n;
    
    const beta = x.reduce((sum, xi, i) => sum + (xi - meanX) * (y[i] - meanY), 0) /
                 x.reduce((sum, xi) => sum + Math.pow(xi - meanX, 2), 0);
    
    return -Math.log(2) / Math.log(beta);
  }

  private testCointegration(spread: number[]): number {
    // Simplified ADF test - returns p-value
    const diffs = spread.slice(1).map((s, i) => s - spread[i]);
    const mean = diffs.reduce((a, b) => a + b, 0) / diffs.length;
    const std = Math.sqrt(diffs.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / diffs.length);
    const tStat = mean / (std / Math.sqrt(diffs.length));
    
    // Convert t-statistic to p-value (simplified)
    return 1 / (1 + Math.exp(-Math.abs(tStat) + 2));
  }

  private calculateSharpeRatio(returns: number[]): number {
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const std = Math.sqrt(returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length);
    return (mean / (std || 0.001)) * Math.sqrt(252);
  }

  private startMachineLearningPredictions(): void {
    setInterval(() => {
      this.currentPrices.forEach((price, symbol) => {
        const features = this.extractFeatures(symbol);
        if (!features) return;
        
        // Random Forest
        const rfPrediction = this.predictRandomForest(features);
        
        // AdaBoost
        const adaPrediction = this.predictAdaBoost(features);
        
        // Gradient Boosting
        const gbPrediction = this.predictGradientBoosting(features);
        
        // XGBoost
        const xgPrediction = this.predictXGBoost(features);
        
        // LightGBM
        const lgbPrediction = this.predictLightGBM(features);
        
        // Ensemble prediction
        const ensemble = (rfPrediction + adaPrediction + gbPrediction + xgPrediction + lgbPrediction) / 5;
        
        // Store and emit ML predictions
        const predictions = {
          randomForest: { prediction: rfPrediction > 0 ? 1 : -1, confidence: Math.abs(rfPrediction) },
          adaBoost: { prediction: adaPrediction > 0 ? 1 : -1, confidence: Math.abs(adaPrediction) },
          gradientBoosting: { prediction: gbPrediction > 0 ? 1 : -1, confidence: Math.abs(gbPrediction) },
          xgboost: { prediction: xgPrediction > 0 ? 1 : -1, confidence: Math.abs(xgPrediction) },
          lightgbm: { prediction: lgbPrediction > 0 ? 1 : -1, confidence: Math.abs(lgbPrediction) },
          ensemble: {
            signal: ensemble > 0.5 ? 'LONG' : ensemble < -0.5 ? 'SHORT' : 'NEUTRAL',
            strength: Math.abs(ensemble),
            agreement: (Math.sign(rfPrediction) + Math.sign(adaPrediction) + Math.sign(gbPrediction) + Math.sign(xgPrediction) + Math.sign(lgbPrediction)) / 5
          }
        };
        
        this.mlPredictions = predictions;
        this.emit('ml_prediction', predictions);
      });
    }, 1000); // Predict every second
  }

  private extractFeatures(symbol: string): number[] | null {
    const prices = this.prices.get(symbol);
    const returns = this.returns.get(symbol);
    const volume = this.volume.get(symbol);
    const factors = this.barraFactors.get(symbol);
    
    if (!prices || !returns || !factors || prices.length < 100) return null;
    
    // Technical indicators
    const rsi = this.calculateRSI(prices, 14);
    const macd = this.calculateMACD(prices);
    const bb = this.calculateBollingerBands(prices, 20);
    const atr = this.calculateATR(prices, 14);
    
    // Microstructure
    const spread = this.calculateSpread(symbol);
    const imbalance = this.calculateOrderImbalance(symbol);
    
    return [
      factors.momentum,
      factors.value,
      factors.growth,
      factors.profitability,
      factors.investment,
      factors.volatility,
      factors.size,
      factors.leverage,
      factors.liquidity,
      factors.beta,
      rsi,
      macd.signal,
      bb.position,
      atr,
      spread,
      imbalance,
      Math.log(volume || 1),
      returns[returns.length - 1] || 0
    ];
  }

  private calculateRSI(prices: number[], period: number = 14): number {
    if (prices.length < period + 1) return 50;
    
    const changes = prices.slice(-period - 1).map((p, i, arr) => 
      i === 0 ? 0 : p - arr[i - 1]
    ).slice(1);
    
    const gains = changes.filter(c => c > 0);
    const losses = changes.filter(c => c < 0).map(Math.abs);
    
    const avgGain = gains.reduce((a, b) => a + b, 0) / period;
    const avgLoss = losses.reduce((a, b) => a + b, 0) / period;
    
    const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  private calculateMACD(prices: number[]): { macd: number, signal: number } {
    const ema12 = this.calculateEMA(prices, 12);
    const ema26 = this.calculateEMA(prices, 26);
    const macdLine = ema12 - ema26;
    const signal = this.calculateEMA([macdLine], 9);
    
    return { macd: macdLine, signal };
  }

  private calculateEMA(prices: number[], period: number): number {
    const k = 2 / (period + 1);
    let ema = prices[0];
    
    for (let i = 1; i < prices.length; i++) {
      ema = prices[i] * k + ema * (1 - k);
    }
    
    return ema;
  }

  private calculateBollingerBands(prices: number[], period: number): { upper: number, lower: number, position: number } {
    const slice = prices.slice(-period);
    const mean = slice.reduce((a, b) => a + b, 0) / period;
    const std = Math.sqrt(slice.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / period);
    
    const upper = mean + 2 * std;
    const lower = mean - 2 * std;
    const current = prices[prices.length - 1];
    const position = (current - lower) / (upper - lower);
    
    return { upper, lower, position };
  }

  private calculateATR(prices: number[], period: number): number {
    const trs: number[] = [];
    
    for (let i = 1; i < prices.length; i++) {
      const high = prices[i] * 1.01; // Simulated high
      const low = prices[i] * 0.99; // Simulated low
      const prevClose = prices[i - 1];
      
      const tr = Math.max(
        high - low,
        Math.abs(high - prevClose),
        Math.abs(low - prevClose)
      );
      
      trs.push(tr);
    }
    
    const recentTRs = trs.slice(-period);
    return recentTRs.reduce((a, b) => a + b, 0) / period;
  }

  private calculateSpread(symbol: string): number {
    const orderBook = this.orderBook.get(symbol);
    if (!orderBook || !orderBook.bids.length || !orderBook.asks.length) {
      return 0.001; // Default spread
    }
    
    const bestBid = orderBook.bids[0].price;
    const bestAsk = orderBook.asks[0].price;
    
    return (bestAsk - bestBid) / ((bestAsk + bestBid) / 2);
  }

  private calculateOrderImbalance(symbol: string): number {
    const orderBook = this.orderBook.get(symbol);
    if (!orderBook) return 0;
    
    const bidVolume = orderBook.bids.reduce((sum, b) => sum + b.quantity, 0);
    const askVolume = orderBook.asks.reduce((sum, a) => sum + a.quantity, 0);
    
    return (bidVolume - askVolume) / (bidVolume + askVolume + 1);
  }

  // Machine Learning Model Implementations (Simplified)
  private createRandomForestModel(): any {
    const trees = Array(100).fill(null).map(() => this.createDecisionTree());
    return {
      trees,
      predict: (features: number[]) => {
        const predictions = trees.map((tree: any) => tree.predict(features));
        return predictions.reduce((a: number, b: number) => a + b, 0) / predictions.length;
      }
    };
  }

  private createDecisionTree(): any {
    return {
      predict: (features: number[]) => {
        // Simplified decision tree logic
        let prediction = 0;
        
        if (features[0] > 0.1) prediction += 0.3; // Momentum
        if (features[1] > 0) prediction += 0.2; // Value
        if (features[5] < 0.3) prediction += 0.2; // Low volatility
        if (features[10] > 70) prediction -= 0.2; // RSI overbought
        if (features[10] < 30) prediction += 0.2; // RSI oversold
        
        return Math.tanh(prediction);
      }
    };
  }

  private createAdaBoostModel(): any {
    const weakLearners = Array(50).fill(null).map(() => ({
      weight: Math.random(),
      threshold: Math.random(),
      feature: Math.floor(Math.random() * 18)
    }));
    return {
      weakLearners,
      predict: (features: number[]) => {
        return weakLearners.reduce((sum: number, learner: any) => {
          const prediction = features[learner.feature] > learner.threshold ? 1 : -1;
          return sum + prediction * learner.weight;
        }, 0) / weakLearners.length;
      }
    };
  }

  private createGradientBoostingModel(): any {
    const estimators = Array(100).fill(null).map(() => ({
      learning_rate: 0.1,
      predict: (features: number[]) => {
        // Gradient boosted tree prediction
        const weighted = features.reduce((sum, f, i) => sum + f * (i + 1) * 0.01, 0);
        return Math.tanh(weighted);
      }
    }));
    return {
      estimators,
      predict: (features: number[]) => {
        return estimators.reduce((sum: number, est: any) => {
          return sum + est.learning_rate * est.predict(features);
        }, 0);
      }
    };
  }

  private createXGBoostModel(): any {
    return {
      predict: (features: number[]) => {
        // Simplified XGBoost logic with regularization
        const linear = features.reduce((sum, f, i) => sum + f * Math.random(), 0);
        const interaction = features[0] * features[1] * 0.1;
        const regularization = 0.01;
        
        return Math.tanh((linear + interaction) * (1 - regularization));
      }
    };
  }

  private createLightGBMModel(): any {
    return {
      predict: (features: number[]) => {
        // Simplified LightGBM with histogram-based splits
        const bins = features.map(f => Math.floor(f * 10) / 10);
        const score = bins.reduce((sum, b, i) => {
          const weight = 1 / (i + 1);
          return sum + b * weight;
        }, 0);
        
        return Math.tanh(score);
      }
    };
  }

  private predictRandomForest(features: number[]): number {
    const model = this.mlModels.get('RandomForest');
    return model ? model.predict(features) : 0;
  }

  private predictAdaBoost(features: number[]): number {
    const model = this.mlModels.get('AdaBoost');
    return model ? model.predict(features) : 0;
  }

  private predictGradientBoosting(features: number[]): number {
    const model = this.mlModels.get('GradientBoosting');
    return model ? model.predict(features) : 0;
  }

  private predictXGBoost(features: number[]): number {
    const model = this.mlModels.get('XGBoost');
    return model ? model.predict(features) : 0;
  }

  private predictLightGBM(features: number[]): number {
    const model = this.mlModels.get('LightGBM');
    return model ? model.predict(features) : 0;
  }

  private startPortfolioOptimization(): void {
    setInterval(() => {
      const symbols = Array.from(this.currentPrices.keys());
      const returns = symbols.map(s => this.returns.get(s) || []);
      
      if (returns.some(r => r.length < 100)) return;
      
      // Calculate expected returns and covariance matrix
      const expectedReturns = returns.map(r => r.reduce((a, b) => a + b, 0) / r.length);
      const covMatrix = this.calculateCovarianceMatrix(returns);
      
      // Robust portfolio optimization (CVaR optimization)
      const weights = this.optimizePortfolioCVaR(expectedReturns, covMatrix, 0.95);
      
      // Calculate portfolio metrics
      const portfolioReturn = expectedReturns.reduce((sum, r, i) => sum + r * weights[i], 0);
      const portfolioRisk = this.calculatePortfolioRisk(weights, covMatrix);
      const sharpeRatio = (portfolioReturn * 252) / (portfolioRisk * Math.sqrt(252));
      
      // Kelly criterion
      const kellyFraction = portfolioReturn / (portfolioRisk * portfolioRisk);
      
      // VaR and CVaR
      const { var95, cvar95 } = this.calculateVaRCVaR(weights, returns, 0.95);
      
      // Max drawdown
      const maxDrawdown = this.calculateMaxDrawdown(weights, returns);
      
      this.portfolio = {
        weights: new Map(symbols.map((s, i) => [s, weights[i]])),
        expectedReturn: portfolioReturn * 252,
        risk: portfolioRisk * Math.sqrt(252),
        sharpeRatio,
        maxDrawdown,
        kellyFraction: Math.min(0.25, Math.max(0, kellyFraction)), // Cap at 25%
        var95,
        cvar95
      };
    }, 3000); // Optimize every 3 seconds
  }

  private calculateCovarianceMatrix(returns: number[][]): number[][] {
    const n = returns.length;
    const matrix: number[][] = [];
    
    for (let i = 0; i < n; i++) {
      matrix[i] = [];
      for (let j = 0; j < n; j++) {
        matrix[i][j] = this.calculateCovariance(returns[i], returns[j]);
      }
    }
    
    return matrix;
  }

  private optimizePortfolioCVaR(expectedReturns: number[], covMatrix: number[][], alpha: number): number[] {
    // Simplified CVaR optimization using gradient descent
    const n = expectedReturns.length;
    let weights = Array(n).fill(1 / n); // Equal weight initialization
    
    const learningRate = 0.01;
    const iterations = 100;
    
    for (let iter = 0; iter < iterations; iter++) {
      // Calculate gradient
      const gradient = this.calculateCVaRGradient(weights, expectedReturns, covMatrix, alpha);
      
      // Update weights
      weights = weights.map((w, i) => w - learningRate * gradient[i]);
      
      // Project to simplex (weights sum to 1, all positive)
      weights = this.projectToSimplex(weights);
    }
    
    return weights;
  }

  private calculateCVaRGradient(weights: number[], returns: number[], covMatrix: number[][], alpha: number): number[] {
    // Simplified gradient calculation for CVaR
    const n = weights.length;
    const gradient = Array(n).fill(0);
    
    for (let i = 0; i < n; i++) {
      // Return component
      gradient[i] -= returns[i];
      
      // Risk component
      for (let j = 0; j < n; j++) {
        gradient[i] += 2 * covMatrix[i][j] * weights[j];
      }
      
      // CVaR penalty
      gradient[i] += (1 - alpha) * Math.sign(weights[i]);
    }
    
    return gradient;
  }

  private projectToSimplex(weights: number[]): number[] {
    // Project weights to simplex (sum to 1, all non-negative)
    const sorted = [...weights].sort((a, b) => b - a);
    const n = weights.length;
    
    let tmpSum = 0;
    let i = 0;
    
    for (i = 0; i < n - 1; i++) {
      tmpSum += sorted[i];
      const tMax = (tmpSum - 1) / (i + 1);
      if (tMax >= sorted[i + 1]) {
        break;
      }
    }
    
    const theta = (tmpSum + sorted[i] - 1) / (i + 1);
    
    return weights.map(w => Math.max(0, w - theta));
  }

  private calculatePortfolioRisk(weights: number[], covMatrix: number[][]): number {
    let risk = 0;
    
    for (let i = 0; i < weights.length; i++) {
      for (let j = 0; j < weights.length; j++) {
        risk += weights[i] * weights[j] * covMatrix[i][j];
      }
    }
    
    return Math.sqrt(risk);
  }

  private calculateVaRCVaR(weights: number[], returns: number[][], alpha: number): { var95: number, cvar95: number } {
    // Calculate portfolio returns
    const portfolioReturns: number[] = [];
    const minLength = Math.min(...returns.map(r => r.length));
    
    for (let t = 0; t < minLength; t++) {
      const portfolioReturn = weights.reduce((sum, w, i) => sum + w * returns[i][t], 0);
      portfolioReturns.push(portfolioReturn);
    }
    
    // Sort returns
    portfolioReturns.sort((a, b) => a - b);
    
    // Calculate VaR
    const varIndex = Math.floor((1 - alpha) * portfolioReturns.length);
    const var95 = -portfolioReturns[varIndex];
    
    // Calculate CVaR (expected shortfall)
    const tailReturns = portfolioReturns.slice(0, varIndex);
    const cvar95 = -tailReturns.reduce((a, b) => a + b, 0) / tailReturns.length;
    
    return { var95, cvar95 };
  }

  private calculateMaxDrawdown(weights: number[], returns: number[][]): number {
    // Calculate portfolio equity curve
    const minLength = Math.min(...returns.map(r => r.length));
    let equity = 100000; // Start with $100k
    let maxEquity = equity;
    let maxDD = 0;
    
    for (let t = 0; t < minLength; t++) {
      const portfolioReturn = weights.reduce((sum, w, i) => sum + w * returns[i][t], 0);
      equity *= (1 + portfolioReturn);
      
      if (equity > maxEquity) {
        maxEquity = equity;
      }
      
      const drawdown = (maxEquity - equity) / maxEquity;
      if (drawdown > maxDD) {
        maxDD = drawdown;
      }
    }
    
    return maxDD;
  }

  private startMarketDataSimulation(): void {
    setInterval(() => {
      // Update prices with realistic market dynamics
      this.currentPrices.forEach((price, symbol) => {
        const prices = this.prices.get(symbol) || [];
        const factors = this.barraFactors.get(symbol);
        
        // Market dynamics
        let drift = 0.00001; // Slight positive drift
        
        if (factors) {
          // Factor-based price movement
          drift += factors.momentum * 0.0001;
          drift += factors.value * 0.00005;
          drift -= factors.volatility * 0.00002;
        }
        
        // Add noise
        const noise = (Math.random() - 0.5) * 0.001;
        
        // Update price
        const newPrice = price * (1 + drift + noise);
        this.currentPrices.set(symbol, newPrice);
        
        // Update price history
        prices.push(newPrice);
        if (prices.length > 1000) prices.shift();
        
        // Update returns
        const returns = this.returns.get(symbol) || [];
        if (prices.length > 1) {
          returns.push((newPrice - prices[prices.length - 2]) / prices[prices.length - 2]);
          if (returns.length > 999) returns.shift();
        }
        
        // Update volume
        const currentVolume = this.volume.get(symbol) || 0;
        const volumeChange = (Math.random() - 0.5) * 0.1;
        this.volume.set(symbol, currentVolume * (1 + volumeChange));
        
        // Update order book
        this.updateOrderBook(symbol, newPrice);
      });
    }, 100); // Update every 100ms for real-time feel
  }

  private updateOrderBook(symbol: string, price: number): void {
    const spread = 0.0001 + Math.random() * 0.0009; // 1-10 bps spread
    const depth = 10;
    
    const bids = Array(depth).fill(null).map((_, i) => ({
      price: price * (1 - spread * (i + 1)),
      quantity: Math.random() * 1000
    }));
    
    const asks = Array(depth).fill(null).map((_, i) => ({
      price: price * (1 + spread * (i + 1)),
      quantity: Math.random() * 1000
    }));
    
    this.orderBook.set(symbol, { bids, asks });
  }

  private broadcastStrategies(): void {
    this.emit('strategies_update', {
      timestamp: Date.now(),
      barraFactors: Object.fromEntries(this.barraFactors),
      pairTrades: this.pairTrades,
      portfolio: this.portfolio,
      prices: Object.fromEntries(this.currentPrices),
      volumes: Object.fromEntries(this.volume)
    });
  }

  // Public getters
  getBarraFactors(): Map<string, BarraFactors> {
    return this.barraFactors;
  }

  getPairTrades(): PairTrade[] {
    return this.pairTrades;
  }

  getPortfolio(): OptimalPortfolio | null {
    return this.portfolio;
  }

  getCurrentPrices(): Map<string, number> {
    return this.currentPrices;
  }

  stop(): void {
    this.isRunning = false;
    this.removeAllListeners();
  }
  
  // Additional getters for dashboard integration
  getLatestBarraFactors(): any {
    return Object.fromEntries(this.barraFactors);
  }
  
  getStatisticalArbitragePairs(): PairTrade[] {
    return this.pairTrades;
  }
  
  getMLPredictions(): any {
    return this.mlPredictions;
  }
  
  getPortfolioOptimization(): OptimalPortfolio | null {
    return this.portfolio;
  }
  
  getStrategyPerformance(): any {
    return {
      barraFactors: Object.fromEntries(this.barraFactors),
      activePairs: this.pairTrades.filter(p => p.signal !== 'HOLD').length,
      mlConsensus: this.mlPredictions?.ensemble?.signal || 'NEUTRAL',
      portfolioSharpe: this.portfolio?.metrics?.sharpeRatio || 0,
      lastUpdate: Date.now()
    };
  }
}

export { AdvancedTradingStrategies };