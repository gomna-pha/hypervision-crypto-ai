/**
 * Real-Time Feature Engineering & Store
 * 
 * Transforms raw market data into features for ML models:
 * - Returns (log, simple, rolling)
 * - Spreads (cross-exchange, basis, funding)
 * - Volatility (realized, implied, EWMA)
 * - Flow imbalances (order flow, volume)
 * - Z-scores (normalized signals)
 * - Rolling windows (MA, EMA, Bollinger)
 * - Lagged & relative features
 * 
 * Includes versioned feature store for consistency
 */

export interface RawMarketData {
  timestamp: Date;
  symbol: string;
  
  // Prices
  spotPrice: number;
  perpPrice?: number;
  bidPrice: number;
  askPrice: number;
  
  // Cross-exchange
  exchangePrices: Record<string, number>; // { 'binance': 96500, 'coinbase': 96530 }
  
  // Volume & liquidity
  volume24h: number;
  bidVolume: number;
  askVolume: number;
  liquidity: number; // Order book depth
  
  // Funding & rates
  fundingRate?: number;
  openInterest?: number;
  
  // Market-wide
  marketCap?: number;
  dominance?: number;
}

export interface EngineeredFeatures {
  timestamp: Date;
  symbol: string;
  version: string; // Feature version for tracking
  
  // Returns
  returns: {
    log1m: number;      // 1-min log return
    log5m: number;      // 5-min log return
    log1h: number;      // 1-hour log return
    simple1h: number;   // 1-hour simple return
  };
  
  // Spreads
  spreads: {
    bidAsk: number;           // Bid-ask spread (bps)
    crossExchange: number[];  // Cross-exchange spreads (bps)
    spotPerp: number;         // Spot-perp basis (bps)
    fundingBasis?: number;    // Funding rate basis (bps)
  };
  
  // Volatility
  volatility: {
    realized1h: number;  // 1-hour realized vol (%)
    realized24h: number; // 24-hour realized vol (%)
    ewma: number;        // EWMA volatility
    parkinson: number;   // Parkinson high-low estimator
  };
  
  // Flow & imbalances
  flow: {
    volumeImbalance: number;  // (buy_vol - sell_vol) / total_vol
    orderImbalance: number;   // (bid_vol - ask_vol) / (bid_vol + ask_vol)
    netFlow: number;          // Net buying pressure
  };
  
  // Normalized features (z-scores)
  zScores: {
    priceZ: number;      // Price z-score vs 24h mean
    volumeZ: number;     // Volume z-score vs 24h mean
    spreadZ: number;     // Spread z-score vs 24h mean
  };
  
  // Rolling statistics
  rolling: {
    sma20: number;       // 20-period SMA
    ema20: number;       // 20-period EMA
    bollingerUpper: number;
    bollingerLower: number;
    bollingerWidth: number;
    rsi14: number;       // 14-period RSI
  };
  
  // Lagged features
  lagged: {
    price_lag1: number;
    price_lag5: number;
    volume_lag1: number;
    volume_lag5: number;
  };
  
  // Relative features
  relative: {
    priceVsSMA: number;   // (price - SMA) / SMA
    volumeVsAvg: number;  // volume / avg_volume
    spreadVsAvg: number;  // spread / avg_spread
  };
}

export interface FeatureStoreConfig {
  windowSizes: {
    short: number;   // 20 periods
    medium: number;  // 50 periods
    long: number;    // 200 periods
  };
  version: string;
  enableVersioning: boolean;
}

export class FeatureEngineer {
  private config: FeatureStoreConfig;
  private priceHistory: Map<string, number[]>; // symbol -> prices
  private volumeHistory: Map<string, number[]>;
  private spreadHistory: Map<string, number[]>;
  private timestampHistory: Map<string, Date[]>;
  private featureStore: Map<string, EngineeredFeatures[]>; // Versioned store
  
  constructor(config: Partial<FeatureStoreConfig> = {}) {
    this.config = {
      windowSizes: {
        short: config.windowSizes?.short || 20,
        medium: config.windowSizes?.medium || 50,
        long: config.windowSizes?.long || 200,
      },
      version: config.version || 'v1.0.0',
      enableVersioning: config.enableVersioning !== false,
    };
    
    this.priceHistory = new Map();
    this.volumeHistory = new Map();
    this.spreadHistory = new Map();
    this.timestampHistory = new Map();
    this.featureStore = new Map();
  }
  
  /**
   * Engineer features from raw market data
   */
  engineer(data: RawMarketData): EngineeredFeatures {
    const symbol = data.symbol;
    
    // Update history
    this.updateHistory(data);
    
    // Calculate features
    const returns = this.calculateReturns(symbol, data);
    const spreads = this.calculateSpreads(data);
    const volatility = this.calculateVolatility(symbol, data);
    const flow = this.calculateFlow(data);
    const zScores = this.calculateZScores(symbol, data);
    const rolling = this.calculateRolling(symbol, data);
    const lagged = this.calculateLagged(symbol, data);
    const relative = this.calculateRelative(symbol, data, rolling);
    
    const features: EngineeredFeatures = {
      timestamp: data.timestamp,
      symbol,
      version: this.config.version,
      returns,
      spreads,
      volatility,
      flow,
      zScores,
      rolling,
      lagged,
      relative,
    };
    
    // Store features
    if (this.config.enableVersioning) {
      this.storeFeatures(symbol, features);
    }
    
    return features;
  }
  
  /**
   * Update historical data
   */
  private updateHistory(data: RawMarketData): void {
    const symbol = data.symbol;
    
    // Initialize if needed
    if (!this.priceHistory.has(symbol)) {
      this.priceHistory.set(symbol, []);
      this.volumeHistory.set(symbol, []);
      this.spreadHistory.set(symbol, []);
      this.timestampHistory.set(symbol, []);
    }
    
    // Add new data
    const prices = this.priceHistory.get(symbol)!;
    const volumes = this.volumeHistory.get(symbol)!;
    const spreads = this.spreadHistory.get(symbol)!;
    const timestamps = this.timestampHistory.get(symbol)!;
    
    prices.push(data.spotPrice);
    volumes.push(data.volume24h);
    spreads.push(data.askPrice - data.bidPrice);
    timestamps.push(data.timestamp);
    
    // Keep only recent history (max 200 periods)
    const maxLength = this.config.windowSizes.long;
    if (prices.length > maxLength) {
      prices.shift();
      volumes.shift();
      spreads.shift();
      timestamps.shift();
    }
  }
  
  /**
   * Calculate returns
   */
  private calculateReturns(symbol: string, data: RawMarketData): EngineeredFeatures['returns'] {
    const prices = this.priceHistory.get(symbol) || [];
    const currentPrice = data.spotPrice;
    
    if (prices.length < 2) {
      return { log1m: 0, log5m: 0, log1h: 0, simple1h: 0 };
    }
    
    // Log returns
    const log1m = prices.length >= 1 ? Math.log(currentPrice / prices[prices.length - 1]) : 0;
    const log5m = prices.length >= 5 ? Math.log(currentPrice / prices[prices.length - 5]) : 0;
    const log1h = prices.length >= 60 ? Math.log(currentPrice / prices[prices.length - 60]) : 0;
    
    // Simple return
    const simple1h = prices.length >= 60 
      ? (currentPrice - prices[prices.length - 60]) / prices[prices.length - 60]
      : 0;
    
    return {
      log1m: log1m * 100, // Convert to %
      log5m: log5m * 100,
      log1h: log1h * 100,
      simple1h: simple1h * 100,
    };
  }
  
  /**
   * Calculate spreads
   */
  private calculateSpreads(data: RawMarketData): EngineeredFeatures['spreads'] {
    // Bid-ask spread
    const bidAsk = (data.askPrice - data.bidPrice) / data.spotPrice * 10000; // bps
    
    // Cross-exchange spreads
    const crossExchange: number[] = [];
    const exchangePrices = Object.values(data.exchangePrices);
    for (let i = 0; i < exchangePrices.length; i++) {
      for (let j = i + 1; j < exchangePrices.length; j++) {
        const spread = Math.abs(exchangePrices[i] - exchangePrices[j]) / data.spotPrice * 10000;
        crossExchange.push(spread);
      }
    }
    
    // Spot-perp basis
    const spotPerp = data.perpPrice 
      ? (data.perpPrice - data.spotPrice) / data.spotPrice * 10000
      : 0;
    
    // Funding basis
    const fundingBasis = data.fundingRate 
      ? data.fundingRate * 10000
      : undefined;
    
    return {
      bidAsk,
      crossExchange,
      spotPerp,
      fundingBasis,
    };
  }
  
  /**
   * Calculate volatility
   */
  private calculateVolatility(symbol: string, data: RawMarketData): EngineeredFeatures['volatility'] {
    const prices = this.priceHistory.get(symbol) || [];
    
    if (prices.length < 2) {
      return { realized1h: 0, realized24h: 0, ewma: 0, parkinson: 0 };
    }
    
    // Realized volatility (standard deviation of returns)
    const returns = this.calculateReturnsSeries(prices);
    const realized1h = prices.length >= 60 
      ? this.stdDev(returns.slice(-60)) * Math.sqrt(60) * 100
      : 0;
    const realized24h = prices.length >= 1440 
      ? this.stdDev(returns.slice(-1440)) * Math.sqrt(1440) * 100
      : this.stdDev(returns) * Math.sqrt(returns.length) * 100;
    
    // EWMA volatility (exponentially weighted)
    const ewma = this.calculateEWMA(returns, 0.94) * 100;
    
    // Parkinson volatility (high-low estimator)
    // Simplified: use recent price range
    const recentPrices = prices.slice(-20);
    const high = Math.max(...recentPrices);
    const low = Math.min(...recentPrices);
    const parkinson = Math.sqrt(Math.log(high / low) ** 2 / (4 * Math.log(2))) * 100;
    
    return {
      realized1h,
      realized24h,
      ewma,
      parkinson,
    };
  }
  
  /**
   * Calculate flow and imbalances
   */
  private calculateFlow(data: RawMarketData): EngineeredFeatures['flow'] {
    // Volume imbalance (approximation: bid/ask volume)
    const totalVolume = data.bidVolume + data.askVolume;
    const volumeImbalance = totalVolume > 0 
      ? (data.bidVolume - data.askVolume) / totalVolume
      : 0;
    
    // Order imbalance (order book)
    const orderImbalance = (data.bidVolume + data.askVolume) > 0
      ? (data.bidVolume - data.askVolume) / (data.bidVolume + data.askVolume)
      : 0;
    
    // Net flow (positive = buying pressure)
    const netFlow = volumeImbalance * data.volume24h;
    
    return {
      volumeImbalance,
      orderImbalance,
      netFlow,
    };
  }
  
  /**
   * Calculate z-scores
   */
  private calculateZScores(symbol: string, data: RawMarketData): EngineeredFeatures['zScores'] {
    const prices = this.priceHistory.get(symbol) || [];
    const volumes = this.volumeHistory.get(symbol) || [];
    const spreads = this.spreadHistory.get(symbol) || [];
    
    const priceZ = this.calculateZScore(data.spotPrice, prices);
    const volumeZ = this.calculateZScore(data.volume24h, volumes);
    const spreadZ = this.calculateZScore(data.askPrice - data.bidPrice, spreads);
    
    return { priceZ, volumeZ, spreadZ };
  }
  
  /**
   * Calculate rolling statistics
   */
  private calculateRolling(symbol: string, data: RawMarketData): EngineeredFeatures['rolling'] {
    const prices = this.priceHistory.get(symbol) || [];
    const currentPrice = data.spotPrice;
    
    if (prices.length < 20) {
      return {
        sma20: currentPrice,
        ema20: currentPrice,
        bollingerUpper: currentPrice * 1.02,
        bollingerLower: currentPrice * 0.98,
        bollingerWidth: 4,
        rsi14: 50,
      };
    }
    
    // SMA
    const sma20 = this.mean(prices.slice(-20));
    
    // EMA
    const ema20 = this.calculateEMA(prices, 20);
    
    // Bollinger Bands (SMA ± 2*stddev)
    const stdDev20 = this.stdDev(prices.slice(-20));
    const bollingerUpper = sma20 + 2 * stdDev20;
    const bollingerLower = sma20 - 2 * stdDev20;
    const bollingerWidth = (bollingerUpper - bollingerLower) / sma20 * 100;
    
    // RSI
    const rsi14 = this.calculateRSI(prices, 14);
    
    return {
      sma20,
      ema20,
      bollingerUpper,
      bollingerLower,
      bollingerWidth,
      rsi14,
    };
  }
  
  /**
   * Calculate lagged features
   */
  private calculateLagged(symbol: string, data: RawMarketData): EngineeredFeatures['lagged'] {
    const prices = this.priceHistory.get(symbol) || [];
    const volumes = this.volumeHistory.get(symbol) || [];
    
    return {
      price_lag1: prices.length >= 1 ? prices[prices.length - 1] : data.spotPrice,
      price_lag5: prices.length >= 5 ? prices[prices.length - 5] : data.spotPrice,
      volume_lag1: volumes.length >= 1 ? volumes[volumes.length - 1] : data.volume24h,
      volume_lag5: volumes.length >= 5 ? volumes[volumes.length - 5] : data.volume24h,
    };
  }
  
  /**
   * Calculate relative features
   */
  private calculateRelative(
    symbol: string,
    data: RawMarketData,
    rolling: EngineeredFeatures['rolling']
  ): EngineeredFeatures['relative'] {
    const volumes = this.volumeHistory.get(symbol) || [];
    const spreads = this.spreadHistory.get(symbol) || [];
    
    const priceVsSMA = rolling.sma20 > 0 
      ? (data.spotPrice - rolling.sma20) / rolling.sma20
      : 0;
    
    const avgVolume = volumes.length > 0 ? this.mean(volumes) : data.volume24h;
    const volumeVsAvg = avgVolume > 0 ? data.volume24h / avgVolume : 1;
    
    const currentSpread = data.askPrice - data.bidPrice;
    const avgSpread = spreads.length > 0 ? this.mean(spreads) : currentSpread;
    const spreadVsAvg = avgSpread > 0 ? currentSpread / avgSpread : 1;
    
    return {
      priceVsSMA,
      volumeVsAvg,
      spreadVsAvg,
    };
  }
  
  // ============= Helper Functions =============
  
  private calculateReturnsSeries(prices: number[]): number[] {
    const returns: number[] = [];
    for (let i = 1; i < prices.length; i++) {
      returns.push(Math.log(prices[i] / prices[i - 1]));
    }
    return returns;
  }
  
  private mean(values: number[]): number {
    return values.reduce((sum, v) => sum + v, 0) / values.length;
  }
  
  private stdDev(values: number[]): number {
    const avg = this.mean(values);
    const variance = values.reduce((sum, v) => sum + Math.pow(v - avg, 2), 0) / values.length;
    return Math.sqrt(variance);
  }
  
  private calculateZScore(value: number, history: number[]): number {
    if (history.length < 2) return 0;
    const avg = this.mean(history);
    const std = this.stdDev(history);
    return std > 0 ? (value - avg) / std : 0;
  }
  
  private calculateEMA(values: number[], period: number): number {
    if (values.length === 0) return 0;
    
    const multiplier = 2 / (period + 1);
    let ema = values[0];
    
    for (let i = 1; i < values.length; i++) {
      ema = (values[i] - ema) * multiplier + ema;
    }
    
    return ema;
  }
  
  private calculateEWMA(returns: number[], lambda: number): number {
    if (returns.length === 0) return 0;
    
    let variance = returns[0] ** 2;
    for (let i = 1; i < returns.length; i++) {
      variance = lambda * variance + (1 - lambda) * returns[i] ** 2;
    }
    
    return Math.sqrt(variance);
  }
  
  private calculateRSI(prices: number[], period: number): number {
    if (prices.length < period + 1) return 50;
    
    const changes = [];
    for (let i = 1; i < prices.length; i++) {
      changes.push(prices[i] - prices[i - 1]);
    }
    
    const recentChanges = changes.slice(-period);
    const gains = recentChanges.filter(c => c > 0);
    const losses = recentChanges.filter(c => c < 0).map(c => Math.abs(c));
    
    const avgGain = gains.length > 0 ? this.mean(gains) : 0;
    const avgLoss = losses.length > 0 ? this.mean(losses) : 0;
    
    if (avgLoss === 0) return 100;
    
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }
  
  /**
   * Store features in versioned store
   */
  private storeFeatures(symbol: string, features: EngineeredFeatures): void {
    if (!this.featureStore.has(symbol)) {
      this.featureStore.set(symbol, []);
    }
    
    const store = this.featureStore.get(symbol)!;
    store.push(features);
    
    // Keep only last 1000 feature sets
    if (store.length > 1000) {
      store.shift();
    }
  }
  
  /**
   * Get feature history for a symbol
   */
  getFeatureHistory(symbol: string, limit?: number): EngineeredFeatures[] {
    const store = this.featureStore.get(symbol) || [];
    return limit ? store.slice(-limit) : store;
  }
  
  /**
   * Get latest features for a symbol
   */
  getLatestFeatures(symbol: string): EngineeredFeatures | null {
    const store = this.featureStore.get(symbol);
    return store && store.length > 0 ? store[store.length - 1] : null;
  }
  
  /**
   * Clear feature store
   */
  clearStore(): void {
    this.featureStore.clear();
    this.priceHistory.clear();
    this.volumeHistory.clear();
    this.spreadHistory.clear();
    this.timestampHistory.clear();
  }
}

/**
 * Example usage:
 * 
 * const engineer = new FeatureEngineer({
 *   windowSizes: { short: 20, medium: 50, long: 200 },
 *   version: 'v1.0.0',
 *   enableVersioning: true,
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
 * const features = engineer.engineer(rawData);
 * console.log('Engineered features:', features);
 */
