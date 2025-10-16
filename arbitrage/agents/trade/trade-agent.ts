/**
 * Trade Agent - Trade Flow Analysis and Execution Quality
 * Analyzes trade-flow patterns, taker/maker imbalances, VWAP calculations,
 * slippage estimation, and unusual trade pattern detection
 */

import { BaseAgent, AgentConfig, AgentOutput } from '../../core/base-agent.js';
import axios from 'axios';

export interface TradeData {
  exchange: string;
  pair: string;
  taker_buy_volume_1m: number;     // Volume of taker buy orders (market buys)
  taker_sell_volume_1m: number;    // Volume of taker sell orders (market sells)
  maker_volume_1m: number;         // Volume of maker orders (limit orders)
  vwap_1m: number;                 // Volume Weighted Average Price
  vwap_5m: number;                 // 5-minute VWAP for comparison
  slippage_estimate: number;       // Estimated slippage for standard order sizes
  trade_latency_ms: number;        // Average trade execution latency
  abnormal_pattern_flag: boolean;   // Flag for unusual trading patterns
  market_impact: number;           // Price impact per unit volume
  order_flow_imbalance: number;    // Imbalance between buy and sell pressure
}

export interface TradeFeatures {
  overall_taker_buy_ratio: number;    // Ratio of taker buys to total taker volume
  cross_exchange_vwap_deviation: number; // VWAP deviation across exchanges
  average_slippage_estimate: number;   // Average slippage across all exchanges
  trade_efficiency_score: number;     // Quality of trade execution
  order_flow_pressure: number;       // Net order flow pressure direction
  pattern_anomaly_score: number;     // Score for unusual patterns
}

export interface VWAPCalculation {
  exchange: string;
  pair: string;
  vwap_1m: number;
  vwap_5m: number;
  vwap_15m: number;
  price_deviation: number;  // How far current price is from VWAP
  volume_profile: number[]; // Volume distribution across price levels
}

export interface SlippageEstimate {
  exchange: string;
  pair: string;
  small_order_slippage: number;    // $1K order
  medium_order_slippage: number;   // $10K order
  large_order_slippage: number;    // $100K order
  liquidity_score: number;         // Overall liquidity quality
}

export class TradeAgent extends BaseAgent {
  private tradeHistory: Map<string, any[]> = new Map();
  private vwapCache: Map<string, VWAPCalculation> = new Map();
  private slippageCache: Map<string, SlippageEstimate> = new Map();
  private lastLatencyMeasurement: Map<string, number> = new Map();
  
  // Configuration
  private readonly EXCHANGES = ['binance', 'coinbase', 'kraken'];
  private readonly PAIRS = ['BTC-USDT', 'ETH-USDT', 'BTC-USD', 'ETH-USD'];
  private readonly VWAP_WINDOWS = [60, 300, 900]; // 1min, 5min, 15min in seconds
  private readonly SLIPPAGE_ORDER_SIZES = [1000, 10000, 100000]; // USD
  private readonly MAX_TRADE_LATENCY_MS = 500;
  private readonly ABNORMAL_VOLUME_THRESHOLD = 3.0; // 3x normal volume

  constructor(config: AgentConfig) {
    super(config);
  }

  protected async collectData(): Promise<AgentOutput> {
    const timestamp = this.getCurrentTimestamp();

    try {
      // Collect trade flow data from all exchanges
      const tradeDataPoints = await this.collectTradeFlowData();
      
      // Calculate VWAP for different time windows
      const vwapCalculations = await this.calculateVWAPMetrics();
      
      // Estimate slippage for different order sizes
      const slippageEstimates = await this.calculateSlippageEstimates();
      
      // Calculate derived features
      const features = this.calculateFeatures(tradeDataPoints, vwapCalculations, slippageEstimates);
      
      // Calculate key signal (trade quality and flow analysis)
      const keySignal = this.calculateKeySignal(features);
      
      // Calculate confidence based on data quality
      const confidence = this.calculateDataConfidence(tradeDataPoints);
      
      return {
        agent_name: 'TradeAgent',
        timestamp,
        key_signal: keySignal,
        confidence,
        features: {
          ...features,
          trade_data_points: tradeDataPoints.length,
          vwap_calculations: vwapCalculations.length,
          slippage_estimates: slippageEstimates.length
        },
        metadata: {
          exchanges_tracked: this.EXCHANGES,
          pairs_tracked: this.PAIRS,
          vwap_windows_sec: this.VWAP_WINDOWS,
          order_sizes_usd: this.SLIPPAGE_ORDER_SIZES
        }
      };

    } catch (error) {
      console.error('TradeAgent data collection failed:', error);
      throw error;
    }
  }

  /**
   * Collect trade flow data from all exchanges
   */
  private async collectTradeFlowData(): Promise<TradeData[]> {
    const tradePromises: Promise<TradeData[]>[] = [];

    for (const exchange of this.EXCHANGES) {
      tradePromises.push(this.collectExchangeTradeData(exchange));
    }

    const results = await Promise.allSettled(tradePromises);
    const tradeData: TradeData[] = [];

    for (const result of results) {
      if (result.status === 'fulfilled' && result.value) {
        tradeData.push(...result.value);
      }
    }

    return tradeData;
  }

  /**
   * Collect trade data from specific exchange
   */
  private async collectExchangeTradeData(exchange: string): Promise<TradeData[]> {
    switch (exchange) {
      case 'binance':
        return await this.collectBinanceTradeData();
      case 'coinbase':
        return await this.collectCoinbaseTradeData();
      case 'kraken':
        return await this.collectKrakenTradeData();
      default:
        return [];
    }
  }

  /**
   * Collect Binance trade flow data
   */
  private async collectBinanceTradeData(): Promise<TradeData[]> {
    try {
      const symbols = ['BTCUSDT', 'ETHUSDT'];
      const tradeData: TradeData[] = [];

      for (const symbol of symbols) {
        try {
          // Measure API latency
          const latencyStart = Date.now();
          
          // Get recent trades
          const tradesResponse = await axios.get('https://api.binance.com/api/v3/aggTrades', {
            params: { symbol, limit: 1000 },
            timeout: 5000
          });
          
          const latencyEnd = Date.now();
          const apiLatency = latencyEnd - latencyStart;
          
          if (tradesResponse.data && Array.isArray(tradesResponse.data)) {
            const tradeInfo = this.processBinanceTrades(
              tradesResponse.data,
              this.normalizePair(symbol),
              apiLatency
            );
            tradeData.push(tradeInfo);
          }
        } catch (error) {
          console.warn(`Binance ${symbol} trade collection error:`, error.message);
        }
      }

      return tradeData;
    } catch (error) {
      console.warn('Binance trade collection error:', error.message);
      return [];
    }
  }

  /**
   * Process Binance trade data
   */
  private processBinanceTrades(trades: any[], pair: string, apiLatency: number): TradeData {
    const now = Date.now();
    const oneMinuteAgo = now - 60000;
    const fiveMinutesAgo = now - 300000;
    
    const recentTrades1m = trades.filter(trade => trade.T >= oneMinuteAgo);
    const recentTrades5m = trades.filter(trade => trade.T >= fiveMinutesAgo);
    
    // Analyze trade flow (1 minute window)
    let takerBuyVolume1m = 0;
    let takerSellVolume1m = 0;
    let makerVolume1m = 0;
    let totalValue1m = 0;
    let totalQuantity1m = 0;
    
    for (const trade of recentTrades1m) {
      const price = parseFloat(trade.p);
      const quantity = parseFloat(trade.q);
      const value = price * quantity;
      
      totalValue1m += value;
      totalQuantity1m += quantity;
      
      if (trade.m) { // Buyer is maker (limit order filled)
        makerVolume1m += value;
        takerSellVolume1m += value; // Taker was selling
      } else { // Buyer is taker (market order)
        takerBuyVolume1m += value;
      }
    }
    
    // Calculate VWAP (1 minute and 5 minute)
    const vwap1m = totalQuantity1m > 0 ? totalValue1m / totalQuantity1m : 0;
    const vwap5m = this.calculateVWAP(recentTrades5m);
    
    // Estimate slippage for standard order size ($10K)
    const slippageEstimate = this.estimateSlippageFromTrades(recentTrades1m, 10000);
    
    // Calculate market impact (price movement per unit volume)
    const marketImpact = this.calculateMarketImpact(recentTrades1m);
    
    // Detect abnormal patterns
    const abnormalPattern = this.detectAbnormalPattern(recentTrades1m, pair, 'binance');
    
    // Calculate order flow imbalance
    const totalTakerVolume = takerBuyVolume1m + takerSellVolume1m;
    const orderFlowImbalance = totalTakerVolume > 0 
      ? (takerBuyVolume1m - takerSellVolume1m) / totalTakerVolume
      : 0;

    return {
      exchange: 'binance',
      pair,
      taker_buy_volume_1m: takerBuyVolume1m,
      taker_sell_volume_1m: takerSellVolume1m,
      maker_volume_1m: makerVolume1m,
      vwap_1m: vwap1m,
      vwap_5m: vwap5m,
      slippage_estimate: slippageEstimate,
      trade_latency_ms: apiLatency,
      abnormal_pattern_flag: abnormalPattern,
      market_impact: marketImpact,
      order_flow_imbalance: orderFlowImbalance
    };
  }

  /**
   * Collect Coinbase trade flow data
   */
  private async collectCoinbaseTradeData(): Promise<TradeData[]> {
    try {
      const products = ['BTC-USD', 'ETH-USD'];
      const tradeData: TradeData[] = [];

      for (const product of products) {
        try {
          const latencyStart = Date.now();
          
          // Get recent trades
          const tradesResponse = await axios.get(`https://api.exchange.coinbase.com/products/${product}/trades`, {
            params: { limit: 100 },
            timeout: 5000
          });
          
          const latencyEnd = Date.now();
          const apiLatency = latencyEnd - latencyStart;
          
          if (tradesResponse.data && Array.isArray(tradesResponse.data)) {
            const tradeInfo = this.processCoinbaseTrades(
              tradesResponse.data,
              this.normalizePair(product),
              apiLatency
            );
            tradeData.push(tradeInfo);
          }
        } catch (error) {
          console.warn(`Coinbase ${product} trade collection error:`, error.message);
        }
      }

      return tradeData;
    } catch (error) {
      console.warn('Coinbase trade collection error:', error.message);
      return [];
    }
  }

  /**
   * Process Coinbase trade data
   */
  private processCoinbaseTrades(trades: any[], pair: string, apiLatency: number): TradeData {
    const now = new Date();
    const oneMinuteAgo = new Date(now.getTime() - 60000);
    const fiveMinutesAgo = new Date(now.getTime() - 300000);
    
    const recentTrades1m = trades.filter(trade => new Date(trade.time) >= oneMinuteAgo);
    const recentTrades5m = trades.filter(trade => new Date(trade.time) >= fiveMinutesAgo);
    
    let takerBuyVolume1m = 0;
    let takerSellVolume1m = 0;
    let totalValue1m = 0;
    let totalSize1m = 0;
    
    for (const trade of recentTrades1m) {
      const price = parseFloat(trade.price);
      const size = parseFloat(trade.size);
      const value = price * size;
      
      totalValue1m += value;
      totalSize1m += size;
      
      if (trade.side === 'buy') {
        takerBuyVolume1m += value;
      } else {
        takerSellVolume1m += value;
      }
    }
    
    const vwap1m = totalSize1m > 0 ? totalValue1m / totalSize1m : 0;
    const vwap5m = this.calculateVWAPFromCoinbaseTrades(recentTrades5m);
    
    const slippageEstimate = this.estimateSlippageFromCoinbaseTrades(recentTrades1m, 10000);
    const marketImpact = this.calculateMarketImpactFromCoinbaseTrades(recentTrades1m);
    const abnormalPattern = this.detectAbnormalPatternFromCoinbaseTrades(recentTrades1m, pair);
    
    const totalTakerVolume = takerBuyVolume1m + takerSellVolume1m;
    const orderFlowImbalance = totalTakerVolume > 0 
      ? (takerBuyVolume1m - takerSellVolume1m) / totalTakerVolume
      : 0;

    return {
      exchange: 'coinbase',
      pair,
      taker_buy_volume_1m: takerBuyVolume1m,
      taker_sell_volume_1m: takerSellVolume1m,
      maker_volume_1m: 0, // Coinbase doesn't easily distinguish maker volume
      vwap_1m: vwap1m,
      vwap_5m: vwap5m,
      slippage_estimate: slippageEstimate,
      trade_latency_ms: apiLatency,
      abnormal_pattern_flag: abnormalPattern,
      market_impact: marketImpact,
      order_flow_imbalance: orderFlowImbalance
    };
  }

  /**
   * Collect Kraken trade flow data
   */
  private async collectKrakenTradeData(): Promise<TradeData[]> {
    try {
      const pairs = ['XBTUSD', 'ETHUSD'];
      const tradeData: TradeData[] = [];

      for (const pair of pairs) {
        try {
          const latencyStart = Date.now();
          
          const tradesResponse = await axios.get('https://api.kraken.com/0/public/Trades', {
            params: { pair },
            timeout: 5000
          });
          
          const latencyEnd = Date.now();
          const apiLatency = latencyEnd - latencyStart;
          
          if (tradesResponse.data?.result?.[pair]) {
            const tradeInfo = this.processKrakenTrades(
              tradesResponse.data.result[pair],
              this.normalizeKrakenPair(pair),
              apiLatency
            );
            tradeData.push(tradeInfo);
          }
        } catch (error) {
          console.warn(`Kraken ${pair} trade collection error:`, error.message);
        }
      }

      return tradeData;
    } catch (error) {
      console.warn('Kraken trade collection error:', error.message);
      return [];
    }
  }

  /**
   * Process Kraken trade data
   */
  private processKrakenTrades(trades: any[], pair: string, apiLatency: number): TradeData {
    const now = Date.now() / 1000; // Kraken uses seconds
    const oneMinuteAgo = now - 60;
    const fiveMinutesAgo = now - 300;
    
    const recentTrades1m = trades.filter(trade => parseFloat(trade[2]) >= oneMinuteAgo);
    const recentTrades5m = trades.filter(trade => parseFloat(trade[2]) >= fiveMinutesAgo);
    
    let takerBuyVolume1m = 0;
    let takerSellVolume1m = 0;
    let totalValue1m = 0;
    let totalVolume1m = 0;
    
    for (const trade of recentTrades1m) {
      const price = parseFloat(trade[0]);
      const volume = parseFloat(trade[1]);
      const value = price * volume;
      const side = trade[3]; // 'b' for buy, 's' for sell
      
      totalValue1m += value;
      totalVolume1m += volume;
      
      if (side === 'b') {
        takerBuyVolume1m += value;
      } else {
        takerSellVolume1m += value;
      }
    }
    
    const vwap1m = totalVolume1m > 0 ? totalValue1m / totalVolume1m : 0;
    const vwap5m = this.calculateVWAPFromKrakenTrades(recentTrades5m);
    
    const slippageEstimate = this.estimateSlippageFromKrakenTrades(recentTrades1m, 10000);
    const marketImpact = this.calculateMarketImpactFromKrakenTrades(recentTrades1m);
    const abnormalPattern = this.detectAbnormalPatternFromKrakenTrades(recentTrades1m, pair);
    
    const totalTakerVolume = takerBuyVolume1m + takerSellVolume1m;
    const orderFlowImbalance = totalTakerVolume > 0 
      ? (takerBuyVolume1m - takerSellVolume1m) / totalTakerVolume
      : 0;

    return {
      exchange: 'kraken',
      pair,
      taker_buy_volume_1m: takerBuyVolume1m,
      taker_sell_volume_1m: takerSellVolume1m,
      maker_volume_1m: 0, // Kraken doesn't easily provide maker/taker distinction
      vwap_1m: vwap1m,
      vwap_5m: vwap5m,
      slippage_estimate: slippageEstimate,
      trade_latency_ms: apiLatency,
      abnormal_pattern_flag: abnormalPattern,
      market_impact: marketImpact,
      order_flow_imbalance: orderFlowImbalance
    };
  }

  /**
   * Calculate VWAP from trade data
   */
  private calculateVWAP(trades: any[]): number {
    if (trades.length === 0) return 0;
    
    let totalValue = 0;
    let totalQuantity = 0;
    
    for (const trade of trades) {
      const price = parseFloat(trade.p);
      const quantity = parseFloat(trade.q);
      
      totalValue += price * quantity;
      totalQuantity += quantity;
    }
    
    return totalQuantity > 0 ? totalValue / totalQuantity : 0;
  }

  /**
   * Calculate VWAP from Coinbase trades
   */
  private calculateVWAPFromCoinbaseTrades(trades: any[]): number {
    if (trades.length === 0) return 0;
    
    let totalValue = 0;
    let totalSize = 0;
    
    for (const trade of trades) {
      const price = parseFloat(trade.price);
      const size = parseFloat(trade.size);
      
      totalValue += price * size;
      totalSize += size;
    }
    
    return totalSize > 0 ? totalValue / totalSize : 0;
  }

  /**
   * Calculate VWAP from Kraken trades
   */
  private calculateVWAPFromKrakenTrades(trades: any[]): number {
    if (trades.length === 0) return 0;
    
    let totalValue = 0;
    let totalVolume = 0;
    
    for (const trade of trades) {
      const price = parseFloat(trade[0]);
      const volume = parseFloat(trade[1]);
      
      totalValue += price * volume;
      totalVolume += volume;
    }
    
    return totalVolume > 0 ? totalValue / totalVolume : 0;
  }

  /**
   * Estimate slippage from recent trades
   */
  private estimateSlippageFromTrades(trades: any[], orderSizeUSD: number): number {
    if (trades.length === 0) return 0.01; // 1% default estimate
    
    // Sort trades by price to simulate orderbook
    const sortedTrades = [...trades].sort((a, b) => parseFloat(a.p) - parseFloat(b.p));
    
    let accumulatedValue = 0;
    let totalQuantity = 0;
    let weightedPrice = 0;
    
    for (const trade of sortedTrades) {
      const price = parseFloat(trade.p);
      const quantity = parseFloat(trade.q);
      const value = price * quantity;
      
      if (accumulatedValue + value <= orderSizeUSD) {
        accumulatedValue += value;
        totalQuantity += quantity;
        weightedPrice += price * quantity;
      } else {
        const remainingValue = orderSizeUSD - accumulatedValue;
        const remainingQuantity = remainingValue / price;
        totalQuantity += remainingQuantity;
        weightedPrice += price * remainingQuantity;
        break;
      }
    }
    
    if (totalQuantity === 0) return 0.01;
    
    const avgExecutionPrice = weightedPrice / totalQuantity;
    const firstTradePrice = parseFloat(sortedTrades[0].p);
    
    const slippage = Math.abs(avgExecutionPrice - firstTradePrice) / firstTradePrice;
    
    return Math.min(0.1, slippage); // Cap at 10%
  }

  /**
   * Estimate slippage from Coinbase trades
   */
  private estimateSlippageFromCoinbaseTrades(trades: any[], orderSizeUSD: number): number {
    if (trades.length === 0) return 0.01;
    
    const sortedTrades = [...trades].sort((a, b) => parseFloat(a.price) - parseFloat(b.price));
    
    let accumulatedValue = 0;
    let totalSize = 0;
    let weightedPrice = 0;
    
    for (const trade of sortedTrades) {
      const price = parseFloat(trade.price);
      const size = parseFloat(trade.size);
      const value = price * size;
      
      if (accumulatedValue + value <= orderSizeUSD) {
        accumulatedValue += value;
        totalSize += size;
        weightedPrice += price * size;
      } else {
        const remainingValue = orderSizeUSD - accumulatedValue;
        const remainingSize = remainingValue / price;
        totalSize += remainingSize;
        weightedPrice += price * remainingSize;
        break;
      }
    }
    
    if (totalSize === 0) return 0.01;
    
    const avgExecutionPrice = weightedPrice / totalSize;
    const firstTradePrice = parseFloat(sortedTrades[0].price);
    
    return Math.min(0.1, Math.abs(avgExecutionPrice - firstTradePrice) / firstTradePrice);
  }

  /**
   * Estimate slippage from Kraken trades
   */
  private estimateSlippageFromKrakenTrades(trades: any[], orderSizeUSD: number): number {
    if (trades.length === 0) return 0.01;
    
    const sortedTrades = [...trades].sort((a, b) => parseFloat(a[0]) - parseFloat(b[0]));
    
    let accumulatedValue = 0;
    let totalVolume = 0;
    let weightedPrice = 0;
    
    for (const trade of sortedTrades) {
      const price = parseFloat(trade[0]);
      const volume = parseFloat(trade[1]);
      const value = price * volume;
      
      if (accumulatedValue + value <= orderSizeUSD) {
        accumulatedValue += value;
        totalVolume += volume;
        weightedPrice += price * volume;
      } else {
        const remainingValue = orderSizeUSD - accumulatedValue;
        const remainingVolume = remainingValue / price;
        totalVolume += remainingVolume;
        weightedPrice += price * remainingVolume;
        break;
      }
    }
    
    if (totalVolume === 0) return 0.01;
    
    const avgExecutionPrice = weightedPrice / totalVolume;
    const firstTradePrice = parseFloat(sortedTrades[0][0]);
    
    return Math.min(0.1, Math.abs(avgExecutionPrice - firstTradePrice) / firstTradePrice);
  }

  /**
   * Calculate market impact from trades
   */
  private calculateMarketImpact(trades: any[]): number {
    if (trades.length < 2) return 0;
    
    const prices = trades.map(trade => parseFloat(trade.p));
    const volumes = trades.map(trade => parseFloat(trade.q));
    
    // Calculate price volatility per unit volume
    let priceChange = 0;
    let totalVolume = 0;
    
    for (let i = 1; i < prices.length; i++) {
      priceChange += Math.abs(prices[i] - prices[i-1]) / prices[i-1];
      totalVolume += volumes[i];
    }
    
    return totalVolume > 0 ? priceChange / totalVolume : 0;
  }

  /**
   * Calculate market impact from Coinbase trades
   */
  private calculateMarketImpactFromCoinbaseTrades(trades: any[]): number {
    if (trades.length < 2) return 0;
    
    const prices = trades.map(trade => parseFloat(trade.price));
    const sizes = trades.map(trade => parseFloat(trade.size));
    
    let priceChange = 0;
    let totalSize = 0;
    
    for (let i = 1; i < prices.length; i++) {
      priceChange += Math.abs(prices[i] - prices[i-1]) / prices[i-1];
      totalSize += sizes[i];
    }
    
    return totalSize > 0 ? priceChange / totalSize : 0;
  }

  /**
   * Calculate market impact from Kraken trades
   */
  private calculateMarketImpactFromKrakenTrades(trades: any[]): number {
    if (trades.length < 2) return 0;
    
    const prices = trades.map(trade => parseFloat(trade[0]));
    const volumes = trades.map(trade => parseFloat(trade[1]));
    
    let priceChange = 0;
    let totalVolume = 0;
    
    for (let i = 1; i < prices.length; i++) {
      priceChange += Math.abs(prices[i] - prices[i-1]) / prices[i-1];
      totalVolume += volumes[i];
    }
    
    return totalVolume > 0 ? priceChange / totalVolume : 0;
  }

  /**
   * Detect abnormal trading patterns
   */
  private detectAbnormalPattern(trades: any[], pair: string, exchange: string): boolean {
    if (trades.length === 0) return false;
    
    const key = `${exchange}_${pair}`;
    const historicalTrades = this.tradeHistory.get(key) || [];
    
    // Store current trades for future reference
    this.tradeHistory.set(key, [...historicalTrades, ...trades].slice(-1000)); // Keep last 1000 trades
    
    if (historicalTrades.length < 10) return false; // Need historical data
    
    // Calculate current volume
    const currentVolume = trades.reduce((sum, trade) => 
      sum + (parseFloat(trade.p) * parseFloat(trade.q)), 0
    );
    
    // Calculate historical average volume
    const historicalVolumes = this.groupTradesByMinute(historicalTrades);
    if (historicalVolumes.length === 0) return false;
    
    const avgHistoricalVolume = historicalVolumes.reduce((sum, vol) => sum + vol, 0) / historicalVolumes.length;
    
    // Check for volume spike
    const volumeRatio = avgHistoricalVolume > 0 ? currentVolume / avgHistoricalVolume : 1;
    
    return volumeRatio >= this.ABNORMAL_VOLUME_THRESHOLD;
  }

  /**
   * Detect abnormal patterns from Coinbase trades
   */
  private detectAbnormalPatternFromCoinbaseTrades(trades: any[], pair: string): boolean {
    if (trades.length === 0) return false;
    
    const currentVolume = trades.reduce((sum, trade) => 
      sum + (parseFloat(trade.price) * parseFloat(trade.size)), 0
    );
    
    // Simple heuristic: check if current volume is unusually high
    const avgTradeSize = currentVolume / trades.length;
    const largeTradeThreshold = 50000; // $50K
    
    return avgTradeSize > largeTradeThreshold;
  }

  /**
   * Detect abnormal patterns from Kraken trades
   */
  private detectAbnormalPatternFromKrakenTrades(trades: any[], pair: string): boolean {
    if (trades.length === 0) return false;
    
    const currentVolume = trades.reduce((sum, trade) => 
      sum + (parseFloat(trade[0]) * parseFloat(trade[1])), 0
    );
    
    const avgTradeSize = currentVolume / trades.length;
    const largeTradeThreshold = 50000;
    
    return avgTradeSize > largeTradeThreshold;
  }

  /**
   * Group trades by minute for historical analysis
   */
  private groupTradesByMinute(trades: any[]): number[] {
    const minuteVolumes: Record<number, number> = {};
    
    for (const trade of trades) {
      const timestamp = trade.T ? trade.T : Date.now(); // Binance format
      const minute = Math.floor(timestamp / 60000) * 60000;
      const volume = parseFloat(trade.p) * parseFloat(trade.q);
      
      minuteVolumes[minute] = (minuteVolumes[minute] || 0) + volume;
    }
    
    return Object.values(minuteVolumes);
  }

  /**
   * Calculate VWAP metrics for different time windows
   */
  private async calculateVWAPMetrics(): Promise<VWAPCalculation[]> {
    const vwapCalculations: VWAPCalculation[] = [];
    
    // For each exchange-pair combination, calculate VWAP metrics
    for (const exchange of this.EXCHANGES) {
      for (const pair of this.PAIRS) {
        try {
          const vwapCalc = await this.calculateExchangeVWAP(exchange, pair);
          if (vwapCalc) {
            vwapCalculations.push(vwapCalc);
            
            // Cache the calculation
            const key = `${exchange}_${pair}`;
            this.vwapCache.set(key, vwapCalc);
          }
        } catch (error) {
          console.warn(`VWAP calculation error for ${exchange} ${pair}:`, error.message);
        }
      }
    }
    
    return vwapCalculations;
  }

  /**
   * Calculate VWAP for specific exchange-pair
   */
  private async calculateExchangeVWAP(exchange: string, pair: string): Promise<VWAPCalculation | null> {
    // This is a simplified implementation
    // In production, this would use cached trade data or streaming calculations
    
    const key = `${exchange}_${pair}`;
    const historicalTrades = this.tradeHistory.get(key) || [];
    
    if (historicalTrades.length === 0) return null;
    
    const now = Date.now();
    const vwap1m = this.calculateVWAPForWindow(historicalTrades, now - 60000);
    const vwap5m = this.calculateVWAPForWindow(historicalTrades, now - 300000);
    const vwap15m = this.calculateVWAPForWindow(historicalTrades, now - 900000);
    
    // Get current price (last trade price)
    const lastTrade = historicalTrades[historicalTrades.length - 1];
    const currentPrice = lastTrade ? parseFloat(lastTrade.p || lastTrade.price || lastTrade[0]) : vwap1m;
    
    const priceDeviation = vwap1m > 0 ? (currentPrice - vwap1m) / vwap1m : 0;
    
    return {
      exchange,
      pair,
      vwap_1m: vwap1m,
      vwap_5m: vwap5m,
      vwap_15m: vwap15m,
      price_deviation: priceDeviation,
      volume_profile: [1, 1, 1] // Simplified volume profile
    };
  }

  /**
   * Calculate VWAP for specific time window
   */
  private calculateVWAPForWindow(trades: any[], cutoffTime: number): number {
    const relevantTrades = trades.filter(trade => {
      const tradeTime = trade.T || trade.time || (parseFloat(trade[2]) * 1000);
      return tradeTime >= cutoffTime;
    });
    
    return this.calculateVWAP(relevantTrades);
  }

  /**
   * Calculate slippage estimates for different order sizes
   */
  private async calculateSlippageEstimates(): Promise<SlippageEstimate[]> {
    const slippageEstimates: SlippageEstimate[] = [];
    
    for (const exchange of this.EXCHANGES) {
      for (const pair of this.PAIRS) {
        try {
          const slippageEst = await this.calculateExchangeSlippage(exchange, pair);
          if (slippageEst) {
            slippageEstimates.push(slippageEst);
            
            // Cache the estimate
            const key = `${exchange}_${pair}`;
            this.slippageCache.set(key, slippageEst);
          }
        } catch (error) {
          console.warn(`Slippage calculation error for ${exchange} ${pair}:`, error.message);
        }
      }
    }
    
    return slippageEstimates;
  }

  /**
   * Calculate slippage estimates for specific exchange-pair
   */
  private async calculateExchangeSlippage(exchange: string, pair: string): Promise<SlippageEstimate | null> {
    const key = `${exchange}_${pair}`;
    const historicalTrades = this.tradeHistory.get(key) || [];
    
    if (historicalTrades.length === 0) return null;
    
    const smallSlippage = this.estimateSlippageFromTrades(historicalTrades, this.SLIPPAGE_ORDER_SIZES[0]);
    const mediumSlippage = this.estimateSlippageFromTrades(historicalTrades, this.SLIPPAGE_ORDER_SIZES[1]);
    const largeSlippage = this.estimateSlippageFromTrades(historicalTrades, this.SLIPPAGE_ORDER_SIZES[2]);
    
    // Calculate liquidity score based on slippage
    const avgSlippage = (smallSlippage + mediumSlippage + largeSlippage) / 3;
    const liquidityScore = Math.max(0, 1 - (avgSlippage / 0.01)); // 1% slippage = 0 score
    
    return {
      exchange,
      pair,
      small_order_slippage: smallSlippage,
      medium_order_slippage: mediumSlippage,
      large_order_slippage: largeSlippage,
      liquidity_score: liquidityScore
    };
  }

  /**
   * Calculate derived trade features
   */
  private calculateFeatures(
    tradeData: TradeData[], 
    vwapCalculations: VWAPCalculation[], 
    slippageEstimates: SlippageEstimate[]
  ): TradeFeatures {
    if (tradeData.length === 0) {
      return {
        overall_taker_buy_ratio: 0.5,
        cross_exchange_vwap_deviation: 0,
        average_slippage_estimate: 0.01,
        trade_efficiency_score: 0,
        order_flow_pressure: 0,
        pattern_anomaly_score: 0
      };
    }
    
    // Overall taker buy ratio
    const totalTakerBuyVolume = tradeData.reduce((sum, t) => sum + t.taker_buy_volume_1m, 0);
    const totalTakerSellVolume = tradeData.reduce((sum, t) => sum + t.taker_sell_volume_1m, 0);
    const totalTakerVolume = totalTakerBuyVolume + totalTakerSellVolume;
    const overallTakerBuyRatio = totalTakerVolume > 0 ? totalTakerBuyVolume / totalTakerVolume : 0.5;
    
    // Cross-exchange VWAP deviation
    const vwapDeviations = vwapCalculations.map(v => Math.abs(v.price_deviation));
    const crossExchangeVwapDeviation = vwapDeviations.length > 0 
      ? vwapDeviations.reduce((sum, d) => sum + d, 0) / vwapDeviations.length
      : 0;
    
    // Average slippage estimate
    const slippageValues = slippageEstimates.map(s => s.medium_order_slippage);
    const averageSlippageEstimate = slippageValues.length > 0
      ? slippageValues.reduce((sum, s) => sum + s, 0) / slippageValues.length
      : 0.01;
    
    // Trade efficiency score (inverse of latency and slippage)
    const avgLatency = tradeData.reduce((sum, t) => sum + t.trade_latency_ms, 0) / tradeData.length;
    const latencyScore = Math.max(0, 1 - (avgLatency / this.MAX_TRADE_LATENCY_MS));
    const slippageScore = Math.max(0, 1 - (averageSlippageEstimate / 0.01));
    const tradeEfficiencyScore = (latencyScore + slippageScore) / 2;
    
    // Order flow pressure (weighted average of imbalances)
    const orderFlowPressure = tradeData.reduce((sum, t) => sum + t.order_flow_imbalance, 0) / tradeData.length;
    
    // Pattern anomaly score
    const anomalyCount = tradeData.filter(t => t.abnormal_pattern_flag).length;
    const patternAnomalyScore = anomalyCount / tradeData.length;
    
    return {
      overall_taker_buy_ratio: overallTakerBuyRatio,
      cross_exchange_vwap_deviation: crossExchangeVwapDeviation,
      average_slippage_estimate: averageSlippageEstimate,
      trade_efficiency_score: tradeEfficiencyScore,
      order_flow_pressure: orderFlowPressure,
      pattern_anomaly_score: patternAnomalyScore
    };
  }

  /**
   * Calculate key signal (trade quality and flow analysis)
   */
  private calculateKeySignal(features: TradeFeatures): number {
    // Weight different trade factors
    const weights = {
      trade_efficiency: 0.3,      // Execution quality (latency + slippage)
      order_flow: 0.25,           // Order flow pressure strength
      vwap_deviation: 0.2,        // Price discovery quality
      anomaly_detection: 0.15,    // Unusual pattern detection
      liquidity_quality: 0.1      // Overall liquidity assessment
    };
    
    // Trade efficiency signal (higher is better)
    const efficiencySignal = features.trade_efficiency_score;
    
    // Order flow signal (absolute value - we want flow, regardless of direction)
    const orderFlowSignal = Math.abs(features.order_flow_pressure);
    
    // VWAP deviation signal (lower is better, so invert)
    const vwapDeviationSignal = Math.max(0, 1 - (features.cross_exchange_vwap_deviation * 10));
    
    // Anomaly detection signal (inverted - fewer anomalies is better)
    const anomalySignal = Math.max(0, 1 - features.pattern_anomaly_score);
    
    // Liquidity quality (inverse of slippage)
    const liquiditySignal = Math.max(0, 1 - (features.average_slippage_estimate / 0.01));
    
    // Weighted composite
    const signal = (
      efficiencySignal * weights.trade_efficiency +
      orderFlowSignal * weights.order_flow +
      vwapDeviationSignal * weights.vwap_deviation +
      anomalySignal * weights.anomaly_detection +
      liquiditySignal * weights.liquidity_quality
    );
    
    return Math.max(0, Math.min(1, signal));
  }

  /**
   * Calculate confidence based on data quality
   */
  private calculateDataConfidence(tradeData: TradeData[]): number {
    if (tradeData.length === 0) return 0;
    
    // Data completeness factor
    const expectedDataPoints = this.EXCHANGES.length * this.PAIRS.length;
    const completeness = Math.min(1, tradeData.length / expectedDataPoints);
    
    // Latency quality factor (lower latency = higher confidence)
    const avgLatency = tradeData.reduce((sum, t) => sum + t.trade_latency_ms, 0) / tradeData.length;
    const latencyFactor = Math.max(0.3, 1 - (avgLatency / 5000)); // 5 second max acceptable latency
    
    // Data freshness factor (always fresh for real-time trade data)
    const freshnessFactor = 1.0;
    
    // Volume adequacy factor
    const avgVolume = tradeData.reduce((sum, t) => sum + t.taker_buy_volume_1m + t.taker_sell_volume_1m, 0) / tradeData.length;
    const volumeFactor = Math.min(1, avgVolume / 10000); // $10K minimum for confidence
    
    return Math.max(0.1, completeness * latencyFactor * freshnessFactor * volumeFactor);
  }

  /**
   * Normalize pair name
   */
  private normalizePair(pair: string): string {
    return pair.toUpperCase()
      .replace('USDT', '-USDT')
      .replace(/^(.+?)USD$/, '$1-USD')
      .replace(/^(.+?)USDT$/, '$1-USDT');
  }

  /**
   * Normalize Kraken pair names
   */
  private normalizeKrakenPair(krakenPair: string): string {
    const mapping: Record<string, string> = {
      'XXBTZUSD': 'BTC-USD',
      'XETHZUSD': 'ETH-USD',
      'XBTUSD': 'BTC-USD',
      'ETHUSD': 'ETH-USD'
    };
    
    return mapping[krakenPair] || krakenPair;
  }

  /**
   * Get trade summary for debugging
   */
  getTradeSummary(): string {
    const totalTradeHistoryPoints = Array.from(this.tradeHistory.values())
      .reduce((sum, trades) => sum + trades.length, 0);
    
    return `${this.tradeHistory.size} exchange-pairs tracked, ${totalTradeHistoryPoints} historical trades`;
  }
}

/**
 * Factory function to create TradeAgent with config
 */
export function createTradeAgent(): TradeAgent {
  const config: AgentConfig = {
    name: 'trade',
    enabled: true,
    polling_interval_ms: 30 * 1000, // 30 seconds
    confidence_min: 0.3,
    data_age_max_ms: 2 * 60 * 1000, // 2 minutes max age
    retry_attempts: 3,
    retry_backoff_ms: 2000
  };

  return new TradeAgent(config);
}

// Export for testing
export { TradeAgent as default };