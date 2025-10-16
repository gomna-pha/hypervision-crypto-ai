/**
 * Volume Agent - Liquidity and Volume Dynamics Analysis
 * Analyzes trading volume patterns, liquidity metrics, and market depth
 * Detects volume spikes, dry-ups, and unusual trading activity
 */

import { BaseAgent, AgentConfig, AgentOutput } from '../../core/base-agent.js';
import axios from 'axios';

export interface VolumeData {
  pair: string;
  exchange: string;
  volume_1m: number;           // Base asset volume in 1 minute
  quote_volume_1m: number;     // Quote currency volume in USD
  trade_count_1m: number;      // Number of trades in 1 minute
  liquidity_index: number;     // Orderbook depth relative to threshold
  buy_sell_volume_ratio: number; // Ratio of buy vs sell volume
  volume_spike_flag: boolean;  // Whether current volume is unusually high
  avg_trade_size: number;      // Average trade size
  large_trade_count: number;   // Count of unusually large trades
}

export interface VolumeFeatures {
  overall_liquidity_index: number;    // Combined liquidity across exchanges
  volume_momentum: number;            // Rate of volume change
  liquidity_asymmetry: number;       // Imbalance between exchanges
  market_depth_score: number;        // Overall market depth quality
  volume_concentration: number;       // How concentrated volume is
  activity_intensity: number;         // Trading activity intensity
}

export interface LiquidityMetrics {
  exchange: string;
  pair: string;
  bid_depth_usd: number;        // USD value at various spread levels
  ask_depth_usd: number;
  bid_depth_levels: number[];   // Depth at 0.1%, 0.5%, 1% spreads
  ask_depth_levels: number[];
  spread_impact: number[];      // Price impact for various order sizes
}

export class VolumeAgent extends BaseAgent {
  private volumeHistory: Map<string, VolumeData[]> = new Map();
  private liquidityCache: Map<string, LiquidityMetrics> = new Map();
  private lastVolumeUpdate: number = 0;
  
  // Configuration
  private readonly EXCHANGES = ['binance', 'coinbase', 'kraken'];
  private readonly PAIRS = ['BTC-USDT', 'ETH-USDT', 'BTC-USD', 'ETH-USD'];
  private readonly VOLUME_SPIKE_THRESHOLD = 2.0; // 2x normal volume
  private readonly MIN_LIQUIDITY_USD = 100000;
  private readonly LARGE_TRADE_THRESHOLD = 50000; // USD value

  constructor(config: AgentConfig) {
    super(config);
  }

  protected async collectData(): Promise<AgentOutput> {
    const timestamp = this.getCurrentTimestamp();

    try {
      // Collect volume data from all exchanges
      const volumeDataPoints = await this.collectVolumeData();
      
      // Calculate liquidity metrics
      const liquidityMetrics = await this.calculateLiquidityMetrics();
      
      // Calculate derived features
      const features = this.calculateFeatures(volumeDataPoints, liquidityMetrics);
      
      // Calculate key signal (liquidity and volume quality score)
      const keySignal = this.calculateKeySignal(features);
      
      // Calculate confidence based on data completeness
      const confidence = this.calculateDataConfidence(volumeDataPoints);
      
      // Update historical data
      this.updateHistoricalData(volumeDataPoints);
      
      return {
        agent_name: 'VolumeAgent',
        timestamp,
        key_signal: keySignal,
        confidence,
        features: {
          ...features,
          volume_data_points: volumeDataPoints.length,
          liquidity_metrics_count: liquidityMetrics.length
        },
        metadata: {
          exchanges_tracked: this.EXCHANGES,
          pairs_tracked: this.PAIRS,
          historical_depth: this.getHistoricalDepth()
        }
      };

    } catch (error) {
      console.error('VolumeAgent data collection failed:', error);
      throw error;
    }
  }

  /**
   * Collect volume data from all exchanges
   */
  private async collectVolumeData(): Promise<VolumeData[]> {
    const volumePromises: Promise<VolumeData[]>[] = [];

    for (const exchange of this.EXCHANGES) {
      volumePromises.push(this.collectExchangeVolumeData(exchange));
    }

    const results = await Promise.allSettled(volumePromises);
    const volumeData: VolumeData[] = [];

    for (const result of results) {
      if (result.status === 'fulfilled' && result.value) {
        volumeData.push(...result.value);
      }
    }

    return volumeData;
  }

  /**
   * Collect volume data from specific exchange
   */
  private async collectExchangeVolumeData(exchange: string): Promise<VolumeData[]> {
    switch (exchange) {
      case 'binance':
        return await this.collectBinanceVolumeData();
      case 'coinbase':
        return await this.collectCoinbaseVolumeData();
      case 'kraken':
        return await this.collectKrakenVolumeData();
      default:
        return [];
    }
  }

  /**
   * Collect Binance volume data
   */
  private async collectBinanceVolumeData(): Promise<VolumeData[]> {
    try {
      const symbols = ['BTCUSDT', 'ETHUSDT'];
      const volumeData: VolumeData[] = [];

      // Get 24hr ticker statistics
      const tickerResponse = await axios.get('https://api.binance.com/api/v3/ticker/24hr', {
        timeout: 5000
      });

      // Get recent trades for volume analysis
      for (const symbol of symbols) {
        try {
          const tradesResponse = await axios.get('https://api.binance.com/api/v3/aggTrades', {
            params: { symbol, limit: 1000 },
            timeout: 5000
          });

          const ticker = tickerResponse.data.find((t: any) => t.symbol === symbol);
          if (ticker && tradesResponse.data) {
            const volumeInfo = this.processBinanceTrades(
              tradesResponse.data,
              ticker,
              this.normalizePair(symbol)
            );
            volumeData.push(volumeInfo);
          }
        } catch (error) {
          console.warn(`Binance ${symbol} volume collection error:`, error.message);
        }
      }

      return volumeData;
    } catch (error) {
      console.warn('Binance volume collection error:', error.message);
      return [];
    }
  }

  /**
   * Process Binance trades for volume analysis
   */
  private processBinanceTrades(trades: any[], ticker: any, pair: string): VolumeData {
    const now = Date.now();
    const oneMinuteAgo = now - 60000;
    
    const recentTrades = trades.filter(trade => trade.T >= oneMinuteAgo);
    
    let volume1m = 0;
    let quoteVolume1m = 0;
    let buyVolume = 0;
    let sellVolume = 0;
    let tradeCount1m = recentTrades.length;
    let largeTrades = 0;
    
    for (const trade of recentTrades) {
      const qty = parseFloat(trade.q);
      const price = parseFloat(trade.p);
      const quoteQty = qty * price;
      
      volume1m += qty;
      quoteVolume1m += quoteQty;
      
      if (trade.m) { // Buyer is maker (sell order)
        sellVolume += quoteQty;
      } else { // Buyer is taker (buy order)
        buyVolume += quoteQty;
      }
      
      if (quoteQty >= this.LARGE_TRADE_THRESHOLD) {
        largeTrades++;
      }
    }

    const avgTradeSize = tradeCount1m > 0 ? quoteVolume1m / tradeCount1m : 0;
    const buySellRatio = sellVolume > 0 ? buyVolume / sellVolume : 1;
    
    // Calculate volume spike detection
    const historical = this.getHistoricalVolume('binance', pair);
    const avgHistoricalVolume = this.calculateAverageVolume(historical);
    const volumeSpike = avgHistoricalVolume > 0 && 
                       (quoteVolume1m / avgHistoricalVolume) >= this.VOLUME_SPIKE_THRESHOLD;

    // Calculate liquidity index from orderbook depth (simplified)
    const liquidityIndex = this.estimateLiquidityIndex(parseFloat(ticker.count), quoteVolume1m);

    return {
      pair,
      exchange: 'binance',
      volume_1m: volume1m,
      quote_volume_1m: quoteVolume1m,
      trade_count_1m: tradeCount1m,
      liquidity_index: liquidityIndex,
      buy_sell_volume_ratio: buySellRatio,
      volume_spike_flag: volumeSpike,
      avg_trade_size: avgTradeSize,
      large_trade_count: largeTrades
    };
  }

  /**
   * Collect Coinbase volume data
   */
  private async collectCoinbaseVolumeData(): Promise<VolumeData[]> {
    try {
      const products = ['BTC-USD', 'ETH-USD'];
      const volumeData: VolumeData[] = [];

      for (const product of products) {
        try {
          // Get 24hr stats
          const statsResponse = await axios.get(`https://api.exchange.coinbase.com/products/${product}/stats`, {
            timeout: 5000
          });

          // Get recent trades
          const tradesResponse = await axios.get(`https://api.exchange.coinbase.com/products/${product}/trades`, {
            params: { limit: 100 },
            timeout: 5000
          });

          if (statsResponse.data && tradesResponse.data) {
            const volumeInfo = this.processCoinbaseTrades(
              tradesResponse.data,
              statsResponse.data,
              this.normalizePair(product)
            );
            volumeData.push(volumeInfo);
          }
        } catch (error) {
          console.warn(`Coinbase ${product} volume collection error:`, error.message);
        }
      }

      return volumeData;
    } catch (error) {
      console.warn('Coinbase volume collection error:', error.message);
      return [];
    }
  }

  /**
   * Process Coinbase trades for volume analysis
   */
  private processCoinbaseTrades(trades: any[], stats: any, pair: string): VolumeData {
    const now = new Date();
    const oneMinuteAgo = new Date(now.getTime() - 60000);
    
    const recentTrades = trades.filter(trade => 
      new Date(trade.time) >= oneMinuteAgo
    );
    
    let volume1m = 0;
    let quoteVolume1m = 0;
    let buyVolume = 0;
    let sellVolume = 0;
    let largeTrades = 0;
    
    for (const trade of recentTrades) {
      const size = parseFloat(trade.size);
      const price = parseFloat(trade.price);
      const quoteSize = size * price;
      
      volume1m += size;
      quoteVolume1m += quoteSize;
      
      if (trade.side === 'buy') {
        buyVolume += quoteSize;
      } else {
        sellVolume += quoteSize;
      }
      
      if (quoteSize >= this.LARGE_TRADE_THRESHOLD) {
        largeTrades++;
      }
    }

    const avgTradeSize = recentTrades.length > 0 ? quoteVolume1m / recentTrades.length : 0;
    const buySellRatio = sellVolume > 0 ? buyVolume / sellVolume : 1;
    
    // Volume spike detection
    const dailyVolume = parseFloat(stats.volume) / (24 * 60); // Convert to per-minute
    const volumeSpike = dailyVolume > 0 && (quoteVolume1m / dailyVolume) >= this.VOLUME_SPIKE_THRESHOLD;
    
    // Estimate liquidity index
    const liquidityIndex = this.estimateLiquidityIndex(recentTrades.length, quoteVolume1m);

    return {
      pair,
      exchange: 'coinbase',
      volume_1m: volume1m,
      quote_volume_1m: quoteVolume1m,
      trade_count_1m: recentTrades.length,
      liquidity_index: liquidityIndex,
      buy_sell_volume_ratio: buySellRatio,
      volume_spike_flag: volumeSpike,
      avg_trade_size: avgTradeSize,
      large_trade_count: largeTrades
    };
  }

  /**
   * Collect Kraken volume data
   */
  private async collectKrakenVolumeData(): Promise<VolumeData[]> {
    try {
      const pairs = ['XBTUSD', 'ETHUSD'];
      const volumeData: VolumeData[] = [];

      // Get ticker information
      const tickerResponse = await axios.get('https://api.kraken.com/0/public/Ticker', {
        params: { pair: pairs.join(',') },
        timeout: 5000
      });

      if (tickerResponse.data?.result) {
        for (const [krakenPair, data] of Object.entries(tickerResponse.data.result)) {
          const normalizedPair = this.normalizeKrakenPair(krakenPair);
          
          // Kraken doesn't provide 1-minute volume directly, so estimate from 24hr
          const dailyVolume = parseFloat((data as any).v[1]); // 24hr volume
          const volume1m = dailyVolume / (24 * 60); // Rough estimate
          
          // Get recent trades for more detailed analysis
          try {
            const tradesResponse = await axios.get('https://api.kraken.com/0/public/Trades', {
              params: { pair: krakenPair },
              timeout: 5000
            });

            if (tradesResponse.data?.result?.[krakenPair]) {
              const volumeInfo = this.processKrakenTrades(
                tradesResponse.data.result[krakenPair],
                data as any,
                normalizedPair
              );
              volumeData.push(volumeInfo);
            }
          } catch (error) {
            console.warn(`Kraken ${krakenPair} trades error:`, error.message);
            
            // Fallback to ticker data only
            const volumeInfo = this.createKrakenVolumeFromTicker(data as any, normalizedPair);
            volumeData.push(volumeInfo);
          }
        }
      }

      return volumeData;
    } catch (error) {
      console.warn('Kraken volume collection error:', error.message);
      return [];
    }
  }

  /**
   * Process Kraken trades for volume analysis
   */
  private processKrakenTrades(trades: any[], ticker: any, pair: string): VolumeData {
    const now = Date.now() / 1000; // Kraken uses seconds
    const oneMinuteAgo = now - 60;
    
    const recentTrades = trades.filter(trade => trade[2] >= oneMinuteAgo);
    
    let volume1m = 0;
    let quoteVolume1m = 0;
    let largeTrades = 0;
    
    for (const trade of recentTrades) {
      const price = parseFloat(trade[0]);
      const size = parseFloat(trade[1]);
      const quoteSize = size * price;
      
      volume1m += size;
      quoteVolume1m += quoteSize;
      
      if (quoteSize >= this.LARGE_TRADE_THRESHOLD) {
        largeTrades++;
      }
    }

    const avgTradeSize = recentTrades.length > 0 ? quoteVolume1m / recentTrades.length : 0;
    
    // Volume spike detection (use 24hr volume as baseline)
    const dailyVolume = parseFloat(ticker.v[1]) / (24 * 60);
    const volumeSpike = dailyVolume > 0 && (volume1m / dailyVolume) >= this.VOLUME_SPIKE_THRESHOLD;
    
    // Estimate liquidity index
    const liquidityIndex = this.estimateLiquidityIndex(recentTrades.length, quoteVolume1m);

    return {
      pair,
      exchange: 'kraken',
      volume_1m: volume1m,
      quote_volume_1m: quoteVolume1m,
      trade_count_1m: recentTrades.length,
      liquidity_index: liquidityIndex,
      buy_sell_volume_ratio: 1.0, // Kraken doesn't provide taker side info easily
      volume_spike_flag: volumeSpike,
      avg_trade_size: avgTradeSize,
      large_trade_count: largeTrades
    };
  }

  /**
   * Create Kraken volume data from ticker only (fallback)
   */
  private createKrakenVolumeFromTicker(ticker: any, pair: string): VolumeData {
    const dailyVolume = parseFloat(ticker.v[1]) / (24 * 60); // Estimate 1-minute volume
    const dailyTrades = parseInt(ticker.t[1]) / (24 * 60); // Estimate 1-minute trades
    
    return {
      pair,
      exchange: 'kraken',
      volume_1m: dailyVolume,
      quote_volume_1m: dailyVolume * parseFloat(ticker.c[0]), // Estimate using current price
      trade_count_1m: Math.round(dailyTrades),
      liquidity_index: this.estimateLiquidityIndex(dailyTrades, dailyVolume),
      buy_sell_volume_ratio: 1.0,
      volume_spike_flag: false, // Can't detect without historical data
      avg_trade_size: dailyTrades > 0 ? dailyVolume / dailyTrades : 0,
      large_trade_count: 0
    };
  }

  /**
   * Calculate liquidity metrics for all exchange-pair combinations
   */
  private async calculateLiquidityMetrics(): Promise<LiquidityMetrics[]> {
    const liquidityMetrics: LiquidityMetrics[] = [];
    
    // For each exchange-pair combination, calculate orderbook depth metrics
    for (const exchange of this.EXCHANGES) {
      for (const pair of this.PAIRS) {
        try {
          const metrics = await this.calculateExchangeLiquidityMetrics(exchange, pair);
          if (metrics) {
            liquidityMetrics.push(metrics);
          }
        } catch (error) {
          console.warn(`Liquidity metrics error for ${exchange} ${pair}:`, error.message);
        }
      }
    }
    
    return liquidityMetrics;
  }

  /**
   * Calculate liquidity metrics for specific exchange-pair
   */
  private async calculateExchangeLiquidityMetrics(exchange: string, pair: string): Promise<LiquidityMetrics | null> {
    try {
      let orderbook: any = null;
      
      switch (exchange) {
        case 'binance':
          orderbook = await this.getBinanceOrderbook(this.denormalizePair(pair, 'binance'));
          break;
        case 'coinbase':
          orderbook = await this.getCoinbaseOrderbook(this.denormalizePair(pair, 'coinbase'));
          break;
        case 'kraken':
          orderbook = await this.getKrakenOrderbook(this.denormalizePair(pair, 'kraken'));
          break;
      }
      
      if (!orderbook) return null;
      
      return this.processOrderbookForLiquidity(orderbook, exchange, pair);
      
    } catch (error) {
      console.warn(`Failed to get orderbook for ${exchange} ${pair}:`, error.message);
      return null;
    }
  }

  /**
   * Get Binance orderbook
   */
  private async getBinanceOrderbook(symbol: string): Promise<any> {
    const response = await axios.get('https://api.binance.com/api/v3/depth', {
      params: { symbol, limit: 100 },
      timeout: 5000
    });
    return response.data;
  }

  /**
   * Get Coinbase orderbook
   */
  private async getCoinbaseOrderbook(productId: string): Promise<any> {
    const response = await axios.get(`https://api.exchange.coinbase.com/products/${productId}/book`, {
      params: { level: 2 },
      timeout: 5000
    });
    return response.data;
  }

  /**
   * Get Kraken orderbook
   */
  private async getKrakenOrderbook(pair: string): Promise<any> {
    const response = await axios.get('https://api.kraken.com/0/public/Depth', {
      params: { pair, count: 100 },
      timeout: 5000
    });
    return response.data?.result?.[Object.keys(response.data.result)[0]];
  }

  /**
   * Process orderbook data to calculate liquidity metrics
   */
  private processOrderbookForLiquidity(orderbook: any, exchange: string, pair: string): LiquidityMetrics {
    const bids = orderbook.bids || orderbook.b || [];
    const asks = orderbook.asks || orderbook.a || [];
    
    if (bids.length === 0 || asks.length === 0) {
      return {
        exchange,
        pair,
        bid_depth_usd: 0,
        ask_depth_usd: 0,
        bid_depth_levels: [0, 0, 0],
        ask_depth_levels: [0, 0, 0],
        spread_impact: [0, 0, 0]
      };
    }
    
    const bestBid = parseFloat(bids[0][0]);
    const bestAsk = parseFloat(asks[0][0]);
    const midPrice = (bestBid + bestAsk) / 2;
    
    // Calculate depth at different spread levels (0.1%, 0.5%, 1.0%)
    const spreadLevels = [0.001, 0.005, 0.01];
    const bidDepthLevels: number[] = [];
    const askDepthLevels: number[] = [];
    const spreadImpact: number[] = [];
    
    for (const spreadLevel of spreadLevels) {
      const bidPriceLevel = bestBid * (1 - spreadLevel);
      const askPriceLevel = bestAsk * (1 + spreadLevel);
      
      let bidDepth = 0;
      let askDepth = 0;
      
      // Calculate bid depth
      for (const [price, size] of bids) {
        const priceNum = parseFloat(price);
        if (priceNum >= bidPriceLevel) {
          bidDepth += priceNum * parseFloat(size);
        } else {
          break;
        }
      }
      
      // Calculate ask depth
      for (const [price, size] of asks) {
        const priceNum = parseFloat(price);
        if (priceNum <= askPriceLevel) {
          askDepth += priceNum * parseFloat(size);
        } else {
          break;
        }
      }
      
      bidDepthLevels.push(bidDepth);
      askDepthLevels.push(askDepth);
      
      // Calculate spread impact (how much price moves with $10k order)
      const testOrderSize = 10000; // $10k
      const impact = this.calculatePriceImpact(bids, asks, testOrderSize, midPrice);
      spreadImpact.push(impact);
    }
    
    return {
      exchange,
      pair,
      bid_depth_usd: bidDepthLevels[1], // 0.5% spread level
      ask_depth_usd: askDepthLevels[1],
      bid_depth_levels: bidDepthLevels,
      ask_depth_levels: askDepthLevels,
      spread_impact: spreadImpact
    };
  }

  /**
   * Calculate price impact for given order size
   */
  private calculatePriceImpact(bids: any[], asks: any[], orderSizeUSD: number, midPrice: number): number {
    // Simulate market buy order
    let remainingSize = orderSizeUSD;
    let totalCost = 0;
    let totalQuantity = 0;
    
    for (const [price, size] of asks) {
      const priceNum = parseFloat(price);
      const sizeNum = parseFloat(size);
      const levelValue = priceNum * sizeNum;
      
      if (remainingSize <= levelValue) {
        const quantityTaken = remainingSize / priceNum;
        totalCost += remainingSize;
        totalQuantity += quantityTaken;
        break;
      } else {
        totalCost += levelValue;
        totalQuantity += sizeNum;
        remainingSize -= levelValue;
      }
    }
    
    if (totalQuantity === 0) return 1; // 100% impact if no liquidity
    
    const avgExecutionPrice = totalCost / totalQuantity;
    const impact = Math.abs(avgExecutionPrice - midPrice) / midPrice;
    
    return Math.min(1, impact); // Cap at 100%
  }

  /**
   * Estimate liquidity index from trade data
   */
  private estimateLiquidityIndex(tradeCount: number, volume: number): number {
    // Simple heuristic: higher trade count and volume = better liquidity
    const tradeScore = Math.min(1, tradeCount / 100); // Normalize to 100 trades/minute
    const volumeScore = Math.min(1, volume / this.MIN_LIQUIDITY_USD);
    
    return (tradeScore + volumeScore) / 2;
  }

  /**
   * Calculate derived volume features
   */
  private calculateFeatures(volumeData: VolumeData[], liquidityMetrics: LiquidityMetrics[]): VolumeFeatures {
    if (volumeData.length === 0) {
      return {
        overall_liquidity_index: 0,
        volume_momentum: 0,
        liquidity_asymmetry: 0,
        market_depth_score: 0,
        volume_concentration: 0,
        activity_intensity: 0
      };
    }
    
    // Overall liquidity index (average across all data points)
    const overallLiquidityIndex = volumeData.reduce((sum, v) => sum + v.liquidity_index, 0) / volumeData.length;
    
    // Volume momentum (change from previous period)
    const volumeMomentum = this.calculateVolumeMomentum(volumeData);
    
    // Liquidity asymmetry (difference between best and worst exchange)
    const liquidityAsymmetry = this.calculateLiquidityAsymmetry(volumeData);
    
    // Market depth score (from orderbook analysis)
    const marketDepthScore = this.calculateMarketDepthScore(liquidityMetrics);
    
    // Volume concentration (how concentrated volume is across exchanges)
    const volumeConcentration = this.calculateVolumeConcentration(volumeData);
    
    // Activity intensity (trades per minute normalized)
    const totalTrades = volumeData.reduce((sum, v) => sum + v.trade_count_1m, 0);
    const activityIntensity = Math.min(1, totalTrades / (this.EXCHANGES.length * 100));
    
    return {
      overall_liquidity_index: overallLiquidityIndex,
      volume_momentum: volumeMomentum,
      liquidity_asymmetry: liquidityAsymmetry,
      market_depth_score: marketDepthScore,
      volume_concentration: volumeConcentration,
      activity_intensity: activityIntensity
    };
  }

  /**
   * Calculate volume momentum from historical data
   */
  private calculateVolumeMomentum(volumeData: VolumeData[]): number {
    const currentVolume = volumeData.reduce((sum, v) => sum + v.quote_volume_1m, 0);
    
    // Get historical average
    const historicalVolumes: number[] = [];
    for (const data of volumeData) {
      const key = `${data.exchange}_${data.pair}`;
      const history = this.volumeHistory.get(key) || [];
      if (history.length > 0) {
        const avgHistorical = history.reduce((sum, h) => sum + h.quote_volume_1m, 0) / history.length;
        historicalVolumes.push(avgHistorical);
      }
    }
    
    if (historicalVolumes.length === 0) return 0;
    
    const avgHistoricalVolume = historicalVolumes.reduce((sum, v) => sum + v, 0) / historicalVolumes.length;
    
    if (avgHistoricalVolume === 0) return 0;
    
    // Calculate momentum as percentage change
    const momentum = (currentVolume - avgHistoricalVolume) / avgHistoricalVolume;
    
    return Math.max(-1, Math.min(1, momentum)); // Clamp to [-1, 1]
  }

  /**
   * Calculate liquidity asymmetry between exchanges
   */
  private calculateLiquidityAsymmetry(volumeData: VolumeData[]): number {
    if (volumeData.length < 2) return 0;
    
    const liquidityIndices = volumeData.map(v => v.liquidity_index);
    const maxLiquidity = Math.max(...liquidityIndices);
    const minLiquidity = Math.min(...liquidityIndices);
    
    if (maxLiquidity === 0) return 0;
    
    return (maxLiquidity - minLiquidity) / maxLiquidity;
  }

  /**
   * Calculate market depth score from liquidity metrics
   */
  private calculateMarketDepthScore(liquidityMetrics: LiquidityMetrics[]): number {
    if (liquidityMetrics.length === 0) return 0;
    
    let totalDepthScore = 0;
    
    for (const metrics of liquidityMetrics) {
      const bidDepth = metrics.bid_depth_usd;
      const askDepth = metrics.ask_depth_usd;
      const avgDepth = (bidDepth + askDepth) / 2;
      
      // Normalize depth score
      const depthScore = Math.min(1, avgDepth / this.MIN_LIQUIDITY_USD);
      totalDepthScore += depthScore;
    }
    
    return totalDepthScore / liquidityMetrics.length;
  }

  /**
   * Calculate volume concentration (Herfindahl index)
   */
  private calculateVolumeConcentration(volumeData: VolumeData[]): number {
    const totalVolume = volumeData.reduce((sum, v) => sum + v.quote_volume_1m, 0);
    
    if (totalVolume === 0) return 1; // Maximum concentration if no volume
    
    let herfindahlIndex = 0;
    
    for (const data of volumeData) {
      const marketShare = data.quote_volume_1m / totalVolume;
      herfindahlIndex += marketShare * marketShare;
    }
    
    return herfindahlIndex; // Higher value = more concentrated
  }

  /**
   * Calculate key signal (volume and liquidity quality score)
   */
  private calculateKeySignal(features: VolumeFeatures): number {
    // Weight different volume factors
    const weights = {
      liquidity_index: 0.3,      // Overall liquidity availability
      market_depth_score: 0.25,  // Orderbook depth quality
      activity_intensity: 0.2,   // Trading activity level
      volume_momentum: 0.15,     // Volume trend direction
      liquidity_asymmetry: -0.1  // Penalty for asymmetric liquidity (negative weight)
    };
    
    const signal = (
      features.overall_liquidity_index * weights.liquidity_index +
      features.market_depth_score * weights.market_depth_score +
      features.activity_intensity * weights.activity_intensity +
      Math.abs(features.volume_momentum) * weights.volume_momentum +
      features.liquidity_asymmetry * weights.liquidity_asymmetry // This reduces the score
    );
    
    return Math.max(0, Math.min(1, signal));
  }

  /**
   * Calculate confidence based on data completeness
   */
  private calculateDataConfidence(volumeData: VolumeData[]): number {
    if (volumeData.length === 0) return 0;
    
    // Data completeness factor
    const expectedDataPoints = this.EXCHANGES.length * this.PAIRS.length;
    const completeness = Math.min(1, volumeData.length / expectedDataPoints);
    
    // Data quality factor (based on non-zero values)
    let qualityScore = 0;
    for (const data of volumeData) {
      let fieldScore = 0;
      const fields = ['volume_1m', 'quote_volume_1m', 'trade_count_1m', 'liquidity_index'];
      
      for (const field of fields) {
        if (data[field as keyof VolumeData] > 0) {
          fieldScore++;
        }
      }
      
      qualityScore += fieldScore / fields.length;
    }
    
    const avgQuality = qualityScore / volumeData.length;
    
    // Recency factor (always fresh for real-time data)
    const recencyFactor = 1.0;
    
    return Math.max(0.1, completeness * avgQuality * recencyFactor);
  }

  /**
   * Update historical volume data
   */
  private updateHistoricalData(volumeData: VolumeData[]): void {
    for (const data of volumeData) {
      const key = `${data.exchange}_${data.pair}`;
      
      if (!this.volumeHistory.has(key)) {
        this.volumeHistory.set(key, []);
      }
      
      const history = this.volumeHistory.get(key)!;
      history.push(data);
      
      // Keep only last 60 data points (1 hour of minute data)
      if (history.length > 60) {
        history.shift();
      }
    }
  }

  /**
   * Get historical volume for specific exchange-pair
   */
  private getHistoricalVolume(exchange: string, pair: string): VolumeData[] {
    const key = `${exchange}_${pair}`;
    return this.volumeHistory.get(key) || [];
  }

  /**
   * Calculate average volume from historical data
   */
  private calculateAverageVolume(historicalData: VolumeData[]): number {
    if (historicalData.length === 0) return 0;
    
    const totalVolume = historicalData.reduce((sum, data) => sum + data.quote_volume_1m, 0);
    return totalVolume / historicalData.length;
  }

  /**
   * Get historical data depth for metadata
   */
  private getHistoricalDepth(): Record<string, number> {
    const depth: Record<string, number> = {};
    
    for (const [key, history] of this.volumeHistory) {
      depth[key] = history.length;
    }
    
    return depth;
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
   * Denormalize pair for exchange APIs
   */
  private denormalizePair(pair: string, exchange: string): string {
    switch (exchange) {
      case 'binance':
        return pair.replace('-', '');
      case 'coinbase':
        return pair;
      case 'kraken':
        return pair.replace('BTC-', 'XBT').replace('-', '');
      default:
        return pair;
    }
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
   * Get volume summary for debugging
   */
  getVolumeSummary(): string {
    const totalEntries = Array.from(this.volumeHistory.values()).length;
    const totalHistoricalPoints = Array.from(this.volumeHistory.values())
      .reduce((sum, history) => sum + history.length, 0);
    
    return `${totalEntries} exchange-pairs tracked, ${totalHistoricalPoints} historical data points`;
  }
}

/**
 * Factory function to create VolumeAgent with config
 */
export function createVolumeAgent(): VolumeAgent {
  const config: AgentConfig = {
    name: 'volume',
    enabled: true,
    polling_interval_ms: 60 * 1000, // 1 minute
    confidence_min: 0.2,
    data_age_max_ms: 2 * 60 * 1000, // 2 minutes max age
    retry_attempts: 3,
    retry_backoff_ms: 2000
  };

  return new VolumeAgent(config);
}

// Export for testing
export { VolumeAgent as default };