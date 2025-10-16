/**
 * Volume Agent - Liquidity and Volume Dynamics Analysis
 * Analyzes volume patterns, liquidity metrics, and detects volume spikes/dry-ups
 * Provides insights into market liquidity conditions affecting arbitrage opportunities
 */

import { BaseAgent, AgentConfig, AgentOutput } from '../../core/base-agent.js';

export interface VolumeData {
  pair: string;
  volume_1m_base: number;           // Volume in base currency (BTC)
  volume_1m_quote: number;          // Volume in quote currency (USD/USDT)
  trade_count_1m: number;           // Number of trades in last minute
  avg_trade_size: number;           // Average trade size
  large_trade_count: number;        // Count of large trades (>$10k)
  buy_volume_1m: number;           // Buy-side volume
  sell_volume_1m: number;          // Sell-side volume
  timestamp: number;
}

export interface VolumeFeatures {
  liquidity_index: number;          // Normalized liquidity score [0, 1]
  buy_sell_volume_ratio: number;    // Ratio of buy to sell volume
  volume_spike_flag: number;        // Volume anomaly detection [0, 1]
  trade_size_momentum: number;      // Change in average trade size [-1, 1]
  market_depth_score: number;       // Aggregate orderbook depth score [0, 1]
  volume_concentration: number;      // How concentrated volume is across exchanges [0, 1]
}

interface ExchangeVolumeData {
  exchange: string;
  pair: string;
  volume_1m: number;
  orderbook_depth: number;
  timestamp: number;
}

export class VolumeAgent extends BaseAgent {
  private volumeData: Map<string, VolumeData> = new Map();
  private volumeHistory: Map<string, VolumeData[]> = new Map();
  private exchangeVolumes: Map<string, ExchangeVolumeData> = new Map();
  
  // Volume thresholds for anomaly detection
  private readonly LARGE_TRADE_THRESHOLD = 10000; // $10k USD
  private readonly VOLUME_SPIKE_MULTIPLIER = 3;   // 3x normal volume
  private readonly MIN_LIQUIDITY_THRESHOLD = 100000; // $100k
  
  constructor(config: AgentConfig) {
    super(config);
  }

  protected async collectData(): Promise<AgentOutput> {
    const timestamp = this.getCurrentTimestamp();

    try {
      // Collect volume data from multiple sources
      await this.collectVolumeMetrics();

      // Get aggregated volume data for primary pairs
      const btcVolumeData = this.getAggregatedVolumeData('BTC-USDT');
      const ethVolumeData = this.getAggregatedVolumeData('ETH-USDT');

      // Calculate volume features
      const features = this.calculateVolumeFeatures(btcVolumeData, ethVolumeData);

      // Calculate key signal (liquidity health score)
      const keySignal = features.liquidity_index;

      // Calculate confidence based on data quality and coverage
      const confidence = this.calculateVolumeConfidence([btcVolumeData, ethVolumeData]);

      return {
        agent_name: 'VolumeAgent',
        timestamp,
        key_signal: keySignal,
        confidence,
        features: {
          ...features,
          btc_volume_data: btcVolumeData ? {
            volume_1m_usd: btcVolumeData.volume_1m_quote,
            trade_count: btcVolumeData.trade_count_1m,
            buy_sell_ratio: btcVolumeData.buy_volume_1m / (btcVolumeData.sell_volume_1m || 1)
          } : null,
          eth_volume_data: ethVolumeData ? {
            volume_1m_usd: ethVolumeData.volume_1m_quote,
            trade_count: ethVolumeData.trade_count_1m,
            buy_sell_ratio: ethVolumeData.buy_volume_1m / (ethVolumeData.sell_volume_1m || 1)
          } : null
        },
        metadata: {
          tracked_pairs: this.volumeData.size,
          exchange_count: new Set(Array.from(this.exchangeVolumes.values()).map(v => v.exchange)).size,
          data_coverage_pct: Math.min(100, (this.exchangeVolumes.size / 6) * 100) // Target 6 exchange-pair combinations
        }
      };

    } catch (error) {
      console.error('VolumeAgent data collection failed:', error);
      throw error;
    }
  }

  /**
   * Collect volume metrics from various sources
   */
  private async collectVolumeMetrics(): Promise<void> {
    const pairs = ['BTC-USDT', 'ETH-USDT', 'BTC-USD', 'ETH-USD'];
    const exchanges = ['binance', 'coinbase', 'kraken'];

    // In production, this would pull real-time data from exchange APIs
    // For now, we'll simulate realistic volume data
    
    for (const exchange of exchanges) {
      for (const pair of pairs) {
        const exchangeVolumeData = await this.fetchExchangeVolumeData(exchange, pair);
        if (exchangeVolumeData) {
          this.exchangeVolumes.set(`${exchange}_${pair}`, exchangeVolumeData);
        }
      }
    }

    // Aggregate volume data by pair
    for (const pair of pairs) {
      const aggregatedData = this.aggregateVolumeByPair(pair);
      if (aggregatedData) {
        this.volumeData.set(pair, aggregatedData);
        this.updateVolumeHistory(pair, aggregatedData);
      }
    }
  }

  /**
   * Fetch volume data from a specific exchange (simulated for demo)
   */
  private async fetchExchangeVolumeData(exchange: string, pair: string): Promise<ExchangeVolumeData | null> {
    try {
      // Simulate API call with realistic data
      const baseVolume = this.getBaseVolumeForPair(pair);
      const exchangeMultiplier = this.getExchangeMultiplier(exchange);
      
      // Add some randomness to simulate market dynamics
      const randomFactor = 0.7 + (Math.random() * 0.6); // 0.7 to 1.3 multiplier
      
      const volume_1m = baseVolume * exchangeMultiplier * randomFactor;
      const orderbook_depth = volume_1m * 50; // Approximate depth relationship

      return {
        exchange,
        pair,
        volume_1m,
        orderbook_depth,
        timestamp: Date.now()
      };

    } catch (error) {
      console.warn(`Failed to fetch volume data from ${exchange} for ${pair}:`, error.message);
      return null;
    }
  }

  /**
   * Get base volume for a trading pair (realistic estimates)
   */
  private getBaseVolumeForPair(pair: string): number {
    const baseVolumes = {
      'BTC-USDT': 50000,   // $50k per minute
      'BTC-USD': 30000,    // $30k per minute
      'ETH-USDT': 25000,   // $25k per minute
      'ETH-USD': 15000     // $15k per minute
    };
    
    return baseVolumes[pair as keyof typeof baseVolumes] || 10000;
  }

  /**
   * Get exchange-specific volume multiplier
   */
  private getExchangeMultiplier(exchange: string): number {
    const multipliers = {
      'binance': 1.5,    // Highest volume
      'coinbase': 1.0,   // Base reference
      'kraken': 0.7      // Lower volume
    };
    
    return multipliers[exchange as keyof typeof multipliers] || 0.5;
  }

  /**
   * Aggregate volume data across exchanges for a pair
   */
  private aggregateVolumeByPair(pair: string): VolumeData | null {
    const exchangeData = Array.from(this.exchangeVolumes.values())
      .filter(data => this.normalizePair(data.pair) === this.normalizePair(pair));

    if (exchangeData.length === 0) return null;

    // Aggregate volume metrics
    const totalVolume = exchangeData.reduce((sum, data) => sum + data.volume_1m, 0);
    const totalDepth = exchangeData.reduce((sum, data) => sum + data.orderbook_depth, 0);
    
    // Simulate additional metrics (in production, calculate from real trade data)
    const tradeCount = Math.floor(totalVolume / 500) + Math.floor(Math.random() * 50);
    const avgTradeSize = totalVolume / (tradeCount || 1);
    const largeTradeCount = Math.floor(tradeCount * 0.05 * Math.random());
    
    // Simulate buy/sell split with some bias
    const buyBias = 0.45 + (Math.random() * 0.1); // 45-55% buy ratio
    const buyVolume = totalVolume * buyBias;
    const sellVolume = totalVolume * (1 - buyBias);

    return {
      pair,
      volume_1m_base: totalVolume / this.getEstimatedPrice(pair),
      volume_1m_quote: totalVolume,
      trade_count_1m: tradeCount,
      avg_trade_size: avgTradeSize,
      large_trade_count: largeTradeCount,
      buy_volume_1m: buyVolume,
      sell_volume_1m: sellVolume,
      timestamp: Date.now()
    };
  }

  /**
   * Get estimated price for volume calculations
   */
  private getEstimatedPrice(pair: string): number {
    const prices = {
      'BTC-USDT': 42000,
      'BTC-USD': 42000,
      'ETH-USDT': 2500,
      'ETH-USD': 2500
    };
    
    return prices[pair as keyof typeof prices] || 1;
  }

  /**
   * Update volume history for trend analysis
   */
  private updateVolumeHistory(pair: string, data: VolumeData): void {
    if (!this.volumeHistory.has(pair)) {
      this.volumeHistory.set(pair, []);
    }

    const history = this.volumeHistory.get(pair)!;
    history.push(data);

    // Keep only last 60 data points (1 hour of minute data)
    if (history.length > 60) {
      history.shift();
    }
  }

  /**
   * Get aggregated volume data for a specific pair
   */
  private getAggregatedVolumeData(pair: string): VolumeData | null {
    return this.volumeData.get(pair) || null;
  }

  /**
   * Calculate volume-related features
   */
  private calculateVolumeFeatures(btcData: VolumeData | null, ethData: VolumeData | null): VolumeFeatures {
    if (!btcData && !ethData) {
      return this.getDefaultVolumeFeatures();
    }

    const primaryData = btcData || ethData!;

    // Liquidity Index: based on total volume and depth
    const liquidityIndex = this.calculateLiquidityIndex(primaryData);

    // Buy/Sell Volume Ratio
    const buySellRatio = primaryData.sell_volume_1m > 0 
      ? primaryData.buy_volume_1m / primaryData.sell_volume_1m 
      : 1;
    const normalizedBuySellRatio = this.normalize(buySellRatio, 0.5, 2, false); // 0.5 to 2.0 range

    // Volume Spike Detection
    const volumeSpikeFlag = this.detectVolumeSpike(primaryData);

    // Trade Size Momentum
    const tradeSizeMomentum = this.calculateTradeSizeMomentum(primaryData);

    // Market Depth Score
    const marketDepthScore = this.calculateMarketDepthScore();

    // Volume Concentration across exchanges
    const volumeConcentration = this.calculateVolumeConcentration(primaryData.pair);

    return {
      liquidity_index: liquidityIndex,
      buy_sell_volume_ratio: normalizedBuySellRatio,
      volume_spike_flag: volumeSpikeFlag,
      trade_size_momentum: tradeSizeMomentum,
      market_depth_score: marketDepthScore,
      volume_concentration: volumeConcentration
    };
  }

  /**
   * Calculate liquidity index based on volume and depth
   */
  private calculateLiquidityIndex(data: VolumeData): number {
    // Combine volume and trade count for liquidity assessment
    const volumeScore = Math.min(1, data.volume_1m_quote / 100000); // Normalize to $100k
    const tradeCountScore = Math.min(1, data.trade_count_1m / 100); // Normalize to 100 trades
    const avgTradeSizeScore = Math.min(1, data.avg_trade_size / 1000); // Normalize to $1k

    // Weighted combination
    return (volumeScore * 0.5) + (tradeCountScore * 0.3) + (avgTradeSizeScore * 0.2);
  }

  /**
   * Detect volume spikes using historical comparison
   */
  private detectVolumeSpike(data: VolumeData): number {
    const history = this.volumeHistory.get(data.pair);
    
    if (!history || history.length < 10) {
      return 0; // Not enough history
    }

    // Calculate average volume over last 10 periods (excluding current)
    const recentHistory = history.slice(-11, -1); // Last 10, excluding current
    const avgVolume = recentHistory.reduce((sum, h) => sum + h.volume_1m_quote, 0) / recentHistory.length;

    if (avgVolume === 0) return 0;

    // Check if current volume is significantly higher
    const volumeMultiplier = data.volume_1m_quote / avgVolume;
    
    if (volumeMultiplier >= this.VOLUME_SPIKE_MULTIPLIER) {
      return 1; // Clear spike detected
    } else if (volumeMultiplier >= this.VOLUME_SPIKE_MULTIPLIER * 0.7) {
      return 0.5; // Moderate spike
    }

    return 0; // No spike
  }

  /**
   * Calculate trade size momentum
   */
  private calculateTradeSizeMomentum(data: VolumeData): number {
    const history = this.volumeHistory.get(data.pair);
    
    if (!history || history.length < 2) return 0;

    const previousData = history[history.length - 1];
    
    if (previousData.avg_trade_size === 0) return 0;

    const sizeChange = (data.avg_trade_size - previousData.avg_trade_size) / previousData.avg_trade_size;
    return this.normalize(sizeChange, -0.5, 0.5, true); // ±50% change range
  }

  /**
   * Calculate market depth score across exchanges
   */
  private calculateMarketDepthScore(): number {
    const depths = Array.from(this.exchangeVolumes.values()).map(data => data.orderbook_depth);
    
    if (depths.length === 0) return 0.5;

    const totalDepth = depths.reduce((sum, depth) => sum + depth, 0);
    const avgDepth = totalDepth / depths.length;
    
    // Normalize to expected depth levels
    return Math.min(1, avgDepth / 1000000); // Normalize to $1M
  }

  /**
   * Calculate volume concentration (how evenly distributed volume is)
   */
  private calculateVolumeConcentration(pair: string): number {
    const pairData = Array.from(this.exchangeVolumes.values())
      .filter(data => this.normalizePair(data.pair) === this.normalizePair(pair));

    if (pairData.length <= 1) return 1; // Fully concentrated

    const volumes = pairData.map(data => data.volume_1m);
    const totalVolume = volumes.reduce((sum, vol) => sum + vol, 0);
    
    if (totalVolume === 0) return 1;

    // Calculate Herfindahl-Hirschman Index (HHI) for concentration
    const hhi = volumes.reduce((sum, vol) => {
      const share = vol / totalVolume;
      return sum + (share * share);
    }, 0);

    // Normalize HHI: 1/n (perfectly distributed) to 1 (fully concentrated)
    const minHHI = 1 / pairData.length;
    const normalizedHHI = (hhi - minHHI) / (1 - minHHI);
    
    return Math.max(0, Math.min(1, normalizedHHI));
  }

  /**
   * Calculate confidence based on data quality
   */
  private calculateVolumeConfidence(volumeDataArray: (VolumeData | null)[]): number {
    const validData = volumeDataArray.filter(data => data !== null) as VolumeData[];
    
    if (validData.length === 0) return 0;

    // Data coverage factor
    const coverageFactor = validData.length / volumeDataArray.length;

    // Exchange coverage factor
    const uniqueExchanges = new Set(Array.from(this.exchangeVolumes.values()).map(v => v.exchange)).size;
    const exchangeCoverageFactor = Math.min(1, uniqueExchanges / 3); // Target 3 exchanges

    // Volume significance factor (higher volume = higher confidence)
    const avgVolume = validData.reduce((sum, data) => sum + data.volume_1m_quote, 0) / validData.length;
    const volumeSignificanceFactor = Math.min(1, avgVolume / 50000); // Target $50k+ volume

    // Data freshness factor
    const now = Date.now();
    const avgAge = validData.reduce((sum, data) => sum + (now - data.timestamp), 0) / validData.length;
    const freshnessFactor = Math.max(0, 1 - (avgAge / 120000)); // 2 minutes max age

    return Math.max(0.1, coverageFactor * exchangeCoverageFactor * volumeSignificanceFactor * freshnessFactor);
  }

  /**
   * Normalize pair names
   */
  private normalizePair(pair: string): string {
    return pair.replace(/[-\/]/g, '-').toUpperCase();
  }

  /**
   * Get default volume features
   */
  private getDefaultVolumeFeatures(): VolumeFeatures {
    return {
      liquidity_index: 0.5,
      buy_sell_volume_ratio: 0.5,
      volume_spike_flag: 0,
      trade_size_momentum: 0,
      market_depth_score: 0.5,
      volume_concentration: 0.5
    };
  }

  /**
   * Get volume summary for debugging
   */
  getVolumeSummary(): string {
    const summaries: string[] = [];
    
    for (const [pair, data] of this.volumeData) {
      const volume = (data.volume_1m_quote / 1000).toFixed(0);
      const trades = data.trade_count_1m;
      const buySellRatio = (data.buy_volume_1m / (data.sell_volume_1m || 1)).toFixed(2);
      summaries.push(`${pair}: $${volume}k vol, ${trades} trades, B/S: ${buySellRatio}`);
    }
    
    return summaries.length > 0 ? summaries.join(', ') : 'No volume data available';
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
    data_age_max_ms: 3 * 60 * 1000, // 3 minutes
    retry_attempts: 3,
    retry_backoff_ms: 2000
  };

  return new VolumeAgent(config);
}

// Export for testing
export { VolumeAgent as default };