/**
 * Trade Agent - Trade Flow Analysis and Execution Monitoring
 * Analyzes trade flow patterns, taker/maker imbalance, VWAP, slippage estimates
 * Monitors unusual trade patterns and execution quality metrics
 */

import { BaseAgent, AgentConfig, AgentOutput } from '../../core/base-agent.js';

export interface TradeData {
  pair: string;
  taker_buy_volume_1m: number;     // Volume of taker buy orders
  taker_sell_volume_1m: number;    // Volume of taker sell orders
  maker_buy_volume_1m: number;     // Volume of maker buy orders
  maker_sell_volume_1m: number;    // Volume of maker sell orders
  vwap_1m: number;                 // Volume-weighted average price
  twap_1m: number;                 // Time-weighted average price
  trade_count_1m: number;          // Number of trades
  avg_trade_size: number;          // Average trade size
  max_trade_size: number;          // Largest single trade
  slippage_estimate_bps: number;   // Estimated slippage in basis points
  trade_latency_ms: number;        // Average trade execution latency
  timestamp: number;
}

export interface TradeFeatures {
  taker_imbalance: number;         // (taker_buy - taker_sell) / total [-1, 1]
  maker_imbalance: number;         // (maker_buy - maker_sell) / total [-1, 1]
  price_efficiency: number;        // How close trades are to mid-price [0, 1]
  execution_quality: number;       // Overall execution quality score [0, 1]
  market_impact: number;           // Estimated market impact [0, 1]
  flow_toxicity: number;           // Measure of informed trading [0, 1]
  urgency_score: number;           // Market urgency based on taker ratio [0, 1]
}

interface ExchangeTradeData {
  exchange: string;
  pair: string;
  trades: TradeExecution[];
  vwap: number;
  total_volume: number;
  timestamp: number;
}

interface TradeExecution {
  price: number;
  size: number;
  side: 'buy' | 'sell';
  taker: boolean;
  timestamp: number;
  trade_id: string;
}

export class TradeAgent extends BaseAgent {
  private tradeData: Map<string, TradeData> = new Map();
  private tradeHistory: Map<string, TradeData[]> = new Map();
  private exchangeTrades: Map<string, ExchangeTradeData> = new Map();
  private recentTrades: Map<string, TradeExecution[]> = new Map();
  
  // Trade analysis parameters
  private readonly LARGE_TRADE_THRESHOLD = 50000; // $50k USD
  private readonly HIGH_FREQUENCY_THRESHOLD = 10;  // 10 trades per minute
  private readonly SLIPPAGE_WARNING_BPS = 20;      // 20 basis points
  private readonly LATENCY_WARNING_MS = 100;       // 100ms execution latency
  
  constructor(config: AgentConfig) {
    super(config);
  }

  protected async collectData(): Promise<AgentOutput> {
    const timestamp = this.getCurrentTimestamp();

    try {
      // Collect trade data from multiple exchanges
      await this.collectTradeMetrics();

      // Get aggregated trade data for primary pairs
      const btcTradeData = this.getAggregatedTradeData('BTC-USDT');
      const ethTradeData = this.getAggregatedTradeData('ETH-USDT');

      // Calculate trade flow features
      const features = this.calculateTradeFeatures(btcTradeData, ethTradeData);

      // Calculate key signal (execution quality score)
      const keySignal = features.execution_quality;

      // Calculate confidence based on trade volume and data quality
      const confidence = this.calculateTradeConfidence([btcTradeData, ethTradeData]);

      return {
        agent_name: 'TradeAgent',
        timestamp,
        key_signal: keySignal,
        confidence,
        features: {
          ...features,
          btc_trade_data: btcTradeData ? {
            vwap_1m: btcTradeData.vwap_1m,
            trade_count: btcTradeData.trade_count_1m,
            taker_imbalance: this.calculateTakerImbalance(btcTradeData),
            slippage_estimate_bps: btcTradeData.slippage_estimate_bps
          } : null,
          eth_trade_data: ethTradeData ? {
            vwap_1m: ethTradeData.vwap_1m,
            trade_count: ethTradeData.trade_count_1m,
            taker_imbalance: this.calculateTakerImbalance(ethTradeData),
            slippage_estimate_bps: ethTradeData.slippage_estimate_bps
          } : null
        },
        metadata: {
          tracked_pairs: this.tradeData.size,
          total_trades_1m: Array.from(this.tradeData.values()).reduce((sum, t) => sum + t.trade_count_1m, 0),
          avg_latency_ms: this.calculateAverageLatency(),
          unusual_patterns_detected: this.detectUnusualPatterns()
        }
      };

    } catch (error) {
      console.error('TradeAgent data collection failed:', error);
      throw error;
    }
  }

  /**
   * Collect trade metrics from various sources
   */
  private async collectTradeMetrics(): Promise<void> {
    const pairs = ['BTC-USDT', 'ETH-USDT', 'BTC-USD', 'ETH-USD'];
    const exchanges = ['binance', 'coinbase', 'kraken'];

    // Simulate collecting recent trades from each exchange
    for (const exchange of exchanges) {
      for (const pair of pairs) {
        const exchangeTradeData = await this.fetchExchangeTradeData(exchange, pair);
        if (exchangeTradeData) {
          this.exchangeTrades.set(`${exchange}_${pair}`, exchangeTradeData);
        }
      }
    }

    // Aggregate trade data by pair
    for (const pair of pairs) {
      const aggregatedData = this.aggregateTradesByPair(pair);
      if (aggregatedData) {
        this.tradeData.set(pair, aggregatedData);
        this.updateTradeHistory(pair, aggregatedData);
      }
    }
  }

  /**
   * Fetch trade data from a specific exchange (simulated for demo)
   */
  private async fetchExchangeTradeData(exchange: string, pair: string): Promise<ExchangeTradeData | null> {
    try {
      // Simulate realistic trade data
      const trades = this.generateSimulatedTrades(exchange, pair);
      const vwap = this.calculateVWAP(trades);
      const totalVolume = trades.reduce((sum, trade) => sum + (trade.price * trade.size), 0);

      return {
        exchange,
        pair,
        trades,
        vwap,
        total_volume: totalVolume,
        timestamp: Date.now()
      };

    } catch (error) {
      console.warn(`Failed to fetch trade data from ${exchange} for ${pair}:`, error.message);
      return null;
    }
  }

  /**
   * Generate simulated trades for testing (replace with real API calls)
   */
  private generateSimulatedTrades(exchange: string, pair: string): TradeExecution[] {
    const trades: TradeExecution[] = [];
    const basePrice = this.getEstimatedPrice(pair);
    const tradeCount = 20 + Math.floor(Math.random() * 30); // 20-50 trades

    for (let i = 0; i < tradeCount; i++) {
      const priceVariation = (Math.random() - 0.5) * 0.002; // ±0.1% price variation
      const price = basePrice * (1 + priceVariation);
      const size = 0.1 + (Math.random() * 2); // 0.1 to 2.1 BTC or ETH
      const side = Math.random() > 0.5 ? 'buy' : 'sell';
      const taker = Math.random() > 0.3; // 70% taker trades
      
      trades.push({
        price,
        size,
        side,
        taker,
        timestamp: Date.now() - Math.floor(Math.random() * 60000), // Within last minute
        trade_id: `${exchange}_${Date.now()}_${i}`
      });
    }

    return trades.sort((a, b) => a.timestamp - b.timestamp);
  }

  /**
   * Calculate Volume-Weighted Average Price
   */
  private calculateVWAP(trades: TradeExecution[]): number {
    let totalVolume = 0;
    let totalValue = 0;

    for (const trade of trades) {
      const value = trade.price * trade.size;
      totalValue += value;
      totalVolume += trade.size;
    }

    return totalVolume > 0 ? totalValue / totalVolume : 0;
  }

  /**
   * Get estimated price for calculations
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
   * Aggregate trades by pair across exchanges
   */
  private aggregateTradesByPair(pair: string): TradeData | null {
    const exchangeData = Array.from(this.exchangeTrades.values())
      .filter(data => this.normalizePair(data.pair) === this.normalizePair(pair));

    if (exchangeData.length === 0) return null;

    // Combine all trades
    const allTrades = exchangeData.flatMap(data => data.trades);
    const now = Date.now();
    const oneMinuteAgo = now - 60000;
    
    // Filter trades from last minute
    const recentTrades = allTrades.filter(trade => trade.timestamp >= oneMinuteAgo);
    
    if (recentTrades.length === 0) return null;

    // Calculate aggregated metrics
    const takerBuyVolume = recentTrades
      .filter(t => t.taker && t.side === 'buy')
      .reduce((sum, t) => sum + (t.price * t.size), 0);
      
    const takerSellVolume = recentTrades
      .filter(t => t.taker && t.side === 'sell')
      .reduce((sum, t) => sum + (t.price * t.size), 0);
      
    const makerBuyVolume = recentTrades
      .filter(t => !t.taker && t.side === 'buy')
      .reduce((sum, t) => sum + (t.price * t.size), 0);
      
    const makerSellVolume = recentTrades
      .filter(t => !t.taker && t.side === 'sell')
      .reduce((sum, t) => sum + (t.price * t.size), 0);

    const vwap = this.calculateVWAP(recentTrades);
    const twap = this.calculateTWAP(recentTrades);
    
    const totalVolume = recentTrades.reduce((sum, t) => sum + (t.price * t.size), 0);
    const avgTradeSize = totalVolume / recentTrades.length;
    const maxTradeSize = Math.max(...recentTrades.map(t => t.price * t.size));
    
    const slippageEstimate = this.estimateSlippage(recentTrades, vwap);
    const avgLatency = this.calculateTradeLatency(recentTrades);

    return {
      pair,
      taker_buy_volume_1m: takerBuyVolume,
      taker_sell_volume_1m: takerSellVolume,
      maker_buy_volume_1m: makerBuyVolume,
      maker_sell_volume_1m: makerSellVolume,
      vwap_1m: vwap,
      twap_1m: twap,
      trade_count_1m: recentTrades.length,
      avg_trade_size: avgTradeSize,
      max_trade_size: maxTradeSize,
      slippage_estimate_bps: slippageEstimate,
      trade_latency_ms: avgLatency,
      timestamp: now
    };
  }

  /**
   * Calculate Time-Weighted Average Price
   */
  private calculateTWAP(trades: TradeExecution[]): number {
    if (trades.length === 0) return 0;
    
    // Simple TWAP calculation (equal time weights)
    const totalPrice = trades.reduce((sum, trade) => sum + trade.price, 0);
    return totalPrice / trades.length;
  }

  /**
   * Estimate slippage based on trade prices vs VWAP
   */
  private estimateSlippage(trades: TradeExecution[], vwap: number): number {
    if (trades.length === 0 || vwap === 0) return 0;

    const slippages = trades.map(trade => {
      const slippage = Math.abs(trade.price - vwap) / vwap;
      return slippage * 10000; // Convert to basis points
    });

    // Return average slippage in basis points
    return slippages.reduce((sum, s) => sum + s, 0) / slippages.length;
  }

  /**
   * Calculate average trade latency (simulated)
   */
  private calculateTradeLatency(trades: TradeExecution[]): number {
    // In production, this would measure actual execution latency
    // For simulation, generate realistic latency based on trade size
    const latencies = trades.map(trade => {
      const baseLatency = 50; // 50ms base latency
      const sizeMultiplier = Math.log10(trade.size + 1); // Larger trades take longer
      return baseLatency + (sizeMultiplier * 10) + (Math.random() * 20);
    });

    return latencies.reduce((sum, l) => sum + l, 0) / latencies.length;
  }

  /**
   * Update trade history for trend analysis
   */
  private updateTradeHistory(pair: string, data: TradeData): void {
    if (!this.tradeHistory.has(pair)) {
      this.tradeHistory.set(pair, []);
    }

    const history = this.tradeHistory.get(pair)!;
    history.push(data);

    // Keep only last 60 data points (1 hour of minute data)
    if (history.length > 60) {
      history.shift();
    }
  }

  /**
   * Get aggregated trade data for a specific pair
   */
  private getAggregatedTradeData(pair: string): TradeData | null {
    return this.tradeData.get(pair) || null;
  }

  /**
   * Calculate trade flow features
   */
  private calculateTradeFeatures(btcData: TradeData | null, ethData: TradeData | null): TradeFeatures {
    if (!btcData && !ethData) {
      return this.getDefaultTradeFeatures();
    }

    const primaryData = btcData || ethData!;

    // Taker Imbalance: (taker_buy - taker_sell) / total_taker
    const takerImbalance = this.calculateTakerImbalance(primaryData);

    // Maker Imbalance: (maker_buy - maker_sell) / total_maker
    const makerImbalance = this.calculateMakerImbalance(primaryData);

    // Price Efficiency: how close VWAP is to TWAP
    const priceEfficiency = this.calculatePriceEfficiency(primaryData);

    // Execution Quality: composite score
    const executionQuality = this.calculateExecutionQuality(primaryData);

    // Market Impact: based on slippage and trade size
    const marketImpact = this.calculateMarketImpact(primaryData);

    // Flow Toxicity: measure of informed trading
    const flowToxicity = this.calculateFlowToxicity(primaryData);

    // Urgency Score: based on taker ratio and latency
    const urgencyScore = this.calculateUrgencyScore(primaryData);

    return {
      taker_imbalance: takerImbalance,
      maker_imbalance: makerImbalance,
      price_efficiency: priceEfficiency,
      execution_quality: executionQuality,
      market_impact: marketImpact,
      flow_toxicity: flowToxicity,
      urgency_score: urgencyScore
    };
  }

  /**
   * Calculate taker imbalance
   */
  private calculateTakerImbalance(data: TradeData): number {
    const totalTaker = data.taker_buy_volume_1m + data.taker_sell_volume_1m;
    if (totalTaker === 0) return 0;
    
    const imbalance = (data.taker_buy_volume_1m - data.taker_sell_volume_1m) / totalTaker;
    return Math.max(-1, Math.min(1, imbalance));
  }

  /**
   * Calculate maker imbalance
   */
  private calculateMakerImbalance(data: TradeData): number {
    const totalMaker = data.maker_buy_volume_1m + data.maker_sell_volume_1m;
    if (totalMaker === 0) return 0;
    
    const imbalance = (data.maker_buy_volume_1m - data.maker_sell_volume_1m) / totalMaker;
    return Math.max(-1, Math.min(1, imbalance));
  }

  /**
   * Calculate price efficiency (VWAP vs TWAP)
   */
  private calculatePriceEfficiency(data: TradeData): number {
    if (data.twap_1m === 0) return 0.5;
    
    const priceDiff = Math.abs(data.vwap_1m - data.twap_1m) / data.twap_1m;
    const efficiency = Math.max(0, 1 - (priceDiff * 100)); // Lower difference = higher efficiency
    
    return Math.min(1, efficiency);
  }

  /**
   * Calculate overall execution quality
   */
  private calculateExecutionQuality(data: TradeData): number {
    // Factors: low slippage, reasonable latency, good price efficiency
    const slippageFactor = Math.max(0, 1 - (data.slippage_estimate_bps / 100)); // Normalize to 100bps
    const latencyFactor = Math.max(0, 1 - (data.trade_latency_ms / 200)); // Normalize to 200ms
    const priceEfficiency = this.calculatePriceEfficiency(data);
    
    return (slippageFactor * 0.4) + (latencyFactor * 0.3) + (priceEfficiency * 0.3);
  }

  /**
   * Calculate market impact
   */
  private calculateMarketImpact(data: TradeData): number {
    // Higher slippage and larger trades = higher market impact
    const slippageImpact = Math.min(1, data.slippage_estimate_bps / 50); // Normalize to 50bps
    const sizeImpact = Math.min(1, data.avg_trade_size / this.LARGE_TRADE_THRESHOLD);
    
    return (slippageImpact * 0.7) + (sizeImpact * 0.3);
  }

  /**
   * Calculate flow toxicity (measure of informed trading)
   */
  private calculateFlowToxicity(data: TradeData): number {
    // High taker ratio + large imbalance + high urgency = higher toxicity
    const totalVolume = data.taker_buy_volume_1m + data.taker_sell_volume_1m + 
                       data.maker_buy_volume_1m + data.maker_sell_volume_1m;
    
    if (totalVolume === 0) return 0;
    
    const takerRatio = (data.taker_buy_volume_1m + data.taker_sell_volume_1m) / totalVolume;
    const takerImbalance = Math.abs(this.calculateTakerImbalance(data));
    const urgency = this.calculateUrgencyScore(data);
    
    return (takerRatio * 0.4) + (takerImbalance * 0.4) + (urgency * 0.2);
  }

  /**
   * Calculate urgency score
   */
  private calculateUrgencyScore(data: TradeData): number {
    // High taker ratio + low latency tolerance = higher urgency
    const totalVolume = data.taker_buy_volume_1m + data.taker_sell_volume_1m + 
                       data.maker_buy_volume_1m + data.maker_sell_volume_1m;
    
    if (totalVolume === 0) return 0;
    
    const takerRatio = (data.taker_buy_volume_1m + data.taker_sell_volume_1m) / totalVolume;
    const latencyPressure = Math.min(1, data.trade_latency_ms / 100); // Normalize to 100ms
    
    return (takerRatio * 0.7) + ((1 - latencyPressure) * 0.3);
  }

  /**
   * Calculate confidence based on trade data quality
   */
  private calculateTradeConfidence(tradeDataArray: (TradeData | null)[]): number {
    const validData = tradeDataArray.filter(data => data !== null) as TradeData[];
    
    if (validData.length === 0) return 0;

    // Volume factor (more trades = higher confidence)
    const avgTradeCount = validData.reduce((sum, data) => sum + data.trade_count_1m, 0) / validData.length;
    const volumeFactor = Math.min(1, avgTradeCount / 50); // Target 50+ trades per minute

    // Slippage factor (lower slippage = higher confidence)
    const avgSlippage = validData.reduce((sum, data) => sum + data.slippage_estimate_bps, 0) / validData.length;
    const slippageFactor = Math.max(0, 1 - (avgSlippage / 100)); // 100bps threshold

    // Latency factor (lower latency = higher confidence)
    const avgLatency = validData.reduce((sum, data) => sum + data.trade_latency_ms, 0) / validData.length;
    const latencyFactor = Math.max(0, 1 - (avgLatency / 200)); // 200ms threshold

    // Data completeness factor
    const completnessFactor = validData.length / tradeDataArray.length;

    return Math.max(0.1, volumeFactor * slippageFactor * latencyFactor * completnessFactor);
  }

  /**
   * Calculate average latency across all pairs
   */
  private calculateAverageLatency(): number {
    const latencies = Array.from(this.tradeData.values()).map(data => data.trade_latency_ms);
    return latencies.length > 0 ? latencies.reduce((sum, l) => sum + l, 0) / latencies.length : 0;
  }

  /**
   * Detect unusual trading patterns
   */
  private detectUnusualPatterns(): number {
    let unusualPatterns = 0;
    
    for (const data of this.tradeData.values()) {
      // Check for unusual volume spikes
      if (data.max_trade_size > this.LARGE_TRADE_THRESHOLD) {
        unusualPatterns++;
      }
      
      // Check for high slippage
      if (data.slippage_estimate_bps > this.SLIPPAGE_WARNING_BPS) {
        unusualPatterns++;
      }
      
      // Check for high latency
      if (data.trade_latency_ms > this.LATENCY_WARNING_MS) {
        unusualPatterns++;
      }
      
      // Check for extreme imbalances
      const takerImbalance = Math.abs(this.calculateTakerImbalance(data));
      if (takerImbalance > 0.8) {
        unusualPatterns++;
      }
    }
    
    return unusualPatterns;
  }

  /**
   * Normalize pair names
   */
  private normalizePair(pair: string): string {
    return pair.replace(/[-\/]/g, '-').toUpperCase();
  }

  /**
   * Get default trade features
   */
  private getDefaultTradeFeatures(): TradeFeatures {
    return {
      taker_imbalance: 0,
      maker_imbalance: 0,
      price_efficiency: 0.5,
      execution_quality: 0.5,
      market_impact: 0.5,
      flow_toxicity: 0.5,
      urgency_score: 0.5
    };
  }

  /**
   * Get trade summary for debugging
   */
  getTradeSummary(): string {
    const summaries: string[] = [];
    
    for (const [pair, data] of this.tradeData) {
      const vwap = data.vwap_1m.toFixed(0);
      const trades = data.trade_count_1m;
      const slippage = data.slippage_estimate_bps.toFixed(1);
      const latency = data.trade_latency_ms.toFixed(0);
      summaries.push(`${pair}: VWAP $${vwap}, ${trades} trades, ${slippage}bps slip, ${latency}ms lat`);
    }
    
    return summaries.length > 0 ? summaries.join(', ') : 'No trade data available';
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
    confidence_min: 0.4,
    data_age_max_ms: 2 * 60 * 1000, // 2 minutes
    retry_attempts: 3,
    retry_backoff_ms: 1000
  };

  return new TradeAgent(config);
}

// Export for testing
export { TradeAgent as default };