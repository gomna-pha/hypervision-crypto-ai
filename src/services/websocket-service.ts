/**
 * PRODUCTION WEBSOCKET SERVICE
 * 
 * Real-time market data streaming from:
 * - Binance (Spot & Futures)
 * - Coinbase Pro
 * - Kraken
 * 
 * Feeds directly into ML pipeline
 */

import { MarketDataFeed, realtimeFeeds } from '../data/realtime-data-feeds-node';
import { RawMarketData } from '../ml/feature-engineering';

export interface AggregatedMarketData {
  symbol: string;
  timestamp: Date;
  
  // Price data
  spotPrice: number;
  perpPrice: number | null;
  bidPrice: number;
  askPrice: number;
  
  // Exchange-specific prices
  exchangePrices: {
    binance: number;
    coinbase: number;
    kraken: number;
  };
  
  // Volume & liquidity
  volume24h: number;
  bidVolume: number;
  askVolume: number;
  liquidity: number;
  
  // Spread data
  bidAskSpread: number; // bps
  crossExchangeSpreads: {
    binanceCoinbase: number;
    binanceKraken: number;
    coinbaseKraken: number;
  };
  
  // Arbitrage opportunities
  bestArbitrageSpread: number;
  bestBuyExchange: string;
  bestSellExchange: string;
  
  // Metadata
  connectedExchanges: string[];
  dataQuality: 'excellent' | 'good' | 'degraded' | 'poor';
}

export class WebSocketService {
  private static instance: WebSocketService;
  private isInitialized = false;
  private latestData: Map<string, AggregatedMarketData> = new Map();
  private subscribers: Map<string, Set<(data: AggregatedMarketData) => void>> = new Map();
  
  private constructor() {}
  
  static getInstance(): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService();
    }
    return WebSocketService.instance;
  }
  
  /**
   * Initialize WebSocket connections
   */
  async initialize(symbols: string[] = ['BTC']): Promise<void> {
    if (this.isInitialized) {
      console.log('[WebSocket Service] Already initialized');
      return;
    }
    
    try {
      console.log('[WebSocket Service] Connecting to exchanges...');
      
      // Connect to all exchanges
      await Promise.all([
        realtimeFeeds.connectBinance(symbols.map(s => `${s}USDT`)),
        realtimeFeeds.connectCoinbase(symbols.map(s => `${s}-USD`)),
        realtimeFeeds.connectKraken(symbols.map(s => `X${s}/USD`)),
      ]);
      
      // Subscribe to price updates
      realtimeFeeds.onPriceUpdate((feed: MarketDataFeed) => {
        this.handlePriceUpdate(feed);
      });
      
      this.isInitialized = true;
      console.log('[WebSocket Service] ✅ Connected to all exchanges');
    } catch (error) {
      console.error('[WebSocket Service] ❌ Failed to initialize:', error);
      throw error;
    }
  }
  
  /**
   * Handle incoming price update
   */
  private handlePriceUpdate(feed: MarketDataFeed): void {
    const symbol = this.normalizeSymbol(feed.symbol);
    
    // Get existing data or create new
    let aggregated = this.latestData.get(symbol);
    if (!aggregated) {
      aggregated = this.createEmptyAggregatedData(symbol);
      this.latestData.set(symbol, aggregated);
    }
    
    // Update exchange-specific price
    aggregated.exchangePrices[feed.exchange as keyof typeof aggregated.exchangePrices] = feed.spotPrice;
    
    // Update timestamp
    aggregated.timestamp = feed.timestamp;
    
    // Recalculate aggregate metrics
    this.recalculateMetrics(aggregated);
    
    // Notify subscribers
    this.notifySubscribers(symbol, aggregated);
  }
  
  /**
   * Recalculate aggregate metrics
   */
  private recalculateMetrics(data: AggregatedMarketData): void {
    const prices = Object.values(data.exchangePrices).filter(p => p > 0);
    
    if (prices.length === 0) return;
    
    // Calculate weighted average price
    data.spotPrice = prices.reduce((sum, p) => sum + p, 0) / prices.length;
    
    // Calculate bid/ask from exchanges
    const { binance, coinbase, kraken } = data.exchangePrices;
    data.bidPrice = Math.min(...prices.filter(p => p > 0));
    data.askPrice = Math.max(...prices.filter(p => p > 0));
    
    // Calculate bid-ask spread in bps
    data.bidAskSpread = ((data.askPrice - data.bidPrice) / data.bidPrice) * 10000;
    
    // Calculate cross-exchange spreads
    if (binance > 0 && coinbase > 0) {
      data.crossExchangeSpreads.binanceCoinbase = Math.abs(binance - coinbase) / Math.min(binance, coinbase) * 10000;
    }
    if (binance > 0 && kraken > 0) {
      data.crossExchangeSpreads.binanceKraken = Math.abs(binance - kraken) / Math.min(binance, kraken) * 10000;
    }
    if (coinbase > 0 && kraken > 0) {
      data.crossExchangeSpreads.coinbaseKraken = Math.abs(coinbase - kraken) / Math.min(coinbase, kraken) * 10000;
    }
    
    // Find best arbitrage opportunity
    const spreads = [
      { spread: data.crossExchangeSpreads.binanceCoinbase, buy: binance < coinbase ? 'binance' : 'coinbase', sell: binance < coinbase ? 'coinbase' : 'binance' },
      { spread: data.crossExchangeSpreads.binanceKraken, buy: binance < kraken ? 'binance' : 'kraken', sell: binance < kraken ? 'kraken' : 'binance' },
      { spread: data.crossExchangeSpreads.coinbaseKraken, buy: coinbase < kraken ? 'coinbase' : 'kraken', sell: coinbase < kraken ? 'kraken' : 'coinbase' },
    ];
    
    const bestSpread = spreads.reduce((best, curr) => curr.spread > best.spread ? curr : best);
    data.bestArbitrageSpread = bestSpread.spread;
    data.bestBuyExchange = bestSpread.buy;
    data.bestSellExchange = bestSpread.sell;
    
    // Update connected exchanges
    data.connectedExchanges = Object.entries(data.exchangePrices)
      .filter(([_, price]) => price > 0)
      .map(([exchange, _]) => exchange);
    
    // Determine data quality
    data.dataQuality = this.assessDataQuality(data);
  }
  
  /**
   * Assess data quality
   */
  private assessDataQuality(data: AggregatedMarketData): 'excellent' | 'good' | 'degraded' | 'poor' {
    const connectedCount = data.connectedExchanges.length;
    const dataAge = Date.now() - data.timestamp.getTime();
    
    if (connectedCount === 3 && dataAge < 1000) return 'excellent';
    if (connectedCount >= 2 && dataAge < 5000) return 'good';
    if (connectedCount >= 1 && dataAge < 10000) return 'degraded';
    return 'poor';
  }
  
  /**
   * Normalize symbol format
   */
  private normalizeSymbol(symbol: string): string {
    return symbol.replace(/USDT|USD|-|\/|X/g, '');
  }
  
  /**
   * Create empty aggregated data
   */
  private createEmptyAggregatedData(symbol: string): AggregatedMarketData {
    return {
      symbol,
      timestamp: new Date(),
      spotPrice: 0,
      perpPrice: null,
      bidPrice: 0,
      askPrice: 0,
      exchangePrices: {
        binance: 0,
        coinbase: 0,
        kraken: 0,
      },
      volume24h: 0,
      bidVolume: 0,
      askVolume: 0,
      liquidity: 5000000, // Default
      bidAskSpread: 0,
      crossExchangeSpreads: {
        binanceCoinbase: 0,
        binanceKraken: 0,
        coinbaseKraken: 0,
      },
      bestArbitrageSpread: 0,
      bestBuyExchange: '',
      bestSellExchange: '',
      connectedExchanges: [],
      dataQuality: 'poor',
    };
  }
  
  /**
   * Subscribe to symbol updates
   */
  subscribe(symbol: string, callback: (data: AggregatedMarketData) => void): () => void {
    if (!this.subscribers.has(symbol)) {
      this.subscribers.set(symbol, new Set());
    }
    
    this.subscribers.get(symbol)!.add(callback);
    
    // Return unsubscribe function
    return () => {
      this.subscribers.get(symbol)?.delete(callback);
    };
  }
  
  /**
   * Notify subscribers
   */
  private notifySubscribers(symbol: string, data: AggregatedMarketData): void {
    const callbacks = this.subscribers.get(symbol);
    if (callbacks) {
      callbacks.forEach(callback => {
        try {
          callback(data);
        } catch (error) {
          console.error('[WebSocket Service] Subscriber callback error:', error);
        }
      });
    }
  }
  
  /**
   * Get latest data for symbol
   */
  getLatestData(symbol: string): AggregatedMarketData | null {
    return this.latestData.get(symbol) || null;
  }
  
  /**
   * Convert to RawMarketData format (for ML pipeline)
   */
  toRawMarketData(aggregated: AggregatedMarketData): RawMarketData {
    return {
      timestamp: aggregated.timestamp,
      symbol: `${aggregated.symbol}-USD`,
      spotPrice: aggregated.spotPrice,
      perpPrice: aggregated.perpPrice || aggregated.spotPrice,
      bidPrice: aggregated.bidPrice,
      askPrice: aggregated.askPrice,
      exchangePrices: aggregated.exchangePrices,
      volume24h: aggregated.volume24h || 1000000,
      bidVolume: aggregated.bidVolume || 500000,
      askVolume: aggregated.askVolume || 500000,
      liquidity: aggregated.liquidity,
      fundingRate: 0.0001, // Default
      openInterest: 1000000000, // Default
    };
  }
  
  /**
   * Get connection status
   */
  getConnectionStatus(): { connected: boolean; exchanges: Record<string, boolean> } {
    const status = realtimeFeeds.getConnectionStatus();
    return {
      connected: Object.values(status).some(s => s),
      exchanges: status,
    };
  }
  
  /**
   * Shutdown
   */
  shutdown(): void {
    realtimeFeeds.disconnectAll();
    this.isInitialized = false;
    this.latestData.clear();
    this.subscribers.clear();
    console.log('[WebSocket Service] 🔌 Disconnected');
  }
}

/**
 * Singleton export
 */
export const websocketService = WebSocketService.getInstance();

/**
 * Example usage:
 * 
 * // Initialize
 * await websocketService.initialize(['BTC', 'ETH']);
 * 
 * // Subscribe to updates
 * const unsubscribe = websocketService.subscribe('BTC', (data) => {
 *   console.log('BTC Price:', data.spotPrice);
 *   console.log('Best Arbitrage:', data.bestArbitrageSpread, 'bps');
 * });
 * 
 * // Get latest data
 * const btcData = websocketService.getLatestData('BTC');
 * 
 * // Convert for ML pipeline
 * const rawData = websocketService.toRawMarketData(btcData!);
 * 
 * // Cleanup
 * unsubscribe();
 */
