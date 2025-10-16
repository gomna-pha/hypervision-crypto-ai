/**
 * Price Agent - Real-Time Exchange Price and Orderbook Data
 * Collects canonical price and orderbook data across exchanges with low latency
 * Connects to Binance, Coinbase, Kraken WebSockets for real-time updates
 */

import { BaseAgent, AgentConfig, AgentOutput } from '../../core/base-agent.js';
import WebSocket from 'ws';

export interface ExchangePriceData {
  exchange: string;
  pair: string;
  best_bid: number;
  best_ask: number;
  mid_price: number;
  last_trade_price: number;
  volume_1m: number;
  vwap_1m: number;
  orderbook_depth_usd_at_0_5pct: number;
  open_interest?: number;
  funding_rate?: number;
  timestamp: number;
}

export interface PriceFeatures {
  spread_pct: number;              // (ask - bid) / mid * 100
  price_momentum_1m: number;       // Price change over 1 minute [-1, 1]
  volume_momentum_1m: number;      // Volume change over 1 minute [-1, 1]
  liquidity_score: number;         // Orderbook depth normalized [0, 1]
  volatility_1m: number;           // Price volatility over 1 minute [0, 1]
  arbitrage_opportunity: number;    // Cross-exchange spread potential [-1, 1]
}

interface OrderBookLevel {
  price: number;
  size: number;
}

interface OrderBook {
  bids: OrderBookLevel[];
  asks: OrderBookLevel[];
  timestamp: number;
}

export class PriceAgent extends BaseAgent {
  private exchanges: Map<string, WebSocket> = new Map();
  private priceData: Map<string, ExchangePriceData> = new Map();
  private orderBooks: Map<string, OrderBook> = new Map();
  private priceHistory: Map<string, ExchangePriceData[]> = new Map();
  private reconnectAttempts: Map<string, number> = new Map();
  private maxReconnectAttempts = 5;

  private readonly exchangeConfigs = {
    binance: {
      wsUrl: 'wss://stream.binance.com:9443/ws',
      pairs: ['btcusdt', 'ethusdt'],
      depthStream: '@depth20@100ms',
      tradeStream: '@aggTrade'
    },
    coinbase: {
      wsUrl: 'wss://ws-feed.exchange.coinbase.com',
      pairs: ['BTC-USD', 'ETH-USD'],
      channels: ['level2', 'matches']
    },
    kraken: {
      wsUrl: 'wss://ws.kraken.com',
      pairs: ['BTC/USD', 'ETH/USD'],
      channels: ['book', 'trade']
    }
  };

  constructor(config: AgentConfig) {
    super(config);
  }

  protected async collectData(): Promise<AgentOutput> {
    const timestamp = this.getCurrentTimestamp();

    try {
      // Ensure WebSocket connections are active
      await this.ensureConnections();

      // Get latest price data from all exchanges
      const allPriceData = Array.from(this.priceData.values());

      if (allPriceData.length === 0) {
        throw new Error('No price data available from any exchange');
      }

      // Calculate cross-exchange features
      const features = this.calculatePriceFeatures(allPriceData);

      // Calculate key signal (arbitrage opportunity score)
      const keySignal = features.arbitrage_opportunity;

      // Calculate confidence based on data freshness and exchange coverage
      const confidence = this.calculatePriceConfidence(allPriceData);

      return {
        agent_name: 'PriceAgent',
        timestamp,
        key_signal: keySignal,
        confidence,
        features: {
          ...features,
          exchange_data: Object.fromEntries(
            allPriceData.map(data => [
              `${data.exchange}_${data.pair}`, 
              {
                mid_price: data.mid_price,
                spread_pct: ((data.best_ask - data.best_bid) / data.mid_price * 100),
                volume_1m: data.volume_1m,
                last_update_age_ms: Date.now() - data.timestamp
              }
            ])
          )
        },
        metadata: {
          active_exchanges: this.exchanges.size,
          price_feeds: allPriceData.length,
          oldest_data_age_ms: Math.max(...allPriceData.map(d => Date.now() - d.timestamp))
        }
      };

    } catch (error) {
      console.error('PriceAgent data collection failed:', error);
      throw error;
    }
  }

  /**
   * Ensure all WebSocket connections are active
   */
  private async ensureConnections(): Promise<void> {
    const promises: Promise<void>[] = [];

    for (const exchangeName of Object.keys(this.exchangeConfigs)) {
      if (!this.exchanges.has(exchangeName) || 
          this.exchanges.get(exchangeName)?.readyState !== WebSocket.OPEN) {
        promises.push(this.connectToExchange(exchangeName));
      }
    }

    if (promises.length > 0) {
      await Promise.all(promises);
    }
  }

  /**
   * Connect to a specific exchange WebSocket
   */
  private async connectToExchange(exchangeName: string): Promise<void> {
    const config = this.exchangeConfigs[exchangeName as keyof typeof this.exchangeConfigs];
    if (!config) {
      throw new Error(`Unknown exchange: ${exchangeName}`);
    }

    try {
      let ws: WebSocket;
      let subscriptionMessage: any;

      switch (exchangeName) {
        case 'binance':
          ws = new WebSocket(config.wsUrl);
          subscriptionMessage = this.createBinanceSubscription(config);
          break;
        
        case 'coinbase':
          ws = new WebSocket(config.wsUrl);
          subscriptionMessage = this.createCoinbaseSubscription(config);
          break;
        
        case 'kraken':
          ws = new WebSocket(config.wsUrl);
          subscriptionMessage = this.createKrakenSubscription(config);
          break;
        
        default:
          throw new Error(`Unsupported exchange: ${exchangeName}`);
      }

      return new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          reject(new Error(`Connection timeout for ${exchangeName}`));
        }, 10000);

        ws.on('open', () => {
          console.log(`Connected to ${exchangeName} WebSocket`);
          ws.send(JSON.stringify(subscriptionMessage));
          
          this.exchanges.set(exchangeName, ws);
          this.reconnectAttempts.set(exchangeName, 0);
          
          clearTimeout(timeout);
          resolve();
        });

        ws.on('message', (data: Buffer) => {
          try {
            this.handleExchangeMessage(exchangeName, JSON.parse(data.toString()));
          } catch (error) {
            console.error(`Error processing ${exchangeName} message:`, error);
          }
        });

        ws.on('error', (error: Error) => {
          console.error(`${exchangeName} WebSocket error:`, error);
          clearTimeout(timeout);
          reject(error);
        });

        ws.on('close', () => {
          console.log(`${exchangeName} WebSocket closed`);
          this.exchanges.delete(exchangeName);
          this.scheduleReconnect(exchangeName);
        });
      });

    } catch (error) {
      console.error(`Failed to connect to ${exchangeName}:`, error);
      throw error;
    }
  }

  /**
   * Create Binance subscription message
   */
  private createBinanceSubscription(config: any): any {
    const streams = config.pairs.flatMap((pair: string) => [
      `${pair}${config.depthStream}`,
      `${pair}${config.tradeStream}`
    ]);

    return {
      method: 'SUBSCRIBE',
      params: streams,
      id: 1
    };
  }

  /**
   * Create Coinbase subscription message
   */
  private createCoinbaseSubscription(config: any): any {
    return {
      type: 'subscribe',
      product_ids: config.pairs,
      channels: config.channels
    };
  }

  /**
   * Create Kraken subscription message
   */
  private createKrakenSubscription(config: any): any {
    return {
      event: 'subscribe',
      pair: config.pairs,
      subscription: {
        name: 'book',
        depth: 25
      }
    };
  }

  /**
   * Handle incoming WebSocket messages from exchanges
   */
  private handleExchangeMessage(exchange: string, message: any): void {
    try {
      switch (exchange) {
        case 'binance':
          this.handleBinanceMessage(message);
          break;
        case 'coinbase':
          this.handleCoinbaseMessage(message);
          break;
        case 'kraken':
          this.handleKrakenMessage(message);
          break;
      }
    } catch (error) {
      console.error(`Error handling ${exchange} message:`, error);
    }
  }

  /**
   * Handle Binance WebSocket messages
   */
  private handleBinanceMessage(message: any): void {
    if (!message.stream) return;

    const [pair, streamType] = message.stream.split('@');
    const data = message.data;

    if (streamType.includes('depth')) {
      // Orderbook update
      const orderBook: OrderBook = {
        bids: data.bids.map((b: string[]) => ({ price: parseFloat(b[0]), size: parseFloat(b[1]) })),
        asks: data.asks.map((a: string[]) => ({ price: parseFloat(a[0]), size: parseFloat(a[1]) })),
        timestamp: Date.now()
      };

      this.orderBooks.set(`binance_${pair.toUpperCase()}`, orderBook);
      this.updatePriceData('binance', pair.toUpperCase(), orderBook);
    }

    if (streamType.includes('aggTrade')) {
      // Trade update - update VWAP and volume
      this.updateTradeData('binance', pair.toUpperCase(), {
        price: parseFloat(data.p),
        quantity: parseFloat(data.q),
        timestamp: data.T
      });
    }
  }

  /**
   * Handle Coinbase WebSocket messages
   */
  private handleCoinbaseMessage(message: any): void {
    if (message.type === 'l2update') {
      // Level 2 orderbook update
      const pair = message.product_id;
      const changes = message.changes;

      // Update existing orderbook or create new one
      const existingBook = this.orderBooks.get(`coinbase_${pair}`) || {
        bids: [],
        asks: [],
        timestamp: Date.now()
      };

      // Apply changes
      changes.forEach((change: string[]) => {
        const [side, price, size] = change;
        const priceNum = parseFloat(price);
        const sizeNum = parseFloat(size);

        const book = side === 'buy' ? existingBook.bids : existingBook.asks;
        
        if (sizeNum === 0) {
          // Remove level
          const index = book.findIndex(level => level.price === priceNum);
          if (index !== -1) book.splice(index, 1);
        } else {
          // Update or add level
          const index = book.findIndex(level => level.price === priceNum);
          if (index !== -1) {
            book[index].size = sizeNum;
          } else {
            book.push({ price: priceNum, size: sizeNum });
            book.sort((a, b) => side === 'buy' ? b.price - a.price : a.price - b.price);
          }
        }
      });

      existingBook.timestamp = Date.now();
      this.orderBooks.set(`coinbase_${pair}`, existingBook);
      this.updatePriceData('coinbase', pair, existingBook);
    }

    if (message.type === 'match') {
      // Trade update
      this.updateTradeData('coinbase', message.product_id, {
        price: parseFloat(message.price),
        quantity: parseFloat(message.size),
        timestamp: new Date(message.time).getTime()
      });
    }
  }

  /**
   * Handle Kraken WebSocket messages (simplified)
   */
  private handleKrakenMessage(message: any): void {
    // Simplified Kraken handling - implement full protocol as needed
    if (Array.isArray(message) && message.length >= 2) {
      const data = message[1];
      if (data.as && data.bs) {
        // Book snapshot/update
        const pair = message[3] || 'BTC/USD';
        
        const orderBook: OrderBook = {
          bids: data.bs?.map((b: string[]) => ({ price: parseFloat(b[0]), size: parseFloat(b[1]) })) || [],
          asks: data.as?.map((a: string[]) => ({ price: parseFloat(a[0]), size: parseFloat(a[1]) })) || [],
          timestamp: Date.now()
        };

        this.orderBooks.set(`kraken_${pair}`, orderBook);
        this.updatePriceData('kraken', pair, orderBook);
      }
    }
  }

  /**
   * Update price data from orderbook
   */
  private updatePriceData(exchange: string, pair: string, orderBook: OrderBook): void {
    if (orderBook.bids.length === 0 || orderBook.asks.length === 0) return;

    const bestBid = orderBook.bids[0].price;
    const bestAsk = orderBook.asks[0].price;
    const midPrice = (bestBid + bestAsk) / 2;

    // Calculate orderbook depth at 0.5%
    const depthPrice = midPrice * 0.005; // 0.5%
    const bidDepth = this.calculateDepth(orderBook.bids, bestBid - depthPrice, 'bid');
    const askDepth = this.calculateDepth(orderBook.asks, bestAsk + depthPrice, 'ask');
    const totalDepthUsd = (bidDepth + askDepth) * midPrice;

    const key = `${exchange}_${pair}`;
    const existingData = this.priceData.get(key);

    const priceData: ExchangePriceData = {
      exchange,
      pair,
      best_bid: bestBid,
      best_ask: bestAsk,
      mid_price: midPrice,
      last_trade_price: existingData?.last_trade_price || midPrice,
      volume_1m: existingData?.volume_1m || 0,
      vwap_1m: existingData?.vwap_1m || midPrice,
      orderbook_depth_usd_at_0_5pct: totalDepthUsd,
      timestamp: orderBook.timestamp
    };

    this.priceData.set(key, priceData);

    // Update price history
    if (!this.priceHistory.has(key)) {
      this.priceHistory.set(key, []);
    }
    
    const history = this.priceHistory.get(key)!;
    history.push(priceData);

    // Keep only last 60 data points (for 1-minute calculations)
    if (history.length > 60) {
      history.shift();
    }
  }

  /**
   * Update trade data (VWAP and volume)
   */
  private updateTradeData(exchange: string, pair: string, trade: any): void {
    const key = `${exchange}_${pair}`;
    const existingData = this.priceData.get(key);
    
    if (!existingData) return;

    // Update last trade price
    existingData.last_trade_price = trade.price;

    // Update 1-minute volume and VWAP (simplified)
    const now = Date.now();
    const oneMinuteAgo = now - 60 * 1000;

    // In production, maintain proper trade history for accurate VWAP
    // For now, use simple approximation
    existingData.volume_1m = (existingData.volume_1m * 0.99) + trade.quantity;
    existingData.vwap_1m = (existingData.vwap_1m * 0.95) + (trade.price * 0.05);

    this.priceData.set(key, existingData);
  }

  /**
   * Calculate orderbook depth up to a price level
   */
  private calculateDepth(levels: OrderBookLevel[], priceLimit: number, side: 'bid' | 'ask'): number {
    let totalSize = 0;
    
    for (const level of levels) {
      if (side === 'bid' && level.price >= priceLimit) {
        totalSize += level.size;
      } else if (side === 'ask' && level.price <= priceLimit) {
        totalSize += level.size;
      } else {
        break;
      }
    }
    
    return totalSize;
  }

  /**
   * Calculate price-related features
   */
  private calculatePriceFeatures(allPriceData: ExchangePriceData[]): PriceFeatures {
    // Group by pair for cross-exchange analysis
    const pairGroups = new Map<string, ExchangePriceData[]>();
    
    allPriceData.forEach(data => {
      const normalizedPair = this.normalizePair(data.pair);
      if (!pairGroups.has(normalizedPair)) {
        pairGroups.set(normalizedPair, []);
      }
      pairGroups.get(normalizedPair)!.push(data);
    });

    // Calculate features for the primary pair (BTC-USD/USDT)
    const btcData = pairGroups.get('BTC-USD') || pairGroups.get('BTC-USDT') || [];
    
    if (btcData.length === 0) {
      return this.getDefaultFeatures();
    }

    // Average spread across exchanges
    const avgSpread = btcData.reduce((sum, data) => 
      sum + ((data.best_ask - data.best_bid) / data.mid_price * 100), 0) / btcData.length;

    // Price momentum (1-minute change)
    const priceMomentum = this.calculatePriceMomentum(btcData[0]);

    // Volume momentum (1-minute change)
    const volumeMomentum = this.calculateVolumeMomentum(btcData[0]);

    // Liquidity score (average depth)
    const avgLiquidity = btcData.reduce((sum, data) => 
      sum + data.orderbook_depth_usd_at_0_5pct, 0) / btcData.length;
    const liquidityScore = this.normalize(avgLiquidity, 0, 1000000, false); // Normalize to $1M

    // Volatility (price standard deviation)
    const volatility = this.calculateVolatility(btcData[0]);

    // Arbitrage opportunity (cross-exchange spread)
    const arbitrageOpportunity = this.calculateArbitrageOpportunity(btcData);

    return {
      spread_pct: avgSpread,
      price_momentum_1m: priceMomentum,
      volume_momentum_1m: volumeMomentum,
      liquidity_score: liquidityScore,
      volatility_1m: volatility,
      arbitrage_opportunity: arbitrageOpportunity
    };
  }

  /**
   * Calculate price momentum over 1 minute
   */
  private calculatePriceMomentum(data: ExchangePriceData): number {
    const key = `${data.exchange}_${data.pair}`;
    const history = this.priceHistory.get(key);
    
    if (!history || history.length < 2) return 0;

    const current = history[history.length - 1];
    const oneMinuteAgo = history.find(h => current.timestamp - h.timestamp >= 60000);
    
    if (!oneMinuteAgo) return 0;

    const priceChange = (current.mid_price - oneMinuteAgo.mid_price) / oneMinuteAgo.mid_price;
    return this.normalize(priceChange, -0.02, 0.02, true); // ±2% range
  }

  /**
   * Calculate volume momentum over 1 minute
   */
  private calculateVolumeMomentum(data: ExchangePriceData): number {
    const key = `${data.exchange}_${data.pair}`;
    const history = this.priceHistory.get(key);
    
    if (!history || history.length < 2) return 0;

    const current = history[history.length - 1];
    const previous = history[history.length - 2];
    
    if (previous.volume_1m === 0) return 0;

    const volumeChange = (current.volume_1m - previous.volume_1m) / previous.volume_1m;
    return this.normalize(volumeChange, -0.5, 0.5, true); // ±50% range
  }

  /**
   * Calculate price volatility over 1 minute
   */
  private calculateVolatility(data: ExchangePriceData): number {
    const key = `${data.exchange}_${data.pair}`;
    const history = this.priceHistory.get(key);
    
    if (!history || history.length < 5) return 0;

    const recentPrices = history.slice(-10).map(h => h.mid_price);
    const mean = recentPrices.reduce((a, b) => a + b, 0) / recentPrices.length;
    const variance = recentPrices.reduce((acc, price) => acc + Math.pow(price - mean, 2), 0) / recentPrices.length;
    const stdDev = Math.sqrt(variance);
    
    const volatilityPct = (stdDev / mean) * 100;
    return this.normalize(volatilityPct, 0, 2, false); // 0-2% volatility range
  }

  /**
   * Calculate cross-exchange arbitrage opportunity
   */
  private calculateArbitrageOpportunity(btcData: ExchangePriceData[]): number {
    if (btcData.length < 2) return 0;

    // Find the exchange with highest bid and lowest ask
    let highestBid = 0;
    let lowestAsk = Infinity;
    let highestBidExchange = '';
    let lowestAskExchange = '';

    btcData.forEach(data => {
      if (data.best_bid > highestBid) {
        highestBid = data.best_bid;
        highestBidExchange = data.exchange;
      }
      if (data.best_ask < lowestAsk) {
        lowestAsk = data.best_ask;
        lowestAskExchange = data.exchange;
      }
    });

    // Calculate potential arbitrage spread
    if (highestBidExchange !== lowestAskExchange && lowestAsk > 0) {
      const arbitrageSpread = (highestBid - lowestAsk) / lowestAsk;
      return this.normalize(arbitrageSpread, -0.01, 0.02, true); // -1% to +2% range
    }

    return 0;
  }

  /**
   * Calculate confidence based on data quality
   */
  private calculatePriceConfidence(allPriceData: ExchangePriceData[]): number {
    if (allPriceData.length === 0) return 0;

    const now = Date.now();
    const maxAge = 5000; // 5 seconds

    // Freshness factor
    const avgAge = allPriceData.reduce((sum, data) => sum + (now - data.timestamp), 0) / allPriceData.length;
    const freshnessFactor = Math.max(0, 1 - (avgAge / maxAge));

    // Exchange coverage factor
    const uniqueExchanges = new Set(allPriceData.map(d => d.exchange)).size;
    const coverageFactor = Math.min(1, uniqueExchanges / 3); // Target 3 exchanges

    // Liquidity factor
    const avgLiquidity = allPriceData.reduce((sum, data) => sum + data.orderbook_depth_usd_at_0_5pct, 0) / allPriceData.length;
    const liquidityFactor = Math.min(1, avgLiquidity / 100000); // $100k minimum

    return Math.max(0.1, freshnessFactor * coverageFactor * liquidityFactor);
  }

  /**
   * Normalize pair names across exchanges
   */
  private normalizePair(pair: string): string {
    return pair.replace(/[-\/]/g, '-').toUpperCase();
  }

  /**
   * Get default features when no data is available
   */
  private getDefaultFeatures(): PriceFeatures {
    return {
      spread_pct: 0.1,
      price_momentum_1m: 0,
      volume_momentum_1m: 0,
      liquidity_score: 0.5,
      volatility_1m: 0.5,
      arbitrage_opportunity: 0
    };
  }

  /**
   * Schedule reconnection to an exchange
   */
  private scheduleReconnect(exchangeName: string): void {
    const attempts = this.reconnectAttempts.get(exchangeName) || 0;
    
    if (attempts >= this.maxReconnectAttempts) {
      console.error(`Max reconnection attempts reached for ${exchangeName}`);
      return;
    }

    const delay = Math.min(30000, 1000 * Math.pow(2, attempts)); // Exponential backoff, max 30s
    
    setTimeout(async () => {
      try {
        console.log(`Reconnecting to ${exchangeName} (attempt ${attempts + 1})`);
        this.reconnectAttempts.set(exchangeName, attempts + 1);
        await this.connectToExchange(exchangeName);
      } catch (error) {
        console.error(`Reconnection failed for ${exchangeName}:`, error);
      }
    }, delay);
  }

  /**
   * Clean up WebSocket connections
   */
  async stop(): Promise<void> {
    // Close all WebSocket connections
    for (const [exchange, ws] of this.exchanges) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    }
    
    this.exchanges.clear();
    this.priceData.clear();
    this.orderBooks.clear();
    this.priceHistory.clear();
    
    await super.stop();
  }

  /**
   * Get price summary for debugging
   */
  getPriceSummary(): string {
    const summaries: string[] = [];
    
    for (const [key, data] of this.priceData) {
      const spread = ((data.best_ask - data.best_bid) / data.mid_price * 100).toFixed(3);
      const age = ((Date.now() - data.timestamp) / 1000).toFixed(1);
      summaries.push(`${key}: $${data.mid_price.toFixed(2)} (${spread}%, ${age}s old)`);
    }
    
    return summaries.length > 0 ? summaries.join(', ') : 'No price data available';
  }
}

/**
 * Factory function to create PriceAgent with config
 */
export function createPriceAgent(): PriceAgent {
  const config: AgentConfig = {
    name: 'price',
    enabled: true,
    polling_interval_ms: 1000, // 1 second for aggregation
    confidence_min: 0.6,
    data_age_max_ms: 5000, // 5 seconds max age
    retry_attempts: 3,
    retry_backoff_ms: 1000
  };

  return new PriceAgent(config);
}

// Export for testing
export { PriceAgent as default };