/**
 * REAL-TIME MARKET DATA FEEDS - Node.js Implementation
 * 
 * WebSocket connections to:
 * - Binance (Spot & Perpetual)
 * - Coinbase (Spot)
 * - Kraken (Spot)
 * - Bybit (Spot & Perpetual)
 * 
 * Aggregates into unified market data stream
 */

import WebSocket from 'ws';

export interface MarketDataFeed {
  symbol: string;
  exchange: string;
  spotPrice: number;
  perpPrice?: number;
  bidPrice: number;
  askPrice: number;
  bidSize: number;
  askSize: number;
  lastTradePrice: number;
  volume24h: number;
  fundingRate?: number;
  openInterest?: number;
  timestamp: Date;
}

export interface OrderBookSnapshot {
  exchange: string;
  symbol: string;
  bids: [number, number][]; // [price, size]
  asks: [number, number][];
  timestamp: Date;
}

export interface FundingRateData {
  exchange: string;
  symbol: string;
  fundingRate: number;
  nextFundingTime: Date;
  timestamp: Date;
}

export class RealtimeDataFeeds {
  private wsConnections: Map<string, WebSocket>;
  private dataCallbacks: Map<string, Function[]>;
  private reconnectIntervals: Map<string, NodeJS.Timeout>;
  private isConnected: Map<string, boolean>;
  
  // Data caches
  private latestPrices: Map<string, MarketDataFeed>;
  private orderBooks: Map<string, OrderBookSnapshot>;
  private fundingRates: Map<string, FundingRateData>;
  
  constructor() {
    this.wsConnections = new Map();
    this.dataCallbacks = new Map();
    this.reconnectIntervals = new Map();
    this.isConnected = new Map();
    this.latestPrices = new Map();
    this.orderBooks = new Map();
    this.fundingRates = new Map();
  }
  
  /**
   * Connect to Binance WebSocket
   */
  async connectBinance(symbols: string[] = ['BTCUSDT']): Promise<void> {
    const streams = symbols.map(s => `${s.toLowerCase()}@ticker`).join('/');
    const wsUrl = `wss://stream.binance.com:9443/stream?streams=${streams}`;
    
    return new Promise((resolve, reject) => {
      try {
        const ws = new WebSocket(wsUrl);
        
        ws.on('open', () => {
          console.log('✅ Binance WebSocket connected');
          this.isConnected.set('binance', true);
          resolve();
        });
        
        ws.on('message', (data: Buffer) => {
          try {
            const message = JSON.parse(data.toString());
            if (message.stream && message.data) {
              this.handleBinanceMessage(message.data);
            }
          } catch (error) {
            console.error('❌ Binance message parse error:', error);
          }
        });
        
        ws.on('error', (error) => {
          console.error('❌ Binance WebSocket error:', error);
          reject(error);
        });
        
        ws.on('close', () => {
          console.warn('⚠️  Binance WebSocket closed, reconnecting...');
          this.isConnected.set('binance', false);
          this.scheduleReconnect('binance', () => this.connectBinance(symbols));
        });
        
        this.wsConnections.set('binance', ws);
      } catch (error) {
        console.error('Failed to connect to Binance:', error);
        reject(error);
      }
    });
  }
  
  /**
   * Connect to Coinbase WebSocket
   */
  async connectCoinbase(symbols: string[] = ['BTC-USD']): Promise<void> {
    const wsUrl = 'wss://ws-feed.exchange.coinbase.com';
    
    return new Promise((resolve, reject) => {
      try {
        const ws = new WebSocket(wsUrl);
        
        ws.on('open', () => {
          console.log('✅ Coinbase WebSocket connected');
          this.isConnected.set('coinbase', true);
          
          // Subscribe to ticker channel
          ws.send(JSON.stringify({
            type: 'subscribe',
            product_ids: symbols,
            channels: ['ticker']
          }));
          
          resolve();
        });
        
        ws.on('message', (data: Buffer) => {
          try {
            const message = JSON.parse(data.toString());
            if (message.type === 'ticker') {
              this.handleCoinbaseMessage(message);
            }
          } catch (error) {
            console.error('❌ Coinbase message parse error:', error);
          }
        });
        
        ws.on('error', (error) => {
          console.error('❌ Coinbase WebSocket error:', error);
          reject(error);
        });
        
        ws.on('close', () => {
          console.warn('⚠️  Coinbase WebSocket closed, reconnecting...');
          this.isConnected.set('coinbase', false);
          this.scheduleReconnect('coinbase', () => this.connectCoinbase(symbols));
        });
        
        this.wsConnections.set('coinbase', ws);
      } catch (error) {
        console.error('Failed to connect to Coinbase:', error);
        reject(error);
      }
    });
  }
  
  /**
   * Connect to Kraken WebSocket
   */
  async connectKraken(symbols: string[] = ['XBT/USD']): Promise<void> {
    const wsUrl = 'wss://ws.kraken.com';
    
    return new Promise((resolve, reject) => {
      try {
        const ws = new WebSocket(wsUrl);
        
        ws.on('open', () => {
          console.log('✅ Kraken WebSocket connected');
          this.isConnected.set('kraken', true);
          
          // Subscribe to ticker
          ws.send(JSON.stringify({
            event: 'subscribe',
            pair: symbols,
            subscription: { name: 'ticker' }
          }));
          
          resolve();
        });
        
        ws.on('message', (data: Buffer) => {
          try {
            const message = JSON.parse(data.toString());
            if (Array.isArray(message) && message[2] === 'ticker') {
              this.handleKrakenMessage(message);
            }
          } catch (error) {
            console.error('❌ Kraken message parse error:', error);
          }
        });
        
        ws.on('error', (error) => {
          console.error('❌ Kraken WebSocket error:', error);
          reject(error);
        });
        
        ws.on('close', () => {
          console.warn('⚠️  Kraken WebSocket closed, reconnecting...');
          this.isConnected.set('kraken', false);
          this.scheduleReconnect('kraken', () => this.connectKraken(symbols));
        });
        
        this.wsConnections.set('kraken', ws);
      } catch (error) {
        console.error('Failed to connect to Kraken:', error);
        reject(error);
      }
    });
  }
  
  /**
   * Handle Binance ticker message
   */
  private handleBinanceMessage(data: any): void {
    const feed: MarketDataFeed = {
      symbol: data.s,
      exchange: 'binance',
      spotPrice: parseFloat(data.c),
      bidPrice: parseFloat(data.b),
      askPrice: parseFloat(data.a),
      bidSize: parseFloat(data.B),
      askSize: parseFloat(data.A),
      lastTradePrice: parseFloat(data.c),
      volume24h: parseFloat(data.v),
      timestamp: new Date(data.E)
    };
    
    this.latestPrices.set(`binance_${data.s}`, feed);
    this.triggerCallbacks('price_update', feed);
  }
  
  /**
   * Handle Coinbase ticker message
   */
  private handleCoinbaseMessage(data: any): void {
    const feed: MarketDataFeed = {
      symbol: data.product_id,
      exchange: 'coinbase',
      spotPrice: parseFloat(data.price),
      bidPrice: parseFloat(data.best_bid),
      askPrice: parseFloat(data.best_ask),
      bidSize: 0,
      askSize: 0,
      lastTradePrice: parseFloat(data.price),
      volume24h: parseFloat(data.volume_24h || 0),
      timestamp: new Date(data.time)
    };
    
    this.latestPrices.set(`coinbase_${data.product_id}`, feed);
    this.triggerCallbacks('price_update', feed);
  }
  
  /**
   * Handle Kraken ticker message
   */
  private handleKrakenMessage(data: any): void {
    const tickerData = data[1];
    const pair = data[3];
    
    const feed: MarketDataFeed = {
      symbol: pair,
      exchange: 'kraken',
      spotPrice: parseFloat(tickerData.c[0]),
      bidPrice: parseFloat(tickerData.b[0]),
      askPrice: parseFloat(tickerData.a[0]),
      bidSize: parseFloat(tickerData.b[1]),
      askSize: parseFloat(tickerData.a[1]),
      lastTradePrice: parseFloat(tickerData.c[0]),
      volume24h: parseFloat(tickerData.v[1]),
      timestamp: new Date()
    };
    
    this.latestPrices.set(`kraken_${pair}`, feed);
    this.triggerCallbacks('price_update', feed);
  }
  
  /**
   * Schedule reconnection
   */
  private scheduleReconnect(exchange: string, connectFn: () => void): void {
    // Clear existing interval
    const existing = this.reconnectIntervals.get(exchange);
    if (existing) {
      clearTimeout(existing);
    }
    
    // Reconnect after 5 seconds
    const interval = setTimeout(() => {
      console.log(`🔄 Reconnecting to ${exchange}...`);
      connectFn();
    }, 5000);
    
    this.reconnectIntervals.set(exchange, interval);
  }
  
  /**
   * Subscribe to data updates
   */
  onPriceUpdate(callback: (feed: MarketDataFeed) => void): void {
    if (!this.dataCallbacks.has('price_update')) {
      this.dataCallbacks.set('price_update', []);
    }
    this.dataCallbacks.get('price_update')!.push(callback);
  }
  
  /**
   * Trigger callbacks
   */
  private triggerCallbacks(event: string, data: any): void {
    const callbacks = this.dataCallbacks.get(event);
    if (callbacks) {
      callbacks.forEach(cb => cb(data));
    }
  }
  
  /**
   * Get latest price for symbol across exchanges
   */
  getLatestPrices(symbol: string): MarketDataFeed[] {
    const feeds: MarketDataFeed[] = [];
    
    for (const [key, feed] of this.latestPrices.entries()) {
      if (feed.symbol.includes(symbol.replace('-', '').replace('/', ''))) {
        feeds.push(feed);
      }
    }
    
    return feeds;
  }
  
  /**
   * Calculate cross-exchange spread
   */
  calculateSpread(symbol: string): { spread: number; buyExchange: string; sellExchange: string; profitable: boolean } {
    const feeds = this.getLatestPrices(symbol);
    
    if (feeds.length < 2) {
      return { spread: 0, buyExchange: '', sellExchange: '', profitable: false };
    }
    
    // Find cheapest and most expensive
    let cheapest = feeds[0];
    let expensive = feeds[0];
    
    for (const feed of feeds) {
      if (feed.askPrice < cheapest.askPrice) cheapest = feed;
      if (feed.bidPrice > expensive.bidPrice) expensive = feed;
    }
    
    const spread = ((expensive.bidPrice - cheapest.askPrice) / cheapest.askPrice) * 10000; // bps
    const fees = 20; // Typical trading fees (bps)
    
    return {
      spread,
      buyExchange: cheapest.exchange,
      sellExchange: expensive.exchange,
      profitable: spread > fees
    };
  }
  
  /**
   * Get connection status
   */
  getConnectionStatus(): Record<string, boolean> {
    return Object.fromEntries(this.isConnected);
  }
  
  /**
   * Disconnect all
   */
  disconnectAll(): void {
    for (const [exchange, ws] of this.wsConnections.entries()) {
      console.log(`🔌 Disconnecting ${exchange}...`);
      ws.close();
    }
    
    for (const [exchange, interval] of this.reconnectIntervals.entries()) {
      clearTimeout(interval);
    }
    
    this.wsConnections.clear();
    this.reconnectIntervals.clear();
    this.isConnected.clear();
  }
}

/**
 * Singleton instance
 */
export const realtimeFeeds = new RealtimeDataFeeds();
