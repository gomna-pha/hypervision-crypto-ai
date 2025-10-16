/**
 * Price Agent - Real-Time Exchange Data Collection
 * Connects to multiple exchange WebSocket feeds for real-time price and orderbook data
 * Provides canonical pricing and spread analysis across exchanges
 */

import { BaseAgent, AgentConfig, AgentOutput } from '../../core/base-agent.js';
import WebSocket from 'ws';
import axios from 'axios';
import crypto from 'crypto';

export interface PriceData {
  exchange: string;
  pair: string;
  best_bid: number;
  best_ask: number;
  mid_price: number;
  last_trade_price: number;
  volume_1m: number;
  vwap_1m: number;
  orderbook_depth_usd: number;
  open_interest?: number;
  funding_rate?: number;
  timestamp: number;
}

export interface SpreadAnalysis {
  pair: string;
  exchanges: string[];
  best_bid_exchange: string;
  best_ask_exchange: string;
  current_spread_pct: number;
  arbitrage_opportunity: boolean;
  potential_profit_usd: number;
  volume_weighted_spread: number;
}

export interface PriceFeatures {
  btc_binance_mid: number;
  btc_coinbase_mid: number;
  eth_binance_mid: number;
  eth_coinbase_mid: number;
  current_spread_pct: number;
  volume_weighted_price: number;
  price_momentum: number;
  volatility_1m: number;
  liquidity_score: number;
}

export class PriceAgent extends BaseAgent {
  private exchanges: Map<string, WebSocket> = new Map();
  private priceData: Map<string, PriceData> = new Map();
  private tradeHistory: Array<{ price: number; volume: number; timestamp: number }> = [];
  private apiKeys: Map<string, { key: string; secret: string }> = new Map();
  private reconnectTimeouts: Map<string, NodeJS.Timeout> = new Map();
  
  // Configuration
  private readonly PAIRS = ['BTC-USDT', 'ETH-USDT', 'BTC-USD', 'ETH-USD'];
  private readonly EXCHANGES = ['binance', 'coinbase', 'kraken'];
  private readonly MIN_ORDERBOOK_DEPTH_USD = 100000;
  private readonly MAX_LATENCY_MS = 200;

  constructor(config: AgentConfig, apiKeys?: Map<string, { key: string; secret: string }>) {
    super(config);
    if (apiKeys) {
      this.apiKeys = apiKeys;
    }
  }

  protected async collectData(): Promise<AgentOutput> {
    const timestamp = this.getCurrentTimestamp();

    try {
      // Ensure WebSocket connections are active
      await this.ensureConnections();
      
      // Get latest price data
      const prices = this.getLatestPrices();
      
      if (prices.length === 0) {
        throw new Error('No price data available');
      }
      
      // Calculate features
      const features = this.calculateFeatures(prices);
      
      // Calculate spread analysis
      const spreadAnalysis = this.analyzeArbitrageSpreads();
      
      // Calculate key signal (arbitrage opportunity score)
      const keySignal = this.calculateKeySignal(features, spreadAnalysis);
      
      // Calculate confidence based on data quality
      const confidence = this.calculateDataConfidence(prices);
      
      return {
        agent_name: 'PriceAgent',
        timestamp,
        key_signal: keySignal,
        confidence,
        features: {
          ...features,
          spread_analysis: spreadAnalysis,
          active_exchanges: this.exchanges.size,
          data_points: prices.length
        },
        metadata: {
          exchanges_connected: Array.from(this.exchanges.keys()),
          pairs_tracked: this.PAIRS,
          last_update_times: this.getLastUpdateTimes()
        }
      };

    } catch (error) {
      console.error('PriceAgent data collection failed:', error);
      throw error;
    }
  }

  /**
   * Ensure WebSocket connections to all exchanges
   */
  private async ensureConnections(): Promise<void> {
    for (const exchange of this.EXCHANGES) {
      if (!this.exchanges.has(exchange) || 
          this.exchanges.get(exchange)?.readyState !== WebSocket.OPEN) {
        await this.connectToExchange(exchange);
      }
    }
  }

  /**
   * Connect to specific exchange WebSocket
   */
  private async connectToExchange(exchange: string): Promise<void> {
    try {
      let ws: WebSocket;
      
      switch (exchange) {
        case 'binance':
          ws = await this.connectBinance();
          break;
        case 'coinbase':
          ws = await this.connectCoinbase();
          break;
        case 'kraken':
          ws = await this.connectKraken();
          break;
        default:
          throw new Error(`Unknown exchange: ${exchange}`);
      }
      
      this.exchanges.set(exchange, ws);
      console.log(`Connected to ${exchange} WebSocket`);
      
    } catch (error) {
      console.error(`Failed to connect to ${exchange}:`, error);
      this.scheduleReconnect(exchange);
    }
  }

  /**
   * Connect to Binance WebSocket
   */
  private async connectBinance(): Promise<WebSocket> {
    const url = 'wss://stream.binance.com:9443/ws/btcusdt@depth@100ms/ethusdt@depth@100ms/btcusdt@aggTrade/ethusdt@aggTrade';
    
    const ws = new WebSocket(url);
    
    ws.on('open', () => {
      console.log('Binance WebSocket connected');
    });
    
    ws.on('message', (data: Buffer) => {
      try {
        const message = JSON.parse(data.toString());
        this.processBinanceMessage(message);
      } catch (error) {
        console.error('Binance message parsing error:', error);
      }
    });
    
    ws.on('error', (error) => {
      console.error('Binance WebSocket error:', error);
      this.scheduleReconnect('binance');
    });
    
    ws.on('close', () => {
      console.log('Binance WebSocket closed');
      this.scheduleReconnect('binance');
    });
    
    return ws;
  }

  /**
   * Connect to Coinbase WebSocket
   */
  private async connectCoinbase(): Promise<WebSocket> {
    const url = 'wss://ws-feed.exchange.coinbase.com';
    
    const ws = new WebSocket(url);
    
    ws.on('open', () => {
      // Subscribe to level2 and matches channels
      const subscribeMessage = {
        type: 'subscribe',
        product_ids: ['BTC-USD', 'ETH-USD', 'BTC-USDT', 'ETH-USDT'],
        channels: ['level2', 'matches', 'ticker']
      };
      
      ws.send(JSON.stringify(subscribeMessage));
      console.log('Coinbase WebSocket connected and subscribed');
    });
    
    ws.on('message', (data: Buffer) => {
      try {
        const message = JSON.parse(data.toString());
        this.processCoinbaseMessage(message);
      } catch (error) {
        console.error('Coinbase message parsing error:', error);
      }
    });
    
    ws.on('error', (error) => {
      console.error('Coinbase WebSocket error:', error);
      this.scheduleReconnect('coinbase');
    });
    
    ws.on('close', () => {
      console.log('Coinbase WebSocket closed');
      this.scheduleReconnect('coinbase');
    });
    
    return ws;
  }

  /**
   * Connect to Kraken WebSocket
   */
  private async connectKraken(): Promise<WebSocket> {
    const url = 'wss://ws.kraken.com';
    
    const ws = new WebSocket(url);
    
    ws.on('open', () => {
      // Subscribe to book and trade channels
      const subscribeMessage = {
        event: 'subscribe',
        pair: ['XBT/USD', 'ETH/USD', 'XBT/USDT', 'ETH/USDT'],
        subscription: { name: 'book', depth: 100 }
      };
      
      ws.send(JSON.stringify(subscribeMessage));
      
      const tradeMessage = {
        event: 'subscribe',
        pair: ['XBT/USD', 'ETH/USD'],
        subscription: { name: 'trade' }
      };
      
      ws.send(JSON.stringify(tradeMessage));
      console.log('Kraken WebSocket connected and subscribed');
    });
    
    ws.on('message', (data: Buffer) => {
      try {
        const message = JSON.parse(data.toString());
        this.processKrakenMessage(message);
      } catch (error) {
        console.error('Kraken message parsing error:', error);
      }
    });
    
    ws.on('error', (error) => {
      console.error('Kraken WebSocket error:', error);
      this.scheduleReconnect('kraken');
    });
    
    ws.on('close', () => {
      console.log('Kraken WebSocket closed');
      this.scheduleReconnect('kraken');
    });
    
    return ws;
  }

  /**
   * Process Binance WebSocket messages
   */
  private processBinanceMessage(message: any): void {
    if (message.stream && message.data) {
      const stream = message.stream;
      const data = message.data;
      
      if (stream.includes('@depth')) {
        // Orderbook update
        this.processBinanceOrderbook(stream, data);
      } else if (stream.includes('@aggTrade')) {
        // Trade update
        this.processBinanceTrade(stream, data);
      }
    }
  }

  /**
   * Process Binance orderbook data
   */
  private processBinanceOrderbook(stream: string, data: any): void {
    const symbol = stream.split('@')[0].toUpperCase();
    const pair = this.normalizePair(symbol);
    
    const bestBid = data.b && data.b.length > 0 ? parseFloat(data.b[0][0]) : 0;
    const bestAsk = data.a && data.a.length > 0 ? parseFloat(data.a[0][0]) : 0;
    
    if (bestBid > 0 && bestAsk > 0) {
      const key = `binance_${pair}`;
      const existing = this.priceData.get(key) || {} as PriceData;
      
      this.priceData.set(key, {
        ...existing,
        exchange: 'binance',
        pair,
        best_bid: bestBid,
        best_ask: bestAsk,
        mid_price: (bestBid + bestAsk) / 2,
        orderbook_depth_usd: this.calculateOrderbookDepth(data),
        timestamp: Date.now()
      });
    }
  }

  /**
   * Process Binance trade data
   */
  private processBinanceTrade(stream: string, data: any): void {
    const symbol = stream.split('@')[0].toUpperCase();
    const pair = this.normalizePair(symbol);
    const price = parseFloat(data.p);
    const quantity = parseFloat(data.q);
    
    // Update trade history
    this.tradeHistory.push({
      price,
      volume: quantity,
      timestamp: data.T
    });
    
    // Keep only last 1000 trades
    if (this.tradeHistory.length > 1000) {
      this.tradeHistory.shift();
    }
    
    // Update price data
    const key = `binance_${pair}`;
    const existing = this.priceData.get(key) || {} as PriceData;
    
    this.priceData.set(key, {
      ...existing,
      exchange: 'binance',
      pair,
      last_trade_price: price,
      vwap_1m: this.calculateVWAP(1),
      volume_1m: this.calculateVolume(1),
      timestamp: Date.now()
    });
  }

  /**
   * Process Coinbase WebSocket messages
   */
  private processCoinbaseMessage(message: any): void {
    if (message.type === 'l2update') {
      this.processCoinbaseL2Update(message);
    } else if (message.type === 'match') {
      this.processCoinbaseMatch(message);
    } else if (message.type === 'ticker') {
      this.processCoinbaseTicker(message);
    }
  }

  /**
   * Process Coinbase L2 orderbook updates
   */
  private processCoinbaseL2Update(message: any): void {
    const pair = this.normalizePair(message.product_id);
    
    // Process changes to find best bid/ask
    let bestBid = 0;
    let bestAsk = Infinity;
    
    for (const change of message.changes) {
      const [side, price, size] = change;
      const priceNum = parseFloat(price);
      const sizeNum = parseFloat(size);
      
      if (side === 'buy' && sizeNum > 0) {
        bestBid = Math.max(bestBid, priceNum);
      } else if (side === 'sell' && sizeNum > 0) {
        bestAsk = Math.min(bestAsk, priceNum);
      }
    }
    
    if (bestBid > 0 && bestAsk < Infinity) {
      const key = `coinbase_${pair}`;
      const existing = this.priceData.get(key) || {} as PriceData;
      
      this.priceData.set(key, {
        ...existing,
        exchange: 'coinbase',
        pair,
        best_bid: bestBid,
        best_ask: bestAsk,
        mid_price: (bestBid + bestAsk) / 2,
        timestamp: Date.now()
      });
    }
  }

  /**
   * Process Coinbase match (trade) data
   */
  private processCoinbaseMatch(message: any): void {
    const pair = this.normalizePair(message.product_id);
    const price = parseFloat(message.price);
    const size = parseFloat(message.size);
    
    this.tradeHistory.push({
      price,
      volume: size,
      timestamp: new Date(message.time).getTime()
    });
    
    if (this.tradeHistory.length > 1000) {
      this.tradeHistory.shift();
    }
  }

  /**
   * Process Coinbase ticker data
   */
  private processCoinbaseTicker(message: any): void {
    const pair = this.normalizePair(message.product_id);
    const key = `coinbase_${pair}`;
    const existing = this.priceData.get(key) || {} as PriceData;
    
    this.priceData.set(key, {
      ...existing,
      exchange: 'coinbase',
      pair,
      last_trade_price: parseFloat(message.price),
      volume_1m: parseFloat(message.volume_24h) / (24 * 60), // Approximate 1m volume
      timestamp: Date.now()
    });
  }

  /**
   * Process Kraken WebSocket messages
   */
  private processKrakenMessage(message: any): void {
    if (Array.isArray(message) && message.length > 0) {
      const channelName = message[message.length - 1];
      
      if (typeof channelName === 'string' && channelName.includes('book')) {
        this.processKrakenOrderbook(message);
      } else if (typeof channelName === 'string' && channelName.includes('trade')) {
        this.processKrakenTrade(message);
      }
    }
  }

  /**
   * Process Kraken orderbook data
   */
  private processKrakenOrderbook(message: any): void {
    if (message.length >= 2) {
      const data = message[1];
      const pairName = message[message.length - 1].split('-')[0];
      const pair = this.normalizeKrakenPair(pairName);
      
      if (data.b && data.a) {
        const bestBid = data.b.length > 0 ? parseFloat(data.b[0][0]) : 0;
        const bestAsk = data.a.length > 0 ? parseFloat(data.a[0][0]) : 0;
        
        if (bestBid > 0 && bestAsk > 0) {
          const key = `kraken_${pair}`;
          const existing = this.priceData.get(key) || {} as PriceData;
          
          this.priceData.set(key, {
            ...existing,
            exchange: 'kraken',
            pair,
            best_bid: bestBid,
            best_ask: bestAsk,
            mid_price: (bestBid + bestAsk) / 2,
            timestamp: Date.now()
          });
        }
      }
    }
  }

  /**
   * Process Kraken trade data
   */
  private processKrakenTrade(message: any): void {
    if (message.length >= 2 && message[1]) {
      const trades = message[1];
      const pairName = message[message.length - 1].split('-')[0];
      
      for (const trade of trades) {
        const price = parseFloat(trade[0]);
        const volume = parseFloat(trade[1]);
        const timestamp = parseFloat(trade[2]) * 1000; // Convert to milliseconds
        
        this.tradeHistory.push({ price, volume, timestamp });
      }
      
      if (this.tradeHistory.length > 1000) {
        this.tradeHistory.splice(0, this.tradeHistory.length - 1000);
      }
    }
  }

  /**
   * Calculate orderbook depth in USD
   */
  private calculateOrderbookDepth(orderbook: any): number {
    let totalDepth = 0;
    
    // Calculate bid side depth
    if (orderbook.b) {
      for (const [price, quantity] of orderbook.b.slice(0, 10)) {
        totalDepth += parseFloat(price) * parseFloat(quantity);
      }
    }
    
    // Calculate ask side depth
    if (orderbook.a) {
      for (const [price, quantity] of orderbook.a.slice(0, 10)) {
        totalDepth += parseFloat(price) * parseFloat(quantity);
      }
    }
    
    return totalDepth;
  }

  /**
   * Calculate Volume Weighted Average Price (VWAP)
   */
  private calculateVWAP(windowMinutes: number): number {
    const windowMs = windowMinutes * 60 * 1000;
    const now = Date.now();
    const cutoff = now - windowMs;
    
    const recentTrades = this.tradeHistory.filter(t => t.timestamp >= cutoff);
    
    if (recentTrades.length === 0) return 0;
    
    let totalValue = 0;
    let totalVolume = 0;
    
    for (const trade of recentTrades) {
      totalValue += trade.price * trade.volume;
      totalVolume += trade.volume;
    }
    
    return totalVolume > 0 ? totalValue / totalVolume : 0;
  }

  /**
   * Calculate volume in time window
   */
  private calculateVolume(windowMinutes: number): number {
    const windowMs = windowMinutes * 60 * 1000;
    const now = Date.now();
    const cutoff = now - windowMs;
    
    const recentTrades = this.tradeHistory.filter(t => t.timestamp >= cutoff);
    
    return recentTrades.reduce((total, trade) => total + trade.volume, 0);
  }

  /**
   * Normalize pair names across exchanges
   */
  private normalizePair(pair: string): string {
    const normalized = pair.toUpperCase()
      .replace('USDT', '-USDT')
      .replace('USD', '-USD')
      .replace('BTC', 'BTC')
      .replace('ETH', 'ETH');
    
    // Ensure proper format
    if (!normalized.includes('-')) {
      if (normalized.includes('BTC')) {
        return normalized.replace('BTC', 'BTC-');
      } else if (normalized.includes('ETH')) {
        return normalized.replace('ETH', 'ETH-');
      }
    }
    
    return normalized;
  }

  /**
   * Normalize Kraken pair names
   */
  private normalizeKrakenPair(pair: string): string {
    const mapping: Record<string, string> = {
      'XBT/USD': 'BTC-USD',
      'XBT/USDT': 'BTC-USDT',
      'ETH/USD': 'ETH-USD',
      'ETH/USDT': 'ETH-USDT'
    };
    
    return mapping[pair] || pair;
  }

  /**
   * Get latest price data
   */
  private getLatestPrices(): PriceData[] {
    const now = Date.now();
    const maxAge = 5000; // 5 seconds
    
    return Array.from(this.priceData.values())
      .filter(price => (now - price.timestamp) < maxAge);
  }

  /**
   * Analyze arbitrage spreads across exchanges
   */
  private analyzeArbitrageSpreads(): SpreadAnalysis[] {
    const spreadAnalyses: SpreadAnalysis[] = [];
    
    for (const pair of this.PAIRS) {
      const pairPrices = Array.from(this.priceData.values())
        .filter(p => p.pair === pair);
      
      if (pairPrices.length >= 2) {
        // Find best bid and ask across exchanges
        let bestBid = 0;
        let bestBidExchange = '';
        let bestAsk = Infinity;
        let bestAskExchange = '';
        
        for (const price of pairPrices) {
          if (price.best_bid > bestBid) {
            bestBid = price.best_bid;
            bestBidExchange = price.exchange;
          }
          
          if (price.best_ask < bestAsk) {
            bestAsk = price.best_ask;
            bestAskExchange = price.exchange;
          }
        }
        
        if (bestBid > 0 && bestAsk < Infinity && bestBidExchange !== bestAskExchange) {
          const spreadPct = ((bestBid - bestAsk) / bestAsk) * 100;
          const arbitrageOpportunity = spreadPct > 0.05; // 0.05% minimum
          
          spreadAnalyses.push({
            pair,
            exchanges: pairPrices.map(p => p.exchange),
            best_bid_exchange: bestBidExchange,
            best_ask_exchange: bestAskExchange,
            current_spread_pct: Math.abs(spreadPct),
            arbitrage_opportunity: arbitrageOpportunity,
            potential_profit_usd: arbitrageOpportunity ? (bestBid - bestAsk) * 100 : 0, // Assume 100 units
            volume_weighted_spread: this.calculateVolumeWeightedSpread(pairPrices)
          });
        }
      }
    }
    
    return spreadAnalyses;
  }

  /**
   * Calculate volume-weighted spread
   */
  private calculateVolumeWeightedSpread(prices: PriceData[]): number {
    let totalWeightedSpread = 0;
    let totalVolume = 0;
    
    for (const price of prices) {
      if (price.best_bid > 0 && price.best_ask > 0) {
        const spread = ((price.best_ask - price.best_bid) / price.best_bid) * 100;
        const volume = price.volume_1m || 1;
        
        totalWeightedSpread += spread * volume;
        totalVolume += volume;
      }
    }
    
    return totalVolume > 0 ? totalWeightedSpread / totalVolume : 0;
  }

  /**
   * Calculate derived features
   */
  private calculateFeatures(prices: PriceData[]): PriceFeatures {
    // Get specific exchange-pair prices
    const btcBinance = prices.find(p => p.exchange === 'binance' && p.pair.includes('BTC'));
    const btcCoinbase = prices.find(p => p.exchange === 'coinbase' && p.pair.includes('BTC'));
    const ethBinance = prices.find(p => p.exchange === 'binance' && p.pair.includes('ETH'));
    const ethCoinbase = prices.find(p => p.exchange === 'coinbase' && p.pair.includes('ETH'));
    
    // Calculate current spread
    let currentSpreadPct = 0;
    if (btcBinance && btcCoinbase) {
      const spreadAbs = Math.abs(btcBinance.mid_price - btcCoinbase.mid_price);
      currentSpreadPct = (spreadAbs / Math.min(btcBinance.mid_price, btcCoinbase.mid_price)) * 100;
    }
    
    // Volume weighted price
    const totalVolume = prices.reduce((sum, p) => sum + (p.volume_1m || 0), 0);
    const volumeWeightedPrice = totalVolume > 0 
      ? prices.reduce((sum, p) => sum + (p.mid_price * (p.volume_1m || 0)), 0) / totalVolume
      : 0;
    
    // Price momentum (simplified)
    const priceMomentum = this.calculatePriceMomentum();
    
    // Volatility (1 minute)
    const volatility1m = this.calculateVolatility(1);
    
    // Liquidity score
    const liquidityScore = this.calculateLiquidityScore(prices);
    
    return {
      btc_binance_mid: btcBinance?.mid_price || 0,
      btc_coinbase_mid: btcCoinbase?.mid_price || 0,
      eth_binance_mid: ethBinance?.mid_price || 0,
      eth_coinbase_mid: ethCoinbase?.mid_price || 0,
      current_spread_pct: currentSpreadPct,
      volume_weighted_price: volumeWeightedPrice,
      price_momentum: priceMomentum,
      volatility_1m: volatility1m,
      liquidity_score: liquidityScore
    };
  }

  /**
   * Calculate price momentum
   */
  private calculatePriceMomentum(): number {
    if (this.tradeHistory.length < 10) return 0;
    
    const recent = this.tradeHistory.slice(-10);
    const older = this.tradeHistory.slice(-20, -10);
    
    if (older.length === 0) return 0;
    
    const recentAvg = recent.reduce((sum, t) => sum + t.price, 0) / recent.length;
    const olderAvg = older.reduce((sum, t) => sum + t.price, 0) / older.length;
    
    return ((recentAvg - olderAvg) / olderAvg) * 100;
  }

  /**
   * Calculate price volatility
   */
  private calculateVolatility(windowMinutes: number): number {
    const windowMs = windowMinutes * 60 * 1000;
    const now = Date.now();
    const cutoff = now - windowMs;
    
    const recentTrades = this.tradeHistory.filter(t => t.timestamp >= cutoff);
    
    if (recentTrades.length < 2) return 0;
    
    const prices = recentTrades.map(t => t.price);
    const mean = prices.reduce((sum, p) => sum + p, 0) / prices.length;
    const variance = prices.reduce((sum, p) => sum + Math.pow(p - mean, 2), 0) / prices.length;
    
    return Math.sqrt(variance) / mean * 100; // Coefficient of variation
  }

  /**
   * Calculate liquidity score
   */
  private calculateLiquidityScore(prices: PriceData[]): number {
    if (prices.length === 0) return 0;
    
    const avgDepth = prices.reduce((sum, p) => sum + (p.orderbook_depth_usd || 0), 0) / prices.length;
    
    // Normalize to 0-1 scale based on minimum required depth
    return Math.min(1, avgDepth / this.MIN_ORDERBOOK_DEPTH_USD);
  }

  /**
   * Calculate key signal (arbitrage opportunity score)
   */
  private calculateKeySignal(features: PriceFeatures, spreads: SpreadAnalysis[]): number {
    // Weight different factors
    const weights = {
      spread: 0.4,        // Current spread opportunities
      liquidity: 0.3,     // Available liquidity
      momentum: 0.2,      // Price momentum
      volatility: 0.1     // Market volatility
    };
    
    // Spread signal (higher spread = higher signal)
    const maxSpread = Math.max(...spreads.map(s => s.current_spread_pct), 0);
    const spreadSignal = Math.min(1, maxSpread / 0.5); // Normalize to 0.5% max
    
    // Liquidity signal
    const liquiditySignal = features.liquidity_score;
    
    // Momentum signal (absolute value, we want movement)
    const momentumSignal = Math.min(1, Math.abs(features.price_momentum) / 2);
    
    // Volatility signal (moderate volatility is good for arbitrage)
    const volatilitySignal = features.volatility_1m > 0 
      ? Math.min(1, 2 - Math.abs(features.volatility_1m - 1)) // Optimal around 1%
      : 0;
    
    // Weighted composite
    const signal = (
      spreadSignal * weights.spread +
      liquiditySignal * weights.liquidity +
      momentumSignal * weights.momentum +
      volatilitySignal * weights.volatility
    );
    
    return Math.max(0, Math.min(1, signal));
  }

  /**
   * Calculate confidence based on data quality
   */
  private calculateDataConfidence(prices: PriceData[]): number {
    if (prices.length === 0) return 0;
    
    const now = Date.now();
    let totalConfidence = 0;
    
    for (const price of prices) {
      // Data freshness factor
      const age = now - price.timestamp;
      const freshnessFactor = Math.max(0, 1 - (age / 5000)); // 5 second max age
      
      // Data completeness factor
      const requiredFields = ['best_bid', 'best_ask', 'mid_price'];
      const completeness = requiredFields.filter(field => 
        price[field as keyof PriceData] && price[field as keyof PriceData] > 0
      ).length / requiredFields.length;
      
      // Exchange reliability factor (simplified)
      const reliabilityFactors: Record<string, number> = {
        binance: 0.95,
        coinbase: 0.90,
        kraken: 0.85
      };
      const reliabilityFactor = reliabilityFactors[price.exchange] || 0.7;
      
      totalConfidence += freshnessFactor * completeness * reliabilityFactor;
    }
    
    // Average confidence across all price sources
    const baseConfidence = totalConfidence / prices.length;
    
    // Bonus for having multiple exchanges
    const diversityBonus = Math.min(0.2, (prices.length - 1) * 0.1);
    
    return Math.max(0.1, Math.min(1, baseConfidence + diversityBonus));
  }

  /**
   * Schedule reconnection for failed exchange
   */
  private scheduleReconnect(exchange: string): void {
    // Clear existing timeout
    const existingTimeout = this.reconnectTimeouts.get(exchange);
    if (existingTimeout) {
      clearTimeout(existingTimeout);
    }
    
    // Schedule reconnection with exponential backoff
    const reconnectDelay = Math.min(30000, 1000 * Math.pow(2, this.errorCount)); // Max 30 seconds
    
    const timeout = setTimeout(async () => {
      console.log(`Attempting to reconnect to ${exchange}...`);
      await this.connectToExchange(exchange);
    }, reconnectDelay);
    
    this.reconnectTimeouts.set(exchange, timeout);
  }

  /**
   * Get last update times for each exchange
   */
  private getLastUpdateTimes(): Record<string, number> {
    const updateTimes: Record<string, number> = {};
    
    for (const [key, price] of this.priceData) {
      updateTimes[key] = price.timestamp;
    }
    
    return updateTimes;
  }

  /**
   * Get price summary for debugging
   */
  getPriceSummary(): string {
    const prices = this.getLatestPrices();
    if (prices.length === 0) return 'No price data available';
    
    const summaries = prices.map(p => 
      `${p.exchange} ${p.pair}: $${p.mid_price.toFixed(2)} (${p.current_spread_pct?.toFixed(3)}%)`
    );
    
    return summaries.join(', ');
  }

  /**
   * Cleanup on agent stop
   */
  async stop(): Promise<void> {
    // Close all WebSocket connections
    for (const [exchange, ws] of this.exchanges) {
      if (ws.readyState === WebSocket.OPEN) {
        ws.close();
      }
    }
    
    // Clear reconnect timers
    for (const timeout of this.reconnectTimeouts.values()) {
      clearTimeout(timeout);
    }
    
    this.exchanges.clear();
    this.reconnectTimeouts.clear();
    
    await super.stop();
  }
}

/**
 * Factory function to create PriceAgent with config
 */
export function createPriceAgent(apiKeys?: Map<string, { key: string; secret: string }>): PriceAgent {
  const config: AgentConfig = {
    name: 'price',
    enabled: true,
    polling_interval_ms: 100, // Very frequent for price data
    confidence_min: 0.7,
    data_age_max_ms: 5000, // 5 seconds max age
    retry_attempts: 3,
    retry_backoff_ms: 1000
  };

  return new PriceAgent(config, apiKeys);
}

// Export for testing
export { PriceAgent as default };