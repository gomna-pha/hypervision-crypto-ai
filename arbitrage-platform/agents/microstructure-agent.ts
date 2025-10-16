import axios from 'axios';
import EventEmitter from 'events';
import WebSocket from 'ws';

export interface MicrostructureData {
  agent_name: string;
  timestamp: string;
  key_signal: number; // -1 to 1 (sell pressure to buy pressure)
  confidence: number; // 0 to 1
  features: {
    bid_ask_spread_bps: number;
    orderbook_imbalance: number;
    volume_weighted_spread: number;
    trade_intensity: number;
    market_impact_estimate: number;
    liquidity_score: number;
    price_momentum_1min: number;
    volume_surge_factor: number;
    tick_direction_bias: number;
  };
}

export interface OrderbookLevel {
  price: number;
  quantity: number;
  timestamp: number;
}

export interface TradeData {
  price: number;
  quantity: number;
  side: 'buy' | 'sell';
  timestamp: number;
  exchange: string;
}

export class MicrostructureAgent extends EventEmitter {
  private isRunning: boolean = false;
  private wsConnections: Map<string, WebSocket> = new Map();
  
  // Visible Parameters for Investors
  public readonly parameters = {
    snapshot_interval_ms: 100,
    orderbook_depth_levels: 20,
    trade_window_seconds: 60,
    volume_window_minutes: 5,
    price_precision_decimals: 2,
    quantity_precision_decimals: 8,
    spread_ema_alpha: 0.1,
    imbalance_ema_alpha: 0.2
  };

  // Visible Constraints for Investors
  public readonly constraints = {
    min_liquidity_usd: 50000,
    max_spread_bps: 500, // 5%
    min_trade_count_per_window: 10,
    max_websocket_latency_ms: 1000,
    required_exchanges: ['binance', 'coinbase'],
    monitored_pairs: ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
    max_reconnection_attempts: 5,
    data_freshness_threshold_ms: 5000
  };

  // Visible Bounds for Investors
  public readonly bounds = {
    signal_range: { min: -1.0, max: 1.0 },
    confidence_range: { min: 0.0, max: 1.0 },
    spread_bps_range: { min: 0.1, max: 1000.0 },
    imbalance_range: { min: -1.0, max: 1.0 },
    momentum_range: { min: -0.1, max: 0.1 }, // 10% max 1-minute move
    volume_surge_max: 10.0, // 10x normal volume
    liquidity_score_range: { min: 0.0, max: 1.0 }
  };

  private currentOrderbooks: Map<string, {
    bids: OrderbookLevel[];
    asks: OrderbookLevel[];
    timestamp: number;
  }> = new Map();
  
  private recentTrades: Map<string, TradeData[]> = new Map();
  private priceHistory: Map<string, Array<{ price: number; timestamp: number }>> = new Map();
  private volumeHistory: Map<string, Array<{ volume: number; timestamp: number }>> = new Map();
  
  private spreadEMA: Map<string, number> = new Map();
  private imbalanceEMA: Map<string, number> = new Map();

  constructor() {
    super();
    console.log('✅ MicrostructureAgent initialized with visible parameters');
  }

  async start(): Promise<void> {
    if (this.isRunning) return;
    
    this.isRunning = true;
    console.log('🚀 Starting MicrostructureAgent with real-time WebSocket feeds...');
    
    // Initialize WebSocket connections to exchanges
    await this.initializeWebSocketConnections();
    
    // Start periodic analysis
    setInterval(() => {
      this.analyzeMicrostructure();
    }, this.parameters.snapshot_interval_ms);
  }

  async stop(): Promise<void> {
    this.isRunning = false;
    
    // Close all WebSocket connections
    for (const [exchange, ws] of this.wsConnections) {
      ws.close();
    }
    this.wsConnections.clear();
    
    console.log('⏹️ MicrostructureAgent stopped');
  }

  private async initializeWebSocketConnections(): Promise<void> {
    for (const exchange of this.constraints.required_exchanges) {
      try {
        await this.connectToExchange(exchange);
      } catch (error) {
        console.error(`❌ Failed to connect to ${exchange}:`, error);
        // Continue with other exchanges
      }
    }
  }

  private async connectToExchange(exchange: string): Promise<void> {
    console.log(`🔌 Connecting to ${exchange} WebSocket...`);
    
    let wsUrl: string;
    let subscribeMessage: any;
    
    switch (exchange.toLowerCase()) {
      case 'binance':
        wsUrl = 'wss://stream.binance.com:9443/ws/btcusdt@depth@100ms';
        // Binance doesn't require subscription message for this endpoint
        break;
        
      case 'coinbase':
        wsUrl = 'wss://ws-feed.exchange.coinbase.com';
        subscribeMessage = {
          type: 'subscribe',
          product_ids: ['BTC-USD', 'ETH-USD'],
          channels: ['level2', 'matches']
        };
        break;
        
      default:
        throw new Error(`Unsupported exchange: ${exchange}`);
    }
    
    const ws = new WebSocket(wsUrl);
    
    ws.on('open', () => {
      console.log(`✅ Connected to ${exchange} WebSocket`);
      
      if (subscribeMessage) {
        ws.send(JSON.stringify(subscribeMessage));
      }
      
      this.wsConnections.set(exchange, ws);
    });
    
    ws.on('message', (data: WebSocket.Data) => {
      try {
        const message = JSON.parse(data.toString());
        this.processWebSocketMessage(exchange, message);
      } catch (error) {
        console.error(`❌ Error processing ${exchange} message:`, error);
      }
    });
    
    ws.on('error', (error) => {
      console.error(`❌ ${exchange} WebSocket error:`, error);
      this.emit('error', { exchange, error });
    });
    
    ws.on('close', () => {
      console.log(`⚠️ ${exchange} WebSocket disconnected`);
      this.wsConnections.delete(exchange);
      
      // Attempt reconnection
      if (this.isRunning) {
        setTimeout(() => this.connectToExchange(exchange), 5000);
      }
    });
  }

  private processWebSocketMessage(exchange: string, message: any): void {
    try {
      switch (exchange.toLowerCase()) {
        case 'binance':
          this.processBinanceMessage(message);
          break;
        case 'coinbase':
          this.processCoinbaseMessage(message);
          break;
        default:
          console.warn(`Unknown exchange: ${exchange}`);
      }
    } catch (error) {
      console.error(`Error processing ${exchange} message:`, error);
    }
  }

  private processBinanceMessage(message: any): void {
    if (message.e === 'depthUpdate') {
      // Orderbook update
      const symbol = message.s; // e.g., 'BTCUSDT'
      const pair = this.normalizePair(symbol);
      
      const orderbook = {
        bids: message.b.map((level: any) => ({
          price: parseFloat(level[0]),
          quantity: parseFloat(level[1]),
          timestamp: Date.now()
        })),
        asks: message.a.map((level: any) => ({
          price: parseFloat(level[0]),
          quantity: parseFloat(level[1]),
          timestamp: Date.now()
        })),
        timestamp: Date.now()
      };
      
      this.currentOrderbooks.set(`binance-${pair}`, orderbook);
      
    } else if (message.e === 'aggTrade') {
      // Trade update
      const symbol = message.s;
      const pair = this.normalizePair(symbol);
      
      const trade: TradeData = {
        price: parseFloat(message.p),
        quantity: parseFloat(message.q),
        side: message.m ? 'sell' : 'buy', // m = true means buyer is market maker (so it's a sell)
        timestamp: message.T,
        exchange: 'binance'
      };
      
      this.addTrade(`binance-${pair}`, trade);
    }
  }

  private processCoinbaseMessage(message: any): void {
    if (message.type === 'l2update') {
      // Level 2 orderbook update
      const pair = this.normalizePair(message.product_id);
      
      // Note: This is simplified - full implementation would maintain orderbook state
      // For demo, we'll simulate orderbook data
      const orderbook = this.getSimulatedOrderbook(pair);
      this.currentOrderbooks.set(`coinbase-${pair}`, orderbook);
      
    } else if (message.type === 'match') {
      // Trade execution
      const pair = this.normalizePair(message.product_id);
      
      const trade: TradeData = {
        price: parseFloat(message.price),
        quantity: parseFloat(message.size),
        side: message.side as 'buy' | 'sell',
        timestamp: new Date(message.time).getTime(),
        exchange: 'coinbase'
      };
      
      this.addTrade(`coinbase-${pair}`, trade);
    }
  }

  private getSimulatedOrderbook(pair: string): any {
    // Generate realistic orderbook for demo
    const basePrice = this.getLastPrice(pair) || 45000; // Default BTC price
    const spread = basePrice * 0.0001; // 1 bps spread
    
    const bids: OrderbookLevel[] = [];
    const asks: OrderbookLevel[] = [];
    
    // Generate bid levels
    for (let i = 0; i < this.parameters.orderbook_depth_levels; i++) {
      bids.push({
        price: basePrice - spread/2 - (i * spread * 0.1),
        quantity: 1 + Math.random() * 5,
        timestamp: Date.now()
      });
    }
    
    // Generate ask levels
    for (let i = 0; i < this.parameters.orderbook_depth_levels; i++) {
      asks.push({
        price: basePrice + spread/2 + (i * spread * 0.1),
        quantity: 1 + Math.random() * 5,
        timestamp: Date.now()
      });
    }
    
    return { bids, asks, timestamp: Date.now() };
  }

  private normalizePair(symbol: string): string {
    // Convert exchange-specific symbols to standard format
    return symbol.replace(/[^A-Z]/g, '').replace('USDT', '/USDT').replace('USD', '/USD');
  }

  private addTrade(key: string, trade: TradeData): void {
    if (!this.recentTrades.has(key)) {
      this.recentTrades.set(key, []);
    }
    
    const trades = this.recentTrades.get(key)!;
    trades.push(trade);
    
    // Keep only recent trades (within window)
    const cutoff = Date.now() - (this.parameters.trade_window_seconds * 1000);
    this.recentTrades.set(key, trades.filter(t => t.timestamp > cutoff));
    
    // Update price history
    this.updatePriceHistory(key.split('-')[1], trade.price, trade.timestamp);
  }

  private updatePriceHistory(pair: string, price: number, timestamp: number): void {
    if (!this.priceHistory.has(pair)) {
      this.priceHistory.set(pair, []);
    }
    
    const history = this.priceHistory.get(pair)!;
    history.push({ price, timestamp });
    
    // Keep only recent history (5 minutes)
    const cutoff = timestamp - (5 * 60 * 1000);
    this.priceHistory.set(pair, history.filter(h => h.timestamp > cutoff));
  }

  private getLastPrice(pair: string): number | null {
    const history = this.priceHistory.get(pair);
    if (history && history.length > 0) {
      return history[history.length - 1].price;
    }
    return null;
  }

  private analyzeMicrostructure(): void {
    if (!this.isRunning) return;
    
    try {
      // Analyze each monitored pair
      for (const pair of this.constraints.monitored_pairs) {
        const analysis = this.analyzeOrderbookForPair(pair);
        if (analysis) {
          this.emit('data', analysis);
        }
      }
    } catch (error) {
      console.error('❌ Microstructure analysis error:', error);
      this.emit('error', error);
    }
  }

  private analyzeOrderbookForPair(pair: string): MicrostructureData | null {
    const exchangeData = this.getExchangeDataForPair(pair);
    if (exchangeData.length === 0) {
      return null;
    }
    
    // Calculate microstructure metrics
    const spreadMetrics = this.calculateSpreadMetrics(exchangeData);
    const imbalanceMetrics = this.calculateImbalanceMetrics(exchangeData);
    const liquidityMetrics = this.calculateLiquidityMetrics(exchangeData);
    const tradeMetrics = this.calculateTradeMetrics(pair);
    const momentumMetrics = this.calculateMomentumMetrics(pair);
    
    // Generate composite microstructure signal
    const signal = this.calculateCompositeSignal(spreadMetrics, imbalanceMetrics, tradeMetrics, momentumMetrics);
    
    // Calculate confidence based on data quality
    const confidence = this.calculateMicrostructureConfidence(exchangeData, tradeMetrics);
    
    const microstructureData: MicrostructureData = {
      agent_name: 'MicrostructureAgent',
      timestamp: new Date().toISOString(),
      key_signal: this.clampToRange(signal, this.bounds.signal_range),
      confidence: this.clampToRange(confidence, this.bounds.confidence_range),
      features: {
        bid_ask_spread_bps: spreadMetrics.avgSpreadBps,
        orderbook_imbalance: imbalanceMetrics.weightedImbalance,
        volume_weighted_spread: spreadMetrics.volumeWeightedSpread,
        trade_intensity: tradeMetrics.intensity,
        market_impact_estimate: tradeMetrics.marketImpact,
        liquidity_score: liquidityMetrics.score,
        price_momentum_1min: momentumMetrics.momentum1min,
        volume_surge_factor: tradeMetrics.volumeSurgeFactor,
        tick_direction_bias: tradeMetrics.tickBias
      }
    };
    
    console.log(`📊 ${pair} Microstructure Signal: ${microstructureData.key_signal.toFixed(3)} (confidence: ${microstructureData.confidence.toFixed(2)})`);
    
    return microstructureData;
  }

  private getExchangeDataForPair(pair: string): Array<{exchange: string; orderbook: any; key: string}> {
    const data: Array<{exchange: string; orderbook: any; key: string}> = [];
    
    for (const exchange of this.constraints.required_exchanges) {
      const key = `${exchange}-${pair}`;
      const orderbook = this.currentOrderbooks.get(key);
      
      if (orderbook && this.isOrderbookFresh(orderbook)) {
        data.push({ exchange, orderbook, key });
      }
    }
    
    return data;
  }

  private isOrderbookFresh(orderbook: any): boolean {
    const age = Date.now() - orderbook.timestamp;
    return age < this.constraints.data_freshness_threshold_ms;
  }

  private calculateSpreadMetrics(exchangeData: any[]): any {
    let totalSpread = 0;
    let totalVolume = 0;
    let volumeWeightedSpread = 0;
    
    for (const { orderbook } of exchangeData) {
      if (orderbook.bids.length === 0 || orderbook.asks.length === 0) continue;
      
      const bestBid = Math.max(...orderbook.bids.map((b: any) => b.price));
      const bestAsk = Math.min(...orderbook.asks.map((a: any) => a.price));
      const midPrice = (bestBid + bestAsk) / 2;
      const spread = bestAsk - bestBid;
      const spreadBps = (spread / midPrice) * 10000;
      
      const bidVolume = orderbook.bids.reduce((sum: number, b: any) => sum + b.quantity, 0);
      const askVolume = orderbook.asks.reduce((sum: number, a: any) => sum + a.quantity, 0);
      const totalOrderbookVolume = bidVolume + askVolume;
      
      totalSpread += spreadBps;
      volumeWeightedSpread += spreadBps * totalOrderbookVolume;
      totalVolume += totalOrderbookVolume;
    }
    
    const avgSpreadBps = exchangeData.length > 0 ? totalSpread / exchangeData.length : 0;
    const volWeightedSpread = totalVolume > 0 ? volumeWeightedSpread / totalVolume : avgSpreadBps;
    
    return {
      avgSpreadBps: this.clampToRange(avgSpreadBps, this.bounds.spread_bps_range),
      volumeWeightedSpread: volWeightedSpread
    };
  }

  private calculateImbalanceMetrics(exchangeData: any[]): any {
    let totalImbalance = 0;
    let weightedImbalance = 0;
    let totalWeight = 0;
    
    for (const { orderbook } of exchangeData) {
      if (orderbook.bids.length === 0 || orderbook.asks.length === 0) continue;
      
      const bidVolume = orderbook.bids.reduce((sum: number, b: any) => sum + b.quantity, 0);
      const askVolume = orderbook.asks.reduce((sum: number, a: any) => sum + a.quantity, 0);
      const totalVolume = bidVolume + askVolume;
      
      if (totalVolume > 0) {
        const imbalance = (bidVolume - askVolume) / totalVolume; // -1 to 1
        totalImbalance += imbalance;
        weightedImbalance += imbalance * totalVolume;
        totalWeight += totalVolume;
      }
    }
    
    const avgImbalance = exchangeData.length > 0 ? totalImbalance / exchangeData.length : 0;
    const volWeightedImbalance = totalWeight > 0 ? weightedImbalance / totalWeight : avgImbalance;
    
    return {
      imbalance: avgImbalance,
      weightedImbalance: this.clampToRange(volWeightedImbalance, this.bounds.imbalance_range)
    };
  }

  private calculateLiquidityMetrics(exchangeData: any[]): any {
    let totalLiquidity = 0;
    let minLiquidity = Infinity;
    
    for (const { orderbook } of exchangeData) {
      const bidLiquidity = orderbook.bids.reduce((sum: number, b: any) => sum + (b.price * b.quantity), 0);
      const askLiquidity = orderbook.asks.reduce((sum: number, a: any) => sum + (a.price * a.quantity), 0);
      const exchangeLiquidity = bidLiquidity + askLiquidity;
      
      totalLiquidity += exchangeLiquidity;
      minLiquidity = Math.min(minLiquidity, exchangeLiquidity);
    }
    
    const avgLiquidity = exchangeData.length > 0 ? totalLiquidity / exchangeData.length : 0;
    const liquidityScore = Math.min(1, avgLiquidity / this.constraints.min_liquidity_usd);
    
    return {
      totalLiquidity,
      avgLiquidity,
      score: liquidityScore
    };
  }

  private calculateTradeMetrics(pair: string): any {
    let totalVolume = 0;
    let buyVolume = 0;
    let sellVolume = 0;
    let tradeCount = 0;
    
    for (const exchange of this.constraints.required_exchanges) {
      const key = `${exchange}-${pair}`;
      const trades = this.recentTrades.get(key) || [];
      
      for (const trade of trades) {
        totalVolume += trade.quantity;
        tradeCount++;
        
        if (trade.side === 'buy') {
          buyVolume += trade.quantity;
        } else {
          sellVolume += trade.quantity;
        }
      }
    }
    
    const intensity = tradeCount / this.parameters.trade_window_seconds; // trades per second
    const volumeSurgeFactor = this.calculateVolumeSurge(pair, totalVolume);
    const tickBias = totalVolume > 0 ? (buyVolume - sellVolume) / totalVolume : 0;
    const marketImpact = this.estimateMarketImpact(pair, totalVolume);
    
    return {
      intensity,
      volumeSurgeFactor: Math.min(this.bounds.volume_surge_max, volumeSurgeFactor),
      tickBias: this.clampToRange(tickBias, this.bounds.imbalance_range),
      marketImpact,
      tradeCount
    };
  }

  private calculateVolumeSurge(pair: string, currentVolume: number): number {
    // Compare current volume to historical average
    const key = `volume-${pair}`;
    if (!this.volumeHistory.has(key)) {
      this.volumeHistory.set(key, []);
    }
    
    const history = this.volumeHistory.get(key)!;
    history.push({ volume: currentVolume, timestamp: Date.now() });
    
    // Keep only recent history
    const cutoff = Date.now() - (this.parameters.volume_window_minutes * 60 * 1000);
    const recentHistory = history.filter(h => h.timestamp > cutoff);
    this.volumeHistory.set(key, recentHistory);
    
    if (recentHistory.length < 10) return 1.0; // Not enough history
    
    const avgVolume = recentHistory.slice(0, -1).reduce((sum, h) => sum + h.volume, 0) / (recentHistory.length - 1);
    return avgVolume > 0 ? currentVolume / avgVolume : 1.0;
  }

  private estimateMarketImpact(pair: string, volume: number): number {
    // Simple market impact model based on volume and liquidity
    const exchangeData = this.getExchangeDataForPair(pair);
    if (exchangeData.length === 0) return 0;
    
    let totalLiquidity = 0;
    for (const { orderbook } of exchangeData) {
      totalLiquidity += orderbook.bids.reduce((sum: number, b: any) => sum + b.quantity, 0);
      totalLiquidity += orderbook.asks.reduce((sum: number, a: any) => sum + a.quantity, 0);
    }
    
    // Impact = sqrt(volume / liquidity) * volatility_factor
    const impactFactor = Math.sqrt(volume / Math.max(1, totalLiquidity)) * 0.01;
    return Math.min(0.1, impactFactor); // Cap at 10%
  }

  private calculateMomentumMetrics(pair: string): any {
    const history = this.priceHistory.get(pair);
    if (!history || history.length < 2) {
      return { momentum1min: 0, momentum5min: 0 };
    }
    
    const now = Date.now();
    const oneMinuteAgo = now - 60 * 1000;
    const fiveMinutesAgo = now - 5 * 60 * 1000;
    
    const currentPrice = history[history.length - 1].price;
    
    // Find prices at different time intervals
    const price1MinAgo = this.findPriceAtTime(history, oneMinuteAgo) || currentPrice;
    const price5MinAgo = this.findPriceAtTime(history, fiveMinutesAgo) || currentPrice;
    
    const momentum1min = (currentPrice - price1MinAgo) / price1MinAgo;
    const momentum5min = (currentPrice - price5MinAgo) / price5MinAgo;
    
    return {
      momentum1min: this.clampToRange(momentum1min, this.bounds.momentum_range),
      momentum5min: this.clampToRange(momentum5min, this.bounds.momentum_range)
    };
  }

  private findPriceAtTime(history: Array<{price: number; timestamp: number}>, targetTime: number): number | null {
    // Find the closest price to the target time
    let closest = null;
    let minDiff = Infinity;
    
    for (const entry of history) {
      const diff = Math.abs(entry.timestamp - targetTime);
      if (diff < minDiff) {
        minDiff = diff;
        closest = entry.price;
      }
    }
    
    return closest;
  }

  private calculateCompositeSignal(spreadMetrics: any, imbalanceMetrics: any, tradeMetrics: any, momentumMetrics: any): number {
    // Weight different microstructure signals
    const weights = {
      imbalance: 0.3,        // Order book imbalance
      momentum: 0.25,        // Price momentum
      tradeFlow: 0.2,        // Trade flow bias
      liquidity: 0.15,       // Liquidity conditions
      intensity: 0.1         // Trading intensity
    };
    
    // Normalize and combine signals
    const imbalanceSignal = imbalanceMetrics.weightedImbalance; // Already -1 to 1
    const momentumSignal = momentumMetrics.momentum1min / this.bounds.momentum_range.max; // Normalize
    const tradeFlowSignal = tradeMetrics.tickBias; // Already -1 to 1
    const liquiditySignal = (spreadMetrics.avgSpreadBps < 20) ? 0.5 : -0.5; // Good spread = positive
    const intensitySignal = Math.min(1, tradeMetrics.intensity / 10) - 0.5; // High intensity = positive
    
    const compositeSignal = 
      imbalanceSignal * weights.imbalance +
      momentumSignal * weights.momentum +
      tradeFlowSignal * weights.tradeFlow +
      liquiditySignal * weights.liquidity +
      intensitySignal * weights.intensity;
    
    return compositeSignal;
  }

  private calculateMicrostructureConfidence(exchangeData: any[], tradeMetrics: any): number {
    let confidence = 1.0;
    
    // Exchange connectivity
    const expectedExchanges = this.constraints.required_exchanges.length;
    const connectedExchanges = exchangeData.length;
    const connectivityScore = connectedExchanges / expectedExchanges;
    
    // Data freshness
    const avgAge = exchangeData.reduce((sum, data) => sum + (Date.now() - data.orderbook.timestamp), 0) / Math.max(1, exchangeData.length);
    const freshnessScore = Math.max(0, 1 - (avgAge / this.constraints.data_freshness_threshold_ms));
    
    // Trade activity
    const activityScore = Math.min(1, tradeMetrics.tradeCount / this.constraints.min_trade_count_per_window);
    
    // Spread quality (tighter spreads = higher confidence)
    const avgSpread = exchangeData.reduce((sum, data) => {
      const bestBid = Math.max(...data.orderbook.bids.map((b: any) => b.price));
      const bestAsk = Math.min(...data.orderbook.asks.map((a: any) => a.price));
      const midPrice = (bestBid + bestAsk) / 2;
      return sum + ((bestAsk - bestBid) / midPrice * 10000);
    }, 0) / Math.max(1, exchangeData.length);
    
    const spreadScore = Math.max(0, 1 - (avgSpread / this.constraints.max_spread_bps));
    
    confidence = connectivityScore * 0.3 + 
                 freshnessScore * 0.3 + 
                 activityScore * 0.2 + 
                 spreadScore * 0.2;
    
    return confidence;
  }

  private clampToRange(value: number, range: { min: number; max: number }): number {
    return Math.max(range.min, Math.min(range.max, value));
  }

  // Public method to get current parameters (for investor transparency)
  getVisibleParameters(): any {
    return {
      parameters: this.parameters,
      constraints: this.constraints,
      bounds: this.bounds,
      active_connections: Array.from(this.wsConnections.keys()),
      orderbook_counts: Object.fromEntries(
        Array.from(this.currentOrderbooks.entries()).map(([key, orderbook]) => [
          key, 
          { bids: orderbook.bids.length, asks: orderbook.asks.length, age_ms: Date.now() - orderbook.timestamp }
        ])
      ),
      recent_trades_counts: Object.fromEntries(
        Array.from(this.recentTrades.entries()).map(([key, trades]) => [key, trades.length])
      )
    };
  }

  // Public method to get current market microstructure snapshot
  getCurrentSnapshot(): any {
    const snapshot: any = {};
    
    for (const pair of this.constraints.monitored_pairs) {
      const exchangeData = this.getExchangeDataForPair(pair);
      
      if (exchangeData.length > 0) {
        const bestPrices = exchangeData.map(data => {
          const bestBid = Math.max(...data.orderbook.bids.map((b: any) => b.price));
          const bestAsk = Math.min(...data.orderbook.asks.map((a: any) => a.price));
          return { exchange: data.exchange, bid: bestBid, ask: bestAsk, spread: bestAsk - bestBid };
        });
        
        snapshot[pair] = bestPrices;
      }
    }
    
    return snapshot;
  }
}