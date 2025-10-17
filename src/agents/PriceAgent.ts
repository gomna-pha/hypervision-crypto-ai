import WebSocket from 'ws';
import axios from 'axios';
import { BaseAgent } from './BaseAgent';
import { PriceAgentOutput, OrderbookUpdate } from '../types';
import config from '../utils/ConfigLoader';

interface ExchangeConnection {
  name: string;
  ws?: WebSocket;
  connected: boolean;
  lastUpdate: number;
  orderbook: {
    bids: Map<number, number>;
    asks: Map<number, number>;
  };
  lastTrade?: {
    price: number;
    volume: number;
    timestamp: number;
  };
  volume24h?: number;
  openInterest?: number;
  fundingRate?: number;
}

interface OrderbookStats {
  bestBid: number;
  bestAsk: number;
  midPrice: number;
  spread: number;
  spreadPct: number;
  depthUsd: number;
}

export class PriceAgent extends BaseAgent {
  private exchanges: Map<string, ExchangeConnection>;
  private pairs: string[];
  private reconnectInterval: number = 5000;
  private maxReconnectAttempts: number = 10;

  constructor(port: number = 3002) {
    super('PriceAgent', port);
    this.exchanges = new Map();
    this.pairs = ['BTC-USDT', 'ETH-USDT']; // Default pairs
  }

  protected async initialize(): Promise<void> {
    const priceConfig = config.get('agents.price');
    const exchangeConfigs = priceConfig?.exchanges || {};

    // Initialize exchange connections
    for (const [exchangeName, exchangeConfig] of Object.entries(exchangeConfigs)) {
      const config = exchangeConfig as any;
      this.exchanges.set(exchangeName, {
        name: exchangeName,
        connected: false,
        lastUpdate: Date.now(),
        orderbook: {
          bids: new Map(),
          asks: new Map(),
        },
      });

      // Connect to WebSocket
      if (config.ws_url) {
        await this.connectExchange(exchangeName, config.ws_url);
      }
    }

    this.logger.info('PriceAgent initialized with exchanges', {
      exchanges: Array.from(this.exchanges.keys()),
    });
  }

  private async connectExchange(exchange: string, wsUrl: string): Promise<void> {
    const connection = this.exchanges.get(exchange);
    if (!connection) return;

    try {
      let url = wsUrl;
      let subscribeMessage: any;

      // Exchange-specific connection logic
      switch (exchange) {
        case 'binance':
          // Binance stream for BTC-USDT
          url = `${wsUrl}/btcusdt@depth20@100ms/btcusdt@aggTrade`;
          break;
        
        case 'coinbase':
          // Coinbase will need subscription after connection
          subscribeMessage = {
            type: 'subscribe',
            product_ids: ['BTC-USD'],
            channels: ['level2', 'matches'],
          };
          break;

        case 'kraken':
          subscribeMessage = {
            event: 'subscribe',
            pair: ['XBT/USD'],
            subscription: {
              name: 'book',
              depth: 25,
            },
          };
          break;
      }

      const ws = new WebSocket(url);
      connection.ws = ws;

      ws.on('open', () => {
        this.logger.info(`Connected to ${exchange} WebSocket`);
        connection.connected = true;
        connection.lastUpdate = Date.now();

        // Send subscription message if needed
        if (subscribeMessage) {
          ws.send(JSON.stringify(subscribeMessage));
        }
      });

      ws.on('message', (data: WebSocket.Data) => {
        try {
          const message = JSON.parse(data.toString());
          this.handleExchangeMessage(exchange, message);
        } catch (error) {
          this.logger.error(`Failed to parse ${exchange} message`, error);
        }
      });

      ws.on('error', (error) => {
        this.logger.error(`${exchange} WebSocket error`, error);
      });

      ws.on('close', () => {
        this.logger.warn(`${exchange} WebSocket disconnected`);
        connection.connected = false;
        
        // Attempt reconnection
        setTimeout(() => {
          this.connectExchange(exchange, wsUrl);
        }, this.reconnectInterval);
      });

    } catch (error) {
      this.logger.error(`Failed to connect to ${exchange}`, error);
    }
  }

  private handleExchangeMessage(exchange: string, message: any): void {
    const connection = this.exchanges.get(exchange);
    if (!connection) return;

    connection.lastUpdate = Date.now();

    // Parse message based on exchange format
    switch (exchange) {
      case 'binance':
        this.handleBinanceMessage(connection, message);
        break;
      case 'coinbase':
        this.handleCoinbaseMessage(connection, message);
        break;
      case 'kraken':
        this.handleKrakenMessage(connection, message);
        break;
    }
  }

  private handleBinanceMessage(connection: ExchangeConnection, message: any): void {
    if (message.e === 'depthUpdate' || message.bids) {
      // Orderbook update
      if (message.bids) {
        connection.orderbook.bids.clear();
        for (const [price, quantity] of message.bids) {
          connection.orderbook.bids.set(parseFloat(price), parseFloat(quantity));
        }
      }
      if (message.asks) {
        connection.orderbook.asks.clear();
        for (const [price, quantity] of message.asks) {
          connection.orderbook.asks.set(parseFloat(price), parseFloat(quantity));
        }
      }
    } else if (message.e === 'aggTrade') {
      // Trade update
      connection.lastTrade = {
        price: parseFloat(message.p),
        volume: parseFloat(message.q),
        timestamp: message.T,
      };
    }
  }

  private handleCoinbaseMessage(connection: ExchangeConnection, message: any): void {
    if (message.type === 'l2update') {
      // Level 2 orderbook updates
      for (const change of message.changes) {
        const [side, price, size] = change;
        const priceNum = parseFloat(price);
        const sizeNum = parseFloat(size);
        
        if (side === 'buy') {
          if (sizeNum === 0) {
            connection.orderbook.bids.delete(priceNum);
          } else {
            connection.orderbook.bids.set(priceNum, sizeNum);
          }
        } else {
          if (sizeNum === 0) {
            connection.orderbook.asks.delete(priceNum);
          } else {
            connection.orderbook.asks.set(priceNum, sizeNum);
          }
        }
      }
    } else if (message.type === 'match') {
      // Trade update
      connection.lastTrade = {
        price: parseFloat(message.price),
        volume: parseFloat(message.size),
        timestamp: new Date(message.time).getTime(),
      };
    }
  }

  private handleKrakenMessage(connection: ExchangeConnection, message: any): void {
    // Kraken message handling
    if (Array.isArray(message) && message.length >= 2) {
      const data = message[1];
      if (data.b) {
        // Bids update
        connection.orderbook.bids.clear();
        for (const [price, volume] of data.b) {
          connection.orderbook.bids.set(parseFloat(price), parseFloat(volume));
        }
      }
      if (data.a) {
        // Asks update
        connection.orderbook.asks.clear();
        for (const [price, volume] of data.a) {
          connection.orderbook.asks.set(parseFloat(price), parseFloat(volume));
        }
      }
    }
  }

  protected async update(): Promise<void> {
    // Process each exchange and publish price data
    for (const [exchangeName, connection] of this.exchanges) {
      if (!connection.connected || connection.orderbook.bids.size === 0) {
        continue;
      }

      const stats = this.calculateOrderbookStats(connection);
      const volume1m = this.calculateRecentVolume(connection);
      
      const output: PriceAgentOutput = {
        agent_name: 'PriceAgent',
        timestamp: new Date().toISOString(),
        pair: 'BTC-USDT',
        exchange: exchangeName,
        best_bid: stats.bestBid,
        best_ask: stats.bestAsk,
        mid_price: stats.midPrice,
        last_trade_price: connection.lastTrade?.price,
        volume_1m: volume1m,
        vwap_1m: this.calculateVWAP(connection),
        orderbook_depth_usd: stats.depthUsd,
        open_interest: connection.openInterest,
        funding_rate: connection.fundingRate,
        key_signal: this.calculateKeySignal(stats),
        confidence: this.calculateConfidence(connection),
        features: {
          spread: stats.spread,
          spread_pct: stats.spreadPct,
          bid_count: connection.orderbook.bids.size,
          ask_count: connection.orderbook.asks.size,
        },
      };

      await this.publishOutput(output);
    }
  }

  private calculateOrderbookStats(connection: ExchangeConnection): OrderbookStats {
    const bids = Array.from(connection.orderbook.bids.entries()).sort((a, b) => b[0] - a[0]);
    const asks = Array.from(connection.orderbook.asks.entries()).sort((a, b) => a[0] - b[0]);

    const bestBid = bids[0]?.[0] || 0;
    const bestAsk = asks[0]?.[0] || 0;
    const midPrice = (bestBid + bestAsk) / 2;
    const spread = bestAsk - bestBid;
    const spreadPct = midPrice > 0 ? (spread / midPrice) * 100 : 0;

    // Calculate depth at 0.5% from mid
    const depthRange = midPrice * 0.005;
    let bidDepthUsd = 0;
    let askDepthUsd = 0;

    for (const [price, volume] of bids) {
      if (price >= midPrice - depthRange) {
        bidDepthUsd += price * volume;
      } else break;
    }

    for (const [price, volume] of asks) {
      if (price <= midPrice + depthRange) {
        askDepthUsd += price * volume;
      } else break;
    }

    return {
      bestBid,
      bestAsk,
      midPrice,
      spread,
      spreadPct,
      depthUsd: bidDepthUsd + askDepthUsd,
    };
  }

  private calculateRecentVolume(connection: ExchangeConnection): number {
    // In production, would track recent trades
    // For now, return estimated value
    return connection.lastTrade?.volume || 0;
  }

  private calculateVWAP(connection: ExchangeConnection): number {
    // Simplified VWAP calculation
    return connection.lastTrade?.price || 0;
  }

  private calculateKeySignal(stats: OrderbookStats): number {
    // Signal based on spread tightness and depth
    const spreadSignal = 1 - Math.min(stats.spreadPct / 0.1, 1); // Tighter spread = higher signal
    const depthSignal = Math.min(stats.depthUsd / 1000000, 1); // More depth = higher signal
    
    return (spreadSignal * 0.6 + depthSignal * 0.4);
  }

  private calculateConfidence(connection: ExchangeConnection): number {
    const timeSinceUpdate = Date.now() - connection.lastUpdate;
    const freshness = Math.max(0, 1 - timeSinceUpdate / 60000); // Decay over 1 minute
    const dataQuality = connection.orderbook.bids.size > 0 ? 1 : 0;
    
    return freshness * 0.7 + dataQuality * 0.3;
  }

  protected async cleanup(): Promise<void> {
    // Close all WebSocket connections
    for (const connection of this.exchanges.values()) {
      if (connection.ws) {
        connection.ws.close();
      }
    }
    this.logger.info('PriceAgent cleanup completed');
  }

  protected getPollingInterval(): number {
    // Price agent publishes based on WebSocket updates, not polling
    return 0;
  }
}

// Export for standalone execution
if (require.main === module) {
  const agent = new PriceAgent();
  agent.start().catch(error => {
    console.error('Failed to start PriceAgent:', error);
    process.exit(1);
  });

  process.on('SIGINT', async () => {
    await agent.stop();
    process.exit(0);
  });
}