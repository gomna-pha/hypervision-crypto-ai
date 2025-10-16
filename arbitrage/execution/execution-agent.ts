/**
 * Execution Agent - Order Execution and Exchange Integration
 * Handles order placement, execution monitoring, and trade reconciliation
 * Supports multiple exchanges with safety checks and risk management
 */

import { ExecutionPlan, DecisionResult } from '../decision/decision-engine.js';
import axios from 'axios';
import crypto from 'crypto';
import { EventEmitter } from 'events';

export interface ExchangeConfig {
  name: string;
  enabled: boolean;
  sandbox: boolean;
  api_key: string;
  api_secret: string;
  passphrase?: string; // For Coinbase
  base_url: string;
  ws_url?: string;
  rate_limit_per_sec: number;
}

export interface OrderRequest {
  exchange: string;
  pair: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'post_only';
  amount: number;
  price?: number;
  time_in_force?: 'GTC' | 'IOC' | 'FOK';
  client_order_id?: string;
}

export interface OrderResponse {
  order_id: string;
  client_order_id?: string;
  exchange: string;
  pair: string;
  side: 'buy' | 'sell';
  type: string;
  amount: number;
  price?: number;
  status: 'pending' | 'open' | 'filled' | 'cancelled' | 'rejected';
  filled_amount: number;
  filled_price?: number;
  fees_paid: number;
  timestamp: number;
  raw_response: any;
}

export interface ArbitrageExecution {
  trade_id: string;
  execution_plan: ExecutionPlan;
  status: 'pending' | 'executing' | 'completed' | 'failed' | 'cancelled';
  buy_order?: OrderResponse;
  sell_order?: OrderResponse;
  start_time: number;
  end_time?: number;
  realized_pnl?: number;
  execution_summary?: ExecutionSummary;
}

export interface ExecutionSummary {
  total_fees_paid: number;
  effective_spread_captured: number;
  slippage_experienced: number;
  execution_time_ms: number;
  success_rate: number;
  notes: string[];
}

export interface RiskCheck {
  passed: boolean;
  warnings: string[];
  blocking_issues: string[];
}

export class ExecutionAgent extends EventEmitter {
  private exchangeConfigs: Map<string, ExchangeConfig> = new Map();
  private activeExecutions: Map<string, ArbitrageExecution> = new Map();
  private orderHistory: Map<string, OrderResponse> = new Map();
  private rateLimiters: Map<string, { lastCall: number; callCount: number }> = new Map();
  private executionStats = {
    total_trades: 0,
    successful_trades: 0,
    total_volume_usd: 0,
    total_fees_paid: 0,
    avg_execution_time_ms: 0
  };

  constructor() {
    super();
    this.initializeExchangeConfigs();
  }

  /**
   * Execute arbitrage plan
   */
  async executeArbitragePlan(executionPlan: ExecutionPlan): Promise<ArbitrageExecution> {
    const tradeId = executionPlan.trade_id;
    console.log(`Starting arbitrage execution for trade ${tradeId}`);

    const execution: ArbitrageExecution = {
      trade_id: tradeId,
      execution_plan: executionPlan,
      status: 'pending',
      start_time: Date.now()
    };

    try {
      // Step 1: Pre-execution risk checks
      const riskCheck = await this.performPreExecutionRiskCheck(executionPlan);
      if (!riskCheck.passed) {
        throw new Error(`Risk check failed: ${riskCheck.blocking_issues.join(', ')}`);
      }

      // Step 2: Update execution status
      execution.status = 'executing';
      this.activeExecutions.set(tradeId, execution);
      this.emit('execution_started', execution);

      // Step 3: Simulate fill estimation for both legs
      const buySimulation = await this.simulateFill(
        executionPlan.arbitrage_plan.buy_exchange,
        executionPlan.arbitrage_plan.pair,
        'buy',
        executionPlan.arbitrage_plan.notional_usd
      );

      const sellSimulation = await this.simulateFill(
        executionPlan.arbitrage_plan.sell_exchange,
        executionPlan.arbitrage_plan.pair,
        'sell',
        executionPlan.arbitrage_plan.notional_usd
      );

      // Step 4: Execute orders simultaneously (or with minimal delay)
      const [buyOrder, sellOrder] = await Promise.allSettled([
        this.placeOrder({
          exchange: executionPlan.arbitrage_plan.buy_exchange,
          pair: executionPlan.arbitrage_plan.pair,
          side: 'buy',
          type: executionPlan.execution_params.order_type,
          amount: this.calculateOrderAmount(
            executionPlan.arbitrage_plan.notional_usd,
            buySimulation.estimated_price
          ),
          price: executionPlan.execution_params.order_type === 'limit' ? buySimulation.estimated_price : undefined,
          client_order_id: `${tradeId}_BUY`
        }),
        this.placeOrder({
          exchange: executionPlan.arbitrage_plan.sell_exchange,
          pair: executionPlan.arbitrage_plan.pair,
          side: 'sell',
          type: executionPlan.execution_params.order_type,
          amount: this.calculateOrderAmount(
            executionPlan.arbitrage_plan.notional_usd,
            sellSimulation.estimated_price
          ),
          price: executionPlan.execution_params.order_type === 'limit' ? sellSimulation.estimated_price : undefined,
          client_order_id: `${tradeId}_SELL`
        })
      ]);

      // Step 5: Process order results
      if (buyOrder.status === 'fulfilled') {
        execution.buy_order = buyOrder.value;
      } else {
        console.error('Buy order failed:', buyOrder.reason);
      }

      if (sellOrder.status === 'fulfilled') {
        execution.sell_order = sellOrder.value;
      } else {
        console.error('Sell order failed:', sellOrder.reason);
      }

      // Step 6: Monitor order fills and manage position
      await this.monitorExecution(execution);

      // Step 7: Calculate final results
      this.calculateExecutionResults(execution);

      execution.status = 'completed';
      execution.end_time = Date.now();

      console.log(`Arbitrage execution completed for trade ${tradeId}: ${execution.status}`);
      this.emit('execution_completed', execution);

      return execution;

    } catch (error) {
      console.error(`Arbitrage execution failed for trade ${tradeId}:`, error);
      
      execution.status = 'failed';
      execution.end_time = Date.now();
      
      this.emit('execution_failed', { execution, error: error.message });
      
      // Attempt cleanup/cancellation
      await this.cleanupFailedExecution(execution);
      
      throw error;
    } finally {
      this.updateExecutionStats(execution);
    }
  }

  /**
   * Perform pre-execution risk checks
   */
  private async performPreExecutionRiskCheck(executionPlan: ExecutionPlan): Promise<RiskCheck> {
    const warnings: string[] = [];
    const blockingIssues: string[] = [];

    try {
      // Check exchange connectivity
      const buyExchange = executionPlan.arbitrage_plan.buy_exchange;
      const sellExchange = executionPlan.arbitrage_plan.sell_exchange;

      const [buyHealth, sellHealth] = await Promise.allSettled([
        this.checkExchangeHealth(buyExchange),
        this.checkExchangeHealth(sellExchange)
      ]);

      if (buyHealth.status === 'rejected') {
        blockingIssues.push(`Buy exchange ${buyExchange} not healthy: ${buyHealth.reason}`);
      }

      if (sellHealth.status === 'rejected') {
        blockingIssues.push(`Sell exchange ${sellExchange} not healthy: ${sellHealth.reason}`);
      }

      // Check account balances (simulated for demo)
      const balanceCheck = await this.checkAccountBalances(executionPlan);
      if (!balanceCheck) {
        warnings.push('Insufficient balance verification - proceeding with simulation');
      }

      // Check market conditions
      const marketCheck = await this.checkMarketConditions(executionPlan);
      if (!marketCheck.suitable) {
        warnings.push(`Market conditions warning: ${marketCheck.reason}`);
      }

      // Check rate limits
      const rateLimitOk = this.checkRateLimits([buyExchange, sellExchange]);
      if (!rateLimitOk) {
        blockingIssues.push('Rate limit exceeded on one or more exchanges');
      }

      return {
        passed: blockingIssues.length === 0,
        warnings,
        blocking_issues: blockingIssues
      };

    } catch (error) {
      blockingIssues.push(`Risk check system error: ${error.message}`);
      return {
        passed: false,
        warnings,
        blocking_issues: blockingIssues
      };
    }
  }

  /**
   * Simulate order fill to estimate execution
   */
  private async simulateFill(
    exchange: string,
    pair: string,
    side: 'buy' | 'sell',
    notionalUsd: number
  ): Promise<{ estimated_price: number; estimated_slippage: number; confidence: number }> {
    try {
      // Get current orderbook
      const orderbook = await this.getOrderbook(exchange, pair);
      
      if (!orderbook || !orderbook.bids || !orderbook.asks) {
        throw new Error(`Invalid orderbook data for ${exchange} ${pair}`);
      }

      const levels = side === 'buy' ? orderbook.asks : orderbook.bids;
      const basePrice = side === 'buy' ? 
        parseFloat(orderbook.asks[0][0]) : 
        parseFloat(orderbook.bids[0][0]);

      // Simulate market impact
      let remainingNotional = notionalUsd;
      let totalCost = 0;
      let totalQuantity = 0;

      for (const [priceStr, quantityStr] of levels) {
        const price = parseFloat(priceStr);
        const quantity = parseFloat(quantityStr);
        const levelValue = price * quantity;

        if (remainingNotional <= levelValue) {
          const partialQuantity = remainingNotional / price;
          totalCost += remainingNotional;
          totalQuantity += partialQuantity;
          break;
        } else {
          totalCost += levelValue;
          totalQuantity += quantity;
          remainingNotional -= levelValue;
        }

        if (remainingNotional <= 0) break;
      }

      if (totalQuantity === 0) {
        throw new Error(`Insufficient liquidity for ${notionalUsd} USD on ${exchange}`);
      }

      const avgExecutionPrice = totalCost / totalQuantity;
      const slippage = Math.abs(avgExecutionPrice - basePrice) / basePrice;
      const confidence = Math.min(1, totalQuantity / (notionalUsd / basePrice));

      return {
        estimated_price: avgExecutionPrice,
        estimated_slippage: slippage,
        confidence: confidence
      };

    } catch (error) {
      console.warn(`Fill simulation failed for ${exchange} ${pair}:`, error.message);
      
      // Fallback to basic estimation
      return {
        estimated_price: 50000, // Default BTC price for demo
        estimated_slippage: 0.002, // 0.2% default slippage
        confidence: 0.5
      };
    }
  }

  /**
   * Place order on exchange
   */
  private async placeOrder(orderRequest: OrderRequest): Promise<OrderResponse> {
    const exchange = orderRequest.exchange;
    const config = this.exchangeConfigs.get(exchange);

    if (!config) {
      throw new Error(`Exchange configuration not found: ${exchange}`);
    }

    if (!config.enabled) {
      throw new Error(`Exchange disabled: ${exchange}`);
    }

    // Check rate limits
    if (!this.checkRateLimit(exchange)) {
      throw new Error(`Rate limit exceeded for ${exchange}`);
    }

    try {
      switch (exchange) {
        case 'binance':
          return await this.placeBinanceOrder(orderRequest, config);
        case 'coinbase':
          return await this.placeCoinbaseOrder(orderRequest, config);
        case 'kraken':
          return await this.placeKrakenOrder(orderRequest, config);
        default:
          throw new Error(`Unsupported exchange: ${exchange}`);
      }
    } catch (error) {
      console.error(`Order placement failed on ${exchange}:`, error.message);
      
      // Return a simulated order for demo purposes
      return this.createSimulatedOrder(orderRequest, 'rejected');
    }
  }

  /**
   * Place Binance order (sandbox/testnet)
   */
  private async placeBinanceOrder(request: OrderRequest, config: ExchangeConfig): Promise<OrderResponse> {
    const symbol = this.formatBinancePair(request.pair);
    const timestamp = Date.now();
    
    // For demo: simulate order placement
    if (config.sandbox) {
      console.log(`[DEMO] Placing Binance order: ${request.side} ${request.amount} ${symbol}`);
      return this.createSimulatedOrder(request, 'filled');
    }

    const params: any = {
      symbol,
      side: request.side.toUpperCase(),
      type: request.type.toUpperCase(),
      quantity: request.amount.toFixed(6),
      timestamp
    };

    if (request.type === 'limit' && request.price) {
      params.price = request.price.toFixed(2);
      params.timeInForce = 'GTC';
    }

    if (request.client_order_id) {
      params.newClientOrderId = request.client_order_id;
    }

    // Create signature
    const queryString = Object.keys(params)
      .sort()
      .map(key => `${key}=${params[key]}`)
      .join('&');
    
    const signature = crypto
      .createHmac('sha256', config.api_secret)
      .update(queryString)
      .digest('hex');

    const url = `${config.base_url}/api/v3/order?${queryString}&signature=${signature}`;

    const response = await axios.post(url, null, {
      headers: {
        'X-MBX-APIKEY': config.api_key
      },
      timeout: 10000
    });

    return this.parseBinanceOrderResponse(response.data, request);
  }

  /**
   * Place Coinbase order (sandbox)
   */
  private async placeCoinbaseOrder(request: OrderRequest, config: ExchangeConfig): Promise<OrderResponse> {
    const productId = this.formatCoinbasePair(request.pair);
    
    // For demo: simulate order placement
    if (config.sandbox) {
      console.log(`[DEMO] Placing Coinbase order: ${request.side} ${request.amount} ${productId}`);
      return this.createSimulatedOrder(request, 'filled');
    }

    const body = {
      product_id: productId,
      side: request.side,
      size: request.amount.toFixed(6),
      type: request.type
    };

    if (request.type === 'limit' && request.price) {
      (body as any).price = request.price.toFixed(2);
    }

    if (request.client_order_id) {
      (body as any).client_oid = request.client_order_id;
    }

    // Create Coinbase signature
    const timestamp = Date.now() / 1000;
    const method = 'POST';
    const requestPath = '/orders';
    const bodyStr = JSON.stringify(body);
    
    const message = timestamp + method + requestPath + bodyStr;
    const signature = crypto
      .createHmac('sha256', Buffer.from(config.api_secret, 'base64'))
      .update(message)
      .digest('base64');

    const response = await axios.post(`${config.base_url}/orders`, body, {
      headers: {
        'CB-ACCESS-KEY': config.api_key,
        'CB-ACCESS-SIGN': signature,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'CB-ACCESS-PASSPHRASE': config.passphrase || '',
        'Content-Type': 'application/json'
      },
      timeout: 10000
    });

    return this.parseCoinbaseOrderResponse(response.data, request);
  }

  /**
   * Place Kraken order
   */
  private async placeKrakenOrder(request: OrderRequest, config: ExchangeConfig): Promise<OrderResponse> {
    // For demo: simulate order placement
    console.log(`[DEMO] Placing Kraken order: ${request.side} ${request.amount} ${request.pair}`);
    return this.createSimulatedOrder(request, 'filled');
  }

  /**
   * Create simulated order response for demo
   */
  private createSimulatedOrder(request: OrderRequest, status: 'filled' | 'rejected'): OrderResponse {
    const orderId = `SIM_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    const basePrice = request.pair.includes('BTC') ? 50000 : 3000; // Simplified pricing
    const filledPrice = request.price || basePrice * (1 + (Math.random() - 0.5) * 0.001); // ±0.05% slippage

    return {
      order_id: orderId,
      client_order_id: request.client_order_id,
      exchange: request.exchange,
      pair: request.pair,
      side: request.side,
      type: request.type,
      amount: request.amount,
      price: request.price,
      status: status,
      filled_amount: status === 'filled' ? request.amount : 0,
      filled_price: status === 'filled' ? filledPrice : undefined,
      fees_paid: status === 'filled' ? request.amount * filledPrice * 0.001 : 0, // 0.1% fee
      timestamp: Date.now(),
      raw_response: { simulated: true, status: status }
    };
  }

  /**
   * Monitor execution progress
   */
  private async monitorExecution(execution: ArbitrageExecution): Promise<void> {
    const maxWaitTime = execution.execution_plan.execution_params.timeout_sec * 1000;
    const startTime = Date.now();

    while (Date.now() - startTime < maxWaitTime) {
      let buyFilled = false;
      let sellFilled = false;

      // Check buy order status
      if (execution.buy_order && execution.buy_order.status !== 'filled') {
        const updatedBuyOrder = await this.getOrderStatus(
          execution.buy_order.exchange,
          execution.buy_order.order_id
        );
        if (updatedBuyOrder) {
          execution.buy_order = updatedBuyOrder;
        }
      }
      buyFilled = execution.buy_order?.status === 'filled';

      // Check sell order status
      if (execution.sell_order && execution.sell_order.status !== 'filled') {
        const updatedSellOrder = await this.getOrderStatus(
          execution.sell_order.exchange,
          execution.sell_order.order_id
        );
        if (updatedSellOrder) {
          execution.sell_order = updatedSellOrder;
        }
      }
      sellFilled = execution.sell_order?.status === 'filled';

      // Both orders filled - execution complete
      if (buyFilled && sellFilled) {
        console.log(`Both orders filled for trade ${execution.trade_id}`);
        break;
      }

      // Check for partial fills and risk conditions
      await this.handlePartialFills(execution);

      // Wait before next check
      await new Promise(resolve => setTimeout(resolve, 1000));
    }

    // Handle any unfilled orders
    await this.handleUnfilledOrders(execution);
  }

  /**
   * Handle partial fills and adjust strategy
   */
  private async handlePartialFills(execution: ArbitrageExecution): Promise<void> {
    const buyOrder = execution.buy_order;
    const sellOrder = execution.sell_order;

    if (!buyOrder || !sellOrder) return;

    // Check for significant imbalance in fills
    const buyFillRatio = buyOrder.filled_amount / buyOrder.amount;
    const sellFillRatio = sellOrder.filled_amount / sellOrder.amount;
    const imbalance = Math.abs(buyFillRatio - sellFillRatio);

    if (imbalance > 0.2) { // 20% imbalance
      console.warn(`Significant fill imbalance detected for trade ${execution.trade_id}: ${imbalance.toFixed(3)}`);
      
      // Could implement hedging logic here
      this.emit('execution_warning', {
        trade_id: execution.trade_id,
        message: 'Significant fill imbalance',
        imbalance: imbalance
      });
    }
  }

  /**
   * Handle unfilled orders at execution timeout
   */
  private async handleUnfilledOrders(execution: ArbitrageExecution): Promise<void> {
    const orders = [execution.buy_order, execution.sell_order].filter(order => 
      order && order.status !== 'filled' && order.status !== 'cancelled'
    );

    for (const order of orders) {
      if (order) {
        try {
          console.log(`Cancelling unfilled order ${order.order_id} on ${order.exchange}`);
          await this.cancelOrder(order.exchange, order.order_id);
          order.status = 'cancelled';
        } catch (error) {
          console.error(`Failed to cancel order ${order.order_id}:`, error.message);
        }
      }
    }
  }

  /**
   * Calculate execution results and P&L
   */
  private calculateExecutionResults(execution: ArbitrageExecution): void {
    const buyOrder = execution.buy_order;
    const sellOrder = execution.sell_order;

    if (!buyOrder || !sellOrder || 
        buyOrder.status !== 'filled' || sellOrder.status !== 'filled') {
      execution.realized_pnl = 0;
      return;
    }

    // Calculate P&L
    const buyNotional = buyOrder.filled_amount * (buyOrder.filled_price || 0);
    const sellNotional = sellOrder.filled_amount * (sellOrder.filled_price || 0);
    const totalFees = buyOrder.fees_paid + sellOrder.fees_paid;
    
    const grossPnL = sellNotional - buyNotional;
    const netPnL = grossPnL - totalFees;
    
    execution.realized_pnl = netPnL;

    // Calculate execution summary
    const executionTime = execution.end_time ? execution.end_time - execution.start_time : 0;
    const plannedSpread = execution.execution_plan.arbitrage_plan.estimated_profit_usd || 0;
    const actualSpread = grossPnL;
    
    execution.execution_summary = {
      total_fees_paid: totalFees,
      effective_spread_captured: actualSpread,
      slippage_experienced: Math.abs(plannedSpread - actualSpread) / Math.abs(plannedSpread),
      execution_time_ms: executionTime,
      success_rate: netPnL > 0 ? 1 : 0,
      notes: [
        `Buy: ${buyOrder.filled_amount.toFixed(6)} @ ${buyOrder.filled_price?.toFixed(2)}`,
        `Sell: ${sellOrder.filled_amount.toFixed(6)} @ ${sellOrder.filled_price?.toFixed(2)}`,
        `Net P&L: $${netPnL.toFixed(2)}`
      ]
    };

    // Store order history
    this.orderHistory.set(buyOrder.order_id, buyOrder);
    this.orderHistory.set(sellOrder.order_id, sellOrder);
  }

  /**
   * Clean up after failed execution
   */
  private async cleanupFailedExecution(execution: ArbitrageExecution): Promise<void> {
    // Cancel any open orders
    const orders = [execution.buy_order, execution.sell_order];
    
    for (const order of orders) {
      if (order && order.status === 'open') {
        try {
          await this.cancelOrder(order.exchange, order.order_id);
        } catch (error) {
          console.error(`Cleanup: Failed to cancel order ${order.order_id}:`, error);
        }
      }
    }

    this.emit('execution_cleanup', execution.trade_id);
  }

  /**
   * Update execution statistics
   */
  private updateExecutionStats(execution: ArbitrageExecution): void {
    this.executionStats.total_trades++;
    
    if (execution.status === 'completed' && execution.realized_pnl && execution.realized_pnl > 0) {
      this.executionStats.successful_trades++;
    }
    
    if (execution.execution_plan.arbitrage_plan.notional_usd) {
      this.executionStats.total_volume_usd += execution.execution_plan.arbitrage_plan.notional_usd;
    }
    
    if (execution.execution_summary?.total_fees_paid) {
      this.executionStats.total_fees_paid += execution.execution_summary.total_fees_paid;
    }
    
    if (execution.execution_summary?.execution_time_ms) {
      const totalTime = this.executionStats.avg_execution_time_ms * (this.executionStats.total_trades - 1);
      this.executionStats.avg_execution_time_ms = 
        (totalTime + execution.execution_summary.execution_time_ms) / this.executionStats.total_trades;
    }
  }

  /**
   * Helper methods for exchange-specific functionality
   */
  private async checkExchangeHealth(exchange: string): Promise<boolean> {
    // Simplified health check - in production would check exchange status APIs
    const config = this.exchangeConfigs.get(exchange);
    return config ? config.enabled : false;
  }

  private async checkAccountBalances(executionPlan: ExecutionPlan): Promise<boolean> {
    // Simplified balance check - always return true for demo
    return true;
  }

  private async checkMarketConditions(executionPlan: ExecutionPlan): Promise<{ suitable: boolean; reason?: string }> {
    // Simplified market condition check
    return { suitable: true };
  }

  private checkRateLimits(exchanges: string[]): boolean {
    const now = Date.now();
    
    for (const exchange of exchanges) {
      const limiter = this.rateLimiters.get(exchange);
      const config = this.exchangeConfigs.get(exchange);
      
      if (!config) continue;
      
      if (limiter) {
        const timeDiff = now - limiter.lastCall;
        if (timeDiff < 1000 && limiter.callCount >= config.rate_limit_per_sec) {
          return false;
        }
      }
    }
    
    return true;
  }

  private checkRateLimit(exchange: string): boolean {
    return this.checkRateLimits([exchange]);
  }

  private updateRateLimit(exchange: string): void {
    const now = Date.now();
    const limiter = this.rateLimiters.get(exchange) || { lastCall: 0, callCount: 0 };
    
    if (now - limiter.lastCall >= 1000) {
      limiter.callCount = 1;
      limiter.lastCall = now;
    } else {
      limiter.callCount++;
    }
    
    this.rateLimiters.set(exchange, limiter);
  }

  // Format pair names for different exchanges
  private formatBinancePair(pair: string): string {
    return pair.replace('-', '').toUpperCase();
  }

  private formatCoinbasePair(pair: string): string {
    return pair.toUpperCase();
  }

  // Utility functions (stubs for demo)
  private async getOrderbook(exchange: string, pair: string): Promise<any> {
    // Simplified orderbook - would fetch from exchange APIs
    return {
      bids: [['50000', '1'], ['49990', '2'], ['49980', '3']],
      asks: [['50010', '1'], ['50020', '2'], ['50030', '3']]
    };
  }

  private async getOrderStatus(exchange: string, orderId: string): Promise<OrderResponse | null> {
    // Simplified order status check - would query exchange APIs
    const cachedOrder = this.orderHistory.get(orderId);
    return cachedOrder || null;
  }

  private async cancelOrder(exchange: string, orderId: string): Promise<void> {
    // Simplified order cancellation - would call exchange APIs
    console.log(`[DEMO] Cancelling order ${orderId} on ${exchange}`);
  }

  private calculateOrderAmount(notionalUsd: number, price: number): number {
    return notionalUsd / price;
  }

  private parseBinanceOrderResponse(data: any, request: OrderRequest): OrderResponse {
    return {
      order_id: data.orderId,
      client_order_id: data.clientOrderId,
      exchange: request.exchange,
      pair: request.pair,
      side: request.side,
      type: request.type,
      amount: parseFloat(data.origQty),
      price: data.price ? parseFloat(data.price) : undefined,
      status: this.mapBinanceStatus(data.status),
      filled_amount: parseFloat(data.executedQty || '0'),
      filled_price: data.fills?.[0]?.price ? parseFloat(data.fills[0].price) : undefined,
      fees_paid: data.fills?.reduce((sum: number, fill: any) => sum + parseFloat(fill.commission), 0) || 0,
      timestamp: data.transactTime || Date.now(),
      raw_response: data
    };
  }

  private parseCoinbaseOrderResponse(data: any, request: OrderRequest): OrderResponse {
    return {
      order_id: data.id,
      client_order_id: data.client_oid,
      exchange: request.exchange,
      pair: request.pair,
      side: request.side,
      type: request.type,
      amount: parseFloat(data.size),
      price: data.price ? parseFloat(data.price) : undefined,
      status: this.mapCoinbaseStatus(data.status),
      filled_amount: parseFloat(data.filled_size || '0'),
      filled_price: data.executed_value && data.filled_size ? 
        parseFloat(data.executed_value) / parseFloat(data.filled_size) : undefined,
      fees_paid: parseFloat(data.fill_fees || '0'),
      timestamp: new Date(data.created_at).getTime(),
      raw_response: data
    };
  }

  private mapBinanceStatus(status: string): OrderResponse['status'] {
    const statusMap: Record<string, OrderResponse['status']> = {
      'NEW': 'open',
      'PARTIALLY_FILLED': 'open',
      'FILLED': 'filled',
      'CANCELED': 'cancelled',
      'REJECTED': 'rejected',
      'EXPIRED': 'cancelled'
    };
    return statusMap[status] || 'pending';
  }

  private mapCoinbaseStatus(status: string): OrderResponse['status'] {
    const statusMap: Record<string, OrderResponse['status']> = {
      'pending': 'pending',
      'open': 'open',
      'active': 'open',
      'done': 'filled',
      'cancelled': 'cancelled',
      'rejected': 'rejected'
    };
    return statusMap[status] || 'pending';
  }

  /**
   * Initialize exchange configurations
   */
  private initializeExchangeConfigs(): void {
    // Binance Testnet/Sandbox
    this.exchangeConfigs.set('binance', {
      name: 'binance',
      enabled: true,
      sandbox: true,
      api_key: process.env.BINANCE_TESTNET_API_KEY || 'demo_key',
      api_secret: process.env.BINANCE_TESTNET_SECRET || 'demo_secret',
      base_url: 'https://testnet.binance.vision',
      rate_limit_per_sec: 10
    });

    // Coinbase Sandbox
    this.exchangeConfigs.set('coinbase', {
      name: 'coinbase',
      enabled: true,
      sandbox: true,
      api_key: process.env.COINBASE_SANDBOX_API_KEY || 'demo_key',
      api_secret: process.env.COINBASE_SANDBOX_SECRET || 'demo_secret',
      passphrase: process.env.COINBASE_SANDBOX_PASSPHRASE || 'demo_passphrase',
      base_url: 'https://api-public.sandbox.exchange.coinbase.com',
      rate_limit_per_sec: 10
    });

    // Kraken (demo mode)
    this.exchangeConfigs.set('kraken', {
      name: 'kraken',
      enabled: true,
      sandbox: true,
      api_key: process.env.KRAKEN_API_KEY || 'demo_key',
      api_secret: process.env.KRAKEN_SECRET || 'demo_secret',
      base_url: 'https://api.kraken.com',
      rate_limit_per_sec: 5
    });
  }

  /**
   * Public API methods
   */
  getActiveExecutions(): ArbitrageExecution[] {
    return Array.from(this.activeExecutions.values());
  }

  getExecutionHistory(limit: number = 50): ArbitrageExecution[] {
    // In production, this would come from database
    return Array.from(this.activeExecutions.values())
      .filter(e => e.status === 'completed' || e.status === 'failed')
      .slice(-limit);
  }

  getExecutionStats() {
    return {
      ...this.executionStats,
      success_rate: this.executionStats.total_trades > 0 ? 
        this.executionStats.successful_trades / this.executionStats.total_trades : 0
    };
  }

  getOrderHistory(limit: number = 100): OrderResponse[] {
    return Array.from(this.orderHistory.values()).slice(-limit);
  }

  async cancelExecution(tradeId: string): Promise<void> {
    const execution = this.activeExecutions.get(tradeId);
    if (!execution) {
      throw new Error(`Execution not found: ${tradeId}`);
    }

    execution.status = 'cancelled';
    await this.cleanupFailedExecution(execution);
  }
}

/**
 * Factory function to create ExecutionAgent
 */
export function createExecutionAgent(): ExecutionAgent {
  return new ExecutionAgent();
}

// Export for testing
export { ExecutionAgent as default };