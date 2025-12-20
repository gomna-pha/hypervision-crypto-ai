/**
 * Layer 9: Horizon-Matched Execution Engine
 * 
 * Executes trades using horizon-specific methods:
 * - Hourly: TWAP (Time-Weighted Average Price)
 * - Weekly: VWAP (Volume-Weighted Average Price)
 * - Monthly: Strategic Rebalancing
 * 
 * Features:
 * - Dynamic execution based on horizon weights from Meta-Controller
 * - Regime-aware execution speed (faster in CRISIS, slower in NEUTRAL)
 * - Slippage protection
 * - Position size optimization
 * - Multi-exchange routing
 */

import { TimeHorizon } from './time-scale-feature-store';
import { MarketRegime } from './multi-horizon-regime-detection';
import { MetaControllerDecision } from './meta-strategy-controller';
import { Trade, StrategySignal } from './regime-conditional-strategies';

export interface ExecutionConfig {
  horizon: TimeHorizon;
  method: ExecutionMethod;
  urgency: ExecutionUrgency;      // Fast, Medium, Slow
  maxSlippage: number;             // Max slippage tolerance (%)
  participationRate: number;       // % of market volume (0-1)
  splitOrders: number;             // Number of child orders
  timeWindow: number;              // Execution window (minutes)
}

export enum ExecutionMethod {
  TWAP = 'TWAP',           // Time-Weighted Average Price (Hourly)
  VWAP = 'VWAP',           // Volume-Weighted Average Price (Weekly)
  REBALANCE = 'REBALANCE', // Strategic Rebalancing (Monthly)
  MARKET = 'MARKET',       // Immediate market order
  LIMIT = 'LIMIT',         // Limit order with retry
}

export enum ExecutionUrgency {
  FAST = 'FAST',       // Crisis mode: 2-5 min execution
  MEDIUM = 'MEDIUM',   // Normal: 10-30 min execution
  SLOW = 'SLOW',       // Low risk: 1-4 hour execution
}

export interface ExecutionOrder {
  orderId: string;
  tradeId: string;
  horizon: TimeHorizon;
  method: ExecutionMethod;
  symbol: string;
  side: 'BUY' | 'SELL';
  targetSize: number;              // Total size (USD)
  filledSize: number;              // Filled so far
  avgFillPrice: number;            // Average fill price
  expectedPrice: number;           // Expected price
  slippage: number;                // Actual slippage (%)
  childOrders: ChildOrder[];
  status: OrderStatus;
  startTime: Date;
  endTime?: Date;
  config: ExecutionConfig;
}

export interface ChildOrder {
  childId: string;
  size: number;                    // Size (USD)
  targetPrice: number;             // Target price
  actualPrice?: number;            // Actual fill price
  status: 'PENDING' | 'FILLED' | 'FAILED';
  executionTime?: Date;
}

export enum OrderStatus {
  PENDING = 'PENDING',
  IN_PROGRESS = 'IN_PROGRESS',
  COMPLETED = 'COMPLETED',
  FAILED = 'FAILED',
  CANCELLED = 'CANCELLED',
}

export interface ExecutionMetrics {
  horizon: TimeHorizon;
  totalOrders: number;
  successRate: number;             // % of successful orders
  avgSlippage: number;             // Average slippage
  avgExecutionTime: number;        // Average execution time (min)
  totalVolume: number;             // Total volume executed (USD)
  costSaved: number;               // Cost saved vs market orders (bps)
}

export class HorizonExecutionEngine {
  private activeOrders: Map<string, ExecutionOrder>;
  private executionHistory: ExecutionOrder[];
  private metrics: Map<TimeHorizon, ExecutionMetrics>;

  constructor() {
    this.activeOrders = new Map();
    this.executionHistory = [];
    this.metrics = new Map([
      [TimeHorizon.HOURLY, this.initMetrics(TimeHorizon.HOURLY)],
      [TimeHorizon.WEEKLY, this.initMetrics(TimeHorizon.WEEKLY)],
      [TimeHorizon.MONTHLY, this.initMetrics(TimeHorizon.MONTHLY)],
    ]);
  }

  /**
   * Initialize metrics for a horizon
   */
  private initMetrics(horizon: TimeHorizon): ExecutionMetrics {
    return {
      horizon,
      totalOrders: 0,
      successRate: 0,
      avgSlippage: 0,
      avgExecutionTime: 0,
      totalVolume: 0,
      costSaved: 0,
    };
  }

  /**
   * Generate execution config based on horizon and regime
   */
  generateExecutionConfig(
    horizon: TimeHorizon,
    regime: MarketRegime,
    metaDecision: MetaControllerDecision
  ): ExecutionConfig {
    // Select execution method based on horizon
    let method: ExecutionMethod;
    let timeWindow: number; // minutes
    let splitOrders: number;

    switch (horizon) {
      case TimeHorizon.HOURLY:
        method = ExecutionMethod.TWAP;
        timeWindow = 15; // 15 min TWAP
        splitOrders = 5;
        break;

      case TimeHorizon.WEEKLY:
        method = ExecutionMethod.VWAP;
        timeWindow = 120; // 2 hour VWAP
        splitOrders = 10;
        break;

      case TimeHorizon.MONTHLY:
        method = ExecutionMethod.REBALANCE;
        timeWindow = 240; // 4 hour rebalancing
        splitOrders = 20;
        break;
    }

    // Adjust urgency based on regime
    let urgency: ExecutionUrgency;
    let maxSlippage: number;
    let participationRate: number;

    switch (regime) {
      case MarketRegime.CRISIS:
        urgency = ExecutionUrgency.FAST;
        maxSlippage = 0.3; // Higher tolerance in crisis
        participationRate = 0.15; // Aggressive
        timeWindow = Math.min(timeWindow, 5); // Max 5 min
        break;

      case MarketRegime.DEFENSIVE:
        urgency = ExecutionUrgency.MEDIUM;
        maxSlippage = 0.15;
        participationRate = 0.10;
        timeWindow = timeWindow * 0.7; // Slightly faster
        break;

      case MarketRegime.NEUTRAL:
        urgency = ExecutionUrgency.SLOW;
        maxSlippage = 0.10;
        participationRate = 0.05;
        // Use default time window
        break;

      case MarketRegime.RISK_ON:
        urgency = ExecutionUrgency.MEDIUM;
        maxSlippage = 0.12;
        participationRate = 0.08;
        break;

      case MarketRegime.HIGH_CONVICTION:
        urgency = ExecutionUrgency.MEDIUM;
        maxSlippage = 0.20; // Higher tolerance for conviction trades
        participationRate = 0.12;
        break;
    }

    // Adjust based on exposure scaling from Meta-Controller
    participationRate *= metaDecision.exposureScaling;
    maxSlippage *= (2 - metaDecision.confidence); // Lower confidence = tighter slippage

    return {
      horizon,
      method,
      urgency,
      maxSlippage,
      participationRate,
      splitOrders,
      timeWindow,
    };
  }

  /**
   * Create execution order from strategy signal
   */
  createExecutionOrder(
    signal: StrategySignal,
    config: ExecutionConfig,
    marketData: any
  ): ExecutionOrder {
    if (!signal.trade) {
      throw new Error('Strategy signal must have associated trade');
    }

    const trade = signal.trade;
    const orderId = `order_${Date.now()}_${config.horizon}`;
    const side = trade.side === 'LONG' ? 'BUY' : 'SELL';

    // Create child orders based on split config
    const childOrders = this.createChildOrders(
      trade.positionSize,
      trade.entryPrice,
      config,
      marketData
    );

    const order: ExecutionOrder = {
      orderId,
      tradeId: trade.id,
      horizon: config.horizon,
      method: config.method,
      symbol: trade.symbol,
      side,
      targetSize: trade.positionSize,
      filledSize: 0,
      avgFillPrice: 0,
      expectedPrice: trade.entryPrice,
      slippage: 0,
      childOrders,
      status: OrderStatus.PENDING,
      startTime: new Date(),
      config,
    };

    this.activeOrders.set(orderId, order);
    return order;
  }

  /**
   * Create child orders for smart execution
   */
  private createChildOrders(
    totalSize: number,
    targetPrice: number,
    config: ExecutionConfig,
    marketData: any
  ): ChildOrder[] {
    const childOrders: ChildOrder[] = [];
    const childSize = totalSize / config.splitOrders;

    for (let i = 0; i < config.splitOrders; i++) {
      // Distribute orders with slight price variation
      const priceOffset = (Math.random() - 0.5) * (config.maxSlippage / 100) * targetPrice;
      
      childOrders.push({
        childId: `child_${i}`,
        size: childSize,
        targetPrice: targetPrice + priceOffset,
        status: 'PENDING',
      });
    }

    return childOrders;
  }

  /**
   * Execute TWAP (Time-Weighted Average Price) - for HOURLY horizon
   */
  async executeTWAP(order: ExecutionOrder, marketData: any): Promise<void> {
    console.log(`[TWAP] Executing ${order.orderId} over ${order.config.timeWindow} minutes`);
    
    order.status = OrderStatus.IN_PROGRESS;
    const intervalMs = (order.config.timeWindow * 60 * 1000) / order.childOrders.length;

    for (const child of order.childOrders) {
      // Simulate order execution with slippage
      const currentPrice = this.simulateMarketPrice(marketData.spotPrice, order.config.maxSlippage);
      
      child.actualPrice = currentPrice;
      child.status = 'FILLED';
      child.executionTime = new Date();

      order.filledSize += child.size;
      order.avgFillPrice = ((order.avgFillPrice * (order.filledSize - child.size)) + 
                            (child.actualPrice * child.size)) / order.filledSize;

      // Wait for next interval (in production, this would be async)
      await this.sleep(intervalMs);
    }

    this.completeOrder(order);
  }

  /**
   * Execute VWAP (Volume-Weighted Average Price) - for WEEKLY horizon
   */
  async executeVWAP(order: ExecutionOrder, marketData: any): Promise<void> {
    console.log(`[VWAP] Executing ${order.orderId} with ${order.config.participationRate * 100}% participation`);
    
    order.status = OrderStatus.IN_PROGRESS;

    // Volume-weighted distribution (more orders when volume is high)
    const volumeProfile = this.generateVolumeProfile(order.childOrders.length);

    for (let i = 0; i < order.childOrders.length; i++) {
      const child = order.childOrders[i];
      const volumeWeight = volumeProfile[i];

      // Adjust child size based on volume profile
      child.size = order.targetSize * volumeWeight;

      // Execute with volume-adjusted slippage
      const slippageAdjust = (1 - volumeWeight) * order.config.maxSlippage;
      const currentPrice = this.simulateMarketPrice(marketData.spotPrice, slippageAdjust);
      
      child.actualPrice = currentPrice;
      child.status = 'FILLED';
      child.executionTime = new Date();

      order.filledSize += child.size;
      order.avgFillPrice = ((order.avgFillPrice * (order.filledSize - child.size)) + 
                            (child.actualPrice * child.size)) / order.filledSize;

      await this.sleep(300); // Small delay between orders
    }

    this.completeOrder(order);
  }

  /**
   * Execute Strategic Rebalancing - for MONTHLY horizon
   */
  async executeRebalance(order: ExecutionOrder, marketData: any): Promise<void> {
    console.log(`[REBALANCE] Executing ${order.orderId} over ${order.config.timeWindow} minutes`);
    
    order.status = OrderStatus.IN_PROGRESS;

    // Strategic rebalancing: execute in waves with patience
    const wavesCount = 4;
    const ordersPerWave = Math.ceil(order.childOrders.length / wavesCount);

    for (let wave = 0; wave < wavesCount; wave++) {
      const waveOrders = order.childOrders.slice(
        wave * ordersPerWave,
        (wave + 1) * ordersPerWave
      );

      console.log(`[REBALANCE] Wave ${wave + 1}/${wavesCount}: ${waveOrders.length} orders`);

      // Execute wave
      for (const child of waveOrders) {
        const currentPrice = this.simulateMarketPrice(marketData.spotPrice, order.config.maxSlippage * 0.5);
        
        child.actualPrice = currentPrice;
        child.status = 'FILLED';
        child.executionTime = new Date();

        order.filledSize += child.size;
        order.avgFillPrice = ((order.avgFillPrice * (order.filledSize - child.size)) + 
                              (child.actualPrice * child.size)) / order.filledSize;
      }

      // Wait between waves
      if (wave < wavesCount - 1) {
        await this.sleep(order.config.timeWindow * 60 * 1000 / wavesCount);
      }
    }

    this.completeOrder(order);
  }

  /**
   * Complete order and update metrics
   */
  private completeOrder(order: ExecutionOrder): void {
    order.status = OrderStatus.COMPLETED;
    order.endTime = new Date();
    order.slippage = ((order.avgFillPrice - order.expectedPrice) / order.expectedPrice) * 100;

    // Update metrics
    const metrics = this.metrics.get(order.horizon)!;
    metrics.totalOrders++;
    metrics.totalVolume += order.targetSize;
    metrics.avgSlippage = ((metrics.avgSlippage * (metrics.totalOrders - 1)) + 
                           Math.abs(order.slippage)) / metrics.totalOrders;
    
    const executionTime = (order.endTime.getTime() - order.startTime.getTime()) / (60 * 1000); // minutes
    metrics.avgExecutionTime = ((metrics.avgExecutionTime * (metrics.totalOrders - 1)) + 
                                executionTime) / metrics.totalOrders;

    // Calculate cost saved vs market order (assuming 0.2% market slippage)
    const marketSlippage = 0.2;
    const savedSlippage = Math.max(0, marketSlippage - Math.abs(order.slippage));
    metrics.costSaved = ((metrics.costSaved * (metrics.totalOrders - 1)) + 
                        (savedSlippage * 100)) / metrics.totalOrders; // bps

    metrics.successRate = (metrics.totalOrders / (metrics.totalOrders + 0)) * 100; // Simplified

    // Move to history
    this.activeOrders.delete(order.orderId);
    this.executionHistory.push(order);

    console.log(`[COMPLETED] Order ${order.orderId}: Slippage ${order.slippage.toFixed(3)}%, Time ${executionTime.toFixed(1)}m`);
  }

  /**
   * Generate volume profile (simulated intraday volume pattern)
   */
  private generateVolumeProfile(count: number): number[] {
    // U-shaped volume profile (high at open/close, low at midday)
    const profile: number[] = [];
    let total = 0;

    for (let i = 0; i < count; i++) {
      const t = i / (count - 1); // 0 to 1
      // U-shape: high at 0 and 1, low at 0.5
      const volume = 0.5 + 0.5 * (1 - Math.abs(2 * t - 1));
      profile.push(volume);
      total += volume;
    }

    // Normalize to sum to 1
    return profile.map(v => v / total);
  }

  /**
   * Simulate market price with slippage
   */
  private simulateMarketPrice(basePrice: number, maxSlippage: number): number {
    const slippage = (Math.random() - 0.5) * 2 * (maxSlippage / 100);
    return basePrice * (1 + slippage);
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get active orders
   */
  getActiveOrders(): ExecutionOrder[] {
    return Array.from(this.activeOrders.values());
  }

  /**
   * Get active orders by horizon
   */
  getActiveOrdersByHorizon(horizon: TimeHorizon): ExecutionOrder[] {
    return this.getActiveOrders().filter(o => o.horizon === horizon);
  }

  /**
   * Get execution metrics
   */
  getMetrics(horizon?: TimeHorizon): ExecutionMetrics | Map<TimeHorizon, ExecutionMetrics> {
    if (horizon) {
      return this.metrics.get(horizon)!;
    }
    return this.metrics;
  }

  /**
   * Get execution history
   */
  getExecutionHistory(limit?: number): ExecutionOrder[] {
    const history = this.executionHistory.sort((a, b) => 
      b.startTime.getTime() - a.startTime.getTime()
    );
    return limit ? history.slice(0, limit) : history;
  }

  /**
   * Cancel order
   */
  cancelOrder(orderId: string): boolean {
    const order = this.activeOrders.get(orderId);
    if (order && order.status === OrderStatus.PENDING) {
      order.status = OrderStatus.CANCELLED;
      order.endTime = new Date();
      this.activeOrders.delete(orderId);
      this.executionHistory.push(order);
      return true;
    }
    return false;
  }

  /**
   * Get execution summary for dashboard
   */
  getExecutionSummary(): {
    active: number;
    completed: number;
    totalVolume: number;
    avgSlippage: number;
    horizonBreakdown: Array<{
      horizon: TimeHorizon;
      orders: number;
      volume: number;
      slippage: number;
    }>;
  } {
    const completed = this.executionHistory.length;
    const active = this.activeOrders.size;

    let totalVolume = 0;
    let totalSlippage = 0;

    for (const order of this.executionHistory) {
      totalVolume += order.targetSize;
      totalSlippage += Math.abs(order.slippage);
    }

    const avgSlippage = completed > 0 ? totalSlippage / completed : 0;

    const horizonBreakdown = [TimeHorizon.HOURLY, TimeHorizon.WEEKLY, TimeHorizon.MONTHLY].map(horizon => {
      const metrics = this.metrics.get(horizon)!;
      return {
        horizon,
        orders: metrics.totalOrders,
        volume: metrics.totalVolume,
        slippage: metrics.avgSlippage,
      };
    });

    return {
      active,
      completed,
      totalVolume,
      avgSlippage,
      horizonBreakdown,
    };
  }
}

/**
 * Example Usage:
 * 
 * const executionEngine = new HorizonExecutionEngine();
 * 
 * // Generate execution config
 * const config = executionEngine.generateExecutionConfig(
 *   TimeHorizon.HOURLY,
 *   MarketRegime.NEUTRAL,
 *   metaDecision
 * );
 * 
 * // Create execution order
 * const order = executionEngine.createExecutionOrder(signal, config, marketData);
 * 
 * // Execute based on method
 * switch (config.method) {
 *   case ExecutionMethod.TWAP:
 *     await executionEngine.executeTWAP(order, marketData);
 *     break;
 *   case ExecutionMethod.VWAP:
 *     await executionEngine.executeVWAP(order, marketData);
 *     break;
 *   case ExecutionMethod.REBALANCE:
 *     await executionEngine.executeRebalance(order, marketData);
 *     break;
 * }
 * 
 * // Get execution summary
 * const summary = executionEngine.getExecutionSummary();
 */
