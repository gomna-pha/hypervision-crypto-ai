import { EventEmitter } from 'events';
import Logger from '../utils/logger';
import { 
  BacktestConfig, 
  BacktestResult, 
  ExecutionResult,
  FusionPrediction,
  ValidationOutcome 
} from '../types';

interface HistoricalTick {
  timestamp: number;
  exchange: string;
  pair: string;
  bid: number;
  ask: number;
  volume: number;
  spread: number;
}

interface BacktestTrade {
  timestamp: number;
  buyExchange: string;
  sellExchange: string;
  pair: string;
  entrySpread: number;
  exitSpread: number;
  prediction: FusionPrediction;
  entryPrice: { buy: number; sell: number };
  exitPrice: { buy: number; sell: number };
  quantity: number;
  pnl: number;
  fees: number;
  slippage: number;
  holdTime: number;
  success: boolean;
}

export class BacktestEngine extends EventEmitter {
  private logger: Logger;
  private config: BacktestConfig;
  private historicalData: Map<string, HistoricalTick[]>;
  private trades: BacktestTrade[];
  private capital: number;
  private positions: Map<string, any>;
  private metrics: BacktestResult | null = null;

  constructor(config: BacktestConfig) {
    super();
    this.logger = Logger.getInstance('BacktestEngine');
    this.config = config;
    this.historicalData = new Map();
    this.trades = [];
    this.capital = config.initial_capital_usd;
    this.positions = new Map();
  }

  /**
   * Load historical data for backtesting
   */
  async loadHistoricalData(
    startDate: Date, 
    endDate: Date, 
    pairs: string[], 
    exchanges: string[]
  ): Promise<void> {
    this.logger.info('Loading historical data', {
      startDate: startDate.toISOString(),
      endDate: endDate.toISOString(),
      pairs,
      exchanges
    });

    // In production, this would load from database or API
    // For now, generate synthetic data
    for (const exchange of exchanges) {
      for (const pair of pairs) {
        const key = `${exchange}-${pair}`;
        const data = this.generateSyntheticData(
          startDate.getTime(),
          endDate.getTime(),
          exchange,
          pair
        );
        this.historicalData.set(key, data);
      }
    }

    this.logger.info(`Loaded ${this.historicalData.size} historical datasets`);
  }

  /**
   * Generate synthetic historical data for testing
   */
  private generateSyntheticData(
    startTime: number,
    endTime: number,
    exchange: string,
    pair: string
  ): HistoricalTick[] {
    const ticks: HistoricalTick[] = [];
    const interval = 60000; // 1 minute intervals
    let basePrice = pair === 'BTC-USDT' ? 45000 : 3000;
    
    for (let t = startTime; t <= endTime; t += interval) {
      // Random walk with mean reversion
      const volatility = 0.002; // 0.2% volatility
      const drift = (Math.random() - 0.5) * volatility;
      basePrice = basePrice * (1 + drift);
      
      // Exchange-specific spread
      const baseSpread = exchange === 'binance' ? 0.0001 : 0.00015;
      const spread = basePrice * (baseSpread + Math.random() * 0.0001);
      
      ticks.push({
        timestamp: t,
        exchange,
        pair,
        bid: basePrice - spread / 2,
        ask: basePrice + spread / 2,
        volume: 100 + Math.random() * 1000,
        spread: spread
      });
    }
    
    return ticks;
  }

  /**
   * Run backtest with LLM predictions
   */
  async runBacktest(
    predictions: FusionPrediction[],
    startDate: Date,
    endDate: Date
  ): Promise<BacktestResult> {
    this.logger.info('Starting backtest run');
    this.trades = [];
    this.capital = this.config.initial_capital_usd;
    
    // Sort predictions by timestamp
    const sortedPredictions = [...predictions].sort((a, b) => {
      return new Date(a.expected_time_s * 1000).getTime() - 
             new Date(b.expected_time_s * 1000).getTime();
    });

    // Process each prediction
    for (const prediction of sortedPredictions) {
      await this.processPrediction(prediction);
    }

    // Calculate final metrics
    this.metrics = this.calculateMetrics();
    
    this.logger.info('Backtest completed', {
      totalTrades: this.trades.length,
      finalCapital: this.capital,
      totalReturn: ((this.capital - this.config.initial_capital_usd) / 
                    this.config.initial_capital_usd * 100).toFixed(2) + '%'
    });

    return this.metrics;
  }

  /**
   * Process a single prediction in backtest
   */
  private async processPrediction(prediction: FusionPrediction): Promise<void> {
    const { buy, sell, notional_usd } = prediction.arbitrage_plan;
    const pair = 'BTC-USDT'; // Default pair for now
    
    // Get historical prices at prediction time
    const timestamp = Date.now(); // Would use actual prediction timestamp
    const buyData = this.getHistoricalPrice(buy, pair, timestamp);
    const sellData = this.getHistoricalPrice(sell, pair, timestamp);
    
    if (!buyData || !sellData) {
      this.logger.warn('No historical data for prediction', { buy, sell });
      return;
    }

    // Calculate actual spread
    const actualSpread = (sellData.bid - buyData.ask) / buyData.ask;
    
    // Check if spread meets minimum threshold
    if (actualSpread < this.config.fee_model.maker_fee_pct + 
                       this.config.fee_model.taker_fee_pct) {
      return; // Skip unprofitable trade
    }

    // Simulate entry
    const quantity = notional_usd / buyData.ask;
    const entryFees = this.calculateFees(notional_usd);
    const entrySlippage = this.calculateSlippage(notional_usd, buyData.volume);
    
    // Simulate holding period
    const holdTime = prediction.expected_time_s * 1000;
    const exitTimestamp = timestamp + holdTime;
    
    // Get exit prices
    const buyExitData = this.getHistoricalPrice(buy, pair, exitTimestamp);
    const sellExitData = this.getHistoricalPrice(sell, pair, exitTimestamp);
    
    if (!buyExitData || !sellExitData) return;

    // Calculate P&L
    const grossProfit = (sellExitData.bid - buyData.ask) * quantity;
    const exitFees = this.calculateFees(grossProfit);
    const exitSlippage = this.calculateSlippage(grossProfit, sellExitData.volume);
    const netPnl = grossProfit - entryFees - exitFees - entrySlippage - exitSlippage;

    // Record trade
    const trade: BacktestTrade = {
      timestamp,
      buyExchange: buy,
      sellExchange: sell,
      pair,
      entrySpread: actualSpread,
      exitSpread: (sellExitData.bid - buyExitData.ask) / buyExitData.ask,
      prediction,
      entryPrice: { buy: buyData.ask, sell: sellData.bid },
      exitPrice: { buy: buyExitData.ask, sell: sellExitData.bid },
      quantity,
      pnl: netPnl,
      fees: entryFees + exitFees,
      slippage: entrySlippage + exitSlippage,
      holdTime: holdTime / 1000,
      success: netPnl > 0
    };

    this.trades.push(trade);
    this.capital += netPnl;

    // Emit trade event
    this.emit('trade', trade);
  }

  /**
   * Get historical price at specific timestamp
   */
  private getHistoricalPrice(
    exchange: string, 
    pair: string, 
    timestamp: number
  ): HistoricalTick | null {
    const key = `${exchange}-${pair}`;
    const data = this.historicalData.get(key);
    
    if (!data || data.length === 0) return null;
    
    // Find closest tick to timestamp
    let closest = data[0];
    let minDiff = Math.abs(data[0].timestamp - timestamp);
    
    for (const tick of data) {
      const diff = Math.abs(tick.timestamp - timestamp);
      if (diff < minDiff) {
        minDiff = diff;
        closest = tick;
      }
    }
    
    return closest;
  }

  /**
   * Calculate trading fees
   */
  private calculateFees(notional: number): number {
    const { maker_fee_pct, taker_fee_pct } = this.config.fee_model;
    // Assume 50/50 maker/taker
    return notional * (maker_fee_pct + taker_fee_pct) / 2;
  }

  /**
   * Calculate slippage based on volume
   */
  private calculateSlippage(notional: number, volume: number): number {
    const { base_slippage_pct, volume_impact_factor } = this.config.slippage_model;
    const volumeImpact = (notional / volume) * volume_impact_factor;
    return notional * (base_slippage_pct + volumeImpact);
  }

  /**
   * Calculate comprehensive metrics
   */
  private calculateMetrics(): BacktestResult {
    const winningTrades = this.trades.filter(t => t.success);
    const losingTrades = this.trades.filter(t => !t.success);
    
    const totalPnl = this.trades.reduce((sum, t) => sum + t.pnl, 0);
    const returns = this.calculateReturns();
    
    // Calculate Sharpe Ratio
    const sharpeRatio = this.calculateSharpeRatio(returns);
    
    // Calculate Sortino Ratio
    const sortinoRatio = this.calculateSortinoRatio(returns);
    
    // Calculate Max Drawdown
    const maxDrawdown = this.calculateMaxDrawdown();
    
    // Find best and worst trades
    const sortedByPnl = [...this.trades].sort((a, b) => b.pnl - a.pnl);
    const bestTrade = sortedByPnl[0];
    const worstTrade = sortedByPnl[sortedByPnl.length - 1];

    return {
      total_trades: this.trades.length,
      winning_trades: winningTrades.length,
      losing_trades: losingTrades.length,
      total_pnl_usd: totalPnl,
      sharpe_ratio: sharpeRatio,
      sortino_ratio: sortinoRatio,
      max_drawdown_pct: maxDrawdown,
      win_rate: winningTrades.length / this.trades.length,
      avg_trade_duration_s: this.trades.reduce((sum, t) => sum + t.holdTime, 0) / this.trades.length,
      avg_profit_per_trade_usd: totalPnl / this.trades.length,
      best_trade: this.tradeToExecutionResult(bestTrade),
      worst_trade: this.tradeToExecutionResult(worstTrade)
    };
  }

  /**
   * Calculate returns series
   */
  private calculateReturns(): number[] {
    let capital = this.config.initial_capital_usd;
    const returns: number[] = [];
    
    for (const trade of this.trades) {
      const prevCapital = capital;
      capital += trade.pnl;
      returns.push((capital - prevCapital) / prevCapital);
    }
    
    return returns;
  }

  /**
   * Calculate Sharpe Ratio
   */
  private calculateSharpeRatio(returns: number[]): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);
    
    // Annualized (assuming daily returns)
    const annualizedReturn = avgReturn * 252;
    const annualizedStdDev = stdDev * Math.sqrt(252);
    
    return stdDev > 0 ? annualizedReturn / annualizedStdDev : 0;
  }

  /**
   * Calculate Sortino Ratio
   */
  private calculateSortinoRatio(returns: number[]): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = returns.reduce((a, b) => a + b, 0) / returns.length;
    const negativeReturns = returns.filter(r => r < 0);
    
    if (negativeReturns.length === 0) return avgReturn > 0 ? Infinity : 0;
    
    const downside = Math.sqrt(
      negativeReturns.reduce((sum, r) => sum + r * r, 0) / negativeReturns.length
    );
    
    // Annualized
    const annualizedReturn = avgReturn * 252;
    const annualizedDownside = downside * Math.sqrt(252);
    
    return annualizedDownside > 0 ? annualizedReturn / annualizedDownside : 0;
  }

  /**
   * Calculate Maximum Drawdown
   */
  private calculateMaxDrawdown(): number {
    if (this.trades.length === 0) return 0;
    
    let peak = this.config.initial_capital_usd;
    let maxDrawdown = 0;
    let capital = this.config.initial_capital_usd;
    
    for (const trade of this.trades) {
      capital += trade.pnl;
      if (capital > peak) {
        peak = capital;
      }
      const drawdown = (peak - capital) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }
    
    return maxDrawdown * 100; // Return as percentage
  }

  /**
   * Convert BacktestTrade to ExecutionResult
   */
  private tradeToExecutionResult(trade: BacktestTrade | undefined): ExecutionResult {
    if (!trade) {
      return {
        execution_plan_id: '',
        buy_order: {} as any,
        sell_order: {} as any,
        realized_spread_pct: 0,
        realized_profit_usd: 0,
        total_fees_usd: 0,
        execution_time_ms: 0,
        slippage_pct: 0
      };
    }

    return {
      execution_plan_id: `backtest-${trade.timestamp}`,
      buy_order: {
        id: `buy-${trade.timestamp}`,
        exchange: trade.buyExchange,
        pair: trade.pair,
        side: 'buy',
        type: 'limit',
        price: trade.entryPrice.buy,
        quantity: trade.quantity,
        status: 'filled',
        filled_price: trade.entryPrice.buy,
        filled_quantity: trade.quantity,
        fees: trade.fees / 2
      },
      sell_order: {
        id: `sell-${trade.timestamp}`,
        exchange: trade.sellExchange,
        pair: trade.pair,
        side: 'sell',
        type: 'limit',
        price: trade.entryPrice.sell,
        quantity: trade.quantity,
        status: 'filled',
        filled_price: trade.entryPrice.sell,
        filled_quantity: trade.quantity,
        fees: trade.fees / 2
      },
      realized_spread_pct: trade.entrySpread * 100,
      realized_profit_usd: trade.pnl,
      total_fees_usd: trade.fees,
      execution_time_ms: trade.holdTime * 1000,
      slippage_pct: (trade.slippage / (trade.quantity * trade.entryPrice.buy)) * 100
    };
  }

  /**
   * Compare backtest results with LLM predictions
   */
  async compareWithLLM(
    backtestResults: BacktestResult,
    llmPredictions: FusionPrediction[]
  ): Promise<{
    accuracy: number;
    precision: number;
    recall: number;
    mae: number;
    correlation: number;
  }> {
    const comparisons: ValidationOutcome[] = [];
    
    for (const trade of this.trades) {
      comparisons.push({
        execution_plan_id: `${trade.timestamp}`,
        predicted_spread_pct: trade.prediction.predicted_spread_pct,
        realized_spread_pct: trade.entrySpread * 100,
        prediction_error: Math.abs(trade.prediction.predicted_spread_pct - trade.entrySpread * 100),
        predicted_profit_usd: trade.prediction.arbitrage_plan.notional_usd * trade.prediction.predicted_spread_pct / 100,
        realized_profit_usd: trade.pnl,
        profit_error: Math.abs((trade.prediction.arbitrage_plan.notional_usd * trade.prediction.predicted_spread_pct / 100) - trade.pnl),
        execution_time_s: trade.holdTime,
        timestamp: new Date(trade.timestamp).toISOString()
      });
    }

    // Calculate metrics
    const truePositives = comparisons.filter(c => c.predicted_spread_pct > 0.5 && c.realized_profit_usd > 0).length;
    const falsePositives = comparisons.filter(c => c.predicted_spread_pct > 0.5 && c.realized_profit_usd <= 0).length;
    const falseNegatives = comparisons.filter(c => c.predicted_spread_pct <= 0.5 && c.realized_profit_usd > 0).length;
    const trueNegatives = comparisons.filter(c => c.predicted_spread_pct <= 0.5 && c.realized_profit_usd <= 0).length;

    const accuracy = (truePositives + trueNegatives) / comparisons.length;
    const precision = truePositives / (truePositives + falsePositives) || 0;
    const recall = truePositives / (truePositives + falseNegatives) || 0;
    const mae = comparisons.reduce((sum, c) => sum + c.prediction_error, 0) / comparisons.length;

    // Calculate correlation
    const predicted = comparisons.map(c => c.predicted_spread_pct);
    const realized = comparisons.map(c => c.realized_spread_pct);
    const correlation = this.calculateCorrelation(predicted, realized);

    return {
      accuracy,
      precision,
      recall,
      mae,
      correlation
    };
  }

  /**
   * Calculate Pearson correlation coefficient
   */
  private calculateCorrelation(x: number[], y: number[]): number {
    const n = x.length;
    if (n === 0) return 0;

    const sumX = x.reduce((a, b) => a + b, 0);
    const sumY = y.reduce((a, b) => a + b, 0);
    const sumXY = x.reduce((total, xi, i) => total + xi * y[i], 0);
    const sumX2 = x.reduce((total, xi) => total + xi * xi, 0);
    const sumY2 = y.reduce((total, yi) => total + yi * yi, 0);

    const numerator = n * sumXY - sumX * sumY;
    const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

    return denominator === 0 ? 0 : numerator / denominator;
  }

  /**
   * Get current metrics
   */
  getMetrics(): BacktestResult | null {
    return this.metrics;
  }

  /**
   * Get trade history
   */
  getTrades(): BacktestTrade[] {
    return this.trades;
  }
}

export default BacktestEngine;