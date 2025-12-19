/**
 * Portfolio Construction & Risk Control
 * 
 * Manages:
 * - Volatility targeting (position sizing)
 * - Capital allocation across strategies
 * - Exposure caps (max per strategy, total)
 * - Drawdown control (stop-loss triggers)
 * - Dynamic strategy weighting
 * - Risk parity & portfolio optimization
 */

import { Trade, StrategySignal } from './regime-conditional-strategies';
import { MarketRegime } from './market-regime-detection';

export interface PortfolioConfig {
  totalCapital: number;         // Total capital (USD)
  maxDrawdown: number;          // Max portfolio drawdown (%)
  targetVolatility: number;     // Target annualized volatility (%)
  maxLeverage: number;          // Max portfolio leverage
  maxExposurePerStrategy: number; // Max % per strategy
  maxTotalExposure: number;     // Max total exposure (%)
  rebalanceFrequency: number;   // Rebalance interval (minutes)
}

export interface PortfolioPosition {
  strategyName: string;
  trades: Trade[];
  totalExposure: number;        // USD
  weight: number;               // Portfolio weight (0-1)
  pnl: number;                  // Unrealized P&L (USD)
  returnPercent: number;        // Return (%)
  sharpeRatio: number;          // Strategy Sharpe
  volatility: number;           // Strategy volatility (%)
}

export interface PortfolioMetrics {
  totalCapital: number;
  usedCapital: number;
  availableCapital: number;
  totalExposure: number;
  totalLeverage: number;
  totalPnL: number;
  totalReturn: number;          // %
  sharpeRatio: number;
  maxDrawdown: number;          // %
  currentDrawdown: number;      // %
  volatility: number;           // %
  numActiveTrades: number;
  positionsByStrategy: Map<string, PortfolioPosition>;
  riskUtilization: number;      // % of risk budget used
  timestamp: Date;
}

export interface RiskConstraint {
  name: string;
  violated: boolean;
  current: number;
  limit: number;
  severity: 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
}

export class PortfolioRiskManager {
  private config: PortfolioConfig;
  private positions: Map<string, PortfolioPosition>;
  private capitalHistory: number[];
  private returnHistory: number[];
  private peakCapital: number;
  
  constructor(config: Partial<PortfolioConfig> = {}) {
    this.config = {
      totalCapital: config.totalCapital || 100000,
      maxDrawdown: config.maxDrawdown || 20,
      targetVolatility: config.targetVolatility || 15,
      maxLeverage: config.maxLeverage || 3,
      maxExposurePerStrategy: config.maxExposurePerStrategy || 30,
      maxTotalExposure: config.maxTotalExposure || 80,
      rebalanceFrequency: config.rebalanceFrequency || 60,
    };
    
    this.positions = new Map();
    this.capitalHistory = [this.config.totalCapital];
    this.returnHistory = [];
    this.peakCapital = this.config.totalCapital;
  }
  
  /**
   * Calculate portfolio metrics
   */
  calculateMetrics(): PortfolioMetrics {
    const positions = Array.from(this.positions.values());
    
    // Total exposure & leverage
    const totalExposure = positions.reduce((sum, p) => sum + p.totalExposure, 0);
    const totalLeverage = totalExposure / this.config.totalCapital;
    
    // P&L
    const totalPnL = positions.reduce((sum, p) => sum + p.pnl, 0);
    const currentCapital = this.config.totalCapital + totalPnL;
    const totalReturn = (currentCapital - this.config.totalCapital) / this.config.totalCapital * 100;
    
    // Drawdown
    this.peakCapital = Math.max(this.peakCapital, currentCapital);
    const currentDrawdown = (this.peakCapital - currentCapital) / this.peakCapital * 100;
    const maxDrawdown = Math.max(...this.capitalHistory.map((c, i) => {
      const peak = Math.max(...this.capitalHistory.slice(0, i + 1));
      return (peak - c) / peak * 100;
    }));
    
    // Sharpe & Volatility
    const sharpeRatio = this.calculateSharpeRatio();
    const volatility = this.calculateVolatility();
    
    // Capital allocation
    const usedCapital = totalExposure / totalLeverage;
    const availableCapital = this.config.totalCapital - usedCapital;
    
    // Risk utilization
    const riskUtilization = (totalLeverage / this.config.maxLeverage) * 100;
    
    return {
      totalCapital: this.config.totalCapital,
      usedCapital,
      availableCapital,
      totalExposure,
      totalLeverage,
      totalPnL,
      totalReturn,
      sharpeRatio,
      maxDrawdown,
      currentDrawdown,
      volatility,
      numActiveTrades: positions.reduce((sum, p) => sum + p.trades.length, 0),
      positionsByStrategy: this.positions,
      riskUtilization,
      timestamp: new Date(),
    };
  }
  
  /**
   * Check risk constraints
   */
  checkRiskConstraints(): RiskConstraint[] {
    const metrics = this.calculateMetrics();
    const constraints: RiskConstraint[] = [];
    
    // 1. Max drawdown
    constraints.push({
      name: 'Max Drawdown',
      violated: metrics.currentDrawdown > this.config.maxDrawdown,
      current: metrics.currentDrawdown,
      limit: this.config.maxDrawdown,
      severity: metrics.currentDrawdown > this.config.maxDrawdown ? 'CRITICAL' : 'LOW',
    });
    
    // 2. Max leverage
    constraints.push({
      name: 'Max Leverage',
      violated: metrics.totalLeverage > this.config.maxLeverage,
      current: metrics.totalLeverage,
      limit: this.config.maxLeverage,
      severity: metrics.totalLeverage > this.config.maxLeverage ? 'HIGH' : 'LOW',
    });
    
    // 3. Max total exposure
    const exposurePercent = metrics.totalExposure / this.config.totalCapital * 100;
    constraints.push({
      name: 'Max Total Exposure',
      violated: exposurePercent > this.config.maxTotalExposure,
      current: exposurePercent,
      limit: this.config.maxTotalExposure,
      severity: exposurePercent > this.config.maxTotalExposure ? 'HIGH' : 'LOW',
    });
    
    // 4. Per-strategy exposure
    for (const [strategyName, position] of this.positions) {
      const strategyExposurePercent = position.totalExposure / this.config.totalCapital * 100;
      if (strategyExposurePercent > this.config.maxExposurePerStrategy) {
        constraints.push({
          name: `${strategyName} Exposure`,
          violated: true,
          current: strategyExposurePercent,
          limit: this.config.maxExposurePerStrategy,
          severity: 'MEDIUM',
        });
      }
    }
    
    return constraints;
  }
  
  /**
   * Calculate position size using volatility targeting
   */
  calculatePositionSize(
    signal: StrategySignal,
    strategyVolatility: number,
    currentExposure: number
  ): number {
    const metrics = this.calculateMetrics();
    
    // Available capital
    const availableCapital = metrics.availableCapital;
    if (availableCapital <= 0) return 0;
    
    // Volatility-adjusted position size
    const targetVol = this.config.targetVolatility / 100;
    const strategyVol = strategyVolatility / 100;
    
    // Base position size (Kelly Criterion approximation)
    const winRate = signal.confidence;
    const winLoss = signal.expectedReturn / signal.risk;
    const kellyFraction = winRate - (1 - winRate) / winLoss;
    const safeKellyFraction = Math.max(0, Math.min(0.25, kellyFraction * 0.5)); // Half-Kelly
    
    // Volatility scaling
    const volScaling = strategyVol > 0 ? targetVol / strategyVol : 1;
    
    // Position size
    let positionSize = availableCapital * safeKellyFraction * volScaling;
    
    // Apply per-strategy exposure limit
    const maxStrategyExposure = this.config.totalCapital * (this.config.maxExposurePerStrategy / 100);
    positionSize = Math.min(positionSize, maxStrategyExposure - currentExposure);
    
    // Apply total exposure limit
    const maxAdditionalExposure = this.config.totalCapital * (this.config.maxTotalExposure / 100) - metrics.totalExposure;
    positionSize = Math.min(positionSize, maxAdditionalExposure);
    
    return Math.max(0, positionSize);
  }
  
  /**
   * Allocate capital using risk parity
   */
  allocateCapitalRiskParity(): Map<string, number> {
    const positions = Array.from(this.positions.values());
    const allocation = new Map<string, number>();
    
    if (positions.length === 0) return allocation;
    
    // Risk contribution = weight × volatility
    // Risk parity: equal risk contribution from each strategy
    const totalInverseVol = positions.reduce((sum, p) => sum + (p.volatility > 0 ? 1 / p.volatility : 0), 0);
    
    for (const position of positions) {
      const weight = position.volatility > 0 
        ? (1 / position.volatility) / totalInverseVol
        : 1 / positions.length;
      
      const capital = this.config.totalCapital * weight;
      allocation.set(position.strategyName, capital);
    }
    
    return allocation;
  }
  
  /**
   * Optimize portfolio weights using mean-variance optimization
   */
  optimizeWeightsMeanVariance(): Map<string, number> {
    const positions = Array.from(this.positions.values());
    const allocation = new Map<string, number>();
    
    if (positions.length === 0) return allocation;
    
    // Simplified mean-variance: maximize Sharpe ratio
    const totalSharpe = positions.reduce((sum, p) => sum + Math.max(0, p.sharpeRatio), 0);
    
    for (const position of positions) {
      const weight = totalSharpe > 0 
        ? Math.max(0, position.sharpeRatio) / totalSharpe
        : 1 / positions.length;
      
      const capital = this.config.totalCapital * weight;
      allocation.set(position.strategyName, capital);
    }
    
    return allocation;
  }
  
  /**
   * Add or update position
   */
  updatePosition(strategyName: string, trades: Trade[]): void {
    const totalExposure = trades.reduce((sum, t) => sum + t.positionSize * t.leverage, 0);
    const pnl = this.calculatePositionPnL(trades);
    const returns = this.calculatePositionReturns(trades);
    
    const position: PortfolioPosition = {
      strategyName,
      trades,
      totalExposure,
      weight: totalExposure / this.config.totalCapital,
      pnl,
      returnPercent: returns.reduce((sum, r) => sum + r, 0),
      sharpeRatio: this.calculateStrategySharp(returns),
      volatility: this.calculateStrategyVolatility(returns),
    };
    
    this.positions.set(strategyName, position);
  }
  
  /**
   * Remove position
   */
  removePosition(strategyName: string): void {
    this.positions.delete(strategyName);
  }
  
  /**
   * Calculate position P&L
   */
  private calculatePositionPnL(trades: Trade[]): number {
    // Simplified: assume current price = entry price (unrealized P&L = 0)
    // In production, calculate using current market price
    return trades.reduce((sum, t) => {
      const priceDiff = t.targetPrice - t.entryPrice;
      const pnl = priceDiff / t.entryPrice * t.positionSize;
      return sum + (t.side === 'LONG' ? pnl : -pnl);
    }, 0);
  }
  
  /**
   * Calculate position returns
   */
  private calculatePositionReturns(trades: Trade[]): number[] {
    return trades.map(t => {
      const expectedReturn = (t.targetPrice - t.entryPrice) / t.entryPrice;
      return t.side === 'LONG' ? expectedReturn : -expectedReturn;
    });
  }
  
  /**
   * Calculate Sharpe ratio
   */
  private calculateSharpeRatio(): number {
    if (this.returnHistory.length < 2) return 0;
    
    const avgReturn = this.mean(this.returnHistory);
    const stdDev = this.stdDev(this.returnHistory);
    
    return stdDev > 0 ? avgReturn / stdDev * Math.sqrt(252) : 0; // Annualized
  }
  
  /**
   * Calculate strategy Sharpe ratio
   */
  private calculateStrategySharp(returns: number[]): number {
    if (returns.length < 2) return 0;
    
    const avgReturn = this.mean(returns);
    const stdDev = this.stdDev(returns);
    
    return stdDev > 0 ? avgReturn / stdDev : 0;
  }
  
  /**
   * Calculate volatility
   */
  private calculateVolatility(): number {
    if (this.returnHistory.length < 2) return 0;
    
    return this.stdDev(this.returnHistory) * Math.sqrt(252) * 100; // Annualized %
  }
  
  /**
   * Calculate strategy volatility
   */
  private calculateStrategyVolatility(returns: number[]): number {
    if (returns.length < 2) return 0;
    
    return this.stdDev(returns) * Math.sqrt(252) * 100; // Annualized %
  }
  
  /**
   * Update capital history
   */
  updateCapitalHistory(): void {
    const metrics = this.calculateMetrics();
    const currentCapital = metrics.totalCapital + metrics.totalPnL;
    
    this.capitalHistory.push(currentCapital);
    
    if (this.capitalHistory.length > 1) {
      const prevCapital = this.capitalHistory[this.capitalHistory.length - 2];
      const returnPercent = (currentCapital - prevCapital) / prevCapital;
      this.returnHistory.push(returnPercent);
    }
    
    // Keep last 1000 data points
    if (this.capitalHistory.length > 1000) {
      this.capitalHistory.shift();
    }
    if (this.returnHistory.length > 1000) {
      this.returnHistory.shift();
    }
  }
  
  /**
   * Emergency risk reduction (close worst-performing positions)
   */
  emergencyRiskReduction(targetReduction: number): string[] {
    const metrics = this.calculateMetrics();
    const positions = Array.from(this.positions.values());
    
    // Sort by P&L (worst first)
    positions.sort((a, b) => a.pnl - b.pnl);
    
    const closedStrategies: string[] = [];
    let totalReduction = 0;
    
    for (const position of positions) {
      if (totalReduction >= targetReduction) break;
      
      closedStrategies.push(position.strategyName);
      totalReduction += position.totalExposure;
    }
    
    return closedStrategies;
  }
  
  // Helper functions
  private mean(values: number[]): number {
    return values.reduce((sum, v) => sum + v, 0) / values.length;
  }
  
  private stdDev(values: number[]): number {
    const avg = this.mean(values);
    const variance = values.reduce((sum, v) => sum + Math.pow(v - avg, 2), 0) / values.length;
    return Math.sqrt(variance);
  }
  
  /**
   * Get configuration
   */
  getConfig(): PortfolioConfig {
    return this.config;
  }
  
  /**
   * Update configuration
   */
  updateConfig(config: Partial<PortfolioConfig>): void {
    this.config = { ...this.config, ...config };
  }
}

/**
 * Example usage:
 * 
 * const riskManager = new PortfolioRiskManager({
 *   totalCapital: 100000,
 *   maxDrawdown: 20,
 *   targetVolatility: 15,
 *   maxLeverage: 3,
 * });
 * 
 * // Check constraints
 * const constraints = riskManager.checkRiskConstraints();
 * for (const constraint of constraints) {
 *   if (constraint.violated) {
 *     console.log(`⚠️ ${constraint.name}: ${constraint.current.toFixed(2)} > ${constraint.limit.toFixed(2)}`);
 *   }
 * }
 * 
 * // Calculate position size
 * const positionSize = riskManager.calculatePositionSize(signal, strategyVol, currentExposure);
 * 
 * // Get metrics
 * const metrics = riskManager.calculateMetrics();
 * console.log('Portfolio metrics:', metrics);
 */
