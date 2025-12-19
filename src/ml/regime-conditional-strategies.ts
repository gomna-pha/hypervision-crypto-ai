/**
 * Regime-Conditional Arbitrage Strategies
 * 
 * Strategy activation conditional on:
 * - Market regime (Crisis, Defensive, Neutral, Risk-On, High Conviction)
 * - XGBoost confidence score
 * - GA-selected signals
 * - Risk constraints
 * 
 * Strategies:
 * 1. Cross-Exchange Spread Trades
 * 2. Funding-Rate Carry Arbitrage
 * 3. Volatility-Driven Basis Trades
 * 4. Regime-Aware Statistical Arbitrage
 */

import { MarketRegime, RegimeState } from './market-regime-detection';
import { MetaModelOutput } from './xgboost-meta-model';
import { SignalGenome } from './genetic-algorithm';
import { AgentSignal } from './agent-signal';

export interface StrategyConfig {
  enabled: boolean;
  minConfidence: number;        // Min XGBoost confidence to activate
  allowedRegimes: MarketRegime[]; // Regimes where strategy is active
  maxPosition: number;          // Max position size (USD)
  maxLeverage: number;          // Max leverage
  stopLoss: number;             // Stop loss (%)
  takeProfit: number;           // Take profit (%)
}

export interface Trade {
  id: string;
  strategy: string;
  symbol: string;
  side: 'LONG' | 'SHORT';
  entryPrice: number;
  targetPrice: number;
  stopLoss: number;
  positionSize: number;         // USD
  leverage: number;
  expectedAlpha: number;        // bps
  confidence: number;           // 0-1
  regime: MarketRegime;
  timestamp: Date;
  status: 'PENDING' | 'ACTIVE' | 'CLOSED';
}

export interface StrategySignal {
  strategy: string;
  action: 'ENTER' | 'EXIT' | 'HOLD';
  confidence: number;
  expectedReturn: number;       // %
  risk: number;                 // %
  trade?: Trade;
  reason: string;
}

export class RegimeConditionalStrategies {
  private strategies: Map<string, StrategyConfig>;
  private activeTrades: Map<string, Trade>;

  constructor() {
    this.strategies = new Map();
    this.activeTrades = new Map();
    this.initializeStrategies();
  }

  /**
   * Initialize strategy configurations
   */
  private initializeStrategies(): void {
    // Strategy 1: Cross-Exchange Spread Trades
    this.strategies.set('cross_exchange_spread', {
      enabled: true,
      minConfidence: 60,
      allowedRegimes: [MarketRegime.NEUTRAL, MarketRegime.RISK_ON, MarketRegime.HIGH_CONVICTION],
      maxPosition: 50000,
      maxLeverage: 3,
      stopLoss: 0.5,
      takeProfit: 1.0,
    });

    // Strategy 2: Funding-Rate Carry Arbitrage
    this.strategies.set('funding_rate_carry', {
      enabled: true,
      minConfidence: 65,
      allowedRegimes: [MarketRegime.NEUTRAL, MarketRegime.RISK_ON],
      maxPosition: 100000,
      maxLeverage: 2,
      stopLoss: 1.0,
      takeProfit: 2.0,
    });

    // Strategy 3: Volatility-Driven Basis Trades
    this.strategies.set('volatility_basis', {
      enabled: true,
      minConfidence: 70,
      allowedRegimes: [MarketRegime.DEFENSIVE, MarketRegime.HIGH_CONVICTION],
      maxPosition: 75000,
      maxLeverage: 2.5,
      stopLoss: 1.5,
      takeProfit: 3.0,
    });

    // Strategy 4: Regime-Aware Statistical Arbitrage
    this.strategies.set('statistical_arbitrage', {
      enabled: true,
      minConfidence: 55,
      allowedRegimes: [MarketRegime.NEUTRAL, MarketRegime.RISK_ON, MarketRegime.HIGH_CONVICTION],
      maxPosition: 60000,
      maxLeverage: 2,
      stopLoss: 1.0,
      takeProfit: 2.5,
    });
  }

  /**
   * Evaluate all strategies and generate signals
   */
  evaluateStrategies(
    regimeState: RegimeState,
    metaModelOutput: MetaModelOutput,
    gaGenome: SignalGenome,
    agentSignals: AgentSignal[],
    marketData: any
  ): StrategySignal[] {
    const signals: StrategySignal[] = [];

    // Evaluate each strategy
    for (const [strategyName, config] of this.strategies.entries()) {
      if (!config.enabled) continue;

      const signal = this.evaluateStrategy(
        strategyName,
        config,
        regimeState,
        metaModelOutput,
        gaGenome,
        agentSignals,
        marketData
      );

      if (signal) {
        signals.push(signal);
      }
    }

    return signals;
  }

  /**
   * Evaluate a single strategy
   */
  private evaluateStrategy(
    strategyName: string,
    config: StrategyConfig,
    regimeState: RegimeState,
    metaModelOutput: MetaModelOutput,
    gaGenome: SignalGenome,
    agentSignals: AgentSignal[],
    marketData: any
  ): StrategySignal | null {
    // Check if regime allows this strategy
    if (!config.allowedRegimes.includes(regimeState.regime)) {
      return null;
    }

    // Check if confidence meets threshold
    if (metaModelOutput.confidenceScore < config.minConfidence) {
      return null;
    }

    // Strategy-specific logic
    switch (strategyName) {
      case 'cross_exchange_spread':
        return this.evaluateCrossExchangeSpread(config, regimeState, metaModelOutput, agentSignals, marketData);
      
      case 'funding_rate_carry':
        return this.evaluateFundingRateCarry(config, regimeState, metaModelOutput, agentSignals, marketData);
      
      case 'volatility_basis':
        return this.evaluateVolatilityBasis(config, regimeState, metaModelOutput, agentSignals, marketData);
      
      case 'statistical_arbitrage':
        return this.evaluateStatisticalArbitrage(config, regimeState, metaModelOutput, agentSignals, marketData);
      
      default:
        return null;
    }
  }

  /**
   * Strategy 1: Cross-Exchange Spread Trades
   */
  private evaluateCrossExchangeSpread(
    config: StrategyConfig,
    regimeState: RegimeState,
    metaModelOutput: MetaModelOutput,
    agentSignals: AgentSignal[],
    marketData: any
  ): StrategySignal | null {
    const crossExAgent = agentSignals.find(s => s.agentId === 'cross_exchange_agent');
    if (!crossExAgent) return null;

    const spread = crossExAgent.features.binanceCoinbaseSpread || 0;
    const spreadZ = crossExAgent.features.spreadZScore || 0;

    // Enter if spread is wide (z-score > 2)
    if (spreadZ > 2 && spread > 20) {
      const side = spread > 0 ? 'SHORT' : 'LONG'; // Buy cheap, sell expensive
      
      return {
        strategy: 'cross_exchange_spread',
        action: 'ENTER',
        confidence: metaModelOutput.confidenceScore / 100,
        expectedReturn: Math.abs(spread) / 10000, // bps to %
        risk: 0.5,
        trade: {
          id: `trade_${Date.now()}`,
          strategy: 'cross_exchange_spread',
          symbol: 'BTC-USD',
          side,
          entryPrice: marketData.spotPrice || 96500,
          targetPrice: marketData.spotPrice * (1 + config.takeProfit / 100),
          stopLoss: marketData.spotPrice * (1 - config.stopLoss / 100),
          positionSize: config.maxPosition * metaModelOutput.exposureScaler,
          leverage: Math.min(config.maxLeverage, metaModelOutput.leverageScaler * 2),
          expectedAlpha: Math.abs(spread),
          confidence: metaModelOutput.confidenceScore / 100,
          regime: regimeState.regime,
          timestamp: new Date(),
          status: 'PENDING',
        },
        reason: `Wide cross-exchange spread detected: ${spread.toFixed(1)} bps (z-score: ${spreadZ.toFixed(2)})`,
      };
    }

    return null;
  }

  /**
   * Strategy 2: Funding-Rate Carry Arbitrage
   */
  private evaluateFundingRateCarry(
    config: StrategyConfig,
    regimeState: RegimeState,
    metaModelOutput: MetaModelOutput,
    agentSignals: AgentSignal[],
    marketData: any
  ): StrategySignal | null {
    const fundingRate = marketData.fundingRate || 0.01; // %
    const openInterest = marketData.openInterest || 1000000000;

    // Positive funding rate = longs pay shorts (short perp, long spot)
    // Negative funding rate = shorts pay longs (long perp, short spot)
    
    if (Math.abs(fundingRate) > 0.05) { // > 5 bps / 8h
      const side = fundingRate > 0 ? 'SHORT' : 'LONG';
      const expectedAlpha = Math.abs(fundingRate) * 3 * 365; // Annualized (3x per day)
      
      return {
        strategy: 'funding_rate_carry',
        action: 'ENTER',
        confidence: metaModelOutput.confidenceScore / 100,
        expectedReturn: expectedAlpha / 100,
        risk: 1.0,
        trade: {
          id: `trade_${Date.now()}`,
          strategy: 'funding_rate_carry',
          symbol: 'BTC-PERP',
          side,
          entryPrice: marketData.perpPrice || 96530,
          targetPrice: marketData.perpPrice * (1 + config.takeProfit / 100),
          stopLoss: marketData.perpPrice * (1 - config.stopLoss / 100),
          positionSize: config.maxPosition * metaModelOutput.exposureScaler,
          leverage: Math.min(config.maxLeverage, metaModelOutput.leverageScaler * 2),
          expectedAlpha: expectedAlpha * 100, // Convert to bps
          confidence: metaModelOutput.confidenceScore / 100,
          regime: regimeState.regime,
          timestamp: new Date(),
          status: 'PENDING',
        },
        reason: `High funding rate: ${(fundingRate * 100).toFixed(2)}% / 8h (annualized: ${expectedAlpha.toFixed(1)}%)`,
      };
    }

    return null;
  }

  /**
   * Strategy 3: Volatility-Driven Basis Trades
   */
  private evaluateVolatilityBasis(
    config: StrategyConfig,
    regimeState: RegimeState,
    metaModelOutput: MetaModelOutput,
    agentSignals: AgentSignal[],
    marketData: any
  ): StrategySignal | null {
    const spotPrice = marketData.spotPrice || 96500;
    const perpPrice = marketData.perpPrice || 96530;
    const volatility = marketData.volatility || 25;

    const basis = (perpPrice - spotPrice) / spotPrice * 10000; // bps
    const basisZScore = (basis - 15) / 10; // Normalize vs historical

    // High volatility + wide basis = opportunity
    if (volatility > 30 && Math.abs(basisZScore) > 2) {
      const side = basis > 0 ? 'SHORT' : 'LONG'; // Mean reversion
      
      return {
        strategy: 'volatility_basis',
        action: 'ENTER',
        confidence: metaModelOutput.confidenceScore / 100,
        expectedReturn: Math.abs(basis) / 10000 * 0.5, // 50% reversion
        risk: 1.5,
        trade: {
          id: `trade_${Date.now()}`,
          strategy: 'volatility_basis',
          symbol: 'BTC-PERP',
          side,
          entryPrice: perpPrice,
          targetPrice: perpPrice * (1 + config.takeProfit / 100 * (side === 'LONG' ? 1 : -1)),
          stopLoss: perpPrice * (1 + config.stopLoss / 100 * (side === 'LONG' ? -1 : 1)),
          positionSize: config.maxPosition * metaModelOutput.exposureScaler * 0.8, // More conservative
          leverage: Math.min(config.maxLeverage, metaModelOutput.leverageScaler * 1.5),
          expectedAlpha: Math.abs(basis) * 0.5,
          confidence: metaModelOutput.confidenceScore / 100,
          regime: regimeState.regime,
          timestamp: new Date(),
          status: 'PENDING',
        },
        reason: `Wide spot-perp basis in high volatility: ${basis.toFixed(1)} bps (vol: ${volatility.toFixed(0)}%)`,
      };
    }

    return null;
  }

  /**
   * Strategy 4: Regime-Aware Statistical Arbitrage
   */
  private evaluateStatisticalArbitrage(
    config: StrategyConfig,
    regimeState: RegimeState,
    metaModelOutput: MetaModelOutput,
    agentSignals: AgentSignal[],
    marketData: any
  ): StrategySignal | null {
    // Use composite signal from all agents
    const compositeSignal = agentSignals.reduce((sum, s) => sum + s.signal * s.confidence, 0) 
      / agentSignals.reduce((sum, s) => sum + s.confidence, 0);

    const priceVsSMA = marketData.priceVsSMA || 0;

    // Mean reversion: price far from SMA + strong agent agreement
    if (Math.abs(priceVsSMA) > 0.02 && Math.abs(compositeSignal) > 0.6) {
      const side = priceVsSMA > 0 ? 'SHORT' : 'LONG'; // Revert to mean
      
      return {
        strategy: 'statistical_arbitrage',
        action: 'ENTER',
        confidence: metaModelOutput.confidenceScore / 100,
        expectedReturn: Math.abs(priceVsSMA) * 0.5, // 50% mean reversion
        risk: 1.0,
        trade: {
          id: `trade_${Date.now()}`,
          strategy: 'statistical_arbitrage',
          symbol: 'BTC-USD',
          side,
          entryPrice: marketData.spotPrice || 96500,
          targetPrice: marketData.sma20 || 96500, // Target = SMA
          stopLoss: marketData.spotPrice * (1 + config.stopLoss / 100 * (side === 'LONG' ? -1 : 1)),
          positionSize: config.maxPosition * metaModelOutput.exposureScaler,
          leverage: Math.min(config.maxLeverage, metaModelOutput.leverageScaler * 2),
          expectedAlpha: Math.abs(priceVsSMA) * 5000, // Convert to bps
          confidence: metaModelOutput.confidenceScore / 100,
          regime: regimeState.regime,
          timestamp: new Date(),
          status: 'PENDING',
        },
        reason: `Statistical mean reversion: price ${(priceVsSMA * 100).toFixed(1)}% from SMA (composite signal: ${compositeSignal.toFixed(2)})`,
      };
    }

    return null;
  }

  /**
   * Add active trade
   */
  addTrade(trade: Trade): void {
    this.activeTrades.set(trade.id, trade);
  }

  /**
   * Get active trades
   */
  getActiveTrades(): Trade[] {
    return Array.from(this.activeTrades.values());
  }

  /**
   * Get trades by strategy
   */
  getTradesByStrategy(strategy: string): Trade[] {
    return this.getActiveTrades().filter(t => t.strategy === strategy);
  }

  /**
   * Close trade
   */
  closeTrade(tradeId: string): void {
    const trade = this.activeTrades.get(tradeId);
    if (trade) {
      trade.status = 'CLOSED';
      this.activeTrades.delete(tradeId);
    }
  }

  /**
   * Get strategy configuration
   */
  getStrategyConfig(strategyName: string): StrategyConfig | undefined {
    return this.strategies.get(strategyName);
  }

  /**
   * Update strategy configuration
   */
  updateStrategyConfig(strategyName: string, config: Partial<StrategyConfig>): void {
    const existing = this.strategies.get(strategyName);
    if (existing) {
      this.strategies.set(strategyName, { ...existing, ...config });
    }
  }
}

/**
 * Example usage:
 * 
 * const strategies = new RegimeConditionalStrategies();
 * 
 * const signals = strategies.evaluateStrategies(
 *   regimeState,
 *   metaModelOutput,
 *   gaGenome,
 *   agentSignals,
 *   marketData
 * );
 * 
 * for (const signal of signals) {
 *   if (signal.action === 'ENTER' && signal.trade) {
 *     console.log(`${signal.strategy}: ${signal.reason}`);
 *     strategies.addTrade(signal.trade);
 *   }
 * }
 */
