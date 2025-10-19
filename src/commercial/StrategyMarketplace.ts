/**
 * Strategy Marketplace - Commercial Algorithm Trading Platform
 * 
 * Industry-standard marketplace for algorithmic trading strategies
 * Enables investors to discover, evaluate, and purchase strategies
 * Implements performance tracking, ranking, and subscription models
 */

import { EventEmitter } from 'events';

// Strategy Performance Metrics (Industry Standard)
interface StrategyMetrics {
  strategyId: string;
  name: string;
  category: 'HFT' | 'StatArb' | 'ML' | 'BarraFactors' | 'Portfolio' | 'Hybrid';
  frequency: 'microsecond' | 'millisecond' | 'second' | 'minute' | 'hour' | 'daily';
  
  // Performance Metrics (GIPS compliant)
  performance: {
    totalReturn: number;          // Cumulative return
    annualizedReturn: number;     // Annualized return
    monthlyReturns: number[];     // Last 12 months
    sharpeRatio: number;          // Risk-adjusted return
    sortinoRatio: number;         // Downside risk-adjusted
    calmarRatio: number;          // Return / Max Drawdown
    informationRatio: number;     // Active return / Tracking error
    treynorRatio: number;         // Return / Beta
    jensenAlpha: number;          // Excess return over CAPM
    betaToMarket: number;         // Market correlation
    r2: number;                   // R-squared to benchmark
  };
  
  // Risk Metrics
  risk: {
    volatility: number;           // Annualized volatility
    maxDrawdown: number;          // Maximum peak-to-trough
    drawdownDuration: number;     // Days in drawdown
    var95: number;                // Value at Risk (95%)
    cvar95: number;               // Conditional VaR (95%)
    downsideDeviation: number;   // Downside volatility
    ulcerIndex: number;           // Drawdown severity
    painIndex: number;            // Average drawdown
    omegaRatio: number;           // Probability-weighted ratio
    tailRatio: number;            // 95th percentile / 5th percentile
  };
  
  // Trading Activity
  activity: {
    totalTrades: number;
    winRate: number;              // Percentage of profitable trades
    avgWin: number;               // Average winning trade
    avgLoss: number;              // Average losing trade
    profitFactor: number;         // Gross profit / Gross loss
    expectancy: number;           // Expected value per trade
    avgHoldingPeriod: number;     // Minutes/Hours/Days
    turnover: number;             // Annual portfolio turnover
    roundTrips: number;           // Completed round trips
    maxConsecutiveWins: number;
    maxConsecutiveLosses: number;
  };
  
  // Market Conditions Performance
  marketRegimes: {
    bullMarket: { return: number; trades: number; };
    bearMarket: { return: number; trades: number; };
    sidewaysMarket: { return: number; trades: number; };
    highVolatility: { return: number; trades: number; };
    lowVolatility: { return: number; trades: number; };
  };
  
  // Strategy Metadata
  metadata: {
    creator: string;              // Strategy developer
    createdDate: Date;
    lastUpdated: Date;
    version: string;
    backtestPeriod: { start: Date; end: Date; };
    liveTrackRecord: { start: Date; end: Date; };
    aum: number;                  // Assets under management
    subscribers: number;           // Active subscribers
    rating: number;               // 1-5 star rating
    reviews: number;              // Number of reviews
  };
  
  // Pricing and Licensing
  pricing: {
    model: 'subscription' | 'performance' | 'hybrid' | 'onetime';
    subscriptionFee?: number;     // Monthly subscription
    performanceFee?: number;      // % of profits (2/20 standard)
    managementFee?: number;       // % of AUM
    oneTimeFee?: number;          // One-time purchase
    minimumInvestment?: number;   // Minimum capital required
    lockupPeriod?: number;        // Days before withdrawal
  };
}

// Investor Profile and Preferences
interface InvestorProfile {
  investorId: string;
  type: 'retail' | 'professional' | 'institutional' | 'family_office' | 'hedge_fund';
  
  // Investment Goals
  goals: {
    primaryGoal: 'capital_preservation' | 'income' | 'growth' | 'aggressive_growth' | 'speculation';
    targetReturn: number;         // Annual target return
    riskTolerance: 'conservative' | 'moderate' | 'aggressive' | 'very_aggressive';
    investmentHorizon: number;    // Months
    liquidityNeeds: 'daily' | 'weekly' | 'monthly' | 'quarterly' | 'annual' | 'none';
  };
  
  // Constraints
  constraints: {
    maxDrawdown: number;          // Maximum acceptable drawdown
    maxVolatility: number;        // Maximum acceptable volatility
    minSharpe: number;            // Minimum Sharpe ratio
    maxLeverage: number;          // Maximum leverage allowed
    restrictedAssets?: string[];  // Assets to exclude
    esgCompliant?: boolean;       // ESG requirements
    shariaCompliant?: boolean;    // Islamic finance compliant
  };
  
  // Portfolio Allocation
  allocation: {
    totalCapital: number;
    maxPerStrategy: number;       // Max allocation per strategy
    diversificationRequirement: number; // Min number of strategies
    rebalancingFrequency: 'daily' | 'weekly' | 'monthly' | 'quarterly';
  };
  
  // Compliance (KYC/AML)
  compliance: {
    kycStatus: 'pending' | 'verified' | 'rejected';
    amlStatus: 'clear' | 'review' | 'flagged';
    accreditedInvestor: boolean;
    qualifiedPurchaser: boolean;
    jurisdiction: string;
    taxResidency: string;
  };
}

// Strategy Subscription
interface StrategySubscription {
  subscriptionId: string;
  investorId: string;
  strategyId: string;
  status: 'trial' | 'active' | 'paused' | 'cancelled' | 'expired';
  
  // Subscription Details
  details: {
    startDate: Date;
    endDate?: Date;
    trialEndDate?: Date;
    allocatedCapital: number;
    actualCapital: number;        // Current value
    pnl: number;                  // Total P&L
    fees: {
      subscriptionPaid: number;
      performancePaid: number;
      managementPaid: number;
      totalPaid: number;
    };
  };
  
  // Performance Tracking
  performance: {
    totalReturn: number;
    dailyReturns: number[];
    currentDrawdown: number;
    trades: number;
    winRate: number;
  };
  
  // Risk Monitoring
  riskMonitoring: {
    currentVaR: number;
    leverageUsed: number;
    marginUsed: number;
    alerts: Array<{
      type: string;
      message: string;
      timestamp: Date;
      severity: 'info' | 'warning' | 'critical';
    }>;
  };
}

export class StrategyMarketplace extends EventEmitter {
  private strategies: Map<string, StrategyMetrics> = new Map();
  private investors: Map<string, InvestorProfile> = new Map();
  private subscriptions: Map<string, StrategySubscription> = new Map();
  private rankings: Map<string, number> = new Map(); // Strategy rankings
  
  constructor() {
    super();
    this.initializeMarketplace();
  }
  
  private initializeMarketplace(): void {
    console.log('🏪 Initializing Strategy Marketplace...');
    
    // Load existing strategies
    this.loadStrategies();
    
    // Calculate rankings
    this.calculateRankings();
    
    // Start performance tracking
    this.startPerformanceTracking();
  }
  
  private loadStrategies(): void {
    // Example strategies with realistic metrics
    const strategies: StrategyMetrics[] = [
      {
        strategyId: 'barra-momentum-01',
        name: 'Barra Momentum Factor Premium',
        category: 'BarraFactors',
        frequency: 'minute',
        performance: {
          totalReturn: 0.4823,
          annualizedReturn: 0.2341,
          monthlyReturns: [0.032, 0.018, -0.009, 0.041, 0.027, 0.013, -0.005, 0.038, 0.022, 0.019, 0.031, 0.024],
          sharpeRatio: 1.87,
          sortinoRatio: 2.43,
          calmarRatio: 3.21,
          informationRatio: 1.65,
          treynorRatio: 0.19,
          jensenAlpha: 0.0823,
          betaToMarket: 0.68,
          r2: 0.71
        },
        risk: {
          volatility: 0.1254,
          maxDrawdown: 0.0729,
          drawdownDuration: 42,
          var95: 0.0198,
          cvar95: 0.0287,
          downsideDeviation: 0.0512,
          ulcerIndex: 0.0234,
          painIndex: 0.0187,
          omegaRatio: 1.92,
          tailRatio: 2.31
        },
        activity: {
          totalTrades: 8745,
          winRate: 0.5834,
          avgWin: 0.0034,
          avgLoss: -0.0021,
          profitFactor: 1.94,
          expectancy: 0.0008,
          avgHoldingPeriod: 240, // minutes
          turnover: 18.5,
          roundTrips: 4372,
          maxConsecutiveWins: 18,
          maxConsecutiveLosses: 9
        },
        marketRegimes: {
          bullMarket: { return: 0.3421, trades: 3890 },
          bearMarket: { return: 0.0821, trades: 2103 },
          sidewaysMarket: { return: 0.0581, trades: 2752 },
          highVolatility: { return: 0.1923, trades: 4211 },
          lowVolatility: { return: 0.2900, trades: 4534 }
        },
        metadata: {
          creator: 'QuantumAlpha Labs',
          createdDate: new Date('2023-01-15'),
          lastUpdated: new Date(),
          version: '2.3.1',
          backtestPeriod: { start: new Date('2020-01-01'), end: new Date('2023-01-01') },
          liveTrackRecord: { start: new Date('2023-01-15'), end: new Date() },
          aum: 45000000,
          subscribers: 127,
          rating: 4.7,
          reviews: 89
        },
        pricing: {
          model: 'hybrid',
          subscriptionFee: 299,
          performanceFee: 0.20,
          managementFee: 0.02,
          minimumInvestment: 10000,
          lockupPeriod: 30
        }
      },
      {
        strategyId: 'statarb-pairs-02',
        name: 'Statistical Arbitrage Cointegration Master',
        category: 'StatArb',
        frequency: 'second',
        performance: {
          totalReturn: 0.6234,
          annualizedReturn: 0.2876,
          monthlyReturns: [0.041, 0.029, 0.037, 0.018, 0.044, 0.032, 0.021, 0.048, 0.039, 0.026, 0.035, 0.042],
          sharpeRatio: 2.34,
          sortinoRatio: 3.12,
          calmarRatio: 4.87,
          informationRatio: 2.01,
          treynorRatio: 0.31,
          jensenAlpha: 0.1234,
          betaToMarket: 0.23,
          r2: 0.18
        },
        risk: {
          volatility: 0.0987,
          maxDrawdown: 0.0591,
          drawdownDuration: 28,
          var95: 0.0156,
          cvar95: 0.0221,
          downsideDeviation: 0.0398,
          ulcerIndex: 0.0187,
          painIndex: 0.0143,
          omegaRatio: 2.45,
          tailRatio: 2.89
        },
        activity: {
          totalTrades: 42318,
          winRate: 0.6123,
          avgWin: 0.0018,
          avgLoss: -0.0012,
          profitFactor: 2.31,
          expectancy: 0.0005,
          avgHoldingPeriod: 45, // minutes
          turnover: 124.5,
          roundTrips: 21159,
          maxConsecutiveWins: 24,
          maxConsecutiveLosses: 7
        },
        marketRegimes: {
          bullMarket: { return: 0.1823, trades: 15234 },
          bearMarket: { return: 0.2134, trades: 11432 },
          sidewaysMarket: { return: 0.2277, trades: 15652 },
          highVolatility: { return: 0.3421, trades: 19234 },
          lowVolatility: { return: 0.2813, trades: 23084 }
        },
        metadata: {
          creator: 'Renaissance Quant Group',
          createdDate: new Date('2022-06-01'),
          lastUpdated: new Date(),
          version: '3.1.4',
          backtestPeriod: { start: new Date('2019-01-01'), end: new Date('2022-06-01') },
          liveTrackRecord: { start: new Date('2022-06-01'), end: new Date() },
          aum: 128000000,
          subscribers: 342,
          rating: 4.9,
          reviews: 256
        },
        pricing: {
          model: 'performance',
          performanceFee: 0.30,
          managementFee: 0.015,
          minimumInvestment: 25000,
          lockupPeriod: 90
        }
      },
      {
        strategyId: 'ml-ensemble-03',
        name: 'ML Ensemble Alpha Generator',
        category: 'ML',
        frequency: 'minute',
        performance: {
          totalReturn: 0.7823,
          annualizedReturn: 0.3421,
          monthlyReturns: [0.052, 0.038, 0.045, 0.029, 0.061, 0.041, 0.033, 0.057, 0.048, 0.035, 0.044, 0.051],
          sharpeRatio: 2.67,
          sortinoRatio: 3.89,
          calmarRatio: 5.23,
          informationRatio: 2.34,
          treynorRatio: 0.38,
          jensenAlpha: 0.1567,
          betaToMarket: 0.41,
          r2: 0.34
        },
        risk: {
          volatility: 0.1281,
          maxDrawdown: 0.0654,
          drawdownDuration: 35,
          var95: 0.0201,
          cvar95: 0.0289,
          downsideDeviation: 0.0467,
          ulcerIndex: 0.0211,
          painIndex: 0.0167,
          omegaRatio: 2.78,
          tailRatio: 3.21
        },
        activity: {
          totalTrades: 15672,
          winRate: 0.5912,
          avgWin: 0.0041,
          avgLoss: -0.0023,
          profitFactor: 2.18,
          expectancy: 0.0012,
          avgHoldingPeriod: 180, // minutes
          turnover: 31.2,
          roundTrips: 7836,
          maxConsecutiveWins: 21,
          maxConsecutiveLosses: 8
        },
        marketRegimes: {
          bullMarket: { return: 0.2934, trades: 5678 },
          bearMarket: { return: 0.1823, trades: 3421 },
          sidewaysMarket: { return: 0.1456, trades: 4123 },
          highVolatility: { return: 0.4123, trades: 7234 },
          lowVolatility: { return: 0.3700, trades: 8438 }
        },
        metadata: {
          creator: 'DeepQuant AI',
          createdDate: new Date('2022-09-01'),
          lastUpdated: new Date(),
          version: '4.2.0',
          backtestPeriod: { start: new Date('2018-01-01'), end: new Date('2022-09-01') },
          liveTrackRecord: { start: new Date('2022-09-01'), end: new Date() },
          aum: 234000000,
          subscribers: 523,
          rating: 4.8,
          reviews: 412
        },
        pricing: {
          model: 'hybrid',
          subscriptionFee: 499,
          performanceFee: 0.25,
          managementFee: 0.02,
          minimumInvestment: 50000,
          lockupPeriod: 60
        }
      }
    ];
    
    strategies.forEach(strategy => {
      this.strategies.set(strategy.strategyId, strategy);
    });
    
    console.log(`✅ Loaded ${strategies.length} strategies to marketplace`);
  }
  
  private calculateRankings(): void {
    // Multi-factor ranking algorithm
    const weights = {
      sharpeRatio: 0.25,
      totalReturn: 0.20,
      calmarRatio: 0.15,
      winRate: 0.10,
      subscribers: 0.10,
      rating: 0.10,
      drawdown: 0.10
    };
    
    const scores = new Map<string, number>();
    
    this.strategies.forEach((strategy, id) => {
      const score = 
        (strategy.performance.sharpeRatio / 3) * weights.sharpeRatio +
        strategy.performance.totalReturn * weights.totalReturn +
        (strategy.performance.calmarRatio / 6) * weights.calmarRatio +
        strategy.activity.winRate * weights.winRate +
        Math.min(strategy.metadata.subscribers / 500, 1) * weights.subscribers +
        (strategy.metadata.rating / 5) * weights.rating +
        (1 - strategy.risk.maxDrawdown) * weights.drawdown;
      
      scores.set(id, score);
    });
    
    // Sort and assign ranks
    const sorted = Array.from(scores.entries()).sort((a, b) => b[1] - a[1]);
    sorted.forEach((entry, index) => {
      this.rankings.set(entry[0], index + 1);
    });
    
    console.log('📊 Strategy rankings calculated');
  }
  
  private startPerformanceTracking(): void {
    // Real-time performance updates
    setInterval(() => {
      this.updateStrategyPerformance();
      this.updateSubscriptionPerformance();
      this.calculateRankings();
      
      this.emit('marketplace_update', {
        strategies: Array.from(this.strategies.values()),
        rankings: Array.from(this.rankings.entries())
      });
    }, 60000); // Update every minute
  }
  
  private updateStrategyPerformance(): void {
    this.strategies.forEach((strategy, id) => {
      // Simulate realistic performance updates
      const dailyReturn = (Math.random() - 0.48) * 0.02; // -0.96% to +1.04% daily
      const volatilityAdjustment = Math.random() * 0.001;
      
      strategy.performance.totalReturn *= (1 + dailyReturn);
      strategy.risk.volatility += volatilityAdjustment - 0.0005;
      strategy.activity.totalTrades += Math.floor(Math.random() * 100);
      
      // Update moving metrics
      if (dailyReturn > 0) {
        strategy.activity.winRate = (strategy.activity.winRate * 0.99) + (0.01 * 1);
      } else {
        strategy.activity.winRate = (strategy.activity.winRate * 0.99) + (0.01 * 0);
      }
      
      // Update drawdown
      if (dailyReturn < 0) {
        strategy.risk.maxDrawdown = Math.max(
          strategy.risk.maxDrawdown,
          Math.abs(dailyReturn)
        );
      }
    });
  }
  
  private updateSubscriptionPerformance(): void {
    this.subscriptions.forEach((subscription, id) => {
      const strategy = this.strategies.get(subscription.strategyId);
      if (!strategy) return;
      
      // Calculate subscription-specific performance
      const dailyReturn = (Math.random() - 0.48) * 0.02;
      subscription.performance.totalReturn *= (1 + dailyReturn);
      subscription.performance.dailyReturns.push(dailyReturn);
      
      // Keep only last 30 days
      if (subscription.performance.dailyReturns.length > 30) {
        subscription.performance.dailyReturns.shift();
      }
      
      // Update P&L
      subscription.details.actualCapital *= (1 + dailyReturn);
      subscription.details.pnl = subscription.details.actualCapital - subscription.details.allocatedCapital;
      
      // Calculate fees
      if (subscription.details.pnl > 0 && strategy.pricing.performanceFee) {
        subscription.details.fees.performancePaid += 
          subscription.details.pnl * strategy.pricing.performanceFee / 365;
      }
      
      if (strategy.pricing.managementFee) {
        subscription.details.fees.managementPaid += 
          subscription.details.actualCapital * strategy.pricing.managementFee / 365;
      }
      
      // Risk monitoring
      subscription.riskMonitoring.currentVaR = strategy.risk.var95 * subscription.details.actualCapital;
      
      // Generate alerts if needed
      if (subscription.performance.currentDrawdown > subscription.details.allocatedCapital * 0.10) {
        subscription.riskMonitoring.alerts.push({
          type: 'drawdown',
          message: `Strategy experiencing ${(subscription.performance.currentDrawdown * 100).toFixed(2)}% drawdown`,
          timestamp: new Date(),
          severity: 'warning'
        });
      }
    });
  }
  
  // Public API Methods
  
  /**
   * Get recommended strategies for an investor based on their profile
   */
  async getRecommendedStrategies(investorId: string): Promise<StrategyMetrics[]> {
    const investor = this.investors.get(investorId);
    if (!investor) {
      throw new Error('Investor profile not found');
    }
    
    const recommendations: StrategyMetrics[] = [];
    
    this.strategies.forEach(strategy => {
      let score = 0;
      
      // Match risk tolerance
      if (investor.goals.riskTolerance === 'conservative' && strategy.risk.volatility < 0.10) {
        score += 30;
      } else if (investor.goals.riskTolerance === 'moderate' && strategy.risk.volatility < 0.15) {
        score += 30;
      } else if (investor.goals.riskTolerance === 'aggressive' && strategy.risk.volatility < 0.25) {
        score += 30;
      } else if (investor.goals.riskTolerance === 'very_aggressive') {
        score += 30;
      }
      
      // Match return target
      const returnDiff = Math.abs(strategy.performance.annualizedReturn - investor.goals.targetReturn);
      score += Math.max(0, 30 - (returnDiff * 100));
      
      // Match drawdown constraint
      if (strategy.risk.maxDrawdown <= investor.constraints.maxDrawdown) {
        score += 20;
      }
      
      // Match Sharpe requirement
      if (strategy.performance.sharpeRatio >= investor.constraints.minSharpe) {
        score += 20;
      }
      
      // Add to recommendations if score is high enough
      if (score >= 50) {
        recommendations.push(strategy);
      }
    });
    
    // Sort by combined score and ranking
    recommendations.sort((a, b) => {
      const rankA = this.rankings.get(a.strategyId) || 999;
      const rankB = this.rankings.get(b.strategyId) || 999;
      return rankA - rankB;
    });
    
    return recommendations.slice(0, 10); // Top 10 recommendations
  }
  
  /**
   * Subscribe an investor to a strategy
   */
  async subscribeToStrategy(
    investorId: string,
    strategyId: string,
    allocatedCapital: number
  ): Promise<StrategySubscription> {
    const investor = this.investors.get(investorId);
    const strategy = this.strategies.get(strategyId);
    
    if (!investor || !strategy) {
      throw new Error('Invalid investor or strategy');
    }
    
    // Check compliance
    if (investor.compliance.kycStatus !== 'verified') {
      throw new Error('KYC verification required');
    }
    
    // Check minimum investment
    if (strategy.pricing.minimumInvestment && 
        allocatedCapital < strategy.pricing.minimumInvestment) {
      throw new Error(`Minimum investment is ${strategy.pricing.minimumInvestment}`);
    }
    
    // Create subscription
    const subscription: StrategySubscription = {
      subscriptionId: `sub_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      investorId,
      strategyId,
      status: 'active',
      details: {
        startDate: new Date(),
        allocatedCapital,
        actualCapital: allocatedCapital,
        pnl: 0,
        fees: {
          subscriptionPaid: strategy.pricing.subscriptionFee || 0,
          performancePaid: 0,
          managementPaid: 0,
          totalPaid: strategy.pricing.subscriptionFee || 0
        }
      },
      performance: {
        totalReturn: 0,
        dailyReturns: [],
        currentDrawdown: 0,
        trades: 0,
        winRate: 0
      },
      riskMonitoring: {
        currentVaR: 0,
        leverageUsed: 0,
        marginUsed: 0,
        alerts: []
      }
    };
    
    this.subscriptions.set(subscription.subscriptionId, subscription);
    
    // Update strategy AUM and subscribers
    strategy.metadata.aum += allocatedCapital;
    strategy.metadata.subscribers += 1;
    
    // Emit subscription event
    this.emit('new_subscription', {
      subscription,
      investor,
      strategy
    });
    
    console.log(`✅ Investor ${investorId} subscribed to strategy ${strategyId}`);
    
    return subscription;
  }
  
  /**
   * Get top performing strategies
   */
  getTopStrategies(limit: number = 10): StrategyMetrics[] {
    const sorted = Array.from(this.rankings.entries())
      .sort((a, b) => a[1] - b[1])
      .slice(0, limit);
    
    return sorted.map(([id]) => this.strategies.get(id)!).filter(s => s);
  }
  
  /**
   * Get strategies by category
   */
  getStrategiesByCategory(category: string): StrategyMetrics[] {
    return Array.from(this.strategies.values())
      .filter(s => s.category === category);
  }
  
  /**
   * Get investor's active subscriptions
   */
  getInvestorSubscriptions(investorId: string): StrategySubscription[] {
    return Array.from(this.subscriptions.values())
      .filter(s => s.investorId === investorId && s.status === 'active');
  }
  
  /**
   * Calculate portfolio allocation recommendations
   */
  calculateOptimalAllocation(
    investorId: string,
    strategies: string[]
  ): Map<string, number> {
    const investor = this.investors.get(investorId);
    if (!investor) {
      throw new Error('Investor not found');
    }
    
    const allocations = new Map<string, number>();
    const totalCapital = investor.allocation.totalCapital;
    const maxPerStrategy = investor.allocation.maxPerStrategy;
    
    // Simple equal-weight allocation with constraints
    const baseAllocation = Math.min(
      totalCapital / strategies.length,
      maxPerStrategy
    );
    
    strategies.forEach(strategyId => {
      allocations.set(strategyId, baseAllocation);
    });
    
    // TODO: Implement sophisticated portfolio optimization
    // using Markowitz, Black-Litterman, or Risk Parity
    
    return allocations;
  }
  
  /**
   * Register a new investor
   */
  async registerInvestor(profile: InvestorProfile): Promise<void> {
    this.investors.set(profile.investorId, profile);
    
    this.emit('investor_registered', profile);
    
    console.log(`✅ Investor ${profile.investorId} registered`);
  }
  
  /**
   * Get marketplace statistics
   */
  getMarketplaceStats(): any {
    const totalAUM = Array.from(this.strategies.values())
      .reduce((sum, s) => sum + s.metadata.aum, 0);
    
    const totalSubscribers = Array.from(this.strategies.values())
      .reduce((sum, s) => sum + s.metadata.subscribers, 0);
    
    const avgSharpe = Array.from(this.strategies.values())
      .reduce((sum, s) => sum + s.performance.sharpeRatio, 0) / this.strategies.size;
    
    const avgReturn = Array.from(this.strategies.values())
      .reduce((sum, s) => sum + s.performance.annualizedReturn, 0) / this.strategies.size;
    
    return {
      totalStrategies: this.strategies.size,
      totalInvestors: this.investors.size,
      activeSubscriptions: Array.from(this.subscriptions.values())
        .filter(s => s.status === 'active').length,
      totalAUM,
      totalSubscribers,
      avgSharpeRatio: avgSharpe.toFixed(2),
      avgAnnualizedReturn: (avgReturn * 100).toFixed(2) + '%',
      topCategory: this.getTopCategory(),
      lastUpdated: new Date()
    };
  }
  
  private getTopCategory(): string {
    const categoryCounts = new Map<string, number>();
    
    this.strategies.forEach(strategy => {
      const count = categoryCounts.get(strategy.category) || 0;
      categoryCounts.set(strategy.category, count + 1);
    });
    
    let topCategory = '';
    let maxCount = 0;
    
    categoryCounts.forEach((count, category) => {
      if (count > maxCount) {
        maxCount = count;
        topCategory = category;
      }
    });
    
    return topCategory;
  }
}

// Export for use in other modules
export type { StrategyMetrics, InvestorProfile, StrategySubscription };