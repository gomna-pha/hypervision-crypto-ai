/**
 * Investor Portal - Commercial Trading Platform
 * 
 * Comprehensive investor interface for strategy discovery, subscription,
 * portfolio management, and performance tracking
 * Implements industry-standard payment processing and compliance
 */

import { EventEmitter } from 'events';
import { StrategyMarketplace, StrategyMetrics, InvestorProfile, StrategySubscription } from './StrategyMarketplace';

// Payment Processing
interface PaymentMethod {
  id: string;
  type: 'credit_card' | 'debit_card' | 'bank_transfer' | 'wire' | 'crypto' | 'paypal';
  provider: 'stripe' | 'plaid' | 'coinbase' | 'wise' | 'paypal';
  
  details: {
    last4?: string;           // Last 4 digits of card/account
    brand?: string;           // Visa, Mastercard, etc.
    bankName?: string;        // Bank name for transfers
    cryptoAddress?: string;   // Crypto wallet address
    isDefault: boolean;
    isVerified: boolean;
    addedDate: Date;
  };
}

// Transaction Records
interface Transaction {
  transactionId: string;
  investorId: string;
  type: 'deposit' | 'withdrawal' | 'subscription_fee' | 'performance_fee' | 'management_fee';
  
  amount: number;
  currency: string;
  status: 'pending' | 'processing' | 'completed' | 'failed' | 'cancelled';
  
  paymentMethod: PaymentMethod;
  
  details: {
    strategyId?: string;
    subscriptionId?: string;
    description: string;
    initiatedAt: Date;
    completedAt?: Date;
    failureReason?: string;
  };
  
  compliance: {
    amlChecked: boolean;
    sanctionsChecked: boolean;
    limitChecked: boolean;
    reportingRequired: boolean;
  };
}

// Portfolio Overview
interface InvestorPortfolio {
  investorId: string;
  
  // Account Summary
  account: {
    totalValue: number;        // Total portfolio value
    cashBalance: number;       // Available cash
    investedAmount: number;    // Amount in strategies
    totalPnL: number;         // All-time P&L
    totalPnLPercent: number;  // All-time P&L %
    dailyPnL: number;         // Today's P&L
    dailyPnLPercent: number;  // Today's P&L %
  };
  
  // Active Strategies
  activeStrategies: Array<{
    strategyId: string;
    name: string;
    allocation: number;        // Amount allocated
    currentValue: number;      // Current value
    pnl: number;              // Strategy P&L
    pnlPercent: number;       // Strategy P&L %
    weight: number;           // Portfolio weight %
  }>;
  
  // Performance Metrics
  performance: {
    totalReturn: number;
    annualizedReturn: number;
    sharpeRatio: number;
    sortinoRatio: number;
    maxDrawdown: number;
    volatility: number;
    beta: number;
    alpha: number;
  };
  
  // Risk Metrics
  risk: {
    portfolioVaR: number;      // Portfolio Value at Risk
    portfolioCVaR: number;     // Conditional VaR
    correlation: number;       // Correlation between strategies
    concentration: number;     // Concentration risk (HHI)
    leverage: number;         // Overall leverage
  };
  
  // Historical Data
  history: {
    dailyReturns: number[];   // Last 30 days
    monthlyReturns: number[]; // Last 12 months
    transactions: Transaction[]; // Recent transactions
    rebalancingHistory: Array<{
      date: Date;
      changes: Map<string, number>;
      reason: string;
    }>;
  };
}

// Onboarding Flow
interface OnboardingStep {
  step: number;
  name: string;
  status: 'pending' | 'in_progress' | 'completed' | 'failed';
  
  data: {
    kycData?: any;            // KYC information
    riskProfile?: any;        // Risk assessment
    investmentGoals?: any;    // Goals and constraints
    fundingSource?: any;      // Initial funding
    regulatoryAgreements?: any; // Terms acceptance
  };
}

export class InvestorPortal extends EventEmitter {
  private marketplace: StrategyMarketplace;
  private portfolios: Map<string, InvestorPortfolio> = new Map();
  private transactions: Map<string, Transaction> = new Map();
  private paymentMethods: Map<string, PaymentMethod[]> = new Map();
  private onboardingFlows: Map<string, OnboardingStep[]> = new Map();
  
  // Payment Processors (would integrate with real services)
  private paymentProcessors = {
    stripe: null,    // Stripe SDK
    plaid: null,     // Plaid SDK
    coinbase: null,  // Coinbase Commerce
    wise: null,      // Wise API
    paypal: null     // PayPal SDK
  };
  
  constructor(marketplace: StrategyMarketplace) {
    super();
    this.marketplace = marketplace;
    this.initialize();
  }
  
  private initialize(): void {
    console.log('🏦 Initializing Investor Portal...');
    
    // Initialize payment processors
    this.initializePaymentProcessors();
    
    // Start portfolio tracking
    this.startPortfolioTracking();
    
    // Start compliance monitoring
    this.startComplianceMonitoring();
  }
  
  private initializePaymentProcessors(): void {
    // In production, initialize real payment SDKs
    console.log('💳 Payment processors initialized (sandbox mode)');
  }
  
  private startPortfolioTracking(): void {
    // Update portfolios every minute
    setInterval(() => {
      this.updateAllPortfolios();
    }, 60000);
  }
  
  private startComplianceMonitoring(): void {
    // Monitor for compliance requirements
    setInterval(() => {
      this.checkComplianceAlerts();
    }, 300000); // Every 5 minutes
  }
  
  /**
   * Onboard a new investor
   */
  async onboardInvestor(email: string, type: string): Promise<string> {
    const investorId = `inv_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Create onboarding flow
    const onboardingSteps: OnboardingStep[] = [
      {
        step: 1,
        name: 'Account Creation',
        status: 'completed',
        data: { email, type, createdAt: new Date() }
      },
      {
        step: 2,
        name: 'KYC Verification',
        status: 'pending',
        data: {}
      },
      {
        step: 3,
        name: 'Risk Assessment',
        status: 'pending',
        data: {}
      },
      {
        step: 4,
        name: 'Investment Goals',
        status: 'pending',
        data: {}
      },
      {
        step: 5,
        name: 'Funding Setup',
        status: 'pending',
        data: {}
      },
      {
        step: 6,
        name: 'Regulatory Agreements',
        status: 'pending',
        data: {}
      }
    ];
    
    this.onboardingFlows.set(investorId, onboardingSteps);
    
    // Create initial portfolio
    const portfolio: InvestorPortfolio = {
      investorId,
      account: {
        totalValue: 0,
        cashBalance: 0,
        investedAmount: 0,
        totalPnL: 0,
        totalPnLPercent: 0,
        dailyPnL: 0,
        dailyPnLPercent: 0
      },
      activeStrategies: [],
      performance: {
        totalReturn: 0,
        annualizedReturn: 0,
        sharpeRatio: 0,
        sortinoRatio: 0,
        maxDrawdown: 0,
        volatility: 0,
        beta: 0,
        alpha: 0
      },
      risk: {
        portfolioVaR: 0,
        portfolioCVaR: 0,
        correlation: 0,
        concentration: 0,
        leverage: 0
      },
      history: {
        dailyReturns: [],
        monthlyReturns: [],
        transactions: [],
        rebalancingHistory: []
      }
    };
    
    this.portfolios.set(investorId, portfolio);
    
    this.emit('investor_onboarded', { investorId, email, type });
    
    console.log(`✅ Investor ${investorId} onboarding started`);
    
    return investorId;
  }
  
  /**
   * Complete KYC verification
   */
  async completeKYC(investorId: string, kycData: any): Promise<boolean> {
    // Simulate KYC verification (would use real service like Jumio, Onfido, etc.)
    const onboarding = this.onboardingFlows.get(investorId);
    if (!onboarding) return false;
    
    const kycStep = onboarding.find(s => s.name === 'KYC Verification');
    if (kycStep) {
      kycStep.status = 'completed';
      kycStep.data.kycData = kycData;
    }
    
    // Create investor profile
    const profile: InvestorProfile = {
      investorId,
      type: kycData.investorType || 'retail',
      goals: {
        primaryGoal: 'growth',
        targetReturn: 0.15,
        riskTolerance: 'moderate',
        investmentHorizon: 36,
        liquidityNeeds: 'quarterly'
      },
      constraints: {
        maxDrawdown: 0.20,
        maxVolatility: 0.25,
        minSharpe: 1.0,
        maxLeverage: 2.0
      },
      allocation: {
        totalCapital: 0,
        maxPerStrategy: 0,
        diversificationRequirement: 3,
        rebalancingFrequency: 'monthly'
      },
      compliance: {
        kycStatus: 'verified',
        amlStatus: 'clear',
        accreditedInvestor: kycData.accredited || false,
        qualifiedPurchaser: kycData.qualified || false,
        jurisdiction: kycData.country || 'US',
        taxResidency: kycData.taxCountry || 'US'
      }
    };
    
    await this.marketplace.registerInvestor(profile);
    
    this.emit('kyc_completed', { investorId, status: 'verified' });
    
    return true;
  }
  
  /**
   * Set investment goals and constraints
   */
  async setInvestmentGoals(
    investorId: string,
    goals: any,
    constraints: any
  ): Promise<void> {
    const onboarding = this.onboardingFlows.get(investorId);
    if (!onboarding) return;
    
    const goalsStep = onboarding.find(s => s.name === 'Investment Goals');
    if (goalsStep) {
      goalsStep.status = 'completed';
      goalsStep.data = { investmentGoals: goals, constraints };
    }
    
    this.emit('goals_set', { investorId, goals, constraints });
  }
  
  /**
   * Add payment method
   */
  async addPaymentMethod(
    investorId: string,
    type: string,
    details: any
  ): Promise<PaymentMethod> {
    const paymentMethod: PaymentMethod = {
      id: `pm_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: type as any,
      provider: this.getProviderForType(type),
      details: {
        ...details,
        isDefault: false,
        isVerified: false,
        addedDate: new Date()
      }
    };
    
    // Verify payment method (would use real payment processor)
    paymentMethod.details.isVerified = true;
    
    // Store payment method
    const methods = this.paymentMethods.get(investorId) || [];
    methods.push(paymentMethod);
    this.paymentMethods.set(investorId, methods);
    
    // Set as default if first method
    if (methods.length === 1) {
      paymentMethod.details.isDefault = true;
    }
    
    this.emit('payment_method_added', { investorId, paymentMethod });
    
    return paymentMethod;
  }
  
  private getProviderForType(type: string): any {
    const providerMap: { [key: string]: any } = {
      'credit_card': 'stripe',
      'debit_card': 'stripe',
      'bank_transfer': 'plaid',
      'wire': 'wise',
      'crypto': 'coinbase',
      'paypal': 'paypal'
    };
    return providerMap[type] || 'stripe';
  }
  
  /**
   * Deposit funds to account
   */
  async depositFunds(
    investorId: string,
    amount: number,
    paymentMethodId: string
  ): Promise<Transaction> {
    const methods = this.paymentMethods.get(investorId) || [];
    const paymentMethod = methods.find(m => m.id === paymentMethodId);
    
    if (!paymentMethod) {
      throw new Error('Payment method not found');
    }
    
    // Create transaction
    const transaction: Transaction = {
      transactionId: `txn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      investorId,
      type: 'deposit',
      amount,
      currency: 'USD',
      status: 'processing',
      paymentMethod,
      details: {
        description: `Deposit ${amount} USD`,
        initiatedAt: new Date()
      },
      compliance: {
        amlChecked: false,
        sanctionsChecked: false,
        limitChecked: false,
        reportingRequired: amount >= 10000 // CTR requirement
      }
    };
    
    this.transactions.set(transaction.transactionId, transaction);
    
    // Simulate payment processing
    setTimeout(() => {
      // Compliance checks
      transaction.compliance.amlChecked = true;
      transaction.compliance.sanctionsChecked = true;
      transaction.compliance.limitChecked = true;
      
      // Complete transaction
      transaction.status = 'completed';
      transaction.details.completedAt = new Date();
      
      // Update portfolio
      const portfolio = this.portfolios.get(investorId);
      if (portfolio) {
        portfolio.account.cashBalance += amount;
        portfolio.account.totalValue += amount;
        portfolio.history.transactions.unshift(transaction);
      }
      
      this.emit('deposit_completed', { investorId, amount, transaction });
    }, 2000);
    
    return transaction;
  }
  
  /**
   * Subscribe to a strategy
   */
  async subscribeToStrategy(
    investorId: string,
    strategyId: string,
    amount: number
  ): Promise<StrategySubscription> {
    const portfolio = this.portfolios.get(investorId);
    if (!portfolio) {
      throw new Error('Portfolio not found');
    }
    
    // Check available balance
    if (portfolio.account.cashBalance < amount) {
      throw new Error('Insufficient funds');
    }
    
    // Subscribe through marketplace
    const subscription = await this.marketplace.subscribeToStrategy(
      investorId,
      strategyId,
      amount
    );
    
    // Update portfolio
    portfolio.account.cashBalance -= amount;
    portfolio.account.investedAmount += amount;
    
    // Add to active strategies
    const strategies = await this.marketplace.getTopStrategies(100);
    const strategy = strategies.find(s => s.strategyId === strategyId);
    
    if (strategy) {
      portfolio.activeStrategies.push({
        strategyId,
        name: strategy.name,
        allocation: amount,
        currentValue: amount,
        pnl: 0,
        pnlPercent: 0,
        weight: (amount / portfolio.account.totalValue) * 100
      });
    }
    
    // Create subscription fee transaction if applicable
    if (strategy && strategy.pricing.subscriptionFee) {
      const feeTransaction: Transaction = {
        transactionId: `txn_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        investorId,
        type: 'subscription_fee',
        amount: strategy.pricing.subscriptionFee,
        currency: 'USD',
        status: 'completed',
        paymentMethod: this.getDefaultPaymentMethod(investorId)!,
        details: {
          strategyId,
          subscriptionId: subscription.subscriptionId,
          description: `Subscription fee for ${strategy.name}`,
          initiatedAt: new Date(),
          completedAt: new Date()
        },
        compliance: {
          amlChecked: true,
          sanctionsChecked: true,
          limitChecked: true,
          reportingRequired: false
        }
      };
      
      this.transactions.set(feeTransaction.transactionId, feeTransaction);
      portfolio.history.transactions.unshift(feeTransaction);
    }
    
    this.emit('strategy_subscribed', {
      investorId,
      strategyId,
      amount,
      subscription
    });
    
    return subscription;
  }
  
  private getDefaultPaymentMethod(investorId: string): PaymentMethod | null {
    const methods = this.paymentMethods.get(investorId) || [];
    return methods.find(m => m.details.isDefault) || methods[0] || null;
  }
  
  /**
   * Rebalance portfolio
   */
  async rebalancePortfolio(investorId: string): Promise<void> {
    const portfolio = this.portfolios.get(investorId);
    if (!portfolio) return;
    
    // Calculate target allocations
    const targetAllocations = this.calculateTargetAllocations(portfolio);
    
    // Execute rebalancing trades
    const changes = new Map<string, number>();
    
    portfolio.activeStrategies.forEach(strategy => {
      const targetAmount = targetAllocations.get(strategy.strategyId) || 0;
      const currentAmount = strategy.currentValue;
      const change = targetAmount - currentAmount;
      
      if (Math.abs(change) > 100) { // Minimum rebalancing threshold
        changes.set(strategy.strategyId, change);
        strategy.allocation = targetAmount;
      }
    });
    
    // Record rebalancing
    portfolio.history.rebalancingHistory.push({
      date: new Date(),
      changes,
      reason: 'Scheduled monthly rebalancing'
    });
    
    this.emit('portfolio_rebalanced', {
      investorId,
      changes: Array.from(changes.entries())
    });
  }
  
  private calculateTargetAllocations(portfolio: InvestorPortfolio): Map<string, number> {
    const allocations = new Map<string, number>();
    const totalValue = portfolio.account.totalValue;
    
    // Simple equal-weight rebalancing
    const targetWeight = 1 / portfolio.activeStrategies.length;
    
    portfolio.activeStrategies.forEach(strategy => {
      allocations.set(strategy.strategyId, totalValue * targetWeight);
    });
    
    return allocations;
  }
  
  /**
   * Update all portfolios
   */
  private updateAllPortfolios(): void {
    this.portfolios.forEach((portfolio, investorId) => {
      this.updatePortfolio(investorId);
    });
  }
  
  /**
   * Update single portfolio
   */
  private updatePortfolio(investorId: string): void {
    const portfolio = this.portfolios.get(investorId);
    if (!portfolio) return;
    
    // Get subscriptions from marketplace
    const subscriptions = this.marketplace.getInvestorSubscriptions(investorId);
    
    // Update strategy values and P&L
    let totalValue = portfolio.account.cashBalance;
    let totalPnL = 0;
    
    portfolio.activeStrategies.forEach(strategy => {
      const subscription = subscriptions.find(s => s.strategyId === strategy.strategyId);
      if (subscription) {
        strategy.currentValue = subscription.details.actualCapital;
        strategy.pnl = subscription.details.pnl;
        strategy.pnlPercent = (subscription.details.pnl / strategy.allocation) * 100;
        
        totalValue += strategy.currentValue;
        totalPnL += strategy.pnl;
      }
    });
    
    // Update account summary
    const previousValue = portfolio.account.totalValue;
    portfolio.account.totalValue = totalValue;
    portfolio.account.investedAmount = totalValue - portfolio.account.cashBalance;
    portfolio.account.totalPnL = totalPnL;
    portfolio.account.totalPnLPercent = (totalPnL / (totalValue - totalPnL)) * 100;
    portfolio.account.dailyPnL = totalValue - previousValue;
    portfolio.account.dailyPnLPercent = (portfolio.account.dailyPnL / previousValue) * 100;
    
    // Update weights
    portfolio.activeStrategies.forEach(strategy => {
      strategy.weight = (strategy.currentValue / totalValue) * 100;
    });
    
    // Calculate portfolio metrics
    this.calculatePortfolioMetrics(portfolio);
    
    // Track daily returns
    portfolio.history.dailyReturns.push(portfolio.account.dailyPnLPercent);
    if (portfolio.history.dailyReturns.length > 30) {
      portfolio.history.dailyReturns.shift();
    }
    
    this.emit('portfolio_updated', {
      investorId,
      portfolio
    });
  }
  
  private calculatePortfolioMetrics(portfolio: InvestorPortfolio): void {
    const returns = portfolio.history.dailyReturns;
    
    if (returns.length > 0) {
      // Calculate basic metrics
      portfolio.performance.totalReturn = portfolio.account.totalPnLPercent / 100;
      portfolio.performance.annualizedReturn = 
        portfolio.performance.totalReturn * (365 / returns.length);
      
      // Calculate volatility
      const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
      const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
      portfolio.performance.volatility = Math.sqrt(variance * 252); // Annualized
      
      // Calculate Sharpe (assuming 2% risk-free rate)
      portfolio.performance.sharpeRatio = 
        (portfolio.performance.annualizedReturn - 0.02) / portfolio.performance.volatility;
      
      // Calculate max drawdown
      let peak = 0;
      let maxDD = 0;
      let cumReturn = 1;
      
      returns.forEach(ret => {
        cumReturn *= (1 + ret / 100);
        if (cumReturn > peak) {
          peak = cumReturn;
        }
        const drawdown = (peak - cumReturn) / peak;
        if (drawdown > maxDD) {
          maxDD = drawdown;
        }
      });
      
      portfolio.performance.maxDrawdown = maxDD;
      
      // Calculate VaR (95% confidence)
      const sortedReturns = [...returns].sort((a, b) => a - b);
      const varIndex = Math.floor(sortedReturns.length * 0.05);
      portfolio.risk.portfolioVaR = sortedReturns[varIndex] || 0;
      
      // Calculate CVaR (average of returns below VaR)
      const tailReturns = sortedReturns.slice(0, varIndex);
      portfolio.risk.portfolioCVaR = 
        tailReturns.reduce((a, b) => a + b, 0) / (tailReturns.length || 1);
      
      // Calculate concentration (HHI)
      const weights = portfolio.activeStrategies.map(s => s.weight / 100);
      portfolio.risk.concentration = weights.reduce((a, b) => a + b * b, 0);
    }
  }
  
  /**
   * Check compliance alerts
   */
  private checkComplianceAlerts(): void {
    // Check for regulatory reporting requirements
    this.transactions.forEach(transaction => {
      if (transaction.compliance.reportingRequired && !transaction.compliance.amlChecked) {
        this.emit('compliance_alert', {
          type: 'CTR_required',
          transaction,
          message: 'Currency Transaction Report required for transactions over $10,000'
        });
      }
    });
    
    // Check for suspicious activity
    this.portfolios.forEach((portfolio, investorId) => {
      // Check for rapid trading
      const recentTransactions = portfolio.history.transactions
        .filter(t => new Date().getTime() - t.details.initiatedAt.getTime() < 86400000);
      
      if (recentTransactions.length > 10) {
        this.emit('compliance_alert', {
          type: 'rapid_trading',
          investorId,
          message: 'Unusual trading activity detected'
        });
      }
    });
  }
  
  /**
   * Get portfolio overview
   */
  getPortfolio(investorId: string): InvestorPortfolio | undefined {
    return this.portfolios.get(investorId);
  }
  
  /**
   * Get transaction history
   */
  getTransactionHistory(investorId: string): Transaction[] {
    return Array.from(this.transactions.values())
      .filter(t => t.investorId === investorId)
      .sort((a, b) => b.details.initiatedAt.getTime() - a.details.initiatedAt.getTime());
  }
  
  /**
   * Get onboarding status
   */
  getOnboardingStatus(investorId: string): OnboardingStep[] | undefined {
    return this.onboardingFlows.get(investorId);
  }
  
  /**
   * Generate performance report
   */
  generatePerformanceReport(investorId: string): any {
    const portfolio = this.portfolios.get(investorId);
    if (!portfolio) return null;
    
    return {
      summary: {
        totalValue: portfolio.account.totalValue,
        totalReturn: portfolio.account.totalPnLPercent,
        sharpeRatio: portfolio.performance.sharpeRatio,
        maxDrawdown: portfolio.performance.maxDrawdown
      },
      strategies: portfolio.activeStrategies.map(s => ({
        name: s.name,
        allocation: s.allocation,
        currentValue: s.currentValue,
        pnl: s.pnl,
        pnlPercent: s.pnlPercent,
        weight: s.weight
      })),
      risk: {
        portfolioVaR: portfolio.risk.portfolioVaR,
        concentration: portfolio.risk.concentration,
        leverage: portfolio.risk.leverage
      },
      transactions: portfolio.history.transactions.slice(0, 10),
      generatedAt: new Date()
    };
  }
}

// Export for use in other modules
export type { InvestorPortfolio, Transaction, PaymentMethod, OnboardingStep };