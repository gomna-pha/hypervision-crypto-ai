import Anthropic from '@anthropic-ai/sdk';
import Logger from '../utils/logger';
import { InstitutionalDataAggregator } from '../agents/InstitutionalDataAggregator';

const logger = Logger.getInstance('InstitutionalLLMStrategy');

interface TradingStrategy {
  id: string;
  type: 'Statistical Arbitrage' | 'Market Making' | 'Momentum' | 'Mean Reversion' | 'Volatility Arbitrage';
  confidence: number;
  expectedReturn: number;
  riskAdjustedReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  timeHorizon: string;
  capitalAllocation: number;
  entryConditions: string[];
  exitConditions: string[];
  riskManagement: {
    stopLoss: number;
    takeProfit: number;
    positionSize: number;
    maxLeverage: number;
  };
  marketRegime: 'Trending' | 'Ranging' | 'Volatile' | 'Calm';
  executionPlan: string[];
}

interface MarketRegimeAnalysis {
  regime: string;
  confidence: number;
  indicators: {
    trendStrength: number;
    volatility: number;
    correlation: number;
    liquidity: number;
  };
  recommendation: string;
}

export class InstitutionalLLMStrategy {
  private anthropic: Anthropic | null = null;
  private dataAggregator: InstitutionalDataAggregator;
  private currentStrategies: TradingStrategy[] = [];
  private marketRegime: MarketRegimeAnalysis | null = null;
  private lastUpdate: number = 0;
  private isRunning: boolean = false;

  constructor() {
    this.dataAggregator = new InstitutionalDataAggregator();
    
    // Initialize Anthropic if API key available
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (apiKey) {
      this.anthropic = new Anthropic({ apiKey });
      logger.info('Anthropic Claude initialized for institutional strategies');
    } else {
      logger.warn('No Anthropic API key, using sophisticated mock strategies');
    }
  }

  async start(): Promise<void> {
    if (this.isRunning) return;
    this.isRunning = true;
    
    logger.info('Starting Institutional LLM Strategy Generator...');
    
    // Start data aggregator
    await this.dataAggregator.start();
    
    // Generate strategies every 5 seconds
    this.generateStrategies();
    setInterval(() => this.generateStrategies(), 5000);
    
    // Analyze market regime every 30 seconds
    this.analyzeMarketRegime();
    setInterval(() => this.analyzeMarketRegime(), 30000);
  }

  private async generateStrategies(): Promise<void> {
    const startTime = Date.now();
    
    try {
      const microstructure = this.dataAggregator.getMicrostructure();
      const opportunities = this.dataAggregator.getArbitrageOpportunities();
      const metrics = this.dataAggregator.getInstitutionalMetrics();
      const marketData = this.dataAggregator.getMarketData();
      
      if (this.anthropic) {
        // Real LLM strategy generation
        const strategies = await this.generateLLMStrategies(
          microstructure,
          opportunities,
          metrics,
          marketData
        );
        this.currentStrategies = strategies;
      } else {
        // Sophisticated mock strategies for demo
        this.currentStrategies = this.generateMockInstitutionalStrategies(
          microstructure,
          opportunities,
          metrics
        );
      }
      
      const processingTime = Date.now() - startTime;
      logger.info(`Generated ${this.currentStrategies.length} institutional strategies in ${processingTime}ms`);
      
      this.lastUpdate = Date.now();
    } catch (error) {
      logger.error('Failed to generate strategies', error);
    }
  }

  private async generateLLMStrategies(
    microstructure: any,
    opportunities: any[],
    metrics: any,
    marketData: Map<string, any>
  ): Promise<TradingStrategy[]> {
    if (!this.anthropic) return [];
    
    const prompt = `You are an institutional-grade algorithmic trading system managing $${(metrics.aum / 1000000).toFixed(1)}M AUM.

Current Market Microstructure:
- Bid-Ask Spread: ${microstructure?.bidAskSpreadBps?.toFixed(2)} bps
- Order Book Imbalance: ${(microstructure?.orderBookImbalance * 100)?.toFixed(2)}%
- Liquidity Score: ${microstructure?.liquidityScore?.toFixed(1)}/100
- Toxicity Score: ${microstructure?.toxicityScore?.toFixed(1)}/100
- VPIN: ${microstructure?.vpin?.toFixed(4)}
- Market Impact (Kyle's Lambda): ${microstructure?.marketImpact?.toFixed(6)}

Top Arbitrage Opportunities:
${opportunities.slice(0, 3).map(opp => 
  `- ${opp.symbol}: ${opp.spreadPercent.toFixed(3)}% spread, Sharpe ${opp.sharpeRatio.toFixed(2)}, Volume $${(opp.volumeAvailable * opp.spreadUsd).toFixed(0)}`
).join('\n')}

Portfolio Metrics:
- YTD Return: ${(metrics.ytdReturn * 100).toFixed(2)}%
- Sharpe Ratio: 2.15
- Sortino Ratio: ${metrics.sortinoRatio.toFixed(2)}
- Max Drawdown: ${(metrics.maxDrawdown * 100).toFixed(2)}%
- Kelly Fraction: ${(metrics.kellyFraction * 100).toFixed(1)}%

Generate 3 institutional-grade trading strategies with the following requirements:
1. Each strategy must have expected Sharpe > 2.0
2. Maximum drawdown < 10%
3. Clear entry/exit conditions
4. Risk management parameters
5. Capital allocation based on Kelly Criterion

Return as JSON array of TradingStrategy objects.`;

    try {
      const response = await this.anthropic.messages.create({
        model: 'claude-3-opus-20240229',
        max_tokens: 2000,
        temperature: 0.3,
        system: 'You are an elite quantitative trading strategist at a top-tier hedge fund.',
        messages: [{ role: 'user', content: prompt }]
      });
      
      const content = response.content[0];
      if (content && 'text' in content) {
        const strategies = JSON.parse(content.text);
        return Array.isArray(strategies) ? strategies : [strategies];
      }
    } catch (error) {
      logger.error('LLM strategy generation failed', error);
    }
    
    return [];
  }

  private generateMockInstitutionalStrategies(
    microstructure: any,
    opportunities: any[],
    metrics: any
  ): TradingStrategy[] {
    const strategies: TradingStrategy[] = [];
    
    // Strategy 1: Statistical Arbitrage
    if (opportunities.length > 0 && opportunities[0].sharpeRatio > 2) {
      strategies.push({
        id: `STAT-ARB-${Date.now()}`,
        type: 'Statistical Arbitrage',
        confidence: 0.92,
        expectedReturn: 0.00145, // 14.5 bps
        riskAdjustedReturn: 0.00128,
        sharpeRatio: 2.43,
        maxDrawdown: 0.045, // 4.5%
        timeHorizon: '5-30 minutes',
        capitalAllocation: metrics.aum * 0.25,
        entryConditions: [
          `Cross-exchange spread > ${opportunities[0].spreadPercent.toFixed(3)}%`,
          'Order book imbalance < 20%',
          'Liquidity score > 70',
          'VPIN < 0.3',
          'Both exchanges showing stable connection'
        ],
        exitConditions: [
          'Spread converges to < 0.05%',
          'Position held for > 30 minutes',
          'Toxicity score > 80',
          'Realized P&L > 10 bps',
          'Stop loss triggered at -5 bps'
        ],
        riskManagement: {
          stopLoss: 0.0005, // 5 bps
          takeProfit: 0.0015, // 15 bps
          positionSize: metrics.kellyFraction * 0.5,
          maxLeverage: 2.0
        },
        marketRegime: 'Ranging',
        executionPlan: [
          'Split order into 10 child orders',
          'Use TWAP over 30 seconds',
          'Route 60% to primary exchange',
          'Reserve 40% for opportunistic fills',
          'Monitor slippage in real-time'
        ]
      });
    }
    
    // Strategy 2: Market Making with Inventory Management
    if (microstructure && microstructure.liquidityScore > 60) {
      const halfSpread = (microstructure.bidAskSpreadBps / 2) / 10000;
      strategies.push({
        id: `MM-INV-${Date.now()}`,
        type: 'Market Making',
        confidence: 0.88,
        expectedReturn: 0.00082, // 8.2 bps
        riskAdjustedReturn: 0.00071,
        sharpeRatio: 2.15,
        maxDrawdown: 0.062,
        timeHorizon: '1-5 minutes',
        capitalAllocation: metrics.aum * 0.20,
        entryConditions: [
          `Bid-ask spread > ${microstructure.bidAskSpreadBps.toFixed(1)} bps`,
          'Order flow toxicity < 40',
          'Inventory skew < 30%',
          'Market depth > $500k each side',
          'No major news events expected'
        ],
        exitConditions: [
          'Inventory imbalance > 40%',
          'Adverse selection detected',
          'Spread compression < 3 bps',
          'Daily P&L target reached',
          'Risk limit breached'
        ],
        riskManagement: {
          stopLoss: 0.0008,
          takeProfit: 0.0012,
          positionSize: 0.15,
          maxLeverage: 1.5
        },
        marketRegime: 'Calm',
        executionPlan: [
          'Post limit orders at mid ± half spread',
          'Adjust quotes based on inventory',
          'Skew prices to reduce position',
          'Cancel and replace every 500ms',
          'Use maker rebates to enhance returns'
        ]
      });
    }
    
    // Strategy 3: Volatility Arbitrage using Options
    strategies.push({
      id: `VOL-ARB-${Date.now()}`,
      type: 'Volatility Arbitrage',
      confidence: 0.85,
      expectedReturn: 0.00198, // 19.8 bps
      riskAdjustedReturn: 0.00156,
      sharpeRatio: 2.31,
      maxDrawdown: 0.078,
      timeHorizon: '1-3 days',
      capitalAllocation: metrics.aum * 0.15,
      entryConditions: [
        'Implied volatility > realized volatility by 15%',
        'Term structure shows backwardation',
        'Options volume > 10,000 contracts',
        'Delta-neutral hedge available',
        'Funding rate favorable'
      ],
      exitConditions: [
        'IV/RV spread < 5%',
        'Position theta decay > $5,000/day',
        'Gamma risk exceeds limits',
        'Volatility regime change detected',
        'P&L target of 20 bps achieved'
      ],
      riskManagement: {
        stopLoss: 0.0015,
        takeProfit: 0.0025,
        positionSize: 0.10,
        maxLeverage: 3.0
      },
      marketRegime: 'Volatile',
      executionPlan: [
        'Sell ATM straddle',
        'Delta hedge with perpetual futures',
        'Rebalance hedge every hour',
        'Monitor Greeks continuously',
        'Adjust for pin risk near expiry'
      ]
    });
    
    // Strategy 4: Cross-Asset Momentum
    const marketData = this.dataAggregator.getMarketData();
    if (marketData && marketData.size > 3) {
      strategies.push({
        id: `CROSS-MOM-${Date.now()}`,
        type: 'Momentum',
        confidence: 0.79,
        expectedReturn: 0.00234,
        riskAdjustedReturn: 0.00178,
        sharpeRatio: 2.08,
        maxDrawdown: 0.095,
        timeHorizon: '4-12 hours',
        capitalAllocation: metrics.aum * 0.10,
        entryConditions: [
          'Price breakout above 20-period high',
          'Volume surge > 2x average',
          'RSI > 70 but < 85',
          'Correlation breakdown detected',
          'Institutional flow detected'
        ],
        exitConditions: [
          'Momentum divergence observed',
          'Volume drops below average',
          'Support level breached',
          'Correlation returns to normal',
          'Trailing stop hit at -8 bps'
        ],
        riskManagement: {
          stopLoss: 0.0008,
          takeProfit: 0.0035,
          positionSize: 0.08,
          maxLeverage: 2.5
        },
        marketRegime: 'Trending',
        executionPlan: [
          'Scale in with 3 entries',
          'Use momentum confirmation',
          'Trail stop at 1.5 ATR',
          'Take partial profits at +15 bps',
          'Full exit on signal reversal'
        ]
      });
    }
    
    return strategies;
  }

  private async analyzeMarketRegime(): Promise<void> {
    const microstructure = this.dataAggregator.getMicrostructure();
    const marketData = this.dataAggregator.getMarketData();
    
    if (!microstructure) return;
    
    // Calculate regime indicators
    const volatility = this.calculateVolatility(marketData);
    const trendStrength = this.calculateTrendStrength(marketData);
    const correlation = this.calculateCorrelation(marketData);
    const liquidity = microstructure.liquidityScore / 100;
    
    // Determine market regime
    let regime: 'Trending' | 'Ranging' | 'Volatile' | 'Calm';
    let confidence = 0;
    
    if (volatility > 0.025 && trendStrength < 0.3) {
      regime = 'Volatile';
      confidence = 0.85;
    } else if (trendStrength > 0.6) {
      regime = 'Trending';
      confidence = 0.88;
    } else if (volatility < 0.015 && trendStrength < 0.3) {
      regime = 'Calm';
      confidence = 0.82;
    } else {
      regime = 'Ranging';
      confidence = 0.75;
    }
    
    this.marketRegime = {
      regime,
      confidence,
      indicators: {
        trendStrength,
        volatility,
        correlation,
        liquidity
      },
      recommendation: this.getRegimeRecommendation(regime)
    };
  }

  private calculateVolatility(marketData: Map<string, any>): number {
    // Calculate realized volatility from price data
    const prices: number[] = [];
    marketData.forEach(data => {
      if (data.binance?.price) prices.push(data.binance.price);
    });
    
    if (prices.length < 2) return 0.02; // Default 2% volatility
    
    const returns = prices.slice(1).map((p, i) => Math.log(p / prices[i]));
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / returns.length;
    
    return Math.sqrt(variance * 365 * 24 * 60 * 60); // Annualized
  }

  private calculateTrendStrength(marketData: Map<string, any>): number {
    // Simple trend strength calculation
    const prices: number[] = [];
    marketData.forEach(data => {
      if (data.binance?.price) prices.push(data.binance.price);
    });
    
    if (prices.length < 3) return 0.5;
    
    const ma = prices.reduce((a, b) => a + b, 0) / prices.length;
    const trend = (prices[prices.length - 1] - ma) / ma;
    
    return Math.min(Math.abs(trend) * 10, 1);
  }

  private calculateCorrelation(marketData: Map<string, any>): number {
    // Calculate average correlation between assets
    // Simplified for demo
    return 0.65 + (Math.random() - 0.5) * 0.2;
  }

  private getRegimeRecommendation(regime: string): string {
    const recommendations: Record<string, string> = {
      'Trending': 'Deploy momentum strategies, increase position sizes on winners',
      'Ranging': 'Focus on mean reversion and market making strategies',
      'Volatile': 'Reduce leverage, focus on volatility arbitrage and hedged positions',
      'Calm': 'Increase market making, sell volatility, optimize for carry trades'
    };
    
    return recommendations[regime] || 'Maintain balanced portfolio allocation';
  }

  // Public getters
  getCurrentStrategies(): TradingStrategy[] {
    return this.currentStrategies;
  }

  getMarketRegime(): MarketRegimeAnalysis | null {
    return this.marketRegime;
  }

  getDataAggregator(): InstitutionalDataAggregator {
    return this.dataAggregator;
  }

  stop(): void {
    this.isRunning = false;
    this.dataAggregator.stop();
  }
}

export default InstitutionalLLMStrategy;