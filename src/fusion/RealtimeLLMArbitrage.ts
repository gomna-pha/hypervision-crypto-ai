import { EventEmitter } from 'events';
import Anthropic from '@anthropic-ai/sdk';
import Logger from '../utils/logger';
import { RealTimeDataAggregator } from '../agents/RealTimeDataAggregator';
import { DecisionEngine } from '../decision/DecisionEngine';

interface ArbitrageStrategy {
  id: string;
  type: 'direct' | 'triangular' | 'futures-spot' | 'multi-exchange';
  pair: string;
  exchanges: string[];
  entryConditions: {
    minSpread: number;
    maxSlippage: number;
    minLiquidity: number;
  };
  exitConditions: {
    targetProfit: number;
    stopLoss: number;
    maxHoldTime: number;
  };
  sizing: {
    notional: number;
    leverage: number;
    riskPercent: number;
  };
  confidence: number;
  expectedReturn: number;
  riskScore: number;
  timestamp: number;
}

interface LLMPrediction {
  strategies: ArbitrageStrategy[];
  marketRegime: 'trending' | 'ranging' | 'volatile' | 'calm';
  riskLevel: 'low' | 'medium' | 'high' | 'extreme';
  recommendations: string[];
  confidence: number;
  reasoning: string;
}

export class RealtimeLLMArbitrage extends EventEmitter {
  private logger: Logger;
  private dataAggregator: RealTimeDataAggregator;
  private decisionEngine: DecisionEngine;
  private anthropic: Anthropic | null = null;
  
  private isRunning: boolean = false;
  private predictionInterval: NodeJS.Timeout | null = null;
  private lastPrediction: LLMPrediction | null = null;
  private activeStrategies: Map<string, ArbitrageStrategy> = new Map();
  
  // Performance tracking
  private predictions: LLMPrediction[] = [];
  private executedTrades: number = 0;
  private successfulTrades: number = 0;
  private totalPnL: number = 0;

  constructor() {
    super();
    this.logger = Logger.getInstance('RealtimeLLMArbitrage');
    this.dataAggregator = new RealTimeDataAggregator();
    this.decisionEngine = new DecisionEngine();
    
    // Initialize Anthropic if API key available
    const apiKey = process.env.ANTHROPIC_API_KEY;
    if (apiKey) {
      this.anthropic = new Anthropic({ apiKey });
      this.logger.info('Anthropic Claude initialized');
    } else {
      this.logger.warn('No Anthropic API key, using mock predictions');
    }
    
    this.setupEventListeners();
  }

  private setupEventListeners(): void {
    // Listen to real-time data updates
    this.dataAggregator.on('arbitrage_opportunity', (opportunity) => {
      this.handleNewOpportunity(opportunity);
    });
    
    this.dataAggregator.on('agent_signal', (signal) => {
      this.processAgentSignal(signal);
    });
    
    // Listen to decision engine
    this.decisionEngine.on('decision', (decision) => {
      this.handleDecision(decision);
    });
  }

  async start(): Promise<void> {
    if (this.isRunning) {
      this.logger.warn('Already running');
      return;
    }
    
    this.isRunning = true;
    
    // Start data aggregator
    await this.dataAggregator.start();
    
    // Start prediction loop
    this.startPredictionLoop();
    
    this.logger.info('Realtime LLM Arbitrage started');
    this.emit('status', { running: true });
  }

  private startPredictionLoop(): void {
    const predict = async () => {
      if (!this.isRunning) return;
      
      try {
        const prediction = await this.generatePrediction();
        this.processPrediction(prediction);
      } catch (error) {
        this.logger.error('Prediction error', error);
      }
    };
    
    // Initial prediction
    predict();
    
    // Regular predictions every 5 seconds
    this.predictionInterval = setInterval(predict, 5000);
  }

  private async generatePrediction(): Promise<LLMPrediction> {
    // Gather all real-time data
    const marketData = this.dataAggregator.getMarketData();
    const economicData = this.dataAggregator.getEconomicData();
    const sentimentData = this.dataAggregator.getSentimentData();
    const microstructure = this.dataAggregator.getMicrostructure();
    const opportunities = this.dataAggregator.getOpportunities();
    const agentSignals = this.dataAggregator.getAgentSignals();
    
    // Build context for LLM
    const context = {
      timestamp: new Date().toISOString(),
      market: {
        btc: marketData.get('binance-btcusdt'),
        eth: marketData.get('binance-ethusdt'),
        sol: marketData.get('binance-solusdt')
      },
      economic: economicData,
      sentiment: sentimentData,
      microstructure: Array.from(microstructure.entries()).slice(0, 3),
      opportunities: opportunities.slice(0, 5),
      agentSignals,
      recentPerformance: {
        trades: this.executedTrades,
        winRate: this.executedTrades > 0 ? this.successfulTrades / this.executedTrades : 0,
        pnl: this.totalPnL
      }
    };
    
    if (this.anthropic) {
      return await this.callLLM(context);
    } else {
      return this.generateMockPrediction(context);
    }
  }

  private async callLLM(context: any): Promise<LLMPrediction> {
    const prompt = this.buildPrompt(context);
    
    try {
      const response = await this.anthropic!.messages.create({
        model: 'claude-3-sonnet-20240229',
        max_tokens: 2000,
        temperature: 0.1,
        system: `You are an advanced arbitrage strategy generator. Analyze real-time market data from multiple agents and generate specific, actionable arbitrage strategies. Return only valid JSON matching the LLMPrediction schema.`,
        messages: [{ role: 'user', content: prompt }]
      });
      
      const content = response.content[0];
      if (content.type === 'text') {
        return JSON.parse(content.text);
      }
      
      throw new Error('Invalid LLM response');
    } catch (error) {
      this.logger.error('LLM call failed', error);
      return this.generateMockPrediction(context);
    }
  }

  private buildPrompt(context: any): string {
    return `Analyze the following real-time market data and generate arbitrage strategies:

MARKET DATA:
${JSON.stringify(context.market, null, 2)}

ECONOMIC INDICATORS:
${JSON.stringify(context.economic, null, 2)}

SENTIMENT DATA:
${JSON.stringify(context.sentiment, null, 2)}

AGENT SIGNALS:
- Economic: ${context.agentSignals.economic.signal.toFixed(3)} (${(context.agentSignals.economic.confidence * 100).toFixed(0)}% confidence)
- Sentiment: ${context.agentSignals.sentiment.signal.toFixed(3)} (${(context.agentSignals.sentiment.confidence * 100).toFixed(0)}% confidence)
- Microstructure: ${context.agentSignals.microstructure.signal.toFixed(3)} (${(context.agentSignals.microstructure.confidence * 100).toFixed(0)}% confidence)
- Cross-Exchange: ${context.agentSignals.crossExchange.signal.toFixed(3)} (${(context.agentSignals.crossExchange.confidence * 100).toFixed(0)}% confidence)

CURRENT OPPORTUNITIES:
${JSON.stringify(context.opportunities, null, 2)}

RECENT PERFORMANCE:
- Executed Trades: ${context.recentPerformance.trades}
- Win Rate: ${(context.recentPerformance.winRate * 100).toFixed(1)}%
- P&L: $${context.recentPerformance.pnl.toFixed(2)}

Generate specific arbitrage strategies with entry/exit conditions, sizing, and risk parameters. Consider the agent signals and current market regime. Return as JSON with structure: { strategies: [], marketRegime: "", riskLevel: "", recommendations: [], confidence: 0, reasoning: "" }`;
  }

  private generateMockPrediction(context: any): LLMPrediction {
    const opportunities = context.opportunities || [];
    const strategies: ArbitrageStrategy[] = [];
    
    // Generate strategies based on opportunities
    for (const opp of opportunities.slice(0, 3)) {
      strategies.push({
        id: `strat-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        type: 'direct',
        pair: opp.pair,
        exchanges: [opp.buyExchange, opp.sellExchange],
        entryConditions: {
          minSpread: opp.spreadPercent * 0.8,
          maxSlippage: 0.002,
          minLiquidity: 100000
        },
        exitConditions: {
          targetProfit: opp.estimatedProfit * 0.8,
          stopLoss: -opp.estimatedProfit * 0.5,
          maxHoldTime: 300 // 5 minutes
        },
        sizing: {
          notional: Math.min(100000, opp.estimatedProfit * 1000),
          leverage: 1,
          riskPercent: 1
        },
        confidence: opp.confidence,
        expectedReturn: opp.estimatedProfit,
        riskScore: opp.riskScore,
        timestamp: Date.now()
      });
    }
    
    // Add a triangular arbitrage strategy
    if (Math.random() > 0.7) {
      strategies.push({
        id: `tri-${Date.now()}`,
        type: 'triangular',
        pair: 'BTC-ETH-USDT',
        exchanges: ['binance'],
        entryConditions: {
          minSpread: 0.15,
          maxSlippage: 0.001,
          minLiquidity: 500000
        },
        exitConditions: {
          targetProfit: 50,
          stopLoss: -25,
          maxHoldTime: 60
        },
        sizing: {
          notional: 50000,
          leverage: 1,
          riskPercent: 0.5
        },
        confidence: 0.72,
        expectedReturn: 45,
        riskScore: 0.35,
        timestamp: Date.now()
      });
    }
    
    // Determine market regime
    const volatility = context.economic?.vix || 20;
    let marketRegime: 'trending' | 'ranging' | 'volatile' | 'calm';
    if (volatility > 30) marketRegime = 'volatile';
    else if (volatility > 20) marketRegime = 'trending';
    else if (volatility > 15) marketRegime = 'ranging';
    else marketRegime = 'calm';
    
    // Determine risk level
    let riskLevel: 'low' | 'medium' | 'high' | 'extreme';
    if (volatility > 35) riskLevel = 'extreme';
    else if (volatility > 25) riskLevel = 'high';
    else if (volatility > 18) riskLevel = 'medium';
    else riskLevel = 'low';
    
    // Generate recommendations
    const recommendations = [
      strategies.length > 0 ? `Execute ${strategies.length} identified strategies` : 'Wait for better opportunities',
      marketRegime === 'volatile' ? 'Reduce position sizes due to volatility' : 'Normal position sizing appropriate',
      context.agentSignals.sentiment.signal < -0.3 ? 'Market sentiment negative, be cautious' : 'Sentiment supportive of arbitrage',
      context.agentSignals.crossExchange.signal > 0.7 ? 'Strong cross-exchange opportunities present' : 'Limited cross-exchange spreads'
    ];
    
    return {
      strategies,
      marketRegime,
      riskLevel,
      recommendations,
      confidence: 0.75 + Math.random() * 0.15,
      reasoning: `Generated ${strategies.length} strategies based on ${opportunities.length} opportunities. Market regime is ${marketRegime} with ${riskLevel} risk. Agent consensus indicates ${context.agentSignals.crossExchange.signal > 0.5 ? 'favorable' : 'challenging'} conditions.`
    };
  }

  private processPrediction(prediction: LLMPrediction): void {
    this.lastPrediction = prediction;
    this.predictions.push(prediction);
    
    // Keep only last 100 predictions
    if (this.predictions.length > 100) {
      this.predictions.shift();
    }
    
    // Process each strategy
    for (const strategy of prediction.strategies) {
      this.evaluateStrategy(strategy);
    }
    
    // Emit prediction for dashboard
    this.emit('prediction', {
      ...prediction,
      timestamp: Date.now(),
      activeStrategies: this.activeStrategies.size
    });
    
    this.logger.info('LLM prediction generated', {
      strategies: prediction.strategies.length,
      regime: prediction.marketRegime,
      risk: prediction.riskLevel,
      confidence: prediction.confidence
    });
  }

  private evaluateStrategy(strategy: ArbitrageStrategy): void {
    // Check if strategy meets criteria
    const meetsThreshold = strategy.confidence >= 0.7 && strategy.riskScore < 0.5;
    
    if (meetsThreshold) {
      this.activeStrategies.set(strategy.id, strategy);
      
      // Send to decision engine
      this.decisionEngine.processDecision(
        {
          predicted_spread_pct: strategy.entryConditions.minSpread,
          confidence: strategy.confidence,
          direction: 'converge',
          expected_time_s: strategy.exitConditions.maxHoldTime,
          arbitrage_plan: {
            buy: strategy.exchanges[0],
            sell: strategy.exchanges[1] || strategy.exchanges[0],
            notional_usd: strategy.sizing.notional
          },
          rationale: `Strategy ${strategy.id}: ${strategy.type} arbitrage`,
          risk_flags: strategy.riskScore > 0.3 ? ['elevated_risk'] : []
        },
        new Map() // Agent data would be passed here
      );
      
      // Simulate execution tracking
      setTimeout(() => {
        this.trackExecution(strategy);
      }, strategy.exitConditions.maxHoldTime * 1000);
    }
  }

  private trackExecution(strategy: ArbitrageStrategy): void {
    this.executedTrades++;
    
    // Simulate success based on confidence
    const success = Math.random() < strategy.confidence;
    if (success) {
      this.successfulTrades++;
      this.totalPnL += strategy.expectedReturn;
    } else {
      this.totalPnL -= strategy.expectedReturn * 0.5;
    }
    
    this.activeStrategies.delete(strategy.id);
    
    this.emit('execution_complete', {
      strategyId: strategy.id,
      success,
      pnl: success ? strategy.expectedReturn : -strategy.expectedReturn * 0.5
    });
  }

  private handleNewOpportunity(opportunity: any): void {
    // Quick evaluation of new opportunities
    if (opportunity.confidence > 0.8 && opportunity.riskScore < 0.3) {
      this.emit('high_confidence_opportunity', opportunity);
    }
  }

  private processAgentSignal(signal: any): void {
    // Track agent signals for dashboard
    this.emit('agent_signal_update', signal);
  }

  private handleDecision(decision: any): void {
    // Track decision engine outcomes
    this.emit('decision_made', decision);
  }

  // ============= API Methods =============

  getStatus(): any {
    return {
      running: this.isRunning,
      lastPrediction: this.lastPrediction,
      activeStrategies: Array.from(this.activeStrategies.values()),
      performance: {
        executedTrades: this.executedTrades,
        successfulTrades: this.successfulTrades,
        winRate: this.executedTrades > 0 ? (this.successfulTrades / this.executedTrades) * 100 : 0,
        totalPnL: this.totalPnL
      },
      agentSignals: this.dataAggregator.getAgentSignals(),
      opportunities: this.dataAggregator.getOpportunities()
    };
  }

  getRealtimeData(): any {
    return {
      market: Array.from(this.dataAggregator.getMarketData().entries()),
      economic: this.dataAggregator.getEconomicData(),
      sentiment: this.dataAggregator.getSentimentData(),
      microstructure: Array.from(this.dataAggregator.getMicrostructure().entries()),
      spreads: Array.from(this.dataAggregator.getCrossExchangeSpreads().values()),
      opportunities: this.dataAggregator.getOpportunities()
    };
  }

  stop(): void {
    this.isRunning = false;
    
    if (this.predictionInterval) {
      clearInterval(this.predictionInterval);
    }
    
    this.dataAggregator.stop();
    
    this.logger.info('Realtime LLM Arbitrage stopped');
    this.emit('status', { running: false });
  }
}

export default RealtimeLLMArbitrage;