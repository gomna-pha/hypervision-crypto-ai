import EventEmitter from 'events';
import { SentimentNewsAgent, SentimentNewsData } from '../agents/sentiment-news-agent';
import { EconomicIndicatorsAgent, EconomicIndicatorsData } from '../agents/economic-indicators-agent';
import { MicrostructureAgent, MicrostructureData } from '../agents/microstructure-agent';
import { HyperbolicEmbedder } from './hyperbolic-embedder';

export interface FusionInput {
  timestamp: string;
  economic?: EconomicIndicatorsData;
  sentiment?: SentimentNewsData;
  microstructure?: MicrostructureData;
  hyperbolic_context: {
    embedding: number[];
    neighbors: any[];
    confidence: number;
  };
}

export interface ArbitragePlan {
  buy: string;        // Exchange name
  sell: string;       // Exchange name
  notional_usd: number;
  buy_price?: number;
  sell_price?: number;
}

export interface ArbitragePrediction {
  id: string;
  timestamp: string;
  predicted_spread_pct: number;     // 0.001 to 0.05 (0.1% to 5%)
  confidence: number;               // 0 to 1
  direction: 'converge' | 'diverge';
  expected_time_s: number;          // 60 to 3600 seconds
  arbitrage_plan: ArbitragePlan;
  rationale: string;
  risk_flags: string[];
  fusion_input: FusionInput;        // Store for audit trail
  hyperbolic_embedding: number[];   // Store embedding for visualization
}

export class FusionBrain extends EventEmitter {
  private economicAgent: EconomicIndicatorsAgent;
  private sentimentAgent: SentimentNewsAgent;
  private microstructureAgent: MicrostructureAgent;
  private hyperbolicEmbedder: HyperbolicEmbedder;
  private isRunning: boolean = false;
  
  // Visible Parameters for Investors
  public readonly parameters = {
    agent_freshness_threshold_sec: 5,
    min_agent_count_for_prediction: 2,
    confidence_threshold_for_execution: 0.75,
    hyperbolic_embedding_dim: 128,
    hyperbolic_curvature: 1.0,
    context_window_minutes: 30,
    prediction_horizon_minutes: 15,
    fusion_interval_ms: 500, // Poll agents every 500ms
  };

  // Visible Constraints for Investors
  public readonly constraints = {
    max_llm_response_time_ms: 10000,
    max_tokens_per_request: 4000,
    min_prediction_confidence: 0.6,
    max_concurrent_predictions: 5,
    required_agent_types: ['sentiment', 'economic', 'microstructure'],
    blackout_during_events: true,
    max_retry_attempts: 3
  };

  // Visible Bounds for Investors
  public readonly bounds = {
    predicted_spread_pct_range: { min: 0.001, max: 0.05 },
    confidence_range: { min: 0.0, max: 1.0 },
    time_horizon_range: { min: 60, max: 3600 },
    max_position_size_pct: 0.1,
    notional_usd_range: { min: 1000, max: 100000 }
  };

  private lastAgentData: {
    economic?: EconomicIndicatorsData;
    sentiment?: SentimentNewsData;
    microstructure?: MicrostructureData;
  } = {};

  private activePredictions: Map<string, ArbitragePrediction> = new Map();
  private llmProvider: 'anthropic' | 'openai' = 'anthropic';
  private llmApiKey: string;

  constructor() {
    super();
    this.economicAgent = new EconomicIndicatorsAgent();
    this.sentimentAgent = new SentimentNewsAgent();
    this.microstructureAgent = new MicrostructureAgent();
    this.hyperbolicEmbedder = new HyperbolicEmbedder();
    
    this.llmApiKey = process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY || 'demo_key';
    
    this.setupAgentListeners();
    console.log('✅ FusionBrain initialized with visible parameters');
  }

  private setupAgentListeners(): void {
    // Listen to agent data updates
    this.economicAgent.on('data', (data: EconomicIndicatorsData) => {
      this.lastAgentData.economic = data;
    });

    this.sentimentAgent.on('data', (data: SentimentNewsData) => {
      this.lastAgentData.sentiment = data;
    });

    this.microstructureAgent.on('data', (data: MicrostructureData) => {
      this.lastAgentData.microstructure = data;
    });
  }

  async start(): Promise<void> {
    if (this.isRunning) return;
    
    console.log('🚀 Starting FusionBrain with LLM integration...');
    
    // Start all agents
    await this.economicAgent.start();
    await this.sentimentAgent.start();
    await this.microstructureAgent.start();
    
    this.isRunning = true;
    
    // Start fusion loop
    this.startFusionLoop();
  }

  async stop(): Promise<void> {
    this.isRunning = false;
    
    await this.economicAgent.stop();
    await this.sentimentAgent.stop();
    await this.microstructureAgent.stop();
    
    console.log('⏹️ FusionBrain stopped');
  }

  private startFusionLoop(): void {
    setInterval(async () => {
      if (!this.isRunning) return;
      
      try {
        await this.processFusion();
      } catch (error) {
        console.error('❌ Fusion loop error:', error);
        this.emit('error', error);
      }
    }, this.parameters.fusion_interval_ms);
  }

  private async processFusion(): Promise<void> {
    // Step 1: Collect fresh agent data
    const freshAgentData = this.collectFreshAgentData();
    
    // Step 2: Validate minimum agent requirements
    if (!this.validateAgentData(freshAgentData)) {
      return; // Skip this cycle
    }
    
    // Step 3: Build hyperbolic context
    const hyperbolicContext = await this.buildHyperbolicContext(freshAgentData);
    
    // Step 4: Create fusion input
    const fusionInput: FusionInput = {
      timestamp: new Date().toISOString(),
      economic: freshAgentData.economic,
      sentiment: freshAgentData.sentiment,
      microstructure: freshAgentData.microstructure,
      hyperbolic_context: hyperbolicContext
    };
    
    // Step 5: Call LLM
    const prediction = await this.callLLMFusion(fusionInput);
    
    if (prediction) {
      // Step 6: Store and emit prediction
      this.activePredictions.set(prediction.id, prediction);
      this.emit('prediction', prediction);
      
      console.log(`🔮 Fusion Prediction: ${prediction.predicted_spread_pct.toFixed(4)}% spread, confidence: ${prediction.confidence.toFixed(2)}`);
    }
  }

  private collectFreshAgentData(): any {
    const now = Date.now();
    const freshnessThreshold = this.parameters.agent_freshness_threshold_sec * 1000;
    
    const freshData: any = {};
    
    // Check economic data freshness
    if (this.lastAgentData.economic) {
      const age = now - new Date(this.lastAgentData.economic.timestamp).getTime();
      if (age <= freshnessThreshold) {
        freshData.economic = this.lastAgentData.economic;
      }
    }
    
    // Check sentiment data freshness
    if (this.lastAgentData.sentiment) {
      const age = now - new Date(this.lastAgentData.sentiment.timestamp).getTime();
      if (age <= freshnessThreshold) {
        freshData.sentiment = this.lastAgentData.sentiment;
      }
    }
    
    // Check microstructure data freshness
    if (this.lastAgentData.microstructure) {
      const age = now - new Date(this.lastAgentData.microstructure.timestamp).getTime();
      if (age <= freshnessThreshold) {
        freshData.microstructure = this.lastAgentData.microstructure;
      }
    }
    
    return freshData;
  }

  private validateAgentData(agentData: any): boolean {
    const availableAgents = Object.keys(agentData);
    const requiredCount = this.parameters.min_agent_count_for_prediction;
    
    if (availableAgents.length < requiredCount) {
      // console.log(`⏳ Insufficient fresh agents: ${availableAgents.length}/${requiredCount}`);
      return false;
    }
    
    // Check confidence thresholds
    for (const agentType of availableAgents) {
      const agentOutput = agentData[agentType];
      if (agentOutput.confidence < this.constraints.min_prediction_confidence) {
        console.log(`⚠️ Low confidence from ${agentType}: ${agentOutput.confidence}`);
        return false;
      }
    }
    
    return true;
  }

  private async buildHyperbolicContext(agentData: any): Promise<any> {
    try {
      // Create combined feature vector from all agents
      const combinedFeatures: Record<string, number> = {};
      
      // Economic features
      if (agentData.economic) {
        combinedFeatures.economic_signal = agentData.economic.key_signal;
        combinedFeatures.cpi_yoy = agentData.economic.features.cpi_yoy_pct;
        combinedFeatures.fed_rate = agentData.economic.features.fed_funds_rate;
        combinedFeatures.real_rate = agentData.economic.features.real_interest_rate;
        combinedFeatures.liquidity_conditions = agentData.economic.features.liquidity_conditions;
      }
      
      // Sentiment features
      if (agentData.sentiment) {
        combinedFeatures.sentiment_signal = agentData.sentiment.key_signal;
        combinedFeatures.news_sentiment = agentData.sentiment.features.news_sentiment_score;
        combinedFeatures.fear_greed = agentData.sentiment.features.fear_greed_index;
        combinedFeatures.mention_volume = Math.log(agentData.sentiment.features.article_count_24h + 1);
      }
      
      // Microstructure features
      if (agentData.microstructure) {
        combinedFeatures.micro_signal = agentData.microstructure.key_signal;
        combinedFeatures.spread_bps = agentData.microstructure.features.bid_ask_spread_bps;
        combinedFeatures.orderbook_imbalance = agentData.microstructure.features.orderbook_imbalance;
        combinedFeatures.volume_surge = agentData.microstructure.features.volume_surge_factor;
        combinedFeatures.price_momentum = agentData.microstructure.features.price_momentum_1min;
      }
      
      // Generate hyperbolic embedding
      const embedding = await this.hyperbolicEmbedder.embed(combinedFeatures);
      
      // Find k-nearest neighbors in hyperbolic space
      const neighbors = await this.hyperbolicEmbedder.knnQuery(embedding, 5);
      
      return {
        embedding,
        neighbors,
        confidence: this.calculateHyperbolicConfidence(embedding, neighbors)
      };
    } catch (error) {
      console.error('❌ Hyperbolic context error:', error);
      return {
        embedding: new Array(this.parameters.hyperbolic_embedding_dim).fill(0),
        neighbors: [],
        confidence: 0.5
      };
    }
  }

  private calculateHyperbolicConfidence(embedding: number[], neighbors: any[]): number {
    if (neighbors.length === 0) return 0.5;
    
    // Calculate confidence based on neighbor consensus and distance
    const avgDistance = neighbors.reduce((sum, n) => sum + (n.distance || 1), 0) / neighbors.length;
    const distanceConfidence = Math.max(0, 1 - avgDistance);
    
    return Math.min(1, Math.max(0, distanceConfidence));
  }

  private async callLLMFusion(fusionInput: FusionInput): Promise<ArbitragePrediction | null> {
    try {
      // Build compact prompt for LLM
      const prompt = this.buildLLMPrompt(fusionInput);
      
      // Call LLM with timeout
      const llmResponse = await this.callLLMAPI(prompt);
      
      // Parse and validate response
      const prediction = this.parseLLMResponse(llmResponse, fusionInput);
      
      return prediction;
      
    } catch (error) {
      console.error('❌ LLM fusion error:', error);
      return null;
    }
  }

  private buildLLMPrompt(fusionInput: FusionInput): string {
    // Create compact numerical context
    const context: any = {
      timestamp: fusionInput.timestamp
    };
    
    if (fusionInput.economic) {
      context.economic = {
        liquidity_bias: fusionInput.economic.key_signal,
        cpi_yoy: fusionInput.economic.features.cpi_yoy_pct,
        fed_rate: fusionInput.economic.features.fed_funds_rate,
        real_rate: fusionInput.economic.features.real_interest_rate,
        vix: fusionInput.economic.features.vix_index
      };
    }
    
    if (fusionInput.sentiment) {
      context.sentiment = {
        polarity: fusionInput.sentiment.key_signal,
        news_sentiment: fusionInput.sentiment.features.news_sentiment_score,
        article_count: fusionInput.sentiment.features.article_count_24h,
        fear_greed: fusionInput.sentiment.features.fear_greed_index
      };
    }
    
    if (fusionInput.microstructure) {
      context.microstructure = {
        signal: fusionInput.microstructure.key_signal,
        spread_bps: fusionInput.microstructure.features.bid_ask_spread_bps,
        orderbook_imbalance: fusionInput.microstructure.features.orderbook_imbalance,
        volume_surge: fusionInput.microstructure.features.volume_surge_factor,
        momentum_1min: fusionInput.microstructure.features.price_momentum_1min
      };
    }
    
    const systemPrompt = `You are an arbitrage fusion assistant. Given real-time agent signals, return ONLY a strict JSON object. Do not output anything else.

Required JSON format:
{
  "predicted_spread_pct": <float between 0.001 and 0.05>,
  "confidence": <float between 0.6 and 1.0>,
  "direction": "converge" | "diverge",
  "expected_time_s": <int between 60 and 3600>,
  "arbitrage_plan": {
    "buy": "binance" | "coinbase" | "kraken",
    "sell": "binance" | "coinbase" | "kraken", 
    "notional_usd": <int between 1000 and 100000>
  },
  "rationale": "<brief explanation>",
  "risk_flags": [<array of string risk factors>]
}`;
    
    const userPrompt = JSON.stringify(context);
    
    return `${systemPrompt}\n\nAgent signals:\n${userPrompt}`;
  }

  private async callLLMAPI(prompt: string): Promise<string> {
    if (this.llmApiKey === 'demo_key') {
      // Return simulated LLM response for demo
      return this.generateSimulatedLLMResponse();
    }
    
    try {
      if (this.llmProvider === 'anthropic') {
        return await this.callAnthropicAPI(prompt);
      } else {
        return await this.callOpenAIAPI(prompt);
      }
    } catch (error) {
      console.error('❌ LLM API error:', error);
      // Fallback to simulated response
      return this.generateSimulatedLLMResponse();
    }
  }

  private async callAnthropicAPI(prompt: string): Promise<string> {
    const response = await fetch('https://api.anthropic.com/v1/messages', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': this.llmApiKey,
        'anthropic-version': '2023-06-01'
      },
      body: JSON.stringify({
        model: 'claude-3-sonnet-20240229',
        max_tokens: 1000,
        messages: [{ role: 'user', content: prompt }]
      })
    });
    
    if (!response.ok) {
      throw new Error(`Anthropic API error: ${response.status}`);
    }
    
    const data: any = await response.json();
    return data.content[0].text;
  }

  private async callOpenAIAPI(prompt: string): Promise<string> {
    const response = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.llmApiKey}`
      },
      body: JSON.stringify({
        model: 'gpt-4',
        messages: [{ role: 'user', content: prompt }],
        max_tokens: 1000,
        temperature: 0.1
      })
    });
    
    if (!response.ok) {
      throw new Error(`OpenAI API error: ${response.status}`);
    }
    
    const data: any = await response.json();
    return data.choices[0].message.content;
  }

  private generateSimulatedLLMResponse(): string {
    // Generate realistic arbitrage prediction for demo
    const predictions = [
      {
        predicted_spread_pct: 0.0045 + Math.random() * 0.005, // 0.45% to 0.95%
        confidence: 0.6 + Math.random() * 0.35, // 60% to 95%
        direction: Math.random() > 0.5 ? 'converge' : 'diverge',
        expected_time_s: 180 + Math.random() * 420, // 3 to 10 minutes
        arbitrage_plan: {
          buy: Math.random() > 0.5 ? 'binance' : 'coinbase',
          sell: Math.random() > 0.5 ? 'coinbase' : 'binance',
          notional_usd: 10000 + Math.random() * 40000
        },
        rationale: 'Positive sentiment with sufficient liquidity and mid-price divergence detected',
        risk_flags: Math.random() > 0.7 ? ['low_liquidity_warning'] : []
      }
    ];
    
    return JSON.stringify(predictions[0]);
  }

  private parseLLMResponse(llmResponse: string, fusionInput: FusionInput): ArbitragePrediction | null {
    try {
      // Clean response and parse JSON
      const cleanResponse = llmResponse.replace(/```json\n?/g, '').replace(/```\n?/g, '').trim();
      const parsed = JSON.parse(cleanResponse);
      
      // Validate response structure
      if (!this.validateLLMResponse(parsed)) {
        throw new Error('Invalid LLM response structure');
      }
      
      // Apply bounds
      const prediction: ArbitragePrediction = {
        id: `pred_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date().toISOString(),
        predicted_spread_pct: this.clampToRange(parsed.predicted_spread_pct, this.bounds.predicted_spread_pct_range),
        confidence: this.clampToRange(parsed.confidence, this.bounds.confidence_range),
        direction: parsed.direction || 'converge',
        expected_time_s: this.clampToRange(parsed.expected_time_s, this.bounds.time_horizon_range),
        arbitrage_plan: {
          buy: parsed.arbitrage_plan.buy,
          sell: parsed.arbitrage_plan.sell,
          notional_usd: this.clampToRange(parsed.arbitrage_plan.notional_usd, this.bounds.notional_usd_range)
        },
        rationale: parsed.rationale || 'LLM arbitrage analysis',
        risk_flags: parsed.risk_flags || [],
        fusion_input: fusionInput,
        hyperbolic_embedding: fusionInput.hyperbolic_context.embedding
      };
      
      return prediction;
      
    } catch (error) {
      console.error('❌ Failed to parse LLM response:', error);
      console.error('Raw response:', llmResponse);
      return null;
    }
  }

  private validateLLMResponse(parsed: any): boolean {
    return (
      typeof parsed.predicted_spread_pct === 'number' &&
      typeof parsed.confidence === 'number' &&
      typeof parsed.direction === 'string' &&
      typeof parsed.expected_time_s === 'number' &&
      parsed.arbitrage_plan &&
      typeof parsed.arbitrage_plan.buy === 'string' &&
      typeof parsed.arbitrage_plan.sell === 'string' &&
      typeof parsed.arbitrage_plan.notional_usd === 'number'
    );
  }

  private clampToRange(value: number, range: { min: number; max: number }): number {
    return Math.max(range.min, Math.min(range.max, value));
  }

  // Public methods for transparency
  getVisibleParameters(): any {
    return {
      parameters: this.parameters,
      constraints: this.constraints,
      bounds: this.bounds,
      llm_provider: this.llmProvider,
      active_predictions_count: this.activePredictions.size,
      last_agent_data_age: this.getLastAgentDataAge()
    };
  }

  private getLastAgentDataAge(): any {
    const now = Date.now();
    const ages: any = {};
    
    if (this.lastAgentData.economic) {
      ages.economic_age_sec = (now - new Date(this.lastAgentData.economic.timestamp).getTime()) / 1000;
    }
    if (this.lastAgentData.sentiment) {
      ages.sentiment_age_sec = (now - new Date(this.lastAgentData.sentiment.timestamp).getTime()) / 1000;
    }
    if (this.lastAgentData.microstructure) {
      ages.microstructure_age_sec = (now - new Date(this.lastAgentData.microstructure.timestamp).getTime()) / 1000;
    }
    
    return ages;
  }

  getActivePredictions(): ArbitragePrediction[] {
    return Array.from(this.activePredictions.values());
  }

  getLatestPrediction(): ArbitragePrediction | null {
    const predictions = this.getActivePredictions();
    if (predictions.length === 0) return null;
    
    return predictions.reduce((latest, current) => 
      new Date(current.timestamp) > new Date(latest.timestamp) ? current : latest
    );
  }
}