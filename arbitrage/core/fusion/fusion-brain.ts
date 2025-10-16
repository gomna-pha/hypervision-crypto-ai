/**
 * Fusion Brain - LLM Core Orchestration System
 * Integrates all agent outputs, hyperbolic embeddings, and contextual analysis
 * Provides structured arbitrage predictions using Claude/GPT-4
 */

import { AgentOutput } from '../base-agent.js';
import { HyperbolicEngine, NearestNeighbors } from '../../hyperbolic/hyperbolic-engine.js';
import axios, { AxiosResponse } from 'axios';

export interface FusionInput {
  agents: Record<string, AgentOutput>;
  hyperbolic_context: Record<string, NearestNeighbors>;
  market_conditions: {
    volatility: number;
    liquidity: number;
    trend: string;
  };
  timestamp: string;
}

export interface ArbitragePlan {
  buy_exchange: string;
  sell_exchange: string;
  pair: string;
  notional_usd: number;
  estimated_profit_usd: number;
  max_position_time_sec: number;
}

export interface FusionPrediction {
  predicted_spread_pct: number;    // Expected spread to capture (e.g., 0.7%)
  confidence: number;              // 0..1
  direction: 'converge' | 'diverge' | 'stable';
  expected_time_s: number;         // Time horizon for prediction
  arbitrage_plan: ArbitragePlan;
  rationale: string;               // LLM explanation
  risk_flags: string[];           // Identified risks
  aos_score: number;              // Arbitrage Opportunity Score
  timestamp: string;
}

export interface LLMConfig {
  provider: 'anthropic' | 'openai' | 'local';
  model: string;
  api_key: string;
  max_tokens: number;
  temperature: number;
  timeout_ms: number;
  fallback_provider?: 'anthropic' | 'openai';
  fallback_model?: string;
  fallback_api_key?: string;
}

/**
 * Fusion Brain Class - Orchestrates LLM-based arbitrage analysis
 */
export class FusionBrain {
  private llmConfig: LLMConfig;
  private hyperbolicEngine: HyperbolicEngine;
  private predictionHistory: FusionPrediction[] = [];
  private systemPrompt: string;
  
  // AOS weights from configuration
  private aosWeights = {
    price: 0.4,
    sentiment: 0.25,
    volume: 0.2,
    image: 0.05,
    risk: -0.1
  };

  constructor(llmConfig: LLMConfig, hyperbolicEngine: HyperbolicEngine) {
    this.llmConfig = llmConfig;
    this.hyperbolicEngine = hyperbolicEngine;
    this.systemPrompt = this.createSystemPrompt();
  }

  /**
   * Main fusion process - analyze agents and generate prediction
   */
  async generatePrediction(agents: Record<string, AgentOutput>): Promise<FusionPrediction> {
    const timestamp = new Date().toISOString();

    try {
      // Step 1: Validate agent data freshness and confidence
      const validAgents = this.validateAgentData(agents);
      
      if (Object.keys(validAgents).length === 0) {
        throw new Error('No valid agent data available for fusion');
      }

      // Step 2: Generate hyperbolic context
      const hyperbolicContext = this.hyperbolicEngine.getContextualNeighbors(
        Object.values(validAgents), 
        10
      );

      // Step 3: Build compact fusion input
      const fusionInput = this.buildFusionInput(validAgents, hyperbolicContext, timestamp);

      // Step 4: Calculate AOS score
      const aosScore = this.calculateAOS(validAgents);

      // Step 5: Query LLM with structured prompt
      const llmResponse = await this.queryLLM(fusionInput);

      // Step 6: Validate and enhance LLM response
      const prediction = this.validateAndEnhancePrediction(llmResponse, aosScore, timestamp);

      // Step 7: Store for audit trail
      this.predictionHistory.push(prediction);
      this.trimPredictionHistory();

      console.log(`Fusion prediction generated: ${prediction.confidence.toFixed(3)} confidence, ${prediction.predicted_spread_pct.toFixed(4)}% spread`);

      return prediction;

    } catch (error) {
      console.error('Fusion Brain prediction failed:', error);
      throw error;
    }
  }

  /**
   * Validate agent data freshness and confidence thresholds
   */
  private validateAgentData(agents: Record<string, AgentOutput>): Record<string, AgentOutput> {
    const validAgents: Record<string, AgentOutput> = {};
    const now = Date.now();
    const maxAge = 5000; // 5 seconds for HFT decisions

    for (const [name, agent] of Object.entries(agents)) {
      if (!agent) continue;

      // Check data freshness
      const age = now - new Date(agent.timestamp).getTime();
      if (age > maxAge) {
        console.warn(`Agent ${name} data is stale (${age}ms old)`);
        continue;
      }

      // Check confidence threshold (from config, default values)
      const minConfidence = this.getMinConfidenceForAgent(name);
      if (agent.confidence < minConfidence) {
        console.warn(`Agent ${name} confidence ${agent.confidence} below threshold ${minConfidence}`);
        continue;
      }

      validAgents[name] = agent;
    }

    return validAgents;
  }

  /**
   * Get minimum confidence threshold for agent type
   */
  private getMinConfidenceForAgent(agentName: string): number {
    const thresholds = {
      'economic': 0.5,
      'sentiment': 0.4,
      'price': 0.6,
      'volume': 0.2,
      'trade': 0.4,
      'image': 0.6
    };
    
    return thresholds[agentName as keyof typeof thresholds] || 0.5;
  }

  /**
   * Build compact fusion input for LLM
   */
  private buildFusionInput(
    agents: Record<string, AgentOutput>, 
    hyperbolicContext: Map<string, NearestNeighbors>,
    timestamp: string
  ): FusionInput {
    // Extract key market conditions
    const priceAgent = agents['price'];
    const volumeAgent = agents['volume'];
    const tradeAgent = agents['trade'];

    const marketConditions = {
      volatility: priceAgent?.features?.volatility_1m || 0.5,
      liquidity: volumeAgent?.features?.liquidity_index || 0.5,
      trend: this.determineTrend(priceAgent)
    };

    return {
      agents,
      hyperbolic_context: Object.fromEntries(hyperbolicContext),
      market_conditions: marketConditions,
      timestamp
    };
  }

  /**
   * Determine market trend from price agent
   */
  private determineTrend(priceAgent?: AgentOutput): string {
    if (!priceAgent?.features) return 'neutral';

    const momentum = priceAgent.features.price_momentum_1m || 0;
    
    if (momentum > 0.1) return 'bullish';
    if (momentum < -0.1) return 'bearish';
    return 'neutral';
  }

  /**
   * Calculate Arbitrage Opportunity Score (AOS)
   */
  private calculateAOS(agents: Record<string, AgentOutput>): number {
    let score = 0;
    let totalWeight = 0;

    // Price component
    if (agents['price']) {
      const priceSignal = agents['price'].key_signal;
      const priceWeight = this.aosWeights.price * agents['price'].confidence;
      score += priceSignal * priceWeight;
      totalWeight += Math.abs(priceWeight);
    }

    // Sentiment component
    if (agents['sentiment']) {
      const sentimentSignal = agents['sentiment'].key_signal;
      const sentimentWeight = this.aosWeights.sentiment * agents['sentiment'].confidence;
      score += sentimentSignal * sentimentWeight;
      totalWeight += Math.abs(sentimentWeight);
    }

    // Volume component
    if (agents['volume']) {
      const volumeSignal = agents['volume'].key_signal;
      const volumeWeight = this.aosWeights.volume * agents['volume'].confidence;
      score += volumeSignal * volumeWeight;
      totalWeight += Math.abs(volumeWeight);
    }

    // Image component
    if (agents['image']) {
      const imageSignal = agents['image'].key_signal;
      const imageWeight = this.aosWeights.image * agents['image'].confidence;
      score += imageSignal * imageWeight;
      totalWeight += Math.abs(imageWeight);
    }

    // Risk component (negative weight)
    const riskScore = this.calculateRiskScore(agents);
    score += riskScore * this.aosWeights.risk;
    totalWeight += Math.abs(this.aosWeights.risk);

    return totalWeight > 0 ? score / totalWeight : 0;
  }

  /**
   * Calculate composite risk score
   */
  private calculateRiskScore(agents: Record<string, AgentOutput>): number {
    let riskFactors = 0;
    let riskCount = 0;

    // Volatility risk
    if (agents['price']?.features?.volatility_1m) {
      riskFactors += agents['price'].features.volatility_1m;
      riskCount++;
    }

    // Liquidity risk
    if (agents['volume']?.features?.liquidity_index) {
      riskFactors += (1 - agents['volume'].features.liquidity_index);
      riskCount++;
    }

    // Execution risk
    if (agents['trade']?.features?.market_impact) {
      riskFactors += agents['trade'].features.market_impact;
      riskCount++;
    }

    return riskCount > 0 ? riskFactors / riskCount : 0.5;
  }

  /**
   * Create system prompt for LLM
   */
  private createSystemPrompt(): string {
    return `You are an arbitrage fusion assistant. You will be given JSON containing the latest agent outputs from economic, sentiment, price, volume, trade, and image agents, along with hyperbolic embedding context.

Your task is to analyze this multi-modal data and predict arbitrage opportunities with high precision.

Return ONLY a strict JSON object with these exact fields:
- predicted_spread_pct (float): Expected net spread to capture (e.g., 0.007 for 0.7%)
- confidence (float 0-1): Your confidence in this prediction
- direction (string): "converge", "diverge", or "stable"
- expected_time_s (integer): Time horizon in seconds (60-3600 range)
- arbitrage_plan (object): {"buy_exchange": string, "sell_exchange": string, "pair": string, "notional_usd": number}
- rationale (string): Brief explanation of your reasoning
- risk_flags (array of strings): Identified risks or concerns

Important guidelines:
1. Only predict spreads between 0.001% and 2.0%
2. Confidence should reflect data quality and market conditions
3. Consider economic backdrop, sentiment momentum, liquidity conditions, and execution quality
4. Flag risks like low liquidity, high volatility, or conflicting signals
5. Notional amounts should be reasonable ($10k-$500k range)
6. Do not output anything other than the JSON object

Example output:
{
  "predicted_spread_pct": 0.007,
  "confidence": 0.86,
  "direction": "converge",
  "expected_time_s": 300,
  "arbitrage_plan": {"buy_exchange": "binance", "sell_exchange": "coinbase", "pair": "BTC-USDT", "notional_usd": 100000},
  "rationale": "Positive sentiment with sufficient liquidity and mid-price divergence",
  "risk_flags": ["moderate_volatility"]
}`;
  }

  /**
   * Query LLM with fusion input
   */
  private async queryLLM(fusionInput: FusionInput): Promise<any> {
    const userMessage = this.createUserMessage(fusionInput);
    
    try {
      return await this.callLLMProvider(this.llmConfig.provider, userMessage);
    } catch (error) {
      console.warn(`Primary LLM provider failed, trying fallback:`, error.message);
      
      if (this.llmConfig.fallback_provider) {
        return await this.callLLMProvider(this.llmConfig.fallback_provider, userMessage);
      }
      
      throw error;
    }
  }

  /**
   * Create user message from fusion input
   */
  private createUserMessage(fusionInput: FusionInput): string {
    const compactInput = {
      economic_agent: this.extractAgentSummary(fusionInput.agents['economic']),
      sentiment_agent: this.extractAgentSummary(fusionInput.agents['sentiment']),
      price_agent: this.extractAgentSummary(fusionInput.agents['price']),
      volume_agent: this.extractAgentSummary(fusionInput.agents['volume']),
      trade_agent: this.extractAgentSummary(fusionInput.agents['trade']),
      image_agent: this.extractAgentSummary(fusionInput.agents['image']),
      market_conditions: fusionInput.market_conditions,
      hyperbolic_insights: this.extractHyperbolicInsights(fusionInput.hyperbolic_context)
    };

    return JSON.stringify(compactInput);
  }

  /**
   * Extract key data from agent output for LLM
   */
  private extractAgentSummary(agent?: AgentOutput): any {
    if (!agent) return null;

    return {
      key_signal: agent.key_signal,
      confidence: agent.confidence,
      features: this.selectTopFeatures(agent.features, 5),
      age_ms: Date.now() - new Date(agent.timestamp).getTime()
    };
  }

  /**
   * Select top numerical features from agent output
   */
  private selectTopFeatures(features: any, maxFeatures: number): Record<string, number> {
    const numericalFeatures: Record<string, number> = {};
    
    const extractNumerical = (obj: any, prefix: string = '') => {
      for (const [key, value] of Object.entries(obj || {})) {
        const fullKey = prefix ? `${prefix}_${key}` : key;
        
        if (typeof value === 'number' && !isNaN(value) && isFinite(value)) {
          numericalFeatures[fullKey] = value;
        } else if (typeof value === 'object' && value !== null && prefix.split('_').length < 2) {
          extractNumerical(value, fullKey);
        }
      }
    };

    extractNumerical(features);

    // Sort by absolute value and take top features
    const sortedFeatures = Object.entries(numericalFeatures)
      .sort(([, a], [, b]) => Math.abs(b) - Math.abs(a))
      .slice(0, maxFeatures);

    return Object.fromEntries(sortedFeatures);
  }

  /**
   * Extract insights from hyperbolic context
   */
  private extractHyperbolicInsights(context: Record<string, NearestNeighbors>): any {
    const insights: any = {};

    for (const [agentName, neighbors] of Object.entries(context)) {
      if (neighbors?.neighbors?.length > 0) {
        const avgSimilarity = neighbors.neighbors.reduce((sum, n) => sum + n.similarity, 0) / neighbors.neighbors.length;
        const diverseAgents = new Set(neighbors.neighbors.map(n => n.agent_name)).size;
        
        insights[agentName] = {
          avg_similarity: avgSimilarity,
          neighbor_diversity: diverseAgents,
          closest_agent: neighbors.neighbors[0]?.agent_name || 'unknown'
        };
      }
    }

    return insights;
  }

  /**
   * Call specific LLM provider
   */
  private async callLLMProvider(provider: string, userMessage: string): Promise<any> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };

    let url: string;
    let requestBody: any;

    switch (provider) {
      case 'anthropic':
        url = 'https://api.anthropic.com/v1/messages';
        headers['Authorization'] = `Bearer ${this.llmConfig.api_key}`;
        headers['anthropic-version'] = '2023-06-01';
        
        requestBody = {
          model: this.llmConfig.model || 'claude-3-sonnet-20240229',
          max_tokens: this.llmConfig.max_tokens,
          temperature: this.llmConfig.temperature,
          system: this.systemPrompt,
          messages: [
            { role: 'user', content: userMessage }
          ]
        };
        break;

      case 'openai':
        url = 'https://api.openai.com/v1/chat/completions';
        headers['Authorization'] = `Bearer ${this.llmConfig.api_key}`;
        
        requestBody = {
          model: this.llmConfig.model || 'gpt-4',
          max_tokens: this.llmConfig.max_tokens,
          temperature: this.llmConfig.temperature,
          messages: [
            { role: 'system', content: this.systemPrompt },
            { role: 'user', content: userMessage }
          ]
        };
        break;

      default:
        throw new Error(`Unsupported LLM provider: ${provider}`);
    }

    const response: AxiosResponse = await axios.post(url, requestBody, {
      headers,
      timeout: this.llmConfig.timeout_ms
    });

    return this.extractLLMResponse(response.data, provider);
  }

  /**
   * Extract response from LLM provider response format
   */
  private extractLLMResponse(responseData: any, provider: string): any {
    let content: string;

    switch (provider) {
      case 'anthropic':
        content = responseData.content?.[0]?.text || '';
        break;
      case 'openai':
        content = responseData.choices?.[0]?.message?.content || '';
        break;
      default:
        throw new Error(`Unknown provider response format: ${provider}`);
    }

    try {
      return JSON.parse(content);
    } catch (error) {
      throw new Error(`Invalid JSON response from LLM: ${content}`);
    }
  }

  /**
   * Validate and enhance LLM prediction
   */
  private validateAndEnhancePrediction(
    llmResponse: any, 
    aosScore: number, 
    timestamp: string
  ): FusionPrediction {
    // Validate required fields
    const requiredFields = ['predicted_spread_pct', 'confidence', 'direction', 'expected_time_s', 'arbitrage_plan', 'rationale'];
    
    for (const field of requiredFields) {
      if (!(field in llmResponse)) {
        throw new Error(`Missing required field in LLM response: ${field}`);
      }
    }

    // Validate and clamp values
    const spread = Math.max(0.00001, Math.min(0.02, parseFloat(llmResponse.predicted_spread_pct)));
    const confidence = Math.max(0, Math.min(1, parseFloat(llmResponse.confidence)));
    const timeHorizon = Math.max(60, Math.min(3600, parseInt(llmResponse.expected_time_s)));

    // Validate arbitrage plan
    const plan = llmResponse.arbitrage_plan;
    if (!plan.buy_exchange || !plan.sell_exchange || !plan.pair) {
      throw new Error('Invalid arbitrage plan in LLM response');
    }

    const notionalUsd = Math.max(1000, Math.min(1000000, parseFloat(plan.notional_usd || 50000)));

    return {
      predicted_spread_pct: spread,
      confidence: confidence,
      direction: ['converge', 'diverge', 'stable'].includes(llmResponse.direction) ? llmResponse.direction : 'stable',
      expected_time_s: timeHorizon,
      arbitrage_plan: {
        buy_exchange: plan.buy_exchange,
        sell_exchange: plan.sell_exchange,
        pair: plan.pair,
        notional_usd: notionalUsd,
        estimated_profit_usd: (notionalUsd * spread) - (notionalUsd * 0.001), // Subtract fees
        max_position_time_sec: timeHorizon
      },
      rationale: String(llmResponse.rationale || 'AI analysis based on multi-modal agent data'),
      risk_flags: Array.isArray(llmResponse.risk_flags) ? llmResponse.risk_flags : [],
      aos_score: aosScore,
      timestamp: timestamp
    };
  }

  /**
   * Get prediction history for analysis
   */
  getPredictionHistory(limit: number = 50): FusionPrediction[] {
    return this.predictionHistory.slice(-limit);
  }

  /**
   * Trim prediction history to prevent memory bloat
   */
  private trimPredictionHistory(): void {
    if (this.predictionHistory.length > 200) {
      this.predictionHistory = this.predictionHistory.slice(-100);
    }
  }

  /**
   * Get fusion statistics
   */
  getFusionStatistics(): {
    total_predictions: number;
    avg_confidence: number;
    avg_spread_predicted: number;
    direction_distribution: Record<string, number>;
    avg_time_horizon_s: number;
  } {
    if (this.predictionHistory.length === 0) {
      return {
        total_predictions: 0,
        avg_confidence: 0,
        avg_spread_predicted: 0,
        direction_distribution: {},
        avg_time_horizon_s: 0
      };
    }

    const totalPredictions = this.predictionHistory.length;
    const avgConfidence = this.predictionHistory.reduce((sum, p) => sum + p.confidence, 0) / totalPredictions;
    const avgSpread = this.predictionHistory.reduce((sum, p) => sum + p.predicted_spread_pct, 0) / totalPredictions;
    const avgTimeHorizon = this.predictionHistory.reduce((sum, p) => sum + p.expected_time_s, 0) / totalPredictions;

    const directionCounts: Record<string, number> = {};
    for (const prediction of this.predictionHistory) {
      directionCounts[prediction.direction] = (directionCounts[prediction.direction] || 0) + 1;
    }

    return {
      total_predictions: totalPredictions,
      avg_confidence: avgConfidence,
      avg_spread_predicted: avgSpread,
      direction_distribution: directionCounts,
      avg_time_horizon_s: avgTimeHorizon
    };
  }
}

/**
 * Factory function to create FusionBrain
 */
export function createFusionBrain(llmConfig: LLMConfig, hyperbolicEngine: HyperbolicEngine): FusionBrain {
  return new FusionBrain(llmConfig, hyperbolicEngine);
}

// Export for testing
export { FusionBrain as default };