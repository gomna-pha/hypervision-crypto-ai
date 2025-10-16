/**
 * Platform Orchestrator - Main System Controller
 * Coordinates all agents, fusion brain, decision engine, and execution
 * Provides the main entry point and system management for the arbitrage platform
 */

import { AgentRegistry, BaseAgent } from './base-agent.js';
import { createEconomicAgent, DEMO_FRED_API_KEY } from '../agents/economic/economic-agent.js';
import { createSentimentAgent } from '../agents/sentiment/sentiment-agent.js';
import { createPriceAgent } from '../agents/price/price-agent.js';
import { createVolumeAgent } from '../agents/volume/volume-agent.js';
import { createTradeAgent } from '../agents/trade/trade-agent.js';
import { createImageAgent } from '../agents/image/image-agent.js';
import { HyperbolicEngine, createHyperbolicEngine } from '../hyperbolic/hyperbolic-engine.js';
import { FusionBrain, createFusionBrain, LLMConfig } from './fusion/fusion-brain.js';
import { DecisionEngine, createDecisionEngine } from '../decision/decision-engine.js';
import { EventEmitter } from 'events';

export interface PlatformConfig {
  agents: {
    enabled_agents: string[];
    polling_intervals: Record<string, number>;
  };
  llm: LLMConfig;
  hyperbolic: {
    embedding_dim: number;
    curvature: number;
  };
  execution: {
    sandbox_mode: boolean;
    max_position_size_usd: number;
  };
  api_keys: {
    fred_api_key?: string;
    twitter_bearer_token?: string;
    news_api_key?: string;
    anthropic_api_key?: string;
    openai_api_key?: string;
  };
}

export interface SystemMetrics {
  uptime_seconds: number;
  total_predictions: number;
  successful_predictions: number;
  active_agents: number;
  avg_agent_confidence: number;
  system_health_score: number;
  last_fusion_timestamp: string;
  circuit_breaker_active: boolean;
}

export interface ArbitrageOpportunity {
  opportunity_id: string;
  timestamp: string;
  pair: string;
  buy_exchange: string;
  sell_exchange: string;
  predicted_spread_pct: number;
  confidence: number;
  estimated_profit_usd: number;
  risk_score: number;
  approved_for_execution: boolean;
  rationale: string;
}

/**
 * Main Platform Orchestrator
 */
export class ArbitragePlatformOrchestrator extends EventEmitter {
  private config: PlatformConfig;
  private agentRegistry: AgentRegistry;
  private hyperbolicEngine: HyperbolicEngine;
  private fusionBrain: FusionBrain;
  private decisionEngine: DecisionEngine;
  
  private isRunning: boolean = false;
  private startTime: number = 0;
  private fusionInterval: NodeJS.Timeout | null = null;
  private healthCheckInterval: NodeJS.Timeout | null = null;
  
  // System state
  private opportunities: ArbitrageOpportunity[] = [];
  private systemMetrics: SystemMetrics;
  
  // Performance tracking
  private predictionCount: number = 0;
  private successfulPredictions: number = 0;

  constructor(config?: Partial<PlatformConfig>) {
    super();
    this.config = this.mergeWithDefaults(config || {});
    this.initializeComponents();
    this.initializeMetrics();
  }

  /**
   * Start the arbitrage platform
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      throw new Error('Platform is already running');
    }

    console.log('🚀 Starting Agent-Based LLM Arbitrage Platform...');
    
    try {
      this.startTime = Date.now();
      
      // Step 1: Start all agents
      console.log('📡 Starting data collection agents...');
      await this.agentRegistry.startAll();
      
      // Step 2: Wait for initial agent data
      console.log('⏳ Waiting for initial agent data...');
      await this.waitForInitialData();
      
      // Step 3: Start fusion brain loop
      console.log('🧠 Starting Fusion Brain analysis loop...');
      this.startFusionLoop();
      
      // Step 4: Start system health monitoring
      console.log('💊 Starting system health monitoring...');
      this.startHealthMonitoring();
      
      this.isRunning = true;
      
      console.log('✅ Agent-Based LLM Arbitrage Platform is now running!');
      console.log(`📊 Active agents: ${this.agentRegistry.getSystemHealth().toString()}`);
      
      this.emit('platform_started', {
        timestamp: new Date().toISOString(),
        active_agents: Object.keys(this.agentRegistry.getSystemHealth()).length
      });
      
    } catch (error) {
      console.error('❌ Failed to start platform:', error);
      throw error;
    }
  }

  /**
   * Stop the arbitrage platform
   */
  async stop(): Promise<void> {
    if (!this.isRunning) {
      return;
    }

    console.log('🛑 Stopping Agent-Based LLM Arbitrage Platform...');
    
    try {
      // Stop fusion loop
      if (this.fusionInterval) {
        clearInterval(this.fusionInterval);
        this.fusionInterval = null;
      }
      
      // Stop health monitoring
      if (this.healthCheckInterval) {
        clearInterval(this.healthCheckInterval);
        this.healthCheckInterval = null;
      }
      
      // Stop all agents
      await this.agentRegistry.stopAll();
      
      this.isRunning = false;
      
      console.log('✅ Platform stopped successfully');
      
      this.emit('platform_stopped', {
        timestamp: new Date().toISOString(),
        uptime_seconds: Math.floor((Date.now() - this.startTime) / 1000)
      });
      
    } catch (error) {
      console.error('❌ Error stopping platform:', error);
      throw error;
    }
  }

  /**
   * Get current arbitrage opportunities
   */
  getOpportunities(limit: number = 20): ArbitrageOpportunity[] {
    return this.opportunities.slice(-limit);
  }

  /**
   * Get system metrics
   */
  getSystemMetrics(): SystemMetrics {
    const uptime = this.isRunning ? Math.floor((Date.now() - this.startTime) / 1000) : 0;
    const agentHealth = this.agentRegistry.getSystemHealth();
    const agentCount = Object.keys(agentHealth).length;
    
    // Calculate average agent confidence
    let totalConfidence = 0;
    let validAgents = 0;
    
    for (const health of Object.values(agentHealth)) {
      if (health.status === 'healthy' && health.last_update !== 'never') {
        // Get latest output from agent (simplified)
        totalConfidence += 0.75; // Placeholder - in production, get from actual agent outputs
        validAgents++;
      }
    }
    
    const avgConfidence = validAgents > 0 ? totalConfidence / validAgents : 0;
    
    // Calculate system health score
    const healthyAgents = Object.values(agentHealth).filter(h => h.status === 'healthy').length;
    const systemHealthScore = agentCount > 0 ? healthyAgents / agentCount : 0;
    
    return {
      uptime_seconds: uptime,
      total_predictions: this.predictionCount,
      successful_predictions: this.successfulPredictions,
      active_agents: agentCount,
      avg_agent_confidence: avgConfidence,
      system_health_score: systemHealthScore,
      last_fusion_timestamp: this.systemMetrics.last_fusion_timestamp,
      circuit_breaker_active: this.decisionEngine.getSystemStatus().circuit_breaker_active
    };
  }

  /**
   * Get system status for monitoring
   */
  getSystemStatus(): {
    running: boolean;
    agent_health: Record<string, any>;
    fusion_stats: any;
    decision_stats: any;
    hyperbolic_stats: any;
  } {
    return {
      running: this.isRunning,
      agent_health: this.agentRegistry.getSystemHealth(),
      fusion_stats: this.fusionBrain.getFusionStatistics(),
      decision_stats: this.decisionEngine.getDecisionStatistics(),
      hyperbolic_stats: this.hyperbolicEngine.getStatistics()
    };
  }

  /**
   * Force a manual analysis (for testing/demo)
   */
  async performAnalysis(): Promise<ArbitrageOpportunity | null> {
    try {
      console.log('🔄 Performing manual arbitrage analysis...');
      return await this.executeFusionCycle();
    } catch (error) {
      console.error('❌ Manual analysis failed:', error);
      return null;
    }
  }

  /**
   * Initialize platform components
   */
  private initializeComponents(): void {
    console.log('🔧 Initializing platform components...');
    
    // Initialize agent registry
    this.agentRegistry = new AgentRegistry();
    
    // Initialize hyperbolic engine
    this.hyperbolicEngine = createHyperbolicEngine({
      embedding_dim: this.config.hyperbolic.embedding_dim,
      curvature: this.config.hyperbolic.curvature
    });
    
    // Initialize fusion brain
    this.fusionBrain = createFusionBrain(this.config.llm, this.hyperbolicEngine);
    
    // Initialize decision engine
    this.decisionEngine = createDecisionEngine();
    
    // Create and register agents
    this.createAgents();
  }

  /**
   * Create and register all agents
   */
  private createAgents(): void {
    const enabledAgents = this.config.agents.enabled_agents;
    
    if (enabledAgents.includes('economic')) {
      const economicAgent = createEconomicAgent(
        this.config.api_keys.fred_api_key || DEMO_FRED_API_KEY
      );
      this.agentRegistry.registerAgent(economicAgent);
    }
    
    if (enabledAgents.includes('sentiment')) {
      const sentimentAgent = createSentimentAgent(
        this.config.api_keys.twitter_bearer_token,
        this.config.api_keys.news_api_key
      );
      this.agentRegistry.registerAgent(sentimentAgent);
    }
    
    if (enabledAgents.includes('price')) {
      const priceAgent = createPriceAgent();
      this.agentRegistry.registerAgent(priceAgent);
    }
    
    if (enabledAgents.includes('volume')) {
      const volumeAgent = createVolumeAgent();
      this.agentRegistry.registerAgent(volumeAgent);
    }
    
    if (enabledAgents.includes('trade')) {
      const tradeAgent = createTradeAgent();
      this.agentRegistry.registerAgent(tradeAgent);
    }
    
    if (enabledAgents.includes('image')) {
      const imageAgent = createImageAgent();
      this.agentRegistry.registerAgent(imageAgent);
    }
    
    console.log(`📋 Registered ${enabledAgents.length} agents: ${enabledAgents.join(', ')}`);
  }

  /**
   * Merge user config with platform defaults
   */
  private mergeWithDefaults(userConfig: Partial<PlatformConfig>): PlatformConfig {
    const defaults: PlatformConfig = {
      agents: {
        enabled_agents: ['economic', 'sentiment', 'price', 'volume', 'trade', 'image'],
        polling_intervals: {
          economic: 3600000,  // 1 hour
          sentiment: 30000,   // 30 seconds
          price: 1000,        // 1 second
          volume: 60000,      // 1 minute
          trade: 30000,       // 30 seconds
          image: 60000        // 1 minute
        }
      },
      llm: {
        provider: 'anthropic',
        model: 'claude-3-sonnet-20240229',
        api_key: userConfig.api_keys?.anthropic_api_key || 'demo_key',
        max_tokens: 1000,
        temperature: 0.1,
        timeout_ms: 10000,
        fallback_provider: 'openai',
        fallback_model: 'gpt-4',
        fallback_api_key: userConfig.api_keys?.openai_api_key
      },
      hyperbolic: {
        embedding_dim: 128,
        curvature: 1.0
      },
      execution: {
        sandbox_mode: true,
        max_position_size_usd: 100000
      },
      api_keys: {
        fred_api_key: 'demo_fred_key',
        ...userConfig.api_keys
      }
    };

    return {
      ...defaults,
      ...userConfig,
      agents: { ...defaults.agents, ...userConfig.agents },
      llm: { ...defaults.llm, ...userConfig.llm },
      hyperbolic: { ...defaults.hyperbolic, ...userConfig.hyperbolic },
      execution: { ...defaults.execution, ...userConfig.execution },
      api_keys: { ...defaults.api_keys, ...userConfig.api_keys }
    };
  }

  /**
   * Initialize system metrics
   */
  private initializeMetrics(): void {
    this.systemMetrics = {
      uptime_seconds: 0,
      total_predictions: 0,
      successful_predictions: 0,
      active_agents: 0,
      avg_agent_confidence: 0,
      system_health_score: 0,
      last_fusion_timestamp: new Date().toISOString(),
      circuit_breaker_active: false
    };
  }

  /**
   * Wait for initial agent data before starting fusion
   */
  private async waitForInitialData(maxWaitMs: number = 30000): Promise<void> {
    const startWait = Date.now();
    
    while (Date.now() - startWait < maxWaitMs) {
      const agentOutputs = this.agentRegistry.getAllOutputs();
      const validOutputs = Object.values(agentOutputs).filter(output => output !== null);
      
      // Need at least 3 agents with data to start
      if (validOutputs.length >= 3) {
        console.log(`✅ Initial data ready from ${validOutputs.length} agents`);
        return;
      }
      
      console.log(`⏳ Waiting for agent data... (${validOutputs.length}/3 ready)`);
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
    
    console.warn('⚠️ Starting with limited agent data');
  }

  /**
   * Start the fusion brain analysis loop
   */
  private startFusionLoop(): void {
    const fusionInterval = 10000; // 10 seconds
    
    this.fusionInterval = setInterval(async () => {
      try {
        await this.executeFusionCycle();
      } catch (error) {
        console.error('Fusion cycle error:', error.message);
        this.emit('fusion_error', { error: error.message, timestamp: new Date().toISOString() });
      }
    }, fusionInterval);
  }

  /**
   * Execute a complete fusion analysis cycle
   */
  private async executeFusionCycle(): Promise<ArbitrageOpportunity | null> {
    try {
      // Get latest agent outputs
      const agentOutputs = this.agentRegistry.getAllOutputs();
      
      // Filter valid outputs
      const validOutputs: Record<string, any> = {};
      for (const [name, output] of Object.entries(agentOutputs)) {
        if (output && output.confidence > 0.1) {
          validOutputs[name] = output;
        }
      }
      
      if (Object.keys(validOutputs).length < 2) {
        console.warn('Insufficient agent data for fusion analysis');
        return null;
      }
      
      // Generate fusion prediction
      const prediction = await this.fusionBrain.generatePrediction(validOutputs);
      this.predictionCount++;
      
      // Make execution decision
      const executionPlan = await this.decisionEngine.makeDecision(prediction, validOutputs);
      
      // Create opportunity record
      const opportunity: ArbitrageOpportunity = {
        opportunity_id: executionPlan.decision_id,
        timestamp: prediction.timestamp,
        pair: prediction.arbitrage_plan.pair,
        buy_exchange: prediction.arbitrage_plan.buy_exchange,
        sell_exchange: prediction.arbitrage_plan.sell_exchange,
        predicted_spread_pct: prediction.predicted_spread_pct,
        confidence: prediction.confidence,
        estimated_profit_usd: prediction.arbitrage_plan.estimated_profit_usd,
        risk_score: executionPlan.risk_assessment.risk_score,
        approved_for_execution: executionPlan.approved,
        rationale: prediction.rationale
      };
      
      // Store opportunity
      this.opportunities.push(opportunity);
      this.trimOpportunityHistory();
      
      // Update metrics
      this.systemMetrics.last_fusion_timestamp = prediction.timestamp;
      if (executionPlan.approved) {
        this.successfulPredictions++;
      }
      
      // Emit events
      this.emit('opportunity_detected', opportunity);
      
      if (executionPlan.approved) {
        this.emit('opportunity_approved', opportunity);
        console.log(`💰 APPROVED: ${opportunity.pair} ${opportunity.buy_exchange}->${opportunity.sell_exchange} ${(opportunity.predicted_spread_pct * 100).toFixed(3)}% spread`);
      } else {
        console.log(`❌ REJECTED: ${opportunity.pair} - Risk: ${opportunity.risk_score.toFixed(3)}, Conf: ${opportunity.confidence.toFixed(3)}`);
      }
      
      return opportunity;
      
    } catch (error) {
      console.error('Fusion cycle failed:', error);
      throw error;
    }
  }

  /**
   * Start system health monitoring
   */
  private startHealthMonitoring(): void {
    this.healthCheckInterval = setInterval(() => {
      const systemHealth = this.agentRegistry.getSystemHealth();
      const unhealthyAgents = Object.entries(systemHealth)
        .filter(([, health]) => health.status !== 'healthy')
        .map(([name]) => name);
      
      if (unhealthyAgents.length > 0) {
        console.warn(`⚠️ Unhealthy agents detected: ${unhealthyAgents.join(', ')}`);
        this.emit('agents_unhealthy', { agents: unhealthyAgents, timestamp: new Date().toISOString() });
      }
      
      // Update system metrics
      this.systemMetrics = this.getSystemMetrics();
      
      // Clean up old data
      this.hyperbolicEngine.clearOldEmbeddings(24 * 60 * 60 * 1000); // 24 hours
      
    }, 30000); // Check every 30 seconds
  }

  /**
   * Trim opportunity history to prevent memory bloat
   */
  private trimOpportunityHistory(): void {
    if (this.opportunities.length > 1000) {
      this.opportunities = this.opportunities.slice(-500);
    }
  }

  /**
   * Get platform summary for investor presentations
   */
  getInvestorSummary(): {
    platform_status: string;
    total_opportunities: number;
    approval_rate: number;
    avg_confidence: number;
    system_uptime: string;
    agent_health: string;
    recent_performance: any;
  } {
    const metrics = this.getSystemMetrics();
    const approvalRate = metrics.total_predictions > 0 
      ? (metrics.successful_predictions / metrics.total_predictions) * 100 
      : 0;
    
    const uptimeHours = Math.floor(metrics.uptime_seconds / 3600);
    const uptimeMinutes = Math.floor((metrics.uptime_seconds % 3600) / 60);
    
    return {
      platform_status: this.isRunning ? 'RUNNING' : 'STOPPED',
      total_opportunities: this.opportunities.length,
      approval_rate: approvalRate,
      avg_confidence: metrics.avg_agent_confidence,
      system_uptime: `${uptimeHours}h ${uptimeMinutes}m`,
      agent_health: `${Math.round(metrics.system_health_score * 100)}%`,
      recent_performance: {
        last_24h_opportunities: this.opportunities.filter(
          opp => Date.now() - new Date(opp.timestamp).getTime() < 24 * 60 * 60 * 1000
        ).length
      }
    };
  }
}

/**
 * Factory function to create platform orchestrator
 */
export function createArbitragePlatform(config?: Partial<PlatformConfig>): ArbitragePlatformOrchestrator {
  return new ArbitragePlatformOrchestrator(config);
}

// Export for testing
export { ArbitragePlatformOrchestrator as default };