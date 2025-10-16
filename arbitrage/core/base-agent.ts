/**
 * Base Agent Framework for LLM Arbitrage Platform
 * Provides common functionality for all agents including REST endpoints,
 * Kafka publishing, error handling, and health checks.
 */

import { EventEmitter } from 'events';
import yaml from 'yaml';
import { readFileSync } from 'fs';

export interface AgentConfig {
  name: string;
  enabled: boolean;
  polling_interval_ms: number;
  confidence_min: number;
  data_age_max_ms: number;
  retry_attempts: number;
  retry_backoff_ms: number;
}

export interface AgentOutput {
  agent_name: string;
  timestamp: string;
  key_signal: number;      // normalized [-1..1] or [0..1]
  confidence: number;      // 0..1
  features: Record<string, any>; // agent-specific numeric features
  metadata?: Record<string, any>;
}

export interface AgentHealthStatus {
  agent_name: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  last_update: string;
  error_count: number;
  uptime_seconds: number;
  data_freshness_ms: number;
}

export abstract class BaseAgent extends EventEmitter {
  protected config: AgentConfig;
  protected isRunning: boolean = false;
  protected lastOutput: AgentOutput | null = null;
  protected errorCount: number = 0;
  protected startTime: number = Date.now();
  protected intervalId: NodeJS.Timeout | null = null;

  constructor(config: AgentConfig) {
    super();
    this.config = config;
  }

  /**
   * Abstract method to be implemented by each agent
   * Should fetch data and return structured output
   */
  protected abstract collectData(): Promise<AgentOutput>;

  /**
   * Start the agent's data collection loop
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      throw new Error(`Agent ${this.config.name} is already running`);
    }

    if (!this.config.enabled) {
      console.log(`Agent ${this.config.name} is disabled, not starting`);
      return;
    }

    this.isRunning = true;
    this.startTime = Date.now();
    
    console.log(`Starting agent: ${this.config.name}`);
    
    // Initial data collection
    await this.executeDataCollection();
    
    // Set up periodic collection
    this.intervalId = setInterval(async () => {
      await this.executeDataCollection();
    }, this.config.polling_interval_ms);

    this.emit('started', this.config.name);
  }

  /**
   * Stop the agent's data collection
   */
  async stop(): Promise<void> {
    if (!this.isRunning) {
      return;
    }

    this.isRunning = false;
    
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;
    }

    console.log(`Stopped agent: ${this.config.name}`);
    this.emit('stopped', this.config.name);
  }

  /**
   * Execute data collection with error handling and retry logic
   */
  private async executeDataCollection(): Promise<void> {
    let attempts = 0;
    const maxAttempts = this.config.retry_attempts || 3;

    while (attempts < maxAttempts) {
      try {
        const output = await this.collectData();
        
        // Validate output
        if (!this.validateOutput(output)) {
          throw new Error('Invalid agent output format');
        }

        // Check confidence threshold
        if (output.confidence < this.config.confidence_min) {
          console.warn(`Agent ${this.config.name} confidence ${output.confidence} below threshold ${this.config.confidence_min}`);
        }

        this.lastOutput = output;
        this.errorCount = 0; // Reset error count on success
        
        // Publish to Kafka (simulate for now)
        await this.publishOutput(output);
        
        // Emit event for subscribers
        this.emit('data', output);
        
        break; // Success, exit retry loop
        
      } catch (error) {
        attempts++;
        this.errorCount++;
        
        console.error(`Agent ${this.config.name} error (attempt ${attempts}/${maxAttempts}):`, error);
        
        if (attempts >= maxAttempts) {
          this.emit('error', {
            agent_name: this.config.name,
            error: error.message,
            attempts: attempts
          });
        } else {
          // Wait before retry with exponential backoff
          const backoffMs = this.config.retry_backoff_ms * Math.pow(2, attempts - 1);
          await this.sleep(backoffMs);
        }
      }
    }
  }

  /**
   * Validate agent output format
   */
  private validateOutput(output: AgentOutput): boolean {
    return (
      typeof output.agent_name === 'string' &&
      typeof output.timestamp === 'string' &&
      typeof output.key_signal === 'number' &&
      typeof output.confidence === 'number' &&
      output.confidence >= 0 && output.confidence <= 1 &&
      typeof output.features === 'object' &&
      output.features !== null
    );
  }

  /**
   * Publish output to Kafka (simplified implementation)
   */
  private async publishOutput(output: AgentOutput): Promise<void> {
    // TODO: Implement actual Kafka publishing
    // For now, just log the output
    console.log(`[KAFKA] Publishing to agents.latest:`, JSON.stringify(output, null, 2));
  }

  /**
   * Get latest agent output
   */
  getLatestOutput(): AgentOutput | null {
    return this.lastOutput;
  }

  /**
   * Get agent health status
   */
  getHealthStatus(): AgentHealthStatus {
    const now = Date.now();
    const uptimeSeconds = Math.floor((now - this.startTime) / 1000);
    const dataFreshnessMs = this.lastOutput 
      ? now - new Date(this.lastOutput.timestamp).getTime()
      : Infinity;

    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    
    if (!this.isRunning) {
      status = 'unhealthy';
    } else if (this.errorCount > 5 || dataFreshnessMs > this.config.data_age_max_ms) {
      status = 'degraded';
    } else if (this.errorCount > 10) {
      status = 'unhealthy';
    }

    return {
      agent_name: this.config.name,
      status,
      last_update: this.lastOutput?.timestamp || 'never',
      error_count: this.errorCount,
      uptime_seconds: uptimeSeconds,
      data_freshness_ms: dataFreshnessMs
    };
  }

  /**
   * REST endpoint handler for getting latest output
   */
  handleLatestRequest(): AgentOutput | null {
    const output = this.getLatestOutput();
    
    if (!output) {
      return null;
    }

    // Check if data is fresh enough
    const age = Date.now() - new Date(output.timestamp).getTime();
    if (age > this.config.data_age_max_ms) {
      console.warn(`Agent ${this.config.name} data is stale (age: ${age}ms)`);
      return null;
    }

    return output;
  }

  /**
   * REST endpoint handler for health check
   */
  handleHealthRequest(): AgentHealthStatus {
    return this.getHealthStatus();
  }

  /**
   * Utility function for sleeping
   */
  protected sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get current timestamp in ISO format
   */
  protected getCurrentTimestamp(): string {
    return new Date().toISOString();
  }

  /**
   * Normalize value to range [0, 1] or [-1, 1]
   */
  protected normalize(value: number, min: number, max: number, bipolar: boolean = false): number {
    if (max === min) return bipolar ? 0 : 0.5;
    
    const normalized = (value - min) / (max - min);
    
    if (bipolar) {
      return (normalized * 2) - 1; // Convert to [-1, 1]
    }
    
    return Math.max(0, Math.min(1, normalized)); // Clamp to [0, 1]
  }

  /**
   * Calculate confidence based on data quality metrics
   */
  protected calculateConfidence(
    dataAge: number, 
    sampleSize: number = 1, 
    errorRate: number = 0
  ): number {
    // Age factor (newer data = higher confidence)
    const ageFactor = Math.max(0, 1 - (dataAge / this.config.data_age_max_ms));
    
    // Sample size factor (more data = higher confidence)
    const sizeFactor = Math.min(1, sampleSize / 10);
    
    // Error factor (fewer errors = higher confidence)
    const errorFactor = Math.max(0, 1 - errorRate);
    
    return Math.max(0.1, ageFactor * sizeFactor * errorFactor);
  }
}

/**
 * Agent Registry for managing multiple agents
 */
export class AgentRegistry {
  private agents: Map<string, BaseAgent> = new Map();
  private config: any;

  constructor(configPath: string = './arbitrage/config/platform.yaml') {
    try {
      const configFile = readFileSync(configPath, 'utf8');
      this.config = yaml.parse(configFile);
    } catch (error) {
      console.error('Failed to load configuration:', error);
      this.config = {};
    }
  }

  /**
   * Register an agent with the registry
   */
  registerAgent(agent: BaseAgent): void {
    const name = agent['config'].name;
    this.agents.set(name, agent);
    
    // Set up event listeners
    agent.on('data', (output) => {
      console.log(`Agent ${name} published data: confidence=${output.confidence}`);
    });
    
    agent.on('error', (error) => {
      console.error(`Agent ${name} error:`, error);
    });
  }

  /**
   * Start all registered agents
   */
  async startAll(): Promise<void> {
    const promises = Array.from(this.agents.values()).map(agent => agent.start());
    await Promise.all(promises);
    console.log(`Started ${this.agents.size} agents`);
  }

  /**
   * Stop all registered agents
   */
  async stopAll(): Promise<void> {
    const promises = Array.from(this.agents.values()).map(agent => agent.stop());
    await Promise.all(promises);
    console.log(`Stopped ${this.agents.size} agents`);
  }

  /**
   * Get agent by name
   */
  getAgent(name: string): BaseAgent | undefined {
    return this.agents.get(name);
  }

  /**
   * Get all agent outputs
   */
  getAllOutputs(): Record<string, AgentOutput | null> {
    const outputs: Record<string, AgentOutput | null> = {};
    
    for (const [name, agent] of this.agents) {
      outputs[name] = agent.getLatestOutput();
    }
    
    return outputs;
  }

  /**
   * Get system health status
   */
  getSystemHealth(): Record<string, AgentHealthStatus> {
    const health: Record<string, AgentHealthStatus> = {};
    
    for (const [name, agent] of this.agents) {
      health[name] = agent.getHealthStatus();
    }
    
    return health;
  }

  /**
   * Check if all required agents are healthy
   */
  isSystemHealthy(): boolean {
    const requiredAgents = ['economic', 'sentiment', 'price', 'volume', 'trade'];
    
    for (const agentName of requiredAgents) {
      const agent = this.agents.get(agentName);
      if (!agent || agent.getHealthStatus().status === 'unhealthy') {
        return false;
      }
    }
    
    return true;
  }
}