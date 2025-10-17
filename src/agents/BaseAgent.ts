import { EventEmitter } from 'events';
import express, { Express, Request, Response } from 'express';
import { Kafka, Producer, Consumer, EachMessagePayload } from 'kafkajs';
import Logger from '../utils/logger';
import config from '../utils/ConfigLoader';
import { BaseAgentOutput } from '../types';

export abstract class BaseAgent extends EventEmitter {
  protected name: string;
  protected logger: Logger;
  protected kafka: Kafka | null = null;
  protected producer: Producer | null = null;
  protected consumer: Consumer | null = null;
  protected app: Express;
  protected port: number;
  protected lastOutput: BaseAgentOutput | null = null;
  protected isRunning: boolean = false;
  protected pollingInterval: NodeJS.Timeout | null = null;

  constructor(name: string, port: number) {
    super();
    this.name = name;
    this.port = port;
    this.logger = Logger.getInstance(name);
    this.app = express();
    this.setupRoutes();
    this.initKafka();
  }

  private async initKafka(): Promise<void> {
    try {
      const kafkaConfig = config.get('kafka');
      if (kafkaConfig && kafkaConfig.brokers) {
        this.kafka = new Kafka({
          clientId: `${this.name}-client`,
          brokers: kafkaConfig.brokers,
          retry: {
            initialRetryTime: 100,
            retries: 8,
          },
        });

        this.producer = this.kafka.producer();
        await this.producer.connect();
        
        this.logger.info('Kafka producer connected');
      }
    } catch (error) {
      this.logger.warn('Kafka initialization failed, running in standalone mode', error);
    }
  }

  private setupRoutes(): void {
    this.app.use(express.json());

    // Health check endpoint
    this.app.get('/health', (req: Request, res: Response) => {
      res.json({
        status: 'healthy',
        agent: this.name,
        running: this.isRunning,
        lastUpdate: this.lastOutput?.timestamp || null,
      });
    });

    // Get latest output
    this.app.get(`/agents/${this.name.toLowerCase()}/latest`, (req: Request, res: Response) => {
      if (this.lastOutput) {
        res.json(this.lastOutput);
      } else {
        res.status(404).json({ error: 'No data available yet' });
      }
    });

    // Get agent configuration
    this.app.get(`/agents/${this.name.toLowerCase()}/config`, (req: Request, res: Response) => {
      const agentConfig = config.get(`agents.${this.name.toLowerCase().replace('Agent', '')}`);
      res.json(agentConfig || {});
    });

    // Trigger manual update
    this.app.post(`/agents/${this.name.toLowerCase()}/update`, async (req: Request, res: Response) => {
      try {
        await this.update();
        res.json({ message: 'Update triggered successfully' });
      } catch (error: any) {
        res.status(500).json({ error: error.message });
      }
    });
  }

  protected async publishOutput(output: BaseAgentOutput): Promise<void> {
    this.lastOutput = output;
    this.emit('output', output);

    // Publish to Kafka if available
    if (this.producer) {
      try {
        const kafkaConfig = config.get('kafka');
        const topic = kafkaConfig?.topics?.agents_latest || 'agents.latest';
        
        await this.producer.send({
          topic,
          messages: [
            {
              key: this.name,
              value: JSON.stringify(output),
              timestamp: Date.now().toString(),
            },
          ],
        });

        this.logger.debug(`Published output to Kafka topic ${topic}`);
      } catch (error) {
        this.logger.error('Failed to publish to Kafka', error);
      }
    }

    // Log metrics
    this.logger.metric(`${this.name}.key_signal`, output.key_signal);
    this.logger.metric(`${this.name}.confidence`, output.confidence);
  }

  protected createOutput(
    keySignal: number, 
    confidence: number, 
    features: Record<string, any>
  ): BaseAgentOutput {
    return {
      agent_name: this.name,
      timestamp: new Date().toISOString(),
      key_signal: keySignal,
      confidence: confidence,
      features,
    };
  }

  public async start(): Promise<void> {
    if (this.isRunning) {
      this.logger.warn('Agent is already running');
      return;
    }

    this.isRunning = true;

    // Start HTTP server
    this.app.listen(this.port, () => {
      this.logger.info(`${this.name} HTTP server listening on port ${this.port}`);
    });

    // Initialize agent-specific components
    await this.initialize();

    // Start polling if configured
    const pollingMs = this.getPollingInterval();
    if (pollingMs > 0) {
      this.startPolling(pollingMs);
    }

    this.logger.info(`${this.name} started successfully`);
  }

  public async stop(): Promise<void> {
    this.isRunning = false;

    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
    }

    if (this.producer) {
      await this.producer.disconnect();
    }

    if (this.consumer) {
      await this.consumer.disconnect();
    }

    await this.cleanup();

    this.logger.info(`${this.name} stopped`);
  }

  private startPolling(intervalMs: number): void {
    this.pollingInterval = setInterval(async () => {
      try {
        await this.update();
      } catch (error) {
        this.logger.error('Polling update failed', error);
      }
    }, intervalMs);

    // Initial update
    this.update().catch(error => {
      this.logger.error('Initial update failed', error);
    });
  }

  // Abstract methods to be implemented by child classes
  protected abstract initialize(): Promise<void>;
  protected abstract update(): Promise<void>;
  protected abstract cleanup(): Promise<void>;
  protected abstract getPollingInterval(): number;

  // Helper methods for child classes
  protected async retryWithBackoff<T>(
    fn: () => Promise<T>,
    maxRetries: number = 3,
    initialDelay: number = 1000
  ): Promise<T> {
    let lastError: Error | undefined;
    
    for (let i = 0; i < maxRetries; i++) {
      try {
        return await fn();
      } catch (error: any) {
        lastError = error;
        const delay = initialDelay * Math.pow(2, i);
        this.logger.warn(`Retry ${i + 1}/${maxRetries} after ${delay}ms`, error.message);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }

    throw lastError;
  }

  protected validateDataFreshness(timestamp: string | number, maxAgeMs: number): boolean {
    const dataTime = typeof timestamp === 'string' 
      ? new Date(timestamp).getTime() 
      : timestamp;
    
    const now = Date.now();
    const age = now - dataTime;
    
    return age <= maxAgeMs;
  }

  protected normalizeSignal(value: number, min: number, max: number): number {
    if (max === min) return 0;
    return Math.max(-1, Math.min(1, (2 * (value - min) / (max - min)) - 1));
  }
}