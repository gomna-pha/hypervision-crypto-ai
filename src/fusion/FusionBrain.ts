import Anthropic from '@anthropic-ai/sdk';
import { EventEmitter } from 'events';
import { Kafka, Consumer, Producer } from 'kafkajs';
import { z } from 'zod';
import Logger from '../utils/logger';
import config from '../utils/ConfigLoader';
import HyperbolicEmbedding from '../hyperbolic/HyperbolicEmbedding';
import {
  FusionInput,
  FusionPrediction,
  AgentOutput,
  EconomicAgentOutput,
  SentimentAgentOutput,
  PriceAgentOutput,
  VolumeAgentOutput,
  TradeAgentOutput,
  ImageAgentOutput,
} from '../types';

// Zod schema for LLM response validation
const FusionPredictionSchema = z.object({
  predicted_spread_pct: z.number(),
  confidence: z.number().min(0).max(1),
  direction: z.enum(['converge', 'diverge']),
  expected_time_s: z.number().positive(),
  arbitrage_plan: z.object({
    buy: z.string(),
    sell: z.string(),
    notional_usd: z.number().positive(),
  }),
  rationale: z.string(),
  risk_flags: z.array(z.string()),
});

interface AgentCache {
  economic?: EconomicAgentOutput;
  sentiment?: SentimentAgentOutput;
  price?: PriceAgentOutput;
  volume?: VolumeAgentOutput;
  trade?: TradeAgentOutput;
  image?: ImageAgentOutput;
  lastUpdate: Map<string, number>;
}

export class FusionBrain extends EventEmitter {
  private logger: Logger;
  private anthropic: Anthropic | null = null;
  private hyperbolic: HyperbolicEmbedding;
  private kafka: Kafka | null = null;
  private consumer: Consumer | null = null;
  private producer: Producer | null = null;
  private agentCache: AgentCache;
  private isRunning: boolean = false;
  private processingInterval: NodeJS.Timeout | null = null;
  private maxDataAge: number = 5000; // 5 seconds

  constructor() {
    super();
    this.logger = Logger.getInstance('FusionBrain');
    this.hyperbolic = new HyperbolicEmbedding();
    this.agentCache = {
      lastUpdate: new Map(),
    };
  }

  async initialize(): Promise<void> {
    try {
      // Initialize Anthropic client
      const apiKey = process.env.ANTHROPIC_API_KEY;
      if (apiKey) {
        this.anthropic = new Anthropic({ apiKey });
        this.logger.info('Anthropic client initialized');
      } else {
        this.logger.warn('ANTHROPIC_API_KEY not set, using mock LLM');
      }

      // Initialize hyperbolic embedding
      await this.hyperbolic.initialize();

      // Initialize Kafka
      await this.initKafka();

      this.logger.info('FusionBrain initialized successfully');
    } catch (error) {
      this.logger.error('Failed to initialize FusionBrain', error);
      throw error;
    }
  }

  private async initKafka(): Promise<void> {
    try {
      const kafkaConfig = config.get('kafka');
      if (!kafkaConfig?.brokers) {
        this.logger.warn('Kafka not configured, running in standalone mode');
        return;
      }

      this.kafka = new Kafka({
        clientId: 'fusion-brain',
        brokers: kafkaConfig.brokers,
      });

      // Initialize consumer
      this.consumer = this.kafka.consumer({
        groupId: kafkaConfig.consumer_group_id || 'fusion-brain-group',
      });

      await this.consumer.connect();
      await this.consumer.subscribe({
        topic: kafkaConfig.topics?.agents_latest || 'agents.latest',
        fromBeginning: false,
      });

      // Initialize producer
      this.producer = this.kafka.producer();
      await this.producer.connect();

      this.logger.info('Kafka connections established');
    } catch (error) {
      this.logger.error('Kafka initialization failed', error);
    }
  }

  async start(): Promise<void> {
    if (this.isRunning) {
      this.logger.warn('FusionBrain is already running');
      return;
    }

    this.isRunning = true;

    // Start consuming agent outputs
    if (this.consumer) {
      this.consumer.run({
        eachMessage: async (payload) => {
          await this.handleAgentMessage(payload);
        },
      });
    }

    // Start processing loop
    this.startProcessingLoop();

    this.logger.info('FusionBrain started');
  }

  private async handleAgentMessage({ message }: any): Promise<void> {
    try {
      const agentOutput = JSON.parse(message.value.toString()) as AgentOutput;
      const agentName = agentOutput.agent_name;

      // Update cache based on agent type
      switch (agentName) {
        case 'EconomicAgent':
          this.agentCache.economic = agentOutput as EconomicAgentOutput;
          break;
        case 'SentimentAgent':
          this.agentCache.sentiment = agentOutput as SentimentAgentOutput;
          break;
        case 'PriceAgent':
          this.agentCache.price = agentOutput as PriceAgentOutput;
          break;
        case 'VolumeAgent':
          this.agentCache.volume = agentOutput as VolumeAgentOutput;
          break;
        case 'TradeAgent':
          this.agentCache.trade = agentOutput as TradeAgentOutput;
          break;
        case 'ImageAgent':
          this.agentCache.image = agentOutput as ImageAgentOutput;
          break;
      }

      this.agentCache.lastUpdate.set(agentName, Date.now());

      // Update hyperbolic embeddings
      await this.hyperbolic.embed(
        `${agentName}-${Date.now()}`,
        agentOutput.features as Record<string, number>
      );

      this.logger.debug(`Received update from ${agentName}`, {
        key_signal: agentOutput.key_signal,
        confidence: agentOutput.confidence,
      });
    } catch (error) {
      this.logger.error('Failed to handle agent message', error);
    }
  }

  private startProcessingLoop(): void {
    const intervalMs = 1000; // Process every second

    this.processingInterval = setInterval(async () => {
      try {
        await this.processFusion();
      } catch (error) {
        this.logger.error('Fusion processing failed', error);
      }
    }, intervalMs);
  }

  private async processFusion(): Promise<void> {
    // Check if we have fresh data from required agents
    if (!this.hasValidData()) {
      return;
    }

    // Build fusion input
    const fusionInput = await this.buildFusionInput();

    // Get LLM prediction
    const prediction = await this.getLLMPrediction(fusionInput);
    if (!prediction) return;

    // Publish prediction
    await this.publishPrediction(prediction);

    this.logger.info('Fusion prediction generated', {
      spread: prediction.predicted_spread_pct,
      confidence: prediction.confidence,
      direction: prediction.direction,
    });
  }

  private hasValidData(): boolean {
    const now = Date.now();
    const requiredAgents = ['PriceAgent']; // Minimum required
    
    for (const agent of requiredAgents) {
      const lastUpdate = this.agentCache.lastUpdate.get(agent) || 0;
      if (now - lastUpdate > this.maxDataAge) {
        return false;
      }
    }

    return true;
  }

  private async buildFusionInput(): Promise<FusionInput> {
    // Get hyperbolic neighbors for context
    const currentEmbedding = await this.hyperbolic.embed(
      'fusion-query',
      this.extractCurrentFeatures()
    );

    const neighbors = await this.hyperbolic.findNeighbors(currentEmbedding, 5);

    return {
      economic_agent: this.agentCache.economic,
      sentiment_agent: this.agentCache.sentiment,
      price_agent: this.agentCache.price,
      volume_agent: this.agentCache.volume,
      trade_agent: this.agentCache.trade,
      image_agent: this.agentCache.image,
      hyperbolic_neighbors: neighbors,
    };
  }

  private extractCurrentFeatures(): Record<string, number> {
    const features: Record<string, number> = {};

    if (this.agentCache.economic) {
      features.economic_signal = this.agentCache.economic.key_signal;
      features.economic_confidence = this.agentCache.economic.confidence;
    }

    if (this.agentCache.sentiment) {
      features.sentiment_signal = this.agentCache.sentiment.key_signal;
      features.sentiment_confidence = this.agentCache.sentiment.confidence;
    }

    if (this.agentCache.price) {
      features.price_spread = 
        (this.agentCache.price.best_ask - this.agentCache.price.best_bid) / 
        this.agentCache.price.mid_price;
      features.price_depth = this.agentCache.price.orderbook_depth_usd;
    }

    return features;
  }

  private async getLLMPrediction(input: FusionInput): Promise<FusionPrediction | null> {
    try {
      const prompt = this.buildPrompt(input);

      if (!this.anthropic) {
        // Return mock prediction for development
        return this.getMockPrediction(input);
      }

      const response = await this.anthropic.messages.create({
        model: 'claude-3-sonnet-20240229',
        max_tokens: 1000,
        temperature: 0.1,
        system: config.get('llm.system_prompt'),
        messages: [
          {
            role: 'user',
            content: prompt,
          },
        ],
      });

      // Parse JSON response
      const content = response.content[0];
      if (content.type === 'text') {
        const jsonStr = content.text.trim();
        const prediction = JSON.parse(jsonStr);
        
        // Validate with Zod
        return FusionPredictionSchema.parse(prediction);
      }

      throw new Error('Invalid LLM response format');
    } catch (error) {
      this.logger.error('LLM prediction failed', error);
      return null;
    }
  }

  private buildPrompt(input: FusionInput): string {
    const data: any = {
      timestamp: new Date().toISOString(),
    };

    if (input.economic_agent) {
      data.economic_agent = {
        key_signal: input.economic_agent.key_signal,
        CPI: input.economic_agent.signals?.CPI,
        FEDFUNDS: input.economic_agent.signals?.FEDFUNDS,
        features: input.economic_agent.features,
      };
    }

    if (input.sentiment_agent) {
      data.sentiment_agent = {
        key_signal: input.sentiment_agent.key_signal,
        features: input.sentiment_agent.features,
      };
    }

    if (input.price_agent) {
      data.price_agent = {
        exchange: input.price_agent.exchange,
        mid_price: input.price_agent.mid_price,
        spread: input.price_agent.best_ask - input.price_agent.best_bid,
        depth: input.price_agent.orderbook_depth_usd,
      };
    }

    if (input.volume_agent) {
      data.volume_agent = {
        liquidity_index: input.volume_agent.features.liquidity_index,
        volume_1m: input.volume_agent.features.volume_1m,
      };
    }

    return JSON.stringify(data, null, 2);
  }

  private getMockPrediction(input: FusionInput): FusionPrediction {
    // Generate mock prediction based on input data
    const hasGoodSignals = 
      (input.economic_agent?.key_signal || 0) > 0 &&
      (input.sentiment_agent?.key_signal || 0) > 0.5;

    const spread = input.price_agent 
      ? (input.price_agent.best_ask - input.price_agent.best_bid) / input.price_agent.mid_price
      : 0.001;

    return {
      predicted_spread_pct: spread * 1.5 + (hasGoodSignals ? 0.002 : 0),
      confidence: 0.75 + Math.random() * 0.15,
      direction: Math.random() > 0.5 ? 'converge' : 'diverge',
      expected_time_s: 300 + Math.floor(Math.random() * 300),
      arbitrage_plan: {
        buy: 'binance',
        sell: 'coinbase',
        notional_usd: 50000,
      },
      rationale: 'Mock prediction based on positive signals and market depth',
      risk_flags: hasGoodSignals ? [] : ['low_confidence_signals'],
    };
  }

  private async publishPrediction(prediction: FusionPrediction): Promise<void> {
    this.emit('prediction', prediction);

    if (this.producer) {
      try {
        const kafkaConfig = config.get('kafka');
        const topic = kafkaConfig?.topics?.fusion_predictions || 'fusion.predictions';

        await this.producer.send({
          topic,
          messages: [
            {
              key: 'fusion-prediction',
              value: JSON.stringify(prediction),
              timestamp: Date.now().toString(),
            },
          ],
        });

        this.logger.debug('Published prediction to Kafka');
      } catch (error) {
        this.logger.error('Failed to publish prediction to Kafka', error);
      }
    }
  }

  async stop(): Promise<void> {
    this.isRunning = false;

    if (this.processingInterval) {
      clearInterval(this.processingInterval);
    }

    if (this.consumer) {
      await this.consumer.disconnect();
    }

    if (this.producer) {
      await this.producer.disconnect();
    }

    await this.hyperbolic.cleanup();

    this.logger.info('FusionBrain stopped');
  }
}

// Export for standalone execution
if (require.main === module) {
  const fusion = new FusionBrain();
  
  fusion.initialize().then(() => {
    return fusion.start();
  }).catch(error => {
    console.error('Failed to start FusionBrain:', error);
    process.exit(1);
  });

  process.on('SIGINT', async () => {
    await fusion.stop();
    process.exit(0);
  });
}

export default FusionBrain;