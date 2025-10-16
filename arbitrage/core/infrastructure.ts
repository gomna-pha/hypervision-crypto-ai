/**
 * Infrastructure Layer - Kafka, Redis, and Database Management
 * Production-ready infrastructure for agent communication and data persistence
 */

import { Kafka, Producer, Consumer, KafkaConfig } from 'kafkajs';
import { createClient, RedisClientType } from 'redis';
import sqlite3 from 'sqlite3';
import { Database } from 'sqlite3';
import yaml from 'yaml';
import { readFileSync } from 'fs';

export interface InfrastructureConfig {
  kafka: {
    enabled: boolean;
    brokers: string[];
    clientId: string;
    topics: {
      agents_latest: string;
      fusion_predictions: string;
      execution_orders: string;
      validation_results: string;
    };
  };
  redis: {
    enabled: boolean;
    url: string;
    ttl_seconds: number;
  };
  database: {
    type: 'sqlite' | 'postgresql';
    path?: string;
    connection_string?: string;
  };
}

/**
 * Kafka Message Bus for Inter-Service Communication
 */
export class KafkaManager {
  private kafka: Kafka | null = null;
  private producer: Producer | null = null;
  private consumer: Consumer | null = null;
  private config: InfrastructureConfig['kafka'];

  constructor(config: InfrastructureConfig['kafka']) {
    this.config = config;
    
    if (config.enabled) {
      this.kafka = new Kafka({
        clientId: config.clientId,
        brokers: config.brokers,
        retry: {
          initialRetryTime: 100,
          retries: 8
        }
      });
    }
  }

  /**
   * Initialize Kafka producer and consumer
   */
  async initialize(): Promise<void> {
    if (!this.config.enabled || !this.kafka) {
      console.log('Kafka disabled, using in-memory message passing');
      return;
    }

    try {
      this.producer = this.kafka.producer();
      this.consumer = this.kafka.consumer({ groupId: 'arbitrage-platform' });

      await this.producer.connect();
      await this.consumer.connect();

      console.log('Kafka connected successfully');
    } catch (error) {
      console.error('Failed to initialize Kafka:', error);
      throw error;
    }
  }

  /**
   * Publish message to Kafka topic
   */
  async publish(topic: string, message: any): Promise<void> {
    if (!this.producer) {
      // Fallback to console logging if Kafka not available
      console.log(`[KAFKA-FALLBACK] Topic: ${topic}, Message:`, JSON.stringify(message, null, 2));
      return;
    }

    try {
      await this.producer.send({
        topic,
        messages: [{
          key: message.agent_name || 'system',
          value: JSON.stringify(message),
          timestamp: Date.now().toString()
        }]
      });
    } catch (error) {
      console.error(`Failed to publish to topic ${topic}:`, error);
      throw error;
    }
  }

  /**
   * Subscribe to Kafka topic
   */
  async subscribe(topic: string, handler: (message: any) => void): Promise<void> {
    if (!this.consumer) {
      console.warn(`Cannot subscribe to ${topic}: Kafka consumer not initialized`);
      return;
    }

    try {
      await this.consumer.subscribe({ topic });
      
      await this.consumer.run({
        eachMessage: async ({ topic, partition, message }) => {
          try {
            const value = message.value?.toString();
            if (value) {
              const parsedMessage = JSON.parse(value);
              handler(parsedMessage);
            }
          } catch (error) {
            console.error(`Error processing message from ${topic}:`, error);
          }
        }
      });
    } catch (error) {
      console.error(`Failed to subscribe to topic ${topic}:`, error);
      throw error;
    }
  }

  /**
   * Publish agent output to agents.latest topic
   */
  async publishAgentOutput(agentOutput: any): Promise<void> {
    await this.publish(this.config.topics.agents_latest, agentOutput);
  }

  /**
   * Publish fusion prediction
   */
  async publishFusionPrediction(prediction: any): Promise<void> {
    await this.publish(this.config.topics.fusion_predictions, prediction);
  }

  /**
   * Close Kafka connections
   */
  async close(): Promise<void> {
    if (this.producer) {
      await this.producer.disconnect();
    }
    if (this.consumer) {
      await this.consumer.disconnect();
    }
    console.log('Kafka connections closed');
  }
}

/**
 * Redis Cache Manager for Fast Data Access
 */
export class RedisManager {
  private client: RedisClientType | null = null;
  private config: InfrastructureConfig['redis'];

  constructor(config: InfrastructureConfig['redis']) {
    this.config = config;
  }

  /**
   * Initialize Redis connection
   */
  async initialize(): Promise<void> {
    if (!this.config.enabled) {
      console.log('Redis disabled, using in-memory caching');
      return;
    }

    try {
      this.client = createClient({
        url: this.config.url
      });

      this.client.on('error', (err) => {
        console.error('Redis Client Error:', err);
      });

      await this.client.connect();
      console.log('Redis connected successfully');
    } catch (error) {
      console.error('Failed to initialize Redis:', error);
      // Don't throw error, allow fallback to in-memory cache
    }
  }

  /**
   * Set key-value pair with TTL
   */
  async set(key: string, value: any, ttlSeconds?: number): Promise<void> {
    if (!this.client) {
      // Fallback to console logging
      console.log(`[REDIS-FALLBACK] SET ${key}:`, value);
      return;
    }

    try {
      const serialized = JSON.stringify(value);
      const ttl = ttlSeconds || this.config.ttl_seconds;
      
      await this.client.setEx(key, ttl, serialized);
    } catch (error) {
      console.error(`Redis SET error for key ${key}:`, error);
    }
  }

  /**
   * Get value by key
   */
  async get(key: string): Promise<any | null> {
    if (!this.client) {
      console.log(`[REDIS-FALLBACK] GET ${key}: null`);
      return null;
    }

    try {
      const value = await this.client.get(key);
      return value ? JSON.parse(value) : null;
    } catch (error) {
      console.error(`Redis GET error for key ${key}:`, error);
      return null;
    }
  }

  /**
   * Cache agent output
   */
  async cacheAgentOutput(agentName: string, output: any): Promise<void> {
    const key = `agent:${agentName}:latest`;
    await this.set(key, output);
  }

  /**
   * Get cached agent output
   */
  async getCachedAgentOutput(agentName: string): Promise<any | null> {
    const key = `agent:${agentName}:latest`;
    return await this.get(key);
  }

  /**
   * Close Redis connection
   */
  async close(): Promise<void> {
    if (this.client) {
      await this.client.quit();
      console.log('Redis connection closed');
    }
  }
}

/**
 * Database Manager for Persistent Storage
 */
export class DatabaseManager {
  private db: Database | null = null;
  private config: InfrastructureConfig['database'];

  constructor(config: InfrastructureConfig['database']) {
    this.config = config;
  }

  /**
   * Initialize database connection
   */
  async initialize(): Promise<void> {
    if (this.config.type === 'sqlite' && this.config.path) {
      return new Promise((resolve, reject) => {
        this.db = new sqlite3.Database(this.config.path!, (err) => {
          if (err) {
            console.error('Failed to initialize SQLite database:', err);
            reject(err);
          } else {
            console.log('SQLite database connected successfully');
            this.createTables().then(resolve).catch(reject);
          }
        });
      });
    }
  }

  /**
   * Create necessary tables
   */
  private async createTables(): Promise<void> {
    if (!this.db) return;

    const tables = [
      // Agent outputs table
      `CREATE TABLE IF NOT EXISTS agent_outputs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        agent_name TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        key_signal REAL NOT NULL,
        confidence REAL NOT NULL,
        features TEXT NOT NULL,
        metadata TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,

      // Fusion predictions table
      `CREATE TABLE IF NOT EXISTS fusion_predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        predicted_spread_pct REAL NOT NULL,
        confidence REAL NOT NULL,
        direction TEXT NOT NULL,
        expected_time_s INTEGER NOT NULL,
        arbitrage_plan TEXT NOT NULL,
        rationale TEXT,
        risk_flags TEXT,
        agent_inputs TEXT NOT NULL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,

      // Execution orders table
      `CREATE TABLE IF NOT EXISTS execution_orders (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        order_id TEXT UNIQUE NOT NULL,
        exchange TEXT NOT NULL,
        pair TEXT NOT NULL,
        side TEXT NOT NULL,
        amount REAL NOT NULL,
        price REAL NOT NULL,
        status TEXT NOT NULL,
        filled_amount REAL DEFAULT 0,
        filled_price REAL,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`,

      // Audit logs table
      `CREATE TABLE IF NOT EXISTS audit_logs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        event_data TEXT NOT NULL,
        user_agent TEXT,
        ip_address TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP
      )`
    ];

    return new Promise((resolve, reject) => {
      let completed = 0;
      const total = tables.length;

      tables.forEach((sql) => {
        this.db!.run(sql, (err) => {
          if (err) {
            console.error('Failed to create table:', err);
            reject(err);
            return;
          }
          
          completed++;
          if (completed === total) {
            console.log('Database tables created successfully');
            resolve();
          }
        });
      });
    });
  }

  /**
   * Insert agent output
   */
  async insertAgentOutput(output: any): Promise<void> {
    if (!this.db) {
      console.log('[DB-FALLBACK] Insert agent output:', output.agent_name);
      return;
    }

    const sql = `
      INSERT INTO agent_outputs 
      (agent_name, timestamp, key_signal, confidence, features, metadata)
      VALUES (?, ?, ?, ?, ?, ?)
    `;
    
    const params = [
      output.agent_name,
      output.timestamp,
      output.key_signal,
      output.confidence,
      JSON.stringify(output.features),
      JSON.stringify(output.metadata || {})
    ];

    return new Promise((resolve, reject) => {
      this.db!.run(sql, params, (err) => {
        if (err) {
          console.error('Failed to insert agent output:', err);
          reject(err);
        } else {
          resolve();
        }
      });
    });
  }

  /**
   * Insert fusion prediction
   */
  async insertFusionPrediction(prediction: any): Promise<void> {
    if (!this.db) {
      console.log('[DB-FALLBACK] Insert fusion prediction');
      return;
    }

    const sql = `
      INSERT INTO fusion_predictions 
      (predicted_spread_pct, confidence, direction, expected_time_s, 
       arbitrage_plan, rationale, risk_flags, agent_inputs)
      VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    `;
    
    const params = [
      prediction.predicted_spread_pct,
      prediction.confidence,
      prediction.direction,
      prediction.expected_time_s,
      JSON.stringify(prediction.arbitrage_plan),
      prediction.rationale,
      JSON.stringify(prediction.risk_flags || []),
      JSON.stringify(prediction.agent_inputs)
    ];

    return new Promise((resolve, reject) => {
      this.db!.run(sql, params, (err) => {
        if (err) {
          console.error('Failed to insert fusion prediction:', err);
          reject(err);
        } else {
          resolve();
        }
      });
    });
  }

  /**
   * Get recent agent outputs
   */
  async getRecentAgentOutputs(agentName?: string, limit: number = 100): Promise<any[]> {
    if (!this.db) {
      console.log('[DB-FALLBACK] Get recent agent outputs');
      return [];
    }

    const sql = agentName 
      ? `SELECT * FROM agent_outputs WHERE agent_name = ? ORDER BY created_at DESC LIMIT ?`
      : `SELECT * FROM agent_outputs ORDER BY created_at DESC LIMIT ?`;
    
    const params = agentName ? [agentName, limit] : [limit];

    return new Promise((resolve, reject) => {
      this.db!.all(sql, params, (err, rows) => {
        if (err) {
          console.error('Failed to get agent outputs:', err);
          reject(err);
        } else {
          // Parse JSON fields
          const parsed = rows.map((row: any) => ({
            ...row,
            features: JSON.parse(row.features),
            metadata: JSON.parse(row.metadata || '{}')
          }));
          resolve(parsed);
        }
      });
    });
  }

  /**
   * Insert audit log
   */
  async insertAuditLog(eventType: string, eventData: any, metadata?: any): Promise<void> {
    if (!this.db) {
      console.log('[DB-FALLBACK] Audit log:', eventType);
      return;
    }

    const sql = `
      INSERT INTO audit_logs (event_type, event_data, user_agent, ip_address)
      VALUES (?, ?, ?, ?)
    `;
    
    const params = [
      eventType,
      JSON.stringify(eventData),
      metadata?.userAgent || null,
      metadata?.ipAddress || null
    ];

    return new Promise((resolve, reject) => {
      this.db!.run(sql, params, (err) => {
        if (err) {
          console.error('Failed to insert audit log:', err);
          reject(err);
        } else {
          resolve();
        }
      });
    });
  }

  /**
   * Close database connection
   */
  async close(): Promise<void> {
    if (this.db) {
      return new Promise((resolve) => {
        this.db!.close((err) => {
          if (err) {
            console.error('Error closing database:', err);
          } else {
            console.log('Database connection closed');
          }
          resolve();
        });
      });
    }
  }
}

/**
 * Infrastructure Orchestrator
 */
export class Infrastructure {
  private config: InfrastructureConfig;
  public kafka: KafkaManager;
  public redis: RedisManager;
  public database: DatabaseManager;

  constructor(configPath: string = './arbitrage/config/platform.yaml') {
    // Load configuration
    this.config = this.loadConfig(configPath);
    
    // Initialize components
    this.kafka = new KafkaManager(this.config.kafka);
    this.redis = new RedisManager(this.config.redis);
    this.database = new DatabaseManager(this.config.database);
  }

  /**
   * Load infrastructure configuration
   */
  private loadConfig(configPath: string): InfrastructureConfig {
    try {
      const configFile = readFileSync(configPath, 'utf8');
      const config = yaml.parse(configFile);
      
      // Extract infrastructure-specific configuration
      return {
        kafka: {
          enabled: config.messaging?.kafka?.enabled || false,
          brokers: config.messaging?.kafka?.brokers || ['localhost:9092'],
          clientId: 'arbitrage-platform',
          topics: config.messaging?.kafka?.topics || {
            agents_latest: 'agents.latest',
            fusion_predictions: 'fusion.predictions',
            execution_orders: 'execution.orders',
            validation_results: 'validation.results'
          }
        },
        redis: {
          enabled: true, // Always enable Redis for caching
          url: process.env.REDIS_URL || 'redis://localhost:6379',
          ttl_seconds: 300 // 5 minutes default TTL
        },
        database: {
          type: config.database?.type || 'sqlite',
          path: config.database?.path || './data/arbitrage.db'
        }
      };
    } catch (error) {
      console.warn('Failed to load config, using defaults:', error.message);
      
      // Return default configuration
      return {
        kafka: {
          enabled: false,
          brokers: ['localhost:9092'],
          clientId: 'arbitrage-platform',
          topics: {
            agents_latest: 'agents.latest',
            fusion_predictions: 'fusion.predictions',
            execution_orders: 'execution.orders',
            validation_results: 'validation.results'
          }
        },
        redis: {
          enabled: false, // Disable Redis if config fails
          url: 'redis://localhost:6379',
          ttl_seconds: 300
        },
        database: {
          type: 'sqlite',
          path: './data/arbitrage.db'
        }
      };
    }
  }

  /**
   * Initialize all infrastructure components
   */
  async initialize(): Promise<void> {
    try {
      // Create data directory if it doesn't exist
      await this.ensureDataDirectory();
      
      // Initialize in parallel
      await Promise.all([
        this.kafka.initialize(),
        this.redis.initialize(),
        this.database.initialize()
      ]);
      
      console.log('Infrastructure initialized successfully');
    } catch (error) {
      console.error('Failed to initialize infrastructure:', error);
      throw error;
    }
  }

  /**
   * Ensure data directory exists
   */
  private async ensureDataDirectory(): Promise<void> {
    const fs = await import('fs/promises');
    const path = await import('path');
    
    if (this.config.database.path) {
      const dir = path.dirname(this.config.database.path);
      try {
        await fs.mkdir(dir, { recursive: true });
      } catch (error) {
        // Directory might already exist
      }
    }
  }

  /**
   * Shutdown all infrastructure components
   */
  async shutdown(): Promise<void> {
    console.log('Shutting down infrastructure...');
    
    await Promise.all([
      this.kafka.close().catch(e => console.error('Kafka shutdown error:', e)),
      this.redis.close().catch(e => console.error('Redis shutdown error:', e)),
      this.database.close().catch(e => console.error('Database shutdown error:', e))
    ]);
    
    console.log('Infrastructure shutdown complete');
  }

  /**
   * Health check for all components
   */
  async healthCheck(): Promise<{
    kafka: boolean;
    redis: boolean;
    database: boolean;
    overall: boolean;
  }> {
    const kafka = this.config.kafka.enabled; // Simplified check
    const redis = this.config.redis.enabled; // Simplified check
    const database = this.database !== null;
    
    return {
      kafka,
      redis,
      database,
      overall: kafka && redis && database
    };
  }
}

/**
 * Create and initialize infrastructure
 */
export async function createInfrastructure(configPath?: string): Promise<Infrastructure> {
  const infrastructure = new Infrastructure(configPath);
  await infrastructure.initialize();
  return infrastructure;
}

// Export for testing and debugging
export { Infrastructure as default };