import fs from 'fs';
import path from 'path';
import yaml from 'yaml';
import dotenv from 'dotenv';
import { z } from 'zod';

// Load environment variables
dotenv.config();

const ConfigSchema = z.object({
  agents: z.object({
    economic: z.object({
      polling_hours: z.number(),
      data_age_max_hours: z.number(),
      confidence_min: z.number(),
    }),
    sentiment: z.object({
      window_sec: z.number(),
      min_mention_volume: z.number(),
      polarity_confidence_min: z.number(),
    }),
    price: z.object({
      snapshot_ms: z.number(),
      min_24h_volume_usd: z.number(),
      orderbook_levels: z.number(),
    }),
    volume: z.object({
      window_sec: z.number(),
      liquidity_index_min: z.number(),
    }),
    trade: z.object({
      max_trade_latency_ms: z.number(),
      slippage_estimate_max_pct: z.number(),
    }),
    image: z.object({
      image_age_max_sec: z.number(),
      visual_confidence_min: z.number(),
    }),
  }),
  constraints: z.object({
    max_open_exposure_pct_of_NAV: z.number(),
    api_health_pause_threshold: z.number(),
    event_blackout_sec: z.number(),
    data_freshness_max_sec: z.number(),
  }),
  bounds: z.object({
    min_spread_pct_for_execution: z.number(),
    llm_confidence_threshold: z.number(),
    max_hold_time_sec: z.number(),
    max_simultaneous_trades: z.number(),
    max_slippage_pct_estimate: z.number(),
    z_threshold_k: z.number(),
  }),
  hyperbolic: z.object({
    model: z.enum(['poincare', 'lorentz']),
    curvature: z.number(),
    embedding_dim: z.number(),
  }),
}).passthrough(); // Allow additional fields

export class ConfigLoader {
  private static instance: ConfigLoader;
  private config: any;

  private constructor() {
    this.loadConfig();
  }

  static getInstance(): ConfigLoader {
    if (!ConfigLoader.instance) {
      ConfigLoader.instance = new ConfigLoader();
    }
    return ConfigLoader.instance;
  }

  private loadConfig(): void {
    const configPath = path.join(process.cwd(), 'config.yaml');
    
    if (!fs.existsSync(configPath)) {
      throw new Error(`Configuration file not found: ${configPath}`);
    }

    const fileContent = fs.readFileSync(configPath, 'utf8');
    let parsedConfig = yaml.parse(fileContent);

    // Replace environment variables
    parsedConfig = this.replaceEnvVars(parsedConfig);

    // Validate config
    try {
      ConfigSchema.parse(parsedConfig);
    } catch (error) {
      console.error('Invalid configuration:', error);
      throw error;
    }

    this.config = parsedConfig;
  }

  private replaceEnvVars(obj: any): any {
    if (typeof obj === 'string') {
      // Replace ${VAR_NAME} with environment variable
      const matches = obj.match(/\$\{([^}]+)\}/g);
      if (matches) {
        let result = obj;
        matches.forEach(match => {
          const varName = match.slice(2, -1);
          const value = process.env[varName];
          if (value !== undefined) {
            result = result.replace(match, value);
          }
        });
        return result;
      }
      return obj;
    } else if (Array.isArray(obj)) {
      return obj.map(item => this.replaceEnvVars(item));
    } else if (obj !== null && typeof obj === 'object') {
      const result: any = {};
      for (const [key, value] of Object.entries(obj)) {
        result[key] = this.replaceEnvVars(value);
      }
      return result;
    }
    return obj;
  }

  get<T = any>(path: string): T {
    const keys = path.split('.');
    let result = this.config;

    for (const key of keys) {
      if (result && typeof result === 'object' && key in result) {
        result = result[key];
      } else {
        return undefined as any;
      }
    }

    return result;
  }

  getAll(): any {
    return this.config;
  }

  reload(): void {
    this.loadConfig();
  }
}

export default ConfigLoader.getInstance();