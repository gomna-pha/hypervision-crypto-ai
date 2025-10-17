import axios from 'axios';
import NodeCache from 'node-cache';
import { BaseAgent } from './BaseAgent';
import { EconomicAgentOutput } from '../types';
import config from '../utils/ConfigLoader';

interface FREDSeriesData {
  date: string;
  value: string;
}

interface EconomicData {
  CPI?: number;
  FEDFUNDS?: number;
  UNRATE?: number;
  M2SL?: number;
  GDP?: number;
  VIX?: number;
}

export class EconomicAgent extends BaseAgent {
  private cache: NodeCache;
  private fredApiKey: string | undefined;
  private seriesIds: Map<string, string>;

  constructor(port: number = 3001) {
    super('EconomicAgent', port);
    this.cache = new NodeCache({ stdTTL: 3600 }); // 1 hour cache
    this.seriesIds = new Map([
      ['CPI', 'CPIAUCSL'],
      ['FEDFUNDS', 'FEDFUNDS'],
      ['UNRATE', 'UNRATE'],
      ['M2SL', 'M2SL'],
      ['GDP', 'GDPC1'],
    ]);
  }

  protected async initialize(): Promise<void> {
    const agentConfig = config.get('agents.economic');
    
    // Get FRED API key from environment
    this.fredApiKey = process.env.FRED_API_KEY;
    
    if (!this.fredApiKey) {
      this.logger.warn('FRED_API_KEY not configured, using mock data');
    }

    this.logger.info('EconomicAgent initialized');
  }

  protected async update(): Promise<void> {
    try {
      const economicData = await this.fetchEconomicData();
      const features = this.calculateFeatures(economicData);
      const keySignal = this.calculateKeySignal(economicData, features);
      const confidence = this.calculateConfidence(economicData);

      const output: EconomicAgentOutput = {
        agent_name: 'EconomicAgent',
        timestamp: new Date().toISOString(),
        key_signal: keySignal,
        confidence: confidence,
        signals: economicData,
        features: features,
      };

      await this.publishOutput(output);
      this.logger.info('Economic data updated', { keySignal, confidence });
    } catch (error) {
      this.logger.error('Failed to update economic data', error);
      throw error;
    }
  }

  private async fetchEconomicData(): Promise<EconomicData> {
    const data: EconomicData = {};

    // If no API key, return mock data for development
    if (!this.fredApiKey) {
      return this.getMockData();
    }

    // Fetch each series from FRED
    for (const [key, seriesId] of this.seriesIds) {
      try {
        // Check cache first
        const cachedValue = this.cache.get<number>(key);
        if (cachedValue !== undefined) {
          data[key as keyof EconomicData] = cachedValue;
          continue;
        }

        // Fetch from FRED API
        const value = await this.fetchFREDSeries(seriesId);
        if (value !== null) {
          data[key as keyof EconomicData] = value;
          this.cache.set(key, value);
        }
      } catch (error) {
        this.logger.error(`Failed to fetch ${key} data`, error);
      }
    }

    // Fetch VIX from alternative source
    try {
      const vix = await this.fetchVIX();
      if (vix !== null) {
        data.VIX = vix;
      }
    } catch (error) {
      this.logger.error('Failed to fetch VIX data', error);
    }

    return data;
  }

  private async fetchFREDSeries(seriesId: string): Promise<number | null> {
    try {
      const url = 'https://api.stlouisfed.org/fred/series/observations';
      
      const response = await axios.get(url, {
        params: {
          series_id: seriesId,
          api_key: this.fredApiKey,
          file_type: 'json',
          limit: 1,
          sort_order: 'desc',
        },
        timeout: 10000,
      });

      const observations = response.data?.observations;
      if (observations && observations.length > 0) {
        const latestValue = parseFloat(observations[0].value);
        if (!isNaN(latestValue)) {
          return latestValue;
        }
      }

      return null;
    } catch (error) {
      this.logger.error(`FRED API error for ${seriesId}`, error);
      return null;
    }
  }

  private async fetchVIX(): Promise<number | null> {
    // In production, this would fetch from a real VIX data source
    // For now, return mock data
    return 15.5 + Math.random() * 10;
  }

  private getMockData(): EconomicData {
    // Mock data for development/testing
    return {
      CPI: 3.2 + Math.random() * 0.5,
      FEDFUNDS: 4.5 + Math.random() * 0.25,
      UNRATE: 4.1 + Math.random() * 0.3,
      M2SL: 21000000 + Math.random() * 1000000,
      GDP: 2.1 + Math.random() * 0.5,
      VIX: 15.5 + Math.random() * 10,
    };
  }

  private calculateFeatures(data: EconomicData): {
    inflation_trend: number;
    real_rate: number;
    liquidity_bias: number;
  } {
    const cpi = data.CPI || 3.0;
    const fedFunds = data.FEDFUNDS || 4.5;
    const m2 = data.M2SL || 21000000;

    // Simple trend calculation (in production, would use historical data)
    const inflationTrend = this.normalizeSignal(cpi - 3.0, -2, 2);
    
    // Real interest rate
    const realRate = fedFunds - cpi;
    
    // Liquidity bias based on M2 growth (simplified)
    const m2Baseline = 20000000;
    const liquidityBias = this.normalizeSignal((m2 - m2Baseline) / m2Baseline, -0.1, 0.1);

    return {
      inflation_trend: inflationTrend,
      real_rate: realRate,
      liquidity_bias: liquidityBias,
    };
  }

  private calculateKeySignal(
    data: EconomicData, 
    features: { inflation_trend: number; real_rate: number; liquidity_bias: number }
  ): number {
    // Composite signal combining multiple factors
    // Positive = risk-on environment, Negative = risk-off
    let signal = 0;

    // VIX component (inverted - low VIX = positive signal)
    if (data.VIX) {
      const vixNorm = this.normalizeSignal(data.VIX, 10, 40);
      signal -= vixNorm * 0.3; // Weight: 30%
    }

    // Real rate component (negative real rates = positive for risk assets)
    signal -= features.real_rate * 0.1; // Weight: 20%

    // Unemployment component (low unemployment = positive)
    if (data.UNRATE) {
      const unrateNorm = this.normalizeSignal(data.UNRATE, 3, 7);
      signal -= unrateNorm * 0.2; // Weight: 20%
    }

    // Liquidity component
    signal += features.liquidity_bias * 0.3; // Weight: 30%

    // Clamp to [-1, 1]
    return Math.max(-1, Math.min(1, signal));
  }

  private calculateConfidence(data: EconomicData): number {
    // Confidence based on data completeness and freshness
    const dataPoints = Object.keys(data).length;
    const maxDataPoints = 6;
    
    const completeness = dataPoints / maxDataPoints;
    
    // In production, would also check data freshness
    const freshness = 0.95; // Assume data is fresh for now
    
    return completeness * freshness;
  }

  protected async cleanup(): Promise<void> {
    this.cache.flushAll();
    this.logger.info('EconomicAgent cleanup completed');
  }

  protected getPollingInterval(): number {
    const hours = config.get<number>('agents.economic.polling_hours') || 1;
    return hours * 60 * 60 * 1000; // Convert hours to milliseconds
  }
}

// Export for standalone execution
if (require.main === module) {
  const agent = new EconomicAgent();
  agent.start().catch(error => {
    console.error('Failed to start EconomicAgent:', error);
    process.exit(1);
  });

  process.on('SIGINT', async () => {
    await agent.stop();
    process.exit(0);
  });
}