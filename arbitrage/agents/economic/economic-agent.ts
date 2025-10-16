/**
 * Economic Agent - Macro Economic Data Collection
 * Collects real-time economic indicators from FRED API and other sources
 * Provides macro context affecting market risk appetite and funding rates
 */

import { BaseAgent, AgentConfig, AgentOutput } from '../../core/base-agent.js';
import axios, { AxiosResponse } from 'axios';

export interface EconomicData {
  CPI: number;           // Consumer Price Index (inflation)
  FEDFUNDS: number;      // Federal Funds Rate
  UNRATE: number;        // Unemployment Rate
  M2SL: number;          // M2 Money Supply
  GDP: number;           // GDP Growth Rate
  VIX: number;           // Volatility Index
}

export interface EconomicFeatures {
  inflation_trend: number;     // CPI trend [-1, 1]
  real_rate: number;          // FEDFUNDS - CPI
  liquidity_bias: number;     // M2 growth rate normalized [-1, 1]
  employment_strength: number; // (100 - UNRATE) / 100
  risk_appetite: number;      // Inverse VIX normalized
  economic_momentum: number;   // GDP growth normalized
}

export class EconomicAgent extends BaseAgent {
  private fredApiKey: string;
  private lastData: EconomicData | null = null;
  private historicalData: EconomicData[] = [];

  constructor(config: AgentConfig, fredApiKey: string) {
    super(config);
    this.fredApiKey = fredApiKey;
  }

  protected async collectData(): Promise<AgentOutput> {
    const timestamp = this.getCurrentTimestamp();

    try {
      // Collect all economic indicators
      const economicData = await this.fetchEconomicIndicators();
      
      // Calculate derived features
      const features = this.calculateFeatures(economicData);
      
      // Calculate key signal (composite economic health score)
      const keySignal = this.calculateKeySignal(features);
      
      // Calculate confidence based on data freshness and completeness
      const confidence = this.calculateDataConfidence(economicData);
      
      // Store for trend analysis
      this.lastData = economicData;
      this.historicalData.push(economicData);
      
      // Keep only last 24 data points (24 hours of hourly data)
      if (this.historicalData.length > 24) {
        this.historicalData.shift();
      }

      return {
        agent_name: 'EconomicAgent',
        timestamp,
        key_signal: keySignal,
        confidence,
        features: {
          ...features,
          raw_data: economicData
        },
        metadata: {
          data_points: Object.keys(economicData).length,
          historical_samples: this.historicalData.length
        }
      };

    } catch (error) {
      console.error('EconomicAgent data collection failed:', error);
      throw error;
    }
  }

  /**
   * Fetch economic indicators from FRED API
   */
  private async fetchEconomicIndicators(): Promise<EconomicData> {
    const indicators = {
      CPI: 'CPIAUCSL',      // Consumer Price Index
      FEDFUNDS: 'FEDFUNDS', // Federal Funds Rate
      UNRATE: 'UNRATE',     // Unemployment Rate
      M2SL: 'M2SL',         // M2 Money Supply
      GDP: 'GDP',           // Gross Domestic Product
      VIX: 'VIXCLS'         // VIX Volatility Index
    };

    const results: Partial<EconomicData> = {};
    const promises: Promise<void>[] = [];

    // Fetch all indicators in parallel
    for (const [key, seriesId] of Object.entries(indicators)) {
      const promise = this.fetchFredSeries(seriesId)
        .then(value => {
          results[key as keyof EconomicData] = value;
        })
        .catch(error => {
          console.warn(`Failed to fetch ${key} (${seriesId}):`, error.message);
          // Use last known value if available
          if (this.lastData && this.lastData[key as keyof EconomicData]) {
            results[key as keyof EconomicData] = this.lastData[key as keyof EconomicData];
          }
        });
      
      promises.push(promise);
    }

    await Promise.all(promises);

    // Validate we have minimum required data
    const requiredFields: (keyof EconomicData)[] = ['CPI', 'FEDFUNDS', 'UNRATE'];
    for (const field of requiredFields) {
      if (results[field] === undefined) {
        throw new Error(`Missing required economic indicator: ${field}`);
      }
    }

    return results as EconomicData;
  }

  /**
   * Fetch single economic series from FRED API
   */
  private async fetchFredSeries(seriesId: string): Promise<number> {
    const url = 'https://api.stlouisfed.org/fred/series/observations';
    const params = {
      series_id: seriesId,
      api_key: this.fredApiKey,
      limit: 1,
      sort_order: 'desc',
      output_type: 'json'
    };

    const response: AxiosResponse = await axios.get(url, { 
      params, 
      timeout: 5000 
    });

    if (response.data.observations && response.data.observations.length > 0) {
      const latestValue = parseFloat(response.data.observations[0].value);
      
      if (isNaN(latestValue)) {
        throw new Error(`Invalid numeric value for series ${seriesId}`);
      }
      
      return latestValue;
    } else {
      throw new Error(`No data available for series ${seriesId}`);
    }
  }

  /**
   * Calculate derived economic features
   */
  private calculateFeatures(data: EconomicData): EconomicFeatures {
    // Calculate inflation trend (using historical data if available)
    let inflationTrend = 0;
    if (this.historicalData.length >= 2) {
      const previous = this.historicalData[this.historicalData.length - 2];
      inflationTrend = this.normalize(
        data.CPI - previous.CPI, 
        -1, 1, true // bipolar [-1, 1]
      );
    }

    // Real interest rate = nominal rate - inflation
    const realRate = (data.FEDFUNDS || 0) - (data.CPI || 0);

    // Liquidity bias from M2 money supply growth
    let liquidityBias = 0;
    if (this.historicalData.length >= 2 && data.M2SL) {
      const previous = this.historicalData[this.historicalData.length - 2];
      const m2Growth = ((data.M2SL - (previous.M2SL || data.M2SL)) / (previous.M2SL || data.M2SL)) * 100;
      liquidityBias = this.normalize(m2Growth, -5, 5, true); // -5% to +5% growth range
    }

    // Employment strength (inverse unemployment)
    const employmentStrength = data.UNRATE ? (100 - data.UNRATE) / 100 : 0.95;

    // Risk appetite (inverse VIX)
    const riskAppetite = data.VIX ? this.normalize(100 - data.VIX, 0, 80, false) : 0.5;

    // Economic momentum from GDP
    const economicMomentum = data.GDP ? this.normalize(data.GDP, -5, 8, false) : 0.5;

    return {
      inflation_trend: inflationTrend,
      real_rate: realRate,
      liquidity_bias: liquidityBias,
      employment_strength: employmentStrength,
      risk_appetite: riskAppetite,
      economic_momentum: economicMomentum
    };
  }

  /**
   * Calculate composite economic health key signal
   */
  private calculateKeySignal(features: EconomicFeatures): number {
    // Weighted composite of economic health indicators
    const weights = {
      inflation_trend: -0.2,      // High inflation is negative
      real_rate: 0.15,           // Positive real rates are good
      liquidity_bias: 0.2,       // More liquidity is positive for risk assets
      employment_strength: 0.2,   // Strong employment is positive
      risk_appetite: 0.15,       // Low VIX is positive for arbitrage
      economic_momentum: 0.1      // GDP growth is positive
    };

    let signal = 0;
    let totalWeight = 0;

    for (const [key, weight] of Object.entries(weights)) {
      const value = features[key as keyof EconomicFeatures];
      if (typeof value === 'number' && !isNaN(value)) {
        signal += value * weight;
        totalWeight += Math.abs(weight);
      }
    }

    // Normalize to [-1, 1] range
    return totalWeight > 0 ? Math.max(-1, Math.min(1, signal / totalWeight)) : 0;
  }

  /**
   * Calculate confidence based on data quality
   */
  private calculateDataConfidence(data: EconomicData): number {
    const totalFields = Object.keys(data).length;
    const validFields = Object.values(data).filter(v => 
      typeof v === 'number' && !isNaN(v) && isFinite(v)
    ).length;

    // Base confidence on data completeness
    const completeness = validFields / totalFields;

    // Adjust for data recency (FRED data can be delayed)
    const recencyFactor = 1.0; // Economic data is typically daily/weekly

    // Historical data availability boosts confidence
    const historyFactor = Math.min(1, this.historicalData.length / 10);

    return Math.max(0.1, completeness * recencyFactor * (0.7 + 0.3 * historyFactor));
  }

  /**
   * Get economic summary for debugging
   */
  getEconomicSummary(): string {
    if (!this.lastData) {
      return 'No economic data available';
    }

    const { CPI, FEDFUNDS, UNRATE, VIX } = this.lastData;
    return `CPI: ${CPI?.toFixed(1)}%, Fed Funds: ${FEDFUNDS?.toFixed(2)}%, Unemployment: ${UNRATE?.toFixed(1)}%, VIX: ${VIX?.toFixed(1)}`;
  }
}

/**
 * Factory function to create EconomicAgent with config
 */
export function createEconomicAgent(fredApiKey: string): EconomicAgent {
  const config: AgentConfig = {
    name: 'economic',
    enabled: true,
    polling_interval_ms: 60 * 60 * 1000, // 1 hour
    confidence_min: 0.5,
    data_age_max_ms: 6 * 60 * 60 * 1000, // 6 hours
    retry_attempts: 3,
    retry_backoff_ms: 5000
  };

  return new EconomicAgent(config, fredApiKey);
}

/**
 * Demo/test FRED API key (replace with real key for production)
 */
export const DEMO_FRED_API_KEY = 'demo_key_replace_with_real_fred_api_key';

// Export for testing
export { EconomicAgent as default };