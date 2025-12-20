/**
 * Time-Scale Feature Store
 * 
 * Partitions features by time horizon: Hourly, Weekly, Monthly
 * Enables multi-horizon analysis and signal generation
 */

import { EngineeredFeatures, RawMarketData } from './feature-engineering';

export enum TimeHorizon {
  HOURLY = 'hourly',
  WEEKLY = 'weekly',
  MONTHLY = 'monthly'
}

export interface TimeScaledFeatures {
  horizon: TimeHorizon;
  features: EngineeredFeatures;
  timestamp: Date;
  aggregationWindow: {
    start: Date;
    end: Date;
    dataPoints: number;
  };
  metadata: {
    volatilityRegime: 'low' | 'normal' | 'high' | 'extreme';
    dataQuality: number; // [0, 1]
    completeness: number; // [0, 1]
  };
}

export interface HorizonCharacteristics {
  horizon: TimeHorizon;
  description: string;
  signalType: 'short_lived' | 'persistent' | 'structural';
  typicalDecayHours: number;
  rebalanceFrequency: string;
  executionStyle: 'intraday_twap' | 'daily_vwap' | 'multiday_rebalance';
}

export const HORIZON_SPECS: Record<TimeHorizon, HorizonCharacteristics> = {
  [TimeHorizon.HOURLY]: {
    horizon: TimeHorizon.HOURLY,
    description: 'Short-term signals (1-24 hours)',
    signalType: 'short_lived',
    typicalDecayHours: 6,
    rebalanceFrequency: 'Every 1-4 hours',
    executionStyle: 'intraday_twap'
  },
  [TimeHorizon.WEEKLY]: {
    horizon: TimeHorizon.WEEKLY,
    description: 'Medium-term signals (3-10 days)',
    signalType: 'persistent',
    typicalDecayHours: 48,
    rebalanceFrequency: 'Daily',
    executionStyle: 'daily_vwap'
  },
  [TimeHorizon.MONTHLY]: {
    horizon: TimeHorizon.MONTHLY,
    description: 'Long-term structural signals (2-8 weeks)',
    signalType: 'structural',
    typicalDecayHours: 168, // 1 week
    rebalanceFrequency: 'Weekly',
    executionStyle: 'multiday_rebalance'
  }
};

export class TimeScaleFeatureStore {
  private hourlyStore: TimeScaledFeatures[] = [];
  private weeklyStore: TimeScaledFeatures[] = [];
  private monthlyStore: TimeScaledFeatures[] = [];
  
  private maxHourlyStored = 168;    // 1 week of hourly data
  private maxWeeklyStored = 52;     // 1 year of weekly data
  private maxMonthlyStored = 24;    // 2 years of monthly data
  
  private rawDataBuffer: RawMarketData[] = [];
  private maxBufferSize = 10000;

  /**
   * Add raw market data to buffer
   */
  addRawData(data: RawMarketData): void {
    this.rawDataBuffer.push(data);
    
    if (this.rawDataBuffer.length > this.maxBufferSize) {
      this.rawDataBuffer.shift();
    }
  }

  /**
   * Compute and store features for all time horizons
   */
  computeAndStoreFeatures(engineeredFeatures: EngineeredFeatures): void {
    const now = new Date();
    
    // Hourly features (latest 1 hour of data)
    const hourlyFeatures = this.createTimeScaledFeatures(
      TimeHorizon.HOURLY,
      engineeredFeatures,
      now
    );
    this.storeHourly(hourlyFeatures);
    
    // Weekly features (aggregate of last 7 days)
    if (this.shouldComputeWeekly()) {
      const weeklyFeatures = this.aggregateFeatures(TimeHorizon.WEEKLY);
      this.storeWeekly(weeklyFeatures);
    }
    
    // Monthly features (aggregate of last 30 days)
    if (this.shouldComputeMonthly()) {
      const monthlyFeatures = this.aggregateFeatures(TimeHorizon.MONTHLY);
      this.storeMonthly(monthlyFeatures);
    }
  }

  /**
   * Create time-scaled features with metadata
   */
  private createTimeScaledFeatures(
    horizon: TimeHorizon,
    features: EngineeredFeatures,
    timestamp: Date
  ): TimeScaledFeatures {
    const windowSizes = {
      [TimeHorizon.HOURLY]: 3600 * 1000,      // 1 hour in ms
      [TimeHorizon.WEEKLY]: 7 * 24 * 3600 * 1000,    // 7 days
      [TimeHorizon.MONTHLY]: 30 * 24 * 3600 * 1000   // 30 days
    };
    
    const windowMs = windowSizes[horizon];
    const start = new Date(timestamp.getTime() - windowMs);
    
    return {
      horizon,
      features,
      timestamp,
      aggregationWindow: {
        start,
        end: timestamp,
        dataPoints: this.rawDataBuffer.filter(d => 
          d.timestamp >= start && d.timestamp <= timestamp
        ).length
      },
      metadata: {
        volatilityRegime: this.classifyVolatilityRegime(features.volatility.realized24h),
        dataQuality: this.assessDataQuality(start, timestamp),
        completeness: this.assessCompleteness(start, timestamp)
      }
    };
  }

  /**
   * Aggregate features for weekly/monthly horizons
   */
  private aggregateFeatures(horizon: TimeHorizon): TimeScaledFeatures {
    const now = new Date();
    const windowMs = horizon === TimeHorizon.WEEKLY 
      ? 7 * 24 * 3600 * 1000 
      : 30 * 24 * 3600 * 1000;
    
    const start = new Date(now.getTime() - windowMs);
    const relevantHourly = this.hourlyStore.filter(h => 
      h.timestamp >= start && h.timestamp <= now
    );
    
    if (relevantHourly.length === 0) {
      // Return zero features if no data
      return this.createEmptyTimeScaledFeatures(horizon, now);
    }
    
    // Aggregate returns
    const aggregatedReturns = {
      log1m: this.average(relevantHourly.map(h => h.features.returns.log1m)),
      log5m: this.average(relevantHourly.map(h => h.features.returns.log5m)),
      log1h: this.average(relevantHourly.map(h => h.features.returns.log1h)),
      simple1h: this.average(relevantHourly.map(h => h.features.returns.simple1h))
    };
    
    // Aggregate spreads
    const aggregatedSpreads = {
      bidAsk: this.average(relevantHourly.map(h => h.features.spreads.bidAsk)),
      crossExchange: [
        this.average(relevantHourly.map(h => h.features.spreads.crossExchange[0])),
        this.average(relevantHourly.map(h => h.features.spreads.crossExchange[1])),
        this.average(relevantHourly.map(h => h.features.spreads.crossExchange[2]))
      ],
      spotPerp: this.average(relevantHourly.map(h => h.features.spreads.spotPerp)),
      fundingBasis: this.average(relevantHourly.map(h => h.features.spreads.fundingBasis))
    };
    
    // Aggregate volatility (use max for realized vol, average for others)
    const aggregatedVolatility = {
      realized1h: this.max(relevantHourly.map(h => h.features.volatility.realized1h)),
      realized24h: this.max(relevantHourly.map(h => h.features.volatility.realized24h)),
      ewma: this.average(relevantHourly.map(h => h.features.volatility.ewma)),
      parkinson: this.max(relevantHourly.map(h => h.features.volatility.parkinson))
    };
    
    // Aggregate flow
    const aggregatedFlow = {
      volumeImbalance: this.average(relevantHourly.map(h => h.features.flow.volumeImbalance)),
      orderImbalance: this.average(relevantHourly.map(h => h.features.flow.orderImbalance)),
      netFlow: this.sum(relevantHourly.map(h => h.features.flow.netFlow))
    };
    
    // Aggregate z-scores
    const aggregatedZScores = {
      priceZ: this.average(relevantHourly.map(h => h.features.zScores.priceZ)),
      volumeZ: this.average(relevantHourly.map(h => h.features.zScores.volumeZ)),
      spreadZ: this.average(relevantHourly.map(h => h.features.zScores.spreadZ))
    };
    
    // Use latest rolling metrics
    const latestRolling = relevantHourly[relevantHourly.length - 1].features.rolling;
    
    const aggregatedFeatures: EngineeredFeatures = {
      returns: aggregatedReturns,
      spreads: aggregatedSpreads,
      volatility: aggregatedVolatility,
      flow: aggregatedFlow,
      zScores: aggregatedZScores,
      rolling: latestRolling
    };
    
    return this.createTimeScaledFeatures(horizon, aggregatedFeatures, now);
  }

  /**
   * Store hourly features
   */
  private storeHourly(features: TimeScaledFeatures): void {
    this.hourlyStore.push(features);
    if (this.hourlyStore.length > this.maxHourlyStored) {
      this.hourlyStore.shift();
    }
  }

  /**
   * Store weekly features
   */
  private storeWeekly(features: TimeScaledFeatures): void {
    this.weeklyStore.push(features);
    if (this.weeklyStore.length > this.maxWeeklyStored) {
      this.weeklyStore.shift();
    }
  }

  /**
   * Store monthly features
   */
  private storeMonthly(features: TimeScaledFeatures): void {
    this.monthlyStore.push(features);
    if (this.monthlyStore.length > this.maxMonthlyStored) {
      this.monthlyStore.shift();
    }
  }

  /**
   * Get features for specific horizon
   */
  getFeatures(horizon: TimeHorizon): TimeScaledFeatures[] {
    switch (horizon) {
      case TimeHorizon.HOURLY:
        return [...this.hourlyStore];
      case TimeHorizon.WEEKLY:
        return [...this.weeklyStore];
      case TimeHorizon.MONTHLY:
        return [...this.monthlyStore];
    }
  }

  /**
   * Get latest features for horizon
   */
  getLatestFeatures(horizon: TimeHorizon): TimeScaledFeatures | null {
    const store = this.getFeatures(horizon);
    return store.length > 0 ? store[store.length - 1] : null;
  }

  /**
   * Get features for all horizons
   */
  getAllHorizonsLatest(): Record<TimeHorizon, TimeScaledFeatures | null> {
    return {
      [TimeHorizon.HOURLY]: this.getLatestFeatures(TimeHorizon.HOURLY),
      [TimeHorizon.WEEKLY]: this.getLatestFeatures(TimeHorizon.WEEKLY),
      [TimeHorizon.MONTHLY]: this.getLatestFeatures(TimeHorizon.MONTHLY)
    };
  }

  /**
   * Determine if we should compute weekly features (once per day)
   */
  private shouldComputeWeekly(): boolean {
    if (this.weeklyStore.length === 0) return true;
    
    const lastWeekly = this.weeklyStore[this.weeklyStore.length - 1];
    const hoursSinceLastWeekly = (Date.now() - lastWeekly.timestamp.getTime()) / (1000 * 3600);
    
    return hoursSinceLastWeekly >= 24;  // Update once per day
  }

  /**
   * Determine if we should compute monthly features (once per week)
   */
  private shouldComputeMonthly(): boolean {
    if (this.monthlyStore.length === 0) return true;
    
    const lastMonthly = this.monthlyStore[this.monthlyStore.length - 1];
    const hoursSinceLastMonthly = (Date.now() - lastMonthly.timestamp.getTime()) / (1000 * 3600);
    
    return hoursSinceLastMonthly >= 168;  // Update once per week
  }

  /**
   * Classify volatility regime
   */
  private classifyVolatilityRegime(realizedVol: number): 'low' | 'normal' | 'high' | 'extreme' {
    if (realizedVol < 15) return 'low';
    if (realizedVol < 30) return 'normal';
    if (realizedVol < 50) return 'high';
    return 'extreme';
  }

  /**
   * Assess data quality based on data availability
   */
  private assessDataQuality(start: Date, end: Date): number {
    const relevantData = this.rawDataBuffer.filter(d => 
      d.timestamp >= start && d.timestamp <= end
    );
    
    if (relevantData.length === 0) return 0;
    
    // Check for missing prices, volumes
    const hasPrice = relevantData.every(d => d.spotPrice > 0);
    const hasVolume = relevantData.every(d => d.volume24h > 0);
    
    return (hasPrice ? 0.5 : 0) + (hasVolume ? 0.5 : 0);
  }

  /**
   * Assess data completeness (ratio of expected vs actual data points)
   */
  private assessCompleteness(start: Date, end: Date): number {
    const windowMs = end.getTime() - start.getTime();
    const expectedPoints = Math.floor(windowMs / (60 * 1000)); // Assuming 1 min granularity
    
    const actualPoints = this.rawDataBuffer.filter(d => 
      d.timestamp >= start && d.timestamp <= end
    ).length;
    
    return Math.min(actualPoints / expectedPoints, 1.0);
  }

  /**
   * Create empty time-scaled features
   */
  private createEmptyTimeScaledFeatures(horizon: TimeHorizon, timestamp: Date): TimeScaledFeatures {
    const emptyFeatures: EngineeredFeatures = {
      returns: { log1m: 0, log5m: 0, log1h: 0, simple1h: 0 },
      spreads: { bidAsk: 0, crossExchange: [0, 0, 0], spotPerp: 0, fundingBasis: 0 },
      volatility: { realized1h: 0, realized24h: 0, ewma: 0, parkinson: 0 },
      flow: { volumeImbalance: 0, orderImbalance: 0, netFlow: 0 },
      zScores: { priceZ: 0, volumeZ: 0, spreadZ: 0 },
      rolling: { sma20: 0, ema20: 0, bollingerUpper: 0, bollingerLower: 0, bollingerWidth: 0, rsi14: 50 }
    };
    
    return this.createTimeScaledFeatures(horizon, emptyFeatures, timestamp);
  }

  // Helper functions
  private average(values: number[]): number {
    if (values.length === 0) return 0;
    return values.reduce((a, b) => a + b, 0) / values.length;
  }

  private sum(values: number[]): number {
    return values.reduce((a, b) => a + b, 0);
  }

  private max(values: number[]): number {
    if (values.length === 0) return 0;
    return Math.max(...values);
  }

  /**
   * Get horizon characteristics
   */
  getHorizonSpecs(horizon: TimeHorizon): HorizonCharacteristics {
    return HORIZON_SPECS[horizon];
  }

  /**
   * Get storage statistics
   */
  getStats(): {
    hourly: { count: number; oldest: Date | null; newest: Date | null };
    weekly: { count: number; oldest: Date | null; newest: Date | null };
    monthly: { count: number; oldest: Date | null; newest: Date | null };
    bufferSize: number;
  } {
    return {
      hourly: {
        count: this.hourlyStore.length,
        oldest: this.hourlyStore[0]?.timestamp ?? null,
        newest: this.hourlyStore[this.hourlyStore.length - 1]?.timestamp ?? null
      },
      weekly: {
        count: this.weeklyStore.length,
        oldest: this.weeklyStore[0]?.timestamp ?? null,
        newest: this.weeklyStore[this.weeklyStore.length - 1]?.timestamp ?? null
      },
      monthly: {
        count: this.monthlyStore.length,
        oldest: this.monthlyStore[0]?.timestamp ?? null,
        newest: this.monthlyStore[this.monthlyStore.length - 1]?.timestamp ?? null
      },
      bufferSize: this.rawDataBuffer.length
    };
  }
}
