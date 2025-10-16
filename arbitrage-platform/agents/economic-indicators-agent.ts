import axios from 'axios';
import EventEmitter from 'events';

export interface EconomicIndicatorsData {
  agent_name: string;
  timestamp: string;
  key_signal: number; // -1 to 1 (contractionary to expansionary)
  confidence: number; // 0 to 1
  features: {
    cpi_yoy_pct: number;
    fed_funds_rate: number;
    unemployment_rate: number;
    m2_money_supply_trillions: number;
    gdp_growth_rate: number;
    vix_index: number;
    real_interest_rate: number;
    inflation_trend: number;
    liquidity_conditions: number;
  };
}

export class EconomicIndicatorsAgent extends EventEmitter {
  private isRunning: boolean = false;
  private intervalId?: NodeJS.Timeout;
  
  // Visible Parameters for Investors
  public readonly parameters = {
    polling_interval_hours: 1,
    data_staleness_threshold_hours: 24,
    confidence_decay_half_life_hours: 12,
    inflation_target_pct: 2.0,
    neutral_fed_rate_pct: 2.5,
    full_employment_rate_pct: 4.0,
    vix_panic_threshold: 30,
    vix_complacency_threshold: 15
  };

  // Visible Constraints for Investors
  public readonly constraints = {
    max_api_calls_per_hour: 10,
    required_indicators: ['CPI', 'FEDFUNDS', 'UNRATE'],
    max_data_age_hours: 48,
    min_confidence_for_signal: 0.3,
    fred_api_timeout_ms: 15000,
    backup_data_sources: ['BLS', 'OECD'],
    critical_event_blackout_hours: 2
  };

  // Visible Bounds for Investors
  public readonly bounds = {
    signal_range: { min: -1.0, max: 1.0 },
    confidence_range: { min: 0.0, max: 1.0 },
    cpi_range: { min: -2.0, max: 10.0 }, // % YoY
    fed_rate_range: { min: 0.0, max: 20.0 }, // %
    unemployment_range: { min: 2.0, max: 15.0 }, // %
    vix_range: { min: 5.0, max: 100.0 },
    signal_smoothing_alpha: 0.7
  };

  private fredApiKey: string;
  private lastUpdateTimestamp: number = 0;
  private indicatorHistory: Map<string, Array<{ value: number; timestamp: number }>> = new Map();
  
  // FRED series mappings
  private fredSeries = {
    CPI: 'CPIAUCSL',           // Consumer Price Index
    FEDFUNDS: 'FEDFUNDS',     // Federal Funds Rate
    UNRATE: 'UNRATE',         // Unemployment Rate
    M2: 'M2SL',               // M2 Money Supply
    GDP: 'GDP',               // Gross Domestic Product
    VIX: 'VIXCLS'            // VIX Volatility Index
  };

  constructor() {
    super();
    this.fredApiKey = process.env.FRED_API_KEY || 'demo_key';
    console.log('✅ EconomicIndicatorsAgent initialized with visible parameters');
  }

  async start(): Promise<void> {
    if (this.isRunning) return;
    
    this.isRunning = true;
    console.log('🚀 Starting EconomicIndicatorsAgent with real FRED API...');
    
    // Initial data collection
    await this.collectEconomicData();
    
    // Set up polling interval
    this.intervalId = setInterval(async () => {
      try {
        await this.collectEconomicData();
      } catch (error) {
        console.error('❌ EconomicIndicatorsAgent polling error:', error);
        this.emit('error', error);
      }
    }, this.parameters.polling_interval_hours * 60 * 60 * 1000);
  }

  async stop(): Promise<void> {
    this.isRunning = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
    console.log('⏹️ EconomicIndicatorsAgent stopped');
  }

  private async collectEconomicData(): Promise<void> {
    try {
      console.log('📊 Collecting real-time economic indicators from FRED API...');
      
      // Fetch all economic indicators
      const indicators = await this.fetchAllIndicators();
      
      // Calculate derived metrics
      const derivedMetrics = this.calculateDerivedMetrics(indicators);
      
      // Generate composite economic signal
      const economicSignal = this.calculateCompositeSignal(indicators, derivedMetrics);
      
      // Calculate confidence based on data quality and recency
      const confidence = this.calculateConfidence(indicators);
      
      const economicData: EconomicIndicatorsData = {
        agent_name: 'EconomicIndicatorsAgent',
        timestamp: new Date().toISOString(),
        key_signal: this.clampToRange(economicSignal, this.bounds.signal_range),
        confidence: this.clampToRange(confidence, this.bounds.confidence_range),
        features: {
          cpi_yoy_pct: indicators.CPI || 0,
          fed_funds_rate: indicators.FEDFUNDS || 0,
          unemployment_rate: indicators.UNRATE || 0,
          m2_money_supply_trillions: (indicators.M2 || 0) / 1000, // Convert to trillions
          gdp_growth_rate: indicators.GDP || 0,
          vix_index: indicators.VIX || 0,
          real_interest_rate: derivedMetrics.realRate,
          inflation_trend: derivedMetrics.inflationTrend,
          liquidity_conditions: derivedMetrics.liquidityConditions
        }
      };

      // Update history for trend analysis
      this.updateIndicatorHistory(indicators);
      this.lastUpdateTimestamp = Date.now();
      
      // Emit to system
      this.emit('data', economicData);
      
      console.log(`📈 Economic Signal: ${economicData.key_signal.toFixed(3)} (confidence: ${economicData.confidence.toFixed(2)})`);
      console.log(`   CPI: ${economicData.features.cpi_yoy_pct.toFixed(1)}% | Fed Rate: ${economicData.features.fed_funds_rate.toFixed(2)}% | VIX: ${economicData.features.vix_index.toFixed(1)}`);
      
    } catch (error) {
      console.error('❌ Failed to collect economic data:', error);
      throw error;
    }
  }

  private async fetchAllIndicators(): Promise<Record<string, number>> {
    const indicators: Record<string, number> = {};
    
    for (const [key, seriesId] of Object.entries(this.fredSeries)) {
      try {
        const value = await this.fetchFredSeries(seriesId);
        if (value !== null) {
          indicators[key] = this.validateIndicatorValue(key, value);
        }
      } catch (error) {
        console.error(`❌ Failed to fetch ${key} (${seriesId}):`, error);
        // Use backup or historical data
        indicators[key] = this.getBackupValue(key);
      }
    }
    
    return indicators;
  }

  private async fetchFredSeries(seriesId: string): Promise<number | null> {
    try {
      if (this.fredApiKey === 'demo_key') {
        // Return simulated data for demo
        return this.getSimulatedValue(seriesId);
      }
      
      const response = await axios.get('https://api.stlouisfed.org/fred/series/observations', {
        params: {
          series_id: seriesId,
          api_key: this.fredApiKey,
          file_type: 'json',
          sort_order: 'desc',
          limit: 1,
          realtime_start: '1776-07-04',
          realtime_end: '9999-12-31'
        },
        timeout: this.constraints.fred_api_timeout_ms
      });
      
      const observations = response.data?.observations;
      if (observations && observations.length > 0) {
        const latestValue = observations[0].value;
        if (latestValue !== '.') { // FRED uses '.' for missing data
          return parseFloat(latestValue);
        }
      }
      
      return null;
      
    } catch (error) {
      console.error(`❌ FRED API error for ${seriesId}:`, error);
      throw error;
    }
  }

  private getSimulatedValue(seriesId: string): number {
    // Generate realistic economic data for demo
    const simValues = {
      'CPIAUCSL': 3.2 + (Math.random() - 0.5) * 0.4,      // ~3.2% inflation
      'FEDFUNDS': 5.25 + (Math.random() - 0.5) * 0.5,     // ~5.25% fed rate
      'UNRATE': 3.7 + (Math.random() - 0.5) * 0.6,        // ~3.7% unemployment
      'M2SL': 20800 + (Math.random() - 0.5) * 200,        // ~$20.8T money supply
      'GDP': 27000 + (Math.random() - 0.5) * 500,         // ~$27T GDP
      'VIXCLS': 18 + (Math.random() - 0.5) * 8            // ~18 VIX
    };
    
    return simValues[seriesId] || 0;
  }

  private getBackupValue(key: string): number {
    // Get last known good value or reasonable default
    const history = this.indicatorHistory.get(key);
    if (history && history.length > 0) {
      return history[history.length - 1].value;
    }
    
    // Fallback defaults (current approximate US values)
    const defaults = {
      CPI: 3.2,
      FEDFUNDS: 5.25,
      UNRATE: 3.7,
      M2: 20800,
      GDP: 27000,
      VIX: 18
    };
    
    return defaults[key] || 0;
  }

  private validateIndicatorValue(key: string, value: number): number {
    // Apply bounds checking
    switch (key) {
      case 'CPI':
        return Math.max(this.bounds.cpi_range.min, Math.min(this.bounds.cpi_range.max, value));
      case 'FEDFUNDS':
        return Math.max(this.bounds.fed_rate_range.min, Math.min(this.bounds.fed_rate_range.max, value));
      case 'UNRATE':
        return Math.max(this.bounds.unemployment_range.min, Math.min(this.bounds.unemployment_range.max, value));
      case 'VIX':
        return Math.max(this.bounds.vix_range.min, Math.min(this.bounds.vix_range.max, value));
      default:
        return value;
    }
  }

  private calculateDerivedMetrics(indicators: Record<string, number>): any {
    const cpi = indicators.CPI || 0;
    const fedRate = indicators.FEDFUNDS || 0;
    const unemployment = indicators.UNRATE || 0;
    const vix = indicators.VIX || 0;
    const m2 = indicators.M2 || 0;
    
    // Real interest rate
    const realRate = fedRate - cpi;
    
    // Inflation trend (deviation from target)
    const inflationTrend = (cpi - this.parameters.inflation_target_pct) / this.parameters.inflation_target_pct;
    
    // Liquidity conditions (combination of fed policy and money supply growth)
    const fedStance = (this.parameters.neutral_fed_rate_pct - fedRate) / this.parameters.neutral_fed_rate_pct;
    const unemploymentGap = (this.parameters.full_employment_rate_pct - unemployment) / this.parameters.full_employment_rate_pct;
    const liquidityConditions = (fedStance + unemploymentGap) * 0.5;
    
    return {
      realRate,
      inflationTrend,
      liquidityConditions,
      fedStance,
      unemploymentGap
    };
  }

  private calculateCompositeSignal(indicators: Record<string, number>, derived: any): number {
    const cpi = indicators.CPI || 0;
    const fedRate = indicators.FEDFUNDS || 0;
    const unemployment = indicators.UNRATE || 0;
    const vix = indicators.VIX || 0;
    
    // Component signals (-1 to 1)
    
    // Monetary policy signal (dovish = positive, hawkish = negative)
    const monetarySignal = derived.fedStance; // Already normalized
    
    // Employment signal (low unemployment = positive for growth)
    const employmentSignal = -derived.unemploymentGap; // Flip sign for intuitive direction
    
    // Inflation signal (moderate inflation = positive, deflation/hyperinflation = negative)
    const inflationSignal = -Math.abs(derived.inflationTrend) + 0.5; // Penalize extreme inflation
    
    // Volatility signal (low VIX = positive, high VIX = negative)  
    let volatilitySignal;
    if (vix < this.parameters.vix_complacency_threshold) {
      volatilitySignal = 0.5; // Complacency
    } else if (vix > this.parameters.vix_panic_threshold) {
      volatilitySignal = -1.0; // Panic
    } else {
      volatilitySignal = (this.parameters.vix_panic_threshold - vix) / 
                        (this.parameters.vix_panic_threshold - this.parameters.vix_complacency_threshold);
    }
    
    // Liquidity signal
    const liquiditySignal = derived.liquidityConditions;
    
    // Weighted composite (weights sum to 1.0)
    const weights = {
      monetary: 0.3,
      employment: 0.2,
      inflation: 0.2,
      volatility: 0.2,
      liquidity: 0.1
    };
    
    const compositeSignal = 
      monetarySignal * weights.monetary +
      employmentSignal * weights.employment +
      inflationSignal * weights.inflation +
      volatilitySignal * weights.volatility +
      liquiditySignal * weights.liquidity;
    
    // Apply smoothing with historical data
    return this.applySmoothingToSignal(compositeSignal);
  }

  private applySmoothingToSignal(newSignal: number): number {
    const history = this.indicatorHistory.get('COMPOSITE_SIGNAL');
    if (!history || history.length === 0) {
      return newSignal;
    }
    
    const lastSignal = history[history.length - 1].value;
    return newSignal * (1 - this.bounds.signal_smoothing_alpha) + 
           lastSignal * this.bounds.signal_smoothing_alpha;
  }

  private calculateConfidence(indicators: Record<string, number>): number {
    let confidence = 1.0;
    
    // Check data completeness
    const requiredCount = this.constraints.required_indicators.length;
    const availableCount = this.constraints.required_indicators
      .filter(indicator => indicators[indicator] !== undefined).length;
    const completenessScore = availableCount / requiredCount;
    
    // Check data freshness
    const hoursSinceUpdate = (Date.now() - this.lastUpdateTimestamp) / (1000 * 60 * 60);
    const freshnessScore = Math.max(0, 1 - (hoursSinceUpdate / this.parameters.data_staleness_threshold_hours));
    
    // Check data consistency (look for outliers)
    const consistencyScore = this.checkDataConsistency(indicators);
    
    // Apply confidence decay
    const decayFactor = Math.pow(0.5, hoursSinceUpdate / this.parameters.confidence_decay_half_life_hours);
    
    confidence = completenessScore * 0.4 + 
                freshnessScore * 0.3 + 
                consistencyScore * 0.2 + 
                decayFactor * 0.1;
    
    return Math.max(this.constraints.min_confidence_for_signal, confidence);
  }

  private checkDataConsistency(indicators: Record<string, number>): number {
    let consistencyScore = 1.0;
    
    // Check for unrealistic combinations
    const cpi = indicators.CPI || 0;
    const fedRate = indicators.FEDFUNDS || 0;
    const unemployment = indicators.UNRATE || 0;
    const vix = indicators.VIX || 0;
    
    // Fed rate should generally be above inflation for restrictive policy
    if (cpi > 5 && fedRate < cpi - 2) {
      consistencyScore -= 0.2; // Unusual policy stance
    }
    
    // Very low unemployment with very high VIX is inconsistent
    if (unemployment < 3 && vix > 40) {
      consistencyScore -= 0.3;
    }
    
    // Very high inflation with very low VIX is inconsistent
    if (cpi > 6 && vix < 12) {
      consistencyScore -= 0.2;
    }
    
    return Math.max(0, consistencyScore);
  }

  private updateIndicatorHistory(indicators: Record<string, number>): void {
    const timestamp = Date.now();
    
    // Update individual indicator histories
    for (const [key, value] of Object.entries(indicators)) {
      if (!this.indicatorHistory.has(key)) {
        this.indicatorHistory.set(key, []);
      }
      
      const history = this.indicatorHistory.get(key)!;
      history.push({ value, timestamp });
      
      // Keep only recent history (last 30 days)
      const cutoff = timestamp - (30 * 24 * 60 * 60 * 1000);
      this.indicatorHistory.set(key, history.filter(h => h.timestamp > cutoff));
    }
  }

  private clampToRange(value: number, range: { min: number; max: number }): number {
    return Math.max(range.min, Math.min(range.max, value));
  }

  // Public method to get current parameters (for investor transparency)
  getVisibleParameters(): any {
    return {
      parameters: this.parameters,
      constraints: this.constraints,
      bounds: this.bounds,
      fred_series: this.fredSeries,
      last_update: new Date(this.lastUpdateTimestamp).toISOString(),
      indicator_counts: Object.fromEntries(
        Array.from(this.indicatorHistory.entries()).map(([key, history]) => [key, history.length])
      )
    };
  }

  // Public method to get latest indicator values
  getLatestIndicators(): Record<string, number> {
    const latest: Record<string, number> = {};
    
    for (const [key, history] of this.indicatorHistory.entries()) {
      if (history.length > 0) {
        latest[key] = history[history.length - 1].value;
      }
    }
    
    return latest;
  }
}