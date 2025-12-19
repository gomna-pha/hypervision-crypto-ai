/**
 * Standardized Agent Signal Format
 * 
 * All agents output signals in this format for consistency
 * and integration with GA/XGBoost meta-model
 */

export interface AgentSignal {
  // Core signal properties
  agentId: string;              // Unique agent identifier
  signal: number;               // -1.0 (bearish) to +1.0 (bullish)
  confidence: number;           // 0.0 (no confidence) to 1.0 (high confidence)
  timestamp: Date;              // Signal generation time
  
  // Features used to generate signal
  features: Record<string, number>;
  
  // Human-readable explanation
  explanation: string;
  
  // Arbitrage-specific fields
  opportunityType: 'spot_perp' | 'cross_exchange' | 'funding_rate' | 'statistical' | 'volatility';
  expectedAlpha: number;        // Expected profit in basis points (bps)
  riskScore: number;            // 0.0 (low risk) to 1.0 (high risk)
  
  // Metadata
  version: string;              // Agent version (for tracking)
  latencyMs: number;            // Signal generation latency
}

export interface AgentConfig {
  refreshInterval: number;      // How often to update signal (ms)
  enabled: boolean;             // Is agent active?
  weight: number;               // Weight in ensemble (0-1)
}

export abstract class BaseAgent {
  protected agentId: string;
  protected version: string;
  protected config: AgentConfig;
  protected lastSignal: AgentSignal | null;

  constructor(agentId: string, version: string, config: Partial<AgentConfig> = {}) {
    this.agentId = agentId;
    this.version = version;
    this.config = {
      refreshInterval: config.refreshInterval || 60000, // Default: 1 minute
      enabled: config.enabled !== undefined ? config.enabled : true,
      weight: config.weight || 0.2, // Default: 20% weight
    };
    this.lastSignal = null;
  }

  /**
   * Abstract method: each agent implements its own logic
   */
  abstract generateSignal(marketData: any): Promise<AgentSignal>;

  /**
   * Get last generated signal
   */
  getLastSignal(): AgentSignal | null {
    return this.lastSignal;
  }

  /**
   * Check if signal is stale (older than refresh interval)
   */
  isSignalStale(): boolean {
    if (!this.lastSignal) return true;
    const now = new Date();
    const age = now.getTime() - this.lastSignal.timestamp.getTime();
    return age > this.config.refreshInterval;
  }

  /**
   * Get agent configuration
   */
  getConfig(): AgentConfig {
    return this.config;
  }

  /**
   * Update agent configuration
   */
  updateConfig(config: Partial<AgentConfig>): void {
    this.config = { ...this.config, ...config };
  }
}

/**
 * Economic Agent - Macro Risk & Liquidity Stress
 */
export class EconomicAgent extends BaseAgent {
  constructor(config?: Partial<AgentConfig>) {
    super('economic_agent', 'v2.0.0', config);
  }

  async generateSignal(marketData: any): Promise<AgentSignal> {
    const startTime = Date.now();

    // Extract features
    const fedRate = marketData.fedRate || 4.25; // %
    const cpi = marketData.cpi || 3.2; // %
    const gdp = marketData.gdp || 2.8; // %
    const vix = marketData.vix || 18; // Volatility index
    const liquidityScore = marketData.liquidityScore || 75; // 0-100

    // Calculate composite score
    // Hawkish Fed (high rates) = bearish
    // High inflation (CPI) = bearish
    // Strong GDP = bullish
    // High VIX = bearish (risk-off)
    // Low liquidity = bearish
    
    const fedScore = (6 - fedRate) / 6 * 100; // Normalize: low rates = bullish
    const cpiScore = (5 - cpi) / 5 * 100; // Normalize: low inflation = bullish
    const gdpScore = gdp / 5 * 100; // Normalize: high GDP = bullish
    const vixScore = (30 - vix) / 30 * 100; // Normalize: low VIX = bullish
    
    const compositeScore = (
      fedScore * 0.3 +
      cpiScore * 0.25 +
      gdpScore * 0.2 +
      vixScore * 0.15 +
      liquidityScore * 0.1
    );

    // Convert to signal (-1 to +1)
    const signal = (compositeScore - 50) / 50;

    // Confidence based on agreement between indicators
    const indicators = [fedScore, cpiScore, gdpScore, vixScore, liquidityScore];
    const variance = this.calculateVariance(indicators);
    const confidence = 1 - Math.min(1, variance / 1000); // Lower variance = higher confidence

    // Risk score (high VIX or low liquidity = high risk)
    const riskScore = (vix / 30 + (100 - liquidityScore) / 100) / 2;

    // Expected alpha (macro signals typically longer-term, lower alpha)
    const expectedAlpha = Math.abs(signal) * 10; // 0-10 bps

    const explanation = this.generateExplanation(
      fedRate,
      cpi,
      gdp,
      vix,
      liquidityScore,
      signal
    );

    const agentSignal: AgentSignal = {
      agentId: this.agentId,
      signal: Math.max(-1, Math.min(1, signal)),
      confidence: Math.max(0, Math.min(1, confidence)),
      timestamp: new Date(),
      features: {
        fedRate,
        cpi,
        gdp,
        vix,
        liquidityScore,
        compositeScore,
      },
      explanation,
      opportunityType: 'statistical', // Macro signals → statistical arbitrage
      expectedAlpha,
      riskScore: Math.max(0, Math.min(1, riskScore)),
      version: this.version,
      latencyMs: Date.now() - startTime,
    };

    this.lastSignal = agentSignal;
    return agentSignal;
  }

  private calculateVariance(values: number[]): number {
    const mean = values.reduce((sum, v) => sum + v, 0) / values.length;
    const squaredDiffs = values.map(v => Math.pow(v - mean, 2));
    return squaredDiffs.reduce((sum, d) => sum + d, 0) / values.length;
  }

  private generateExplanation(
    fedRate: number,
    cpi: number,
    gdp: number,
    vix: number,
    liquidityScore: number,
    signal: number
  ): string {
    const sentiment = signal > 0.3 ? 'bullish' : signal < -0.3 ? 'bearish' : 'neutral';
    
    let explanation = `Economic Agent: ${sentiment.toUpperCase()} outlook. `;
    
    if (fedRate > 4.5) {
      explanation += `Hawkish Fed (${fedRate.toFixed(2)}%) suppressing risk appetite. `;
    } else if (fedRate < 3.5) {
      explanation += `Dovish Fed (${fedRate.toFixed(2)}%) supporting risk assets. `;
    }

    if (cpi > 4.0) {
      explanation += `High inflation (${cpi.toFixed(1)}%) creates macro uncertainty. `;
    }

    if (vix > 25) {
      explanation += `Elevated VIX (${vix.toFixed(0)}) signals market stress. `;
    }

    if (liquidityScore < 60) {
      explanation += `Low liquidity (${liquidityScore}) increases execution risk.`;
    }

    return explanation;
  }
}

/**
 * Sentiment Agent - Narrative & Flow Momentum
 */
export class SentimentAgent extends BaseAgent {
  constructor(config?: Partial<AgentConfig>) {
    super('sentiment_agent', 'v2.0.0', config);
  }

  async generateSignal(marketData: any): Promise<AgentSignal> {
    const startTime = Date.now();

    // Extract features
    const fearGreed = marketData.fearGreed || 50; // 0-100 (0=fear, 100=greed)
    const googleTrends = marketData.googleTrends || 50; // 0-100
    const socialSentiment = marketData.socialSentiment || 50; // 0-100
    const volumeRatio = marketData.volumeRatio || 1.0; // Current/avg volume

    // Calculate composite score
    const compositeScore = (
      fearGreed * 0.4 +
      googleTrends * 0.3 +
      socialSentiment * 0.2 +
      (volumeRatio > 1 ? volumeRatio * 10 : 50) * 0.1
    );

    // Convert to signal (-1 to +1)
    // Contrarian strategy: extreme fear = buy, extreme greed = sell
    let signal: number;
    if (fearGreed < 25) {
      signal = 0.5; // Extreme fear → bullish (contrarian)
    } else if (fearGreed > 75) {
      signal = -0.5; // Extreme greed → bearish (contrarian)
    } else {
      signal = (compositeScore - 50) / 50; // Follow sentiment in neutral zone
    }

    // Confidence based on sentiment extremes (more confident at extremes)
    const distanceFrom50 = Math.abs(fearGreed - 50);
    const confidence = distanceFrom50 / 50;

    // Risk score (extreme sentiment = higher risk)
    const riskScore = distanceFrom50 / 50 * 0.7;

    // Expected alpha (sentiment trades can be profitable in extremes)
    const expectedAlpha = distanceFrom50 / 50 * 50; // 0-50 bps

    const explanation = this.generateExplanation(
      fearGreed,
      googleTrends,
      socialSentiment,
      signal
    );

    const agentSignal: AgentSignal = {
      agentId: this.agentId,
      signal: Math.max(-1, Math.min(1, signal)),
      confidence: Math.max(0, Math.min(1, confidence)),
      timestamp: new Date(),
      features: {
        fearGreed,
        googleTrends,
        socialSentiment,
        volumeRatio,
        compositeScore,
      },
      explanation,
      opportunityType: 'statistical',
      expectedAlpha,
      riskScore: Math.max(0, Math.min(1, riskScore)),
      version: this.version,
      latencyMs: Date.now() - startTime,
    };

    this.lastSignal = agentSignal;
    return agentSignal;
  }

  private generateExplanation(
    fearGreed: number,
    googleTrends: number,
    socialSentiment: number,
    signal: number
  ): string {
    let explanation = 'Sentiment Agent: ';

    if (fearGreed < 25) {
      explanation += `EXTREME FEAR (${fearGreed}) - Contrarian buy signal. `;
    } else if (fearGreed > 75) {
      explanation += `EXTREME GREED (${fearGreed}) - Contrarian sell signal. `;
    } else {
      explanation += `Neutral sentiment (${fearGreed}). `;
    }

    if (googleTrends > 70) {
      explanation += `High retail attention (${googleTrends}). `;
    }

    if (socialSentiment < 30) {
      explanation += `Negative social sentiment (${socialSentiment}).`;
    }

    return explanation;
  }
}

/**
 * Cross-Exchange Agent - Price / Basis Mispricing
 */
export class CrossExchangeAgent extends BaseAgent {
  constructor(config?: Partial<AgentConfig>) {
    super('cross_exchange_agent', 'v2.0.0', config);
  }

  async generateSignal(marketData: any): Promise<AgentSignal> {
    const startTime = Date.now();

    // Extract features
    const binancePrice = marketData.binancePrice || 96500;
    const coinbasePrice = marketData.coinbasePrice || 96530;
    const krakenPrice = marketData.krakenPrice || 96520;

    const binanceLiquidity = marketData.binanceLiquidity || 1000000; // USD
    const coinbaseLiquidity = marketData.coinbaseLiquidity || 800000;

    // Calculate spreads
    const binanceCoinbaseSpread = (coinbasePrice - binancePrice) / binancePrice * 10000; // bps
    const binanceKrakenSpread = (krakenPrice - binancePrice) / binancePrice * 10000; // bps

    // Calculate z-scores (how many std devs from mean)
    const spreadMean = 15; // Historical mean spread (bps)
    const spreadStd = 10; // Historical std dev
    const spreadZScore = (Math.abs(binanceCoinbaseSpread) - spreadMean) / spreadStd;

    // Signal: positive if spread is wide (arbitrage opportunity)
    const signal = Math.min(1, spreadZScore / 3); // Normalize by 3 std devs

    // Confidence based on liquidity
    const minLiquidity = Math.min(binanceLiquidity, coinbaseLiquidity);
    const confidence = Math.min(1, minLiquidity / 1000000); // $1M = 100% confidence

    // Risk score (low liquidity = high risk)
    const riskScore = 1 - confidence;

    // Expected alpha (actual spread minus fees)
    const fees = 20; // Typical fees (bps)
    const expectedAlpha = Math.max(0, Math.abs(binanceCoinbaseSpread) - fees);

    const explanation = this.generateExplanation(
      binancePrice,
      coinbasePrice,
      binanceCoinbaseSpread,
      expectedAlpha
    );

    const agentSignal: AgentSignal = {
      agentId: this.agentId,
      signal: Math.max(-1, Math.min(1, signal)),
      confidence: Math.max(0, Math.min(1, confidence)),
      timestamp: new Date(),
      features: {
        binancePrice,
        coinbasePrice,
        krakenPrice,
        binanceCoinbaseSpread,
        binanceKrakenSpread,
        spreadZScore,
        binanceLiquidity,
        coinbaseLiquidity,
      },
      explanation,
      opportunityType: 'cross_exchange',
      expectedAlpha,
      riskScore: Math.max(0, Math.min(1, riskScore)),
      version: this.version,
      latencyMs: Date.now() - startTime,
    };

    this.lastSignal = agentSignal;
    return agentSignal;
  }

  private generateExplanation(
    binancePrice: number,
    coinbasePrice: number,
    spread: number,
    expectedAlpha: number
  ): string {
    const cheaper = binancePrice < coinbasePrice ? 'Binance' : 'Coinbase';
    const expensive = cheaper === 'Binance' ? 'Coinbase' : 'Binance';

    return (
      `Cross-Exchange Agent: ${cheaper} trading $${Math.abs(spread * binancePrice / 10000).toFixed(2)} ` +
      `${cheaper === 'Binance' ? 'below' : 'above'} ${expensive}. ` +
      `Spread: ${Math.abs(spread).toFixed(1)} bps. ` +
      `Expected alpha: ${expectedAlpha.toFixed(1)} bps after fees.`
    );
  }
}

/**
 * On-Chain Agent - Flow Pressure & Structural Bias
 */
export class OnChainAgent extends BaseAgent {
  constructor(config?: Partial<AgentConfig>) {
    super('on_chain_agent', 'v2.0.0', config);
  }

  async generateSignal(marketData: any): Promise<AgentSignal> {
    const startTime = Date.now();

    // Extract features
    const exchangeNetflow = marketData.exchangeNetflow || -5000; // BTC (negative = outflow)
    const whaleTransactions = marketData.whaleTransactions || 50; // Count
    const sopr = marketData.sopr || 1.02; // Spent Output Profit Ratio
    const mvrv = marketData.mvrv || 1.8; // Market Value to Realized Value

    // Calculate composite score
    // Negative netflow (outflow) = bullish (accumulation)
    // High whale activity = significant
    // SOPR > 1 = profit-taking (bearish), SOPR < 1 = loss-taking (bullish)
    // MVRV > 2 = overvalued (bearish), MVRV < 1 = undervalued (bullish)

    const netflowScore = -exchangeNetflow / 10000 * 50 + 50; // Normalize
    const soprScore = (1 - (sopr - 1)) * 100; // SOPR close to 1 = neutral
    const mvrvScore = (2.5 - mvrv) / 2.5 * 100; // Lower MVRV = bullish

    const compositeScore = (
      netflowScore * 0.4 +
      soprScore * 0.3 +
      mvrvScore * 0.3
    );

    // Convert to signal
    const signal = (compositeScore - 50) / 50;

    // Confidence based on whale activity
    const confidence = Math.min(1, whaleTransactions / 100);

    // Risk score (high whale activity = higher risk due to potential manipulation)
    const riskScore = Math.min(1, whaleTransactions / 100) * 0.6;

    // Expected alpha (on-chain signals are slower, lower alpha)
    const expectedAlpha = Math.abs(signal) * 15; // 0-15 bps

    const explanation = this.generateExplanation(
      exchangeNetflow,
      whaleTransactions,
      sopr,
      mvrv,
      signal
    );

    const agentSignal: AgentSignal = {
      agentId: this.agentId,
      signal: Math.max(-1, Math.min(1, signal)),
      confidence: Math.max(0, Math.min(1, confidence)),
      timestamp: new Date(),
      features: {
        exchangeNetflow,
        whaleTransactions,
        sopr,
        mvrv,
        compositeScore,
      },
      explanation,
      opportunityType: 'statistical',
      expectedAlpha,
      riskScore: Math.max(0, Math.min(1, riskScore)),
      version: this.version,
      latencyMs: Date.now() - startTime,
    };

    this.lastSignal = agentSignal;
    return agentSignal;
  }

  private generateExplanation(
    exchangeNetflow: number,
    whaleTransactions: number,
    sopr: number,
    mvrv: number,
    signal: number
  ): string {
    let explanation = 'On-Chain Agent: ';

    if (exchangeNetflow < -3000) {
      explanation += `Strong outflow (${exchangeNetflow.toFixed(0)} BTC) - accumulation phase. `;
    } else if (exchangeNetflow > 3000) {
      explanation += `Strong inflow (+${exchangeNetflow.toFixed(0)} BTC) - distribution phase. `;
    }

    if (whaleTransactions > 80) {
      explanation += `High whale activity (${whaleTransactions} txs). `;
    }

    if (sopr > 1.05) {
      explanation += `Profit-taking (SOPR ${sopr.toFixed(2)}). `;
    } else if (sopr < 0.95) {
      explanation += `Loss-taking (SOPR ${sopr.toFixed(2)}). `;
    }

    return explanation;
  }
}

/**
 * CNN Pattern Agent - Temporal Arbitrage Patterns
 */
export class CNNPatternAgent extends BaseAgent {
  constructor(config?: Partial<AgentConfig>) {
    super('cnn_pattern_agent', 'v2.0.0', config);
  }

  async generateSignal(marketData: any): Promise<AgentSignal> {
    const startTime = Date.now();

    // Extract features
    const pattern = marketData.pattern || 'Bull Flag'; // Detected pattern
    const patternConfidence = marketData.patternConfidence || 0.75; // 0-1
    const fearGreed = marketData.fearGreed || 50; // For sentiment reinforcement

    // Determine signal direction based on pattern
    const bullishPatterns = ['Bull Flag', 'Cup & Handle', 'Double Bottom', 'Inverse Head & Shoulders'];
    const bearishPatterns = ['Bear Flag', 'Head & Shoulders', 'Double Top'];

    let baseSignal: number;
    if (bullishPatterns.includes(pattern)) {
      baseSignal = patternConfidence;
    } else if (bearishPatterns.includes(pattern)) {
      baseSignal = -patternConfidence;
    } else {
      baseSignal = 0;
    }

    // Sentiment reinforcement (Baumeister et al., 2001)
    let reinforcementMultiplier = 1.0;
    if (baseSignal > 0 && fearGreed < 25) {
      reinforcementMultiplier = 1.15 + (25 - fearGreed) / 25 * 0.15; // 1.15-1.30×
    } else if (baseSignal < 0 && fearGreed > 75) {
      reinforcementMultiplier = 1.10 + (fearGreed - 75) / 25 * 0.15; // 1.10-1.25×
    } else if ((baseSignal > 0 && fearGreed > 75) || (baseSignal < 0 && fearGreed < 25)) {
      reinforcementMultiplier = 0.75; // Conflicting signals
    }

    const signal = baseSignal * reinforcementMultiplier;
    const confidence = Math.min(0.96, patternConfidence * reinforcementMultiplier);

    // Risk score (pattern trades can fail)
    const riskScore = 1 - confidence;

    // Expected alpha (pattern trades typically short-term, higher alpha)
    const expectedAlpha = Math.abs(signal) * 40; // 0-40 bps

    const explanation = this.generateExplanation(
      pattern,
      patternConfidence,
      fearGreed,
      reinforcementMultiplier
    );

    const agentSignal: AgentSignal = {
      agentId: this.agentId,
      signal: Math.max(-1, Math.min(1, signal)),
      confidence: Math.max(0, Math.min(1, confidence)),
      timestamp: new Date(),
      features: {
        pattern,
        patternConfidence,
        fearGreed,
        reinforcementMultiplier,
        baseSignal,
      },
      explanation,
      opportunityType: 'volatility',
      expectedAlpha,
      riskScore: Math.max(0, Math.min(1, riskScore)),
      version: this.version,
      latencyMs: Date.now() - startTime,
    };

    this.lastSignal = agentSignal;
    return agentSignal;
  }

  private generateExplanation(
    pattern: string,
    patternConfidence: number,
    fearGreed: number,
    reinforcementMultiplier: number
  ): string {
    const direction = pattern.includes('Bull') || pattern.includes('Bottom') || pattern.includes('Inverse') || pattern.includes('Cup') ? 'bullish' : 'bearish';

    let explanation = `CNN Pattern Agent: Detected ${pattern} (${(patternConfidence * 100).toFixed(0)}% confidence) - ${direction} signal. `;

    if (reinforcementMultiplier > 1.1) {
      explanation += `Sentiment reinforcement (${reinforcementMultiplier.toFixed(2)}×) boosts confidence. `;
    } else if (reinforcementMultiplier < 0.9) {
      explanation += `Conflicting sentiment (${reinforcementMultiplier.toFixed(2)}×) reduces confidence.`;
    }

    return explanation;
  }
}
