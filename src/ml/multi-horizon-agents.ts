/**
 * Multi-Horizon Agent System
 * 
 * Creates 15 agents (5 types × 3 horizons) for the HyperVision AI system.
 * Each horizon has distinct characteristics:
 * 
 * - Hourly: Short-lived signals (6h decay), intraday opportunities
 * - Weekly: Persistent signals (48h decay), medium-term trends
 * - Monthly: Structural signals (168h decay), long-term positioning
 */

import { AgentSignal } from './agent-signal';
import { TimeScaledFeatures, TimeHorizon } from './time-scale-feature-store';

// ============================================================================
// Base Agent Interface
// ============================================================================

export interface HorizonAgent {
  type: 'economic' | 'sentiment' | 'crossExchange' | 'onChain' | 'cnnPattern';
  horizon: TimeHorizon;
  decayHours: number;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal;
}

// ============================================================================
// HOURLY AGENT POOL (6-hour signal decay)
// ============================================================================

export class HourlyEconomicAgent implements HorizonAgent {
  type = 'economic' as const;
  horizon: TimeHorizon = 'hourly';
  decayHours = 6;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { returns, volatility, spreadZScore } = features;
    
    // Hourly economic signals: React to immediate macro events
    // Focus: Fed announcements, CPI releases, flash PMI
    let signal = 0;
    let confidence = 0;
    
    // High volatility spike + negative returns = risk-off
    if (volatility > 0.03 && returns < -0.01) {
      signal = -1;
      confidence = 0.75;
    }
    // Low volatility + positive spread = opportunity
    else if (volatility < 0.015 && spreadZScore > 1.5) {
      signal = 1;
      confidence = 0.65;
    }
    // Neutral
    else {
      signal = 0;
      confidence = 0.50;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: volatility > 0.025 ? 0.8 : 0.3,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

export class HourlySentimentAgent implements HorizonAgent {
  type = 'sentiment' as const;
  horizon: TimeHorizon = 'hourly';
  decayHours = 6;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { returns, volumeZScore } = features;
    
    // Hourly sentiment: Social media sentiment shifts
    // Focus: Twitter trends, news sentiment, Reddit activity
    let signal = 0;
    let confidence = 0;
    
    // High volume + positive returns = bullish sentiment surge
    if (volumeZScore > 2.0 && returns > 0.005) {
      signal = 1;
      confidence = 0.70;
    }
    // High volume + negative returns = panic
    else if (volumeZScore > 2.0 && returns < -0.005) {
      signal = -1;
      confidence = 0.65;
    }
    else {
      signal = 0;
      confidence = 0.45;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: volumeZScore > 2.5 ? 0.75 : 0.35,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

export class HourlyCrossExchangeAgent implements HorizonAgent {
  type = 'crossExchange' as const;
  horizon: TimeHorizon = 'hourly';
  decayHours = 6;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { crossExchangeSpread, spreadZScore, flowImbalance } = features;
    
    // Hourly cross-exchange: Arbitrage spreads
    // Focus: Flash arbitrage, funding rate spikes
    let signal = 0;
    let confidence = 0;
    
    // Wide spread + imbalance = arbitrage opportunity
    if (Math.abs(crossExchangeSpread) > 0.002 && spreadZScore > 2.0) {
      signal = crossExchangeSpread > 0 ? 1 : -1;
      confidence = 0.80;
    }
    // Flow imbalance suggests directional pressure
    else if (Math.abs(flowImbalance) > 0.3) {
      signal = flowImbalance > 0 ? 1 : -1;
      confidence = 0.60;
    }
    else {
      signal = 0;
      confidence = 0.40;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: Math.abs(spreadZScore) > 2.5 ? 0.70 : 0.30,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

export class HourlyOnChainAgent implements HorizonAgent {
  type = 'onChain' as const;
  horizon: TimeHorizon = 'hourly';
  decayHours = 6;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { returns, volumeZScore } = features;
    
    // Hourly on-chain: Whale movements, large transfers
    // Focus: Exchange inflows/outflows, whale alerts
    let signal = 0;
    let confidence = 0;
    
    // Large volume + price move = whale activity
    if (volumeZScore > 2.5 && Math.abs(returns) > 0.01) {
      signal = returns > 0 ? 1 : -1;
      confidence = 0.75;
    }
    // Moderate whale activity
    else if (volumeZScore > 1.5) {
      signal = returns > 0 ? 1 : -1;
      confidence = 0.55;
    }
    else {
      signal = 0;
      confidence = 0.40;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: volumeZScore > 3.0 ? 0.80 : 0.35,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

export class HourlyCNNPatternAgent implements HorizonAgent {
  type = 'cnnPattern' as const;
  horizon: TimeHorizon = 'hourly';
  decayHours = 6;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { returns, volatility, spreadZScore } = features;
    
    // Hourly CNN patterns: 1-hour chart patterns
    // Focus: Head & shoulders, double tops/bottoms, flags
    let signal = 0;
    let confidence = 0;
    
    // Reversal pattern: High volatility after trending
    if (volatility > 0.025 && Math.abs(returns) > 0.015) {
      signal = returns > 0 ? -1 : 1; // Contrarian
      confidence = 0.70;
    }
    // Continuation pattern: Steady returns with low volatility
    else if (volatility < 0.015 && Math.abs(returns) > 0.005) {
      signal = returns > 0 ? 1 : -1;
      confidence = 0.65;
    }
    else {
      signal = 0;
      confidence = 0.50;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: volatility > 0.03 ? 0.75 : 0.30,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

// ============================================================================
// WEEKLY AGENT POOL (48-hour signal decay)
// ============================================================================

export class WeeklyEconomicAgent implements HorizonAgent {
  type = 'economic' as const;
  horizon: TimeHorizon = 'weekly';
  decayHours = 48;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { returns, volatility, spreadZScore } = features;
    
    // Weekly economic signals: Medium-term macro trends
    // Focus: PMI trends, unemployment, retail sales
    let signal = 0;
    let confidence = 0;
    
    // Sustained positive returns = economic expansion
    if (returns > 0.02 && volatility < 0.02) {
      signal = 1;
      confidence = 0.80;
    }
    // Sustained negative returns = contraction
    else if (returns < -0.02 && volatility > 0.025) {
      signal = -1;
      confidence = 0.75;
    }
    else {
      signal = 0;
      confidence = 0.55;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: volatility > 0.03 ? 0.70 : 0.25,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

export class WeeklySentimentAgent implements HorizonAgent {
  type = 'sentiment' as const;
  horizon: TimeHorizon = 'weekly';
  decayHours = 48;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { returns, volumeZScore } = features;
    
    // Weekly sentiment: Sentiment trends
    // Focus: Fear & Greed Index trends, social volume
    let signal = 0;
    let confidence = 0;
    
    // Sustained bullish sentiment
    if (returns > 0.015 && volumeZScore > 1.0) {
      signal = 1;
      confidence = 0.75;
    }
    // Sustained bearish sentiment
    else if (returns < -0.015 && volumeZScore > 1.0) {
      signal = -1;
      confidence = 0.70;
    }
    else {
      signal = 0;
      confidence = 0.50;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: volumeZScore > 2.0 ? 0.65 : 0.30,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

export class WeeklyCrossExchangeAgent implements HorizonAgent {
  type = 'crossExchange' as const;
  horizon: TimeHorizon = 'weekly';
  decayHours = 48;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { crossExchangeSpread, spreadZScore, basis } = features;
    
    // Weekly cross-exchange: Persistent basis
    // Focus: Funding carry trades, persistent mispricing
    let signal = 0;
    let confidence = 0;
    
    // Persistent positive basis = carry opportunity
    if (basis > 0.001 && spreadZScore > 1.5) {
      signal = 1;
      confidence = 0.85;
    }
    // Persistent negative basis = short opportunity
    else if (basis < -0.001 && spreadZScore < -1.5) {
      signal = -1;
      confidence = 0.80;
    }
    else {
      signal = 0;
      confidence = 0.45;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: Math.abs(spreadZScore) > 2.0 ? 0.60 : 0.25,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

export class WeeklyOnChainAgent implements HorizonAgent {
  type = 'onChain' as const;
  horizon: TimeHorizon = 'weekly';
  decayHours = 48;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { returns, volatility } = features;
    
    // Weekly on-chain: Network growth trends
    // Focus: Active addresses, transaction volume, miner behavior
    let signal = 0;
    let confidence = 0;
    
    // Steady growth with low volatility = healthy network
    if (returns > 0.01 && volatility < 0.02) {
      signal = 1;
      confidence = 0.80;
    }
    // Network contraction
    else if (returns < -0.01 && volatility > 0.025) {
      signal = -1;
      confidence = 0.75;
    }
    else {
      signal = 0;
      confidence = 0.50;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: volatility > 0.03 ? 0.65 : 0.25,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

export class WeeklyCNNPatternAgent implements HorizonAgent {
  type = 'cnnPattern' as const;
  horizon: TimeHorizon = 'weekly';
  decayHours = 48;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { returns, volatility, spreadZScore } = features;
    
    // Weekly CNN patterns: Daily chart patterns
    // Focus: Cup & handle, ascending triangles, wedges
    let signal = 0;
    let confidence = 0;
    
    // Breakout pattern: Returns accelerating with volume
    if (returns > 0.02 && volatility > 0.02 && spreadZScore > 1.0) {
      signal = 1;
      confidence = 0.80;
    }
    // Breakdown pattern
    else if (returns < -0.02 && volatility > 0.025 && spreadZScore < -1.0) {
      signal = -1;
      confidence = 0.75;
    }
    else {
      signal = 0;
      confidence = 0.55;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: volatility > 0.035 ? 0.70 : 0.30,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

// ============================================================================
// MONTHLY AGENT POOL (168-hour = 1 week signal decay)
// ============================================================================

export class MonthlyEconomicAgent implements HorizonAgent {
  type = 'economic' as const;
  horizon: TimeHorizon = 'monthly';
  decayHours = 168;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { returns, volatility } = features;
    
    // Monthly economic signals: Long-term macro cycles
    // Focus: GDP, inflation cycles, central bank policy
    let signal = 0;
    let confidence = 0;
    
    // Structural bull market: Consistent returns, low volatility
    if (returns > 0.05 && volatility < 0.015) {
      signal = 1;
      confidence = 0.90;
    }
    // Structural bear market: Consistent losses
    else if (returns < -0.05 && volatility > 0.02) {
      signal = -1;
      confidence = 0.85;
    }
    else {
      signal = 0;
      confidence = 0.60;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: volatility > 0.025 ? 0.60 : 0.20,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

export class MonthlySentimentAgent implements HorizonAgent {
  type = 'sentiment' as const;
  horizon: TimeHorizon = 'monthly';
  decayHours = 168;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { returns } = features;
    
    // Monthly sentiment: Long-term sentiment cycles
    // Focus: Institutional sentiment, adoption trends
    let signal = 0;
    let confidence = 0;
    
    // Persistent bullish sentiment
    if (returns > 0.04) {
      signal = 1;
      confidence = 0.85;
    }
    // Persistent bearish sentiment
    else if (returns < -0.04) {
      signal = -1;
      confidence = 0.80;
    }
    else {
      signal = 0;
      confidence = 0.55;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: Math.abs(returns) > 0.06 ? 0.55 : 0.20,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

export class MonthlyCrossExchangeAgent implements HorizonAgent {
  type = 'crossExchange' as const;
  horizon: TimeHorizon = 'monthly';
  decayHours = 168;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { basis, spreadZScore } = features;
    
    // Monthly cross-exchange: Structural mispricing
    // Focus: Long-term basis divergence, liquidity gaps
    let signal = 0;
    let confidence = 0;
    
    // Persistent structural mispricing
    if (Math.abs(basis) > 0.002 && Math.abs(spreadZScore) > 1.0) {
      signal = basis > 0 ? 1 : -1;
      confidence = 0.90;
    }
    else {
      signal = 0;
      confidence = 0.50;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: Math.abs(spreadZScore) > 1.5 ? 0.50 : 0.20,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

export class MonthlyOnChainAgent implements HorizonAgent {
  type = 'onChain' as const;
  horizon: TimeHorizon = 'monthly';
  decayHours = 168;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { returns, volatility } = features;
    
    // Monthly on-chain: Adoption metrics
    // Focus: Long-term holder behavior, network value
    let signal = 0;
    let confidence = 0;
    
    // Long-term adoption growth
    if (returns > 0.03 && volatility < 0.015) {
      signal = 1;
      confidence = 0.90;
    }
    // Declining network value
    else if (returns < -0.03) {
      signal = -1;
      confidence = 0.85;
    }
    else {
      signal = 0;
      confidence = 0.60;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: volatility > 0.02 ? 0.55 : 0.15,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

export class MonthlyCNNPatternAgent implements HorizonAgent {
  type = 'cnnPattern' as const;
  horizon: TimeHorizon = 'monthly';
  decayHours = 168;
  
  generateSignal(features: TimeScaledFeatures): AgentSignal {
    const { returns, volatility } = features;
    
    // Monthly CNN patterns: Weekly chart patterns
    // Focus: Major trend changes, multi-year patterns
    let signal = 0;
    let confidence = 0;
    
    // Major bullish trend
    if (returns > 0.05 && volatility < 0.02) {
      signal = 1;
      confidence = 0.85;
    }
    // Major bearish trend
    else if (returns < -0.05) {
      signal = -1;
      confidence = 0.80;
    }
    else {
      signal = 0;
      confidence = 0.60;
    }
    
    return {
      agentType: this.type,
      direction: signal as -1 | 0 | 1,
      strength: Math.abs(signal) * confidence,
      confidence,
      riskScore: volatility > 0.025 ? 0.60 : 0.20,
      timestamp: new Date(),
      decayRate: 1 / this.decayHours,
    };
  }
}

// ============================================================================
// CROSS-HORIZON SYNC MODULE
// ============================================================================

export interface HorizonAlignment {
  hourly_weekly: number;  // [-1, 1]: -1 = opposite, 0 = neutral, 1 = aligned
  weekly_monthly: number;
  hourly_monthly: number;
  overall: number;
}

export interface HorizonConflict {
  type: 'short_long_divergence' | 'medium_long_divergence' | 'all_horizons_divergence';
  severity: 'low' | 'medium' | 'high';
  signals: {
    hourly?: AgentSignal;
    weekly?: AgentSignal;
    monthly?: AgentSignal;
  };
  description: string;
}

export class CrossHorizonSync {
  /**
   * Calculate alignment scores between horizons
   */
  calculateAlignment(
    hourlySignals: AgentSignal[],
    weeklySignals: AgentSignal[],
    monthlySignals: AgentSignal[]
  ): HorizonAlignment {
    const hourlyAvg = this.averageSignal(hourlySignals);
    const weeklyAvg = this.averageSignal(weeklySignals);
    const monthlyAvg = this.averageSignal(monthlySignals);
    
    // Calculate pairwise alignment (cosine similarity)
    const hourly_weekly = this.signalSimilarity(hourlyAvg, weeklyAvg);
    const weekly_monthly = this.signalSimilarity(weeklyAvg, monthlyAvg);
    const hourly_monthly = this.signalSimilarity(hourlyAvg, monthlyAvg);
    
    // Overall alignment (average of pairwise)
    const overall = (hourly_weekly + weekly_monthly + hourly_monthly) / 3;
    
    return {
      hourly_weekly,
      weekly_monthly,
      hourly_monthly,
      overall,
    };
  }
  
  /**
   * Detect conflicts between horizons
   */
  detectConflicts(
    hourlySignals: AgentSignal[],
    weeklySignals: AgentSignal[],
    monthlySignals: AgentSignal[]
  ): HorizonConflict[] {
    const conflicts: HorizonConflict[] = [];
    
    const hourlyAvg = this.averageSignal(hourlySignals);
    const weeklyAvg = this.averageSignal(weeklySignals);
    const monthlyAvg = this.averageSignal(monthlySignals);
    
    // Short-long divergence (hourly vs monthly)
    if (hourlyAvg * monthlyAvg < 0 && Math.abs(hourlyAvg) > 0.3 && Math.abs(monthlyAvg) > 0.3) {
      conflicts.push({
        type: 'short_long_divergence',
        severity: Math.abs(hourlyAvg - monthlyAvg) > 1.5 ? 'high' : 'medium',
        signals: {
          hourly: hourlySignals[0],
          monthly: monthlySignals[0],
        },
        description: `Hourly signals ${hourlyAvg > 0 ? 'bullish' : 'bearish'}, Monthly signals ${monthlyAvg > 0 ? 'bullish' : 'bearish'}`,
      });
    }
    
    // Medium-long divergence (weekly vs monthly)
    if (weeklyAvg * monthlyAvg < 0 && Math.abs(weeklyAvg) > 0.3 && Math.abs(monthlyAvg) > 0.3) {
      conflicts.push({
        type: 'medium_long_divergence',
        severity: Math.abs(weeklyAvg - monthlyAvg) > 1.2 ? 'high' : 'medium',
        signals: {
          weekly: weeklySignals[0],
          monthly: monthlySignals[0],
        },
        description: `Weekly signals ${weeklyAvg > 0 ? 'bullish' : 'bearish'}, Monthly signals ${monthlyAvg > 0 ? 'bullish' : 'bearish'}`,
      });
    }
    
    // All horizons diverging
    if (hourlyAvg * weeklyAvg < 0 && weeklyAvg * monthlyAvg < 0) {
      conflicts.push({
        type: 'all_horizons_divergence',
        severity: 'high',
        signals: {
          hourly: hourlySignals[0],
          weekly: weeklySignals[0],
          monthly: monthlySignals[0],
        },
        description: 'All three horizons show conflicting signals',
      });
    }
    
    return conflicts;
  }
  
  /**
   * Manage correlation across horizons
   */
  manageCorrelation(
    hourlySignals: AgentSignal[],
    weeklySignals: AgentSignal[],
    monthlySignals: AgentSignal[]
  ): Map<string, number> {
    const correlations = new Map<string, number>();
    
    // Calculate correlation by agent type
    const agentTypes: Array<'economic' | 'sentiment' | 'crossExchange' | 'onChain' | 'cnnPattern'> = [
      'economic', 'sentiment', 'crossExchange', 'onChain', 'cnnPattern'
    ];
    
    for (const agentType of agentTypes) {
      const hourly = hourlySignals.find(s => s.agentType === agentType);
      const weekly = weeklySignals.find(s => s.agentType === agentType);
      const monthly = monthlySignals.find(s => s.agentType === agentType);
      
      if (hourly && weekly && monthly) {
        const correlation = this.calculateThreeWayCorrelation(
          hourly.direction,
          weekly.direction,
          monthly.direction
        );
        correlations.set(agentType, correlation);
      }
    }
    
    return correlations;
  }
  
  // ============================================================================
  // Helper Methods
  // ============================================================================
  
  private averageSignal(signals: AgentSignal[]): number {
    if (signals.length === 0) return 0;
    const sum = signals.reduce((acc, s) => acc + s.direction * s.confidence, 0);
    return sum / signals.length;
  }
  
  private signalSimilarity(signal1: number, signal2: number): number {
    // Cosine similarity for signals in [-1, 1]
    // Returns: 1 = same direction, 0 = orthogonal, -1 = opposite
    if (signal1 === 0 || signal2 === 0) return 0;
    return (signal1 * signal2) / (Math.abs(signal1) * Math.abs(signal2));
  }
  
  private calculateThreeWayCorrelation(s1: number, s2: number, s3: number): number {
    // Average pairwise correlation
    const corr12 = s1 * s2;
    const corr23 = s2 * s3;
    const corr13 = s1 * s3;
    return (corr12 + corr23 + corr13) / 3;
  }
}

// ============================================================================
// MULTI-HORIZON AGENT POOL (Main Class)
// ============================================================================

export interface MultiHorizonAgentPool {
  hourly: {
    economic: HourlyEconomicAgent;
    sentiment: HourlySentimentAgent;
    crossExchange: HourlyCrossExchangeAgent;
    onChain: HourlyOnChainAgent;
    cnnPattern: HourlyCNNPatternAgent;
  };
  weekly: {
    economic: WeeklyEconomicAgent;
    sentiment: WeeklySentimentAgent;
    crossExchange: WeeklyCrossExchangeAgent;
    onChain: WeeklyOnChainAgent;
    cnnPattern: WeeklyCNNPatternAgent;
  };
  monthly: {
    economic: MonthlyEconomicAgent;
    sentiment: MonthlySentimentAgent;
    crossExchange: MonthlyCrossExchangeAgent;
    onChain: MonthlyOnChainAgent;
    cnnPattern: MonthlyCNNPatternAgent;
  };
}

export class MultiHorizonAgentSystem {
  private agents: MultiHorizonAgentPool;
  private crossHorizonSync: CrossHorizonSync;
  
  constructor() {
    // Initialize all 15 agents (5 types × 3 horizons)
    this.agents = {
      hourly: {
        economic: new HourlyEconomicAgent(),
        sentiment: new HourlySentimentAgent(),
        crossExchange: new HourlyCrossExchangeAgent(),
        onChain: new HourlyOnChainAgent(),
        cnnPattern: new HourlyCNNPatternAgent(),
      },
      weekly: {
        economic: new WeeklyEconomicAgent(),
        sentiment: new WeeklySentimentAgent(),
        crossExchange: new WeeklyCrossExchangeAgent(),
        onChain: new WeeklyOnChainAgent(),
        cnnPattern: new WeeklyCNNPatternAgent(),
      },
      monthly: {
        economic: new MonthlyEconomicAgent(),
        sentiment: new MonthlySentimentAgent(),
        crossExchange: new MonthlyCrossExchangeAgent(),
        onChain: new MonthlyOnChainAgent(),
        cnnPattern: new MonthlyCNNPatternAgent(),
      },
    };
    
    this.crossHorizonSync = new CrossHorizonSync();
  }
  
  /**
   * Generate signals from all agents for all horizons
   */
  generateAllSignals(
    hourlyFeatures: TimeScaledFeatures,
    weeklyFeatures: TimeScaledFeatures,
    monthlyFeatures: TimeScaledFeatures
  ): {
    hourly: AgentSignal[];
    weekly: AgentSignal[];
    monthly: AgentSignal[];
    alignment: HorizonAlignment;
    conflicts: HorizonConflict[];
    correlations: Map<string, number>;
  } {
    // Generate hourly signals
    const hourlySignals: AgentSignal[] = [
      this.agents.hourly.economic.generateSignal(hourlyFeatures),
      this.agents.hourly.sentiment.generateSignal(hourlyFeatures),
      this.agents.hourly.crossExchange.generateSignal(hourlyFeatures),
      this.agents.hourly.onChain.generateSignal(hourlyFeatures),
      this.agents.hourly.cnnPattern.generateSignal(hourlyFeatures),
    ];
    
    // Generate weekly signals
    const weeklySignals: AgentSignal[] = [
      this.agents.weekly.economic.generateSignal(weeklyFeatures),
      this.agents.weekly.sentiment.generateSignal(weeklyFeatures),
      this.agents.weekly.crossExchange.generateSignal(weeklyFeatures),
      this.agents.weekly.onChain.generateSignal(weeklyFeatures),
      this.agents.weekly.cnnPattern.generateSignal(weeklyFeatures),
    ];
    
    // Generate monthly signals
    const monthlySignals: AgentSignal[] = [
      this.agents.monthly.economic.generateSignal(monthlyFeatures),
      this.agents.monthly.sentiment.generateSignal(monthlyFeatures),
      this.agents.monthly.crossExchange.generateSignal(monthlyFeatures),
      this.agents.monthly.onChain.generateSignal(monthlyFeatures),
      this.agents.monthly.cnnPattern.generateSignal(monthlyFeatures),
    ];
    
    // Cross-horizon analysis
    const alignment = this.crossHorizonSync.calculateAlignment(
      hourlySignals,
      weeklySignals,
      monthlySignals
    );
    
    const conflicts = this.crossHorizonSync.detectConflicts(
      hourlySignals,
      weeklySignals,
      monthlySignals
    );
    
    const correlations = this.crossHorizonSync.manageCorrelation(
      hourlySignals,
      weeklySignals,
      monthlySignals
    );
    
    return {
      hourly: hourlySignals,
      weekly: weeklySignals,
      monthly: monthlySignals,
      alignment,
      conflicts,
      correlations,
    };
  }
  
  /**
   * Get agents for a specific horizon
   */
  getAgentsByHorizon(horizon: TimeHorizon): HorizonAgent[] {
    const pool = this.agents[horizon];
    return [
      pool.economic,
      pool.sentiment,
      pool.crossExchange,
      pool.onChain,
      pool.cnnPattern,
    ];
  }
}
