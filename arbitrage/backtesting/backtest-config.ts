import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';

export interface BacktestStrategy {
  id: string;
  name: string;
  description: string;
  parameters: {
    minConfidence: number;
    maxRisk: number;
    maxPositionSize: number;
    timeHorizon: number;
    riskRewardRatio: number;
    maxConcurrentTrades: number;
  };
  filters: {
    minSpread: number;
    minLiquidity: number;
    maxVolatility: number;
    allowedExchanges: string[];
    allowedSymbols: string[];
  };
  entryConditions: {
    sentimentThreshold: number;
    volumeSpike: number;
    priceDeviation: number;
    technicalIndicators: string[];
  };
  exitConditions: {
    profitTarget: number;
    stopLoss: number;
    maxHoldTime: number;
    reversalSignal: boolean;
  };
}

export interface BacktestScenario {
  id: string;
  name: string;
  description: string;
  strategy: BacktestStrategy;
  config: {
    startDate: string; // ISO date string
    endDate: string;
    initialCapital: number;
    maxPositionSize: number;
    maxConcurrentPositions: number;
    commission: number;
    slippage: number;
    riskFreeRate: number;
    timeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
    symbols: string[];
    exchanges: string[];
  };
  marketConditions: {
    volatilityRegime: 'low' | 'medium' | 'high';
    trendDirection: 'bullish' | 'bearish' | 'sideways';
    liquidityCondition: 'normal' | 'stressed' | 'crisis';
    newsEvents: boolean;
  };
  benchmarks: {
    buyAndHold: boolean;
    marketNeutral: boolean;
    riskParity: boolean;
    customBenchmark?: string;
  };
}

export interface BacktestSuite {
  id: string;
  name: string;
  description: string;
  scenarios: BacktestScenario[];
  validationRules: {
    minSharpeRatio: number;
    maxDrawdown: number;
    minWinRate: number;
    minProfitFactor: number;
    maxVaR: number;
  };
  reportingOptions: {
    includeEquityCurve: boolean;
    includeDrawdownChart: boolean;
    includeMonthlyPerformance: boolean;
    includeRiskMetrics: boolean;
    includeSensitivityAnalysis: boolean;
  };
}

export class BacktestConfigManager {
  private configPath: string;
  private strategiesPath: string;
  private scenariosPath: string;
  private suitesPath: string;

  constructor(basePath: string = './arbitrage/backtesting/configs') {
    this.configPath = basePath;
    this.strategiesPath = join(basePath, 'strategies');
    this.scenariosPath = join(basePath, 'scenarios');
    this.suitesPath = join(basePath, 'suites');
  }

  // Strategy Management
  createDefaultStrategies(): BacktestStrategy[] {
    const strategies: BacktestStrategy[] = [
      {
        id: 'conservative_arbitrage',
        name: 'Conservative Arbitrage',
        description: 'Low-risk arbitrage strategy focusing on high-probability opportunities',
        parameters: {
          minConfidence: 0.8,
          maxRisk: 0.02,
          maxPositionSize: 0.1,
          timeHorizon: 300, // 5 minutes
          riskRewardRatio: 2.0,
          maxConcurrentTrades: 3
        },
        filters: {
          minSpread: 0.001, // 10 basis points
          minLiquidity: 10000,
          maxVolatility: 0.05,
          allowedExchanges: ['binance', 'coinbase', 'kraken'],
          allowedSymbols: ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
        },
        entryConditions: {
          sentimentThreshold: 0.6,
          volumeSpike: 1.5,
          priceDeviation: 0.002,
          technicalIndicators: ['rsi', 'bollinger_bands']
        },
        exitConditions: {
          profitTarget: 0.01,
          stopLoss: 0.005,
          maxHoldTime: 600, // 10 minutes
          reversalSignal: true
        }
      },
      {
        id: 'aggressive_arbitrage',
        name: 'Aggressive Arbitrage',
        description: 'High-frequency arbitrage strategy for experienced traders',
        parameters: {
          minConfidence: 0.6,
          maxRisk: 0.05,
          maxPositionSize: 0.25,
          timeHorizon: 120, // 2 minutes
          riskRewardRatio: 1.5,
          maxConcurrentTrades: 10
        },
        filters: {
          minSpread: 0.0005, // 5 basis points
          minLiquidity: 5000,
          maxVolatility: 0.15,
          allowedExchanges: ['binance', 'coinbase', 'kraken', 'bybit', 'okx'],
          allowedSymbols: ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'SOL/USDT']
        },
        entryConditions: {
          sentimentThreshold: 0.4,
          volumeSpike: 1.2,
          priceDeviation: 0.001,
          technicalIndicators: ['macd', 'ema_crossover']
        },
        exitConditions: {
          profitTarget: 0.008,
          stopLoss: 0.003,
          maxHoldTime: 300, // 5 minutes
          reversalSignal: false
        }
      },
      {
        id: 'market_making',
        name: 'Market Making Strategy',
        description: 'Provide liquidity while capturing bid-ask spreads',
        parameters: {
          minConfidence: 0.7,
          maxRisk: 0.03,
          maxPositionSize: 0.15,
          timeHorizon: 60, // 1 minute
          riskRewardRatio: 1.2,
          maxConcurrentTrades: 20
        },
        filters: {
          minSpread: 0.0002, // 2 basis points
          minLiquidity: 20000,
          maxVolatility: 0.08,
          allowedExchanges: ['binance', 'coinbase'],
          allowedSymbols: ['BTC/USDT', 'ETH/USDT']
        },
        entryConditions: {
          sentimentThreshold: 0.5,
          volumeSpike: 1.0,
          priceDeviation: 0.0005,
          technicalIndicators: ['orderbook_imbalance', 'volume_profile']
        },
        exitConditions: {
          profitTarget: 0.003,
          stopLoss: 0.002,
          maxHoldTime: 180, // 3 minutes
          reversalSignal: true
        }
      }
    ];

    return strategies;
  }

  createDefaultScenarios(): BacktestScenario[] {
    const strategies = this.createDefaultStrategies();
    
    const scenarios: BacktestScenario[] = [
      {
        id: 'bull_market_2021',
        name: 'Bull Market 2021',
        description: 'Test strategy performance during strong bull market conditions',
        strategy: strategies[0], // Conservative strategy
        config: {
          startDate: '2021-01-01',
          endDate: '2021-12-31',
          initialCapital: 100000,
          maxPositionSize: 10000,
          maxConcurrentPositions: 5,
          commission: 10, // 10 basis points
          slippage: 5, // 5 basis points
          riskFreeRate: 0.02,
          timeframe: '1h',
          symbols: ['BTC/USDT', 'ETH/USDT'],
          exchanges: ['binance', 'coinbase']
        },
        marketConditions: {
          volatilityRegime: 'high',
          trendDirection: 'bullish',
          liquidityCondition: 'normal',
          newsEvents: true
        },
        benchmarks: {
          buyAndHold: true,
          marketNeutral: false,
          riskParity: false
        }
      },
      {
        id: 'bear_market_2022',
        name: 'Bear Market 2022',
        description: 'Test strategy resilience during market downturn',
        strategy: strategies[1], // Aggressive strategy
        config: {
          startDate: '2022-01-01',
          endDate: '2022-12-31',
          initialCapital: 100000,
          maxPositionSize: 15000,
          maxConcurrentPositions: 10,
          commission: 8,
          slippage: 6,
          riskFreeRate: 0.03,
          timeframe: '15m',
          symbols: ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
          exchanges: ['binance', 'coinbase', 'kraken']
        },
        marketConditions: {
          volatilityRegime: 'high',
          trendDirection: 'bearish',
          liquidityCondition: 'stressed',
          newsEvents: true
        },
        benchmarks: {
          buyAndHold: true,
          marketNeutral: true,
          riskParity: true
        }
      },
      {
        id: 'sideways_market_2023',
        name: 'Sideways Market 2023',
        description: 'Test strategy in range-bound market conditions',
        strategy: strategies[2], // Market making strategy
        config: {
          startDate: '2023-01-01',
          endDate: '2023-06-30',
          initialCapital: 50000,
          maxPositionSize: 5000,
          maxConcurrentPositions: 15,
          commission: 6,
          slippage: 3,
          riskFreeRate: 0.035,
          timeframe: '5m',
          symbols: ['BTC/USDT', 'ETH/USDT'],
          exchanges: ['binance', 'coinbase']
        },
        marketConditions: {
          volatilityRegime: 'low',
          trendDirection: 'sideways',
          liquidityCondition: 'normal',
          newsEvents: false
        },
        benchmarks: {
          buyAndHold: true,
          marketNeutral: true,
          riskParity: false
        }
      },
      {
        id: 'high_volatility_crisis',
        name: 'High Volatility Crisis',
        description: 'Test strategy during extreme market volatility and liquidity crunch',
        strategy: strategies[0], // Conservative strategy for crisis
        config: {
          startDate: '2020-03-01',
          endDate: '2020-05-31',
          initialCapital: 100000,
          maxPositionSize: 5000,
          maxConcurrentPositions: 3,
          commission: 15,
          slippage: 20,
          riskFreeRate: 0.001,
          timeframe: '1h',
          symbols: ['BTC/USDT', 'ETH/USDT'],
          exchanges: ['binance', 'coinbase']
        },
        marketConditions: {
          volatilityRegime: 'high',
          trendDirection: 'bearish',
          liquidityCondition: 'crisis',
          newsEvents: true
        },
        benchmarks: {
          buyAndHold: true,
          marketNeutral: true,
          riskParity: true
        }
      }
    ];

    return scenarios;
  }

  createDefaultSuites(): BacktestSuite[] {
    const scenarios = this.createDefaultScenarios();
    
    const suites: BacktestSuite[] = [
      {
        id: 'comprehensive_validation',
        name: 'Comprehensive Strategy Validation',
        description: 'Full validation suite covering multiple market conditions',
        scenarios: scenarios,
        validationRules: {
          minSharpeRatio: 0.8,
          maxDrawdown: 0.15,
          minWinRate: 0.55,
          minProfitFactor: 1.2,
          maxVaR: 0.05
        },
        reportingOptions: {
          includeEquityCurve: true,
          includeDrawdownChart: true,
          includeMonthlyPerformance: true,
          includeRiskMetrics: true,
          includeSensitivityAnalysis: true
        }
      },
      {
        id: 'quick_validation',
        name: 'Quick Strategy Validation',
        description: 'Fast validation for strategy development',
        scenarios: [scenarios[2]], // Just sideways market
        validationRules: {
          minSharpeRatio: 0.5,
          maxDrawdown: 0.20,
          minWinRate: 0.50,
          minProfitFactor: 1.0,
          maxVaR: 0.08
        },
        reportingOptions: {
          includeEquityCurve: true,
          includeDrawdownChart: false,
          includeMonthlyPerformance: false,
          includeRiskMetrics: true,
          includeSensitivityAnalysis: false
        }
      },
      {
        id: 'stress_testing',
        name: 'Stress Testing Suite',
        description: 'Test strategy under extreme market conditions',
        scenarios: [scenarios[1], scenarios[3]], // Bear market and crisis
        validationRules: {
          minSharpeRatio: 0.3,
          maxDrawdown: 0.25,
          minWinRate: 0.45,
          minProfitFactor: 0.8,
          maxVaR: 0.10
        },
        reportingOptions: {
          includeEquityCurve: true,
          includeDrawdownChart: true,
          includeMonthlyPerformance: true,
          includeRiskMetrics: true,
          includeSensitivityAnalysis: true
        }
      }
    ];

    return suites;
  }

  // Configuration I/O
  saveStrategy(strategy: BacktestStrategy): void {
    const filePath = join(this.strategiesPath, `${strategy.id}.json`);
    writeFileSync(filePath, JSON.stringify(strategy, null, 2));
  }

  loadStrategy(strategyId: string): BacktestStrategy {
    const filePath = join(this.strategiesPath, `${strategyId}.json`);
    const content = readFileSync(filePath, 'utf-8');
    return JSON.parse(content);
  }

  saveScenario(scenario: BacktestScenario): void {
    const filePath = join(this.scenariosPath, `${scenario.id}.json`);
    writeFileSync(filePath, JSON.stringify(scenario, null, 2));
  }

  loadScenario(scenarioId: string): BacktestScenario {
    const filePath = join(this.scenariosPath, `${scenarioId}.json`);
    const content = readFileSync(filePath, 'utf-8');
    return JSON.parse(content);
  }

  saveSuite(suite: BacktestSuite): void {
    const filePath = join(this.suitesPath, `${suite.id}.json`);
    writeFileSync(filePath, JSON.stringify(suite, null, 2));
  }

  loadSuite(suiteId: string): BacktestSuite {
    const filePath = join(this.suitesPath, `${suiteId}.json`);
    const content = readFileSync(filePath, 'utf-8');
    return JSON.parse(content);
  }

  // Validation and Utilities
  validateStrategy(strategy: BacktestStrategy): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Validate parameters
    if (strategy.parameters.minConfidence < 0 || strategy.parameters.minConfidence > 1) {
      errors.push('minConfidence must be between 0 and 1');
    }
    if (strategy.parameters.maxRisk <= 0 || strategy.parameters.maxRisk > 1) {
      errors.push('maxRisk must be between 0 and 1');
    }
    if (strategy.parameters.maxPositionSize <= 0 || strategy.parameters.maxPositionSize > 1) {
      errors.push('maxPositionSize must be between 0 and 1');
    }
    if (strategy.parameters.timeHorizon <= 0) {
      errors.push('timeHorizon must be positive');
    }
    if (strategy.parameters.riskRewardRatio <= 0) {
      errors.push('riskRewardRatio must be positive');
    }

    // Validate filters
    if (strategy.filters.minSpread < 0) {
      errors.push('minSpread must be non-negative');
    }
    if (strategy.filters.minLiquidity <= 0) {
      errors.push('minLiquidity must be positive');
    }
    if (strategy.filters.maxVolatility <= 0) {
      errors.push('maxVolatility must be positive');
    }

    // Validate exit conditions
    if (strategy.exitConditions.profitTarget <= 0) {
      errors.push('profitTarget must be positive');
    }
    if (strategy.exitConditions.stopLoss <= 0) {
      errors.push('stopLoss must be positive');
    }
    if (strategy.exitConditions.profitTarget <= strategy.exitConditions.stopLoss) {
      errors.push('profitTarget should be greater than stopLoss');
    }

    return { valid: errors.length === 0, errors };
  }

  validateScenario(scenario: BacktestScenario): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Validate dates
    const startDate = new Date(scenario.config.startDate);
    const endDate = new Date(scenario.config.endDate);
    
    if (isNaN(startDate.getTime())) {
      errors.push('Invalid startDate format');
    }
    if (isNaN(endDate.getTime())) {
      errors.push('Invalid endDate format');
    }
    if (startDate >= endDate) {
      errors.push('startDate must be before endDate');
    }

    // Validate config
    if (scenario.config.initialCapital <= 0) {
      errors.push('initialCapital must be positive');
    }
    if (scenario.config.maxPositionSize <= 0) {
      errors.push('maxPositionSize must be positive');
    }
    if (scenario.config.maxConcurrentPositions <= 0) {
      errors.push('maxConcurrentPositions must be positive');
    }
    if (scenario.config.commission < 0) {
      errors.push('commission must be non-negative');
    }
    if (scenario.config.slippage < 0) {
      errors.push('slippage must be non-negative');
    }

    // Validate arrays
    if (scenario.config.symbols.length === 0) {
      errors.push('symbols array cannot be empty');
    }
    if (scenario.config.exchanges.length === 0) {
      errors.push('exchanges array cannot be empty');
    }

    // Validate strategy
    const strategyValidation = this.validateStrategy(scenario.strategy);
    if (!strategyValidation.valid) {
      errors.push(...strategyValidation.errors.map(e => `Strategy: ${e}`));
    }

    return { valid: errors.length === 0, errors };
  }

  // Parameter Optimization
  generateParameterSweep(
    baseStrategy: BacktestStrategy, 
    parameterRanges: { [key: string]: { min: number; max: number; step: number } }
  ): BacktestStrategy[] {
    const strategies: BacktestStrategy[] = [];
    const parameterNames = Object.keys(parameterRanges);
    
    const generateCombinations = (index: number, currentParams: any): void => {
      if (index === parameterNames.length) {
        // Create new strategy with current parameter combination
        const newStrategy: BacktestStrategy = JSON.parse(JSON.stringify(baseStrategy));
        
        // Apply parameter changes
        for (const [paramName, value] of Object.entries(currentParams)) {
          this.setNestedProperty(newStrategy, paramName, value);
        }
        
        newStrategy.id = `${baseStrategy.id}_${Object.values(currentParams).join('_')}`;
        newStrategy.name = `${baseStrategy.name} (${Object.entries(currentParams)
          .map(([k, v]) => `${k}=${v}`).join(', ')})`;
        
        strategies.push(newStrategy);
        return;
      }
      
      const paramName = parameterNames[index];
      const range = parameterRanges[paramName];
      
      for (let value = range.min; value <= range.max; value += range.step) {
        generateCombinations(index + 1, { ...currentParams, [paramName]: value });
      }
    };
    
    generateCombinations(0, {});
    return strategies;
  }

  private setNestedProperty(obj: any, path: string, value: any): void {
    const keys = path.split('.');
    let current = obj;
    
    for (let i = 0; i < keys.length - 1; i++) {
      if (!(keys[i] in current)) {
        current[keys[i]] = {};
      }
      current = current[keys[i]];
    }
    
    current[keys[keys.length - 1]] = value;
  }

  // Market Condition Analysis
  analyzeMarketConditions(scenario: BacktestScenario): {
    expectedPerformance: string;
    riskFactors: string[];
    recommendations: string[];
  } {
    const conditions = scenario.marketConditions;
    const strategy = scenario.strategy;
    
    let expectedPerformance = 'moderate';
    const riskFactors: string[] = [];
    const recommendations: string[] = [];
    
    // Analyze volatility regime
    if (conditions.volatilityRegime === 'high') {
      if (strategy.filters.maxVolatility < 0.1) {
        expectedPerformance = 'poor';
        riskFactors.push('Strategy not designed for high volatility');
        recommendations.push('Increase maxVolatility threshold or adjust position sizing');
      } else {
        expectedPerformance = 'good';
      }
    }
    
    // Analyze trend direction
    if (conditions.trendDirection === 'bearish') {
      if (strategy.parameters.maxRisk > 0.03) {
        riskFactors.push('High risk exposure during bearish conditions');
        recommendations.push('Reduce maxRisk parameter during bear markets');
      }
    }
    
    // Analyze liquidity conditions
    if (conditions.liquidityCondition === 'crisis') {
      if (strategy.filters.minLiquidity < 20000) {
        riskFactors.push('Low liquidity threshold during crisis conditions');
        recommendations.push('Increase minLiquidity requirement significantly');
      }
      if (strategy.parameters.maxConcurrentTrades > 5) {
        riskFactors.push('Too many concurrent trades during liquidity crisis');
        recommendations.push('Reduce maxConcurrentTrades during crisis');
      }
    }
    
    // Analyze news impact
    if (conditions.newsEvents) {
      if (strategy.exitConditions.maxHoldTime > 600) {
        riskFactors.push('Long hold times during news-heavy periods');
        recommendations.push('Reduce maxHoldTime when news events are frequent');
      }
    }
    
    return { expectedPerformance, riskFactors, recommendations };
  }

  // Export configurations for external tools
  exportToYAML(config: BacktestStrategy | BacktestScenario | BacktestSuite): string {
    // Convert to YAML format (simplified implementation)
    return JSON.stringify(config, null, 2);
  }

  // Import from external formats
  importFromJSON(jsonString: string): BacktestStrategy | BacktestScenario | BacktestSuite {
    return JSON.parse(jsonString);
  }
}