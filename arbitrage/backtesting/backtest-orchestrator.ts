import { EventEmitter } from 'events';
import { BacktestingEngine, BacktestResult, BacktestConfig } from './backtesting-engine';
import { LiveValidator, WalkForwardResult, ComparisonResult } from './live-validator';
import { BacktestConfigManager, BacktestStrategy, BacktestScenario, BacktestSuite } from './backtest-config';
import { KafkaManager, RedisManager, DatabaseManager } from '../core/infrastructure';
import { FusionBrain } from '../core/fusion/fusion-brain';
import { DecisionEngine } from '../decision/decision-engine';

export interface BacktestJob {
  id: string;
  type: 'single' | 'suite' | 'parameter_sweep' | 'walk_forward';
  status: 'queued' | 'running' | 'completed' | 'failed';
  config: BacktestConfig | BacktestSuite;
  priority: 'low' | 'medium' | 'high';
  submittedAt: number;
  startedAt?: number;
  completedAt?: number;
  results?: BacktestResult | BacktestResult[];
  error?: string;
  progress: number;
}

export interface BacktestReport {
  id: string;
  jobId: string;
  type: 'performance' | 'risk' | 'comparison' | 'walk_forward';
  summary: {
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    totalTrades: number;
    profitFactor: number;
  };
  detailed: {
    equityCurve: { timestamp: number; equity: number; drawdown: number }[];
    monthlyReturns: { month: string; return: number }[];
    riskMetrics: any;
    tradeAnalysis: any;
    performanceBySymbol: Map<string, any>;
    performanceByExchange: Map<string, any>;
  };
  charts: {
    equityCurveChart: string; // Base64 encoded chart
    drawdownChart: string;
    monthlyPerformanceChart: string;
    riskMetricsChart: string;
  };
  recommendations: string[];
  timestamp: number;
}

export interface InvestorDashboardData {
  livePerformance: {
    currentPnL: number;
    todayPnL: number;
    weekPnL: number;
    monthPnL: number;
    yearPnL: number;
    currentDrawdown: number;
    sharpeRatio: number;
    winRate: number;
    activeTrades: number;
  };
  backtestValidation: {
    latestBacktest: BacktestResult;
    liveVsBacktest: ComparisonResult;
    walkForwardResults: WalkForwardResult[];
    predictionAccuracy: number;
    modelReliability: number;
  };
  riskMetrics: {
    portfolioVaR: number;
    expectedShortfall: number;
    leverageRatio: number;
    concentrationRisk: number;
    liquidityRisk: number;
  };
  systemHealth: {
    agentStatus: { [agentName: string]: 'online' | 'offline' | 'error' };
    fusionBrainHealth: number;
    executionLatency: number;
    dataFreshnessScore: number;
    overallHealth: number;
  };
  opportunities: {
    active: any[];
    pipeline: any[];
    recentExecutions: any[];
  };
  alerts: {
    level: 'info' | 'warning' | 'critical';
    message: string;
    timestamp: number;
  }[];
}

export class BacktestOrchestrator extends EventEmitter {
  private backtestEngine: BacktestingEngine;
  private liveValidator: LiveValidator;
  private configManager: BacktestConfigManager;
  private kafka: KafkaManager;
  private redis: RedisManager;
  private db: DatabaseManager;
  private fusionBrain: FusionBrain;
  private decisionEngine: DecisionEngine;
  
  private jobQueue: BacktestJob[] = [];
  private activeJobs: Map<string, BacktestJob> = new Map();
  private isProcessing: boolean = false;
  private maxConcurrentJobs: number = 3;
  
  constructor() {
    super();
    this.backtestEngine = new BacktestingEngine();
    this.liveValidator = new LiveValidator();
    this.configManager = new BacktestConfigManager();
    this.kafka = new KafkaManager();
    this.redis = new RedisManager();
    this.db = new DatabaseManager();
    this.fusionBrain = new FusionBrain();
    this.decisionEngine = new DecisionEngine();
    
    this.setupEventListeners();
  }

  private setupEventListeners(): void {
    // Listen for new predictions to trigger validation backtests
    this.kafka.subscribe('fusion-predictions', (prediction) => {
      this.triggerValidationBacktest(prediction);
    });

    // Listen for execution results to compare with backtests
    this.kafka.subscribe('execution-results', (execution) => {
      this.updateLiveComparison(execution);
    });

    // Backtest engine events
    this.backtestEngine.on('backtest-progress', (progress) => {
      this.emit('job-progress', progress);
    });

    this.backtestEngine.on('backtest-complete', (result) => {
      this.handleBacktestComplete(result);
    });

    // Live validator events
    this.liveValidator.on('validation-completed', (validation) => {
      this.emit('validation-completed', validation);
    });

    this.liveValidator.on('walk-forward-complete', (result) => {
      this.handleWalkForwardComplete(result);
    });
  }

  async initialize(): Promise<void> {
    await this.backtestEngine.initialize();
    await this.liveValidator.initialize();
    await this.kafka.connect();
    await this.redis.connect();
    await this.db.connect();
    await this.fusionBrain.initialize();
    await this.decisionEngine.initialize();
    
    await this.createOrchestratorTables();
    
    // Initialize default configurations
    await this.initializeDefaultConfigurations();
    
    // Start job processing
    this.startJobProcessor();
    
    console.log('BacktestOrchestrator initialized');
  }

  private async createOrchestratorTables(): Promise<void> {
    const queries = [
      `CREATE TABLE IF NOT EXISTS backtest_jobs (
        id TEXT PRIMARY KEY,
        type TEXT NOT NULL,
        status TEXT NOT NULL,
        config TEXT NOT NULL,
        priority TEXT NOT NULL,
        submitted_at INTEGER NOT NULL,
        started_at INTEGER,
        completed_at INTEGER,
        results TEXT,
        error TEXT,
        progress REAL DEFAULT 0
      )`,
      `CREATE TABLE IF NOT EXISTS backtest_reports (
        id TEXT PRIMARY KEY,
        job_id TEXT NOT NULL,
        type TEXT NOT NULL,
        summary TEXT NOT NULL,
        detailed TEXT NOT NULL,
        charts TEXT,
        recommendations TEXT,
        timestamp INTEGER NOT NULL,
        FOREIGN KEY (job_id) REFERENCES backtest_jobs (id)
      )`,
      `CREATE TABLE IF NOT EXISTS investor_dashboard_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp INTEGER NOT NULL,
        data TEXT NOT NULL
      )`
    ];

    for (const query of queries) {
      await this.db.run(query);
    }
  }

  private async initializeDefaultConfigurations(): Promise<void> {
    const strategies = this.configManager.createDefaultStrategies();
    const scenarios = this.configManager.createDefaultScenarios();
    const suites = this.configManager.createDefaultSuites();
    
    // Store in Redis for quick access
    await this.redis.set('default_strategies', JSON.stringify(strategies));
    await this.redis.set('default_scenarios', JSON.stringify(scenarios));
    await this.redis.set('default_suites', JSON.stringify(suites));
    
    console.log(`Initialized ${strategies.length} strategies, ${scenarios.length} scenarios, ${suites.length} suites`);
  }

  // Job Management
  async submitBacktest(config: BacktestConfig, priority: 'low' | 'medium' | 'high' = 'medium'): Promise<string> {
    const jobId = `backtest_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const job: BacktestJob = {
      id: jobId,
      type: 'single',
      status: 'queued',
      config,
      priority,
      submittedAt: Date.now(),
      progress: 0
    };
    
    this.jobQueue.push(job);
    this.sortJobQueue();
    
    // Store in database
    await this.db.run(
      `INSERT INTO backtest_jobs 
       (id, type, status, config, priority, submitted_at, progress) 
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [job.id, job.type, job.status, JSON.stringify(job.config), 
       job.priority, job.submittedAt, job.progress]
    );
    
    this.emit('job-submitted', job);
    console.log(`Submitted backtest job ${jobId}`);
    
    return jobId;
  }

  async submitSuite(suite: BacktestSuite, priority: 'low' | 'medium' | 'high' = 'medium'): Promise<string> {
    const jobId = `suite_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const job: BacktestJob = {
      id: jobId,
      type: 'suite',
      status: 'queued',
      config: suite,
      priority,
      submittedAt: Date.now(),
      progress: 0
    };
    
    this.jobQueue.push(job);
    this.sortJobQueue();
    
    await this.db.run(
      `INSERT INTO backtest_jobs 
       (id, type, status, config, priority, submitted_at, progress) 
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [job.id, job.type, job.status, JSON.stringify(job.config), 
       job.priority, job.submittedAt, job.progress]
    );
    
    this.emit('job-submitted', job);
    console.log(`Submitted suite job ${jobId}`);
    
    return jobId;
  }

  async submitParameterSweep(
    baseStrategy: BacktestStrategy,
    parameterRanges: any,
    baseConfig: BacktestConfig,
    priority: 'low' | 'medium' | 'high' = 'low'
  ): Promise<string> {
    const jobId = `sweep_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    const job: BacktestJob = {
      id: jobId,
      type: 'parameter_sweep',
      status: 'queued',
      config: { baseStrategy, parameterRanges, baseConfig },
      priority,
      submittedAt: Date.now(),
      progress: 0
    };
    
    this.jobQueue.push(job);
    this.sortJobQueue();
    
    await this.db.run(
      `INSERT INTO backtest_jobs 
       (id, type, status, config, priority, submitted_at, progress) 
       VALUES (?, ?, ?, ?, ?, ?, ?)`,
      [job.id, job.type, job.status, JSON.stringify(job.config), 
       job.priority, job.submittedAt, job.progress]
    );
    
    this.emit('job-submitted', job);
    console.log(`Submitted parameter sweep job ${jobId}`);
    
    return jobId;
  }

  private sortJobQueue(): void {
    const priorityOrder = { 'high': 3, 'medium': 2, 'low': 1 };
    
    this.jobQueue.sort((a, b) => {
      const priorityDiff = priorityOrder[b.priority] - priorityOrder[a.priority];
      if (priorityDiff !== 0) return priorityDiff;
      return a.submittedAt - b.submittedAt; // FIFO within same priority
    });
  }

  private startJobProcessor(): void {
    setInterval(() => {
      this.processJobQueue();
    }, 5000); // Check every 5 seconds
  }

  private async processJobQueue(): Promise<void> {
    if (this.isProcessing || this.activeJobs.size >= this.maxConcurrentJobs || this.jobQueue.length === 0) {
      return;
    }
    
    this.isProcessing = true;
    
    try {
      const job = this.jobQueue.shift()!;
      await this.executeJob(job);
    } catch (error) {
      console.error('Error processing job queue:', error);
    } finally {
      this.isProcessing = false;
    }
  }

  private async executeJob(job: BacktestJob): Promise<void> {
    console.log(`Starting execution of job ${job.id} (${job.type})`);
    
    job.status = 'running';
    job.startedAt = Date.now();
    this.activeJobs.set(job.id, job);
    
    // Update database
    await this.db.run(
      `UPDATE backtest_jobs SET status = ?, started_at = ? WHERE id = ?`,
      [job.status, job.startedAt, job.id]
    );
    
    try {
      switch (job.type) {
        case 'single':
          job.results = await this.backtestEngine.runBacktest(job.config as BacktestConfig);
          break;
        case 'suite':
          job.results = await this.executeSuite(job.config as BacktestSuite);
          break;
        case 'parameter_sweep':
          job.results = await this.executeParameterSweep(job.config as any);
          break;
        case 'walk_forward':
          await this.liveValidator.runWalkForwardAnalysis();
          break;
      }
      
      job.status = 'completed';
      job.completedAt = Date.now();
      job.progress = 100;
      
    } catch (error) {
      job.status = 'failed';
      job.error = error instanceof Error ? error.message : String(error);
      job.completedAt = Date.now();
      
      console.error(`Job ${job.id} failed:`, error);
    }
    
    // Update database
    await this.db.run(
      `UPDATE backtest_jobs 
       SET status = ?, completed_at = ?, results = ?, error = ?, progress = ? 
       WHERE id = ?`,
      [job.status, job.completedAt, JSON.stringify(job.results || null), 
       job.error || null, job.progress, job.id]
    );
    
    this.activeJobs.delete(job.id);
    this.emit('job-completed', job);
    
    console.log(`Job ${job.id} completed with status ${job.status}`);
  }

  private async executeSuite(suite: BacktestSuite): Promise<BacktestResult[]> {
    const results: BacktestResult[] = [];
    
    for (let i = 0; i < suite.scenarios.length; i++) {
      const scenario = suite.scenarios[i];
      
      // Convert scenario to backtest config
      const config: BacktestConfig = {
        startDate: new Date(scenario.config.startDate).getTime(),
        endDate: new Date(scenario.config.endDate).getTime(),
        initialCapital: scenario.config.initialCapital,
        maxPositionSize: scenario.config.maxPositionSize,
        maxConcurrentPositions: scenario.config.maxConcurrentPositions,
        commission: scenario.config.commission,
        slippage: scenario.config.slippage,
        riskFreeRate: scenario.config.riskFreeRate,
        timeframe: scenario.config.timeframe,
        symbols: scenario.config.symbols,
        exchanges: scenario.config.exchanges
      };
      
      const result = await this.backtestEngine.runBacktest(config);
      results.push(result);
      
      // Update progress
      const progress = ((i + 1) / suite.scenarios.length) * 100;
      this.emit('suite-progress', { suiteId: suite.id, progress, scenario: scenario.name });
    }
    
    return results;
  }

  private async executeParameterSweep(config: any): Promise<BacktestResult[]> {
    const { baseStrategy, parameterRanges, baseConfig } = config;
    
    const strategies = this.configManager.generateParameterSweep(baseStrategy, parameterRanges);
    const results: BacktestResult[] = [];
    
    for (let i = 0; i < strategies.length; i++) {
      const strategy = strategies[i];
      
      // Apply strategy parameters to config (simplified)
      const sweepConfig = { ...baseConfig };
      
      const result = await this.backtestEngine.runBacktest(sweepConfig);
      results.push(result);
      
      // Update progress
      const progress = ((i + 1) / strategies.length) * 100;
      this.emit('sweep-progress', { progress, strategy: strategy.name });
    }
    
    return results;
  }

  // Automatic validation backtests triggered by live predictions
  private async triggerValidationBacktest(prediction: any): Promise<void> {
    const config: BacktestConfig = {
      startDate: Date.now() - 7 * 24 * 60 * 60 * 1000, // Last 7 days
      endDate: Date.now(),
      initialCapital: 10000,
      maxPositionSize: 1000,
      maxConcurrentPositions: 3,
      commission: 10,
      slippage: 5,
      riskFreeRate: 0.02,
      timeframe: '1h',
      symbols: [prediction.opportunities[0]?.symbol || 'BTC/USDT'],
      exchanges: ['binance', 'coinbase']
    };
    
    await this.submitBacktest(config, 'high');
    console.log(`Triggered validation backtest for prediction ${prediction.id}`);
  }

  private handleBacktestComplete(result: BacktestResult): void {
    // Generate comprehensive report
    this.generateBacktestReport(result);
    
    // Update live comparison data
    this.updateLiveComparison(result);
    
    // Cache key metrics for dashboard
    this.cacheBacktestMetrics(result);
  }

  private async generateBacktestReport(result: BacktestResult): Promise<void> {
    const report: BacktestReport = {
      id: `report_${result.id}`,
      jobId: result.id,
      type: 'performance',
      summary: {
        totalReturn: (result.equityCurve[result.equityCurve.length - 1]?.equity - result.config.initialCapital) / result.config.initialCapital,
        sharpeRatio: result.metrics.sharpeRatio,
        maxDrawdown: result.metrics.maxDrawdown,
        winRate: result.metrics.winRate,
        totalTrades: result.metrics.totalTrades,
        profitFactor: result.metrics.profitFactor
      },
      detailed: {
        equityCurve: result.equityCurve,
        monthlyReturns: this.calculateMonthlyReturns(result.equityCurve),
        riskMetrics: this.calculateRiskMetrics(result),
        tradeAnalysis: this.analyzeTradePatterns(result.positions),
        performanceBySymbol: result.performanceBySymbol,
        performanceByExchange: result.performanceByExchange
      },
      charts: {
        equityCurveChart: this.generateEquityCurveChart(result.equityCurve),
        drawdownChart: this.generateDrawdownChart(result.equityCurve),
        monthlyPerformanceChart: this.generateMonthlyChart(result.equityCurve),
        riskMetricsChart: this.generateRiskChart(result.metrics)
      },
      recommendations: this.generateRecommendations(result),
      timestamp: Date.now()
    };
    
    // Store report
    await this.db.run(
      `INSERT INTO backtest_reports 
       (id, job_id, type, summary, detailed, charts, recommendations, timestamp)
       VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
      [
        report.id, report.jobId, report.type, JSON.stringify(report.summary),
        JSON.stringify(report.detailed), JSON.stringify(report.charts),
        JSON.stringify(report.recommendations), report.timestamp
      ]
    );
    
    this.emit('report-generated', report);
  }

  private calculateMonthlyReturns(equityCurve: any[]): { month: string; return: number }[] {
    const monthlyReturns: { month: string; return: number }[] = [];
    const monthlyData: { [month: string]: { start: number; end: number } } = {};
    
    for (const point of equityCurve) {
      const date = new Date(point.timestamp);
      const monthKey = `${date.getFullYear()}-${String(date.getMonth() + 1).padStart(2, '0')}`;
      
      if (!monthlyData[monthKey]) {
        monthlyData[monthKey] = { start: point.equity, end: point.equity };
      } else {
        monthlyData[monthKey].end = point.equity;
      }
    }
    
    for (const [month, data] of Object.entries(monthlyData)) {
      const returnPct = (data.end - data.start) / data.start;
      monthlyReturns.push({ month, return: returnPct });
    }
    
    return monthlyReturns.sort((a, b) => a.month.localeCompare(b.month));
  }

  private calculateRiskMetrics(result: BacktestResult): any {
    return {
      valueAtRisk95: this.calculateVaR(result.equityCurve, 0.95),
      expectedShortfall: this.calculateExpectedShortfall(result.equityCurve, 0.95),
      skewness: this.calculateSkewness(result.equityCurve),
      kurtosis: this.calculateKurtosis(result.equityCurve),
      tailRatio: this.calculateTailRatio(result.positions)
    };
  }

  private calculateVaR(equityCurve: any[], confidence: number): number {
    const returns = equityCurve.slice(1).map((point, i) => 
      (point.equity - equityCurve[i].equity) / equityCurve[i].equity
    );
    
    returns.sort((a, b) => a - b);
    const index = Math.floor((1 - confidence) * returns.length);
    return returns[index] || 0;
  }

  private calculateExpectedShortfall(equityCurve: any[], confidence: number): number {
    const returns = equityCurve.slice(1).map((point, i) => 
      (point.equity - equityCurve[i].equity) / equityCurve[i].equity
    );
    
    returns.sort((a, b) => a - b);
    const varIndex = Math.floor((1 - confidence) * returns.length);
    const tailReturns = returns.slice(0, varIndex);
    
    return tailReturns.length > 0 ? 
      tailReturns.reduce((a, b) => a + b, 0) / tailReturns.length : 0;
  }

  private calculateSkewness(equityCurve: any[]): number {
    const returns = equityCurve.slice(1).map((point, i) => 
      (point.equity - equityCurve[i].equity) / equityCurve[i].equity
    );
    
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);
    
    if (stdDev === 0) return 0;
    
    const skewness = returns.reduce((sum, r) => sum + Math.pow((r - mean) / stdDev, 3), 0) / returns.length;
    return skewness;
  }

  private calculateKurtosis(equityCurve: any[]): number {
    const returns = equityCurve.slice(1).map((point, i) => 
      (point.equity - equityCurve[i].equity) / equityCurve[i].equity
    );
    
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);
    
    if (stdDev === 0) return 0;
    
    const kurtosis = returns.reduce((sum, r) => sum + Math.pow((r - mean) / stdDev, 4), 0) / returns.length;
    return kurtosis - 3; // Excess kurtosis
  }

  private calculateTailRatio(positions: any[]): number {
    const closedPositions = positions.filter(p => p.status === 'closed' && p.pnl !== undefined);
    if (closedPositions.length === 0) return 0;
    
    const profits = closedPositions.filter(p => p.pnl > 0).map(p => p.pnl);
    const losses = closedPositions.filter(p => p.pnl < 0).map(p => Math.abs(p.pnl));
    
    if (profits.length === 0 || losses.length === 0) return 0;
    
    profits.sort((a, b) => b - a);
    losses.sort((a, b) => b - a);
    
    const top10PercentProfits = profits.slice(0, Math.max(1, Math.floor(profits.length * 0.1)));
    const top10PercentLosses = losses.slice(0, Math.max(1, Math.floor(losses.length * 0.1)));
    
    const avgTopProfits = top10PercentProfits.reduce((a, b) => a + b, 0) / top10PercentProfits.length;
    const avgTopLosses = top10PercentLosses.reduce((a, b) => a + b, 0) / top10PercentLosses.length;
    
    return avgTopLosses > 0 ? avgTopProfits / avgTopLosses : 0;
  }

  private analyzeTradePatterns(positions: any[]): any {
    const closedPositions = positions.filter(p => p.status === 'closed');
    
    return {
      averageHoldTime: this.calculateAverageHoldTime(closedPositions),
      profitByTimeOfDay: this.analyzeProfitByTimeOfDay(closedPositions),
      profitByDayOfWeek: this.analyzeProfitByDayOfWeek(closedPositions),
      winStreakAnalysis: this.analyzeWinStreaks(closedPositions),
      positionSizeAnalysis: this.analyzePositionSizes(closedPositions)
    };
  }

  private calculateAverageHoldTime(positions: any[]): number {
    if (positions.length === 0) return 0;
    
    const holdTimes = positions
      .filter(p => p.exitTime)
      .map(p => p.exitTime - p.entryTime);
    
    return holdTimes.reduce((a, b) => a + b, 0) / holdTimes.length;
  }

  private analyzeProfitByTimeOfDay(positions: any[]): any {
    const hourlyProfits: { [hour: number]: number[] } = {};
    
    for (const position of positions) {
      if (position.pnl === undefined) continue;
      
      const hour = new Date(position.entryTime).getHours();
      if (!hourlyProfits[hour]) hourlyProfits[hour] = [];
      hourlyProfits[hour].push(position.pnl);
    }
    
    const result: { [hour: number]: number } = {};
    for (const [hour, profits] of Object.entries(hourlyProfits)) {
      result[parseInt(hour)] = profits.reduce((a, b) => a + b, 0) / profits.length;
    }
    
    return result;
  }

  private analyzeProfitByDayOfWeek(positions: any[]): any {
    const dailyProfits: { [day: number]: number[] } = {};
    
    for (const position of positions) {
      if (position.pnl === undefined) continue;
      
      const day = new Date(position.entryTime).getDay();
      if (!dailyProfits[day]) dailyProfits[day] = [];
      dailyProfits[day].push(position.pnl);
    }
    
    const result: { [day: number]: number } = {};
    for (const [day, profits] of Object.entries(dailyProfits)) {
      result[parseInt(day)] = profits.reduce((a, b) => a + b, 0) / profits.length;
    }
    
    return result;
  }

  private analyzeWinStreaks(positions: any[]): any {
    let currentStreak = 0;
    let maxWinStreak = 0;
    let maxLossStreak = 0;
    let currentLossStreak = 0;
    
    for (const position of positions) {
      if (position.pnl === undefined) continue;
      
      if (position.pnl > 0) {
        currentStreak++;
        currentLossStreak = 0;
        maxWinStreak = Math.max(maxWinStreak, currentStreak);
      } else {
        currentLossStreak++;
        currentStreak = 0;
        maxLossStreak = Math.max(maxLossStreak, currentLossStreak);
      }
    }
    
    return { maxWinStreak, maxLossStreak };
  }

  private analyzePositionSizes(positions: any[]): any {
    const sizes = positions.map(p => Math.abs(p.quantity));
    const profits = positions.filter(p => p.pnl !== undefined).map(p => p.pnl);
    
    return {
      averageSize: sizes.reduce((a, b) => a + b, 0) / sizes.length,
      maxSize: Math.max(...sizes),
      minSize: Math.min(...sizes),
      sizeStdDev: this.calculateStandardDeviation(sizes),
      correlationSizeProfit: this.calculateCorrelation(sizes, profits)
    };
  }

  private calculateStandardDeviation(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const variance = values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) / values.length;
    return Math.sqrt(variance);
  }

  private calculateCorrelation(x: number[], y: number[]): number {
    if (x.length !== y.length || x.length === 0) return 0;
    
    const meanX = x.reduce((a, b) => a + b, 0) / x.length;
    const meanY = y.reduce((a, b) => a + b, 0) / y.length;
    
    let numerator = 0;
    let sumXSquared = 0;
    let sumYSquared = 0;
    
    for (let i = 0; i < x.length; i++) {
      const diffX = x[i] - meanX;
      const diffY = y[i] - meanY;
      numerator += diffX * diffY;
      sumXSquared += diffX * diffX;
      sumYSquared += diffY * diffY;
    }
    
    const denominator = Math.sqrt(sumXSquared * sumYSquared);
    return denominator === 0 ? 0 : numerator / denominator;
  }

  // Chart generation (simplified - would use actual charting library in production)
  private generateEquityCurveChart(equityCurve: any[]): string {
    // Return base64 encoded chart data (placeholder)
    return Buffer.from(`Equity Curve Chart Data: ${equityCurve.length} points`).toString('base64');
  }

  private generateDrawdownChart(equityCurve: any[]): string {
    return Buffer.from(`Drawdown Chart Data: ${equityCurve.length} points`).toString('base64');
  }

  private generateMonthlyChart(equityCurve: any[]): string {
    return Buffer.from(`Monthly Performance Chart Data: ${equityCurve.length} points`).toString('base64');
  }

  private generateRiskChart(metrics: any): string {
    return Buffer.from(`Risk Metrics Chart Data: ${JSON.stringify(metrics)}`).toString('base64');
  }

  private generateRecommendations(result: BacktestResult): string[] {
    const recommendations: string[] = [];
    const metrics = result.metrics;
    
    if (metrics.sharpeRatio < 0.5) {
      recommendations.push('Consider reducing risk exposure - Sharpe ratio is below acceptable threshold');
    }
    
    if (metrics.maxDrawdown > 0.2) {
      recommendations.push('Implement stricter position sizing - Maximum drawdown exceeds 20%');
    }
    
    if (metrics.winRate < 0.4) {
      recommendations.push('Review entry criteria - Win rate is below 40%');
    }
    
    if (metrics.profitFactor < 1.2) {
      recommendations.push('Optimize exit strategy - Profit factor suggests poor risk/reward ratio');
    }
    
    if (metrics.totalTrades < 50) {
      recommendations.push('Extend backtest period - Sample size may be insufficient for statistical significance');
    }
    
    return recommendations;
  }

  private async cacheBacktestMetrics(result: BacktestResult): Promise<void> {
    const cacheData = {
      id: result.id,
      timestamp: result.endTime,
      sharpeRatio: result.metrics.sharpeRatio,
      maxDrawdown: result.metrics.maxDrawdown,
      winRate: result.metrics.winRate,
      totalReturn: result.metrics.netPnL / result.config.initialCapital,
      totalTrades: result.metrics.totalTrades
    };
    
    await this.redis.set(`backtest_metrics:${result.id}`, JSON.stringify(cacheData), 3600);
  }

  private updateLiveComparison(data: any): void {
    // Update live vs backtest comparison data
    this.emit('live-comparison-updated', data);
  }

  private handleWalkForwardComplete(result: WalkForwardResult): void {
    this.emit('walk-forward-completed', result);
    console.log(`Walk-forward analysis completed with degradation: ${result.degradation}`);
  }

  // Dashboard Data Generation
  async generateInvestorDashboard(): Promise<InvestorDashboardData> {
    // Get live performance data
    const liveMetrics = this.liveValidator.getLiveMetrics();
    
    // Get latest backtest results
    const backtestResults = await this.backtestEngine.getBacktestResults(1);
    const latestBacktest = backtestResults[0];
    
    // Get walk-forward results
    const walkForwardResults = await this.liveValidator.getWalkForwardResults(5);
    
    // Get system health metrics
    const systemHealth = await this.getSystemHealth();
    
    // Generate dashboard data
    const dashboardData: InvestorDashboardData = {
      livePerformance: {
        currentPnL: await this.calculateCurrentPnL(),
        todayPnL: await this.calculatePeriodPnL(24 * 60 * 60 * 1000),
        weekPnL: await this.calculatePeriodPnL(7 * 24 * 60 * 60 * 1000),
        monthPnL: await this.calculatePeriodPnL(30 * 24 * 60 * 60 * 1000),
        yearPnL: await this.calculatePeriodPnL(365 * 24 * 60 * 60 * 1000),
        currentDrawdown: await this.calculateCurrentDrawdown(),
        sharpeRatio: liveMetrics.averageAccuracy * 2, // Simplified conversion
        winRate: liveMetrics.averageAccuracy,
        activeTrades: this.liveValidator.getActivePredictions().length
      },
      backtestValidation: {
        latestBacktest: latestBacktest,
        liveVsBacktest: await this.getLatestBacktestComparison(),
        walkForwardResults: walkForwardResults,
        predictionAccuracy: liveMetrics.averageAccuracy,
        modelReliability: liveMetrics.confidenceCalibration
      },
      riskMetrics: {
        portfolioVaR: await this.calculatePortfolioVaR(),
        expectedShortfall: await this.calculatePortfolioES(),
        leverageRatio: 1.0, // Simplified
        concentrationRisk: 0.2, // Simplified
        liquidityRisk: 0.1 // Simplified
      },
      systemHealth: systemHealth,
      opportunities: {
        active: await this.getActiveOpportunities(),
        pipeline: await this.getPipelineOpportunities(),
        recentExecutions: await this.getRecentExecutions()
      },
      alerts: await this.getSystemAlerts()
    };
    
    // Cache dashboard data
    await this.redis.set('investor_dashboard', JSON.stringify(dashboardData), 30); // 30 second TTL
    
    // Store snapshot in database
    await this.db.run(
      `INSERT INTO investor_dashboard_snapshots (timestamp, data) VALUES (?, ?)`,
      [Date.now(), JSON.stringify(dashboardData)]
    );
    
    return dashboardData;
  }

  private async calculateCurrentPnL(): Promise<number> {
    // Calculate current P&L from live positions
    return 1250.75; // Placeholder
  }

  private async calculatePeriodPnL(periodMs: number): Promise<number> {
    // Calculate P&L for specific time period
    return Math.random() * 1000 - 500; // Placeholder
  }

  private async calculateCurrentDrawdown(): Promise<number> {
    // Calculate current drawdown from peak equity
    return 0.03; // Placeholder
  }

  private async getLatestBacktestComparison(): Promise<ComparisonResult> {
    const comparisons = await this.liveValidator.getBacktestComparisons(1);
    return comparisons[0] || {} as ComparisonResult;
  }

  private async calculatePortfolioVaR(): Promise<number> {
    return 0.02; // Placeholder - 2% daily VaR
  }

  private async calculatePortfolioES(): Promise<number> {
    return 0.035; // Placeholder - 3.5% expected shortfall
  }

  private async getSystemHealth(): Promise<any> {
    return {
      agentStatus: {
        economic: 'online',
        sentiment: 'online',
        price: 'online',
        volume: 'online',
        trade: 'online',
        image: 'online'
      },
      fusionBrainHealth: 0.95,
      executionLatency: 150, // ms
      dataFreshnessScore: 0.98,
      overallHealth: 0.94
    };
  }

  private async getActiveOpportunities(): Promise<any[]> {
    return []; // Would return current arbitrage opportunities
  }

  private async getPipelineOpportunities(): Promise<any[]> {
    return []; // Would return pending opportunities
  }

  private async getRecentExecutions(): Promise<any[]> {
    return []; // Would return recent trade executions
  }

  private async getSystemAlerts(): Promise<any[]> {
    return [
      {
        level: 'info' as const,
        message: 'System operating normally',
        timestamp: Date.now()
      }
    ];
  }

  // Public API methods
  async getJobStatus(jobId: string): Promise<BacktestJob | null> {
    const row = await this.db.get(
      `SELECT * FROM backtest_jobs WHERE id = ?`,
      [jobId]
    );
    
    if (!row) return null;
    
    return {
      id: row.id,
      type: row.type,
      status: row.status,
      config: JSON.parse(row.config),
      priority: row.priority,
      submittedAt: row.submitted_at,
      startedAt: row.started_at,
      completedAt: row.completed_at,
      results: row.results ? JSON.parse(row.results) : undefined,
      error: row.error,
      progress: row.progress
    };
  }

  async getBacktestReport(reportId: string): Promise<BacktestReport | null> {
    const row = await this.db.get(
      `SELECT * FROM backtest_reports WHERE id = ?`,
      [reportId]
    );
    
    if (!row) return null;
    
    return {
      id: row.id,
      jobId: row.job_id,
      type: row.type,
      summary: JSON.parse(row.summary),
      detailed: JSON.parse(row.detailed),
      charts: JSON.parse(row.charts || '{}'),
      recommendations: JSON.parse(row.recommendations),
      timestamp: row.timestamp
    };
  }

  async getJobQueue(): Promise<BacktestJob[]> {
    return [...this.jobQueue];
  }

  async getActiveJobs(): Promise<BacktestJob[]> {
    return Array.from(this.activeJobs.values());
  }

  async stop(): Promise<void> {
    this.isProcessing = false;
    this.jobQueue.length = 0;
    this.activeJobs.clear();
    
    await this.backtestEngine.stop();
    await this.liveValidator.stop();
    await this.kafka.disconnect();
    await this.redis.disconnect();
    await this.db.close();
  }
}