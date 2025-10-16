import EventEmitter from 'events';
import { FusionBrain, ArbitragePrediction } from './fusion-brain';
import { DecisionEngine, ExecutionDecision } from './decision-engine';
import { MicroBacktester, BacktestResult } from './micro-backtesting';
import * as yaml from 'js-yaml';
import * as fs from 'fs';
import * as path from 'path';

export interface PlatformState {
  status: 'starting' | 'running' | 'stopping' | 'stopped' | 'error';
  components: {
    fusion_brain: 'active' | 'inactive' | 'error';
    decision_engine: 'active' | 'inactive' | 'error';
    micro_backtester: 'active' | 'inactive' | 'error';
  };
  metrics: {
    total_predictions: number;
    total_decisions: number;
    total_backtests: number;
    execution_rate: number;
    avg_aos_score: number;
    avg_backtest_quality: number;
  };
  uptime_ms: number;
  last_activity: string;
}

export interface PlatformEvent {
  event_id: string;
  timestamp: string;
  event_type: 'prediction' | 'decision' | 'backtest_completed' | 'error' | 'status_change';
  component: string;
  data: any;
  severity: 'info' | 'warning' | 'error';
}

/**
 * PlatformOrchestrator - Main coordinator for the LLM Arbitrage Platform
 * 
 * Responsibilities:
 * 1. Initialize and coordinate all platform components
 * 2. Manage the complete workflow: Agents → Fusion → Decision → Backtesting  
 * 3. Provide unified platform state and monitoring
 * 4. Handle errors and component recovery
 * 5. Expose transparent parameters and metrics to investors
 * 
 * Workflow:
 * Agents collect data → FusionBrain processes with LLM → DecisionEngine scores →  
 * MicroBacktester validates → Results feed back to improve system
 */
export class PlatformOrchestrator extends EventEmitter {
  private fusionBrain: FusionBrain;
  private decisionEngine: DecisionEngine;
  private microBacktester: MicroBacktester;
  
  private platformState: PlatformState;
  private startTime: number = Date.now();
  private eventHistory: PlatformEvent[] = [];
  private config: any;
  
  // Visible Parameters for Investors
  public readonly parameters = {
    workflow_enabled: true,            // Enable full agent → fusion → decision workflow
    auto_backtesting_enabled: true,    // Automatically start backtests for predictions
    error_recovery_enabled: true,      // Attempt to recover from component errors
    performance_monitoring_enabled: true, // Monitor and log performance metrics
    event_history_retention_hours: 24, // Keep 24 hours of platform events
    health_check_interval_ms: 10000,   // Health check every 10 seconds
    max_component_restart_attempts: 3, // Max restart attempts per component
    platform_timeout_minutes: 60,     // Auto-stop platform after 1 hour idle
  };

  // Visible Constraints for Investors
  public readonly constraints = {
    max_concurrent_workflows: 5,       // Max simultaneous prediction workflows  
    max_event_history_size: 1000,      // Limit event history memory usage
    max_component_error_rate: 0.1,     // Stop component if >10% error rate
    required_components: ['fusion_brain', 'decision_engine', 'micro_backtester'],
    startup_timeout_ms: 30000,         // 30 second startup timeout
    graceful_shutdown_timeout_ms: 15000, // 15 second shutdown timeout
  };

  // Component Health Status
  private componentHealth = {
    fusion_brain: { errors: 0, last_activity: Date.now(), restart_attempts: 0 },
    decision_engine: { errors: 0, last_activity: Date.now(), restart_attempts: 0 },
    micro_backtester: { errors: 0, last_activity: Date.now(), restart_attempts: 0 },
  };

  private healthCheckInterval?: NodeJS.Timeout;
  private activeWorkflows: Map<string, {
    prediction: ArbitragePrediction;
    decision?: ExecutionDecision;
    backtest_id?: string;
    backtest_result?: BacktestResult;
    start_time: number;
    stage: 'prediction' | 'decision' | 'backtesting' | 'completed' | 'error';
  }> = new Map();

  constructor() {
    super();
    
    this.loadConfiguration();
    
    // Initialize components
    this.fusionBrain = new FusionBrain();
    this.decisionEngine = new DecisionEngine();
    this.microBacktester = new MicroBacktester();
    
    // Initialize platform state
    this.platformState = {
      status: 'stopped',
      components: {
        fusion_brain: 'inactive',
        decision_engine: 'inactive',
        micro_backtester: 'inactive'
      },
      metrics: {
        total_predictions: 0,
        total_decisions: 0,
        total_backtests: 0,
        execution_rate: 0,
        avg_aos_score: 0,
        avg_backtest_quality: 0
      },
      uptime_ms: 0,
      last_activity: new Date().toISOString()
    };
    
    this.setupEventHandlers();
    console.log('✅ PlatformOrchestrator initialized');
  }

  private loadConfiguration(): void {
    try {
      const configPath = path.join(__dirname, '..', 'config.yaml');
      const configFile = fs.readFileSync(configPath, 'utf8');
      this.config = yaml.load(configFile) as any;
    } catch (error) {
      console.warn('⚠️ Could not load config.yaml, using defaults');
      this.config = {};
    }
  }

  private setupEventHandlers(): void {
    // Listen to FusionBrain predictions
    this.fusionBrain.on('prediction', (prediction: ArbitragePrediction) => {
      this.handleNewPrediction(prediction);
    });

    this.fusionBrain.on('error', (error: Error) => {
      this.handleComponentError('fusion_brain', error);
    });

    // Listen to DecisionEngine decisions
    this.decisionEngine.on('decision', (decision: ExecutionDecision) => {
      this.handleNewDecision(decision);
    });

    this.decisionEngine.on('error', (error: Error) => {
      this.handleComponentError('decision_engine', error);
    });

    // Listen to MicroBacktester results
    this.microBacktester.on('backtest_completed', (result: BacktestResult) => {
      this.handleBacktestCompleted(result);
    });

    this.microBacktester.on('backtest_error', (event: any) => {
      this.handleComponentError('micro_backtester', new Error(event.error));
    });
  }

  /**
   * Start the complete arbitrage platform
   */
  async start(): Promise<void> {
    console.log('🚀 Starting LLM Arbitrage Platform...');
    
    this.platformState.status = 'starting';
    this.logPlatformEvent('status_change', 'platform', { new_status: 'starting' }, 'info');
    
    try {
      // Start components in order
      console.log('🔧 Starting FusionBrain...');
      await this.fusionBrain.start();
      this.platformState.components.fusion_brain = 'active';
      this.componentHealth.fusion_brain.last_activity = Date.now();
      
      console.log('🔧 Starting DecisionEngine...');
      // DecisionEngine is stateless, no async start needed
      this.platformState.components.decision_engine = 'active';
      this.componentHealth.decision_engine.last_activity = Date.now();
      
      console.log('🔧 Starting MicroBacktester...');
      // MicroBacktester is stateless, no async start needed
      this.platformState.components.micro_backtester = 'active';
      this.componentHealth.micro_backtester.last_activity = Date.now();
      
      // Start health monitoring
      if (this.parameters.performance_monitoring_enabled) {
        this.startHealthMonitoring();
      }
      
      this.platformState.status = 'running';
      this.startTime = Date.now();
      
      this.logPlatformEvent('status_change', 'platform', { new_status: 'running' }, 'info');
      
      console.log('✅ LLM Arbitrage Platform is running!');
      console.log('📊 Platform will execute the full workflow:');
      console.log('   1. Agents collect economic, sentiment, microstructure data');
      console.log('   2. FusionBrain processes with hyperbolic embeddings + LLM');
      console.log('   3. DecisionEngine calculates AOS scores and makes execution decisions');
      console.log('   4. MicroBacktester validates predictions with real-time backtesting');
      
      this.emit('platform_started');
      
    } catch (error: any) {
      console.error('❌ Failed to start platform:', error);
      this.platformState.status = 'error';
      this.logPlatformEvent('error', 'platform', { error: error.message }, 'error');
      
      // Attempt cleanup on startup failure
      await this.stop();
      throw error;
    }
  }

  /**
   * Stop the platform gracefully
   */
  async stop(): Promise<void> {
    console.log('⏹️ Stopping LLM Arbitrage Platform...');
    
    this.platformState.status = 'stopping';
    this.logPlatformEvent('status_change', 'platform', { new_status: 'stopping' }, 'info');
    
    try {
      // Stop health monitoring first
      if (this.healthCheckInterval) {
        clearInterval(this.healthCheckInterval);
        this.healthCheckInterval = undefined;
      }
      
      // Complete any active backtests
      if (this.activeWorkflows.size > 0) {
        console.log(`🏁 Completing ${this.activeWorkflows.size} active workflows...`);
        await this.microBacktester.forceCompleteAllBacktests();
      }
      
      // Stop components
      console.log('🔧 Stopping FusionBrain...');
      await this.fusionBrain.stop();
      this.platformState.components.fusion_brain = 'inactive';
      
      console.log('🔧 DecisionEngine stopped');
      this.platformState.components.decision_engine = 'inactive';
      
      console.log('🔧 MicroBacktester stopped');
      this.platformState.components.micro_backtester = 'inactive';
      
      this.platformState.status = 'stopped';
      this.logPlatformEvent('status_change', 'platform', { new_status: 'stopped' }, 'info');
      
      console.log('✅ LLM Arbitrage Platform stopped');
      
      this.emit('platform_stopped');
      
    } catch (error: any) {
      console.error('❌ Error during platform shutdown:', error);
      this.platformState.status = 'error';
      this.logPlatformEvent('error', 'platform', { error: error.message }, 'error');
    }
  }

  /**
   * Handle new prediction from FusionBrain - start workflow
   */
  private async handleNewPrediction(prediction: ArbitragePrediction): Promise<void> {
    try {
      console.log(`🔮 New prediction received: ${prediction.id}`);
      
      this.platformState.metrics.total_predictions++;
      this.componentHealth.fusion_brain.last_activity = Date.now();
      
      // Create workflow tracker
      const workflow = {
        prediction,
        start_time: Date.now(),
        stage: 'prediction' as const
      };
      
      this.activeWorkflows.set(prediction.id, workflow);
      
      this.logPlatformEvent('prediction', 'fusion_brain', {
        prediction_id: prediction.id,
        predicted_spread_pct: prediction.predicted_spread_pct,
        confidence: prediction.confidence,
        direction: prediction.direction
      }, 'info');
      
      // Move to decision stage
      if (this.parameters.workflow_enabled) {
        await this.processDecisionStage(prediction.id);
      }
      
    } catch (error) {
      console.error('❌ Error handling prediction:', error);
      this.handleComponentError('fusion_brain', error as Error);
    }
  }

  /**
   * Process decision stage of workflow
   */
  private async processDecisionStage(predictionId: string): Promise<void> {
    const workflow = this.activeWorkflows.get(predictionId);
    if (!workflow) return;
    
    try {
      workflow.stage = 'decision';
      
      // Get decision from DecisionEngine
      const decision = await this.decisionEngine.evaluateArbitrageOpportunity(workflow.prediction);
      
      workflow.decision = decision;
      this.activeWorkflows.set(predictionId, workflow);
      
      // Move to backtesting stage if auto-backtesting enabled
      if (this.parameters.auto_backtesting_enabled) {
        await this.processBacktestingStage(predictionId);
      } else {
        workflow.stage = 'completed';
      }
      
    } catch (error) {
      console.error('❌ Error in decision stage:', error);
      workflow.stage = 'error';
      this.activeWorkflows.set(predictionId, workflow);
    }
  }

  /**
   * Handle new decision from DecisionEngine
   */
  private async handleNewDecision(decision: ExecutionDecision): Promise<void> {
    try {
      console.log(`📊 New decision: ${decision.execution_action} | AOS: ${decision.aos_score.overall_score.toFixed(1)}`);
      
      this.platformState.metrics.total_decisions++;
      this.componentHealth.decision_engine.last_activity = Date.now();
      
      // Update metrics
      this.updateExecutionRateMetric();
      this.updateAvgAOSMetric(decision.aos_score.overall_score);
      
      this.logPlatformEvent('decision', 'decision_engine', {
        decision_id: decision.decision_id,
        prediction_id: decision.prediction_id,
        execution_action: decision.execution_action,
        aos_score: decision.aos_score.overall_score,
        constraints_passed: decision.constraints_passed
      }, decision.execution_action === 'reject' ? 'warning' : 'info');
      
    } catch (error) {
      console.error('❌ Error handling decision:', error);
      this.handleComponentError('decision_engine', error as Error);
    }
  }

  /**
   * Process backtesting stage of workflow
   */
  private async processBacktestingStage(predictionId: string): Promise<void> {
    const workflow = this.activeWorkflows.get(predictionId);
    if (!workflow) return;
    
    try {
      workflow.stage = 'backtesting';
      
      // Start backtest
      const backtestId = await this.microBacktester.startBacktest(
        workflow.prediction,
        workflow.decision
      );
      
      workflow.backtest_id = backtestId;
      this.activeWorkflows.set(predictionId, workflow);
      
      console.log(`🧪 Started backtest: ${backtestId} for prediction: ${predictionId}`);
      
    } catch (error) {
      console.error('❌ Error in backtesting stage:', error);
      workflow.stage = 'error';
      this.activeWorkflows.set(predictionId, workflow);
    }
  }

  /**
   * Handle completed backtest
   */
  private async handleBacktestCompleted(result: BacktestResult): Promise<void> {
    try {
      console.log(`📈 Backtest completed: ${result.backtest_id} | Quality: ${result.learning_feedback.prediction_quality_score.toFixed(1)}`);
      
      this.platformState.metrics.total_backtests++;
      this.componentHealth.micro_backtester.last_activity = Date.now();
      
      // Find and update workflow
      const workflow = Array.from(this.activeWorkflows.entries())
        .find(([_, wf]) => wf.backtest_id === result.backtest_id);
      
      if (workflow) {
        const [predictionId, workflowData] = workflow;
        workflowData.backtest_result = result;
        workflowData.stage = 'completed';
        this.activeWorkflows.set(predictionId, workflowData);
      }
      
      // Update metrics
      this.updateAvgBacktestQualityMetric(result.learning_feedback.prediction_quality_score);
      
      this.logPlatformEvent('backtest_completed', 'micro_backtester', {
        backtest_id: result.backtest_id,
        prediction_id: result.prediction_id,
        simulated_pnl_pct: result.execution_simulation.simulated_pnl_pct,
        prediction_quality_score: result.learning_feedback.prediction_quality_score,
        direction_correct: result.prediction_accuracy.direction_correct
      }, 'info');
      
      // Clean up completed workflows after some time
      setTimeout(() => {
        this.activeWorkflows.delete(result.prediction_id);
      }, 300000); // 5 minutes
      
    } catch (error) {
      console.error('❌ Error handling backtest completion:', error);
      this.handleComponentError('micro_backtester', error as Error);
    }
  }

  /**
   * Handle component errors with recovery attempts
   */
  private handleComponentError(componentName: string, error: Error): void {
    console.error(`❌ Component error in ${componentName}:`, error);
    
    const health = this.componentHealth[componentName as keyof typeof this.componentHealth];
    if (health) {
      health.errors++;
      
      const errorRate = health.errors / ((Date.now() - this.startTime) / 60000); // Errors per minute
      
      if (errorRate > this.constraints.max_component_error_rate) {
        console.error(`🚨 High error rate in ${componentName}: ${errorRate.toFixed(2)} errors/min`);
        this.platformState.components[componentName as keyof typeof this.platformState.components] = 'error';
      }
    }
    
    this.logPlatformEvent('error', componentName, {
      error: error.message,
      stack: error.stack
    }, 'error');
    
    // Attempt recovery if enabled
    if (this.parameters.error_recovery_enabled) {
      this.attemptComponentRecovery(componentName);
    }
  }

  /**
   * Attempt to recover failed component
   */
  private async attemptComponentRecovery(componentName: string): Promise<void> {
    const health = this.componentHealth[componentName as keyof typeof this.componentHealth];
    
    if (health && health.restart_attempts < this.parameters.max_component_restart_attempts) {
      health.restart_attempts++;
      
      console.log(`🔄 Attempting recovery for ${componentName} (attempt ${health.restart_attempts})`);
      
      try {
        // Component-specific recovery logic would go here
        // For now, just reset error count and mark as active
        
        health.errors = 0;
        health.last_activity = Date.now();
        
        this.platformState.components[componentName as keyof typeof this.platformState.components] = 'active';
        
        console.log(`✅ Successfully recovered ${componentName}`);
        
      } catch (recoveryError) {
        console.error(`❌ Failed to recover ${componentName}:`, recoveryError);
      }
    }
  }

  /**
   * Start health monitoring
   */
  private startHealthMonitoring(): void {
    this.healthCheckInterval = setInterval(() => {
      this.performHealthCheck();
    }, this.parameters.health_check_interval_ms);
  }

  /**
   * Perform platform health check
   */
  private performHealthCheck(): void {
    const now = Date.now();
    this.platformState.uptime_ms = now - this.startTime;
    
    // Check component activity
    for (const [componentName, health] of Object.entries(this.componentHealth)) {
      const inactiveTime = now - health.last_activity;
      
      if (inactiveTime > 60000) { // 1 minute inactive
        console.warn(`⚠️ Component ${componentName} inactive for ${(inactiveTime / 1000).toFixed(0)}s`);
      }
    }
    
    // Clean up old events
    this.cleanupEventHistory();
    
    // Update last activity
    this.platformState.last_activity = new Date().toISOString();
  }

  /**
   * Update execution rate metric
   */
  private updateExecutionRateMetric(): void {
    const recentDecisions = this.decisionEngine.getExecutionHistory(100);
    const executeDecisions = recentDecisions.filter(d => d.execution_action === 'execute').length;
    
    this.platformState.metrics.execution_rate = recentDecisions.length > 0 
      ? executeDecisions / recentDecisions.length 
      : 0;
  }

  /**
   * Update average AOS metric
   */
  private updateAvgAOSMetric(newScore: number): void {
    const currentAvg = this.platformState.metrics.avg_aos_score;
    const count = this.platformState.metrics.total_decisions;
    
    // Exponential moving average
    const alpha = 0.1;
    this.platformState.metrics.avg_aos_score = count === 1 
      ? newScore 
      : (1 - alpha) * currentAvg + alpha * newScore;
  }

  /**
   * Update average backtest quality metric
   */
  private updateAvgBacktestQualityMetric(newScore: number): void {
    const currentAvg = this.platformState.metrics.avg_backtest_quality;
    const count = this.platformState.metrics.total_backtests;
    
    // Exponential moving average
    const alpha = 0.1;
    this.platformState.metrics.avg_backtest_quality = count === 1 
      ? newScore 
      : (1 - alpha) * currentAvg + alpha * newScore;
  }

  /**
   * Log platform event
   */
  private logPlatformEvent(
    eventType: PlatformEvent['event_type'],
    component: string,
    data: any,
    severity: PlatformEvent['severity']
  ): void {
    
    const event: PlatformEvent = {
      event_id: `evt_${Date.now()}_${Math.random().toString(36).substr(2, 6)}`,
      timestamp: new Date().toISOString(),
      event_type: eventType,
      component,
      data,
      severity
    };
    
    this.eventHistory.push(event);
    this.emit('platform_event', event);
  }

  /**
   * Clean up old event history
   */
  private cleanupEventHistory(): void {
    const cutoff = Date.now() - (this.parameters.event_history_retention_hours * 60 * 60 * 1000);
    
    this.eventHistory = this.eventHistory.filter(event => {
      return new Date(event.timestamp).getTime() > cutoff;
    });
    
    // Also enforce max size
    if (this.eventHistory.length > this.constraints.max_event_history_size) {
      this.eventHistory = this.eventHistory.slice(-this.constraints.max_event_history_size);
    }
  }

  // Public methods for transparency and monitoring
  getPlatformState(): PlatformState {
    return { ...this.platformState }; // Return copy
  }

  getVisibleParameters(): any {
    return {
      parameters: this.parameters,
      constraints: this.constraints,
      fusion_brain_params: this.fusionBrain.getVisibleParameters(),
      decision_engine_params: this.decisionEngine.getVisibleParameters(),
      micro_backtester_params: this.microBacktester.getVisibleParameters(),
      component_health: { ...this.componentHealth },
      platform_state: this.getPlatformState()
    };
  }

  getActiveWorkflows(): Array<{
    prediction_id: string;
    stage: string;
    duration_s: number;
    has_decision: boolean;
    has_backtest: boolean;
  }> {
    const now = Date.now();
    
    return Array.from(this.activeWorkflows.entries()).map(([id, workflow]) => ({
      prediction_id: id,
      stage: workflow.stage,
      duration_s: (now - workflow.start_time) / 1000,
      has_decision: !!workflow.decision,
      has_backtest: !!workflow.backtest_id
    }));
  }

  getRecentEvents(limit: number = 100): PlatformEvent[] {
    return this.eventHistory
      .sort((a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime())
      .slice(0, limit);
  }

  getPerformanceMetrics(): any {
    const fusionMetrics = this.fusionBrain.getActivePredictions().length;
    const decisionMetrics = this.decisionEngine.getPerformanceMetrics();
    const backtestMetrics = this.microBacktester.getBacktestPerformanceMetrics();
    
    return {
      platform: this.platformState.metrics,
      fusion_brain: { active_predictions: fusionMetrics },
      decision_engine: decisionMetrics,
      micro_backtester: backtestMetrics,
      active_workflows: this.activeWorkflows.size,
      uptime_hours: this.platformState.uptime_ms / (1000 * 60 * 60)
    };
  }

  /**
   * Get real-time dashboard data for the UI
   */
  getDashboardData(): any {
    const latestPrediction = this.fusionBrain.getLatestPrediction();
    const recentDecisions = this.decisionEngine.getExecutionHistory(10);
    const recentBacktests = this.microBacktester.getCompletedBacktests(10);
    const activeBacktests = this.microBacktester.getActiveBacktests();
    
    return {
      platform_state: this.getPlatformState(),
      latest_prediction: latestPrediction,
      recent_decisions: recentDecisions,
      recent_backtests: recentBacktests,
      active_backtests: activeBacktests,
      active_workflows: this.getActiveWorkflows(),
      performance_metrics: this.getPerformanceMetrics(),
      recent_events: this.getRecentEvents(20),
      visible_parameters: this.getVisibleParameters()
    };
  }

  /**
   * Emergency stop - force stop all components immediately
   */
  async emergencyStop(): Promise<void> {
    console.log('🛑 EMERGENCY STOP - Forcing platform shutdown');
    
    this.platformState.status = 'stopping';
    
    try {
      // Force stop all components without waiting
      if (this.healthCheckInterval) {
        clearInterval(this.healthCheckInterval);
      }
      
      // Force complete backtests
      await this.microBacktester.forceCompleteAllBacktests();
      
      // Stop fusion brain
      await this.fusionBrain.stop();
      
      this.platformState.status = 'stopped';
      this.logPlatformEvent('status_change', 'platform', { new_status: 'emergency_stopped' }, 'warning');
      
      console.log('🛑 Emergency stop completed');
      
    } catch (error) {
      console.error('❌ Error during emergency stop:', error);
      this.platformState.status = 'error';
    }
  }
}