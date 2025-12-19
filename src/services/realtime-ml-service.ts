/**
 * REAL-TIME ML SERVICE
 * 
 * Integrates WebSocket feeds with ML pipeline
 * Runs ML predictions on every price update
 */

import { websocketService, AggregatedMarketData } from './websocket-service';
import { MLOrchestrator, MLPipelineOutput } from '../ml/ml-orchestrator';

export interface RealtimeMLOutput extends MLPipelineOutput {
  // Additional metadata
  dataSource: 'websocket' | 'simulated';
  dataQuality: 'excellent' | 'good' | 'degraded' | 'poor';
  connectedExchanges: string[];
  updateLatencyMs: number;
}

export class RealtimeMLService {
  private static instance: RealtimeMLService;
  private orchestrator: MLOrchestrator;
  private isRunning = false;
  private latestOutput: Map<string, RealtimeMLOutput> = new Map();
  private updateCallbacks: Set<(output: RealtimeMLOutput) => void> = new Set();
  
  // Performance tracking
  private updateCount = 0;
  private totalLatency = 0;
  private lastUpdate = Date.now();
  
  private constructor() {
    this.orchestrator = new MLOrchestrator({
      enableGA: true,
      enableHyperbolic: true,
      enableXGBoost: true,
      enableStrategies: true,
      enableRiskManager: true,
      totalCapital: 100000,
    });
  }
  
  static getInstance(): RealtimeMLService {
    if (!RealtimeMLService.instance) {
      RealtimeMLService.instance = new RealtimeMLService();
    }
    return RealtimeMLService.instance;
  }
  
  /**
   * Start real-time ML pipeline
   */
  async start(symbols: string[] = ['BTC']): Promise<void> {
    if (this.isRunning) {
      console.log('[Realtime ML] Already running');
      return;
    }
    
    try {
      console.log('[Realtime ML] Starting real-time ML pipeline...');
      
      // Initialize WebSocket service
      await websocketService.initialize(symbols);
      
      // Subscribe to market data updates
      for (const symbol of symbols) {
        websocketService.subscribe(symbol, (data) => this.handleMarketUpdate(data));
      }
      
      this.isRunning = true;
      console.log('[Realtime ML] ✅ Real-time ML pipeline active');
    } catch (error) {
      console.error('[Realtime ML] ❌ Failed to start:', error);
      throw error;
    }
  }
  
  /**
   * Handle market data update
   */
  private async handleMarketUpdate(marketData: AggregatedMarketData): Promise<void> {
    const startTime = Date.now();
    
    try {
      // Convert to RawMarketData format
      const rawData = websocketService.toRawMarketData(marketData);
      
      // Run ML pipeline
      const mlOutput = await this.orchestrator.runPipeline(rawData);
      
      // Create realtime output
      const realtimeOutput: RealtimeMLOutput = {
        ...mlOutput,
        dataSource: 'websocket',
        dataQuality: marketData.dataQuality,
        connectedExchanges: marketData.connectedExchanges,
        updateLatencyMs: Date.now() - startTime,
      };
      
      // Store latest output
      this.latestOutput.set(marketData.symbol, realtimeOutput);
      
      // Update performance metrics
      this.updateCount++;
      this.totalLatency += realtimeOutput.updateLatencyMs;
      this.lastUpdate = Date.now();
      
      // Notify subscribers
      this.notifySubscribers(realtimeOutput);
      
      // Log performance (throttled)
      if (this.updateCount % 10 === 0) {
        const avgLatency = this.totalLatency / this.updateCount;
        console.log(`[Realtime ML] Updates: ${this.updateCount}, Avg Latency: ${avgLatency.toFixed(0)}ms, Regime: ${mlOutput.regimeState.current}, Action: ${mlOutput.metaModelOutput?.action || 'N/A'}`);
      }
    } catch (error) {
      console.error('[Realtime ML] Pipeline error:', error);
    }
  }
  
  /**
   * Subscribe to ML updates
   */
  onUpdate(callback: (output: RealtimeMLOutput) => void): () => void {
    this.updateCallbacks.add(callback);
    
    // Return unsubscribe function
    return () => {
      this.updateCallbacks.delete(callback);
    };
  }
  
  /**
   * Notify subscribers
   */
  private notifySubscribers(output: RealtimeMLOutput): void {
    this.updateCallbacks.forEach(callback => {
      try {
        callback(output);
      } catch (error) {
        console.error('[Realtime ML] Subscriber callback error:', error);
      }
    });
  }
  
  /**
   * Get latest output for symbol
   */
  getLatestOutput(symbol: string): RealtimeMLOutput | null {
    return this.latestOutput.get(symbol) || null;
  }
  
  /**
   * Get all latest outputs
   */
  getAllLatestOutputs(): RealtimeMLOutput[] {
    return Array.from(this.latestOutput.values());
  }
  
  /**
   * Get performance metrics
   */
  getPerformanceMetrics(): {
    updateCount: number;
    avgLatencyMs: number;
    lastUpdateAgo: number;
    isHealthy: boolean;
  } {
    const avgLatency = this.updateCount > 0 ? this.totalLatency / this.updateCount : 0;
    const lastUpdateAgo = Date.now() - this.lastUpdate;
    const isHealthy = this.isRunning && lastUpdateAgo < 30000; // < 30 seconds
    
    return {
      updateCount: this.updateCount,
      avgLatencyMs: avgLatency,
      lastUpdateAgo,
      isHealthy,
    };
  }
  
  /**
   * Get system status
   */
  getStatus(): {
    isRunning: boolean;
    websocketConnected: boolean;
    exchanges: Record<string, boolean>;
    symbols: string[];
    performance: ReturnType<typeof this.getPerformanceMetrics>;
  } {
    const wsStatus = websocketService.getConnectionStatus();
    
    return {
      isRunning: this.isRunning,
      websocketConnected: wsStatus.connected,
      exchanges: wsStatus.exchanges,
      symbols: Array.from(this.latestOutput.keys()),
      performance: this.getPerformanceMetrics(),
    };
  }
  
  /**
   * Stop real-time ML pipeline
   */
  stop(): void {
    if (!this.isRunning) {
      console.log('[Realtime ML] Not running');
      return;
    }
    
    websocketService.shutdown();
    this.isRunning = false;
    this.updateCallbacks.clear();
    console.log('[Realtime ML] 🛑 Stopped');
  }
}

/**
 * Singleton export
 */
export const realtimeMLService = RealtimeMLService.getInstance();

/**
 * Example usage:
 * 
 * // Start real-time ML pipeline
 * await realtimeMLService.start(['BTC', 'ETH']);
 * 
 * // Subscribe to ML updates
 * const unsubscribe = realtimeMLService.onUpdate((output) => {
 *   console.log('Regime:', output.regimeState.current);
 *   console.log('ML Action:', output.metaModelOutput?.action);
 *   console.log('Data Quality:', output.dataQuality);
 *   console.log('Latency:', output.updateLatencyMs, 'ms');
 * });
 * 
 * // Get latest output
 * const btcOutput = realtimeMLService.getLatestOutput('BTC');
 * 
 * // Get system status
 * const status = realtimeMLService.getStatus();
 * console.log('Status:', status);
 * 
 * // Cleanup
 * unsubscribe();
 * realtimeMLService.stop();
 */
