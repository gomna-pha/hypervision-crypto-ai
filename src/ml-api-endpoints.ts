/**
 * ML API Endpoints
 * 
 * New endpoints for advanced ML features:
 * - /api/ml/pipeline - Run full ML pipeline
 * - /api/ml/regime - Get market regime
 * - /api/ml/meta-model - Get XGBoost confidence
 * - /api/ml/strategies - Get strategy signals
 * - /api/ml/portfolio - Get portfolio metrics with risk
 * - /api/ml/ga-optimize - Run GA optimization
 */

import { Hono } from 'hono';
import { MLOrchestrator } from './ml/ml-orchestrator';
import type { RawMarketData } from './ml/feature-engineering';

// Initialize ML Orchestrator (singleton)
let mlOrchestrator: MLOrchestrator | null = null;

function getMLOrchestrator(): MLOrchestrator {
  if (!mlOrchestrator) {
    mlOrchestrator = new MLOrchestrator({
      enableGA: true,
      enableHyperbolic: true,
      enableXGBoost: true,
      enableStrategies: true,
      enableRiskManager: true,
      gaGenerations: 30, // Reduced for faster response
      totalCapital: 100000,
    });
  }
  return mlOrchestrator;
}

export function registerMLEndpoints(app: Hono) {
  /**
   * POST /api/ml/pipeline
   * Run full ML pipeline with all components
   */
  app.post('/api/ml/pipeline', async (c) => {
    try {
      const startTime = Date.now();
      
      // Parse request body (or use defaults)
      const body = await c.req.json().catch(() => ({}));
      
      // Construct raw market data (use provided data or defaults)
      const rawData: RawMarketData = {
        timestamp: new Date(),
        symbol: body.symbol || 'BTC-USD',
        spotPrice: body.spotPrice || 96500,
        perpPrice: body.perpPrice || 96530,
        bidPrice: body.bidPrice || 96495,
        askPrice: body.askPrice || 96505,
        exchangePrices: body.exchangePrices || {
          binance: 96500,
          coinbase: 96530,
          kraken: 96520,
        },
        volume24h: body.volume24h || 1000000,
        bidVolume: body.bidVolume || 500000,
        askVolume: body.askVolume || 500000,
        liquidity: body.liquidity || 5000000,
        fundingRate: body.fundingRate || 0.01,
        openInterest: body.openInterest || 1000000000,
      };
      
      // Run ML pipeline
      const orchestrator = getMLOrchestrator();
      const output = await orchestrator.runPipeline(rawData);
      
      const totalTime = Date.now() - startTime;
      
      return c.json({
        success: true,
        data: {
          // Features
          features: {
            returns: output.features.returns,
            spreads: output.features.spreads,
            volatility: output.features.volatility,
            flow: output.features.flow,
            zScores: output.features.zScores,
            rolling: output.features.rolling,
          },
          
          // Agent signals
          agentSignals: output.agentSignals.map(s => ({
            agentId: s.agentId,
            signal: s.signal,
            confidence: s.confidence,
            expectedAlpha: s.expectedAlpha,
            riskScore: s.riskScore,
            explanation: s.explanation,
          })),
          
          // GA genome
          gaGenome: output.gaGenome ? {
            fitness: output.gaGenome.fitness,
            activeSignals: output.gaGenome.activeSignals,
            weights: output.gaGenome.weights,
            generation: output.gaGenome.generation,
          } : null,
          
          // Market regime
          regime: {
            current: output.regimeState.regime,
            confidence: output.regimeState.confidence,
            transitionProb: output.regimeState.transitionProb,
          },
          
          // Meta-model
          metaModel: output.metaModelOutput ? {
            confidenceScore: output.metaModelOutput.confidenceScore,
            action: output.metaModelOutput.action,
            signalAgreement: output.metaModelOutput.signalAgreement,
            signalDivergence: output.metaModelOutput.signalDivergence,
            exposureScaler: output.metaModelOutput.exposureScaler,
            leverageScaler: output.metaModelOutput.leverageScaler,
          } : null,
          
          // Strategy signals
          strategies: output.strategySignals.map(s => ({
            strategy: s.strategy,
            action: s.action,
            confidence: s.confidence,
            expectedReturn: s.expectedReturn,
            risk: s.risk,
            reason: s.reason,
            trade: s.trade ? {
              id: s.trade.id,
              symbol: s.trade.symbol,
              side: s.trade.side,
              positionSize: s.trade.positionSize,
              leverage: s.trade.leverage,
              expectedAlpha: s.trade.expectedAlpha,
            } : null,
          })),
          
          // Portfolio metrics
          portfolio: output.portfolioMetrics ? {
            totalCapital: output.portfolioMetrics.totalCapital,
            totalExposure: output.portfolioMetrics.totalExposure,
            totalLeverage: output.portfolioMetrics.totalLeverage,
            totalPnL: output.portfolioMetrics.totalPnL,
            totalReturn: output.portfolioMetrics.totalReturn,
            sharpeRatio: output.portfolioMetrics.sharpeRatio,
            maxDrawdown: output.portfolioMetrics.maxDrawdown,
            currentDrawdown: output.portfolioMetrics.currentDrawdown,
            volatility: output.portfolioMetrics.volatility,
            numActiveTrades: output.portfolioMetrics.numActiveTrades,
            riskUtilization: output.portfolioMetrics.riskUtilization,
          } : null,
          
          // Risk constraints
          riskConstraints: output.riskConstraints.map(rc => ({
            name: rc.name,
            violated: rc.violated,
            current: rc.current,
            limit: rc.limit,
            severity: rc.severity,
          })),
        },
        metadata: {
          latencyMs: totalTime,
          pipelineLatencyMs: output.latencyMs,
          timestamp: output.timestamp,
        },
      });
    } catch (error: any) {
      console.error('[ML Pipeline API] Error:', error);
      return c.json({
        success: false,
        error: error.message || 'Failed to run ML pipeline',
      }, 500);
    }
  });

  /**
   * GET /api/ml/regime
   * Get current market regime
   */
  app.get('/api/ml/regime', async (c) => {
    try {
      const orchestrator = getMLOrchestrator();
      const components = orchestrator.getComponents();
      
      const currentRegime = components.regimeDetector.getCurrentRegime();
      const history = components.regimeDetector.getRegimeHistory();
      const persistence = components.regimeDetector.getRegimePersistence();
      
      return c.json({
        success: true,
        data: {
          current: currentRegime,
          persistence,
          history: history.slice(-20), // Last 20 regimes
        },
      });
    } catch (error: any) {
      return c.json({ success: false, error: error.message }, 500);
    }
  });

  /**
   * GET /api/ml/strategies
   * Get active strategy signals
   */
  app.get('/api/ml/strategies', async (c) => {
    try {
      const orchestrator = getMLOrchestrator();
      const components = orchestrator.getComponents();
      
      if (!components.strategies) {
        return c.json({ success: false, error: 'Strategies not enabled' }, 400);
      }
      
      const activeTrades = components.strategies.getActiveTrades();
      
      return c.json({
        success: true,
        data: {
          activeTrades: activeTrades.map(t => ({
            id: t.id,
            strategy: t.strategy,
            symbol: t.symbol,
            side: t.side,
            entryPrice: t.entryPrice,
            targetPrice: t.targetPrice,
            stopLoss: t.stopLoss,
            positionSize: t.positionSize,
            leverage: t.leverage,
            expectedAlpha: t.expectedAlpha,
            confidence: t.confidence,
            regime: t.regime,
            status: t.status,
            timestamp: t.timestamp,
          })),
          numActiveTrades: activeTrades.length,
        },
      });
    } catch (error: any) {
      return c.json({ success: false, error: error.message }, 500);
    }
  });

  /**
   * GET /api/ml/portfolio
   * Get comprehensive portfolio metrics
   */
  app.get('/api/ml/portfolio', async (c) => {
    try {
      const orchestrator = getMLOrchestrator();
      const components = orchestrator.getComponents();
      
      if (!components.riskManager) {
        return c.json({ success: false, error: 'Risk manager not enabled' }, 400);
      }
      
      const metrics = components.riskManager.calculateMetrics();
      const constraints = components.riskManager.checkRiskConstraints();
      
      return c.json({
        success: true,
        data: {
          metrics: {
            totalCapital: metrics.totalCapital,
            usedCapital: metrics.usedCapital,
            availableCapital: metrics.availableCapital,
            totalExposure: metrics.totalExposure,
            totalLeverage: metrics.totalLeverage,
            totalPnL: metrics.totalPnL,
            totalReturn: metrics.totalReturn,
            sharpeRatio: metrics.sharpeRatio,
            maxDrawdown: metrics.maxDrawdown,
            currentDrawdown: metrics.currentDrawdown,
            volatility: metrics.volatility,
            numActiveTrades: metrics.numActiveTrades,
            riskUtilization: metrics.riskUtilization,
          },
          riskConstraints: constraints.map(rc => ({
            name: rc.name,
            violated: rc.violated,
            current: rc.current,
            limit: rc.limit,
            severity: rc.severity,
          })),
          positions: Array.from(metrics.positionsByStrategy.values()).map(p => ({
            strategyName: p.strategyName,
            totalExposure: p.totalExposure,
            weight: p.weight,
            pnl: p.pnl,
            returnPercent: p.returnPercent,
            sharpeRatio: p.sharpeRatio,
            volatility: p.volatility,
            numTrades: p.trades.length,
          })),
        },
      });
    } catch (error: any) {
      return c.json({ success: false, error: error.message }, 500);
    }
  });

  /**
   * POST /api/ml/ga-optimize
   * Run genetic algorithm optimization (expensive operation)
   */
  app.post('/api/ml/ga-optimize', async (c) => {
    try {
      const startTime = Date.now();
      
      // This is a computationally expensive operation
      // Should be called infrequently (e.g., once per hour)
      
      const orchestrator = getMLOrchestrator();
      const components = orchestrator.getComponents();
      
      if (!components.gaSelector) {
        return c.json({ success: false, error: 'GA not enabled' }, 400);
      }
      
      // Get current population stats
      const population = components.gaSelector.getPopulation();
      const bestGenome = components.gaSelector.getBestGenome();
      const generation = components.gaSelector.getGeneration();
      
      return c.json({
        success: true,
        data: {
          bestGenome: bestGenome ? {
            id: bestGenome.id,
            fitness: bestGenome.fitness,
            activeSignals: bestGenome.activeSignals,
            weights: bestGenome.weights,
            generation: bestGenome.generation,
            age: bestGenome.age,
          } : null,
          populationSize: population.length,
          currentGeneration: generation,
          avgFitness: population.reduce((sum, g) => sum + g.fitness, 0) / population.length,
        },
        metadata: {
          latencyMs: Date.now() - startTime,
        },
      });
    } catch (error: any) {
      return c.json({ success: false, error: error.message }, 500);
    }
  });
}
