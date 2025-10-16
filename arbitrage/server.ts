/**
 * Agent-Based LLM Arbitrage Platform - REST API Server
 * Provides HTTP endpoints for platform management, monitoring, and analysis
 * Designed for investor demos and production deployment
 */

import { Hono } from 'hono';
import { cors } from 'hono/cors';
import { createArbitragePlatform, PlatformConfig } from './core/orchestrator.js';

// Global platform instance
let platform: any = null;
let isInitialized = false;

// Default configuration
const SERVER_CONFIG: Partial<PlatformConfig> = {
  agents: {
    enabled_agents: ['economic', 'sentiment', 'price', 'volume', 'trade', 'image'],
    polling_intervals: {
      economic: 10000,  // 10 seconds for demo
      sentiment: 5000,  // 5 seconds
      price: 2000,      // 2 seconds
      volume: 5000,     // 5 seconds
      trade: 5000,      // 5 seconds
      image: 10000      // 10 seconds
    }
  },
  llm: {
    provider: 'anthropic',
    model: 'claude-3-sonnet-20240229',
    api_key: 'demo_anthropic_key', // Replace with real key
    max_tokens: 1000,
    temperature: 0.1,
    timeout_ms: 15000,
    fallback_provider: 'openai',
    fallback_model: 'gpt-4',
    fallback_api_key: 'demo_openai_key'
  },
  execution: {
    sandbox_mode: true,
    max_position_size_usd: 100000
  },
  api_keys: {
    fred_api_key: 'demo_fred_key',
    // Add real API keys here for production
  }
};

const app = new Hono();

// Enable CORS for all routes
app.use('*', cors({
  origin: '*',
  allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowHeaders: ['Content-Type', 'Authorization'],
}));

/**
 * Initialize platform if not already done
 */
async function ensurePlatformInitialized() {
  if (!isInitialized) {
    platform = createArbitragePlatform(SERVER_CONFIG);
    isInitialized = true;
  }
  return platform;
}

/**
 * Health check endpoint
 */
app.get('/health', (c) => {
  return c.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    service: 'arbitrage-platform-api',
    version: '1.0.0'
  });
});

/**
 * Platform status endpoint
 */
app.get('/api/platform/status', async (c) => {
  try {
    await ensurePlatformInitialized();
    
    const status = platform?.getSystemStatus() || {
      running: false,
      agent_health: {},
      fusion_stats: {},
      decision_stats: {},
      hyperbolic_stats: {}
    };
    
    return c.json({
      success: true,
      data: {
        ...status,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    return c.json({
      success: false,
      error: error.message
    }, 500);
  }
});

/**
 * System metrics endpoint
 */
app.get('/api/platform/metrics', async (c) => {
  try {
    await ensurePlatformInitialized();
    
    const metrics = platform?.getSystemMetrics() || {
      uptime_seconds: 0,
      total_predictions: 0,
      successful_predictions: 0,
      active_agents: 0,
      avg_agent_confidence: 0,
      system_health_score: 0,
      last_fusion_timestamp: new Date().toISOString(),
      circuit_breaker_active: false
    };
    
    return c.json({
      success: true,
      data: {
        ...metrics,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    return c.json({
      success: false,
      error: error.message
    }, 500);
  }
});

/**
 * Start platform endpoint
 */
app.post('/api/platform/start', async (c) => {
  try {
    await ensurePlatformInitialized();
    
    if (platform.isRunning) {
      return c.json({
        success: false,
        error: 'Platform is already running'
      }, 400);
    }
    
    await platform.start();
    
    return c.json({
      success: true,
      message: 'Platform started successfully',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    return c.json({
      success: false,
      error: error.message
    }, 500);
  }
});

/**
 * Stop platform endpoint
 */
app.post('/api/platform/stop', async (c) => {
  try {
    if (!platform) {
      return c.json({
        success: false,
        error: 'Platform is not initialized'
      }, 400);
    }
    
    await platform.stop();
    
    return c.json({
      success: true,
      message: 'Platform stopped successfully',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    return c.json({
      success: false,
      error: error.message
    }, 500);
  }
});

/**
 * Get arbitrage opportunities endpoint
 */
app.get('/api/opportunities', async (c) => {
  try {
    await ensurePlatformInitialized();
    
    const limitParam = c.req.query('limit');
    const limit = limitParam ? parseInt(limitParam) : 50;
    
    const opportunities = platform?.getOpportunities(limit) || [];
    
    return c.json({
      success: true,
      data: {
        opportunities,
        count: opportunities.length,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    return c.json({
      success: false,
      error: error.message
    }, 500);
  }
});

/**
 * Manual analysis endpoint
 */
app.post('/api/analysis/manual', async (c) => {
  try {
    await ensurePlatformInitialized();
    
    const opportunity = await platform.performAnalysis();
    
    return c.json({
      success: true,
      data: {
        opportunity,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    return c.json({
      success: false,
      error: error.message
    }, 500);
  }
});

/**
 * Investor summary endpoint
 */
app.get('/api/investor/summary', async (c) => {
  try {
    await ensurePlatformInitialized();
    
    const summary = platform?.getInvestorSummary() || {
      platform_status: 'STOPPED',
      total_opportunities: 0,
      approval_rate: 0,
      avg_confidence: 0,
      system_uptime: '0h 0m',
      agent_health: '0%',
      recent_performance: {}
    };
    
    return c.json({
      success: true,
      data: {
        ...summary,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    return c.json({
      success: false,
      error: error.message
    }, 500);
  }
});

/**
 * Agent health endpoint
 */
app.get('/api/agents/health', async (c) => {
  try {
    await ensurePlatformInitialized();
    
    const status = platform?.getSystemStatus();
    const agentHealth = status?.agent_health || {};
    
    return c.json({
      success: true,
      data: {
        agents: agentHealth,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    return c.json({
      success: false,
      error: error.message
    }, 500);
  }
});

/**
 * Configuration endpoint
 */
app.get('/api/platform/config', (c) => {
  return c.json({
    success: true,
    data: {
      enabled_agents: SERVER_CONFIG.agents?.enabled_agents || [],
      sandbox_mode: SERVER_CONFIG.execution?.sandbox_mode || true,
      llm_provider: SERVER_CONFIG.llm?.provider || 'anthropic',
      hyperbolic_dimension: SERVER_CONFIG.hyperbolic?.embedding_dim || 128,
      timestamp: new Date().toISOString()
    }
  });
});

/**
 * Live WebSocket endpoint for real-time updates
 */
app.get('/api/live', async (c) => {
  // WebSocket upgrade would go here in production
  // For now, return instructions
  return c.json({
    message: 'WebSocket endpoint for real-time updates',
    endpoint: '/ws/live',
    events: [
      'opportunity_detected',
      'opportunity_approved', 
      'agent_health_changed',
      'platform_status_changed'
    ],
    usage: 'Connect via WebSocket for real-time arbitrage opportunity updates'
  });
});

/**
 * Demo endpoint that starts platform and returns live data
 */
app.post('/api/demo/start', async (c) => {
  try {
    await ensurePlatformInitialized();
    
    // Start platform if not running
    if (!platform.isRunning) {
      await platform.start();
      
      // Wait a few seconds for initial data
      await new Promise(resolve => setTimeout(resolve, 5000));
    }
    
    // Return initial data
    const metrics = platform.getSystemMetrics();
    const opportunities = platform.getOpportunities(10);
    const status = platform.getSystemStatus();
    
    return c.json({
      success: true,
      message: 'Demo started successfully',
      data: {
        platform_running: status.running,
        initial_metrics: metrics,
        recent_opportunities: opportunities,
        agent_count: Object.keys(status.agent_health).length
      },
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    return c.json({
      success: false,
      error: error.message
    }, 500);
  }
});

/**
 * API documentation endpoint
 */
app.get('/api/docs', (c) => {
  const apiDocs = {
    name: 'Agent-Based LLM Arbitrage Platform API',
    version: '1.0.0',
    description: 'REST API for managing and monitoring the arbitrage trading platform',
    endpoints: {
      'GET /health': 'Health check',
      'GET /api/platform/status': 'Get platform status and agent health',
      'GET /api/platform/metrics': 'Get system metrics and performance data',
      'POST /api/platform/start': 'Start the arbitrage platform',
      'POST /api/platform/stop': 'Stop the arbitrage platform',
      'GET /api/opportunities': 'Get recent arbitrage opportunities',
      'POST /api/analysis/manual': 'Trigger manual analysis',
      'GET /api/investor/summary': 'Get investor-focused summary',
      'GET /api/agents/health': 'Get detailed agent health information',
      'GET /api/platform/config': 'Get platform configuration',
      'POST /api/demo/start': 'Start demo mode with live data'
    },
    features: [
      'Real-time multi-agent data collection',
      'Hyperbolic geometry embeddings',
      'LLM-powered fusion analysis',
      'Deterministic risk management',
      'Production-ready monitoring',
      'Investor-grade reporting'
    ]
  };
  
  return c.json(apiDocs);
});

/**
 * Root endpoint
 */
app.get('/', (c) => {
  return c.json({
    service: 'Agent-Based LLM Arbitrage Platform',
    status: 'running',
    version: '1.0.0',
    description: 'Advanced AI-powered arbitrage trading system with real-time LLM integration',
    docs: '/api/docs',
    health: '/health',
    demo: 'POST /api/demo/start'
  });
});

/**
 * Error handler
 */
app.onError((err, c) => {
  console.error('API Error:', err);
  
  return c.json({
    success: false,
    error: 'Internal server error',
    message: err.message,
    timestamp: new Date().toISOString()
  }, 500);
});

/**
 * 404 handler
 */
app.notFound((c) => {
  return c.json({
    success: false,
    error: 'Endpoint not found',
    available_endpoints: '/api/docs',
    timestamp: new Date().toISOString()
  }, 404);
});

// Graceful shutdown
process.on('SIGINT', async () => {
  console.log('\n🛑 Received shutdown signal, stopping platform...');
  
  if (platform && platform.isRunning) {
    try {
      await platform.stop();
      console.log('✅ Platform stopped successfully');
    } catch (error) {
      console.error('❌ Error stopping platform:', error.message);
    }
  }
  
  process.exit(0);
});

process.on('SIGTERM', async () => {
  console.log('\n🛑 Received termination signal, stopping platform...');
  
  if (platform && platform.isRunning) {
    try {
      await platform.stop();
      console.log('✅ Platform stopped successfully');
    } catch (error) {
      console.error('❌ Error stopping platform:', error.message);
    }
  }
  
  process.exit(0);
});

export default app;
export { app, platform };