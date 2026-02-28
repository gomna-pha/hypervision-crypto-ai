import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serveStatic } from 'hono/cloudflare-workers'

// Streamlined dashboard and API
import registerStreamlinedDashboard from './streamlined-dashboard'
import {
  getAgentSignals,
  getMarketRegime,
  getGAStatus,
  getPortfolioMetrics,
  getHyperbolicEmbeddings
} from './streamlined-api'

// ML endpoints (maintaining core functionality)
import { registerMLEndpoints } from './ml-api-endpoints'

// Real API services
import { detectAllRealOpportunities } from './api-services'

const app = new Hono()

// ============================================================================
// MIDDLEWARE
// ============================================================================

// Enable CORS for API routes
app.use('/api/*', cors())

// Serve static files from public directory
app.use('/static/*', serveStatic({ root: './public' }))

// ============================================================================
// MAIN DASHBOARD - Streamlined Dual-Interface
// ============================================================================

// Register STREAMLINED dashboard as main route (dual-interface: user + research)
registerStreamlinedDashboard(app)

// ============================================================================
// CORE API ENDPOINTS - Simplified & Clean
// ============================================================================

// 5 Agent Signals (Economic, Sentiment, Cross-Exchange, On-Chain, CNN)
app.get('/api/agents', getAgentSignals)

// Market Regime Detection (Crisis/Recovery/Late Cycle/Neutral)
app.get('/api/regime', getMarketRegime)

// Genetic Algorithm Status & Results
app.get('/api/ga/status', getGAStatus)

// Portfolio Metrics (Balance, Sharpe, etc.)
app.get('/api/portfolio/metrics', getPortfolioMetrics)

// Hyperbolic Embeddings (Signal Hierarchy)
app.get('/api/hyperbolic/embeddings', getHyperbolicEmbeddings)

// Live Arbitrage Opportunities
app.get('/api/opportunities', async (c) => {
  try {
    const realOpportunities = await detectAllRealOpportunities()
    console.log(`[API] Found ${realOpportunities.length} opportunities`)
    return c.json(realOpportunities)
  } catch (error) {
    console.error('[API /api/opportunities] Error:', error)
    return c.json([])
  }
})

// ============================================================================
// ADVANCED ML ENDPOINTS - Research Features
// ============================================================================

// Register ML endpoints (GA, Regime Detection, Hyperbolic Embeddings)
registerMLEndpoints(app)

// ============================================================================
// EXPORT
// ============================================================================

export default app
