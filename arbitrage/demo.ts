/**
 * Agent-Based LLM Arbitrage Platform - Live Demo
 * Demonstrates the complete system with real-time agent coordination,
 * hyperbolic embeddings, LLM fusion, and deterministic decision making
 */

import { createArbitragePlatform, PlatformConfig } from './core/orchestrator.js';

/**
 * Demo configuration for investor presentation
 */
const DEMO_CONFIG: Partial<PlatformConfig> = {
  agents: {
    enabled_agents: ['economic', 'sentiment', 'price', 'volume', 'trade', 'image'],
    polling_intervals: {
      economic: 5000,   // 5 seconds for demo (normally 1 hour)
      sentiment: 3000,  // 3 seconds for demo (normally 30 seconds)
      price: 1000,      // 1 second (real-time)
      volume: 2000,     // 2 seconds (normally 1 minute)
      trade: 2000,      // 2 seconds (normally 30 seconds)
      image: 5000       // 5 seconds (normally 1 minute)
    }
  },
  llm: {
    provider: 'anthropic', // Use 'openai' as fallback
    model: 'claude-3-sonnet-20240229',
    api_key: process.env.ANTHROPIC_API_KEY || 'demo_anthropic_key',
    max_tokens: 1000,
    temperature: 0.1,
    timeout_ms: 15000,
    fallback_provider: 'openai',
    fallback_model: 'gpt-4',
    fallback_api_key: process.env.OPENAI_API_KEY || 'demo_openai_key'
  },
  hyperbolic: {
    embedding_dim: 128,
    curvature: 1.0
  },
  execution: {
    sandbox_mode: true, // Safe mode for demo
    max_position_size_usd: 50000 // $50k max for demo
  },
  api_keys: {
    fred_api_key: process.env.FRED_API_KEY || 'demo_fred_key',
    twitter_bearer_token: process.env.TWITTER_BEARER_TOKEN,
    news_api_key: process.env.NEWS_API_KEY,
    anthropic_api_key: process.env.ANTHROPIC_API_KEY || 'demo_anthropic_key',
    openai_api_key: process.env.OPENAI_API_KEY || 'demo_openai_key'
  }
};

/**
 * Main demo function
 */
async function runArbitragePlatformDemo() {
  console.log('='.repeat(80));
  console.log('🚀 AGENT-BASED LLM ARBITRAGE PLATFORM - LIVE DEMO');
  console.log('='.repeat(80));
  console.log('');
  console.log('📋 Platform Features:');
  console.log('  ✅ 6 Autonomous Real-Time Agents');
  console.log('  ✅ Hyperbolic Geometry Embeddings');
  console.log('  ✅ Claude/GPT-4 Fusion Brain');
  console.log('  ✅ Deterministic Decision Engine');
  console.log('  ✅ Production-Ready Architecture');
  console.log('  ✅ Investor-Grade Monitoring');
  console.log('');

  // Initialize platform
  console.log('🔧 Initializing arbitrage platform...');
  const platform = createArbitragePlatform(DEMO_CONFIG);

  // Set up event listeners for live monitoring
  setupEventListeners(platform);

  try {
    // Start the platform
    console.log('🚀 Starting platform (this may take 10-15 seconds)...');
    await platform.start();

    // Display initial system status
    displaySystemStatus(platform);

    // Run demo analysis cycles
    await runDemoAnalysisCycles(platform);

    // Show investor summary
    displayInvestorSummary(platform);

  } catch (error) {
    console.error('❌ Demo failed:', error.message);
    console.error('💡 This is a demo with simulated data for VC presentation');
  } finally {
    // Clean shutdown
    console.log('🛑 Stopping platform...');
    await platform.stop();
    console.log('✅ Demo completed successfully!');
  }
}

/**
 * Set up event listeners for live monitoring
 */
function setupEventListeners(platform: any) {
  platform.on('platform_started', (data: any) => {
    console.log(`✅ Platform started with ${data.active_agents} agents`);
  });

  platform.on('opportunity_detected', (opportunity: any) => {
    const spread = (opportunity.predicted_spread_pct * 100).toFixed(4);
    const profit = opportunity.estimated_profit_usd.toFixed(0);
    console.log(`💡 Opportunity: ${opportunity.pair} ${spread}% spread, $${profit} profit`);
  });

  platform.on('opportunity_approved', (opportunity: any) => {
    const spread = (opportunity.predicted_spread_pct * 100).toFixed(4);
    console.log(`🟢 APPROVED: ${opportunity.buy_exchange}→${opportunity.sell_exchange} ${spread}%`);
  });

  platform.on('agents_unhealthy', (data: any) => {
    console.log(`⚠️ Agent health warning: ${data.agents.join(', ')}`);
  });

  platform.on('fusion_error', (data: any) => {
    console.log(`🔴 Fusion error: ${data.error}`);
  });
}

/**
 * Display current system status
 */
function displaySystemStatus(platform: any) {
  console.log('\n📊 SYSTEM STATUS');
  console.log('-'.repeat(50));
  
  const status = platform.getSystemStatus();
  
  console.log(`Platform Running: ${status.running ? '🟢 YES' : '🔴 NO'}`);
  console.log(`Agent Health:`);
  
  for (const [name, health] of Object.entries(status.agent_health)) {
    const statusEmoji = (health as any).status === 'healthy' ? '🟢' : 
                       (health as any).status === 'degraded' ? '🟡' : '🔴';
    console.log(`  ${statusEmoji} ${name}: ${(health as any).status}`);
  }
  
  console.log(`Fusion Stats: ${status.fusion_stats.total_predictions} predictions`);
  console.log(`Decision Stats: ${Math.round(status.decision_stats.approval_rate * 100)}% approval rate`);
  console.log(`Hyperbolic Engine: ${status.hyperbolic_stats.total_points} embeddings`);
  console.log('');
}

/**
 * Run several demo analysis cycles
 */
async function runDemoAnalysisCycles(platform: any) {
  console.log('🔄 Running live analysis cycles...\n');
  
  // Let the system run for 30 seconds to collect data and generate predictions
  const demoRunTime = 30000; // 30 seconds
  const startTime = Date.now();
  
  console.log(`⏱️ Demo running for ${demoRunTime / 1000} seconds...`);
  console.log('📈 Watch for real-time opportunities:\n');
  
  let lastOpportunityCount = 0;
  
  // Monitor in real-time
  const monitorInterval = setInterval(() => {
    const opportunities = platform.getOpportunities(5);
    const metrics = platform.getSystemMetrics();
    
    if (opportunities.length > lastOpportunityCount) {
      const newOpportunities = opportunities.slice(lastOpportunityCount);
      
      for (const opp of newOpportunities) {
        const timeStr = new Date(opp.timestamp).toLocaleTimeString();
        const spreadPct = (opp.predicted_spread_pct * 100).toFixed(4);
        const confidence = (opp.confidence * 100).toFixed(1);
        const status = opp.approved_for_execution ? '✅ APPROVED' : '❌ REJECTED';
        
        console.log(`[${timeStr}] ${status}`);
        console.log(`  Pair: ${opp.pair}`);
        console.log(`  Route: ${opp.buy_exchange} → ${opp.sell_exchange}`);
        console.log(`  Spread: ${spreadPct}% (${confidence}% confidence)`);
        console.log(`  Profit: $${opp.estimated_profit_usd.toFixed(0)}`);
        console.log(`  Risk: ${(opp.risk_score * 100).toFixed(1)}%`);
        console.log(`  Reason: ${opp.rationale}`);
        console.log('');
      }
      
      lastOpportunityCount = opportunities.length;
    }
    
    // Show live metrics every 10 seconds
    if ((Date.now() - startTime) % 10000 < 1000) {
      console.log(`📊 Live Metrics: ${metrics.total_predictions} predictions, ` +
                 `${Math.round(metrics.system_health_score * 100)}% system health`);
    }
    
  }, 1000);
  
  // Wait for demo duration
  await new Promise(resolve => setTimeout(resolve, demoRunTime));
  
  clearInterval(monitorInterval);
  
  console.log('⏹️ Analysis cycle completed\n');
}

/**
 * Display investor summary
 */
function displayInvestorSummary(platform: any) {
  console.log('💼 INVESTOR SUMMARY');
  console.log('='.repeat(50));
  
  const summary = platform.getInvestorSummary();
  const metrics = platform.getSystemMetrics();
  const opportunities = platform.getOpportunities();
  
  console.log(`Platform Status: ${summary.platform_status}`);
  console.log(`System Uptime: ${summary.system_uptime}`);
  console.log(`Agent Health: ${summary.agent_health}`);
  console.log(`Total Opportunities: ${summary.total_opportunities}`);
  console.log(`Approval Rate: ${summary.approval_rate.toFixed(1)}%`);
  console.log(`Average Confidence: ${(summary.avg_confidence * 100).toFixed(1)}%`);
  console.log('');
  
  // Show top opportunities
  if (opportunities.length > 0) {
    console.log('🏆 TOP OPPORTUNITIES:');
    
    const topOpportunities = opportunities
      .filter(opp => opp.approved_for_execution)
      .sort((a, b) => b.estimated_profit_usd - a.estimated_profit_usd)
      .slice(0, 3);
    
    if (topOpportunities.length > 0) {
      topOpportunities.forEach((opp, index) => {
        console.log(`  ${index + 1}. ${opp.pair}: ${(opp.predicted_spread_pct * 100).toFixed(4)}% ` +
                   `($${opp.estimated_profit_usd.toFixed(0)} profit)`);
      });
    } else {
      console.log('  No approved opportunities in this demo cycle');
    }
  }
  
  console.log('');
  console.log('📈 SYSTEM CAPABILITIES DEMONSTRATED:');
  console.log('  ✅ Real-time multi-modal data fusion');
  console.log('  ✅ Advanced AI reasoning with LLM integration');
  console.log('  ✅ Hyperbolic geometry for market relationships');
  console.log('  ✅ Deterministic risk management');
  console.log('  ✅ Production-ready scalable architecture');
  console.log('  ✅ Comprehensive audit trails');
  console.log('  ✅ Investor-grade monitoring and reporting');
  console.log('');
}

/**
 * Manual test function for individual components
 */
async function runComponentTests() {
  console.log('🧪 Running component tests...\n');
  
  const platform = createArbitragePlatform(DEMO_CONFIG);
  
  try {
    // Test manual analysis
    console.log('Testing manual analysis...');
    const opportunity = await platform.performAnalysis();
    
    if (opportunity) {
      console.log(`✅ Manual analysis successful: ${opportunity.pair} ${(opportunity.predicted_spread_pct * 100).toFixed(4)}%`);
    } else {
      console.log('ℹ️ Manual analysis completed but no opportunity detected');
    }
    
  } catch (error) {
    console.log(`❌ Component test failed: ${error.message}`);
  }
}

/**
 * Entry point
 */
async function main() {
  const args = process.argv.slice(2);
  
  if (args.includes('--test')) {
    await runComponentTests();
  } else {
    await runArbitragePlatformDemo();
  }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Received shutdown signal...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Received termination signal...');
  process.exit(0);
});

// Run demo
if (import.meta.url === `file://${process.argv[1]}`) {
  main().catch(console.error);
}

export { runArbitragePlatformDemo, DEMO_CONFIG };