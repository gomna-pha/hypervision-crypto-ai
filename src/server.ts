/**
 * PRODUCTION SERVER - Real-Time Arbitrage Trading System
 * 
 * This is the main entry point for the Node.js server.
 * Supports WebSocket connections, real-time data processing, and execution.
 */

import { serve } from '@hono/node-server';
import app from './index';
import { realtimeMLService } from './services/realtime-ml-service';

const PORT = parseInt(process.env.PORT || '8787');

// Start server
const server = serve({
  fetch: app.fetch,
  port: PORT,
});

console.log(`
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│     🚀 HYPERVISION CRYPTO AI - REAL-TIME ARBITRAGE          │
│                                                              │
│  Status: PRODUCTION                                          │
│  Port: ${PORT}                                               │
│  Environment: ${process.env.NODE_ENV || 'development'}      │
│                                                              │
└──────────────────────────────────────────────────────────────┘

Initializing real-time services...
`);

// Initialize real-time ML service
const SYMBOLS = (process.env.TRADING_SYMBOLS || 'BTC,ETH,SOL').split(',');

realtimeMLService.start(SYMBOLS)
  .then(() => {
    console.log(`
✅ Real-Time ML Service Started
   Symbols: ${SYMBOLS.join(', ')}
   WebSocket Connections: Active
   ML Pipeline: Running

🌐 Server ready at http://localhost:${PORT}
📊 Dashboard: http://localhost:${PORT}/
🔌 API Health: http://localhost:${PORT}/health
📈 ML Status: http://localhost:${PORT}/api/ml/realtime/status

Press Ctrl+C to stop
`);
  })
  .catch((error) => {
    console.error('❌ Failed to start real-time ML service:', error);
    console.log('⚠️  Server running without real-time data feeds');
    console.log(`🌐 Server ready at http://localhost:${PORT}`);
  });

// Graceful shutdown
process.on('SIGTERM', () => {
  console.log('\n🛑 SIGTERM received, shutting down gracefully...');
  realtimeMLService.stop();
  process.exit(0);
});

process.on('SIGINT', () => {
  console.log('\n🛑 SIGINT received, shutting down gracefully...');
  realtimeMLService.stop();
  process.exit(0);
});

// Handle uncaught errors
process.on('unhandledRejection', (reason, promise) => {
  console.error('❌ Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
  console.error('❌ Uncaught Exception:', error);
  process.exit(1);
});

export default server;
