/**
 * Server Runner - Node.js HTTP Server for Arbitrage Platform
 * Runs the Hono app on Node.js with proper TypeScript support
 */

import { serve } from '@hono/node-server';
import app from './arbitrage/server.js';

const port = process.env.PORT || 4000;

console.log('🚀 Starting Agent-Based LLM Arbitrage Platform API Server...');
console.log(`🌐 Server will be available at: http://localhost:${port}`);
console.log('📊 API Documentation: http://localhost:${port}/api/docs');
console.log('🏥 Health Check: http://localhost:${port}/health');
console.log('🎯 Demo Endpoint: POST http://localhost:${port}/api/demo/start');
console.log('');

serve({
  fetch: app.fetch,
  port: port
}, (info) => {
  console.log(`✅ Server is running on port ${info.port}`);
  console.log('');
  console.log('🔗 Quick Start URLs:');
  console.log(`   Health Check: http://localhost:${port}/health`);
  console.log(`   Platform Status: http://localhost:${port}/api/platform/status`);
  console.log(`   Start Demo: curl -X POST http://localhost:${port}/api/demo/start`);
  console.log(`   Get Opportunities: http://localhost:${port}/api/opportunities`);
  console.log('');
  console.log('Press Ctrl+C to stop the server');
});

// Graceful shutdown
process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down server...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Terminating server...');
  process.exit(0);
});