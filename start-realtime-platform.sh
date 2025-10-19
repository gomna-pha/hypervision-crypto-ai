#!/bin/bash

# LLM Arbitrage Platform - Real-Time Startup Script
# Launches all components with live data feeds

echo "🚀 Starting LLM Arbitrage Platform with Real-Time Data..."
echo "=================================================="

# Clean up any existing processes
echo "Cleaning up existing processes..."
pm2 delete all 2>/dev/null || true
fuser -k 3000/tcp 2>/dev/null || true
fuser -k 3001/tcp 2>/dev/null || true
fuser -k 3002/tcp 2>/dev/null || true
fuser -k 3003/tcp 2>/dev/null || true
fuser -k 3004/tcp 2>/dev/null || true
fuser -k 3005/tcp 2>/dev/null || true

# Build the project
echo "Building project..."
npm run build

# Start individual agent services (if not using central aggregator)
# These are now optional since RealTimeDataAggregator handles all data collection
# echo "Starting agent services..."
# npx tsx src/agents/EconomicAgent.ts &
# npx tsx src/agents/SentimentAgent.ts &
# npx tsx src/agents/PriceAgent.ts &
# npx tsx src/agents/VolumeFlowAgent.ts &
# npx tsx src/agents/TradeFlowAgent.ts &

# Start the professional dashboard with real-time components
echo "Starting Professional Dashboard with Real-Time Data..."
pm2 start src/dashboard/professional-dashboard.ts --name "llm-arbitrage-dashboard" --interpreter "tsx"

# Wait for services to start
sleep 3

# Check status
echo ""
echo "✅ Platform Status:"
echo "==================="
pm2 list

echo ""
echo "📊 Access Points:"
echo "=================="
echo "Professional Dashboard: http://localhost:3000"
echo ""
echo "Real-Time Features:"
echo "• WebSocket connections to Binance, Coinbase, Kraken"
echo "• Live economic data from multiple sources"
echo "• Real-time sentiment analysis"
echo "• Cross-exchange arbitrage monitoring"
echo "• LLM-powered strategy generation every 5 seconds"
echo ""
echo "Agent Signals Updated:"
echo "• Economic: Live GDP, inflation, rates data"
echo "• Sentiment: Live social media & news sentiment"
echo "• Microstructure: Live order book analysis"
echo "• Cross-Exchange: Live spread monitoring"
echo ""
echo "All agents are now providing REAL-TIME data to the LLM"
echo "for continuous arbitrage opportunity detection!"
echo ""
echo "Monitor logs with: pm2 logs llm-arbitrage-dashboard"
echo "Stop platform with: pm2 stop all"