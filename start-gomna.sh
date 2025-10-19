#!/bin/bash

echo "🍫 Starting Gomna Arbitrage Trades Platform"
echo "=================================================="
echo "Premium Trading Platform with Cream & Cocoa Theme"
echo ""

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing dependencies..."
    npm install
fi

# Check if PM2 is installed
if ! command -v pm2 &> /dev/null; then
    echo "📦 Installing PM2..."
    npm install -g pm2
fi

# Kill any existing processes on port 3000
echo "🧹 Cleaning up port 3000..."
fuser -k 3000/tcp 2>/dev/null || true
pm2 delete all 2>/dev/null || true

# Start the Gomna platform
echo "🎯 Starting Gomna Arbitrage Trades..."
pm2 start gomna-arbitrage-platform.js --name gomna-trades

# Wait for startup
sleep 3

# Show status
echo ""
echo "✅ Gomna Arbitrage Trades is running!"
echo "=================================================="
pm2 status
echo ""
echo "🍫 Access the platform at: http://localhost:3000"
echo "📝 View logs: pm2 logs gomna-trades"
echo "🛑 Stop platform: pm2 stop gomna-trades"
echo ""
echo "🎨 Features:"
echo "  • Premium cream color scheme (95% cream tones)"
echo "  • Cocoa brown accents (0.5% brown tones)"
echo "  • 3D rendered cocoa pod logo"
echo "  • Professional italicized branding"
echo "  • Real-time arbitrage detection"
echo "  • Complete transparency maintained"
echo ""