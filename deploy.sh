#!/bin/bash

# HyperVision AI - GitHub Pages Deployment Script
echo "🚀 Deploying HyperVision AI Trading Platform to GitHub Pages..."

# Build the application
echo "📦 Building application..."
npm run build

# Install gh-pages if not present
if ! npm list gh-pages --depth=0 > /dev/null 2>&1; then
    echo "📥 Installing gh-pages..."
    npm install --save-dev gh-pages
fi

# Deploy to GitHub Pages
echo "🌐 Deploying to GitHub Pages..."
npx gh-pages -d build -m "deploy: Update HyperVision AI Trading Platform"

echo "✅ Deployment complete!"
echo "🔗 Your site will be available at: https://gomna-pha.github.io/hypervision-crypto-ai/"