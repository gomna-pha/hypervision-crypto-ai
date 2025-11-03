// PM2 Configuration for Trading Intelligence Platform
// Production-Ready with Live Data Feeds

module.exports = {
  apps: [
    {
      name: 'trading-intelligence',
      script: 'npx',
      // Pass all API keys as wrangler bindings
      // Add your API keys to .dev.vars file, they'll be loaded automatically
      // Or uncomment and add them here directly for PM2
      args: `wrangler pages dev dist --d1=webapp-production --local --ip 0.0.0.0 --port 3000 \
--binding GEMINI_API_KEY=AIzaSyCG4nVE1101YRsNh0OSq94VoHQe-CDv4og \
--binding FRED_API_KEY=a436d248d2c5b81f11f9410c067a1eb6`,
      
      // Uncomment below to add more API keys via PM2 (or use .dev.vars)
      // args: `wrangler pages dev dist --d1=webapp-production --local --ip 0.0.0.0 --port 3000 \
      // --binding GEMINI_API_KEY=AIzaSyCG4nVE1101YRsNh0OSq94VoHQe-CDv4og \
      // --binding COINGECKO_API_KEY=CG-your-key-here \
      // --binding FRED_API_KEY=your-fred-key-here \
      // --binding SERPAPI_KEY=your-serpapi-key-here`,
      
      env: {
        NODE_ENV: 'development',
        PORT: 3000
      },
      watch: false,
      instances: 1,
      exec_mode: 'fork'
    }
  ]
}
