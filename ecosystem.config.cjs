module.exports = {
  apps: [
    {
      name: 'trading-intelligence',
      script: 'npx',
      args: 'wrangler pages dev dist --d1=webapp-production --local --ip 0.0.0.0 --port 3000 --binding GEMINI_API_KEY=AIzaSyCG4nVE1101YRsNh0OSq94VoHQe-CDv4og',
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
