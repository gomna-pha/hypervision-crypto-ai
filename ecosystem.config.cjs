module.exports = {
  apps: [
    {
      name: 'enhanced-dashboard',
      script: 'npx',
      args: 'tsx src/dashboard/enhanced-server.ts',
      cwd: '/home/user/webapp',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '512M',
      env: {
        NODE_ENV: 'development',
        DASHBOARD_PORT: 3000,
        LOG_LEVEL: 'info'
      },
      error_file: './logs/dashboard-error.log',
      out_file: './logs/dashboard-out.log',
      time: true
    },
    {
      name: 'economic-agent',
      script: 'npx',
      args: 'tsx src/agents/EconomicAgent.ts',
      cwd: '/home/user/webapp',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        NODE_ENV: 'development',
        PORT: 3001,
        LOG_LEVEL: 'info'
      },
      error_file: './logs/economic-agent-error.log',
      out_file: './logs/economic-agent-out.log',
      time: true
    },
    {
      name: 'price-agent',
      script: 'npx',
      args: 'tsx src/agents/PriceAgent.ts',
      cwd: '/home/user/webapp',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        NODE_ENV: 'development',
        PORT: 3002,
        LOG_LEVEL: 'info'
      },
      error_file: './logs/price-agent-error.log',
      out_file: './logs/price-agent-out.log',
      time: true
    },
    {
      name: 'sentiment-agent',
      script: 'npx',
      args: 'tsx src/agents/SentimentAgent.ts',
      cwd: '/home/user/webapp',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '500M',
      env: {
        NODE_ENV: 'development',
        PORT: 3003,
        LOG_LEVEL: 'info'
      },
      error_file: './logs/sentiment-agent-error.log',
      out_file: './logs/sentiment-agent-out.log',
      time: true
    },
    {
      name: 'fusion-brain',
      script: 'npx',
      args: 'tsx src/fusion/FusionBrain.ts',
      cwd: '/home/user/webapp',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '1G',
      env: {
        NODE_ENV: 'development',
        LOG_LEVEL: 'info'
      },
      error_file: './logs/fusion-brain-error.log',
      out_file: './logs/fusion-brain-out.log',
      time: true
    }
  ]
};