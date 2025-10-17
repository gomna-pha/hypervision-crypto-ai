module.exports = {
  apps: [
    {
      name: 'economic-agent',
      script: 'tsx',
      args: 'src/agents/EconomicAgent.ts',
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
      log_file: './logs/economic-agent-combined.log',
      time: true
    },
    {
      name: 'price-agent',
      script: 'tsx',
      args: 'src/agents/PriceAgent.ts',
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
      log_file: './logs/price-agent-combined.log',
      time: true
    },
    {
      name: 'fusion-brain',
      script: 'tsx',
      args: 'src/fusion/FusionBrain.ts',
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
      log_file: './logs/fusion-brain-combined.log',
      time: true
    },
    {
      name: 'main-orchestrator',
      script: 'tsx',
      args: 'src/index.ts start',
      instances: 1,
      exec_mode: 'fork',
      autorestart: true,
      watch: false,
      max_memory_restart: '2G',
      env: {
        NODE_ENV: 'development',
        LOG_LEVEL: 'info'
      },
      error_file: './logs/main-error.log',
      out_file: './logs/main-out.log',
      log_file: './logs/main-combined.log',
      time: true
    }
  ]
};