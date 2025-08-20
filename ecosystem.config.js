module.exports = {
  apps: [{
    name: 'crypto-app',
    script: 'npm',
    args: 'start',
    cwd: '/home/user/webapp',
    env: {
      NODE_ENV: 'development',
      BROWSER: 'none'
    }
  }]
};
