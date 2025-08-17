module.exports = {
  apps: [{
    name: "hypervision-crypto",
    script: "npm",
    args: "start",
    env: {
      PORT: 3000,
      BROWSER: "none",
      HOST: "0.0.0.0"
    },
    watch: false,
    autorestart: true,
    max_memory_restart: "500M"
  }]
};