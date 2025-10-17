import express from 'express';
import http from 'http';
import path from 'path';
import axios from 'axios';
import Logger from '../utils/logger';

const app = express();
const server = http.createServer(app);
const logger = Logger.getInstance('Dashboard');

const PORT = process.env.DASHBOARD_PORT || 3000;

// Serve static files
app.use(express.static(path.join(__dirname, 'public')));
app.use(express.json());

// API endpoints to fetch agent data
app.get('/api/agents/status', async (req, res) => {
  const agents = [
    { name: 'Economic Agent', port: 3001, endpoint: '/health' },
    { name: 'Price Agent', port: 3002, endpoint: '/health' },
  ];

  const statuses = await Promise.all(
    agents.map(async (agent) => {
      try {
        const response = await axios.get(`http://localhost:${agent.port}${agent.endpoint}`, {
          timeout: 2000
        });
        return {
          name: agent.name,
          status: response.data.status || 'unknown',
          running: response.data.running || false,
          lastUpdate: response.data.lastUpdate,
        };
      } catch (error) {
        return {
          name: agent.name,
          status: 'offline',
          running: false,
          lastUpdate: null,
        };
      }
    })
  );

  res.json(statuses);
});

app.get('/api/agents/economic/latest', async (req, res) => {
  try {
    const response = await axios.get('http://localhost:3001/agents/economicagent/latest', {
      timeout: 2000
    });
    res.json(response.data);
  } catch (error) {
    res.status(503).json({ error: 'Economic Agent not available' });
  }
});

app.get('/api/agents/price/latest', async (req, res) => {
  try {
    const response = await axios.get('http://localhost:3002/agents/priceagent/latest', {
      timeout: 2000
    });
    res.json(response.data);
  } catch (error) {
    res.status(503).json({ error: 'Price Agent not available' });
  }
});

// Serve the main dashboard HTML
app.get('/', (req, res) => {
  res.send(`
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>LLM Arbitrage Platform Dashboard</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://cdn.jsdelivr.net/npm/@fortawesome/fontawesome-free@6.4.0/css/all.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        @keyframes pulse-green {
            0%, 100% { background-color: rgb(34, 197, 94); }
            50% { background-color: rgb(74, 222, 128); }
        }
        @keyframes pulse-red {
            0%, 100% { background-color: rgb(239, 68, 68); }
            50% { background-color: rgb(248, 113, 113); }
        }
        .status-online { animation: pulse-green 2s infinite; }
        .status-offline { animation: pulse-red 2s infinite; }
    </style>
</head>
<body class="bg-gray-900 text-gray-100">
    <div class="container mx-auto p-6">
        <!-- Header -->
        <div class="bg-gray-800 rounded-lg p-6 mb-6 border border-gray-700">
            <div class="flex items-center justify-between">
                <div class="flex items-center">
                    <i class="fas fa-chart-line text-4xl text-blue-500 mr-4"></i>
                    <div>
                        <h1 class="text-3xl font-bold text-white">LLM Arbitrage Platform</h1>
                        <p class="text-gray-400">Real-time Agent-Based Trading Intelligence</p>
                    </div>
                </div>
                <div class="text-right">
                    <p class="text-sm text-gray-500">System Time</p>
                    <p class="text-xl font-mono" id="system-time">--:--:--</p>
                </div>
            </div>
        </div>

        <!-- Agent Status Grid -->
        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
            <div id="agent-status-container" class="col-span-3 grid grid-cols-1 md:grid-cols-3 gap-4">
                <!-- Agent status cards will be inserted here -->
            </div>
        </div>

        <!-- Main Content Grid -->
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <!-- Economic Data Panel -->
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h2 class="text-xl font-bold mb-4 flex items-center">
                    <i class="fas fa-globe-americas mr-2 text-green-500"></i>
                    Economic Indicators
                </h2>
                <div id="economic-data" class="space-y-3">
                    <div class="flex justify-center items-center h-32">
                        <i class="fas fa-spinner fa-spin text-4xl text-gray-600"></i>
                    </div>
                </div>
            </div>

            <!-- Price Data Panel -->
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h2 class="text-xl font-bold mb-4 flex items-center">
                    <i class="fas fa-coins mr-2 text-yellow-500"></i>
                    Price & Market Data
                </h2>
                <div id="price-data" class="space-y-3">
                    <div class="flex justify-center items-center h-32">
                        <i class="fas fa-spinner fa-spin text-4xl text-gray-600"></i>
                    </div>
                </div>
            </div>

            <!-- Fusion Brain Status -->
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h2 class="text-xl font-bold mb-4 flex items-center">
                    <i class="fas fa-brain mr-2 text-purple-500"></i>
                    Fusion Brain Status
                </h2>
                <div id="fusion-status" class="space-y-3">
                    <div class="bg-gray-700 p-3 rounded">
                        <div class="flex justify-between items-center">
                            <span class="text-gray-400">LLM Provider</span>
                            <span class="font-mono">Anthropic Claude</span>
                        </div>
                    </div>
                    <div class="bg-gray-700 p-3 rounded">
                        <div class="flex justify-between items-center">
                            <span class="text-gray-400">Hyperbolic Model</span>
                            <span class="font-mono">Poincaré Ball</span>
                        </div>
                    </div>
                    <div class="bg-gray-700 p-3 rounded">
                        <div class="flex justify-between items-center">
                            <span class="text-gray-400">Embedding Dim</span>
                            <span class="font-mono">128</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Arbitrage Opportunities -->
            <div class="bg-gray-800 rounded-lg p-6 border border-gray-700">
                <h2 class="text-xl font-bold mb-4 flex items-center">
                    <i class="fas fa-exchange-alt mr-2 text-blue-500"></i>
                    Arbitrage Opportunities
                </h2>
                <div id="arbitrage-ops" class="space-y-3">
                    <div class="bg-gray-700 p-4 rounded border-l-4 border-green-500">
                        <div class="flex justify-between items-center mb-2">
                            <span class="font-bold text-green-400">BTC-USDT</span>
                            <span class="text-sm text-gray-400">Mock Data</span>
                        </div>
                        <div class="grid grid-cols-2 gap-2 text-sm">
                            <div>Buy: Binance</div>
                            <div>Sell: Coinbase</div>
                            <div>Spread: 0.12%</div>
                            <div>Confidence: 85%</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- System Metrics -->
        <div class="mt-6 bg-gray-800 rounded-lg p-6 border border-gray-700">
            <h2 class="text-xl font-bold mb-4 flex items-center">
                <i class="fas fa-tachometer-alt mr-2 text-red-500"></i>
                System Performance
            </h2>
            <div class="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div class="bg-gray-700 p-4 rounded text-center">
                    <p class="text-gray-400 text-sm">CPU Usage</p>
                    <p class="text-2xl font-bold">12%</p>
                </div>
                <div class="bg-gray-700 p-4 rounded text-center">
                    <p class="text-gray-400 text-sm">Memory</p>
                    <p class="text-2xl font-bold">384MB</p>
                </div>
                <div class="bg-gray-700 p-4 rounded text-center">
                    <p class="text-gray-400 text-sm">Latency</p>
                    <p class="text-2xl font-bold">45ms</p>
                </div>
                <div class="bg-gray-700 p-4 rounded text-center">
                    <p class="text-gray-400 text-sm">Messages/sec</p>
                    <p class="text-2xl font-bold">127</p>
                </div>
            </div>
        </div>
    </div>

    <script>
        // Update system time
        function updateTime() {
            const now = new Date();
            document.getElementById('system-time').textContent = now.toLocaleTimeString();
        }
        setInterval(updateTime, 1000);
        updateTime();

        // Fetch and display agent status
        async function updateAgentStatus() {
            try {
                const response = await fetch('/api/agents/status');
                const agents = await response.json();
                
                const container = document.getElementById('agent-status-container');
                container.innerHTML = agents.map(agent => \`
                    <div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
                        <div class="flex items-center justify-between mb-3">
                            <h3 class="font-bold">\${agent.name}</h3>
                            <div class="\${agent.status === 'healthy' ? 'status-online' : 'status-offline'} w-3 h-3 rounded-full"></div>
                        </div>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between">
                                <span class="text-gray-400">Status</span>
                                <span class="\${agent.status === 'healthy' ? 'text-green-400' : 'text-red-400'}">\${agent.status}</span>
                            </div>
                            <div class="flex justify-between">
                                <span class="text-gray-400">Running</span>
                                <span>\${agent.running ? 'Yes' : 'No'}</span>
                            </div>
                        </div>
                    </div>
                \`).join('');
            } catch (error) {
                console.error('Failed to fetch agent status:', error);
            }
        }

        // Fetch economic data
        async function updateEconomicData() {
            try {
                const response = await fetch('/api/agents/economic/latest');
                const data = await response.json();
                
                const container = document.getElementById('economic-data');
                if (data.signals) {
                    container.innerHTML = Object.entries(data.signals).map(([key, value]) => \`
                        <div class="bg-gray-700 p-3 rounded">
                            <div class="flex justify-between items-center">
                                <span class="text-gray-400">\${key}</span>
                                <span class="font-mono">\${typeof value === 'number' ? value.toFixed(2) : value}</span>
                            </div>
                        </div>
                    \`).join('');
                }
            } catch (error) {
                document.getElementById('economic-data').innerHTML = '<p class="text-gray-500 text-center">No data available</p>';
            }
        }

        // Fetch price data
        async function updatePriceData() {
            try {
                const response = await fetch('/api/agents/price/latest');
                const data = await response.json();
                
                const container = document.getElementById('price-data');
                container.innerHTML = \`
                    <div class="bg-gray-700 p-3 rounded">
                        <div class="flex justify-between items-center">
                            <span class="text-gray-400">Exchange</span>
                            <span class="font-mono">\${data.exchange || 'N/A'}</span>
                        </div>
                    </div>
                    <div class="bg-gray-700 p-3 rounded">
                        <div class="flex justify-between items-center">
                            <span class="text-gray-400">Mid Price</span>
                            <span class="font-mono">$\${data.mid_price ? data.mid_price.toFixed(2) : 'N/A'}</span>
                        </div>
                    </div>
                    <div class="bg-gray-700 p-3 rounded">
                        <div class="flex justify-between items-center">
                            <span class="text-gray-400">Spread</span>
                            <span class="font-mono">$\${data.best_ask && data.best_bid ? (data.best_ask - data.best_bid).toFixed(2) : 'N/A'}</span>
                        </div>
                    </div>
                \`;
            } catch (error) {
                document.getElementById('price-data').innerHTML = '<p class="text-gray-500 text-center">No data available</p>';
            }
        }

        // Update all data
        function updateDashboard() {
            updateAgentStatus();
            updateEconomicData();
            updatePriceData();
        }

        // Initial load and periodic updates
        updateDashboard();
        setInterval(updateDashboard, 5000);
    </script>
</body>
</html>
  `);
});

server.listen(PORT, '0.0.0.0', () => {
  logger.info(`Dashboard server running on port ${PORT}`);
  console.log(`\n🚀 Dashboard available at http://localhost:${PORT}\n`);
});

// Graceful shutdown
process.on('SIGINT', () => {
  logger.info('Shutting down dashboard server...');
  server.close(() => {
    process.exit(0);
  });
});