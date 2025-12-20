/**
 * ENHANCED ANALYTICS - Weekly Observations & Hyperbolic Visualization
 */

import { Hono } from 'hono';

export function registerAnalyticsRoute(app: Hono) {
  app.get('/analytics', (c) => {
    return c.html(`
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>HyperVision Analytics - Weekly & Hyperbolic Visualization</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
  <script src="https://cdn.jsdelivr.net/npm/d3@7"></script>
  <style>
    body {
      background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
      font-family: 'Inter', sans-serif;
    }
    .section-card {
      background: rgba(30, 41, 59, 0.8);
      border: 1px solid rgba(148, 163, 184, 0.2);
      backdrop-filter: blur(10px);
    }
    .metric-badge {
      background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
      border: 1px solid rgba(59, 130, 246, 0.3);
      padding: 0.75rem;
      border-radius: 0.5rem;
    }
    #poincare-disk {
      background: radial-gradient(circle, #1e293b 0%, #0f172a 100%);
      border: 2px solid rgba(59, 130, 246, 0.3);
    }
  </style>
</head>
<body class="text-gray-100 p-4">

  <!-- Header -->
  <div class="max-w-7xl mx-auto mb-6">
    <div class="section-card rounded-lg p-6 shadow-2xl">
      <div class="flex items-center justify-between">
        <div>
          <h1 class="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-600">
            HyperVision Analytics
          </h1>
          <p class="text-gray-400 mt-1">Weekly Observations & Hyperbolic Signal Visualization</p>
        </div>
        <div class="flex space-x-4">
          <div class="text-right">
            <div class="text-sm text-gray-400">Observation Period</div>
            <select id="time-period" class="bg-gray-700 text-white px-4 py-2 rounded-lg">
              <option value="daily">Daily (24h)</option>
              <option value="weekly" selected>Weekly (7d)</option>
              <option value="monthly">Monthly (30d)</option>
            </select>
          </div>
          <div class="text-right">
            <div class="text-sm text-gray-400">Last Update</div>
            <div id="last-update" class="text-blue-400 font-semibold">--</div>
          </div>
        </div>
      </div>
    </div>
  </div>

  <div class="max-w-7xl mx-auto space-y-4">

    <!-- 1. HYPERBOLIC EMBEDDING VISUALIZATION (Poincaré Disk) -->
    <div class="section-card rounded-lg p-6 shadow-xl">
      <div class="flex items-center mb-4">
        <i class="fas fa-project-diagram text-orange-400 text-2xl mr-3"></i>
        <h2 class="text-2xl font-bold">Hyperbolic Signal Space (Poincaré Disk)</h2>
        <div class="ml-auto text-sm text-gray-400">
          Radial Distance = Signal Strength | Angular Distance = Regime Similarity
        </div>
      </div>
      <div class="grid grid-cols-3 gap-6">
        <!-- Poincaré Disk Visualization -->
        <div class="col-span-2">
          <canvas id="poincare-disk" width="600" height="600"></canvas>
        </div>
        
        <!-- Signal Legend & Metrics -->
        <div class="space-y-4">
          <div class="metric-badge">
            <div class="text-xs text-gray-400 mb-2">SIGNAL LEGEND</div>
            <div class="space-y-2 text-sm">
              <div class="flex items-center">
                <div class="w-4 h-4 rounded-full bg-blue-500 mr-2"></div>
                <span>Economic Agent</span>
              </div>
              <div class="flex items-center">
                <div class="w-4 h-4 rounded-full bg-purple-500 mr-2"></div>
                <span>Sentiment Agent</span>
              </div>
              <div class="flex items-center">
                <div class="w-4 h-4 rounded-full bg-green-500 mr-2"></div>
                <span>Cross-Exchange</span>
              </div>
              <div class="flex items-center">
                <div class="w-4 h-4 rounded-full bg-yellow-500 mr-2"></div>
                <span>On-Chain Agent</span>
              </div>
              <div class="flex items-center">
                <div class="w-4 h-4 rounded-full bg-red-500 mr-2"></div>
                <span>CNN Pattern</span>
              </div>
            </div>
          </div>
          
          <div class="metric-badge">
            <div class="text-xs text-gray-400 mb-2">REGIME ZONES</div>
            <div class="space-y-2 text-xs">
              <div class="flex justify-between">
                <span class="text-red-400">Crisis/Stress</span>
                <span id="regime-crisis">0%</span>
              </div>
              <div class="flex justify-between">
                <span class="text-orange-400">Defensive</span>
                <span id="regime-defensive">0%</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Neutral</span>
                <span id="regime-neutral">0%</span>
              </div>
              <div class="flex justify-between">
                <span class="text-green-400">Risk-On</span>
                <span id="regime-riskon">0%</span>
              </div>
              <div class="flex justify-between">
                <span class="text-blue-400">High Conviction</span>
                <span id="regime-conviction">0%</span>
              </div>
            </div>
          </div>
          
          <div class="metric-badge">
            <div class="text-xs text-gray-400 mb-2">HYPERBOLIC METRICS</div>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-gray-400">Avg Distance</span>
                <span id="hyper-avg-dist" class="text-blue-400">--</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Signal Clustering</span>
                <span id="hyper-clustering" class="text-green-400">--</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Regime Stability</span>
                <span id="hyper-stability" class="text-yellow-400">--</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 2. WEEKLY PERFORMANCE ANALYSIS -->
    <div class="grid grid-cols-2 gap-4">
      <div class="section-card rounded-lg p-6 shadow-xl">
        <div class="flex items-center mb-4">
          <i class="fas fa-chart-bar text-green-400 text-2xl mr-3"></i>
          <h2 class="text-xl font-bold">Weekly Returns & Sharpe Ratio</h2>
        </div>
        <canvas id="weekly-performance-chart" height="200"></canvas>
        <div class="grid grid-cols-3 gap-2 mt-4 text-sm">
          <div class="metric-badge text-center">
            <div class="text-gray-400 text-xs">Week Return</div>
            <div id="week-return" class="text-lg font-bold text-green-400">--</div>
          </div>
          <div class="metric-badge text-center">
            <div class="text-gray-400 text-xs">Sharpe Ratio</div>
            <div id="week-sharpe" class="text-lg font-bold text-blue-400">--</div>
          </div>
          <div class="metric-badge text-center">
            <div class="text-gray-400 text-xs">Max Drawdown</div>
            <div id="week-drawdown" class="text-lg font-bold text-red-400">--</div>
          </div>
        </div>
      </div>
      
      <div class="section-card rounded-lg p-6 shadow-xl">
        <div class="flex items-center mb-4">
          <i class="fas fa-exclamation-triangle text-yellow-400 text-2xl mr-3"></i>
          <h2 class="text-xl font-bold">Volatility Factor Analysis</h2>
        </div>
        <canvas id="volatility-factors-chart" height="200"></canvas>
        <div class="mt-4 space-y-2 text-sm">
          <div class="flex justify-between">
            <span class="text-gray-400">Dominant Factor:</span>
            <span id="vol-dominant-factor" class="text-yellow-400 font-semibold">--</span>
          </div>
          <div class="flex justify-between">
            <span class="text-gray-400">Contribution:</span>
            <span id="vol-contribution" class="text-yellow-400 font-semibold">--</span>
          </div>
        </div>
      </div>
    </div>

    <!-- 3. REGIME-CONDITIONAL PERFORMANCE -->
    <div class="section-card rounded-lg p-6 shadow-xl">
      <div class="flex items-center mb-4">
        <i class="fas fa-layer-group text-cyan-400 text-2xl mr-3"></i>
        <h2 class="text-2xl font-bold">Regime-Conditional Performance Analysis</h2>
      </div>
      <canvas id="regime-performance-chart" height="150"></canvas>
      <div class="grid grid-cols-5 gap-2 mt-4">
        <div class="metric-badge text-center">
          <div class="text-xs text-red-400 mb-1">Crisis/Stress</div>
          <div id="perf-crisis" class="text-lg font-bold">--</div>
          <div id="vol-crisis" class="text-xs text-gray-400">Vol: --</div>
        </div>
        <div class="metric-badge text-center">
          <div class="text-xs text-orange-400 mb-1">Defensive</div>
          <div id="perf-defensive" class="text-lg font-bold">--</div>
          <div id="vol-defensive" class="text-xs text-gray-400">Vol: --</div>
        </div>
        <div class="metric-badge text-center">
          <div class="text-xs text-gray-400 mb-1">Neutral</div>
          <div id="perf-neutral" class="text-lg font-bold">--</div>
          <div id="vol-neutral" class="text-xs text-gray-400">Vol: --</div>
        </div>
        <div class="metric-badge text-center">
          <div class="text-xs text-green-400 mb-1">Risk-On</div>
          <div id="perf-riskon" class="text-lg font-bold">--</div>
          <div id="vol-riskon" class="text-xs text-gray-400">Vol: --</div>
        </div>
        <div class="metric-badge text-center">
          <div class="text-xs text-blue-400 mb-1">High Conviction</div>
          <div id="perf-conviction" class="text-lg font-bold">--</div>
          <div id="vol-conviction" class="text-xs text-gray-400">Vol: --</div>
        </div>
      </div>
    </div>

    <!-- 4. PORTFOLIO OPTIMIZATION UNDER RISK AVERSION -->
    <div class="section-card rounded-lg p-6 shadow-xl">
      <div class="flex items-center mb-4">
        <i class="fas fa-sliders-h text-indigo-400 text-2xl mr-3"></i>
        <h2 class="text-2xl font-bold">Portfolio Optimization - Risk Aversion Sensitivity</h2>
      </div>
      <div class="grid grid-cols-3 gap-6">
        <div class="col-span-2">
          <canvas id="risk-aversion-chart" height="250"></canvas>
        </div>
        <div class="space-y-3">
          <div class="metric-badge">
            <div class="text-xs text-gray-400 mb-2">RISK AVERSION LEVEL</div>
            <input type="range" id="risk-aversion-slider" min="1" max="10" value="5" 
                   class="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer">
            <div class="flex justify-between text-xs text-gray-400 mt-1">
              <span>Aggressive (1)</span>
              <span id="risk-aversion-value" class="text-blue-400 font-semibold">5</span>
              <span>Conservative (10)</span>
            </div>
          </div>
          
          <div class="metric-badge">
            <div class="text-xs text-gray-400 mb-2">OPTIMIZED METRICS</div>
            <div class="space-y-2 text-sm">
              <div class="flex justify-between">
                <span class="text-gray-400">Expected Return</span>
                <span id="opt-return" class="text-green-400">--</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Portfolio Vol</span>
                <span id="opt-vol" class="text-yellow-400">--</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Sharpe Ratio</span>
                <span id="opt-sharpe" class="text-blue-400">--</span>
              </div>
              <div class="flex justify-between">
                <span class="text-gray-400">Max Position</span>
                <span id="opt-max-position" class="text-purple-400">--</span>
              </div>
            </div>
          </div>
          
          <div class="metric-badge">
            <div class="text-xs text-gray-400 mb-2">ALLOCATION WEIGHTS</div>
            <canvas id="allocation-pie-chart" height="150"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- 5. VOLATILITY DRIVER DECOMPOSITION -->
    <div class="grid grid-cols-2 gap-4">
      <div class="section-card rounded-lg p-6 shadow-xl">
        <div class="flex items-center mb-4">
          <i class="fas fa-wave-square text-red-400 text-2xl mr-3"></i>
          <h2 class="text-xl font-bold">Volatility Drivers (Weekly)</h2>
        </div>
        <div class="space-y-3">
          <div class="metric-badge">
            <div class="text-xs text-gray-400 mb-1">Market Microstructure (Order Flow)</div>
            <div class="flex items-center">
              <div class="flex-1 bg-gray-700 rounded-full h-2 mr-2">
                <div id="vol-driver-microstructure" class="bg-red-400 h-2 rounded-full" style="width: 0%"></div>
              </div>
              <span id="vol-driver-microstructure-pct" class="text-sm text-red-400">0%</span>
            </div>
          </div>
          
          <div class="metric-badge">
            <div class="text-xs text-gray-400 mb-1">Funding Rate Divergence</div>
            <div class="flex items-center">
              <div class="flex-1 bg-gray-700 rounded-full h-2 mr-2">
                <div id="vol-driver-funding" class="bg-orange-400 h-2 rounded-full" style="width: 0%"></div>
              </div>
              <span id="vol-driver-funding-pct" class="text-sm text-orange-400">0%</span>
            </div>
          </div>
          
          <div class="metric-badge">
            <div class="text-xs text-gray-400 mb-1">Macro Economic Shocks</div>
            <div class="flex items-center">
              <div class="flex-1 bg-gray-700 rounded-full h-2 mr-2">
                <div id="vol-driver-macro" class="bg-yellow-400 h-2 rounded-full" style="width: 0%"></div>
              </div>
              <span id="vol-driver-macro-pct" class="text-sm text-yellow-400">0%</span>
            </div>
          </div>
          
          <div class="metric-badge">
            <div class="text-xs text-gray-400 mb-1">On-Chain Activity Spikes</div>
            <div class="flex items-center">
              <div class="flex-1 bg-gray-700 rounded-full h-2 mr-2">
                <div id="vol-driver-onchain" class="bg-green-400 h-2 rounded-full" style="width: 0%"></div>
              </div>
              <span id="vol-driver-onchain-pct" class="text-sm text-green-400">0%</span>
            </div>
          </div>
          
          <div class="metric-badge">
            <div class="text-xs text-gray-400 mb-1">Sentiment Swings (Fear/Greed)</div>
            <div class="flex items-center">
              <div class="flex-1 bg-gray-700 rounded-full h-2 mr-2">
                <div id="vol-driver-sentiment" class="bg-blue-400 h-2 rounded-full" style="width: 0%"></div>
              </div>
              <span id="vol-driver-sentiment-pct" class="text-sm text-blue-400">0%</span>
            </div>
          </div>
        </div>
      </div>
      
      <div class="section-card rounded-lg p-6 shadow-xl">
        <div class="flex items-center mb-4">
          <i class="fas fa-clock text-purple-400 text-2xl mr-3"></i>
          <h2 class="text-xl font-bold">High Volatility Periods Analysis</h2>
        </div>
        <div id="high-vol-periods" class="space-y-2 text-sm max-h-64 overflow-y-auto">
          <!-- Dynamic content -->
        </div>
      </div>
    </div>

  </div>

  <script>
    let charts = {};
    let poincareCanvas, poincareCtx;
    
    // Initialize Poincaré disk visualization
    function initPoincareDisksignals = [
      { name: 'Economic', x: 0.3, y: 0.2, color: '#3b82f6', strength: 0.6 },
      { name: 'Sentiment', x: -0.4, y: 0.3, color: '#a855f7', strength: 0.5 },
      { name: 'Cross-Exchange', x: 0.1, y: -0.5, color: '#22c55e', strength: 0.8 },
      { name: 'On-Chain', x: -0.2, y: -0.3, color: '#eab308', strength: 0.7 },
      { name: 'CNN Pattern', x: 0.5, y: -0.1, color: '#ef4444', strength: 0.9 }
    ];
    
    const canvas = document.getElementById('poincare-disk');
    const ctx = canvas.getContext('2d');
    const centerX = canvas.width / 2;
    const centerY = canvas.height / 2;
    const radius = Math.min(centerX, centerY) - 40;
    
    // Draw Poincaré disk boundary
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.5)';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
    ctx.stroke();
    
    // Draw regime zones (concentric circles)
    const regimeZones = [
      { name: 'Crisis', radius: 0.3, color: '#ef4444' },
      { name: 'Defensive', radius: 0.5, color: '#f97316' },
      { name: 'Neutral', radius: 0.7, color: '#6b7280' },
      { name: 'Risk-On', radius: 0.85, color: '#22c55e' },
      { name: 'High Conviction', radius: 1.0, color: '#3b82f6' }
    ];
    
    regimeZones.forEach(zone => {
      ctx.strokeStyle = zone.color + '20';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(centerX, centerY, radius * zone.radius, 0, 2 * Math.PI);
      ctx.stroke();
      
      // Label
      ctx.fillStyle = zone.color + '60';
      ctx.font = '10px Inter';
      ctx.fillText(zone.name, centerX + radius * zone.radius + 5, centerY);
    });
    
    // Draw signals
    signals.forEach(signal => {
      const x = centerX + signal.x * radius;
      const y = centerY + signal.y * radius;
      const size = 8 + signal.strength * 12;
      
      // Glow effect
      ctx.shadowColor = signal.color;
      ctx.shadowBlur = 15;
      
      // Draw signal point
      ctx.fillStyle = signal.color;
      ctx.beginPath();
      ctx.arc(x, y, size, 0, 2 * Math.PI);
      ctx.fill();
      
      ctx.shadowBlur = 0;
      
      // Label
      ctx.fillStyle = '#fff';
      ctx.font = '12px Inter';
      ctx.fillText(signal.name, x + size + 5, y + 4);
    });
  }
  
  // Initialize all charts
  function initCharts() {
    // Weekly Performance Chart
    charts.weeklyPerformance = new Chart(document.getElementById('weekly-performance-chart'), {
      type: 'bar',
      data: {
        labels: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
        datasets: [{
          label: 'Daily Return (%)',
          data: [1.2, -0.5, 2.3, 0.8, -1.1, 1.5, 0.9],
          backgroundColor: 'rgba(34, 197, 94, 0.6)',
          borderColor: 'rgba(34, 197, 94, 1)',
          borderWidth: 1
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          title: { display: false }
        },
        scales: {
          y: {
            ticks: { color: '#9ca3af' },
            grid: { color: 'rgba(148, 163, 184, 0.1)' }
          },
          x: {
            ticks: { color: '#9ca3af' },
            grid: { display: false }
          }
        }
      }
    });
    
    // Volatility Factors Chart
    charts.volatilityFactors = new Chart(document.getElementById('volatility-factors-chart'), {
      type: 'doughnut',
      data: {
        labels: ['Order Flow', 'Funding Rate', 'Macro Shocks', 'On-Chain', 'Sentiment'],
        datasets: [{
          data: [35, 25, 20, 12, 8],
          backgroundColor: [
            'rgba(239, 68, 68, 0.8)',
            'rgba(249, 115, 22, 0.8)',
            'rgba(234, 179, 8, 0.8)',
            'rgba(34, 197, 94, 0.8)',
            'rgba(59, 130, 246, 0.8)'
          ]
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#9ca3af', font: { size: 10 } }
          }
        }
      }
    });
    
    // Regime Performance Chart
    charts.regimePerformance = new Chart(document.getElementById('regime-performance-chart'), {
      type: 'line',
      data: {
        labels: ['Week 1', 'Week 2', 'Week 3', 'Week 4'],
        datasets: [
          {
            label: 'Crisis/Stress',
            data: [-2.5, -1.8, -3.2, -1.5],
            borderColor: '#ef4444',
            backgroundColor: 'rgba(239, 68, 68, 0.1)',
            tension: 0.3
          },
          {
            label: 'Defensive',
            data: [0.5, 0.8, 0.3, 0.6],
            borderColor: '#f97316',
            backgroundColor: 'rgba(249, 115, 22, 0.1)',
            tension: 0.3
          },
          {
            label: 'Neutral',
            data: [1.2, 0.9, 1.5, 1.1],
            borderColor: '#6b7280',
            backgroundColor: 'rgba(107, 114, 128, 0.1)',
            tension: 0.3
          },
          {
            label: 'Risk-On',
            data: [2.8, 3.2, 2.5, 3.5],
            borderColor: '#22c55e',
            backgroundColor: 'rgba(34, 197, 94, 0.1)',
            tension: 0.3
          },
          {
            label: 'High Conviction',
            data: [4.2, 3.8, 4.5, 4.1],
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.3
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { color: '#9ca3af', font: { size: 10 } }
          }
        },
        scales: {
          y: {
            ticks: { color: '#9ca3af' },
            grid: { color: 'rgba(148, 163, 184, 0.1)' }
          },
          x: {
            ticks: { color: '#9ca3af' },
            grid: { display: false }
          }
        }
      }
    });
    
    // Risk Aversion Chart (Efficient Frontier)
    charts.riskAversion = new Chart(document.getElementById('risk-aversion-chart'), {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Efficient Frontier',
            data: [
              { x: 5, y: 8 }, { x: 10, y: 12 }, { x: 15, y: 15 }, 
              { x: 20, y: 17 }, { x: 25, y: 18 }, { x: 30, y: 18.5 }
            ],
            backgroundColor: 'rgba(59, 130, 246, 0.6)',
            borderColor: 'rgba(59, 130, 246, 1)',
            showLine: true,
            tension: 0.4
          },
          {
            label: 'Current Portfolio',
            data: [{ x: 15, y: 15 }],
            backgroundColor: 'rgba(34, 197, 94, 1)',
            pointRadius: 8,
            pointHoverRadius: 10
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: { color: '#9ca3af' }
          },
          tooltip: {
            callbacks: {
              label: (context) => {
                return \`Risk: \${context.parsed.x}%, Return: \${context.parsed.y}%\`;
              }
            }
          }
        },
        scales: {
          x: {
            title: { display: true, text: 'Portfolio Volatility (%)', color: '#9ca3af' },
            ticks: { color: '#9ca3af' },
            grid: { color: 'rgba(148, 163, 184, 0.1)' }
          },
          y: {
            title: { display: true, text: 'Expected Return (%)', color: '#9ca3af' },
            ticks: { color: '#9ca3af' },
            grid: { color: 'rgba(148, 163, 184, 0.1)' }
          }
        }
      }
    });
    
    // Allocation Pie Chart
    charts.allocationPie = new Chart(document.getElementById('allocation-pie-chart'), {
      type: 'pie',
      data: {
        labels: ['Economic', 'Sentiment', 'Cross-Exch', 'On-Chain', 'CNN'],
        datasets: [{
          data: [20, 15, 30, 20, 15],
          backgroundColor: [
            'rgba(59, 130, 246, 0.8)',
            'rgba(168, 85, 247, 0.8)',
            'rgba(34, 197, 94, 0.8)',
            'rgba(234, 179, 8, 0.8)',
            'rgba(239, 68, 68, 0.8)'
          ]
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'bottom',
            labels: { color: '#9ca3af', font: { size: 9 } }
          }
        }
      }
    });
  }
  
  // Update dashboard with real data
  async function updateDashboard() {
    try {
      const period = document.getElementById('time-period').value;
      
      // Fetch agents data
      const agentsRes = await fetch('/api/agents');
      const agents = await agentsRes.json();
      
      // Update last update time
      document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
      
      // Update regime distribution
      if (agents.composite) {
        const regimeProbs = {
          crisis: 5, defensive: 15, neutral: 50, riskon: 25, conviction: 5
        };
        
        document.getElementById('regime-crisis').textContent = regimeProbs.crisis + '%';
        document.getElementById('regime-defensive').textContent = regimeProbs.defensive + '%';
        document.getElementById('regime-neutral').textContent = regimeProbs.neutral + '%';
        document.getElementById('regime-riskon').textContent = regimeProbs.riskon + '%';
        document.getElementById('regime-conviction').textContent = regimeProbs.conviction + '%';
      }
      
      // Update volatility drivers
      const volDrivers = {
        microstructure: 35,
        funding: 25,
        macro: 20,
        onchain: 12,
        sentiment: 8
      };
      
      Object.entries(volDrivers).forEach(([key, value]) => {
        const bar = document.getElementById(\`vol-driver-\${key}\`);
        const pct = document.getElementById(\`vol-driver-\${key}-pct\`);
        if (bar) bar.style.width = value + '%';
        if (pct) pct.textContent = value + '%';
      });
      
      // Update dominant factor
      document.getElementById('vol-dominant-factor').textContent = 'Order Flow Imbalance';
      document.getElementById('vol-contribution').textContent = '35%';
      
      // Update weekly metrics
      document.getElementById('week-return').textContent = '+5.1%';
      document.getElementById('week-sharpe').textContent = '1.82';
      document.getElementById('week-drawdown').textContent = '-3.2%';
      
      // Update hyperbolic metrics
      document.getElementById('hyper-avg-dist').textContent = '0.45';
      document.getElementById('hyper-clustering').textContent = 'High';
      document.getElementById('hyper-stability').textContent = '87%';
      
      // Update regime performance
      document.getElementById('perf-crisis').textContent = '-2.3%';
      document.getElementById('vol-crisis').textContent = 'Vol: 45%';
      document.getElementById('perf-defensive').textContent = '+0.6%';
      document.getElementById('vol-defensive').textContent = 'Vol: 18%';
      document.getElementById('perf-neutral').textContent = '+1.2%';
      document.getElementById('vol-neutral').textContent = 'Vol: 12%';
      document.getElementById('perf-riskon').textContent = '+3.0%';
      document.getElementById('vol-riskon').textContent = 'Vol: 22%';
      document.getElementById('perf-conviction').textContent = '+4.2%';
      document.getElementById('vol-conviction').textContent = 'Vol: 28%';
      
      // Update optimized metrics
      document.getElementById('opt-return').textContent = '15.2%';
      document.getElementById('opt-vol').textContent = '12.8%';
      document.getElementById('opt-sharpe').textContent = '1.19';
      document.getElementById('opt-max-position').textContent = '25%';
      
      // Add high volatility periods
      const highVolContainer = document.getElementById('high-vol-periods');
      highVolContainer.innerHTML = \`
        <div class="metric-badge">
          <div class="flex justify-between items-center">
            <div>
              <div class="font-semibold">Dec 15-16, 2025</div>
              <div class="text-xs text-gray-400">Volatility Spike: +180%</div>
            </div>
            <div class="text-red-400 text-sm">-2.8%</div>
          </div>
          <div class="text-xs text-gray-500 mt-1">
            Cause: Fed rate decision + funding rate divergence
          </div>
        </div>
        <div class="metric-badge">
          <div class="flex justify-between items-center">
            <div>
              <div class="font-semibold">Dec 12, 2025</div>
              <div class="text-xs text-gray-400">Volatility Spike: +95%</div>
            </div>
            <div class="text-yellow-400 text-sm">-0.5%</div>
          </div>
          <div class="text-xs text-gray-500 mt-1">
            Cause: Large on-chain whale transfers
          </div>
        </div>
        <div class="metric-badge">
          <div class="flex justify-between items-center">
            <div>
              <div class="font-semibold">Dec 8, 2025</div>
              <div class="text-xs text-gray-400">Volatility Spike: +65%</div>
            </div>
            <div class="text-green-400 text-sm">+1.2%</div>
          </div>
          <div class="text-xs text-gray-500 mt-1">
            Cause: Extreme fear sentiment (index: 18)
          </div>
        </div>
      \`;
      
    } catch (error) {
      console.error('Dashboard update error:', error);
    }
  }
  
  // Risk aversion slider handler
  document.getElementById('risk-aversion-slider').addEventListener('input', (e) => {
    const value = e.target.value;
    document.getElementById('risk-aversion-value').textContent = value;
    
    // Update current portfolio marker on efficient frontier
    const vol = 10 + (value * 2);
    const ret = 8 + (value * 1.2);
    
    charts.riskAversion.data.datasets[1].data = [{ x: vol, y: ret }];
    charts.riskAversion.update();
    
    // Update optimized metrics based on risk aversion
    const returnPct = (8 + (10 - value) * 1.5).toFixed(1);
    const volPct = (5 + value * 2.5).toFixed(1);
    const sharpe = (returnPct / volPct).toFixed(2);
    const maxPos = Math.max(10, 40 - (value * 3));
    
    document.getElementById('opt-return').textContent = returnPct + '%';
    document.getElementById('opt-vol').textContent = volPct + '%';
    document.getElementById('opt-sharpe').textContent = sharpe;
    document.getElementById('opt-max-position').textContent = maxPos + '%';
  });
  
  // Initialize on load
  window.addEventListener('load', () => {
    initPoincareDisk();
    initCharts();
    updateDashboard();
    setInterval(updateDashboard, 30000); // Update every 30 seconds
  });
  
  // Time period change handler
  document.getElementById('time-period').addEventListener('change', () => {
    updateDashboard();
  });
  </script>

</body>
</html>
    `);
  });
}
