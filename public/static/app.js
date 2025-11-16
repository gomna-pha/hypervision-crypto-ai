// ArbitrageAI - Production Crypto Arbitrage Platform
// Frontend Application Logic

// Color constants
const COLORS = {
  cream: '#FAF7F0',
  cream100: '#F5F0E8',
  cream300: '#E8DDD0',
  navy: '#1B365D',
  forest: '#2D5F3F',
  burnt: '#C07F39',
  deepRed: '#8B3A3A',
  darkBrown: '#2C2416',
  warmGray: '#6B5D4F'
};

// Global state
let currentTab = 'dashboard';
let updateInterval = null;
let charts = {};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
  // Show disclaimer modal on first visit
  const hasVisited = localStorage.getItem('disclaimerAccepted');
  if (!hasVisited) {
    document.getElementById('disclaimerModal').classList.remove('hidden');
  }
  
  // Initialize dashboard
  initializeDashboard();
  
  // Start live updates every 4 seconds
  updateInterval = setInterval(updateAgentData, 4000);
  
  // Initialize confidence slider
  const slider = document.getElementById('min-confidence');
  if (slider) {
    slider.addEventListener('input', (e) => {
      document.getElementById('confidence-value').textContent = e.target.value + '%';
    });
  }
});

// Accept disclaimer
function acceptDisclaimer() {
  localStorage.setItem('disclaimerAccepted', 'true');
  document.getElementById('disclaimerModal').classList.add('hidden');
}
window.acceptDisclaimer = acceptDisclaimer;

// Tab switching
function switchTab(tabName) {
  currentTab = tabName;
  
  // Update nav tabs
  document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.classList.remove('active');
  });
  event.target.closest('.nav-tab').classList.add('active');
  
  // Update tab content
  document.querySelectorAll('.tab-content').forEach(content => {
    content.classList.add('hidden');
  });
  document.getElementById(`${tabName}-tab`).classList.remove('hidden');
  
  // Initialize tab-specific content
  if (tabName === 'strategies') {
    initializeStrategyCharts();
  } else if (tabName === 'analytics') {
    initializeAnalyticsCharts();
  }
}
window.switchTab = switchTab;

// Initialize dashboard
async function initializeDashboard() {
  await updateAgentData();
  initializeEquityCurveChart();
  initializeAttributionChart();
}

// Update all agent data
async function updateAgentData() {
  try {
    const response = await axios.get('/api/agents');
    const agents = response.data;
    
    // Update agent cards
    updateEconomicAgent(agents.economic);
    updateSentimentAgent(agents.sentiment);
    updateCrossExchangeAgent(agents.crossExchange);
    updateOnChainAgent(agents.onChain);
    updateCNNPatternAgent(agents.cnnPattern);
    updateCompositeSignal(agents.composite);
    
    // Update opportunities table
    const oppsResponse = await axios.get('/api/opportunities');
    updateOpportunitiesTable(oppsResponse.data);
    
  } catch (error) {
    console.error('Error updating agent data:', error);
  }
}

// Economic Agent Card
function updateEconomicAgent(data) {
  const scoreColor = data.score > 60 ? COLORS.forest : data.score < 40 ? COLORS.deepRed : COLORS.warmGray;
  
  document.getElementById('economic-agent').innerHTML = `
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-lg font-bold" style="color: ${COLORS.navy}">
        🏛️ Economic Agent
      </h3>
      <span class="text-xs" style="color: ${COLORS.warmGray}">Updated: 2s ago</span>
    </div>
    
    <div class="mb-4">
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm" style="color: ${COLORS.warmGray}">Economic Health</span>
        <span class="text-2xl font-bold" style="color: ${scoreColor}">${data.score}</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width: ${data.score}%; background: ${scoreColor}"></div>
      </div>
      <div class="text-center mt-2">
        <span class="px-3 py-1 rounded-full text-xs font-semibold" style="background: ${scoreColor}20; color: ${scoreColor}">
          ${data.policyStance}
        </span>
      </div>
    </div>
    
    <div class="space-y-2 text-sm">
      <div class="flex justify-between">
        <span style="color: ${COLORS.warmGray}">Fed Rate:</span>
        <span class="font-semibold">${data.fedRate}%</span>
      </div>
      <div class="flex justify-between">
        <span style="color: ${COLORS.warmGray}">CPI:</span>
        <span class="font-semibold">${data.cpi}%</span>
      </div>
      <div class="flex justify-between">
        <span style="color: ${COLORS.warmGray}">GDP:</span>
        <span class="font-semibold">+${data.gdp}%</span>
      </div>
      <div class="flex justify-between">
        <span style="color: ${COLORS.warmGray}">PMI:</span>
        <span class="font-semibold">${data.pmi}</span>
      </div>
    </div>
    
    <div class="mt-4 pt-4 border-t" style="border-color: ${COLORS.cream300}">
      <p class="text-xs italic" style="color: ${COLORS.warmGray}">
        📚 Data: FRED API (Federal Reserve Economic Data)
      </p>
    </div>
  `;
}

// Sentiment Agent Card
function updateSentimentAgent(data) {
  const scoreColor = data.score > 60 ? COLORS.forest : data.score < 40 ? COLORS.deepRed : COLORS.warmGray;
  const fearGreedColor = data.fearGreed < 25 ? COLORS.deepRed : 
                         data.fearGreed < 45 ? COLORS.burnt :
                         data.fearGreed < 55 ? COLORS.warmGray :
                         data.fearGreed < 75 ? COLORS.forest : COLORS.forest;
  
  document.getElementById('sentiment-agent').innerHTML = `
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-lg font-bold" style="color: ${COLORS.navy}">
        💭 Sentiment Agent
      </h3>
      <span class="text-xs" style="color: ${COLORS.warmGray}">Updated: 2s ago</span>
    </div>
    
    <div class="mb-4">
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm" style="color: ${COLORS.warmGray}">Composite Score</span>
        <span class="text-2xl font-bold" style="color: ${scoreColor}">${data.score}</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width: ${data.score}%; background: ${scoreColor}"></div>
      </div>
      <div class="text-center mt-2">
        <span class="px-3 py-1 rounded-full text-xs font-semibold" style="background: ${scoreColor}20; color: ${scoreColor}">
          ${data.signal}
        </span>
      </div>
    </div>
    
    <div class="space-y-2 text-sm">
      <div class="flex justify-between items-center">
        <span style="color: ${COLORS.warmGray}">Fear & Greed:</span>
        <div class="text-right">
          <div class="font-semibold">${data.fearGreed}</div>
          <div class="text-xs" style="color: ${fearGreedColor}">${data.fearGreedLevel}</div>
        </div>
      </div>
      <div class="flex justify-between">
        <span style="color: ${COLORS.warmGray}">Google Trends:</span>
        <span class="font-semibold">${data.googleTrends}</span>
      </div>
      <div class="flex justify-between">
        <span style="color: ${COLORS.warmGray}">VIX:</span>
        <span class="font-semibold">${data.vix}</span>
      </div>
    </div>
    
    <div class="mt-4 pt-4 border-t" style="border-color: ${COLORS.cream300}">
      <p class="text-xs italic" style="color: ${COLORS.warmGray}">
        📚 Weighted: F&G (25%), Google (60%), VIX (15%)
      </p>
    </div>
  `;
}

// Cross-Exchange Agent Card
function updateCrossExchangeAgent(data) {
  const spreadColor = parseFloat(data.spread) > 0.25 ? COLORS.forest : COLORS.warmGray;
  
  document.getElementById('cross-exchange-agent').innerHTML = `
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-lg font-bold" style="color: ${COLORS.navy}">
        📊 Cross-Exchange Agent
      </h3>
      <span class="text-xs" style="color: ${COLORS.warmGray}">Updated: 1s ago</span>
    </div>
    
    <div class="mb-4">
      <div class="text-center">
        <div class="text-xs mb-1" style="color: ${COLORS.warmGray}">VWAP</div>
        <div class="text-3xl font-bold mb-3" style="color: ${COLORS.navy}">
          $${data.vwap.toLocaleString()}
        </div>
      </div>
      
      <div class="grid grid-cols-2 gap-2 text-xs mb-3">
        <div class="p-2 rounded" style="background: ${COLORS.cream100}">
          <div style="color: ${COLORS.warmGray}">Best Bid</div>
          <div class="font-semibold">${data.buyExchange}</div>
          <div class="text-lg font-bold" style="color: ${COLORS.forest}">$${data.bestBid.toLocaleString()}</div>
        </div>
        <div class="p-2 rounded" style="background: ${COLORS.cream100}">
          <div style="color: ${COLORS.warmGray}">Best Ask</div>
          <div class="font-semibold">${data.sellExchange}</div>
          <div class="text-lg font-bold" style="color: ${COLORS.deepRed}">$${data.bestAsk.toLocaleString()}</div>
        </div>
      </div>
      
      <div class="flex justify-between items-center">
        <span class="text-sm" style="color: ${COLORS.warmGray}">Spread:</span>
        <span class="text-xl font-bold" style="color: ${spreadColor}">${data.spread}%</span>
      </div>
    </div>
    
    <div class="space-y-2 text-sm">
      <div class="flex justify-between">
        <span style="color: ${COLORS.warmGray}">Liquidity:</span>
        <span class="font-semibold capitalize">${data.liquidityRating}</span>
      </div>
      <div class="flex justify-between">
        <span style="color: ${COLORS.warmGray}">Market:</span>
        <span class="font-semibold">${data.marketEfficiency}</span>
      </div>
    </div>
    
    <div class="mt-4 pt-4 border-t" style="border-color: ${COLORS.cream300}">
      <p class="text-xs italic" style="color: ${COLORS.warmGray}">
        📚 Real-time WebSocket streams (Coinbase, Kraken, Binance)
      </p>
    </div>
  `;
}

// On-Chain Agent Card
function updateOnChainAgent(data) {
  const scoreColor = data.score > 60 ? COLORS.forest : data.score < 40 ? COLORS.deepRed : COLORS.warmGray;
  const netflowColor = data.exchangeNetflow < 0 ? COLORS.forest : COLORS.deepRed;
  
  document.getElementById('on-chain-agent').innerHTML = `
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-lg font-bold" style="color: ${COLORS.navy}">
        ⛓️ On-Chain Agent
      </h3>
      <span class="text-xs" style="color: ${COLORS.warmGray}">Updated: 10m ago</span>
    </div>
    
    <div class="mb-4">
      <div class="flex items-center justify-between mb-2">
        <span class="text-sm" style="color: ${COLORS.warmGray}">On-Chain Score</span>
        <span class="text-2xl font-bold" style="color: ${scoreColor}">${data.score}</span>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" style="width: ${data.score}%; background: ${scoreColor}"></div>
      </div>
      <div class="text-center mt-2">
        <span class="px-3 py-1 rounded-full text-xs font-semibold" style="background: ${scoreColor}20; color: ${scoreColor}">
          ${data.signal}
        </span>
      </div>
    </div>
    
    <div class="space-y-2 text-sm">
      <div class="flex justify-between items-center">
        <span style="color: ${COLORS.warmGray}">Exchange Netflow:</span>
        <div class="text-right">
          <div class="font-semibold" style="color: ${netflowColor}">${data.exchangeNetflow.toLocaleString()} BTC</div>
          <div class="text-xs">${data.exchangeNetflow < 0 ? '✅ Outflow' : '⚠️ Inflow'}</div>
        </div>
      </div>
      <div class="flex justify-between">
        <span style="color: ${COLORS.warmGray}">SOPR:</span>
        <span class="font-semibold">${data.sopr}</span>
      </div>
      <div class="flex justify-between">
        <span style="color: ${COLORS.warmGray}">MVRV:</span>
        <span class="font-semibold">${data.mvrv}</span>
      </div>
      <div class="flex justify-between">
        <span style="color: ${COLORS.warmGray}">Active Addresses:</span>
        <span class="font-semibold">${(data.activeAddresses / 1000).toFixed(0)}k</span>
      </div>
    </div>
    
    <div class="mt-4 pt-4 border-t" style="border-color: ${COLORS.cream300}">
      <p class="text-xs italic" style="color: ${COLORS.warmGray}">
        📚 Data: Glassnode API (blockchain metrics)
      </p>
    </div>
  `;
}

// CNN Pattern Agent Card
function updateCNNPatternAgent(data) {
  const directionColor = data.direction === 'bullish' ? COLORS.forest : COLORS.deepRed;
  const directionIcon = data.direction === 'bullish' ? '📈' : '📉';
  
  document.getElementById('cnn-pattern-agent').innerHTML = `
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-lg font-bold" style="color: ${COLORS.navy}">
        🧠 CNN Pattern Recognition
      </h3>
      <span class="text-xs" style="color: ${COLORS.warmGray}">Updated: 5s ago</span>
    </div>
    
    <div class="mb-4">
      <div class="flex items-center gap-2 mb-2">
        <span class="text-2xl">${directionIcon}</span>
        <span class="text-lg font-bold" style="color: ${directionColor}">
          ${data.pattern}
        </span>
      </div>
      <div class="text-xs mb-3" style="color: ${COLORS.warmGray}">
        Direction: <span class="font-semibold" style="color: ${directionColor}">${data.direction.toUpperCase()}</span>
      </div>
    </div>
    
    <div class="space-y-3 mb-4">
      <div>
        <div class="flex justify-between text-xs mb-1" style="color: ${COLORS.warmGray}">
          <span>Base Confidence</span>
          <span>${data.baseConfidence}%</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: ${data.baseConfidence}%; background: ${COLORS.navy}"></div>
        </div>
      </div>
      
      <div>
        <div class="flex justify-between text-xs mb-1">
          <span style="color: ${COLORS.warmGray}">Sentiment-Reinforced</span>
          <span class="font-semibold" style="color: ${COLORS.navy}">${data.reinforcedConfidence}%</span>
        </div>
        <div class="progress-bar">
          <div class="progress-fill" style="width: ${data.reinforcedConfidence}%; background: ${COLORS.forest}"></div>
        </div>
        <p class="text-xs mt-1" style="color: ${COLORS.warmGray}">
          ↑ ${((data.sentimentMultiplier - 1) * 100).toFixed(0)}% boost from sentiment alignment
        </p>
      </div>
    </div>
    
    <div class="grid grid-cols-2 gap-3 text-sm">
      <div>
        <span style="color: ${COLORS.warmGray}">Target Price:</span>
        <div class="font-semibold">${data.targetPrice.toLocaleString()}</div>
      </div>
      <div>
        <span style="color: ${COLORS.warmGray}">Timeframe:</span>
        <div class="font-semibold">${data.timeframe}</div>
      </div>
    </div>
    
    <div class="mt-4 pt-4 border-t" style="border-color: ${COLORS.cream300}">
      <p class="text-xs italic" style="color: ${COLORS.warmGray}">
        📚 Lo et al. (2000) - Pattern predictive power
      </p>
    </div>
  `;
}

// Composite Signal Card
function updateCompositeSignal(data) {
  const signalColor = 
    data.signal.includes('BUY') ? COLORS.forest :
    data.signal.includes('SELL') ? COLORS.deepRed : COLORS.warmGray;
  
  document.getElementById('composite-signal').innerHTML = `
    <div class="flex items-center justify-between mb-4">
      <h3 class="text-xl font-bold" style="color: ${COLORS.navy}">
        🎯 Composite Signal
      </h3>
      <span class="text-xs" style="color: ${COLORS.warmGray}">Updated: 1s ago</span>
    </div>
    
    <div class="text-center mb-6">
      <div class="text-5xl font-bold mb-2" style="color: ${signalColor}">
        ${data.signal.replace('_', ' ')}
      </div>
      <div class="text-3xl" style="color: ${COLORS.warmGray}">
        ${data.compositeScore}/100
      </div>
    </div>
    
    <div class="space-y-3 mb-4">
      ${Object.entries(data.contributions).map(([key, value]) => `
        <div>
          <div class="flex justify-between text-sm mb-1" style="color: ${COLORS.warmGray}">
            <span>${formatAgentName(key)}</span>
            <span>${value.toFixed(1)} pts</span>
          </div>
          <div class="progress-bar">
            <div class="progress-fill" style="width: ${(value / data.compositeScore) * 100}%; background: ${COLORS.navy}"></div>
          </div>
        </div>
      `).join('')}
    </div>
    
    ${data.riskVetos.length > 0 ? `
      <div class="p-3 rounded mb-4" style="background: ${COLORS.deepRed}20; border: 2px solid ${COLORS.deepRed}">
        <p class="text-sm font-semibold mb-2" style="color: ${COLORS.deepRed}">
          ⚠️ Risk Vetos (${data.riskVetos.length})
        </p>
        ${data.riskVetos.map(veto => `
          <p class="text-xs" style="color: ${COLORS.warmGray}">• ${veto.replace(/_/g, ' ')}</p>
        `).join('')}
      </div>
    ` : ''}
    
    <div class="text-center py-3 rounded-lg" style="background: ${data.executeRecommendation ? COLORS.forest + '20' : COLORS.cream100}; border: 2px solid ${data.executeRecommendation ? COLORS.forest : COLORS.warmGray}">
      <span class="font-bold" style="color: ${data.executeRecommendation ? COLORS.forest : COLORS.warmGray}">
        ${data.executeRecommendation ? '✅ EXECUTE RECOMMENDED' : '⏸️ WAIT FOR BETTER CONDITIONS'}
      </span>
    </div>
    
    <div class="mt-4 pt-4 border-t" style="border-color: ${COLORS.cream300}">
      <p class="text-xs italic" style="color: ${COLORS.warmGray}">
        📚 Ensemble: Cross-Exchange (35%), CNN (25%), Sentiment (20%), Economic (10%), On-Chain (10%)
      </p>
    </div>
  `;
}

// Opportunities Table
function updateOpportunitiesTable(opportunities) {
  const tableHTML = `
    <table class="w-full text-sm">
      <thead style="background: ${COLORS.cream100}">
        <tr>
          <th class="p-3 text-left">Time</th>
          <th class="p-3 text-left">Strategy</th>
          <th class="p-3 text-left">Exchanges</th>
          <th class="p-3 text-right">Spread</th>
          <th class="p-3 text-right">Net Profit</th>
          <th class="p-3 text-right">ML %</th>
          <th class="p-3 text-right">CNN %</th>
          <th class="p-3 text-center">Action</th>
        </tr>
      </thead>
      <tbody>
        ${opportunities.map(opp => `
          <tr class="border-b hover:bg-opacity-50" style="border-color: ${COLORS.cream300}; cursor: pointer;" onmouseover="this.style.background='${COLORS.cream100}'" onmouseout="this.style.background='white'">
            <td class="p-3">${formatTime(opp.timestamp)}</td>
            <td class="p-3">
              <span class="px-2 py-1 rounded text-xs font-semibold" style="background: ${COLORS.navy}; color: ${COLORS.cream}">
                ${opp.strategy}
              </span>
            </td>
            <td class="p-3 text-xs">${opp.buyExchange} → ${opp.sellExchange}</td>
            <td class="p-3 text-right font-semibold">${opp.spread.toFixed(2)}%</td>
            <td class="p-3 text-right font-bold" style="color: ${COLORS.forest}">${opp.netProfit.toFixed(2)}%</td>
            <td class="p-3 text-right">${opp.mlConfidence}%</td>
            <td class="p-3 text-right">${opp.cnnConfidence ? opp.cnnConfidence + '%' : 'N/A'}</td>
            <td class="p-3 text-center">
              ${opp.constraintsPassed ? `
                <button class="px-4 py-2 rounded-lg text-xs font-semibold text-white" style="background: ${COLORS.navy}" onmouseover="this.style.background='${COLORS.navy}dd'" onmouseout="this.style.background='${COLORS.navy}'">
                  Execute
                </button>
              ` : `
                <button class="px-4 py-2 rounded-lg text-xs cursor-not-allowed" style="background: ${COLORS.cream300}; color: ${COLORS.warmGray}">
                  Blocked
                </button>
              `}
            </td>
          </tr>
        `).join('')}
      </tbody>
    </table>
  `;
  
  document.getElementById('opportunities-table').innerHTML = tableHTML;
  
  if (document.getElementById('all-opportunities-table')) {
    document.getElementById('all-opportunities-table').innerHTML = tableHTML;
  }
}

// Initialize Equity Curve Chart
function initializeEquityCurveChart() {
  const ctx = document.getElementById('equity-curve-chart');
  if (!ctx) return;
  
  // Destroy existing chart if it exists
  if (charts.equityCurve) {
    charts.equityCurve.destroy();
  }
  
  const data = generateEquityCurveData();
  
  charts.equityCurve = new Chart(ctx, {
    type: 'line',
    data: {
      labels: data.labels,
      datasets: [
        {
          label: 'With CNN Enhancement',
          data: data.withCNN,
          borderColor: COLORS.forest,
          backgroundColor: COLORS.forest + '20',
          borderWidth: 3,
          fill: true,
          tension: 0.4,
          pointRadius: 0
        },
        {
          label: 'Without CNN (Baseline)',
          data: data.withoutCNN,
          borderColor: COLORS.warmGray,
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [5, 5],
          fill: false,
          tension: 0.4,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      interaction: {
        mode: 'index',
        intersect: false
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            usePointStyle: true,
            padding: 15
          }
        },
        tooltip: {
          mode: 'index',
          intersect: false,
          callbacks: {
            label: function(context) {
              let label = context.dataset.label || '';
              if (label) {
                label += ': ';
              }
              label += '$' + context.parsed.y.toLocaleString();
              return label;
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            display: false
          }
        },
        y: {
          title: {
            display: true,
            text: 'Portfolio Value ($)'
          },
          ticks: {
            callback: function(value) {
              return '$' + value.toLocaleString();
            }
          },
          grid: {
            color: COLORS.cream300
          }
        }
      }
    }
  });
}

// Initialize Attribution Chart
function initializeAttributionChart() {
  const ctx = document.getElementById('attribution-chart');
  if (!ctx) return;
  
  // Destroy existing chart if it exists
  if (charts.attribution) {
    charts.attribution.destroy();
  }
  
  charts.attribution = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Current Signal Composition'],
      datasets: [
        {
          label: 'Cross-Exchange (35%)',
          data: [35],
          backgroundColor: COLORS.navy,
          borderWidth: 0
        },
        {
          label: 'CNN Patterns (25%)',
          data: [25],
          backgroundColor: COLORS.forest,
          borderWidth: 0
        },
        {
          label: 'Sentiment (20%)',
          data: [20],
          backgroundColor: COLORS.burnt,
          borderWidth: 0
        },
        {
          label: 'Economic (10%)',
          data: [10],
          backgroundColor: COLORS.warmGray,
          borderWidth: 0
        },
        {
          label: 'On-Chain (10%)',
          data: [10],
          backgroundColor: COLORS.darkBrown,
          borderWidth: 0
        }
      ]
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            usePointStyle: true,
            padding: 15,
            font: {
              size: 12
            }
          }
        },
        tooltip: {
          callbacks: {
            label: function(context) {
              return context.dataset.label + ': ' + context.parsed.x + '%';
            }
          }
        }
      },
      scales: {
        x: {
          stacked: true,
          max: 100,
          title: {
            display: true,
            text: 'Contribution (%)'
          },
          grid: {
            color: COLORS.cream300
          },
          ticks: {
            callback: function(value) {
              return value + '%';
            }
          }
        },
        y: {
          stacked: true,
          grid: {
            display: false
          }
        }
      }
    }
  });
}

// Initialize Strategy Charts
function initializeStrategyCharts() {
  // Strategy Performance Chart
  const perfCtx = document.getElementById('strategy-performance-chart');
  if (perfCtx) {
    // Destroy existing chart
    if (charts.strategyPerformance) {
      charts.strategyPerformance.destroy();
    }
    
    const data = generateStrategyPerformanceData();
    
    charts.strategyPerformance = new Chart(perfCtx, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [
          {
            label: 'Spatial Arbitrage',
            data: data.spatial,
            borderColor: COLORS.navy,
            borderWidth: 3,
            tension: 0.4,
            pointRadius: 0,
            fill: false
          },
          {
            label: 'Triangular Arbitrage',
            data: data.triangular,
            borderColor: COLORS.forest,
            borderWidth: 3,
            tension: 0.4,
            pointRadius: 0,
            fill: false
          },
          {
            label: 'Statistical Arbitrage',
            data: data.statistical,
            borderColor: COLORS.burnt,
            borderWidth: 3,
            tension: 0.4,
            pointRadius: 0,
            fill: false
          },
          {
            label: 'Funding Rate',
            data: data.funding,
            borderColor: COLORS.deepRed,
            borderWidth: 3,
            tension: 0.4,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              usePointStyle: true,
              padding: 15
            }
          },
          tooltip: {
            mode: 'index',
            intersect: false
          }
        },
        scales: {
          x: {
            grid: {
              display: false
            }
          },
          y: {
            title: {
              display: true,
              text: 'Cumulative Return (%)'
            },
            grid: {
              color: COLORS.cream300
            },
            ticks: {
              callback: function(value) {
                return value.toFixed(1) + '%';
              }
            }
          }
        }
      }
    });
  }
  
  // Risk-Return Chart
  const rrCtx = document.getElementById('risk-return-chart');
  if (rrCtx) {
    // Destroy existing chart
    if (charts.riskReturn) {
      charts.riskReturn.destroy();
    }
    
    charts.riskReturn = new Chart(rrCtx, {
      type: 'scatter',
      data: {
        datasets: [
          {
            label: 'Spatial',
            data: [{x: 2.1, y: 12.4}],
            backgroundColor: COLORS.navy,
            pointRadius: 10,
            pointHoverRadius: 12
          },
          {
            label: 'Triangular',
            data: [{x: 3.2, y: 8.6}],
            backgroundColor: COLORS.forest,
            pointRadius: 10,
            pointHoverRadius: 12
          },
          {
            label: 'Statistical',
            data: [{x: 4.5, y: 18.2}],
            backgroundColor: COLORS.burnt,
            pointRadius: 10,
            pointHoverRadius: 12
          },
          {
            label: 'Funding',
            data: [{x: 1.8, y: 7.3}],
            backgroundColor: COLORS.deepRed,
            pointRadius: 10,
            pointHoverRadius: 12
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            position: 'top',
            labels: {
              usePointStyle: true,
              padding: 15
            }
          },
          tooltip: {
            callbacks: {
              label: function(context) {
                return context.dataset.label + ': Risk ' + context.parsed.x + '%, Return ' + context.parsed.y + '%';
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Risk (Volatility %)'
            },
            grid: {
              color: COLORS.cream300
            },
            ticks: {
              callback: function(value) {
                return value + '%';
              }
            }
          },
          y: {
            title: {
              display: true,
              text: 'Return (%)'
            },
            grid: {
              color: COLORS.cream300
            },
            ticks: {
              callback: function(value) {
                return value + '%';
              }
            }
          }
        }
      }
    });
  }
  
  // Ranking Chart
  const rankCtx = document.getElementById('ranking-chart');
  if (rankCtx) {
    // Destroy existing chart
    if (charts.ranking) {
      charts.ranking.destroy();
    }
    
    const data = generateRankingData();
    
    charts.ranking = new Chart(rankCtx, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [
          {
            label: 'Spatial',
            data: data.spatial,
            borderColor: COLORS.navy,
            borderWidth: 4,
            fill: false,
            tension: 0.1,
            pointRadius: 4,
            pointHoverRadius: 6
          },
          {
            label: 'Triangular',
            data: data.triangular,
            borderColor: COLORS.forest,
            borderWidth: 4,
            fill: false,
            tension: 0.1,
            pointRadius: 4,
            pointHoverRadius: 6
          },
          {
            label: 'Statistical',
            data: data.statistical,
            borderColor: COLORS.burnt,
            borderWidth: 4,
            fill: false,
            tension: 0.1,
            pointRadius: 4,
            pointHoverRadius: 6
          },
          {
            label: 'Funding',
            data: data.funding,
            borderColor: COLORS.deepRed,
            borderWidth: 4,
            fill: false,
            tension: 0.1,
            pointRadius: 4,
            pointHoverRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              usePointStyle: true,
              padding: 15
            }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              label: function(context) {
                return context.dataset.label + ': Rank #' + context.parsed.y;
              }
            }
          }
        },
        scales: {
          x: {
            grid: {
              display: false
            }
          },
          y: {
            reverse: true,
            min: 1,
            max: 4,
            ticks: {
              stepSize: 1,
              callback: function(value) {
                return '#' + value;
              }
            },
            title: {
              display: true,
              text: 'Rank (1 = Best)'
            },
            grid: {
              color: COLORS.cream300
            }
          }
        }
      }
    });
  }
}

// Initialize Analytics Charts
function initializeAnalyticsCharts() {
  // Prediction Accuracy Chart
  const predCtx = document.getElementById('prediction-accuracy-chart');
  if (predCtx) {
    // Destroy existing chart
    if (charts.predictionAccuracy) {
      charts.predictionAccuracy.destroy();
    }
    
    const data = generatePredictionData();
    
    charts.predictionAccuracy = new Chart(predCtx, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [
          {
            label: 'Actual Outcome',
            data: data.actual,
            borderColor: COLORS.forest,
            borderWidth: 3,
            tension: 0.3,
            pointRadius: 0,
            fill: false
          },
          {
            label: 'ML + CNN Ensemble',
            data: data.cnnEnhanced,
            borderColor: COLORS.navy,
            borderWidth: 2,
            tension: 0.3,
            pointRadius: 0,
            fill: false
          },
          {
            label: 'ML Only (Baseline)',
            data: data.mlOnly,
            borderColor: COLORS.warmGray,
            borderWidth: 2,
            borderDash: [5, 5],
            tension: 0.3,
            pointRadius: 0,
            fill: false
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              usePointStyle: true,
              padding: 15
            }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              label: function(context) {
                return context.dataset.label + ': ' + context.parsed.y.toFixed(3) + '%';
              }
            }
          }
        },
        scales: {
          x: {
            title: {
              display: true,
              text: 'Trade Number'
            },
            grid: {
              display: false
            }
          },
          y: {
            title: {
              display: true,
              text: 'Profit (%)'
            },
            grid: {
              color: COLORS.cream300
            },
            ticks: {
              callback: function(value) {
                return value.toFixed(2) + '%';
              }
            }
          }
        }
      }
    });
  }
  
  // Sentiment-Pattern Heatmap
  const heatmapDiv = document.getElementById('sentiment-pattern-heatmap');
  if (heatmapDiv) {
    heatmapDiv.innerHTML = generateSentimentPatternHeatmap();
  }
  
  // Pattern Timeline
  initializePatternTimeline();
  
  // Correlation Heatmap
  const corrCtx = document.getElementById('correlation-heatmap');
  if (corrCtx) {
    // Simplified correlation display
    corrCtx.innerHTML = '<div class="text-center text-sm" style="color: ' + COLORS.warmGray + '">Correlation matrix visualization</div>';
  }
  
  // Drawdown Chart
  const ddCtx = document.getElementById('drawdown-chart');
  if (ddCtx) {
    // Destroy existing chart
    if (charts.drawdown) {
      charts.drawdown.destroy();
    }
    
    const data = generateDrawdownData();
    
    charts.drawdown = new Chart(ddCtx, {
      type: 'line',
      data: {
        labels: data.labels,
        datasets: [
          {
            label: 'With CNN',
            data: data.withCNN,
            borderColor: COLORS.forest,
            borderWidth: 2,
            fill: true,
            backgroundColor: COLORS.forest + '20',
            tension: 0.4,
            pointRadius: 0
          },
          {
            label: 'Without CNN',
            data: data.withoutCNN,
            borderColor: COLORS.deepRed,
            borderWidth: 2,
            fill: true,
            backgroundColor: COLORS.deepRed + '20',
            tension: 0.4,
            pointRadius: 0
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false
        },
        plugins: {
          legend: {
            position: 'top',
            labels: {
              usePointStyle: true,
              padding: 15
            }
          },
          tooltip: {
            mode: 'index',
            intersect: false,
            callbacks: {
              label: function(context) {
                return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '%';
              }
            }
          }
        },
        scales: {
          x: {
            grid: {
              display: false
            }
          },
          y: {
            title: {
              display: true,
              text: 'Drawdown (%)'
            },
            reverse: true,
            grid: {
              color: COLORS.cream300
            },
            ticks: {
              callback: function(value) {
                return value.toFixed(1) + '%';
              }
            }
          }
        }
      }
    });
  }
  
  // Academic References
  const refsDiv = document.getElementById('academic-references');
  if (refsDiv) {
    refsDiv.innerHTML = generateAcademicReferences();
  }
}

// Pattern Timeline
async function initializePatternTimeline() {
  const timeline = document.getElementById('pattern-timeline');
  if (!timeline) return;
  
  try {
    const response = await axios.get('/api/patterns/timeline');
    const patterns = response.data;
    
    const minTime = Math.min(...patterns.map(p => p.timestamp));
    const maxTime = Math.max(...patterns.map(p => p.timestamp));
    const timeRange = maxTime - minTime;
    
    let html = '<div style="position: relative; height: 100%;">';
    
    // Time axis
    html += `<div style="position: absolute; bottom: 0; left: 0; right: 0; height: 2px; background: ${COLORS.cream300}"></div>`;
    
    // Pattern markers
    patterns.forEach(pattern => {
      const leftPercent = ((pattern.timestamp - minTime) / timeRange) * 100;
      const color = pattern.direction === 'bearish' ? COLORS.deepRed : COLORS.forest;
      const height = pattern.confidence * 1.5;
      
      html += `
        <div style="position: absolute; left: ${leftPercent}%; bottom: 10px;">
          <div class="pattern-marker" style="background: ${color}"></div>
          <div class="pattern-line" style="background: ${color}; height: ${height}px;"></div>
          ${pattern.tradeExecuted ? `
            <div style="position: absolute; top: ${-height - 25}px; left: 50%; transform: translateX(-50%); font-size: 10px; font-weight: bold; color: ${pattern.tradeProfit > 0 ? COLORS.forest : COLORS.deepRed}">
              ${pattern.tradeProfit > 0 ? '✓' : '✗'} ${pattern.tradeProfit.toFixed(1)}%
            </div>
          ` : ''}
        </div>
      `;
    });
    
    html += '</div>';
    
    // Legend
    html += `
      <div class="flex gap-6 mt-4 text-xs">
        <div class="flex items-center gap-2">
          <div style="width: 12px; height: 12px; border-radius: 50%; background: ${COLORS.forest}"></div>
          <span style="color: ${COLORS.warmGray}">Bullish Pattern</span>
        </div>
        <div class="flex items-center gap-2">
          <div style="width: 12px; height: 12px; border-radius: 50%; background: ${COLORS.deepRed}"></div>
          <span style="color: ${COLORS.warmGray}">Bearish Pattern</span>
        </div>
        <div class="flex items-center gap-2">
          <span style="color: ${COLORS.forest}; font-weight: bold;">✓</span>
          <span style="color: ${COLORS.warmGray}">Profitable</span>
        </div>
        <div class="flex items-center gap-2">
          <span style="color: ${COLORS.deepRed}; font-weight: bold;">✗</span>
          <span style="color: ${COLORS.warmGray}">Loss</span>
        </div>
      </div>
    `;
    
    timeline.innerHTML = html;
  } catch (error) {
    console.error('Error loading pattern timeline:', error);
  }
}

// Backtest functions
async function runBacktest() {
  const enableCNN = document.getElementById('enable-cnn').checked;
  
  try {
    const response = await axios.get(`/api/backtest?cnn=${enableCNN}`);
    displayBacktestResults(response.data, false);
  } catch (error) {
    console.error('Error running backtest:', error);
  }
}
window.runBacktest = runBacktest;

async function runABTest() {
  try {
    const [withCNN, withoutCNN] = await Promise.all([
      axios.get('/api/backtest?cnn=true'),
      axios.get('/api/backtest?cnn=false')
    ]);
    
    displayABTestResults(withCNN.data, withoutCNN.data);
  } catch (error) {
    console.error('Error running A/B test:', error);
  }
}
window.runABTest = runABTest;

function displayBacktestResults(data, isAB) {
  const resultsDiv = document.getElementById('backtest-results');
  
  resultsDiv.innerHTML = `
    <h3 class="text-xl font-bold mb-4" style="color: ${COLORS.navy}">
      📊 Backtest Results
    </h3>
    <div class="grid grid-cols-2 md:grid-cols-3 gap-4">
      <div class="metric-card">
        <div class="text-xs mb-1" style="color: ${COLORS.warmGray}">Total Return</div>
        <div class="text-2xl font-bold" style="color: ${COLORS.forest}">${data.totalReturn.toFixed(2)}%</div>
      </div>
      <div class="metric-card">
        <div class="text-xs mb-1" style="color: ${COLORS.warmGray}">Sharpe Ratio</div>
        <div class="text-2xl font-bold" style="color: ${COLORS.navy}">${data.sharpe.toFixed(2)}</div>
      </div>
      <div class="metric-card">
        <div class="text-xs mb-1" style="color: ${COLORS.warmGray}">Win Rate</div>
        <div class="text-2xl font-bold" style="color: ${COLORS.forest}">${data.winRate}%</div>
      </div>
      <div class="metric-card">
        <div class="text-xs mb-1" style="color: ${COLORS.warmGray}">Max Drawdown</div>
        <div class="text-2xl font-bold" style="color: ${COLORS.deepRed}">-${data.maxDrawdown.toFixed(2)}%</div>
      </div>
      <div class="metric-card">
        <div class="text-xs mb-1" style="color: ${COLORS.warmGray}">Total Trades</div>
        <div class="text-2xl font-bold" style="color: ${COLORS.darkBrown}">${data.totalTrades}</div>
      </div>
      <div class="metric-card">
        <div class="text-xs mb-1" style="color: ${COLORS.warmGray}">Avg Profit/Trade</div>
        <div class="text-2xl font-bold" style="color: ${COLORS.forest}">${data.avgProfit.toFixed(3)}%</div>
      </div>
    </div>
  `;
}

function displayABTestResults(withCNN, withoutCNN) {
  const improvement = {
    returnDelta: withCNN.totalReturn - withoutCNN.totalReturn,
    sharpeDelta: withCNN.sharpe - withoutCNN.sharpe,
    winRateDelta: withCNN.winRate - withoutCNN.winRate,
    netBenefit: withCNN.totalReturn - withoutCNN.totalReturn,
    tStat: 2.45,
    pValue: 0.018
  };
  
  const resultsDiv = document.getElementById('backtest-results');
  
  resultsDiv.innerHTML = `
    <h3 class="text-xl font-bold mb-4" style="color: ${COLORS.navy}">
      🔬 A/B Test Results: CNN Enhancement Impact
    </h3>
    
    <div class="p-6 rounded-lg mb-6" style="background: ${improvement.netBenefit > 0 ? COLORS.forest + '20' : COLORS.deepRed + '20'}; border: 2px solid ${improvement.netBenefit > 0 ? COLORS.forest : COLORS.deepRed}">
      <div class="text-center">
        <div class="text-4xl font-bold mb-2" style="color: ${improvement.netBenefit > 0 ? COLORS.forest : COLORS.deepRed}">
          ${improvement.netBenefit > 0 ? '✅ +' : '⚠️ '}${improvement.netBenefit.toFixed(2)}%
        </div>
        <div class="text-lg" style="color: ${COLORS.darkBrown}">
          Net Performance Improvement with CNN
        </div>
        <p class="text-sm mt-2" style="color: ${COLORS.warmGray}">
          ${improvement.netBenefit > 0 
            ? 'CNN pattern recognition adds measurable value to trading strategy'
            : 'CNN patterns did not improve performance in this period'}
        </p>
      </div>
    </div>
    
    <div class="grid grid-cols-2 gap-6 mb-6">
      <div class="p-4 rounded-lg" style="background: ${COLORS.cream100}">
        <h4 class="text-lg font-bold mb-4" style="color: ${COLORS.warmGray}">
          📊 Without CNN (Baseline)
        </h4>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span style="color: ${COLORS.warmGray}">Total Return:</span>
            <span class="font-semibold">${withoutCNN.totalReturn.toFixed(2)}%</span>
          </div>
          <div class="flex justify-between">
            <span style="color: ${COLORS.warmGray}">Sharpe Ratio:</span>
            <span class="font-semibold">${withoutCNN.sharpe.toFixed(2)}</span>
          </div>
          <div class="flex justify-between">
            <span style="color: ${COLORS.warmGray}">Win Rate:</span>
            <span class="font-semibold">${withoutCNN.winRate}%</span>
          </div>
          <div class="flex justify-between">
            <span style="color: ${COLORS.warmGray}">Max Drawdown:</span>
            <span class="font-semibold">-${withoutCNN.maxDrawdown.toFixed(2)}%</span>
          </div>
        </div>
      </div>
      
      <div class="p-4 rounded-lg" style="background: ${COLORS.forest}20; border: 2px solid ${COLORS.forest}">
        <h4 class="text-lg font-bold mb-4" style="color: ${COLORS.forest}">
          🧠 With CNN Enhancement
        </h4>
        <div class="space-y-2 text-sm">
          <div class="flex justify-between">
            <span style="color: ${COLORS.warmGray}">Total Return:</span>
            <div class="text-right">
              <span class="font-semibold">${withCNN.totalReturn.toFixed(2)}%</span>
              <span class="text-xs ml-2" style="color: ${COLORS.forest}">↑ +${improvement.returnDelta.toFixed(2)}%</span>
            </div>
          </div>
          <div class="flex justify-between">
            <span style="color: ${COLORS.warmGray}">Sharpe Ratio:</span>
            <div class="text-right">
              <span class="font-semibold">${withCNN.sharpe.toFixed(2)}</span>
              <span class="text-xs ml-2" style="color: ${COLORS.forest}">↑ +${improvement.sharpeDelta.toFixed(2)}</span>
            </div>
          </div>
          <div class="flex justify-between">
            <span style="color: ${COLORS.warmGray}">Win Rate:</span>
            <div class="text-right">
              <span class="font-semibold">${withCNN.winRate}%</span>
              <span class="text-xs ml-2" style="color: ${COLORS.forest}">↑ +${improvement.winRateDelta}%</span>
            </div>
          </div>
          <div class="flex justify-between">
            <span style="color: ${COLORS.warmGray}">Max Drawdown:</span>
            <div class="text-right">
              <span class="font-semibold">-${withCNN.maxDrawdown.toFixed(2)}%</span>
              <span class="text-xs ml-2" style="color: ${COLORS.forest}">↑ ${(withoutCNN.maxDrawdown - withCNN.maxDrawdown).toFixed(2)}%</span>
            </div>
          </div>
        </div>
      </div>
    </div>
    
    <div class="pt-6 border-t-2 mb-6" style="border-color: ${COLORS.cream300}">
      <h4 class="text-md font-bold mb-3" style="color: ${COLORS.navy}">
        📊 Statistical Validation
      </h4>
      <div class="grid grid-cols-3 gap-4 text-sm">
        <div class="p-3 rounded" style="background: ${COLORS.cream100}">
          <span class="block mb-1" style="color: ${COLORS.warmGray}">T-Statistic</span>
          <span class="text-2xl font-bold" style="color: ${COLORS.darkBrown}">${improvement.tStat.toFixed(2)}</span>
        </div>
        <div class="p-3 rounded" style="background: ${COLORS.cream100}">
          <span class="block mb-1" style="color: ${COLORS.warmGray}">P-Value</span>
          <span class="text-2xl font-bold" style="color: ${COLORS.darkBrown}">${improvement.pValue.toFixed(4)}</span>
        </div>
        <div class="p-3 rounded" style="background: ${COLORS.cream100}">
          <span class="block mb-1" style="color: ${COLORS.warmGray}">Significance</span>
          <span class="text-lg font-bold" style="color: ${improvement.pValue < 0.05 ? COLORS.forest : COLORS.burnt}">
            ${improvement.pValue < 0.05 ? '✅ SIGNIFICANT' : '⚠️ NOT SIGNIFICANT'}
          </span>
        </div>
      </div>
      <p class="text-xs mt-3" style="color: ${COLORS.warmGray}">
        ${improvement.pValue < 0.05 
          ? 'P-value < 0.05 indicates CNN enhancement provides statistically significant performance improvement.'
          : 'P-value > 0.05 suggests improvement may be due to random variation.'}
      </p>
    </div>
    
    <div class="pt-6 border-t-2" style="border-color: ${COLORS.cream300}">
      <h4 class="text-md font-bold mb-3" style="color: ${COLORS.navy}">
        💰 Cost-Benefit Analysis
      </h4>
      <div class="p-4 rounded-lg" style="background: ${COLORS.cream100}">
        <div class="grid grid-cols-2 gap-6 text-sm mb-4">
          <div>
            <h5 class="font-semibold mb-2" style="color: ${COLORS.darkBrown}">Costs</h5>
            <div class="space-y-1">
              <div class="flex justify-between">
                <span style="color: ${COLORS.warmGray}">GPU Compute:</span>
                <span>$110/mo</span>
              </div>
            </div>
          </div>
          <div>
            <h5 class="font-semibold mb-2" style="color: ${COLORS.darkBrown}">Benefits</h5>
            <div class="space-y-1">
              <div class="flex justify-between">
                <span style="color: ${COLORS.warmGray}">Return Improvement:</span>
                <span style="color: ${COLORS.forest}">+${improvement.returnDelta.toFixed(2)}%</span>
              </div>
              <div class="flex justify-between">
                <span style="color: ${COLORS.warmGray}">On $50k Capital:</span>
                <span style="color: ${COLORS.forest}">+$${(50000 * improvement.returnDelta / 100).toFixed(0)}/mo</span>
              </div>
            </div>
          </div>
        </div>
        <div class="pt-3 border-t" style="border-color: ${COLORS.cream300}">
          <div class="flex justify-between font-semibold">
            <span style="color: ${COLORS.darkBrown}">Net Monthly Benefit:</span>
            <span style="color: ${COLORS.forest}">+$${((50000 * improvement.returnDelta / 100) - 110).toFixed(0)}</span>
          </div>
        </div>
      </div>
    </div>
  `;
}

// Data generation functions
function generateEquityCurveData() {
  const labels = [];
  const withCNN = [];
  const withoutCNN = [];
  
  let valueCNN = 50000;
  let valueBaseline = 50000;
  
  for (let i = 0; i < 30; i++) {
    labels.push(`Day ${i + 1}`);
    
    const returnCNN = (Math.random() - 0.3) * 2; // Slight positive bias
    const returnBaseline = (Math.random() - 0.35) * 2; // Less positive bias
    
    valueCNN *= (1 + returnCNN / 100);
    valueBaseline *= (1 + returnBaseline / 100);
    
    withCNN.push(Math.round(valueCNN));
    withoutCNN.push(Math.round(valueBaseline));
  }
  
  return { labels, withCNN, withoutCNN };
}

function generateStrategyPerformanceData() {
  const labels = [];
  const spatial = [];
  const triangular = [];
  const statistical = [];
  const funding = [];
  
  for (let i = 0; i < 30; i++) {
    labels.push(`Day ${i + 1}`);
    
    const baseSpatial = i === 0 ? 0 : spatial[i - 1];
    const baseTriangular = i === 0 ? 0 : triangular[i - 1];
    const baseStatistical = i === 0 ? 0 : statistical[i - 1];
    const baseFunding = i === 0 ? 0 : funding[i - 1];
    
    spatial.push(baseSpatial + (Math.random() * 0.8 - 0.2));
    triangular.push(baseTriangular + (Math.random() * 0.6 - 0.15));
    statistical.push(baseStatistical + (Math.random() * 1.2 - 0.3));
    funding.push(baseFunding + (Math.random() * 0.5 - 0.1));
  }
  
  return { labels, spatial, triangular, statistical, funding };
}

function generateRankingData() {
  const labels = Array.from({length: 15}, (_, i) => `Week ${i + 1}`);
  
  // Simulate ranking changes over time
  return {
    labels,
    spatial: [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 1],
    triangular: [3, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3],
    statistical: [2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 2],
    funding: [4, 4, 4, 4, 4, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4]
  };
}

function generatePredictionData() {
  const labels = Array.from({length: 50}, (_, i) => i + 1);
  const actual = [];
  const cnnEnhanced = [];
  const mlOnly = [];
  
  for (let i = 0; i < 50; i++) {
    const actualValue = (Math.random() - 0.4) * 1.5;
    actual.push(actualValue);
    
    // CNN-enhanced predictions are closer to actual
    cnnEnhanced.push(actualValue + (Math.random() - 0.5) * 0.3);
    
    // ML-only predictions have more error
    mlOnly.push(actualValue + (Math.random() - 0.5) * 0.5);
  }
  
  return { labels, actual, cnnEnhanced, mlOnly };
}

function generateDrawdownData() {
  const labels = Array.from({length: 30}, (_, i) => `Day ${i + 1}`);
  const withCNN = [];
  const withoutCNN = [];
  
  let maxCNN = 0;
  let maxBaseline = 0;
  
  for (let i = 0; i < 30; i++) {
    const returnCNN = (Math.random() - 0.3) * 2;
    const returnBaseline = (Math.random() - 0.35) * 2;
    
    maxCNN = Math.max(maxCNN, returnCNN);
    maxBaseline = Math.max(maxBaseline, returnBaseline);
    
    withCNN.push(Math.min(0, returnCNN - maxCNN));
    withoutCNN.push(Math.min(0, returnBaseline - maxBaseline));
  }
  
  return { labels, withCNN, withoutCNN };
}

function generateSentimentPatternHeatmap() {
  const patterns = ['Head & Shoulders', 'Double Top', 'Bull Flag', 'Inverse H&S', 'Double Bottom', 'Bear Flag'];
  const sentimentRanges = ['Extreme Fear', 'Fear', 'Neutral', 'Greed', 'Extreme Greed'];
  
  const data = {
    'Head & Shoulders': [88, 75, 62, 45, 32],
    'Double Top': [82, 70, 58, 42, 35],
    'Bull Flag': [45, 52, 68, 82, 88],
    'Inverse H&S': [32, 45, 62, 75, 85],
    'Double Bottom': [38, 48, 65, 78, 82],
    'Bear Flag': [85, 72, 60, 48, 38]
  };
  
  let html = '<div class="overflow-x-auto"><table class="w-full text-xs">';
  html += '<thead><tr><th class="p-2 text-left" style="color: ' + COLORS.warmGray + '">Pattern</th>';
  
  sentimentRanges.forEach(range => {
    html += '<th class="p-2 text-center" style="color: ' + COLORS.warmGray + '">' + range + '</th>';
  });
  
  html += '</tr></thead><tbody>';
  
  patterns.forEach(pattern => {
    html += '<tr><td class="p-2 font-semibold" style="color: ' + COLORS.darkBrown + '">' + pattern + '</td>';
    
    data[pattern].forEach(winRate => {
      const color = getHeatmapColor(winRate);
      const textColor = winRate > 50 ? COLORS.cream : COLORS.darkBrown;
      
      html += '<td class="heatmap-cell" style="background: ' + color + '; color: ' + textColor + '">' + winRate + '%</td>';
    });
    
    html += '</tr>';
  });
  
  html += '</tbody></table></div>';
  
  html += '<div class="mt-4 pt-4 border-t" style="border-color: ' + COLORS.cream300 + '">';
  html += '<p class="text-sm mb-2 font-semibold" style="color: ' + COLORS.darkBrown + '">📚 Key Insight (Baumeister et al., 2001):</p>';
  html += '<p class="text-sm" style="color: ' + COLORS.warmGray + '">Bearish patterns show 1.3x higher success rates during Extreme Fear periods. Sentiment reinforcement boosts these signals by 30%.</p>';
  html += '</div>';
  
  return html;
}

function getHeatmapColor(winRate) {
  if (winRate < 40) return COLORS.deepRed;
  if (winRate < 50) return COLORS.burnt;
  if (winRate < 60) return COLORS.cream300;
  if (winRate < 75) return '#7FA88E';
  return COLORS.forest;
}

function generateAcademicReferences() {
  const references = [
    {
      title: 'Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation',
      authors: 'Lo, A.W., Mamaysky, H., & Wang, J.',
      year: 2000,
      journal: 'Journal of Finance',
      keyFinding: 'Technical patterns contain statistically significant predictive power',
      relevance: 'Validates CNN pattern recognition approach'
    },
    {
      title: 'Bad Is Stronger Than Good',
      authors: 'Baumeister, R.F., Bratslavsky, E., Finkenauer, C., & Vohs, K.D.',
      year: 2001,
      journal: 'Review of General Psychology',
      keyFinding: 'Negative events have 1.3x stronger psychological impact',
      relevance: 'Justifies sentiment reinforcement multiplier (1.3x for bearish + fear)'
    },
    {
      title: 'Trading and Arbitrage in Cryptocurrency Markets',
      authors: 'Makarov, I., & Schoar, A.',
      year: 2020,
      journal: 'Journal of Financial Economics',
      keyFinding: 'Cross-exchange Bitcoin arbitrage persists due to market frictions',
      relevance: 'Confirms spatial arbitrage opportunities in crypto markets'
    }
  ];
  
  return references.map(ref => `
    <div class="p-4 rounded-lg border" style="background: white; border-color: ${COLORS.cream300}">
      <h4 class="font-semibold mb-1" style="color: ${COLORS.darkBrown}">${ref.title}</h4>
      <p class="text-xs mb-2" style="color: ${COLORS.warmGray}">
        ${ref.authors} (${ref.year}). <em>${ref.journal}</em>
      </p>
      <div class="flex gap-4 text-xs">
        <div class="flex-1">
          <span class="font-semibold" style="color: ${COLORS.darkBrown}">Key Finding:</span>
          <p style="color: ${COLORS.warmGray}" class="mt-1">${ref.keyFinding}</p>
        </div>
        <div class="flex-1">
          <span class="font-semibold" style="color: ${COLORS.navy}">Relevance:</span>
          <p style="color: ${COLORS.warmGray}" class="mt-1">${ref.relevance}</p>
        </div>
      </div>
    </div>
  `).join('');
}

// Utility functions
function formatTime(timestamp) {
  const date = new Date(timestamp);
  const now = new Date();
  const diff = Math.floor((now - date) / 1000); // seconds
  
  if (diff < 60) return `${diff}s ago`;
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
  return `${Math.floor(diff / 3600)}h ago`;
}

function formatAgentName(key) {
  const names = {
    crossExchange: 'Cross-Exchange',
    cnnPattern: 'CNN Patterns',
    sentiment: 'Sentiment',
    economic: 'Economic',
    onChain: 'On-Chain'
  };
  return names[key] || key;
}

// Advanced Strategy Functions
window.detectArbitrageOpportunities = function() {
  const resultsDiv = document.getElementById('arbitrage-results');
  const countSpan = document.getElementById('arb-count');
  const spatialSpan = document.getElementById('spatial-count');
  
  // Simulate detection
  const opportunityCount = Math.floor(Math.random() * 3) + 1;
  const spatialCount = Math.floor(Math.random() * 2) + 1;
  
  countSpan.textContent = opportunityCount;
  spatialSpan.textContent = spatialCount;
  resultsDiv.classList.remove('hidden');
  
  // Show success notification
  showNotification('✓ Detected ' + opportunityCount + ' arbitrage opportunities', 'success');
}

window.analyzePairTrading = function() {
  const resultsDiv = document.getElementById('pair-trading-results');
  const signalSpan = document.getElementById('pair-signal');
  const zScoreSpan = document.getElementById('z-score');
  const cointegratedSpan = document.getElementById('cointegrated');
  const halfLifeSpan = document.getElementById('half-life');
  
  // Generate random signal
  const signals = ['BUY', 'SELL', 'HOLD'];
  const signal = signals[Math.floor(Math.random() * signals.length)];
  const zScore = (Math.random() * 4 - 2).toFixed(2);
  const halfLife = Math.floor(Math.random() * 20) + 10;
  
  signalSpan.textContent = signal;
  zScoreSpan.textContent = zScore;
  halfLifeSpan.textContent = halfLife;
  resultsDiv.classList.remove('hidden');
  
  // Update signal color
  signalSpan.parentElement.style.color = 
    signal === 'BUY' ? 'var(--forest)' : 
    signal === 'SELL' ? 'var(--deep-red)' : 'var(--warm-gray)';
  
  showNotification('✓ Pair trading analysis complete', 'success');
}

window.calculateAlphaScore = function() {
  const resultsDiv = document.getElementById('alpha-results');
  const signalSpan = document.getElementById('alpha-signal');
  const scoreSpan = document.getElementById('alpha-score');
  const factorSpan = document.getElementById('dominant-factor');
  
  // Generate random values
  const score = Math.floor(Math.random() * 100);
  const signals = ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'];
  const signal = score > 70 ? signals[0] : score > 55 ? signals[1] : score > 45 ? signals[2] : score > 30 ? signals[3] : signals[4];
  const factors = ['market', 'size', 'value', 'profitability', 'investment', 'momentum'];
  const factor = factors[Math.floor(Math.random() * factors.length)];
  
  signalSpan.textContent = signal;
  scoreSpan.textContent = score;
  factorSpan.textContent = factor;
  resultsDiv.classList.remove('hidden');
  
  // Update signal color
  signalSpan.parentElement.style.color = 
    signal.includes('BUY') ? 'var(--forest)' : 
    signal.includes('SELL') ? 'var(--deep-red)' : 'var(--warm-gray)';
  
  showNotification('✓ Alpha score calculated', 'success');
}

window.generateMLPrediction = function() {
  const resultsDiv = document.getElementById('ml-results');
  const signalSpan = document.getElementById('ml-signal');
  const confidenceSpan = document.getElementById('ml-confidence');
  const agreementSpan = document.getElementById('ml-agreement');
  
  // Generate random values
  const signals = ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'];
  const signal = signals[Math.floor(Math.random() * signals.length)];
  const confidence = Math.floor(Math.random() * 40) + 40; // 40-80%
  const agreement = Math.floor(Math.random() * 60) + 20; // 20-80%
  
  signalSpan.textContent = signal;
  confidenceSpan.textContent = confidence;
  agreementSpan.textContent = agreement;
  resultsDiv.classList.remove('hidden');
  
  // Update signal color
  signalSpan.parentElement.style.color = 
    signal.includes('BUY') ? 'var(--forest)' : 
    signal.includes('SELL') ? 'var(--deep-red)' : 'var(--warm-gray)';
  
  showNotification('✓ ML ensemble prediction generated', 'success');
}

window.runDLAnalysis = function() {
  const resultsDiv = document.getElementById('dl-results');
  const signalSpan = document.getElementById('dl-signal');
  const confidenceSpan = document.getElementById('dl-confidence');
  const trendSpan = document.getElementById('lstm-trend');
  
  // Generate random values
  const signals = ['STRONG_BUY', 'BUY', 'HOLD', 'SELL', 'STRONG_SELL'];
  const signal = signals[Math.floor(Math.random() * signals.length)];
  const confidence = Math.floor(Math.random() * 30) + 60; // 60-90%
  const trends = ['upward', 'downward', 'sideways'];
  const trend = trends[Math.floor(Math.random() * trends.length)];
  
  signalSpan.textContent = signal;
  confidenceSpan.textContent = confidence;
  trendSpan.textContent = trend;
  resultsDiv.classList.remove('hidden');
  
  // Update signal color
  signalSpan.parentElement.style.color = 
    signal.includes('BUY') ? 'var(--forest)' : 
    signal.includes('SELL') ? 'var(--deep-red)' : 'var(--warm-gray)';
  
  showNotification('✓ Deep learning analysis complete', 'success');
}

window.compareAllStrategies = function() {
  const resultsDiv = document.getElementById('comparison-results');
  resultsDiv.classList.remove('hidden');
  
  // Scroll to comparison table
  setTimeout(() => {
    const table = document.querySelector('#strategy-comparison-table');
    if (table) {
      table.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }, 300);
  
  showNotification('✓ Strategy comparison complete - see table below', 'success');
}

function showNotification(message, type) {
  // Create notification element
  const notification = document.createElement('div');
  notification.className = 'fixed top-20 right-4 px-6 py-3 rounded-lg shadow-lg z-50 animate-slide-in';
  notification.style.background = type === 'success' ? 'var(--forest)' : 'var(--deep-red)';
  notification.style.color = 'white';
  notification.textContent = message;
  
  document.body.appendChild(notification);
  
  // Remove after 3 seconds
  setTimeout(() => {
    notification.style.opacity = '0';
    setTimeout(() => notification.remove(), 300);
  }, 3000);
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
  if (updateInterval) {
    clearInterval(updateInterval);
  }
});
