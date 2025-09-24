/**
 * GOMNA HFT ARBITRAGE UI COMPONENTS
 * Advanced user interface for High Frequency Trading and Arbitrage strategies
 * Integrates with the existing HyperVision platform
 */

class HFTArbitrageUI {
    constructor(container, arbitrageEngine) {
        this.container = container;
        this.arbitrageEngine = arbitrageEngine;
        this.updateIntervals = new Map();
        this.charts = new Map();
        this.isInitialized = false;
        
        this.init();
    }

    async init() {
        console.log('🎯 Initializing HFT Arbitrage UI...');
        
        this.createMainLayout();
        this.setupEventListeners();
        this.startRealTimeUpdates();
        
        this.isInitialized = true;
        console.log('✅ HFT Arbitrage UI initialized');
    }

    createMainLayout() {
        this.container.innerHTML = `
            <div class="hft-arbitrage-dashboard">
                <!-- Strategy Control Panel -->
                <div class="strategy-control-panel">
                    <div class="panel-header">
                        <h3>🚀 HFT Arbitrage Strategies</h3>
                        <div class="status-indicators">
                            <span class="latency-indicator" id="latency-status">Latency: <span id="avg-latency">--</span>ms</span>
                            <span class="connection-indicator" id="connection-status">Exchanges: <span id="connected-count">0</span>/4</span>
                        </div>
                    </div>
                    
                    <div class="strategy-toggles">
                        <div class="strategy-toggle" data-strategy="cross_exchange">
                            <div class="toggle-switch">
                                <input type="checkbox" id="toggle-cross-exchange">
                                <label for="toggle-cross-exchange" class="toggle-label">
                                    <span class="toggle-slider"></span>
                                </label>
                            </div>
                            <div class="strategy-info">
                                <h4>Cross-Exchange Arbitrage</h4>
                                <p>Exploit price differences across exchanges</p>
                                <span class="profit-indicator">Profit: <span id="cross-exchange-profit">$0</span></span>
                            </div>
                        </div>
                        
                        <div class="strategy-toggle" data-strategy="statistical">
                            <div class="toggle-switch">
                                <input type="checkbox" id="toggle-statistical">
                                <label for="toggle-statistical" class="toggle-label">
                                    <span class="toggle-slider"></span>
                                </label>
                            </div>
                            <div class="strategy-info">
                                <h4>Statistical Arbitrage</h4>
                                <p>Mean reversion and pairs trading</p>
                                <span class="profit-indicator">Profit: <span id="statistical-profit">$0</span></span>
                            </div>
                        </div>
                        
                        <div class="strategy-toggle" data-strategy="news_based">
                            <div class="toggle-switch">
                                <input type="checkbox" id="toggle-news-based">
                                <label for="toggle-news-based" class="toggle-label">
                                    <span class="toggle-slider"></span>
                                </label>
                            </div>
                            <div class="strategy-info">
                                <h4>News-Based Arbitrage</h4>
                                <p>React to breaking news with FinBERT</p>
                                <span class="profit-indicator">Profit: <span id="news-based-profit">$0</span></span>
                            </div>
                        </div>
                        
                        <div class="strategy-toggle" data-strategy="index_arbitrage">
                            <div class="toggle-switch">
                                <input type="checkbox" id="toggle-index-arbitrage">
                                <label for="toggle-index-arbitrage" class="toggle-label">
                                    <span class="toggle-slider"></span>
                                </label>
                            </div>
                            <div class="strategy-info">
                                <h4>Index Arbitrage</h4>
                                <p>ETF vs basket constituent trading</p>
                                <span class="profit-indicator">Profit: <span id="index-arbitrage-profit">$0</span></span>
                            </div>
                        </div>
                        
                        <div class="strategy-toggle" data-strategy="latency_arbitrage">
                            <div class="toggle-switch">
                                <input type="checkbox" id="toggle-latency-arbitrage">
                                <label for="toggle-latency-arbitrage" class="toggle-label">
                                    <span class="toggle-slider"></span>
                                </label>
                            </div>
                            <div class="strategy-info">
                                <h4>Latency Arbitrage</h4>
                                <p>Ultra-low latency execution advantage</p>
                                <span class="profit-indicator">Profit: <span id="latency-arbitrage-profit">$0</span></span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Real-Time Opportunities Display -->
                <div class="opportunities-panel">
                    <div class="panel-header">
                        <h3>🎯 Active Opportunities</h3>
                        <div class="opportunity-stats">
                            <span>Found: <span id="opportunities-count">0</span></span>
                            <span>Executed: <span id="executed-count">0</span></span>
                            <span>Success Rate: <span id="success-rate">0%</span></span>
                        </div>
                    </div>
                    <div class="opportunities-list" id="opportunities-list">
                        <!-- Dynamic opportunity cards will be inserted here -->
                    </div>
                </div>

                <!-- Performance Dashboard -->
                <div class="performance-dashboard">
                    <div class="performance-metrics">
                        <div class="metric-card">
                            <h4>Latency Performance</h4>
                            <canvas id="latency-chart" width="300" height="200"></canvas>
                            <div class="metric-details">
                                <span>Avg: <span id="avg-latency-detail">--</span>ms</span>
                                <span>P95: <span id="p95-latency">--</span>ms</span>
                                <span>P99: <span id="p99-latency">--</span>ms</span>
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <h4>Profit & Loss</h4>
                            <canvas id="pnl-chart" width="300" height="200"></canvas>
                            <div class="metric-details">
                                <span class="pnl-total">Total: <span id="total-pnl">$0</span></span>
                                <span class="pnl-today">Today: <span id="today-pnl">$0</span></span>
                                <span class="pnl-hourly">Last Hour: <span id="hourly-pnl">$0</span></span>
                            </div>
                        </div>
                        
                        <div class="metric-card">
                            <h4>Execution Statistics</h4>
                            <canvas id="execution-chart" width="300" height="200"></canvas>
                            <div class="metric-details">
                                <span>Orders/Min: <span id="orders-per-minute">0</span></span>
                                <span>Fill Rate: <span id="fill-rate">0%</span></span>
                                <span>Slippage: <span id="avg-slippage">0</span>bps</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Sentiment & News Analysis -->
                <div class="sentiment-panel">
                    <div class="panel-header">
                        <h3>📰 Market Sentiment (FinBERT)</h3>
                        <div class="sentiment-summary">
                            <span class="sentiment-score">Overall: <span id="overall-sentiment">Neutral</span></span>
                            <span class="news-count">News Items: <span id="news-item-count">0</span></span>
                        </div>
                    </div>
                    <div class="sentiment-content">
                        <div class="news-feed" id="news-feed">
                            <!-- Real-time news analysis will appear here -->
                        </div>
                        <div class="social-sentiment" id="social-sentiment">
                            <!-- Social media sentiment analysis -->
                        </div>
                    </div>
                </div>

                <!-- Order Book Visualization -->
                <div class="orderbook-panel">
                    <div class="panel-header">
                        <h3>📊 Multi-Exchange Order Books</h3>
                        <select id="symbol-selector">
                            <option value="BTC/USD">BTC/USD</option>
                            <option value="ETH/USD">ETH/USD</option>
                            <option value="SOL/USD">SOL/USD</option>
                        </select>
                    </div>
                    <div class="orderbook-grid" id="orderbook-grid">
                        <!-- Order book visualizations for each exchange -->
                    </div>
                </div>

                <!-- Risk Management Controls -->
                <div class="risk-management-panel">
                    <div class="panel-header">
                        <h3>⚠️ Risk Management</h3>
                        <button id="emergency-stop" class="emergency-stop-btn">EMERGENCY STOP</button>
                    </div>
                    <div class="risk-controls">
                        <div class="risk-limit">
                            <label>Max Position Size ($)</label>
                            <input type="number" id="max-position-size" value="100000" min="1000" max="1000000">
                        </div>
                        <div class="risk-limit">
                            <label>Max Daily Loss ($)</label>
                            <input type="number" id="max-daily-loss" value="5000" min="100" max="50000">
                        </div>
                        <div class="risk-limit">
                            <label>Max Latency (ms)</label>
                            <input type="number" id="max-latency-limit" value="5" min="1" max="100">
                        </div>
                        <div class="risk-status">
                            <span class="risk-indicator">Risk Level: <span id="current-risk-level">Low</span></span>
                            <span class="exposure-indicator">Exposure: <span id="current-exposure">$0</span></span>
                        </div>
                    </div>
                </div>
            </div>
        `;

        this.initializeCharts();
        this.createOrderBookVisualizations();
    }

    setupEventListeners() {
        // Strategy toggle listeners
        document.querySelectorAll('.strategy-toggle input[type="checkbox"]').forEach(toggle => {
            toggle.addEventListener('change', (e) => {
                const strategy = e.target.id.replace('toggle-', '').replace('-', '_');
                
                if (e.target.checked) {
                    this.enableStrategy(strategy);
                } else {
                    this.disableStrategy(strategy);
                }
            });
        });

        // Emergency stop button
        document.getElementById('emergency-stop').addEventListener('click', () => {
            this.emergencyStop();
        });

        // Risk limit updates
        ['max-position-size', 'max-daily-loss', 'max-latency-limit'].forEach(id => {
            document.getElementById(id).addEventListener('change', (e) => {
                this.updateRiskLimits();
            });
        });

        // Symbol selector for order book
        document.getElementById('symbol-selector').addEventListener('change', (e) => {
            this.updateOrderBookDisplay(e.target.value);
        });
    }

    enableStrategy(strategyName) {
        console.log(`✅ Enabling strategy: ${strategyName}`);
        
        if (this.arbitrageEngine) {
            this.arbitrageEngine.enableStrategy(strategyName);
        }
        
        // Update UI to show strategy is active
        const strategyElement = document.querySelector(`[data-strategy="${strategyName}"]`);
        if (strategyElement) {
            strategyElement.classList.add('strategy-active');
        }

        // Show success notification
        this.showNotification(`${strategyName} strategy enabled`, 'success');
    }

    disableStrategy(strategyName) {
        console.log(`⏹️ Disabling strategy: ${strategyName}`);
        
        if (this.arbitrageEngine) {
            this.arbitrageEngine.disableStrategy(strategyName);
        }
        
        // Update UI to show strategy is inactive
        const strategyElement = document.querySelector(`[data-strategy="${strategyName}"]`);
        if (strategyElement) {
            strategyElement.classList.remove('strategy-active');
        }

        // Show notification
        this.showNotification(`${strategyName} strategy disabled`, 'info');
    }

    emergencyStop() {
        console.log('🚨 EMERGENCY STOP ACTIVATED');
        
        // Disable all strategies
        document.querySelectorAll('.strategy-toggle input[type="checkbox"]').forEach(toggle => {
            toggle.checked = false;
            const strategy = toggle.id.replace('toggle-', '').replace('-', '_');
            this.disableStrategy(strategy);
        });

        // Show critical alert
        this.showNotification('EMERGENCY STOP: All strategies disabled', 'error');
        
        // Update emergency stop button state
        const emergencyBtn = document.getElementById('emergency-stop');
        emergencyBtn.textContent = 'STOPPED';
        emergencyBtn.style.backgroundColor = '#ff4444';
        emergencyBtn.disabled = true;

        // Re-enable after 10 seconds
        setTimeout(() => {
            emergencyBtn.textContent = 'EMERGENCY STOP';
            emergencyBtn.style.backgroundColor = '';
            emergencyBtn.disabled = false;
        }, 10000);
    }

    updateRiskLimits() {
        const maxPosition = document.getElementById('max-position-size').value;
        const maxDailyLoss = document.getElementById('max-daily-loss').value;
        const maxLatency = document.getElementById('max-latency-limit').value;

        console.log('📊 Updating risk limits:', { maxPosition, maxDailyLoss, maxLatency });

        if (this.arbitrageEngine && this.arbitrageEngine.hftConfig) {
            this.arbitrageEngine.hftConfig.maxPositionSize = parseFloat(maxPosition);
            this.arbitrageEngine.hftConfig.maxLatencyMs = parseFloat(maxLatency);
        }

        this.showNotification('Risk limits updated', 'success');
    }

    startRealTimeUpdates() {
        // Update performance metrics every second
        this.updateIntervals.set('performance', setInterval(() => {
            this.updatePerformanceMetrics();
        }, 1000));

        // Update opportunities every 100ms for HFT
        this.updateIntervals.set('opportunities', setInterval(() => {
            this.updateOpportunities();
        }, 100));

        // Update sentiment analysis every 5 seconds
        this.updateIntervals.set('sentiment', setInterval(() => {
            this.updateSentimentAnalysis();
        }, 5000));

        // Update order books every 50ms
        this.updateIntervals.set('orderbook', setInterval(() => {
            this.updateOrderBooks();
        }, 50));

        // Update latency stats every 500ms
        this.updateIntervals.set('latency', setInterval(() => {
            this.updateLatencyStats();
        }, 500));
    }

    updatePerformanceMetrics() {
        if (!this.arbitrageEngine) return;

        try {
            const stats = this.arbitrageEngine.getArbitrageStats();
            
            // Update opportunity counts
            document.getElementById('opportunities-count').textContent = stats.totalOpportunities || 0;
            
            // Update strategy profits (placeholder calculation)
            const profitability = stats.profitabilityByStrategy || {};
            Object.entries(profitability).forEach(([strategy, data]) => {
                const profitElement = document.getElementById(`${strategy.replace('_', '-')}-profit`);
                if (profitElement && data.totalProfit) {
                    profitElement.textContent = `$${data.totalProfit.toFixed(2)}`;
                }
            });

            // Update charts
            this.updatePnLChart();
            this.updateExecutionChart();

        } catch (error) {
            console.error('Error updating performance metrics:', error);
        }
    }

    updateOpportunities() {
        if (!this.arbitrageEngine) return;

        try {
            const opportunities = Array.from(this.arbitrageEngine.activeOpportunities.values());
            const opportunitiesList = document.getElementById('opportunities-list');
            
            // Clear existing opportunities
            opportunitiesList.innerHTML = '';
            
            // Display recent opportunities (last 10)
            const recentOpportunities = opportunities
                .sort((a, b) => b.timestamp - a.timestamp)
                .slice(0, 10);

            recentOpportunities.forEach(opportunity => {
                const opportunityCard = this.createOpportunityCard(opportunity);
                opportunitiesList.appendChild(opportunityCard);
            });

            // Update executed count
            const executedCount = opportunities.filter(opp => opp.status === 'executed').length;
            document.getElementById('executed-count').textContent = executedCount;

            // Update success rate
            const successRate = opportunities.length > 0 ? 
                (executedCount / opportunities.length * 100).toFixed(1) : 0;
            document.getElementById('success-rate').textContent = `${successRate}%`;

        } catch (error) {
            console.error('Error updating opportunities:', error);
        }
    }

    createOpportunityCard(opportunity) {
        const card = document.createElement('div');
        card.className = `opportunity-card ${opportunity.status}`;
        
        const profitBps = opportunity.profit ? (opportunity.profit * 10000).toFixed(1) : 'N/A';
        const timeAgo = this.getTimeAgo(opportunity.timestamp);
        
        card.innerHTML = `
            <div class="opportunity-header">
                <span class="strategy-badge">${opportunity.strategyName}</span>
                <span class="profit-badge">${profitBps} bps</span>
            </div>
            <div class="opportunity-details">
                <div class="symbol">${opportunity.symbol || 'Multiple'}</div>
                <div class="action">${opportunity.action || opportunity.direction || 'N/A'}</div>
                <div class="size">$${(opportunity.size || opportunity.quantity * (opportunity.price || 100)).toLocaleString()}</div>
            </div>
            <div class="opportunity-footer">
                <span class="timestamp">${timeAgo}</span>
                <span class="status ${opportunity.status}">${opportunity.status.toUpperCase()}</span>
            </div>
        `;

        return card;
    }

    updateSentimentAnalysis() {
        if (!this.arbitrageEngine || !this.arbitrageEngine.sentimentAnalyzer) return;

        // Simulate sentiment data (would be real in production)
        const sentimentData = {
            overall: Math.random() > 0.5 ? 'Bullish' : Math.random() > 0.3 ? 'Bearish' : 'Neutral',
            newsItems: Math.floor(Math.random() * 50) + 10,
            recentNews: [
                {
                    title: "Federal Reserve signals dovish stance",
                    sentiment: 0.65,
                    confidence: 0.89,
                    impact: 'High',
                    timestamp: Date.now() - Math.random() * 300000
                },
                {
                    title: "Tech earnings beat expectations",
                    sentiment: 0.78,
                    confidence: 0.92,
                    impact: 'Medium',
                    timestamp: Date.now() - Math.random() * 600000
                }
            ]
        };

        // Update sentiment display
        document.getElementById('overall-sentiment').textContent = sentimentData.overall;
        document.getElementById('news-item-count').textContent = sentimentData.newsItems;

        // Update news feed
        const newsFeed = document.getElementById('news-feed');
        newsFeed.innerHTML = sentimentData.recentNews.map(news => `
            <div class="news-item ${news.sentiment > 0 ? 'bullish' : 'bearish'}">
                <div class="news-title">${news.title}</div>
                <div class="news-metrics">
                    <span class="sentiment">Sentiment: ${(news.sentiment * 100).toFixed(0)}%</span>
                    <span class="confidence">Confidence: ${(news.confidence * 100).toFixed(0)}%</span>
                    <span class="impact">Impact: ${news.impact}</span>
                </div>
                <div class="news-time">${this.getTimeAgo(news.timestamp)}</div>
            </div>
        `).join('');
    }

    updateOrderBooks() {
        const symbol = document.getElementById('symbol-selector').value;
        const orderBookGrid = document.getElementById('orderbook-grid');
        
        if (!orderBookGrid.hasChildNodes()) {
            this.createOrderBookVisualizations();
        }

        // Update each exchange's order book display
        ['binance', 'coinbase', 'kraken', 'okx'].forEach(exchange => {
            this.updateExchangeOrderBook(exchange, symbol);
        });
    }

    createOrderBookVisualizations() {
        const orderBookGrid = document.getElementById('orderbook-grid');
        orderBookGrid.innerHTML = '';

        ['binance', 'coinbase', 'kraken', 'okx'].forEach(exchange => {
            const orderBookCard = document.createElement('div');
            orderBookCard.className = 'orderbook-card';
            orderBookCard.innerHTML = `
                <div class="orderbook-header">
                    <h4>${exchange.toUpperCase()}</h4>
                    <span class="spread">Spread: <span id="${exchange}-spread">--</span></span>
                </div>
                <div class="orderbook-content" id="${exchange}-orderbook">
                    <div class="asks">
                        <div class="orderbook-side-header">ASKS</div>
                        <div class="orders" id="${exchange}-asks"></div>
                    </div>
                    <div class="spread-indicator" id="${exchange}-spread-indicator">
                        <span class="mid-price">$<span id="${exchange}-mid">--</span></span>
                    </div>
                    <div class="bids">
                        <div class="orderbook-side-header">BIDS</div>
                        <div class="orders" id="${exchange}-bids"></div>
                    </div>
                </div>
            `;
            orderBookGrid.appendChild(orderBookCard);
        });
    }

    updateExchangeOrderBook(exchange, symbol) {
        // Simulate order book data (would be real WebSocket data in production)
        const midPrice = 50000 + (Math.random() - 0.5) * 1000;
        const spread = 0.5 + Math.random() * 2;
        
        const bids = Array.from({ length: 5 }, (_, i) => ({
            price: midPrice - spread/2 - (i * 0.5),
            size: Math.random() * 10 + 0.1
        }));
        
        const asks = Array.from({ length: 5 }, (_, i) => ({
            price: midPrice + spread/2 + (i * 0.5),
            size: Math.random() * 10 + 0.1
        }));

        // Update display
        const bidsContainer = document.getElementById(`${exchange}-bids`);
        const asksContainer = document.getElementById(`${exchange}-asks`);
        const midPriceElement = document.getElementById(`${exchange}-mid`);
        const spreadElement = document.getElementById(`${exchange}-spread`);

        if (bidsContainer) {
            bidsContainer.innerHTML = bids.map(bid => `
                <div class="order-row bid">
                    <span class="price">${bid.price.toFixed(2)}</span>
                    <span class="size">${bid.size.toFixed(3)}</span>
                </div>
            `).join('');
        }

        if (asksContainer) {
            asksContainer.innerHTML = asks.map(ask => `
                <div class="order-row ask">
                    <span class="price">${ask.price.toFixed(2)}</span>
                    <span class="size">${ask.size.toFixed(3)}</span>
                </div>
            `).join('');
        }

        if (midPriceElement) {
            midPriceElement.textContent = midPrice.toFixed(2);
        }

        if (spreadElement) {
            spreadElement.textContent = `$${spread.toFixed(2)}`;
        }
    }

    updateLatencyStats() {
        if (!this.arbitrageEngine || !this.arbitrageEngine.executionLatencyTracker) return;

        try {
            const stats = this.arbitrageEngine.executionLatencyTracker.getStats();
            
            // Update latency indicators
            const avgLatency = stats.execute ? stats.execute.avg : 0;
            document.getElementById('avg-latency').textContent = avgLatency.toFixed(2);
            document.getElementById('avg-latency-detail').textContent = avgLatency.toFixed(2);
            
            if (stats.execute) {
                document.getElementById('p95-latency').textContent = stats.execute.p95.toFixed(2);
                document.getElementById('p99-latency').textContent = stats.execute.p99.toFixed(2);
            }

            // Update latency chart
            this.updateLatencyChart(stats);

            // Update connection status
            const connectedCount = this.arbitrageEngine.exchangeConnections ? 
                this.arbitrageEngine.exchangeConnections.size : 0;
            document.getElementById('connected-count').textContent = connectedCount;

        } catch (error) {
            console.error('Error updating latency stats:', error);
        }
    }

    initializeCharts() {
        // Initialize Chart.js charts for performance visualization
        const latencyCtx = document.getElementById('latency-chart');
        const pnlCtx = document.getElementById('pnl-chart');
        const executionCtx = document.getElementById('execution-chart');

        if (latencyCtx) {
            this.charts.set('latency', new Chart(latencyCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Average Latency (ms)',
                        data: [],
                        borderColor: '#00ff88',
                        backgroundColor: 'rgba(0, 255, 136, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 10
                        }
                    }
                }
            }));
        }

        if (pnlCtx) {
            this.charts.set('pnl', new Chart(pnlCtx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'P&L ($)',
                        data: [],
                        borderColor: '#ff6b6b',
                        backgroundColor: 'rgba(255, 107, 107, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            }));
        }

        if (executionCtx) {
            this.charts.set('execution', new Chart(executionCtx, {
                type: 'bar',
                data: {
                    labels: ['Success', 'Failed', 'Partial'],
                    datasets: [{
                        label: 'Executions',
                        data: [0, 0, 0],
                        backgroundColor: ['#4ecdc4', '#ff6b6b', '#ffe66d']
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false
                }
            }));
        }
    }

    updateLatencyChart(stats) {
        const chart = this.charts.get('latency');
        if (!chart || !stats.execute) return;

        const now = new Date().toLocaleTimeString();
        
        // Add new data point
        chart.data.labels.push(now);
        chart.data.datasets[0].data.push(stats.execute.avg);
        
        // Keep only last 20 data points
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        chart.update('none');
    }

    updatePnLChart() {
        const chart = this.charts.get('pnl');
        if (!chart) return;

        // Simulate P&L data (would be real calculations in production)
        const now = new Date().toLocaleTimeString();
        const randomPnL = (Math.random() - 0.4) * 1000; // Slight positive bias
        
        chart.data.labels.push(now);
        chart.data.datasets[0].data.push(randomPnL);
        
        // Keep only last 20 data points
        if (chart.data.labels.length > 20) {
            chart.data.labels.shift();
            chart.data.datasets[0].data.shift();
        }
        
        chart.update('none');

        // Update total P&L display
        const totalPnL = chart.data.datasets[0].data.reduce((sum, val) => sum + val, 0);
        document.getElementById('total-pnl').textContent = `$${totalPnL.toFixed(2)}`;
    }

    updateExecutionChart() {
        const chart = this.charts.get('execution');
        if (!chart) return;

        // Simulate execution data
        const success = Math.floor(Math.random() * 50) + 30;
        const failed = Math.floor(Math.random() * 10);
        const partial = Math.floor(Math.random() * 5);

        chart.data.datasets[0].data = [success, failed, partial];
        chart.update('none');

        // Update metrics
        const total = success + failed + partial;
        const fillRate = total > 0 ? ((success + partial) / total * 100).toFixed(1) : 0;
        document.getElementById('fill-rate').textContent = `${fillRate}%`;
        document.getElementById('orders-per-minute').textContent = total;
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.className = `hft-notification ${type}`;
        notification.innerHTML = `
            <span class="notification-icon">${this.getNotificationIcon(type)}</span>
            <span class="notification-message">${message}</span>
            <button class="notification-close">&times;</button>
        `;

        // Add to page
        document.body.appendChild(notification);

        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 5000);

        // Close button functionality
        notification.querySelector('.notification-close').addEventListener('click', () => {
            notification.parentNode.removeChild(notification);
        });
    }

    getNotificationIcon(type) {
        const icons = {
            success: '✅',
            error: '❌',
            warning: '⚠️',
            info: 'ℹ️'
        };
        return icons[type] || icons.info;
    }

    getTimeAgo(timestamp) {
        const seconds = Math.floor((Date.now() - timestamp) / 1000);
        
        if (seconds < 60) return `${seconds}s ago`;
        if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
        if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
        return `${Math.floor(seconds / 86400)}d ago`;
    }

    destroy() {
        // Clean up intervals and event listeners
        for (const interval of this.updateIntervals.values()) {
            clearInterval(interval);
        }
        this.updateIntervals.clear();

        // Destroy charts
        for (const chart of this.charts.values()) {
            chart.destroy();
        }
        this.charts.clear();
    }
}

// CSS Styles for HFT Arbitrage UI
const HFT_ARBITRAGE_CSS = `
    .hft-arbitrage-dashboard {
        display: grid;
        grid-template-columns: 1fr 1fr;
        grid-template-rows: auto auto auto;
        gap: 20px;
        padding: 20px;
        background: #1a1a1a;
        color: #ffffff;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    .strategy-control-panel {
        grid-column: 1 / -1;
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #4a5568;
    }

    .panel-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 20px;
        border-bottom: 1px solid #4a5568;
        padding-bottom: 15px;
    }

    .panel-header h3 {
        margin: 0;
        color: #00ff88;
        font-size: 1.4em;
        font-weight: 600;
    }

    .status-indicators {
        display: flex;
        gap: 20px;
        font-size: 0.9em;
    }

    .latency-indicator, .connection-indicator {
        background: rgba(0, 255, 136, 0.1);
        padding: 5px 10px;
        border-radius: 6px;
        border: 1px solid #00ff88;
    }

    .strategy-toggles {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 15px;
    }

    .strategy-toggle {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 15px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        border: 1px solid #4a5568;
        transition: all 0.3s ease;
    }

    .strategy-toggle:hover {
        border-color: #00ff88;
        background: rgba(0, 255, 136, 0.05);
    }

    .strategy-toggle.strategy-active {
        border-color: #00ff88;
        background: rgba(0, 255, 136, 0.1);
    }

    .toggle-switch {
        position: relative;
        width: 50px;
        height: 25px;
    }

    .toggle-switch input {
        display: none;
    }

    .toggle-label {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: #4a5568;
        border-radius: 25px;
        cursor: pointer;
        transition: background 0.3s;
    }

    .toggle-slider {
        position: absolute;
        top: 2px;
        left: 2px;
        width: 21px;
        height: 21px;
        background: #ffffff;
        border-radius: 50%;
        transition: transform 0.3s;
    }

    .toggle-switch input:checked + .toggle-label {
        background: #00ff88;
    }

    .toggle-switch input:checked + .toggle-label .toggle-slider {
        transform: translateX(25px);
    }

    .strategy-info h4 {
        margin: 0 0 5px 0;
        color: #ffffff;
        font-size: 1.1em;
    }

    .strategy-info p {
        margin: 0 0 8px 0;
        color: #a0aec0;
        font-size: 0.9em;
    }

    .profit-indicator {
        font-size: 0.9em;
        font-weight: 600;
        color: #00ff88;
    }

    .opportunities-panel {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #4a5568;
        max-height: 600px;
        overflow-y: auto;
    }

    .opportunity-stats {
        display: flex;
        gap: 15px;
        font-size: 0.9em;
        color: #a0aec0;
    }

    .opportunities-list {
        max-height: 500px;
        overflow-y: auto;
    }

    .opportunity-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
        border-left: 4px solid #4a5568;
        transition: all 0.3s ease;
    }

    .opportunity-card.executed {
        border-left-color: #00ff88;
        background: rgba(0, 255, 136, 0.05);
    }

    .opportunity-card:hover {
        background: rgba(255, 255, 255, 0.08);
    }

    .opportunity-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .strategy-badge {
        background: rgba(0, 255, 136, 0.2);
        color: #00ff88;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: 600;
    }

    .profit-badge {
        background: rgba(255, 107, 107, 0.2);
        color: #ff6b6b;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        font-weight: 600;
    }

    .opportunity-details {
        display: flex;
        justify-content: space-between;
        margin-bottom: 8px;
        font-size: 0.9em;
    }

    .opportunity-footer {
        display: flex;
        justify-content: space-between;
        font-size: 0.8em;
        color: #a0aec0;
    }

    .status.executed {
        color: #00ff88;
    }

    .performance-dashboard {
        grid-column: 1 / -1;
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #4a5568;
    }

    .performance-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
        gap: 20px;
    }

    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #4a5568;
    }

    .metric-card h4 {
        margin: 0 0 15px 0;
        color: #00ff88;
        font-size: 1.1em;
    }

    .metric-details {
        display: flex;
        justify-content: space-between;
        margin-top: 10px;
        font-size: 0.9em;
    }

    .sentiment-panel {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #4a5568;
    }

    .sentiment-content {
        display: grid;
        grid-template-columns: 2fr 1fr;
        gap: 20px;
    }

    .news-feed {
        max-height: 400px;
        overflow-y: auto;
    }

    .news-item {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 6px;
        padding: 10px;
        margin-bottom: 8px;
        border-left: 3px solid #4a5568;
    }

    .news-item.bullish {
        border-left-color: #00ff88;
    }

    .news-item.bearish {
        border-left-color: #ff6b6b;
    }

    .news-title {
        font-weight: 600;
        margin-bottom: 5px;
    }

    .news-metrics {
        display: flex;
        gap: 10px;
        font-size: 0.8em;
        color: #a0aec0;
        margin-bottom: 5px;
    }

    .orderbook-panel {
        background: linear-gradient(135deg, #2d3748 0%, #1a202c 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #4a5568;
    }

    .orderbook-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }

    .orderbook-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px;
        border: 1px solid #4a5568;
    }

    .orderbook-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
        padding-bottom: 8px;
        border-bottom: 1px solid #4a5568;
    }

    .orderbook-header h4 {
        margin: 0;
        color: #00ff88;
        font-size: 1em;
    }

    .orderbook-content {
        font-family: 'Courier New', monospace;
        font-size: 0.85em;
    }

    .orderbook-side-header {
        text-align: center;
        color: #a0aec0;
        font-weight: 600;
        margin-bottom: 5px;
        font-size: 0.9em;
    }

    .order-row {
        display: flex;
        justify-content: space-between;
        padding: 2px 0;
    }

    .order-row.ask {
        color: #ff6b6b;
    }

    .order-row.bid {
        color: #00ff88;
    }

    .spread-indicator {
        text-align: center;
        padding: 8px 0;
        margin: 5px 0;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 4px;
        color: #ffe66d;
        font-weight: 600;
    }

    .risk-management-panel {
        grid-column: 1 / -1;
        background: linear-gradient(135deg, #742a2a 0%, #2d1b1b 100%);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid #c53030;
    }

    .emergency-stop-btn {
        background: #c53030;
        color: white;
        border: none;
        padding: 10px 20px;
        border-radius: 6px;
        font-weight: 600;
        cursor: pointer;
        transition: background 0.3s;
    }

    .emergency-stop-btn:hover {
        background: #9c2626;
    }

    .risk-controls {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 15px;
        margin-top: 15px;
    }

    .risk-limit {
        display: flex;
        flex-direction: column;
        gap: 5px;
    }

    .risk-limit label {
        color: #fbb6ce;
        font-size: 0.9em;
        font-weight: 600;
    }

    .risk-limit input {
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid #c53030;
        border-radius: 4px;
        padding: 8px;
        color: white;
        font-size: 0.9em;
    }

    .risk-status {
        display: flex;
        flex-direction: column;
        gap: 5px;
        color: #fbb6ce;
        font-size: 0.9em;
    }

    .hft-notification {
        position: fixed;
        top: 20px;
        right: 20px;
        background: #2d3748;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid;
        display: flex;
        align-items: center;
        gap: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        z-index: 10000;
        min-width: 300px;
        animation: slideIn 0.3s ease;
    }

    .hft-notification.success {
        border-left-color: #00ff88;
    }

    .hft-notification.error {
        border-left-color: #ff6b6b;
    }

    .hft-notification.warning {
        border-left-color: #ffe66d;
    }

    .hft-notification.info {
        border-left-color: #4ecdc4;
    }

    .notification-close {
        background: none;
        border: none;
        color: #a0aec0;
        font-size: 1.2em;
        cursor: pointer;
        margin-left: auto;
    }

    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    /* Responsive design */
    @media (max-width: 1200px) {
        .hft-arbitrage-dashboard {
            grid-template-columns: 1fr;
        }
        
        .sentiment-content {
            grid-template-columns: 1fr;
        }
    }

    @media (max-width: 768px) {
        .strategy-toggles {
            grid-template-columns: 1fr;
        }
        
        .performance-metrics {
            grid-template-columns: 1fr;
        }
        
        .orderbook-grid {
            grid-template-columns: 1fr 1fr;
        }
    }

    @media (max-width: 480px) {
        .orderbook-grid {
            grid-template-columns: 1fr;
        }
        
        .hft-notification {
            left: 10px;
            right: 10px;
            min-width: auto;
        }
    }
`;

// Add CSS to document
if (typeof document !== 'undefined') {
    const styleSheet = document.createElement('style');
    styleSheet.textContent = HFT_ARBITRAGE_CSS;
    document.head.appendChild(styleSheet);
}

// Export for global usage
if (typeof window !== 'undefined') {
    window.HFTArbitrageUI = HFTArbitrageUI;
}

console.log('🎯 HFT Arbitrage UI loaded successfully');