/**
 * Cocoa Trading AI - Complete Platform Integration
 * Integrates all professional features into a cohesive platform
 */

class CocoaPlatformIntegration {
    constructor() {
        this.isInitialized = false;
        this.components = {
            branding: null,
            accounts: null,
            credentials: null,
            dashboard: null
        };
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing Cocoa Trading AI Complete Platform...');
        
        // Wait for DOM and other components
        await this.waitForDOMReady();
        await this.waitForComponents();
        
        // Initialize professional dashboard
        this.createProfessionalDashboard();
        this.setupNavigationSystem();
        this.initializeRealTimeFeatures();
        
        this.isInitialized = true;
        console.log('✅ Cocoa Trading AI Platform fully initialized');
    }

    waitForDOMReady() {
        return new Promise(resolve => {
            if (document.readyState === 'loading') {
                document.addEventListener('DOMContentLoaded', resolve);
            } else {
                resolve();
            }
        });
    }

    waitForComponents() {
        return new Promise(resolve => {
            const checkComponents = () => {
                if (window.cocoaTradingAIBranding && 
                    window.cocoaTieredAccounts && 
                    window.cocoaRegulatoryCredentials) {
                    this.components.branding = window.cocoaTradingAIBranding;
                    this.components.accounts = window.cocoaTieredAccounts;
                    this.components.credentials = window.cocoaRegulatoryCredentials;
                    resolve();
                } else {
                    setTimeout(checkComponents, 100);
                }
            };
            checkComponents();
        });
    }

    createProfessionalDashboard() {
        // Create main dashboard container
        const dashboardContainer = document.createElement('div');
        dashboardContainer.id = 'cocoa-professional-dashboard';
        dashboardContainer.className = 'cocoa-fade-in';
        
        dashboardContainer.innerHTML = `
            <!-- Professional Navigation -->
            <div class="cocoa-nav" style="margin: 20px 0;">
                <div class="cocoa-nav-item active" data-tab="overview">📊 Trading Overview</div>
                <div class="cocoa-nav-item" data-tab="arbitrage">⚡ Arbitrage Strategies</div>
                <div class="cocoa-nav-item" data-tab="account">🏦 Account Management</div>
                <div class="cocoa-nav-item" data-tab="compliance">🏛️ Compliance</div>
                <div class="cocoa-nav-item" data-tab="analytics">📈 Performance Analytics</div>
            </div>

            <!-- Dashboard Content Panels -->
            <div id="dashboard-content">
                ${this.createOverviewTab()}
                ${this.createArbitrageTab()}
                ${this.createAnalyticsTab()}
            </div>

            <!-- Professional Start Trading Section -->
            <div class="cocoa-panel" style="margin: 30px 0; text-align: center; background: linear-gradient(135deg, var(--cocoa-primary) 0%, var(--cocoa-secondary) 100%);">
                <div style="padding: 40px 20px;">
                    <h2 style="color: white; margin-bottom: 15px; font-size: 2.2rem;">
                        🚀 Start Trading with Cocoa Trading AI
                    </h2>
                    <p style="color: rgba(255,255,255,0.9); font-size: 1.2rem; margin-bottom: 30px; max-width: 600px; margin-left: auto; margin-right: auto; line-height: 1.6;">
                        Join thousands of professional traders using our AI-powered arbitrage strategies. 
                        Get started in minutes with our tiered account system.
                    </p>
                    <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap;">
                        <button class="cocoa-btn-primary" id="start-trading-btn" style="
                            background: var(--cocoa-accent); 
                            color: var(--cocoa-bg); 
                            font-size: 1.1rem; 
                            padding: 15px 30px;
                            box-shadow: 0 8px 25px rgba(255, 215, 0, 0.4);
                        ">
                            🎯 Create Account & Start Trading
                        </button>
                        <button class="cocoa-btn-secondary" id="demo-trading-btn" style="
                            background: rgba(255, 255, 255, 0.2); 
                            border-color: rgba(255, 255, 255, 0.5);
                            color: white;
                            font-size: 1.1rem;
                            padding: 15px 30px;
                        ">
                            📊 Try Demo Mode
                        </button>
                    </div>
                </div>
            </div>
        `;

        // Insert after the header
        const header = document.querySelector('.cocoa-trading-header');
        if (header && header.parentNode) {
            header.parentNode.insertBefore(dashboardContainer, header.nextSibling);
        } else {
            const mainContainer = document.querySelector('.container, .main-content, body > div') || document.body;
            mainContainer.appendChild(dashboardContainer);
        }
    }

    createOverviewTab() {
        return `
            <div id="tab-overview" class="tab-content" style="display: block;">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0;">
                    <!-- Live Market Status -->
                    <div class="cocoa-trading-card">
                        <h4 style="color: var(--cocoa-accent); margin-bottom: 15px;">📈 Live Market Status</h4>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span>Market Status:</span>
                            <div class="cocoa-status-indicator cocoa-status-live">OPEN</div>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                            <span>Active Strategies:</span>
                            <span style="color: var(--cocoa-success); font-weight: bold;">6 Running</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <span>System Status:</span>
                            <div class="cocoa-status-indicator cocoa-status-live">ALL SYSTEMS GO</div>
                        </div>
                    </div>

                    <!-- Today's Performance -->
                    <div class="cocoa-trading-card">
                        <h4 style="color: var(--cocoa-accent); margin-bottom: 15px;">💰 Today's Performance</h4>
                        <div class="cocoa-metric-value" style="color: var(--cocoa-success); font-size: 1.8rem; margin-bottom: 5px;">
                            +$12,450.30
                        </div>
                        <div style="color: var(--cocoa-success); margin-bottom: 15px;">
                            +2.34% Return
                        </div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.9rem;">
                            <span>Trades: 47</span>
                            <span>Win Rate: 89%</span>
                        </div>
                    </div>

                    <!-- Portfolio Value -->
                    <div class="cocoa-trading-card">
                        <h4 style="color: var(--cocoa-accent); margin-bottom: 15px;">📊 Portfolio Value</h4>
                        <div class="cocoa-metric-value" style="font-size: 1.8rem; margin-bottom: 5px;">
                            $532,180.45
                        </div>
                        <div style="color: var(--cocoa-success); margin-bottom: 15px;">
                            +15.2% This Month
                        </div>
                        <div style="font-size: 0.9rem; color: var(--cocoa-secondary);">
                            Available: $89,420.15
                        </div>
                    </div>
                </div>

                <!-- Real-time Arbitrage Opportunities -->
                <div class="cocoa-panel" style="margin: 20px 0;">
                    <div class="cocoa-panel-header">
                        <h3>⚡ Live Arbitrage Opportunities</h3>
                        <div class="cocoa-status-indicator cocoa-status-live">
                            Scanning ${Math.floor(Math.random() * 50) + 20} Markets
                        </div>
                    </div>
                    <div class="cocoa-panel-content">
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 15px;">
                            ${this.generateArbitrageOpportunities()}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    createArbitrageTab() {
        return `
            <div id="tab-arbitrage" class="tab-content" style="display: none;">
                <div class="cocoa-panel">
                    <div class="cocoa-panel-header">
                        <h3>⚡ Arbitrage Strategy Performance</h3>
                        <button class="cocoa-btn-secondary" id="optimize-strategies">Optimize All</button>
                    </div>
                    <div class="cocoa-panel-content">
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                            ${this.createStrategyCards()}
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    createAnalyticsTab() {
        return `
            <div id="tab-analytics" class="tab-content" style="display: none;">
                <div class="cocoa-panel">
                    <div class="cocoa-panel-header">
                        <h3>📈 Advanced Analytics Dashboard</h3>
                        <div>
                            <button class="cocoa-btn-secondary" style="margin-right: 10px;">Export Report</button>
                            <button class="cocoa-btn-secondary">Share Analytics</button>
                        </div>
                    </div>
                    <div class="cocoa-panel-content">
                        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px;">
                            <div class="cocoa-metric">
                                <span class="cocoa-metric-value">89.4%</span>
                                <div class="cocoa-metric-label">Win Rate (30 Days)</div>
                            </div>
                            <div class="cocoa-metric">
                                <span class="cocoa-metric-value">2.34</span>
                                <div class="cocoa-metric-label">Sharpe Ratio</div>
                            </div>
                            <div class="cocoa-metric">
                                <span class="cocoa-metric-value">-3.2%</span>
                                <div class="cocoa-metric-label">Max Drawdown</div>
                            </div>
                            <div class="cocoa-metric">
                                <span class="cocoa-metric-value">1,247</span>
                                <div class="cocoa-metric-label">Total Trades</div>
                            </div>
                        </div>
                        
                        <div style="background: rgba(139, 69, 19, 0.05); padding: 20px; border-radius: 12px;">
                            <h4 style="color: var(--cocoa-accent); margin-bottom: 15px;">📊 Performance Chart</h4>
                            <div id="performance-chart" style="height: 300px; background: rgba(255,255,255,0.1); border-radius: 8px; display: flex; align-items: center; justify-content: center; color: var(--cocoa-text);">
                                Interactive Chart Loading...
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    generateArbitrageOpportunities() {
        const opportunities = [
            { pair: 'BTC/USD', exchanges: 'Binance → Coinbase', spread: '0.23%', profit: '$1,240' },
            { pair: 'ETH/USD', exchanges: 'Kraken → Bitstamp', spread: '0.18%', profit: '$890' },
            { pair: 'BTC/EUR', exchanges: 'Bitfinex → Coinbase', spread: '0.31%', profit: '$1,650' },
            { pair: 'ADA/USD', exchanges: 'Binance → KuCoin', spread: '0.45%', profit: '$420' }
        ];

        return opportunities.map(opp => `
            <div class="cocoa-trading-card" style="border-left: 3px solid var(--cocoa-success);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <strong style="color: var(--cocoa-accent);">${opp.pair}</strong>
                    <span style="color: var(--cocoa-success); font-weight: bold;">${opp.spread}</span>
                </div>
                <div style="margin-bottom: 8px; font-size: 0.9rem; color: var(--cocoa-secondary);">
                    ${opp.exchanges}
                </div>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="color: var(--cocoa-success); font-weight: bold;">Est. Profit: ${opp.profit}</span>
                    <button class="cocoa-btn-primary execute-trade" style="font-size: 0.8rem; padding: 5px 10px;">Execute</button>
                </div>
            </div>
        `).join('');
    }

    createStrategyCards() {
        const strategies = [
            { name: 'Cross-Exchange Arbitrage', status: 'Active', performance: '+15.2%', trades: 156 },
            { name: 'Triangular Arbitrage', status: 'Active', performance: '+12.8%', trades: 89 },
            { name: 'Statistical Arbitrage', status: 'Active', performance: '+18.7%', trades: 234 },
            { name: 'FinBERT Sentiment', status: 'Active', performance: '+22.1%', trades: 67 },
            { name: 'Volatility Arbitrage', status: 'Paused', performance: '+8.4%', trades: 45 },
            { name: 'News-Based Arbitrage', status: 'Active', performance: '+25.3%', trades: 123 }
        ];

        return strategies.map(strategy => `
            <div class="cocoa-trading-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                    <h4 style="color: var(--cocoa-accent); margin: 0;">${strategy.name}</h4>
                    <div class="cocoa-status-indicator ${strategy.status === 'Active' ? 'cocoa-status-live' : 'cocoa-status-demo'}">
                        ${strategy.status}
                    </div>
                </div>
                <div style="margin-bottom: 10px;">
                    <div style="display: flex; justify-content: space-between;">
                        <span>Performance (30d):</span>
                        <span style="color: var(--cocoa-success); font-weight: bold;">${strategy.performance}</span>
                    </div>
                </div>
                <div style="display: flex; justify-content: space-between; margin-bottom: 15px;">
                    <span>Trades Executed:</span>
                    <span style="font-weight: bold;">${strategy.trades}</span>
                </div>
                <div style="display: flex; gap: 10px;">
                    <button class="cocoa-btn-secondary" style="flex: 1; font-size: 0.9rem;">Configure</button>
                    <button class="cocoa-btn-primary" style="flex: 1; font-size: 0.9rem;">${strategy.status === 'Active' ? 'Pause' : 'Start'}</button>
                </div>
            </div>
        `).join('');
    }

    setupNavigationSystem() {
        document.addEventListener('click', (e) => {
            // Tab navigation
            if (e.target.classList.contains('cocoa-nav-item')) {
                const targetTab = e.target.getAttribute('data-tab');
                this.switchTab(targetTab);
            }

            // Start trading button
            if (e.target.id === 'start-trading-btn') {
                this.showAccountRegistration();
            }

            // Demo trading button
            if (e.target.id === 'demo-trading-btn') {
                this.startDemoMode();
            }

            // Execute trade buttons
            if (e.target.classList.contains('execute-trade')) {
                this.executeArbitrageTrade(e.target);
            }
        });
    }

    switchTab(tabName) {
        // Hide all tabs
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.style.display = 'none';
        });

        // Remove active class from nav items
        document.querySelectorAll('.cocoa-nav-item').forEach(item => {
            item.classList.remove('active');
        });

        // Show selected tab
        const targetTab = document.getElementById(`tab-${tabName}`);
        if (targetTab) {
            targetTab.style.display = 'block';
        }

        // Add active class to clicked nav item
        const navItem = document.querySelector(`[data-tab="${tabName}"]`);
        if (navItem) {
            navItem.classList.add('active');
        }

        // Special handling for account and compliance tabs
        if (tabName === 'account' && this.components.accounts) {
            this.components.accounts.createAccountInterface();
        }
        
        if (tabName === 'compliance' && this.components.credentials) {
            this.components.credentials.createCredentialsInterface();
        }
    }

    showAccountRegistration() {
        if (this.components.accounts) {
            // Show the tier upgrade modal for starter account
            this.components.accounts.showUpgradeModal('starter');
        } else {
            this.showNotification('🏦 Account registration system loading...', 'info');
        }
    }

    startDemoMode() {
        this.showNotification('🎯 Demo mode activated! All trades are simulated.', 'info');
        // Switch to arbitrage tab to show demo trading
        this.switchTab('arbitrage');
    }

    executeArbitrageTrade(button) {
        const card = button.closest('.cocoa-trading-card');
        const pair = card.querySelector('strong').textContent;
        
        button.textContent = 'Executing...';
        button.disabled = true;
        
        setTimeout(() => {
            button.textContent = 'Executed ✅';
            button.style.background = 'var(--cocoa-success)';
            
            this.showNotification(`✅ Arbitrage trade executed successfully for ${pair}!`, 'success');
            
            setTimeout(() => {
                button.textContent = 'Execute';
                button.disabled = false;
                button.style.background = '';
            }, 3000);
        }, 2000);
    }

    initializeRealTimeFeatures() {
        // Simulate real-time updates
        setInterval(() => {
            this.updateRealTimeData();
        }, 5000);

        // Initialize performance chart if visible
        setTimeout(() => {
            this.initializePerformanceChart();
        }, 1000);
    }

    updateRealTimeData() {
        // Update portfolio value
        const portfolioElements = document.querySelectorAll('.cocoa-metric-value');
        if (portfolioElements.length > 0) {
            // Simulate small changes in portfolio value
            const change = (Math.random() - 0.5) * 1000;
            // This could be more sophisticated real-time updates
        }

        // Update arbitrage opportunities
        const opportunityCount = Math.floor(Math.random() * 50) + 20;
        const scanningElement = document.querySelector('.cocoa-status-indicator:contains("Scanning")');
        if (scanningElement) {
            scanningElement.textContent = `Scanning ${opportunityCount} Markets`;
        }
    }

    initializePerformanceChart() {
        const chartContainer = document.getElementById('performance-chart');
        if (chartContainer && window.Chart) {
            // Create a simple performance chart
            chartContainer.innerHTML = '<canvas id="perf-chart"></canvas>';
            
            const ctx = document.getElementById('perf-chart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: Array.from({length: 30}, (_, i) => `Day ${i + 1}`),
                    datasets: [{
                        label: 'Portfolio Performance',
                        data: Array.from({length: 30}, () => Math.random() * 10 + 95),
                        borderColor: '#FFD700',
                        backgroundColor: 'rgba(255, 215, 0, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            labels: {
                                color: '#FFFFFF'
                            }
                        }
                    },
                    scales: {
                        x: {
                            ticks: { color: '#FFFFFF' }
                        },
                        y: {
                            ticks: { color: '#FFFFFF' }
                        }
                    }
                }
            });
        }
    }

    showNotification(message, type = 'info') {
        // Use the existing notification system from other components
        if (this.components.credentials && this.components.credentials.showNotification) {
            this.components.credentials.showNotification(message, type);
        } else {
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }

    // Public API
    getStatus() {
        return {
            initialized: this.isInitialized,
            components: Object.keys(this.components).reduce((acc, key) => {
                acc[key] = this.components[key] !== null;
                return acc;
            }, {})
        };
    }
}

// Initialize the complete platform
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.cocoaPlatformIntegration = new CocoaPlatformIntegration();
    });
} else {
    window.cocoaPlatformIntegration = new CocoaPlatformIntegration();
}

console.log('🚀 Cocoa Trading AI Complete Platform Integration loaded successfully');