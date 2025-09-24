/**
 * GOMNA ALGORITHMIC MARKETPLACE UI
 * Revolutionary interface for buying, selling, and executing trades based on live algorithm performance
 * Real-time arbitrage strategy marketplace with performance-based execution
 */

class AlgorithmicMarketplaceUI {
    constructor() {
        this.marketplace = null;
        this.investorSystem = null; // Will be injected by platform manager
        this.publicAPIs = null;     // Simple public APIs (no auth needed)
        this.updateInterval = null;
        this.refreshRate = 1000; // 1 second updates
        this.isInitialized = false;
        
        this.init();
    }

    async init() {
        console.log('🎨 Initializing Algorithmic Marketplace UI...');
        
        // Wait for marketplace to be available
        if (typeof AlgorithmicMarketplace === 'undefined') {
            console.log('⏳ Waiting for AlgorithmicMarketplace...');
            setTimeout(() => this.init(), 500);
            return;
        }

        this.marketplace = new AlgorithmicMarketplace();
        this.setupUI();
        this.startRealTimeUpdates();
        this.isInitialized = true;
        
        console.log('✅ Algorithmic Marketplace UI ready');
    }

    setupUI() {
        // Create marketplace tab in existing platform
        this.createMarketplaceTab();
        
        // Setup event listeners
        this.setupEventListeners();
    }

    createMarketplaceTab() {
        // Add marketplace tab to existing tab system
        const tabContainer = document.querySelector('.tab-container') || this.createTabContainer();
        
        // Create marketplace tab button
        const marketplaceTab = document.createElement('button');
        marketplaceTab.className = 'tab-button px-6 py-3 rounded-lg bg-cream-100 hover:bg-cream-200 transition-colors duration-200 border border-cream-300';
        marketplaceTab.setAttribute('data-tab', 'marketplace');
        marketplaceTab.innerHTML = `
            <div class="flex items-center space-x-2">
                <div class="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span>🏪 Algorithm Marketplace</span>
            </div>
        `;
        
        tabContainer.appendChild(marketplaceTab);
        
        // Create marketplace content panel
        this.createMarketplaceContent();
    }

    createTabContainer() {
        // If no tab container exists, create one
        const container = document.createElement('div');
        container.className = 'tab-container flex space-x-2 mb-6';
        
        const mainContent = document.getElementById('main-content') || document.body;
        mainContent.insertBefore(container, mainContent.firstChild);
        
        return container;
    }

    createMarketplaceContent() {
        const contentContainer = document.getElementById('content-container') || this.createContentContainer();
        
        const marketplacePanel = document.createElement('div');
        marketplacePanel.id = 'marketplace-panel';
        marketplacePanel.className = 'tab-content hidden';
        marketplacePanel.innerHTML = this.getMarketplaceHTML();
        
        contentContainer.appendChild(marketplacePanel);
    }

    createContentContainer() {
        const container = document.createElement('div');
        container.id = 'content-container';
        container.className = 'content-container';
        
        const mainContent = document.getElementById('main-content') || document.body;
        mainContent.appendChild(container);
        
        return container;
    }

    getMarketplaceHTML() {
        return `
            <div class="algorithmic-marketplace gradient-bg min-h-screen p-6">
                <!-- Header Section -->
                <div class="text-center mb-8">
                    <h1 class="text-4xl font-bold text-cream-800 mb-4">🏪 Algorithmic Trading Marketplace</h1>
                    <p class="text-xl text-cream-600 max-w-4xl mx-auto">
                        Revolutionary platform where investors execute trades based on live algorithm performance. 
                        Buy premium arbitrage strategies and let AI generate profits in real-time.
                    </p>
                </div>

                <!-- User Account Status -->
                <div class="user-account-status glass-effect rounded-xl p-6 mb-8" id="user-account-status">
                    <div class="text-center py-8">
                        <div class="mb-6">
                            <h2 class="text-3xl font-bold text-cream-800 mb-2">🎯 Start Your Algorithmic Trading Journey</h2>
                            <p class="text-lg text-cream-600">Join thousands of investors using AI-powered trading strategies</p>
                        </div>
                        
                        <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
                            <div class="bg-green-50 p-4 rounded-lg border border-green-200">
                                <div class="text-2xl mb-2">🤖</div>
                                <div class="font-semibold text-green-800">6 Professional Algorithms</div>
                                <div class="text-sm text-green-600">From $299 to $1299</div>
                            </div>
                            <div class="bg-blue-50 p-4 rounded-lg border border-blue-200">
                                <div class="text-2xl mb-2">📊</div>
                                <div class="font-semibold text-blue-800">Real-Time Performance</div>
                                <div class="text-sm text-blue-600">Live market data feeds</div>
                            </div>
                            <div class="bg-purple-50 p-4 rounded-lg border border-purple-200">
                                <div class="text-2xl mb-2">💰</div>
                                <div class="font-semibold text-purple-800">$10,000 Starting Balance</div>
                                <div class="text-sm text-purple-600">Ready to invest immediately</div>
                            </div>
                        </div>
                        
                        <div class="space-y-3">
                            <button id="show-auth-panel" class="w-full md:w-auto px-8 py-4 bg-gradient-to-r from-blue-500 to-blue-600 text-white text-lg font-bold rounded-lg hover:from-blue-600 hover:to-blue-700 transition-colors shadow-lg">
                                🚀 Create Investor Account & Start Trading
                            </button>
                            <div class="text-sm text-cream-500">
                                Already have an account? <button class="text-blue-500 hover:underline" onclick="document.getElementById('show-auth-panel').click()">Sign In</button>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Portfolio Overview -->
                <div class="portfolio-overview glass-effect rounded-xl p-6 mb-8" id="portfolio-overview" style="display: none;">
                    <h2 class="text-2xl font-bold text-cream-800 mb-4">💼 Your Trading Portfolio</h2>
                    <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                        <div class="metric-card rounded-lg p-4">
                            <div class="text-sm text-cream-600">Balance</div>
                            <div class="text-2xl font-bold text-green-600" id="portfolio-balance">$10,000.00</div>
                        </div>
                        <div class="metric-card rounded-lg p-4">
                            <div class="text-sm text-cream-600">Total P&L</div>
                            <div class="text-2xl font-bold" id="portfolio-pnl">$0.00</div>
                        </div>
                        <div class="metric-card rounded-lg p-4">
                            <div class="text-sm text-cream-600">Active Positions</div>
                            <div class="text-2xl font-bold text-blue-600" id="active-positions">0</div>
                        </div>
                        <div class="metric-card rounded-lg p-4">
                            <div class="text-sm text-cream-600">Owned Algorithms</div>
                            <div class="text-2xl font-bold text-purple-600" id="owned-algorithms">0</div>
                        </div>
                    </div>
                </div>

                <!-- Algorithm Marketplace -->
                <div class="algorithm-marketplace glass-effect rounded-xl p-6 mb-8">
                    <h2 class="text-2xl font-bold text-cream-800 mb-4">🤖 Professional Trading Algorithms</h2>
                    <div class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6" id="algorithms-grid">
                        <!-- Algorithm cards will be dynamically populated -->
                    </div>
                </div>

                <!-- Live Trading Control Panel -->
                <div class="trading-controls glass-effect rounded-xl p-6 mb-8">
                    <h2 class="text-2xl font-bold text-cream-800 mb-4">🎛️ Live Trading Controls</h2>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                        <div class="bg-green-50 p-4 rounded-lg border border-green-200">
                            <h3 class="font-semibold text-green-800 mb-2">Auto-Trading</h3>
                            <label class="flex items-center space-x-2">
                                <input type="checkbox" id="auto-trading-toggle" class="text-green-600">
                                <span class="text-sm text-green-700">Enable automatic execution</span>
                            </label>
                        </div>
                        <div class="bg-blue-50 p-4 rounded-lg border border-blue-200">
                            <h3 class="font-semibold text-blue-800 mb-2">Risk Management</h3>
                            <div class="space-y-2">
                                <div class="flex justify-between text-sm">
                                    <span>Max Position Size:</span>
                                    <span id="max-position-display">$5,000</span>
                                </div>
                                <input type="range" id="max-position-slider" min="1000" max="50000" value="5000" step="1000" class="w-full">
                            </div>
                        </div>
                        <div class="bg-yellow-50 p-4 rounded-lg border border-yellow-200">
                            <h3 class="font-semibold text-yellow-800 mb-2">Performance Target</h3>
                            <div class="space-y-2">
                                <div class="flex justify-between text-sm">
                                    <span>Daily P&L Target:</span>
                                    <span id="daily-target-display">$500</span>
                                </div>
                                <input type="range" id="daily-target-slider" min="100" max="2000" value="500" step="100" class="w-full">
                            </div>
                        </div>
                    </div>
                    
                    <div class="flex justify-center space-x-4">
                        <button id="start-all-algorithms" class="px-6 py-3 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors font-medium">
                            ▶️ Start All Owned Algorithms
                        </button>
                        <button id="pause-all-algorithms" class="px-6 py-3 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 transition-colors font-medium">
                            ⏸️ Pause All Algorithms
                        </button>
                        <button id="emergency-stop-trading" class="px-6 py-3 bg-red-500 text-white rounded-lg hover:bg-red-600 transition-colors font-medium">
                            🛑 Emergency Stop
                        </button>
                    </div>
                </div>

                <!-- Live Trading Signals -->
                <div class="live-signals glass-effect rounded-xl p-6 mb-8">
                    <div class="flex justify-between items-center mb-4">
                        <h2 class="text-2xl font-bold text-cream-800">⚡ Live Trading Signals</h2>
                        <div class="flex items-center space-x-2">
                            <div class="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                            <span class="text-sm text-cream-600">Real-time updates every 1s</span>
                        </div>
                    </div>
                    
                    <!-- Signal Filters -->
                    <div class="mb-4 flex flex-wrap gap-2">
                        <button class="signal-filter px-3 py-1 text-xs bg-cream-200 hover:bg-cream-300 rounded-full transition-colors" data-filter="all">All</button>
                        <button class="signal-filter px-3 py-1 text-xs bg-green-100 hover:bg-green-200 text-green-700 rounded-full transition-colors" data-filter="ARBITRAGE">Arbitrage</button>
                        <button class="signal-filter px-3 py-1 text-xs bg-blue-100 hover:bg-blue-200 text-blue-700 rounded-full transition-colors" data-filter="PAIRS_TRADE">Pairs</button>
                        <button class="signal-filter px-3 py-1 text-xs bg-purple-100 hover:bg-purple-200 text-purple-700 rounded-full transition-colors" data-filter="NEWS_SENTIMENT">News</button>
                        <button class="signal-filter px-3 py-1 text-xs bg-red-100 hover:bg-red-200 text-red-700 rounded-full transition-colors" data-filter="LATENCY_ARBITRAGE">HFT</button>
                    </div>
                    
                    <div class="overflow-x-auto">
                        <table class="w-full text-sm">
                            <thead>
                                <tr class="border-b border-cream-300">
                                    <th class="text-left p-3">Time</th>
                                    <th class="text-left p-3">Algorithm</th>
                                    <th class="text-left p-3">Signal Type</th>
                                    <th class="text-left p-3">Asset</th>
                                    <th class="text-left p-3">Action</th>
                                    <th class="text-left p-3">Confidence</th>
                                    <th class="text-left p-3">Expected Profit</th>
                                    <th class="text-left p-3">Execute</th>
                                    <th class="text-left p-3">Status</th>
                                </tr>
                            </thead>
                            <tbody id="live-signals-table">
                                <!-- Live signals will be populated here -->
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Active Trades -->
                <div class="active-trades glass-effect rounded-xl p-6 mb-8">
                    <h2 class="text-2xl font-bold text-cream-800 mb-4">📈 Active Trades</h2>
                    <div class="overflow-x-auto">
                        <table class="w-full text-sm">
                            <thead>
                                <tr class="border-b border-cream-300">
                                    <th class="text-left p-3">Time</th>
                                    <th class="text-left p-3">Algorithm</th>
                                    <th class="text-left p-3">Asset</th>
                                    <th class="text-left p-3">Side</th>
                                    <th class="text-left p-3">Quantity</th>
                                    <th class="text-left p-3">Entry Price</th>
                                    <th class="text-left p-3">Current P&L</th>
                                    <th class="text-left p-3">Status</th>
                                </tr>
                            </thead>
                            <tbody id="active-trades-table">
                                <!-- Active trades will be populated here -->
                            </tbody>
                        </table>
                    </div>
                </div>

                <!-- Performance Dashboard -->
                <div class="performance-dashboard glass-effect rounded-xl p-6">
                    <h2 class="text-2xl font-bold text-cream-800 mb-4">📊 Real-Time Performance Dashboard</h2>
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div class="chart-container">
                            <h3 class="text-lg font-semibold mb-4">Portfolio P&L Chart</h3>
                            <canvas id="pnl-chart" width="400" height="200"></canvas>
                        </div>
                        <div class="chart-container">
                            <h3 class="text-lg font-semibold mb-4">Algorithm Performance</h3>
                            <canvas id="performance-chart" width="400" height="200"></canvas>
                        </div>
                    </div>
                </div>
            </div>
        `;
    }

    setupEventListeners() {
        // Tab switching
        document.addEventListener('click', (e) => {
            if (e.target.matches('[data-tab="marketplace"]') || e.target.closest('[data-tab="marketplace"]')) {
                this.showMarketplaceTab();
            }
        });

        // Show auth panel when requested
        document.addEventListener('click', (e) => {
            if (e.target.matches('#show-auth-panel')) {
                if (this.investorSystem) {
                    this.investorSystem.showAuthPanel('register');
                } else {
                    this.showNotification('Account system not yet loaded, please wait...', 'info');
                }
            }
        });

        // Algorithm purchase buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('.buy-algorithm-btn')) {
                // Check if user is logged in through investor system
                if (!this.investorSystem || !this.investorSystem.isLoggedIn()) {
                    e.preventDefault();
                    if (this.investorSystem) {
                        this.investorSystem.showAuthPanel('register');
                        this.showNotification('Please create an account to purchase algorithms', 'info');
                    } else {
                        this.showNotification('Account system loading...', 'info');
                    }
                    return;
                }
                
                const algorithmId = e.target.getAttribute('data-algorithm');
                this.purchaseAlgorithm(algorithmId);
            }
        });

        // Algorithm activation/deactivation
        document.addEventListener('click', (e) => {
            if (e.target.matches('.activate-algorithm-btn')) {
                const algorithmId = e.target.getAttribute('data-algorithm');
                this.toggleAlgorithm(algorithmId);
            }
        });

        // Trading control handlers
        document.addEventListener('click', (e) => {
            if (e.target.matches('#start-all-algorithms')) {
                this.startAllOwnedAlgorithms();
            } else if (e.target.matches('#pause-all-algorithms')) {
                this.pauseAllAlgorithms();
            } else if (e.target.matches('#emergency-stop-trading')) {
                this.emergencyStopTrading();
            }
        });

        // Manual signal execution
        document.addEventListener('click', (e) => {
            if (e.target.matches('.execute-signal-btn')) {
                const signalId = e.target.getAttribute('data-signal');
                this.executeSignalManually(signalId);
            }
        });

        // Signal filters
        document.addEventListener('click', (e) => {
            if (e.target.matches('.signal-filter')) {
                const filter = e.target.getAttribute('data-filter');
                this.filterSignals(filter);
                
                // Update active filter UI
                document.querySelectorAll('.signal-filter').forEach(btn => {
                    btn.classList.remove('bg-cream-300');
                    btn.classList.add('bg-cream-200');
                });
                e.target.classList.remove('bg-cream-200');
                e.target.classList.add('bg-cream-300');
            }
        });

        // Risk management controls
        document.addEventListener('input', (e) => {
            if (e.target.matches('#max-position-slider')) {
                const value = e.target.value;
                document.getElementById('max-position-display').textContent = `$${parseInt(value).toLocaleString()}`;
            } else if (e.target.matches('#daily-target-slider')) {
                const value = e.target.value;
                document.getElementById('daily-target-display').textContent = `$${parseInt(value).toLocaleString()}`;
            }
        });
    }

    showMarketplaceTab() {
        // Hide all other tabs
        document.querySelectorAll('.tab-content').forEach(tab => {
            tab.classList.add('hidden');
        });

        // Show marketplace tab
        const marketplacePanel = document.getElementById('marketplace-panel');
        if (marketplacePanel) {
            marketplacePanel.classList.remove('hidden');
        }

        // Update tab button states
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('bg-cream-300', 'border-cream-500');
            btn.classList.add('bg-cream-100', 'border-cream-300');
        });

        const marketplaceTabBtn = document.querySelector('[data-tab="marketplace"]');
        if (marketplaceTabBtn) {
            marketplaceTabBtn.classList.remove('bg-cream-100', 'border-cream-300');
            marketplaceTabBtn.classList.add('bg-cream-300', 'border-cream-500');
        }
    }

    startRealTimeUpdates() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }

        this.updateInterval = setInterval(() => {
            this.updateUI();
        }, this.refreshRate);
    }

    updateUI() {
        if (!this.marketplace || !this.isInitialized) return;

        this.updateUserAccountStatus();
        this.updatePortfolioOverview();
        this.updateAlgorithmsGrid();
        this.updateLiveSignals();
        this.updateActiveTrades();
        this.updateCharts();
    }

    updateUserAccountStatus() {
        const accountStatus = document.getElementById('user-account-status');
        const portfolioOverview = document.getElementById('portfolio-overview');
        
        if (this.investorSystem && this.investorSystem.isLoggedIn()) {
            // User is logged in - show portfolio, hide login prompt
            if (accountStatus) accountStatus.style.display = 'none';
            if (portfolioOverview) portfolioOverview.style.display = 'block';
            
            // Update marketplace with user's portfolio data
            const currentUser = this.investorSystem.getCurrentUser();
            if (currentUser) {
                this.marketplace.userPortfolio = {
                    balance: currentUser.portfolioValue || 10000,
                    ownedAlgorithms: new Set(currentUser.algorithmsOwned || []),
                    activePositions: new Map(),
                    totalPnL: 0,
                    trades: []
                };
            }
        } else {
            // User not logged in - show login prompt, hide portfolio
            if (accountStatus) accountStatus.style.display = 'block';
            if (portfolioOverview) portfolioOverview.style.display = 'none';
            
            // Reset marketplace to demo mode
            this.marketplace.userPortfolio = {
                balance: 0,
                ownedAlgorithms: new Set(),
                activePositions: new Map(),
                totalPnL: 0,
                trades: []
            };
        }
    }

    updatePortfolioOverview() {
        const portfolio = this.marketplace.getPortfolio();

        const balanceEl = document.getElementById('portfolio-balance');
        const pnlEl = document.getElementById('portfolio-pnl');
        const positionsEl = document.getElementById('active-positions');
        const algorithmsEl = document.getElementById('owned-algorithms');

        if (balanceEl) {
            balanceEl.textContent = `$${portfolio.balance.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
        }

        if (pnlEl) {
            const pnl = portfolio.totalPnL || 0;
            pnlEl.textContent = `$${pnl.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
            pnlEl.className = `text-2xl font-bold ${pnl >= 0 ? 'text-green-600' : 'text-red-600'}`;
        }

        if (positionsEl) {
            positionsEl.textContent = portfolio.activePositions.length;
        }

        if (algorithmsEl) {
            algorithmsEl.textContent = portfolio.ownedAlgorithms.length;
        }
    }

    updateAlgorithmsGrid() {
        const grid = document.getElementById('algorithms-grid');
        if (!grid) return;

        const algorithms = this.marketplace.getAllAlgorithms();
        const portfolio = this.marketplace.getPortfolio();

        grid.innerHTML = algorithms.map(algo => this.createAlgorithmCard(algo, portfolio)).join('');
    }

    createAlgorithmCard(algorithm, portfolio) {
        const isOwned = portfolio.ownedAlgorithms.some(owned => owned.id === algorithm.id);
        const isActive = algorithm.isActive;

        return `
            <div class="algorithm-card metric-card rounded-lg p-6 border-l-4 border-${this.getCategoryColor(algorithm.category)}-500">
                <div class="flex items-start justify-between mb-4">
                    <div>
                        <h3 class="text-lg font-bold text-cream-800">${algorithm.name}</h3>
                        <p class="text-sm text-cream-600">${algorithm.description}</p>
                        <div class="text-xs text-cream-500 mt-1">
                            v${algorithm.version} • by ${algorithm.creator}
                        </div>
                    </div>
                    <div class="text-right">
                        <div class="text-xl font-bold text-green-600">$${algorithm.price}</div>
                        <div class="text-xs text-cream-600">${algorithm.subscribers} subscribers</div>
                    </div>
                </div>

                <!-- Performance Metrics -->
                <div class="grid grid-cols-2 gap-3 mb-4">
                    <div class="text-center p-2 bg-cream-50 rounded">
                        <div class="text-sm font-semibold">${(algorithm.performance.winRate * 100).toFixed(1)}%</div>
                        <div class="text-xs text-cream-600">Win Rate</div>
                    </div>
                    <div class="text-center p-2 bg-cream-50 rounded">
                        <div class="text-sm font-semibold">${algorithm.performance.sharpeRatio.toFixed(2)}</div>
                        <div class="text-xs text-cream-600">Sharpe Ratio</div>
                    </div>
                    <div class="text-center p-2 bg-cream-50 rounded">
                        <div class="text-sm font-semibold">${(algorithm.performance.avgReturn * 100).toFixed(2)}%</div>
                        <div class="text-xs text-cream-600">Avg Return</div>
                    </div>
                    <div class="text-center p-2 bg-cream-50 rounded">
                        <div class="text-sm font-semibold">${algorithm.performance.trades}</div>
                        <div class="text-xs text-cream-600">Total Trades</div>
                    </div>
                </div>

                <!-- Features -->
                <div class="mb-4">
                    <div class="text-xs font-semibold text-cream-700 mb-2">Key Features:</div>
                    <div class="space-y-1">
                        ${algorithm.features.slice(0, 3).map(feature => 
                            `<div class="text-xs text-cream-600">• ${feature}</div>`
                        ).join('')}
                    </div>
                </div>

                <!-- Action Buttons -->
                <div class="flex space-x-2">
                    ${!isOwned ? `
                        <button class="buy-algorithm-btn flex-1 bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded text-sm font-medium transition-colors"
                                data-algorithm="${algorithm.id}">
                            💳 Purchase Algorithm
                        </button>
                    ` : `
                        <button class="activate-algorithm-btn flex-1 ${isActive ? 'bg-red-500 hover:bg-red-600' : 'bg-blue-500 hover:bg-blue-600'} text-white px-4 py-2 rounded text-sm font-medium transition-colors"
                                data-algorithm="${algorithm.id}">
                            ${isActive ? '⏹️ Deactivate' : '▶️ Activate'}
                        </button>
                    `}
                    <button class="bg-cream-200 hover:bg-cream-300 text-cream-700 px-4 py-2 rounded text-sm font-medium transition-colors">
                        📊 Details
                    </button>
                </div>

                ${isActive ? `
                    <div class="mt-3 p-2 bg-green-50 border border-green-200 rounded text-xs">
                        🟢 <strong>LIVE:</strong> Generating signals and executing trades automatically
                    </div>
                ` : ''}
            </div>
        `;
    }

    getCategoryColor(category) {
        const colors = {
            'arbitrage': 'green',
            'statistical': 'blue',
            'sentiment': 'purple',
            'index': 'yellow',
            'hft': 'red',
            'volatility': 'indigo'
        };
        return colors[category] || 'gray';
    }

    updateLiveSignals() {
        const table = document.getElementById('live-signals-table');
        if (!table) return;

        const signals = this.marketplace.getLiveSignals();
        const portfolio = this.marketplace.getPortfolio();
        const autoTradingEnabled = document.getElementById('auto-trading-toggle')?.checked || false;
        
        table.innerHTML = signals.slice(0, 10).map(signal => {
            const algorithm = this.marketplace.getAlgorithm(signal.algorithm);
            const isOwned = portfolio.ownedAlgorithms.some(owned => owned.id === signal.algorithm);
            
            return `
                <tr class="border-b border-cream-200 hover:bg-cream-50 ${!isOwned ? 'opacity-50' : ''}" data-signal-type="${signal.type}">
                    <td class="p-3">${new Date(signal.timestamp).toLocaleTimeString()}</td>
                    <td class="p-3">
                        <div class="font-medium text-xs">
                            ${algorithm?.name || signal.algorithm}
                            ${isOwned ? '<span class="ml-1 text-green-500">✓</span>' : '<span class="ml-1 text-gray-400">🔒</span>'}
                        </div>
                    </td>
                    <td class="p-3">
                        <span class="px-2 py-1 bg-${this.getSignalTypeColor(signal.type)}-100 text-${this.getSignalTypeColor(signal.type)}-700 rounded-full text-xs">
                            ${signal.type}
                        </span>
                    </td>
                    <td class="p-3">${signal.symbol || signal.asset1 || signal.etf || 'N/A'}</td>
                    <td class="p-3">
                        <span class="font-medium text-xs ${signal.action.includes('BUY') ? 'text-green-600' : 'text-red-600'}">
                            ${signal.action}
                        </span>
                    </td>
                    <td class="p-3">
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div class="bg-green-500 h-2 rounded-full" style="width: ${(signal.confidence * 100)}%"></div>
                        </div>
                        <div class="text-xs text-gray-500 mt-1">${(signal.confidence * 100).toFixed(0)}%</div>
                    </td>
                    <td class="p-3 font-medium text-green-600">$${(signal.expectedProfit || 0).toFixed(2)}</td>
                    <td class="p-3">
                        ${isOwned ? `
                            <button class="execute-signal-btn px-2 py-1 bg-blue-500 hover:bg-blue-600 text-white text-xs rounded transition-colors" 
                                    data-signal="${signal.id}" ${autoTradingEnabled ? 'disabled' : ''}>
                                ${autoTradingEnabled ? '⚙️ Auto' : '▶️ Execute'}
                            </button>
                        ` : `
                            <span class="text-xs text-gray-400">Need Algorithm</span>
                        `}
                    </td>
                    <td class="p-3">
                        <span class="px-2 py-1 ${isOwned ? 'bg-green-100 text-green-700' : 'bg-gray-100 text-gray-500'} rounded-full text-xs ${isOwned ? 'animate-pulse' : ''}">
                            ${isOwned ? (autoTradingEnabled ? 'AUTO' : 'MANUAL') : 'LOCKED'}
                        </span>
                    </td>
                </tr>
            `;
        }).join('');

        if (signals.length === 0) {
            table.innerHTML = `
                <tr>
                    <td colspan="9" class="p-8 text-center text-cream-500">
                        No live signals at the moment. Purchase and activate algorithms to start generating signals.
                    </td>
                </tr>
            `;
        }
    }

    startAllOwnedAlgorithms() {
        const portfolio = this.marketplace.getPortfolio();
        let activatedCount = 0;
        
        portfolio.ownedAlgorithms.forEach(algorithm => {
            if (!algorithm.isActive) {
                this.marketplace.activateAlgorithm(algorithm.id);
                activatedCount++;
            }
        });
        
        if (activatedCount > 0) {
            this.showNotification(`✅ Activated ${activatedCount} algorithms for live trading`, 'success');
        } else {
            this.showNotification(`ℹ️ All owned algorithms are already active`, 'info');
        }
    }

    pauseAllAlgorithms() {
        const algorithms = this.marketplace.getAllAlgorithms();
        let pausedCount = 0;
        
        algorithms.forEach(algorithm => {
            if (algorithm.isActive) {
                this.marketplace.deactivateAlgorithm(algorithm.id);
                pausedCount++;
            }
        });
        
        if (pausedCount > 0) {
            this.showNotification(`⏸️ Paused ${pausedCount} algorithms`, 'warning');
        } else {
            this.showNotification(`ℹ️ No active algorithms to pause`, 'info');
        }
    }

    emergencyStopTrading() {
        // Stop all algorithms
        const algorithms = this.marketplace.getAllAlgorithms();
        algorithms.forEach(algorithm => {
            if (algorithm.isActive) {
                this.marketplace.deactivateAlgorithm(algorithm.id);
            }
        });
        
        // Disable auto-trading
        const autoToggle = document.getElementById('auto-trading-toggle');
        if (autoToggle) autoToggle.checked = false;
        
        this.showNotification(`🛑 EMERGENCY STOP: All trading halted`, 'error');
    }

    executeSignalManually(signalId) {
        const signals = this.marketplace.getLiveSignals();
        const signal = signals.find(s => s.id === signalId);
        
        if (!signal) {
            this.showNotification(`❌ Signal not found or expired`, 'error');
            return;
        }
        
        // Execute the signal
        this.marketplace.executeAlgorithmTrade(signal.algorithm, signal);
        this.showNotification(`🎯 Manually executed ${signal.type} signal`, 'success');
    }

    filterSignals(filterType) {
        const rows = document.querySelectorAll('#live-signals-table tr[data-signal-type]');
        
        rows.forEach(row => {
            if (filterType === 'all' || row.getAttribute('data-signal-type') === filterType) {
                row.style.display = '';
            } else {
                row.style.display = 'none';
            }
        });
    }

    getSignalTypeColor(type) {
        const colors = {
            'ARBITRAGE': 'green',
            'PAIRS_TRADE': 'blue',
            'NEWS_SENTIMENT': 'purple',
            'INDEX_ARBITRAGE': 'yellow',
            'LATENCY_ARBITRAGE': 'red',
            'VOLATILITY_ARBITRAGE': 'indigo'
        };
        return colors[type] || 'gray';
    }

    updateActiveTrades() {
        const table = document.getElementById('active-trades-table');
        if (!table) return;

        const portfolio = this.marketplace.getPortfolio();
        const activeTrades = portfolio.activePositions;

        table.innerHTML = activeTrades.map(trade => `
            <tr class="border-b border-cream-200 hover:bg-cream-50">
                <td class="p-3">${new Date(trade.timestamp).toLocaleTimeString()}</td>
                <td class="p-3">
                    <div class="font-medium">${trade.algorithmName}</div>
                </td>
                <td class="p-3">${trade.signal.symbol || 'Multi-Asset'}</td>
                <td class="p-3">
                    <span class="font-medium ${trade.side === 'BUY' ? 'text-green-600' : 'text-red-600'}">
                        ${trade.side}
                    </span>
                </td>
                <td class="p-3">${trade.quantity}</td>
                <td class="p-3">$${trade.executionPrice.toFixed(2)}</td>
                <td class="p-3">
                    <span class="font-bold ${trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'}">
                        $${trade.pnl.toFixed(2)}
                    </span>
                </td>
                <td class="p-3">
                    <span class="px-2 py-1 bg-blue-100 text-blue-700 rounded-full text-xs animate-pulse">
                        ${trade.status}
                    </span>
                </td>
            </tr>
        `).join('');

        if (activeTrades.length === 0) {
            table.innerHTML = `
                <tr>
                    <td colspan="8" class="p-8 text-center text-cream-500">
                        No active trades. Purchase and activate algorithms to start trading.
                    </td>
                </tr>
            `;
        }
    }

    updateCharts() {
        // Update P&L chart with Chart.js
        this.updatePnLChart();
        
        // Update performance chart with Chart.js  
        this.updatePerformanceChart();
    }

    updatePnLChart() {
        const canvas = document.getElementById('pnl-chart');
        if (!canvas) return;

        const portfolio = this.marketplace.getPortfolio();
        
        // Initialize or update Chart.js P&L chart
        if (!this.pnlChart) {
            const ctx = canvas.getContext('2d');
            
            // Generate mock historical P&L data
            const labels = [];
            const pnlData = [];
            const now = Date.now();
            
            for (let i = 29; i >= 0; i--) {
                const time = new Date(now - i * 60000); // 1 minute intervals
                labels.push(time.toLocaleTimeString());
                
                // Simulate realistic P&L progression
                const baseValue = i === 29 ? 0 : pnlData[pnlData.length - 1] || 0;
                const change = (Math.random() - 0.45) * 50; // Slight positive bias
                pnlData.push(baseValue + change);
            }
            
            // Add current P&L
            pnlData[pnlData.length - 1] = portfolio.totalPnL;

            this.pnlChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Portfolio P&L',
                        data: pnlData,
                        borderColor: portfolio.totalPnL >= 0 ? '#10b981' : '#ef4444',
                        backgroundColor: portfolio.totalPnL >= 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.2
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: false,
                            grid: {
                                color: 'rgba(232, 220, 199, 0.3)'
                            },
                            ticks: {
                                callback: function(value) {
                                    return '$' + value.toFixed(0);
                                }
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(232, 220, 199, 0.3)'
                            }
                        }
                    },
                    animation: {
                        duration: 1000
                    }
                }
            });
        } else {
            // Update existing chart
            const newTime = new Date().toLocaleTimeString();
            this.pnlChart.data.labels.push(newTime);
            this.pnlChart.data.datasets[0].data.push(portfolio.totalPnL);
            
            // Keep only last 30 data points
            if (this.pnlChart.data.labels.length > 30) {
                this.pnlChart.data.labels.shift();
                this.pnlChart.data.datasets[0].data.shift();
            }
            
            // Update colors based on current P&L
            this.pnlChart.data.datasets[0].borderColor = portfolio.totalPnL >= 0 ? '#10b981' : '#ef4444';
            this.pnlChart.data.datasets[0].backgroundColor = portfolio.totalPnL >= 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)';
            
            this.pnlChart.update('none');
        }
    }

    updatePerformanceChart() {
        const canvas = document.getElementById('performance-chart');
        if (!canvas) return;

        const algorithms = this.marketplace.getAllAlgorithms();
        const portfolio = this.marketplace.getPortfolio();
        
        // Initialize or update Chart.js performance chart
        if (!this.performanceChart) {
            const ctx = canvas.getContext('2d');
            
            const labels = algorithms.map(algo => algo.name.replace(' Pro', '').replace(' Elite', '').replace(' Master', ''));
            const winRates = algorithms.map(algo => (algo.performance.winRate * 100).toFixed(1));
            const sharpeRatios = algorithms.map(algo => algo.performance.sharpeRatio.toFixed(2));
            const owned = algorithms.map(algo => portfolio.ownedAlgorithms.some(owned => owned.id === algo.id));

            this.performanceChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [{
                        label: 'Win Rate (%)',
                        data: winRates,
                        backgroundColor: algorithms.map(algo => {
                            const isOwned = portfolio.ownedAlgorithms.some(owned => owned.id === algo.id);
                            return isOwned ? 'rgba(34, 197, 94, 0.7)' : 'rgba(156, 163, 175, 0.7)';
                        }),
                        borderColor: algorithms.map(algo => {
                            const isOwned = portfolio.ownedAlgorithms.some(owned => owned.id === algo.id);
                            return isOwned ? 'rgba(34, 197, 94, 1)' : 'rgba(156, 163, 175, 1)';
                        }),
                        borderWidth: 1
                    }, {
                        label: 'Sharpe Ratio',
                        data: sharpeRatios,
                        backgroundColor: 'rgba(59, 130, 246, 0.5)',
                        borderColor: 'rgba(59, 130, 246, 1)',
                        borderWidth: 1,
                        yAxisID: 'y1'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: true,
                            position: 'top'
                        }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            title: {
                                display: true,
                                text: 'Win Rate (%)'
                            },
                            grid: {
                                color: 'rgba(232, 220, 199, 0.3)'
                            }
                        },
                        y1: {
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {
                                display: true,
                                text: 'Sharpe Ratio'
                            },
                            grid: {
                                drawOnChartArea: false,
                            },
                            min: 0,
                            max: 4
                        },
                        x: {
                            grid: {
                                color: 'rgba(232, 220, 199, 0.3)'
                            }
                        }
                    },
                    animation: {
                        duration: 1500
                    }
                }
            });
        } else {
            // Update chart data
            algorithms.forEach((algo, index) => {
                const isOwned = portfolio.ownedAlgorithms.some(owned => owned.id === algo.id);
                this.performanceChart.data.datasets[0].data[index] = (algo.performance.winRate * 100).toFixed(1);
                this.performanceChart.data.datasets[1].data[index] = algo.performance.sharpeRatio.toFixed(2);
                
                // Update colors based on ownership
                this.performanceChart.data.datasets[0].backgroundColor[index] = isOwned ? 'rgba(34, 197, 94, 0.7)' : 'rgba(156, 163, 175, 0.7)';
                this.performanceChart.data.datasets[0].borderColor[index] = isOwned ? 'rgba(34, 197, 94, 1)' : 'rgba(156, 163, 175, 1)';
            });
            
            this.performanceChart.update('none');
        }
    }

    purchaseAlgorithm(algorithmId) {
        const algorithm = this.marketplace.getAlgorithm(algorithmId);
        if (!algorithm) return;

        // Show purchase confirmation modal with payment options
        this.showPurchaseModal(algorithm);
    }

    showPurchaseModal(algorithm) {
        // Remove any existing modals
        document.querySelectorAll('.purchase-modal').forEach(modal => modal.remove());

        const modal = document.createElement('div');
        modal.className = 'purchase-modal fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50';
        modal.innerHTML = `
            <div class="bg-white rounded-xl shadow-2xl max-w-md w-full mx-4 p-6">
                <div class="text-center mb-6">
                    <h3 class="text-2xl font-bold text-gray-900 mb-2">Purchase Algorithm</h3>
                    <div class="text-lg font-semibold text-cream-700">${algorithm.name}</div>
                    <div class="text-3xl font-bold text-green-600 mt-2">$${algorithm.price.toLocaleString()}</div>
                </div>

                <div class="mb-6">
                    <h4 class="font-semibold text-gray-800 mb-3">Algorithm Features:</h4>
                    <ul class="space-y-1">
                        ${algorithm.features.map(feature => `<li class="text-sm text-gray-600">• ${feature}</li>`).join('')}
                    </ul>
                </div>

                <div class="mb-6">
                    <h4 class="font-semibold text-gray-800 mb-3">Performance Metrics:</h4>
                    <div class="grid grid-cols-2 gap-3">
                        <div class="bg-green-50 p-3 rounded">
                            <div class="text-sm text-gray-600">Win Rate</div>
                            <div class="font-bold text-green-600">${(algorithm.performance.winRate * 100).toFixed(1)}%</div>
                        </div>
                        <div class="bg-blue-50 p-3 rounded">
                            <div class="text-sm text-gray-600">Sharpe Ratio</div>
                            <div class="font-bold text-blue-600">${algorithm.performance.sharpeRatio.toFixed(2)}</div>
                        </div>
                    </div>
                </div>

                <div class="mb-6">
                    <h4 class="font-semibold text-gray-800 mb-3">Payment Method:</h4>
                    <div class="space-y-3">
                        <label class="flex items-center space-x-3 p-3 border rounded cursor-pointer hover:bg-gray-50">
                            <input type="radio" name="payment" value="balance" checked class="text-blue-600">
                            <div>
                                <div class="font-medium">Account Balance</div>
                                <div class="text-sm text-gray-500">Current balance: $${this.marketplace.getPortfolio().balance.toLocaleString()}</div>
                            </div>
                        </label>
                        <label class="flex items-center space-x-3 p-3 border rounded cursor-pointer hover:bg-gray-50">
                            <input type="radio" name="payment" value="card" class="text-blue-600">
                            <div>
                                <div class="font-medium">Credit Card</div>
                                <div class="text-sm text-gray-500">Instant purchase with card</div>
                            </div>
                        </label>
                        <label class="flex items-center space-x-3 p-3 border rounded cursor-pointer hover:bg-gray-50">
                            <input type="radio" name="payment" value="crypto" class="text-blue-600">
                            <div>
                                <div class="font-medium">Cryptocurrency</div>
                                <div class="text-sm text-gray-500">Pay with BTC, ETH, or USDC</div>
                            </div>
                        </label>
                    </div>
                </div>

                <div class="flex space-x-3">
                    <button class="cancel-purchase flex-1 px-4 py-2 bg-gray-200 text-gray-800 rounded-lg hover:bg-gray-300 transition-colors">
                        Cancel
                    </button>
                    <button class="confirm-purchase flex-1 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors">
                        🛒 Purchase Algorithm
                    </button>
                </div>
            </div>
        `;

        document.body.appendChild(modal);

        // Event listeners
        modal.querySelector('.cancel-purchase').addEventListener('click', () => {
            modal.remove();
        });

        modal.querySelector('.confirm-purchase').addEventListener('click', () => {
            const paymentMethod = modal.querySelector('input[name="payment"]:checked').value;
            this.processPurchase(algorithm.id, paymentMethod);
            modal.remove();
        });

        // Close on backdrop click
        modal.addEventListener('click', (e) => {
            if (e.target === modal) {
                modal.remove();
            }
        });
    }

    processPurchase(algorithmId, paymentMethod) {
        this.showNotification(`💳 Processing payment via ${paymentMethod}...`, 'info');

        // Simulate payment processing
        setTimeout(() => {
            if (paymentMethod === 'balance') {
                // Use investor system for purchase validation
                if (this.investorSystem) {
                    const algorithm = this.marketplace.getAlgorithm(algorithmId);
                    const investorResult = this.investorSystem.purchaseAlgorithm(algorithmId, algorithm.price);
                    
                    if (investorResult.success) {
                        // Update marketplace
                        const result = this.marketplace.buyAlgorithm(algorithmId);
                        
                        if (result.success) {
                            this.showNotification(`✅ ${result.message}`, 'success');
                            this.showNotification(`🎯 Algorithm activated and ready for trading!`, 'success');
                            
                            // Auto-activate the purchased algorithm
                            setTimeout(() => {
                                this.marketplace.activateAlgorithm(algorithmId);
                                
                                // Notify about live data connection
                                setTimeout(() => {
                                    if (this.publicAPIs && this.publicAPIs.isConnected()) {
                                        this.showNotification('📡 Algorithm now using live market data feeds!', 'success');
                                    } else {
                                        this.showNotification('⚡ Algorithm activated with demo data', 'info');
                                    }
                                }, 2000);
                            }, 1000);
                        } else {
                            this.showNotification(`❌ ${result.error}`, 'error');
                        }
                    } else {
                        this.showNotification(`❌ ${investorResult.error}`, 'error');
                    }
                } else {
                    this.showNotification('❌ Account system not available', 'error');
                }
            } else {
                // Simulate external payment processing
                setTimeout(() => {
                    const algorithm = this.marketplace.getAlgorithm(algorithmId);
                    
                    // Update investor account
                    if (this.investorSystem && this.investorSystem.getCurrentUser()) {
                        const user = this.investorSystem.getCurrentUser();
                        user.portfolioValue += algorithm.price; // Add funds
                        
                        const investorResult = this.investorSystem.purchaseAlgorithm(algorithmId, algorithm.price);
                        if (investorResult.success) {
                            // Update marketplace
                            this.marketplace.userPortfolio.balance += algorithm.price;
                            const result = this.marketplace.buyAlgorithm(algorithmId);
                            
                            if (result.success) {
                                this.showNotification(`✅ Payment successful! ${result.message}`, 'success');
                                this.showNotification(`🎯 Algorithm activated and ready for trading!`, 'success');
                                
                                // Auto-activate the purchased algorithm
                                setTimeout(() => {
                                    this.marketplace.activateAlgorithm(algorithmId);
                                }, 1000);
                            }
                        }
                    }
                }, 2000);
            }
        }, 1500);
    }

    toggleAlgorithm(algorithmId) {
        const algorithm = this.marketplace.getAlgorithm(algorithmId);
        
        if (algorithm.isActive) {
            const result = this.marketplace.deactivateAlgorithm(algorithmId);
            if (result.success) {
                this.showNotification(`⏹️ ${result.message}`, 'info');
            }
        } else {
            const result = this.marketplace.activateAlgorithm(algorithmId);
            if (result.success) {
                this.showNotification(`▶️ ${result.message}`, 'success');
            } else {
                this.showNotification(`❌ ${result.error}`, 'error');
            }
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `fixed top-4 right-4 p-4 rounded-lg shadow-lg z-50 ${this.getNotificationClass(type)}`;
        notification.textContent = message;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }

    getNotificationClass(type) {
        const classes = {
            'success': 'bg-green-500 text-white',
            'error': 'bg-red-500 text-white',
            'info': 'bg-blue-500 text-white',
            'warning': 'bg-yellow-500 text-black'
        };
        return classes[type] || classes.info;
    }

    destroy() {
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        this.isInitialized = false;
    }
}

// Export for global usage
if (typeof window !== 'undefined') {
    window.AlgorithmicMarketplaceUI = AlgorithmicMarketplaceUI;
}

console.log('🎨 Algorithmic Marketplace UI loaded successfully');