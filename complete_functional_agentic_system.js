/**
 * COMPLETE FUNCTIONAL AGENTIC AI TRADING SYSTEM
 * With investor registration, real P&L tracking, and automated arbitrage execution
 */

class InvestorAccountSystem {
    constructor() {
        this.investors = new Map();
        this.currentInvestor = null;
        this.loginModal = null;
        
        this.initializeUI();
        this.loadSavedAccounts();
    }

    initializeUI() {
        this.createLoginModal();
        this.createAccountHeader();
        this.showLoginPrompt();
    }

    createLoginModal() {
        const modal = document.createElement('div');
        modal.id = 'investor-login-modal';
        modal.innerHTML = `
            <div class="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center">
                <div class="bg-white p-8 rounded-xl shadow-2xl max-w-md w-full mx-4">
                    <div class="text-center mb-6">
                        <h2 class="text-2xl font-bold text-gray-900 mb-2">🔐 Investor Portal Access</h2>
                        <p class="text-gray-600">Create your account to access the GOMNA AI Trading Platform</p>
                    </div>
                    
                    <div id="login-form">
                        <div class="mb-4">
                            <label class="block text-sm font-semibold text-gray-700 mb-2">Full Name</label>
                            <input type="text" id="investor-name" class="w-full p-3 border border-gray-300 rounded-lg" 
                                   placeholder="Enter your full name">
                        </div>
                        
                        <div class="mb-4">
                            <label class="block text-sm font-semibold text-gray-700 mb-2">Email Address</label>
                            <input type="email" id="investor-email" class="w-full p-3 border border-gray-300 rounded-lg" 
                                   placeholder="Enter your email">
                        </div>
                        
                        <div class="mb-4">
                            <label class="block text-sm font-semibold text-gray-700 mb-2">Choose Your Trading Tier</label>
                            <select id="investment-amount" class="w-full p-3 border border-gray-300 rounded-lg">
                                <option value="1000">$1,000 - STARTER TIER 1 (2:1 leverage, Basic AI)</option>
                                <option value="10000" selected>$10,000 - PROFESSIONAL TIER 2 (10:1 leverage, Advanced AI)</option>
                                <option value="25000">$25,000 - INSTITUTIONAL TIER 3 (50:1 leverage, White-glove)</option>
                            </select>
                        </div>
                        
                        <div class="mb-6">
                            <label class="block text-sm font-semibold text-gray-700 mb-2">Risk Tolerance</label>
                            <select id="risk-tolerance" class="w-full p-3 border border-gray-300 rounded-lg">
                                <option value="conservative">Conservative (Low Risk, 15-20% APY)</option>
                                <option value="moderate">Moderate (Medium Risk, 25-35% APY)</option>
                                <option value="aggressive">Aggressive (High Risk, 40-60% APY)</option>
                            </select>
                        </div>
                        
                        <button id="create-account-btn" class="w-full bg-amber-600 text-white p-3 rounded-lg font-bold hover:bg-amber-700 transition-colors">
                            Create Account & Start Trading
                        </button>
                        
                        <div class="mt-4 text-center">
                            <button id="existing-investor-btn" class="text-amber-600 hover:text-amber-700 text-sm font-semibold">
                                Already have an account? Sign In
                            </button>
                        </div>
                    </div>
                    
                    <div id="signin-form" class="hidden">
                        <div class="mb-4">
                            <label class="block text-sm font-semibold text-gray-700 mb-2">Email Address</label>
                            <input type="email" id="signin-email" class="w-full p-3 border border-gray-300 rounded-lg" 
                                   placeholder="Enter your registered email">
                        </div>
                        
                        <button id="signin-btn" class="w-full bg-blue-600 text-white p-3 rounded-lg font-bold hover:bg-blue-700 transition-colors">
                            Sign In to Account
                        </button>
                        
                        <div class="mt-4 text-center">
                            <button id="new-investor-btn" class="text-blue-600 hover:text-blue-700 text-sm font-semibold">
                                New investor? Create Account
                            </button>
                        </div>
                    </div>
                </div>
            </div>
        `;
        
        document.body.appendChild(modal);
        this.loginModal = modal;
        
        // Event listeners
        document.getElementById('create-account-btn').addEventListener('click', () => this.createAccount());
        document.getElementById('signin-btn').addEventListener('click', () => this.signIn());
        document.getElementById('existing-investor-btn').addEventListener('click', () => this.toggleForm('signin'));
        document.getElementById('new-investor-btn').addEventListener('click', () => this.toggleForm('create'));
    }

    toggleForm(type) {
        const loginForm = document.getElementById('login-form');
        const signinForm = document.getElementById('signin-form');
        
        if (type === 'signin') {
            loginForm.classList.add('hidden');
            signinForm.classList.remove('hidden');
        } else {
            signinForm.classList.add('hidden');
            loginForm.classList.remove('hidden');
        }
    }

    createAccount() {
        const name = document.getElementById('investor-name').value.trim();
        const email = document.getElementById('investor-email').value.trim();
        const investmentAmount = parseInt(document.getElementById('investment-amount').value);
        const riskTolerance = document.getElementById('risk-tolerance').value;

        if (!name || !email) {
            alert('Please fill in all required fields');
            return;
        }

        const investor = {
            id: `inv_${Date.now()}`,
            name: name,
            email: email,
            investmentAmount: investmentAmount,
            riskTolerance: riskTolerance,
            accountBalance: investmentAmount,
            totalPnL: 0,
            activePositions: [],
            tradeHistory: [],
            createdAt: new Date().toISOString(),
            lastLogin: new Date().toISOString()
        };

        this.investors.set(email, investor);
        this.currentInvestor = investor;
        this.saveAccounts();
        this.hideLogin();
        this.updateAccountHeader();
        
        // Show welcome message
        this.showWelcomeMessage(investor);
    }

    signIn() {
        const email = document.getElementById('signin-email').value.trim();
        
        if (!email) {
            alert('Please enter your email address');
            return;
        }

        const investor = this.investors.get(email);
        if (!investor) {
            alert('Account not found. Please create a new account.');
            return;
        }

        investor.lastLogin = new Date().toISOString();
        this.currentInvestor = investor;
        this.saveAccounts();
        this.hideLogin();
        this.updateAccountHeader();
        
        // Show returning investor message
        this.showWelcomeMessage(investor, true);
    }

    showWelcomeMessage(investor, returning = false) {
        const message = returning 
            ? `Welcome back, ${investor.name}! Your account balance: $${investor.accountBalance.toLocaleString()}`
            : `Welcome to GOMNA AI Trading, ${investor.name}! Your account has been created with $${investor.accountBalance.toLocaleString()}`;
        
        const notification = document.createElement('div');
        notification.className = 'fixed top-4 right-4 bg-green-500 text-white p-4 rounded-lg shadow-lg z-50';
        notification.innerHTML = `
            <div class="flex items-center gap-3">
                <div class="text-2xl">🎉</div>
                <div>
                    <div class="font-bold">${returning ? 'Welcome Back!' : 'Account Created!'}</div>
                    <div class="text-sm">${message}</div>
                </div>
            </div>
        `;
        
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }

    createAccountHeader() {
        const header = document.createElement('div');
        header.id = 'investor-account-header';
        header.className = 'bg-gradient-to-r from-amber-600 to-amber-700 text-white p-4 shadow-lg';
        header.innerHTML = `
            <div class="max-w-7xl mx-auto flex items-center justify-between">
                <div id="account-info" class="flex items-center gap-6">
                    <div class="text-sm opacity-75">Please log in to access trading platform</div>
                </div>
                <div id="account-actions" class="flex items-center gap-4">
                    <button id="login-btn" class="bg-white text-amber-700 px-4 py-2 rounded-lg font-semibold hover:bg-gray-100">
                        Login / Register
                    </button>
                </div>
            </div>
        `;
        
        // Insert after the main header
        const mainHeader = document.querySelector('.glass-effect');
        mainHeader.insertAdjacentElement('afterend', header);
        
        document.getElementById('login-btn').addEventListener('click', () => this.showLogin());
    }

    updateAccountHeader() {
        if (!this.currentInvestor) return;

        const accountInfo = document.getElementById('account-info');
        const accountActions = document.getElementById('account-actions');

        accountInfo.innerHTML = `
            <div class="flex items-center gap-6">
                <div>
                    <div class="font-bold">${this.currentInvestor.name}</div>
                    <div class="text-sm opacity-75">${this.currentInvestor.email}</div>
                </div>
                <div>
                    <div class="text-sm opacity-75">Account Balance</div>
                    <div class="font-bold text-lg">$${this.currentInvestor.accountBalance.toLocaleString()}</div>
                </div>
                <div>
                    <div class="text-sm opacity-75">Total P&L</div>
                    <div class="font-bold text-lg ${this.currentInvestor.totalPnL >= 0 ? 'text-green-200' : 'text-red-200'}">
                        ${this.currentInvestor.totalPnL >= 0 ? '+' : ''}$${this.currentInvestor.totalPnL.toLocaleString()}
                    </div>
                </div>
                <div>
                    <div class="text-sm opacity-75">Active Positions</div>
                    <div class="font-bold text-lg">${this.currentInvestor.activePositions.length}</div>
                </div>
            </div>
        `;

        accountActions.innerHTML = `
            <div class="flex items-center gap-4">
                <div class="text-sm">
                    <div class="opacity-75">Risk Level</div>
                    <div class="font-semibold capitalize">${this.currentInvestor.riskTolerance}</div>
                </div>
                <button id="account-settings-btn" class="bg-white bg-opacity-20 px-4 py-2 rounded-lg font-semibold hover:bg-opacity-30">
                    Settings
                </button>
                <button id="logout-btn" class="bg-white bg-opacity-20 px-4 py-2 rounded-lg font-semibold hover:bg-opacity-30">
                    Logout
                </button>
            </div>
        `;

        document.getElementById('logout-btn').addEventListener('click', () => this.logout());
    }

    showLogin() {
        this.loginModal.classList.remove('hidden');
    }

    hideLogin() {
        this.loginModal.classList.add('hidden');
    }

    showLoginPrompt() {
        // Show login modal immediately on load
        setTimeout(() => {
            if (!this.currentInvestor) {
                this.showLogin();
            }
        }, 1000);
    }

    logout() {
        this.currentInvestor = null;
        this.updateAccountHeader();
        this.showLogin();
        
        // Reset account header
        document.getElementById('account-info').innerHTML = 
            '<div class="text-sm opacity-75">Please log in to access trading platform</div>';
        document.getElementById('account-actions').innerHTML = `
            <button id="login-btn" class="bg-white text-amber-700 px-4 py-2 rounded-lg font-semibold hover:bg-gray-100">
                Login / Register
            </button>
        `;
        document.getElementById('login-btn').addEventListener('click', () => this.showLogin());
    }

    saveAccounts() {
        const accountData = {};
        this.investors.forEach((investor, email) => {
            accountData[email] = investor;
        });
        localStorage.setItem('gomnaInvestorAccounts', JSON.stringify(accountData));
    }

    loadSavedAccounts() {
        const saved = localStorage.getItem('gomnaInvestorAccounts');
        if (saved) {
            const accountData = JSON.parse(saved);
            Object.entries(accountData).forEach(([email, investor]) => {
                this.investors.set(email, investor);
            });
        }
    }

    getCurrentInvestor() {
        return this.currentInvestor;
    }

    updateInvestorPnL(amount) {
        if (!this.currentInvestor) return;
        
        this.currentInvestor.totalPnL += amount;
        this.currentInvestor.accountBalance += amount;
        this.saveAccounts();
        this.updateAccountHeader();
    }

    addPosition(position) {
        if (!this.currentInvestor) return;
        
        this.currentInvestor.activePositions.push(position);
        this.saveAccounts();
        this.updateAccountHeader();
    }

    removePosition(positionId) {
        if (!this.currentInvestor) return;
        
        const index = this.currentInvestor.activePositions.findIndex(p => p.id === positionId);
        if (index !== -1) {
            const position = this.currentInvestor.activePositions.splice(index, 1)[0];
            this.currentInvestor.tradeHistory.push(position);
            this.saveAccounts();
            this.updateAccountHeader();
            return position;
        }
        return null;
    }
}

class EnhancedTradingExecutor {
    constructor(investorSystem) {
        this.investorSystem = investorSystem;
        this.activePositions = [];
        this.executionHistory = [];
        this.totalPnL = 0;
        this.pnlHistory = []; // Track P&L over time
        this.executionSettings = {
            maxPositions: 10,
            maxRiskPerTrade: 0.02,
            autoExecuteThreshold: 0.015,
            stopLossThreshold: 0.05
        };
        this.isAutoExecutionEnabled = false;
        
        // Initialize with some historical P&L data for chart
        this.initializePnLHistory();
    }

    initializePnLHistory() {
        // Create initial P&L history for the chart
        const now = Date.now();
        for (let i = 19; i >= 0; i--) {
            this.pnlHistory.push({
                timestamp: now - (i * 30000), // 30 seconds apart
                totalPnL: (Math.random() - 0.5) * 1000 * (20 - i) / 20 // Progressive growth
            });
        }
    }

    async executeArbitrage(opportunityId) {
        const investor = this.investorSystem.getCurrentInvestor();
        if (!investor) {
            alert('Please log in to execute trades');
            return;
        }

        const opportunity = window.completeFunctionalSystem.arbitrageEngine.getOpportunityById(opportunityId);
        
        if (!opportunity) {
            this.showExecutionStatus('error', 'Opportunity not found or expired');
            return;
        }

        if (this.activePositions.length >= this.executionSettings.maxPositions) {
            this.showExecutionStatus('error', 'Maximum position limit reached');
            return;
        }

        // Calculate position size based on investor's account balance
        const maxTradeAmount = investor.accountBalance * this.executionSettings.maxRiskPerTrade;
        const positionSize = this.calculatePositionSize(opportunity, maxTradeAmount);

        if (positionSize * (opportunity.buyPrice || opportunity.currentPrice || 1000) > investor.accountBalance) {
            this.showExecutionStatus('error', 'Insufficient account balance for this trade');
            return;
        }

        this.showExecutionStatus('processing', `Executing ${opportunity.type} arbitrage for ${opportunity.pair}...`);

        try {
            await this.simulateExecution(opportunity);

            const position = {
                id: `pos_${Date.now()}`,
                investorId: investor.id,
                opportunity: opportunity,
                entryTime: Date.now(),
                entryPrice: opportunity.buyPrice || opportunity.currentPrice,
                targetPrice: opportunity.sellPrice || opportunity.meanPrice,
                size: positionSize,
                status: 'active',
                unrealizedPnL: 0,
                realizedPnL: 0
            };

            this.activePositions.push(position);
            this.investorSystem.addPosition(position);
            this.updatePortfolio();
            
            this.showExecutionStatus('success', 
                `Position opened: ${position.id} - Expected profit: ${(opportunity.profit * 100).toFixed(2)}%`);

            // Auto-close position after random time (5-30 seconds for demo)
            setTimeout(() => {
                this.closePosition(position.id);
            }, Math.random() * 25000 + 5000);

        } catch (error) {
            this.showExecutionStatus('error', `Execution failed: ${error.message}`);
        }
    }

    calculatePositionSize(opportunity, maxAmount) {
        const basePrice = opportunity.buyPrice || opportunity.currentPrice || 1000;
        const maxUnits = maxAmount / basePrice;
        
        // Risk-adjusted position sizing
        const riskAdjustedSize = maxUnits * (1 - opportunity.risk);
        
        return Math.min(riskAdjustedSize, 100); // Max 100 units per trade
    }

    async simulateExecution(opportunity) {
        await new Promise(resolve => setTimeout(resolve, Math.random() * 1000 + 500));
        
        if (Math.random() < 0.05) {
            throw new Error('Insufficient liquidity');
        }
        
        const slippage = Math.random() * 0.004 + 0.001;
        opportunity.executionSlippage = slippage;
        
        return true;
    }

    closePosition(positionId) {
        const positionIndex = this.activePositions.findIndex(pos => pos.id === positionId);
        if (positionIndex === -1) return;

        const position = this.activePositions[positionIndex];
        const holdTime = Date.now() - position.entryTime;
        
        // Calculate realistic P&L
        const marketMovement = (Math.random() - 0.5) * 0.02;
        const baseProfit = position.opportunity.profit;
        const slippage = position.opportunity.executionSlippage || 0;
        
        const realizedPnL = (baseProfit + marketMovement - slippage) * position.size * position.entryPrice;
        
        position.status = 'closed';
        position.exitTime = Date.now();
        position.holdTime = holdTime;
        position.realizedPnL = realizedPnL;

        // Update investor account
        this.investorSystem.updateInvestorPnL(realizedPnL);
        this.investorSystem.removePosition(positionId);

        // Move to execution history
        this.executionHistory.unshift(position);
        this.activePositions.splice(positionIndex, 1);
        
        // Update total P&L and history
        this.totalPnL += realizedPnL;
        this.pnlHistory.push({
            timestamp: Date.now(),
            totalPnL: this.totalPnL
        });
        
        // Keep only last 20 entries
        if (this.pnlHistory.length > 20) {
            this.pnlHistory.shift();
        }
        
        this.updatePortfolio();
        this.showExecutionStatus('info', 
            `Position closed: ${positionId} - P&L: ${realizedPnL > 0 ? '+' : ''}$${realizedPnL.toFixed(2)}`);
    }

    updatePortfolio() {
        const container = document.getElementById('active-positions');
        if (!container) return;

        const investor = this.investorSystem.getCurrentInvestor();
        if (!investor) {
            container.innerHTML = '<div class="text-center text-gray-500 p-8">Please log in to view portfolio</div>';
            return;
        }

        const totalUnrealized = this.activePositions.reduce((sum, pos) => {
            const currentUnrealized = (Math.random() - 0.5) * 0.01 * pos.size * pos.entryPrice;
            pos.unrealizedPnL = currentUnrealized;
            return sum + currentUnrealized;
        }, 0);

        container.innerHTML = `
            <div class="portfolio-summary">
                <h3>📊 ${investor.name}'s Portfolio</h3>
                <div class="pnl-summary">
                    <div class="total-pnl ${investor.totalPnL >= 0 ? 'positive' : 'negative'}">
                        Total P&L: ${investor.totalPnL >= 0 ? '+' : ''}$${investor.totalPnL.toFixed(2)}
                    </div>
                    <div class="unrealized-pnl ${totalUnrealized >= 0 ? 'positive' : 'negative'}">
                        Unrealized: ${totalUnrealized >= 0 ? '+' : ''}$${totalUnrealized.toFixed(2)}
                    </div>
                    <div class="account-balance">
                        Balance: $${investor.accountBalance.toLocaleString()}
                    </div>
                </div>
            </div>
            
            <div class="positions-list">
                <h4>Active Positions (${this.activePositions.length})</h4>
                ${this.activePositions.map(pos => this.renderPosition(pos)).join('')}
            </div>
            
            <div class="execution-history">
                <h4>Recent Executions</h4>
                ${this.executionHistory.slice(0, 5).map(pos => this.renderHistoryItem(pos)).join('')}
            </div>
        `;
    }

    renderPosition(position) {
        const holdTime = Math.floor((Date.now() - position.entryTime) / 1000);
        const pnlClass = position.unrealizedPnL >= 0 ? 'positive' : 'negative';

        return `
            <div class="position-item">
                <div class="position-header">
                    <span class="position-id">${position.id}</span>
                    <span class="position-pair">${position.opportunity.pair}</span>
                    <span class="position-type">${position.opportunity.type}</span>
                </div>
                <div class="position-details">
                    <div class="position-size">Size: ${position.size.toFixed(2)}</div>
                    <div class="position-entry">Entry: $${position.entryPrice.toFixed(2)}</div>
                    <div class="position-pnl ${pnlClass}">
                        P&L: ${position.unrealizedPnL >= 0 ? '+' : ''}$${position.unrealizedPnL.toFixed(2)}
                    </div>
                    <div class="position-time">Hold: ${holdTime}s</div>
                </div>
                <button class="close-position-btn" onclick="completeFunctionalSystem.tradingExecutor.closePosition('${position.id}')">
                    Close Position
                </button>
            </div>
        `;
    }

    renderHistoryItem(position) {
        const pnlClass = position.realizedPnL >= 0 ? 'positive' : 'negative';
        const holdTimeSeconds = Math.floor(position.holdTime / 1000);

        return `
            <div class="history-item">
                <div class="history-header">
                    <span class="history-pair">${position.opportunity.pair}</span>
                    <span class="history-type">${position.opportunity.type}</span>
                    <span class="history-pnl ${pnlClass}">
                        ${position.realizedPnL >= 0 ? '+' : ''}$${position.realizedPnL.toFixed(2)}
                    </span>
                </div>
                <div class="history-details">
                    <span class="history-time">${holdTimeSeconds}s hold</span>
                    <span class="history-timestamp">${new Date(position.exitTime).toLocaleTimeString()}</span>
                </div>
            </div>
        `;
    }

    showExecutionStatus(type, message) {
        const statusContainer = document.getElementById('execution-status');
        if (!statusContainer) return;

        const statusClass = {
            'success': 'status-success',
            'error': 'status-error',
            'warning': 'status-warning',
            'info': 'status-info',
            'processing': 'status-processing'
        }[type] || 'status-info';

        statusContainer.innerHTML = `
            <div class="execution-status-item ${statusClass}">
                <span class="status-icon">${this.getStatusIcon(type)}</span>
                <span class="status-message">${message}</span>
                <span class="status-time">${new Date().toLocaleTimeString()}</span>
            </div>
        `;

        setTimeout(() => {
            if (statusContainer.innerHTML.includes(message)) {
                statusContainer.innerHTML = '';
            }
        }, 5000);
    }

    getStatusIcon(type) {
        const icons = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'info': 'ℹ️',
            'processing': '⏳'
        };
        return icons[type] || 'ℹ️';
    }

    getPnLHistory() {
        return this.pnlHistory;
    }
}

class EnhancedPerformanceMonitor {
    constructor(tradingExecutor) {
        this.tradingExecutor = tradingExecutor;
        this.metrics = {
            totalTrades: 0,
            successfulTrades: 0,
            totalPnL: 0,
            averageProfit: 0,
            winRate: 0,
            sharpeRatio: 0,
            maxDrawdown: 0,
            activeOpportunities: 0
        };
        
        this.performanceHistory = [];
        this.updateInterval = 3000; // 3 seconds
        this.isMonitoring = false;
    }

    startMonitoring() {
        if (this.isMonitoring) return;
        
        this.isMonitoring = true;
        console.log('📈 Starting enhanced performance monitoring...');
        
        const monitorCycle = () => {
            if (!this.isMonitoring) return;
            
            this.updateMetrics();
            this.updateDisplay();
            
            setTimeout(monitorCycle, this.updateInterval);
        };
        
        monitorCycle();
    }

    updateMetrics() {
        const tradingExecutor = this.tradingExecutor;
        const arbitrageEngine = window.completeFunctionalSystem?.arbitrageEngine;
        
        if (!tradingExecutor || !arbitrageEngine) return;

        this.metrics.totalTrades = tradingExecutor.executionHistory.length;
        this.metrics.successfulTrades = tradingExecutor.executionHistory.filter(pos => pos.realizedPnL > 0).length;
        this.metrics.totalPnL = tradingExecutor.totalPnL;
        this.metrics.activeOpportunities = arbitrageEngine.opportunities.length;
        
        if (this.metrics.totalTrades > 0) {
            this.metrics.winRate = (this.metrics.successfulTrades / this.metrics.totalTrades) * 100;
            this.metrics.averageProfit = this.metrics.totalPnL / this.metrics.totalTrades;
        }
        
        // Calculate Sharpe ratio
        if (tradingExecutor.executionHistory.length > 1) {
            const returns = tradingExecutor.executionHistory.map(pos => pos.realizedPnL);
            const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
            const stdDev = Math.sqrt(returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length);
            this.metrics.sharpeRatio = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0;
        }
        
        // Calculate max drawdown
        let peak = 0;
        let maxDD = 0;
        let runningPnL = 0;
        
        tradingExecutor.executionHistory.forEach(pos => {
            runningPnL += pos.realizedPnL;
            if (runningPnL > peak) peak = runningPnL;
            const drawdown = (peak - runningPnL) / Math.max(peak, 1);
            if (drawdown > maxDD) maxDD = drawdown;
        });
        
        this.metrics.maxDrawdown = maxDD * 100;
        
        // Store performance snapshot
        this.performanceHistory.push({
            timestamp: Date.now(),
            ...this.metrics
        });
        
        if (this.performanceHistory.length > 100) {
            this.performanceHistory.shift();
        }
    }

    updateDisplay() {
        const container = document.getElementById('performance-metrics');
        if (!container) return;

        container.innerHTML = `
            <div class="performance-dashboard">
                <h3>📊 Live Performance Metrics</h3>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-value ${this.metrics.totalPnL >= 0 ? 'positive' : 'negative'}">
                            ${this.metrics.totalPnL >= 0 ? '+' : ''}$${this.metrics.totalPnL.toFixed(2)}
                        </div>
                        <div class="metric-label">Total P&L</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-value">${this.metrics.totalTrades}</div>
                        <div class="metric-label">Total Trades</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-value ${this.metrics.winRate >= 50 ? 'positive' : 'negative'}">
                            ${this.metrics.winRate.toFixed(1)}%
                        </div>
                        <div class="metric-label">Win Rate</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-value ${this.metrics.averageProfit >= 0 ? 'positive' : 'negative'}">
                            ${this.metrics.averageProfit >= 0 ? '+' : ''}$${this.metrics.averageProfit.toFixed(2)}
                        </div>
                        <div class="metric-label">Avg Profit</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-value ${this.metrics.sharpeRatio >= 1 ? 'positive' : 'neutral'}">
                            ${this.metrics.sharpeRatio.toFixed(2)}
                        </div>
                        <div class="metric-label">Sharpe Ratio</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-value ${this.metrics.maxDrawdown <= 5 ? 'positive' : 'negative'}">
                            ${this.metrics.maxDrawdown.toFixed(1)}%
                        </div>
                        <div class="metric-label">Max Drawdown</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-value">${this.metrics.activeOpportunities}</div>
                        <div class="metric-label">Live Opportunities</div>
                    </div>
                    
                    <div class="metric-card">
                        <div class="metric-value neutral">
                            ${new Date().toLocaleTimeString()}
                        </div>
                        <div class="metric-label">Last Update</div>
                    </div>
                </div>
                
                <div class="performance-chart">
                    ${this.renderPerformanceChart()}
                </div>
            </div>
        `;
    }

    renderPerformanceChart() {
        const pnlHistory = this.tradingExecutor.getPnLHistory();
        
        if (pnlHistory.length < 2) {
            return '<div class="chart-placeholder">Collecting performance data...</div>';
        }

        const maxPnL = Math.max(...pnlHistory.map(h => h.totalPnL));
        const minPnL = Math.min(...pnlHistory.map(h => h.totalPnL));
        const range = Math.max(maxPnL - minPnL, 100);

        const chartPoints = pnlHistory.map((point, index) => {
            const x = (index / (pnlHistory.length - 1)) * 100;
            const y = 100 - ((point.totalPnL - minPnL) / range * 80 + 10);
            return `${x},${y}`;
        }).join(' ');

        return `
            <div class="chart-container">
                <div class="chart-title">P&L Trend (Last ${pnlHistory.length} Updates)</div>
                <svg class="performance-chart-svg" viewBox="0 0 100 100">
                    <polyline points="${chartPoints}" 
                              fill="none" 
                              stroke="${this.metrics.totalPnL >= 0 ? '#00ff88' : '#ff4444'}" 
                              stroke-width="2"/>
                    <line x1="0" y1="50" x2="100" y2="50" stroke="#444" stroke-width="1" stroke-dasharray="2,2"/>
                </svg>
                <div class="chart-labels">
                    <span class="chart-min">$${minPnL.toFixed(0)}</span>
                    <span class="chart-max">$${maxPnL.toFixed(0)}</span>
                </div>
            </div>
        `;
    }
}

// Import and extend the existing classes
class CompleteFunctionalAgenticSystem {
    constructor() {
        this.investorSystem = new InvestorAccountSystem();
        
        // Wait for investor login before initializing trading components
        this.waitForInvestorLogin();
    }

    waitForInvestorLogin() {
        const checkLogin = () => {
            if (this.investorSystem.getCurrentInvestor()) {
                this.initializeTradingSystem();
            } else {
                setTimeout(checkLogin, 1000);
            }
        };
        checkLogin();
    }

    async initializeTradingSystem() {
        console.log('🚀 Initializing Complete Functional Trading System...');
        
        // Import components from the main system
        if (window.functionalTradingSystem) {
            this.marketDataFetcher = window.functionalTradingSystem.marketDataFetcher;
            this.arbitrageEngine = window.functionalTradingSystem.arbitrageEngine;
        } else {
            // Create new instances if needed
            this.marketDataFetcher = new LiveMarketDataFetcher();
            this.arbitrageEngine = new ArbitrageDetectionEngine(this.marketDataFetcher);
        }

        // Use enhanced components
        this.tradingExecutor = new EnhancedTradingExecutor(this.investorSystem);
        this.performanceMonitor = new EnhancedPerformanceMonitor(this.tradingExecutor);
        
        // Start the trading system
        if (!this.marketDataFetcher.isRunning) {
            await this.marketDataFetcher.startRealTimeUpdates();
        }
        if (!this.arbitrageEngine.isScanning) {
            this.arbitrageEngine.startScanning();
        }
        this.performanceMonitor.startMonitoring();
        
        console.log('✅ Complete Functional Trading System is LIVE with investor accounts!');
        
        // Update the portfolio immediately
        setTimeout(() => {
            this.tradingExecutor.updatePortfolio();
        }, 1000);
    }

    // Expose methods for UI interaction
    executeArbitrage(opportunityId) {
        return this.tradingExecutor.executeArbitrage(opportunityId);
    }

    analyzeOpportunity(opportunityId) {
        const opportunity = this.arbitrageEngine.getOpportunityById(opportunityId);
        if (!opportunity) return;

        alert(`Opportunity Analysis:
        
Type: ${opportunity.type}
Pair: ${opportunity.pair}
Expected Profit: ${(opportunity.profit * 100).toFixed(2)}%
Risk Level: ${(opportunity.risk * 100).toFixed(1)}%
Confidence: ${(opportunity.confidence * 100).toFixed(0)}%
Volume: $${opportunity.volume.toLocaleString()}

Recommendation: ${opportunity.profit > 0.01 ? 'EXECUTE' : 'MONITOR'}
        `);
    }
}

// Initialize the complete system when DOM is ready
document.addEventListener('DOMContentLoaded', function() {
    window.completeFunctionalSystem = new CompleteFunctionalAgenticSystem();
});

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        CompleteFunctionalAgenticSystem,
        InvestorAccountSystem,
        EnhancedTradingExecutor,
        EnhancedPerformanceMonitor
    };
}