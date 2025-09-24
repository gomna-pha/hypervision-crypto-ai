/**
 * HyperVision AI - Investor Customization Interface
 * 
 * Allows institutional investors to customize arbitrage algorithm parameters
 * Features:
 * - Strategy template selection and customization
 * - Real-time backtesting and validation
 * - Risk parameter configuration
 * - Performance monitoring and reporting
 * - Compliance and audit trail
 */

class InvestorCustomizationInterface {
    constructor() {
        this.strategies = {
            index_futures_spot: {
                name: "Index/Futures-Spot Arbitrage",
                priority: 1,
                description: "Exploits price discrepancies between index futures and underlying spot assets",
                roi_estimate: "15-25% annually",
                complexity: "Low",
                parameters: {
                    min_spread_bps: { value: 5, min: 1, max: 50, description: "Minimum spread in basis points" },
                    max_position_size: { value: 1000000, min: 100000, max: 10000000, description: "Maximum position size ($)" },
                    max_latency_ms: { value: 50, min: 10, max: 200, description: "Maximum acceptable latency (ms)" },
                    risk_multiplier: { value: 0.1, min: 0.01, max: 1.0, description: "Risk adjustment factor" }
                }
            },
            triangular_crypto: {
                name: "Triangular Crypto Arbitrage", 
                priority: 2,
                description: "Exploits triangular currency relationships in crypto markets",
                roi_estimate: "20-35% annually",
                complexity: "Medium",
                parameters: {
                    min_profit_bps: { value: 10, min: 5, max: 100, description: "Minimum profit in basis points" },
                    max_slippage_bps: { value: 5, min: 1, max: 20, description: "Maximum acceptable slippage (bps)" },
                    trading_pairs: { value: ["BTC/ETH/USDT", "BTC/ETH/USDC"], type: "array", description: "Trading pair combinations" },
                    execution_timeout_ms: { value: 100, min: 50, max: 1000, description: "Execution timeout (ms)" }
                }
            },
            statistical_pairs: {
                name: "Statistical Pairs Arbitrage",
                priority: 3, 
                description: "Mean-reverting pairs trading with hyperbolic embeddings",
                roi_estimate: "12-20% annually",
                complexity: "High",
                parameters: {
                    lookback_periods: { value: 252, min: 50, max: 1000, description: "Historical lookback periods" },
                    z_entry_threshold: { value: 2.0, min: 1.0, max: 4.0, description: "Z-score entry threshold" },
                    z_exit_threshold: { value: 0.5, min: 0.1, max: 1.5, description: "Z-score exit threshold" },
                    max_holding_days: { value: 30, min: 1, max: 90, description: "Maximum holding period (days)" }
                }
            },
            news_sentiment: {
                name: "News/Sentiment-Triggered Arbitrage",
                priority: 4,
                description: "FinBERT-powered news sentiment arbitrage",
                roi_estimate: "25-40% annually", 
                complexity: "High",
                parameters: {
                    sentiment_threshold: { value: 0.7, min: 0.1, max: 1.0, description: "Minimum sentiment confidence" },
                    news_decay_factor: { value: 0.1, min: 0.01, max: 1.0, description: "News signal decay rate" },
                    max_news_age_minutes: { value: 15, min: 1, max: 120, description: "Maximum news age (minutes)" },
                    social_weight: { value: 0.3, min: 0.0, max: 1.0, description: "Social media signal weight" }
                }
            },
            latency_arbitrage: {
                name: "Statistical Latency Arbitrage",
                priority: 5,
                description: "Microstructure patterns and predictable price movements", 
                roi_estimate: "30-50% annually",
                complexity: "Very High",
                parameters: {
                    max_execution_latency_us: { value: 100, min: 10, max: 1000, description: "Max execution latency (microseconds)" },
                    pattern_confidence_threshold: { value: 0.8, min: 0.5, max: 0.99, description: "Pattern confidence threshold" },
                    position_decay_ms: { value: 50, min: 10, max: 500, description: "Position decay time (ms)" }
                }
            }
        };

        this.currentInvestor = null;
        this.customConfigurations = new Map();
        this.backtestResults = new Map();
        this.realTimeMonitor = null;
        
        this.initializeInterface();
    }

    initializeInterface() {
        this.createCustomizationPanels();
        this.setupEventListeners();
        this.initializeCharts();
        this.loadInvestorProfiles();
    }

    createCustomizationPanels() {
        const container = document.getElementById('investor-customization');
        if (!container) return;

        container.innerHTML = `
            <div class="customization-interface bg-white rounded-lg shadow-lg p-6">
                <!-- Header -->
                <div class="interface-header mb-8">
                    <h2 class="text-3xl font-bold text-gray-800 mb-2">🎯 Algorithm Customization Suite</h2>
                    <p class="text-gray-600">Customize arbitrage strategies for your institutional requirements</p>
                </div>

                <!-- Investor Profile Selection -->
                <div class="investor-profile mb-8">
                    <h3 class="text-xl font-semibold mb-4">👤 Investor Profile</h3>
                    <div class="profile-selector grid grid-cols-1 md:grid-cols-3 gap-4">
                        <div class="profile-card bg-blue-50 p-4 rounded-lg cursor-pointer border-2 border-transparent hover:border-blue-300" data-profile="hedge-fund">
                            <h4 class="font-semibold text-blue-800">🏦 Hedge Fund</h4>
                            <p class="text-sm text-blue-600">High risk, high return strategies</p>
                            <div class="mt-2">
                                <span class="text-xs bg-blue-200 px-2 py-1 rounded">$10M+ AUM</span>
                            </div>
                        </div>
                        <div class="profile-card bg-green-50 p-4 rounded-lg cursor-pointer border-2 border-transparent hover:border-green-300" data-profile="pension-fund">
                            <h4 class="font-semibold text-green-800">🏛️ Pension Fund</h4>
                            <p class="text-sm text-green-600">Conservative, stable returns</p>
                            <div class="mt-2">
                                <span class="text-xs bg-green-200 px-2 py-1 rounded">$100M+ AUM</span>
                            </div>
                        </div>
                        <div class="profile-card bg-purple-50 p-4 rounded-lg cursor-pointer border-2 border-transparent hover:border-purple-300" data-profile="prop-trading">
                            <h4 class="font-semibold text-purple-800">⚡ Prop Trading</h4>
                            <p class="text-sm text-purple-600">Ultra-low latency strategies</p>
                            <div class="mt-2">
                                <span class="text-xs bg-purple-200 px-2 py-1 rounded">Speed Focus</span>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Strategy Selection -->
                <div class="strategy-selection mb-8">
                    <h3 class="text-xl font-semibold mb-4">📊 Strategy Selection & Prioritization</h3>
                    <div id="strategy-grid" class="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
                        <!-- Strategy cards will be populated here -->
                    </div>
                </div>

                <!-- Parameter Customization -->
                <div id="parameter-customization" class="parameter-section mb-8" style="display: none;">
                    <h3 class="text-xl font-semibold mb-4">⚙️ Parameter Customization</h3>
                    <div id="parameter-panels" class="space-y-6">
                        <!-- Parameter controls will be populated here -->
                    </div>
                </div>

                <!-- Real-time Backtesting -->
                <div id="backtesting-section" class="backtest-section mb-8" style="display: none;">
                    <h3 class="text-xl font-semibold mb-4">🧪 Real-time Strategy Validation</h3>
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div class="backtest-controls bg-gray-50 p-4 rounded-lg">
                            <h4 class="font-semibold mb-3">Backtest Configuration</h4>
                            <div class="space-y-3">
                                <div>
                                    <label class="block text-sm font-medium mb-1">Time Period</label>
                                    <select id="backtest-period" class="w-full p-2 border rounded">
                                        <option value="1d">Last 24 Hours</option>
                                        <option value="7d">Last Week</option>
                                        <option value="30d" selected>Last Month</option>
                                        <option value="90d">Last Quarter</option>
                                    </select>
                                </div>
                                <div>
                                    <label class="block text-sm font-medium mb-1">Initial Capital</label>
                                    <input type="number" id="initial-capital" value="1000000" class="w-full p-2 border rounded" placeholder="$1,000,000">
                                </div>
                                <button id="run-backtest" class="w-full bg-blue-600 text-white py-2 rounded hover:bg-blue-700">
                                    🚀 Run Backtest
                                </button>
                            </div>
                        </div>
                        <div class="backtest-results bg-gray-50 p-4 rounded-lg">
                            <h4 class="font-semibold mb-3">Performance Metrics</h4>
                            <div id="performance-metrics" class="space-y-2">
                                <div class="metric-row flex justify-between">
                                    <span>Total Return:</span>
                                    <span id="total-return" class="font-semibold">--</span>
                                </div>
                                <div class="metric-row flex justify-between">
                                    <span>Sharpe Ratio:</span>
                                    <span id="sharpe-ratio" class="font-semibold">--</span>
                                </div>
                                <div class="metric-row flex justify-between">
                                    <span>Max Drawdown:</span>
                                    <span id="max-drawdown" class="font-semibold">--</span>
                                </div>
                                <div class="metric-row flex justify-between">
                                    <span>Win Rate:</span>
                                    <span id="win-rate" class="font-semibold">--</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div class="mt-6">
                        <canvas id="backtest-chart" width="800" height="400"></canvas>
                    </div>
                </div>

                <!-- Risk Management -->
                <div id="risk-management" class="risk-section mb-8" style="display: none;">
                    <h3 class="text-xl font-semibold mb-4">🛡️ Risk Management & Compliance</h3>
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div class="risk-controls bg-red-50 p-4 rounded-lg">
                            <h4 class="font-semibold mb-3 text-red-800">Risk Limits</h4>
                            <div class="space-y-3">
                                <div>
                                    <label class="block text-sm font-medium mb-1">Max Daily Loss</label>
                                    <div class="flex items-center space-x-2">
                                        <input type="range" id="max-daily-loss" min="10000" max="1000000" value="50000" class="flex-1">
                                        <span id="max-daily-loss-value" class="text-sm font-mono">$50,000</span>
                                    </div>
                                </div>
                                <div>
                                    <label class="block text-sm font-medium mb-1">Max Position Size</label>
                                    <div class="flex items-center space-x-2">
                                        <input type="range" id="max-position-size" min="100000" max="10000000" value="1000000" class="flex-1">
                                        <span id="max-position-size-value" class="text-sm font-mono">$1,000,000</span>
                                    </div>
                                </div>
                                <div>
                                    <label class="block text-sm font-medium mb-1">Max Drawdown</label>
                                    <div class="flex items-center space-x-2">
                                        <input type="range" id="max-drawdown-limit" min="5" max="50" value="10" class="flex-1">
                                        <span id="max-drawdown-value" class="text-sm font-mono">10%</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                        <div class="compliance-settings bg-blue-50 p-4 rounded-lg">
                            <h4 class="font-semibold mb-3 text-blue-800">Compliance Settings</h4>
                            <div class="space-y-3">
                                <div class="flex items-center justify-between">
                                    <span class="text-sm">Enable Position Reporting</span>
                                    <input type="checkbox" id="position-reporting" checked class="toggle">
                                </div>
                                <div class="flex items-center justify-between">
                                    <span class="text-sm">Real-time Risk Monitoring</span>
                                    <input type="checkbox" id="risk-monitoring" checked class="toggle">
                                </div>
                                <div class="flex items-center justify-between">
                                    <span class="text-sm">Audit Trail Logging</span>
                                    <input type="checkbox" id="audit-logging" checked class="toggle">
                                </div>
                                <div class="flex items-center justify-between">
                                    <span class="text-sm">Circuit Breaker</span>
                                    <input type="checkbox" id="circuit-breaker" checked class="toggle">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <!-- Live Deployment -->
                <div id="deployment-section" class="deployment-section mb-8" style="display: none;">
                    <h3 class="text-xl font-semibold mb-4">🚀 Strategy Deployment</h3>
                    <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div class="deployment-config bg-green-50 p-4 rounded-lg">
                            <h4 class="font-semibold mb-3 text-green-800">Deployment Configuration</h4>
                            <div class="space-y-3">
                                <div>
                                    <label class="block text-sm font-medium mb-1">Environment</label>
                                    <select id="deployment-env" class="w-full p-2 border rounded">
                                        <option value="sandbox">Sandbox (Paper Trading)</option>
                                        <option value="staging">Staging Environment</option>
                                        <option value="production">Production (Live Trading)</option>
                                    </select>
                                </div>
                                <div>
                                    <label class="block text-sm font-medium mb-1">Capital Allocation</label>
                                    <input type="number" id="deployment-capital" value="100000" class="w-full p-2 border rounded" placeholder="$100,000">
                                </div>
                                <div>
                                    <label class="block text-sm font-medium mb-1">Strategy Weight (%)</label>
                                    <input type="range" id="strategy-weight" min="1" max="100" value="25" class="w-full">
                                    <span id="strategy-weight-value" class="text-sm font-mono">25%</span>
                                </div>
                            </div>
                        </div>
                        <div class="deployment-status bg-gray-50 p-4 rounded-lg">
                            <h4 class="font-semibold mb-3">Current Status</h4>
                            <div id="deployment-status" class="space-y-2">
                                <div class="status-indicator flex items-center space-x-2">
                                    <div class="w-3 h-3 bg-yellow-400 rounded-full"></div>
                                    <span class="text-sm">Configuration Ready</span>
                                </div>
                                <div class="status-indicator flex items-center space-x-2">
                                    <div class="w-3 h-3 bg-gray-400 rounded-full"></div>
                                    <span class="text-sm">Waiting for Deployment</span>
                                </div>
                            </div>
                            <button id="deploy-strategy" class="w-full mt-4 bg-green-600 text-white py-2 rounded hover:bg-green-700">
                                📡 Deploy Strategy
                            </button>
                        </div>
                    </div>
                </div>

                <!-- Configuration Export/Import -->
                <div class="config-management mt-8">
                    <h3 class="text-xl font-semibold mb-4">💾 Configuration Management</h3>
                    <div class="flex space-x-4">
                        <button id="export-config" class="bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700">
                            📤 Export Configuration
                        </button>
                        <button id="import-config" class="bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700">
                            📥 Import Configuration
                        </button>
                        <button id="save-template" class="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
                            💾 Save as Template
                        </button>
                    </div>
                    <input type="file" id="config-file-input" accept=".json" style="display: none;">
                </div>
            </div>
        `;

        this.populateStrategyGrid();
    }

    populateStrategyGrid() {
        const strategyGrid = document.getElementById('strategy-grid');
        if (!strategyGrid) return;

        strategyGrid.innerHTML = '';

        Object.entries(this.strategies).forEach(([strategyId, strategy]) => {
            const strategyCard = document.createElement('div');
            strategyCard.className = 'strategy-card bg-white border-2 border-gray-200 rounded-lg p-4 cursor-pointer hover:border-blue-300 transition-colors';
            strategyCard.dataset.strategy = strategyId;

            const complexityColor = {
                'Low': 'bg-green-100 text-green-800',
                'Medium': 'bg-yellow-100 text-yellow-800', 
                'High': 'bg-orange-100 text-orange-800',
                'Very High': 'bg-red-100 text-red-800'
            }[strategy.complexity] || 'bg-gray-100 text-gray-800';

            strategyCard.innerHTML = `
                <div class="flex items-start justify-between mb-3">
                    <h4 class="font-semibold text-gray-800">${strategy.name}</h4>
                    <div class="flex items-center space-x-2">
                        <span class="text-xs ${complexityColor} px-2 py-1 rounded">${strategy.complexity}</span>
                        <span class="text-xs bg-blue-100 text-blue-800 px-2 py-1 rounded">Priority ${strategy.priority}</span>
                    </div>
                </div>
                <p class="text-sm text-gray-600 mb-3">${strategy.description}</p>
                <div class="flex items-center justify-between">
                    <span class="text-sm font-semibold text-green-600">${strategy.roi_estimate}</span>
                    <input type="checkbox" class="strategy-checkbox" data-strategy="${strategyId}">
                </div>
            `;

            strategyGrid.appendChild(strategyCard);
        });
    }

    setupEventListeners() {
        // Profile selection
        document.querySelectorAll('.profile-card').forEach(card => {
            card.addEventListener('click', (e) => {
                this.selectInvestorProfile(e.currentTarget.dataset.profile);
            });
        });

        // Strategy selection
        document.addEventListener('change', (e) => {
            if (e.target.classList.contains('strategy-checkbox')) {
                this.toggleStrategy(e.target.dataset.strategy, e.target.checked);
            }
        });

        // Strategy card clicks
        document.addEventListener('click', (e) => {
            const strategyCard = e.target.closest('.strategy-card');
            if (strategyCard) {
                const checkbox = strategyCard.querySelector('.strategy-checkbox');
                checkbox.checked = !checkbox.checked;
                this.toggleStrategy(strategyCard.dataset.strategy, checkbox.checked);
            }
        });

        // Parameter updates
        document.addEventListener('input', (e) => {
            if (e.target.classList.contains('parameter-input')) {
                this.updateParameter(e.target.dataset.strategy, e.target.dataset.parameter, e.target.value);
            }
        });

        // Risk management sliders
        document.addEventListener('input', (e) => {
            if (e.target.id === 'max-daily-loss') {
                document.getElementById('max-daily-loss-value').textContent = `$${parseInt(e.target.value).toLocaleString()}`;
            } else if (e.target.id === 'max-position-size') {
                document.getElementById('max-position-size-value').textContent = `$${parseInt(e.target.value).toLocaleString()}`;
            } else if (e.target.id === 'max-drawdown-limit') {
                document.getElementById('max-drawdown-value').textContent = `${e.target.value}%`;
            } else if (e.target.id === 'strategy-weight') {
                document.getElementById('strategy-weight-value').textContent = `${e.target.value}%`;
            }
        });

        // Backtest button
        document.getElementById('run-backtest')?.addEventListener('click', () => {
            this.runBacktest();
        });

        // Deployment button
        document.getElementById('deploy-strategy')?.addEventListener('click', () => {
            this.deployStrategy();
        });

        // Configuration management
        document.getElementById('export-config')?.addEventListener('click', () => {
            this.exportConfiguration();
        });

        document.getElementById('import-config')?.addEventListener('click', () => {
            document.getElementById('config-file-input').click();
        });

        document.getElementById('config-file-input')?.addEventListener('change', (e) => {
            this.importConfiguration(e.target.files[0]);
        });

        document.getElementById('save-template')?.addEventListener('click', () => {
            this.saveTemplate();
        });
    }

    selectInvestorProfile(profile) {
        // Update UI
        document.querySelectorAll('.profile-card').forEach(card => {
            card.classList.remove('border-blue-500', 'bg-blue-100');
        });
        
        document.querySelector(`[data-profile="${profile}"]`).classList.add('border-blue-500', 'bg-blue-100');

        this.currentInvestor = profile;

        // Apply profile-specific defaults
        this.applyProfileDefaults(profile);
        
        // Show next sections
        document.getElementById('parameter-customization').style.display = 'block';
        document.getElementById('backtesting-section').style.display = 'block';
        document.getElementById('risk-management').style.display = 'block';
        document.getElementById('deployment-section').style.display = 'block';
    }

    applyProfileDefaults(profile) {
        const defaults = {
            'hedge-fund': {
                risk_multiplier: 0.5,
                max_daily_loss: 100000,
                max_drawdown: 15,
                strategies: ['index_futures_spot', 'triangular_crypto', 'news_sentiment']
            },
            'pension-fund': {
                risk_multiplier: 0.1,
                max_daily_loss: 25000,
                max_drawdown: 5,
                strategies: ['index_futures_spot', 'statistical_pairs']
            },
            'prop-trading': {
                risk_multiplier: 1.0,
                max_daily_loss: 200000,
                max_drawdown: 25,
                strategies: ['triangular_crypto', 'latency_arbitrage', 'news_sentiment']
            }
        };

        const profileDefaults = defaults[profile] || defaults['hedge-fund'];

        // Update risk sliders
        document.getElementById('max-daily-loss').value = profileDefaults.max_daily_loss;
        document.getElementById('max-daily-loss-value').textContent = `$${profileDefaults.max_daily_loss.toLocaleString()}`;
        document.getElementById('max-drawdown-limit').value = profileDefaults.max_drawdown;
        document.getElementById('max-drawdown-value').textContent = `${profileDefaults.max_drawdown}%`;

        // Auto-select recommended strategies
        profileDefaults.strategies.forEach(strategyId => {
            const checkbox = document.querySelector(`[data-strategy="${strategyId}"]`);
            if (checkbox) {
                checkbox.checked = true;
                this.toggleStrategy(strategyId, true);
            }
        });
    }

    toggleStrategy(strategyId, enabled) {
        if (enabled) {
            this.createParameterPanel(strategyId);
        } else {
            this.removeParameterPanel(strategyId);
        }
        
        // Update strategy card appearance
        const strategyCard = document.querySelector(`[data-strategy="${strategyId}"]`);
        if (strategyCard) {
            if (enabled) {
                strategyCard.classList.add('border-blue-500', 'bg-blue-50');
            } else {
                strategyCard.classList.remove('border-blue-500', 'bg-blue-50');
            }
        }
    }

    createParameterPanel(strategyId) {
        const strategy = this.strategies[strategyId];
        if (!strategy) return;

        const panelsContainer = document.getElementById('parameter-panels');
        
        // Remove existing panel if it exists
        const existingPanel = document.getElementById(`panel-${strategyId}`);
        if (existingPanel) {
            existingPanel.remove();
        }

        const panel = document.createElement('div');
        panel.id = `panel-${strategyId}`;
        panel.className = 'parameter-panel bg-gray-50 rounded-lg p-4 border-l-4 border-blue-500';

        let parametersHTML = '';
        Object.entries(strategy.parameters).forEach(([paramName, paramConfig]) => {
            if (paramConfig.type === 'array') {
                parametersHTML += `
                    <div class="parameter-control">
                        <label class="block text-sm font-medium mb-1">${paramConfig.description}</label>
                        <div class="flex flex-wrap gap-2">
                            ${paramConfig.value.map(item => `<span class="bg-blue-100 text-blue-800 px-2 py-1 rounded text-sm">${item}</span>`).join('')}
                        </div>
                    </div>
                `;
            } else {
                parametersHTML += `
                    <div class="parameter-control">
                        <label class="block text-sm font-medium mb-1">${paramConfig.description}</label>
                        <div class="flex items-center space-x-4">
                            <input type="range" 
                                   class="parameter-input flex-1" 
                                   data-strategy="${strategyId}"
                                   data-parameter="${paramName}"
                                   min="${paramConfig.min}" 
                                   max="${paramConfig.max}" 
                                   step="${(paramConfig.max - paramConfig.min) / 100}"
                                   value="${paramConfig.value}">
                            <span class="parameter-value font-mono text-sm w-20 text-right" id="value-${strategyId}-${paramName}">
                                ${paramConfig.value}
                            </span>
                        </div>
                    </div>
                `;
            }
        });

        panel.innerHTML = `
            <div class="flex items-center justify-between mb-4">
                <h4 class="font-semibold text-gray-800">${strategy.name}</h4>
                <div class="flex items-center space-x-2">
                    <span class="text-xs bg-green-100 text-green-800 px-2 py-1 rounded">${strategy.roi_estimate}</span>
                    <button class="remove-strategy text-red-500 hover:text-red-700" data-strategy="${strategyId}">✕</button>
                </div>
            </div>
            <div class="grid grid-cols-1 md:grid-cols-2 gap-4">
                ${parametersHTML}
            </div>
        `;

        panelsContainer.appendChild(panel);

        // Add remove button listener
        panel.querySelector('.remove-strategy').addEventListener('click', (e) => {
            const strategyId = e.target.dataset.strategy;
            const checkbox = document.querySelector(`[data-strategy="${strategyId}"]`);
            if (checkbox) {
                checkbox.checked = false;
                this.toggleStrategy(strategyId, false);
            }
        });
    }

    removeParameterPanel(strategyId) {
        const panel = document.getElementById(`panel-${strategyId}`);
        if (panel) {
            panel.remove();
        }
    }

    updateParameter(strategyId, paramName, value) {
        // Update the display value
        const valueDisplay = document.getElementById(`value-${strategyId}-${paramName}`);
        if (valueDisplay) {
            valueDisplay.textContent = parseFloat(value).toFixed(2);
        }

        // Update internal configuration
        if (this.strategies[strategyId] && this.strategies[strategyId].parameters[paramName]) {
            this.strategies[strategyId].parameters[paramName].value = parseFloat(value);
        }

        // Trigger real-time validation if enabled
        this.validateConfiguration();
    }

    validateConfiguration() {
        // Real-time configuration validation
        const enabledStrategies = this.getEnabledStrategies();
        
        if (enabledStrategies.length === 0) {
            this.showValidationWarning('At least one strategy must be selected');
            return false;
        }

        // Check for parameter conflicts
        const conflicts = this.checkParameterConflicts(enabledStrategies);
        if (conflicts.length > 0) {
            this.showValidationWarning(`Parameter conflicts detected: ${conflicts.join(', ')}`);
            return false;
        }

        this.clearValidationWarnings();
        return true;
    }

    getEnabledStrategies() {
        return Array.from(document.querySelectorAll('.strategy-checkbox:checked'))
            .map(checkbox => checkbox.dataset.strategy);
    }

    checkParameterConflicts(strategies) {
        const conflicts = [];
        
        // Example conflict checks
        if (strategies.includes('latency_arbitrage') && strategies.includes('statistical_pairs')) {
            const latencyParam = this.strategies.latency_arbitrage.parameters.max_execution_latency_us.value;
            const pairsLatency = this.strategies.statistical_pairs.parameters.lookback_periods.value * 100; // Rough estimate
            
            if (latencyParam > 500 && pairsLatency > 25000) {
                conflicts.push('High latency settings conflict between strategies');
            }
        }

        return conflicts;
    }

    showValidationWarning(message) {
        // Implementation for showing validation warnings
        console.warn('Validation Warning:', message);
    }

    clearValidationWarnings() {
        // Implementation for clearing validation warnings
    }

    async runBacktest() {
        const runButton = document.getElementById('run-backtest');
        runButton.textContent = '⏳ Running Backtest...';
        runButton.disabled = true;

        try {
            const config = this.exportConfiguration(false);
            const period = document.getElementById('backtest-period').value;
            const initialCapital = parseFloat(document.getElementById('initial-capital').value);

            // Simulate backtest (in production, this would call the real backtesting engine)
            const results = await this.simulateBacktest(config, period, initialCapital);

            // Update UI with results
            this.displayBacktestResults(results);

            // Store results
            this.backtestResults.set(Date.now(), results);

        } catch (error) {
            console.error('Backtest error:', error);
            alert('Backtest failed: ' + error.message);
        } finally {
            runButton.textContent = '🚀 Run Backtest';
            runButton.disabled = false;
        }
    }

    async simulateBacktest(config, period, initialCapital) {
        // Simulate realistic backtest results
        await new Promise(resolve => setTimeout(resolve, 2000)); // Simulate processing time

        const strategies = this.getEnabledStrategies();
        const days = { '1d': 1, '7d': 7, '30d': 30, '90d': 90 }[period] || 30;

        // Simulate returns based on strategy mix and parameters
        let totalReturn = 0;
        let maxDrawdown = 0;
        let winRate = 0;
        let sharpeRatio = 0;
        let volatility = 0;

        strategies.forEach(strategyId => {
            const strategy = this.strategies[strategyId];
            const roiRange = strategy.roi_estimate.match(/(\d+)-(\d+)/);
            if (roiRange) {
                const minRoi = parseInt(roiRange[1]) / 100;
                const maxRoi = parseInt(roiRange[2]) / 100;
                const avgRoi = (minRoi + maxRoi) / 2;
                
                // Annualized to period
                const periodReturn = avgRoi * (days / 365);
                totalReturn += periodReturn / strategies.length;
            }
        });

        // Add realistic variations
        totalReturn *= (0.8 + Math.random() * 0.4); // 80-120% of expected
        maxDrawdown = Math.abs(totalReturn) * (0.1 + Math.random() * 0.2); // 10-30% of return
        winRate = 0.55 + Math.random() * 0.25; // 55-80% win rate
        volatility = Math.abs(totalReturn) * (1 + Math.random());
        sharpeRatio = totalReturn / (volatility || 0.1);

        // Generate daily P&L series
        const dailyReturns = [];
        let cumulativeReturn = 0;
        
        for (let i = 0; i < days; i++) {
            const dailyReturn = (totalReturn / days) + (Math.random() - 0.5) * 0.01;
            cumulativeReturn += dailyReturn;
            dailyReturns.push({
                date: new Date(Date.now() - (days - i) * 24 * 60 * 60 * 1000),
                return: dailyReturn,
                cumulative: cumulativeReturn,
                value: initialCapital * (1 + cumulativeReturn)
            });
        }

        return {
            totalReturn: totalReturn * 100, // Convert to percentage
            sharpeRatio,
            maxDrawdown: maxDrawdown * 100,
            winRate: winRate * 100,
            finalValue: initialCapital * (1 + totalReturn),
            dailyReturns,
            strategies: strategies,
            period: period,
            initialCapital
        };
    }

    displayBacktestResults(results) {
        document.getElementById('total-return').textContent = `${results.totalReturn.toFixed(2)}%`;
        document.getElementById('sharpe-ratio').textContent = results.sharpeRatio.toFixed(2);
        document.getElementById('max-drawdown').textContent = `${results.maxDrawdown.toFixed(2)}%`;
        document.getElementById('win-rate').textContent = `${results.winRate.toFixed(1)}%`;

        // Update chart
        this.updateBacktestChart(results.dailyReturns);
    }

    updateBacktestChart(dailyReturns) {
        const canvas = document.getElementById('backtest-chart');
        const ctx = canvas.getContext('2d');

        if (this.backtestChart) {
            this.backtestChart.destroy();
        }

        this.backtestChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: dailyReturns.map(d => d.date.toLocaleDateString()),
                datasets: [{
                    label: 'Portfolio Value',
                    data: dailyReturns.map(d => d.value),
                    borderColor: 'rgb(59, 130, 246)',
                    backgroundColor: 'rgba(59, 130, 246, 0.1)',
                    fill: true,
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return '$' + value.toLocaleString();
                            }
                        }
                    }
                },
                plugins: {
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return 'Value: $' + context.parsed.y.toLocaleString();
                            }
                        }
                    }
                }
            }
        });
    }

    async deployStrategy() {
        if (!this.validateConfiguration()) {
            alert('Please fix configuration issues before deploying');
            return;
        }

        const deployButton = document.getElementById('deploy-strategy');
        deployButton.textContent = '⏳ Deploying...';
        deployButton.disabled = true;

        try {
            const config = this.exportConfiguration(false);
            const environment = document.getElementById('deployment-env').value;
            const capital = parseFloat(document.getElementById('deployment-capital').value);

            // Simulate deployment process
            await this.simulateDeployment(config, environment, capital);

            // Update status indicators
            this.updateDeploymentStatus('deployed');

            alert(`Strategy successfully deployed to ${environment} environment with $${capital.toLocaleString()} capital allocation`);

        } catch (error) {
            console.error('Deployment error:', error);
            alert('Deployment failed: ' + error.message);
        } finally {
            deployButton.textContent = '📡 Deploy Strategy';
            deployButton.disabled = false;
        }
    }

    async simulateDeployment(config, environment, capital) {
        // Simulate deployment steps
        const steps = [
            'Validating configuration...',
            'Connecting to execution engine...',
            'Initializing strategy parameters...',
            'Starting market data feeds...',
            'Deploying to ' + environment + '...',
            'Deployment complete'
        ];

        for (let step of steps) {
            console.log(step);
            await new Promise(resolve => setTimeout(resolve, 500));
        }
    }

    updateDeploymentStatus(status) {
        const statusContainer = document.getElementById('deployment-status');
        
        if (status === 'deployed') {
            statusContainer.innerHTML = `
                <div class="status-indicator flex items-center space-x-2">
                    <div class="w-3 h-3 bg-green-400 rounded-full"></div>
                    <span class="text-sm">Configuration Active</span>
                </div>
                <div class="status-indicator flex items-center space-x-2">
                    <div class="w-3 h-3 bg-green-400 rounded-full"></div>
                    <span class="text-sm">Strategy Deployed</span>
                </div>
                <div class="status-indicator flex items-center space-x-2">
                    <div class="w-3 h-3 bg-blue-400 rounded-full animate-pulse"></div>
                    <span class="text-sm">Live Trading Active</span>
                </div>
            `;
        }
    }

    exportConfiguration(download = true) {
        const config = {
            investor_profile: this.currentInvestor,
            strategies: {},
            risk_management: {
                max_daily_loss: parseInt(document.getElementById('max-daily-loss')?.value || 50000),
                max_position_size: parseInt(document.getElementById('max-position-size')?.value || 1000000),
                max_drawdown: parseFloat(document.getElementById('max-drawdown-limit')?.value || 10) / 100
            },
            compliance: {
                position_reporting: document.getElementById('position-reporting')?.checked || true,
                risk_monitoring: document.getElementById('risk-monitoring')?.checked || true,
                audit_logging: document.getElementById('audit-logging')?.checked || true,
                circuit_breaker: document.getElementById('circuit-breaker')?.checked || true
            },
            deployment: {
                environment: document.getElementById('deployment-env')?.value || 'sandbox',
                capital_allocation: parseFloat(document.getElementById('deployment-capital')?.value || 100000),
                strategy_weight: parseFloat(document.getElementById('strategy-weight')?.value || 25) / 100
            },
            timestamp: new Date().toISOString(),
            version: '1.0'
        };

        // Get enabled strategies and their parameters
        this.getEnabledStrategies().forEach(strategyId => {
            config.strategies[strategyId] = {
                enabled: true,
                parameters: { ...this.strategies[strategyId].parameters }
            };
        });

        if (download) {
            const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `hypervision-config-${Date.now()}.json`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        return config;
    }

    async importConfiguration(file) {
        if (!file) return;

        try {
            const text = await file.text();
            const config = JSON.parse(text);

            // Validate configuration structure
            if (!config.investor_profile || !config.strategies) {
                throw new Error('Invalid configuration file format');
            }

            // Apply configuration
            this.applyImportedConfiguration(config);

            alert('Configuration imported successfully');

        } catch (error) {
            console.error('Import error:', error);
            alert('Failed to import configuration: ' + error.message);
        }
    }

    applyImportedConfiguration(config) {
        // Set investor profile
        if (config.investor_profile) {
            this.selectInvestorProfile(config.investor_profile);
        }

        // Enable strategies and set parameters
        Object.entries(config.strategies || {}).forEach(([strategyId, strategyConfig]) => {
            if (strategyConfig.enabled) {
                const checkbox = document.querySelector(`[data-strategy="${strategyId}"]`);
                if (checkbox) {
                    checkbox.checked = true;
                    this.toggleStrategy(strategyId, true);

                    // Set parameters
                    Object.entries(strategyConfig.parameters || {}).forEach(([paramName, paramConfig]) => {
                        if (paramConfig.value !== undefined) {
                            const input = document.querySelector(`[data-strategy="${strategyId}"][data-parameter="${paramName}"]`);
                            if (input) {
                                input.value = paramConfig.value;
                                this.updateParameter(strategyId, paramName, paramConfig.value);
                            }
                        }
                    });
                }
            }
        });

        // Set risk management parameters
        if (config.risk_management) {
            const riskConfig = config.risk_management;
            if (riskConfig.max_daily_loss) {
                const slider = document.getElementById('max-daily-loss');
                if (slider) slider.value = riskConfig.max_daily_loss;
            }
            if (riskConfig.max_position_size) {
                const slider = document.getElementById('max-position-size');
                if (slider) slider.value = riskConfig.max_position_size;
            }
            if (riskConfig.max_drawdown) {
                const slider = document.getElementById('max-drawdown-limit');
                if (slider) slider.value = riskConfig.max_drawdown * 100;
            }
        }

        // Set compliance settings
        if (config.compliance) {
            Object.entries(config.compliance).forEach(([setting, value]) => {
                const checkbox = document.getElementById(setting.replace('_', '-'));
                if (checkbox) checkbox.checked = value;
            });
        }
    }

    saveTemplate() {
        const templateName = prompt('Enter template name:');
        if (!templateName) return;

        const config = this.exportConfiguration(false);
        config.template_name = templateName;

        // Save to localStorage (in production, save to server)
        const templates = JSON.parse(localStorage.getItem('hypervision_templates') || '[]');
        templates.push(config);
        localStorage.setItem('hypervision_templates', JSON.stringify(templates));

        alert(`Template "${templateName}" saved successfully`);
    }

    loadInvestorProfiles() {
        // Load saved templates and configurations
        const templates = JSON.parse(localStorage.getItem('hypervision_templates') || '[]');
        console.log('Loaded templates:', templates);
    }

    initializeCharts() {
        // Initialize Chart.js defaults
        Chart.defaults.font.family = 'Inter, system-ui, sans-serif';
        Chart.defaults.color = '#374151';
    }
}

// Initialize the customization interface when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    // Create container if it doesn't exist
    if (!document.getElementById('investor-customization')) {
        const container = document.createElement('div');
        container.id = 'investor-customization';
        container.className = 'container mx-auto px-4 py-8';
        document.body.appendChild(container);
    }

    // Initialize interface
    window.investorInterface = new InvestorCustomizationInterface();
});

// Export for external use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = InvestorCustomizationInterface;
}