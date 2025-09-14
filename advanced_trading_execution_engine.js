/**
 * GOMNA ADVANCED TRADING EXECUTION ENGINE
 * Novel AI-driven trading platform with seamless account creation and execution
 * Supports manual trading, automated AI agents, and hybrid approaches
 */

class GomnaTradeExecutionEngine {
    constructor(paymentSystem, config = {}) {
        this.paymentSystem = paymentSystem;
        this.config = {
            apiBaseUrl: config.apiBaseUrl || '/api/trading',
            maxPositionSize: config.maxPositionSize || 0.25, // 25% max position
            riskTolerance: config.riskTolerance || 'moderate',
            enableAI: config.enableAI || true,
            ...config
        };
        
        // Core components
        this.orderBook = new Map();
        this.portfolioManager = null;
        this.aiAgents = new Map();
        this.riskManager = null;
        this.marketDataStream = null;
        
        // Trading state
        this.isConnected = false;
        this.currentUser = null;
        this.activePositions = new Map();
        this.tradingEnabled = false;
        
        this.init();
    }

    async init() {
        try {
            this.portfolioManager = new PortfolioManager(this);
            this.riskManager = new RiskManager(this);
            this.marketDataStream = new MarketDataStream(this);
            
            await this.setupTradingEnvironment();
            console.log('🚀 Gomna Trading Engine initialized successfully');
        } catch (error) {
            console.error('❌ Failed to initialize trading engine:', error);
        }
    }

    /**
     * SEAMLESS ACCOUNT CREATION & TRADING SETUP
     */
    async createTradingAccount(userData, tradingPreferences = {}) {
        try {
            // Step 1: Create payment account through integrated system
            const paymentResult = await this.paymentSystem.startUserOnboarding(userData);
            
            if (!paymentResult.success) {
                throw new Error('Failed to create payment account: ' + paymentResult.error);
            }

            // Step 2: Setup trading-specific components
            const tradingSetup = await this.setupTradingProfile({
                userId: paymentResult.userId,
                walletId: paymentResult.walletId,
                preferences: {
                    riskProfile: tradingPreferences.riskProfile || 'moderate',
                    tradingStyle: tradingPreferences.tradingStyle || 'balanced',
                    aiAssistance: tradingPreferences.aiAssistance !== false,
                    autoRebalancing: tradingPreferences.autoRebalancing || false,
                    maxDrawdown: tradingPreferences.maxDrawdown || 0.15, // 15%
                    preferredAssets: tradingPreferences.preferredAssets || ['SPY', 'BTC', 'ETH'],
                    notificationPreferences: tradingPreferences.notifications || {
                        email: true,
                        sms: false,
                        push: true,
                        slack: false
                    }
                }
            });

            this.currentUser = {
                ...paymentResult,
                ...tradingSetup,
                createdAt: new Date().toISOString()
            };

            // Step 3: Initialize AI trading agents if enabled
            if (tradingPreferences.aiAssistance !== false) {
                await this.initializeAIAgents();
            }

            return {
                success: true,
                user: this.currentUser,
                message: 'Trading account created successfully! Ready to start trading.'
            };

        } catch (error) {
            console.error('Account creation failed:', error);
            return { success: false, error: error.message };
        }
    }

    async setupTradingProfile(profileData) {
        const { userId, walletId, preferences } = profileData;
        
        try {
            const tradingProfile = await this.apiCall('/trading/profile/create', {
                method: 'POST',
                body: JSON.stringify({
                    userId,
                    walletId,
                    riskProfile: preferences.riskProfile,
                    tradingStyle: preferences.tradingStyle,
                    maxDrawdown: preferences.maxDrawdown,
                    preferredAssets: preferences.preferredAssets,
                    aiSettings: {
                        enabled: preferences.aiAssistance,
                        autoRebalancing: preferences.autoRebalancing,
                        riskManagement: true,
                        marketAnalysis: true
                    }
                })
            });

            return {
                tradingProfileId: tradingProfile.profileId,
                riskLimits: tradingProfile.riskLimits,
                tradingPermissions: tradingProfile.permissions,
                aiConfiguration: tradingProfile.aiConfig
            };
        } catch (error) {
            throw new Error('Failed to setup trading profile: ' + error.message);
        }
    }

    /**
     * AI AGENT SYSTEM - NOVEL APPROACH
     */
    async initializeAIAgents() {
        const agentConfigs = [
            {
                id: 'market_analyzer',
                type: 'MarketAnalysisAgent',
                description: 'Real-time market sentiment and trend analysis',
                capabilities: ['sentiment_analysis', 'trend_detection', 'volatility_prediction'],
                enabled: true
            },
            {
                id: 'portfolio_optimizer',
                type: 'PortfolioOptimizationAgent',
                description: 'Hyperbolic geometry portfolio optimization',
                capabilities: ['risk_parity', 'momentum_allocation', 'rebalancing'],
                enabled: true
            },
            {
                id: 'risk_guardian',
                type: 'RiskManagementAgent',
                description: 'Dynamic risk monitoring and protection',
                capabilities: ['stop_loss_management', 'position_sizing', 'drawdown_protection'],
                enabled: true
            },
            {
                id: 'execution_optimizer',
                type: 'ExecutionOptimizationAgent',
                description: 'Smart order routing and timing',
                capabilities: ['order_splitting', 'market_impact_reduction', 'timing_optimization'],
                enabled: true
            },
            {
                id: 'opportunity_scout',
                type: 'OpportunityDetectionAgent',
                description: 'Identifies trading opportunities across markets',
                capabilities: ['arbitrage_detection', 'momentum_signals', 'mean_reversion'],
                enabled: this.currentUser?.preferences?.aggressive || false
            }
        ];

        for (const config of agentConfigs) {
            if (config.enabled) {
                const agent = await this.createAIAgent(config);
                this.aiAgents.set(config.id, agent);
            }
        }

        console.log(`🤖 Initialized ${this.aiAgents.size} AI trading agents`);
    }

    async createAIAgent(config) {
        const AgentClass = this.getAgentClass(config.type);
        const agent = new AgentClass({
            id: config.id,
            userId: this.currentUser.userId,
            capabilities: config.capabilities,
            tradingEngine: this,
            riskLimits: this.currentUser.riskLimits
        });

        await agent.initialize();
        return agent;
    }

    getAgentClass(type) {
        const agentClasses = {
            'MarketAnalysisAgent': MarketAnalysisAgent,
            'PortfolioOptimizationAgent': PortfolioOptimizationAgent,
            'RiskManagementAgent': RiskManagementAgent,
            'ExecutionOptimizationAgent': ExecutionOptimizationAgent,
            'OpportunityDetectionAgent': OpportunityDetectionAgent
        };
        return agentClasses[type] || BaseAIAgent;
    }

    /**
     * MANUAL TRADING INTERFACE
     */
    async executeTrade(tradeData) {
        const { 
            symbol, 
            side, 
            quantity, 
            orderType = 'market', 
            price = null,
            stopLoss = null,
            takeProfit = null,
            timeInForce = 'DAY'
        } = tradeData;

        try {
            // Pre-trade risk assessment
            const riskAssessment = await this.riskManager.assessTrade({
                symbol,
                side,
                quantity,
                currentPortfolio: this.portfolioManager.getCurrentPortfolio()
            });

            if (!riskAssessment.approved) {
                return {
                    success: false,
                    error: `Trade rejected: ${riskAssessment.reason}`,
                    suggestions: riskAssessment.suggestions
                };
            }

            // Get AI agent recommendations
            const aiRecommendations = await this.getAIRecommendations(tradeData);

            // Execute the trade
            const order = await this.submitOrder({
                symbol,
                side,
                quantity: riskAssessment.adjustedQuantity || quantity,
                orderType,
                price,
                stopLoss,
                takeProfit,
                timeInForce,
                metadata: {
                    source: 'manual',
                    riskScore: riskAssessment.riskScore,
                    aiRecommendations
                }
            });

            return {
                success: true,
                orderId: order.orderId,
                status: order.status,
                aiInsights: aiRecommendations,
                riskAssessment
            };

        } catch (error) {
            console.error('Trade execution failed:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * AI AUTOMATED TRADING
     */
    async enableAutomatedTrading(settings = {}) {
        try {
            const automationConfig = {
                enabled: true,
                agentPermissions: settings.agentPermissions || {
                    market_analyzer: { canTrade: false, canRecommend: true },
                    portfolio_optimizer: { canTrade: true, canRecommend: true, maxAllocation: 0.1 },
                    risk_guardian: { canTrade: true, canRecommend: true, emergencyStop: true },
                    execution_optimizer: { canTrade: false, canOptimize: true },
                    opportunity_scout: { canTrade: settings.aggressive || false, canRecommend: true }
                },
                safetyLimits: {
                    maxTradesPerHour: settings.maxTradesPerHour || 5,
                    maxDailyRisk: settings.maxDailyRisk || 0.05, // 5%
                    requireHumanApproval: settings.requireHumanApproval || false,
                    emergencyStopLoss: settings.emergencyStopLoss || 0.15 // 15%
                }
            };

            // Start automated trading loops for each agent
            for (const [agentId, agent] of this.aiAgents) {
                const permissions = automationConfig.agentPermissions[agentId];
                if (permissions?.canTrade) {
                    agent.startAutomation(permissions);
                }
            }

            this.tradingEnabled = true;
            
            return {
                success: true,
                message: 'Automated trading enabled with AI agents',
                activeAgents: Array.from(this.aiAgents.keys()),
                safetyLimits: automationConfig.safetyLimits
            };

        } catch (error) {
            console.error('Failed to enable automated trading:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * HYBRID TRADING - NOVEL APPROACH
     */
    async enableHybridTrading(hybridConfig = {}) {
        // Combine human oversight with AI execution
        const config = {
            aiAssistanceLevel: hybridConfig.aiAssistanceLevel || 'moderate', // low, moderate, high
            humanApprovalRequired: hybridConfig.humanApprovalRequired || ['large_orders', 'new_assets'],
            aiCanSuggest: hybridConfig.aiCanSuggest !== false,
            aiCanExecuteSmall: hybridConfig.aiCanExecuteSmall !== false,
            collaborationMode: hybridConfig.collaborationMode || 'advisory', // advisory, collaborative, autonomous
            ...hybridConfig
        };

        try {
            // Setup hybrid workflow
            await this.setupHybridWorkflow(config);
            
            return {
                success: true,
                message: 'Hybrid trading mode activated',
                config: config
            };
        } catch (error) {
            return { success: false, error: error.message };
        }
    }

    async setupHybridWorkflow(config) {
        // Create intelligent routing between human and AI decisions
        this.hybridRouter = new HybridTradingRouter({
            engine: this,
            config: config,
            decisionMatrix: {
                'small_trade': { threshold: 1000, requireApproval: false, aiCanExecute: true },
                'medium_trade': { threshold: 10000, requireApproval: config.humanApprovalRequired.includes('medium_orders'), aiCanExecute: true },
                'large_trade': { threshold: 50000, requireApproval: true, aiCanExecute: false },
                'new_asset': { requireApproval: config.humanApprovalRequired.includes('new_assets'), aiCanExecute: false },
                'high_risk': { requireApproval: true, aiCanExecute: false }
            }
        });
    }

    /**
     * REAL-TIME MARKET DATA & EXECUTION
     */
    async setupTradingEnvironment() {
        // Initialize market data feeds
        await this.marketDataStream.connect([
            'stocks', 'crypto', 'forex', 'commodities'
        ]);

        // Setup order routing
        await this.initializeOrderRouting();

        // Start risk monitoring
        this.riskManager.startMonitoring();

        this.isConnected = true;
    }

    async initializeOrderRouting() {
        console.log('🔧 Initializing order routing system...');
        
        // Configure execution venues and routing algorithms
        this.executionVenues = new Map([
            ['stocks', { primary: 'NYSE', secondary: 'NASDAQ', latency: 2 }],
            ['crypto', { primary: 'Coinbase', secondary: 'Binance', latency: 1 }],
            ['forex', { primary: 'FXCM', secondary: 'OANDA', latency: 3 }],
            ['commodities', { primary: 'CME', secondary: 'CBOT', latency: 4 }]
        ]);
        
        // Initialize smart order routing
        this.smartRouting = {
            algorithm: 'TWAP', // Time-weighted average price
            sliceSize: 0.1, // 10% of order size per slice
            maxLatency: 50, // milliseconds
            darkPoolAccess: true,
            ecnAccess: true
        };
        
        // Setup order validation rules
        this.orderValidation = {
            maxOrderSize: 1000000, // $1M
            minOrderSize: 1, // $1
            allowedOrderTypes: ['market', 'limit', 'stop', 'stop_limit'],
            marketHours: { start: '09:30', end: '16:00', timezone: 'EST' }
        };
        
        console.log('✅ Order routing initialized successfully');
    }

    async submitOrder(orderData) {
        try {
            // Route through AI optimization if enabled
            if (this.aiAgents.has('execution_optimizer')) {
                const optimizer = this.aiAgents.get('execution_optimizer');
                orderData = await optimizer.optimizeOrder(orderData);
            }

            // Submit to execution venue
            const result = await this.apiCall('/trading/orders/submit', {
                method: 'POST',
                body: JSON.stringify({
                    ...orderData,
                    userId: this.currentUser.userId,
                    walletId: this.currentUser.walletId,
                    timestamp: Date.now()
                })
            });

            // Track in portfolio
            this.portfolioManager.addOrder(result);
            
            return result;

        } catch (error) {
            throw new Error('Order submission failed: ' + error.message);
        }
    }

    async getAIRecommendations(tradeData) {
        const recommendations = {};
        
        for (const [agentId, agent] of this.aiAgents) {
            try {
                const rec = await agent.analyzeTradeOpportunity(tradeData);
                recommendations[agentId] = rec;
            } catch (error) {
                console.error(`AI recommendation failed for ${agentId}:`, error);
            }
        }

        return recommendations;
    }

    /**
     * PORTFOLIO & PERFORMANCE TRACKING
     */
    getPortfolioSummary() {
        return this.portfolioManager.getSummary();
    }

    getPerformanceMetrics() {
        return this.portfolioManager.getPerformanceMetrics();
    }

    /**
     * UTILITY METHODS
     */
    async apiCall(endpoint, options = {}) {
        const url = `${this.config.apiBaseUrl}${endpoint}`;
        const defaultHeaders = {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${this.currentUser?.apiToken}`,
            'X-User-ID': this.currentUser?.userId
        };

        const response = await fetch(url, {
            headers: { ...defaultHeaders, ...(options.headers || {}) },
            ...options
        });

        if (!response.ok) {
            throw new Error(`API call failed: ${response.status} ${response.statusText}`);
        }

        return await response.json();
    }

    /**
     * EVENT SYSTEM
     */
    setupEventListeners() {
        document.addEventListener('trading:order-filled', (e) => {
            this.handleOrderFilled(e.detail);
        });

        document.addEventListener('trading:risk-alert', (e) => {
            this.handleRiskAlert(e.detail);
        });

        document.addEventListener('trading:ai-recommendation', (e) => {
            this.handleAIRecommendation(e.detail);
        });
    }

    handleOrderFilled(orderData) {
        this.portfolioManager.updatePosition(orderData);
        
        // Emit success notification
        document.dispatchEvent(new CustomEvent('trading:notification', {
            detail: {
                type: 'success',
                message: `Order filled: ${orderData.symbol} ${orderData.side} ${orderData.quantity}`,
                data: orderData
            }
        }));
    }

    handleRiskAlert(alertData) {
        console.warn('Risk Alert:', alertData);
        
        // Auto-handle critical alerts
        if (alertData.severity === 'critical') {
            this.emergencyStopTrading(alertData.reason);
        }
    }

    async emergencyStopTrading(reason) {
        this.tradingEnabled = false;
        
        // Stop all AI agents
        for (const agent of this.aiAgents.values()) {
            agent.stopAutomation();
        }
        
        // Notify user
        document.dispatchEvent(new CustomEvent('trading:emergency-stop', {
            detail: { reason }
        }));
    }
}

/**
 * SUPPORTING CLASSES
 */

class PortfolioManager {
    constructor(tradingEngine) {
        this.tradingEngine = tradingEngine;
        this.positions = new Map();
        this.orders = new Map();
        this.performance = {
            totalReturn: 0,
            sharpeRatio: 0,
            maxDrawdown: 0,
            winRate: 0
        };
    }

    getCurrentPortfolio() {
        return {
            positions: Array.from(this.positions.values()),
            totalValue: this.calculateTotalValue(),
            allocation: this.calculateAllocation()
        };
    }

    calculateTotalValue() {
        let total = 0;
        for (const position of this.positions.values()) {
            total += position.marketValue;
        }
        return total;
    }

    calculateAllocation() {
        const total = this.calculateTotalValue();
        const allocation = {};
        
        for (const [symbol, position] of this.positions) {
            allocation[symbol] = position.marketValue / total;
        }
        
        return allocation;
    }

    addOrder(orderData) {
        this.orders.set(orderData.orderId, orderData);
    }

    updatePosition(orderData) {
        const symbol = orderData.symbol;
        const existing = this.positions.get(symbol);
        
        if (existing) {
            // Update existing position
            this.positions.set(symbol, this.mergePositions(existing, orderData));
        } else {
            // Create new position
            this.positions.set(symbol, this.createPosition(orderData));
        }
    }

    getSummary() {
        return {
            totalValue: this.calculateTotalValue(),
            positionCount: this.positions.size,
            allocation: this.calculateAllocation(),
            performance: this.performance
        };
    }

    getPerformanceMetrics() {
        // Calculate real-time performance metrics
        return this.performance;
    }
}

class RiskManager {
    constructor(tradingEngine) {
        this.tradingEngine = tradingEngine;
        this.riskLimits = {
            maxPositionSize: 0.25,
            maxDailyLoss: 0.05,
            maxDrawdown: 0.15,
            maxLeverage: 2.0
        };
        this.dailyPnL = 0;
        this.maxDrawdownReached = 0;
    }

    async assessTrade(tradeData) {
        const checks = [
            this.checkPositionSize(tradeData),
            this.checkDailyLimits(tradeData),
            this.checkDrawdownLimits(tradeData),
            this.checkCorrelationRisk(tradeData)
        ];

        const results = await Promise.all(checks);
        const failed = results.filter(r => !r.passed);

        if (failed.length > 0) {
            return {
                approved: false,
                reason: failed.map(f => f.reason).join(', '),
                suggestions: failed.map(f => f.suggestion).filter(Boolean)
            };
        }

        return {
            approved: true,
            riskScore: this.calculateRiskScore(tradeData),
            adjustedQuantity: this.getOptimalQuantity(tradeData)
        };
    }

    checkPositionSize(tradeData) {
        const portfolio = this.tradingEngine.portfolioManager.getCurrentPortfolio();
        const proposedValue = tradeData.quantity * (tradeData.price || this.getMarketPrice(tradeData.symbol));
        const positionRatio = proposedValue / portfolio.totalValue;

        if (positionRatio > this.riskLimits.maxPositionSize) {
            return {
                passed: false,
                reason: `Position size exceeds ${this.riskLimits.maxPositionSize * 100}% limit`,
                suggestion: `Reduce quantity to ${Math.floor(tradeData.quantity * this.riskLimits.maxPositionSize / positionRatio)}`
            };
        }

        return { passed: true };
    }

    checkDailyLimits(tradeData) {
        if (Math.abs(this.dailyPnL) > this.riskLimits.maxDailyLoss) {
            return {
                passed: false,
                reason: 'Daily loss limit exceeded',
                suggestion: 'Wait until next trading day'
            };
        }

        return { passed: true };
    }

    startMonitoring() {
        // Real-time risk monitoring
        setInterval(() => {
            this.checkPortfolioRisk();
        }, 5000); // Check every 5 seconds
    }

    calculateDrawdown(portfolio) {
        // Calculate maximum drawdown from peak to current value
        if (!portfolio || !portfolio.history || portfolio.history.length < 2) {
            return 0;
        }
        
        const values = portfolio.history.map(p => p.totalValue);
        let peak = values[0];
        let maxDrawdown = 0;
        
        for (let i = 1; i < values.length; i++) {
            if (values[i] > peak) {
                peak = values[i];
            }
            
            const drawdown = (peak - values[i]) / peak;
            if (drawdown > maxDrawdown) {
                maxDrawdown = drawdown;
            }
        }
        
        return maxDrawdown;
    }

    checkPortfolioRisk() {
        const portfolio = this.tradingEngine.portfolioManager.getCurrentPortfolio();
        
        // Check drawdown
        const currentDrawdown = this.calculateDrawdown(portfolio);
        if (currentDrawdown > this.riskLimits.maxDrawdown) {
            this.tradingEngine.handleRiskAlert({
                type: 'drawdown',
                severity: 'critical',
                message: `Drawdown limit exceeded: ${currentDrawdown * 100}%`
            });
        }
    }
}

class MarketDataStream {
    constructor(tradingEngine) {
        this.tradingEngine = tradingEngine;
        this.websocket = null;
        this.subscriptions = new Set();
        this.marketData = new Map();
    }

    async connect(instruments) {
        try {
            this.websocket = new WebSocket('wss://api.gomna.ai/market-data');
            
            this.websocket.onopen = () => {
                console.log('📡 Market data stream connected');
                this.subscribe(instruments);
            };

            this.websocket.onmessage = (event) => {
                this.handleMarketData(JSON.parse(event.data));
            };

            this.websocket.onerror = (error) => {
                console.error('Market data stream error:', error);
            };

        } catch (error) {
            console.error('Failed to connect market data stream:', error);
        }
    }

    subscribe(instruments) {
        const message = {
            action: 'subscribe',
            instruments: instruments
        };
        
        this.websocket.send(JSON.stringify(message));
    }

    handleMarketData(data) {
        this.marketData.set(data.symbol, data);
        
        // Notify AI agents of new data
        for (const agent of this.tradingEngine.aiAgents.values()) {
            agent.onMarketData(data);
        }
    }
}

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { 
        GomnaTradeExecutionEngine,
        PortfolioManager,
        RiskManager,
        MarketDataStream
    };
}

// Global access
window.GomnaTradeExecutionEngine = GomnaTradeExecutionEngine;

console.log('🚀 Advanced Trading Execution Engine loaded successfully');