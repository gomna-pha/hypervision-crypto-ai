/**
 * GOMNA HYPERVISION ARBITRAGE INTEGRATION
 * Seamlessly integrates the new HFT Arbitrage system with the existing HyperVision platform
 * Maintains compatibility while adding advanced arbitrage capabilities
 */

class HyperVisionArbitrageIntegration {
    constructor() {
        this.arbitrageEngine = null;
        this.arbitrageUI = null;
        this.tradingEngine = null;
        this.isIntegrated = false;
        this.integrationMode = 'enhanced'; // 'enhanced' or 'standalone'
        
        this.init();
    }

    async init() {
        console.log('🔗 Initializing HyperVision Arbitrage Integration...');
        
        try {
            // Wait for existing platform components to load
            await this.waitForPlatformComponents();
            
            // Initialize arbitrage system
            await this.initializeArbitrageSystem();
            
            // Integrate with existing UI
            await this.integrateWithExistingUI();
            
            // Setup event bridges
            this.setupEventBridges();
            
            // Add arbitrage controls to existing panels
            this.enhanceExistingPanels();
            
            this.isIntegrated = true;
            console.log('✅ HyperVision Arbitrage Integration complete');
            
            // Show integration notification
            this.showIntegrationNotification();
            
        } catch (error) {
            console.error('❌ Arbitrage integration failed:', error);
            this.handleIntegrationFailure(error);
        }
    }

    async waitForPlatformComponents() {
        // Wait for essential HyperVision components to be available
        const maxWaitTime = 30000; // 30 seconds
        const checkInterval = 500; // 500ms
        let elapsed = 0;

        return new Promise((resolve, reject) => {
            const checkComponents = () => {
                const componentsReady = 
                    typeof window.GomnaTradeExecutionEngine !== 'undefined' &&
                    document.querySelector('.trading-dashboard') !== null;

                if (componentsReady) {
                    console.log('✅ Platform components ready');
                    resolve();
                } else if (elapsed >= maxWaitTime) {
                    reject(new Error('Platform components did not load in time'));
                } else {
                    elapsed += checkInterval;
                    setTimeout(checkComponents, checkInterval);
                }
            };

            checkComponents();
        });
    }

    async initializeArbitrageSystem() {
        console.log('🚀 Initializing arbitrage system...');
        
        // Get existing trading engine if available
        this.tradingEngine = window.tradingEngine || null;
        
        // Initialize arbitrage engine
        this.arbitrageEngine = new ArbitrageEngine(this.tradingEngine);
        
        // Wait for arbitrage engine to be ready
        await new Promise(resolve => {
            const checkReady = () => {
                if (this.arbitrageEngine.isConnected || 
                    (this.arbitrageEngine.strategies && this.arbitrageEngine.strategies.size > 0)) {
                    resolve();
                } else {
                    setTimeout(checkReady, 100);
                }
            };
            checkReady();
        });
        
        console.log('✅ Arbitrage engine initialized');
    }

    async integrateWithExistingUI() {
        console.log('🎨 Integrating with existing UI...');
        
        // Create arbitrage panel container
        const arbitragePanelContainer = this.createArbitragePanelContainer();
        
        // Initialize arbitrage UI
        this.arbitrageUI = new HFTArbitrageUI(arbitragePanelContainer, this.arbitrageEngine);
        
        // Add to existing dashboard
        this.addToExistingDashboard(arbitragePanelContainer);
        
        console.log('✅ UI integration complete');
    }

    createArbitragePanelContainer() {
        const container = document.createElement('div');
        container.id = 'arbitrage-panel-container';
        container.className = 'panel-container arbitrage-enhanced';
        container.style.cssText = `
            grid-column: 1 / -1;
            margin: 20px 0;
            border-radius: 12px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
            border: 2px solid #00ff88;
            box-shadow: 0 8px 32px rgba(0, 255, 136, 0.2);
            overflow: hidden;
        `;
        
        return container;
    }

    addToExistingDashboard(container) {
        // Find the best location to insert the arbitrage panel
        const tradingDashboard = document.querySelector('.trading-dashboard') || 
                                 document.querySelector('.main-dashboard') ||
                                 document.querySelector('#dashboard') ||
                                 document.body;

        if (tradingDashboard) {
            // Insert after AI trading agent panel if it exists
            const aiPanel = document.querySelector('.ai-trading-panel') || 
                           document.querySelector('[data-panel="trading"]') ||
                           tradingDashboard.children[0];
            
            if (aiPanel && aiPanel.nextSibling) {
                tradingDashboard.insertBefore(container, aiPanel.nextSibling);
            } else {
                tradingDashboard.appendChild(container);
            }
            
            console.log('✅ Arbitrage panel added to dashboard');
        } else {
            console.warn('⚠️ Dashboard not found, appending to body');
            document.body.appendChild(container);
        }
    }

    setupEventBridges() {
        console.log('🌉 Setting up event bridges...');
        
        // Bridge arbitrage events to existing platform
        if (this.arbitrageEngine) {
            // Listen for arbitrage opportunities
            this.arbitrageEngine.on = this.arbitrageEngine.on || ((event, callback) => {
                document.addEventListener(`arbitrage:${event}`, callback);
            });

            // Simulate event system if not present
            this.simulateArbitrageEvents();
        }

        // Bridge existing platform events to arbitrage system
        this.bridgeExistingEvents();
        
        // Setup trade execution bridge
        this.setupTradeExecutionBridge();
        
        console.log('✅ Event bridges established');
    }

    simulateArbitrageEvents() {
        // Simulate arbitrage opportunity events
        setInterval(() => {
            const mockOpportunity = {
                strategy: 'cross_exchange',
                symbol: 'BTC/USD',
                profit: Math.random() * 0.005 + 0.001, // 0.1% to 0.6% profit
                confidence: Math.random() * 0.4 + 0.6, // 60% to 100% confidence
                timestamp: Date.now()
            };

            document.dispatchEvent(new CustomEvent('arbitrage:opportunity_found', {
                detail: mockOpportunity
            }));
        }, 5000 + Math.random() * 10000); // Every 5-15 seconds

        // Simulate execution events
        setInterval(() => {
            const mockExecution = {
                executionId: 'arb_' + Date.now(),
                strategy: 'statistical',
                profit: Math.random() * 0.003 + 0.0005, // 0.05% to 0.35% profit
                status: Math.random() > 0.1 ? 'success' : 'failed',
                latency: Math.random() * 3 + 1, // 1-4ms latency
                timestamp: Date.now()
            };

            document.dispatchEvent(new CustomEvent('arbitrage:execution_complete', {
                detail: mockExecution
            }));
        }, 8000 + Math.random() * 12000); // Every 8-20 seconds
    }

    bridgeExistingEvents() {
        // Listen for existing platform events and forward to arbitrage system
        
        // Market data updates
        document.addEventListener('market:price_update', (e) => {
            if (this.arbitrageEngine && this.arbitrageEngine.marketDataStream) {
                this.arbitrageEngine.marketDataStream.handleMarketData(e.detail);
            }
        });

        // Trade executions
        document.addEventListener('trading:order-filled', (e) => {
            this.handleTradeExecution(e.detail);
        });

        // Risk alerts
        document.addEventListener('trading:risk-alert', (e) => {
            this.handleRiskAlert(e.detail);
        });
    }

    setupTradeExecutionBridge() {
        // Override or enhance existing trade execution
        if (window.tradingEngine && typeof window.tradingEngine.executeTrade === 'function') {
            const originalExecuteTrade = window.tradingEngine.executeTrade.bind(window.tradingEngine);
            
            window.tradingEngine.executeTrade = async (tradeData) => {
                // Check for arbitrage opportunities first
                const arbOpportunity = await this.checkArbitrageOpportunity(tradeData);
                
                if (arbOpportunity && arbOpportunity.profitable) {
                    console.log('🎯 Arbitrage opportunity detected, optimizing trade...');
                    return this.executeArbitrageOptimizedTrade(tradeData, arbOpportunity);
                }
                
                // Fall back to original execution
                return originalExecuteTrade(tradeData);
            };
            
            console.log('✅ Trade execution bridge established');
        }
    }

    async checkArbitrageOpportunity(tradeData) {
        if (!this.arbitrageEngine) return null;

        try {
            // Check if trade could be part of arbitrage strategy
            const symbol = tradeData.symbol;
            const side = tradeData.side || tradeData.action;
            const quantity = tradeData.quantity;

            // Simulate arbitrage opportunity check
            const opportunity = {
                profitable: Math.random() > 0.7, // 30% chance of arbitrage opportunity
                strategy: 'cross_exchange',
                additionalProfit: Math.random() * 0.002, // Up to 0.2% additional profit
                executionPlan: {
                    primaryExchange: 'binance',
                    secondaryExchange: 'coinbase',
                    optimalTiming: Date.now() + 100 // Execute in 100ms
                }
            };

            return opportunity;
        } catch (error) {
            console.error('Arbitrage opportunity check failed:', error);
            return null;
        }
    }

    async executeArbitrageOptimizedTrade(tradeData, arbOpportunity) {
        try {
            console.log('🚀 Executing arbitrage-optimized trade...');
            
            // Execute the arbitrage strategy
            const arbResult = await this.arbitrageEngine.executeArbitrageOpportunity({
                type: arbOpportunity.strategy,
                symbol: tradeData.symbol,
                side: tradeData.side || tradeData.action,
                quantity: tradeData.quantity,
                executionPlan: arbOpportunity.executionPlan
            }, arbOpportunity.strategy);

            // Execute the original trade as well
            const originalResult = await window.tradingEngine.executeTrade(tradeData);

            // Combine results
            return {
                success: true,
                arbitrageProfit: arbOpportunity.additionalProfit,
                combinedExecution: {
                    arbitrage: arbResult,
                    original: originalResult
                },
                message: `Trade executed with ${(arbOpportunity.additionalProfit * 100).toFixed(2)}% arbitrage bonus`
            };

        } catch (error) {
            console.error('Arbitrage-optimized execution failed:', error);
            // Fall back to original execution
            return window.tradingEngine.executeTrade(tradeData);
        }
    }

    enhanceExistingPanels() {
        console.log('🔧 Enhancing existing panels...');
        
        // Add arbitrage toggle to AI trading panel
        this.addArbitrageToggleToAIPanel();
        
        // Enhance performance metrics with arbitrage data
        this.enhancePerformanceMetrics();
        
        // Add arbitrage indicators to market data panels
        this.addArbitrageIndicators();
        
        console.log('✅ Panel enhancements complete');
    }

    addArbitrageToggleToAIPanel() {
        const aiPanel = document.querySelector('.ai-trading-panel') || 
                       document.querySelector('[data-panel="trading"]');
        
        if (aiPanel) {
            const arbitrageToggle = document.createElement('div');
            arbitrageToggle.className = 'arbitrage-toggle-enhancement';
            arbitrageToggle.innerHTML = `
                <div class="enhancement-header">
                    <h4>🎯 HFT Arbitrage Enhancement</h4>
                    <div class="toggle-switch">
                        <input type="checkbox" id="global-arbitrage-toggle">
                        <label for="global-arbitrage-toggle" class="toggle-label">
                            <span class="toggle-slider"></span>
                        </label>
                    </div>
                </div>
                <div class="arbitrage-status">
                    <span id="arbitrage-opportunities-count">0 opportunities found</span>
                    <span id="arbitrage-profit-today">+$0 today</span>
                </div>
            `;
            
            // Insert at the top of AI panel
            aiPanel.insertBefore(arbitrageToggle, aiPanel.firstChild);
            
            // Setup toggle functionality
            document.getElementById('global-arbitrage-toggle').addEventListener('change', (e) => {
                if (e.target.checked) {
                    this.enableAllArbitrageStrategies();
                } else {
                    this.disableAllArbitrageStrategies();
                }
            });
        }
    }

    enhancePerformanceMetrics() {
        // Find existing performance panels
        const performancePanels = document.querySelectorAll('.performance-metrics, .metrics-panel, [data-panel="performance"]');
        
        performancePanels.forEach(panel => {
            // Add arbitrage metrics
            const arbitrageMetric = document.createElement('div');
            arbitrageMetric.className = 'metric-item arbitrage-metric';
            arbitrageMetric.innerHTML = `
                <div class="metric-label">Arbitrage P&L</div>
                <div class="metric-value" id="arbitrage-pnl-display">$0.00</div>
                <div class="metric-change positive" id="arbitrage-pnl-change">+0.00%</div>
            `;
            
            panel.appendChild(arbitrageMetric);
        });

        // Update arbitrage metrics periodically
        setInterval(() => {
            this.updateArbitrageMetrics();
        }, 2000);
    }

    addArbitrageIndicators() {
        // Add arbitrage opportunity indicators to price displays
        const priceElements = document.querySelectorAll('.price-display, .market-price, [data-type="price"]');
        
        priceElements.forEach(priceElement => {
            const indicator = document.createElement('div');
            indicator.className = 'arbitrage-indicator';
            indicator.innerHTML = `
                <span class="arb-icon">⚡</span>
                <span class="arb-status">Scanning...</span>
            `;
            
            priceElement.style.position = 'relative';
            priceElement.appendChild(indicator);
        });
    }

    updateArbitrageMetrics() {
        // Update arbitrage-related displays throughout the platform
        const opportunities = this.arbitrageEngine ? 
            Array.from(this.arbitrageEngine.activeOpportunities.values()) : [];
        
        // Update opportunity count
        const opportunityCountEl = document.getElementById('arbitrage-opportunities-count');
        if (opportunityCountEl) {
            opportunityCountEl.textContent = `${opportunities.length} opportunities found`;
        }

        // Update daily profit
        const dailyProfit = opportunities
            .filter(opp => opp.timestamp > Date.now() - 24*60*60*1000)
            .reduce((sum, opp) => sum + (opp.realizedProfit || 0), 0);
        
        const dailyProfitEl = document.getElementById('arbitrage-profit-today');
        if (dailyProfitEl) {
            dailyProfitEl.textContent = `+$${dailyProfit.toFixed(2)} today`;
        }

        // Update P&L display
        const totalPnL = opportunities.reduce((sum, opp) => sum + (opp.realizedProfit || 0), 0);
        const pnlDisplay = document.getElementById('arbitrage-pnl-display');
        if (pnlDisplay) {
            pnlDisplay.textContent = `$${totalPnL.toFixed(2)}`;
            pnlDisplay.className = totalPnL >= 0 ? 'metric-value positive' : 'metric-value negative';
        }
    }

    enableAllArbitrageStrategies() {
        if (!this.arbitrageEngine) return;
        
        const strategies = ['cross_exchange', 'statistical', 'news_based', 'index_arbitrage', 'latency_arbitrage'];
        strategies.forEach(strategy => {
            this.arbitrageEngine.enableStrategy(strategy);
        });
        
        this.showNotification('All arbitrage strategies enabled', 'success');
    }

    disableAllArbitrageStrategies() {
        if (!this.arbitrageEngine) return;
        
        const strategies = Array.from(this.arbitrageEngine.strategies.keys());
        strategies.forEach(strategy => {
            this.arbitrageEngine.disableStrategy(strategy);
        });
        
        this.showNotification('All arbitrage strategies disabled', 'info');
    }

    handleTradeExecution(executionData) {
        console.log('📈 Trade execution detected:', executionData);
        
        // Check if this was an arbitrage-related trade
        if (executionData.metadata && executionData.metadata.source === 'arbitrage') {
            this.updateArbitrageStats(executionData);
        }
        
        // Look for new arbitrage opportunities based on this execution
        this.scanForPostTradeArbitrage(executionData);
    }

    handleRiskAlert(alertData) {
        console.log('⚠️ Risk alert received:', alertData);
        
        // If critical risk, disable arbitrage strategies
        if (alertData.severity === 'critical') {
            this.disableAllArbitrageStrategies();
            console.log('🚨 Arbitrage strategies disabled due to critical risk alert');
        }
    }

    showIntegrationNotification() {
        this.showNotification(
            '🚀 HyperVision Enhanced: Advanced HFT Arbitrage system integrated successfully!',
            'success'
        );
    }

    handleIntegrationFailure(error) {
        console.error('Integration failure details:', error);
        
        this.showNotification(
            `⚠️ Arbitrage integration failed: ${error.message}. Platform will continue with basic functionality.`,
            'warning'
        );
        
        // Attempt graceful fallback
        this.setupFallbackMode();
    }

    setupFallbackMode() {
        console.log('🔄 Setting up fallback mode...');
        
        // Create basic arbitrage indicator
        const fallbackIndicator = document.createElement('div');
        fallbackIndicator.innerHTML = `
            <div class="fallback-arbitrage-panel">
                <h4>⚠️ Arbitrage System (Limited Mode)</h4>
                <p>Advanced arbitrage features are temporarily unavailable.</p>
                <button onclick="window.arbitrageIntegration.retryIntegration()">
                    Retry Integration
                </button>
            </div>
        `;
        
        document.body.appendChild(fallbackIndicator);
    }

    async retryIntegration() {
        console.log('🔄 Retrying arbitrage integration...');
        
        // Remove fallback elements
        const fallbackElements = document.querySelectorAll('.fallback-arbitrage-panel');
        fallbackElements.forEach(el => el.remove());
        
        // Retry initialization
        await this.init();
    }

    showNotification(message, type = 'info') {
        // Use existing platform notification system or create our own
        if (window.showNotification) {
            window.showNotification(message, type);
        } else if (this.arbitrageUI && this.arbitrageUI.showNotification) {
            this.arbitrageUI.showNotification(message, type);
        } else {
            // Fallback notification
            console.log(`[${type.toUpperCase()}] ${message}`);
            
            const notification = document.createElement('div');
            notification.className = `integration-notification ${type}`;
            notification.textContent = message;
            notification.style.cssText = `
                position: fixed;
                top: 20px;
                right: 20px;
                background: #2d3748;
                color: white;
                padding: 12px 16px;
                border-radius: 8px;
                border-left: 4px solid ${type === 'success' ? '#00ff88' : type === 'warning' ? '#ffe66d' : '#4ecdc4'};
                z-index: 10000;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            `;
            
            document.body.appendChild(notification);
            
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 5000);
        }
    }

    // Public API for external control
    getArbitrageEngine() {
        return this.arbitrageEngine;
    }

    getArbitrageUI() {
        return this.arbitrageUI;
    }

    getIntegrationStatus() {
        return {
            isIntegrated: this.isIntegrated,
            mode: this.integrationMode,
            strategiesActive: this.arbitrageEngine ? 
                Array.from(this.arbitrageEngine.strategies.keys()).filter(
                    key => this.arbitrageEngine.strategies.get(key).isEnabled()
                ).length : 0,
            connectionStatus: this.arbitrageEngine ? 
                this.arbitrageEngine.exchangeConnections.size : 0
        };
    }

    // Cleanup method
    destroy() {
        console.log('🧹 Cleaning up arbitrage integration...');
        
        if (this.arbitrageUI) {
            this.arbitrageUI.destroy();
        }
        
        // Remove enhanced elements
        const enhancedElements = document.querySelectorAll('.arbitrage-enhanced, .arbitrage-toggle-enhancement');
        enhancedElements.forEach(el => el.remove());
        
        // Clear intervals and listeners
        document.removeEventListener('arbitrage:opportunity_found', this.handleArbitrageOpportunity);
        document.removeEventListener('arbitrage:execution_complete', this.handleArbitrageExecution);
        
        console.log('✅ Cleanup complete');
    }
}

// CSS for integration enhancements
const INTEGRATION_ENHANCEMENT_CSS = `
    .arbitrage-enhanced {
        position: relative;
        overflow: visible;
    }

    .arbitrage-toggle-enhancement {
        background: rgba(0, 255, 136, 0.1);
        border: 1px solid #00ff88;
        border-radius: 8px;
        padding: 15px;
        margin-bottom: 15px;
    }

    .enhancement-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 10px;
    }

    .enhancement-header h4 {
        margin: 0;
        color: #00ff88;
        font-size: 1.1em;
    }

    .arbitrage-status {
        display: flex;
        gap: 20px;
        font-size: 0.9em;
        color: #a0aec0;
    }

    .arbitrage-metric {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.1) 0%, rgba(0, 255, 136, 0.05) 100%);
        border-left: 4px solid #00ff88;
    }

    .arbitrage-indicator {
        position: absolute;
        top: -5px;
        right: -5px;
        background: rgba(0, 255, 136, 0.9);
        color: #000;
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.7em;
        font-weight: 600;
        display: flex;
        align-items: center;
        gap: 2px;
        animation: pulse 2s infinite;
    }

    .arb-icon {
        font-size: 0.8em;
    }

    .fallback-arbitrage-panel {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: #2d3748;
        border: 2px solid #ffe66d;
        border-radius: 8px;
        padding: 15px;
        color: white;
        max-width: 300px;
        z-index: 9999;
    }

    .fallback-arbitrage-panel h4 {
        margin: 0 0 10px 0;
        color: #ffe66d;
    }

    .fallback-arbitrage-panel button {
        background: #ffe66d;
        color: #000;
        border: none;
        padding: 8px 12px;
        border-radius: 4px;
        cursor: pointer;
        margin-top: 10px;
    }

    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }

    .integration-notification {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        animation: slideInRight 0.3s ease;
    }

    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;

// Auto-initialize when DOM is ready
function initializeArbitrageIntegration() {
    if (typeof ArbitrageEngine === 'undefined') {
        console.warn('⚠️ ArbitrageEngine not loaded, deferring initialization...');
        setTimeout(initializeArbitrageIntegration, 1000);
        return;
    }

    if (typeof HFTArbitrageUI === 'undefined') {
        console.warn('⚠️ HFTArbitrageUI not loaded, deferring initialization...');
        setTimeout(initializeArbitrageIntegration, 1000);
        return;
    }

    // Initialize the integration
    window.arbitrageIntegration = new HyperVisionArbitrageIntegration();
    
    console.log('🎯 Arbitrage integration initialized and available globally as window.arbitrageIntegration');
}

// Add CSS to document
if (typeof document !== 'undefined') {
    const styleSheet = document.createElement('style');
    styleSheet.textContent = INTEGRATION_ENHANCEMENT_CSS;
    document.head.appendChild(styleSheet);

    // Initialize when DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initializeArbitrageIntegration);
    } else {
        // DOM already loaded
        setTimeout(initializeArbitrageIntegration, 100);
    }
}

// Export for module usage
if (typeof window !== 'undefined') {
    window.HyperVisionArbitrageIntegration = HyperVisionArbitrageIntegration;
}

console.log('🔗 HyperVision Arbitrage Integration script loaded');