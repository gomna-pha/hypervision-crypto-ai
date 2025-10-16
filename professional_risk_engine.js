/**
 * Professional Risk Engine for HFT Operations
 * Based on NautilusTrader architecture - Critical safety component
 * 
 * Features:
 * - Pre-trade risk validation
 * - Real-time position monitoring
 * - Emergency stop mechanisms
 * - Risk limit enforcement
 * - Regulatory compliance checks
 */

class ProfessionalRiskEngine {
    constructor() {
        this.isActive = false;
        this.emergencyStopActive = false;
        
        // Risk limits configuration
        this.riskLimits = {
            // Position limits
            maxPositionSize: 1000000,      // $1M max position per symbol
            maxPortfolioValue: 10000000,    // $10M max total portfolio
            maxConcentration: 0.25,         // 25% max in single asset
            
            // P&L limits
            maxDailyLoss: 500000,          // $500K max daily loss
            maxDrawdown: 0.10,             // 10% max drawdown
            stopLossThreshold: 0.05,        // 5% stop loss trigger
            
            // Trading limits
            maxOrderSize: 100000,          // $100K max single order
            maxOrdersPerSecond: 100,       // 100 orders/second max
            maxDailyVolume: 50000000,      // $50M max daily volume
            
            // Volatility limits
            maxVolatility: 0.50,           // 50% max volatility
            maxCorrelation: 0.80,          // 80% max correlation
            
            // Leverage limits
            maxLeverage: 4.0,              // 4:1 max leverage
            marginRequirement: 0.25        // 25% margin requirement
        };
        
        // Current risk state
        this.riskState = {
            currentPositions: new Map(),
            dailyPnL: 0,
            unrealizedPnL: 0,
            currentDrawdown: 0,
            maxDrawdownToday: 0,
            ordersToday: 0,
            volumeToday: 0,
            lastResetTime: new Date().setHours(0,0,0,0)
        };
        
        // Risk monitoring
        this.riskAlerts = [];
        this.riskViolations = new Map();
        
        // Performance tracking
        this.riskCheckLatency = [];
        this.riskProcessingTimes = [];
        
        this.initialize();
    }
    
    initialize() {
        console.log('🛡️ Initializing Professional Risk Engine...');
        
        // Setup risk monitoring
        this.setupRiskMonitoring();
        
        // Setup emergency procedures
        this.setupEmergencyProcedures();
        
        // Start daily reset timer
        this.setupDailyReset();
        
        // Initialize risk validation
        this.initializeRiskValidation();
        
        this.isActive = true;
        
        console.log('✅ Professional Risk Engine initialized');
        console.log(`🎯 Max Daily Loss: $${this.riskLimits.maxDailyLoss.toLocaleString()}`);
        console.log(`⚡ Max Position: $${this.riskLimits.maxPositionSize.toLocaleString()}`);
    }
    
    // === PRE-TRADE RISK VALIDATION ===
    
    async validateOrder(order) {
        if (!this.isActive) {
            return { valid: false, reason: 'Risk engine not active' };
        }
        
        if (this.emergencyStopActive) {
            return { valid: false, reason: 'Emergency stop active - all trading halted' };
        }
        
        const startTime = performance.now();
        
        try {
            // Comprehensive risk checks
            const riskChecks = {
                orderSize: this.validateOrderSize(order),
                positionLimits: this.validatePositionLimits(order),
                portfolioLimits: this.validatePortfolioLimits(order),
                concentrationRisk: this.validateConcentrationRisk(order),
                dailyLimits: this.validateDailyLimits(order),
                volatilityLimits: this.validateVolatilityLimits(order),
                leverageLimits: this.validateLeverageLimits(order),
                correlationRisk: this.validateCorrelationRisk(order),
                liquidityRisk: this.validateLiquidityRisk(order),
                regulatoryChecks: this.validateRegulatoryRequirements(order)
            };
            
            // Check if all validations pass
            const failedChecks = Object.entries(riskChecks)
                .filter(([check, result]) => !result.valid)
                .map(([check, result]) => ({ check, reason: result.reason }));
            
            const isValid = failedChecks.length === 0;
            
            if (!isValid) {
                this.recordRiskViolation(order, failedChecks);
            }
            
            const latency = performance.now() - startTime;
            this.riskCheckLatency.push(latency);
            
            return {
                valid: isValid,
                failedChecks,
                riskScore: this.calculateRiskScore(order),
                latency
            };
            
        } catch (error) {
            console.error('❌ Risk validation error:', error);
            return { valid: false, reason: 'Risk validation system error' };
        }
    }
    
    // Order size validation
    validateOrderSize(order) {
        const orderValue = order.quantity * order.price;
        
        if (orderValue > this.riskLimits.maxOrderSize) {
            return {
                valid: false,
                reason: `Order size $${orderValue.toLocaleString()} exceeds limit $${this.riskLimits.maxOrderSize.toLocaleString()}`
            };
        }
        
        return { valid: true };
    }
    
    // Position limits validation
    validatePositionLimits(order) {
        const currentPosition = this.riskState.currentPositions.get(order.symbol) || { quantity: 0, value: 0 };
        const newQuantity = currentPosition.quantity + (order.side === 'BUY' ? order.quantity : -order.quantity);
        const newPositionValue = Math.abs(newQuantity * order.price);
        
        if (newPositionValue > this.riskLimits.maxPositionSize) {
            return {
                valid: false,
                reason: `Position value $${newPositionValue.toLocaleString()} would exceed limit $${this.riskLimits.maxPositionSize.toLocaleString()}`
            };
        }
        
        return { valid: true };
    }
    
    // Portfolio limits validation
    validatePortfolioLimits(order) {
        const currentPortfolioValue = this.calculatePortfolioValue();
        const orderValue = order.quantity * order.price;
        const newPortfolioValue = currentPortfolioValue + orderValue;
        
        if (newPortfolioValue > this.riskLimits.maxPortfolioValue) {
            return {
                valid: false,
                reason: `Portfolio value would exceed limit $${this.riskLimits.maxPortfolioValue.toLocaleString()}`
            };
        }
        
        return { valid: true };
    }
    
    // Concentration risk validation
    validateConcentrationRisk(order) {
        const portfolioValue = this.calculatePortfolioValue();
        const symbolValue = this.getSymbolValue(order.symbol);
        const orderValue = order.quantity * order.price;
        const newSymbolValue = symbolValue + orderValue;
        const concentration = newSymbolValue / portfolioValue;
        
        if (concentration > this.riskLimits.maxConcentration) {
            return {
                valid: false,
                reason: `Concentration ${(concentration * 100).toFixed(1)}% exceeds limit ${(this.riskLimits.maxConcentration * 100).toFixed(1)}%`
            };
        }
        
        return { valid: true };
    }
    
    // Daily limits validation
    validateDailyLimits(order) {
        const orderValue = order.quantity * order.price;
        
        // Check daily loss limit
        if (this.riskState.dailyPnL < -this.riskLimits.maxDailyLoss) {
            return {
                valid: false,
                reason: `Daily loss limit exceeded: $${Math.abs(this.riskState.dailyPnL).toLocaleString()}`
            };
        }
        
        // Check daily volume limit
        if (this.riskState.volumeToday + orderValue > this.riskLimits.maxDailyVolume) {
            return {
                valid: false,
                reason: `Daily volume limit would be exceeded`
            };
        }
        
        // Check orders per second limit
        const recentOrders = this.getRecentOrderCount(1000); // Last second
        if (recentOrders >= this.riskLimits.maxOrdersPerSecond) {
            return {
                valid: false,
                reason: `Order rate limit exceeded: ${recentOrders}/second`
            };
        }
        
        return { valid: true };
    }
    
    // Volatility limits validation
    validateVolatilityLimits(order) {
        const volatility = this.getSymbolVolatility(order.symbol);
        
        if (volatility > this.riskLimits.maxVolatility) {
            return {
                valid: false,
                reason: `Symbol volatility ${(volatility * 100).toFixed(1)}% exceeds limit ${(this.riskLimits.maxVolatility * 100).toFixed(1)}%`
            };
        }
        
        return { valid: true };
    }
    
    // Leverage limits validation
    validateLeverageLimits(order) {
        const equity = this.getAccountEquity();
        const currentExposure = this.getTotalExposure();
        const orderValue = order.quantity * order.price;
        const newExposure = currentExposure + orderValue;
        const leverage = newExposure / equity;
        
        if (leverage > this.riskLimits.maxLeverage) {
            return {
                valid: false,
                reason: `Leverage ${leverage.toFixed(2)}:1 exceeds limit ${this.riskLimits.maxLeverage}:1`
            };
        }
        
        return { valid: true };
    }
    
    // Correlation risk validation
    validateCorrelationRisk(order) {
        const correlations = this.getPortfolioCorrelations(order.symbol);
        const highCorrelations = correlations.filter(corr => corr.value > this.riskLimits.maxCorrelation);
        
        if (highCorrelations.length > 0) {
            return {
                valid: false,
                reason: `High correlation detected with existing positions`
            };
        }
        
        return { valid: true };
    }
    
    // Liquidity risk validation
    validateLiquidityRisk(order) {
        const liquidity = this.getSymbolLiquidity(order.symbol);
        const orderSize = order.quantity * order.price;
        
        // Check if order is too large relative to available liquidity
        if (orderSize > liquidity.availableVolume * 0.1) { // Max 10% of available volume
            return {
                valid: false,
                reason: `Order size too large relative to market liquidity`
            };
        }
        
        return { valid: true };
    }
    
    // Regulatory compliance validation
    validateRegulatoryRequirements(order) {
        // Pattern Day Trading rule (US)
        if (this.isDayTradingViolation(order)) {
            return {
                valid: false,
                reason: 'Pattern Day Trading rule violation'
            };
        }
        
        // Position reporting thresholds
        if (this.isPositionReportingRequired(order)) {
            this.flagForReporting(order);
        }
        
        return { valid: true };
    }
    
    // === REAL-TIME RISK MONITORING ===
    
    setupRiskMonitoring() {
        // Real-time position monitoring (every 100ms)
        setInterval(() => {
            this.monitorPositions();
        }, 100);
        
        // P&L monitoring (every 500ms)
        setInterval(() => {
            this.monitorPnL();
        }, 500);
        
        // Portfolio risk assessment (every second)
        setInterval(() => {
            this.assessPortfolioRisk();
        }, 1000);
        
        console.log('✅ Real-time risk monitoring active');
    }
    
    monitorPositions() {
        this.riskState.currentPositions.forEach((position, symbol) => {
            const currentPrice = this.getCurrentPrice(symbol);
            const positionValue = Math.abs(position.quantity) * currentPrice;
            const unrealizedPnL = this.calculateUnrealizedPnL(position, currentPrice);
            
            // Check position limits
            if (positionValue > this.riskLimits.maxPositionSize) {
                this.triggerRiskAlert('POSITION_LIMIT_BREACH', {
                    symbol,
                    value: positionValue,
                    limit: this.riskLimits.maxPositionSize
                });
            }
            
            // Check stop loss
            const loss = unrealizedPnL / (position.quantity * position.averagePrice);
            if (loss < -this.riskLimits.stopLossThreshold) {
                this.triggerStopLoss(symbol, position);
            }
        });
    }
    
    monitorPnL() {
        // Calculate current P&L
        const totalPnL = this.calculateTotalPnL();
        this.riskState.dailyPnL = totalPnL.realized;
        this.riskState.unrealizedPnL = totalPnL.unrealized;
        
        // Check daily loss limit
        if (this.riskState.dailyPnL < -this.riskLimits.maxDailyLoss) {
            this.triggerEmergencyStop('DAILY_LOSS_LIMIT_BREACH', {
                currentLoss: this.riskState.dailyPnL,
                limit: this.riskLimits.maxDailyLoss
            });
        }
        
        // Calculate drawdown
        this.updateDrawdown();
    }
    
    assessPortfolioRisk() {
        const riskMetrics = this.calculateRiskMetrics();
        
        // Check overall portfolio risk
        if (riskMetrics.portfolioVaR > this.riskLimits.maxPortfolioValue * 0.02) { // 2% VaR limit
            this.triggerRiskAlert('HIGH_PORTFOLIO_RISK', riskMetrics);
        }
        
        // Check correlation concentration
        if (riskMetrics.maxCorrelation > this.riskLimits.maxCorrelation) {
            this.triggerRiskAlert('HIGH_CORRELATION_RISK', riskMetrics);
        }
    }
    
    // === EMERGENCY PROCEDURES ===
    
    setupEmergencyProcedures() {
        // Setup emergency stop mechanisms
        this.emergencyProcedures = {
            stopLoss: this.triggerStopLoss.bind(this),
            emergencyStop: this.triggerEmergencyStop.bind(this),
            positionClose: this.closeAllPositions.bind(this),
            systemHalt: this.haltAllTrading.bind(this)
        };
        
        // Setup keyboard shortcuts for manual emergency stops
        this.setupEmergencyShortcuts();
    }
    
    triggerEmergencyStop(reason, data = {}) {
        console.error('🚨 EMERGENCY STOP TRIGGERED:', reason);
        
        this.emergencyStopActive = true;
        
        // Record emergency stop
        this.recordEmergencyStop(reason, data);
        
        // Halt all new trading immediately
        this.haltAllTrading();
        
        // Cancel all pending orders
        this.cancelAllOrders();
        
        // Close risky positions (optional, based on reason)
        if (reason.includes('LOSS') || reason.includes('RISK')) {
            this.closeRiskyPositions();
        }
        
        // Notify risk management team
        this.notifyEmergencyStop(reason, data);
        
        // Emit emergency stop event
        this.emitEvent('EMERGENCY_STOP', { reason, data, timestamp: Date.now() });
    }
    
    triggerStopLoss(symbol, position) {
        console.warn(`⚠️ Stop loss triggered for ${symbol}`);
        
        const stopLossOrder = {
            symbol,
            side: position.quantity > 0 ? 'SELL' : 'BUY',
            quantity: Math.abs(position.quantity),
            type: 'MARKET',
            reason: 'STOP_LOSS'
        };
        
        // Execute stop loss order immediately
        this.executeEmergencyOrder(stopLossOrder);
        
        // Record stop loss event
        this.recordStopLoss(symbol, position);
    }
    
    haltAllTrading() {
        console.warn('🛑 All trading halted by risk engine');
        
        // Emit trading halt event
        this.emitEvent('TRADING_HALTED', { timestamp: Date.now() });
        
        // Set halt flag
        this.tradingHalted = true;
    }
    
    // === RISK CALCULATIONS ===
    
    calculateRiskScore(order) {
        const factors = {
            orderSize: this.getOrderSizeRisk(order),
            volatility: this.getVolatilityRisk(order),
            concentration: this.getConcentrationRisk(order),
            correlation: this.getCorrelationRisk(order),
            liquidity: this.getLiquidityRisk(order)
        };
        
        // Weighted risk score (0-100)
        const weights = { orderSize: 0.2, volatility: 0.25, concentration: 0.25, correlation: 0.15, liquidity: 0.15 };
        
        return Object.entries(factors).reduce((score, [factor, risk]) => {
            return score + (risk * weights[factor]);
        }, 0);
    }
    
    calculateRiskMetrics() {
        const positions = Array.from(this.riskState.currentPositions.values());
        
        return {
            portfolioValue: this.calculatePortfolioValue(),
            portfolioVaR: this.calculateVaR(positions),
            maxDrawdown: this.riskState.maxDrawdownToday,
            currentDrawdown: this.riskState.currentDrawdown,
            leverage: this.calculateCurrentLeverage(),
            concentration: this.calculateMaxConcentration(),
            maxCorrelation: this.calculateMaxCorrelation(),
            sharpeRatio: this.calculateSharpeRatio()
        };
    }
    
    // === UTILITY METHODS ===
    
    getCurrentPrice(symbol) {
        // Integration with market data system
        return window.hftOptimizedMarketplace?.getLatestPrice(symbol) || 100;
    }
    
    calculatePortfolioValue() {
        let totalValue = 0;
        
        this.riskState.currentPositions.forEach((position, symbol) => {
            const currentPrice = this.getCurrentPrice(symbol);
            totalValue += Math.abs(position.quantity) * currentPrice;
        });
        
        return totalValue;
    }
    
    updatePosition(symbol, quantity, price, side) {
        const position = this.riskState.currentPositions.get(symbol) || {
            quantity: 0,
            averagePrice: 0,
            unrealizedPnL: 0,
            entryTime: Date.now()
        };
        
        const tradeQuantity = side === 'BUY' ? quantity : -quantity;
        const newQuantity = position.quantity + tradeQuantity;
        
        // Update average price using weighted average
        if (newQuantity !== 0) {
            const totalValue = (position.quantity * position.averagePrice) + (tradeQuantity * price);
            position.averagePrice = totalValue / newQuantity;
        }
        
        position.quantity = newQuantity;
        
        // Remove position if quantity is zero
        if (newQuantity === 0) {
            this.riskState.currentPositions.delete(symbol);
        } else {
            this.riskState.currentPositions.set(symbol, position);
        }
        
        // Update daily volume
        this.riskState.volumeToday += quantity * price;
        this.riskState.ordersToday++;
    }
    
    // Daily reset functionality
    setupDailyReset() {
        const now = new Date();
        const tomorrow = new Date(now);
        tomorrow.setDate(tomorrow.getDate() + 1);
        tomorrow.setHours(0, 0, 0, 0);
        
        const msUntilMidnight = tomorrow.getTime() - now.getTime();
        
        setTimeout(() => {
            this.resetDailyLimits();
            
            // Setup daily reset interval
            setInterval(() => {
                this.resetDailyLimits();
            }, 24 * 60 * 60 * 1000); // Every 24 hours
            
        }, msUntilMidnight);
    }
    
    resetDailyLimits() {
        this.riskState.dailyPnL = 0;
        this.riskState.volumeToday = 0;
        this.riskState.ordersToday = 0;
        this.riskState.maxDrawdownToday = 0;
        this.riskState.lastResetTime = Date.now();
        
        console.log('🔄 Daily risk limits reset');
    }
    
    // Event emission for integration
    emitEvent(eventType, data) {
        const event = new CustomEvent('riskEngineEvent', {
            detail: { eventType, data, timestamp: Date.now() }
        });
        document.dispatchEvent(event);
    }
    
    // Performance monitoring
    getRiskEngineMetrics() {
        return {
            active: this.isActive,
            emergencyStop: this.emergencyStopActive,
            riskCheckLatency: this.calculateStats(this.riskCheckLatency),
            dailyStats: {
                ordersProcessed: this.riskState.ordersToday,
                volumeTraded: this.riskState.volumeToday,
                currentPnL: this.riskState.dailyPnL
            },
            riskLimits: this.riskLimits,
            alertCount: this.riskAlerts.length,
            violationCount: this.riskViolations.size
        };
    }
    
    calculateStats(array) {
        if (array.length === 0) return { avg: 0, min: 0, max: 0, count: 0 };
        
        const sorted = [...array].sort((a, b) => a - b);
        return {
            avg: array.reduce((sum, val) => sum + val, 0) / array.length,
            min: sorted[0],
            max: sorted[sorted.length - 1],
            p95: sorted[Math.floor(sorted.length * 0.95)],
            count: array.length
        };
    }
    
    // Risk engine status
    getStatus() {
        return {
            active: this.isActive,
            emergencyStop: this.emergencyStopActive,
            tradingHalted: this.tradingHalted || false,
            riskMetrics: this.calculateRiskMetrics(),
            currentPositions: this.riskState.currentPositions.size,
            dailyPnL: this.riskState.dailyPnL,
            alertCount: this.riskAlerts.length,
            uptime: Date.now() - this.riskState.lastResetTime
        };
    }
    
    // Manual controls for risk management team
    manualEmergencyStop(reason = 'Manual override') {
        this.triggerEmergencyStop('MANUAL_EMERGENCY_STOP', { reason });
    }
    
    resumeTrading(authorization) {
        if (!authorization || !this.validateAuthorization(authorization)) {
            throw new Error('Invalid authorization for trading resume');
        }
        
        this.emergencyStopActive = false;
        this.tradingHalted = false;
        
        console.log('✅ Trading resumed by risk management');
        this.emitEvent('TRADING_RESUMED', { timestamp: Date.now() });
    }
    
    validateAuthorization(auth) {
        // In production, this would validate proper authorization
        return auth === 'RISK_MANAGER_OVERRIDE';
    }
}

// Export for use in trading system
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ProfessionalRiskEngine;
}

console.log('🛡️ Professional Risk Engine module loaded');