/**
 * SERVER-SIDE COMPREHENSIVE FIXES
 * ===============================
 * Addresses backend statistical, mathematical, and engineering issues
 */

const fs = require('fs');
const path = require('path');

class ServerFixes {
    constructor() {
        this.statisticalEngine = new ServerStatisticalEngine();
        this.validationEngine = new ServerValidationEngine();
        this.performanceMonitor = new ServerPerformanceMonitor();
        
        this.initializeFixes();
    }
    
    initializeFixes() {
        console.log('🔧 SERVER: Initializing comprehensive fixes...');
        
        // Replace Math.random in server contexts
        this.replaceServerRandomGeneration();
        
        // Add API validation
        this.addAPIValidation();
        
        // Add performance monitoring
        this.addServerMonitoring();
        
        console.log('✅ SERVER: All fixes applied');
    }
    
    replaceServerRandomGeneration() {
        // Override Math.random for server-side calculations
        const originalRandom = Math.random;
        
        Math.random = () => {
            // Use crypto.randomBytes for better randomness in server
            if (typeof require !== 'undefined') {
                try {
                    const crypto = require('crypto');
                    const buffer = crypto.randomBytes(4);
                    return buffer.readUInt32BE(0) / 0xFFFFFFFF;
                } catch (e) {
                    // Fallback to original
                    return originalRandom();
                }
            }
            return originalRandom();
        };
    }
    
    addAPIValidation() {
        // Validation middleware
        this.validateMarketData = (data) => {
            return this.validationEngine.validateMarketData(data);
        };
        
        this.validatePortfolioData = (data) => {
            return this.validationEngine.validatePortfolioData(data);
        };
        
        this.validateSignalData = (data) => {
            return this.validationEngine.validateSignalData(data);
        };
    }
    
    addServerMonitoring() {
        this.performanceMonitor.start();
    }
    
    // Enhanced market data generation
    getRealisticMarketData(symbol) {
        return this.statisticalEngine.generateMarketData(symbol);
    }
    
    getRealisticPortfolioMetrics() {
        return this.statisticalEngine.generatePortfolioMetrics();
    }
    
    getRealisticSignals(symbol) {
        return this.statisticalEngine.generateSignals(symbol);
    }
}

class ServerStatisticalEngine {
    constructor() {
        this.priceModels = new Map();
        this.initializeModels();
    }
    
    initializeModels() {
        const assets = ['BTC-USD', 'ETH-USD', 'SPY', 'QQQ', 'TLT', 'GLD'];
        
        assets.forEach(asset => {
            this.priceModels.set(asset, {
                currentPrice: this.getBasePrice(asset),
                volatility: this.getVolatility(asset),
                trend: 0,
                lastUpdate: Date.now(),
                history: []
            });
        });
    }
    
    getBasePrice(asset) {
        const basePrices = {
            'BTC-USD': 45000,
            'ETH-USD': 2800,
            'SPY': 450,
            'QQQ': 380,
            'TLT': 95,
            'GLD': 180
        };
        return basePrices[asset] || 100;
    }
    
    getVolatility(asset) {
        const volatilities = {
            'BTC-USD': 0.04,
            'ETH-USD': 0.045,
            'SPY': 0.015,
            'QQQ': 0.02,
            'TLT': 0.012,
            'GLD': 0.018
        };
        return volatilities[asset] || 0.02;
    }
    
    generateMarketData(symbol) {
        const model = this.priceModels.get(symbol);
        if (!model) {
            return this.getDefaultMarketData(symbol);
        }
        
        // Generate realistic price using Geometric Brownian Motion
        const dt = (Date.now() - model.lastUpdate) / (1000 * 60 * 60 * 24 * 365);
        const dW = this.boxMullerRandom();
        
        const priceChange = model.currentPrice * (
            model.trend * dt + 
            model.volatility * Math.sqrt(dt) * dW
        );
        
        model.currentPrice = Math.max(model.currentPrice + priceChange, model.currentPrice * 0.1);
        model.lastUpdate = Date.now();
        
        return {
            symbol,
            price: model.currentPrice,
            change24h: priceChange,
            volume24h: this.generateRealisticVolume(symbol, model.currentPrice),
            high24h: model.currentPrice * (1 + Math.abs(dW) * model.volatility),
            low24h: model.currentPrice * (1 - Math.abs(dW) * model.volatility),
            timestamp: Date.now()
        };
    }
    
    boxMullerRandom() {
        const u1 = Math.random();
        const u2 = Math.random();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    
    generateRealisticVolume(symbol, price) {
        const baseVolumes = {
            'BTC-USD': 50000000000,
            'ETH-USD': 20000000000,
            'SPY': 80000000000,
            'QQQ': 60000000000,
            'TLT': 10000000000,
            'GLD': 15000000000
        };
        
        const baseVolume = baseVolumes[symbol] || 10000000000;
        const volatilityFactor = 1 + Math.abs(this.boxMullerRandom()) * 0.5;
        
        return baseVolume * volatilityFactor;
    }
    
    generatePortfolioMetrics() {
        const baseMetrics = {
            totalValue: 2847563,
            sharpeRatio: 2.34,
            sortinoRatio: 3.87,
            calmarRatio: 3.42,
            informationRatio: 1.42,
            maxDrawdown: -6.8,
            winRate: 73.8,
            profitFactor: 2.3
        };
        
        // Add realistic variations using statistical distributions
        return Object.fromEntries(
            Object.entries(baseMetrics).map(([key, value]) => {
                const variation = this.boxMullerRandom() * 0.05; // 5% std dev
                const newValue = value * (1 + variation);
                return [key, newValue];
            })
        );
    }
    
    generateSignals(symbol) {
        const confidence = 0.7 + Math.random() * 0.25;
        
        return {
            symbol,
            action: Math.random() > 0.5 ? 'BUY' : 'SELL',
            confidence,
            indicators: {
                rsi: 30 + this.boxMullerRandom() * 20 + 25,
                macd: this.boxMullerRandom() * 5,
                bollingerPosition: this.boxMullerRandom() * 0.5,
                sentimentScore: Math.max(0, Math.min(1, 0.5 + this.boxMullerRandom() * 0.3)),
                orderImbalance: this.boxMullerRandom() * 0.3
            },
            mlPrediction: {
                nextHourDirection: Math.random() > 0.5 ? 'UP' : 'DOWN',
                confidence: 0.8 + Math.random() * 0.15,
                expectedMove: this.boxMullerRandom() * 2.5
            },
            timestamp: Date.now()
        };
    }
    
    getDefaultMarketData(symbol) {
        return {
            symbol,
            price: this.getBasePrice(symbol),
            change24h: 0,
            volume24h: 1000000000,
            high24h: this.getBasePrice(symbol) * 1.02,
            low24h: this.getBasePrice(symbol) * 0.98,
            timestamp: Date.now()
        };
    }
}

class ServerValidationEngine {
    constructor() {
        this.validationRules = this.initializeValidationRules();
    }
    
    initializeValidationRules() {
        return {
            price: {
                min: 0.01,
                max: 1000000,
                required: true
            },
            volume: {
                min: 1000,
                max: 1000000000000,
                required: true
            },
            sharpeRatio: {
                min: -10,
                max: 10,
                required: true
            },
            correlation: {
                min: -1,
                max: 1,
                required: true
            }
        };
    }
    
    validateMarketData(data) {
        const issues = [];
        
        // Validate price
        if (!data.price || data.price < this.validationRules.price.min) {
            issues.push('Invalid price');
        }
        
        // Validate volume
        if (!data.volume24h || data.volume24h < this.validationRules.volume.min) {
            issues.push('Invalid volume');
        }
        
        // Validate high >= low
        if (data.high24h < data.low24h) {
            issues.push('High price less than low price');
        }
        
        // Validate price within high/low range
        if (data.price > data.high24h || data.price < data.low24h) {
            issues.push('Price outside high/low range');
        }
        
        return {
            valid: issues.length === 0,
            issues,
            correctedData: issues.length > 0 ? this.correctMarketData(data) : data
        };
    }
    
    validatePortfolioData(data) {
        const issues = [];
        
        // Validate Sharpe ratio
        if (data.sharpeRatio && (data.sharpeRatio < this.validationRules.sharpeRatio.min || 
                                 data.sharpeRatio > this.validationRules.sharpeRatio.max)) {
            issues.push('Invalid Sharpe ratio');
        }
        
        // Validate win rate
        if (data.winRate && (data.winRate < 0 || data.winRate > 100)) {
            issues.push('Invalid win rate');
        }
        
        // Validate max drawdown
        if (data.maxDrawdown && data.maxDrawdown > 0) {
            issues.push('Max drawdown should be negative');
        }
        
        return {
            valid: issues.length === 0,
            issues,
            correctedData: issues.length > 0 ? this.correctPortfolioData(data) : data
        };
    }
    
    validateSignalData(data) {
        const issues = [];
        
        // Validate confidence
        if (data.confidence && (data.confidence < 0 || data.confidence > 1)) {
            issues.push('Invalid confidence range');
        }
        
        // Validate RSI
        if (data.indicators && data.indicators.rsi && 
            (data.indicators.rsi < 0 || data.indicators.rsi > 100)) {
            issues.push('Invalid RSI range');
        }
        
        return {
            valid: issues.length === 0,
            issues,
            correctedData: issues.length > 0 ? this.correctSignalData(data) : data
        };
    }
    
    correctMarketData(data) {
        const corrected = { ...data };
        
        // Fix price
        if (!corrected.price || corrected.price < this.validationRules.price.min) {
            corrected.price = 100;
        }
        
        // Fix volume
        if (!corrected.volume24h || corrected.volume24h < this.validationRules.volume.min) {
            corrected.volume24h = 1000000;
        }
        
        // Fix high/low consistency
        if (corrected.high24h < corrected.low24h) {
            const avg = (corrected.high24h + corrected.low24h) / 2;
            corrected.high24h = avg * 1.01;
            corrected.low24h = avg * 0.99;
        }
        
        // Ensure price is within range
        corrected.price = Math.max(corrected.low24h, Math.min(corrected.high24h, corrected.price));
        
        return corrected;
    }
    
    correctPortfolioData(data) {
        const corrected = { ...data };
        
        // Fix Sharpe ratio
        if (corrected.sharpeRatio) {
            corrected.sharpeRatio = Math.max(-10, Math.min(10, corrected.sharpeRatio));
        }
        
        // Fix win rate
        if (corrected.winRate) {
            corrected.winRate = Math.max(0, Math.min(100, corrected.winRate));
        }
        
        // Fix max drawdown
        if (corrected.maxDrawdown && corrected.maxDrawdown > 0) {
            corrected.maxDrawdown = -Math.abs(corrected.maxDrawdown);
        }
        
        return corrected;
    }
    
    correctSignalData(data) {
        const corrected = { ...data };
        
        // Fix confidence
        if (corrected.confidence) {
            corrected.confidence = Math.max(0, Math.min(1, corrected.confidence));
        }
        
        // Fix RSI
        if (corrected.indicators && corrected.indicators.rsi) {
            corrected.indicators.rsi = Math.max(0, Math.min(100, corrected.indicators.rsi));
        }
        
        return corrected;
    }
}

class ServerPerformanceMonitor {
    constructor() {
        this.metrics = {
            requests: 0,
            errors: 0,
            responseTime: [],
            memoryUsage: [],
            cpuUsage: []
        };
        
        this.startTime = Date.now();
    }
    
    start() {
        // Monitor memory and CPU every minute
        setInterval(() => {
            this.collectSystemMetrics();
        }, 60000);
        
        console.log('📊 Server performance monitoring started');
    }
    
    collectSystemMetrics() {
        const usage = process.memoryUsage();
        
        this.metrics.memoryUsage.push({
            rss: usage.rss,
            heapUsed: usage.heapUsed,
            heapTotal: usage.heapTotal,
            external: usage.external,
            timestamp: Date.now()
        });
        
        // CPU usage (simplified)
        const cpuUsage = process.cpuUsage();
        this.metrics.cpuUsage.push({
            user: cpuUsage.user,
            system: cpuUsage.system,
            timestamp: Date.now()
        });
        
        // Keep only last 100 measurements
        if (this.metrics.memoryUsage.length > 100) {
            this.metrics.memoryUsage = this.metrics.memoryUsage.slice(-100);
        }
        
        if (this.metrics.cpuUsage.length > 100) {
            this.metrics.cpuUsage = this.metrics.cpuUsage.slice(-100);
        }
    }
    
    recordRequest(duration) {
        this.metrics.requests++;
        this.metrics.responseTime.push({
            duration,
            timestamp: Date.now()
        });
        
        // Keep only last 1000 response times
        if (this.metrics.responseTime.length > 1000) {
            this.metrics.responseTime = this.metrics.responseTime.slice(-1000);
        }
    }
    
    recordError() {
        this.metrics.errors++;
    }
    
    getMetrics() {
        const now = Date.now();
        const uptime = now - this.startTime;
        
        // Calculate averages
        const recentResponseTimes = this.metrics.responseTime.filter(
            rt => rt.timestamp > now - 3600000 // Last hour
        ).map(rt => rt.duration);
        
        const avgResponseTime = recentResponseTimes.length > 0 
            ? recentResponseTimes.reduce((a, b) => a + b, 0) / recentResponseTimes.length 
            : 0;
        
        const recentMemory = this.metrics.memoryUsage.slice(-1)[0];
        
        return {
            uptime,
            requests: this.metrics.requests,
            errors: this.metrics.errors,
            errorRate: this.metrics.requests > 0 ? this.metrics.errors / this.metrics.requests : 0,
            avgResponseTime,
            currentMemory: recentMemory,
            timestamp: now
        };
    }
    
    getTotalRequests() {
        return this.metrics.requests;
    }
    
    getErrorRate() {
        return this.metrics.requests > 0 ? this.metrics.errors / this.metrics.requests : 0;
    }
    
    getAverageResponseTime() {
        const now = Date.now();
        const recentResponseTimes = this.metrics.responseTime.filter(
            rt => rt.timestamp > now - 3600000 // Last hour
        ).map(rt => rt.duration);
        
        return recentResponseTimes.length > 0 
            ? recentResponseTimes.reduce((a, b) => a + b, 0) / recentResponseTimes.length 
            : 0;
    }
}

module.exports = {
    ServerFixes,
    ServerStatisticalEngine,
    ServerValidationEngine,
    ServerPerformanceMonitor
};