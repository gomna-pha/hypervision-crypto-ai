/**
 * GOMNA ENHANCED ARBITRAGE & HFT ENGINE
 * Advanced High Frequency Trading platform with customizable arbitrage strategies
 * Implements professor's recommendations for low-latency execution and statistical arbitrage
 */

class ArbitrageEngine {
    constructor(tradingEngine) {
        this.tradingEngine = tradingEngine;
        this.strategies = new Map();
        this.activeOpportunities = new Map();
        this.executionLatencyTracker = new LatencyTracker();
        this.sentimentAnalyzer = new FinBERTSentimentAnalyzer();
        this.volumeAnalyzer = new TradingVolumeAnalyzer();
        this.statisticalModels = new StatisticalArbitrageModels();
        
        // HFT Configuration
        this.hftConfig = {
            maxLatencyMs: 5, // Ultra-low latency requirement
            minProfitBps: 2, // 2 basis points minimum profit
            maxPositionSize: 1000000, // $1M max position
            riskAdjustment: 0.95, // Risk adjustment factor
            enableNewsBased: true,
            enableStatistical: true,
            enableIndexArb: true,
            enableCrossExchange: true
        };

        this.exchangeConnections = new Map();
        this.priceFeeds = new Map();
        this.orderBooks = new Map();
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing Enhanced Arbitrage & HFT Engine...');
        
        try {
            // Initialize core components with performance optimization
            await this.initializeStrategies();
            
            // Initialize other components in parallel for better performance
            await Promise.all([
                this.setupExchangeConnections(),
                this.initializeMarketDataFeeds(),
                this.setupSentimentAnalysis()
            ]);
            
            // Start monitoring loops after initialization
            this.startArbitrageScanning();
            this.startLatencyMonitoring();
            
            console.log('✅ Arbitrage Engine initialized successfully');
            
        } catch (error) {
            console.error('❌ Arbitrage Engine initialization failed:', error);
            // Continue with limited functionality
            this.handleInitializationError(error);
        }
    }

    async initializeStrategies() {
        // 1. Cross-Exchange Arbitrage
        this.registerStrategy('cross_exchange', new CrossExchangeArbitrageStrategy({
            exchanges: ['binance', 'coinbase', 'kraken', 'okx'],
            minSpread: 0.02, // 2% minimum spread
            maxExecutionTime: 100, // 100ms max execution
            riskLimit: 0.1 // 10% of portfolio
        }));

        // 2. Statistical Arbitrage (Pairs Trading)
        this.registerStrategy('statistical', new StatisticalArbitrageStrategy({
            lookbackPeriod: 252, // 1 year of trading days
            zscore_threshold: 2.0,
            cointegrationThreshold: 0.05,
            meanReversionWindow: 20
        }));

        // 3. News-Based Arbitrage
        this.registerStrategy('news_based', new NewsBasedArbitrageStrategy({
            sentimentThreshold: 0.3,
            newsLatencyMs: 50, // React within 50ms of news
            confidenceThreshold: 0.8,
            timeDecayFactor: 0.95
        }));

        // 4. Index Arbitrage (ETF vs Basket)
        this.registerStrategy('index_arbitrage', new IndexArbitrageStrategy({
            etfs: ['SPY', 'QQQ', 'IWM', 'VTI'],
            basketTrackingError: 0.001,
            rebalanceFrequency: 300000 // 5 minutes
        }));

        // 5. Latency Arbitrage (HFT)
        this.registerStrategy('latency_arbitrage', new LatencyArbitrageStrategy({
            maxLatency: 1, // 1ms max latency
            venues: ['ARCA', 'NASDAQ', 'NYSE', 'BATS'],
            orderTypes: ['IOC', 'FOK'], // Immediate or Cancel, Fill or Kill
            tickSizeOptimization: true
        }));

        // 6. Volatility Arbitrage
        this.registerStrategy('volatility_arbitrage', new VolatilityArbitrageStrategy({
            impliedVolThreshold: 0.05,
            realizedVolWindow: 30,
            gammaThreshold: 0.1,
            vegaLimit: 10000
        }));

        console.log(`📊 Registered ${this.strategies.size} arbitrage strategies`);
    }

    registerStrategy(name, strategy) {
        this.strategies.set(name, strategy);
        strategy.setEngine(this);
    }

    async setupExchangeConnections() {
        const exchanges = [
            { name: 'binance', ws: 'wss://stream.binance.com:9443/ws', rest: 'https://api.binance.com' },
            { name: 'coinbase', ws: 'wss://ws-feed.pro.coinbase.com', rest: 'https://api.pro.coinbase.com' },
            { name: 'kraken', ws: 'wss://ws.kraken.com', rest: 'https://api.kraken.com' },
            { name: 'okx', ws: 'wss://ws.okx.com:8443/ws/v5/public', rest: 'https://www.okx.com' }
        ];

        for (const exchange of exchanges) {
            try {
                const connection = new ExchangeConnection(exchange);
                await connection.connect();
                this.exchangeConnections.set(exchange.name, connection);
                console.log(`✅ Connected to ${exchange.name}`);
            } catch (error) {
                console.error(`❌ Failed to connect to ${exchange.name}:`, error);
            }
        }
    }

    async initializeMarketDataFeeds() {
        // High-frequency market data feeds
        const feeds = [
            { provider: 'Level2_OrderBook', symbols: ['BTC/USD', 'ETH/USD', 'SOL/USD'] },
            { provider: 'Trade_Stream', symbols: ['BTC/USD', 'ETH/USD', 'SOL/USD'] },
            { provider: 'Options_Chain', symbols: ['SPY', 'QQQ', 'TSLA'] },
            { provider: 'Futures_Data', symbols: ['ES', 'NQ', 'RTY'] }
        ];

        for (const feed of feeds) {
            const dataFeed = new HighFrequencyDataFeed(feed);
            await dataFeed.initialize();
            this.priceFeeds.set(feed.provider, dataFeed);
        }
    }

    async setupSentimentAnalysis() {
        // Initialize FinBERT and social media sentiment analysis
        await this.sentimentAnalyzer.initialize();
        
        // News sources integration
        const newsSources = [
            { name: 'reuters', endpoint: 'https://api.reuters.com/v1/news' },
            { name: 'bloomberg', endpoint: 'https://api.bloomberg.com/v1/news' },
            { name: 'twitter', endpoint: 'https://api.twitter.com/2/tweets/search' },
            { name: 'reddit', endpoint: 'https://api.reddit.com/r/trading' }
        ];

        for (const source of newsSources) {
            this.sentimentAnalyzer.addNewsSource(source);
        }
    }

    startArbitrageScanning() {
        // Ultra high-frequency scanning - every 100ms
        setInterval(() => {
            this.scanForOpportunities();
        }, 100);

        // News-based scanning - every 10ms for breaking news
        setInterval(() => {
            this.scanNewsBasedOpportunities();
        }, 10);

        // Statistical model updates - every second
        setInterval(() => {
            this.updateStatisticalModels();
        }, 1000);
    }

    async scanForOpportunities() {
        const startTime = performance.now();

        try {
            // Parallel execution of all strategy scans
            const scanPromises = Array.from(this.strategies.entries()).map(async ([name, strategy]) => {
                if (strategy.isEnabled()) {
                    const opportunities = await strategy.scan();
                    return { name, opportunities: opportunities || [] };
                }
                return { name, opportunities: [] };
            });

            const results = await Promise.all(scanPromises);
            
            // Process opportunities by priority
            for (const { name, opportunities } of results) {
                for (const opportunity of opportunities) {
                    await this.evaluateOpportunity(opportunity, name);
                }
            }

        } catch (error) {
            console.error('Arbitrage scanning error:', error);
        }

        const executionTime = performance.now() - startTime;
        this.executionLatencyTracker.recordLatency('scan', executionTime);
        
        // Alert if scanning takes too long
        if (executionTime > this.hftConfig.maxLatencyMs) {
            console.warn(`⚠️ Scanning latency exceeded threshold: ${executionTime.toFixed(2)}ms`);
        }
    }

    async evaluateOpportunity(opportunity, strategyName) {
        const startTime = performance.now();

        try {
            // Quick profit calculation
            const profitBps = this.calculateProfitBasisPoints(opportunity);
            
            if (profitBps < this.hftConfig.minProfitBps) {
                return; // Skip low-profit opportunities
            }

            // Risk assessment
            const riskScore = await this.assessRisk(opportunity);
            if (riskScore > 0.5) {
                return; // Skip high-risk opportunities
            }

            // Check if opportunity still valid (ultra-low latency check)
            const isValid = await this.validateOpportunity(opportunity);
            if (!isValid) {
                return;
            }

            // Execute if all checks pass
            await this.executeArbitrageOpportunity(opportunity, strategyName);

        } catch (error) {
            console.error(`Opportunity evaluation failed for ${strategyName}:`, error);
        }

        const executionTime = performance.now() - startTime;
        this.executionLatencyTracker.recordLatency('evaluate', executionTime);
    }

    async executeArbitrageOpportunity(opportunity, strategyName) {
        const executionId = `arb_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const startTime = performance.now();

        try {
            console.log(`🎯 Executing ${strategyName} arbitrage:`, {
                id: executionId,
                profit_bps: this.calculateProfitBasisPoints(opportunity),
                size: opportunity.size
            });

            // Atomic execution for cross-exchange arbitrage
            if (strategyName === 'cross_exchange') {
                await this.executeCrossExchangeArbitrage(opportunity, executionId);
            }
            // Statistical arbitrage execution
            else if (strategyName === 'statistical') {
                await this.executeStatisticalArbitrage(opportunity, executionId);
            }
            // News-based execution
            else if (strategyName === 'news_based') {
                await this.executeNewsBasedArbitrage(opportunity, executionId);
            }
            // Index arbitrage execution
            else if (strategyName === 'index_arbitrage') {
                await this.executeIndexArbitrage(opportunity, executionId);
            }
            // Latency arbitrage execution
            else if (strategyName === 'latency_arbitrage') {
                await this.executeLatencyArbitrage(opportunity, executionId);
            }

            const executionTime = performance.now() - startTime;
            this.executionLatencyTracker.recordLatency('execute', executionTime);

            console.log(`✅ Arbitrage executed in ${executionTime.toFixed(2)}ms`);

            // Store successful execution
            this.activeOpportunities.set(executionId, {
                ...opportunity,
                strategyName,
                executionTime,
                status: 'executed',
                timestamp: Date.now()
            });

        } catch (error) {
            console.error(`❌ Arbitrage execution failed:`, error);
            
            // Risk management - halt strategy if too many failures
            const strategy = this.strategies.get(strategyName);
            if (strategy) {
                strategy.recordFailure();
            }
        }
    }

    async executeCrossExchangeArbitrage(opportunity, executionId) {
        const { buyExchange, sellExchange, symbol, quantity, buyPrice, sellPrice } = opportunity;

        // Simultaneous order placement to minimize slippage
        const [buyOrder, sellOrder] = await Promise.all([
            this.placeOrder(buyExchange, 'buy', symbol, quantity, buyPrice),
            this.placeOrder(sellExchange, 'sell', symbol, quantity, sellPrice)
        ]);

        return { buyOrder, sellOrder, executionId };
    }

    async executeStatisticalArbitrage(opportunity, executionId) {
        const { asset1, asset2, action, ratio, confidence } = opportunity;

        if (action === 'long_short') {
            // Long asset1, short asset2
            const [longOrder, shortOrder] = await Promise.all([
                this.placeOrder('primary', 'buy', asset1.symbol, asset1.quantity, asset1.price),
                this.placeOrder('primary', 'sell', asset2.symbol, asset2.quantity, asset2.price)
            ]);
            return { longOrder, shortOrder, executionId };
        }
    }

    async executeNewsBasedArbitrage(opportunity, executionId) {
        const { symbol, direction, confidence, newsImpact, timeDecay } = opportunity;

        // Quick execution based on sentiment analysis
        const adjustedQuantity = this.calculateNewsBasedPosition(opportunity);
        
        const order = await this.placeOrder(
            'primary',
            direction,
            symbol,
            adjustedQuantity,
            opportunity.targetPrice
        );

        // Set automatic exit based on time decay
        setTimeout(() => {
            this.closePosition(order.orderId, 'time_decay');
        }, opportunity.holdingPeriod);

        return { order, executionId };
    }

    async executeIndexArbitrage(opportunity, executionId) {
        const { etf, basket, action, deviation } = opportunity;

        if (action === 'buy_etf_sell_basket') {
            // Buy ETF, sell constituent stocks
            const etfOrder = await this.placeOrder('primary', 'buy', etf.symbol, etf.quantity, etf.price);
            
            const basketOrders = await Promise.all(
                basket.constituents.map(stock => 
                    this.placeOrder('primary', 'sell', stock.symbol, stock.quantity, stock.price)
                )
            );

            return { etfOrder, basketOrders, executionId };
        }
    }

    async executeLatencyArbitrage(opportunity, executionId) {
        const { venue1, venue2, symbol, quantity, priceGap } = opportunity;

        // Ultra-fast execution using IOC orders
        const order = await this.placeIOCOrder(
            venue2,
            opportunity.direction,
            symbol,
            quantity,
            opportunity.targetPrice
        );

        return { order, executionId };
    }

    // Advanced order placement with ultra-low latency
    async placeOrder(exchange, side, symbol, quantity, price, orderType = 'limit') {
        const startTime = performance.now();
        
        try {
            const exchangeConnection = this.exchangeConnections.get(exchange);
            if (!exchangeConnection) {
                throw new Error(`Exchange ${exchange} not connected`);
            }

            const order = await exchangeConnection.placeOrder({
                symbol,
                side,
                quantity,
                price,
                type: orderType,
                timeInForce: 'IOC', // Immediate or Cancel for HFT
                timestamp: Date.now()
            });

            const latency = performance.now() - startTime;
            this.executionLatencyTracker.recordLatency('order_placement', latency);

            return order;

        } catch (error) {
            console.error('Order placement failed:', error);
            throw error;
        }
    }

    async placeIOCOrder(venue, side, symbol, quantity, price) {
        // Immediate or Cancel order for latency arbitrage
        return await this.placeOrder(venue, side, symbol, quantity, price, 'IOC');
    }

    // Risk Management and Validation
    async validateOpportunity(opportunity) {
        // Check if prices are still valid (latency check)
        const currentPrices = await this.getCurrentPrices(opportunity.symbols);
        const priceDeviation = this.calculatePriceDeviation(opportunity.prices, currentPrices);
        
        return priceDeviation < 0.001; // 0.1% max deviation
    }

    async assessRisk(opportunity) {
        const factors = {
            liquidityRisk: await this.assessLiquidityRisk(opportunity),
            marketRisk: this.assessMarketRisk(opportunity),
            executionRisk: this.assessExecutionRisk(opportunity),
            correlationRisk: this.assessCorrelationRisk(opportunity)
        };

        // Weighted risk score
        const weights = { liquidityRisk: 0.3, marketRisk: 0.3, executionRisk: 0.25, correlationRisk: 0.15 };
        
        return Object.entries(factors).reduce((total, [risk, value]) => {
            return total + (value * weights[risk]);
        }, 0);
    }

    calculateProfitBasisPoints(opportunity) {
        if (opportunity.type === 'cross_exchange') {
            const spread = opportunity.sellPrice - opportunity.buyPrice;
            return (spread / opportunity.buyPrice) * 10000; // Convert to basis points
        }
        return opportunity.expectedReturn * 10000;
    }

    // Sentiment Analysis Integration
    async scanNewsBasedOpportunities() {
        const recentNews = await this.sentimentAnalyzer.getRecentNews();
        const socialSentiment = await this.sentimentAnalyzer.getSocialSentiment();

        for (const newsItem of recentNews) {
            if (newsItem.confidence > 0.8 && newsItem.impact > 0.3) {
                const opportunity = await this.createNewsBasedOpportunity(newsItem, socialSentiment);
                if (opportunity) {
                    await this.evaluateOpportunity(opportunity, 'news_based');
                }
            }
        }
    }

    async createNewsBasedOpportunity(newsItem, socialSentiment) {
        const affectedSymbols = newsItem.entities.filter(e => e.type === 'STOCK_SYMBOL');
        
        if (affectedSymbols.length === 0) return null;

        const symbol = affectedSymbols[0].value;
        const sentiment = newsItem.sentiment;
        const socialConfirmation = socialSentiment[symbol] || { score: 0 };

        // Combined sentiment scoring
        const combinedSentiment = (sentiment.score * 0.7) + (socialConfirmation.score * 0.3);
        const direction = combinedSentiment > 0 ? 'buy' : 'sell';

        return {
            type: 'news_based',
            symbol,
            direction,
            confidence: Math.abs(combinedSentiment),
            newsImpact: newsItem.impact,
            timeDecay: this.calculateTimeDecay(newsItem.timestamp),
            targetPrice: await this.calculateNewsTargetPrice(symbol, combinedSentiment),
            quantity: this.calculateNewsBasedPosition({ sentiment: combinedSentiment, impact: newsItem.impact }),
            holdingPeriod: this.calculateOptimalHoldingPeriod(newsItem)
        };
    }

    // Performance Monitoring
    startLatencyMonitoring() {
        setInterval(() => {
            const stats = this.executionLatencyTracker.getStats();
            
            console.log('📊 HFT Performance Stats:', {
                avgScanLatency: `${stats.scan.avg.toFixed(2)}ms`,
                avgExecutionLatency: `${stats.execute.avg.toFixed(2)}ms`,
                avgOrderLatency: `${stats.order_placement.avg.toFixed(2)}ms`,
                totalOpportunities: this.activeOpportunities.size
            });

            // Alert if performance degrades
            if (stats.execute.avg > this.hftConfig.maxLatencyMs) {
                console.warn('⚠️ HFT latency degraded, optimizing...');
                this.optimizePerformance();
            }
        }, 10000); // Every 10 seconds
    }

    optimizePerformance() {
        // Dynamic performance optimization
        console.log('🔧 Optimizing HFT performance...');
        
        // Reduce scanning frequency if needed
        if (this.executionLatencyTracker.getStats().scan.avg > 5) {
            console.log('Reducing scan frequency to improve performance');
        }

        // Optimize memory usage
        this.cleanupOldOpportunities();
        
        // Garbage collection hint
        if (global.gc) {
            global.gc();
        }
    }

    cleanupOldOpportunities() {
        const cutoffTime = Date.now() - (5 * 60 * 1000); // 5 minutes ago
        
        for (const [id, opportunity] of this.activeOpportunities.entries()) {
            if (opportunity.timestamp < cutoffTime) {
                this.activeOpportunities.delete(id);
            }
        }
    }

    // Strategy Management
    enableStrategy(strategyName, config = {}) {
        const strategy = this.strategies.get(strategyName);
        if (strategy) {
            strategy.enable(config);
            console.log(`✅ Strategy ${strategyName} enabled`);
        }
    }

    disableStrategy(strategyName) {
        const strategy = this.strategies.get(strategyName);
        if (strategy) {
            strategy.disable();
            console.log(`⏹️ Strategy ${strategyName} disabled`);
        }
    }

    getArbitrageStats() {
        const stats = {
            totalOpportunities: this.activeOpportunities.size,
            strategiesActive: Array.from(this.strategies.entries())
                .filter(([name, strategy]) => strategy.isEnabled()).length,
            avgLatency: this.executionLatencyTracker.getStats(),
            profitabilityByStrategy: this.calculateStrategyProfitability()
        };

        return stats;
    }

    calculateStrategyProfitability() {
        const profitability = {};
        
        for (const [id, opportunity] of this.activeOpportunities.entries()) {
            const strategy = opportunity.strategyName;
            if (!profitability[strategy]) {
                profitability[strategy] = { count: 0, totalProfit: 0 };
            }
            
            profitability[strategy].count++;
            profitability[strategy].totalProfit += opportunity.realizedProfit || 0;
        }

        return profitability;
    }

    // Utility methods for calculations
    calculateTimeDecay(timestamp) {
        const elapsed = Date.now() - timestamp;
        return Math.exp(-elapsed / (5 * 60 * 1000)); // 5-minute decay
    }

    calculateNewsTargetPrice(symbol, sentiment) {
        // Placeholder - would integrate with price prediction model
        const currentPrice = this.getCurrentPrice(symbol);
        const impact = sentiment * 0.02; // 2% max impact
        return currentPrice * (1 + impact);
    }

    calculateNewsBasedPosition(opportunity) {
        const baseSize = 1000; // Base position size
        const confidenceMultiplier = opportunity.confidence || opportunity.sentiment;
        const impactMultiplier = opportunity.impact || 1;
        
        return Math.floor(baseSize * confidenceMultiplier * impactMultiplier);
    }

    calculateOptimalHoldingPeriod(newsItem) {
        // News impact typically decays within 15-30 minutes
        const baseHolding = 15 * 60 * 1000; // 15 minutes
        const impactAdjustment = newsItem.impact * 15 * 60 * 1000; // Up to 15 more minutes for high impact
        
        return baseHolding + impactAdjustment;
    }

    getCurrentPrice(symbol) {
        // Placeholder - would fetch from real-time feed
        return 100 + (Math.random() * 20);
    }

    async getCurrentPrices(symbols) {
        // Fetch current prices for validation
        const prices = {};
        for (const symbol of symbols) {
            prices[symbol] = this.getCurrentPrice(symbol);
        }
        return prices;
    }

    calculatePriceDeviation(oldPrices, newPrices) {
        let totalDeviation = 0;
        let count = 0;
        
        for (const [symbol, oldPrice] of Object.entries(oldPrices)) {
            if (newPrices[symbol]) {
                totalDeviation += Math.abs((newPrices[symbol] - oldPrice) / oldPrice);
                count++;
            }
        }
        
        return count > 0 ? totalDeviation / count : 0;
    }

    // Risk assessment methods
    async assessLiquidityRisk(opportunity) {
        // Check order book depth and bid-ask spreads
        return 0.1; // Placeholder
    }

    assessMarketRisk(opportunity) {
        // Assess market volatility and correlation risks
        return 0.15; // Placeholder
    }

    assessExecutionRisk(opportunity) {
        // Assess slippage and partial fill risks
        return 0.1; // Placeholder
    }

    assessCorrelationRisk(opportunity) {
        // Assess correlation between assets
        return 0.05; // Placeholder
    }

    handleInitializationError(error) {
        console.warn('⚠️ Arbitrage engine running in limited mode due to:', error.message);
        
        // Set up basic functionality even if full initialization failed
        this.strategies.set('demo', {
            isEnabled: () => true,
            scan: async () => {
                return [{
                    type: 'demo',
                    symbol: 'BTC/USD',
                    profit: 0.001,
                    status: 'simulated',
                    timestamp: Date.now()
                }];
            }
        });
        
        // Emit limited mode event
        if (typeof document !== 'undefined') {
            document.dispatchEvent(new CustomEvent('arbitrage:limited_mode', {
                detail: { error: error.message }
            }));
        }
    }
}

/**
 * SUPPORTING CLASSES FOR ARBITRAGE STRATEGIES
 */

class LatencyTracker {
    constructor() {
        this.metrics = {
            scan: [],
            evaluate: [],
            execute: [],
            order_placement: []
        };
        this.maxHistory = 1000; // Keep last 1000 measurements
    }

    recordLatency(operation, latencyMs) {
        if (this.metrics[operation]) {
            this.metrics[operation].push(latencyMs);
            
            // Trim history if too long
            if (this.metrics[operation].length > this.maxHistory) {
                this.metrics[operation] = this.metrics[operation].slice(-this.maxHistory);
            }
        }
    }

    getStats() {
        const stats = {};
        
        for (const [operation, latencies] of Object.entries(this.metrics)) {
            if (latencies.length > 0) {
                stats[operation] = {
                    avg: latencies.reduce((a, b) => a + b, 0) / latencies.length,
                    min: Math.min(...latencies),
                    max: Math.max(...latencies),
                    p95: this.calculatePercentile(latencies, 0.95),
                    p99: this.calculatePercentile(latencies, 0.99),
                    count: latencies.length
                };
            } else {
                stats[operation] = { avg: 0, min: 0, max: 0, p95: 0, p99: 0, count: 0 };
            }
        }
        
        return stats;
    }

    calculatePercentile(arr, percentile) {
        const sorted = [...arr].sort((a, b) => a - b);
        const index = Math.floor(sorted.length * percentile);
        return sorted[index] || 0;
    }
}

// Strategy Base Classes
class BaseArbitrageStrategy {
    constructor(config) {
        this.config = config;
        this.enabled = false;
        this.engine = null;
        this.failures = 0;
        this.maxFailures = 10;
    }

    setEngine(engine) {
        this.engine = engine;
    }

    enable(config = {}) {
        this.config = { ...this.config, ...config };
        this.enabled = true;
        this.failures = 0;
    }

    disable() {
        this.enabled = false;
    }

    isEnabled() {
        return this.enabled && this.failures < this.maxFailures;
    }

    recordFailure() {
        this.failures++;
        if (this.failures >= this.maxFailures) {
            console.warn(`Strategy disabled due to too many failures: ${this.constructor.name}`);
            this.enabled = false;
        }
    }

    async scan() {
        // To be implemented by each strategy
        throw new Error('scan() method must be implemented by strategy');
    }
}

class CrossExchangeArbitrageStrategy extends BaseArbitrageStrategy {
    async scan() {
        const opportunities = [];
        
        try {
            const symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD'];
            
            for (const symbol of symbols) {
                const prices = await this.getPricesFromExchanges(symbol);
                const opportunity = this.findBestSpread(symbol, prices);
                
                if (opportunity && opportunity.profit > this.config.minSpread) {
                    opportunities.push(opportunity);
                }
            }
        } catch (error) {
            console.error('Cross-exchange scan error:', error);
        }

        return opportunities;
    }

    async getPricesFromExchanges(symbol) {
        const exchanges = this.config.exchanges;
        const prices = {};

        for (const exchange of exchanges) {
            try {
                prices[exchange] = await this.getPrice(exchange, symbol);
            } catch (error) {
                console.error(`Failed to get price from ${exchange}:`, error);
            }
        }

        return prices;
    }

    findBestSpread(symbol, prices) {
        let bestOpportunity = null;
        let maxProfit = 0;

        const exchanges = Object.keys(prices);
        
        for (let i = 0; i < exchanges.length; i++) {
            for (let j = i + 1; j < exchanges.length; j++) {
                const exchange1 = exchanges[i];
                const exchange2 = exchanges[j];
                
                const price1 = prices[exchange1];
                const price2 = prices[exchange2];
                
                if (price1 && price2) {
                    const profit1 = (price2.bid - price1.ask) / price1.ask;
                    const profit2 = (price1.bid - price2.ask) / price2.ask;
                    
                    if (profit1 > maxProfit) {
                        maxProfit = profit1;
                        bestOpportunity = {
                            type: 'cross_exchange',
                            symbol,
                            buyExchange: exchange1,
                            sellExchange: exchange2,
                            buyPrice: price1.ask,
                            sellPrice: price2.bid,
                            profit: profit1,
                            quantity: Math.min(price1.askSize, price2.bidSize)
                        };
                    }
                    
                    if (profit2 > maxProfit) {
                        maxProfit = profit2;
                        bestOpportunity = {
                            type: 'cross_exchange',
                            symbol,
                            buyExchange: exchange2,
                            sellExchange: exchange1,
                            buyPrice: price2.ask,
                            sellPrice: price1.bid,
                            profit: profit2,
                            quantity: Math.min(price2.askSize, price1.bidSize)
                        };
                    }
                }
            }
        }

        return bestOpportunity;
    }

    async getPrice(exchange, symbol) {
        // Placeholder - would fetch real price data
        const basePrice = 50000 + (Math.random() * 1000);
        const spread = basePrice * 0.001; // 0.1% spread
        
        return {
            bid: basePrice - spread/2,
            ask: basePrice + spread/2,
            bidSize: 100 + (Math.random() * 900),
            askSize: 100 + (Math.random() * 900)
        };
    }
}

class StatisticalArbitrageStrategy extends BaseArbitrageStrategy {
    async scan() {
        const opportunities = [];
        
        try {
            const pairs = await this.getCointegrationPairs();
            
            for (const pair of pairs) {
                const zscore = await this.calculateZScore(pair);
                
                if (Math.abs(zscore) > this.config.zscore_threshold) {
                    const opportunity = {
                        type: 'statistical',
                        asset1: pair.asset1,
                        asset2: pair.asset2,
                        zscore,
                        action: zscore > 0 ? 'short_long' : 'long_short',
                        confidence: Math.min(Math.abs(zscore) / 3, 1),
                        expectedReturn: Math.abs(zscore) * 0.01,
                        ratio: pair.hedgeRatio
                    };
                    
                    opportunities.push(opportunity);
                }
            }
        } catch (error) {
            console.error('Statistical arbitrage scan error:', error);
        }

        return opportunities;
    }

    async getCointegrationPairs() {
        // Placeholder - would run cointegration tests on asset pairs
        return [
            { asset1: { symbol: 'AAPL', price: 175 }, asset2: { symbol: 'MSFT', price: 380 }, hedgeRatio: 0.46 },
            { asset1: { symbol: 'JPM', price: 150 }, asset2: { symbol: 'BAC', price: 35 }, hedgeRatio: 4.28 }
        ];
    }

    async calculateZScore(pair) {
        // Calculate z-score of the spread
        // Placeholder calculation
        return (Math.random() - 0.5) * 6; // Random z-score between -3 and 3
    }
}

class NewsBasedArbitrageStrategy extends BaseArbitrageStrategy {
    async scan() {
        const opportunities = [];
        
        try {
            const recentNews = await this.getRecentMarketNews();
            
            for (const newsItem of recentNews) {
                if (newsItem.sentiment.confidence > this.config.confidenceThreshold) {
                    const opportunity = await this.createNewsOpportunity(newsItem);
                    if (opportunity) {
                        opportunities.push(opportunity);
                    }
                }
            }
        } catch (error) {
            console.error('News-based arbitrage scan error:', error);
        }

        return opportunities;
    }

    async getRecentMarketNews() {
        // Placeholder - would fetch real news data
        return [
            {
                title: "AAPL reports strong earnings",
                sentiment: { score: 0.85, confidence: 0.92 },
                symbols: ['AAPL'],
                timestamp: Date.now() - 30000, // 30 seconds ago
                impact: 0.8
            }
        ];
    }

    async createNewsOpportunity(newsItem) {
        if (newsItem.symbols.length === 0) return null;

        const symbol = newsItem.symbols[0];
        const sentiment = newsItem.sentiment.score;
        
        return {
            type: 'news_based',
            symbol,
            direction: sentiment > 0 ? 'buy' : 'sell',
            confidence: newsItem.sentiment.confidence,
            newsImpact: newsItem.impact,
            timeDecay: this.calculateTimeDecay(newsItem.timestamp),
            expectedReturn: Math.abs(sentiment) * newsItem.impact * 0.02
        };
    }

    calculateTimeDecay(timestamp) {
        const elapsed = Date.now() - timestamp;
        return Math.exp(-elapsed / (this.config.newsLatencyMs * 1000));
    }
}

class IndexArbitrageStrategy extends BaseArbitrageStrategy {
    async scan() {
        const opportunities = [];
        
        try {
            for (const etf of this.config.etfs) {
                const trackingError = await this.calculateTrackingError(etf);
                
                if (Math.abs(trackingError) > this.config.basketTrackingError) {
                    const opportunity = {
                        type: 'index_arbitrage',
                        etf: { symbol: etf, price: await this.getETFPrice(etf) },
                        basket: await this.getBasketPrices(etf),
                        deviation: trackingError,
                        action: trackingError > 0 ? 'sell_etf_buy_basket' : 'buy_etf_sell_basket',
                        expectedReturn: Math.abs(trackingError)
                    };
                    
                    opportunities.push(opportunity);
                }
            }
        } catch (error) {
            console.error('Index arbitrage scan error:', error);
        }

        return opportunities;
    }

    async calculateTrackingError(etf) {
        // Calculate deviation between ETF and its underlying basket
        // Placeholder calculation
        return (Math.random() - 0.5) * 0.002; // ±0.2% tracking error
    }

    async getETFPrice(etf) {
        // Placeholder ETF price
        return 450 + (Math.random() * 10);
    }

    async getBasketPrices(etf) {
        // Placeholder basket constituents
        return {
            constituents: [
                { symbol: 'AAPL', price: 175, weight: 0.07 },
                { symbol: 'MSFT', price: 380, weight: 0.065 },
                { symbol: 'GOOGL', price: 140, weight: 0.04 }
            ]
        };
    }
}

class LatencyArbitrageStrategy extends BaseArbitrageStrategy {
    async scan() {
        const opportunities = [];
        
        try {
            for (const venue of this.config.venues) {
                const latency = await this.measureVenueLatency(venue);
                
                if (latency < this.config.maxLatency) {
                    const priceDiscrepancies = await this.findPriceDiscrepancies(venue);
                    opportunities.push(...priceDiscrepancies);
                }
            }
        } catch (error) {
            console.error('Latency arbitrage scan error:', error);
        }

        return opportunities;
    }

    async measureVenueLatency(venue) {
        // Measure order routing latency to venue
        // Placeholder
        return Math.random() * 2; // 0-2ms latency
    }

    async findPriceDiscrepancies(venue) {
        // Find price discrepancies exploitable with low latency
        // Placeholder
        return [{
            type: 'latency_arbitrage',
            venue1: venue,
            venue2: 'alternative_venue',
            symbol: 'AAPL',
            priceGap: 0.01,
            direction: 'buy',
            targetPrice: 175.25,
            quantity: 1000
        }];
    }
}

class VolatilityArbitrageStrategy extends BaseArbitrageStrategy {
    async scan() {
        const opportunities = [];
        
        try {
            const symbols = ['AAPL', 'GOOGL', 'TSLA'];
            
            for (const symbol of symbols) {
                const impliedVol = await this.getImpliedVolatility(symbol);
                const realizedVol = await this.getRealizedVolatility(symbol);
                
                const volDiff = impliedVol - realizedVol;
                
                if (Math.abs(volDiff) > this.config.impliedVolThreshold) {
                    const opportunity = {
                        type: 'volatility_arbitrage',
                        symbol,
                        impliedVol,
                        realizedVol,
                        volDifference: volDiff,
                        strategy: volDiff > 0 ? 'sell_volatility' : 'buy_volatility',
                        expectedReturn: Math.abs(volDiff) * 0.1
                    };
                    
                    opportunities.push(opportunity);
                }
            }
        } catch (error) {
            console.error('Volatility arbitrage scan error:', error);
        }

        return opportunities;
    }

    async getImpliedVolatility(symbol) {
        // Get implied volatility from options market
        return 0.20 + (Math.random() * 0.10); // 20-30% IV
    }

    async getRealizedVolatility(symbol) {
        // Calculate realized volatility from historical prices
        return 0.18 + (Math.random() * 0.08); // 18-26% RV
    }
}

// Supporting Classes
class FinBERTSentimentAnalyzer {
    constructor() {
        this.newsSources = [];
        this.sentimentCache = new Map();
        this.socialSentimentCache = new Map();
    }

    async initialize() {
        console.log('🤖 Initializing FinBERT sentiment analyzer...');
        // Initialize FinBERT model (placeholder)
        this.model = { ready: true };
    }

    addNewsSource(source) {
        this.newsSources.push(source);
    }

    async getRecentNews() {
        // Fetch and analyze recent news
        const news = [
            {
                title: "Federal Reserve signals potential rate cut",
                content: "The Federal Reserve indicated it may lower interest rates...",
                timestamp: Date.now() - 60000,
                entities: [{ type: 'STOCK_SYMBOL', value: 'SPY' }],
                sentiment: { score: 0.65, confidence: 0.89 },
                impact: 0.7
            }
        ];

        return news;
    }

    async getSocialSentiment() {
        // Get social media sentiment
        return {
            'AAPL': { score: 0.72, volume: 15420 },
            'TSLA': { score: -0.23, volume: 8910 },
            'GOOGL': { score: 0.45, volume: 6730 }
        };
    }
}

class TradingVolumeAnalyzer {
    constructor() {
        this.volumeData = new Map();
        this.patterns = new Map();
    }

    async getVolumeProfile(symbol) {
        // Get volume profile for symbol
        return {
            avgVolume: 1000000,
            currentVolume: 1200000,
            volumeRatio: 1.2,
            unusualActivity: true
        };
    }

    async detectVolumeAnomalies() {
        // Detect unusual volume patterns
        return [
            {
                symbol: 'AAPL',
                volumeSpike: 2.5, // 2.5x normal volume
                confidence: 0.85,
                direction: 'bullish'
            }
        ];
    }
}

class StatisticalArbitrageModels {
    constructor() {
        this.models = new Map();
        this.correlationMatrix = new Map();
    }

    async updateModels() {
        // Update statistical models with latest data
        console.log('📈 Updating statistical arbitrage models...');
    }

    async getCointegrationPairs() {
        // Return cointegrated asset pairs
        return [
            { asset1: 'AAPL', asset2: 'MSFT', correlation: 0.85, hedgeRatio: 0.46 },
            { asset1: 'JPM', asset2: 'BAC', correlation: 0.92, hedgeRatio: 4.28 }
        ];
    }
}

class ExchangeConnection {
    constructor(exchangeConfig) {
        this.config = exchangeConfig;
        this.websocket = null;
        this.connected = false;
        this.orderBook = new Map();
    }

    async connect() {
        try {
            // Connect to exchange WebSocket
            this.websocket = { connected: true }; // Placeholder
            this.connected = true;
            console.log(`📡 Connected to ${this.config.name}`);
        } catch (error) {
            console.error(`Failed to connect to ${this.config.name}:`, error);
            throw error;
        }
    }

    async placeOrder(orderData) {
        // Place order on exchange
        const order = {
            orderId: `${this.config.name}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            ...orderData,
            status: 'filled',
            timestamp: Date.now()
        };

        return order;
    }

    async getOrderBook(symbol) {
        // Get current order book
        return {
            symbol,
            bids: [[50000, 1.5], [49990, 2.1], [49980, 0.8]],
            asks: [[50010, 1.2], [50020, 1.8], [50030, 2.4]],
            timestamp: Date.now()
        };
    }
}

class HighFrequencyDataFeed {
    constructor(config) {
        this.config = config;
        this.subscribers = new Set();
        this.latestData = new Map();
    }

    async initialize() {
        console.log(`📊 Initializing ${this.config.provider} data feed...`);
        // Start data feed
        setInterval(() => {
            this.generateData();
        }, 10); // 10ms updates for HFT
    }

    generateData() {
        // Generate high-frequency market data
        for (const symbol of this.config.symbols) {
            const data = {
                symbol,
                price: 50000 + (Math.random() * 1000),
                volume: Math.floor(Math.random() * 10000),
                timestamp: Date.now()
            };

            this.latestData.set(symbol, data);
            this.notifySubscribers(data);
        }
    }

    notifySubscribers(data) {
        for (const callback of this.subscribers) {
            callback(data);
        }
    }

    subscribe(callback) {
        this.subscribers.add(callback);
    }

    unsubscribe(callback) {
        this.subscribers.delete(callback);
    }
}

// Export for global usage
if (typeof window !== 'undefined') {
    window.ArbitrageEngine = ArbitrageEngine;
    window.CrossExchangeArbitrageStrategy = CrossExchangeArbitrageStrategy;
    window.StatisticalArbitrageStrategy = StatisticalArbitrageStrategy;
    window.NewsBasedArbitrageStrategy = NewsBasedArbitrageStrategy;
    window.IndexArbitrageStrategy = IndexArbitrageStrategy;
    window.LatencyArbitrageStrategy = LatencyArbitrageStrategy;
    window.VolatilityArbitrageStrategy = VolatilityArbitrageStrategy;
}

console.log('🚀 Enhanced Arbitrage & HFT Engine loaded successfully');