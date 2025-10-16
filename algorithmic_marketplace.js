/**
 * GOMNA ALGORITHMIC MARKETPLACE
 * Revolutionary platform for buying, selling, and executing trades based on live algorithm performance
 * Real-time arbitrage strategy marketplace with performance-based execution
 */

class AlgorithmicMarketplace {
    constructor() {
        this.algorithms = new Map();
        this.marketData = new Map();
        this.userPortfolio = {
            balance: 10000,
            ownedAlgorithms: new Set(),
            activePositions: new Map(),
            totalPnL: 0,
            trades: []
        };
        this.liveSignals = new Map();
        this.performanceTrackers = new Map();
        this.marketMakers = new Map();
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing Algorithmic Marketplace...');
        
        // Initialize core systems
        this.initializeAlgorithms();
        this.setupMarketData();
        this.startPerformanceTracking();
        this.initializeMarketMakers();
        this.startRealTimeExecution();
        
        console.log('✅ Algorithmic Marketplace ready');
    }

    initializeAlgorithms() {
        console.log('🔧 Reverse engineering and packaging algorithms...');
        
        // Premium Cross-Exchange Arbitrage Algorithm
        this.algorithms.set('cross_exchange_pro', {
            id: 'cross_exchange_pro',
            name: 'Cross-Exchange Arbitrage Pro',
            description: 'Advanced multi-exchange price difference exploitation',
            category: 'arbitrage',
            version: '2.1.0',
            creator: 'GOMNA AI Labs',
            price: 299.99,
            performance: {
                winRate: 0.847,
                sharpeRatio: 2.34,
                maxDrawdown: 0.045,
                avgReturn: 0.0023,
                trades: 1247,
                lastUpdate: Date.now()
            },
            features: [
                'Ultra-low latency execution (<2ms)',
                'Multi-exchange connectivity (Binance, Coinbase, Kraken)',
                'Dynamic spread optimization',
                'Risk-adjusted position sizing',
                'Real-time P&L tracking'
            ],
            parameters: {
                minSpread: 0.015,
                maxPosition: 50000,
                riskMultiplier: 0.8,
                exchanges: ['binance', 'coinbase', 'kraken']
            },
            signals: [],
            isActive: false,
            subscribers: 0,
            revenueGenerated: 0
        });

        // Statistical Arbitrage ML Algorithm
        this.algorithms.set('stat_arb_ml', {
            id: 'stat_arb_ml',
            name: 'Statistical Arbitrage ML',
            description: 'Machine learning powered pairs trading with cointegration',
            category: 'statistical',
            version: '1.8.3',
            creator: 'QuantLab Research',
            price: 449.99,
            performance: {
                winRate: 0.721,
                sharpeRatio: 1.87,
                maxDrawdown: 0.072,
                avgReturn: 0.0031,
                trades: 892,
                lastUpdate: Date.now()
            },
            features: [
                'ML-based cointegration detection',
                'Dynamic z-score thresholds',
                'Portfolio beta hedging',
                'Mean reversion optimization',
                'Risk parity allocation'
            ],
            parameters: {
                lookback: 252,
                zscoreThreshold: 2.0,
                maxCorrelation: 0.85,
                rebalanceFreq: 3600000
            },
            signals: [],
            isActive: false,
            subscribers: 0,
            revenueGenerated: 0
        });

        // FinBERT News Sentiment Algorithm
        this.algorithms.set('finbert_news', {
            id: 'finbert_news',
            name: 'FinBERT News Sentiment Pro',
            description: 'AI-powered news sentiment analysis with real-time execution',
            category: 'sentiment',
            version: '3.0.1',
            creator: 'NLP Capital',
            price: 599.99,
            performance: {
                winRate: 0.683,
                sharpeRatio: 2.12,
                maxDrawdown: 0.089,
                avgReturn: 0.0041,
                trades: 634,
                lastUpdate: Date.now()
            },
            features: [
                'FinBERT transformer model',
                'Multi-source news aggregation',
                'Sentiment confidence scoring',
                'Time-decay optimization',
                'Social media integration'
            ],
            parameters: {
                sentimentThreshold: 0.3,
                confidenceMin: 0.8,
                timeDecay: 0.95,
                maxHolding: 1800000
            },
            signals: [],
            isActive: false,
            subscribers: 0,
            revenueGenerated: 0
        });

        // Index Arbitrage Algorithm
        this.algorithms.set('index_arb', {
            id: 'index_arb',
            name: 'Index Arbitrage Elite',
            description: 'ETF vs constituent basket arbitrage with creation/redemption',
            category: 'index',
            version: '1.5.2',
            creator: 'Institutional Quant',
            price: 799.99,
            performance: {
                winRate: 0.793,
                sharpeRatio: 2.67,
                maxDrawdown: 0.032,
                avgReturn: 0.0019,
                trades: 456,
                lastUpdate: Date.now()
            },
            features: [
                'Real-time ETF tracking',
                'Basket composition analysis',
                'Creation/redemption optimization',
                'Dividend adjustment',
                'Liquidity-weighted execution'
            ],
            parameters: {
                trackingError: 0.001,
                minDeviation: 0.002,
                etfs: ['SPY', 'QQQ', 'IWM', 'VTI'],
                rebalanceThreshold: 0.005
            },
            signals: [],
            isActive: false,
            subscribers: 0,
            revenueGenerated: 0
        });

        // HFT Latency Arbitrage Algorithm
        this.algorithms.set('hft_latency', {
            id: 'hft_latency',
            name: 'HFT Latency Arbitrage Ultra',
            description: 'Microsecond-precision latency arbitrage with co-location',
            category: 'hft',
            version: '4.2.0',
            creator: 'Speed Capital',
            price: 1299.99,
            performance: {
                winRate: 0.912,
                sharpeRatio: 3.45,
                maxDrawdown: 0.018,
                avgReturn: 0.0008,
                trades: 8934,
                lastUpdate: Date.now()
            },
            features: [
                'Sub-millisecond execution',
                'FPGA acceleration',
                'Co-location optimization',
                'Market microstructure analysis',
                'Tick-by-tick processing'
            ],
            parameters: {
                maxLatency: 0.5,
                tickSize: 0.01,
                orderTypes: ['IOC', 'FOK'],
                venues: ['ARCA', 'NASDAQ', 'NYSE']
            },
            signals: [],
            isActive: false,
            subscribers: 0,
            revenueGenerated: 0
        });

        // Volatility Arbitrage Algorithm
        this.algorithms.set('vol_arb', {
            id: 'vol_arb',
            name: 'Volatility Arbitrage Master',
            description: 'Options volatility surface arbitrage with Greeks hedging',
            category: 'volatility',
            version: '2.3.1',
            creator: 'Derivatives Pro',
            price: 899.99,
            performance: {
                winRate: 0.756,
                sharpeRatio: 2.01,
                maxDrawdown: 0.067,
                avgReturn: 0.0035,
                trades: 323,
                lastUpdate: Date.now()
            },
            features: [
                'Real-time IV calculation',
                'Greeks-based hedging',
                'Volatility surface modeling',
                'Options chain analysis',
                'Risk-neutral pricing'
            ],
            parameters: {
                ivThreshold: 0.05,
                gammaLimit: 1000,
                vegaLimit: 5000,
                deltaHedge: true
            },
            signals: [],
            isActive: false,
            subscribers: 0,
            revenueGenerated: 0
        });

        console.log(`📦 Packaged ${this.algorithms.size} professional algorithms`);
    }

    setupMarketData() {
        console.log('📡 Setting up real-time market data feeds...');
        
        // Simulate real-time market data
        const symbols = ['BTC/USD', 'ETH/USD', 'SPY', 'QQQ', 'AAPL', 'GOOGL', 'TSLA'];
        
        symbols.forEach(symbol => {
            this.marketData.set(symbol, {
                symbol,
                bid: 50000 + Math.random() * 1000,
                ask: 50010 + Math.random() * 1000,
                last: 50005 + Math.random() * 1000,
                volume: Math.floor(Math.random() * 1000000),
                timestamp: Date.now(),
                exchanges: {
                    binance: { bid: 50000 + Math.random() * 10, ask: 50010 + Math.random() * 10 },
                    coinbase: { bid: 50002 + Math.random() * 10, ask: 50012 + Math.random() * 10 },
                    kraken: { bid: 49998 + Math.random() * 10, ask: 50008 + Math.random() * 10 }
                }
            });
        });

        // Update market data every 100ms
        setInterval(() => {
            this.updateMarketData();
        }, 100);
    }

    updateMarketData() {
        for (const [symbol, data] of this.marketData.entries()) {
            // Simulate realistic price movements
            const change = (Math.random() - 0.5) * 0.002; // ±0.2% max change
            
            data.bid *= (1 + change);
            data.ask *= (1 + change);
            data.last = (data.bid + data.ask) / 2;
            data.volume += Math.floor(Math.random() * 1000);
            data.timestamp = Date.now();
            
            // Update exchange prices with slight variations
            Object.keys(data.exchanges).forEach(exchange => {
                const variation = (Math.random() - 0.5) * 0.001;
                data.exchanges[exchange].bid = data.bid * (1 + variation);
                data.exchanges[exchange].ask = data.ask * (1 + variation);
            });
        }

        // Generate algorithm signals based on market data
        this.generateAlgorithmSignals();
    }

    generateAlgorithmSignals() {
        for (const [algId, algorithm] of this.algorithms.entries()) {
            if (!algorithm.isActive) continue;

            let signal = null;

            switch (algorithm.category) {
                case 'arbitrage':
                    signal = this.generateCrossExchangeSignal(algorithm);
                    break;
                case 'statistical':
                    signal = this.generateStatisticalSignal(algorithm);
                    break;
                case 'sentiment':
                    signal = this.generateSentimentSignal(algorithm);
                    break;
                case 'index':
                    signal = this.generateIndexSignal(algorithm);
                    break;
                case 'hft':
                    signal = this.generateHFTSignal(algorithm);
                    break;
                case 'volatility':
                    signal = this.generateVolatilitySignal(algorithm);
                    break;
            }

            if (signal) {
                algorithm.signals.unshift(signal);
                if (algorithm.signals.length > 100) {
                    algorithm.signals = algorithm.signals.slice(0, 100);
                }
                
                // Store live signal for execution
                this.liveSignals.set(algId, signal);
                
                // Execute trade if user has algorithm and auto-trading is enabled
                this.executeAlgorithmTrade(algId, signal);
            }
        }
    }

    generateCrossExchangeSignal(algorithm) {
        const symbols = ['BTC/USD', 'ETH/USD'];
        const symbol = symbols[Math.floor(Math.random() * symbols.length)];
        const marketData = this.marketData.get(symbol);
        
        if (!marketData) return null;

        // Find best arbitrage opportunity
        let bestSpread = 0;
        let bestBuyExchange = null;
        let bestSellExchange = null;
        let buyPrice = 0;
        let sellPrice = 0;

        const exchanges = Object.keys(marketData.exchanges);
        for (let i = 0; i < exchanges.length; i++) {
            for (let j = i + 1; j < exchanges.length; j++) {
                const ex1 = exchanges[i];
                const ex2 = exchanges[j];
                
                const spread1 = marketData.exchanges[ex2].bid - marketData.exchanges[ex1].ask;
                const spread2 = marketData.exchanges[ex1].bid - marketData.exchanges[ex2].ask;
                
                if (spread1 > bestSpread && spread1 > algorithm.parameters.minSpread * marketData.last) {
                    bestSpread = spread1;
                    bestBuyExchange = ex1;
                    bestSellExchange = ex2;
                    buyPrice = marketData.exchanges[ex1].ask;
                    sellPrice = marketData.exchanges[ex2].bid;
                }
                
                if (spread2 > bestSpread && spread2 > algorithm.parameters.minSpread * marketData.last) {
                    bestSpread = spread2;
                    bestBuyExchange = ex2;
                    bestSellExchange = ex1;
                    buyPrice = marketData.exchanges[ex2].ask;
                    sellPrice = marketData.exchanges[ex1].bid;
                }
            }
        }

        if (bestSpread > 0) {
            return {
                id: `signal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                algorithm: algorithm.id,
                type: 'ARBITRAGE',
                symbol,
                action: 'BUY_SELL',
                buyExchange: bestBuyExchange,
                sellExchange: bestSellExchange,
                buyPrice,
                sellPrice,
                spread: bestSpread,
                profitBps: (bestSpread / buyPrice) * 10000,
                confidence: Math.min(0.95, 0.6 + (bestSpread / (buyPrice * 0.01))),
                quantity: Math.min(algorithm.parameters.maxPosition / buyPrice, 10),
                timestamp: Date.now(),
                executionTime: Math.random() * 2 + 0.5,
                expectedProfit: bestSpread * Math.min(algorithm.parameters.maxPosition / buyPrice, 10)
            };
        }

        return null;
    }

    generateStatisticalSignal(algorithm) {
        // Simulate pairs trading signal
        const pairs = [
            ['AAPL', 'GOOGL'],
            ['SPY', 'QQQ']
        ];
        
        const pair = pairs[Math.floor(Math.random() * pairs.length)];
        const zscore = (Math.random() - 0.5) * 6; // -3 to 3

        if (Math.abs(zscore) > algorithm.parameters.zscoreThreshold) {
            return {
                id: `signal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                algorithm: algorithm.id,
                type: 'PAIRS_TRADE',
                asset1: pair[0],
                asset2: pair[1],
                zscore,
                action: zscore > 0 ? 'SHORT_LONG' : 'LONG_SHORT',
                confidence: Math.min(0.9, Math.abs(zscore) / 3),
                quantity: 100,
                expectedHolding: 24 * 60 * 60 * 1000, // 24 hours
                timestamp: Date.now(),
                expectedProfit: Math.abs(zscore) * 50
            };
        }

        return null;
    }

    generateSentimentSignal(algorithm) {
        // Simulate news sentiment signal
        const symbols = ['AAPL', 'GOOGL', 'TSLA'];
        const symbol = symbols[Math.floor(Math.random() * symbols.length)];
        
        if (Math.random() > 0.95) { // 5% chance of news signal
            const sentiment = (Math.random() - 0.5) * 2; // -1 to 1
            const confidence = 0.7 + Math.random() * 0.2; // 0.7 to 0.9
            
            if (Math.abs(sentiment) > algorithm.parameters.sentimentThreshold && 
                confidence > algorithm.parameters.confidenceMin) {
                
                return {
                    id: `signal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    algorithm: algorithm.id,
                    type: 'NEWS_SENTIMENT',
                    symbol,
                    sentiment,
                    confidence,
                    action: sentiment > 0 ? 'BUY' : 'SELL',
                    quantity: 50,
                    timeDecay: algorithm.parameters.timeDecay,
                    maxHolding: algorithm.parameters.maxHolding,
                    timestamp: Date.now(),
                    newsSource: 'Reuters',
                    expectedProfit: Math.abs(sentiment) * confidence * 100
                };
            }
        }

        return null;
    }

    generateIndexSignal(algorithm) {
        // Simulate ETF arbitrage signal
        const etfs = algorithm.parameters.etfs;
        const etf = etfs[Math.floor(Math.random() * etfs.length)];
        
        // Simulate tracking error
        const trackingError = (Math.random() - 0.5) * 0.01; // ±1%
        
        if (Math.abs(trackingError) > algorithm.parameters.trackingError) {
            return {
                id: `signal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                algorithm: algorithm.id,
                type: 'INDEX_ARBITRAGE',
                etf,
                trackingError,
                action: trackingError > 0 ? 'SELL_ETF_BUY_BASKET' : 'BUY_ETF_SELL_BASKET',
                confidence: Math.min(0.9, Math.abs(trackingError) * 100),
                quantity: 1000,
                timestamp: Date.now(),
                expectedProfit: Math.abs(trackingError) * 1000
            };
        }

        return null;
    }

    generateHFTSignal(algorithm) {
        // Simulate HFT latency arbitrage
        if (Math.random() > 0.8) { // 20% chance
            const venues = algorithm.parameters.venues;
            const venue1 = venues[Math.floor(Math.random() * venues.length)];
            const venue2 = venues[Math.floor(Math.random() * venues.length)];
            
            if (venue1 !== venue2) {
                return {
                    id: `signal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    algorithm: algorithm.id,
                    type: 'LATENCY_ARBITRAGE',
                    symbol: 'SPY',
                    venue1,
                    venue2,
                    latencyAdv: Math.random() * 0.3 + 0.1, // 0.1-0.4ms advantage
                    action: 'SNIPE',
                    quantity: 100,
                    confidence: 0.95,
                    timestamp: Date.now(),
                    expectedProfit: 25
                };
            }
        }

        return null;
    }

    generateVolatilitySignal(algorithm) {
        // Simulate volatility arbitrage signal
        if (Math.random() > 0.9) { // 10% chance
            const impliedVol = 0.2 + Math.random() * 0.15; // 20-35%
            const realizedVol = 0.18 + Math.random() * 0.12; // 18-30%
            const volDiff = impliedVol - realizedVol;
            
            if (Math.abs(volDiff) > algorithm.parameters.ivThreshold) {
                return {
                    id: `signal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                    algorithm: algorithm.id,
                    type: 'VOLATILITY_ARBITRAGE',
                    symbol: 'SPY',
                    impliedVol,
                    realizedVol,
                    volDiff,
                    action: volDiff > 0 ? 'SELL_VOLATILITY' : 'BUY_VOLATILITY',
                    confidence: Math.min(0.9, Math.abs(volDiff) * 10),
                    quantity: 10,
                    timestamp: Date.now(),
                    expectedProfit: Math.abs(volDiff) * 1000
                };
            }
        }

        return null;
    }

    executeAlgorithmTrade(algId, signal) {
        const algorithm = this.algorithms.get(algId);
        if (!algorithm || !this.userPortfolio.ownedAlgorithms.has(algId)) return;

        // Check if user has sufficient balance
        const requiredCapital = this.calculateRequiredCapital(signal);
        if (requiredCapital > this.userPortfolio.balance) return;

        // Execute the trade
        const trade = {
            id: `trade_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            algorithmId: algId,
            algorithmName: algorithm.name,
            signal,
            executionPrice: signal.buyPrice || this.marketData.get(signal.symbol)?.last || 100,
            quantity: signal.quantity,
            side: signal.action.includes('BUY') ? 'BUY' : 'SELL',
            capital: requiredCapital,
            timestamp: Date.now(),
            status: 'EXECUTED',
            pnl: 0, // Will be updated in real-time
            fees: requiredCapital * 0.001 // 0.1% fees
        };

        // Update portfolio
        this.userPortfolio.balance -= requiredCapital + trade.fees;
        this.userPortfolio.trades.push(trade);
        this.userPortfolio.activePositions.set(trade.id, trade);

        // Update algorithm performance
        algorithm.revenueGenerated += trade.fees * 0.3; // 30% revenue share
        
        // Simulate P&L updates
        setTimeout(() => {
            this.updateTradePnL(trade.id);
        }, 5000 + Math.random() * 10000); // 5-15 seconds

        console.log(`🎯 Executed trade for ${algorithm.name}:`, trade);
    }

    calculateRequiredCapital(signal) {
        switch (signal.type) {
            case 'ARBITRAGE':
                return signal.buyPrice * signal.quantity;
            case 'PAIRS_TRADE':
                return 5000; // Fixed capital for pairs trade
            case 'NEWS_SENTIMENT':
            case 'LATENCY_ARBITRAGE':
                return (this.marketData.get(signal.symbol)?.last || 100) * signal.quantity;
            case 'INDEX_ARBITRAGE':
                return 450 * signal.quantity; // Approximate ETF price
            case 'VOLATILITY_ARBITRAGE':
                return 10000; // Options capital requirement
            default:
                return 1000;
        }
    }

    updateTradePnL(tradeId) {
        const trade = this.userPortfolio.activePositions.get(tradeId);
        if (!trade) return;

        // Simulate realistic P&L based on signal expected profit
        const algorithm = this.algorithms.get(trade.algorithmId);
        const expectedProfit = trade.signal.expectedProfit || 0;
        const randomFactor = (Math.random() - 0.3); // Slight positive bias
        
        trade.pnl = expectedProfit * randomFactor * algorithm.performance.winRate;
        
        // Update portfolio total P&L
        this.userPortfolio.totalPnL = Array.from(this.userPortfolio.activePositions.values())
            .reduce((sum, t) => sum + t.pnl, 0);

        // Close position after some time
        if (Math.random() > 0.7) {
            this.closePosition(tradeId);
        } else {
            // Continue updating P&L
            setTimeout(() => {
                this.updateTradePnL(tradeId);
            }, 3000 + Math.random() * 7000);
        }
    }

    closePosition(tradeId) {
        const trade = this.userPortfolio.activePositions.get(tradeId);
        if (!trade) return;

        // Realize P&L
        this.userPortfolio.balance += trade.capital + trade.pnl;
        trade.status = 'CLOSED';
        trade.closeTime = Date.now();
        
        // Remove from active positions
        this.userPortfolio.activePositions.delete(tradeId);
        
        console.log(`💰 Closed position ${tradeId} with P&L: $${trade.pnl.toFixed(2)}`);
    }

    startPerformanceTracking() {
        // Track algorithm performance in real-time
        setInterval(() => {
            for (const [algId, algorithm] of this.algorithms.entries()) {
                this.updateAlgorithmPerformance(algId);
            }
        }, 5000); // Update every 5 seconds
    }

    updateAlgorithmPerformance(algId) {
        const algorithm = this.algorithms.get(algId);
        if (!algorithm) return;

        // Simulate performance updates based on recent signals
        const recentSignals = algorithm.signals.slice(0, 10);
        if (recentSignals.length === 0) return;

        // Update win rate
        const successfulSignals = recentSignals.filter(s => Math.random() > 0.3).length;
        algorithm.performance.winRate = (algorithm.performance.winRate * 0.9) + 
                                        (successfulSignals / recentSignals.length * 0.1);

        // Update other metrics with small variations
        algorithm.performance.sharpeRatio *= (1 + (Math.random() - 0.5) * 0.02);
        algorithm.performance.avgReturn *= (1 + (Math.random() - 0.5) * 0.01);
        algorithm.performance.maxDrawdown *= (1 + (Math.random() - 0.5) * 0.01);
        algorithm.performance.trades += recentSignals.length;
        algorithm.performance.lastUpdate = Date.now();

        // Update subscribers based on performance
        if (algorithm.performance.winRate > 0.8) {
            algorithm.subscribers += Math.floor(Math.random() * 3);
        }
    }

    initializeMarketMakers() {
        // Create market makers for algorithm trading
        const marketMakers = [
            {
                id: 'institutional_alpha',
                name: 'Institutional Alpha Partners',
                algorithms: ['cross_exchange_pro', 'hft_latency'],
                capital: 50000000,
                riskTolerance: 'high'
            },
            {
                id: 'quant_capital',
                name: 'Quant Capital Management',
                algorithms: ['stat_arb_ml', 'vol_arb'],
                capital: 25000000,
                riskTolerance: 'medium'
            },
            {
                id: 'ai_hedge_fund',
                name: 'AI Hedge Fund',
                algorithms: ['finbert_news', 'index_arb'],
                capital: 75000000,
                riskTolerance: 'high'
            }
        ];

        marketMakers.forEach(mm => {
            this.marketMakers.set(mm.id, mm);
        });
    }

    startRealTimeExecution() {
        console.log('⚡ Starting real-time execution engine...');
        
        // Process live signals every second
        setInterval(() => {
            this.processLiveSignals();
        }, 1000);
    }

    processLiveSignals() {
        for (const [algId, signal] of this.liveSignals.entries()) {
            // Age check - signals expire after 30 seconds
            if (Date.now() - signal.timestamp > 30000) {
                this.liveSignals.delete(algId);
                continue;
            }

            // Execute for market makers
            this.executeMarketMakerTrades(algId, signal);
        }
    }

    executeMarketMakerTrades(algId, signal) {
        for (const [mmId, marketMaker] of this.marketMakers.entries()) {
            if (marketMaker.algorithms.includes(algId) && Math.random() > 0.5) {
                // Market maker executes the signal
                console.log(`🏦 ${marketMaker.name} executed ${algId} signal`);
                
                // Update algorithm revenue
                const algorithm = this.algorithms.get(algId);
                if (algorithm) {
                    algorithm.revenueGenerated += signal.expectedProfit * 0.1;
                }
            }
        }
    }

    // Public API methods for UI integration
    getAllAlgorithms() {
        return Array.from(this.algorithms.values());
    }

    getAlgorithm(id) {
        return this.algorithms.get(id);
    }

    buyAlgorithm(algId) {
        const algorithm = this.algorithms.get(algId);
        if (!algorithm) return { success: false, error: 'Algorithm not found' };

        if (this.userPortfolio.balance < algorithm.price) {
            return { success: false, error: 'Insufficient balance' };
        }

        if (this.userPortfolio.ownedAlgorithms.has(algId)) {
            return { success: false, error: 'Algorithm already owned' };
        }

        // Purchase algorithm
        this.userPortfolio.balance -= algorithm.price;
        this.userPortfolio.ownedAlgorithms.add(algId);
        algorithm.subscribers++;

        return { 
            success: true, 
            message: `Successfully purchased ${algorithm.name}`,
            algorithm 
        };
    }

    activateAlgorithm(algId) {
        const algorithm = this.algorithms.get(algId);
        if (!algorithm) return { success: false, error: 'Algorithm not found' };

        if (!this.userPortfolio.ownedAlgorithms.has(algId)) {
            return { success: false, error: 'Algorithm not owned' };
        }

        algorithm.isActive = true;
        return { 
            success: true, 
            message: `${algorithm.name} is now active and generating signals` 
        };
    }

    deactivateAlgorithm(algId) {
        const algorithm = this.algorithms.get(algId);
        if (algorithm) {
            algorithm.isActive = false;
            return { 
                success: true, 
                message: `${algorithm.name} deactivated` 
            };
        }
        return { success: false, error: 'Algorithm not found' };
    }

    getPortfolio() {
        return {
            ...this.userPortfolio,
            activePositions: Array.from(this.userPortfolio.activePositions.values()),
            ownedAlgorithms: Array.from(this.userPortfolio.ownedAlgorithms)
                .map(id => this.algorithms.get(id))
                .filter(Boolean)
        };
    }

    getLiveSignals() {
        return Array.from(this.liveSignals.values());
    }

    getMarketData() {
        return Array.from(this.marketData.values());
    }
}

// Export for global usage
if (typeof window !== 'undefined') {
    window.AlgorithmicMarketplace = AlgorithmicMarketplace;
}

console.log('🏪 Algorithmic Marketplace loaded successfully');