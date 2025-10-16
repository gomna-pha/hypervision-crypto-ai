/**
 * Cocoa Trading AI - Advanced Portfolio Performance Metrics Interface
 * Professional portfolio analytics and performance tracking
 * 
 * Features:
 * - Real-time portfolio performance tracking
 * - Advanced metrics (Sharpe, Sortino, Alpha, Beta)
 * - Risk analytics and drawdown analysis
 * - Performance attribution and benchmarking
 */

class PortfolioPerformanceMetrics {
    constructor() {
        this.portfolioData = {};
        this.performanceMetrics = {};
        this.benchmarks = {};
        this.charts = {};
        this.isInitialized = false;
        
        this.init();
    }

    async init() {
        try {
            console.log('📈 Initializing Portfolio Performance Metrics...');
            
            await this.loadPortfolioData();
            this.calculatePerformanceMetrics();
            this.createMetricsInterface();
            this.setupRealTimeTracking();
            this.initializeCharts();
            
            this.isInitialized = true;
            console.log('✅ Portfolio Performance Metrics initialized successfully');
        } catch (error) {
            console.error('❌ Error initializing portfolio metrics:', error);
        }
    }

    async loadPortfolioData() {
        // Simulate loading comprehensive portfolio data
        this.portfolioData = {
            totalValue: 1250000,
            startingValue: 1000000,
            dailyReturn: 0.0234,
            totalReturn: 0.25,
            positions: [
                { symbol: 'BTC', value: 450000, allocation: 36, pnl: 125000, pnlPercent: 38.46 },
                { symbol: 'ETH', value: 320000, allocation: 25.6, pnl: 85000, pnlPercent: 36.17 },
                { symbol: 'BNB', value: 180000, allocation: 14.4, pnl: 23000, pnlPercent: 14.67 },
                { symbol: 'SOL', value: 150000, allocation: 12, pnl: 35000, pnlPercent: 30.43 },
                { symbol: 'ADA', value: 100000, allocation: 8, pnl: 12000, pnlPercent: 13.64 },
                { symbol: 'DOT', value: 50000, allocation: 4, pnl: -5000, pnlPercent: -9.09 }
            ],
            performanceHistory: this.generatePerformanceHistory(),
            riskMetrics: {
                volatility: 0.285,
                maxDrawdown: 0.082,
                varDaily: 0.0385,
                beta: 1.15,
                correlation: 0.78
            }
        };

        this.benchmarks = {
            btc: { name: 'Bitcoin', return: 0.195, volatility: 0.412 },
            sp500: { name: 'S&P 500', return: 0.108, volatility: 0.158 },
            crypto_index: { name: 'Crypto Index', return: 0.234, volatility: 0.356 }
        };

        console.log('✅ Portfolio data loaded');
    }

    generatePerformanceHistory() {
        // Generate 30 days of performance history
        const history = [];
        let value = 1000000;
        const now = new Date();
        
        for (let i = 30; i >= 0; i--) {
            const date = new Date(now.getTime() - i * 24 * 60 * 60 * 1000);
            const dailyReturn = (Math.random() - 0.45) * 0.05; // Slight positive bias
            value *= (1 + dailyReturn);
            
            history.push({
                date: date.toISOString().split('T')[0],
                value: value,
                return: dailyReturn,
                drawdown: Math.max(0, (Math.max(...history.map(h => h?.value || value)) - value) / Math.max(...history.map(h => h?.value || value)))
            });
        }
        
        return history;
    }

    calculatePerformanceMetrics() {
        const returns = this.portfolioData.performanceHistory.map(h => h.return);
        const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const returnStdDev = Math.sqrt(returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length);
        
        // Risk-free rate (assume 3% annually / 365 days)
        const riskFreeRate = 0.03 / 365;
        
        this.performanceMetrics = {
            totalReturn: this.portfolioData.totalReturn,
            annualizedReturn: Math.pow(1 + this.portfolioData.totalReturn, 365 / 30) - 1, // Annualized from 30 days
            volatility: returnStdDev * Math.sqrt(365), // Annualized volatility
            sharpeRatio: (avgReturn - riskFreeRate) / returnStdDev * Math.sqrt(365),
            sortinoRatio: this.calculateSortinoRatio(returns, riskFreeRate),
            maxDrawdown: Math.max(...this.portfolioData.performanceHistory.map(h => h.drawdown)),
            calmarRatio: (avgReturn * 365) / Math.max(...this.portfolioData.performanceHistory.map(h => h.drawdown)),
            winRate: returns.filter(r => r > 0).length / returns.length,
            profitFactor: this.calculateProfitFactor(returns),
            alpha: this.calculateAlpha(avgReturn, this.portfolioData.riskMetrics.beta, this.benchmarks.crypto_index.return / 365),
            beta: this.portfolioData.riskMetrics.beta,
            informationRatio: this.calculateInformationRatio(returns),
            treynorRatio: (avgReturn - riskFreeRate) / this.portfolioData.riskMetrics.beta,
            upturnCapture: this.calculateUpturnCapture(returns),
            downturnCapture: this.calculateDownturnCapture(returns)
        };

        console.log('✅ Performance metrics calculated');
    }

    calculateSortinoRatio(returns, riskFreeRate) {
        const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
        const downsideReturns = returns.filter(r => r < riskFreeRate);
        const downsideStdDev = Math.sqrt(downsideReturns.reduce((sum, r) => sum + Math.pow(r - riskFreeRate, 2), 0) / downsideReturns.length);
        return (avgReturn - riskFreeRate) / downsideStdDev * Math.sqrt(365);
    }

    calculateProfitFactor(returns) {
        const profits = returns.filter(r => r > 0).reduce((sum, r) => sum + r, 0);
        const losses = Math.abs(returns.filter(r => r < 0).reduce((sum, r) => sum + r, 0));
        return losses === 0 ? Infinity : profits / losses;
    }

    calculateAlpha(avgReturn, beta, marketReturn) {
        return (avgReturn - (0.03 / 365 + beta * (marketReturn - 0.03 / 365))) * 365;
    }

    calculateInformationRatio(returns) {
        const benchmarkReturns = Array(returns.length).fill(this.benchmarks.crypto_index.return / 365);
        const excessReturns = returns.map((r, i) => r - benchmarkReturns[i]);
        const avgExcessReturn = excessReturns.reduce((sum, r) => sum + r, 0) / excessReturns.length;
        const trackingError = Math.sqrt(excessReturns.reduce((sum, r) => sum + Math.pow(r - avgExcessReturn, 2), 0) / excessReturns.length);
        return avgExcessReturn / trackingError * Math.sqrt(365);
    }

    calculateUpturnCapture(returns) {
        const benchmarkReturn = this.benchmarks.crypto_index.return / 365;
        const upReturns = returns.filter(r => r > 0);
        const avgUpReturn = upReturns.reduce((sum, r) => sum + r, 0) / upReturns.length;
        return avgUpReturn / benchmarkReturn;
    }

    calculateDownturnCapture(returns) {
        const benchmarkReturn = this.benchmarks.crypto_index.return / 365;
        const downReturns = returns.filter(r => r < 0);
        const avgDownReturn = downReturns.reduce((sum, r) => sum + r, 0) / downReturns.length;
        return Math.abs(avgDownReturn) / Math.abs(benchmarkReturn);
    }

    createMetricsInterface() {
        const metricsContainer = document.createElement('div');
        metricsContainer.id = 'portfolio-performance-metrics';
        metricsContainer.className = 'cocoa-panel cocoa-fade-in';
        
        metricsContainer.innerHTML = `
            <div class="cocoa-panel-header">
                <h3>📈 Portfolio Performance Analytics</h3>
                <div class="performance-status">
                    <div class="performance-indicator positive">
                        <span class="status-dot"></span>
                        <span>+${(this.portfolioData.dailyReturn * 100).toFixed(2)}% Today</span>
                    </div>
                </div>
            </div>
            <div class="cocoa-panel-content">
                <div class="metrics-layout">
                    <div class="portfolio-overview-section">
                        ${this.createPortfolioOverview()}
                    </div>
                    <div class="key-metrics-section">
                        ${this.createKeyMetrics()}
                    </div>
                    <div class="risk-metrics-section">
                        ${this.createRiskMetrics()}
                    </div>
                    <div class="performance-attribution-section">
                        ${this.createPerformanceAttribution()}
                    </div>
                    <div class="benchmark-comparison-section">
                        ${this.createBenchmarkComparison()}
                    </div>
                    <div class="portfolio-composition-section">
                        ${this.createPortfolioComposition()}
                    </div>
                </div>
            </div>
        `;

        // Add metrics-specific styles
        const metricsStyles = document.createElement('style');
        metricsStyles.textContent = `
            .portfolio-performance-metrics {
                margin: 20px 0;
            }

            .performance-status {
                display: flex;
                align-items: center;
                gap: 15px;
            }

            .performance-indicator {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 6px 12px;
                border-radius: 20px;
                font-weight: 600;
                font-size: 0.9rem;
            }

            .performance-indicator.positive {
                background: rgba(16, 185, 129, 0.2);
                color: var(--cocoa-success);
                border: 1px solid var(--cocoa-success);
            }

            .performance-indicator.negative {
                background: rgba(239, 68, 68, 0.2);
                color: var(--cocoa-error);
                border: 1px solid var(--cocoa-error);
            }

            .performance-indicator .status-dot {
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: currentColor;
            }

            .metrics-layout {
                display: grid;
                grid-template-columns: 1fr 1fr;
                grid-template-rows: auto auto auto;
                gap: 20px;
                margin-top: 20px;
            }

            .portfolio-overview-section {
                grid-column: 1 / -1;
            }

            .key-metrics-section,
            .risk-metrics-section,
            .performance-attribution-section,
            .benchmark-comparison-section,
            .portfolio-composition-section {
                background: rgba(26, 26, 26, 0.6);
                border: 1px solid rgba(212, 165, 116, 0.2);
                border-radius: 12px;
                padding: 20px;
            }

            /* Portfolio Overview */
            .portfolio-overview-cards {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 15px;
            }

            .overview-card {
                background: rgba(139, 69, 19, 0.1);
                border: 1px solid rgba(139, 69, 19, 0.2);
                border-radius: 12px;
                padding: 20px;
                text-align: center;
                transition: all 0.3s ease;
            }

            .overview-card:hover {
                border-color: var(--cocoa-accent);
                transform: translateY(-2px);
            }

            .overview-value {
                font-size: 2rem;
                font-weight: 700;
                color: var(--cocoa-accent);
                margin: 10px 0;
            }

            .overview-label {
                color: var(--cocoa-text);
                font-size: 0.9rem;
                opacity: 0.8;
            }

            .overview-change {
                font-weight: 600;
                font-size: 1rem;
                margin-top: 8px;
            }

            .overview-change.positive {
                color: var(--cocoa-success);
            }

            .overview-change.negative {
                color: var(--cocoa-error);
            }

            /* Metrics Grid */
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                gap: 15px;
            }

            .metric-item {
                text-align: center;
                padding: 15px 10px;
                background: rgba(139, 69, 19, 0.05);
                border-radius: 8px;
                border: 1px solid rgba(139, 69, 19, 0.1);
            }

            .metric-value {
                font-size: 1.5rem;
                font-weight: 700;
                color: var(--cocoa-accent);
                margin-bottom: 5px;
            }

            .metric-label {
                color: var(--cocoa-text);
                font-size: 0.85rem;
                opacity: 0.8;
            }

            .metric-description {
                color: var(--cocoa-text);
                font-size: 0.75rem;
                opacity: 0.6;
                margin-top: 3px;
                line-height: 1.2;
            }

            /* Risk Metrics Specific */
            .risk-indicator {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 10px 15px;
                margin: 8px 0;
                background: rgba(139, 69, 19, 0.05);
                border-radius: 8px;
                border-left: 4px solid var(--risk-color, var(--cocoa-secondary));
            }

            .risk-indicator.low {
                --risk-color: var(--cocoa-success);
            }

            .risk-indicator.medium {
                --risk-color: var(--cocoa-warning);
            }

            .risk-indicator.high {
                --risk-color: var(--cocoa-error);
            }

            .risk-label {
                font-weight: 600;
                color: var(--cocoa-text);
            }

            .risk-value {
                font-weight: 700;
                color: var(--risk-color, var(--cocoa-accent));
            }

            /* Performance Attribution */
            .attribution-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 12px 0;
                border-bottom: 1px solid rgba(212, 165, 116, 0.1);
            }

            .attribution-item:last-child {
                border-bottom: none;
            }

            .attribution-symbol {
                font-weight: 600;
                color: var(--cocoa-text);
                display: flex;
                align-items: center;
                gap: 10px;
            }

            .attribution-contribution {
                font-weight: 600;
                text-align: right;
            }

            .attribution-contribution.positive {
                color: var(--cocoa-success);
            }

            .attribution-contribution.negative {
                color: var(--cocoa-error);
            }

            /* Benchmark Comparison */
            .benchmark-item {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding: 15px;
                margin: 10px 0;
                background: rgba(139, 69, 19, 0.05);
                border-radius: 10px;
                border: 1px solid rgba(139, 69, 19, 0.1);
            }

            .benchmark-name {
                font-weight: 600;
                color: var(--cocoa-text);
            }

            .benchmark-comparison {
                text-align: right;
            }

            .benchmark-return {
                font-weight: 600;
                color: var(--cocoa-secondary);
                margin-bottom: 3px;
            }

            .benchmark-outperformance {
                font-size: 0.9rem;
                font-weight: 600;
            }

            .benchmark-outperformance.positive {
                color: var(--cocoa-success);
            }

            .benchmark-outperformance.negative {
                color: var(--cocoa-error);
            }

            /* Portfolio Composition */
            .composition-item {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 12px 0;
                border-bottom: 1px solid rgba(212, 165, 116, 0.1);
            }

            .composition-item:last-child {
                border-bottom: none;
            }

            .composition-asset {
                display: flex;
                align-items: center;
                gap: 15px;
                flex: 1;
            }

            .asset-symbol {
                font-weight: 600;
                color: var(--cocoa-text);
                width: 50px;
            }

            .asset-bar-container {
                flex: 1;
                height: 20px;
                background: rgba(139, 69, 19, 0.1);
                border-radius: 10px;
                overflow: hidden;
                position: relative;
            }

            .asset-bar {
                height: 100%;
                background: linear-gradient(90deg, var(--cocoa-primary), var(--cocoa-secondary));
                border-radius: 10px;
                transition: width 0.6s ease;
            }

            .asset-allocation {
                position: absolute;
                right: 8px;
                top: 50%;
                transform: translateY(-50%);
                font-size: 0.8rem;
                font-weight: 600;
                color: var(--cocoa-text);
            }

            .composition-values {
                text-align: right;
                margin-left: 15px;
            }

            .asset-value {
                font-weight: 600;
                color: var(--cocoa-accent);
                margin-bottom: 2px;
            }

            .asset-pnl {
                font-size: 0.85rem;
                font-weight: 600;
            }

            .asset-pnl.positive {
                color: var(--cocoa-success);
            }

            .asset-pnl.negative {
                color: var(--cocoa-error);
            }

            /* Section Headers */
            .section-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid rgba(139, 69, 19, 0.3);
            }

            .section-title {
                color: var(--cocoa-secondary);
                font-weight: 600;
                font-size: 1.1rem;
                display: flex;
                align-items: center;
                gap: 8px;
            }

            /* Responsive Design */
            @media (max-width: 1024px) {
                .metrics-layout {
                    grid-template-columns: 1fr;
                }
            }

            @media (max-width: 768px) {
                .portfolio-overview-cards {
                    grid-template-columns: 1fr 1fr;
                }
                
                .metrics-grid {
                    grid-template-columns: 1fr 1fr;
                }
                
                .composition-asset {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 8px;
                }
                
                .asset-bar-container {
                    width: 100%;
                }
            }
        `;
        
        document.head.appendChild(metricsStyles);

        // Insert into the main content area
        const targetContainer = document.querySelector('.container, .main-content') || document.body;
        const marketplaceTab = document.querySelector('#marketplace-tab') || targetContainer.querySelector('[data-tab="marketplace"]');
        
        if (marketplaceTab) {
            marketplaceTab.appendChild(metricsContainer);
        } else {
            targetContainer.appendChild(metricsContainer);
        }

        console.log('✅ Portfolio performance metrics interface created');
    }

    createPortfolioOverview() {
        const totalPnL = this.portfolioData.totalValue - this.portfolioData.startingValue;
        const totalPnLPercent = (totalPnL / this.portfolioData.startingValue) * 100;
        
        return `
            <div class="section-header">
                <div class="section-title">
                    💼 Portfolio Overview
                </div>
            </div>
            <div class="portfolio-overview-cards">
                <div class="overview-card">
                    <div class="overview-label">Total Portfolio Value</div>
                    <div class="overview-value">$${this.portfolioData.totalValue.toLocaleString()}</div>
                    <div class="overview-change ${totalPnLPercent >= 0 ? 'positive' : 'negative'}">
                        ${totalPnLPercent >= 0 ? '+' : ''}$${totalPnL.toLocaleString()} (${totalPnLPercent.toFixed(2)}%)
                    </div>
                </div>
                <div class="overview-card">
                    <div class="overview-label">Daily Return</div>
                    <div class="overview-value">${(this.portfolioData.dailyReturn * 100).toFixed(2)}%</div>
                    <div class="overview-change ${this.portfolioData.dailyReturn >= 0 ? 'positive' : 'negative'}">
                        ${this.portfolioData.dailyReturn >= 0 ? '+' : ''}$${(this.portfolioData.totalValue * this.portfolioData.dailyReturn).toLocaleString()}
                    </div>
                </div>
                <div class="overview-card">
                    <div class="overview-label">Annualized Return</div>
                    <div class="overview-value">${(this.performanceMetrics.annualizedReturn * 100).toFixed(1)}%</div>
                    <div class="overview-change positive">vs ${(this.benchmarks.crypto_index.return * 100).toFixed(1)}% benchmark</div>
                </div>
                <div class="overview-card">
                    <div class="overview-label">Sharpe Ratio</div>
                    <div class="overview-value">${this.performanceMetrics.sharpeRatio.toFixed(2)}</div>
                    <div class="overview-change ${this.performanceMetrics.sharpeRatio > 1 ? 'positive' : 'negative'}">
                        ${this.performanceMetrics.sharpeRatio > 1 ? 'Excellent' : this.performanceMetrics.sharpeRatio > 0.5 ? 'Good' : 'Poor'}
                    </div>
                </div>
            </div>
        `;
    }

    createKeyMetrics() {
        return `
            <div class="section-header">
                <div class="section-title">📊 Key Performance Metrics</div>
            </div>
            <div class="metrics-grid">
                <div class="metric-item">
                    <div class="metric-value">${(this.performanceMetrics.totalReturn * 100).toFixed(1)}%</div>
                    <div class="metric-label">Total Return</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">${(this.performanceMetrics.volatility * 100).toFixed(1)}%</div>
                    <div class="metric-label">Volatility</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">${this.performanceMetrics.sortinoRatio.toFixed(2)}</div>
                    <div class="metric-label">Sortino Ratio</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">${(this.performanceMetrics.maxDrawdown * 100).toFixed(1)}%</div>
                    <div class="metric-label">Max Drawdown</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">${(this.performanceMetrics.winRate * 100).toFixed(0)}%</div>
                    <div class="metric-label">Win Rate</div>
                </div>
                <div class="metric-item">
                    <div class="metric-value">${this.performanceMetrics.calmarRatio.toFixed(2)}</div>
                    <div class="metric-label">Calmar Ratio</div>
                </div>
            </div>
        `;
    }

    createRiskMetrics() {
        return `
            <div class="section-header">
                <div class="section-title">⚠️ Risk Analysis</div>
            </div>
            <div class="risk-metrics-content">
                <div class="risk-indicator ${this.performanceMetrics.volatility < 0.2 ? 'low' : this.performanceMetrics.volatility < 0.4 ? 'medium' : 'high'}">
                    <span class="risk-label">Portfolio Volatility</span>
                    <span class="risk-value">${(this.performanceMetrics.volatility * 100).toFixed(1)}%</span>
                </div>
                <div class="risk-indicator ${this.performanceMetrics.maxDrawdown < 0.05 ? 'low' : this.performanceMetrics.maxDrawdown < 0.15 ? 'medium' : 'high'}">
                    <span class="risk-label">Maximum Drawdown</span>
                    <span class="risk-value">${(this.performanceMetrics.maxDrawdown * 100).toFixed(1)}%</span>
                </div>
                <div class="risk-indicator ${Math.abs(this.performanceMetrics.beta - 1) < 0.2 ? 'low' : Math.abs(this.performanceMetrics.beta - 1) < 0.5 ? 'medium' : 'high'}">
                    <span class="risk-label">Beta (vs Crypto Market)</span>
                    <span class="risk-value">${this.performanceMetrics.beta.toFixed(2)}</span>
                </div>
                <div class="risk-indicator ${this.portfolioData.riskMetrics.varDaily < 0.03 ? 'low' : this.portfolioData.riskMetrics.varDaily < 0.05 ? 'medium' : 'high'}">
                    <span class="risk-label">Daily VaR (95%)</span>
                    <span class="risk-value">${(this.portfolioData.riskMetrics.varDaily * 100).toFixed(1)}%</span>
                </div>
            </div>
        `;
    }

    createPerformanceAttribution() {
        return `
            <div class="section-header">
                <div class="section-title">🎯 Performance Attribution</div>
            </div>
            <div class="performance-attribution-content">
                ${this.portfolioData.positions.map(position => {
                    const contribution = (position.pnl / (this.portfolioData.totalValue - (this.portfolioData.totalValue - this.portfolioData.startingValue))) * 100;
                    return `
                        <div class="attribution-item">
                            <div class="attribution-symbol">
                                <span>${position.symbol}</span>
                                <span style="font-size: 0.8rem; opacity: 0.7;">${position.allocation}%</span>
                            </div>
                            <div class="attribution-contribution ${contribution >= 0 ? 'positive' : 'negative'}">
                                ${contribution >= 0 ? '+' : ''}${contribution.toFixed(1)}%
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    createBenchmarkComparison() {
        return `
            <div class="section-header">
                <div class="section-title">📈 Benchmark Comparison</div>
            </div>
            <div class="benchmark-comparison-content">
                ${Object.entries(this.benchmarks).map(([key, benchmark]) => {
                    const outperformance = this.performanceMetrics.annualizedReturn - benchmark.return;
                    return `
                        <div class="benchmark-item">
                            <div class="benchmark-name">${benchmark.name}</div>
                            <div class="benchmark-comparison">
                                <div class="benchmark-return">${(benchmark.return * 100).toFixed(1)}%</div>
                                <div class="benchmark-outperformance ${outperformance >= 0 ? 'positive' : 'negative'}">
                                    ${outperformance >= 0 ? '+' : ''}${(outperformance * 100).toFixed(1)}% vs benchmark
                                </div>
                            </div>
                        </div>
                    `;
                }).join('')}
            </div>
        `;
    }

    createPortfolioComposition() {
        return `
            <div class="section-header">
                <div class="section-title">🥧 Portfolio Composition</div>
            </div>
            <div class="portfolio-composition-content">
                ${this.portfolioData.positions.map(position => `
                    <div class="composition-item">
                        <div class="composition-asset">
                            <div class="asset-symbol">${position.symbol}</div>
                            <div class="asset-bar-container">
                                <div class="asset-bar" style="width: ${position.allocation}%"></div>
                                <div class="asset-allocation">${position.allocation}%</div>
                            </div>
                        </div>
                        <div class="composition-values">
                            <div class="asset-value">$${position.value.toLocaleString()}</div>
                            <div class="asset-pnl ${position.pnl >= 0 ? 'positive' : 'negative'}">
                                ${position.pnl >= 0 ? '+' : ''}${position.pnlPercent.toFixed(1)}%
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;
    }

    setupRealTimeTracking() {
        // Set up real-time portfolio tracking
        setInterval(() => {
            this.updatePortfolioMetrics();
        }, 10000); // Update every 10 seconds

        console.log('✅ Real-time portfolio tracking setup');
    }

    updatePortfolioMetrics() {
        // Simulate real-time portfolio updates
        const marketMovement = (Math.random() - 0.5) * 0.02; // ±1% movement
        
        // Update portfolio value
        this.portfolioData.totalValue *= (1 + marketMovement);
        this.portfolioData.dailyReturn = marketMovement;
        
        // Update individual positions
        this.portfolioData.positions.forEach(position => {
            const individualMovement = marketMovement + (Math.random() - 0.5) * 0.01;
            position.value *= (1 + individualMovement);
            position.pnl = position.value * (position.pnlPercent / 100);
        });

        // Recalculate metrics
        this.calculatePerformanceMetrics();
        
        // Update UI elements
        this.updatePortfolioDisplay();
    }

    updatePortfolioDisplay() {
        // Update key display elements
        const totalValueElement = document.querySelector('.overview-card .overview-value');
        if (totalValueElement) {
            totalValueElement.textContent = `$${this.portfolioData.totalValue.toLocaleString()}`;
        }

        const dailyReturnElement = document.querySelector('.overview-card:nth-child(2) .overview-value');
        if (dailyReturnElement) {
            dailyReturnElement.textContent = `${(this.portfolioData.dailyReturn * 100).toFixed(2)}%`;
        }

        // Update performance indicator in header
        const performanceIndicator = document.querySelector('.performance-indicator');
        if (performanceIndicator) {
            const isPositive = this.portfolioData.dailyReturn >= 0;
            performanceIndicator.className = `performance-indicator ${isPositive ? 'positive' : 'negative'}`;
            performanceIndicator.innerHTML = `
                <span class="status-dot"></span>
                <span>${isPositive ? '+' : ''}${(this.portfolioData.dailyReturn * 100).toFixed(2)}% Today</span>
            `;
        }
    }

    initializeCharts() {
        // Initialize charts for portfolio visualization
        if (typeof Chart !== 'undefined') {
            this.createPerformanceChart();
            this.createAllocationChart();
        } else {
            console.warn('Chart.js not loaded, skipping chart initialization');
        }
    }

    createPerformanceChart() {
        // Performance history chart would be implemented here
        console.log('📊 Performance chart initialization');
    }

    createAllocationChart() {
        // Asset allocation pie chart would be implemented here
        console.log('🥧 Allocation chart initialization');
    }

    // Public API methods
    getPortfolioMetrics() {
        return {
            portfolio: this.portfolioData,
            metrics: this.performanceMetrics,
            benchmarks: this.benchmarks
        };
    }

    exportPerformanceReport() {
        const report = {
            generatedAt: new Date().toISOString(),
            portfolio: this.portfolioData,
            metrics: this.performanceMetrics,
            benchmarks: this.benchmarks
        };
        
        const dataStr = JSON.stringify(report, null, 2);
        const dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);
        
        const exportFileDefaultName = `portfolio_report_${new Date().toISOString().split('T')[0]}.json`;
        
        const linkElement = document.createElement('a');
        linkElement.setAttribute('href', dataUri);
        linkElement.setAttribute('download', exportFileDefaultName);
        linkElement.click();
        
        console.log('📄 Performance report exported');
    }

    addBenchmark(name, data) {
        this.benchmarks[name.toLowerCase().replace(/\s+/g, '_')] = data;
        console.log(`✅ Added benchmark: ${name}`);
    }
}

// Auto-initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.portfolioPerformanceMetrics = new PortfolioPerformanceMetrics();
    });
} else {
    window.portfolioPerformanceMetrics = new PortfolioPerformanceMetrics();
}

// Export for module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = PortfolioPerformanceMetrics;
}

console.log('📈 Portfolio Performance Metrics loaded and ready for advanced analytics');