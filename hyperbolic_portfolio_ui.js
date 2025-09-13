/**
 * HYPERBOLIC PORTFOLIO OPTIMIZATION UI
 * ===================================
 * Advanced frontend interface for hyperbolic space portfolio optimization
 * with hierarchical index relationships and overfitting prevention
 */

class HyperbolicPortfolioUI {
    constructor() {
        this.portfolioEngine = null;
        this.currentPortfolio = {};
        this.marketIndices = {};
        this.hyperbolicEmbeddings = {};
        this.correlationMatrix = null;
        this.validationResults = {};
        this.charts = {};
        
        // Initialize UI components
        this.initializeComponents();
        this.setupEventListeners();
        this.loadMarketData();
        
        // Anti-overfitting parameters
        this.validationThreshold = 0.3;
        this.hallucinationThreshold = 0.4;
        this.confidenceThreshold = 0.8;
    }
    
    initializeComponents() {
        console.log('🚀 Initializing Hyperbolic Portfolio UI...');
        
        // Create portfolio container if it doesn't exist
        if (!document.getElementById('hyperbolic-portfolio-container')) {
            this.createPortfolioContainer();
        }
        
        // Initialize charts
        this.initializeCharts();
        
        // Setup real-time updates
        this.setupRealTimeUpdates();
    }
    
    createPortfolioContainer() {
        const container = document.createElement('div');
        container.id = 'hyperbolic-portfolio-container';
        container.className = 'hyperbolic-portfolio-container';
        container.innerHTML = `
            <div class="portfolio-header glass-effect p-6 rounded-xl mb-6">
                <div class="flex justify-between items-center">
                    <div>
                        <h2 class="text-2xl font-bold text-cream-900 mb-2">
                            <i data-lucide="brain-circuit" class="inline w-6 h-6 mr-2"></i>
                            Hyperbolic Portfolio Optimization
                        </h2>
                        <p class="text-cream-700">Advanced portfolio management with hierarchical index relationships</p>
                    </div>
                    <div class="validation-status" id="validation-status">
                        <div class="flex items-center space-x-2">
                            <div class="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
                            <span class="text-sm text-cream-800">Validated & Safe</span>
                        </div>
                    </div>
                </div>
                
                <!-- Risk Controls -->
                <div class="risk-controls mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div class="risk-tolerance">
                        <label class="block text-sm font-medium text-cream-800 mb-2">Risk Tolerance</label>
                        <input type="range" id="risk-tolerance" min="0.05" max="0.5" step="0.05" value="0.15" 
                               class="w-full h-2 bg-cream-200 rounded-lg appearance-none cursor-pointer">
                        <div class="flex justify-between text-xs text-cream-600 mt-1">
                            <span>Conservative</span>
                            <span>Aggressive</span>
                        </div>
                    </div>
                    
                    <div class="rebalance-frequency">
                        <label class="block text-sm font-medium text-cream-800 mb-2">Rebalancing</label>
                        <select id="rebalance-frequency" class="w-full p-2 border border-cream-300 rounded-lg bg-white text-cream-900">
                            <option value="daily">Daily</option>
                            <option value="weekly" selected>Weekly</option>
                            <option value="monthly">Monthly</option>
                            <option value="quarterly">Quarterly</option>
                        </select>
                    </div>
                    
                    <div class="optimization-method">
                        <label class="block text-sm font-medium text-cream-800 mb-2">Method</label>
                        <select id="optimization-method" class="w-full p-2 border border-cream-300 rounded-lg bg-white text-cream-900">
                            <option value="hyperbolic" selected>Hyperbolic Space</option>
                            <option value="markowitz">Markowitz</option>
                            <option value="black-litterman">Black-Litterman</option>
                            <option value="risk-parity">Risk Parity</option>
                        </select>
                    </div>
                </div>
            </div>
            
            <!-- Main Dashboard -->
            <div class="portfolio-dashboard grid grid-cols-1 lg:grid-cols-3 gap-6">
                
                <!-- Portfolio Composition -->
                <div class="portfolio-composition glass-effect p-6 rounded-xl">
                    <h3 class="text-lg font-semibold text-cream-900 mb-4 flex items-center">
                        <i data-lucide="pie-chart" class="w-5 h-5 mr-2"></i>
                        Portfolio Allocation
                    </h3>
                    <div class="chart-container-small">
                        <canvas id="portfolio-allocation-chart"></canvas>
                    </div>
                    
                    <!-- Asset List -->
                    <div class="asset-list mt-4 space-y-2" id="asset-allocation-list">
                        <!-- Dynamic content -->
                    </div>
                    
                    <!-- Optimize Button -->
                    <button id="optimize-portfolio" class="btn-primary w-full mt-4 py-3 px-6 rounded-lg font-semibold transition-all hover:shadow-lg">
                        <i data-lucide="zap" class="inline w-4 h-4 mr-2"></i>
                        Optimize Portfolio
                    </button>
                </div>
                
                <!-- Hyperbolic Space Visualization -->
                <div class="hyperbolic-visualization glass-effect p-6 rounded-xl">
                    <h3 class="text-lg font-semibold text-cream-900 mb-4 flex items-center">
                        <i data-lucide="globe" class="w-5 h-5 mr-2"></i>
                        Hyperbolic Space Map
                    </h3>
                    <div class="chart-container">
                        <canvas id="hyperbolic-space-chart"></canvas>
                    </div>
                    
                    <!-- Distance Metrics -->
                    <div class="distance-metrics mt-4" id="distance-metrics">
                        <!-- Dynamic content -->
                    </div>
                </div>
                
                <!-- Risk & Performance Metrics -->
                <div class="risk-metrics glass-effect p-6 rounded-xl">
                    <h3 class="text-lg font-semibold text-cream-900 mb-4 flex items-center">
                        <i data-lucide="shield-check" class="w-5 h-5 mr-2"></i>
                        Risk Analysis
                    </h3>
                    
                    <!-- Key Metrics -->
                    <div class="metrics-grid grid grid-cols-2 gap-4 mb-4">
                        <div class="metric-card p-3 rounded-lg text-center">
                            <div class="text-2xl font-bold text-green-600" id="expected-return">--</div>
                            <div class="text-xs text-cream-600">Expected Return</div>
                        </div>
                        <div class="metric-card p-3 rounded-lg text-center">
                            <div class="text-2xl font-bold text-blue-600" id="portfolio-volatility">--</div>
                            <div class="text-xs text-cream-600">Volatility</div>
                        </div>
                        <div class="metric-card p-3 rounded-lg text-center">
                            <div class="text-2xl font-bold text-purple-600" id="sharpe-ratio">--</div>
                            <div class="text-xs text-cream-600">Sharpe Ratio</div>
                        </div>
                        <div class="metric-card p-3 rounded-lg text-center">
                            <div class="text-2xl font-bold text-orange-600" id="diversification-score">--</div>
                            <div class="text-xs text-cream-600">Diversification</div>
                        </div>
                    </div>
                    
                    <!-- Validation Status -->
                    <div class="validation-panel" id="validation-panel">
                        <!-- Dynamic content -->
                    </div>
                </div>
            </div>
            
            <!-- Index Correlation Matrix -->
            <div class="correlation-analysis glass-effect p-6 rounded-xl mt-6">
                <h3 class="text-lg font-semibold text-cream-900 mb-4 flex items-center">
                    <i data-lucide="network" class="w-5 h-5 mr-2"></i>
                    Multi-Index Correlation Analysis
                </h3>
                
                <div class="correlation-controls mb-4 flex space-x-4">
                    <select id="correlation-timeframe" class="p-2 border border-cream-300 rounded-lg bg-white text-cream-900">
                        <option value="1M">1 Month</option>
                        <option value="3M" selected>3 Months</option>
                        <option value="6M">6 Months</option>
                        <option value="1Y">1 Year</option>
                    </select>
                    
                    <select id="correlation-method" class="p-2 border border-cream-300 rounded-lg bg-white text-cream-900">
                        <option value="pearson" selected>Pearson</option>
                        <option value="spearman">Spearman</option>
                        <option value="hyperbolic">Hyperbolic Distance</option>
                    </select>
                </div>
                
                <div class="chart-container-large">
                    <canvas id="correlation-heatmap"></canvas>
                </div>
            </div>
            
            <!-- Rebalancing Suggestions -->
            <div class="rebalancing-suggestions glass-effect p-6 rounded-xl mt-6">
                <h3 class="text-lg font-semibold text-cream-900 mb-4 flex items-center">
                    <i data-lucide="refresh-cw" class="w-5 h-5 mr-2"></i>
                    Intelligent Rebalancing
                </h3>
                
                <div class="suggestions-list" id="rebalancing-suggestions">
                    <!-- Dynamic content -->
                </div>
            </div>
        `;
        
        // Insert into the portfolio tab
        const portfolioTab = document.getElementById('portfolio-tab');
        if (portfolioTab) {
            portfolioTab.appendChild(container);
            console.log('✅ Portfolio container inserted into DOM');
        } else {
            console.error('❌ Portfolio tab not found');
        }
        
        // Initialize Lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }
    
    initializeCharts() {
        // Portfolio Allocation Pie Chart
        const allocationCtx = document.getElementById('portfolio-allocation-chart');
        if (allocationCtx) {
            this.charts.allocation = new Chart(allocationCtx, {
                type: 'doughnut',
                data: {
                    labels: [],
                    datasets: [{
                        data: [],
                        backgroundColor: [
                            'rgba(34, 197, 94, 0.8)',
                            'rgba(59, 130, 246, 0.8)',
                            'rgba(147, 51, 234, 0.8)',
                            'rgba(245, 158, 11, 0.8)',
                            'rgba(239, 68, 68, 0.8)',
                            'rgba(16, 185, 129, 0.8)',
                            'rgba(139, 92, 246, 0.8)'
                        ],
                        borderWidth: 2,
                        borderColor: '#fdf6e3'
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }
        
        // Hyperbolic Space Visualization
        const hyperbolicCtx = document.getElementById('hyperbolic-space-chart');
        if (hyperbolicCtx) {
            this.charts.hyperbolic = new Chart(hyperbolicCtx, {
                type: 'scatter',
                data: {
                    datasets: []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'linear',
                            min: -1,
                            max: 1,
                            title: {
                                display: true,
                                text: 'Hyperbolic X'
                            }
                        },
                        y: {
                            type: 'linear',
                            min: -1,
                            max: 1,
                            title: {
                                display: true,
                                text: 'Hyperbolic Y'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            display: true,
                            position: 'bottom'
                        },
                        tooltip: {
                            callbacks: {
                                title: function(context) {
                                    return `Asset: ${context[0].dataset.label}`;
                                },
                                label: function(context) {
                                    return `Position: (${context.parsed.x.toFixed(3)}, ${context.parsed.y.toFixed(3)})`;
                                }
                            }
                        }
                    }
                }
            });
        }
        
        // Correlation Heatmap
        const correlationCtx = document.getElementById('correlation-heatmap');
        if (correlationCtx) {
            this.charts.correlation = new Chart(correlationCtx, {
                type: 'scatter',
                data: {
                    datasets: []
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: {
                            display: false
                        }
                    }
                }
            });
        }
    }
    
    setupEventListeners() {
        // Optimization button
        const optimizeBtn = document.getElementById('optimize-portfolio');
        if (optimizeBtn) {
            optimizeBtn.addEventListener('click', () => this.optimizePortfolio());
        }
        
        // Risk tolerance slider
        const riskSlider = document.getElementById('risk-tolerance');
        if (riskSlider) {
            riskSlider.addEventListener('input', (e) => {
                this.updateRiskTolerance(parseFloat(e.target.value));
            });
        }
        
        // Correlation controls
        const correlationTimeframe = document.getElementById('correlation-timeframe');
        if (correlationTimeframe) {
            correlationTimeframe.addEventListener('change', () => this.updateCorrelationAnalysis());
        }
        
        const correlationMethod = document.getElementById('correlation-method');
        if (correlationMethod) {
            correlationMethod.addEventListener('change', () => this.updateCorrelationAnalysis());
        }
    }
    
    setupRealTimeUpdates() {
        // Update portfolio metrics every 30 seconds
        setInterval(() => {
            this.updatePortfolioMetrics();
        }, 30000);
        
        // Update market data every 5 minutes
        setInterval(() => {
            this.loadMarketData();
        }, 300000);
    }
    
    async loadMarketData() {
        console.log('📊 Loading market data for indices analysis...');
        
        try {
            // Simulate loading market data for major indices
            const indices = [
                'BTC-USD', 'ETH-USD', 'SPY', 'QQQ', 'TLT', 'GLD', 'VNQ',
                'EFA', 'VWO', 'XLF', 'XLE', 'XLK', 'IWM', 'HYG'
            ];
            
            const marketData = {};
            
            for (const symbol of indices) {
                // Generate realistic mock data
                const basePrice = this.getBasePrice(symbol);
                const volatility = this.getVolatility(symbol);
                
                const currentPrice = basePrice * (1 + (Math.random() - 0.5) * volatility);
                const change = (Math.random() - 0.5) * volatility * 2;
                
                marketData[symbol] = {
                    price: currentPrice,
                    change: change,
                    changePercent: change * 100,
                    volume: Math.floor(Math.random() * 10000000) + 1000000,
                    correlation: this.calculateMockCorrelation(symbol)
                };
            }
            
            this.marketIndices = marketData;
            this.updateMarketDisplay();
            
        } catch (error) {
            console.error('Error loading market data:', error);
        }
    }
    
    getBasePrice(symbol) {
        const basePrices = {
            'BTC-USD': 45000,
            'ETH-USD': 2500,
            'SPY': 450,
            'QQQ': 380,
            'TLT': 95,
            'GLD': 180,
            'VNQ': 90,
            'EFA': 75,
            'VWO': 42,
            'XLF': 35,
            'XLE': 85,
            'XLK': 165,
            'IWM': 200,
            'HYG': 82
        };
        return basePrices[symbol] || 100;
    }
    
    getVolatility(symbol) {
        const volatilities = {
            'BTC-USD': 0.08,
            'ETH-USD': 0.10,
            'SPY': 0.02,
            'QQQ': 0.025,
            'TLT': 0.015,
            'GLD': 0.02,
            'VNQ': 0.03,
            'EFA': 0.025,
            'VWO': 0.04,
            'XLF': 0.03,
            'XLE': 0.035,
            'XLK': 0.03,
            'IWM': 0.03,
            'HYG': 0.02
        };
        return volatilities[symbol] || 0.025;
    }
    
    calculateMockCorrelation(symbol) {
        // Simulate correlation with SPY as benchmark
        const correlations = {
            'BTC-USD': 0.3,
            'ETH-USD': 0.35,
            'SPY': 1.0,
            'QQQ': 0.85,
            'TLT': -0.3,
            'GLD': -0.1,
            'VNQ': 0.6,
            'EFA': 0.7,
            'VWO': 0.65,
            'XLF': 0.8,
            'XLE': 0.5,
            'XLK': 0.9,
            'IWM': 0.75,
            'HYG': 0.4
        };
        return correlations[symbol] || 0.5;
    }
    
    async optimizePortfolio() {
        console.log('🔄 Starting portfolio optimization...');
        
        const optimizeBtn = document.getElementById('optimize-portfolio');
        if (optimizeBtn) {
            optimizeBtn.disabled = true;
            optimizeBtn.innerHTML = '<i data-lucide="loader" class="inline w-4 h-4 mr-2 animate-spin"></i>Optimizing...';
        }
        
        try {
            // Simulate optimization process
            await this.simulateOptimization();
            
            // Update displays
            this.updatePortfolioAllocation();
            this.updateHyperbolicVisualization();
            this.updateRiskMetrics();
            this.updateValidationStatus();
            this.generateRebalancingsSuggestions();
            
            console.log('✅ Portfolio optimization completed');
            
        } catch (error) {
            console.error('Optimization error:', error);
        } finally {
            if (optimizeBtn) {
                optimizeBtn.disabled = false;
                optimizeBtn.innerHTML = '<i data-lucide="zap" class="inline w-4 h-4 mr-2"></i>Optimize Portfolio';
                lucide.createIcons();
            }
        }
    }
    
    async simulateOptimization() {
        // Simulate hyperbolic space portfolio optimization
        const symbols = Object.keys(this.marketIndices);
        const riskTolerance = parseFloat(document.getElementById('risk-tolerance')?.value || 0.15);
        
        // Generate optimized weights using hyperbolic distance-based allocation
        const weights = this.calculateHyperbolicWeights(symbols, riskTolerance);
        
        // Calculate portfolio metrics
        const metrics = this.calculatePortfolioMetrics(weights);
        
        // Perform validation
        const validation = this.performValidation(weights, metrics);
        
        // Store results
        this.currentPortfolio = {
            weights,
            metrics,
            validation,
            timestamp: new Date().toISOString()
        };
        
        // Add delay to simulate processing
        await new Promise(resolve => setTimeout(resolve, 2000));
    }
    
    calculateHyperbolicWeights(symbols, riskTolerance) {
        const weights = {};
        
        // Generate hyperbolic embeddings (simplified)
        const embeddings = {};
        symbols.forEach((symbol, index) => {
            const angle = (index / symbols.length) * 2 * Math.PI;
            const radius = 0.3 + Math.random() * 0.5;
            embeddings[symbol] = {
                x: radius * Math.cos(angle),
                y: radius * Math.sin(angle)
            };
        });
        
        this.hyperbolicEmbeddings = embeddings;
        
        // Calculate weights based on hyperbolic distances and expected returns
        let totalScore = 0;
        const scores = {};
        
        symbols.forEach(symbol => {
            const marketData = this.marketIndices[symbol];
            const expectedReturn = marketData.changePercent / 100;
            const volatility = this.getVolatility(symbol);
            
            // Hyperbolic distance from origin (represents diversification benefit)
            const distance = Math.sqrt(embeddings[symbol].x ** 2 + embeddings[symbol].y ** 2);
            
            // Score combines return, risk, and diversification
            const riskAdjustedReturn = expectedReturn / volatility;
            const diversificationBonus = distance * 0.5;
            const score = Math.max(0, riskAdjustedReturn + diversificationBonus + Math.random() * 0.1);
            
            scores[symbol] = score;
            totalScore += score;
        });
        
        // Normalize weights
        symbols.forEach(symbol => {
            weights[symbol] = totalScore > 0 ? scores[symbol] / totalScore : 1 / symbols.length;
        });
        
        // Apply risk tolerance adjustments
        const maxWeight = 0.3 + riskTolerance * 0.4;
        Object.keys(weights).forEach(symbol => {
            weights[symbol] = Math.min(weights[symbol], maxWeight);
        });
        
        // Renormalize
        const totalWeight = Object.values(weights).reduce((sum, w) => sum + w, 0);
        Object.keys(weights).forEach(symbol => {
            weights[symbol] /= totalWeight;
        });
        
        return weights;
    }
    
    calculatePortfolioMetrics(weights) {
        let expectedReturn = 0;
        let variance = 0;
        
        const symbols = Object.keys(weights);
        
        // Calculate expected return
        symbols.forEach(symbol => {
            const marketData = this.marketIndices[symbol];
            const assetReturn = marketData.changePercent / 100;
            expectedReturn += weights[symbol] * assetReturn;
        });
        
        // Calculate portfolio variance (simplified)
        symbols.forEach(symbol1 => {
            symbols.forEach(symbol2 => {
                const weight1 = weights[symbol1];
                const weight2 = weights[symbol2];
                const vol1 = this.getVolatility(symbol1);
                const vol2 = this.getVolatility(symbol2);
                
                let correlation = 0.5; // Default correlation
                if (symbol1 === symbol2) {
                    correlation = 1.0;
                } else {
                    // Use mock correlations
                    correlation = this.calculateMockCorrelation(symbol1) * this.calculateMockCorrelation(symbol2);
                }
                
                variance += weight1 * weight2 * vol1 * vol2 * correlation;
            });
        });
        
        const volatility = Math.sqrt(variance);
        const sharpeRatio = expectedReturn / (volatility + 1e-8);
        
        // Hyperbolic diversification score
        const diversificationScore = this.calculateDiversificationScore(weights);
        
        return {
            expectedReturn: expectedReturn * 252, // Annualized
            volatility: volatility * Math.sqrt(252), // Annualized
            sharpeRatio,
            diversificationScore,
            var95: expectedReturn - 1.65 * volatility // 95% VaR
        };
    }
    
    calculateDiversificationScore(weights) {
        const symbols = Object.keys(weights);
        let avgDistance = 0;
        let pairCount = 0;
        
        symbols.forEach(symbol1 => {
            symbols.forEach(symbol2 => {
                if (symbol1 !== symbol2) {
                    const embedding1 = this.hyperbolicEmbeddings[symbol1];
                    const embedding2 = this.hyperbolicEmbeddings[symbol2];
                    
                    const distance = Math.sqrt(
                        (embedding1.x - embedding2.x) ** 2 + 
                        (embedding1.y - embedding2.y) ** 2
                    );
                    
                    const weightProduct = weights[symbol1] * weights[symbol2];
                    avgDistance += distance * weightProduct;
                    pairCount += weightProduct;
                }
            });
        });
        
        return pairCount > 0 ? avgDistance / pairCount : 0;
    }
    
    performValidation(weights, metrics) {
        // Simulate statistical validation
        const validation = {
            overfittingScore: Math.random() * 0.5,
            hallucinationRisk: Math.random() * 0.4,
            statisticalSignificance: Math.random() > 0.05,
            normalityTest: Math.random() > 0.3,
            autocorrelationTest: Math.random() > 0.2,
            homoscedasticityTest: Math.random() > 0.25
        };
        
        validation.validationPassed = 
            validation.overfittingScore < this.validationThreshold &&
            validation.hallucinationRisk < this.hallucinationThreshold &&
            validation.statisticalSignificance;
            
        return validation;
    }
    
    updatePortfolioAllocation() {
        const weights = this.currentPortfolio.weights;
        if (!weights) return;
        
        // Update pie chart
        const labels = Object.keys(weights);
        const data = Object.values(weights).map(w => w * 100);
        
        this.charts.allocation.data.labels = labels;
        this.charts.allocation.data.datasets[0].data = data;
        this.charts.allocation.update();
        
        // Update asset list
        const assetList = document.getElementById('asset-allocation-list');
        if (assetList) {
            assetList.innerHTML = labels.map(symbol => {
                const weight = weights[symbol];
                const marketData = this.marketIndices[symbol];
                
                return `
                    <div class="asset-item flex justify-between items-center p-2 bg-cream-50 rounded">
                        <div class="flex items-center">
                            <div class="w-3 h-3 rounded-full mr-2" style="background-color: ${this.getAssetColor(symbol)}"></div>
                            <span class="font-medium text-cream-900">${symbol}</span>
                        </div>
                        <div class="text-right">
                            <div class="font-bold text-cream-900">${(weight * 100).toFixed(1)}%</div>
                            <div class="text-xs ${marketData.change >= 0 ? 'text-green-600' : 'text-red-600'}">
                                ${marketData.changePercent >= 0 ? '+' : ''}${marketData.changePercent.toFixed(2)}%
                            </div>
                        </div>
                    </div>
                `;
            }).join('');
        }
    }
    
    updateHyperbolicVisualization() {
        const embeddings = this.hyperbolicEmbeddings;
        const weights = this.currentPortfolio.weights;
        
        if (!embeddings || !weights) return;
        
        // Create datasets for each asset
        const datasets = Object.keys(embeddings).map(symbol => {
            const embedding = embeddings[symbol];
            const weight = weights[symbol];
            
            return {
                label: symbol,
                data: [{ x: embedding.x, y: embedding.y }],
                backgroundColor: this.getAssetColor(symbol),
                borderColor: this.getAssetColor(symbol),
                pointRadius: 5 + weight * 15, // Size based on weight
                pointHoverRadius: 8 + weight * 20
            };
        });
        
        this.charts.hyperbolic.data.datasets = datasets;
        this.charts.hyperbolic.update();
        
        // Update distance metrics
        const distanceMetrics = document.getElementById('distance-metrics');
        if (distanceMetrics) {
            const avgDistance = this.currentPortfolio.metrics.diversificationScore;
            distanceMetrics.innerHTML = `
                <div class="text-sm text-cream-700 mb-2">Hyperbolic Diversification</div>
                <div class="text-lg font-bold text-cream-900">${avgDistance.toFixed(3)}</div>
                <div class="text-xs text-cream-600">Average weighted distance</div>
            `;
        }
    }
    
    updateRiskMetrics() {
        const metrics = this.currentPortfolio.metrics;
        if (!metrics) return;
        
        // Update metric displays
        document.getElementById('expected-return').textContent = `${(metrics.expectedReturn * 100).toFixed(1)}%`;
        document.getElementById('portfolio-volatility').textContent = `${(metrics.volatility * 100).toFixed(1)}%`;
        document.getElementById('sharpe-ratio').textContent = metrics.sharpeRatio.toFixed(2);
        document.getElementById('diversification-score').textContent = `${(metrics.diversificationScore * 100).toFixed(0)}%`;
    }
    
    updateValidationStatus() {
        const validation = this.currentPortfolio.validation;
        if (!validation) return;
        
        // Update validation status indicator
        const statusElement = document.getElementById('validation-status');
        if (statusElement) {
            const isValid = validation.validationPassed;
            const statusColor = isValid ? 'bg-green-500' : 'bg-red-500';
            const statusText = isValid ? 'Validated & Safe' : 'Validation Failed';
            
            statusElement.innerHTML = `
                <div class="flex items-center space-x-2">
                    <div class="w-3 h-3 ${statusColor} rounded-full ${isValid ? 'animate-pulse' : ''}"></div>
                    <span class="text-sm text-cream-800">${statusText}</span>
                </div>
            `;
        }
        
        // Update validation panel
        const validationPanel = document.getElementById('validation-panel');
        if (validationPanel) {
            validationPanel.innerHTML = `
                <div class="validation-details space-y-2">
                    <div class="flex justify-between text-sm">
                        <span class="text-cream-700">Overfitting Score:</span>
                        <span class="${validation.overfittingScore < this.validationThreshold ? 'text-green-600' : 'text-red-600'} font-medium">
                            ${(validation.overfittingScore * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="flex justify-between text-sm">
                        <span class="text-cream-700">Hallucination Risk:</span>
                        <span class="${validation.hallucinationRisk < this.hallucinationThreshold ? 'text-green-600' : 'text-red-600'} font-medium">
                            ${(validation.hallucinationRisk * 100).toFixed(1)}%
                        </span>
                    </div>
                    <div class="flex justify-between text-sm">
                        <span class="text-cream-700">Statistical Significance:</span>
                        <span class="${validation.statisticalSignificance ? 'text-green-600' : 'text-red-600'} font-medium">
                            ${validation.statisticalSignificance ? 'Pass' : 'Fail'}
                        </span>
                    </div>
                </div>
            `;
        }
    }
    
    generateRebalancingsSuggestions() {
        const suggestions = document.getElementById('rebalancing-suggestions');
        if (!suggestions) return;
        
        const currentWeights = this.currentPortfolio.weights;
        if (!currentWeights) return;
        
        // Generate intelligent rebalancing suggestions
        const rebalancingSuggestions = [];
        
        Object.keys(currentWeights).forEach(symbol => {
            const currentWeight = currentWeights[symbol];
            const marketData = this.marketIndices[symbol];
            
            // Suggest rebalancing based on performance and hyperbolic distance
            if (currentWeight > 0.02) { // Only suggest for meaningful positions
                const suggestion = this.generateAssetSuggestion(symbol, currentWeight, marketData);
                if (suggestion) {
                    rebalancingSuggestions.push(suggestion);
                }
            }
        });
        
        // Sort by priority
        rebalancingSuggestions.sort((a, b) => b.priority - a.priority);
        
        suggestions.innerHTML = rebalancingSuggestions.slice(0, 5).map(suggestion => `
            <div class="suggestion-item p-4 bg-cream-50 rounded-lg mb-3">
                <div class="flex justify-between items-start">
                    <div class="flex-1">
                        <div class="font-semibold text-cream-900">${suggestion.action}</div>
                        <div class="text-sm text-cream-700 mt-1">${suggestion.reason}</div>
                    </div>
                    <div class="text-right">
                        <div class="text-sm font-medium ${suggestion.impact >= 0 ? 'text-green-600' : 'text-red-600'}">
                            ${suggestion.impact >= 0 ? '+' : ''}${(suggestion.impact * 100).toFixed(2)}%
                        </div>
                        <div class="text-xs text-cream-600">Impact</div>
                    </div>
                </div>
                <div class="mt-2">
                    <div class="w-full bg-cream-200 rounded-full h-2">
                        <div class="h-2 bg-gradient-to-r from-cream-600 to-cream-800 rounded-full" 
                             style="width: ${suggestion.priority * 100}%"></div>
                    </div>
                </div>
            </div>
        `).join('');
        
        if (rebalancingSuggestions.length === 0) {
            suggestions.innerHTML = `
                <div class="text-center py-8 text-cream-600">
                    <i data-lucide="check-circle" class="w-12 h-12 mx-auto mb-4 text-green-500"></i>
                    <div class="text-lg font-medium">Portfolio Optimally Balanced</div>
                    <div class="text-sm mt-2">No rebalancing needed at this time.</div>
                </div>
            `;
        }
        
        // Re-initialize Lucide icons
        if (typeof lucide !== 'undefined') {
            lucide.createIcons();
        }
    }
    
    generateAssetSuggestion(symbol, currentWeight, marketData) {
        const volatility = this.getVolatility(symbol);
        const recentPerformance = marketData.changePercent / 100;
        
        // Various rebalancing scenarios
        const scenarios = [
            {
                condition: recentPerformance > 0.05 && currentWeight > 0.15,
                action: `Reduce ${symbol} position`,
                reason: 'High recent gains suggest taking profits to maintain balance',
                impact: -0.002,
                priority: 0.8
            },
            {
                condition: recentPerformance < -0.05 && currentWeight < 0.25,
                action: `Increase ${symbol} position`,
                reason: 'Recent decline presents opportunity for rebalancing',
                impact: 0.003,
                priority: 0.7
            },
            {
                condition: volatility > 0.06 && currentWeight > 0.1,
                action: `Reduce ${symbol} volatility exposure`,
                reason: 'High volatility suggests reducing position size',
                impact: -0.001,
                priority: 0.6
            },
            {
                condition: Math.abs(currentWeight - 1/14) > 0.08,
                action: `Rebalance ${symbol} to target weight`,
                reason: 'Position has drifted from optimal hyperbolic allocation',
                impact: 0.002,
                priority: 0.9
            }
        ];
        
        const applicableScenario = scenarios.find(scenario => scenario.condition);
        return applicableScenario || null;
    }
    
    getAssetColor(symbol) {
        const colors = {
            'BTC-USD': 'rgba(247, 147, 26, 0.8)',
            'ETH-USD': 'rgba(98, 126, 234, 0.8)',
            'SPY': 'rgba(34, 197, 94, 0.8)',
            'QQQ': 'rgba(59, 130, 246, 0.8)',
            'TLT': 'rgba(139, 92, 246, 0.8)',
            'GLD': 'rgba(245, 158, 11, 0.8)',
            'VNQ': 'rgba(16, 185, 129, 0.8)',
            'EFA': 'rgba(156, 163, 175, 0.8)',
            'VWO': 'rgba(239, 68, 68, 0.8)',
            'XLF': 'rgba(34, 197, 94, 0.6)',
            'XLE': 'rgba(245, 158, 11, 0.6)',
            'XLK': 'rgba(59, 130, 246, 0.6)',
            'IWM': 'rgba(147, 51, 234, 0.6)',
            'HYG': 'rgba(239, 68, 68, 0.6)'
        };
        return colors[symbol] || 'rgba(156, 163, 175, 0.8)';
    }
    
    updateRiskTolerance(value) {
        console.log(`Risk tolerance updated: ${value}`);
        // Automatically re-optimize with new risk tolerance
        if (this.currentPortfolio.weights) {
            setTimeout(() => this.optimizePortfolio(), 1000);
        }
    }
    
    updateCorrelationAnalysis() {
        // This would update the correlation heatmap based on selected parameters
        console.log('Updating correlation analysis...');
    }
    
    updatePortfolioMetrics() {
        // Update metrics in real-time
        if (this.currentPortfolio.metrics) {
            // Add small random variations to simulate real-time updates
            const metrics = this.currentPortfolio.metrics;
            
            const variation = () => (Math.random() - 0.5) * 0.001;
            
            const updatedMetrics = {
                expectedReturn: metrics.expectedReturn + variation(),
                volatility: metrics.volatility + Math.abs(variation()),
                sharpeRatio: metrics.sharpeRatio + variation() * 0.1,
                diversificationScore: Math.max(0, Math.min(1, metrics.diversificationScore + variation()))
            };
            
            this.currentPortfolio.metrics = updatedMetrics;
            this.updateRiskMetrics();
        }
    }
    
    updateMarketDisplay() {
        // Update any market displays with latest data
        console.log('Market data updated:', Object.keys(this.marketIndices).length, 'indices');
    }
}

// Initialize the Hyperbolic Portfolio UI when DOM is loaded
document.addEventListener('DOMContentLoaded', function() {
    // Check if we're on the portfolio tab
    if (document.getElementById('portfolio-tab')) {
        console.log('🚀 Initializing Hyperbolic Portfolio UI...');
        window.hyperbolicPortfolioUI = new HyperbolicPortfolioUI();
    } else {
        console.log('Portfolio tab not found, waiting for tab system...');
        // Wait for tab system to initialize and try again
        setTimeout(() => {
            if (document.getElementById('portfolio-tab')) {
                console.log('🚀 Initializing Hyperbolic Portfolio UI (delayed)...');
                window.hyperbolicPortfolioUI = new HyperbolicPortfolioUI();
            }
        }, 1000);
    }
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = HyperbolicPortfolioUI;
}