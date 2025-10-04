// Trading Dashboard JavaScript

class TradingDashboard {
    constructor() {
        this.currentSection = 'dashboard'
        this.refreshInterval = null
        this.portfolio = null
        this.candlestickChart = null
        this.currentSymbol = 'BTC'
        this.currentTimeframe = '1m'
        this.patternAlerts = []
        this.hyperbolicAnalysis = {}
        this.backtests = {}
        this.paperAccount = null
        this.autoTradingEnabled = false
        this.equityCurveChart = null
        this.drawdownChart = null
        this.clusteringData = null
        this.clusteringAnimation = null
        this.currentVisualization = 'patterns'
        this.init()
    }

    init() {
        this.setupNavigation()
        this.setupEventListeners()
        this.startDataRefresh()
        this.updateClock()
        setInterval(() => this.updateClock(), 1000)
    }

    setupNavigation() {
        const navItems = document.querySelectorAll('.nav-item')
        const sections = document.querySelectorAll('.section')

        navItems.forEach(item => {
            item.addEventListener('click', () => {
                const sectionName = item.dataset.section
                
                // Update active nav item
                navItems.forEach(nav => nav.classList.remove('active'))
                item.classList.add('active')
                
                // Show corresponding section
                sections.forEach(section => section.classList.remove('active'))
                document.getElementById(sectionName).classList.add('active')
                
                this.currentSection = sectionName
                this.loadSectionData(sectionName)
            })
        })
    }

    setupEventListeners() {
        // Chat functionality
        const chatInput = document.getElementById('chat-input')
        const sendButton = document.getElementById('send-message')
        const quickQueries = document.querySelectorAll('.quick-query')

        if (chatInput && sendButton) {
            chatInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter') {
                    this.sendChatMessage()
                }
            })
            
            sendButton.addEventListener('click', () => {
                this.sendChatMessage()
            })
        }

        quickQueries.forEach(button => {
            button.addEventListener('click', () => {
                const query = button.dataset.query
                chatInput.value = query
                this.sendChatMessage()
            })
        })

        // Execute arbitrage buttons (will be added dynamically)
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('execute-arbitrage')) {
                const opportunityData = JSON.parse(e.target.dataset.opportunity)
                this.executeArbitrage(opportunityData)
            }
        })

        // Advanced Chart Controls
        this.setupChartControls()
        
        // Visualization Toggle Controls
        this.setupVisualizationToggle()
    }

    updateClock() {
        const now = new Date()
        const timeString = now.toLocaleTimeString('en-US', { 
            hour12: true,
            hour: '2-digit',
            minute: '2-digit',
            second: '2-digit'
        })
        document.getElementById('current-time').textContent = timeString
    }

    startDataRefresh() {
        this.loadSectionData(this.currentSection)
        this.refreshInterval = setInterval(() => {
            if (this.currentSection === 'dashboard') {
                this.loadMarketData()
                this.loadArbitrageOpportunities()
                this.loadOrderBook()
            }
        }, 2000) // Refresh every 2 seconds for live effect
    }

    async loadSectionData(section) {
        switch (section) {
            case 'dashboard':
                await Promise.all([
                    this.loadMarketData(),
                    this.loadArbitrageOpportunities(),
                    this.loadOrderBook(),
                    this.drawPoincareVisualization(),
                    this.initializeCandlestickChart(),
                    this.startHyperbolicAnalysis()
                ])
                // Initialize clustering for hyperbolic space engine
                if (this.currentVisualization === 'clustering') {
                    this.initializeAssetClustering()
                }
                break
            case 'portfolio':
                await this.loadPortfolioData()
                break
            case 'markets':
                await this.loadGlobalMarkets()
                break
            case 'transparency':
                this.loadModelTransparency()
                break
            case 'backtesting':
                this.initializeBacktesting()
                break
            case 'paper-trading':
                this.initializePaperTrading()
                break
        }
    }

    async loadMarketData() {
        try {
            const response = await axios.get('/api/market-data')
            const data = response.data
            
            const feedsContainer = document.getElementById('market-feeds')
            feedsContainer.innerHTML = Object.entries(data).map(([symbol, info]) => `
                <div class="flex justify-between items-center border-b border-gray-700 pb-2">
                    <div>
                        <div class="font-semibold">${symbol}/USD</div>
                        <div class="text-sm text-gray-400">
                            Vol: ${(info.volume / 1000).toFixed(1)}K | 
                            Trades: ${(info.trades / 1000).toFixed(0)}K
                        </div>
                    </div>
                    <div class="text-right">
                        <div class="text-lg font-bold">$${info.price.toFixed(2)}</div>
                        <div class="text-sm ${info.change24h > 0 ? 'text-profit' : 'text-loss'}">
                            ${info.change24h > 0 ? '+' : ''}${info.change24h.toFixed(2)}%
                        </div>
                    </div>
                </div>
            `).join('')

            // Update spreads
            const spreadsContainer = document.getElementById('spreads')
            spreadsContainer.innerHTML = `
                <div class="flex justify-between">
                    <span>Binance-Coinbase:</span>
                    <span class="text-profit">+$${(Math.random() * 20 + 5).toFixed(2)}</span>
                </div>
                <div class="flex justify-between">
                    <span>Kraken-Bybit:</span>
                    <span class="text-profit">+$${(Math.random() * 25 + 10).toFixed(2)}</span>
                </div>
                <div class="flex justify-between">
                    <span>Futures-Spot:</span>
                    <span class="text-profit">+$${(Math.random() * 200 + 100).toFixed(2)}</span>
                </div>
            `
            
        } catch (error) {
            console.error('Error loading market data:', error)
        }
    }

    async loadArbitrageOpportunities() {
        try {
            const response = await axios.get('/api/arbitrage-opportunities')
            const opportunities = response.data
            
            const container = document.getElementById('arbitrage-opportunities')
            const lastScan = document.getElementById('last-scan')
            
            lastScan.textContent = new Date().toLocaleTimeString('en-US', { 
                hour12: true,
                hour: '2-digit',
                minute: '2-digit',
                second: '2-digit'
            })
            
            container.innerHTML = opportunities.map((opp, index) => {
                const icon = opp.type.includes('Cross') ? '📊' : 
                           opp.type.includes('Triangular') ? '📐' : '🧠'
                
                return `
                    <div class="border border-gray-700 rounded-lg p-4 hover:border-accent transition-colors">
                        <div class="flex items-center justify-between mb-3">
                            <div class="flex items-center">
                                <span class="text-xl mr-2">${icon}</span>
                                <div>
                                    <div class="font-semibold text-accent">${opp.type}</div>
                                    <div class="text-sm text-gray-400">${opp.pair}</div>
                                </div>
                            </div>
                            <div class="text-right">
                                <div class="text-lg font-bold text-profit">+${opp.profit}% ($${opp.profitUSD})</div>
                                <div class="text-sm text-gray-400">Est. ${opp.executionTime} execution</div>
                            </div>
                        </div>
                        
                        ${opp.buyPrice ? `
                            <div class="grid grid-cols-3 gap-4 text-sm mb-3">
                                <div>
                                    <div class="text-gray-400">Buy Price:</div>
                                    <div class="font-semibold">$${opp.buyPrice.toFixed(2)}</div>
                                </div>
                                <div>
                                    <div class="text-gray-400">Sell Price:</div>
                                    <div class="font-semibold">$${opp.sellPrice.toFixed(2)}</div>
                                </div>
                                <div>
                                    <div class="text-gray-400">Volume:</div>
                                    <div class="font-semibold">${opp.volume} BTC</div>
                                </div>
                            </div>
                        ` : ''}
                        
                        ${opp.zScore ? `
                            <div class="grid grid-cols-3 gap-4 text-sm mb-3">
                                <div>
                                    <div class="text-gray-400">Z-Score:</div>
                                    <div class="font-semibold">${opp.zScore}σ</div>
                                </div>
                                <div>
                                    <div class="text-gray-400">Correlation:</div>
                                    <div class="font-semibold">${opp.correlation}</div>
                                </div>
                                <div>
                                    <div class="text-gray-400">FinBERT:</div>
                                    <div class="font-semibold">+${opp.finBERT}</div>
                                </div>
                            </div>
                        ` : ''}
                        
                        <div class="flex space-x-2">
                            <button class="execute-arbitrage bg-accent text-dark-bg px-4 py-2 rounded text-sm font-semibold hover:bg-opacity-80 flex items-center"
                                    data-opportunity='${JSON.stringify(opp)}'>
                                ⚡ Execute Arbitrage
                            </button>
                            <button class="bg-gray-700 text-white px-4 py-2 rounded text-sm hover:bg-gray-600">
                                📊 Details
                            </button>
                        </div>
                    </div>
                `
            }).join('')
            
        } catch (error) {
            console.error('Error loading arbitrage opportunities:', error)
        }
    }

    async loadOrderBook() {
        try {
            const response = await axios.get('/api/orderbook/BTC')
            const data = response.data
            
            const container = document.getElementById('order-book')
            container.innerHTML = `
                <div class="grid grid-cols-2 gap-4 text-sm">
                    <div>
                        <div class="font-semibold text-loss mb-2">ASKS</div>
                        ${data.asks.slice(0, 5).map(ask => `
                            <div class="flex justify-between py-1">
                                <span>$${ask.price}</span>
                                <span class="text-gray-400">${ask.volume} BTC</span>
                            </div>
                        `).join('')}
                    </div>
                    <div>
                        <div class="font-semibold text-profit mb-2">BIDS</div>
                        ${data.bids.slice(0, 5).map(bid => `
                            <div class="flex justify-between py-1">
                                <span>$${bid.price}</span>
                                <span class="text-gray-400">${bid.volume} BTC</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
                <div class="mt-4 text-center">
                    <div class="text-sm text-gray-400">Spread: <span class="text-warning">$${data.spread}</span></div>
                </div>
            `
            
        } catch (error) {
            console.error('Error loading order book:', error)
        }
    }

    async loadPortfolioData() {
        try {
            const response = await axios.get('/api/portfolio')
            this.portfolio = response.data
            
            const container = document.getElementById('portfolio-content')
            container.innerHTML = `
                <div class="mb-6">
                    <div class="text-3xl font-bold mb-2">$${this.portfolio.totalValue.toLocaleString()}</div>
                    <div class="text-profit text-lg">+${this.portfolio.monthlyChange}% MTD</div>
                </div>
                
                <div class="grid grid-cols-4 gap-6 mb-6">
                    ${Object.entries(this.portfolio.assets).map(([symbol, asset]) => `
                        <div class="bg-gray-800 rounded-lg p-4">
                            <div class="text-lg font-semibold">${asset.percentage}%</div>
                            <div class="text-sm text-gray-400">${symbol === 'STABLE' ? 'Stablecoins' : symbol === 'OTHER' ? 'Other Assets' : symbol}</div>
                            <div class="text-sm font-semibold">$${asset.value.toLocaleString()}</div>
                            <div class="text-xs ${asset.pnlPercent > 0 ? 'text-profit' : asset.pnlPercent < 0 ? 'text-loss' : 'text-gray-400'}">
                                ${asset.pnlPercent > 0 ? '+' : ''}${asset.pnlPercent.toFixed(1)}%
                            </div>
                        </div>
                    `).join('')}
                </div>
                
                <div class="bg-gray-800 rounded-lg p-4">
                    <h4 class="font-semibold mb-4">Active Positions</h4>
                    <div class="overflow-x-auto">
                        <table class="w-full text-sm">
                            <thead>
                                <tr class="border-b border-gray-700">
                                    <th class="text-left py-2">Asset</th>
                                    <th class="text-right py-2">Quantity</th>
                                    <th class="text-right py-2">Avg Price</th>
                                    <th class="text-right py-2">Current Price</th>
                                    <th class="text-right py-2">P&L</th>
                                </tr>
                            </thead>
                            <tbody>
                                ${Object.entries(this.portfolio.assets).filter(([k,v]) => k !== 'STABLE' && k !== 'OTHER').map(([symbol, asset]) => `
                                    <tr class="border-b border-gray-700">
                                        <td class="py-2 font-semibold">${symbol}/USD</td>
                                        <td class="text-right">${asset.quantity}</td>
                                        <td class="text-right">$${asset.avgPrice.toLocaleString()}</td>
                                        <td class="text-right">$${asset.currentPrice.toLocaleString()}</td>
                                        <td class="text-right ${asset.pnl > 0 ? 'text-profit' : 'text-loss'}">
                                            ${asset.pnl > 0 ? '+' : ''}$${asset.pnl.toLocaleString()} (${asset.pnlPercent > 0 ? '+' : ''}${asset.pnlPercent}%)
                                        </td>
                                    </tr>
                                `).join('')}
                            </tbody>
                        </table>
                    </div>
                </div>
            `
            
            this.drawPortfolioChart()
            
        } catch (error) {
            console.error('Error loading portfolio data:', error)
        }
    }

    async loadGlobalMarkets() {
        try {
            const response = await axios.get('/api/global-markets')
            const data = response.data
            
            const container = document.getElementById('global-markets-content')
            container.innerHTML = `
                <div class="grid grid-cols-5 gap-6">
                    <div>
                        <h4 class="font-semibold mb-4 text-accent">💎 Cryptocurrency</h4>
                        ${Object.entries(data.crypto).map(([symbol, info]) => `
                            <div class="flex justify-between items-center py-2">
                                <div>
                                    <div class="font-semibold">${symbol}</div>
                                    <div class="text-sm ${info.change > 0 ? 'text-profit' : 'text-loss'}">
                                        ${info.change > 0 ? '+' : ''}${info.change.toFixed(2)}%
                                    </div>
                                </div>
                                <div class="text-right">
                                    <div class="font-semibold">$${info.price.toLocaleString()}</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                    
                    <div>
                        <h4 class="font-semibold mb-4 text-accent">🇺🇸 US Equity</h4>
                        ${Object.entries(data.equity).map(([symbol, info]) => `
                            <div class="flex justify-between items-center py-2">
                                <div>
                                    <div class="font-semibold">${symbol}</div>
                                    <div class="text-sm ${info.change > 0 ? 'text-profit' : 'text-loss'}">
                                        ${info.change > 0 ? '+' : ''}${info.change.toFixed(2)}%
                                    </div>
                                </div>
                                <div class="text-right">
                                    <div class="font-semibold">${info.price.toLocaleString()}</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                    
                    <div>
                        <h4 class="font-semibold mb-4 text-accent">🌍 International</h4>
                        ${Object.entries(data.international).map(([symbol, info]) => `
                            <div class="flex justify-between items-center py-2">
                                <div>
                                    <div class="font-semibold">${symbol}</div>
                                    <div class="text-sm ${info.change > 0 ? 'text-profit' : 'text-loss'}">
                                        ${info.change > 0 ? '+' : ''}${info.change.toFixed(2)}%
                                    </div>
                                </div>
                                <div class="text-right">
                                    <div class="font-semibold">${info.price.toLocaleString()}</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                    
                    <div>
                        <h4 class="font-semibold mb-4 text-accent">🏆 Commodities</h4>
                        ${Object.entries(data.commodities).map(([symbol, info]) => `
                            <div class="flex justify-between items-center py-2">
                                <div>
                                    <div class="font-semibold">${symbol}</div>
                                    <div class="text-sm ${info.change > 0 ? 'text-profit' : 'text-loss'}">
                                        ${info.change > 0 ? '+' : ''}${info.change.toFixed(2)}%
                                    </div>
                                </div>
                                <div class="text-right">
                                    <div class="font-semibold">$${info.price.toFixed(2)}</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                    
                    <div>
                        <h4 class="font-semibold mb-4 text-accent">💱 Major Forex</h4>
                        ${Object.entries(data.forex).map(([symbol, info]) => `
                            <div class="flex justify-between items-center py-2">
                                <div>
                                    <div class="font-semibold">${symbol}</div>
                                    <div class="text-sm ${info.change > 0 ? 'text-profit' : 'text-loss'}">
                                        ${info.change > 0 ? '+' : ''}${info.change.toFixed(2)}%
                                    </div>
                                </div>
                                <div class="text-right">
                                    <div class="font-semibold">${info.price.toFixed(4)}</div>
                                </div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `
            
        } catch (error) {
            console.error('Error loading global markets:', error)
        }
    }

    loadModelTransparency() {
        const container = document.getElementById('transparency-content')
        container.innerHTML = `
            <div class="grid grid-cols-2 gap-6">
                <div>
                    <h4 class="font-semibold mb-4 text-accent">Model Architecture</h4>
                    <div class="bg-gray-800 rounded-lg p-4 mb-4">
                        <div class="font-semibold mb-2">Hyperbolic Neural Network</div>
                        <ul class="text-sm text-gray-300 space-y-1">
                            <li>• Poincaré Ball Model (Curvature: -1.0)</li>
                            <li>• Embedding Dimension: 768</li>
                            <li>• Geodesic Distance Optimization</li>
                            <li>• Möbius Transformations</li>
                        </ul>
                    </div>
                    
                    <div class="bg-gray-800 rounded-lg p-4">
                        <div class="font-semibold mb-2">Ensemble Components</div>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between">
                                <span>Hyperbolic CNN:</span>
                                <span class="text-accent">40% weight</span>
                            </div>
                            <div class="flex justify-between">
                                <span>LSTM-Transformer:</span>
                                <span class="text-accent">25% weight</span>
                            </div>
                            <div class="flex justify-between">
                                <span>FinBERT Sentiment:</span>
                                <span class="text-accent">20% weight</span>
                            </div>
                            <div class="flex justify-between">
                                <span>Classical Arbitrage:</span>
                                <span class="text-accent">15% weight</span>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div>
                    <h4 class="font-semibold mb-4 text-accent">Live Model Performance</h4>
                    <div class="grid grid-cols-2 gap-4 mb-4">
                        <div class="bg-gray-800 rounded-lg p-4 text-center">
                            <div class="text-2xl font-bold text-profit">91.2%</div>
                            <div class="text-sm text-gray-400">Model Accuracy</div>
                        </div>
                        <div class="bg-gray-800 rounded-lg p-4 text-center">
                            <div class="text-2xl font-bold text-accent">0.968</div>
                            <div class="text-sm text-gray-400">AUC-ROC Score</div>
                        </div>
                        <div class="bg-gray-800 rounded-lg p-4 text-center">
                            <div class="text-2xl font-bold text-profit">89.7%</div>
                            <div class="text-sm text-gray-400">Precision</div>
                        </div>
                        <div class="bg-gray-800 rounded-lg p-4 text-center">
                            <div class="text-2xl font-bold text-profit">92.8%</div>
                            <div class="text-sm text-gray-400">Recall</div>
                        </div>
                    </div>
                    
                    <div class="bg-gray-800 rounded-lg p-4">
                        <div class="font-semibold mb-2">Feature Importance</div>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between">
                                <span>Price Momentum</span>
                                <span class="text-accent">0.31</span>
                            </div>
                            <div class="flex justify-between">
                                <span>Volume Profile</span>
                                <span class="text-accent">0.24</span>
                            </div>
                            <div class="flex justify-between">
                                <span>Cross-Exchange Spreads</span>
                                <span class="text-accent">0.19</span>
                            </div>
                            <div class="flex justify-between">
                                <span>Sentiment Score</span>
                                <span class="text-accent">0.15</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        `
    }

    drawPortfolioChart() {
        if (!this.portfolio) return
        
        const canvas = document.getElementById('portfolio-chart')
        const ctx = canvas.getContext('2d')
        
        const assets = Object.entries(this.portfolio.assets)
        const colors = ['#00d4aa', '#ff6b6b', '#4ecdc4', '#45b7d1']
        
        new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: assets.map(([symbol, asset]) => {
                    const name = symbol === 'STABLE' ? 'Stablecoins' : 
                                symbol === 'OTHER' ? 'Other Assets' : symbol
                    return `${name} (${asset.percentage}%)`
                }),
                datasets: [{
                    data: assets.map(([_, asset]) => asset.percentage),
                    backgroundColor: colors,
                    borderColor: '#1a1f29',
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: {
                            color: '#ffffff',
                            font: {
                                family: 'monospace'
                            }
                        }
                    }
                }
            }
        })
    }

    drawPoincareVisualization() {
        const canvas = document.getElementById('poincare-disk')
        const ctx = canvas.getContext('2d')
        const centerX = canvas.width / 2
        const centerY = canvas.height / 2
        const radius = 100
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        
        // Draw Poincaré disk boundary
        ctx.strokeStyle = '#00d4aa'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI)
        ctx.stroke()
        
        // Draw geodesic paths
        ctx.strokeStyle = '#4ecdc4'
        ctx.lineWidth = 1
        
        for (let i = 0; i < 20; i++) {
            const angle1 = (Math.PI * 2 * i) / 20
            const angle2 = angle1 + Math.PI / 3
            
            const x1 = centerX + Math.cos(angle1) * radius * 0.8
            const y1 = centerY + Math.sin(angle1) * radius * 0.8
            const x2 = centerX + Math.cos(angle2) * radius * 0.6
            const y2 = centerY + Math.sin(angle2) * radius * 0.6
            
            ctx.beginPath()
            ctx.moveTo(x1, y1)
            ctx.lineTo(x2, y2)
            ctx.stroke()
        }
        
        // Draw data points
        ctx.fillStyle = '#ff6b6b'
        for (let i = 0; i < 50; i++) {
            const angle = Math.random() * Math.PI * 2
            const r = Math.random() * radius * 0.9
            const x = centerX + Math.cos(angle) * r
            const y = centerY + Math.sin(angle) * r
            
            ctx.beginPath()
            ctx.arc(x, y, 2, 0, 2 * Math.PI)
            ctx.fill()
        }
        
        // Animate by redrawing periodically
        setTimeout(() => this.drawPoincareVisualization(), 3000)
    }

    // Enhanced Asset Clustering Visualization System
    initializeHyperbolicVisualizations() {
        this.setupVisualizationToggle()
        this.initializeAssetClustering()
    }

    setupVisualizationToggle() {
        const patternsBtn = document.getElementById('viz-toggle-patterns')
        const clusteringBtn = document.getElementById('viz-toggle-clustering')
        
        if (patternsBtn && clusteringBtn) {
            patternsBtn.addEventListener('click', () => {
                this.switchVisualization('patterns')
            })
            
            clusteringBtn.addEventListener('click', () => {
                this.switchVisualization('clustering')
            })
        }
    }

    switchVisualization(type) {
        const patternsView = document.getElementById('poincare-patterns-view')
        const clusteringView = document.getElementById('poincare-clustering-view')
        const patternsBtn = document.getElementById('viz-toggle-patterns')
        const clusteringBtn = document.getElementById('viz-toggle-clustering')
        
        if (type === 'patterns') {
            patternsView.classList.remove('hidden')
            clusteringView.classList.add('hidden')
            patternsBtn.classList.add('bg-accent', 'text-dark-bg')
            patternsBtn.classList.remove('text-gray-300')
            clusteringBtn.classList.remove('bg-accent', 'text-dark-bg')
            clusteringBtn.classList.add('text-gray-300')
            this.currentVisualization = 'patterns'
        } else if (type === 'clustering') {
            patternsView.classList.add('hidden')
            clusteringView.classList.remove('hidden')
            clusteringBtn.classList.add('bg-accent', 'text-dark-bg')
            clusteringBtn.classList.remove('text-gray-300')
            patternsBtn.classList.remove('bg-accent', 'text-dark-bg')
            patternsBtn.classList.add('text-gray-300')
            this.currentVisualization = 'clustering'
            this.startAssetClusteringUpdates()
        }
    }

    initializeAssetClustering() {
        // Initialize clustering canvas if not already done
        const canvas = document.getElementById('asset-clustering-disk')
        if (canvas && !canvas.initialized) {
            canvas.initialized = true
            // Start loading clustering data
            this.loadClusteringData()
        }
    }

    async loadClusteringData() {
        try {
            const response = await axios.get('/api/asset-clustering')
            this.clusteringData = response.data.clustering
            
            if (this.currentVisualization === 'clustering') {
                this.drawAssetClustering()
                this.updateClusteringMetrics()
            }
        } catch (error) {
            console.error('Error loading clustering data:', error)
        }
    }

    drawAssetClustering() {
        const canvas = document.getElementById('asset-clustering-disk')
        if (!canvas || !this.clusteringData) return
        
        const ctx = canvas.getContext('2d')
        const centerX = canvas.width / 2
        const centerY = canvas.height / 2
        const radius = 100
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        
        // Draw Poincaré disk boundary
        ctx.strokeStyle = '#00d4aa'
        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI)
        ctx.stroke()
        
        // Draw center indicator (market stability)
        ctx.fillStyle = '#4ecdc4'
        ctx.beginPath()
        ctx.arc(centerX, centerY, 3, 0, 2 * Math.PI)
        ctx.fill()
        
        // Draw correlation connection lines first (so they appear behind assets)
        this.drawCorrelationLines(ctx, centerX, centerY, radius)
        
        // Draw asset nodes
        this.drawAssetNodes(ctx, centerX, centerY, radius)
        
        // Draw cluster boundaries
        this.drawClusterBoundaries(ctx, centerX, centerY, radius)
        
        // Update legend
        this.updateAssetLegend()
    }

    drawCorrelationLines(ctx, centerX, centerY, radius) {
        if (!this.clusteringData.assets) return
        
        const assets = this.clusteringData.assets
        
        // Draw lines between highly correlated assets (correlation > 0.5)
        for (let i = 0; i < assets.length; i++) {
            for (let j = i + 1; j < assets.length; j++) {
                const asset1 = assets[i]
                const asset2 = assets[j]
                const correlation = Math.abs(asset1.correlations[asset2.symbol] || 0)
                
                if (correlation > 0.3) { // Only show significant correlations
                    const x1 = centerX + asset1.x * radius
                    const y1 = centerY + asset1.y * radius
                    const x2 = centerX + asset2.x * radius
                    const y2 = centerY + asset2.y * radius
                    
                    // Line style based on correlation strength
                    ctx.strokeStyle = correlation > 0.7 ? '#2ed573' : 
                                     correlation > 0.5 ? '#ffa502' : '#4ecdc4'
                    ctx.lineWidth = Math.max(1, correlation * 3)
                    ctx.globalAlpha = 0.6
                    
                    // Draw geodesic-like curve (simplified)
                    ctx.beginPath()
                    ctx.moveTo(x1, y1)
                    
                    // Create curved line through disk center for more hyperbolic appearance
                    const midX = centerX + (asset1.x + asset2.x) * radius * 0.3
                    const midY = centerY + (asset1.y + asset2.y) * radius * 0.3
                    ctx.quadraticCurveTo(midX, midY, x2, y2)
                    ctx.stroke()
                    
                    ctx.globalAlpha = 1.0
                }
            }
        }
    }

    drawAssetNodes(ctx, centerX, centerY, radius) {
        if (!this.clusteringData.assets) return
        
        this.clusteringData.assets.forEach(asset => {
            const x = centerX + asset.x * radius
            const y = centerY + asset.y * radius
            
            // Node size based on market cap (logarithmic scale)
            const nodeSize = Math.max(8, Math.min(20, Math.log10(asset.marketCap / 1e9) * 3))
            
            // Color based on price change
            let fillColor
            if (asset.priceChange > 0.01) fillColor = '#2ed573'      // Green for gains
            else if (asset.priceChange < -0.01) fillColor = '#ff4757' // Red for losses  
            else fillColor = '#4ecdc4'                                // Blue for stable
            
            // Volatility glow effect
            const glowRadius = nodeSize + asset.volatility * 100
            const gradient = ctx.createRadialGradient(x, y, nodeSize, x, y, glowRadius)
            gradient.addColorStop(0, fillColor)
            gradient.addColorStop(1, 'transparent')
            
            // Draw glow
            ctx.fillStyle = gradient
            ctx.globalAlpha = 0.3
            ctx.beginPath()
            ctx.arc(x, y, glowRadius, 0, 2 * Math.PI)
            ctx.fill()
            
            // Draw main node
            ctx.globalAlpha = 1.0
            ctx.fillStyle = fillColor
            ctx.beginPath()
            ctx.arc(x, y, nodeSize, 0, 2 * Math.PI)
            ctx.fill()
            
            // Border for arbitrage opportunities
            if (this.hasArbitrageOpportunity(asset.symbol)) {
                ctx.strokeStyle = '#ffa502'
                ctx.lineWidth = 2
                ctx.stroke()
            }
            
            // Asset symbol text
            ctx.fillStyle = '#ffffff'
            ctx.font = 'bold 10px monospace'
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'
            ctx.fillText(asset.symbol, x, y)
        })
    }

    drawClusterBoundaries(ctx, centerX, centerY, radius) {
        // Draw transparent regions for different asset classes
        const cryptoAssets = this.clusteringData.assets.filter(a => ['BTC', 'ETH', 'SOL'].includes(a.symbol))
        
        if (cryptoAssets.length > 0) {
            // Calculate crypto cluster boundary
            const avgX = cryptoAssets.reduce((sum, a) => sum + a.x, 0) / cryptoAssets.length
            const avgY = cryptoAssets.reduce((sum, a) => sum + a.y, 0) / cryptoAssets.length
            const avgDistance = cryptoAssets.reduce((sum, a) => sum + Math.sqrt(a.x*a.x + a.y*a.y), 0) / cryptoAssets.length
            
            // Draw cluster region
            ctx.fillStyle = 'rgba(0, 212, 170, 0.1)'
            ctx.beginPath()
            ctx.arc(centerX + avgX * radius, centerY + avgY * radius, avgDistance * radius * 1.2, 0, 2 * Math.PI)
            ctx.fill()
            
            // Cluster label
            ctx.fillStyle = '#00d4aa'
            ctx.font = '8px monospace'
            ctx.textAlign = 'center'
            ctx.fillText('CRYPTO', centerX + avgX * radius, centerY + avgY * radius - avgDistance * radius * 1.5)
        }
    }

    updateClusteringMetrics() {
        if (!this.clusteringData) return
        
        // Calculate average correlation
        let totalCorrelation = 0
        let correlationCount = 0
        
        this.clusteringData.assets.forEach(asset1 => {
            this.clusteringData.assets.forEach(asset2 => {
                if (asset1.symbol !== asset2.symbol) {
                    totalCorrelation += Math.abs(asset1.correlations[asset2.symbol] || 0)
                    correlationCount++
                }
            })
        })
        
        const avgCorrelation = correlationCount > 0 ? totalCorrelation / correlationCount : 0
        
        // Update UI elements
        document.getElementById('avg-correlation').textContent = avgCorrelation.toFixed(3)
        
        // Cluster stability (based on correlation variance)
        const stability = avgCorrelation > 0.6 ? 'High' : avgCorrelation > 0.3 ? 'Medium' : 'Low'
        const stabilityElement = document.getElementById('cluster-stability')
        stabilityElement.textContent = stability
        stabilityElement.className = `text-${stability === 'High' ? 'profit' : stability === 'Medium' ? 'warning' : 'loss'}`
    }

    updateAssetLegend() {
        if (!this.clusteringData) return
        
        const legendContainer = document.getElementById('asset-legend')
        if (!legendContainer) return
        
        const legendHtml = this.clusteringData.assets.map(asset => {
            const priceChangePercent = (asset.priceChange * 100).toFixed(2)
            const changeColor = asset.priceChange > 0 ? 'text-profit' : asset.priceChange < 0 ? 'text-loss' : 'text-gray-400'
            
            return `
                <div class="flex items-center justify-between py-1">
                    <div class="flex items-center">
                        <div class="w-2 h-2 rounded-full mr-2" style="background-color: ${
                            asset.priceChange > 0.01 ? '#2ed573' : 
                            asset.priceChange < -0.01 ? '#ff4757' : '#4ecdc4'
                        }"></div>
                        <span class="font-semibold">${asset.symbol}</span>
                    </div>
                    <span class="${changeColor}">${priceChangePercent > 0 ? '+' : ''}${priceChangePercent}%</span>
                </div>
            `
        }).join('')
        
        legendContainer.innerHTML = legendHtml
    }

    hasArbitrageOpportunity(symbol) {
        // Check if asset has current arbitrage opportunities
        // This would integrate with existing arbitrage detection
        return Math.random() > 0.7 // Simplified for demo
    }

    startAssetClusteringUpdates() {
        if (this.clusteringAnimation) {
            clearInterval(this.clusteringAnimation)
        }
        
        // Update clustering visualization every 5 seconds
        this.clusteringAnimation = setInterval(() => {
            if (this.currentVisualization === 'clustering') {
                this.loadClusteringData()
            }
        }, 5000)
    }

    stopAssetClusteringUpdates() {
        if (this.clusteringAnimation) {
            clearInterval(this.clusteringAnimation)
            this.clusteringAnimation = null
        }
    }

    async executeArbitrage(opportunity) {
        try {
            const response = await axios.post('/api/execute-arbitrage', opportunity)
            const result = response.data
            
            if (result.success) {
                this.showNotification(`✅ Arbitrage executed successfully! Transaction ID: ${result.transactionId}`, 'success')
            } else {
                this.showNotification(`❌ Execution failed: ${result.message}`, 'error')
            }
            
        } catch (error) {
            console.error('Error executing arbitrage:', error)
            this.showNotification('❌ Network error during execution', 'error')
        }
    }

    async sendChatMessage() {
        const input = document.getElementById('chat-input')
        const message = input.value.trim()
        
        if (!message) return
        
        // Add user message to chat
        this.addChatMessage(message, 'user')
        input.value = ''
        
        try {
            const response = await axios.post('/api/ai-query', { query: message })
            const result = response.data
            
            // Add AI response to chat
            this.addChatMessage(result.response, 'ai', result.confidence)
            
        } catch (error) {
            console.error('Error sending chat message:', error)
            this.addChatMessage('Sorry, I encountered an error processing your request.', 'ai')
        }
    }

    addChatMessage(message, sender, confidence = null) {
        const container = document.getElementById('chat-container')
        const messageDiv = document.createElement('div')
        messageDiv.className = `chat-message ${sender}-message mb-4`
        
        const senderName = sender === 'ai' ? 'GOMNA AI' : 'You'
        const senderClass = sender === 'ai' ? 'text-accent' : 'text-white'
        const confidenceText = confidence ? ` (${confidence}% confidence)` : ''
        
        messageDiv.innerHTML = `
            <div class="font-semibold ${senderClass} mb-1">${senderName}${confidenceText}</div>
            <div>${message}</div>
        `
        
        container.appendChild(messageDiv)
        container.scrollTop = container.scrollHeight
    }

    // Advanced Hyperbolic CNN Chart Analysis System
    setupChartControls() {
        // Symbol selection buttons
        const symbolButtons = document.querySelectorAll('.symbol-btn')
        symbolButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                symbolButtons.forEach(b => {
                    b.classList.remove('active', 'bg-accent', 'text-dark-bg')
                    b.classList.add('bg-gray-700', 'text-white')
                })
                e.target.classList.add('active', 'bg-accent', 'text-dark-bg')
                e.target.classList.remove('bg-gray-700', 'text-white')
                
                this.currentSymbol = e.target.dataset.symbol
                this.loadCandlestickData()
            })
        })

        // Timeframe selection buttons
        const timeframeButtons = document.querySelectorAll('.timeframe-btn')
        timeframeButtons.forEach(btn => {
            btn.addEventListener('click', (e) => {
                timeframeButtons.forEach(b => {
                    b.classList.remove('active', 'bg-accent', 'text-dark-bg')
                    b.classList.add('bg-gray-700', 'text-white')
                })
                e.target.classList.add('active', 'bg-accent', 'text-dark-bg')
                e.target.classList.remove('bg-gray-700', 'text-white')
                
                this.currentTimeframe = e.target.dataset.timeframe
                this.loadCandlestickData()
            })
        })

        // Pattern analysis button
        const analyzeButton = document.getElementById('analyze-chart')
        if (analyzeButton) {
            analyzeButton.addEventListener('click', () => {
                this.performHyperbolicAnalysis()
            })
        }

        // Pattern-based arbitrage execution
        const executePatternButton = document.getElementById('execute-pattern-arbitrage')
        if (executePatternButton) {
            executePatternButton.addEventListener('click', () => {
                this.executePatternBasedArbitrage()
            })
        }
    }

    async initializeCandlestickChart() {
        const canvas = document.getElementById('candlestick-chart')
        if (!canvas) return

        const ctx = canvas.getContext('2d')
        
        // Initialize with empty chart
        this.candlestickChart = new Chart(ctx, {
            type: 'candlestick',
            data: {
                datasets: [{
                    label: `${this.currentSymbol}/USD`,
                    data: [],
                    borderColor: '#00d4aa',
                    backgroundColor: 'rgba(0, 212, 170, 0.1)',
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        type: 'time',
                        time: {
                            unit: 'minute'
                        },
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#ffffff'
                        }
                    },
                    y: {
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: '#ffffff'
                        }
                    }
                },
                plugins: {
                    legend: {
                        labels: {
                            color: '#ffffff'
                        }
                    },
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                        backgroundColor: 'rgba(26, 31, 41, 0.9)',
                        titleColor: '#00d4aa',
                        bodyColor: '#ffffff',
                        borderColor: '#00d4aa',
                        borderWidth: 1
                    }
                },
                interaction: {
                    mode: 'nearest',
                    axis: 'x',
                    intersect: false
                }
            }
        })

        // Load initial data
        await this.loadCandlestickData()
    }

    async loadCandlestickData() {
        try {
            const response = await axios.get(`/api/candlestick/${this.currentSymbol}/${this.currentTimeframe}`)
            const data = response.data

            // Convert data to Chart.js candlestick format
            const chartData = data.data.map(candle => ({
                x: candle.timestamp,
                o: candle.open,
                h: candle.high,
                l: candle.low,
                c: candle.close
            }))

            if (this.candlestickChart) {
                this.candlestickChart.data.datasets[0].data = chartData
                this.candlestickChart.data.datasets[0].label = `${this.currentSymbol}/USD (${this.currentTimeframe})`
                this.candlestickChart.update('none')
            }

            // Trigger automatic pattern analysis
            setTimeout(() => this.performHyperbolicAnalysis(), 1000)

        } catch (error) {
            console.error('Error loading candlestick data:', error)
        }
    }

    async performHyperbolicAnalysis() {
        try {
            const response = await axios.get(`/api/pattern-analysis/${this.currentSymbol}/${this.currentTimeframe}`)
            const analysis = response.data

            this.updatePatternAnalysisUI(analysis)
            this.checkPatternAlerts(analysis)

        } catch (error) {
            console.error('Error performing hyperbolic analysis:', error)
        }
    }

    updatePatternAnalysisUI(analysis) {
        const { pattern, arbitrageTiming } = analysis

        // Update pattern analysis panel
        document.getElementById('detected-pattern').textContent = pattern.pattern.replace(/_/g, ' ').toUpperCase()
        document.getElementById('pattern-confidence').textContent = `${pattern.confidence}%`
        document.getElementById('pattern-signal').textContent = pattern.signal.replace(/_/g, ' ').toUpperCase()
        document.getElementById('arbitrage-relevance').textContent = `${pattern.arbitrageRelevance}%`

        // Update hyperbolic metrics
        document.getElementById('geodesic-efficiency').textContent = `${pattern.geodesicEfficiency}%`
        document.getElementById('hyperbolic-distance').textContent = pattern.hyperbolicDistance

        // Update arbitrage timing
        document.getElementById('timing-action').textContent = arbitrageTiming.timing.toUpperCase()
        document.getElementById('optimal-entry').textContent = arbitrageTiming.optimalEntry || 'N/A'
        document.getElementById('risk-level').textContent = arbitrageTiming.riskLevel.toUpperCase()
        document.getElementById('timing-recommendation').textContent = arbitrageTiming.recommendation

        // Update colors based on signal
        const signalElement = document.getElementById('pattern-signal')
        const actionElement = document.getElementById('timing-action')
        
        signalElement.className = pattern.signal.includes('bullish') ? 'text-profit' : 
                                 pattern.signal.includes('bearish') ? 'text-loss' : 'text-warning'
        
        actionElement.className = arbitrageTiming.timing === 'buy' ? 'text-profit' :
                                 arbitrageTiming.timing === 'sell' ? 'text-loss' : 'text-warning'

        // Enable/disable execution button
        const executeButton = document.getElementById('execute-pattern-arbitrage')
        if (pattern.arbitrageRelevance > 75 && pattern.confidence > 80) {
            executeButton.disabled = false
            executeButton.classList.remove('opacity-50')
        } else {
            executeButton.disabled = true
            executeButton.classList.add('opacity-50')
        }

        // Store analysis for later use
        this.hyperbolicAnalysis = analysis
        
        // Check auto trading if enabled
        this.checkAutoTrading()
    }

    checkPatternAlerts(analysis) {
        const { pattern, arbitrageTiming } = analysis
        
        // Generate alert if high-confidence pattern detected
        if (pattern.confidence > 90 && pattern.arbitrageRelevance > 85) {
            const alert = {
                timestamp: new Date().toLocaleTimeString(),
                symbol: this.currentSymbol,
                timeframe: this.currentTimeframe,
                pattern: pattern.pattern,
                confidence: pattern.confidence,
                action: arbitrageTiming.timing,
                message: `High-confidence ${pattern.pattern} detected for ${this.currentSymbol} (${this.currentTimeframe})`
            }
            
            this.addPatternAlert(alert)
        }
    }

    addPatternAlert(alert) {
        this.patternAlerts.unshift(alert)
        if (this.patternAlerts.length > 10) {
            this.patternAlerts.pop()
        }

        const alertsContainer = document.getElementById('pattern-alerts')
        if (alertsContainer) {
            alertsContainer.innerHTML = this.patternAlerts.map(alert => `
                <div class="flex items-center justify-between p-2 bg-gray-900 rounded text-sm">
                    <div class="flex items-center space-x-2">
                        <span class="w-2 h-2 bg-accent rounded-full animate-pulse"></span>
                        <span class="font-semibold">${alert.symbol}</span>
                        <span class="text-gray-400">${alert.timeframe}</span>
                        <span class="text-warning">${alert.pattern.replace(/_/g, ' ')}</span>
                    </div>
                    <div class="flex items-center space-x-2">
                        <span class="text-accent">${alert.confidence}%</span>
                        <span class="text-xs text-gray-400">${alert.timestamp}</span>
                    </div>
                </div>
            `).join('')
        }
    }

    async executePatternBasedArbitrage() {
        if (!this.hyperbolicAnalysis) {
            this.showNotification('No pattern analysis available', 'error')
            return
        }

        try {
            const opportunityData = {
                type: 'Hyperbolic CNN Pattern-Based',
                symbol: this.currentSymbol,
                timeframe: this.currentTimeframe,
                pattern: this.hyperbolicAnalysis.pattern.pattern,
                confidence: this.hyperbolicAnalysis.pattern.confidence,
                arbitrageRelevance: this.hyperbolicAnalysis.pattern.arbitrageRelevance,
                timing: this.hyperbolicAnalysis.arbitrageTiming.timing,
                geodesicEfficiency: this.hyperbolicAnalysis.pattern.geodesicEfficiency
            }

            const response = await axios.post('/api/execute-arbitrage', opportunityData)
            const result = response.data

            if (result.success) {
                this.showNotification(`✅ Pattern-based arbitrage executed! Pattern: ${opportunityData.pattern}, Confidence: ${opportunityData.confidence}%`, 'success')
                
                // Add to pattern alerts
                this.addPatternAlert({
                    timestamp: new Date().toLocaleTimeString(),
                    symbol: this.currentSymbol,
                    timeframe: this.currentTimeframe,
                    pattern: opportunityData.pattern,
                    confidence: opportunityData.confidence,
                    action: 'EXECUTED',
                    message: `Pattern-based arbitrage executed: ${opportunityData.pattern}`
                })
            } else {
                this.showNotification(`❌ Pattern execution failed: ${result.message}`, 'error')
            }

        } catch (error) {
            console.error('Error executing pattern-based arbitrage:', error)
            this.showNotification('❌ Network error during pattern execution', 'error')
        }
    }

    async startHyperbolicAnalysis() {
        // Start continuous hyperbolic analysis updates
        setInterval(async () => {
            if (this.currentSection === 'dashboard') {
                await this.loadCandlestickData()
                
                // Load multi-timeframe analysis
                await this.loadMultiTimeframeAnalysis()
            }
        }, 5000) // Update every 5 seconds for pattern analysis
    }

    async loadMultiTimeframeAnalysis() {
        try {
            const response = await axios.get('/api/hyperbolic-analysis')
            const analysis = response.data

            // Update navigation with pattern confidence indicators
            const symbols = ['BTC', 'ETH', 'SOL']
            symbols.forEach(symbol => {
                const button = document.querySelector(`.symbol-btn[data-symbol="${symbol}"]`)
                if (button && analysis[symbol]) {
                    const maxConfidence = Math.max(
                        ...Object.values(analysis[symbol]).map(tf => tf.confidence)
                    )
                    
                    // Add confidence indicator
                    let indicator = button.querySelector('.confidence-indicator')
                    if (!indicator) {
                        indicator = document.createElement('span')
                        indicator.className = 'confidence-indicator ml-1 w-2 h-2 rounded-full inline-block'
                        button.appendChild(indicator)
                    }
                    
                    if (maxConfidence > 90) {
                        indicator.className = 'confidence-indicator ml-1 w-2 h-2 rounded-full inline-block bg-profit animate-pulse'
                    } else if (maxConfidence > 75) {
                        indicator.className = 'confidence-indicator ml-1 w-2 h-2 rounded-full inline-block bg-warning'
                    } else {
                        indicator.className = 'confidence-indicator ml-1 w-2 h-2 rounded-full inline-block bg-gray-500'
                    }
                }
            })

        } catch (error) {
            console.error('Error loading multi-timeframe analysis:', error)
        }
    }

    async sendChatMessage() {
        const input = document.getElementById('chat-input')
        const message = input.value.trim()
        
        if (!message) return
        
        // Add user message to chat
        this.addChatMessage(message, 'user')
        input.value = ''
        
        try {
            // Include current chart data in query for enhanced AI analysis
            const chartData = this.hyperbolicAnalysis || {}
            
            const response = await axios.post('/api/ai-query', { 
                query: message,
                chartData: chartData
            })
            const result = response.data
            
            // Add AI response to chat with enhanced formatting for chart analysis
            this.addChatMessage(result.response, 'ai', result.confidence)
            
            // If chart data is included, show additional insights
            if (result.patternAnalysis) {
                setTimeout(() => {
                    this.addChatMessage(`📊 **Additional Chart Insights**: The hyperbolic CNN detected ${result.patternAnalysis.pattern} with ${result.patternAnalysis.geodesicEfficiency}% geodesic efficiency. This pattern has ${result.patternAnalysis.arbitrageRelevance}% relevance for arbitrage opportunities.`, 'ai', result.confidence)
                }, 1000)
            }
            
        } catch (error) {
            console.error('Error sending chat message:', error)
            this.addChatMessage('Sorry, I encountered an error processing your request.', 'ai')
        }
    }

    // Advanced Backtesting System
    initializeBacktesting() {
        // Setup backtesting event listeners
        this.setupBacktestingControls()
        this.initializeBacktestCharts()
    }

    setupBacktestingControls() {
        // Run backtest button
        const runBacktestBtn = document.getElementById('run-backtest')
        if (runBacktestBtn) {
            runBacktestBtn.addEventListener('click', () => this.runBacktest())
        }

        // Monte Carlo simulation button
        const runMonteCarloBtn = document.getElementById('run-monte-carlo')
        if (runMonteCarloBtn) {
            runMonteCarloBtn.addEventListener('click', () => this.runMonteCarloSimulation())
        }

        // Compare strategies button
        const compareBtn = document.getElementById('compare-strategies')
        if (compareBtn) {
            compareBtn.addEventListener('click', () => this.compareStrategies())
        }
    }

    initializeBacktestCharts() {
        // Initialize equity curve chart
        const equityCanvas = document.getElementById('equity-curve-chart')
        if (equityCanvas) {
            const ctx = equityCanvas.getContext('2d')
            this.equityCurveChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Equity',
                        data: [],
                        borderColor: '#00d4aa',
                        backgroundColor: 'rgba(0, 212, 170, 0.1)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'time',
                            time: { unit: 'day' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#ffffff' }
                        },
                        y: {
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#ffffff' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#ffffff' } }
                    }
                }
            })
        }

        // Initialize drawdown chart
        const drawdownCanvas = document.getElementById('drawdown-chart')
        if (drawdownCanvas) {
            const ctx = drawdownCanvas.getContext('2d')
            this.drawdownChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Drawdown %',
                        data: [],
                        borderColor: '#ff4757',
                        backgroundColor: 'rgba(255, 71, 87, 0.1)',
                        tension: 0.1,
                        fill: true
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            type: 'time',
                            time: { unit: 'day' },
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#ffffff' }
                        },
                        y: {
                            reverse: true,
                            grid: { color: 'rgba(255, 255, 255, 0.1)' },
                            ticks: { color: '#ffffff' }
                        }
                    },
                    plugins: {
                        legend: { labels: { color: '#ffffff' } }
                    }
                }
            })
        }
    }

    async runBacktest() {
        try {
            const strategyConfig = this.getBacktestConfiguration()
            
            // Show loading state
            const resultsContainer = document.getElementById('backtest-results')
            resultsContainer.innerHTML = `
                <div class="flex items-center justify-center py-8">
                    <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-accent mr-3"></div>
                    <span>Running backtest...</span>
                </div>
            `

            const response = await axios.post('/api/backtest/run', strategyConfig)
            const { results } = response.data

            this.displayBacktestResults(results)
            this.updateBacktestCharts(results)
            
            // Store results
            this.backtests[strategyConfig.strategyId] = results

            this.showNotification(`✅ Backtest completed! Total Return: ${results.metrics.totalReturn.toFixed(2)}%`, 'success')

        } catch (error) {
            console.error('Backtest error:', error)
            this.showNotification('❌ Backtest failed. Please check your parameters.', 'error')
            
            document.getElementById('backtest-results').innerHTML = `
                <div class="text-center text-loss py-8">
                    <i class="fas fa-exclamation-triangle text-2xl mb-2"></i>
                    <div>Backtest failed. Please try again.</div>
                </div>
            `
        }
    }

    getBacktestConfiguration() {
        return {
            strategyId: `backtest_${Date.now()}`,
            strategyName: document.getElementById('strategy-name').value || 'Unnamed Strategy',
            symbol: document.getElementById('backtest-symbol').value,
            timeframe: document.getElementById('backtest-timeframe').value,
            strategyType: document.getElementById('strategy-type').value,
            initialCapital: parseFloat(document.getElementById('initial-capital').value),
            startDate: new Date(Date.now() - 365 * 24 * 60 * 60 * 1000).toISOString(),
            endDate: new Date().toISOString(),
            parameters: {
                minConfidence: parseFloat(document.getElementById('min-confidence').value),
                riskPerTrade: parseFloat(document.getElementById('risk-per-trade').value) / 100,
                stopLoss: parseFloat(document.getElementById('stop-loss').value) / 100,
                takeProfit: parseFloat(document.getElementById('take-profit').value) / 100,
                minArbitrageRelevance: parseFloat(document.getElementById('min-arbitrage-relevance').value)
            }
        }
    }

    displayBacktestResults(results) {
        const { metrics, trades } = results
        
        const resultsHtml = `
            <div class="grid grid-cols-4 gap-4 mb-6">
                <div class="text-center">
                    <div class="text-2xl font-bold ${metrics.totalReturn > 0 ? 'text-profit' : 'text-loss'}">
                        ${metrics.totalReturn.toFixed(2)}%
                    </div>
                    <div class="text-sm text-gray-400">Total Return</div>
                </div>
                <div class="text-center">
                    <div class="text-2xl font-bold text-accent">${metrics.winRate.toFixed(1)}%</div>
                    <div class="text-sm text-gray-400">Win Rate</div>
                </div>
                <div class="text-center">
                    <div class="text-2xl font-bold text-warning">${metrics.sharpeRatio}</div>
                    <div class="text-sm text-gray-400">Sharpe Ratio</div>
                </div>
                <div class="text-center">
                    <div class="text-2xl font-bold text-loss">-${metrics.maxDrawdown.toFixed(2)}%</div>
                    <div class="text-sm text-gray-400">Max Drawdown</div>
                </div>
            </div>
            
            <div class="grid grid-cols-2 gap-6">
                <div>
                    <h5 class="font-semibold mb-3 text-accent">📊 Performance Metrics</h5>
                    <div class="space-y-2 text-sm">
                        <div class="flex justify-between">
                            <span>Initial Capital:</span>
                            <span class="text-accent">$${metrics.initialCapital.toLocaleString()}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Final Capital:</span>
                            <span class="text-accent">$${metrics.finalCapital.toLocaleString()}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Total P&L:</span>
                            <span class="${metrics.totalPnL > 0 ? 'text-profit' : 'text-loss'}">
                                ${metrics.totalPnL > 0 ? '+' : ''}$${metrics.totalPnL.toLocaleString()}
                            </span>
                        </div>
                        <div class="flex justify-between">
                            <span>Total Trades:</span>
                            <span class="text-accent">${metrics.totalTrades}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Profit Factor:</span>
                            <span class="text-accent">${metrics.profitFactor.toFixed(2)}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Calmar Ratio:</span>
                            <span class="text-accent">${metrics.calmarRatio}</span>
                        </div>
                    </div>
                </div>
                
                <div>
                    <h5 class="font-semibold mb-3 text-accent">📈 Trade Analysis</h5>
                    <div class="space-y-2 text-sm">
                        <div class="flex justify-between">
                            <span>Winning Trades:</span>
                            <span class="text-profit">${metrics.winningTrades}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Losing Trades:</span>
                            <span class="text-loss">${metrics.losingTrades}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Avg Trade:</span>
                            <span class="${metrics.avgTrade > 0 ? 'text-profit' : 'text-loss'}">
                                ${metrics.avgTrade > 0 ? '+' : ''}$${metrics.avgTrade.toFixed(2)}
                            </span>
                        </div>
                        <div class="flex justify-between">
                            <span>Avg Winner:</span>
                            <span class="text-profit">+$${metrics.avgWin.toFixed(2)}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Avg Loser:</span>
                            <span class="text-loss">-$${metrics.avgLoss.toFixed(2)}</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Best Trade:</span>
                            <span class="text-profit">
                                +$${Math.max(...trades.map(t => t.pnl), 0).toFixed(2)}
                            </span>
                        </div>
                        <div class="flex justify-between">
                            <span>Worst Trade:</span>
                            <span class="text-loss">
                                $${Math.min(...trades.map(t => t.pnl), 0).toFixed(2)}
                            </span>
                        </div>
                    </div>
                </div>
            </div>
        `
        
        document.getElementById('backtest-results').innerHTML = resultsHtml
    }

    updateBacktestCharts(results) {
        // Update equity curve
        if (this.equityCurveChart && results.equity) {
            const equityData = results.equity.slice(0, -1) // Skip last incomplete point
            this.equityCurveChart.data.labels = equityData.map(point => new Date(point.timestamp))
            this.equityCurveChart.data.datasets[0].data = equityData.map(point => point.equity)
            this.equityCurveChart.update()
        }

        // Update drawdown chart
        if (this.drawdownChart && results.drawdowns) {
            const drawdownData = results.drawdowns.slice(0, -1)
            this.drawdownChart.data.labels = drawdownData.map(point => new Date(point.timestamp))
            this.drawdownChart.data.datasets[0].data = drawdownData.map(point => point.drawdown)
            this.drawdownChart.update()
        }
    }

    async runMonteCarloSimulation() {
        try {
            const strategyConfig = this.getBacktestConfiguration()
            
            this.showNotification('🎲 Running Monte Carlo simulation...', 'info')
            
            const response = await axios.post('/api/monte-carlo', {
                strategyConfig,
                iterations: 1000
            })
            
            const simulation = response.data.simulation
            
            this.showNotification(
                `✅ Monte Carlo completed! Avg Return: ${simulation.summary.avgReturn.toFixed(2)}%, ` +
                `Profit Probability: ${simulation.summary.profitProbability.toFixed(1)}%`, 
                'success'
            )
            
            // Display Monte Carlo results
            this.displayMonteCarloResults(simulation)
            
        } catch (error) {
            console.error('Monte Carlo error:', error)
            this.showNotification('❌ Monte Carlo simulation failed', 'error')
        }
    }

    displayMonteCarloResults(simulation) {
        const { summary } = simulation
        
        const monteCarloHtml = `
            <div class="mt-6 bg-gray-900 rounded-lg p-4">
                <h5 class="font-semibold mb-3 text-purple-400">🎲 Monte Carlo Simulation Results</h5>
                <div class="grid grid-cols-3 gap-4">
                    <div class="text-center">
                        <div class="text-lg font-bold text-accent">${summary.avgReturn.toFixed(2)}%</div>
                        <div class="text-xs text-gray-400">Average Return</div>
                    </div>
                    <div class="text-center">
                        <div class="text-lg font-bold text-profit">${summary.profitProbability.toFixed(1)}%</div>
                        <div class="text-xs text-gray-400">Profit Probability</div>
                    </div>
                    <div class="text-center">
                        <div class="text-lg font-bold text-warning">${summary.stdReturn.toFixed(2)}%</div>
                        <div class="text-xs text-gray-400">Return Std Dev</div>
                    </div>
                </div>
                <div class="mt-4 grid grid-cols-2 gap-4 text-sm">
                    <div class="space-y-1">
                        <div class="flex justify-between">
                            <span>Best Case:</span>
                            <span class="text-profit">+${summary.maxReturn.toFixed(2)}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Worst Case:</span>
                            <span class="text-loss">${summary.minReturn.toFixed(2)}%</span>
                        </div>
                    </div>
                    <div class="space-y-1">
                        <div class="flex justify-between">
                            <span>Median Return:</span>
                            <span class="text-accent">${summary.medianReturn.toFixed(2)}%</span>
                        </div>
                        <div class="flex justify-between">
                            <span>Avg Drawdown:</span>
                            <span class="text-loss">-${summary.avgDrawdown.toFixed(2)}%</span>
                        </div>
                    </div>
                </div>
            </div>
        `
        
        document.getElementById('backtest-results').innerHTML += monteCarloHtml
    }

    async compareStrategies() {
        // This would implement strategy comparison logic
        this.showNotification('🔄 Strategy comparison feature coming soon!', 'info')
    }

    // Paper Trading System
    initializePaperTrading() {
        this.setupPaperTradingControls()
        this.loadPaperAccounts()
    }

    setupPaperTradingControls() {
        // Create account button
        const createAccountBtn = document.getElementById('create-paper-account')
        if (createAccountBtn) {
            createAccountBtn.addEventListener('click', () => this.createPaperAccount())
        }

        // Place order button
        const placeOrderBtn = document.getElementById('place-paper-order')
        if (placeOrderBtn) {
            placeOrderBtn.addEventListener('click', () => this.placePaperOrder())
        }

        // Order type change listener
        const orderTypeSelect = document.getElementById('paper-order-type')
        if (orderTypeSelect) {
            orderTypeSelect.addEventListener('change', (e) => {
                const limitPriceContainer = document.getElementById('limit-price-container')
                if (e.target.value === 'LIMIT') {
                    limitPriceContainer.classList.remove('hidden')
                } else {
                    limitPriceContainer.classList.add('hidden')
                }
            })
        }

        // Auto trading toggle
        const autoTradingToggle = document.getElementById('auto-trading-toggle')
        if (autoTradingToggle) {
            autoTradingToggle.addEventListener('change', (e) => {
                this.autoTradingEnabled = e.target.checked
                const status = document.getElementById('auto-trading-status')
                if (status) {
                    status.textContent = this.autoTradingEnabled 
                        ? 'Auto trading enabled - monitoring patterns...'
                        : 'Auto trading disabled'
                    status.className = this.autoTradingEnabled 
                        ? 'text-xs text-profit' 
                        : 'text-xs text-gray-400'
                }
            })
        }
    }

    async createPaperAccount() {
        try {
            const accountName = document.getElementById('paper-account-name').value || 'Default Account'
            const initialBalance = parseFloat(document.getElementById('paper-initial-balance').value) || 100000
            
            const response = await axios.post('/api/paper-trading/account', {
                accountId: `account_${Date.now()}`,
                initialBalance
            })
            
            this.paperAccount = response.data.account
            this.updatePaperAccountUI()
            this.showNotification(`✅ Paper trading account created with $${initialBalance.toLocaleString()}`, 'success')
            
        } catch (error) {
            console.error('Account creation error:', error)
            this.showNotification('❌ Failed to create paper trading account', 'error')
        }
    }

    async placePaperOrder() {
        try {
            if (!this.paperAccount) {
                this.showNotification('❌ Please create a paper trading account first', 'error')
                return
            }

            const orderData = {
                accountId: this.paperAccount.accountId,
                symbol: document.getElementById('paper-symbol').value,
                side: document.getElementById('paper-side').value,
                quantity: parseFloat(document.getElementById('paper-quantity').value),
                orderType: document.getElementById('paper-order-type').value,
                price: document.getElementById('paper-order-type').value === 'LIMIT' 
                    ? parseFloat(document.getElementById('paper-limit-price').value) 
                    : null,
                stopLoss: parseFloat(document.getElementById('paper-stop-loss').value) || null,
                takeProfit: parseFloat(document.getElementById('paper-take-profit').value) || null
            }

            const response = await axios.post('/api/paper-trading/order', orderData)
            const order = response.data.order
            
            if (order.status === 'EXECUTED') {
                this.showNotification(`✅ Order executed: ${order.side} ${order.quantity} ${order.symbol} at $${order.executedPrice}`, 'success')
            } else if (order.status === 'REJECTED') {
                this.showNotification(`❌ Order rejected: ${order.rejectionReason}`, 'error')
            } else {
                this.showNotification(`📋 Order placed: ${order.side} ${order.quantity} ${order.symbol}`, 'info')
            }
            
            this.refreshPaperAccount()
            
        } catch (error) {
            console.error('Order placement error:', error)
            this.showNotification('❌ Failed to place order', 'error')
        }
    }

    async refreshPaperAccount() {
        if (!this.paperAccount) return
        
        try {
            const response = await axios.get(`/api/paper-trading/account/${this.paperAccount.accountId}`)
            this.paperAccount = response.data.account
            this.updatePaperAccountUI()
        } catch (error) {
            console.error('Account refresh error:', error)
        }
    }

    updatePaperAccountUI() {
        if (!this.paperAccount) return
        
        // Update account summary
        document.getElementById('paper-balance').textContent = `$${this.paperAccount.balance.toLocaleString()}`
        document.getElementById('paper-equity').textContent = `$${(this.paperAccount.currentValue || this.paperAccount.balance).toLocaleString()}`
        
        const totalPnL = this.paperAccount.metrics.totalPnL
        const pnlElement = document.getElementById('paper-pnl')
        pnlElement.textContent = `${totalPnL >= 0 ? '+' : ''}$${totalPnL.toLocaleString()}`
        pnlElement.className = `text-2xl font-bold ${totalPnL >= 0 ? 'text-profit' : 'text-loss'}`
        
        const totalReturn = this.paperAccount.totalReturn || 0
        const returnElement = document.getElementById('paper-return')
        returnElement.textContent = `${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%`
        returnElement.className = `text-2xl font-bold ${totalReturn >= 0 ? 'text-profit' : 'text-loss'}`
        
        // Update positions
        this.updatePaperPositions()
        
        // Update trade history
        this.updatePaperTradeHistory()
    }

    updatePaperPositions() {
        const positionsContainer = document.getElementById('paper-positions')
        const positions = Object.values(this.paperAccount.positions || {}).filter(pos => pos.quantity > 0)
        
        if (positions.length === 0) {
            positionsContainer.innerHTML = '<div class="text-center text-gray-400 py-4">No positions yet...</div>'
            return
        }
        
        const positionsHtml = `
            <div class="overflow-x-auto">
                <table class="w-full text-sm">
                    <thead>
                        <tr class="border-b border-gray-700">
                            <th class="text-left py-2">Symbol</th>
                            <th class="text-right py-2">Quantity</th>
                            <th class="text-right py-2">Avg Price</th>
                            <th class="text-right py-2">Current Price</th>
                            <th class="text-right py-2">Unrealized P&L</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${positions.map(pos => `
                            <tr class="border-b border-gray-700">
                                <td class="py-2 font-semibold">${pos.symbol}</td>
                                <td class="text-right">${pos.quantity}</td>
                                <td class="text-right">$${pos.avgPrice.toLocaleString()}</td>
                                <td class="text-right">$${this.getCurrentPrice(pos.symbol).toLocaleString()}</td>
                                <td class="text-right ${pos.unrealizedPnL >= 0 ? 'text-profit' : 'text-loss'}">
                                    ${pos.unrealizedPnL >= 0 ? '+' : ''}$${pos.unrealizedPnL.toLocaleString()}
                                </td>
                            </tr>
                        `).join('')}
                    </tbody>
                </table>
            </div>
        `
        
        positionsContainer.innerHTML = positionsHtml
    }

    updatePaperTradeHistory() {
        const historyContainer = document.getElementById('paper-trade-history')
        const trades = this.paperAccount.tradeHistory || []
        
        if (trades.length === 0) {
            historyContainer.innerHTML = '<div class="text-center text-gray-400 py-4">No trades yet...</div>'
            return
        }
        
        const tradesHtml = trades.slice(-10).reverse().map(trade => `
            <div class="flex items-center justify-between p-2 bg-gray-900 rounded mb-2">
                <div class="flex items-center space-x-3">
                    <span class="font-semibold ${trade.side === 'BUY' ? 'text-profit' : 'text-loss'}">
                        ${trade.side}
                    </span>
                    <span>${trade.quantity} ${trade.symbol}</span>
                    <span class="text-gray-400">@$${trade.executedPrice}</span>
                </div>
                <div class="text-right text-sm">
                    <div class="text-gray-400">${new Date(trade.executedAt).toLocaleTimeString()}</div>
                    ${trade.realizedPnL ? `<div class="${trade.realizedPnL >= 0 ? 'text-profit' : 'text-loss'}">
                        ${trade.realizedPnL >= 0 ? '+' : ''}$${trade.realizedPnL.toFixed(2)}
                    </div>` : ''}
                </div>
            </div>
        `).join('')
        
        historyContainer.innerHTML = tradesHtml
    }

    getCurrentPrice(symbol) {
        // This would get current market price
        const basePrice = { BTC: 67234.56, ETH: 3456.08, SOL: 123.45 }
        return basePrice[symbol] || 0
    }

    async loadPaperAccounts() {
        try {
            const response = await axios.get('/api/paper-trading/accounts')
            const accounts = response.data.accounts
            
            if (accounts.length > 0) {
                this.paperAccount = accounts[0] // Load first account by default
                this.updatePaperAccountUI()
            }
        } catch (error) {
            console.error('Loading accounts error:', error)
        }
    }

    // Auto trading based on pattern analysis
    async checkAutoTrading() {
        if (!this.autoTradingEnabled || !this.paperAccount || !this.hyperbolicAnalysis) return
        
        const { pattern, arbitrageTiming } = this.hyperbolicAnalysis
        
        if (pattern && pattern.confidence > 85 && pattern.arbitrageRelevance > 80) {
            if (arbitrageTiming.timing === 'buy') {
                await this.executeAutoTrade('BUY', this.currentSymbol, 0.01)
            } else if (arbitrageTiming.timing === 'sell') {
                await this.executeAutoTrade('SELL', this.currentSymbol, 0.01)
            }
        }
    }

    async executeAutoTrade(side, symbol, quantity) {
        try {
            const response = await axios.post('/api/paper-trading/order', {
                accountId: this.paperAccount.accountId,
                symbol,
                side,
                quantity,
                orderType: 'MARKET'
            })
            
            if (response.data.order.status === 'EXECUTED') {
                this.showNotification(`🤖 Auto trade executed: ${side} ${quantity} ${symbol}`, 'success')
                this.refreshPaperAccount()
            }
        } catch (error) {
            console.error('Auto trade error:', error)
        }
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div')
        notification.className = `fixed top-4 right-4 px-6 py-3 rounded-lg text-white font-semibold z-50 ${
            type === 'success' ? 'bg-profit' : 
            type === 'error' ? 'bg-loss' : 'bg-accent'
        }`
        notification.textContent = message
        
        document.body.appendChild(notification)
        
        setTimeout(() => {
            notification.remove()
        }, 5000)
    }

    // Visualization Toggle Setup
    setupVisualizationToggle() {
        const patternsToggle = document.getElementById('viz-toggle-patterns')
        const clusteringToggle = document.getElementById('viz-toggle-clustering')
        
        if (patternsToggle && clusteringToggle) {
            patternsToggle.addEventListener('click', () => {
                this.switchVisualization('patterns')
            })
            
            clusteringToggle.addEventListener('click', () => {
                this.switchVisualization('clustering')
            })
        }
    }

    switchVisualization(type) {
        this.currentVisualization = type
        
        // Update toggle buttons
        const patternsToggle = document.getElementById('viz-toggle-patterns')
        const clusteringToggle = document.getElementById('viz-toggle-clustering')
        
        if (type === 'patterns') {
            patternsToggle.className = 'px-3 py-1 rounded text-xs font-semibold bg-accent text-dark-bg'
            clusteringToggle.className = 'px-3 py-1 rounded text-xs font-semibold text-gray-300 hover:text-white'
            
            // Show patterns view, hide clustering view
            document.getElementById('poincare-patterns-view').classList.remove('hidden')
            document.getElementById('poincare-clustering-view').classList.add('hidden')
        } else {
            patternsToggle.className = 'px-3 py-1 rounded text-xs font-semibold text-gray-300 hover:text-white'
            clusteringToggle.className = 'px-3 py-1 rounded text-xs font-semibold bg-accent text-dark-bg'
            
            // Show clustering view, hide patterns view
            document.getElementById('poincare-patterns-view').classList.add('hidden')
            document.getElementById('poincare-clustering-view').classList.remove('hidden')
            
            // Initialize clustering visualization
            this.initializeAssetClustering()
        }
    }

    // Asset Clustering Functionality
    async initializeAssetClustering() {
        try {
            const response = await axios.get('/api/asset-clustering')
            this.clusteringData = response.data.clustering
            this.drawAssetClustering()
            
            // Start real-time clustering updates
            if (this.clusteringAnimation) {
                clearInterval(this.clusteringAnimation)
            }
            this.clusteringAnimation = setInterval(() => {
                if (this.currentVisualization === 'clustering') {
                    this.updateAssetClustering()
                }
            }, 2000)
        } catch (error) {
            console.error('Failed to initialize asset clustering:', error)
        }
    }

    async updateAssetClustering() {
        try {
            const response = await axios.get('/api/asset-clustering')
            this.clusteringData = response.data.clustering
            this.drawAssetClustering()
        } catch (error) {
            console.error('Failed to update clustering:', error)
        }
    }

    drawAssetClustering() {
        const canvas = document.getElementById('asset-clustering-disk')
        if (!canvas || !this.clusteringData) return
        
        const ctx = canvas.getContext('2d')
        const centerX = canvas.width / 2
        const centerY = canvas.height / 2
        const radius = Math.min(centerX, centerY) - 20
        
        // Clear canvas
        ctx.clearRect(0, 0, canvas.width, canvas.height)
        
        // Draw Poincaré disk boundary
        ctx.beginPath()
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI)
        ctx.strokeStyle = '#00ff9f'
        ctx.lineWidth = 2
        ctx.stroke()
        
        // Draw correlation lines between assets
        this.drawCorrelationLines(ctx, centerX, centerY, radius)
        
        // Draw asset nodes
        this.drawAssetNodes(ctx, centerX, centerY, radius)
        
        // Update clustering metrics
        this.updateClusteringMetrics()
    }

    drawCorrelationLines(ctx, centerX, centerY, radius) {
        if (!this.clusteringData.assets) return
        
        // Enhanced correlation visualization for 15 assets
        // Only show significant correlations to avoid visual clutter
        const significantThreshold = 0.3
        
        for (let i = 0; i < this.clusteringData.assets.length; i++) {
            const asset1 = this.clusteringData.assets[i]
            
            for (let j = i + 1; j < this.clusteringData.assets.length; j++) {
                const asset2 = this.clusteringData.assets[j]
                const correlation = asset1.correlations ? asset1.correlations[asset2.symbol] : 0
                
                if (!correlation || Math.abs(correlation) < significantThreshold) continue
                
                // Convert coordinates to canvas positions
                const pos1 = this.hyperbolicToCanvas(asset1, centerX, centerY, radius)
                const pos2 = this.hyperbolicToCanvas(asset2, centerX, centerY, radius)
                
                // Calculate correlation strength and visual properties
                const strength = Math.abs(correlation)
                const alpha = Math.min(0.8, strength * 1.5)
                const lineWidth = Math.max(0.5, strength * 3)
                
                // Category-aware correlation coloring
                let correlationColor
                if (asset1.category === asset2.category) {
                    // Intra-category correlations (same asset class)
                    correlationColor = correlation > 0 ? 
                        `rgba(0, 255, 159, ${alpha})` :     // Strong green for positive intra-category
                        `rgba(255, 159, 0, ${alpha})`       // Orange for negative intra-category
                } else {
                    // Inter-category correlations (cross-asset class)
                    correlationColor = correlation > 0 ?
                        `rgba(100, 200, 255, ${alpha})` :   // Blue for positive cross-category  
                        `rgba(255, 71, 87, ${alpha})`       // Red for negative cross-category
                }
                
                // Draw correlation line with gradient effect
                ctx.beginPath()
                ctx.moveTo(pos1.x, pos1.y)
                ctx.lineTo(pos2.x, pos2.y)
                ctx.strokeStyle = correlationColor
                ctx.lineWidth = lineWidth
                
                // Add dash pattern for negative correlations
                if (correlation < 0) {
                    ctx.setLineDash([3, 3])
                } else {
                    ctx.setLineDash([])
                }
                
                ctx.stroke()
                ctx.setLineDash([]) // Reset dash pattern
                
                // Draw correlation strength indicator for very strong correlations
                if (strength > 0.7) {
                    const midX = (pos1.x + pos2.x) / 2
                    const midY = (pos1.y + pos2.y) / 2
                    
                    ctx.fillStyle = '#ffffff'
                    ctx.font = '7px monospace'
                    ctx.textAlign = 'center'
                    ctx.shadowColor = 'rgba(0,0,0,0.8)'
                    ctx.shadowBlur = 1
                    ctx.fillText(correlation.toFixed(2), midX, midY - 2)
                    ctx.shadowBlur = 0
                }
            }
        }
    }

    drawAssetNodes(ctx, centerX, centerY, radius) {
        if (!this.clusteringData.assets) return
        
        // Enhanced visualization for all 15 assets across 5 categories
        const categoryColors = {
            crypto: '#00ff9f',      // Bright Green
            equity: '#3742fa',      // Blue  
            international: '#ff6b35', // Orange
            commodities: '#f39c12',  // Golden
            forex: '#9c59d1'        // Purple
        }
        
        this.clusteringData.assets.forEach(asset => {
            const pos = this.hyperbolicToCanvas(asset, centerX, centerY, radius)
            
            // Dynamic node size based on market cap and fusion signal strength
            const maxMarketCap = Math.max(...this.clusteringData.assets.map(a => a.marketCap || 1000000000))
            const baseSize = 6 + (asset.marketCap / maxMarketCap) * 8
            const fusionMultiplier = 1 + Math.abs(asset.fusionSignal || 0) * 2
            const nodeSize = baseSize * fusionMultiplier
            
            // Category-based coloring with performance overlay
            const categoryColor = categoryColors[asset.category] || '#ffffff'
            
            // Draw main node circle  
            ctx.beginPath()
            ctx.arc(pos.x, pos.y, nodeSize, 0, 2 * Math.PI)
            
            // Performance-based color intensity
            const intensity = asset.priceChange > 0 ? 
                Math.min(1, Math.abs(asset.priceChange) * 50 + 0.3) : 0.8
            
            if (asset.priceChange > 0) {
                ctx.fillStyle = `rgba(0, 255, 159, ${intensity})` // Green for gains
            } else {
                ctx.fillStyle = `rgba(255, 71, 87, ${intensity})` // Red for losses
            }
            
            ctx.fill()
            
            // Category border
            ctx.strokeStyle = categoryColor
            ctx.lineWidth = 2
            ctx.stroke()
            
            // Draw asset symbol
            ctx.fillStyle = '#ffffff'
            ctx.font = `${Math.max(8, nodeSize * 0.7)}px monospace`
            ctx.textAlign = 'center'
            ctx.shadowColor = 'rgba(0,0,0,0.8)'
            ctx.shadowBlur = 2
            ctx.fillText(asset.symbol, pos.x, pos.y + 2)
            ctx.shadowBlur = 0
            
            // Fusion signal indicator (pulsing ring)
            if (Math.abs(asset.fusionSignal || 0) > 0.1) {
                const pulseRadius = nodeSize + 3 + Math.sin(Date.now() * 0.005) * 2
                ctx.beginPath()
                ctx.arc(pos.x, pos.y, pulseRadius, 0, 2 * Math.PI)
                ctx.strokeStyle = `rgba(255, 255, 255, ${Math.abs(asset.fusionSignal) * 0.8})`
                ctx.lineWidth = 1
                ctx.stroke()
            }
            
            // Volatility ring (outer)
            if (asset.volatility > 0.005) {
                ctx.beginPath()
                ctx.arc(pos.x, pos.y, nodeSize + 6, 0, 2 * Math.PI)
                ctx.strokeStyle = `rgba(255, 255, 255, ${Math.min(asset.volatility * 20, 0.6)})`
                ctx.lineWidth = 1
                ctx.setLineDash([2, 2])
                ctx.stroke()
                ctx.setLineDash([])
            }
        })
    }

    hyperbolicToCanvas(position, centerX, centerY, radius) {
        // Scale hyperbolic coordinates to canvas
        const scaledX = position.x * radius * 0.8
        const scaledY = position.y * radius * 0.8
        
        return {
            x: centerX + scaledX,
            y: centerY + scaledY
        }
    }

    updateClusteringMetrics() {
        if (!this.clusteringData) return
        
        const { assets, fusionComponents, assetCategories, totalAssets } = this.clusteringData
        
        // Enhanced multi-modal metrics
        const assetCount = totalAssets || assets.length
        
        // Calculate fusion-weighted correlation statistics
        const correlations = []
        const categoryCorrelations = {}
        
        assetCategories.forEach(cat => {
            categoryCorrelations[cat] = []
        })
        
        assets.forEach(asset => {
            if (asset.correlations) {
                Object.values(asset.correlations).forEach(corr => {
                    if (corr !== 1 && !isNaN(corr)) {
                        correlations.push(corr)
                    }
                })
                
                // Category-specific correlations
                if (categoryCorrelations[asset.category]) {
                    assets.filter(a => a.category === asset.category && a.symbol !== asset.symbol).forEach(sameCategory => {
                        const corr = asset.correlations[sameCategory.symbol]
                        if (corr !== undefined && !isNaN(corr)) {
                            categoryCorrelations[asset.category].push(corr)
                        }
                    })
                }
            }
        })
        
        // Advanced metrics calculation
        const avgCorrelation = correlations.length > 0 ? correlations.reduce((a, b) => a + b) / correlations.length : 0
        const correlationVariance = correlations.length > 0 ? 
            correlations.reduce((sum, corr) => sum + Math.pow(corr - avgCorrelation, 2), 0) / correlations.length : 0
        
        // Multi-modal clustering coefficient
        const fusionScore = fusionComponents ? 
            (fusionComponents.hyperbolicCNN + fusionComponents.finBERT + fusionComponents.lstmTransformer) : 0.85
        const clusteringCoefficient = Math.max(0, 1 - (correlationVariance / 0.3)) * fusionScore
        
        // Calculate category diversity (how well distributed across categories)
        const categoryDistribution = {}
        assets.forEach(asset => {
            categoryDistribution[asset.category] = (categoryDistribution[asset.category] || 0) + 1
        })
        const diversityScore = Math.min(Object.keys(categoryDistribution).length / 5, 1) // Max 5 categories
        
        // Update UI elements with enhanced metrics
        document.getElementById('cluster-asset-count').textContent = `${assetCount} (${assetCategories.length} categories)`
        document.getElementById('correlation-variance').textContent = correlationVariance.toFixed(4)
        document.getElementById('clustering-coefficient').textContent = `${clusteringCoefficient.toFixed(3)} (fusion-weighted)`
        
        // Add fusion component display if elements exist
        const fusionDisplay = document.getElementById('fusion-components')
        if (fusionDisplay && fusionComponents) {
            fusionDisplay.innerHTML = `
                <div class="text-xs">
                    <div>CNN: ${(fusionComponents.hyperbolicCNN * 100).toFixed(0)}%</div>
                    <div>LSTM: ${(fusionComponents.lstmTransformer * 100).toFixed(0)}%</div>
                    <div>FinBERT: ${(fusionComponents.finBERT * 100).toFixed(0)}%</div>
                    <div>Arbitrage: ${(fusionComponents.classicalArbitrage * 100).toFixed(0)}%</div>
                </div>
            `
        }
        
        // Update diversity metrics if element exists
        const diversityDisplay = document.getElementById('category-diversity')
        if (diversityDisplay) {
            diversityDisplay.textContent = `${(diversityScore * 100).toFixed(1)}%`
        }
    }
}

// Initialize the dashboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new TradingDashboard()
})