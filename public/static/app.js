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
                this.loadSocialSentiment()
                this.loadEconomicIndicators()
                // Always update clustering data and metrics for real-time display
                this.loadClusteringData()
                // Force update clustering metrics even if not in clustering view
                this.forceClusteringMetricsUpdate()
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
                    this.loadSocialSentiment(),
                    this.loadEconomicIndicators(),
                    this.drawPoincareVisualization(),
                    this.initializeCandlestickChart(),
                    this.startHyperbolicAnalysis()
                ])
                // Initialize clustering for hyperbolic space engine (always load data for metrics)
                this.initializeAssetClustering()
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
                console.log('🧪 Loading backtesting section...')
                // Add small delay to ensure DOM is rendered
                setTimeout(() => {
                    this.initializeBacktesting()
                }, 200)
                break
            case 'paper-trading':
                this.initializePaperTrading()
                break
            case 'economic-data':
                this.initializeEconomicData()
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

    async loadSocialSentiment() {
        try {
            const response = await axios.get('/api/sentiment-summary')
            const data = response.data
            
            const sentimentContainer = document.getElementById('social-sentiment')
            if (sentimentContainer) {
                const getSentimentColor = (score) => {
                    if (score > 70) return 'text-profit'
                    if (score > 30) return 'text-warning' 
                    return 'text-loss'
                }
                
                const getSentimentIcon = (score) => {
                    if (score > 70) return '😄'
                    if (score > 50) return '😐'
                    return '😰'
                }
                
                sentimentContainer.innerHTML = `
                    <div class="space-y-3">
                        <div class="flex justify-between items-center">
                            <span class="font-semibold">Market Mood</span>
                            <span class="${getSentimentColor(data.overall)} font-bold">
                                ${getSentimentIcon(data.overall)} ${data.overall.toFixed(0)}%
                            </span>
                        </div>
                        
                        <div class="bg-gray-800 rounded p-2 space-y-2">
                            ${Object.entries(data.assets).map(([asset, sentiment]) => `
                                <div class="flex justify-between text-sm">
                                    <span>${asset}:</span>
                                    <span class="${getSentimentColor(sentiment)}">${sentiment.toFixed(0)}%</span>
                                </div>
                            `).join('')}
                        </div>
                        
                        <div class="text-xs text-gray-400 space-y-1">
                            <div class="flex justify-between">
                                <span>Fear & Greed:</span>
                                <span class="${getSentimentColor(data.fearGreedIndex)}">${data.fearGreedIndex}</span>
                            </div>
                            <div class="flex justify-between">
                                <span>Social Volume:</span>
                                <span class="text-accent">${(data.socialVolume.twitter / 1000).toFixed(0)}K</span>
                            </div>
                            ${data.trending.twitter.length > 0 ? `
                                <div class="text-warning">🔥 Trending: ${data.trending.twitter.join(', ')}</div>
                            ` : ''}
                        </div>
                    </div>
                `
            }
        } catch (error) {
            console.error('Error loading social sentiment:', error)
        }
    }

    async loadEconomicIndicators() {
        try {
            const response = await axios.get('/api/economic-indicators')
            const data = response.data
            
            const economicContainer = document.getElementById('economic-indicators')
            if (economicContainer) {
                const formatEconValue = (indicator) => {
                    const change = indicator.change || 0
                    const changeColor = change > 0 ? 'text-profit' : change < 0 ? 'text-loss' : 'text-gray-400'
                    const changeIcon = change > 0 ? '↗️' : change < 0 ? '↘️' : '→'
                    return `
                        <span class="text-accent">${indicator.current.toFixed(2)}</span>
                        <span class="${changeColor} text-xs ml-1">${changeIcon}${Math.abs(change).toFixed(2)}</span>
                    `
                }
                
                economicContainer.innerHTML = `
                    <div class="space-y-2 text-xs">
                        <div class="flex justify-between">
                            <span>US GDP:</span>
                            ${formatEconValue(data.us.gdp)}%
                        </div>
                        <div class="flex justify-between">
                            <span>Inflation:</span>
                            ${formatEconValue(data.us.inflation)}%
                        </div>
                        <div class="flex justify-between">
                            <span>Fed Rate:</span>
                            ${formatEconValue(data.us.interestRate)}%
                        </div>
                        <div class="flex justify-between">
                            <span>Unemployment:</span>
                            ${formatEconValue(data.us.unemployment)}%
                        </div>
                        <div class="flex justify-between">
                            <span>DXY Index:</span>
                            ${formatEconValue(data.us.dollarIndex)}
                        </div>
                        <div class="border-t border-gray-700 pt-2 mt-2">
                            <div class="flex justify-between">
                                <span>BTC Dominance:</span>
                                <span class="text-accent">${data.crypto.bitcoinDominance.current.toFixed(1)}%</span>
                            </div>
                            <div class="flex justify-between">
                                <span>Total Market Cap:</span>
                                <span class="text-accent">$${data.crypto.totalMarketCap.current.toFixed(2)}T</span>
                            </div>
                            <div class="flex justify-between">
                                <span>DeFi TVL:</span>
                                <span class="text-accent">$${data.crypto.defiTvl.current.toFixed(1)}B</span>
                            </div>
                        </div>
                    </div>
                `
            }
        } catch (error) {
            console.error('Error loading economic indicators:', error)
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
        if (!canvas) return
        
        const ctx = canvas.getContext('2d')
        const centerX = canvas.width / 2
        const centerY = canvas.height / 2
        const radius = Math.min(canvas.width, canvas.height) * 0.4 // Scale with canvas size
        const currentTime = Date.now()
        
        // Clear canvas with subtle background
        ctx.fillStyle = 'rgba(15, 20, 25, 0.95)'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        
        // Enhanced Poincaré disk boundary with glow
        const boundaryGradient = ctx.createRadialGradient(centerX, centerY, radius - 5, centerX, centerY, radius + 10)
        boundaryGradient.addColorStop(0, 'transparent')
        boundaryGradient.addColorStop(0.8, 'rgba(0, 212, 170, 0.3)')
        boundaryGradient.addColorStop(1, 'rgba(0, 212, 170, 0.8)')
        
        ctx.fillStyle = boundaryGradient
        ctx.beginPath()
        ctx.arc(centerX, centerY, radius + 5, 0, 2 * Math.PI)
        ctx.fill()
        
        // Main disk boundary
        ctx.strokeStyle = '#00d4aa'
        ctx.lineWidth = 3
        ctx.shadowColor = '#00d4aa'
        ctx.shadowBlur = 8
        ctx.beginPath()
        ctx.arc(centerX, centerY, radius, 0, 2 * Math.PI)
        ctx.stroke()
        ctx.shadowBlur = 0
        
        // Enhanced animated geodesic paths with pattern recognition
        const patternData = this.hyperbolicAnalysis || {}
        const confidence = patternData.pattern?.confidence || 75
        const pathCount = Math.floor(15 + (confidence / 100) * 10)
        
        for (let i = 0; i < pathCount; i++) {
            const animationOffset = (currentTime + i * 100) / 2000
            const baseAngle = (Math.PI * 2 * i) / pathCount
            const angle1 = baseAngle + Math.sin(animationOffset) * 0.2
            const angle2 = angle1 + Math.PI / 3 + Math.cos(animationOffset * 1.5) * 0.1
            
            // Dynamic radius based on pattern strength
            const radius1 = radius * (0.7 + Math.sin(animationOffset + i) * 0.2)
            const radius2 = radius * (0.5 + Math.cos(animationOffset * 1.2 + i) * 0.2)
            
            const x1 = centerX + Math.cos(angle1) * radius1
            const y1 = centerY + Math.sin(angle1) * radius1
            const x2 = centerX + Math.cos(angle2) * radius2
            const y2 = centerY + Math.sin(angle2) * radius2
            
            // Color based on pattern confidence and distance from center
            const distance = Math.sqrt((x1 - centerX) ** 2 + (y1 - centerY) ** 2) / radius
            const intensity = 1 - distance
            const alpha = 0.3 + intensity * 0.4 + Math.sin(animationOffset + i) * 0.2
            
            const pathGradient = ctx.createLinearGradient(x1, y1, x2, y2)
            pathGradient.addColorStop(0, `rgba(78, 205, 196, ${alpha})`)
            pathGradient.addColorStop(0.5, `rgba(0, 212, 170, ${alpha * 1.2})`)
            pathGradient.addColorStop(1, `rgba(78, 205, 196, ${alpha})`)
            
            ctx.strokeStyle = pathGradient
            ctx.lineWidth = 1 + intensity
            ctx.globalAlpha = alpha
            
            ctx.beginPath()
            ctx.moveTo(x1, y1)
            
            // Curved geodesic path
            const controlX = centerX + (x1 - centerX + x2 - centerX) * 0.3
            const controlY = centerY + (y1 - centerY + y2 - centerY) * 0.3
            ctx.quadraticCurveTo(controlX, controlY, x2, y2)
            ctx.stroke()
        }
        
        ctx.globalAlpha = 1.0
        
        // Enhanced dynamic data points representing market patterns
        const pointCount = Math.floor(30 + (confidence / 100) * 20)
        
        for (let i = 0; i < pointCount; i++) {
            const animationPhase = (currentTime + i * 50) / 1500
            const baseAngle = (Math.PI * 2 * i) / pointCount + Math.sin(animationPhase) * 0.5
            const baseRadius = (0.3 + Math.sin(animationPhase + i) * 0.4) * radius
            
            const x = centerX + Math.cos(baseAngle) * baseRadius
            const y = centerY + Math.sin(baseAngle) * baseRadius
            
            // Point size and color based on position and pattern strength
            const distanceFromCenter = Math.sqrt((x - centerX) ** 2 + (y - centerY) ** 2)
            const normalizedDistance = distanceFromCenter / radius
            
            const pointSize = 2 + (1 - normalizedDistance) * 3 + Math.sin(animationPhase * 2 + i) * 1
            
            // Color coding for different pattern types
            let pointColor
            if (normalizedDistance < 0.3) {
                pointColor = '#ff6b6b'  // Core pattern (red)
            } else if (normalizedDistance < 0.6) {
                pointColor = '#ffa502'  // Mid pattern (orange)
            } else {
                pointColor = '#4ecdc4'  // Edge pattern (teal)
            }
            
            // Pulsing glow effect
            const pulse = Math.sin(animationPhase * 3 + i) * 0.3 + 0.7
            
            // Draw point glow
            const pointGradient = ctx.createRadialGradient(x, y, 0, x, y, pointSize * 2)
            pointGradient.addColorStop(0, pointColor)
            pointGradient.addColorStop(1, 'transparent')
            
            ctx.fillStyle = pointGradient
            ctx.globalAlpha = pulse * 0.6
            ctx.beginPath()
            ctx.arc(x, y, pointSize * 2, 0, 2 * Math.PI)
            ctx.fill()
            
            // Draw main point
            ctx.fillStyle = pointColor
            ctx.globalAlpha = pulse
            ctx.beginPath()
            ctx.arc(x, y, pointSize, 0, 2 * Math.PI)
            ctx.fill()
        }
        
        ctx.globalAlpha = 1.0
        
        // Add pattern confidence indicator in center
        if (confidence > 0) {
            ctx.fillStyle = `rgba(0, 212, 170, 0.8)`
            ctx.font = 'bold 12px Arial, sans-serif'
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'
            ctx.fillText(`${confidence}%`, centerX, centerY - 8)
            
            ctx.font = '8px Arial, sans-serif'
            ctx.fillText('Confidence', centerX, centerY + 5)
        }
        
        // Continuous animation
        setTimeout(() => this.drawPoincareVisualization(), 50)
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
            
            // Load clustering data immediately when switching to this view
            this.loadClusteringData()
            this.startAssetClusteringUpdates()
        }
    }

    initializeAssetClustering() {
        // Always load clustering data for metrics display
        if (!this.clusteringData) {
            console.log('Initializing clustering data for metrics...')
            this.loadClusteringData()
        }
        
        // Initialize clustering canvas if not already done
        const canvas = document.getElementById('asset-clustering-disk')
        if (canvas && !canvas.initialized) {
            canvas.initialized = true
            console.log('Clustering canvas initialized')
        }
    }

    async loadClusteringData() {
        try {
            const response = await axios.get('/api/asset-clustering')
            
            if (response.data && response.data.clustering) {
                this.clusteringData = response.data.clustering
                
                // Always update metrics when data is loaded
                this.updateClusteringMetrics()
                
                // Draw visualization if in clustering mode
                if (this.currentVisualization === 'clustering') {
                    this.drawAssetClustering()
                }
            } else {
                console.error('Invalid clustering data structure:', response.data)
            }
        } catch (error) {
            console.error('Error loading clustering data:', error)
        }
    }

    forceClusteringMetricsUpdate() {
        // Force update timestamp even when clustering view is not active
        const timestampElement = document.getElementById('clustering-timestamp')
        if (timestampElement) {
            const now = new Date()
            timestampElement.textContent = now.toLocaleTimeString('en-US', { 
                hour12: false, 
                hour: '2-digit', 
                minute: '2-digit', 
                second: '2-digit' 
            })
        }
    }

    drawAssetClustering() {
        const canvas = document.getElementById('asset-clustering-disk')
        if (!canvas || !this.clusteringData) return
        
        const ctx = canvas.getContext('2d')
        const centerX = canvas.width / 2
        const centerY = canvas.height / 2
        const radius = Math.min(canvas.width, canvas.height) * 0.4 // Scale with canvas size
        
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
        const currentTime = Date.now()
        
        // Draw enhanced correlation lines with animations
        for (let i = 0; i < assets.length; i++) {
            for (let j = i + 1; j < assets.length; j++) {
                const asset1 = assets[i]
                const asset2 = assets[j]
                const correlation = Math.abs(asset1.correlations[asset2.symbol] || 0)
                
                if (correlation > 0.25) { // Show more correlations for better visualization
                    const x1 = centerX + asset1.x * radius
                    const y1 = centerY + asset1.y * radius
                    const x2 = centerX + asset2.x * radius
                    const y2 = centerY + asset2.y * radius
                    
                    // Enhanced line styling with gradients
                    let strokeColor, glowColor
                    if (correlation > 0.7) {
                        strokeColor = '#2ed573'
                        glowColor = '#2ed573'
                    } else if (correlation > 0.5) {
                        strokeColor = '#ffa502'
                        glowColor = '#ffa502'
                    } else if (correlation > 0.35) {
                        strokeColor = '#4ecdc4'
                        glowColor = '#4ecdc4'
                    } else {
                        strokeColor = '#6c757d'
                        glowColor = '#6c757d'
                    }
                    
                    // Animated line width and opacity
                    const animationPhase = (currentTime + (i * j * 100)) / 2000
                    const pulse = Math.sin(animationPhase) * 0.2 + 0.8
                    const baseWidth = Math.max(0.5, correlation * 4)
                    const animatedWidth = baseWidth * pulse
                    
                    // Draw glow effect first
                    const glowGradient = ctx.createLinearGradient(x1, y1, x2, y2)
                    glowGradient.addColorStop(0, glowColor + '40')
                    glowGradient.addColorStop(0.5, glowColor + '80')
                    glowGradient.addColorStop(1, glowColor + '40')
                    
                    ctx.strokeStyle = glowGradient
                    ctx.lineWidth = animatedWidth + 2
                    ctx.globalAlpha = 0.3 * pulse
                    
                    // Enhanced geodesic curve calculation
                    ctx.beginPath()
                    ctx.moveTo(x1, y1)
                    
                    // Calculate curved path that follows hyperbolic geometry
                    const distance = Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    const curvature = Math.min(0.5, correlation) * distance * 0.3
                    
                    // Control points for smooth hyperbolic curve
                    const midX = (x1 + x2) / 2 + (y2 - y1) * curvature / distance
                    const midY = (y1 + y2) / 2 - (x2 - x1) * curvature / distance
                    
                    ctx.quadraticCurveTo(midX, midY, x2, y2)
                    ctx.stroke()
                    
                    // Draw main correlation line
                    const mainGradient = ctx.createLinearGradient(x1, y1, x2, y2)
                    mainGradient.addColorStop(0, strokeColor + 'CC')
                    mainGradient.addColorStop(0.5, strokeColor)
                    mainGradient.addColorStop(1, strokeColor + 'CC')
                    
                    ctx.strokeStyle = mainGradient
                    ctx.lineWidth = animatedWidth
                    ctx.globalAlpha = 0.7 + 0.3 * pulse
                    
                    ctx.beginPath()
                    ctx.moveTo(x1, y1)
                    ctx.quadraticCurveTo(midX, midY, x2, y2)
                    ctx.stroke()
                    
                    // Add correlation strength indicator (small text at midpoint)
                    if (correlation > 0.5) {
                        const textX = midX
                        const textY = midY
                        const correlationText = (correlation * 100).toFixed(0) + '%'
                        
                        ctx.fillStyle = 'rgba(0, 0, 0, 0.8)'
                        ctx.font = '8px Arial, sans-serif'
                        ctx.textAlign = 'center'
                        ctx.textBaseline = 'middle'
                        ctx.fillText(correlationText, textX + 1, textY + 1)
                        
                        ctx.fillStyle = '#ffffff'
                        ctx.fillText(correlationText, textX, textY)
                    }
                    
                    ctx.globalAlpha = 1.0
                }
            }
        }
    }

    drawAssetNodes(ctx, centerX, centerY, radius) {
        if (!this.clusteringData.assets) return
        
        const currentTime = Date.now()
        
        this.clusteringData.assets.forEach((asset, index) => {
            const x = centerX + asset.x * radius
            const y = centerY + asset.y * radius
            
            // Enhanced node size with pulsing effect (scaled for larger canvas)
            const baseSize = Math.max(12, Math.min(28, Math.log10(asset.marketCap / 1e9) * 4.5))
            const pulse = Math.sin((currentTime + index * 100) / 1000) * 1.5
            const nodeSize = baseSize + pulse
            
            // Enhanced color system with gradients
            let fillColor, secondaryColor
            if (asset.priceChange > 0.01) {
                fillColor = '#2ed573'
                secondaryColor = '#20bf6b'
            } else if (asset.priceChange < -0.01) {
                fillColor = '#ff4757'
                secondaryColor = '#ff3838'
            } else {
                fillColor = '#4ecdc4'
                secondaryColor = '#26d0ce'
            }
            
            // Enhanced volatility glow with animation
            const volatilityMultiplier = 1 + Math.sin((currentTime + index * 150) / 800) * 0.3
            const glowRadius = nodeSize + (asset.volatility * 120 * volatilityMultiplier)
            
            // Multi-layer glow effect
            const innerGradient = ctx.createRadialGradient(x, y, nodeSize * 0.3, x, y, nodeSize)
            innerGradient.addColorStop(0, fillColor)
            innerGradient.addColorStop(1, secondaryColor)
            
            const outerGradient = ctx.createRadialGradient(x, y, nodeSize, x, y, glowRadius)
            outerGradient.addColorStop(0, fillColor + '80')
            outerGradient.addColorStop(0.7, fillColor + '20')
            outerGradient.addColorStop(1, 'transparent')
            
            // Draw outer glow
            ctx.fillStyle = outerGradient
            ctx.globalAlpha = 0.4
            ctx.beginPath()
            ctx.arc(x, y, glowRadius, 0, 2 * Math.PI)
            ctx.fill()
            
            // Draw main node with gradient
            ctx.globalAlpha = 1.0
            ctx.fillStyle = innerGradient
            ctx.beginPath()
            ctx.arc(x, y, nodeSize, 0, 2 * Math.PI)
            ctx.fill()
            
            // Enhanced border effects
            if (this.hasArbitrageOpportunity(asset.symbol)) {
                // Animated arbitrage border
                const borderPulse = Math.sin((currentTime + index * 200) / 600) * 0.5 + 0.5
                ctx.strokeStyle = `rgba(255, 165, 2, ${0.8 + borderPulse * 0.2})`
                ctx.lineWidth = 3
                ctx.stroke()
            }
            
            // High correlation indicator
            const maxCorrelation = Math.max(...Object.values(asset.correlations || {}))
            if (maxCorrelation > 0.7) {
                ctx.strokeStyle = '#ffa502'
                ctx.lineWidth = 2
                ctx.setLineDash([3, 3])
                ctx.stroke()
                ctx.setLineDash([])
            }
            
            // Enhanced text rendering with better shadows
            const fontSize = Math.max(10, nodeSize * 0.6)
            
            // Multiple shadow layers for depth
            ctx.fillStyle = 'rgba(0, 0, 0, 0.8)'
            ctx.font = `bold ${fontSize}px 'Segoe UI', Arial, sans-serif`
            ctx.textAlign = 'center'
            ctx.textBaseline = 'middle'
            ctx.fillText(asset.symbol, x + 2, y + 2) // Heavy shadow
            
            ctx.fillStyle = 'rgba(0, 0, 0, 0.4)'
            ctx.fillText(asset.symbol, x + 1, y + 1) // Light shadow
            
            // Main text with subtle glow
            ctx.fillStyle = '#ffffff'
            ctx.shadowColor = fillColor
            ctx.shadowBlur = 3
            ctx.fillText(asset.symbol, x, y)
            ctx.shadowBlur = 0
            
            // Enhanced price change indicator
            const priceChangeText = `${(asset.priceChange * 100).toFixed(1)}%`
            const trendIcon = asset.priceChange > 0 ? '▲' : asset.priceChange < 0 ? '▼' : '●'
            const changeColor = asset.priceChange > 0 ? '#2ed573' : asset.priceChange < 0 ? '#ff4757' : '#4ecdc4'
            
            ctx.fillStyle = changeColor
            ctx.font = `bold ${Math.max(8, fontSize * 0.7)}px 'Segoe UI', Arial, sans-serif`
            ctx.shadowColor = changeColor
            ctx.shadowBlur = 2
            ctx.fillText(`${trendIcon} ${priceChangeText}`, x, y + nodeSize + 15)
            ctx.shadowBlur = 0
            
            // Market cap indicator (small circle)
            const capSize = Math.min(nodeSize * 0.3, 6)
            ctx.fillStyle = asset.marketCap > 1e12 ? '#ffd700' : asset.marketCap > 1e11 ? '#c0c0c0' : '#cd7f32'
            ctx.beginPath()
            ctx.arc(x + nodeSize * 0.7, y - nodeSize * 0.7, capSize, 0, 2 * Math.PI)
            ctx.fill()
        })
    }

    drawClusterBoundaries(ctx, centerX, centerY, radius) {
        // Define asset categories with colors
        const categories = {
            'crypto': { 
                symbols: ['BTC', 'ETH', 'SOL'], 
                color: 'rgba(0, 212, 170, 0.15)',
                labelColor: '#00d4aa',
                name: 'CRYPTO'
            },
            'equity': { 
                symbols: ['SP500', 'NASDAQ', 'DOW'], 
                color: 'rgba(46, 213, 115, 0.15)',
                labelColor: '#2ed573',
                name: 'EQUITY'
            },
            'international': { 
                symbols: ['FTSE', 'NIKKEI', 'DAX'], 
                color: 'rgba(255, 165, 2, 0.15)',
                labelColor: '#ffa502',
                name: 'INTL'
            },
            'commodities': { 
                symbols: ['GOLD', 'SILVER', 'OIL'], 
                color: 'rgba(255, 71, 87, 0.15)',
                labelColor: '#ff4757',
                name: 'COMM'
            },
            'forex': { 
                symbols: ['EURUSD', 'GBPUSD', 'USDJPY'], 
                color: 'rgba(78, 205, 196, 0.15)',
                labelColor: '#4ecdc4',
                name: 'FOREX'
            }
        }
        
        // Draw cluster regions for each category
        Object.entries(categories).forEach(([categoryKey, category]) => {
            const categoryAssets = this.clusteringData.assets.filter(a => 
                category.symbols.includes(a.symbol)
            )
            
            if (categoryAssets.length > 0) {
                // Calculate cluster center and boundary
                const avgX = categoryAssets.reduce((sum, a) => sum + a.x, 0) / categoryAssets.length
                const avgY = categoryAssets.reduce((sum, a) => sum + a.y, 0) / categoryAssets.length
                const maxDistance = Math.max(...categoryAssets.map(a => Math.sqrt(a.x*a.x + a.y*a.y)))
                
                // Draw cluster region
                ctx.fillStyle = category.color
                ctx.beginPath()
                ctx.arc(centerX + avgX * radius, centerY + avgY * radius, maxDistance * radius * 0.8, 0, 2 * Math.PI)
                ctx.fill()
                
                // Cluster border
                ctx.strokeStyle = category.labelColor
                ctx.lineWidth = 1
                ctx.setLineDash([5, 5])
                ctx.stroke()
                ctx.setLineDash([])
                
                // Enhanced cluster label
                const labelX = centerX + avgX * radius
                const labelY = centerY + avgY * radius - maxDistance * radius * 0.9
                
                // Label background
                ctx.fillStyle = category.labelColor
                const labelWidth = category.name.length * 8 + 10
                ctx.fillRect(labelX - labelWidth/2, labelY - 10, labelWidth, 20)
                
                // Label text
                ctx.fillStyle = '#000000'
                ctx.font = 'bold 11px Arial, sans-serif'
                ctx.textAlign = 'center'
                ctx.textBaseline = 'middle'
                ctx.fillText(category.name, labelX, labelY)
            }
        })
    }

    updateClusteringMetrics() {
        if (!this.clusteringData || !this.clusteringData.assets) {
            return
        }
        
        // Calculate real-time average correlation with enhanced accuracy
        let totalCorrelation = 0
        let correlationCount = 0
        let correlationValues = []
        
        this.clusteringData.assets.forEach(asset1 => {
            if (asset1.correlations) {
                this.clusteringData.assets.forEach(asset2 => {
                    if (asset1.symbol !== asset2.symbol && asset1.correlations[asset2.symbol] !== undefined) {
                        const corrValue = Math.abs(asset1.correlations[asset2.symbol])
                        totalCorrelation += corrValue
                        correlationValues.push(corrValue)
                        correlationCount++
                    }
                })
            }
        })
        
        const avgCorrelation = correlationCount > 0 ? totalCorrelation / correlationCount : 0
        
        // Calculate correlation variance for better stability assessment
        let correlationVariance = 0
        if (correlationValues.length > 0) {
            correlationVariance = correlationValues.reduce((sum, val) => sum + Math.pow(val - avgCorrelation, 2), 0) / correlationValues.length
        }
        
        // Enhanced stability calculation
        let stability = 'Low'
        let stabilityClass = 'text-loss'
        let stabilityIcon = '🔴'
        
        if (avgCorrelation > 0.6 && correlationVariance < 0.1) {
            stability = 'Very High'
            stabilityClass = 'text-profit animate-pulse'
            stabilityIcon = '🟢'
        } else if (avgCorrelation > 0.4 && correlationVariance < 0.15) {
            stability = 'High'
            stabilityClass = 'text-profit'
            stabilityIcon = '🟢'
        } else if (avgCorrelation > 0.25) {
            stability = 'Medium'
            stabilityClass = 'text-warning'
            stabilityIcon = '🟡'
        }
        
        // Animate metric updates
        this.animateMetricUpdate('cluster-asset-count', this.clusteringData.totalAssets || this.clusteringData.assets.length)
        this.animateMetricUpdate('avg-correlation', avgCorrelation.toFixed(3))
        
        // Update cluster stability with enhanced presentation
        const stabilityElement = document.getElementById('cluster-stability')
        if (stabilityElement) {
            stabilityElement.innerHTML = `${stabilityIcon} ${stability}`
            stabilityElement.className = `${stabilityClass} font-semibold`
        }
        
        // Update timestamp to show real-time updates
        const timestampElement = document.getElementById('clustering-timestamp')
        if (timestampElement) {
            const now = new Date()
            timestampElement.textContent = now.toLocaleTimeString('en-US', { 
                hour12: false, 
                hour: '2-digit', 
                minute: '2-digit', 
                second: '2-digit' 
            })
        }
        
        // Update asset legend with enhanced information
        this.updateEnhancedAssetLegend()
        
        // Add real-time correlation heatmap
        this.updateCorrelationHeatmap()
        
        // Update clustering performance metrics
        this.updateClusteringPerformance(avgCorrelation, correlationVariance, stabilityIcon)
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
    
    // Enhanced animation for metric updates
    animateMetricUpdate(elementId, newValue) {
        const element = document.getElementById(elementId)
        if (!element) return
        
        // Add smooth update animation with glow effect
        element.style.transition = 'all 0.4s cubic-bezier(0.4, 0, 0.2, 1)'
        element.style.transform = 'scale(1.15)'
        element.style.textShadow = '0 0 10px rgba(0, 212, 170, 0.8)'
        
        setTimeout(() => {
            element.textContent = newValue
            element.style.transform = 'scale(1)'
            element.style.textShadow = 'none'
        }, 200)
    }
    
    // Enhanced asset legend with real-time data
    updateEnhancedAssetLegend() {
        if (!this.clusteringData) return
        
        const legendContainer = document.getElementById('asset-legend')
        if (!legendContainer) return
        
        const legendHtml = this.clusteringData.assets.map(asset => {
            const priceChangePercent = (asset.priceChange * 100).toFixed(2)
            const changeColor = asset.priceChange > 0 ? 'text-profit' : asset.priceChange < 0 ? 'text-loss' : 'text-gray-400'
            const trendIcon = asset.priceChange > 0 ? '📈' : asset.priceChange < 0 ? '📉' : '➡️'
            
            // Calculate volatility indicator
            const volatilityLevel = asset.volatility > 0.005 ? 'High' : asset.volatility > 0.003 ? 'Med' : 'Low'
            const volatilityColor = asset.volatility > 0.005 ? 'text-loss' : asset.volatility > 0.003 ? 'text-warning' : 'text-profit'
            
            return `
                <div class="flex items-center justify-between py-2 hover:bg-gray-700 rounded px-2 transition-colors">
                    <div class="flex items-center">
                        <div class="w-3 h-3 rounded-full mr-3 animate-pulse" style="background-color: ${
                            asset.priceChange > 0.01 ? '#2ed573' : 
                            asset.priceChange < -0.01 ? '#ff4757' : '#4ecdc4'
                        }"></div>
                        <div>
                            <span class="font-semibold">${asset.symbol}</span>
                            <div class="text-xs ${volatilityColor}">Vol: ${volatilityLevel}</div>
                        </div>
                    </div>
                    <div class="text-right">
                        <div class="${changeColor} font-semibold">${trendIcon} ${priceChangePercent > 0 ? '+' : ''}${priceChangePercent}%</div>
                        <div class="text-xs text-gray-400">$${asset.currentPrice ? asset.currentPrice.toLocaleString() : 'N/A'}</div>
                    </div>
                </div>
            `
        }).join('')
        
        legendContainer.innerHTML = legendHtml
    }
    
    // Real-time correlation heatmap
    updateCorrelationHeatmap() {
        if (!this.clusteringData || !this.clusteringData.assets) return
        
        let heatmapContainer = document.getElementById('correlation-heatmap')
        if (!heatmapContainer) {
            // Create heatmap container if it doesn't exist
            const legendContainer = document.getElementById('asset-legend')
            if (legendContainer && legendContainer.parentNode) {
                heatmapContainer = document.createElement('div')
                heatmapContainer.id = 'correlation-heatmap'
                heatmapContainer.className = 'mt-4 p-3 bg-gray-800 rounded-lg'
                heatmapContainer.innerHTML = '<div class="text-gray-400 font-semibold mb-2 text-xs uppercase tracking-wide">Correlation Heatmap:</div>'
                legendContainer.parentNode.insertBefore(heatmapContainer, legendContainer.nextSibling)
            }
        }
        
        if (!heatmapContainer) return
        
        // Create simplified correlation matrix display
        const topCorrelations = []
        
        this.clusteringData.assets.forEach((asset1, i) => {
            if (asset1.correlations) {
                this.clusteringData.assets.forEach((asset2, j) => {
                    if (i < j && asset1.correlations[asset2.symbol] !== undefined) {
                        const correlation = asset1.correlations[asset2.symbol]
                        if (Math.abs(correlation) > 0.3) {
                            topCorrelations.push({
                                pair: `${asset1.symbol}-${asset2.symbol}`,
                                correlation: correlation,
                                strength: Math.abs(correlation)
                            })
                        }
                    }
                })
            }
        })
        
        // Sort by correlation strength and take top 6
        topCorrelations.sort((a, b) => b.strength - a.strength)
        const displayCorrelations = topCorrelations.slice(0, 6)
        
        const heatmapHtml = `
            <div class="text-gray-400 font-semibold mb-2 text-xs uppercase tracking-wide">Top Correlations:</div>
            <div class="grid grid-cols-2 gap-1 text-xs">
                ${displayCorrelations.map(item => {
                    const color = item.correlation > 0.5 ? 'text-profit' : 
                                 item.correlation > 0.3 ? 'text-warning' : 'text-gray-400'
                    const bgColor = item.correlation > 0.5 ? 'bg-green-900' : 
                                   item.correlation > 0.3 ? 'bg-yellow-900' : 'bg-gray-700'
                    return `
                        <div class="${bgColor} rounded p-1 text-center">
                            <div class="font-semibold text-xs">${item.pair}</div>
                            <div class="${color}">${(item.correlation * 100).toFixed(0)}%</div>
                        </div>
                    `
                }).join('')}
            </div>
        `
        
        heatmapContainer.innerHTML = heatmapHtml
    }
    
    // Clustering performance indicators
    updateClusteringPerformance(avgCorrelation, correlationVariance, stabilityIcon) {
        let performanceContainer = document.getElementById('clustering-performance')
        if (!performanceContainer) {
            // Create performance container
            const mainContainer = document.querySelector('#poincare-clustering-view')
            if (mainContainer) {
                performanceContainer = document.createElement('div')
                performanceContainer.id = 'clustering-performance'
                performanceContainer.className = 'mt-4 grid grid-cols-3 gap-2 text-xs'
                mainContainer.appendChild(performanceContainer)
            }
        }
        
        if (!performanceContainer) return
        
        // Calculate clustering efficiency score
        const efficiency = Math.min(100, Math.max(0, (avgCorrelation * 100) - (correlationVariance * 50)))
        const efficiencyColor = efficiency > 70 ? 'text-profit' : efficiency > 40 ? 'text-warning' : 'text-loss'
        
        // Calculate network density
        const networkDensity = (avgCorrelation * 100).toFixed(1)
        const densityColor = networkDensity > 50 ? 'text-accent' : 'text-gray-400'
        
        performanceContainer.innerHTML = `
            <div class="bg-gray-800 rounded p-2 text-center">
                <div class="${efficiencyColor} font-bold">${efficiency.toFixed(0)}%</div>
                <div class="text-gray-400">Efficiency</div>
            </div>
            <div class="bg-gray-800 rounded p-2 text-center">
                <div class="${densityColor} font-bold">${networkDensity}%</div>
                <div class="text-gray-400">Density</div>
            </div>
            <div class="bg-gray-800 rounded p-2 text-center">
                <div class="font-bold">${stabilityIcon}</div>
                <div class="text-gray-400">Status</div>
            </div>
        `
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
        console.log('Initializing backtest charts...')
        
        // Initialize equity curve chart
        const equityCanvas = document.getElementById('equity-curve-chart')
        if (equityCanvas) {
            console.log('Found equity curve canvas, initializing...')
            try {
                const ctx = equityCanvas.getContext('2d')
                this.equityCurveChart = new Chart(ctx, {
                    type: 'line',
                    data: {
                        labels: [],
                        datasets: [{
                            label: 'Portfolio Equity ($)',
                            data: [],
                            borderColor: '#00d4aa',
                            backgroundColor: 'rgba(0, 212, 170, 0.1)',
                            tension: 0.1,
                            fill: true
                        }]
                    },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        scales: {
                            x: {
                                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                                ticks: { color: '#ffffff' }
                            },
                            y: {
                                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                                ticks: { color: '#ffffff' }
                            }
                        },
                        plugins: {
                            legend: { labels: { color: '#ffffff' } },
                            tooltip: {
                                backgroundColor: 'rgba(26, 31, 41, 0.9)',
                                titleColor: '#00d4aa',
                                bodyColor: '#ffffff'
                            }
                        }
                    }
                })
                console.log('✅ Equity curve chart initialized successfully')
            } catch (error) {
                console.error('❌ Error initializing equity curve chart:', error)
            }
        } else {
            console.error('❌ Equity curve canvas not found')
        }

        // Initialize drawdown chart
        const drawdownCanvas = document.getElementById('drawdown-chart')
        if (drawdownCanvas) {
            console.log('Found drawdown canvas, initializing...')
            try {
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
                                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                                ticks: { color: '#ffffff' }
                            },
                            y: {
                                min: 0,
                                reverse: true,
                                grid: { color: 'rgba(255, 255, 255, 0.1)' },
                                ticks: { 
                                    color: '#ffffff',
                                    callback: function(value) {
                                        return '-' + value.toFixed(1) + '%'
                                    }
                                }
                            }
                        },
                        plugins: {
                            legend: { labels: { color: '#ffffff' } },
                            tooltip: {
                                backgroundColor: 'rgba(26, 31, 41, 0.9)',
                                titleColor: '#ff4757',
                                bodyColor: '#ffffff'
                            }
                        }
                    }
                })
                console.log('✅ Drawdown chart initialized successfully')
            } catch (error) {
                console.error('❌ Error initializing drawdown chart:', error)
            }
        } else {
            console.error('❌ Drawdown canvas not found')
        }
    }

    async runBacktest() {
        try {
            const strategyConfig = this.getBacktestConfiguration()
            
            // Validate configuration
            if (!this.validateBacktestConfig(strategyConfig)) {
                this.showNotification('❌ Invalid configuration. Please check all parameters.', 'error')
                return
            }
            
            // Show loading state
            const resultsContainer = document.getElementById('backtest-results')
            resultsContainer.innerHTML = `
                <div class="flex items-center justify-center py-8">
                    <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-accent mr-3"></div>
                    <span>Running enterprise-grade backtest...</span>
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

    validateBacktestConfig(config) {
        if (!config.strategyName || config.strategyName.trim() === '') return false
        if (!config.symbol || !config.timeframe) return false
        if (config.initialCapital < 1000 || config.initialCapital > 10000000) return false
        if (config.parameters.minConfidence < 50 || config.parameters.minConfidence > 100) return false
        if (config.parameters.riskPerTrade < 0.001 || config.parameters.riskPerTrade > 0.1) return false
        return true
    }

    async runMonteCarloSimulation() {
        try {
            const strategyConfig = this.getBacktestConfiguration()
            
            if (!this.validateBacktestConfig(strategyConfig)) {
                this.showNotification('❌ Invalid configuration for Monte Carlo simulation.', 'error')
                return
            }

            const resultsContainer = document.getElementById('backtest-results')
            resultsContainer.innerHTML = `
                <div class="flex items-center justify-center py-8">
                    <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-500 mr-3"></div>
                    <span>Running Monte Carlo simulation (1000 iterations)...</span>
                </div>
            `

            const response = await axios.post('/api/backtest/monte-carlo', {
                ...strategyConfig,
                iterations: 1000
            })
            const { results } = response.data

            this.displayMonteCarloResults(results)
            this.showNotification('✅ Monte Carlo simulation completed!', 'success')

        } catch (error) {
            console.error('Monte Carlo simulation error:', error)
            this.showNotification('❌ Monte Carlo simulation failed.', 'error')
        }
    }

    async compareStrategies() {
        try {
            const baseConfig = this.getBacktestConfiguration()
            
            if (!this.validateBacktestConfig(baseConfig)) {
                this.showNotification('❌ Invalid base configuration for strategy comparison.', 'error')
                return
            }

            const resultsContainer = document.getElementById('backtest-results')
            resultsContainer.innerHTML = `
                <div class="flex items-center justify-center py-8">
                    <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-orange-500 mr-3"></div>
                    <span>Comparing strategies across multiple configurations...</span>
                </div>
            `

            const response = await axios.post('/api/backtest/compare', baseConfig)
            const { results } = response.data

            this.displayStrategyComparison(results)
            this.showNotification('✅ Strategy comparison completed!', 'success')

        } catch (error) {
            console.error('Strategy comparison error:', error)
            this.showNotification('❌ Strategy comparison failed.', 'error')
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
                            <span>Profit Factor:</span>
                            <span class="${metrics.profitFactor > 1.5 ? 'text-profit' : metrics.profitFactor > 1 ? 'text-warning' : 'text-loss'}">
                                ${metrics.profitFactor.toFixed(2)}x
                            </span>
                        </div>
                        <div class="flex justify-between">
                            <span>Calmar Ratio:</span>
                            <span class="${metrics.calmarRatio > 1 ? 'text-profit' : 'text-warning'}">
                                ${metrics.calmarRatio}
                            </span>
                        </div>
                        <div class="flex justify-between">
                            <span>Recovery Factor:</span>
                            <span class="text-accent">
                                ${((metrics.totalReturn / Math.max(metrics.maxDrawdown, 1))).toFixed(2)}
                            </span>
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

    displayMonteCarloResults(results) {
        const { summary, iterations } = results
        
        const monteCarloHtml = `
            <div class="bg-gradient-to-br from-purple-900/50 to-pink-900/50 rounded-lg p-6 mt-4">
                <h4 class="text-lg font-bold text-purple-300 mb-4">
                    🎲 Monte Carlo Analysis (${iterations.toLocaleString()} Iterations)
                </h4>
                
                <div class="grid grid-cols-3 gap-4 mb-6">
                    <div class="text-center">
                        <div class="text-2xl font-bold text-purple-300">
                            ${summary.avgReturn.toFixed(2)}%
                        </div>
                        <div class="text-sm text-gray-400">Average Return</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold text-purple-300">
                            ${summary.profitProbability.toFixed(1)}%
                        </div>
                        <div class="text-sm text-gray-400">Profit Probability</div>
                    </div>
                    <div class="text-center">
                        <div class="text-2xl font-bold text-purple-300">
                            ${summary.stdReturn.toFixed(2)}%
                        </div>
                        <div class="text-sm text-gray-400">Return Volatility</div>
                    </div>
                </div>
                
                <div class="grid grid-cols-4 gap-4">
                    <div class="text-center bg-black/30 rounded p-3">
                        <div class="text-lg font-semibold text-green-400">${summary.maxReturn.toFixed(2)}%</div>
                        <div class="text-xs text-gray-400">Best Case</div>
                    </div>
                    <div class="text-center bg-black/30 rounded p-3">
                        <div class="text-lg font-semibold text-purple-300">${summary.medianReturn.toFixed(2)}%</div>
                        <div class="text-xs text-gray-400">Median Return</div>
                    </div>
                    <div class="text-center bg-black/30 rounded p-3">
                        <div class="text-lg font-semibold text-red-400">${summary.minReturn.toFixed(2)}%</div>
                        <div class="text-xs text-gray-400">Worst Case</div>
                    </div>
                    <div class="text-center bg-black/30 rounded p-3">
                        <div class="text-lg font-semibold text-orange-400">${summary.maxDrawdown.toFixed(2)}%</div>
                        <div class="text-xs text-gray-400">Max Drawdown</div>
                    </div>
                </div>
                
                <div class="mt-4 p-3 bg-black/30 rounded">
                    <div class="text-sm text-gray-300">
                        <strong>Risk Assessment:</strong> 
                        ${summary.profitProbability > 70 ? 
                            '<span class="text-green-400">Low Risk - High probability of profit</span>' :
                            summary.profitProbability > 50 ? 
                            '<span class="text-yellow-400">Medium Risk - Moderate profit probability</span>' :
                            '<span class="text-red-400">High Risk - Low profit probability</span>'
                        }
                    </div>
                </div>
            </div>
        `
        
        document.getElementById('backtest-results').innerHTML += monteCarloHtml
    }

    displayStrategyComparison(results) {
        const { strategies, benchmark } = results
        
        const comparisonHtml = `
            <div class="bg-gradient-to-br from-orange-900/50 to-red-900/50 rounded-lg p-6 mt-4">
                <h4 class="text-lg font-bold text-orange-300 mb-4">
                    ⚖️ Strategy Performance Comparison
                </h4>
                
                <div class="overflow-x-auto">
                    <table class="w-full text-sm">
                        <thead>
                            <tr class="border-b border-gray-600">
                                <th class="text-left py-2 text-orange-300">Strategy</th>
                                <th class="text-center py-2 text-orange-300">Return</th>
                                <th class="text-center py-2 text-orange-300">Sharpe</th>
                                <th class="text-center py-2 text-orange-300">Max DD</th>
                                <th class="text-center py-2 text-orange-300">Win Rate</th>
                                <th class="text-center py-2 text-orange-300">Rating</th>
                            </tr>
                        </thead>
                        <tbody>
                            ${strategies.map((strategy, index) => `
                                <tr class="border-b border-gray-700 ${index === 0 ? 'bg-green-900/30' : ''}">
                                    <td class="py-3 font-medium">${strategy.name}</td>
                                    <td class="text-center ${strategy.metrics.finalReturn > 0 ? 'text-green-400' : 'text-red-400'}">
                                        ${strategy.metrics.finalReturn > 0 ? '+' : ''}${strategy.metrics.finalReturn.toFixed(2)}%
                                    </td>
                                    <td class="text-center">${strategy.metrics.sharpeRatio}</td>
                                    <td class="text-center text-red-400">${strategy.metrics.maxDrawdown.toFixed(2)}%</td>
                                    <td class="text-center">${strategy.metrics.winRate.toFixed(1)}%</td>
                                    <td class="text-center">
                                        <span class="px-2 py-1 rounded text-xs ${
                                            strategy.riskAdjustedReturn > 2 ? 'bg-green-900 text-green-200' :
                                            strategy.riskAdjustedReturn > 1 ? 'bg-yellow-900 text-yellow-200' :
                                            'bg-red-900 text-red-200'
                                        }">
                                            ${strategy.riskAdjustedReturn > 2 ? 'Excellent' :
                                              strategy.riskAdjustedReturn > 1 ? 'Good' : 'Poor'}
                                        </span>
                                    </td>
                                </tr>
                            `).join('')}
                        </tbody>
                    </table>
                </div>
                
                <div class="mt-4 grid grid-cols-2 gap-4">
                    <div class="p-3 bg-black/30 rounded">
                        <div class="text-sm font-medium text-orange-300 mb-1">🏆 Best Strategy</div>
                        <div class="text-white">${strategies[0]?.name || 'N/A'}</div>
                        <div class="text-xs text-gray-400 mt-1">
                            Return: ${strategies[0]?.metrics?.finalReturn?.toFixed(2) || 'N/A'}% | 
                            Sharpe: ${strategies[0]?.metrics?.sharpeRatio || 'N/A'}
                        </div>
                    </div>
                    <div class="p-3 bg-black/30 rounded">
                        <div class="text-sm font-medium text-orange-300 mb-1">📊 Benchmark</div>
                        <div class="text-white">Buy & Hold ${benchmark.symbol}</div>
                        <div class="text-xs text-gray-400 mt-1">
                            Return: ${benchmark.return.toFixed(2)}% | 
                            Volatility: ${benchmark.volatility.toFixed(2)}%
                        </div>
                    </div>
                </div>
            </div>
        `
        
        document.getElementById('backtest-results').innerHTML += comparisonHtml
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div')
        notification.className = `fixed top-4 right-4 px-6 py-3 rounded-lg shadow-lg z-50 transition-all duration-300 ${
            type === 'success' ? 'bg-green-600 text-white' :
            type === 'error' ? 'bg-red-600 text-white' :
            type === 'warning' ? 'bg-yellow-600 text-black' :
            'bg-blue-600 text-white'
        }`
        notification.textContent = message
        
        document.body.appendChild(notification)
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            notification.style.opacity = '0'
            setTimeout(() => {
                document.body.removeChild(notification)
            }, 300)
        }, 5000)
    }

    updateBacktestCharts(results) {
        console.log('📊 Updating backtest charts with results:', results)
        
        // Update equity curve
        if (this.equityCurveChart && results.equity && results.equity.length > 0) {
            console.log('📈 Updating equity curve with', results.equity.length, 'data points')
            const equityData = results.equity
            
            this.equityCurveChart.data.labels = equityData.map(point => new Date(point.timestamp))
            this.equityCurveChart.data.datasets[0].data = equityData.map(point => point.equity)
            this.equityCurveChart.update('none')
            console.log('✅ Equity curve updated successfully')
        } else {
            console.error('❌ Cannot update equity curve:', {
                chartExists: !!this.equityCurveChart,
                equityDataExists: !!results.equity,
                equityDataLength: results.equity?.length
            })
        }

        // Update drawdown chart  
        if (this.drawdownChart && results.drawdowns && results.drawdowns.length > 0) {
            console.log('📉 Updating drawdown chart with', results.drawdowns.length, 'data points')
            const drawdownData = results.drawdowns
            
            this.drawdownChart.data.labels = drawdownData.map(point => new Date(point.timestamp))
            this.drawdownChart.data.datasets[0].data = drawdownData.map(point => Math.abs(point.drawdown))
            this.drawdownChart.update('none')
            console.log('✅ Drawdown chart updated successfully')
        } else {
            console.error('❌ Cannot update drawdown chart:', {
                chartExists: !!this.drawdownChart,
                drawdownDataExists: !!results.drawdowns,
                drawdownDataLength: results.drawdowns?.length
            })
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

    async initializeEconomicData() {
        await this.loadEconomicDashboard()
        await this.loadSentimentDashboard()
        this.createEconomicTrendsChart()
        
        // Auto-refresh economic data every 5 seconds
        if (this.economicDataInterval) {
            clearInterval(this.economicDataInterval)
        }
        this.economicDataInterval = setInterval(() => {
            if (this.currentSection === 'economic-data') {
                this.loadEconomicDashboard()
                this.loadSentimentDashboard()
            }
        }, 5000)
    }

    async loadEconomicDashboard() {
        try {
            const response = await axios.get('/api/economic-indicators')
            const data = response.data
            
            const dashboardContainer = document.getElementById('economic-dashboard')
            if (dashboardContainer) {
                const formatIndicator = (name, value, unit = '%', icon = '📊') => `
                    <div class="bg-gray-800 rounded-lg p-4">
                        <div class="flex items-center justify-between mb-2">
                            <span class="text-sm font-semibold">${icon} ${name}</span>
                            <span class="text-xs ${value.change > 0 ? 'text-profit' : value.change < 0 ? 'text-loss' : 'text-gray-400'}">
                                ${value.change > 0 ? '↗️' : value.change < 0 ? '↘️' : '→'} ${Math.abs(value.change || 0).toFixed(2)}
                            </span>
                        </div>
                        <div class="text-2xl font-bold text-accent">${value.current.toFixed(2)}${unit}</div>
                        <div class="text-xs text-gray-400">
                            Prev: ${value.previous.toFixed(2)}${unit} | Forecast: ${value.forecast.toFixed(2)}${unit}
                        </div>
                    </div>
                `
                
                dashboardContainer.innerHTML = `
                    ${formatIndicator('GDP Growth', data.us.gdp, '%', '🏛️')}
                    ${formatIndicator('Inflation Rate', data.us.inflation, '%', '💰')}
                    ${formatIndicator('Fed Funds Rate', data.us.interestRate, '%', '🏦')}
                    ${formatIndicator('Unemployment', data.us.unemployment, '%', '👷')}
                    ${formatIndicator('Dollar Index', data.us.dollarIndex, '', '💵')}
                    ${formatIndicator('Consumer Confidence', data.us.consumerConfidence, '', '🛒')}
                    ${formatIndicator('BTC Dominance', data.crypto.bitcoinDominance, '%', '₿')}
                    ${formatIndicator('Total Crypto Cap', data.crypto.totalMarketCap, 'T', '💎')}
                    ${formatIndicator('DeFi TVL', data.crypto.defiTvl, 'B', '🔗')}
                `
            }
        } catch (error) {
            console.error('Error loading economic dashboard:', error)
        }
    }

    async loadSentimentDashboard() {
        try {
            const [sentimentResponse, socialResponse] = await Promise.all([
                axios.get('/api/sentiment-summary'),
                axios.get('/api/social-sentiment')
            ])
            
            const sentiment = sentimentResponse.data
            const social = socialResponse.data
            
            const sentimentContainer = document.getElementById('sentiment-dashboard')
            if (sentimentContainer) {
                const getSentimentEmoji = (score) => {
                    if (score > 80) return '🚀'
                    if (score > 60) return '😄' 
                    if (score > 40) return '😐'
                    if (score > 20) return '😰'
                    return '😱'
                }
                
                const getSentimentColor = (score) => {
                    if (score > 70) return 'text-profit'
                    if (score > 30) return 'text-warning'
                    return 'text-loss'
                }
                
                sentimentContainer.innerHTML = `
                    <div class="bg-gray-800 rounded-lg p-4">
                        <div class="text-center mb-3">
                            <div class="text-3xl mb-1">${getSentimentEmoji(sentiment.overall)}</div>
                            <div class="text-xl font-bold ${getSentimentColor(sentiment.overall)}">
                                ${sentiment.overall.toFixed(0)}%
                            </div>
                            <div class="text-sm text-gray-400">Overall Market Mood</div>
                        </div>
                        
                        <div class="space-y-2">
                            ${Object.entries(sentiment.assets).map(([asset, score]) => `
                                <div class="flex justify-between items-center">
                                    <span class="text-sm">${getSentimentEmoji(score)} ${asset}:</span>
                                    <span class="${getSentimentColor(score)} font-semibold">${score.toFixed(0)}%</span>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                    
                    <div class="bg-gray-800 rounded-lg p-4">
                        <h5 class="font-semibold mb-2">📱 Social Media Activity</h5>
                        <div class="space-y-2 text-sm">
                            <div class="flex justify-between">
                                <span>Twitter Volume:</span>
                                <span class="text-accent">${(sentiment.socialVolume.twitter / 1000).toFixed(0)}K</span>
                            </div>
                            <div class="flex justify-between">
                                <span>Reddit Posts:</span>
                                <span class="text-accent">${sentiment.socialVolume.reddit}</span>
                            </div>
                            <div class="flex justify-between">
                                <span>Fear & Greed Index:</span>
                                <span class="${getSentimentColor(sentiment.fearGreedIndex)}">${sentiment.fearGreedIndex}</span>
                            </div>
                            ${sentiment.trending.twitter.length > 0 ? `
                                <div class="mt-2 p-2 bg-warning bg-opacity-20 rounded">
                                    <div class="text-warning text-xs font-semibold">🔥 TRENDING:</div>
                                    <div class="text-xs">${sentiment.trending.twitter.join(', ')}</div>
                                </div>
                            ` : ''}
                        </div>
                    </div>
                `
            }
        } catch (error) {
            console.error('Error loading sentiment dashboard:', error)
        }
    }

    createEconomicTrendsChart() {
        const canvas = document.getElementById('trends-chart')
        if (!canvas) return
        
        const ctx = canvas.getContext('2d')
        
        // Generate sample trend data
        const labels = []
        const inflationData = []
        const gdpData = []
        const unemploymentData = []
        
        for (let i = 11; i >= 0; i--) {
            const date = new Date()
            date.setMonth(date.getMonth() - i)
            labels.push(date.toLocaleDateString('en-US', { month: 'short', year: '2-digit' }))
            
            inflationData.push(3.2 + (Math.random() - 0.5) * 1.5)
            gdpData.push(2.1 + (Math.random() - 0.5) * 0.8)
            unemploymentData.push(3.8 + (Math.random() - 0.5) * 0.6)
        }
        
        new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Inflation %',
                        data: inflationData,
                        borderColor: '#ff4757',
                        backgroundColor: 'rgba(255, 71, 87, 0.1)',
                        tension: 0.3,
                        borderWidth: 2
                    },
                    {
                        label: 'GDP Growth %',
                        data: gdpData,
                        borderColor: '#00ff9f',
                        backgroundColor: 'rgba(0, 255, 159, 0.1)',
                        tension: 0.3,
                        borderWidth: 2
                    },
                    {
                        label: 'Unemployment %',
                        data: unemploymentData,
                        borderColor: '#ffa502',
                        backgroundColor: 'rgba(255, 165, 2, 0.1)',
                        tension: 0.3,
                        borderWidth: 2
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { 
                            color: '#ffffff',
                            font: { size: 10 }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { 
                            color: '#ffffff',
                            font: { size: 9 }
                        }
                    },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.1)' },
                        ticks: { 
                            color: '#ffffff',
                            font: { size: 9 }
                        }
                    }
                }
            }
        })
    }
}

// Initialize the dashboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new TradingDashboard()
})