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
}

// Initialize the dashboard when the page loads
document.addEventListener('DOMContentLoaded', () => {
    new TradingDashboard()
})