// Trading Dashboard JavaScript

class TradingDashboard {
    constructor() {
        this.currentSection = 'dashboard'
        this.refreshInterval = null
        this.portfolio = null
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
                    this.drawPoincareVisualization()
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