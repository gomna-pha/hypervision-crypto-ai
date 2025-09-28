// Real-time Trading Integration for Backtesting and Paper Trading
// This module connects to the WebSocket server for live data streaming

class RealtimeTrading {
    constructor() {
        this.ws = null;
        this.reconnectInterval = 5000;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 10;
        this.isConnected = false;
        this.subscriptions = new Set();
        this.callbacks = new Map();
        this.marketData = new Map();
        this.backtestResults = null;
        this.paperTradingAccounts = new Map();
    }

    connect(port = 9000) {
        // Detect if we're in a sandbox environment
        const hostname = window.location.hostname;
        let wsUrl;
        
        if (hostname.includes('e2b.dev')) {
            // Sandbox environment - use public URL
            const sandboxHost = hostname.replace('8080', '9000').replace('8081', '9000').replace('8000', '9000');
            wsUrl = `wss://${sandboxHost}/ws`;
        } else {
            // Local development
            wsUrl = `ws://localhost:${port}/ws`;
        }
        
        console.log(`Connecting to WebSocket server at ${wsUrl}...`);
        
        try {
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log('WebSocket connected successfully');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.onConnect();
                
                // Subscribe to default symbols
                this.subscribeToMarketData(['BTC/USD', 'ETH/USD', 'SOL/USD', 'AVAX/USD']);
            };
            
            this.ws.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleMessage(data);
                } catch (error) {
                    console.error('Error parsing WebSocket message:', error);
                }
            };
            
            this.ws.onerror = (error) => {
                console.error('WebSocket error:', error);
                this.onError(error);
            };
            
            this.ws.onclose = () => {
                console.log('WebSocket connection closed');
                this.isConnected = false;
                this.onDisconnect();
                this.attemptReconnect();
            };
        } catch (error) {
            console.error('Failed to create WebSocket connection:', error);
            this.attemptReconnect();
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            console.log(`Attempting to reconnect (${this.reconnectAttempts}/${this.maxReconnectAttempts})...`);
            setTimeout(() => this.connect(), this.reconnectInterval);
        } else {
            console.error('Max reconnection attempts reached');
            this.onMaxReconnectAttemptsReached();
        }
    }
    
    send(message) {
        if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(message));
        } else {
            console.warn('WebSocket not connected, queuing message');
            // Queue message for later sending
        }
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'market_data':
                this.handleMarketData(data);
                break;
            case 'backtest_update':
                this.handleBacktestUpdate(data);
                break;
            case 'paper_trading_update':
                this.handlePaperTradingUpdate(data);
                break;
            case 'order_execution':
                this.handleOrderExecution(data);
                break;
            case 'error':
                this.handleError(data);
                break;
            default:
                console.log('Unknown message type:', data.type);
        }
    }
    
    handleMarketData(data) {
        const { symbol, bid, ask, last, volume, timestamp } = data.data;
        
        // Store market data
        this.marketData.set(symbol, {
            bid, ask, last, volume, timestamp,
            spread: ask - bid,
            mid: (bid + ask) / 2
        });
        
        // Update UI with real-time prices
        this.updateMarketDataUI(symbol, data.data);
        
        // Trigger callbacks
        const callbacks = this.callbacks.get('market_data');
        if (callbacks) {
            callbacks.forEach(cb => cb(data.data));
        }
    }
    
    handleBacktestUpdate(data) {
        const { metrics, trades, equity_curve, status } = data.data;
        
        // Update backtest results
        this.backtestResults = data.data;
        
        // Update UI
        this.updateBacktestUI(data.data);
        
        // Trigger callbacks
        const callbacks = this.callbacks.get('backtest_update');
        if (callbacks) {
            callbacks.forEach(cb => cb(data.data));
        }
    }
    
    handlePaperTradingUpdate(data) {
        const { account_id, positions, orders, balance, pnl } = data.data;
        
        // Update paper trading account
        this.paperTradingAccounts.set(account_id, data.data);
        
        // Update UI
        this.updatePaperTradingUI(account_id, data.data);
        
        // Trigger callbacks
        const callbacks = this.callbacks.get('paper_trading_update');
        if (callbacks) {
            callbacks.forEach(cb => cb(data.data));
        }
    }
    
    handleOrderExecution(data) {
        const { order_id, status, fill_price, fill_quantity, timestamp } = data.data;
        
        console.log(`Order ${order_id} ${status} at ${fill_price}`);
        
        // Update order status in UI
        this.updateOrderStatusUI(data.data);
        
        // Trigger callbacks
        const callbacks = this.callbacks.get('order_execution');
        if (callbacks) {
            callbacks.forEach(cb => cb(data.data));
        }
    }
    
    handleError(data) {
        console.error('Server error:', data.error);
        // Show error notification in UI
        this.showErrorNotification(data.error);
    }
    
    // Subscribe to market data for specific symbols
    subscribeToMarketData(symbols) {
        this.send({
            action: 'subscribe',
            data: {
                type: 'market_data',
                symbols: symbols
            }
        });
        
        symbols.forEach(symbol => this.subscriptions.add(symbol));
    }
    
    // Start real-time backtesting
    startRealtimeBacktest(strategyId, config) {
        this.send({
            action: 'start_backtest',
            data: {
                strategy_id: strategyId,
                config: config,
                realtime: true
            }
        });
    }
    
    // Create paper trading account
    createPaperTradingAccount(accountConfig) {
        this.send({
            action: 'create_paper_account',
            data: accountConfig
        });
    }
    
    // Place paper trading order
    placePaperOrder(accountId, order) {
        this.send({
            action: 'place_order',
            data: {
                account_id: accountId,
                order: order
            }
        });
    }
    
    // UI Update Methods
    updateMarketDataUI(symbol, data) {
        // Update price displays
        const priceElements = document.querySelectorAll(`[data-symbol="${symbol}"]`);
        priceElements.forEach(elem => {
            if (elem.dataset.field === 'price') {
                elem.textContent = `$${data.last.toFixed(2)}`;
                
                // Add price change animation
                elem.classList.add('price-update');
                setTimeout(() => elem.classList.remove('price-update'), 300);
            } else if (elem.dataset.field === 'volume') {
                elem.textContent = this.formatVolume(data.volume);
            } else if (elem.dataset.field === 'spread') {
                elem.textContent = `$${(data.ask - data.bid).toFixed(4)}`;
            }
        });
        
        // Update charts if visible
        if (window.updatePriceChart) {
            window.updatePriceChart(symbol, data);
        }
    }
    
    updateBacktestUI(results) {
        // Update backtest metrics
        if (results.metrics) {
            document.getElementById('backtestTotalReturn').textContent = 
                `${(results.metrics.total_return * 100).toFixed(2)}%`;
            document.getElementById('backtestSharpe').textContent = 
                results.metrics.sharpe_ratio.toFixed(2);
            document.getElementById('backtestWinRate').textContent = 
                `${(results.metrics.win_rate * 100).toFixed(1)}%`;
            document.getElementById('backtestMaxDrawdown').textContent = 
                `${(results.metrics.max_drawdown * 100).toFixed(2)}%`;
            
            // Update status
            const statusElem = document.getElementById('backtestStatus');
            if (statusElem) {
                statusElem.textContent = results.status || 'Running';
                statusElem.className = results.status === 'completed' ? 'status-complete' : 'status-running';
            }
        }
        
        // Update equity curve chart
        if (results.equity_curve && window.updateEquityChart) {
            window.updateEquityChart(results.equity_curve);
        }
        
        // Update trade list
        if (results.trades && results.trades.length > 0) {
            this.updateTradeListUI(results.trades.slice(-20));
        }
    }
    
    updatePaperTradingUI(accountId, accountData) {
        // Update account balance
        document.getElementById('paperBalance').textContent = 
            `$${accountData.balance.toFixed(2)}`;
        document.getElementById('paperTotalPnL').textContent = 
            `$${accountData.pnl.toFixed(2)}`;
        
        // Update positions
        if (accountData.positions && accountData.positions.length > 0) {
            const positionsHTML = accountData.positions.map(pos => `
                <tr>
                    <td>${pos.symbol}</td>
                    <td>${pos.quantity}</td>
                    <td>$${pos.entry_price.toFixed(2)}</td>
                    <td>$${pos.current_price.toFixed(2)}</td>
                    <td class="${pos.pnl >= 0 ? 'positive' : 'negative'}">
                        $${pos.pnl.toFixed(2)}
                    </td>
                    <td>
                        <button onclick="realtimeTrading.closePaperPosition('${accountId}', '${pos.symbol}')" 
                                class="btn-close">Close</button>
                    </td>
                </tr>
            `).join('');
            
            document.getElementById('paperPositions').innerHTML = positionsHTML;
        }
        
        // Update open orders
        if (accountData.orders && accountData.orders.length > 0) {
            this.updateOrdersUI(accountData.orders);
        }
    }
    
    updateOrderStatusUI(orderData) {
        const orderElem = document.getElementById(`order-${orderData.order_id}`);
        if (orderElem) {
            orderElem.querySelector('.order-status').textContent = orderData.status;
            if (orderData.status === 'filled') {
                orderElem.querySelector('.fill-price').textContent = 
                    `Filled @ $${orderData.fill_price.toFixed(2)}`;
            }
        }
    }
    
    updateTradeListUI(trades) {
        const tradesHTML = trades.map(trade => `
            <div class="trade-item">
                <div class="trade-header">
                    <span class="trade-symbol">${trade.symbol}</span>
                    <span class="trade-side ${trade.side.toLowerCase()}">${trade.side}</span>
                    <span class="trade-time">${new Date(trade.timestamp).toLocaleTimeString()}</span>
                </div>
                <div class="trade-details">
                    <span>Qty: ${trade.quantity}</span>
                    <span>Price: $${trade.price.toFixed(2)}</span>
                    <span class="trade-pnl ${trade.pnl >= 0 ? 'positive' : 'negative'}">
                        P&L: $${trade.pnl.toFixed(2)}
                    </span>
                </div>
            </div>
        `).join('');
        
        const container = document.getElementById('recentTrades');
        if (container) {
            container.innerHTML = tradesHTML;
        }
    }
    
    updateOrdersUI(orders) {
        const ordersHTML = orders.map(order => `
            <div id="order-${order.id}" class="order-item">
                <div class="order-header">
                    <span>${order.symbol} ${order.side}</span>
                    <span class="order-status">${order.status}</span>
                </div>
                <div class="order-details">
                    <span>Qty: ${order.quantity}</span>
                    <span>${order.type}: ${order.type === 'LIMIT' ? `$${order.limit_price}` : 'Market'}</span>
                    <span class="fill-price"></span>
                </div>
                ${order.status === 'pending' ? `
                    <button onclick="realtimeTrading.cancelOrder('${order.id}')" 
                            class="btn-cancel">Cancel</button>
                ` : ''}
            </div>
        `).join('');
        
        document.getElementById('paperOrders').innerHTML = ordersHTML || 
            '<div class="no-data">No open orders</div>';
    }
    
    showErrorNotification(error) {
        const notification = document.createElement('div');
        notification.className = 'error-notification';
        notification.textContent = error;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.remove();
        }, 5000);
    }
    
    // Utility methods
    formatVolume(volume) {
        if (volume >= 1e9) return `${(volume / 1e9).toFixed(2)}B`;
        if (volume >= 1e6) return `${(volume / 1e6).toFixed(2)}M`;
        if (volume >= 1e3) return `${(volume / 1e3).toFixed(2)}K`;
        return volume.toFixed(0);
    }
    
    closePaperPosition(accountId, symbol) {
        const position = this.paperTradingAccounts.get(accountId)?.positions?.find(p => p.symbol === symbol);
        if (position) {
            this.placePaperOrder(accountId, {
                symbol: symbol,
                side: 'SELL',
                type: 'MARKET',
                quantity: position.quantity
            });
        }
    }
    
    cancelOrder(orderId) {
        this.send({
            action: 'cancel_order',
            data: { order_id: orderId }
        });
    }
    
    // Event callbacks
    on(event, callback) {
        if (!this.callbacks.has(event)) {
            this.callbacks.set(event, new Set());
        }
        this.callbacks.get(event).add(callback);
    }
    
    off(event, callback) {
        const callbacks = this.callbacks.get(event);
        if (callbacks) {
            callbacks.delete(callback);
        }
    }
    
    // Connection event handlers
    onConnect() {
        console.log('Real-time trading system connected');
        // Update UI to show connected status
        const statusIndicator = document.getElementById('wsStatus');
        if (statusIndicator) {
            statusIndicator.className = 'status-connected';
            statusIndicator.textContent = 'Connected';
        }
    }
    
    onDisconnect() {
        console.log('Real-time trading system disconnected');
        // Update UI to show disconnected status
        const statusIndicator = document.getElementById('wsStatus');
        if (statusIndicator) {
            statusIndicator.className = 'status-disconnected';
            statusIndicator.textContent = 'Disconnected';
        }
    }
    
    onError(error) {
        console.error('WebSocket error:', error);
    }
    
    onMaxReconnectAttemptsReached() {
        console.error('Could not establish WebSocket connection after multiple attempts');
        this.showErrorNotification('Unable to connect to real-time data feed. Please refresh the page.');
    }
}

// Initialize global instance
window.realtimeTrading = new RealtimeTrading();

// Auto-connect when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Try to connect to WebSocket server
    window.realtimeTrading.connect(9000);
    
    // Set up event listeners for real-time updates
    window.realtimeTrading.on('market_data', (data) => {
        // Custom market data handler
        console.log('Market data update:', data);
    });
    
    window.realtimeTrading.on('backtest_update', (data) => {
        // Custom backtest update handler
        console.log('Backtest update:', data);
    });
    
    window.realtimeTrading.on('paper_trading_update', (data) => {
        // Custom paper trading update handler
        console.log('Paper trading update:', data);
    });
});

// Export for use in other modules
if (typeof module !== 'undefined' && module.exports) {
    module.exports = RealtimeTrading;
}