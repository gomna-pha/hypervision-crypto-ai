/**
 * Trading API Integration
 * Real execution endpoints for institutional investors
 */

class TradingAPI {
    constructor() {
        this.baseURL = window.location.origin + '/api/v1';
        this.wsURL = window.location.origin.replace('http', 'ws') + '/ws';
        this.apiKey = null;
        this.apiSecret = null;
        this.connected = false;
        this.ws = null;
        
        // Exchange connectors
        this.exchanges = {
            binance: {
                spot: 'https://api.binance.com/api/v3',
                futures: 'https://fapi.binance.com/fapi/v1',
                testnet: 'https://testnet.binance.vision/api/v3'
            },
            coinbase: {
                api: 'https://api.exchange.coinbase.com',
                sandbox: 'https://api-public.sandbox.exchange.coinbase.com'
            },
            kraken: {
                api: 'https://api.kraken.com/0',
                futures: 'https://futures.kraken.com/derivatives/api/v3'
            }
        };
    }

    /**
     * Initialize API with credentials
     */
    async initialize(apiKey, apiSecret, exchange = 'binance', testMode = false) {
        this.apiKey = apiKey;
        this.apiSecret = apiSecret;
        this.exchange = exchange;
        this.testMode = testMode;
        
        try {
            const response = await this.authenticateAPI();
            if (response.success) {
                this.connected = true;
                this.connectWebSocket();
                return { success: true, message: 'API connected successfully' };
            }
        } catch (error) {
            console.error('API initialization failed:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Authenticate with exchange API
     */
    async authenticateAPI() {
        const timestamp = Date.now();
        const signature = this.generateSignature(timestamp);
        
        const response = await fetch(`${this.baseURL}/auth/connect`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-KEY': this.apiKey,
                'X-SIGNATURE': signature,
                'X-TIMESTAMP': timestamp
            },
            body: JSON.stringify({
                exchange: this.exchange,
                testMode: this.testMode
            })
        });
        
        return await response.json();
    }

    /**
     * Generate HMAC signature for authentication
     */
    generateSignature(timestamp) {
        const message = `${timestamp}${this.apiKey}`;
        // In production, use crypto-js or similar for HMAC-SHA256
        return btoa(message); // Simplified for demo
    }

    /**
     * Connect WebSocket for real-time updates
     */
    connectWebSocket() {
        this.ws = new WebSocket(this.wsURL);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.subscribeToStreams();
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleRealtimeUpdate(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            // Reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(), 5000);
        };
    }

    /**
     * Subscribe to real-time data streams
     */
    subscribeToStreams() {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify({
                action: 'subscribe',
                streams: ['trades', 'orders', 'positions', 'balance']
            }));
        }
    }

    /**
     * Handle real-time updates
     */
    handleRealtimeUpdate(data) {
        switch(data.type) {
            case 'trade':
                this.onTradeUpdate(data);
                break;
            case 'order':
                this.onOrderUpdate(data);
                break;
            case 'position':
                this.onPositionUpdate(data);
                break;
            case 'balance':
                this.onBalanceUpdate(data);
                break;
        }
    }

    /**
     * Place Market Order
     */
    async placeMarketOrder(symbol, side, quantity, options = {}) {
        const order = {
            symbol: symbol.toUpperCase(),
            side: side.toUpperCase(),
            type: 'MARKET',
            quantity: quantity,
            timestamp: Date.now(),
            ...options
        };

        try {
            const response = await fetch(`${this.baseURL}/orders/market`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-KEY': this.apiKey,
                    'X-SIGNATURE': this.generateSignature(order.timestamp)
                },
                body: JSON.stringify(order)
            });

            const result = await response.json();
            
            if (result.success) {
                this.onOrderPlaced(result.order);
            }
            
            return result;
        } catch (error) {
            console.error('Order placement failed:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Place Limit Order
     */
    async placeLimitOrder(symbol, side, quantity, price, options = {}) {
        const order = {
            symbol: symbol.toUpperCase(),
            side: side.toUpperCase(),
            type: 'LIMIT',
            quantity: quantity,
            price: price,
            timeInForce: options.timeInForce || 'GTC',
            timestamp: Date.now(),
            ...options
        };

        try {
            const response = await fetch(`${this.baseURL}/orders/limit`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-KEY': this.apiKey,
                    'X-SIGNATURE': this.generateSignature(order.timestamp)
                },
                body: JSON.stringify(order)
            });

            return await response.json();
        } catch (error) {
            console.error('Limit order failed:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Place Stop Loss Order
     */
    async placeStopLoss(symbol, quantity, stopPrice, options = {}) {
        const order = {
            symbol: symbol.toUpperCase(),
            side: 'SELL',
            type: 'STOP_LOSS',
            quantity: quantity,
            stopPrice: stopPrice,
            timestamp: Date.now(),
            ...options
        };

        try {
            const response = await fetch(`${this.baseURL}/orders/stop-loss`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-KEY': this.apiKey,
                    'X-SIGNATURE': this.generateSignature(order.timestamp)
                },
                body: JSON.stringify(order)
            });

            return await response.json();
        } catch (error) {
            console.error('Stop loss order failed:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Place OCO (One-Cancels-Other) Order
     */
    async placeOCOOrder(symbol, side, quantity, price, stopPrice, stopLimitPrice) {
        const order = {
            symbol: symbol.toUpperCase(),
            side: side.toUpperCase(),
            quantity: quantity,
            price: price,
            stopPrice: stopPrice,
            stopLimitPrice: stopLimitPrice,
            timestamp: Date.now()
        };

        try {
            const response = await fetch(`${this.baseURL}/orders/oco`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-KEY': this.apiKey,
                    'X-SIGNATURE': this.generateSignature(order.timestamp)
                },
                body: JSON.stringify(order)
            });

            return await response.json();
        } catch (error) {
            console.error('OCO order failed:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Cancel Order
     */
    async cancelOrder(orderId, symbol) {
        try {
            const response = await fetch(`${this.baseURL}/orders/${orderId}`, {
                method: 'DELETE',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-KEY': this.apiKey,
                    'X-SIGNATURE': this.generateSignature(Date.now())
                },
                body: JSON.stringify({ symbol: symbol })
            });

            return await response.json();
        } catch (error) {
            console.error('Order cancellation failed:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Get Open Orders
     */
    async getOpenOrders(symbol = null) {
        try {
            const params = symbol ? `?symbol=${symbol}` : '';
            const response = await fetch(`${this.baseURL}/orders/open${params}`, {
                headers: {
                    'X-API-KEY': this.apiKey,
                    'X-SIGNATURE': this.generateSignature(Date.now())
                }
            });

            return await response.json();
        } catch (error) {
            console.error('Failed to fetch open orders:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Get Account Balance
     */
    async getBalance() {
        try {
            const response = await fetch(`${this.baseURL}/account/balance`, {
                headers: {
                    'X-API-KEY': this.apiKey,
                    'X-SIGNATURE': this.generateSignature(Date.now())
                }
            });

            return await response.json();
        } catch (error) {
            console.error('Failed to fetch balance:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Get Trading History
     */
    async getTradingHistory(symbol = null, limit = 100) {
        try {
            const params = new URLSearchParams();
            if (symbol) params.append('symbol', symbol);
            params.append('limit', limit);
            
            const response = await fetch(`${this.baseURL}/trades/history?${params}`, {
                headers: {
                    'X-API-KEY': this.apiKey,
                    'X-SIGNATURE': this.generateSignature(Date.now())
                }
            });

            return await response.json();
        } catch (error) {
            console.error('Failed to fetch trading history:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Get Current Positions
     */
    async getPositions() {
        try {
            const response = await fetch(`${this.baseURL}/positions`, {
                headers: {
                    'X-API-KEY': this.apiKey,
                    'X-SIGNATURE': this.generateSignature(Date.now())
                }
            });

            return await response.json();
        } catch (error) {
            console.error('Failed to fetch positions:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Execute Agentic Trade
     */
    async executeAgenticTrade(signal) {
        // Validate signal
        if (!signal || !signal.symbol || !signal.action) {
            return { success: false, error: 'Invalid signal' };
        }

        // Calculate position size using Kelly Criterion
        const positionSize = this.calculatePositionSize(signal);
        
        // Place order based on signal
        let result;
        if (signal.action === 'BUY' || signal.action === 'SELL') {
            if (signal.orderType === 'MARKET') {
                result = await this.placeMarketOrder(
                    signal.symbol,
                    signal.action,
                    positionSize,
                    {
                        stopLoss: signal.stopLoss,
                        takeProfit: signal.takeProfit,
                        agentId: signal.agentId
                    }
                );
            } else {
                result = await this.placeLimitOrder(
                    signal.symbol,
                    signal.action,
                    positionSize,
                    signal.price,
                    {
                        stopLoss: signal.stopLoss,
                        takeProfit: signal.takeProfit,
                        agentId: signal.agentId
                    }
                );
            }
        } else if (signal.action === 'CLOSE') {
            result = await this.closePosition(signal.symbol, signal.positionId);
        }

        return result;
    }

    /**
     * Calculate Position Size (Kelly Criterion)
     */
    calculatePositionSize(signal) {
        const winProbability = signal.confidence / 100;
        const winLossRatio = signal.riskRewardRatio || 2;
        
        // Kelly Criterion: f* = (p * b - q) / b
        // where p = win probability, q = loss probability, b = win/loss ratio
        const kellyFraction = (winProbability * winLossRatio - (1 - winProbability)) / winLossRatio;
        
        // Apply safety factor (typically 0.25 to 0.5 of Kelly)
        const safetyFactor = 0.25;
        const optimalFraction = Math.max(0, Math.min(0.25, kellyFraction * safetyFactor));
        
        // Get account balance and calculate position size
        const accountBalance = signal.accountBalance || 100000;
        const positionValue = accountBalance * optimalFraction;
        const positionSize = positionValue / signal.price;
        
        return Math.floor(positionSize * 1000) / 1000; // Round to 3 decimals
    }

    /**
     * Close Position
     */
    async closePosition(symbol, positionId) {
        try {
            const response = await fetch(`${this.baseURL}/positions/${positionId}/close`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-KEY': this.apiKey,
                    'X-SIGNATURE': this.generateSignature(Date.now())
                },
                body: JSON.stringify({ symbol: symbol })
            });

            return await response.json();
        } catch (error) {
            console.error('Failed to close position:', error);
            return { success: false, error: error.message };
        }
    }

    /**
     * Event handlers
     */
    onOrderPlaced(order) {
        console.log('Order placed:', order);
        // Update UI
        if (window.updateOrdersUI) {
            window.updateOrdersUI(order);
        }
    }

    onTradeUpdate(trade) {
        console.log('Trade update:', trade);
        // Update UI
        if (window.updateTradesUI) {
            window.updateTradesUI(trade);
        }
    }

    onOrderUpdate(order) {
        console.log('Order update:', order);
    }

    onPositionUpdate(position) {
        console.log('Position update:', position);
    }

    onBalanceUpdate(balance) {
        console.log('Balance update:', balance);
    }
}

// Export for use in main application
if (typeof module !== 'undefined' && module.exports) {
    module.exports = TradingAPI;
}

// Initialize global instance
window.tradingAPI = new TradingAPI();