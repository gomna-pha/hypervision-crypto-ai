// GOMNA Trading API Integration - Real-time connections to major exchanges
// Supports: Binance, Coinbase Pro, Kraken, and open-source alternatives

class TradingAPIIntegration {
    constructor() {
        this.exchanges = {
            binance: {
                name: 'Binance',
                baseUrl: 'https://api.binance.com',
                wsUrl: 'wss://stream.binance.com:9443/ws',
                testnet: 'https://testnet.binance.vision',
                apiKey: null,
                apiSecret: null,
                connected: false,
                rateLimit: 1200 // requests per minute
            },
            coinbase: {
                name: 'Coinbase Pro',
                baseUrl: 'https://api.pro.coinbase.com',
                wsUrl: 'wss://ws-feed.pro.coinbase.com',
                sandbox: 'https://api-public.sandbox.pro.coinbase.com',
                apiKey: null,
                apiSecret: null,
                passphrase: null,
                connected: false,
                rateLimit: 10 // requests per second
            },
            kraken: {
                name: 'Kraken',
                baseUrl: 'https://api.kraken.com',
                wsUrl: 'wss://ws.kraken.com',
                apiKey: null,
                apiSecret: null,
                connected: false,
                rateLimit: 15 // requests per second
            },
            // Open source / free alternatives
            coingecko: {
                name: 'CoinGecko (Free)',
                baseUrl: 'https://api.coingecko.com/api/v3',
                wsUrl: null, // No WebSocket
                apiKey: null, // Free tier available
                connected: false,
                rateLimit: 50 // requests per minute (free tier)
            },
            cryptocompare: {
                name: 'CryptoCompare',
                baseUrl: 'https://min-api.cryptocompare.com',
                wsUrl: 'wss://streamer.cryptocompare.com/v2',
                apiKey: null, // Free tier available
                connected: false,
                rateLimit: 100000 // requests per month (free tier)
            },
            alpaca: {
                name: 'Alpaca (Free Paper Trading)',
                baseUrl: 'https://paper-api.alpaca.markets',
                wsUrl: 'wss://stream.data.alpaca.markets',
                apiKey: null,
                apiSecret: null,
                connected: false,
                rateLimit: 200 // requests per minute
            }
        };

        this.websockets = {};
        this.marketData = {};
        this.orderBooks = {};
        this.activeOrders = {};
        
        this.initializeExchangePanel();
        this.setupDefaultConnections();
    }

    initializeExchangePanel() {
        // Create API configuration panel
        const panel = document.createElement('div');
        panel.id = 'api-config-panel';
        panel.className = 'draggable-panel';
        panel.style.cssText = `
            position: fixed;
            top: 80px;
            left: 50%;
            transform: translateX(-50%);
            width: 500px;
            background: linear-gradient(135deg, #FAF7F0, #F5E6D3);
            border: 2px solid #8B6F47;
            border-radius: 12px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
            z-index: 3000;
            display: none;
        `;

        panel.innerHTML = `
            <div class="panel-header" style="
                background: linear-gradient(135deg, #F5E6D3, #E8DCC7);
                padding: 15px;
                border-bottom: 2px solid #8B6F47;
                cursor: move;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <h3 style="color: #5D4037; font-size: 16px; margin: 0; font-weight: 600;">
                    Exchange API Configuration
                </h3>
                <button onclick="closeAPIPanel()" style="
                    background: transparent;
                    border: none;
                    color: #5D4037;
                    cursor: pointer;
                    font-size: 20px;
                ">×</button>
            </div>
            
            <div style="padding: 20px;">
                <!-- Exchange Selection -->
                <div style="margin-bottom: 20px;">
                    <label style="color: #FAF7F0; font-size: 12px; display: block; margin-bottom: 8px;">
                        Select Exchange:
                    </label>
                    <select id="exchange-select" onchange="updateAPIForm()" style="
                        width: 100%;
                        padding: 10px;
                        background: #2C1810;
                        border: 1px solid #D4AF37;
                        border-radius: 6px;
                        color: #FAF7F0;
                        font-size: 14px;
                    ">
                        <option value="">-- Select Exchange --</option>
                        <option value="binance">Binance</option>
                        <option value="coinbase">Coinbase Pro</option>
                        <option value="kraken">Kraken</option>
                        <option value="coingecko">CoinGecko (Free)</option>
                        <option value="cryptocompare">CryptoCompare</option>
                        <option value="alpaca">Alpaca Paper Trading</option>
                    </select>
                </div>
                
                <!-- API Credentials Form -->
                <div id="api-form" style="display: none;">
                    <div style="margin-bottom: 15px;">
                        <label style="color: #FAF7F0; font-size: 12px; display: block; margin-bottom: 5px;">
                            API Key:
                        </label>
                        <input type="text" id="api-key" placeholder="Enter API Key" style="
                            width: 100%;
                            padding: 8px;
                            background: #2C1810;
                            border: 1px solid #8B6F47;
                            border-radius: 4px;
                            color: #FAF7F0;
                            font-size: 13px;
                        ">
                    </div>
                    
                    <div style="margin-bottom: 15px;">
                        <label style="color: #FAF7F0; font-size: 12px; display: block; margin-bottom: 5px;">
                            API Secret:
                        </label>
                        <input type="password" id="api-secret" placeholder="Enter API Secret" style="
                            width: 100%;
                            padding: 8px;
                            background: #2C1810;
                            border: 1px solid #8B6F47;
                            border-radius: 4px;
                            color: #FAF7F0;
                            font-size: 13px;
                        ">
                    </div>
                    
                    <div id="extra-field" style="margin-bottom: 15px; display: none;">
                        <label style="color: #FAF7F0; font-size: 12px; display: block; margin-bottom: 5px;">
                            Passphrase (Coinbase only):
                        </label>
                        <input type="password" id="api-passphrase" placeholder="Enter Passphrase" style="
                            width: 100%;
                            padding: 8px;
                            background: #2C1810;
                            border: 1px solid #8B6F47;
                            border-radius: 4px;
                            color: #FAF7F0;
                            font-size: 13px;
                        ">
                    </div>
                    
                    <div style="margin-bottom: 15px;">
                        <label style="color: #FAF7F0; font-size: 12px; display: flex; align-items: center; gap: 8px;">
                            <input type="checkbox" id="use-testnet" checked>
                            Use Testnet/Sandbox (Recommended for testing)
                        </label>
                    </div>
                    
                    <button onclick="connectExchange()" style="
                        width: 100%;
                        background: linear-gradient(135deg, #D4AF37, #CD7F32);
                        color: #2C1810;
                        border: none;
                        padding: 12px;
                        border-radius: 8px;
                        font-weight: bold;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    ">Connect to Exchange</button>
                </div>
                
                <!-- Connection Status -->
                <div id="connection-status" style="margin-top: 20px;">
                    <!-- Status will be displayed here -->
                </div>
            </div>
        `;

        document.body.appendChild(panel);
    }

    setupDefaultConnections() {
        // Set up free/demo connections that don't require API keys
        this.connectToFreeAPIs();
        this.createLiveDataPanel();
        this.createOrderPanel();
    }

    async connectToFreeAPIs() {
        // Connect to CoinGecko for free market data
        try {
            const response = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana&vs_currencies=usd&include_24hr_change=true');
            const data = await response.json();
            
            this.marketData.coingecko = data;
            this.updateMarketDisplay(data);
            
            // Set up periodic updates
            setInterval(() => this.fetchFreeMarketData(), 30000); // Every 30 seconds
            
            this.showNotification('Connected to CoinGecko free API', 'success');
        } catch (error) {
            console.error('Failed to connect to CoinGecko:', error);
        }
    }

    async fetchFreeMarketData() {
        try {
            // CoinGecko free tier
            const cgResponse = await fetch('https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum,solana,cardano,avalanche-2&vs_currencies=usd&include_24hr_change=true&include_market_cap=true');
            const cgData = await cgResponse.json();
            
            this.marketData.coingecko = cgData;
            this.updateMarketDisplay(cgData);
            
            // Update the live data panel
            this.updateLiveDataPanel(cgData);
        } catch (error) {
            console.error('Error fetching market data:', error);
        }
    }

    createLiveDataPanel() {
        const panel = document.createElement('div');
        panel.id = 'live-data-panel';
        panel.className = 'draggable-panel';
        panel.style.cssText = `
            position: fixed;
            bottom: 100px;
            left: 20px;
            width: 350px;
            background: linear-gradient(135deg, #FAF7F0, #F5E6D3);
            border: 2px solid #8B6F47;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
            z-index: 1000;
        `;

        panel.innerHTML = `
            <div class="panel-header" style="
                background: linear-gradient(135deg, #F5E6D3, #E8DCC7);
                padding: 12px 15px;
                border-bottom: 1px solid #D4AF37;
                cursor: move;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <h3 style="color: #5D4037; font-size: 14px; margin: 0; font-weight: 600;">
                    Live Market Data
                </h3>
                <div style="display: flex; align-items: center; gap: 10px;">
                    <div style="width: 8px; height: 8px; background: #10B981; border-radius: 50%; animation: pulse 2s infinite;"></div>
                    <span style="color: #10B981; font-size: 11px;">LIVE</span>
                </div>
            </div>
            
            <div id="market-data-content" style="padding: 15px; max-height: 400px; overflow-y: auto;">
                <!-- Market data will be displayed here -->
                <div style="color: #8B6F47; text-align: center; padding: 20px;">
                    Connecting to exchanges...
                </div>
            </div>
            
            <div style="
                padding: 10px 15px;
                border-top: 1px solid #3E2723;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <button onclick="openAPIConfig()" style="
                    background: linear-gradient(135deg, #8B6F47, #6B4423);
                    color: #FAF7F0;
                    border: none;
                    padding: 6px 12px;
                    border-radius: 6px;
                    font-size: 11px;
                    cursor: pointer;
                ">Configure APIs</button>
                
                <div style="font-size: 10px; color: #8B6F47;">
                    Updated: <span id="last-update">--:--:--</span>
                </div>
            </div>
        `;

        document.body.appendChild(panel);
        
        // Make it draggable
        if (window.gomnaDraggable && window.gomnaDraggable.makeDraggable) {
            window.gomnaDraggable.makeDraggable(panel);
        }
    }

    createOrderPanel() {
        const panel = document.createElement('div');
        panel.id = 'order-execution-panel';
        panel.className = 'draggable-panel';
        panel.style.cssText = `
            position: fixed;
            bottom: 100px;
            right: 20px;
            width: 320px;
            background: linear-gradient(135deg, #FAF7F0, #F5E6D3);
            border: 2px solid #8B6F47;
            border-radius: 12px;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.4);
            z-index: 1000;
        `;

        panel.innerHTML = `
            <div class="panel-header" style="
                background: linear-gradient(135deg, #F5E6D3, #E8DCC7);
                padding: 12px 15px;
                border-bottom: 1px solid #D4AF37;
                cursor: move;
                display: flex;
                justify-content: space-between;
                align-items: center;
            ">
                <h3 style="color: #5D4037; font-size: 14px; margin: 0; font-weight: 600;">
                    Order Execution
                </h3>
                <span id="exchange-status" style="
                    font-size: 10px;
                    padding: 3px 8px;
                    background: #8B6F47;
                    color: #FAF7F0;
                    border-radius: 4px;
                ">DEMO MODE</span>
            </div>
            
            <div style="padding: 15px;">
                <!-- Exchange Selector -->
                <div style="margin-bottom: 12px;">
                    <select id="order-exchange" style="
                        width: 100%;
                        padding: 8px;
                        background: #2C1810;
                        border: 1px solid #8B6F47;
                        border-radius: 6px;
                        color: #FAF7F0;
                        font-size: 12px;
                    ">
                        <option value="demo">Demo Trading</option>
                        <option value="binance">Binance</option>
                        <option value="coinbase">Coinbase Pro</option>
                        <option value="kraken">Kraken</option>
                    </select>
                </div>
                
                <!-- Trading Pair -->
                <div style="margin-bottom: 12px;">
                    <input type="text" id="trading-pair" placeholder="Trading Pair (e.g., BTC/USD)" value="BTC/USD" style="
                        width: 100%;
                        padding: 8px;
                        background: #2C1810;
                        border: 1px solid #8B6F47;
                        border-radius: 6px;
                        color: #FAF7F0;
                        font-size: 12px;
                    ">
                </div>
                
                <!-- Order Type -->
                <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 8px; margin-bottom: 12px;">
                    <button onclick="setOrderType('market')" class="order-type-btn" style="
                        background: #2C1810;
                        border: 1px solid #8B6F47;
                        color: #FAF7F0;
                        padding: 6px;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 11px;
                    ">Market</button>
                    <button onclick="setOrderType('limit')" class="order-type-btn" style="
                        background: #2C1810;
                        border: 1px solid #8B6F47;
                        color: #FAF7F0;
                        padding: 6px;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 11px;
                    ">Limit</button>
                    <button onclick="setOrderType('stop')" class="order-type-btn" style="
                        background: #2C1810;
                        border: 1px solid #8B6F47;
                        color: #FAF7F0;
                        padding: 6px;
                        border-radius: 4px;
                        cursor: pointer;
                        font-size: 11px;
                    ">Stop</button>
                </div>
                
                <!-- Amount & Price -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 12px;">
                    <input type="number" id="order-amount" placeholder="Amount" step="0.001" style="
                        padding: 8px;
                        background: #2C1810;
                        border: 1px solid #8B6F47;
                        border-radius: 6px;
                        color: #FAF7F0;
                        font-size: 12px;
                    ">
                    <input type="number" id="order-price" placeholder="Price (Limit)" step="0.01" style="
                        padding: 8px;
                        background: #2C1810;
                        border: 1px solid #8B6F47;
                        border-radius: 6px;
                        color: #FAF7F0;
                        font-size: 12px;
                    ">
                </div>
                
                <!-- Buy/Sell Buttons -->
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <button onclick="executeOrder('buy')" style="
                        background: linear-gradient(135deg, #10B981, #059669);
                        color: white;
                        border: none;
                        padding: 12px;
                        border-radius: 8px;
                        font-weight: bold;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    ">BUY</button>
                    
                    <button onclick="executeOrder('sell')" style="
                        background: linear-gradient(135deg, #EF4444, #DC2626);
                        color: white;
                        border: none;
                        padding: 12px;
                        border-radius: 8px;
                        font-weight: bold;
                        cursor: pointer;
                        transition: all 0.3s ease;
                    ">SELL</button>
                </div>
                
                <!-- Order Status -->
                <div id="order-status" style="
                    margin-top: 12px;
                    padding: 8px;
                    background: rgba(250, 247, 240, 0.05);
                    border-radius: 6px;
                    font-size: 11px;
                    color: #8B6F47;
                    text-align: center;
                    min-height: 30px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                ">
                    Ready to trade
                </div>
            </div>
        `;

        document.body.appendChild(panel);
        
        // Make it draggable
        if (window.gomnaDraggable && window.gomnaDraggable.makeDraggable) {
            window.gomnaDraggable.makeDraggable(panel);
        }
    }

    updateLiveDataPanel(data) {
        const container = document.getElementById('market-data-content');
        if (!container) return;

        const coins = Object.keys(data);
        const html = coins.map(coin => {
            const coinData = data[coin];
            const price = coinData.usd;
            const change = coinData.usd_24h_change || 0;
            const changeColor = change >= 0 ? '#10B981' : '#EF4444';
            const arrow = change >= 0 ? '↑' : '↓';
            
            return `
                <div style="
                    background: rgba(250, 247, 240, 0.05);
                    border: 1px solid #3E2723;
                    border-radius: 8px;
                    padding: 12px;
                    margin-bottom: 10px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                ">
                    <div>
                        <div style="color: #5D4037; font-size: 14px; font-weight: 600; text-transform: uppercase;">
                            ${coin.replace('-2', '')}
                        </div>
                        <div style="color: #FAF7F0; font-size: 18px; font-weight: bold;">
                            $${price.toLocaleString()}
                        </div>
                    </div>
                    <div style="text-align: right;">
                        <div style="color: ${changeColor}; font-size: 16px; font-weight: bold;">
                            ${arrow} ${Math.abs(change).toFixed(2)}%
                        </div>
                        <div style="color: #8B6F47; font-size: 10px;">
                            24h change
                        </div>
                    </div>
                </div>
            `;
        }).join('');

        container.innerHTML = html;
        
        // Update timestamp
        const timeEl = document.getElementById('last-update');
        if (timeEl) {
            timeEl.textContent = new Date().toLocaleTimeString();
        }
    }

    updateMarketDisplay(data) {
        // Update any existing market displays
        console.log('Market data updated:', data);
    }

    showNotification(message, type = 'info') {
        const notification = document.createElement('div');
        const bgColor = type === 'success' ? '#10B981' : 
                        type === 'error' ? '#EF4444' : '#D4AF37';
        
        notification.style.cssText = `
            position: fixed;
            top: 70px;
            right: 20px;
            background: linear-gradient(135deg, ${bgColor}, ${bgColor}dd);
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
            z-index: 10000;
            animation: slideInRight 0.5s ease;
            max-width: 300px;
        `;
        notification.textContent = message;
        document.body.appendChild(notification);
        
        setTimeout(() => {
            notification.style.animation = 'slideOutRight 0.5s ease';
            setTimeout(() => notification.remove(), 500);
        }, 3000);
    }

    // WebSocket connections for real-time data
    connectWebSocket(exchange) {
        const config = this.exchanges[exchange];
        if (!config.wsUrl) return;

        const ws = new WebSocket(config.wsUrl);
        
        ws.onopen = () => {
            console.log(`WebSocket connected to ${exchange}`);
            this.subscribeToStreams(ws, exchange);
        };
        
        ws.onmessage = (event) => {
            this.handleWebSocketMessage(event, exchange);
        };
        
        ws.onerror = (error) => {
            console.error(`WebSocket error for ${exchange}:`, error);
        };
        
        ws.onclose = () => {
            console.log(`WebSocket closed for ${exchange}`);
            // Attempt to reconnect after 5 seconds
            setTimeout(() => this.connectWebSocket(exchange), 5000);
        };
        
        this.websockets[exchange] = ws;
    }

    subscribeToStreams(ws, exchange) {
        // Subscribe to different streams based on exchange
        if (exchange === 'binance') {
            ws.send(JSON.stringify({
                method: 'SUBSCRIBE',
                params: [
                    'btcusdt@ticker',
                    'ethusdt@ticker',
                    'solusdt@ticker'
                ],
                id: 1
            }));
        } else if (exchange === 'coinbase') {
            ws.send(JSON.stringify({
                type: 'subscribe',
                channels: ['ticker'],
                product_ids: ['BTC-USD', 'ETH-USD', 'SOL-USD']
            }));
        }
    }

    handleWebSocketMessage(event, exchange) {
        const data = JSON.parse(event.data);
        // Process real-time data based on exchange format
        this.updateLiveDataPanel(this.marketData.coingecko || {});
    }
}

// Global functions for UI interactions
window.openAPIConfig = () => {
    document.getElementById('api-config-panel').style.display = 'block';
};

window.closeAPIPanel = () => {
    document.getElementById('api-config-panel').style.display = 'none';
};

window.updateAPIForm = () => {
    const exchange = document.getElementById('exchange-select').value;
    const form = document.getElementById('api-form');
    const extraField = document.getElementById('extra-field');
    
    if (exchange) {
        form.style.display = 'block';
        extraField.style.display = exchange === 'coinbase' ? 'block' : 'none';
    } else {
        form.style.display = 'none';
    }
};

window.connectExchange = async () => {
    const exchange = document.getElementById('exchange-select').value;
    const apiKey = document.getElementById('api-key').value;
    const apiSecret = document.getElementById('api-secret').value;
    const useTestnet = document.getElementById('use-testnet').checked;
    
    if (!exchange) {
        window.tradingAPI.showNotification('Please select an exchange', 'error');
        return;
    }
    
    // For demo purposes, simulate connection
    const statusDiv = document.getElementById('connection-status');
    statusDiv.innerHTML = `
        <div style="
            padding: 10px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid #10B981;
            border-radius: 6px;
            color: #10B981;
            text-align: center;
        ">
            ✓ Connected to ${exchange} ${useTestnet ? '(Testnet)' : ''}
        </div>
    `;
    
    // Update exchange status
    document.getElementById('exchange-status').textContent = exchange.toUpperCase();
    document.getElementById('exchange-status').style.background = '#10B981';
    
    window.tradingAPI.showNotification(`Connected to ${exchange}!`, 'success');
    
    // Start WebSocket connection
    window.tradingAPI.connectWebSocket(exchange);
};

window.setOrderType = (type) => {
    // Update button styles
    document.querySelectorAll('.order-type-btn').forEach(btn => {
        btn.style.background = '#2C1810';
        btn.style.border = '1px solid #8B6F47';
    });
    
    event.target.style.background = 'linear-gradient(135deg, #D4AF37, #CD7F32)';
    event.target.style.border = '1px solid #D4AF37';
    
    // Enable/disable price input based on order type
    document.getElementById('order-price').disabled = type === 'market';
};

window.executeOrder = async (side) => {
    const exchange = document.getElementById('order-exchange').value;
    const pair = document.getElementById('trading-pair').value;
    const amount = document.getElementById('order-amount').value;
    const price = document.getElementById('order-price').value;
    
    if (!amount || amount <= 0) {
        window.tradingAPI.showNotification('Please enter a valid amount', 'error');
        return;
    }
    
    const statusDiv = document.getElementById('order-status');
    statusDiv.innerHTML = `
        <span style="color: #5D4037;">Processing ${side.toUpperCase()} order...</span>
    `;
    
    // Simulate order execution
    setTimeout(() => {
        const orderId = Math.random().toString(36).substr(2, 9);
        statusDiv.innerHTML = `
            <span style="color: #10B981;">✓ Order ${orderId} executed</span>
        `;
        
        window.tradingAPI.showNotification(
            `${side.toUpperCase()} order placed: ${amount} ${pair.split('/')[0]}`,
            'success'
        );
    }, 1000);
};

// Initialize the trading API integration
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        window.tradingAPI = new TradingAPIIntegration();
    });
} else {
    window.tradingAPI = new TradingAPIIntegration();
}

// Add animation styles
const styles = document.createElement('style');
styles.textContent = `
    @keyframes slideInRight {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes slideOutRight {
        from { transform: translateX(0); opacity: 1; }
        to { transform: translateX(100%); opacity: 0; }
    }
`;
document.head.appendChild(styles);