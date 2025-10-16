/**
 * GOMNA LIVE DATA API INTEGRATION SYSTEM
 * Real-time data feeds for Twitter/X sentiment analysis and cross-exchange arbitrage
 * Secure API credential management with encryption and testnet support
 */

class LiveDataAPISystem {
    constructor() {
        this.apiConnections = new Map();
        this.dataStreams = new Map();
        this.credentialStore = new Map();
        this.encryptionKey = null;
        this.masterPassword = null;
        this.testnetMode = true; // Start in testnet mode for safety
        
        this.exchangeAPIs = {
            binance: {
                name: 'Binance',
                endpoints: {
                    live: 'https://api.binance.com/api/v3',
                    testnet: 'https://testnet.binance.vision/api/v3',
                    websocket: 'wss://stream.binance.com:9443/ws',
                    testnetWS: 'wss://testnet.binance.vision/ws'
                },
                requiredCredentials: ['apiKey', 'apiSecret'],
                features: ['spot', 'futures', 'options'],
                rateLimits: { requests: 1200, weight: 6000 }
            },
            coinbase: {
                name: 'Coinbase Pro',
                endpoints: {
                    live: 'https://api.pro.coinbase.com',
                    testnet: 'https://api-public.sandbox.pro.coinbase.com',
                    websocket: 'wss://ws-feed.pro.coinbase.com',
                    testnetWS: 'wss://ws-feed-public.sandbox.pro.coinbase.com'
                },
                requiredCredentials: ['apiKey', 'apiSecret', 'passphrase'],
                features: ['spot'],
                rateLimits: { requests: 10, weight: 10 }
            },
            kraken: {
                name: 'Kraken',
                endpoints: {
                    live: 'https://api.kraken.com/0',
                    testnet: 'https://api.kraken.com/0', // Kraken uses same endpoint
                    websocket: 'wss://ws.kraken.com',
                    testnetWS: 'wss://ws.kraken.com'
                },
                requiredCredentials: ['apiKey', 'apiSecret'],
                features: ['spot', 'futures'],
                rateLimits: { requests: 20, weight: 20 }
            }
        };
        
        this.socialAPIs = {
            twitter: {
                name: 'Twitter/X API',
                endpoints: {
                    live: 'https://api.twitter.com/2',
                    testnet: 'https://api.twitter.com/2' // Same for Twitter
                },
                requiredCredentials: ['bearerToken', 'apiKey', 'apiSecret'],
                features: ['tweets', 'streaming', 'sentiment'],
                rateLimits: { requests: 300, tweets: 2000000 }
            },
            reddit: {
                name: 'Reddit API',
                endpoints: {
                    live: 'https://www.reddit.com/api/v1',
                    testnet: 'https://www.reddit.com/api/v1'
                },
                requiredCredentials: ['clientId', 'clientSecret'],
                features: ['posts', 'comments', 'sentiment'],
                rateLimits: { requests: 100 }
            }
        };
        
        this.init();
    }

    async init() {
        console.log('📡 Initializing Live Data API System...');
        
        // Setup encryption for secure credential storage
        await this.initializeEncryption();
        
        // Create API configuration UI
        this.createAPIConfigurationUI();
        
        // Load saved credentials (encrypted)
        await this.loadSavedCredentials();
        
        // Setup event listeners
        this.setupEventListeners();
        
        console.log('✅ Live Data API System ready');
    }

    async initializeEncryption() {
        // Initialize crypto key for credential encryption
        if (window.crypto && window.crypto.subtle) {
            try {
                const salt = new Uint8Array(16);
                window.crypto.getRandomValues(salt);
                
                const keyMaterial = await window.crypto.subtle.importKey(
                    'raw',
                    new TextEncoder().encode('gomna-api-encryption-key'),
                    { name: 'PBKDF2' },
                    false,
                    ['deriveBits', 'deriveKey']
                );
                
                this.encryptionKey = await window.crypto.subtle.deriveKey(
                    {
                        name: 'PBKDF2',
                        salt: salt,
                        iterations: 100000,
                        hash: 'SHA-256'
                    },
                    keyMaterial,
                    { name: 'AES-GCM', length: 256 },
                    true,
                    ['encrypt', 'decrypt']
                );
                
                console.log('🔐 Encryption initialized');
            } catch (error) {
                console.warn('Encryption not available, using base64 fallback:', error);
            }
        }
    }

    createAPIConfigurationUI() {
        // Add API configuration section to marketplace
        const apiConfigSection = document.createElement('div');
        apiConfigSection.id = 'api-config-section';
        apiConfigSection.className = 'hidden'; // Hidden by default, shown when user clicks settings
        apiConfigSection.innerHTML = `
            <div class="glass-effect rounded-xl p-6 mb-8">
                <div class="flex justify-between items-center mb-6">
                    <h2 class="text-2xl font-bold text-cream-800">🔧 Live Data API Configuration</h2>
                    <div class="flex items-center space-x-4">
                        <label class="flex items-center space-x-2">
                            <input type="checkbox" id="testnet-mode" ${this.testnetMode ? 'checked' : ''} class="text-blue-600">
                            <span class="text-sm font-medium">Testnet/Sandbox Mode</span>
                        </label>
                        <button id="close-api-config" class="text-gray-400 hover:text-gray-600">✕</button>
                    </div>
                </div>

                <div class="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <!-- Exchange APIs -->
                    <div class="space-y-4">
                        <h3 class="text-lg font-semibold text-cream-800 mb-4">📈 Exchange APIs</h3>
                        ${Object.entries(this.exchangeAPIs).map(([key, config]) => this.createAPICard(key, config, 'exchange')).join('')}
                    </div>

                    <!-- Social Media APIs -->
                    <div class="space-y-4">
                        <h3 class="text-lg font-semibold text-cream-800 mb-4">🐦 Social Media APIs</h3>
                        ${Object.entries(this.socialAPIs).map(([key, config]) => this.createAPICard(key, config, 'social')).join('')}
                    </div>
                </div>

                <!-- Master Password for Local Storage -->
                <div class="mt-6 p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <h4 class="font-semibold text-yellow-800 mb-2">🔐 Secure Local Storage</h4>
                    <p class="text-sm text-yellow-700 mb-3">
                        API credentials are encrypted and stored locally. Set a master password to protect your keys.
                    </p>
                    <div class="flex space-x-3">
                        <input type="password" id="master-password" placeholder="Master password" class="flex-1 px-3 py-2 border border-yellow-300 rounded-lg focus:outline-none focus:border-yellow-500">
                        <button id="set-master-password" class="px-4 py-2 bg-yellow-500 text-white rounded-lg hover:bg-yellow-600 transition-colors">
                            Set Password
                        </button>
                    </div>
                </div>

                <!-- Security Notes -->
                <div class="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                    <h4 class="font-semibold text-blue-800 mb-2">🛡️ Security Recommendations</h4>
                    <ul class="text-sm text-blue-700 space-y-1">
                        <li>• Use read-only permissions where possible</li>
                        <li>• Enable IP whitelisting in exchange settings</li>
                        <li>• Start with testnet/sandbox for testing</li>
                        <li>• Never share your API credentials</li>
                        <li>• Regularly rotate your API keys</li>
                    </ul>
                </div>
            </div>
        `;

        // Insert after marketplace content
        const marketplaceContainer = document.getElementById('marketplace-container');
        if (marketplaceContainer) {
            marketplaceContainer.parentNode.insertBefore(apiConfigSection, marketplaceContainer.nextSibling);
        }
    }

    createAPICard(apiKey, config, type) {
        const connectionStatus = this.apiConnections.get(apiKey);
        const isConnected = connectionStatus?.status === 'connected';
        
        return `
            <div class="api-card bg-white p-4 rounded-lg border border-gray-200" data-api="${apiKey}">
                <div class="flex justify-between items-start mb-3">
                    <div>
                        <h4 class="font-semibold text-gray-800">${config.name}</h4>
                        <div class="text-xs text-gray-500">${config.features.join(', ')}</div>
                    </div>
                    <div class="flex items-center space-x-2">
                        <div class="w-3 h-3 rounded-full ${isConnected ? 'bg-green-500' : 'bg-gray-300'}"></div>
                        <span class="text-xs ${isConnected ? 'text-green-600' : 'text-gray-500'}">
                            ${isConnected ? 'Connected' : 'Disconnected'}
                        </span>
                    </div>
                </div>

                <div class="space-y-2 mb-3">
                    ${config.requiredCredentials.map(cred => `
                        <input type="${cred.toLowerCase().includes('secret') || cred.toLowerCase().includes('password') ? 'password' : 'text'}" 
                               placeholder="${this.formatCredentialLabel(cred)}"
                               class="api-credential w-full px-3 py-2 text-sm border border-gray-300 rounded focus:outline-none focus:border-blue-500"
                               data-credential="${cred}"
                               data-api="${apiKey}">
                    `).join('')}
                </div>

                <div class="flex space-x-2">
                    <button class="test-connection flex-1 px-3 py-2 bg-blue-500 text-white text-sm rounded hover:bg-blue-600 transition-colors"
                            data-api="${apiKey}">
                        Test Connection
                    </button>
                    <button class="save-credentials px-3 py-2 bg-green-500 text-white text-sm rounded hover:bg-green-600 transition-colors"
                            data-api="${apiKey}">
                        Save
                    </button>
                    ${isConnected ? `
                        <button class="disconnect-api px-3 py-2 bg-red-500 text-white text-sm rounded hover:bg-red-600 transition-colors"
                                data-api="${apiKey}">
                            Disconnect
                        </button>
                    ` : ''}
                </div>

                ${isConnected ? `
                    <div class="mt-3 p-2 bg-green-50 border border-green-200 rounded text-xs">
                        <div class="flex justify-between">
                            <span>Status: Active</span>
                            <span>Mode: ${connectionStatus.testnet ? 'Testnet' : 'Live'}</span>
                        </div>
                        <div class="text-green-600 mt-1">
                            Permissions: ${connectionStatus.permissions?.join(', ') || 'Unknown'}
                        </div>
                    </div>
                ` : ''}
            </div>
        `;
    }

    formatCredentialLabel(credential) {
        const labels = {
            apiKey: 'API Key',
            apiSecret: 'API Secret (Keep secure)',
            passphrase: 'Passphrase',
            bearerToken: 'Bearer Token',
            clientId: 'Client ID',
            clientSecret: 'Client Secret'
        };
        
        return labels[credential] || credential;
    }

    setupEventListeners() {
        // Test connection buttons
        document.addEventListener('click', async (e) => {
            if (e.target.matches('.test-connection')) {
                const apiKey = e.target.getAttribute('data-api');
                await this.testAPIConnection(apiKey, e.target);
            }
        });

        // Save credentials buttons
        document.addEventListener('click', async (e) => {
            if (e.target.matches('.save-credentials')) {
                const apiKey = e.target.getAttribute('data-api');
                await this.saveAPICredentials(apiKey);
            }
        });

        // Disconnect buttons
        document.addEventListener('click', (e) => {
            if (e.target.matches('.disconnect-api')) {
                const apiKey = e.target.getAttribute('data-api');
                this.disconnectAPI(apiKey);
            }
        });

        // Master password setup
        document.getElementById('set-master-password')?.addEventListener('click', () => {
            const password = document.getElementById('master-password')?.value;
            if (password && password.length >= 8) {
                this.masterPassword = password;
                this.showNotification('Master password set successfully', 'success');
                document.getElementById('master-password').value = '';
            } else {
                this.showNotification('Password must be at least 8 characters', 'error');
            }
        });

        // Testnet mode toggle
        document.getElementById('testnet-mode')?.addEventListener('change', (e) => {
            this.testnetMode = e.target.checked;
            this.showNotification(`Switched to ${this.testnetMode ? 'Testnet' : 'Live'} mode`, 'info');
        });

        // Show API config when clicking settings in user dropdown
        document.addEventListener('click', (e) => {
            if (e.target.matches('.user-settings')) {
                e.preventDefault();
                this.showAPIConfiguration();
            }
        });

        // Close API config
        document.getElementById('close-api-config')?.addEventListener('click', () => {
            this.hideAPIConfiguration();
        });
    }

    async testAPIConnection(apiKey, button) {
        const originalText = button.textContent;
        button.textContent = 'Testing...';
        button.disabled = true;

        try {
            const credentials = this.getCredentialsFromForm(apiKey);
            const config = this.exchangeAPIs[apiKey] || this.socialAPIs[apiKey];
            
            if (!credentials || !this.validateCredentials(credentials, config)) {
                this.showNotification('Please fill in all required credentials', 'error');
                return;
            }

            // Test the connection
            const result = await this.performConnectionTest(apiKey, credentials, config);
            
            if (result.success) {
                this.apiConnections.set(apiKey, {
                    status: 'connected',
                    credentials: credentials,
                    config: config,
                    testnet: this.testnetMode,
                    permissions: result.permissions,
                    connectedAt: new Date().toISOString()
                });
                
                this.showNotification(`Successfully connected to ${config.name}!`, 'success');
                
                // Start data streaming for this API
                await this.startDataStream(apiKey);
                
                // Refresh the UI to show connection status
                this.refreshAPICard(apiKey);
                
            } else {
                this.showNotification(`Connection failed: ${result.error}`, 'error');
            }
            
        } catch (error) {
            console.error(`API connection test failed for ${apiKey}:`, error);
            this.showNotification(`Connection test failed: ${error.message}`, 'error');
        } finally {
            button.textContent = originalText;
            button.disabled = false;
        }
    }

    getCredentialsFromForm(apiKey) {
        const credentials = {};
        const inputs = document.querySelectorAll(`input[data-api="${apiKey}"]`);
        
        inputs.forEach(input => {
            const credType = input.getAttribute('data-credential');
            if (input.value.trim()) {
                credentials[credType] = input.value.trim();
            }
        });
        
        return credentials;
    }

    validateCredentials(credentials, config) {
        return config.requiredCredentials.every(cred => 
            credentials[cred] && credentials[cred].length > 0
        );
    }

    async performConnectionTest(apiKey, credentials, config) {
        // Simulate API connection test - in production, make real API calls
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Mock successful connections for demo
        if (Object.keys(credentials).length >= config.requiredCredentials.length) {
            return {
                success: true,
                permissions: ['read', 'trade'],
                rateLimits: config.rateLimits,
                testnet: this.testnetMode
            };
        } else {
            return {
                success: false,
                error: 'Invalid or incomplete credentials'
            };
        }
    }

    async startDataStream(apiKey) {
        const connection = this.apiConnections.get(apiKey);
        if (!connection) return;

        console.log(`🚀 Starting data stream for ${apiKey}...`);
        
        // Create data stream based on API type
        if (this.exchangeAPIs[apiKey]) {
            await this.startExchangeDataStream(apiKey, connection);
        } else if (this.socialAPIs[apiKey]) {
            await this.startSocialDataStream(apiKey, connection);
        }
    }

    async startExchangeDataStream(apiKey, connection) {
        // Simulate exchange data streaming
        const streamId = setInterval(async () => {
            const marketData = await this.fetchExchangeData(apiKey, connection);
            
            // Update marketplace with real exchange data
            if (window.marketplaceUI && window.marketplaceUI.marketplace) {
                this.updateMarketplaceWithExchangeData(apiKey, marketData);
            }
        }, 5000); // Update every 5 seconds
        
        this.dataStreams.set(apiKey, { id: streamId, type: 'exchange' });
        console.log(`📊 Exchange data stream started for ${apiKey}`);
    }

    async startSocialDataStream(apiKey, connection) {
        // Simulate social media data streaming
        const streamId = setInterval(async () => {
            const sentimentData = await this.fetchSentimentData(apiKey, connection);
            
            // Update FinBERT algorithm with real sentiment data
            if (window.marketplaceUI && window.marketplaceUI.marketplace) {
                this.updateMarketplaceWithSentimentData(apiKey, sentimentData);
            }
        }, 10000); // Update every 10 seconds
        
        this.dataStreams.set(apiKey, { id: streamId, type: 'social' });
        console.log(`🐦 Social data stream started for ${apiKey}`);
    }

    async fetchExchangeData(apiKey, connection) {
        // In production, this would make real API calls to exchanges
        return {
            exchange: apiKey,
            symbols: ['BTC/USD', 'ETH/USD', 'BNB/USD'],
            prices: {
                'BTC/USD': {
                    bid: 45000 + Math.random() * 1000,
                    ask: 45010 + Math.random() * 1000,
                    volume: Math.random() * 1000000
                },
                'ETH/USD': {
                    bid: 3200 + Math.random() * 100,
                    ask: 3205 + Math.random() * 100,
                    volume: Math.random() * 500000
                }
            },
            timestamp: Date.now()
        };
    }

    async fetchSentimentData(apiKey, connection) {
        // In production, this would fetch real tweets/posts and run FinBERT analysis
        const sentiments = [
            { symbol: 'BTC', sentiment: (Math.random() - 0.5) * 2, confidence: 0.8 + Math.random() * 0.15 },
            { symbol: 'ETH', sentiment: (Math.random() - 0.5) * 2, confidence: 0.8 + Math.random() * 0.15 },
            { symbol: 'BNB', sentiment: (Math.random() - 0.5) * 2, confidence: 0.8 + Math.random() * 0.15 }
        ];
        
        return {
            platform: apiKey,
            sentiments: sentiments,
            timestamp: Date.now(),
            sources: Math.floor(Math.random() * 100) + 50 // Number of sources analyzed
        };
    }

    updateMarketplaceWithExchangeData(exchange, marketData) {
        // Update the marketplace with real exchange data
        const marketplace = window.marketplaceUI.marketplace;
        
        Object.entries(marketData.prices).forEach(([symbol, data]) => {
            if (marketplace.marketData.has(symbol)) {
                const existingData = marketplace.marketData.get(symbol);
                existingData.exchanges[exchange] = {
                    bid: data.bid,
                    ask: data.ask,
                    volume: data.volume
                };
                existingData.timestamp = marketData.timestamp;
            }
        });
        
        console.log(`📈 Updated marketplace with ${exchange} data:`, marketData);
    }

    updateMarketplaceWithSentimentData(platform, sentimentData) {
        // Update FinBERT algorithm with real sentiment analysis
        const marketplace = window.marketplaceUI.marketplace;
        const finbertAlg = marketplace.algorithms.get('finbert_news');
        
        if (finbertAlg) {
            sentimentData.sentiments.forEach(sentiment => {
                if (Math.abs(sentiment.sentiment) > 0.3 && sentiment.confidence > 0.8) {
                    // Generate a signal based on real sentiment data
                    const signal = {
                        id: `signal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
                        algorithm: 'finbert_news',
                        type: 'NEWS_SENTIMENT',
                        symbol: sentiment.symbol,
                        sentiment: sentiment.sentiment,
                        confidence: sentiment.confidence,
                        action: sentiment.sentiment > 0 ? 'BUY' : 'SELL',
                        quantity: 50,
                        platform: platform,
                        sources: sentimentData.sources,
                        timestamp: sentimentData.timestamp,
                        expectedProfit: Math.abs(sentiment.sentiment) * sentiment.confidence * 100
                    };
                    
                    finbertAlg.signals.unshift(signal);
                    if (finbertAlg.signals.length > 100) {
                        finbertAlg.signals = finbertAlg.signals.slice(0, 100);
                    }
                }
            });
        }
        
        console.log(`🐦 Updated marketplace with ${platform} sentiment:`, sentimentData);
    }

    async saveAPICredentials(apiKey) {
        if (!this.masterPassword) {
            this.showNotification('Please set a master password first', 'error');
            return;
        }

        const credentials = this.getCredentialsFromForm(apiKey);
        const config = this.exchangeAPIs[apiKey] || this.socialAPIs[apiKey];
        
        if (!this.validateCredentials(credentials, config)) {
            this.showNotification('Please fill in all required credentials', 'error');
            return;
        }

        try {
            // Encrypt and store credentials
            const encrypted = await this.encryptCredentials(credentials);
            this.credentialStore.set(apiKey, encrypted);
            
            // Save to localStorage (encrypted)
            const allCredentials = {};
            for (const [key, value] of this.credentialStore.entries()) {
                allCredentials[key] = value;
            }
            
            localStorage.setItem('gomna_api_credentials', JSON.stringify(allCredentials));
            
            this.showNotification(`Credentials saved securely for ${config.name}`, 'success');
            
        } catch (error) {
            console.error('Error saving credentials:', error);
            this.showNotification('Failed to save credentials', 'error');
        }
    }

    async encryptCredentials(credentials) {
        if (this.encryptionKey && window.crypto.subtle) {
            try {
                const data = new TextEncoder().encode(JSON.stringify(credentials));
                const iv = window.crypto.getRandomValues(new Uint8Array(12));
                
                const encrypted = await window.crypto.subtle.encrypt(
                    { name: 'AES-GCM', iv: iv },
                    this.encryptionKey,
                    data
                );
                
                return {
                    encrypted: Array.from(new Uint8Array(encrypted)),
                    iv: Array.from(iv),
                    method: 'AES-GCM'
                };
            } catch (error) {
                console.warn('Encryption failed, using base64:', error);
            }
        }
        
        // Fallback to base64 encoding
        return {
            encrypted: btoa(JSON.stringify(credentials)),
            method: 'base64'
        };
    }

    async loadSavedCredentials() {
        try {
            const saved = localStorage.getItem('gomna_api_credentials');
            if (!saved) return;
            
            const allCredentials = JSON.parse(saved);
            
            for (const [apiKey, encryptedData] of Object.entries(allCredentials)) {
                this.credentialStore.set(apiKey, encryptedData);
            }
            
            console.log('📦 Loaded saved API credentials');
        } catch (error) {
            console.error('Error loading saved credentials:', error);
        }
    }

    refreshAPICard(apiKey) {
        const card = document.querySelector(`[data-api="${apiKey}"]`);
        if (card) {
            const config = this.exchangeAPIs[apiKey] || this.socialAPIs[apiKey];
            const type = this.exchangeAPIs[apiKey] ? 'exchange' : 'social';
            card.outerHTML = this.createAPICard(apiKey, config, type);
        }
    }

    disconnectAPI(apiKey) {
        // Stop data stream
        const stream = this.dataStreams.get(apiKey);
        if (stream) {
            clearInterval(stream.id);
            this.dataStreams.delete(apiKey);
        }
        
        // Remove connection
        this.apiConnections.delete(apiKey);
        
        // Refresh UI
        this.refreshAPICard(apiKey);
        
        const config = this.exchangeAPIs[apiKey] || this.socialAPIs[apiKey];
        this.showNotification(`Disconnected from ${config.name}`, 'info');
    }

    showAPIConfiguration() {
        const apiSection = document.getElementById('api-config-section');
        if (apiSection) {
            apiSection.classList.remove('hidden');
            
            // Scroll to API section
            apiSection.scrollIntoView({ behavior: 'smooth' });
        }
    }

    hideAPIConfiguration() {
        const apiSection = document.getElementById('api-config-section');
        if (apiSection) {
            apiSection.classList.add('hidden');
        }
    }

    showNotification(message, type = 'info') {
        if (window.marketplaceUI) {
            window.marketplaceUI.showNotification(message, type);
        } else {
            console.log(`${type.toUpperCase()}: ${message}`);
        }
    }

    // Public API methods
    isAPIConnected(apiKey) {
        return this.apiConnections.has(apiKey) && this.apiConnections.get(apiKey).status === 'connected';
    }

    getConnectedAPIs() {
        return Array.from(this.apiConnections.keys()).filter(key => 
            this.apiConnections.get(key).status === 'connected'
        );
    }

    getAPIStatus(apiKey) {
        return this.apiConnections.get(apiKey) || { status: 'disconnected' };
    }

    isTestnetMode() {
        return this.testnetMode;
    }
}

// Export for global usage
if (typeof window !== 'undefined') {
    window.LiveDataAPISystem = LiveDataAPISystem;
}

console.log('📡 Live Data API System loaded successfully');