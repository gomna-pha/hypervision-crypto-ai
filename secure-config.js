/**
 * Secure Configuration Manager
 * Handles encrypted API keys and sensitive data
 */

class SecureConfig {
    constructor() {
        this.encrypted = true;
        this.keys = new Map();
        this.iv = crypto.getRandomValues(new Uint8Array(16));
    }

    /**
     * Encrypt sensitive data
     */
    async encrypt(text, password) {
        const encoder = new TextEncoder();
        const data = encoder.encode(text);
        
        const key = await this.deriveKey(password);
        const encrypted = await crypto.subtle.encrypt(
            { name: 'AES-GCM', iv: this.iv },
            key,
            data
        );
        
        return {
            encrypted: btoa(String.fromCharCode(...new Uint8Array(encrypted))),
            iv: btoa(String.fromCharCode(...this.iv))
        };
    }

    /**
     * Decrypt sensitive data
     */
    async decrypt(encryptedData, password) {
        const key = await this.deriveKey(password);
        const data = Uint8Array.from(atob(encryptedData), c => c.charCodeAt(0));
        
        const decrypted = await crypto.subtle.decrypt(
            { name: 'AES-GCM', iv: this.iv },
            key,
            data
        );
        
        const decoder = new TextDecoder();
        return decoder.decode(decrypted);
    }

    /**
     * Derive encryption key from password
     */
    async deriveKey(password) {
        const encoder = new TextEncoder();
        const salt = encoder.encode('gomna-ai-trading-platform');
        
        const keyMaterial = await crypto.subtle.importKey(
            'raw',
            encoder.encode(password),
            { name: 'PBKDF2' },
            false,
            ['deriveKey']
        );
        
        return crypto.subtle.deriveKey(
            {
                name: 'PBKDF2',
                salt: salt,
                iterations: 100000,
                hash: 'SHA-256'
            },
            keyMaterial,
            { name: 'AES-GCM', length: 256 },
            false,
            ['encrypt', 'decrypt']
        );
    }

    /**
     * Store API credentials securely
     */
    async storeCredentials(exchange, apiKey, apiSecret, masterPassword) {
        const encryptedKey = await this.encrypt(apiKey, masterPassword);
        const encryptedSecret = await this.encrypt(apiSecret, masterPassword);
        
        // Store in localStorage with encryption
        const credentials = {
            exchange,
            apiKey: encryptedKey.encrypted,
            apiSecret: encryptedSecret.encrypted,
            iv: encryptedKey.iv,
            timestamp: Date.now(),
            expires: Date.now() + (24 * 60 * 60 * 1000) // 24 hours
        };
        
        localStorage.setItem(`gomna_${exchange}_creds`, JSON.stringify(credentials));
        
        // Also store in memory for session
        this.keys.set(exchange, {
            apiKey,
            apiSecret,
            connected: true
        });
        
        return true;
    }

    /**
     * Retrieve API credentials
     */
    async getCredentials(exchange, masterPassword) {
        // Try memory first
        if (this.keys.has(exchange)) {
            return this.keys.get(exchange);
        }
        
        // Try localStorage
        const stored = localStorage.getItem(`gomna_${exchange}_creds`);
        if (!stored) return null;
        
        const credentials = JSON.parse(stored);
        
        // Check expiry
        if (credentials.expires < Date.now()) {
            localStorage.removeItem(`gomna_${exchange}_creds`);
            return null;
        }
        
        // Decrypt
        this.iv = Uint8Array.from(atob(credentials.iv), c => c.charCodeAt(0));
        const apiKey = await this.decrypt(credentials.apiKey, masterPassword);
        const apiSecret = await this.decrypt(credentials.apiSecret, masterPassword);
        
        // Store in memory
        this.keys.set(exchange, {
            apiKey,
            apiSecret,
            connected: true
        });
        
        return { apiKey, apiSecret };
    }

    /**
     * Clear all stored credentials
     */
    clearCredentials() {
        this.keys.clear();
        const exchanges = ['binance', 'coinbase', 'kraken'];
        exchanges.forEach(exchange => {
            localStorage.removeItem(`gomna_${exchange}_creds`);
        });
    }

    /**
     * Validate API credentials format
     */
    validateCredentials(apiKey, apiSecret) {
        // Basic validation
        if (!apiKey || !apiSecret) return false;
        if (apiKey.length < 20 || apiSecret.length < 20) return false;
        
        // Check for common patterns
        const apiKeyPattern = /^[A-Za-z0-9\-_]{20,}$/;
        const apiSecretPattern = /^[A-Za-z0-9\-_\/+=]{30,}$/;
        
        return apiKeyPattern.test(apiKey) && apiSecretPattern.test(apiSecret);
    }

    /**
     * Generate secure random password
     */
    generateSecurePassword(length = 32) {
        const charset = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789!@#$%^&*()_+-=[]{}|;:,.<>?';
        const values = crypto.getRandomValues(new Uint8Array(length));
        return Array.from(values, byte => charset[byte % charset.length]).join('');
    }

    /**
     * Hash sensitive data for comparison
     */
    async hashData(data) {
        const encoder = new TextEncoder();
        const dataBuffer = encoder.encode(data);
        const hashBuffer = await crypto.subtle.digest('SHA-256', dataBuffer);
        const hashArray = Array.from(new Uint8Array(hashBuffer));
        return hashArray.map(b => b.toString(16).padStart(2, '0')).join('');
    }

    /**
     * Check if running in secure context
     */
    isSecureContext() {
        return window.isSecureContext && location.protocol === 'https:';
    }

    /**
     * Get API configuration for specific exchange
     */
    getExchangeConfig(exchange) {
        const configs = {
            binance: {
                baseURL: 'https://api.binance.com',
                wsURL: 'wss://stream.binance.com:9443',
                testnet: 'https://testnet.binance.vision',
                requiredPermissions: ['SPOT', 'MARGIN'],
                rateLimit: 1200,
                orderTypes: ['LIMIT', 'MARKET', 'STOP_LOSS', 'STOP_LOSS_LIMIT', 'TAKE_PROFIT', 'TAKE_PROFIT_LIMIT']
            },
            coinbase: {
                baseURL: 'https://api.exchange.coinbase.com',
                wsURL: 'wss://ws-feed.exchange.coinbase.com',
                sandbox: 'https://api-public.sandbox.exchange.coinbase.com',
                requiredPermissions: ['trade', 'view'],
                rateLimit: 10,
                orderTypes: ['limit', 'market', 'stop', 'stop_limit']
            },
            kraken: {
                baseURL: 'https://api.kraken.com',
                wsURL: 'wss://ws.kraken.com',
                requiredPermissions: ['Query Funds', 'Query Orders', 'Cancel/Close Orders', 'Create Orders'],
                rateLimit: 15,
                orderTypes: ['limit', 'market', 'stop-loss', 'take-profit', 'stop-loss-limit', 'take-profit-limit']
            }
        };
        
        return configs[exchange] || null;
    }
}

// Export for use in application
if (typeof module !== 'undefined' && module.exports) {
    module.exports = SecureConfig;
}

// Initialize global instance
window.secureConfig = new SecureConfig();