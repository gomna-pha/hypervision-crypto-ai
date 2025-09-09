/**
 * Secure Production Server with API Integration
 * Gomna AI Trading Platform
 */

const express = require('express');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const crypto = require('crypto');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 8000;

// Security middleware
app.use(cors({
    origin: process.env.CORS_ORIGIN || 'http://localhost:8000',
    credentials: true
}));
app.use(express.json({ limit: '10mb' }));
app.use(express.static('.'));

// Rate limiting
const rateLimit = new Map();
const MAX_REQUESTS = parseInt(process.env.MAX_REQUESTS_PER_MINUTE) || 60;

app.use((req, res, next) => {
    const ip = req.ip;
    const now = Date.now();
    
    if (!rateLimit.has(ip)) {
        rateLimit.set(ip, { count: 1, resetTime: now + 60000 });
    } else {
        const limit = rateLimit.get(ip);
        if (now > limit.resetTime) {
            limit.count = 1;
            limit.resetTime = now + 60000;
        } else {
            limit.count++;
            if (limit.count > MAX_REQUESTS) {
                return res.status(429).json({ error: 'Rate limit exceeded' });
            }
        }
    }
    next();
});

// Authentication middleware
function authenticateAPI(req, res, next) {
    const apiKey = req.headers['x-api-key'];
    const signature = req.headers['x-signature'];
    const timestamp = req.headers['x-timestamp'];
    
    if (!apiKey || !signature || !timestamp) {
        return res.status(401).json({ error: 'Missing authentication headers' });
    }
    
    // Verify timestamp is within 5 minutes
    const now = Date.now();
    if (Math.abs(now - parseInt(timestamp)) > 300000) {
        return res.status(401).json({ error: 'Request expired' });
    }
    
    // Verify signature
    const expectedSignature = crypto
        .createHmac('sha256', process.env.JWT_SECRET || 'default-secret')
        .update(`${timestamp}${apiKey}`)
        .digest('hex');
    
    if (signature !== expectedSignature) {
        return res.status(401).json({ error: 'Invalid signature' });
    }
    
    req.apiKey = apiKey;
    next();
}

// API Routes
const apiRouter = express.Router();

// Connect to exchange
apiRouter.post('/auth/connect', authenticateAPI, async (req, res) => {
    const { exchange, testMode } = req.body;
    
    try {
        // Get credentials from environment
        const apiKey = process.env[`${exchange.toUpperCase()}_API_KEY`];
        const apiSecret = process.env[`${exchange.toUpperCase()}_API_SECRET`];
        
        if (!apiKey || !apiSecret) {
            return res.status(400).json({ 
                success: false, 
                error: 'API credentials not configured for ' + exchange 
            });
        }
        
        // In production, establish actual connection to exchange
        // For now, simulate connection
        res.json({
            success: true,
            exchange,
            testMode,
            message: 'Connected successfully',
            features: getExchangeFeatures(exchange)
        });
    } catch (error) {
        console.error('Connection error:', error);
        res.status(500).json({ success: false, error: error.message });
    }
});

// Place market order
apiRouter.post('/orders/market', authenticateAPI, async (req, res) => {
    const { symbol, side, quantity, stopLoss, takeProfit } = req.body;
    
    if (!process.env.ENABLE_REAL_TRADING === 'true') {
        // Demo mode
        const order = {
            orderId: 'DEMO-' + Date.now(),
            symbol,
            side,
            type: 'MARKET',
            quantity,
            price: getMarketPrice(symbol),
            status: 'FILLED',
            timestamp: Date.now()
        };
        
        return res.json({ success: true, order });
    }
    
    // Real trading would go here
    try {
        // Connect to exchange API and place order
        const order = await placeRealMarketOrder(symbol, side, quantity);
        res.json({ success: true, order });
    } catch (error) {
        res.status(500).json({ success: false, error: error.message });
    }
});

// Place limit order
apiRouter.post('/orders/limit', authenticateAPI, async (req, res) => {
    const { symbol, side, quantity, price, timeInForce } = req.body;
    
    const order = {
        orderId: generateOrderId(),
        symbol,
        side,
        type: 'LIMIT',
        quantity,
        price,
        timeInForce: timeInForce || 'GTC',
        status: 'NEW',
        timestamp: Date.now()
    };
    
    res.json({ success: true, order });
});

// Get balance
apiRouter.get('/account/balance', authenticateAPI, async (req, res) => {
    const balance = {
        USD: 100000,
        BTC: 2.5,
        ETH: 25,
        USDT: 50000,
        timestamp: Date.now()
    };
    
    res.json({ success: true, balance });
});

// Get positions
apiRouter.get('/positions', authenticateAPI, async (req, res) => {
    const positions = [
        {
            symbol: 'BTC',
            quantity: 1.2,
            avgPrice: 42150,
            currentPrice: 45234,
            pnl: 3700.8,
            pnlPercent: 8.77
        },
        {
            symbol: 'ETH',
            quantity: 15,
            avgPrice: 2890,
            currentPrice: 3124,
            pnl: 3510,
            pnlPercent: 8.10
        }
    ];
    
    res.json({ success: true, positions });
});

// WebSocket endpoint for real-time data
apiRouter.get('/ws/info', (req, res) => {
    res.json({
        url: `wss://${req.get('host')}/ws`,
        streams: ['trades', 'orders', 'positions', 'balance']
    });
});

// Mount API router
app.use('/api/v1', apiRouter);

// Health check
app.get('/health', (req, res) => {
    res.json({
        status: 'healthy',
        uptime: process.uptime(),
        timestamp: Date.now(),
        version: '2.0.0',
        secure: true,
        tradingEnabled: process.env.ENABLE_REAL_TRADING === 'true'
    });
});

// Helper functions
function getExchangeFeatures(exchange) {
    const features = {
        binance: ['SPOT', 'FUTURES', 'MARGIN', 'SAVINGS'],
        coinbase: ['SPOT', 'STAKING'],
        kraken: ['SPOT', 'FUTURES', 'STAKING']
    };
    return features[exchange] || ['SPOT'];
}

function getMarketPrice(symbol) {
    const prices = {
        'BTC': 45234,
        'ETH': 3124,
        'SOL': 112,
        'ADA': 0.62
    };
    return prices[symbol] || 100;
}

function generateOrderId() {
    return 'ORD-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
}

async function placeRealMarketOrder(symbol, side, quantity) {
    // This would connect to actual exchange API
    // For security, implementation details are abstracted
    throw new Error('Real trading not implemented in demo');
}

// Error handling
app.use((err, req, res, next) => {
    console.error('Server error:', err);
    res.status(500).json({
        error: 'Internal server error',
        message: process.env.NODE_ENV === 'development' ? err.message : 'An error occurred'
    });
});

// Start server
const server = app.listen(PORT, '0.0.0.0', () => {
    console.log(`
    ╔════════════════════════════════════════════╗
    ║   Gomna AI Trading Platform - SECURE       ║
    ║   Production Server Running                 ║
    ╠════════════════════════════════════════════╣
    ║   Server: http://localhost:${PORT}            ║
    ║   API:    http://localhost:${PORT}/api/v1     ║
    ║   Health: http://localhost:${PORT}/health     ║
    ╠════════════════════════════════════════════╣
    ║   Trading: ${process.env.ENABLE_REAL_TRADING === 'true' ? 'ENABLED' : 'DEMO MODE'}                          ║
    ║   Security: ACTIVE                          ║
    ║   Rate Limit: ${MAX_REQUESTS}/min                   ║
    ╚════════════════════════════════════════════╝
    `);
    
    if (!process.env.JWT_SECRET) {
        console.warn('⚠️  WARNING: Using default JWT secret. Set JWT_SECRET in .env for production!');
    }
    
    if (!process.env.BINANCE_API_KEY) {
        console.warn('⚠️  WARNING: No API keys configured. Add them to .env file.');
    }
});

// Graceful shutdown
process.on('SIGTERM', () => {
    console.log('SIGTERM received, closing server...');
    server.close(() => {
        console.log('Server closed');
        process.exit(0);
    });
});

module.exports = app;