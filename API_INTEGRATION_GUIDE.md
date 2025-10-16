# GOMNA Trading API Integration Guide

## Overview
GOMNA platform now supports real-time trading through multiple exchange APIs including Binance, Coinbase Pro, Kraken, and various open-source alternatives.

## Supported Exchanges

### 1. **Binance** 🟠
- **Type**: Centralized Exchange
- **API Documentation**: [Binance API Docs](https://binance-docs.github.io/apidocs/spot/en/)
- **Features**: Spot, Futures, Margin, Staking
- **Rate Limit**: 1200 weight per minute
- **Testnet Available**: ✅ Yes (testnet.binance.vision)

#### Setup Instructions:
1. Create account at [Binance](https://www.binance.com)
2. Enable 2FA for security
3. Go to API Management
4. Create new API key
5. Save API Key and Secret securely
6. For testing, use [Binance Testnet](https://testnet.binance.vision)

### 2. **Coinbase Pro** 🔵
- **Type**: Centralized Exchange
- **API Documentation**: [Coinbase Docs](https://docs.cloud.coinbase.com/exchange/docs)
- **Features**: Spot Trading, Staking
- **Rate Limit**: 10 requests per second (public), 15 (private)
- **Sandbox Available**: ✅ Yes

#### Setup Instructions:
1. Create account at [Coinbase Pro](https://pro.coinbase.com)
2. Navigate to API settings
3. Create new API key with permissions:
   - View (read access)
   - Trade (place orders)
   - Transfer (optional)
4. Save API Key, Secret, and Passphrase
5. Use [Sandbox](https://public.sandbox.pro.coinbase.com) for testing

### 3. **Kraken** 🟣
- **Type**: Centralized Exchange
- **API Documentation**: [Kraken API](https://docs.kraken.com/rest/)
- **Features**: Spot, Futures, Margin, Staking
- **Rate Limit**: 15-20 calls per second (tier-based)
- **Demo Account**: ✅ Available

#### Setup Instructions:
1. Create account at [Kraken](https://www.kraken.com)
2. Complete verification
3. Go to Settings → API
4. Generate new key
5. Set permissions (Query, Trade)
6. Save API Key and Private Key

## Open Source & Free APIs

### 4. **CoinGecko** 🦎 (FREE)
- **Type**: Market Data Provider
- **API Documentation**: [CoinGecko API](https://www.coingecko.com/en/api)
- **Features**: Price data, Market cap, Volume, Historical data
- **Rate Limit**: 50 calls/minute (free), 500 (pro)
- **API Key**: Optional (higher limits with key)

#### No Setup Required!
- Works immediately without authentication
- Already integrated in GOMNA platform
- Auto-connects on platform load

### 5. **CryptoCompare** 📊
- **Type**: Data Aggregator
- **API Documentation**: [CryptoCompare API](https://min-api.cryptocompare.com/documentation)
- **Features**: Market data, Historical, News, Social sentiment
- **Rate Limit**: 100,000 calls/month (free)
- **API Key**: Optional but recommended

### 6. **Alpaca Markets** 🦙
- **Type**: Commission-free Trading
- **API Documentation**: [Alpaca API](https://alpaca.markets/docs/)
- **Features**: Stocks, Crypto, Paper Trading
- **Rate Limit**: 200 requests/minute
- **Paper Trading**: ✅ Free unlimited paper trading

#### Setup Instructions:
1. Sign up at [Alpaca](https://alpaca.markets)
2. Get free paper trading account
3. Generate API keys from dashboard
4. Use paper trading for testing

## How to Connect in GOMNA

### Quick Start (Demo Mode)
1. Platform loads with **CoinGecko free API** automatically
2. Live prices update every 30 seconds
3. No configuration needed!

### Connect Exchange API
1. Click **"Configure APIs"** button in Live Market Data panel
2. Select your exchange from dropdown
3. Enter credentials:
   - API Key
   - API Secret
   - Passphrase (Coinbase only)
4. Check "Use Testnet/Sandbox" for testing
5. Click "Connect to Exchange"

### Place Orders
1. Use the **Order Execution** panel
2. Select exchange (or use Demo mode)
3. Enter trading pair (e.g., BTC/USD)
4. Choose order type:
   - **Market**: Execute immediately at current price
   - **Limit**: Set specific price
   - **Stop**: Trigger at stop price
5. Enter amount
6. Click BUY (green) or SELL (red)

## Security Best Practices

### API Key Security
- **Never share** API keys
- **Use IP whitelisting** when available
- **Enable 2FA** on exchange accounts
- **Limit permissions** to only what's needed
- **Use testnet/sandbox** for development
- **Rotate keys** regularly

### Safe Trading
- Start with **paper trading** or **testnet**
- Test with **small amounts** first
- Set **stop-loss** orders
- Monitor **rate limits**
- Keep **audit logs**

## WebSocket Connections

Real-time data streams are available via WebSocket:

```javascript
// Binance WebSocket
wss://stream.binance.com:9443/ws/btcusdt@ticker

// Coinbase WebSocket
wss://ws-feed.pro.coinbase.com

// Kraken WebSocket
wss://ws.kraken.com
```

## Rate Limits

| Exchange | Limit | Interval | Notes |
|----------|-------|----------|-------|
| Binance | 1200 weight | 1 minute | Weight varies by endpoint |
| Coinbase | 10-15 | 1 second | Higher for private endpoints |
| Kraken | 15-20 | 1 second | Tier-based system |
| CoinGecko | 50 | 1 minute | Free tier |
| Alpaca | 200 | 1 minute | All endpoints |

## Error Handling

Common API errors and solutions:

| Error Code | Meaning | Solution |
|------------|---------|----------|
| 401 | Unauthorized | Check API credentials |
| 403 | Forbidden | Verify permissions |
| 429 | Rate Limited | Reduce request frequency |
| 500 | Server Error | Retry with backoff |
| 503 | Maintenance | Check exchange status |

## Testing Endpoints

### Get Current Price (Examples)

**Binance:**
```
GET https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT
```

**CoinGecko (Free):**
```
GET https://api.coingecko.com/api/v3/simple/price?ids=bitcoin&vs_currencies=usd
```

**Kraken:**
```
GET https://api.kraken.com/0/public/Ticker?pair=XBTUSD
```

## Environment Variables

For production, use environment variables:

```javascript
BINANCE_API_KEY=your_key_here
BINANCE_API_SECRET=your_secret_here
COINBASE_API_KEY=your_key_here
COINBASE_API_SECRET=your_secret_here
COINBASE_PASSPHRASE=your_passphrase_here
```

## Support & Resources

- **Binance Support**: [support.binance.com](https://www.binance.com/en/support)
- **Coinbase Support**: [help.coinbase.com](https://help.coinbase.com)
- **Kraken Support**: [support.kraken.com](https://support.kraken.com)
- **CoinGecko API**: [coingecko.com/api](https://www.coingecko.com/en/api)
- **Discord Community**: Join exchange Discord servers
- **GitHub Issues**: Report bugs on respective GitHub repos

## Compliance Notice

⚠️ **Important**: 
- Trading cryptocurrencies involves risk
- Ensure compliance with local regulations
- Some features may not be available in all jurisdictions
- Use testnet/sandbox for development and testing
- Never invest more than you can afford to lose

## Next Steps

1. ✅ Start with **Demo Mode** (already active)
2. 📝 Get **free API keys** from CoinGecko or CryptoCompare
3. 🧪 Test with **Binance Testnet** or **Coinbase Sandbox**
4. 💰 Graduate to **live trading** when ready

---

*GOMNA - Professional Trading with Hyperbolic CNN Intelligence*