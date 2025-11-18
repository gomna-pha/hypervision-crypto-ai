# 🔍 API Replacement Plan - Replace Simulated Data with Free APIs

## Current Simulated Data Sources

### 1. **Economic Agent** (generateEconomicData)
**Current:** Simulated Fed Rate, CPI, GDP, PMI
**Free API Options:**
- ✅ **FRED API** (Federal Reserve Economic Data) - FREE
  - Fed Funds Rate: `FEDFUNDS`
  - CPI: `CPIAUCSL`
  - GDP: `GDP`
  - Unemployment: `UNRATE`
  - API: https://api.stlouisfed.org/fred/series/observations
  - Key: FREE with registration

### 2. **Sentiment Agent** (generateSentimentData)
**Current:** Simulated Fear & Greed, Google Trends, VIX
**Free API Options:**
- ✅ **Alternative.me Fear & Greed Index** - FREE, NO KEY
  - API: https://api.alternative.me/fng/
  - Real-time crypto sentiment (0-100)
- ⚠️ **Google Trends** - No free API (requires SerpAPI $$$)
- ✅ **VIX Alternative** - Use crypto volatility index from exchanges

### 3. **Cross-Exchange Agent** (generateCrossExchangeData)
**Current:** Simulated BTC prices and spreads
**Free API Options:**
- ✅ **CoinGecko API** - FREE, NO KEY (50 calls/min)
  - Get BTC prices from multiple exchanges
  - API: https://api.coingecko.com/api/v3/simple/price
- ✅ **Binance API** - FREE, NO KEY
  - Real-time ticker: https://api.binance.com/api/v3/ticker/price
- ✅ **Coinbase API** - FREE, NO KEY
  - Ticker: https://api.coinbase.com/v2/prices/BTC-USD/spot

### 4. **On-Chain Agent** (generateOnChainData)
**Current:** Simulated exchange flows, SOPR, MVRV
**Free API Options:**
- ✅ **Blockchain.info API** - FREE, NO KEY
  - Market stats: https://blockchain.info/stats
- ✅ **CryptoCompare API** - FREE (100k calls/month)
  - On-chain data: https://min-api.cryptocompare.com/
  - Key: FREE with registration
- ⚠️ **Glassnode** - Has free tier but limited

### 5. **CNN Pattern Agent** (generateCNNPatternData)
**Current:** Random pattern selection
**Keep Simulated:** Technical patterns require complex calculations
**Alternative:** Use real price data to detect patterns algorithmically

### 6. **Opportunities** (generateOpportunities)
**Current:** Hardcoded 35 opportunities
**Enhancement:** Calculate real arbitrage opportunities from exchange APIs

## Implementation Priority

### Phase 1: Core Price Data (CRITICAL)
1. ✅ **CoinGecko API** - Multi-exchange BTC/ETH prices
2. ✅ **Alternative.me** - Real Fear & Greed Index
3. ✅ **Binance/Coinbase** - Real-time ticker data

### Phase 2: Economic Data (IMPORTANT)
4. ✅ **FRED API** - Real Fed Rate, CPI (requires API key)

### Phase 3: On-Chain Data (NICE TO HAVE)
5. ✅ **Blockchain.info** - Basic on-chain metrics
6. ⚠️ **CryptoCompare** - Enhanced on-chain data (requires API key)

### Phase 4: Keep Simulated (ACCEPTABLE)
- CNN Pattern Recognition (complex calculations)
- Google Trends (no free API)
- Backtest data (historical simulation)

## Free APIs Summary

| Data Source | API | Cost | Key Required | Rate Limit |
|-------------|-----|------|--------------|------------|
| BTC Prices | CoinGecko | FREE | NO | 50/min |
| Fear & Greed | Alternative.me | FREE | NO | Unlimited |
| Exchange Prices | Binance | FREE | NO | 1200/min |
| Exchange Prices | Coinbase | FREE | NO | 10,000/day |
| Economic Data | FRED | FREE | YES | 120/min |
| On-Chain | Blockchain.info | FREE | NO | Unknown |
| On-Chain | CryptoCompare | FREE | YES | 100k/month |

## Recommendation

**Immediate Implementation (No Keys Required):**
1. ✅ CoinGecko API - Multi-exchange prices
2. ✅ Alternative.me - Fear & Greed Index
3. ✅ Binance API - Real-time BTC/ETH prices
4. ✅ Coinbase API - Cross-exchange arbitrage data

**Optional (Requires Free Keys):**
5. FRED API - Real economic data (user can add key)
6. CryptoCompare - Enhanced on-chain data (user can add key)

**Keep Simulated:**
- Google Trends (no free API)
- CNN Patterns (complex calculations)
- Backtest data (historical simulation)
- SOPR, MVRV (requires premium APIs)

## Implementation Plan

1. Create API service layer with error handling
2. Replace Economic Agent with FRED API (optional key)
3. Replace Sentiment Agent with Alternative.me API
4. Replace Cross-Exchange with CoinGecko + Binance + Coinbase
5. Update On-Chain with Blockchain.info API
6. Keep CNN and backtest simulated
7. Add fallback to simulated data if APIs fail
8. Add caching to respect rate limits
