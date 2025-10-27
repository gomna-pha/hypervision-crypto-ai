# API Keys Setup Guide for Trading Intelligence Platform

## 🎯 Production-Ready APIs for Venture Capital Presentation

This guide will help you obtain all necessary API keys for live data feeds with timestamps.

---

## 1. 💰 CoinGecko API (Cryptocurrency Prices)

**What it provides:** Real-time crypto prices, market cap, volume, 24h changes

**Free Tier:** 10-50 calls/minute (sufficient for production)

### Steps to Get API Key:

1. **Visit:** https://www.coingecko.com/en/api
2. **Click:** "Get Your API Key"
3. **Sign up** for a free account (no credit card required)
4. **Choose Plan:** 
   - **Demo Plan** (FREE): 10 calls/min - Good for testing
   - **Analyst Plan** ($129/month): 500 calls/min - Recommended for VC demo
5. **Get API Key:** Go to Dashboard → API Keys → Copy your key

**API Key Format:** `CG-xxxxxxxxxxxxxxxxxxxxxx`

**Test Command:**
```bash
curl -X GET "https://api.coingecko.com/api/v3/simple/price?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true&include_last_updated_at=true" \
  -H "x-cg-demo-api-key: YOUR_API_KEY"
```

---

## 2. 📊 FRED API (Federal Reserve Economic Data)

**What it provides:** Fed funds rate, CPI, inflation, unemployment, GDP

**Free Tier:** Unlimited calls (100% FREE, maintained by St. Louis Fed)

### Steps to Get API Key:

1. **Visit:** https://fred.stlouisfed.org/
2. **Click:** "My Account" → "Create Account" (top right)
3. **Sign up** with email (free account)
4. **Request API Key:** 
   - Go to https://fredaccount.stlouisfed.org/apikeys
   - Click "Request API Key"
   - Fill in application name: "Trading Intelligence Platform"
   - Accept terms and submit
5. **Get API Key:** Copy your key from the dashboard

**API Key Format:** `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx` (32 characters)

**Test Command:**
```bash
curl "https://api.stlouisfed.org/fred/series/observations?series_id=FEDFUNDS&api_key=YOUR_API_KEY&file_type=json&limit=1&sort_order=desc"
```

**Key Economic Series IDs:**
- `FEDFUNDS` - Federal Funds Rate
- `CPIAUCSL` - Consumer Price Index (CPI)
- `FPCPITOTLZGUSA` - Inflation Rate
- `UNRATE` - Unemployment Rate
- `GDP` - Gross Domestic Product

---

## 3. 🔍 Google Trends (Serpapi - Easier Alternative)

**What it provides:** Search interest trends, related queries, geographic interest

**Note:** Google Trends doesn't have official API. We'll use SerpApi which scrapes Google Trends.

### Steps to Get SerpApi Key:

1. **Visit:** https://serpapi.com/
2. **Click:** "Register" (top right)
3. **Sign up** for free account
4. **Choose Plan:**
   - **Free Plan:** 100 searches/month - Good for testing
   - **Starter Plan** ($50/month): 5,000 searches/month - Recommended for VC
5. **Get API Key:** Dashboard → API Key → Copy

**API Key Format:** `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

**Test Command:**
```bash
curl "https://serpapi.com/search.json?engine=google_trends&q=bitcoin&api_key=YOUR_API_KEY"
```

**Alternative (Free but Limited):** Use `pytrends` library or Google Trends unofficial API

---

## 4. 🌍 IMF API (International Monetary Fund Data)

**What it provides:** Global economic indicators, GDP, inflation, debt ratios

**Free Tier:** 100% FREE, no registration required (public API)

### No API Key Needed!

IMF provides open JSON API without authentication.

**Test Command:**
```bash
curl "https://www.imf.org/external/datamapper/api/v1/NGDP_RPCH"
```

**Key Indicators:**
- `NGDP_RPCH` - Real GDP Growth
- `PCPIPCH` - Inflation (Average Consumer Prices)
- `GGXWDG_NGDP` - General Government Gross Debt (% of GDP)
- `BCA_NGDPD` - Current Account Balance (% of GDP)

**Documentation:** https://www.imf.org/external/datamapper/api/help

---

## 5. 🔄 Exchange APIs (Binance, Coinbase, Kraken)

**What it provides:** Real-time order books, spreads, liquidity, arbitrage opportunities

**All have FREE public endpoints** (no authentication for market data)

### 5a. Binance API (Largest Exchange)

**No API Key Required** for public market data

**Test Command:**
```bash
curl "https://api.binance.com/api/v3/ticker/24hr?symbol=BTCUSDT"
```

**Documentation:** https://binance-docs.github.io/apidocs/spot/en/

---

### 5b. Coinbase API (US-Based)

**No API Key Required** for public market data

**Test Command:**
```bash
curl "https://api.coinbase.com/v2/prices/BTC-USD/spot"
```

**Documentation:** https://docs.cloud.coinbase.com/sign-in-with-coinbase/docs/api-prices

---

### 5c. Kraken API (European Leader)

**No API Key Required** for public market data

**Test Command:**
```bash
curl "https://api.kraken.com/0/public/Ticker?pair=XBTUSD"
```

**Documentation:** https://docs.kraken.com/rest/

---

## 6. 🤖 Gemini AI API (Already Configured)

**What it provides:** AI-powered analysis, sentiment synthesis, market intelligence

**Your Current Key:** `AIzaSyCG4nVE1101YRsNh0OSq94VoHQe-CDv4og`

**Status:** ✅ Already integrated in your platform

---

## 7. 📰 News API (Optional - For Enhanced Sentiment)

**What it provides:** Financial news headlines, sentiment analysis sources

**Free Tier:** 100 requests/day

### Steps to Get API Key:

1. **Visit:** https://newsapi.org/
2. **Click:** "Get API Key"
3. **Sign up** for Developer plan (FREE)
4. **Get API Key:** Copy from dashboard

**API Key Format:** `xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx`

**Test Command:**
```bash
curl "https://newsapi.org/v2/everything?q=bitcoin&sortBy=publishedAt&apiKey=YOUR_API_KEY"
```

---

## 🔐 Configuration Summary

Once you have the keys, provide them to me in this format:

```
COINGECKO_API_KEY=CG-xxxxxxxxxxxxxxxxxxxxxx
FRED_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
SERPAPI_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
GEMINI_API_KEY=AIzaSyCG4nVE1101YRsNh0OSq94VoHQe-CDv4og (already have)
NEWSAPI_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx (optional)
```

**IMF and Exchange APIs** don't require keys (public endpoints).

---

## 💰 Cost Breakdown for VC-Ready Platform

| Service | Free Tier | Recommended Plan | Monthly Cost |
|---------|-----------|------------------|--------------|
| CoinGecko | 10 calls/min | Analyst (500 calls/min) | $129 |
| FRED | Unlimited | Free | $0 |
| SerpApi (Google Trends) | 100/month | Starter (5,000/month) | $50 |
| IMF | Unlimited | Free | $0 |
| Binance/Coinbase/Kraken | Unlimited | Free (public data) | $0 |
| Gemini AI | Pay per use | ~$5-20/month | ~$10 |
| News API | 100/day | Developer (1,000/day) | $0-$449 |
| **TOTAL** | **~$15/month** | **~$189/month** | **Professional** |

---

## 🚀 Quick Start Options

### Option 1: Minimal Cost Setup (Recommended for Start)
- CoinGecko FREE (10 calls/min)
- FRED FREE
- Skip SerpApi, use basic sentiment
- IMF FREE
- Exchange APIs FREE
- **Total: ~$5-10/month** (Gemini AI only)

### Option 2: Full Production (Recommended for VC)
- CoinGecko Analyst ($129/month)
- FRED FREE
- SerpApi Starter ($50/month)
- IMF FREE
- Exchange APIs FREE
- **Total: ~$189/month**

### Option 3: Start Free, Scale Later
- Use ALL free tiers initially
- Upgrade after VC funding
- **Total: ~$5-10/month**

---

## 📋 Next Steps

1. **Get the API keys** using the links above
2. **Provide them to me** in the format shown
3. **I'll implement** all live data feeds with:
   - Real-time timestamps
   - Constraint filters on each agent
   - Error handling and fallbacks
   - Rate limiting and caching
   - Production-ready monitoring

4. **We'll deploy** to Cloudflare Pages with production domain

---

## 🆘 Need Help?

If you encounter any issues getting API keys:
- Let me know which service is causing problems
- I can provide alternative data sources
- We can start with free tiers and upgrade later

**Ready to proceed?** Get your API keys and share them with me!
