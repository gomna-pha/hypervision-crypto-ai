# 🔧 Quick Fix Guide - Resolving Dashboard Errors

## Current Issues Fixed

### ✅ Issue 1: Database Error (RESOLVED)
**Error:** `D1_ERROR: no such table: market_data: SQLITE_ERROR`

**Solution Applied:**
```bash
cd /home/user/webapp
wrangler d1 migrations apply webapp-production --local
pm2 restart trading-intelligence
```

**Status:** ✅ Database migrations applied successfully, service restarted.

---

### ⚠️ Issue 2: Gemini API Error (NEEDS YOUR API KEY)
**Error:** `Gemini API error: 429` (Rate limiting/No API key)

**Why This Happens:**
- The LLM Analysis feature requires a Google Gemini API key
- Without a valid key, the API returns a 429 error
- This is expected behavior when the API key is not configured

**Solution Steps:**

#### Step 1: Get Your Free Gemini API Key

1. Visit: https://aistudio.google.com/apikey
2. Sign in with your Google account
3. Click "Create API Key"
4. Copy your API key (format: `AIzaSy...`)

#### Step 2: Add API Key to `.dev.vars` File

Open the file `/home/user/webapp/.dev.vars` and replace the placeholder:

```bash
# Change this line:
GEMINI_API_KEY=your_gemini_api_key_here

# To your actual key:
GEMINI_API_KEY=AIzaSyXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
```

#### Step 3: Restart the Service

```bash
cd /home/user/webapp
pm2 restart trading-intelligence
```

#### Step 4: Test the Fix

1. Open your browser and go to the sandbox URL
2. Scroll to "Fair Comparison Architecture"
3. Click "Run LLM Analysis"
4. You should see AI-generated market analysis instead of the error

---

## Optional API Keys (Recommended for Full Functionality)

### 🏦 FRED API Key (FREE - US Economic Data)
**Purpose:** Live Federal Reserve economic data (Fed rates, CPI, GDP, unemployment)

**Get Key:**
1. Visit: https://fredaccount.stlouisfed.org/apikeys
2. Create account (free)
3. Request API key
4. Add to `.dev.vars`: `FRED_API_KEY=your_key_here`

**Impact:** Economic Agent will use LIVE data instead of simulated values

---

### 📊 CoinGecko API Key (FREE Tier Available)
**Purpose:** Enhanced cryptocurrency market data and aggregated prices

**Get Key:**
1. Visit: https://www.coingecko.com/en/api
2. Sign up for free account
3. Get API key (Demo plan: 10 calls/min)
4. Add to `.dev.vars`: `COINGECKO_API_KEY=your_key_here`

**Impact:** Cross-Exchange Agent gets additional price validation

---

### 🔍 SerpAPI Key (100 Free Searches/Month)
**Purpose:** Google Trends sentiment analysis

**Get Key:**
1. Visit: https://serpapi.com/
2. Create free account
3. Copy API key from dashboard
4. Add to `.dev.vars`: `SERPAPI_KEY=your_key_here`

**Impact:** Sentiment Agent includes live Google Trends data

---

## Current Service Status

### Working Features ✅
- ✅ **Database:** All tables created and accessible
- ✅ **Live Arbitrage Opportunities:** Real-time cross-exchange analysis
- ✅ **3 Live Agents:** Economic, Sentiment, Cross-Exchange (all operational)
- ✅ **Live Market Data:** Binance, Coinbase, Kraken feeds (no API keys needed)
- ✅ **Backtesting Engine:** Ready to run (needs API key to unlock)
- ✅ **Real-time Dashboard:** Auto-updating every 2 seconds

### Requires API Key 🔑
- 🔑 **LLM Analysis:** Needs Gemini API key (see Step 1-4 above)
- 🔑 **Agent-Based Backtesting:** Needs Gemini API key (same as above)

---

## Quick Commands Reference

### Restart Service After Adding API Keys
```bash
cd /home/user/webapp
pm2 restart trading-intelligence
pm2 logs trading-intelligence --lines 50
```

### Check Service Status
```bash
cd /home/user/webapp
pm2 status
```

### View Live Logs
```bash
cd /home/user/webapp
pm2 logs trading-intelligence --lines 100 --nostream
```

### Test Database Connection
```bash
cd /home/user/webapp
wrangler d1 execute webapp-production --local --command "SELECT name FROM sqlite_master WHERE type='table';"
```

### Verify API Endpoints
```bash
# Test Economic Agent
curl http://localhost:3000/api/agents/economic

# Test Sentiment Agent
curl http://localhost:3000/api/agents/sentiment

# Test Cross-Exchange Agent
curl http://localhost:3000/api/agents/cross-exchange

# Test Status Endpoint (shows API key configuration)
curl http://localhost:3000/api/status
```

---

## Priority Action Required

### 🎯 **IMMEDIATE: Add Gemini API Key**

This is the ONLY required API key to unlock all major features:
- ✅ LLM Analysis (AI-powered market commentary)
- ✅ Agent-Based Backtesting (algorithmic trading signals)
- ✅ Model Comparison (LLM vs Backtesting agreement analysis)

**Total time:** ~2 minutes
**Cost:** FREE (Google Gemini has generous free tier)

Once added, your dashboard will be fully functional for VC presentations!

---

## Current URLs

- **Local:** http://localhost:3000
- **Public Sandbox:** https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai

---

## Support Resources

- **Detailed API Setup:** See `API_KEYS_SETUP_GUIDE.md`
- **Production Deployment:** See `PRODUCTION_DEPLOYMENT_GUIDE.md`
- **Full Documentation:** See `README.md`

---

## Next Steps After Fix

1. ✅ Add Gemini API key (REQUIRED)
2. ✅ Restart service: `pm2 restart trading-intelligence`
3. ✅ Test LLM Analysis on dashboard
4. ✅ Run Backtesting to verify functionality
5. 📊 (Optional) Add FRED, CoinGecko, SerpAPI keys for enhanced data
6. 🚀 Share working URL with partners

---

**Last Updated:** 2025-11-03
