# 🔍 Live Platform Inspection Guide

## 🌐 Live Platform URL

**🔗 PRIMARY URL**: https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai

**Status**: ✅ **LIVE AND OPERATIONAL** (Updated: 2025-11-03)

---

## 📋 Quick Inspection Checklist

Use this guide to verify all production fixes are working correctly on the live platform.

---

## 🎯 Critical Features to Test

### ✅ **1. Cross-Exchange Agent (Real Spreads)**

**What to Test**: Verify real-time cross-exchange spread calculations (no 0.00% displays)

**Endpoint**: 
```
GET https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai/api/agents/cross-exchange?symbol=BTC
```

**What to Look For**:
```json
{
  "success": true,
  "agent": "cross-exchange",
  "data": {
    "market_depth_analysis": {
      "liquidity_metrics": {
        "average_spread_percent": "0.022",  // ✅ Real spread (not "0.000")
        "max_spread_percent": "0.022",      // ✅ Real spread (not "0.000")
        "spread_signal": "tight",           // ✅ Based on constraints
        "spread_type": "cross-exchange"     // ✅ Correct spread type
      },
      "arbitrage_opportunities": {
        "count": 0,                         // ✅ No fake opportunities
        "minimum_spread_threshold": 0.3     // ✅ Constraint applied
      }
    }
  }
}
```

**✅ SUCCESS CRITERIA**:
- `average_spread_percent` shows real value (NOT "0.000")
- `spread_type` is "cross-exchange" (not "bid-ask")
- No fake arbitrage opportunities
- Constraint threshold = 0.3% enforced

---

### ✅ **2. Advanced Arbitrage (Consistent with Cross-Exchange)**

**What to Test**: Verify arbitrage calculations match Cross-Exchange Agent (no simulated data)

**Endpoint**:
```
GET https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai/api/strategies/arbitrage/advanced?symbol=BTC
```

**What to Look For**:
```json
{
  "success": true,
  "strategy": "advanced_arbitrage",
  "arbitrage_opportunities": {
    "spatial": {
      "opportunities": [],                 // ✅ No fake opportunities
      "count": 0,                          // ✅ Matches Cross-Exchange Agent
      "average_spread": 0.036,             // ✅ Real spread (similar to Cross-Exchange)
      "max_spread": 0.036,                 // ✅ Real spread
      "total_pairs_analyzed": 1            // ✅ Shows calculation count
    },
    "triangular": {
      "opportunities": [],                 // ✅ No simulation
      "count": 0
    },
    "total_opportunities": 0               // ✅ Consistent total
  }
}
```

**✅ SUCCESS CRITERIA**:
- `average_spread` shows real value (NOT 0)
- No fake opportunities with simulated profits
- Spreads are consistent with Cross-Exchange Agent (within ±0.1%)
- `total_pairs_analyzed` shows calculation transparency

---

### ✅ **3. LLM Analysis (Production-Grade Error Handling)**

**What to Test**: Verify Gemini API retry logic and graceful fallback on 429 errors

**Endpoint**:
```
POST https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai/api/llm/analyze-enhanced
Content-Type: application/json

{"symbol": "BTC", "timeframe": "1h"}
```

**Success Response (Gemini API working)**:
```json
{
  "success": true,
  "analysis": "Here's an analysis of BTC/USD based on...",
  "model": "gemini-2.0-flash-exp",           // ✅ Gemini succeeded
  "data_sources": [
    "Economic Agent",
    "Sentiment Agent", 
    "Cross-Exchange Agent"
  ],
  "timestamp": "2025-11-03T17:32:14.308Z"
}
```

**Graceful Fallback Response (429 Rate Limit)**:
```json
{
  "success": true,                           // ✅ Still success!
  "analysis": "Based on the live agent data...",
  "model": "template-fallback-rate-limited", // ✅ Fallback used
  "note": "Using template analysis due to Gemini API rate limits",
  "data_sources": [
    "Economic Agent",
    "Sentiment Agent",
    "Cross-Exchange Agent"
  ]
}
```

**✅ SUCCESS CRITERIA**:
- `success: true` ALWAYS (never false)
- No raw error messages like "Gemini API error: 429"
- If `model` includes "fallback", check for user-friendly `note` field
- Analysis still uses all 3 live agent data sources
- Response time reasonable (retry logic: 2s, 4s, 8s backoff)

---

### ✅ **4. Economic Agent (Live FRED Data)**

**What to Test**: Verify Federal Reserve economic data is live

**Endpoint**:
```
GET https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai/api/agents/economic?symbol=BTC
```

**What to Look For**:
```json
{
  "success": true,
  "agent": "economic",
  "data": {
    "fed_rates": {
      "current_rate": 4.22,                  // ✅ Current Fed rate
      "rate_change": -0.25,
      "trend": "easing"
    },
    "inflation_metrics": {
      "cpi": 2.7,                            // ✅ Current CPI
      "core_cpi": 3.3
    },
    "data_freshness": "LIVE",                // ✅ LIVE data
    "timestamp": 1762191134308
  }
}
```

**✅ SUCCESS CRITERIA**:
- `data_freshness` shows "LIVE"
- Fed rate reflects current FOMC rate (check Fed website)
- CPI/inflation data is recent (within last month)
- No error messages

---

### ✅ **5. Sentiment Agent (Live Google Trends)**

**What to Test**: Verify Google Trends sentiment data via SerpAPI

**Endpoint**:
```
GET https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai/api/agents/sentiment?symbol=BTC
```

**What to Look For**:
```json
{
  "success": true,
  "agent": "sentiment",
  "data": {
    "google_trends": {
      "interest_over_time": [...],           // ✅ Array of data points
      "related_queries": [...],              // ✅ Live search queries
      "current_interest": 65,                // ✅ Current search interest
      "trend": "stable"
    },
    "data_freshness": "LIVE",                // ✅ LIVE data
    "data_source": "Google Trends via SerpAPI"
  }
}
```

**✅ SUCCESS CRITERIA**:
- `data_freshness` shows "LIVE"
- `interest_over_time` has recent data points
- `related_queries` shows current search terms
- Source clearly states "Google Trends via SerpAPI"

---

## 🧪 Testing in Browser

### **Method 1: Use Browser DevTools**

1. Open live URL: https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai
2. Press `F12` to open Developer Tools
3. Navigate to different tabs (Economic Agent, Sentiment Agent, Cross-Exchange Agent)
4. Check **Console** tab for errors
5. Check **Network** tab to see API calls and responses

### **Method 2: Direct API Testing with cURL**

```bash
# Test Cross-Exchange Agent
curl "https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai/api/agents/cross-exchange?symbol=BTC" | jq

# Test Advanced Arbitrage
curl "https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai/api/strategies/arbitrage/advanced?symbol=BTC" | jq

# Test LLM Analysis
curl -X POST "https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai/api/llm/analyze-enhanced" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC","timeframe":"1h"}' | jq

# Test Economic Agent
curl "https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai/api/agents/economic?symbol=BTC" | jq

# Test Sentiment Agent
curl "https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai/api/agents/sentiment?symbol=BTC" | jq
```

### **Method 3: Postman/Insomnia Collection**

Import these endpoints into Postman:
- Base URL: `https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai`
- Add all 5 endpoints listed above
- Test sequentially to verify consistency

---

## 📊 Expected Results Summary

| Feature | Metric | Expected Value | Status |
|---------|--------|----------------|--------|
| Cross-Exchange Agent | Average Spread | 0.020% - 0.150% | ✅ Real data |
| Advanced Arbitrage | Average Spread | 0.020% - 0.150% | ✅ Real data |
| Cross-Exchange Agent | Opportunities | 0 (market tight) | ✅ No fake data |
| Advanced Arbitrage | Opportunities | 0 (market tight) | ✅ No fake data |
| LLM Analysis | Success Rate | 100% | ✅ Never fails |
| LLM Analysis | Model | gemini-2.0-flash-exp or template-fallback-* | ✅ Graceful |
| Economic Agent | Data Freshness | LIVE | ✅ FRED API |
| Sentiment Agent | Data Freshness | LIVE | ✅ SerpAPI |
| Arbitrage Threshold | Constraint | 0.3% minimum | ✅ Applied |

---

## 🔍 Common Issues to Check

### ❌ **Issue 1: Seeing "0.000%" spreads**
**Expected**: Real spreads (0.020% - 0.150%)  
**If seeing 0.000%**: Clear browser cache, refresh page  
**Root Cause**: Old compiled code in browser cache

### ❌ **Issue 2: "Gemini API error: 429" visible to user**
**Expected**: Graceful fallback with note  
**If seeing raw error**: Check browser console for JavaScript errors  
**Root Cause**: Frontend not handling response correctly

### ❌ **Issue 3: Arbitrage showing fake opportunities**
**Expected**: 0 opportunities (or very rare real ones > 0.3%)  
**If seeing fake opportunities**: Check `spread_percent` matches spread calculation  
**Root Cause**: Simulation code not fully removed

### ❌ **Issue 4: Different spreads between features**
**Expected**: Within ±0.1% (due to timing differences)  
**If >0.2% difference**: Refresh both to sync timestamps  
**Root Cause**: API calls at different times

---

## 📚 Documentation References

- **PR #7**: https://github.com/gomna-pha/hypervision-crypto-ai/pull/7
  - 10,300+ word technical analysis
  - Before/After comparisons
  - Implementation details

- **Files**:
  - `CROSS_EXCHANGE_INCONSISTENCY_ANALYSIS.md`: Root cause analysis
  - `FIX_SUMMARY.md`: Quick reference guide
  - `README.md`: Platform overview with live URL

---

## 🚀 Deployment Information

- **Platform**: Cloudflare Workers (Serverless)
- **Database**: Cloudflare D1 (SQLite)
- **Build**: Vite (TypeScript → JavaScript)
- **Process Manager**: PM2 (ecosystem.config.cjs)
- **Source**: `src/index.tsx` → Compiled to `dist/_worker.js`

**Last Deployment**: 2025-11-03  
**Version**: Production v2.0 with PR #7 fixes  
**Status**: ✅ All systems operational

---

## 👨‍💻 For Developers

### Local Development
```bash
# Clone repository
git clone https://github.com/gomna-pha/hypervision-crypto-ai.git
cd hypervision-crypto-ai

# Install dependencies
npm install

# Build
npm run build

# Start with PM2
pm2 start ecosystem.config.cjs
```

### Verify Fixes Locally
```bash
# Check Cross-Exchange spread
curl -s "http://localhost:3000/api/agents/cross-exchange?symbol=BTC" | jq '.data.market_depth_analysis.liquidity_metrics'

# Check Advanced Arbitrage spread
curl -s "http://localhost:3000/api/strategies/arbitrage/advanced?symbol=BTC" | jq '.arbitrage_opportunities.spatial'

# Test LLM retry logic (may take up to 14 seconds with retries)
time curl -X POST "http://localhost:3000/api/llm/analyze-enhanced" \
  -H "Content-Type: application/json" \
  -d '{"symbol":"BTC","timeframe":"1h"}' | jq '.model'
```

---

## ✅ Final Verification Checklist

Before presenting to investors/stakeholders:

- [ ] Live URL loads successfully
- [ ] All 3 agent tabs display real data
- [ ] Cross-Exchange shows real spreads (not 0.000%)
- [ ] Advanced Arbitrage shows real spreads (not 0.000%)
- [ ] LLM Analysis never shows raw errors
- [ ] Economic Agent shows current Fed rate
- [ ] Sentiment Agent shows Google Trends data
- [ ] No console errors in browser DevTools
- [ ] Network tab shows successful API calls (200/201 status)
- [ ] Database migrations applied (check admin panel)

---

## 🎯 Contact & Support

**GitHub Repository**: https://github.com/gomna-pha/hypervision-crypto-ai  
**Pull Request #7**: https://github.com/gomna-pha/hypervision-crypto-ai/pull/7  
**Live Platform**: https://3000-ihto4gjgifvzp5h3din6i-d0b9e1e2.sandbox.novita.ai

For technical issues, check:
1. PR #7 description (comprehensive fix details)
2. `CROSS_EXCHANGE_INCONSISTENCY_ANALYSIS.md` (root cause analysis)
3. Browser console logs (for frontend errors)
4. PM2 logs: `pm2 logs trading-intelligence` (for backend errors)

---

**🎉 Platform is production-ready for investor demonstrations!**

All critical fixes verified and operational as of 2025-11-03.
