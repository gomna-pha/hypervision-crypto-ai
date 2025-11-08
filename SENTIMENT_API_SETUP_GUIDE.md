# 🔑 Sentiment API Setup Guide

This guide shows you how to get API keys for live sentiment data.

---

## ✅ **Already Implemented (Working Now)**

### **1. Fear & Greed Index** ✅ FREE

**Status**: 🟢 **LIVE AND WORKING**

**API**: Alternative.me Fear & Greed Index  
**Cost**: **FREE** (no signup, no key needed)  
**Current Value**: 21 (Extreme Fear)  
**Source**: https://api.alternative.me/fng/

**Setup**: ✅ **No setup needed - already working!**

**Verification**:
```bash
curl "https://api.alternative.me/fng/" | jq '.data[0]'
# Returns: {"value": "21", "value_classification": "Extreme Fear", ...}
```

---

## ⚠️ **Needs API Key (Optional - FREE Tier Available)**

### **2. VIX Volatility Index** ⚠️ FREE TIER

**Status**: 🟡 **Implemented, needs API key to activate**

**Current Status**: Using fallback estimate (20.0)  
**To Activate**: Get free Financial Modeling Prep API key

#### **How to Get FMP API Key** (5 minutes):

1. **Go to**: https://financialmodelingprep.com/developer/docs/
2. **Click**: "Get your Free API Key"
3. **Sign up**: Email + password
4. **Verify email**: Check inbox
5. **Get key**: Dashboard shows your API key

**Free Tier**:
- ✅ 250 requests/day
- ✅ VIX, stock prices, economic data
- ✅ No credit card required

#### **Add to Your Platform**:

**Option 1: Environment Variable**
```bash
# Add to .env file
FMP_API_KEY=your_key_here
```

**Option 2: PM2 Config**
```javascript
// ecosystem.config.cjs
module.exports = {
  apps: [{
    name: 'trading-intelligence',
    env: {
      FMP_API_KEY: 'your_key_here'
    }
  }]
}
```

**Option 3: Wrangler Binding**
```bash
# In wrangler pages dev command
--binding FMP_API_KEY=your_key_here
```

**After Adding Key**:
```bash
# Restart service
pm2 restart trading-intelligence

# Test
curl "http://localhost:3000/api/agents/sentiment?symbol=BTC" | jq '.data.sentiment_metrics.volatility_index_vix'
# Should show: "source": "Financial Modeling Prep (LIVE)"
```

---

## 💰 **Paid APIs (For Production)**

### **3. Social Media Volume** 💰 PAID

**Current Status**: 🔴 Still using random simulation  
**Why**: No free APIs available for crypto social volume

#### **Option A: LunarCrush** (Recommended)

**What it provides**:
- Social media mentions across Twitter, Reddit, YouTube
- Social volume, social dominance, social score
- Real-time crypto social analytics

**Pricing**:
- **Basic**: $50/month (3,000 API calls/day)
- **Pro**: $99/month (10,000 API calls/day)
- **Enterprise**: Custom pricing

**How to Get**:
1. Go to: https://lunarcrush.com/developers/api
2. Sign up for account
3. Choose plan (start with Basic)
4. Get API key from dashboard

**Integration** (once you have key):
```typescript
async function fetchSocialVolume(apiKey: string, symbol: string) {
  const response = await fetch(
    `https://api.lunarcrush.com/v2?data=assets&key=${apiKey}&symbol=${symbol}`
  )
  const data = await response.json()
  return data.data[0].social_volume
}
```

**Add to config**:
```bash
# .env
LUNARCRUSH_API_KEY=your_key_here
```

#### **Option B: The TIE** (Alternative)

**Pricing**: $299/month (professional tier)  
**Website**: https://www.thetie.io/

#### **Option C: Do It Yourself** (FREE but complex)

Use free APIs:
- Twitter API (v2 free tier: 500K tweets/month)
- Reddit API (free, no limits)
- YouTube API (free, 10K requests/day)

Aggregate manually - requires coding.

---

### **4. Institutional Flow** 💰 PAID

**Current Status**: 🔴 Still using random simulation  
**Why**: Requires on-chain analytics services

#### **Option A: Glassnode** (Recommended)

**What it provides**:
- Exchange inflows/outflows
- Whale transactions
- On-chain institutional activity
- Derivative flows

**Pricing**:
- **Starter**: $29/month (limited metrics)
- **Advanced**: $399/month (full metrics)
- **Professional**: $799/month (real-time)

**How to Get**:
1. Go to: https://glassnode.com/
2. Sign up for account
3. Choose plan (Starter good for MVP)
4. Get API key from settings

**Integration** (once you have key):
```typescript
async function fetchInstitutionalFlow(apiKey: string) {
  const response = await fetch(
    `https://api.glassnode.com/v1/metrics/transactions/transfers_volume_exchanges_net?a=BTC&api_key=${apiKey}`
  )
  const data = await response.json()
  return data[data.length - 1].v // Latest net flow
}
```

**Add to config**:
```bash
# .env
GLASSNODE_API_KEY=your_key_here
```

#### **Option B: CryptoQuant** (Alternative)

**Pricing**: Starts at $99/month  
**Website**: https://cryptoquant.com/

#### **Option C: Whale Alert** (Alternative)

**Pricing**: $50-500/month  
**Website**: https://whale-alert.io/

---

## 📊 **Quick Setup Summary**

### **Immediate (FREE)**:
1. ✅ **Fear & Greed**: Already working!
2. ⏱️ **VIX**: Get FMP key (5 min, free)

### **Production (PAID)**:
3. 💰 **Social Volume**: LunarCrush ($50/month)
4. 💰 **Institutional Flow**: Glassnode ($29-799/month)

---

## 🚀 **Recommended Rollout Plan**

### **Phase 1: MVP/Demo** (Current)
- ✅ Fear & Greed: **LIVE** (Alternative.me)
- ⚠️ VIX: Fallback estimate (20.0)
- ⚠️ Social: Random simulation (100K-120K)
- ⚠️ Institutional: Random simulation (-12M to -2M)

**Cost**: $0/month  
**Data Quality**: 25% live, 75% simulated  
**Suitable for**: Demos, VC presentations (with disclosure)

---

### **Phase 2: Beta** (Recommended Next)
- ✅ Fear & Greed: **LIVE** (Alternative.me)
- ✅ VIX: **LIVE** (FMP API - free tier)
- ⚠️ Social: Random simulation (100K-120K)
- ⚠️ Institutional: Random simulation (-12M to -2M)

**Cost**: $0/month (FMP free tier)  
**Data Quality**: 50% live, 50% simulated  
**Suitable for**: Beta testing, early users

**Action**: Sign up for FMP (5 minutes)

---

### **Phase 3: Production** (Before Launch)
- ✅ Fear & Greed: **LIVE** (Alternative.me)
- ✅ VIX: **LIVE** (FMP API)
- ✅ Social: **LIVE** (LunarCrush)
- ✅ Institutional: **LIVE** (Glassnode Starter)

**Cost**: $79/month ($50 LunarCrush + $29 Glassnode)  
**Data Quality**: 100% live  
**Suitable for**: Production, real trading

**Action**: Subscribe to LunarCrush + Glassnode

---

### **Phase 4: Enterprise** (Scale)
- ✅ Fear & Greed: **LIVE** (Alternative.me)
- ✅ VIX: **LIVE** (FMP API Pro)
- ✅ Social: **LIVE** (LunarCrush Pro)
- ✅ Institutional: **LIVE** (Glassnode Advanced)

**Cost**: $548/month  
**Data Quality**: 100% live, high-frequency updates  
**Suitable for**: Large user base, institutional clients

---

## 🔧 **Testing Your Setup**

After adding API keys, verify they work:

### **Test Fear & Greed** (should work now):
```bash
curl "http://localhost:3000/api/agents/sentiment?symbol=BTC" | \
  jq '.data.sentiment_metrics.fear_greed_index'

# Expected: 
# "value": 21, 
# "source": "Alternative.me (LIVE)", 
# "data_freshness": "LIVE"
```

### **Test VIX** (after adding FMP key):
```bash
curl "http://localhost:3000/api/agents/sentiment?symbol=BTC" | \
  jq '.data.sentiment_metrics.volatility_index_vix'

# Expected after key: 
# "value": 16.5, 
# "source": "Financial Modeling Prep (LIVE)", 
# "data_freshness": "LIVE"
```

### **Verify Data Consistency**:
```bash
# Run twice - values should NOT change if live
curl -s "http://localhost:3000/api/agents/sentiment?symbol=BTC" | \
  jq '.data.sentiment_metrics.fear_greed_index.value'
# First run: 21

curl -s "http://localhost:3000/api/agents/sentiment?symbol=BTC" | \
  jq '.data.sentiment_metrics.fear_greed_index.value'
# Second run: 21 (SAME = LIVE DATA ✅)
```

---

## 💡 **Cost-Benefit Analysis**

| Phase | Monthly Cost | Data Quality | Use Case |
|-------|-------------|--------------|----------|
| Phase 1 (Current) | $0 | 25% live | MVP/Demo |
| Phase 2 (FMP) | $0 | 50% live | Beta |
| Phase 3 (Full) | $79 | 100% live | Production |
| Phase 4 (Scale) | $548 | 100% live HF | Enterprise |

**Recommendation**: 
- **Now**: Stay on Phase 1, add FMP key (free, 5 min)
- **Before production**: Phase 3 ($79/month)
- **After 1000 users**: Phase 4 ($548/month)

---

## ✅ **Current Status**

**What's LIVE** ✅:
- Fear & Greed Index (Alternative.me - FREE)
- Economic data (FRED API - FREE)
- Exchange prices (Binance, Coinbase, Kraken - FREE)

**What's Estimated** ⚠️:
- VIX (needs FMP key - FREE tier available)
- Social Volume (needs LunarCrush - $50/month)
- Institutional Flow (needs Glassnode - $29/month)

**Next Step**: Get FMP key (5 minutes, free) to activate VIX

---

**Questions?** Check the API documentation:
- FMP: https://financialmodelingprep.com/developer/docs/
- LunarCrush: https://lunarcrush.com/developers/api
- Glassnode: https://docs.glassnode.com/
