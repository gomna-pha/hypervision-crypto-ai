# 🚀 RAILWAY DEPLOYMENT GUIDE - Real-Time Arbitrage System

## Prerequisites

1. **Railway Account**: Sign up at https://railway.app
2. **GitHub Repository**: https://github.com/gomna-pha/hypervision-crypto-ai
3. **Railway CLI** (optional): `npm install -g @railway/cli`

---

## Option 1: Deploy via Railway Dashboard (Easiest)

### Step 1: Create New Project

1. Go to https://railway.app/new
2. Click "Deploy from GitHub repo"
3. Select `gomna-pha/hypervision-crypto-ai`
4. Click "Deploy Now"

### Step 2: Configure Environment Variables

In Railway dashboard → Variables tab, add:

```
NODE_ENV=production
PORT=8787
TRADING_SYMBOLS=BTC,ETH,SOL
```

###  Step 3: Wait for Deployment

Railway will automatically:
- Detect Node.js project
- Run `npm install`
- Run `npm run build`
- Start with `npm start`

### Step 4: Get Your URL

- Railway will assign a URL like: `https://your-app.up.railway.app`
- Click "Generate Domain" if needed

---

## Option 2: Deploy via Railway CLI (Advanced)

### Step 1: Install Railway CLI

```bash
npm install -g @railway/cli
```

### Step 2: Login

```bash
railway login
```

### Step 3: Initialize Project

```bash
cd /home/user/webapp
railway init
```

### Step 4: Deploy

```bash
railway up
```

### Step 5: Set Environment Variables

```bash
railway variables set NODE_ENV=production
railway variables set PORT=8787
railway variables set TRADING_SYMBOLS=BTC,ETH,SOL
```

### Step 6: Get URL

```bash
railway domain
```

---

## Option 3: Deploy with Docker

### Step 1: Build Docker Image

```bash
cd /home/user/webapp
docker build -t arbitrage-ai .
```

### Step 2: Run Locally (Test)

```bash
docker run -p 8787:8787 \
  -e NODE_ENV=production \
  -e TRADING_SYMBOLS=BTC,ETH,SOL \
  arbitrage-ai
```

### Step 3: Push to Railway

Railway supports Docker deployments. In your Railway project:
1. Settings → Deployments
2. Select "Docker" as builder
3. Railway will use the `Dockerfile` in your repo

---

## Verification Steps

### 1. Check Server Health

```bash
curl https://your-app.up.railway.app/health
```

**Expected response:**
```json
{
  "status": "ok",
  "timestamp": "2025-12-19T...",
  "uptime": 123.45
}
```

### 2. Start Real-Time Pipeline

```bash
curl -X POST https://your-app.up.railway.app/api/ml/realtime/start \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["BTC", "ETH", "SOL"]}'
```

**Expected response:**
```json
{
  "success": true,
  "message": "Real-time ML pipeline started",
  "data": {
    "isRunning": true,
    "websocketConnected": true,
    "exchanges": {
      "binance": true,
      "coinbase": true,
      "kraken": true
    }
  }
}
```

### 3. Check WebSocket Status

```bash
curl https://your-app.up.railway.app/api/ml/realtime/ws-status
```

**Expected response:**
```json
{
  "success": true,
  "data": {
    "websocket": {
      "connected": true,
      "exchanges": {
        "binance": true,
        "coinbase": true,
        "kraken": true
      }
    },
    "latestData": {
      "BTC": {
        "spotPrice": 96500,
        "dataQuality": "excellent",
        "bestArbitrageSpread": 15.2
      }
    }
  }
}
```

### 4. Get Live ML Output

```bash
curl https://your-app.up.railway.app/api/ml/realtime/output/BTC
```

**Expected response:**
```json
{
  "success": true,
  "data": {
    "symbol": "BTC-USD",
    "dataSource": "websocket",
    "dataQuality": "excellent",
    "spotPrice": 96500,
    "regime": "neutral",
    "regimeConfidence": 0.65,
    "metaModel": {
      "confidence": 0.72,
      "action": "WAIT"
    },
    "latencyMs": 145
  }
}
```

---

## Monitoring

### Railway Logs

```bash
railway logs
```

Or view in dashboard: Project → Deployments → Logs

### Expected Log Output

```
🚀 HYPERVISION CRYPTO AI - REAL-TIME ARBITRAGE

Status: PRODUCTION
Port: 8787
Environment: production

Initializing real-time services...

✅ Binance WebSocket connected
✅ Coinbase WebSocket connected
✅ Kraken WebSocket connected

✅ Real-Time ML Service Started
   Symbols: BTC, ETH, SOL
   WebSocket Connections: Active
   ML Pipeline: Running

🌐 Server ready at http://localhost:8787
```

---

## Troubleshooting

### Issue: WebSocket Connection Fails

**Solution**: Check if symbols are correctly formatted

```bash
# Binance expects: BTCUSDT, ETHUSDT
# Coinbase expects: BTC-USD, ETH-USD
# Kraken expects: XBT/USD, ETH/USD

# The system auto-converts, but verify logs
railway logs | grep WebSocket
```

### Issue: High Memory Usage

**Solution**: Upgrade Railway plan or reduce symbols

```bash
# Monitor memory
railway status

# Reduce symbols to 1-2 for free tier
railway variables set TRADING_SYMBOLS=BTC,ETH
```

### Issue: Build Fails

**Solution**: Check build logs

```bash
railway logs --deployment
```

Common fixes:
- Ensure `package.json` has `"start": "node dist/server.js"`
- Verify TypeScript compiles: `npm run build`
- Check Node.js version (requires 18+)

---

## Cost Estimate

### Railway Pricing

| Tier | Cost | Resources | Best For |
|------|------|-----------|----------|
| **Free** | $0/month | $5 credit/month | Testing (1-2 symbols) |
| **Hobby** | $5/month | Shared CPU, 512MB RAM | Small trading (3-5 symbols) |
| **Pro** | $20/month | 2GB RAM, more CPU | Full production (10+ symbols) |

### Recommended Setup

- **Phase 1 (Testing)**: Free tier
  - 1-2 symbols (BTC, ETH)
  - ~$0-5/month usage
  
- **Phase 2 (Production)**: Hobby tier
  - 3-5 symbols
  - $5/month flat rate
  
- **Phase 3 (Scale)**: Pro tier
  - 10+ symbols
  - Add InfluxDB, Redis
  - ~$20-30/month

---

## Next Steps After Deployment

### 1. Frontend Integration

Update frontend to use Railway URL:

```typescript
// In src/ml-api-endpoints.ts or frontend config
const API_BASE_URL = 'https://your-app.up.railway.app';
```

### 2. Add Custom Domain (Optional)

Railway dashboard → Settings → Domains → Add Custom Domain

### 3. Set Up Monitoring

Add health check monitoring:
- Uptime Robot: https://uptimerobot.com (free)
- Better Uptime: https://betteruptime.com (free tier)

### 4. Enable Auto-Deploy

Railway automatically deploys on git push to main branch.

To disable:
- Settings → Deployments → Manual deploys

---

## Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `NODE_ENV` | `development` | Set to `production` for Railway |
| `PORT` | `8787` | Server port (Railway auto-assigns) |
| `TRADING_SYMBOLS` | `BTC,ETH,SOL` | Comma-separated symbols to track |
| `ENABLE_GA` | `true` | Enable genetic algorithm |
| `ENABLE_XGBOOST` | `true` | Enable XGBoost meta-model |
| `TOTAL_CAPITAL` | `100000` | Portfolio capital (USD) |
| `MAX_LEVERAGE` | `3` | Maximum leverage allowed |
| `MAX_DRAWDOWN` | `20` | Max drawdown percentage |

---

## Security Best Practices

### 1. API Keys (for future execution layer)

Store in Railway environment variables (not in code):

```bash
railway variables set BINANCE_API_KEY=your_key
railway variables set BINANCE_API_SECRET=your_secret
```

### 2. Rate Limiting

Add rate limiting to API endpoints (TODO: implement)

### 3. HTTPS Only

Railway provides free SSL certificates automatically

### 4. CORS Configuration

Configure allowed origins for production (TODO: implement)

---

## Quick Reference

### Essential Commands

```bash
# Deploy
railway up

# View logs
railway logs

# Check status
railway status

# Set variables
railway variables set KEY=value

# Get URL
railway domain

# Open dashboard
railway open
```

### Essential URLs

```bash
# Health check
https://your-app.up.railway.app/health

# Start pipeline
https://your-app.up.railway.app/api/ml/realtime/start

# Get status
https://your-app.up.railway.app/api/ml/realtime/status

# Dashboard
https://your-app.up.railway.app/
```

---

## Support

- **Railway Docs**: https://docs.railway.app
- **Railway Discord**: https://discord.gg/railway
- **Project Issues**: https://github.com/gomna-pha/hypervision-crypto-ai/issues

---

**Last Updated**: 2025-12-19  
**Status**: Ready for deployment  
**Estimated Setup Time**: 15-30 minutes
