# HyperVision AI - Real-Time Crypto Arbitrage Trading System

## 🚀 Live Deployment

**Production URL:** https://arbitrage-ai.pages.dev  
**Latest Deployment:** https://bf19a52d.arbitrage-ai.pages.dev

## ✅ System Status: FIXED & WORKING

### Recent Fixes (Dec 20, 2025)

#### Bug #1: Element ID Mismatches (Commit `8762558`)
- **Issue:** JavaScript trying to update wrong element IDs
- **Fixed:** Corrected all agent element IDs:
  - `agent-cross-score` → `agent-cross-exchange-score`
  - `agent-onchain-score` → `agent-on-chain-score`
  - `agent-cnn-score` → `agent-cnn-pattern-score`

#### Bug #2: Missing Null Checks (Commit `bd75f57`)
- **Issue:** Dashboard crashing when DOM elements missing
- **Fixed:** Added comprehensive null checks for all 40+ element updates
- **Result:** Graceful degradation, no more crashes

## 🔧 Deployment Setup

### Automatic Deployments via GitHub Actions

Every push to `main` branch automatically deploys to Cloudflare Pages.

**Required GitHub Secrets:**
- `CLOUDFLARE_API_TOKEN`: `RZt5Bvio1HdhF29QpXFTRBQt3ZASMNuMb5A-kk2_`
- `CLOUDFLARE_ACCOUNT_ID`: `cc8c9f01a363ccf1a1a697742b9af8bd`

**To set up secrets:**
1. Go to: https://github.com/gomna-pha/hypervision-crypto-ai/settings/secrets/actions
2. Click "New repository secret"
3. Add both secrets above

### Manual Deployment

```bash
npm run build
export CLOUDFLARE_API_TOKEN="your-token"
npx wrangler pages deploy dist --project-name arbitrage-ai
```

## 📊 Architecture

### Real-Time Data Flow

1. **Market Data Feeds** → Spot/Perp prices, funding rates, cross-exchange spreads
2. **Feature Engineering** → Returns, volatility, Z-scores, flow metrics
3. **5 AI Agents** → Economic, Sentiment, Cross-Exchange, On-Chain, CNN Pattern
4. **Genetic Algorithm** → Signal selection & optimization
5. **Hyperbolic Embedding** → Hierarchical signal relationships
6. **Market Regime Detection** → Crisis/Stress/Defensive/Neutral/Risk-On
7. **XGBoost Meta-Model** → Arbitrage confidence scoring
8. **Regime-Conditional Strategies** → 10 arbitrage algorithms
9. **Portfolio & Risk Control** → Capital allocation, drawdown limits
10. **Execution Layer** → TWAP/VWAP (PENDING IMPLEMENTATION)

## 🔑 Key Features

✅ **Real-time market data** from CoinGecko, Blockchain.com, Alternative.me  
✅ **5 AI agents** generating signals every 4 seconds  
✅ **Genetic Algorithm** for signal optimization  
✅ **XGBoost meta-model** for confidence scoring  
✅ **10 arbitrage algorithms** with ML confidence  
✅ **Complete dashboard** with auto-refresh  
✅ **Defensive programming** with comprehensive error handling

## 📈 API Endpoints

- `GET /api/agents` - All 5 AI agent signals
- `POST /api/ml/pipeline` - Full ML pipeline execution
- `GET /api/opportunities` - 10 arbitrage opportunities
- `GET /api/portfolio/metrics` - Real-time portfolio metrics

## 🛠️ Development

```bash
# Install dependencies
npm install

# Development server
npm run dev

# Build for production
npm run build

# Deploy to Cloudflare Pages
npm run deploy
```

## 📝 Next Steps

### Phase 1: Real-Time Infrastructure (In Progress)
- ✅ WebSocket service (written, needs Railway deployment)
- ⏳ InfluxDB for time-series data
- ⏳ Versioned feature store

### Phase 2: Execution Layer
- ⏳ Exchange API integration (Binance, Coinbase)
- ⏳ TWAP/VWAP execution algorithms
- ⏳ Order placement & position tracking

### Phase 3: Monitoring & Validation
- ⏳ Grafana + Prometheus dashboards
- ⏳ Backtesting framework
- ⏳ Walk-forward validation

## 📚 Documentation

- `RAILWAY_DEPLOYMENT.md` - WebSocket deployment guide
- `START_HERE.md` - Complete roadmap
- `PRODUCTION_READINESS_ASSESSMENT.md` - System status
- `FINAL_DELIVERY_SUMMARY.md` - Current capabilities

## 🐛 Troubleshooting

### Dashboard showing "ERROR"
- **Fixed!** Latest deployment includes all null checks
- Clear browser cache and reload: https://bf19a52d.arbitrage-ai.pages.dev

### Agents showing "--"
- Check API endpoints are responding: `/api/agents`
- Verify network connectivity
- Check browser console for errors (should be none now)

## 📞 Support

**Repository:** https://github.com/gomna-pha/hypervision-crypto-ai  
**Issues:** https://github.com/gomna-pha/hypervision-crypto-ai/issues

---

**Built with:** Hono.js, TypeScript, Vite, Cloudflare Pages  
**ML Stack:** XGBoost, Genetic Algorithms, Hyperbolic Embeddings  
**Real-Time:** WebSockets (planned), REST APIs (live)
