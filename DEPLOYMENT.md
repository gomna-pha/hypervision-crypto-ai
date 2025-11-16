# 🚀 Deployment Guide - Get Your Permanent URL

This guide will help you deploy ArbitrageAI to Cloudflare Pages and get a permanent, shareable URL like `https://arbitrage-ai.pages.dev`.

---

## ✅ Why Cloudflare Pages?

- **Free Tier**: Unlimited bandwidth & requests
- **Global CDN**: 300+ edge locations worldwide
- **Fast Deployment**: < 30 seconds build time
- **Permanent URL**: Never expires (unlike sandbox URLs)
- **Custom Domains**: Add your own domain for free
- **HTTPS**: Automatic SSL certificate
- **No Server Management**: Serverless architecture

---

## 📋 Prerequisites

1. **Cloudflare Account** (free)
   - Go to https://dash.cloudflare.com/sign-up
   - Create free account (no credit card required)

2. **Cloudflare API Token** (for deployment)
   - Login to Cloudflare Dashboard
   - Go to: My Profile → API Tokens → Create Token
   - Use template: "Edit Cloudflare Workers"
   - Copy the token (save it securely)

3. **Node.js 18+** (already installed in sandbox)

---

## 🎯 Deployment Steps

### Method 1: Using Sidebar Deploy Tab (Recommended)

1. **Configure API Key**
   ```
   → Click "Deploy" tab in sidebar
   → Follow setup instructions
   → Paste your Cloudflare API token
   → Save configuration
   ```

2. **Deploy with One Click**
   ```
   → The platform will guide you through deployment
   → Your URL will be: https://webapp.pages.dev
   ```

### Method 2: Manual Deployment (Command Line)

1. **Build the Project**
   ```bash
   cd /home/user/webapp
   npm run build
   ```
   
   This creates the `dist/` folder with:
   - `_worker.js` (compiled Hono application)
   - `_routes.json` (routing configuration)
   - Static assets from `public/static/`

2. **Set Cloudflare API Token**
   ```bash
   export CLOUDFLARE_API_TOKEN="your-token-here"
   ```

3. **Create Cloudflare Pages Project** (first time only)
   ```bash
   npx wrangler pages project create arbitrage-ai \
     --production-branch main \
     --compatibility-date 2025-11-16
   ```
   
   Replace `arbitrage-ai` with your preferred project name.

4. **Deploy to Cloudflare**
   ```bash
   npx wrangler pages deploy dist --project-name arbitrage-ai
   ```
   
   **Output**:
   ```
   ✨ Success! Uploaded 3 files
   ✨ Deployment complete! Take a peek at:
   
   🌎 Production: https://arbitrage-ai.pages.dev
   🌳 Branch: https://main.arbitrage-ai.pages.dev
   ```

5. **Save Your URL**
   ```bash
   # Production URL (permanent):
   https://arbitrage-ai.pages.dev
   
   # This URL will NEVER expire!
   ```

---

## 🔄 Update Deployment (After Changes)

Whenever you make changes to the code:

```bash
cd /home/user/webapp
npm run build
npx wrangler pages deploy dist --project-name arbitrage-ai
```

Or use the npm script:
```bash
npm run deploy:prod
```

---

## 🌐 Custom Domain Setup (Optional)

Make your URL even cleaner: `https://trading.yourdomain.com`

1. **Add Custom Domain**
   ```bash
   npx wrangler pages domain add trading.yourdomain.com \
     --project-name arbitrage-ai
   ```

2. **Update DNS Records**
   - Add CNAME record: `trading` → `arbitrage-ai.pages.dev`
   - Wait 5-10 minutes for propagation

3. **Access Your Custom URL**
   ```
   https://trading.yourdomain.com
   ```

---

## 🎨 Update GitHub README with URL

After deployment, update your GitHub README:

```markdown
## 🔗 Live Platform

**Production URL**: https://arbitrage-ai.pages.dev

Try the live platform now! No installation required.
```

---

## 📊 Verify Deployment

Test these endpoints to ensure everything works:

```bash
# 1. Homepage
curl https://arbitrage-ai.pages.dev

# 2. AI Agents API
curl https://arbitrage-ai.pages.dev/api/agents

# 3. Opportunities API
curl https://arbitrage-ai.pages.dev/api/opportunities

# 4. Backtest API
curl "https://arbitrage-ai.pages.dev/api/backtest?cnn=true&strategy=Deep%20Learning"
```

All should return proper responses.

---

## 🔧 Troubleshooting

### Issue: "No wrangler.toml found"
**Solution**: The project uses `wrangler.jsonc` instead. This is normal.

### Issue: "Authentication error"
**Solution**: 
1. Verify your API token is correct
2. Check token permissions (needs "Edit Cloudflare Workers")
3. Re-export: `export CLOUDFLARE_API_TOKEN="your-token"`

### Issue: "Project already exists"
**Solution**: Skip the `create` step, go directly to `deploy`:
```bash
npx wrangler pages deploy dist --project-name arbitrage-ai
```

### Issue: "Build failed"
**Solution**:
```bash
# Clean and rebuild
rm -rf dist node_modules
npm install
npm run build
```

---

## 💡 Deployment Best Practices

### 1. Test Locally First
```bash
npm run build
pm2 start ecosystem.config.cjs
curl http://localhost:3000
```

### 2. Use Git Version Control
```bash
git add .
git commit -m "Ready for deployment"
git push origin main
```

### 3. Deploy to Production
```bash
npm run build
npx wrangler pages deploy dist --project-name arbitrage-ai
```

### 4. Verify in Browser
- Open: https://arbitrage-ai.pages.dev
- Test all 4 tabs (Dashboard, Strategies, Backtest, Analytics)
- Click "Execute" on opportunities
- Run backtest with different strategies
- Check autonomous trading agent

---

## 📈 Post-Deployment Checklist

✅ Homepage loads correctly  
✅ All 4 navigation tabs work  
✅ AI agents display dynamic data  
✅ Charts render properly  
✅ Opportunities table loads  
✅ Execute buttons work  
✅ Backtest runs successfully  
✅ Analytics charts display  
✅ Autonomous agent controls work  
✅ Mobile responsive (test on phone)  

---

## 🌟 Share Your URL

Your permanent URL is now ready to share:

**For GitHub**:
```markdown
🔗 **Live Demo**: https://arbitrage-ai.pages.dev
```

**For Social Media**:
```
Check out my AI-powered crypto trading platform! 🚀
https://arbitrage-ai.pages.dev

Features:
✅ 13 Trading Strategies
✅ 5 AI Agents
✅ Autonomous Trading
✅ Comprehensive Analytics
```

**For Email**:
```
View the live platform: https://arbitrage-ai.pages.dev

This is a production-ready cryptocurrency arbitrage platform with:
- Multi-strategy portfolio (23.7% return)
- CNN pattern recognition
- Autonomous trading agent
- Professional backtesting
```

---

## 🚀 Next Steps

1. **Deploy Now**: Follow the steps above to get your permanent URL
2. **Update GitHub README**: Add your live URL
3. **Share with VCs**: Use the permanent link in pitch decks
4. **Monitor Performance**: Check Cloudflare Analytics dashboard
5. **Add Custom Domain**: Make it even more professional

---

## 📞 Need Help?

- **Cloudflare Docs**: https://developers.cloudflare.com/pages/
- **Wrangler CLI**: https://developers.cloudflare.com/workers/wrangler/
- **Hono Framework**: https://hono.dev/

---

**Remember**: The sandbox URL (`https://3000-....sandbox.novita.ai`) expires after a few hours. Your Cloudflare Pages URL (`https://arbitrage-ai.pages.dev`) is **permanent** and will work forever! 🎉
