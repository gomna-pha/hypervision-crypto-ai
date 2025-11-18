# 🚀 Auto-Deploy Setup for ArbitrageAI

## ✅ What I've Created for You

I've set up **GitHub Actions** to automatically deploy your code to Cloudflare Pages whenever you push to the `main` branch.

**File Created:** `.github/workflows/cloudflare-pages.yml`

---

## 🔧 Setup Instructions (5 Minutes)

### Step 1: Add GitHub Secrets

You need to add 2 secrets to your GitHub repository:

1. **Go to your GitHub repository:**
   - Visit: https://github.com/gomna-pha/hypervision-crypto-ai

2. **Navigate to Settings:**
   - Click **Settings** tab (top right)
   - Click **Secrets and variables** → **Actions** (left sidebar)
   - Click **New repository secret** button

3. **Add Secret #1 - CLOUDFLARE_API_TOKEN:**
   - Name: `CLOUDFLARE_API_TOKEN`
   - Value: `RZt5Bvio1HdhF29QpXFTRBQt3ZASMNuMb5A-kk2_`
   - Click **Add secret**

4. **Add Secret #2 - CLOUDFLARE_ACCOUNT_ID:**
   - Click **New repository secret** again
   - Name: `CLOUDFLARE_ACCOUNT_ID`
   - Value: `cc8c9f01a363ccf1a1a697742b9af8bd`
   - Click **Add secret**

---

## 🎯 How Auto-Deploy Works

Once secrets are added, the workflow automatically:

```
You push to GitHub main branch
         ↓
GitHub Actions triggers
         ↓
Installs dependencies (npm ci)
         ↓
Builds project (npm run build)
         ↓
Deploys to Cloudflare Pages
         ↓
Production updated at https://arbitrage-ai.pages.dev
```

**Total time:** ~2-3 minutes from push to live! ⚡

---

## 📝 Workflow Details

**File Location:** `.github/workflows/cloudflare-pages.yml`

**Triggers on:**
- Push to `main` branch

**Steps:**
1. ✅ Checkout code
2. ✅ Setup Node.js 18
3. ✅ Install dependencies
4. ✅ Build project
5. ✅ Deploy to Cloudflare Pages

**Project Name:** `arbitrage-ai`

---

## 🧪 Test Auto-Deploy

After adding the secrets, test it:

### Method 1: Make a Small Change

```bash
# On your local machine
cd /path/to/hypervision-crypto-ai

# Make a small change
echo "# Updated" >> README.md

# Commit and push
git add README.md
git commit -m "Test auto-deploy"
git push origin main
```

### Method 2: Watch Deployment

1. **Go to:** https://github.com/gomna-pha/hypervision-crypto-ai/actions
2. **You'll see:** "Deploy to Cloudflare Pages" workflow running
3. **Wait:** 2-3 minutes
4. **Check:** https://arbitrage-ai.pages.dev

---

## ✅ Verification Checklist

After setup, verify:

- [ ] Both secrets added to GitHub
  - [ ] CLOUDFLARE_API_TOKEN
  - [ ] CLOUDFLARE_ACCOUNT_ID
- [ ] Workflow file pushed to GitHub
- [ ] Test push triggers deployment
- [ ] GitHub Actions runs successfully (green checkmark)
- [ ] Production updates automatically

---

## 🔍 Monitoring Deployments

### View Workflow Runs:
https://github.com/gomna-pha/hypervision-crypto-ai/actions

### View Cloudflare Deployments:
https://dash.cloudflare.com/pages/arbitrage-ai/deployments

### Check Production:
https://arbitrage-ai.pages.dev

---

## 📋 Common Issues & Solutions

### Issue 1: Workflow doesn't trigger
**Solution:** Make sure `.github/workflows/cloudflare-pages.yml` is pushed to GitHub

```bash
cd /home/user/webapp
git add .github/workflows/cloudflare-pages.yml
git commit -m "Add auto-deploy workflow"
git push origin main
```

### Issue 2: Deployment fails with "Authentication failed"
**Solution:** Double-check the secrets:
- CLOUDFLARE_API_TOKEN is correct
- CLOUDFLARE_ACCOUNT_ID is correct
- No extra spaces in secret values

### Issue 3: Build fails
**Solution:** Check GitHub Actions logs:
1. Go to Actions tab
2. Click on the failed workflow
3. Check the error message
4. Usually means missing dependencies or build errors

---

## 🚀 Future Deployments

From now on, deploying is as simple as:

```bash
# Make changes to your code
# ... edit files ...

# Commit changes
git add .
git commit -m "Add new feature"

# Push to GitHub
git push origin main

# Wait 2-3 minutes
# Production automatically updates! ✅
```

**No manual wrangler commands needed!** 🎉

---

## 📊 What Gets Deployed

Every push to `main` branch deploys:
- ✅ All changes in `src/index.tsx`
- ✅ All changes in `public/static/app.js`
- ✅ All changes in `public/static/styles.css`
- ✅ Updated build output to `dist/`
- ✅ Live at https://arbitrage-ai.pages.dev

---

## 🔐 Security Notes

**GitHub Secrets are secure:**
- ✅ Encrypted at rest
- ✅ Not visible in logs
- ✅ Only accessible by workflows
- ✅ Can be rotated anytime

**To rotate API token:**
1. Create new Cloudflare API token
2. Update GitHub secret
3. Next deployment uses new token

---

## 📞 Support

**If auto-deploy doesn't work:**

1. Check workflow syntax: `.github/workflows/cloudflare-pages.yml`
2. Verify secrets in GitHub: Settings → Secrets and variables → Actions
3. Check GitHub Actions logs for errors
4. Check Cloudflare dashboard for deployment status

**Workflow is ready to use once you add the secrets!** 🚀

---

## ✅ Quick Setup Summary

1. **Add 2 secrets to GitHub:**
   - CLOUDFLARE_API_TOKEN: `RZt5Bvio1HdhF29QpXFTRBQt3ZASMNuMb5A-kk2_`
   - CLOUDFLARE_ACCOUNT_ID: `cc8c9f01a363ccf1a1a697742b9af8bd`

2. **Push workflow file to GitHub:**
   ```bash
   git push origin main
   ```

3. **Test by making any change and pushing**

4. **Watch it auto-deploy!** ✅

---

**Created:** 2025-11-18  
**Status:** ✅ Ready to use (just add secrets)  
**Auto-Deploy:** Enabled after setup  
**Deployment Time:** 2-3 minutes per push
