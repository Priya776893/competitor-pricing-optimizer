# 🚀 Quick Deployment Guide

## Deploy to Streamlit Cloud (FREE) - 5 Minutes!

### Step 1: Push to GitHub

```bash
# Initialize git (if not done)
git init
git add .
git commit -m "Ready for deployment"

# Create repo on GitHub, then:
git remote add origin https://github.com/YOUR_USERNAME/competitor-pricing-optimizer.git
git branch -M main
git push -u origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to **[share.streamlit.io](https://share.streamlit.io)**
2. Sign in with GitHub
3. Click **"New app"**
4. Fill in:
   - **Repository**: `YOUR_USERNAME/competitor-pricing-optimizer`
   - **Branch**: `main`
   - **Main file path**: `app.py`
   - **App URL**: `competitor-pricing-optimizer` (or your choice)
5. Click **"Deploy"**

### Step 3: Wait & Done! 🎉

- Build takes 2-3 minutes
- Your app will be live at: `https://competitor-pricing-optimizer.streamlit.app`
- Auto-updates on every git push!

---

## Important Notes

### Option A: Include Models in Repo (Recommended for Demo)

For the app to work immediately, include trained models:

```bash
# Remove models from .gitignore temporarily
git add models/*.pkl models/*.json
git commit -m "Add trained models"
git push
```

### Option B: Generate on First Run

The app will auto-generate data and train models on first run (takes ~30 seconds).

---

## Alternative: Hugging Face Spaces

1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Create new Space → Select "Streamlit"
3. Upload your files
4. Done!

---

## Troubleshooting

**Build fails?**
- Check `requirements.txt` has all packages
- Ensure `app.py` is in root directory
- Check build logs in Streamlit Cloud dashboard

**App loads but shows errors?**
- Models might need to be generated (first run takes time)
- Check logs in Streamlit Cloud

**Need help?**
- See full guide in `DEPLOYMENT.md`
- Check Streamlit Cloud docs: [docs.streamlit.io/streamlit-cloud](https://docs.streamlit.io/streamlit-cloud)

---

**Your app URL will be**: `https://YOUR-APP-NAME.streamlit.app`

Share it on LinkedIn! 🚀

