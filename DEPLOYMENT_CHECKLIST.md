# ✅ Deployment Checklist

Use this checklist before deploying to ensure everything is ready.

## Pre-Deployment

- [ ] All code is working locally
- [ ] `streamlit run app.py` works without errors
- [ ] Models are trained (`python train.py` runs successfully)
- [ ] Data files exist or can be generated
- [ ] `requirements.txt` has all dependencies
- [ ] No hardcoded absolute paths
- [ ] `.gitignore` is configured correctly

## GitHub Setup

- [ ] Git repository initialized
- [ ] All files committed
- [ ] Repository pushed to GitHub
- [ ] Repository is **Public** (required for free Streamlit Cloud)

## Streamlit Cloud Deployment

- [ ] Signed up at [share.streamlit.io](https://share.streamlit.io)
- [ ] Connected GitHub account
- [ ] Created new app
- [ ] Selected correct repository
- [ ] Set main file path: `app.py`
- [ ] App URL is set
- [ ] Clicked "Deploy"

## Post-Deployment

- [ ] App loads without errors
- [ ] All tabs work correctly
- [ ] Data visualizations display
- [ ] Price prediction works
- [ ] Tested on different devices/browsers
- [ ] Shared URL on LinkedIn/GitHub

## Troubleshooting

If deployment fails:
- [ ] Check build logs in Streamlit Cloud
- [ ] Verify `requirements.txt` is correct
- [ ] Ensure `app.py` is in root directory
- [ ] Check Python version (3.8+)
- [ ] Verify all imports work

## Optional: Include Models in Repo

If you want the app to work immediately (without generating models on first run):

```bash
# Temporarily allow models in git
git add models/*.pkl models/*.json
git commit -m "Add trained models for deployment"
git push
```

**Note**: Model files are small (~1-2 MB), so this is fine for demo purposes.

---

**Ready to deploy?** Follow [README_DEPLOYMENT.md](README_DEPLOYMENT.md) for step-by-step instructions!

