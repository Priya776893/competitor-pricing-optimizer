#!/bin/bash
# Script to prepare project for deployment
# Run this before pushing to GitHub

echo "🚀 Preparing Competitor Pricing Optimizer for Deployment..."
echo ""

# Check if models exist
if [ ! -f "models/prediction_model.pkl" ]; then
    echo "⚠️  Models not found. Training models..."
    python train.py
fi

# Check if data exists
if [ ! -f "data/processed/products_processed.csv" ]; then
    echo "⚠️  Processed data not found. Generating..."
    python src/scraper.py --use-sample
    python train.py
fi

echo ""
echo "✅ Preparation complete!"
echo ""
echo "Next steps:"
echo "1. Review .gitignore - decide if you want to commit models"
echo "2. If committing models, run: git add models/*.pkl models/*.json"
echo "3. git add ."
echo "4. git commit -m 'Ready for deployment'"
echo "5. git push"
echo ""
echo "Then deploy on Streamlit Cloud: https://share.streamlit.io"

