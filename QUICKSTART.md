# 🚀 Quick Start Guide

Get your Competitor Pricing Optimizer up and running in minutes!

## Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 2: Generate Sample Data

Since web scraping can be complex and may face restrictions, we'll start with sample data:

```bash
python src/scraper.py --use-sample --output data/raw/products.csv
```

This generates 500 sample products with realistic pricing, ratings, and reviews.

## Step 3: Preprocess Data

```bash
python src/preprocessing.py
```

This will:
- Clean the data
- Engineer features
- Encode categorical variables
- Save processed data to `data/processed/products_processed.csv`

## Step 4: Train Models

```bash
python train.py
```

This will:
- Train K-means clustering model (market segmentation)
- Train XGBoost price prediction model
- Save models to `models/` directory
- Display performance metrics

## Step 5: Launch Dashboard

```bash
streamlit run app.py
```

The dashboard will open in your browser at `http://localhost:8501`

## What You'll See

1. **Dashboard Tab**: Overview of products, price/rating distributions, top products
2. **Market Analysis Tab**: Market segmentation visualization, cluster statistics, 3D positioning
3. **Price Prediction Tab**: Input product features and get optimal price predictions
4. **Insights Tab**: Business insights, brand performance, pricing opportunities

## Optional: Explore Data

Open the Jupyter notebook for detailed analysis:

```bash
jupyter notebook notebooks/eda.ipynb
```

## Troubleshooting

### Import Errors
If you encounter import errors, make sure you're running commands from the project root directory.

### Models Not Found
If the dashboard shows "Models not loaded", make sure you've run `python train.py` first.

### Data Not Found
If you see "No data found", run the scraper with `--use-sample` flag first.

## Next Steps

- **Real Scraping**: Modify `src/scraper.py` to scrape real e-commerce sites (respect robots.txt!)
- **Custom Categories**: Add more product categories
- **Hyperparameter Tuning**: Enable in `train.py` for better model performance
- **Deploy**: Push to Streamlit Cloud for public access

## Expected Performance

After training, you should see:
- **Test RMSE**: <10% of average price
- **Test R²**: >0.85
- **Silhouette Score**: >0.5 (for clustering)

Happy optimizing! 🎯

