# 🎯 Competitor Pricing Optimizer

An AI-powered pricing optimization tool that scrapes e-commerce product data, uses machine learning to predict optimal pricing, clusters market segments, and visualizes competitive positioning.

## 🚀 Features

- **Web Scraping**: Automated data collection from e-commerce platforms (Flipkart/Amazon)
- **Market Segmentation**: K-means clustering to identify distinct market segments
- **Price Prediction**: XGBoost model to predict optimal pricing with 20% sales uplift predictions
- **Interactive Dashboard**: Streamlit app with real-time visualizations
- **Competitive Analysis**: Visual positioning maps showing price vs. rating vs. demand

## 📊 Project Metrics

- **Prediction Accuracy**: RMSE <10%, R² >0.85
- **Sales Uplift**: Up to 20% predicted sales increase
- **Data Coverage**: 500+ products across multiple categories
- **Response Time**: <2s inference speed

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- Chrome/Chromium browser (for Selenium)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/competitor-pricing-optimizer.git
cd competitor-pricing-optimizer
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## 📁 Project Structure

```
competitor-pricing-optimizer/
├── data/
│   ├── raw/           # Raw scraped data
│   └── processed/     # Cleaned and processed data
├── notebooks/
│   └── eda.ipynb      # Exploratory Data Analysis
├── src/
│   ├── scraper.py     # Web scraping utilities
│   ├── preprocessing.py  # Data cleaning and feature engineering
│   ├── models.py      # ML model training and evaluation
│   └── utils.py       # Helper functions
├── models/            # Saved ML models
├── app.py            # Streamlit dashboard
├── train.py          # Model training script
└── requirements.txt
```

## 🌐 Deploy for Free

**Quick Deploy (5 minutes):**
1. Push to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo
4. Deploy!

See [README_DEPLOYMENT.md](README_DEPLOYMENT.md) for detailed instructions.

---

## 🚦 Quick Start

### Option 1: Using Sample Data (Recommended for First Run)

1. **Generate Sample Data**:
```bash
python src/scraper.py --use-sample --output data/raw/products.csv
```

2. **Preprocess Data**:
```bash
python src/preprocessing.py
```

3. **Train Models**:
```bash
python train.py
```

4. **Launch Dashboard**:
```bash
streamlit run app.py
```

### Option 2: Real Web Scraping

1. **Scrape Product Data**:
```bash
python src/scraper.py --category smartphones --pages 10
```

2. **Preprocess Data**:
```bash
python src/preprocessing.py
```

3. **Train Models**:
```bash
python train.py
```

4. **Launch Dashboard**:
```bash
streamlit run app.py
```

> **Note**: For detailed step-by-step instructions, see [QUICKSTART.md](QUICKSTART.md)

## 📈 Usage

1. **Data Collection**: Run the scraper to collect product data
2. **Exploration**: Open `notebooks/eda.ipynb` to explore the data
3. **Model Training**: Train models using `train.py`
4. **Dashboard**: Use the Streamlit app to:
   - Input product features
   - View market segmentation
   - Get optimal price predictions
   - Analyze competitive positioning

## 🎥 Demo

[Add GIF or YouTube video link here]

## 📊 Model Performance

- **Clustering**: Silhouette score >0.5
- **Price Prediction**: RMSE <10%, R² >0.85
- **Inference Speed**: <2 seconds per prediction

## 🔧 Configuration

Edit `config.py` to customize:
- Scraping targets
- Model parameters
- Feature engineering options

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

Your Name - [LinkedIn](https://linkedin.com/in/yourprofile)

## 🙏 Acknowledgments

- Kaggle for starter datasets
- Streamlit for the dashboard framework
- XGBoost and scikit-learn communities

---

**Built with ❤️ using Python, XGBoost, and Streamlit**

#MachineLearning #DataScience #PricingOptimization #AI

