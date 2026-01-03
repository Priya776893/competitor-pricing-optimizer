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

## 📈 Usage

1. **Data Collection**: Run the scraper to collect product data
2. **Exploration**: Open `notebooks/eda.ipynb` to explore the data
3. **Model Training**: Train models using `train.py`
4. **Dashboard**: Use the Streamlit app to:
   - Input product features
   - View market segmentation
   - Get optimal price predictions
   - Analyze competitive positioning

## 🎥 Demo Link -> https://competitor-pricing-optimizer-priya-prasad.streamlit.app/
<img width="1919" height="1029" alt="Screenshot 2026-01-03 222703" src="https://github.com/user-attachments/assets/4374929d-e23d-45f9-8618-a6ad5d2ee77a" />


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

Priya Prasad - [LinkedIn](https://www.linkedin.com/in/priya-prasad1312/)

## 🙏 Acknowledgments

- Kaggle for starter datasets
- Streamlit for the dashboard framework
- XGBoost and scikit-learn communities

---

**Built with ❤️ using Python, XGBoost, and Streamlit**

#MachineLearning #DataScience #PricingOptimization #AI

