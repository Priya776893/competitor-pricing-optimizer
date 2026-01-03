"""
Configuration file for Competitor Pricing Optimizer
"""

# Scraping Configuration
SCRAPING_CONFIG = {
    'delay': 2,  # Delay between requests (seconds)
    'timeout': 10,  # Request timeout (seconds)
    'max_retries': 3,
    'user_agents': [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36'
    ],
    'headers': {
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    }
}

# Data Paths
DATA_PATHS = {
    'raw': 'data/raw',
    'processed': 'data/processed',
    'models': 'models'
}

# Model Configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'n_clusters': 5,  # Default number of clusters
    'xgboost_params': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
}

# Feature Engineering
FEATURE_CONFIG = {
    'price_columns': ['price', 'original_price', 'discounted_price'],
    'rating_columns': ['rating', 'reviews_count'],
    'categorical_columns': ['brand', 'category', 'specs'],
    'target_column': 'optimal_price'
}

