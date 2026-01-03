"""
Training script for Competitor Pricing Optimizer
Trains clustering and prediction models
"""

import pandas as pd
import numpy as np
import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import DataPreprocessor
from src.models import PricingOptimizer
from config import DATA_PATHS

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Main training function"""
    logger.info("="*60)
    logger.info("Competitor Pricing Optimizer - Model Training")
    logger.info("="*60)
    
    # Step 1: Preprocess data
    logger.info("\n[Step 1/3] Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    input_path = 'data/raw/products.csv'
    output_path = 'data/processed/products_processed.csv'
    
    try:
        df_processed = preprocessor.process(input_path, output_path)
        logger.info(f"✓ Preprocessing complete: {len(df_processed)} products")
    except FileNotFoundError:
        logger.error(f"Raw data file not found: {input_path}")
        logger.info("Please run the scraper first: python src/scraper.py")
        return
    
    # Step 2: Prepare features
    logger.info("\n[Step 2/3] Preparing features...")
    X, y = preprocessor.prepare_features(df_processed, is_training=True)
    logger.info(f"✓ Features prepared: {X.shape[1]} features, {len(X)} samples")
    
    # Step 3: Train models
    logger.info("\n[Step 3/3] Training models...")
    
    optimizer = PricingOptimizer()
    
    # Train clustering
    logger.info("\n--- Training Clustering Model ---")
    cluster_labels = optimizer.train_clustering(X, n_clusters=5)
    df_processed['cluster'] = cluster_labels
    
    # Analyze clusters
    logger.info("\nCluster Analysis:")
    cluster_summary = df_processed.groupby('cluster').agg({
        'price': ['mean', 'std', 'count'],
        'rating': 'mean',
        'demand_proxy': 'mean',
        'brand': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
    }).round(2)
    logger.info(f"\n{cluster_summary}")
    
    # Train prediction model
    logger.info("\n--- Training Price Prediction Model ---")
    metrics = optimizer.train_prediction(X, y, tune_hyperparameters=False)
    
    # Step 4: Save models
    logger.info("\n--- Saving Models ---")
    optimizer.save_models()
    
    # Step 5: Save preprocessor (scaler and encoders)
    import joblib
    preprocessor_path = f"{DATA_PATHS['models']}/preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    logger.info(f"✓ Preprocessor saved to {preprocessor_path}")
    
    # Step 6: Save processed data with clusters
    df_processed.to_csv(output_path, index=False)
    logger.info(f"✓ Processed data with clusters saved to {output_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete! ✓")
    logger.info("="*60)
    logger.info(f"\nModel Performance Summary:")
    logger.info(f"  Test RMSE: {metrics['test_rmse']:.2f} ({metrics['test_rmse_pct']:.2f}%)")
    logger.info(f"  Test R²: {metrics['test_r2']:.4f}")
    logger.info(f"  Number of Clusters: {optimizer.n_clusters}")
    logger.info(f"\nModels saved to: {DATA_PATHS['models']}/")
    logger.info(f"Ready to use in Streamlit app: streamlit run app.py")


if __name__ == "__main__":
    main()

