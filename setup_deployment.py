"""
Setup script to prepare the app for deployment
Generates data and models if they don't exist
"""

import os
import sys
import subprocess
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def check_and_setup():
    """Check if data and models exist, create if not"""
    
    # Check for raw data
    raw_data_path = 'data/raw/products.csv'
    if not os.path.exists(raw_data_path):
        logger.info("Raw data not found. Generating sample data...")
        try:
            subprocess.run([sys.executable, 'src/scraper.py', '--use-sample', '--output', raw_data_path], check=True)
            logger.info("✓ Sample data generated")
        except Exception as e:
            logger.error(f"Error generating data: {e}")
            return False
    
    # Check for processed data
    processed_data_path = 'data/processed/products_processed.csv'
    if not os.path.exists(processed_data_path):
        logger.info("Processed data not found. Running preprocessing...")
        try:
            from src.preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor()
            preprocessor.process(raw_data_path, processed_data_path)
            logger.info("✓ Data processed")
        except Exception as e:
            logger.error(f"Error processing data: {e}")
            return False
    
    # Check for models
    model_files = [
        'models/prediction_model.pkl',
        'models/clustering_model.pkl',
        'models/preprocessor.pkl',
        'models/feature_columns.json'
    ]
    
    missing_models = [f for f in model_files if not os.path.exists(f)]
    
    if missing_models:
        logger.info("Models not found. Training models...")
        try:
            subprocess.run([sys.executable, 'train.py'], check=True)
            logger.info("✓ Models trained")
        except Exception as e:
            logger.error(f"Error training models: {e}")
            return False
    
    logger.info("✓ Setup complete! App is ready to run.")
    return True

if __name__ == "__main__":
    success = check_and_setup()
    sys.exit(0 if success else 1)

