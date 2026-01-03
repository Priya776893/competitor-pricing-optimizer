"""
Data preprocessing and feature engineering for Competitor Pricing Optimizer
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import FEATURE_CONFIG, DATA_PATHS
from src.utils import (
    calculate_discount_percentage, 
    calculate_demand_proxy,
    load_dataframe,
    save_dataframe
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """Data preprocessing and feature engineering class"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.target_column = FEATURE_CONFIG['target_column']
    
    def load_raw_data(self, filepath: str) -> pd.DataFrame:
        """Load raw data"""
        logger.info(f"Loading raw data from {filepath}")
        df = load_dataframe(filepath)
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the raw data"""
        logger.info("Cleaning data...")
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['name', 'url'], keep='first')
        logger.info(f"Removed {initial_count - len(df)} duplicates")
        
        # Handle missing values
        # Fill missing prices with median
        if 'price' in df.columns:
            df['price'].fillna(df['price'].median(), inplace=True)
        
        # Fill missing ratings with median
        if 'rating' in df.columns:
            df['rating'].fillna(df['rating'].median(), inplace=True)
        
        # Fill missing reviews_count with 0
        if 'reviews_count' in df.columns:
            df['reviews_count'].fillna(0, inplace=True)
        
        # Fill missing brands with 'Unknown'
        if 'brand' in df.columns:
            df['brand'].fillna('Unknown', inplace=True)
        
        # Remove rows with critical missing data
        df = df.dropna(subset=['price', 'name'])
        
        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer new features"""
        logger.info("Engineering features...")
        
        # Calculate discount percentage
        if 'original_price' in df.columns and 'price' in df.columns:
            df['discount_percentage'] = df.apply(
                lambda row: calculate_discount_percentage(
                    row.get('original_price', row['price']),
                    row['price']
                ), axis=1
            )
        else:
            df['discount_percentage'] = 0.0
        
        # Calculate demand proxy
        if 'rating' in df.columns and 'reviews_count' in df.columns:
            df['demand_proxy'] = df.apply(
                lambda row: calculate_demand_proxy(
                    row['rating'],
                    row['reviews_count']
                ), axis=1
            )
        else:
            df['demand_proxy'] = 0.0
        
        # Price per rating (value metric)
        df['price_per_rating'] = df['price'] / (df['rating'] + 0.1)  # Add small value to avoid division by zero
        
        # Log transformations for skewed features
        df['log_price'] = np.log1p(df['price'])
        df['log_demand'] = np.log1p(df['demand_proxy'])
        df['log_reviews'] = np.log1p(df['reviews_count'])
        
        # Calculate optimal price (target variable)
        # Optimal price = competitor average * (1 + demand_factor)
        # This is a simplified heuristic - in practice, this would be learned
        competitor_avg_price = df['price'].mean()
        demand_factor = (df['demand_proxy'] - df['demand_proxy'].min()) / (df['demand_proxy'].max() - df['demand_proxy'].min() + 1e-6)
        df['optimal_price'] = competitor_avg_price * (1 + 0.2 * demand_factor)  # 20% uplift potential
        
        # Calculate price uplift percentage
        df['price_uplift_pct'] = ((df['optimal_price'] - df['price']) / df['price']) * 100
        
        logger.info("Feature engineering complete")
        return df
    
    def encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        logger.info("Encoding categorical variables...")
        
        df_encoded = df.copy()
        
        # Encode brand
        if 'brand' in df.columns:
            if 'brand' not in self.label_encoders:
                self.label_encoders['brand'] = LabelEncoder()
                df_encoded['brand_encoded'] = self.label_encoders['brand'].fit_transform(df['brand'])
            else:
                # Handle unseen brands
                known_brands = set(self.label_encoders['brand'].classes_)
                df_encoded['brand_encoded'] = df['brand'].apply(
                    lambda x: self.label_encoders['brand'].transform([x])[0] 
                    if x in known_brands else 0
                )
        
        # Encode category
        if 'category' in df.columns:
            if 'category' not in self.label_encoders:
                self.label_encoders['category'] = LabelEncoder()
                df_encoded['category_encoded'] = self.label_encoders['category'].fit_transform(df['category'])
            else:
                known_categories = set(self.label_encoders['category'].classes_)
                df_encoded['category_encoded'] = df['category'].apply(
                    lambda x: self.label_encoders['category'].transform([x])[0]
                    if x in known_categories else 0
                )
        
        logger.info("Categorical encoding complete")
        return df_encoded
    
    def prepare_features(self, df: pd.DataFrame, is_training: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling"""
        logger.info("Preparing features for modeling...")
        
        # Select feature columns
        feature_cols = [
            'price', 'rating', 'reviews_count', 'discount_percentage',
            'demand_proxy', 'price_per_rating', 'log_price', 'log_demand',
            'log_reviews'
        ]
        
        # Add encoded categoricals if available
        if 'brand_encoded' in df.columns:
            feature_cols.append('brand_encoded')
        if 'category_encoded' in df.columns:
            feature_cols.append('category_encoded')
        
        # Filter to available columns
        available_features = [col for col in feature_cols if col in df.columns]
        
        X = df[available_features].copy()
        y = df[self.target_column].copy() if self.target_column in df.columns else None
        
        # Handle missing values in features
        X = X.fillna(X.median())
        
        # Scale features if training
        if is_training:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = pd.DataFrame(
                self.scaler.transform(X),
                columns=X.columns,
                index=X.index
            )
        
        self.feature_columns = X_scaled.columns.tolist()
        logger.info(f"Prepared {len(self.feature_columns)} features")
        
        return X_scaled, y
    
    def process(self, input_path: str, output_path: str) -> pd.DataFrame:
        """Complete preprocessing pipeline"""
        logger.info("Starting preprocessing pipeline...")
        
        # Load data
        df = self.load_raw_data(input_path)
        
        # Clean data
        df = self.clean_data(df)
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Encode categoricals
        df = self.encode_categoricals(df)
        
        # Save processed data
        save_dataframe(df, output_path)
        logger.info(f"Processed data saved to {output_path}")
        
        return df


def main():
    """Main preprocessing function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocess product data')
    parser.add_argument('--input', type=str, default='data/raw/products.csv', help='Input file path')
    parser.add_argument('--output', type=str, default='data/processed/products_processed.csv', help='Output file path')
    
    args = parser.parse_args()
    
    preprocessor = DataPreprocessor()
    df = preprocessor.process(args.input, args.output)
    
    logger.info(f"Preprocessing complete! Processed {len(df)} products")
    logger.info(f"\nData summary:")
    logger.info(df.describe())


if __name__ == "__main__":
    main()

