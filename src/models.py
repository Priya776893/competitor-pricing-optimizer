"""
Machine Learning models for Competitor Pricing Optimizer
Includes K-means clustering and XGBoost price prediction
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, silhouette_score
import xgboost as xgb
import joblib
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Tuple, Dict
from config import MODEL_CONFIG, DATA_PATHS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PricingOptimizer:
    """Main ML model class for pricing optimization"""
    
    def __init__(self):
        self.clustering_model = None
        self.prediction_model = None
        self.n_clusters = MODEL_CONFIG['n_clusters']
        self.random_state = MODEL_CONFIG['random_state']
        self.feature_columns = []
        self.cluster_labels = None
    
    def find_optimal_clusters(self, X: pd.DataFrame, max_k: int = 10) -> int:
        """
        Find optimal number of clusters using elbow method and silhouette score
        
        Args:
            X: Feature matrix
            max_k: Maximum number of clusters to test
        
        Returns:
            Optimal number of clusters
        """
        logger.info("Finding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        k_range = range(2, min(max_k + 1, len(X) // 10))
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(X)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(X, labels))
            logger.info(f"K={k}: Inertia={kmeans.inertia_:.2f}, Silhouette={silhouette_scores[-1]:.3f}")
        
        # Choose k with highest silhouette score
        optimal_k = k_range[np.argmax(silhouette_scores)]
        logger.info(f"Optimal number of clusters: {optimal_k}")
        
        return optimal_k
    
    def train_clustering(self, X: pd.DataFrame, n_clusters: int = None) -> np.ndarray:
        """
        Train K-means clustering model
        
        Args:
            X: Feature matrix
            n_clusters: Number of clusters (if None, will find optimal)
        
        Returns:
            Cluster labels
        """
        logger.info("Training clustering model...")
        
        if n_clusters is None:
            n_clusters = self.find_optimal_clusters(X)
        
        self.n_clusters = n_clusters
        self.clustering_model = KMeans(
            n_clusters=n_clusters,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.cluster_labels = self.clustering_model.fit_predict(X)
        
        # Calculate silhouette score
        silhouette = silhouette_score(X, self.cluster_labels)
        logger.info(f"Clustering complete! Silhouette score: {silhouette:.3f}")
        
        return self.cluster_labels
    
    def train_prediction(self, X: pd.DataFrame, y: pd.Series, tune_hyperparameters: bool = False) -> Dict:
        """
        Train XGBoost price prediction model
        
        Args:
            X: Feature matrix
            y: Target variable (optimal_price)
            tune_hyperparameters: Whether to perform hyperparameter tuning
        
        Returns:
            Dictionary with model performance metrics
        """
        logger.info("Training price prediction model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=MODEL_CONFIG['test_size'], random_state=self.random_state
        )
        
        logger.info(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")
        
        # Base model
        base_params = MODEL_CONFIG['xgboost_params'].copy()
        
        if tune_hyperparameters:
            logger.info("Performing hyperparameter tuning...")
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1, 0.15],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            xgb_model = xgb.XGBRegressor(random_state=self.random_state)
            grid_search = GridSearchCV(
                xgb_model, param_grid, cv=5, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
            grid_search.fit(X_train, y_train)
            self.prediction_model = grid_search.best_estimator_
            logger.info(f"Best parameters: {grid_search.best_params_}")
        else:
            self.prediction_model = xgb.XGBRegressor(**base_params)
            self.prediction_model.fit(X_train, y_train)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Evaluate model
        y_train_pred = self.prediction_model.predict(X_train)
        y_test_pred = self.prediction_model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        
        # Calculate percentage errors
        train_rmse_pct = (train_rmse / y_train.mean()) * 100
        test_rmse_pct = (test_rmse / y_test.mean()) * 100
        
        metrics = {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_rmse_pct': train_rmse_pct,
            'test_rmse_pct': test_rmse_pct,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_mae': train_mae,
            'test_mae': test_mae
        }
        
        logger.info("\n" + "="*50)
        logger.info("Model Performance Metrics:")
        logger.info(f"Train RMSE: {train_rmse:.2f} ({train_rmse_pct:.2f}%)")
        logger.info(f"Test RMSE: {test_rmse:.2f} ({test_rmse_pct:.2f}%)")
        logger.info(f"Train R²: {train_r2:.4f}")
        logger.info(f"Test R²: {test_r2:.4f}")
        logger.info(f"Train MAE: {train_mae:.2f}")
        logger.info(f"Test MAE: {test_mae:.2f}")
        logger.info("="*50)
        
        # Check if model meets requirements
        if test_rmse_pct < 10 and test_r2 > 0.85:
            logger.info("✅ Model meets requirements (RMSE <10%, R² >0.85)")
        else:
            logger.warning("⚠️ Model does not meet all requirements")
        
        return metrics
    
    def predict_price(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict optimal price for given features
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted optimal prices
        """
        if self.prediction_model is None:
            raise ValueError("Model not trained! Call train_prediction() first.")
        
        return self.prediction_model.predict(X)
    
    def predict_cluster(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict cluster for given features
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted cluster labels
        """
        if self.clustering_model is None:
            raise ValueError("Clustering model not trained! Call train_clustering() first.")
        
        return self.clustering_model.predict(X)
    
    def save_models(self, model_dir: str = DATA_PATHS['models']):
        """Save trained models"""
        import os
        os.makedirs(model_dir, exist_ok=True)
        
        if self.clustering_model:
            clustering_path = f"{model_dir}/clustering_model.pkl"
            joblib.dump(self.clustering_model, clustering_path)
            logger.info(f"Clustering model saved to {clustering_path}")
        
        if self.prediction_model:
            prediction_path = f"{model_dir}/prediction_model.pkl"
            joblib.dump(self.prediction_model, prediction_path)
            logger.info(f"Prediction model saved to {prediction_path}")
        
        # Save feature columns
        if self.feature_columns:
            import json
            features_path = f"{model_dir}/feature_columns.json"
            with open(features_path, 'w') as f:
                json.dump(self.feature_columns, f)
            logger.info(f"Feature columns saved to {features_path}")
    
    def load_models(self, model_dir: str = DATA_PATHS['models']):
        """Load trained models"""
        import os
        import json
        
        clustering_path = f"{model_dir}/clustering_model.pkl"
        prediction_path = f"{model_dir}/prediction_model.pkl"
        features_path = f"{model_dir}/feature_columns.json"
        
        if os.path.exists(clustering_path):
            self.clustering_model = joblib.load(clustering_path)
            logger.info(f"Clustering model loaded from {clustering_path}")
        
        if os.path.exists(prediction_path):
            self.prediction_model = joblib.load(prediction_path)
            logger.info(f"Prediction model loaded from {prediction_path}")
        
        if os.path.exists(features_path):
            with open(features_path, 'r') as f:
                self.feature_columns = json.load(f)
            logger.info(f"Feature columns loaded from {features_path}")

