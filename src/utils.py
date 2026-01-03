"""
Utility functions for the Competitor Pricing Optimizer
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def clean_price(price_str: str) -> float:
    """
    Clean and convert price string to float
    
    Args:
        price_str: Price string (e.g., "₹15,999" or "$199.99")
    
    Returns:
        Cleaned price as float
    """
    if pd.isna(price_str) or price_str == '':
        return np.nan
    
    # Remove currency symbols and commas
    price_str = str(price_str)
    price_str = re.sub(r'[₹$€£, ]', '', price_str)
    
    # Extract numbers
    numbers = re.findall(r'\d+\.?\d*', price_str)
    if numbers:
        return float(numbers[0])
    return np.nan


def clean_rating(rating_str: str) -> float:
    """
    Clean and convert rating string to float
    
    Args:
        rating_str: Rating string (e.g., "4.5 out of 5")
    
    Returns:
        Cleaned rating as float
    """
    if pd.isna(rating_str) or rating_str == '':
        return np.nan
    
    rating_str = str(rating_str)
    # Extract first number
    numbers = re.findall(r'\d+\.?\d*', rating_str)
    if numbers:
        rating = float(numbers[0])
        # Normalize to 0-5 scale if needed
        if rating > 5:
            rating = rating / 10
        return rating
    return np.nan


def clean_reviews_count(reviews_str: str) -> int:
    """
    Clean and convert reviews count string to int
    
    Args:
        reviews_str: Reviews string (e.g., "1,234 reviews" or "2.5K")
    
    Returns:
        Cleaned reviews count as int
    """
    if pd.isna(reviews_str) or reviews_str == '':
        return 0
    
    reviews_str = str(reviews_str).lower()
    
    # Remove common words
    reviews_str = re.sub(r'reviews?|ratings?|and|more', '', reviews_str)
    reviews_str = reviews_str.strip()
    
    # Handle K, M suffixes
    if 'k' in reviews_str:
        number = re.findall(r'\d+\.?\d*', reviews_str)
        if number:
            return int(float(number[0]) * 1000)
    elif 'm' in reviews_str:
        number = re.findall(r'\d+\.?\d*', reviews_str)
        if number:
            return int(float(number[0]) * 1000000)
    else:
        # Remove commas and extract number
        number = re.sub(r'[, ]', '', reviews_str)
        numbers = re.findall(r'\d+', number)
        if numbers:
            return int(numbers[0])
    
    return 0


def calculate_discount_percentage(original_price: float, discounted_price: float) -> float:
    """
    Calculate discount percentage
    
    Args:
        original_price: Original price
        discounted_price: Discounted price
    
    Returns:
        Discount percentage
    """
    if pd.isna(original_price) or pd.isna(discounted_price) or original_price == 0:
        return 0.0
    
    discount = ((original_price - discounted_price) / original_price) * 100
    return round(discount, 2)


def calculate_demand_proxy(rating: float, reviews_count: int) -> float:
    """
    Calculate demand proxy (rating * reviews_count)
    
    Args:
        rating: Product rating
        reviews_count: Number of reviews
    
    Returns:
        Demand proxy score
    """
    if pd.isna(rating) or pd.isna(reviews_count):
        return 0.0
    
    return rating * reviews_count


def save_dataframe(df: pd.DataFrame, filepath: str, index: bool = False):
    """
    Save dataframe to CSV with logging
    
    Args:
        df: DataFrame to save
        filepath: Output filepath
        index: Whether to save index
    """
    try:
        df.to_csv(filepath, index=index)
        logger.info(f"Data saved to {filepath} ({len(df)} rows)")
    except Exception as e:
        logger.error(f"Error saving data to {filepath}: {e}")
        raise


def load_dataframe(filepath: str) -> pd.DataFrame:
    """
    Load dataframe from CSV with logging
    
    Args:
        filepath: Input filepath
    
    Returns:
        Loaded DataFrame
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Data loaded from {filepath} ({len(df)} rows)")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {filepath}: {e}")
        raise

