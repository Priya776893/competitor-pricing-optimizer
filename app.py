"""
Streamlit Dashboard for Competitor Pricing Optimizer
Interactive visualization and price prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.preprocessing import DataPreprocessor
from src.models import PricingOptimizer
from config import DATA_PATHS

# Page configuration
st.set_page_config(
    page_title="Competitor Pricing Optimizer",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .prediction-box {
        background-color: #e8f5e9;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False


@st.cache_data(ttl=60)  # Cache for 60 seconds
def load_data():
    """Load processed data, generate if not exists"""
    data_path = 'data/processed/products_processed.csv'
    if os.path.exists(data_path):
        try:
            return pd.read_csv(data_path)
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    else:
        # Try to generate data if not exists (for deployment)
        try:
            import subprocess
            import sys
            # Generate sample data
            raw_path = 'data/raw/products.csv'
            if not os.path.exists(raw_path):
                subprocess.run([sys.executable, 'src/scraper.py', '--use-sample', '--output', raw_path], 
                             check=False, capture_output=True)
            # Process data
            if os.path.exists(raw_path):
                from src.preprocessing import DataPreprocessor
                preprocessor = DataPreprocessor()
                preprocessor.process(raw_path, data_path)
                if os.path.exists(data_path):
                    return pd.read_csv(data_path)
        except Exception as e:
            st.warning(f"Could not auto-generate data: {e}")
    return None


@st.cache_resource(ttl=60)  # Cache for 60 seconds
def load_models():
    """Load trained models, train if not exists"""
    optimizer = PricingOptimizer()
    try:
        optimizer.load_models()
        if optimizer.prediction_model is None:
            # Try to train models if they don't exist (for deployment)
            try:
                import subprocess
                import sys
                subprocess.run([sys.executable, 'train.py'], 
                             check=False, capture_output=True, timeout=300)
                optimizer.load_models()
            except Exception as e:
                return None
        if optimizer.prediction_model is None:
            return None
        return optimizer
    except Exception as e:
        return None


def main():
    """Main Streamlit app"""
    
    # Header
    st.markdown('<h1 class="main-header">🎯 Competitor Pricing Optimizer</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Force reload button
        if st.button("🔄 Refresh Data & Models", use_container_width=True, type="primary"):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()
        
        st.markdown("---")
        
        # Prediction inputs
        st.header("🔮 Price Prediction")
        
        price = st.number_input("Current Price (₹)", min_value=0.0, value=15000.0, step=100.0)
        rating = st.slider("Rating", min_value=0.0, max_value=5.0, value=4.0, step=0.1)
        reviews_count = st.number_input("Reviews Count", min_value=0, value=1000, step=100)
        discount_pct = st.slider("Discount %", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
        
        brands = ['Samsung', 'Apple', 'Xiaomi', 'OnePlus', 'Realme', 'Vivo', 'Oppo', 'Motorola']
        brand = st.selectbox("Brand", brands)
        category = st.selectbox("Category", ['smartphones', 'laptops', 'tablets'])
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Dashboard", "🔍 Market Analysis", "🎯 Price Prediction", "📈 Insights"])
    
    # Load data and models
    df = load_data()
    optimizer = load_models()
    
    if df is None:
        st.warning("⚠️ No data found! Please run the scraper and preprocessing first.")
        st.info("Run: `python src/scraper.py --use-sample` then `python train.py`")
        return
    
    if optimizer is None or optimizer.prediction_model is None:
        st.warning("⚠️ Models not loaded! Please train models first.")
        st.info("Run: `python train.py`")
        return
    
    # Tab 1: Dashboard
    with tab1:
        st.header("📊 Product Dashboard")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Products", len(df))
        
        with col2:
            avg_price = df['price'].mean()
            st.metric("Average Price", f"₹{avg_price:,.0f}")
        
        with col3:
            avg_rating = df['rating'].mean()
            st.metric("Average Rating", f"{avg_rating:.2f}")
        
        with col4:
            total_reviews = df['reviews_count'].sum()
            st.metric("Total Reviews", f"{total_reviews:,}")
        
        st.markdown("---")
        
        # Price distribution
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price Distribution")
            fig_price = px.histogram(
                df, x='price', nbins=50,
                title="Product Price Distribution",
                labels={'price': 'Price (₹)', 'count': 'Count'}
            )
            fig_price.update_layout(height=400)
            st.plotly_chart(fig_price, use_container_width=True)
        
        with col2:
            st.subheader("Rating Distribution")
            fig_rating = px.histogram(
                df, x='rating', nbins=20,
                title="Product Rating Distribution",
                labels={'rating': 'Rating', 'count': 'Count'}
            )
            fig_rating.update_layout(height=400)
            st.plotly_chart(fig_rating, use_container_width=True)
        
        # Top products
        st.subheader("🏆 Top Products by Demand")
        top_products = df.nlargest(10, 'demand_proxy')[['name', 'brand', 'price', 'rating', 'reviews_count', 'demand_proxy']]
        st.dataframe(top_products, use_container_width=True)
    
    # Tab 2: Market Analysis
    with tab2:
        st.header("🔍 Market Segmentation Analysis")
        
        if 'cluster' not in df.columns:
            st.warning("Clusters not found. Please retrain models.")
            return
        
        # Cluster visualization
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Price vs Rating by Cluster")
            fig_cluster = px.scatter(
                df, x='price', y='rating', color='cluster',
                size='demand_proxy', hover_data=['name', 'brand'],
                title="Market Segmentation",
                labels={'price': 'Price (₹)', 'rating': 'Rating', 'cluster': 'Segment'}
            )
            fig_cluster.update_layout(height=500)
            st.plotly_chart(fig_cluster, use_container_width=True)
        
        with col2:
            st.subheader("Cluster Statistics")
            cluster_stats = df.groupby('cluster').agg({
                'price': ['mean', 'count'],
                'rating': 'mean',
                'demand_proxy': 'mean',
                'brand': lambda x: x.mode()[0] if len(x.mode()) > 0 else 'Unknown'
            }).round(2)
            cluster_stats.columns = ['Avg Price', 'Count', 'Avg Rating', 'Avg Demand', 'Top Brand']
            st.dataframe(cluster_stats, use_container_width=True)
        
        # 3D visualization
        st.subheader("3D Market Positioning")
        fig_3d = px.scatter_3d(
            df, x='price', y='rating', z='demand_proxy',
            color='cluster', size='reviews_count',
            hover_data=['name', 'brand'],
            title="3D Market Positioning",
            labels={'price': 'Price (₹)', 'rating': 'Rating', 'demand_proxy': 'Demand', 'cluster': 'Segment'}
        )
        fig_3d.update_layout(height=600)
        st.plotly_chart(fig_3d, use_container_width=True)
    
    # Tab 3: Price Prediction
    with tab3:
        st.header("🎯 Optimal Price Prediction")
        
        # Load preprocessor (with fitted scaler)
        preprocessor_path = f"{DATA_PATHS['models']}/preprocessor.pkl"
        if not os.path.exists(preprocessor_path):
            st.error("Preprocessor not found! Please retrain models: `python train.py`")
            return
        
        preprocessor = joblib.load(preprocessor_path)
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'price': [price],
            'rating': [rating],
            'reviews_count': [reviews_count],
            'discount_percentage': [discount_pct],
            'brand': [brand],
            'category': [category]
        })
        
        # Calculate derived features
        input_data['original_price'] = price / (1 - discount_pct/100) if discount_pct > 0 else price
        input_data['demand_proxy'] = rating * reviews_count
        input_data['price_per_rating'] = price / (rating + 0.1)
        input_data['log_price'] = np.log1p(price)
        input_data['log_demand'] = np.log1p(input_data['demand_proxy'])
        input_data['log_reviews'] = np.log1p(reviews_count)
        
        # Encode categoricals using the fitted encoders
        if 'brand' in preprocessor.label_encoders:
            try:
                input_data['brand_encoded'] = preprocessor.label_encoders['brand'].transform([brand])[0]
            except ValueError:
                # Brand not in training data, use 0
                input_data['brand_encoded'] = 0
        else:
            input_data['brand_encoded'] = 0
        
        if 'category' in preprocessor.label_encoders:
            try:
                input_data['category_encoded'] = preprocessor.label_encoders['category'].transform([category])[0]
            except ValueError:
                # Category not in training data, use 0
                input_data['category_encoded'] = 0
        else:
            input_data['category_encoded'] = 0
        
        # Prepare features
        feature_cols = optimizer.feature_columns
        X_input = input_data[feature_cols].fillna(0)
        
        # Scale features using fitted scaler
        X_scaled = pd.DataFrame(
            preprocessor.scaler.transform(X_input),
            columns=feature_cols
        )
        
        # Predict
        predicted_price = optimizer.predict_price(X_scaled)[0]
        predicted_cluster = optimizer.predict_cluster(X_scaled)[0]
        
        # Calculate uplift
        price_difference = predicted_price - price
        uplift_pct = (price_difference / price) * 100
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.metric("Current Price", f"₹{price:,.2f}")
            st.metric("Optimal Price", f"₹{predicted_price:,.2f}")
            st.metric("Price Difference", f"₹{price_difference:,.2f}")
            st.metric("Uplift Potential", f"{uplift_pct:.1f}%")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.subheader("Market Segment")
            st.info(f"**Segment {predicted_cluster}**")
            
            # Get segment characteristics
            if 'cluster' in df.columns:
                segment_data = df[df['cluster'] == predicted_cluster]
                if len(segment_data) > 0:
                    st.write(f"**Segment Average Price:** ₹{segment_data['price'].mean():,.0f}")
                    st.write(f"**Segment Average Rating:** {segment_data['rating'].mean():.2f}")
                    st.write(f"**Products in Segment:** {len(segment_data)}")
        
        # Visualization
        st.subheader("Price Comparison")
        comparison_df = pd.DataFrame({
            'Type': ['Current Price', 'Optimal Price'],
            'Price': [price, predicted_price]
        })
        
        fig_comparison = px.bar(
            comparison_df, x='Type', y='Price',
            title="Current vs Optimal Price",
            labels={'Price': 'Price (₹)'}
        )
        fig_comparison.update_layout(height=400)
        st.plotly_chart(fig_comparison, use_container_width=True)
    
    # Tab 4: Insights
    with tab4:
        st.header("📈 Business Insights")
        
        # Key metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_uplift = df['price_uplift_pct'].mean() if 'price_uplift_pct' in df.columns else 0
            st.metric("Average Uplift Potential", f"{avg_uplift:.1f}%")
        
        with col2:
            high_uplift = len(df[df['price_uplift_pct'] > 15]) if 'price_uplift_pct' in df.columns else 0
            st.metric("High Uplift Products (>15%)", high_uplift)
        
        with col3:
            total_potential = df['price_uplift_pct'].sum() if 'price_uplift_pct' in df.columns else 0
            st.metric("Total Uplift Potential", f"{total_potential:.1f}%")
        
        # Brand analysis
        st.subheader("Brand Performance")
        brand_analysis = df.groupby('brand').agg({
            'price': 'mean',
            'rating': 'mean',
            'demand_proxy': 'mean',
            'name': 'count'
        }).round(2)
        brand_analysis.columns = ['Avg Price', 'Avg Rating', 'Avg Demand', 'Product Count']
        brand_analysis = brand_analysis.sort_values('Avg Demand', ascending=False)
        st.dataframe(brand_analysis, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        if 'price_uplift_pct' in df.columns:
            top_opportunities = df.nlargest(5, 'price_uplift_pct')[
                ['name', 'brand', 'price', 'price_uplift_pct', 'rating']
            ]
            st.write("**Top Pricing Opportunities:**")
            st.dataframe(top_opportunities, use_container_width=True)


if __name__ == "__main__":
    main()

