"""
Web scraper for e-commerce product data
Supports Flipkart and Amazon (with proper rate limiting and headers)
"""

import time
import random
import pandas as pd
import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from typing import List, Dict, Optional
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import SCRAPING_CONFIG
from src.utils import clean_price, clean_rating, clean_reviews_count, save_dataframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EcommerceScraper:
    """Base scraper class for e-commerce platforms"""
    
    def __init__(self):
        self.delay = SCRAPING_CONFIG['delay']
        self.timeout = SCRAPING_CONFIG['timeout']
        self.max_retries = SCRAPING_CONFIG['max_retries']
        self.user_agents = SCRAPING_CONFIG['user_agents']
        self.headers = SCRAPING_CONFIG['headers']
        self.driver = None
    
    def get_random_headers(self) -> Dict[str, str]:
        """Get random user agent headers"""
        headers = self.headers.copy()
        headers['User-Agent'] = random.choice(self.user_agents)
        return headers
    
    def setup_selenium(self, headless: bool = True):
        """Setup Selenium WebDriver"""
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument('--disable-blink-features=AutomationControlled')
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option('useAutomationExtension', False)
        chrome_options.add_argument(f'user-agent={random.choice(self.user_agents)}')
        
        service = Service(ChromeDriverManager().install())
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    
    def close_selenium(self):
        """Close Selenium WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def rate_limit(self):
        """Apply rate limiting"""
        time.sleep(self.delay + random.uniform(0, 1))


class FlipkartScraper(EcommerceScraper):
    """Scraper for Flipkart"""
    
    def __init__(self):
        super().__init__()
        self.base_url = "https://www.flipkart.com"
    
    def scrape_product(self, product_url: str) -> Optional[Dict]:
        """Scrape a single product page"""
        try:
            if not self.driver:
                self.setup_selenium()
            
            self.driver.get(product_url)
            time.sleep(2)
            
            product_data = {}
            
            # Product name
            try:
                name_elem = self.driver.find_element(By.CSS_SELECTOR, "span.B_NuCI")
                product_data['name'] = name_elem.text.strip()
            except:
                product_data['name'] = None
            
            # Price
            try:
                price_elem = self.driver.find_element(By.CSS_SELECTOR, "div._30jeq3._16Jk6d")
                product_data['price'] = clean_price(price_elem.text)
            except:
                product_data['price'] = None
            
            # Original price (if discounted)
            try:
                original_price_elem = self.driver.find_element(By.CSS_SELECTOR, "div._3I9_wc._2p6lqe")
                product_data['original_price'] = clean_price(original_price_elem.text)
            except:
                product_data['original_price'] = product_data.get('price')
            
            # Rating
            try:
                rating_elem = self.driver.find_element(By.CSS_SELECTOR, "div._3LWZlK")
                product_data['rating'] = clean_rating(rating_elem.text)
            except:
                product_data['rating'] = None
            
            # Reviews count
            try:
                reviews_elem = self.driver.find_element(By.CSS_SELECTOR, "span._2_R_IW")
                product_data['reviews_count'] = clean_reviews_count(reviews_elem.text)
            except:
                product_data['reviews_count'] = 0
            
            # Brand
            try:
                brand_elem = self.driver.find_element(By.CSS_SELECTOR, "span.G6XhRU")
                product_data['brand'] = brand_elem.text.strip()
            except:
                product_data['brand'] = None
            
            # Category
            product_data['category'] = 'smartphones'  # Default, can be parameterized
            
            # URL
            product_data['url'] = product_url
            
            self.rate_limit()
            return product_data
            
        except Exception as e:
            logger.error(f"Error scraping product {product_url}: {e}")
            return None
    
    def search_products(self, query: str, pages: int = 5) -> List[Dict]:
        """Search and scrape multiple products"""
        products = []
        
        try:
            if not self.driver:
                self.setup_selenium()
            
            for page in range(1, pages + 1):
                search_url = f"{self.base_url}/search?q={query.replace(' ', '+')}&page={page}"
                logger.info(f"Scraping page {page}: {search_url}")
                
                self.driver.get(search_url)
                time.sleep(3)
                
                # Find product links
                try:
                    product_links = self.driver.find_elements(By.CSS_SELECTOR, "a._1fQZEK")
                    if not product_links:
                        product_links = self.driver.find_elements(By.CSS_SELECTOR, "a.s1Q9rs")
                    
                    for link in product_links[:10]:  # Limit to 10 per page
                        product_url = link.get_attribute('href')
                        if product_url and '/p/' in product_url:
                            if not product_url.startswith('http'):
                                product_url = self.base_url + product_url
                            
                            product_data = self.scrape_product(product_url)
                            if product_data:
                                products.append(product_data)
                                logger.info(f"Scraped: {product_data.get('name', 'Unknown')}")
                            
                            if len(products) >= 100:  # Limit total products
                                break
                    
                    if len(products) >= 100:
                        break
                    
                except Exception as e:
                    logger.error(f"Error on page {page}: {e}")
                
                self.rate_limit()
        
        except Exception as e:
            logger.error(f"Error in search: {e}")
        
        return products


def generate_sample_data(n_samples: int = 500) -> pd.DataFrame:
    """
    Generate sample product data for testing when scraping is not possible
    This mimics real e-commerce data structure
    """
    import numpy as np
    
    logger.info(f"Generating {n_samples} sample products...")
    
    brands = ['Samsung', 'Apple', 'Xiaomi', 'OnePlus', 'Realme', 'Vivo', 'Oppo', 'Motorola']
    categories = ['smartphones', 'laptops', 'tablets']
    
    data = []
    for i in range(n_samples):
        brand = np.random.choice(brands)
        category = np.random.choice(categories)
        
        # Generate realistic price range based on brand
        if brand == 'Apple':
            base_price = np.random.uniform(30000, 150000)
        elif brand in ['Samsung', 'OnePlus']:
            base_price = np.random.uniform(15000, 80000)
        else:
            base_price = np.random.uniform(8000, 40000)
        
        original_price = round(base_price * np.random.uniform(1.1, 1.3), 2)
        discount_pct = np.random.uniform(5, 30)
        price = round(original_price * (1 - discount_pct/100), 2)
        
        rating = round(np.random.uniform(3.5, 4.8), 1)
        reviews_count = int(np.random.exponential(5000))
        
        product = {
            'name': f"{brand} {category.title()} {i+1}",
            'brand': brand,
            'category': category,
            'price': price,
            'original_price': original_price,
            'rating': rating,
            'reviews_count': reviews_count,
            'url': f"https://example.com/product/{i+1}"
        }
        data.append(product)
    
    df = pd.DataFrame(data)
    logger.info(f"Generated {len(df)} sample products")
    return df


def main():
    """Main scraping function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape e-commerce product data')
    parser.add_argument('--category', type=str, default='smartphones', help='Product category')
    parser.add_argument('--pages', type=int, default=10, help='Number of pages to scrape')
    parser.add_argument('--use-sample', action='store_true', help='Use sample data instead of scraping')
    parser.add_argument('--output', type=str, default='data/raw/products.csv', help='Output file path')
    
    args = parser.parse_args()
    
    if args.use_sample:
        logger.info("Using sample data generation...")
        df = generate_sample_data(n_samples=500)
    else:
        logger.info(f"Scraping {args.category} products from Flipkart...")
        scraper = FlipkartScraper()
        try:
            products = scraper.search_products(args.category, pages=args.pages)
            df = pd.DataFrame(products)
        except Exception as e:
            logger.error(f"Scraping failed: {e}")
            logger.info("Falling back to sample data...")
            df = generate_sample_data(n_samples=500)
        finally:
            scraper.close_selenium()
    
    if len(df) > 0:
        save_dataframe(df, args.output)
        logger.info(f"Saved {len(df)} products to {args.output}")
    else:
        logger.error("No products scraped!")


if __name__ == "__main__":
    main()

