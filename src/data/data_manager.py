"""
Data Manager - Handles market data retrieval and caching
"""

import pandas as pd
import numpy as np
import yfinance as yf
from vnstock import Quote
import os
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
from loguru import logger

from ..utils.config_manager import ConfigManager


class DataManager:
    """Manages market data retrieval and caching"""
    
    def __init__(self, config: ConfigManager):
        """Initialize data manager"""
        self.config = config
        self.cache_dir = config.get("data.cache_dir", "data/cache")
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Ensure cache directory exists"""
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_historical_data(self, symbols: List[str], start_date: str, 
                          end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Get historical market data for multiple symbols"""
        logger.info(f"Fetching historical data for {symbols} from {start_date} to {end_date}")
        
        # Check cache first
        cache_key = f"{'_'.join(symbols)}_{start_date}_{end_date}_{interval}.pkl"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if self.config.get("data.cache_data", True) and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info("Data loaded from cache")
                return data
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        # Fetch data from source
        data_source = self.config.get("data.source", "yfinance")
        
        if data_source == "yfinance":
            data = self._fetch_yfinance_data(symbols, start_date, end_date, interval)
        # elif data_source == "alpha_vantage":
        #     data = self._fetch_alpha_vantage_data(symbols, start_date, end_date, interval)
        elif data_source == "vnstock":
            data = self._fetch_vnstock_data(symbols, start_date, end_date, interval)
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
        
        # Cache the data
        if self.config.get("data.cache_data", True):
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.info("Data cached successfully")
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
        
        return data
    
    def _fetch_yfinance_data(self, symbols: List[str], start_date: str, 
                           end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance"""
        try:
            # Download data for all symbols
            tickers = yf.Tickers(' '.join(symbols))
            # Only use start and end dates, no period parameter
            data = tickers.history(start=start_date, end=end_date, interval=interval)
            
            # Extract close prices for all symbols
            close_data = {}
            for symbol in symbols:
                if f"{symbol} Close" in data.columns:
                    close_data[symbol] = data[f"{symbol} Close"]
                else:
                    logger.warning(f"No data found for {symbol}")
                    close_data[symbol] = pd.Series(index=data.index, dtype=float)
            
            return pd.DataFrame(close_data)
            
        except Exception as e:
            logger.error(f"Error fetching data from Yahoo Finance: {e}")
            raise
    
    def _fetch_vnstock_data(self, symbols: List[str], start_date: str, 
                           end_date: str, interval: str) -> pd.DataFrame:
        """Fetch data from VNStock"""
        try:
            # Fetch data for each symbol
            all_data = {}
            for symbol in symbols:
                try:
                    data = Quote(symbol=symbol, source='tcbs').history(start=start_date, end=end_date, period=interval)
                    if not data.empty:
                        # Extract close prices
                        all_data[symbol] = data['close']
                    else:
                        logger.warning(f"No data found for {symbol}")
                        all_data[symbol] = pd.Series(dtype=float)
                except Exception as e:
                    logger.error(f"Error fetching data for {symbol}: {e}")
                    all_data[symbol] = pd.Series(dtype=float)
            
            return pd.DataFrame(all_data)
        except Exception as e:
            logger.error(f"Error fetching data from VNStock: {e}")
            raise
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a symbol"""
        try:
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            ticker = Quote(symbol=symbol, source='tcbs')
            latest_data = ticker.history(start=start_date, end=end_date, period="1d")
            if not latest_data.empty:
                return latest_data['close'].iloc[-1]
            return None
        except Exception as e:
            logger.error(f"Error fetching latest price for {symbol}: {e}")
            return None
    
    # def get_real_time_data(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
    #     """Get real-time market data"""
    #     real_time_data = {}
        
    #     for symbol in symbols:
    #         try:
    #             ticker = yf.Ticker(symbol)
    #             info = ticker.info
                
    #             real_time_data[symbol] = {
    #                 'price': info.get('regularMarketPrice', 0),
    #                 'volume': info.get('volume', 0),
    #                 'market_cap': info.get('marketCap', 0),
    #                 'pe_ratio': info.get('trailingPE', 0),
    #                 'dividend_yield': info.get('dividendYield', 0),
    #                 'timestamp': datetime.now()
    #             }
    #         except Exception as e:
    #             logger.error(f"Error fetching real-time data for {symbol}: {e}")
    #             real_time_data[symbol] = {
    #                 'price': 0,
    #                 'volume': 0,
    #                 'market_cap': 0,
    #                 'pe_ratio': 0,
    #                 'dividend_yield': 0,
    #                 'timestamp': datetime.now()
    #             }
        
    #     return real_time_data

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate data quality"""
        if data.empty:
            logger.error("Data is empty")
            return False
        
        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_pct > 0.1:  # More than 10% missing data
            logger.warning(f"High percentage of missing data: {missing_pct:.2%}")
        
        # Check for negative prices
        if (data < 0).any().any():
            logger.error("Found negative prices in data")
            return False
        
        # Check for zero prices
        zero_prices = (data == 0).sum().sum()
        if zero_prices > 0:
            logger.warning(f"Found {zero_prices} zero prices in data")
        
        return True
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for analysis"""
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        # Forward fill missing values (use previous day's price)
        data = data.fillna(method='ffill')
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        return data 