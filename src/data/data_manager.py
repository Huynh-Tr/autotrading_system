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
from typing import List, Dict, Optional, Any, Tuple

# Try to import loguru, fallback to basic logging if not available
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    # Set up basic logging if loguru is not available
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

# Use absolute imports instead of relative imports
try:
    from ..utils.config_manager import ConfigManager
except ImportError:
    # Fallback to absolute imports
    from src.utils.config_manager import ConfigManager


class DataManager:
    """Manages market data retrieval and caching with full OHLCV data"""
    
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
        """Get historical market data for multiple symbols with full OHLCV data"""
        logger.info(f"Fetching historical OHLCV data for {symbols} from {start_date} to {end_date}")
        
        # Check cache first
        cache_key = f"{'_'.join(symbols)}_{start_date}_{end_date}_{interval}_ohlcv.pkl"
        cache_path = os.path.join(self.cache_dir, cache_key)
        
        if self.config.get("data.cache_data", True) and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    data = pickle.load(f)
                logger.info("OHLCV data loaded from cache")
                return data
            except Exception as e:
                logger.warning(f"Failed to load from cache: {e}")
        
        # Fetch data from source
        data_source = self.config.get("data.source", "yfinance")
        
        if data_source == "yfinance":
            data = self._fetch_yfinance_ohlcv_data(symbols, start_date, end_date, interval)
        elif data_source == "vnstock":
            data = self._fetch_vnstock_ohlcv_data(symbols, start_date, end_date, interval)
        else:
            raise ValueError(f"Unsupported data source: {data_source}")
        
        # Cache the data
        if self.config.get("data.cache_data", True):
            try:
                with open(cache_path, 'wb') as f:
                    pickle.dump(data, f)
                logger.info("OHLCV data cached successfully")
            except Exception as e:
                logger.warning(f"Failed to cache data: {e}")
        
        return data
    
    def get_close_data(self, symbols: List[str], start_date: str, 
                      end_date: str, interval: str = "1d") -> pd.DataFrame:
        """Get only close prices for backward compatibility"""
        ohlcv_data = self.get_historical_data(symbols, start_date, end_date, interval)
        return self._extract_close_prices(ohlcv_data, symbols)
    
    def _fetch_yfinance_ohlcv_data(self, symbols: List[str], start_date: str, 
                                  end_date: str, interval: str) -> pd.DataFrame:
        """Fetch full OHLCV data from Yahoo Finance"""
        try:
            # Download data for all symbols
            tickers = yf.Tickers(' '.join(symbols))
            data = tickers.history(start=start_date, end=end_date, interval=interval)
            
            # Create a comprehensive OHLCV DataFrame
            ohlcv_data = {}
            
            for symbol in symbols:
                symbol_data = {}
                
                # Extract OHLCV data for each symbol
                if f"{symbol} Time" in data.columns:
                    symbol_data['time'] = data[f"{symbol} Time"]
                if f"{symbol} Open" in data.columns:
                    symbol_data['open'] = data[f"{symbol} Open"]
                if f"{symbol} High" in data.columns:
                    symbol_data['high'] = data[f"{symbol} High"]
                if f"{symbol} Low" in data.columns:
                    symbol_data['low'] = data[f"{symbol} Low"]
                if f"{symbol} Close" in data.columns:
                    symbol_data['close'] = data[f"{symbol} Close"]
                if f"{symbol} Volume" in data.columns:
                    symbol_data['volume'] = data[f"{symbol} Volume"]
                
                if symbol_data:
                    # Create MultiIndex for the symbol
                    symbol_df = pd.DataFrame(symbol_data)
                    symbol_df.columns = pd.MultiIndex.from_product([[symbol], symbol_df.columns])
                    ohlcv_data[symbol] = symbol_df
                else:
                    logger.warning(f"No OHLCV data found for {symbol}")
                    # Create empty DataFrame with proper structure
                    empty_data = pd.DataFrame(index=data.index, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                    empty_data.columns = pd.MultiIndex.from_product([[symbol], empty_data.columns])
                    ohlcv_data[symbol] = empty_data
            
            # Combine all symbols into one DataFrame
            if ohlcv_data:
                combined_data = pd.concat(ohlcv_data.values(), axis=1)
                return combined_data
            else:
                return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching OHLCV data from Yahoo Finance: {e}")
            raise
    
    def _fetch_vnstock_ohlcv_data(self, symbols: List[str], start_date: str, 
                                 end_date: str, interval: str) -> pd.DataFrame:
        """Fetch full OHLCV data from VNStock"""
        try:
            all_data = {}
            
            for symbol in symbols:
                try:
                    data = Quote(symbol=symbol, source='tcbs').history(start=start_date, end=end_date, period=interval)
                    
                    if not data.empty:
                        # Extract OHLCV data
                        symbol_data = {
                            'time': data.get('time', pd.Series(dtype=float)),
                            'open': data.get('open', pd.Series(dtype=float)),
                            'high': data.get('high', pd.Series(dtype=float)),
                            'low': data.get('low', pd.Series(dtype=float)),
                            'close': data.get('close', pd.Series(dtype=float)),
                            'volume': data.get('volume', pd.Series(dtype=float))
                        }
                        
                        # Create MultiIndex for the symbol
                        symbol_df = pd.DataFrame(symbol_data)
                        symbol_df.columns = pd.MultiIndex.from_product([[symbol], symbol_df.columns])
                        all_data[symbol] = symbol_df
                    else:
                        logger.warning(f"No OHLCV data found for {symbol}")
                        # Create empty DataFrame with proper structure
                        empty_data = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                        empty_data.columns = pd.MultiIndex.from_product([[symbol], empty_data.columns])
                        all_data[symbol] = empty_data
                        
                except Exception as e:
                    logger.error(f"Error fetching OHLCV data for {symbol}: {e}")
                    # Create empty DataFrame with proper structure
                    empty_data = pd.DataFrame(columns=['time', 'open', 'high', 'low', 'close', 'volume'])
                    empty_data.columns = pd.MultiIndex.from_product([[symbol], empty_data.columns])
                    all_data[symbol] = empty_data
            
            # Combine all symbols into one DataFrame
            if all_data:
                combined_data = pd.concat(all_data.values(), axis=1)
                return combined_data
            else:
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching OHLCV data from VNStock: {e}")
            raise
    
    def _extract_close_prices(self, ohlcv_data: pd.DataFrame, symbols: List[str]) -> pd.DataFrame:
        """Extract only close prices from OHLCV data for backward compatibility"""
        close_data = {}
        
        for symbol in symbols:
            if (symbol, 'close') in ohlcv_data.columns:
                close_data[symbol] = ohlcv_data[(symbol, 'close')]
            else:
                logger.warning(f"No close data found for {symbol}")
                close_data[symbol] = pd.Series(dtype=float)
        
        return pd.DataFrame(close_data)
    
    def get_symbol_ohlcv(self, ohlcv_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Extract OHLCV data for a specific symbol"""
        if symbol not in ohlcv_data.columns.get_level_values(0):
            logger.warning(f"Symbol {symbol} not found in data")
            return pd.DataFrame()
        
        # Extract all columns for the symbol
        symbol_data = ohlcv_data[symbol]
        return symbol_data
    
    def get_price_data(self, ohlcv_data: pd.DataFrame, symbol: str, price_type: str = 'close') -> pd.Series:
        """Get specific price data (open, high, low, close) for a symbol"""
        if price_type not in ['open', 'high', 'low', 'close']:
            raise ValueError(f"Invalid price type: {price_type}. Must be one of: open, high, low, close")
        
        if (symbol, price_type) in ohlcv_data.columns:
            return ohlcv_data[(symbol, price_type)]
        else:
            logger.warning(f"No {price_type} data found for {symbol}")
            return pd.Series(dtype=float)
    
    def get_volume_data(self, ohlcv_data: pd.DataFrame, symbol: str) -> pd.Series:
        """Get volume data for a symbol"""
        if (symbol, 'volume') in ohlcv_data.columns:
            return ohlcv_data[(symbol, 'volume')]
        else:
            logger.warning(f"No volume data found for {symbol}")
            return pd.Series(dtype=float)
    
    # def calculate_technical_indicators(self, ohlcv_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        # """Calculate technical indicators from OHLCV data"""
        # symbol_data = self.get_symbol_ohlcv(ohlcv_data, symbol)
        
        # if symbol_data.empty:
        #     return pd.DataFrame()
        
        # # Calculate basic technical indicators
        # indicators = pd.DataFrame(index=symbol_data.index)
        
        # # Price-based indicators
        # indicators['close'] = symbol_data['close']
        # indicators['open'] = symbol_data['open']
        # indicators['high'] = symbol_data['high']
        # indicators['low'] = symbol_data['low']
        # indicators['volume'] = symbol_data['volume']
        
        # # Calculate returns
        # indicators['returns'] = symbol_data['close'].pct_change()
        
        # # Calculate price ranges
        # indicators['daily_range'] = symbol_data['high'] - symbol_data['low']
        # indicators['body_size'] = abs(symbol_data['close'] - symbol_data['open'])
        
        # # Calculate moving averages
        # indicators['sma_20'] = symbol_data['close'].rolling(window=20).mean()
        # indicators['sma_50'] = symbol_data['close'].rolling(window=50).mean()
        
        # # Calculate volatility
        # indicators['volatility'] = indicators['returns'].rolling(window=20).std()
        
        # return indicators
    
    def get_latest_ohlcv(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get latest OHLCV data for a symbol"""
        try:
            from datetime import datetime, timedelta
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            ohlcv_data = self.get_historical_data([symbol], start_date, end_date, "1d")
            symbol_data = self.get_symbol_ohlcv(ohlcv_data, symbol)
            
            if not symbol_data.empty:
                latest = symbol_data.iloc[-1]
                return {
                    'open': latest['open'],
                    'high': latest['high'],
                    'low': latest['low'],
                    'close': latest['close'],
                    'volume': latest['volume']
                }
            return None
        except Exception as e:
            logger.error(f"Error fetching latest OHLCV for {symbol}: {e}")
            return None
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest close price for a symbol (backward compatibility)"""
        latest_ohlcv = self.get_latest_ohlcv(symbol)
        return latest_ohlcv['close'] if latest_ohlcv else None
    
    def validate_ohlcv_data(self, data: pd.DataFrame) -> bool:
        """Validate OHLCV data quality"""
        if data.empty:
            logger.error("OHLCV data is empty")
            return False
        
        # Check for missing values
        missing_pct = data.isnull().sum().sum() / (data.shape[0] * data.shape[1])
        if missing_pct > 0.1:  # More than 10% missing data
            logger.warning(f"High percentage of missing OHLCV data: {missing_pct:.2%}")
        
        # Check for negative prices
        price_columns = [col for col in data.columns if col[1] in ['open', 'high', 'low', 'close']]
        if (data[price_columns] < 0).any().any():
            logger.error("Found negative prices in OHLCV data")
            return False
        
        # Check for zero prices
        zero_prices = (data[price_columns] == 0).sum().sum()
        if zero_prices > 0:
            logger.warning(f"Found {zero_prices} zero prices in OHLCV data")
        
        # Check for logical inconsistencies (high < low, etc.)
        for symbol in data.columns.get_level_values(0).unique():
            symbol_data = self.get_symbol_ohlcv(data, symbol)
            if not symbol_data.empty:
                # Check if high >= low
                invalid_high_low = (symbol_data['high'] < symbol_data['low']).sum()
                if invalid_high_low > 0:
                    logger.warning(f"Found {invalid_high_low} records where high < low for {symbol}")
                
                # Check if close is between high and low
                invalid_close = ((symbol_data['close'] > symbol_data['high']) | 
                               (symbol_data['close'] < symbol_data['low'])).sum()
                if invalid_close > 0:
                    logger.warning(f"Found {invalid_close} records where close is outside high-low range for {symbol}")
        
        return True
    
    def clean_ohlcv_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare OHLCV data for analysis"""
        # Remove rows with all NaN values
        data = data.dropna(how='all')
        
        # Forward fill missing values (use previous day's values)
        data = data.ffill()
        
        # Remove any remaining NaN values
        data = data.dropna()
        
        # Fix logical inconsistencies
        for symbol in data.columns.get_level_values(0).unique():
            symbol_data = self.get_symbol_ohlcv(data, symbol)
            if not symbol_data.empty:
                # Ensure high >= low
                symbol_data['high'] = symbol_data[['high', 'low']].max(axis=1)
                symbol_data['low'] = symbol_data[['high', 'low']].min(axis=1)
                
                # Ensure close is between high and low
                symbol_data['close'] = symbol_data['close'].clip(
                    lower=symbol_data['low'], 
                    upper=symbol_data['high']
                )
                
                # Update the main DataFrame
                data[symbol] = symbol_data
        
        return data 