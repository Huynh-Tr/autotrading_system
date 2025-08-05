"""
OHLCV Data Format Utilities

This module provides utilities for working with standardized OHLCV data format
across all components of the autotrading system.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

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


def extract_price_data(data: pd.DataFrame, symbol: str, price_type: str = 'close') -> pd.Series:
    """
    Extract price data from standardized OHLCV DataFrame
    
    Args:
        data: DataFrame with standardized OHLCV format
        symbol: Symbol to extract data for
        price_type: Type of price data ('open', 'high', 'low', 'close', 'volume')
        
    Returns:
        Price series for the specified symbol and type
    """
    if data.empty:
        logger.warning(f"Empty data provided for {symbol}")
        return pd.Series(dtype=float)
    
    # Handle MultiIndex columns (multi-symbol data)
    if isinstance(data.columns, pd.MultiIndex):
        if (symbol, price_type) in data.columns:
            return data[(symbol, price_type)]
        else:
            logger.warning(f"No {price_type} data found for {symbol}")
            return pd.Series(dtype=float)
    
    # Handle single symbol data (standard OHLCV format)
    elif price_type in data.columns:
        return data[price_type]
    
    else:
        logger.warning(f"No {price_type} column found in data")
        return pd.Series(dtype=float)


def get_symbol_data(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Extract all OHLCV data for a specific symbol
    
    Args:
        data: DataFrame with standardized OHLCV format
        symbol: Symbol to extract data for
        
    Returns:
        DataFrame with OHLCV data for the symbol
    """
    if data.empty:
        logger.warning(f"Empty data provided for {symbol}")
        return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Handle MultiIndex columns (multi-symbol data)
    if isinstance(data.columns, pd.MultiIndex):
        symbol_columns = [col for col in data.columns if col[0] == symbol]
        if symbol_columns:
            symbol_data = data[symbol_columns]
            # Rename columns to remove symbol prefix
            symbol_data.columns = [col[1] for col in symbol_columns]
            return symbol_data
        else:
            logger.warning(f"No data found for {symbol}")
            return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    # Handle single symbol data (already in standard format)
    else:
        return data


def get_symbols_from_data(data: pd.DataFrame) -> List[str]:
    """
    Extract list of symbols from standardized OHLCV DataFrame
    
    Args:
        data: DataFrame with standardized OHLCV format
        
    Returns:
        List of symbols in the data
    """
    if data is None or data.empty:
        return []
    
    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        logger.warning(f"Expected DataFrame, got {type(data)}")
        return []
    
    try:
        # Handle MultiIndex columns (multi-symbol data)
        if isinstance(data.columns, pd.MultiIndex):
            return list(data.columns.get_level_values(0).unique())
        
        # Handle single symbol data
        else:
            # If it's single symbol data, we need to infer the symbol
            # This is a fallback - ideally the data should have symbol information
            return ['SINGLE_SYMBOL']
    except Exception as e:
        logger.error(f"Error extracting symbols from data: {e}")
        return []


# def validate_ohlcv_data(data: pd.DataFrame) -> bool:
    # """
    # Validate that data is in standardized OHLCV format
    
    # Args:
    #     data: DataFrame to validate
        
    # Returns:
    #     True if data is valid, False otherwise
    # """
    # if data.empty:
    #     logger.warning("Data is empty")
    #     return False
    
    # # Check for required columns
    # required_columns = ['open', 'high', 'low', 'close', 'volume']
    
    # # Handle MultiIndex columns
    # if isinstance(data.columns, pd.MultiIndex):
    #     # Check if all required columns exist for at least one symbol
    #     symbols = data.columns.get_level_values(0).unique()
    #     for symbol in symbols:
    #         symbol_columns = [col[1] for col in data.columns if col[0] == symbol]
    #         if all(col in symbol_columns for col in required_columns):
    #             return True
    #     logger.warning("No symbol has all required OHLCV columns")
    #     return False
    
    # # Handle single symbol data
    # else:
    #     if all(col in data.columns for col in required_columns):
    #         return True
    #     else:
    #         logger.warning("Missing required OHLCV columns")
    #         return False


# def convert_to_standard_format(data: pd.DataFrame, symbol: str = None) -> pd.DataFrame:
    # """
    # Convert any DataFrame to standardized OHLCV format
    
    # Args:
    #     data: DataFrame to convert
    #     symbol: Symbol name (optional)
        
    # Returns:
    #     DataFrame in standardized OHLCV format
    # """
    # if data.empty:
    #     return pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
    
    # # If already in standard format, return as is
    # if validate_ohlcv_data(data):
    #     return data
    
    # # Handle conversion from various formats
    # result = data.copy()
    
    # # Common column name mappings
    # column_mappings = {
    #     'date': 'time', 'Date': 'time', 'DATE': 'time',
    #     'timestamp': 'time', 'Timestamp': 'time', 'TIMESTAMP': 'time',
    #     'datetime': 'time', 'DateTime': 'time', 'DATETIME': 'time',
    #     'open': 'open', 'Open': 'open', 'OPEN': 'open',
    #     'high': 'high', 'High': 'high', 'HIGH': 'high',
    #     'low': 'low', 'Low': 'low', 'LOW': 'low',
    #     'close': 'close', 'Close': 'close', 'CLOSE': 'close',
    #     'volume': 'volume', 'Volume': 'volume', 'VOLUME': 'volume',
    #     'vol': 'volume', 'Vol': 'volume', 'VOL': 'volume'
    # }
    
    # # Rename columns
    # for old_col, new_col in column_mappings.items():
    #     if old_col in result.columns:
    #         result = result.rename(columns={old_col: new_col})
    
    # # Ensure time column is datetime and set as index
    # if 'time' in result.columns:
    #     result['time'] = pd.to_datetime(result['time'], errors='coerce')
    #     result = result.dropna(subset=['time'])
    #     result.set_index('time', inplace=True)
    
    # # Ensure all price columns are numeric
    # price_columns = ['open', 'high', 'low', 'close', 'volume']
    # for col in price_columns:
    #     if col in result.columns:
    #         result[col] = pd.to_numeric(result[col], errors='coerce')
    
    # # Reorder columns to standard format
    # available_columns = [col for col in price_columns if col in result.columns]
    # if available_columns:
    #     result = result[available_columns]
    
    # logger.info(f"Converted data for {symbol if symbol else 'unknown'}: {result.shape}")
    # return result


def get_current_price(data: pd.DataFrame, symbol: str, price_type: str = 'close') -> Optional[float]:
    """
    Get the most recent price for a symbol
    
    Args:
        data: DataFrame or Series with standardized OHLCV format
        symbol: Symbol to get price for
        price_type: Type of price ('open', 'high', 'low', 'close')
        
    Returns:
        Most recent price or None if not available
    """
    # Handle Series input (single row from DataFrame)
    if isinstance(data, pd.Series):
        # For Series, we need to extract the price directly
        try:
            # Handle MultiIndex columns (multi-symbol data)
            if isinstance(data.index, pd.MultiIndex):
                # Look for the specific symbol and price type
                for col_name in data.index:
                    if col_name[0] == symbol and col_name[1] == price_type:
                        price_value = data[col_name]
                        return float(price_value) if pd.notna(price_value) else None
                return None
            else:
                # Single symbol data - check if the price type column exists
                if price_type in data.index:
                    price_value = data[price_type]
                    return float(price_value) if pd.notna(price_value) else None
                return None
        except Exception as e:
            logger.error(f"Error extracting price from Series: {e}")
            return None
    
    # Handle DataFrame input (original behavior)
    price_series = extract_price_data(data, symbol, price_type)
    
    if price_series.empty:
        return None
    
    # Get the most recent non-null price
    recent_price = price_series.dropna().iloc[-1] if not price_series.dropna().empty else None
    
    return float(recent_price) if recent_price is not None else None


# def get_price_range(data: pd.DataFrame, symbol: str, 
    #                start_date: Optional[datetime] = None, 
    #                end_date: Optional[datetime] = None) -> pd.DataFrame:
    # """
    # Get price data for a specific date range
    
    # Args:
    #     data: DataFrame with standardized OHLCV format
    #     symbol: Symbol to get data for
    #     start_date: Start date (optional)
    #     end_date: End date (optional)
        
    # Returns:
    #     DataFrame with price data for the specified range
    # """
    # symbol_data = get_symbol_data(data, symbol)
    
    # if symbol_data.empty:
    #     return pd.DataFrame()
    
    # # Filter by date range if specified
    # if start_date or end_date:
    #     if start_date:
    #         symbol_data = symbol_data[symbol_data.index >= start_date]
    #     if end_date:
    #         symbol_data = symbol_data[symbol_data.index <= end_date]
    
    # return symbol_data


# def calculate_returns(data: pd.DataFrame, symbol: str, 
    #                  price_type: str = 'close', 
    #                  periods: int = 1) -> pd.Series:
    # """
    # Calculate returns for a symbol
    
    # Args:
    #     data: DataFrame with standardized OHLCV format
    #     symbol: Symbol to calculate returns for
    #     price_type: Type of price to use ('close', 'open', etc.)
    #     periods: Number of periods for return calculation
        
    # Returns:
    #     Series with returns
    # """
    # price_series = extract_price_data(data, symbol, price_type)
    
    # if price_series.empty:
    #     return pd.Series(dtype=float)
    
    # return price_series.pct_change(periods=periods)


# def get_volume_data(data: pd.DataFrame, symbol: str) -> pd.Series:
    # """
    # Get volume data for a symbol
    
    # Args:
    #     data: DataFrame with standardized OHLCV format
    #     symbol: Symbol to get volume for
        
    # Returns:
    #     Volume series
    # """
    # return extract_price_data(data, symbol, 'volume')


# def format_data_for_indicators(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    # """
    # Format data for use with technical indicators
    
    # Args:
    #     data: DataFrame with standardized OHLCV format
    #     symbol: Symbol to format data for
        
    # Returns:
    #     DataFrame formatted for indicators
    # """
    # symbol_data = get_symbol_data(data, symbol)
    
    # if symbol_data.empty:
    #     return pd.DataFrame()
    
    # # Ensure we have the required columns
    # required_columns = ['open', 'high', 'low', 'close', 'volume']
    # missing_columns = [col for col in required_columns if col not in symbol_data.columns]
    
    # if missing_columns:
    #     logger.warning(f"Missing columns for {symbol}: {missing_columns}")
    #     # Add missing columns with NaN values
    #     for col in missing_columns:
    #         symbol_data[col] = np.nan
    
    # return symbol_data[required_columns]


# def get_multi_symbol_data(data: pd.DataFrame, symbols: List[str]) -> Dict[str, pd.DataFrame]:
    # """
    # Extract data for multiple symbols
    
    # Args:
    #     data: DataFrame with standardized OHLCV format
    #     symbols: List of symbols to extract
        
    # Returns:
    #     Dictionary mapping symbols to their data
    # """
    # result = {}
    
    # for symbol in symbols:
    #     symbol_data = get_symbol_data(data, symbol)
    #     if not symbol_data.empty:
    #         result[symbol] = symbol_data
    
    # return result


# def merge_ohlcv_data(data_list: List[pd.DataFrame]) -> pd.DataFrame:
    # """
    # Merge multiple OHLCV DataFrames
    
    # Args:
    #     data_list: List of DataFrames to merge
        
    # Returns:
    #     Merged DataFrame
    # """
    # if not data_list:
    #     return pd.DataFrame()
    
    # # Start with the first DataFrame
    # result = data_list[0].copy()
    
    # # Merge with remaining DataFrames
    # for data in data_list[1:]:
    #     if not data.empty:
    #         result = pd.concat([result, data], axis=1)
    
    # return result


# def validate_and_clean_data(data: pd.DataFrame) -> pd.DataFrame:
    # """
    # Validate and clean OHLCV data
    
    # Args:
    #     data: DataFrame to validate and clean
        
    # Returns:
    #     Cleaned DataFrame
    # """
    # if not validate_ohlcv_data(data):
    #     logger.warning("Invalid OHLCV data format")
    #     return pd.DataFrame()
    
    # # Remove rows with all NaN values
    # cleaned_data = data.dropna(how='all')
    
    # # Forward fill missing values
    # cleaned_data = cleaned_data.ffill()
    
    # # Remove any remaining NaN values
    # cleaned_data = cleaned_data.dropna()
    
    # return cleaned_data 