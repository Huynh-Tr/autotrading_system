"""
Bollinger Bands Indicator

This module provides functions to calculate Bollinger Bands indicator.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple

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


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, 
                            std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands
    
    Args:
        prices: Price series (typically closing prices)
        period: Period for SMA calculation (default: 20)
        std_dev: Number of standard deviations (default: 2)
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    # Ensure prices is numeric
    try:
        prices_numeric = pd.to_numeric(prices, errors='coerce')
        if prices_numeric.isna().all():
            logger.warning("All prices are non-numeric, returning NaN series")
            nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
            return nan_series, nan_series, nan_series
        
        # Calculate Bollinger Bands only on numeric data
        middle_band = prices_numeric.rolling(window=period).mean()
        std = prices_numeric.rolling(window=period).std()
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        return upper_band, middle_band, lower_band
    except Exception as e:
        logger.warning(f"Error calculating Bollinger Bands: {e}")
        nan_series = pd.Series([np.nan] * len(prices), index=prices.index)
        return nan_series, nan_series, nan_series


def calculate_bollinger_components(prices: pd.Series, period: int = 20, 
                                 std_dev: int = 2) -> Dict[str, pd.Series]:
    """
    Calculate Bollinger Bands components and return as dictionary
    
    Args:
        prices: Price series (typically closing prices)
        period: Period for SMA calculation (default: 20)
        std_dev: Number of standard deviations (default: 2)
        
    Returns:
        Dictionary containing Bollinger Bands components
    """
    upper_band, middle_band, lower_band = calculate_bollinger_bands(prices, period, std_dev)
    
    return {
        'upper_band': upper_band,
        'middle_band': middle_band,
        'lower_band': lower_band,
        'bandwidth': (upper_band - lower_band) / middle_band,
        'percent_b': (prices - lower_band) / (upper_band - lower_band)
    } 