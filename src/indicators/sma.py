"""
SMA (Simple Moving Average) Indicator

This module provides functions to calculate SMA indicators.
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


def calculate_sma(prices: pd.Series, window: int) -> pd.Series:
    """
    Calculate Simple Moving Average
    
    Args:
        prices: Price series (typically closing prices)
        window: Window size for SMA calculation
        
    Returns:
        SMA series
    """
    # Ensure prices is numeric
    try:
        prices_numeric = pd.to_numeric(prices, errors='coerce')
        if prices_numeric.isna().all():
            logger.warning("All prices are non-numeric, returning NaN series")
            return pd.Series([np.nan] * len(prices), index=prices.index)
        
        # Calculate SMA only on numeric data
        return prices_numeric.rolling(window=window).mean()
    except Exception as e:
        logger.warning(f"Error calculating SMA: {e}")
        return pd.Series([np.nan] * len(prices), index=prices.index)


def calculate_sma_crossover(prices: pd.Series, short_window: int, long_window: int) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate SMA crossover components
    
    Args:
        prices: Price series (typically closing prices)
        short_window: Short SMA window
        long_window: Long SMA window
        
    Returns:
        Tuple of (short_sma, long_sma)
    """
    short_sma = calculate_sma(prices, short_window)
    long_sma = calculate_sma(prices, long_window)
    
    return short_sma, long_sma


def calculate_sma_components(prices: pd.Series, short_window: int, long_window: int) -> Dict[str, pd.Series]:
    """
    Calculate SMA crossover components and return as dictionary
    
    Args:
        prices: Price series (typically closing prices)
        short_window: Short SMA window
        long_window: Long SMA window
        
    Returns:
        Dictionary containing SMA components
    """
    short_sma, long_sma = calculate_sma_crossover(prices, short_window, long_window)
    
    return {
        'short_sma': short_sma,
        'long_sma': long_sma,
        'sma_diff': short_sma - long_sma
    } 