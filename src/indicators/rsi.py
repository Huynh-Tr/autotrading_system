"""
RSI (Relative Strength Index) Indicator

This module provides functions to calculate RSI indicator.
"""

import pandas as pd
import numpy as np
from typing import Dict

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


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index) indicator
    
    Args:
        prices: Price series (typically closing prices)
        period: Period for RSI calculation (default: 14)
        
    Returns:
        RSI series
    """
    # Ensure prices is numeric
    try:
        prices_numeric = pd.to_numeric(prices, errors='coerce')
        if prices_numeric.isna().all():
            logger.warning("All prices are non-numeric, returning NaN series")
            return pd.Series([np.nan] * len(prices), index=prices.index)
    except Exception as e:
        logger.warning(f"Error converting prices to numeric: {e}")
        return pd.Series([np.nan] * len(prices), index=prices.index)
    
    delta = prices_numeric.diff()
    
    # Separate gains and losses - ensure numeric comparison
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period).mean()
    avg_losses = losses.rolling(window=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_rsi_components(prices: pd.Series, period: int = 14, 
                           overbought_threshold: int = 70, 
                           oversold_threshold: int = 30) -> Dict[str, pd.Series]:
    """
    Calculate RSI and related components
    
    Args:
        prices: Price series (typically closing prices)
        period: Period for RSI calculation (default: 14)
        overbought_threshold: Overbought level (default: 70)
        oversold_threshold: Oversold level (default: 30)
        
    Returns:
        Dictionary containing RSI and threshold lines
    """
    rsi = calculate_rsi(prices, period)
    
    return {
        'rsi': rsi,
        'overbought_line': pd.Series([overbought_threshold] * len(prices), index=prices.index),
        'oversold_line': pd.Series([oversold_threshold] * len(prices), index=prices.index)
    } 