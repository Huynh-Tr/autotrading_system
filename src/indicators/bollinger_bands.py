"""
Bollinger Bands Indicator

This module provides functions to calculate Bollinger Bands indicator.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


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
    middle_band = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper_band = middle_band + (std * std_dev)
    lower_band = middle_band - (std * std_dev)
    
    return upper_band, middle_band, lower_band


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