"""
MACD (Moving Average Convergence Divergence) Indicator

This module provides functions to calculate MACD indicator components.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def calculate_macd(prices: pd.Series, fast_period: int = 12, 
                  slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD indicator components
    
    Args:
        prices: Price series (typically closing prices)
        fast_period: Period for fast EMA (default: 12)
        slow_period: Period for slow EMA (default: 26)
        signal_period: Period for signal line EMA (default: 9)
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    # Calculate exponential moving averages
    ema_fast = prices.ewm(span=fast_period).mean()
    ema_slow = prices.ewm(span=slow_period).mean()
    
    # MACD line
    macd_line = ema_fast - ema_slow
    
    # Signal line
    signal_line = macd_line.ewm(span=signal_period).mean()
    
    # Histogram
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def calculate_macd_components(prices: pd.Series, fast_period: int = 12, 
                            slow_period: int = 26, signal_period: int = 9) -> dict:
    """
    Calculate all MACD components and return as dictionary
    
    Args:
        prices: Price series (typically closing prices)
        fast_period: Period for fast EMA (default: 12)
        slow_period: Period for slow EMA (default: 26)
        signal_period: Period for signal line EMA (default: 9)
        
    Returns:
        Dictionary containing all MACD components
    """
    macd_line, signal_line, histogram = calculate_macd(prices, fast_period, slow_period, signal_period)
    
    return {
        'macd_line': macd_line,
        'signal_line': signal_line,
        'histogram': histogram,
        'zero_line': pd.Series([0] * len(prices), index=prices.index)
    } 