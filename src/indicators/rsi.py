"""
RSI (Relative Strength Index) Indicator

This module provides functions to calculate RSI indicator.
"""

import pandas as pd
import numpy as np
from typing import Dict


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate RSI (Relative Strength Index) indicator
    
    Args:
        prices: Price series (typically closing prices)
        period: Period for RSI calculation (default: 14)
        
    Returns:
        RSI series
    """
    delta = prices.diff()
    
    # Separate gains and losses
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