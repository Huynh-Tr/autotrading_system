"""
SMA + RSI Combined Indicator

This module provides functions to calculate combined SMA and RSI indicators
for the custom1 strategy.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional

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

# Import existing indicators
try:
    from .sma import calculate_sma_crossover
    from .rsi import calculate_rsi
except ImportError:
    # Fallback to absolute imports
    from src.indicators.sma import calculate_sma_crossover
    from src.indicators.rsi import calculate_rsi


def calculate_sma_rsi_combined(prices: pd.Series, 
                              short_window: int = 20, 
                              long_window: int = 50,
                              rsi_period: int = 14) -> Dict[str, pd.Series]:
    """
    Calculate combined SMA and RSI indicators
    
    Args:
        prices: Price series (typically closing prices)
        short_window: Short SMA window
        long_window: Long SMA window
        rsi_period: RSI period
        
    Returns:
        Dictionary containing combined indicators
    """
    # Calculate SMA components
    short_sma, long_sma = calculate_sma_crossover(prices, short_window, long_window)
    
    # Calculate RSI
    rsi = calculate_rsi(prices, rsi_period)
    
    # Calculate SMA crossover signal
    # Calculate crossover signals using vectorized operations
    golden_cross = (short_sma > long_sma) & (short_sma.shift(1) <= long_sma.shift(1))
    death_cross = (short_sma < long_sma) & (short_sma.shift(1) >= long_sma.shift(1))
    
    sma_signal = pd.Series(
        np.where(golden_cross, 1,  # Golden cross (buy signal)
                 np.where(death_cross, -1, 0)),  # Death cross (sell signal)
        index=prices.index
    )
    
    # Calculate RSI signal
    rsi_signal = pd.Series(
        np.where(rsi < 30, 1,  # Oversold (buy signal)
                 np.where(rsi > 70, -1, 0)),  # Overbought (sell signal)
        index=prices.index
    )
    
    # Calculate combined signal strength
    combined_signal = pd.Series(
        sma_signal.values + rsi_signal.values,
        index=prices.index
    )
    
    return {
        'short_sma': short_sma,
        'long_sma': long_sma,
        'rsi': rsi,
        'sma_signal': sma_signal,
        'rsi_signal': rsi_signal,
        'combined_signal': combined_signal,
        'sma_crossover': short_sma - long_sma
    }


def calculate_sma_rsi_momentum(prices: pd.Series, 
                              short_window: int = 20, 
                              long_window: int = 50,
                              rsi_period: int = 14) -> pd.Series:
    """
    Calculate momentum based on SMA and RSI combination
    
    Args:
        prices: Price series
        short_window: Short SMA window
        long_window: Long SMA window
        rsi_period: RSI period
        
    Returns:
        Momentum series
    """
    indicators = calculate_sma_rsi_combined(prices, short_window, long_window, rsi_period)
    
    # Calculate momentum based on SMA trend and RSI divergence
    sma_trend = indicators['sma_crossover'].diff()
    rsi_momentum = indicators['rsi'].diff()
    
    # Combined momentum
    momentum = (sma_trend * 0.6) + (rsi_momentum * 0.4)
    
    return momentum


def get_sma_rsi_signal_strength(prices: pd.Series, 
                                short_window: int = 20, 
                                long_window: int = 50,
                                rsi_period: int = 14) -> Dict[str, float]:
    """
    Get signal strength for SMA+RSI combination
    
    Args:
        prices: Price series
        short_window: Short SMA window
        long_window: Long SMA window
        rsi_period: RSI period
        
    Returns:
        Dictionary with signal strength metrics
    """
    indicators = calculate_sma_rsi_combined(prices, short_window, long_window, rsi_period)
    
    # Get latest values
    latest_sma_signal = indicators['sma_signal'].iloc[-1]
    latest_rsi_signal = indicators['rsi_signal'].iloc[-1]
    latest_combined = indicators['combined_signal'].iloc[-1]
    latest_rsi = indicators['rsi'].iloc[-1]
    latest_sma_crossover = indicators['sma_crossover'].iloc[-1]
    
    # Calculate signal strength
    sma_strength = abs(latest_sma_crossover) / prices.iloc[-1]  # Normalized by price
    rsi_strength = abs(latest_rsi - 50) / 50  # Distance from neutral (50)
    
    return {
        'sma_signal': latest_sma_signal,
        'rsi_signal': latest_rsi_signal,
        'combined_signal': latest_combined,
        'sma_strength': sma_strength,
        'rsi_strength': rsi_strength,
        'total_strength': (sma_strength + rsi_strength) / 2,
        'rsi_value': latest_rsi,
        'sma_crossover_value': latest_sma_crossover
    }


def validate_sma_rsi_parameters(short_window: int, long_window: int, rsi_period: int) -> bool:
    """
    Validate SMA+RSI parameters
    
    Args:
        short_window: Short SMA window
        long_window: Long SMA window
        rsi_period: RSI period
        
    Returns:
        True if parameters are valid
    """
    if short_window <= 0 or long_window <= 0 or rsi_period <= 0:
        logger.error("All windows must be positive")
        return False
    
    if short_window >= long_window:
        logger.error("Short window must be less than long window")
        return False
    
    if rsi_period > short_window:
        logger.warning("RSI period is larger than short SMA window")
    
    return True 