"""
Technical Indicators Module

This module provides comprehensive technical analysis indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
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

from .sma import calculate_sma
from .rsi import calculate_rsi
from .macd import calculate_macd, calculate_macd_components
from .bollinger_bands import calculate_bollinger_bands, calculate_bollinger_components


def calculate_ema(prices: pd.Series, span: int) -> pd.Series:
    """
    Calculate Exponential Moving Average
    
    Args:
        prices: Price series (typically closing prices)
        span: Span for EMA calculation
        
    Returns:
        EMA series
    """
    # Ensure prices is numeric
    try:
        prices_numeric = pd.to_numeric(prices, errors='coerce')
        if prices_numeric.isna().all():
            logger.warning("All prices are non-numeric, returning NaN series")
            return pd.Series([np.nan] * len(prices), index=prices.index)
        
        # Calculate EMA only on numeric data
        return prices_numeric.ewm(span=span).mean()
    except Exception as e:
        logger.warning(f"Error calculating EMA: {e}")
        return pd.Series([np.nan] * len(prices), index=prices.index)


def calculate_all_technical_indicators(data: pd.DataFrame, 
                                    symbols: Optional[List[str]] = None,
                                    include_volume: bool = True) -> pd.DataFrame:
    """
    Calculate all technical indicators for multiple symbols
    
    Args:
        data: DataFrame with price data (columns are symbols)
        symbols: List of symbols to process (if None, uses all columns)
        include_volume: Whether to include volume-based indicators
        
    Returns:
        DataFrame with original data plus all technical indicators
    """
    if symbols is None:
        symbols = list(data.columns)
    
    result = data.copy()
    
    for symbol in symbols:
        if symbol not in data.columns:
            continue
            
        price_series = data[symbol]
        
        # Ensure price_series is numeric
        try:
            price_series_numeric = pd.to_numeric(price_series, errors='coerce')
            if price_series_numeric.isna().all():
                logger.warning(f"All prices for {symbol} are non-numeric, skipping indicators")
                continue
        except Exception as e:
            logger.warning(f"Error converting prices to numeric for {symbol}: {e}")
            continue
        
        # Simple Moving Averages
        result[f'{symbol}_SMA_20'] = calculate_sma(price_series_numeric, 20)
        result[f'{symbol}_SMA_50'] = calculate_sma(price_series_numeric, 50)
        
        # Exponential Moving Averages
        result[f'{symbol}_EMA_12'] = calculate_ema(price_series_numeric, 12)
        result[f'{symbol}_EMA_26'] = calculate_ema(price_series_numeric, 26)
        
        # RSI
        result[f'{symbol}_RSI'] = calculate_rsi(price_series_numeric, 14)
        
        # MACD
        macd_line, signal_line, histogram = calculate_macd(price_series_numeric, 12, 26, 9)
        result[f'{symbol}_MACD'] = macd_line
        result[f'{symbol}_MACD_Signal'] = signal_line
        result[f'{symbol}_MACD_Histogram'] = histogram
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(price_series_numeric, 20, 2)
        result[f'{symbol}_BB_Upper'] = bb_upper
        result[f'{symbol}_BB_Middle'] = bb_middle
        result[f'{symbol}_BB_Lower'] = bb_lower
        
        # Additional indicators - ensure numeric operations
        try:
            result[f'{symbol}_Price_Change'] = price_series_numeric.pct_change()
            result[f'{symbol}_Price_Change_Pct'] = price_series_numeric.pct_change() * 100
            
            # Volatility indicators
            result[f'{symbol}_Volatility_20'] = price_series_numeric.rolling(window=20).std()
            result[f'{symbol}_Volatility_50'] = price_series_numeric.rolling(window=50).std()
            
            # Price levels
            result[f'{symbol}_High_20'] = price_series_numeric.rolling(window=20).max()
            result[f'{symbol}_Low_20'] = price_series_numeric.rolling(window=20).min()
            result[f'{symbol}_High_50'] = price_series_numeric.rolling(window=50).max()
            result[f'{symbol}_Low_50'] = price_series_numeric.rolling(window=50).min()
        except Exception as e:
            logger.warning(f"Error calculating advanced indicators for {symbol}: {e}")
    
    return result


def calculate_indicators_for_symbol(price_data: pd.Series, 
                                 symbol: str,
                                 include_advanced: bool = True) -> Dict[str, pd.Series]:
    """
    Calculate all indicators for a single symbol
    
    Args:
        price_data: Price series for the symbol
        symbol: Symbol name
        include_advanced: Whether to include advanced indicators
        
    Returns:
        Dictionary with all calculated indicators
    """
    indicators = {}
    
    # Ensure price_data is numeric
    try:
        price_data_numeric = pd.to_numeric(price_data, errors='coerce')
        if price_data_numeric.isna().all():
            logger.warning(f"All prices for {symbol} are non-numeric, returning empty indicators")
            return indicators
    except Exception as e:
        logger.warning(f"Error converting prices to numeric for {symbol}: {e}")
        return indicators
    
    # Basic indicators
    indicators[f'{symbol}_SMA_20'] = calculate_sma(price_data_numeric, 20)
    indicators[f'{symbol}_SMA_50'] = calculate_sma(price_data_numeric, 50)
    indicators[f'{symbol}_EMA_12'] = calculate_ema(price_data_numeric, 12)
    indicators[f'{symbol}_EMA_26'] = calculate_ema(price_data_numeric, 26)
    indicators[f'{symbol}_RSI'] = calculate_rsi(price_data_numeric, 14)
    
    # MACD components
    macd_line, signal_line, histogram = calculate_macd(price_data_numeric, 12, 26, 9)
    indicators[f'{symbol}_MACD'] = macd_line
    indicators[f'{symbol}_MACD_Signal'] = signal_line
    indicators[f'{symbol}_MACD_Histogram'] = histogram
    
    # Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(price_data_numeric, 20, 2)
    indicators[f'{symbol}_BB_Upper'] = bb_upper
    indicators[f'{symbol}_BB_Middle'] = bb_middle
    indicators[f'{symbol}_BB_Lower'] = bb_lower
    
    if include_advanced:
        # Advanced indicators - ensure numeric operations
        try:
            indicators[f'{symbol}_Price_Change'] = price_data_numeric.pct_change()
            indicators[f'{symbol}_Price_Change_Pct'] = price_data_numeric.pct_change() * 100
            indicators[f'{symbol}_Volatility_20'] = price_data_numeric.rolling(window=20).std()
            indicators[f'{symbol}_Volatility_50'] = price_data_numeric.rolling(window=50).std()
            indicators[f'{symbol}_High_20'] = price_data_numeric.rolling(window=20).max()
            indicators[f'{symbol}_Low_20'] = price_data_numeric.rolling(window=20).min()
            indicators[f'{symbol}_High_50'] = price_data_numeric.rolling(window=50).max()
            indicators[f'{symbol}_Low_50'] = price_data_numeric.rolling(window=50).min()
        except Exception as e:
            logger.warning(f"Error calculating advanced indicators for {symbol}: {e}")
        
        # Additional moving averages
        indicators[f'{symbol}_SMA_10'] = calculate_sma(price_data_numeric, 10)
        indicators[f'{symbol}_SMA_30'] = calculate_sma(price_data_numeric, 30)
        indicators[f'{symbol}_EMA_9'] = calculate_ema(price_data_numeric, 9)
        indicators[f'{symbol}_EMA_21'] = calculate_ema(price_data_numeric, 21)
    
    return indicators


def get_indicator_summary(data: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """
    Get summary statistics for all indicators of a symbol
    
    Args:
        data: DataFrame with price and indicator data
        symbol: Symbol to analyze
        
    Returns:
        Dictionary with indicator summaries
    """
    summary = {}
    
    if f'{symbol}_SMA_20' in data.columns:
        summary['sma_20'] = {
            'current': data[f'{symbol}_SMA_20'].iloc[-1],
            'min': data[f'{symbol}_SMA_20'].min(),
            'max': data[f'{symbol}_SMA_20'].max(),
            'trend': 'up' if data[f'{symbol}_SMA_20'].iloc[-1] > data[f'{symbol}_SMA_20'].iloc[-2] else 'down'
        }
    
    if f'{symbol}_RSI' in data.columns:
        rsi_current = data[f'{symbol}_RSI'].iloc[-1]
        summary['rsi'] = {
            'current': rsi_current,
            'status': 'overbought' if rsi_current > 70 else 'oversold' if rsi_current < 30 else 'neutral',
            'min': data[f'{symbol}_RSI'].min(),
            'max': data[f'{symbol}_RSI'].max()
        }
    
    if f'{symbol}_MACD' in data.columns:
        macd_current = data[f'{symbol}_MACD'].iloc[-1]
        signal_current = data[f'{symbol}_MACD_Signal'].iloc[-1]
        summary['macd'] = {
            'current': macd_current,
            'signal': signal_current,
            'histogram': macd_current - signal_current,
            'signal_type': 'bullish' if macd_current > signal_current else 'bearish'
        }
    
    if f'{symbol}_BB_Upper' in data.columns:
        price_current = data[symbol].iloc[-1]
        bb_upper = data[f'{symbol}_BB_Upper'].iloc[-1]
        bb_lower = data[f'{symbol}_BB_Lower'].iloc[-1]
        summary['bollinger_bands'] = {
            'position': 'above_upper' if price_current > bb_upper else 'below_lower' if price_current < bb_lower else 'within_bands',
            'upper': bb_upper,
            'lower': bb_lower,
            'width': bb_upper - bb_lower
        }
    
    return summary


def validate_indicators(data: pd.DataFrame, symbol: str) -> bool:
    """
    Validate that indicators are calculated correctly
    
    Args:
        data: DataFrame with price and indicator data
        symbol: Symbol to validate
        
    Returns:
        True if indicators are valid, False otherwise
    """
    try:
        # Check if price data exists
        if symbol not in data.columns:
            return False
        
        # Check if basic indicators exist
        required_indicators = [
            f'{symbol}_SMA_20',
            f'{symbol}_RSI',
            f'{symbol}_MACD'
        ]
        
        for indicator in required_indicators:
            if indicator not in data.columns:
                return False
            
            # Check for infinite or NaN values
            if data[indicator].isnull().all() or np.isinf(data[indicator]).any():
                return False
        
        # Check RSI bounds
        if f'{symbol}_RSI' in data.columns:
            rsi = data[f'{symbol}_RSI']
            if rsi.min() < 0 or rsi.max() > 100:
                return False
        
        return True
        
    except Exception:
        return False


def get_indicator_metadata() -> Dict[str, Dict[str, Any]]:
    """
    Get metadata for all available indicators
    
    Returns:
        Dictionary with indicator metadata
    """
    return {
        'sma': {
            'name': 'Simple Moving Average',
            'description': 'Average of closing prices over a specified period',
            'parameters': ['period'],
            'default_periods': [10, 20, 30, 50],
            'type': 'trend'
        },
        'ema': {
            'name': 'Exponential Moving Average',
            'description': 'Weighted average that gives more importance to recent prices',
            'parameters': ['span'],
            'default_spans': [9, 12, 21, 26],
            'type': 'trend'
        },
        'rsi': {
            'name': 'Relative Strength Index',
            'description': 'Momentum oscillator measuring speed and change of price movements',
            'parameters': ['period'],
            'default_period': 14,
            'overbought': 70,
            'oversold': 30,
            'type': 'momentum'
        },
        'macd': {
            'name': 'Moving Average Convergence Divergence',
            'description': 'Trend-following momentum indicator',
            'parameters': ['fast_period', 'slow_period', 'signal_period'],
            'default_fast': 12,
            'default_slow': 26,
            'default_signal': 9,
            'type': 'trend'
        },
        'bollinger_bands': {
            'name': 'Bollinger Bands',
            'description': 'Volatility indicator with upper and lower bands',
            'parameters': ['period', 'std_dev'],
            'default_period': 20,
            'default_std_dev': 2,
            'type': 'volatility'
        }
    }


def export_indicators_to_csv(data: pd.DataFrame, 
                           symbols: List[str], 
                           filename: str) -> bool:
    """
    Export indicator data to CSV file
    
    Args:
        data: DataFrame with price and indicator data
        symbols: List of symbols to export
        filename: Output CSV filename
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        # Select only price and indicator columns for specified symbols
        columns_to_export = []
        for symbol in symbols:
            # Add price column
            if symbol in data.columns:
                columns_to_export.append(symbol)
            
            # Add indicator columns for this symbol
            symbol_indicators = [col for col in data.columns if col.startswith(f'{symbol}_')]
            columns_to_export.extend(symbol_indicators)
        
        # Export to CSV
        export_data = data[columns_to_export]
        export_data.to_csv(filename, index=True)
        
        return True
        
    except Exception as e:
        print(f"Error exporting indicators: {e}")
        return False


def calculate_correlation_matrix(data: pd.DataFrame, 
                              symbols: List[str], 
                              indicator: str = 'SMA_20') -> pd.DataFrame:
    """
    Calculate correlation matrix for a specific indicator across symbols
    
    Args:
        data: DataFrame with price and indicator data
        symbols: List of symbols to analyze
        indicator: Indicator to correlate (e.g., 'SMA_20', 'RSI')
        
    Returns:
        Correlation matrix DataFrame
    """
    correlation_data = {}
    
    for symbol in symbols:
        indicator_col = f'{symbol}_{indicator}'
        if indicator_col in data.columns:
            correlation_data[symbol] = data[indicator_col]
    
    if correlation_data:
        correlation_df = pd.DataFrame(correlation_data)
        return correlation_df.corr()
    else:
        return pd.DataFrame()


# Convenience functions for backward compatibility
def calculate_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Legacy function for backward compatibility
    
    Args:
        data: DataFrame with price data
        
    Returns:
        DataFrame with technical indicators
    """
    return calculate_all_technical_indicators(data)


def get_indicators_for_symbol(price_data: pd.Series, symbol: str) -> Dict[str, pd.Series]:
    """
    Legacy function for backward compatibility
    
    Args:
        price_data: Price series for the symbol
        symbol: Symbol name
        
    Returns:
        Dictionary with calculated indicators
    """
    return calculate_indicators_for_symbol(price_data, symbol) 