"""
RSI Strategy - Relative Strength Index trading strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

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
    from .base_strategy import BaseStrategy
    from ..indicators.rsi import calculate_rsi, calculate_rsi_components
except ImportError:
    # Fallback to absolute imports
    from src.strategies.base_strategy import BaseStrategy
    from src.indicators.rsi import calculate_rsi, calculate_rsi_components


class RSIStrategy(BaseStrategy):
    """Relative Strength Index (RSI) Trading Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize RSI strategy"""
        super().__init__("rsi", config)
        
        # Strategy parameters
        self.period = config.get('period', 14)
        self.overbought_threshold = config.get('overbought_threshold', 70)
        self.oversold_threshold = config.get('oversold_threshold', 30)
        self.confirmation_period = config.get('confirmation_period', 2)
        
        # Validate parameters
        if self.period <= 0:
            raise ValueError("RSI period must be positive")
        
        if self.overbought_threshold <= self.oversold_threshold:
            raise ValueError("Overbought threshold must be greater than oversold threshold")
        
        if self.overbought_threshold > 100 or self.oversold_threshold < 0:
            raise ValueError("RSI thresholds must be between 0 and 100")
        
        logger.info(f"RSI Strategy initialized: period={self.period}, "
                   f"overbought={self.overbought_threshold}, oversold={self.oversold_threshold}")
    
    def generate_signals(self, historical_data: pd.DataFrame, current_data: pd.Series) -> Dict[str, str]:
        """
        Generate trading signals based on RSI using standardized OHLCV format
        
        Args:
            historical_data: Historical market data up to current point (standardized OHLCV)
            current_data: Current day's market data
            
        Returns:
            Dict mapping symbol to signal ('buy', 'sell', 'hold')
        """
        from ..utils.ohlcv_utils import get_symbols_from_data, extract_price_data
        
        signals = {}
        
        # Get symbols from the data
        symbols = get_symbols_from_data(historical_data)
        
        for symbol in symbols:
            # Extract close price data for this symbol
            close_series = extract_price_data(historical_data, symbol, 'close')
            
            if close_series.empty:
                logger.warning(f"No close data found for {symbol}")
                signals[symbol] = 'hold'
                continue
            
            # Drop NaN values
            symbol_data = close_series.dropna()
            
            if len(symbol_data) >= self.period + self.confirmation_period:
                signal = self._generate_signal_for_symbol(symbol_data, symbol)
                signals[symbol] = signal
            else:
                signals[symbol] = 'hold'  # Not enough data
        
        return signals
    
    def _generate_signal_for_symbol(self, price_data: pd.Series, symbol: str) -> str:
        """Generate signal for a single symbol"""
        if len(price_data) < self.period + self.confirmation_period:
            return 'hold'  # Not enough data
        
        # Calculate RSI
        rsi = self._calculate_rsi(price_data, self.period)
        
        # Get current and previous RSI values
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi
        
        # Check for oversold condition (buy signal)
        if current_rsi < self.oversold_threshold and prev_rsi >= self.oversold_threshold:
            # RSI crossed below oversold threshold
            logger.info(f"RSI oversold signal for {symbol}: {current_rsi:.2f} < {self.oversold_threshold}")
            return 'buy'
        
        # Check for overbought condition (sell signal)
        elif current_rsi > self.overbought_threshold and prev_rsi <= self.overbought_threshold:
            # RSI crossed above overbought threshold
            logger.info(f"RSI overbought signal for {symbol}: {current_rsi:.2f} > {self.overbought_threshold}")
            return 'sell'
        
        # Check for confirmation signals
        elif self.confirmation_period > 1:
            # Look for sustained oversold/overbought conditions
            recent_rsi = rsi.tail(self.confirmation_period)
            
            # Sustained oversold (buy signal)
            if all(rsi_val < self.oversold_threshold for rsi_val in recent_rsi):
                logger.info(f"RSI sustained oversold for {symbol}: {current_rsi:.2f}")
                return 'buy'
            
            # Sustained overbought (sell signal)
            elif all(rsi_val > self.overbought_threshold for rsi_val in recent_rsi):
                logger.info(f"RSI sustained overbought for {symbol}: {current_rsi:.2f}")
                return 'sell'
        
        return 'hold'
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI for the given price series"""
        return calculate_rsi(prices, period)
    
    # def validate_config(self) -> bool:
        # """Validate strategy configuration"""
        # if self.period <= 0:
        #     logger.error("RSI period must be positive")
        #     return False
        
        # if self.overbought_threshold <= self.oversold_threshold:
        #     logger.error("Overbought threshold must be greater than oversold threshold")
        #     return False
        
        # if self.overbought_threshold > 100 or self.oversold_threshold < 0:
        #     logger.error("RSI thresholds must be between 0 and 100")
        #     return False
        
        # if self.confirmation_period <= 0:
        #     logger.error("Confirmation period must be positive")
        #     return False
        
        # return True
    
    def get_indicators(self, price_data: pd.Series) -> Dict[str, pd.Series]:
        """Get strategy indicators for the given price data"""
        if len(price_data) < self.period:
            return {}
        
        # Calculate RSI
        rsi = calculate_rsi(price_data, self.period)
        
        return {
            'RSI': rsi,
            'Overbought': pd.Series([self.overbought_threshold] * len(rsi), index=rsi.index),
            'Oversold': pd.Series([self.oversold_threshold] * len(rsi), index=rsi.index)
        }
    
    # def get_summary(self) -> Dict[str, Any]:
        # """Get strategy summary"""
        # return {
        #     'name': self.name,
        #     'strategy_type': 'RSI Strategy',
        #     'period': self.period,
        #     'overbought_threshold': self.overbought_threshold,
        #     'oversold_threshold': self.oversold_threshold,
        #     'confirmation_period': self.confirmation_period,
        #     'description': f'RSI Strategy with {self.period} period, {self.oversold_threshold}/{self.overbought_threshold} levels'
        # } 