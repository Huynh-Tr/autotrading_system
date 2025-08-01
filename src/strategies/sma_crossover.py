"""
SMA Crossover Strategy - Simple Moving Average crossover trading strategy
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
    from ..indicators.sma import calculate_sma_crossover, calculate_sma_components
except ImportError:
    # Fallback to absolute imports
    from src.strategies.base_strategy import BaseStrategy
    from src.indicators.sma import calculate_sma_crossover, calculate_sma_components


class SMACrossoverStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize SMA Crossover strategy"""
        super().__init__("sma_crossover", config)
        
        # Strategy parameters
        self.short_window = config.get('short_window', 20)
        self.long_window = config.get('long_window', 50)
        
        # Validate parameters
        if self.short_window >= self.long_window:
            raise ValueError("Short window must be less than long window")
        
        logger.info(f"SMA Crossover Strategy initialized: {self.short_window}/{self.long_window}")
    
    def generate_signals(self, historical_data: pd.DataFrame, current_data: pd.Series) -> Dict[str, str]:
        """
        Generate trading signals based on SMA crossover using standardized OHLCV format
        
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
            
            if len(symbol_data) >= self.long_window:
                signal = self._generate_signal_for_symbol(symbol_data, symbol)
                signals[symbol] = signal
            else:
                signals[symbol] = 'hold'  # Not enough data
        
        return signals
    
    def _generate_signal_for_symbol(self, price_data: pd.Series, symbol: str) -> str:
        """Generate signal for a single symbol"""
        if len(price_data) < self.long_window:
            return 'hold'  # Not enough data
        
        # Calculate SMAs
        short_sma, long_sma = calculate_sma_crossover(price_data, self.short_window, self.long_window)
        
        # Get current and previous values
        current_short = short_sma.iloc[-1]
        current_long = long_sma.iloc[-1]
        prev_short = short_sma.iloc[-2] if len(short_sma) > 1 else current_short
        prev_long = long_sma.iloc[-2] if len(long_sma) > 1 else current_long
        
        # Check for crossover
        current_cross_up = current_short > current_long
        prev_cross_up = prev_short > prev_long
        
        # Generate signals
        if current_cross_up and not prev_cross_up:
            # Golden cross (short SMA crosses above long SMA)
            logger.info(f"Golden cross detected for {symbol}: {current_short:.2f} > {current_long:.2f}")
            return 'buy'
        elif not current_cross_up and prev_cross_up:
            # Death cross (short SMA crosses below long SMA)
            logger.info(f"Death cross detected for {symbol}: {current_short:.2f} < {current_long:.2f}")
            return 'sell'
        else:
            return 'hold'
    
    def validate_config(self) -> bool:
        """Validate strategy configuration"""
        if self.short_window <= 0 or self.long_window <= 0:
            logger.error("SMA windows must be positive")
            return False
        
        if self.short_window >= self.long_window:
            logger.error("Short window must be less than long window")
            return False
        
        return True
    
    def get_indicators(self, price_data: pd.Series) -> Dict[str, pd.Series]:
        """Get strategy indicators for the given price data"""
        if len(price_data) < self.long_window:
            return {}
        
        # Calculate SMA components
        short_sma, long_sma = calculate_sma_components(price_data, self.short_window, self.long_window)
        
        return {
            f'{self.short_window}_SMA': short_sma,
            f'{self.long_window}_SMA': long_sma
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get strategy summary"""
        return {
            'name': self.name,
            'strategy_type': 'SMA Crossover',
            'short_window': self.short_window,
            'long_window': self.long_window,
            'description': f'SMA Crossover with {self.short_window}/{self.long_window} periods'
        } 