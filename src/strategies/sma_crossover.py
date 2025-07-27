"""
SMA Crossover Strategy - Simple Moving Average crossover trading strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from loguru import logger

from .base_strategy import BaseStrategy


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
    
    def generate_signals(self, data: pd.Series) -> Dict[str, str]:
        """
        Generate trading signals based on SMA crossover
        
        Args:
            data: Series containing price data with SMA indicators
            
        Returns:
            Dict mapping symbol to signal ('buy', 'sell', 'hold')
        """
        signals = {}
        
        # Extract symbols from data (assuming data has columns for each symbol)
        if isinstance(data, pd.Series):
            # Single symbol data
            symbol = data.name if hasattr(data, 'name') else 'UNKNOWN'
            signal = self._generate_signal_for_symbol(data, symbol)
            signals[symbol] = signal
        else:
            # Multiple symbols data
            for symbol in data.columns:
                if isinstance(symbol, str) and not symbol.endswith('_SMA'):
                    signal = self._generate_signal_for_symbol(data[symbol], symbol)
                    signals[symbol] = signal
        
        return signals
    
    def _generate_signal_for_symbol(self, price_data: pd.Series, symbol: str) -> str:
        """Generate signal for a single symbol"""
        if len(price_data) < self.long_window:
            return 'hold'  # Not enough data
        
        # Calculate SMAs
        short_sma = price_data.rolling(window=self.short_window).mean()
        long_sma = price_data.rolling(window=self.long_window).mean()
        
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
        """Get strategy indicators for analysis"""
        short_sma = price_data.rolling(window=self.short_window).mean()
        long_sma = price_data.rolling(window=self.long_window).mean()
        
        return {
            'short_sma': short_sma,
            'long_sma': long_sma,
            'sma_diff': short_sma - long_sma
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get strategy summary with parameters"""
        summary = super().get_summary()
        summary.update({
            'short_window': self.short_window,
            'long_window': self.long_window,
            'strategy_type': 'SMA Crossover'
        })
        return summary 