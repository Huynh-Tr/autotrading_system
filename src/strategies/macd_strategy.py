"""
MACD Strategy - Moving Average Convergence Divergence trading strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from loguru import logger

from .base_strategy import BaseStrategy
from ..indicators.macd import calculate_macd, calculate_macd_components


class MACDStrategy(BaseStrategy):
    """Moving Average Convergence Divergence (MACD) Trading Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize MACD strategy"""
        super().__init__("macd", config)
        
        # Strategy parameters
        self.fast_period = config.get('fast_period', 12)
        self.slow_period = config.get('slow_period', 26)
        self.signal_period = config.get('signal_period', 9)
        self.histogram_threshold = config.get('histogram_threshold', 0.0)
        self.confirmation_period = config.get('confirmation_period', 1)
        
        # Validate parameters
        if self.fast_period >= self.slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        if self.signal_period <= 0:
            raise ValueError("Signal period must be positive")
        
        logger.info(f"MACD Strategy initialized: fast={self.fast_period}, "
                   f"slow={self.slow_period}, signal={self.signal_period}")
    
    def generate_signals(self, historical_data: pd.DataFrame, current_data: pd.Series) -> Dict[str, str]:
        """
        Generate trading signals based on MACD
        
        Args:
            historical_data: Historical market data up to current point
            current_data: Current day's market data
            
        Returns:
            Dict mapping symbol to signal ('buy', 'sell', 'hold')
        """
        signals = {}
        
        # Process each symbol
        for symbol in historical_data.columns:
            if isinstance(symbol, str) and not symbol.endswith('_MACD'):
                # Get historical price data for this symbol
                symbol_data = historical_data[symbol].dropna()
                
                min_periods = max(self.slow_period, self.signal_period) + self.confirmation_period
                if len(symbol_data) >= min_periods:
                    signal = self._generate_signal_for_symbol(symbol_data, symbol)
                    signals[symbol] = signal
                else:
                    signals[symbol] = 'hold'  # Not enough data
        
        return signals
    
    def _generate_signal_for_symbol(self, price_data: pd.Series, symbol: str) -> str:
        """Generate signal for a single symbol"""
        min_periods = max(self.slow_period, self.signal_period) + self.confirmation_period
        
        if len(price_data) < min_periods:
            return 'hold'  # Not enough data
        
        # Calculate MACD components
        macd_line, signal_line, histogram = self._calculate_macd(price_data)
        
        # Get current and previous values
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        current_histogram = histogram.iloc[-1]
        
        prev_macd = macd_line.iloc[-2] if len(macd_line) > 1 else current_macd
        prev_signal = signal_line.iloc[-2] if len(signal_line) > 1 else current_signal
        prev_histogram = histogram.iloc[-2] if len(histogram) > 1 else current_histogram
        
        # Signal 1: MACD line crosses above signal line (bullish crossover)
        if (current_macd > current_signal and prev_macd <= prev_signal and 
            current_histogram > self.histogram_threshold):
            logger.info(f"MACD bullish crossover for {symbol}: MACD={current_macd:.4f}, Signal={current_signal:.4f}")
            return 'buy'
        
        # Signal 2: MACD line crosses below signal line (bearish crossover)
        elif (current_macd < current_signal and prev_macd >= prev_signal and 
              current_histogram < -self.histogram_threshold):
            logger.info(f"MACD bearish crossover for {symbol}: MACD={current_macd:.4f}, Signal={current_signal:.4f}")
            return 'sell'
        
        # Signal 3: Histogram turns positive (bullish momentum)
        elif (current_histogram > 0 and prev_histogram <= 0 and 
              current_macd > current_signal):
            logger.info(f"MACD histogram bullish for {symbol}: Histogram={current_histogram:.4f}")
            return 'buy'
        
        # Signal 4: Histogram turns negative (bearish momentum)
        elif (current_histogram < 0 and prev_histogram >= 0 and 
              current_macd < current_signal):
            logger.info(f"MACD histogram bearish for {symbol}: Histogram={current_histogram:.4f}")
            return 'sell'
        
        # Signal 5: Zero line crossover (strong trend signal)
        elif (current_macd > 0 and prev_macd <= 0):
            logger.info(f"MACD zero line bullish crossover for {symbol}: MACD={current_macd:.4f}")
            return 'buy'
        
        elif (current_macd < 0 and prev_macd >= 0):
            logger.info(f"MACD zero line bearish crossover for {symbol}: MACD={current_macd:.4f}")
            return 'sell'
        
        # Signal 6: Confirmation signals for sustained trends
        elif self.confirmation_period > 1:
            recent_macd = macd_line.tail(self.confirmation_period)
            recent_signal = signal_line.tail(self.confirmation_period)
            recent_histogram = histogram.tail(self.confirmation_period)
            
            # Sustained bullish trend
            if (all(m > s for m, s in zip(recent_macd, recent_signal)) and 
                all(h > self.histogram_threshold for h in recent_histogram)):
                logger.info(f"MACD sustained bullish for {symbol}")
                return 'buy'
            
            # Sustained bearish trend
            elif (all(m < s for m, s in zip(recent_macd, recent_signal)) and 
                  all(h < -self.histogram_threshold for h in recent_histogram)):
                logger.info(f"MACD sustained bearish for {symbol}")
                return 'sell'
        
        return 'hold'
    
    def _calculate_macd(self, prices: pd.Series) -> tuple:
        """Calculate MACD indicator components"""
        return calculate_macd(prices, self.fast_period, self.slow_period, self.signal_period)
    
    def validate_config(self) -> bool:
        """Validate strategy configuration"""
        if self.fast_period >= self.slow_period:
            logger.error("Fast period must be less than slow period")
            return False
        
        if self.signal_period <= 0:
            logger.error("Signal period must be positive")
            return False
        
        if self.fast_period <= 0 or self.slow_period <= 0:
            logger.error("Fast and slow periods must be positive")
            return False
        
        return True
    
    def get_indicators(self, price_data: pd.Series) -> Dict[str, pd.Series]:
        """Get strategy indicators for analysis"""
        return calculate_macd_components(price_data, self.fast_period, self.slow_period, self.signal_period)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get strategy summary with parameters"""
        summary = super().get_summary()
        summary.update({
            'fast_period': self.fast_period,
            'slow_period': self.slow_period,
            'signal_period': self.signal_period,
            'histogram_threshold': self.histogram_threshold,
            'confirmation_period': self.confirmation_period,
            'strategy_type': 'MACD'
        })
        return summary 