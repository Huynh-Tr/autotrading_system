"""
RSI Strategy - Relative Strength Index trading strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from loguru import logger

from .base_strategy import BaseStrategy


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
    
    def generate_signals(self, data: pd.Series) -> Dict[str, str]:
        """
        Generate trading signals based on RSI
        
        Args:
            data: Series containing price data with RSI indicators
            
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
                if isinstance(symbol, str) and not symbol.endswith('_RSI'):
                    signal = self._generate_signal_for_symbol(data[symbol], symbol)
                    signals[symbol] = signal
        
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
        """Calculate RSI indicator"""
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
    
    def validate_config(self) -> bool:
        """Validate strategy configuration"""
        if self.period <= 0:
            logger.error("RSI period must be positive")
            return False
        
        if self.overbought_threshold <= self.oversold_threshold:
            logger.error("Overbought threshold must be greater than oversold threshold")
            return False
        
        if self.overbought_threshold > 100 or self.oversold_threshold < 0:
            logger.error("RSI thresholds must be between 0 and 100")
            return False
        
        return True
    
    def get_indicators(self, price_data: pd.Series) -> Dict[str, pd.Series]:
        """Get strategy indicators for analysis"""
        rsi = self._calculate_rsi(price_data, self.period)
        
        return {
            'rsi': rsi,
            'overbought_line': pd.Series([self.overbought_threshold] * len(price_data), index=price_data.index),
            'oversold_line': pd.Series([self.oversold_threshold] * len(price_data), index=price_data.index)
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get strategy summary with parameters"""
        summary = super().get_summary()
        summary.update({
            'period': self.period,
            'overbought_threshold': self.overbought_threshold,
            'oversold_threshold': self.oversold_threshold,
            'confirmation_period': self.confirmation_period,
            'strategy_type': 'RSI'
        })
        return summary 