"""
RSI Strategy - Relative Strength Index trading strategy
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from loguru import logger

from .base_strategy import BaseStrategy
from ..indicators.rsi import calculate_rsi, calculate_rsi_components


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
        Generate trading signals based on RSI
        
        Args:
            historical_data: Historical market data up to current point
            current_data: Current day's market data
            
        Returns:
            Dict mapping symbol to signal ('buy', 'sell', 'hold')
        """
        signals = {}
        
        # Process each symbol
        for symbol in historical_data.columns:
            if isinstance(symbol, str) and not symbol.endswith('_RSI'):
                # Get historical price data for this symbol
                symbol_data = historical_data[symbol].dropna()
                
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
        """Calculate RSI indicator"""
        return calculate_rsi(prices, period)
    
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
        return calculate_rsi_components(price_data, self.period, self.overbought_threshold, self.oversold_threshold)
    
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