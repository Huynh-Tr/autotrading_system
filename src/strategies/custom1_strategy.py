"""
Custom1 Strategy - Combined SMA and RSI Strategy

This strategy combines SMA crossover signals with RSI momentum
to create a more robust trading approach.
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
    from ..indicators.sma_rsi_combined import (
        calculate_sma_rsi_combined, 
        get_sma_rsi_signal_strength,
        validate_sma_rsi_parameters
    )
except ImportError:
    # Fallback to absolute imports
    from src.strategies.base_strategy import BaseStrategy
    from src.indicators.sma_rsi_combined import (
        calculate_sma_rsi_combined, 
        get_sma_rsi_signal_strength,
        validate_sma_rsi_parameters
    )


class Custom1Strategy(BaseStrategy):
    """Custom1 Strategy - Combined SMA and RSI Strategy"""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize Custom1 strategy"""
        super().__init__("custom1", config)
        
        # Strategy parameters
        self.short_window = config.get('short_window', 20)
        self.long_window = config.get('long_window', 50)
        self.rsi_period = config.get('rsi_period', 14)
        self.rsi_oversold = config.get('rsi_oversold', 30)
        self.rsi_overbought = config.get('rsi_overbought', 70)
        self.confirmation_period = config.get('confirmation_period', 2)
        self.min_signal_strength = config.get('min_signal_strength', 0.3)
        
        # Validate parameters
        if not validate_sma_rsi_parameters(self.short_window, self.long_window, self.rsi_period):
            raise ValueError("Invalid SMA+RSI parameters")
        
        logger.info(f"Custom1 Strategy initialized: SMA({self.short_window}/{self.long_window}) + RSI({self.rsi_period})")
    
    def generate_signals(self, historical_data: pd.DataFrame, current_data: pd.Series) -> Dict[str, str]:
        """
        Generate trading signals based on combined SMA and RSI analysis
        
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
        """Generate signal for a single symbol using combined SMA+RSI analysis"""
        if len(price_data) < self.long_window:
            return 'hold'  # Not enough data
        
        # Calculate combined indicators
        indicators = calculate_sma_rsi_combined(
            price_data, 
            self.short_window, 
            self.long_window, 
            self.rsi_period
        )
        
        # Get signal strength
        signal_strength = get_sma_rsi_signal_strength(
            price_data, 
            self.short_window, 
            self.long_window, 
            self.rsi_period
        )
        
        # Get latest values
        latest_combined_signal = signal_strength['combined_signal']
        latest_rsi = signal_strength['rsi_value']
        latest_sma_crossover = signal_strength['sma_crossover_value']
        total_strength = signal_strength['total_strength']
        
        # Check if signal strength is sufficient
        if total_strength < self.min_signal_strength:
            return 'hold'
        
        # Generate signals based on combined analysis
        if latest_combined_signal >= 1:  # Strong buy signal
            # Check for confirmation: RSI not overbought and SMA trend positive
            if latest_rsi < self.rsi_overbought and latest_sma_crossover > 0:
                logger.info(f"Strong buy signal for {symbol}: SMA+RSI combined")
                return 'buy'
        
        elif latest_combined_signal <= -1:  # Strong sell signal
            # Check for confirmation: RSI not oversold and SMA trend negative
            if latest_rsi > self.rsi_oversold and latest_sma_crossover < 0:
                logger.info(f"Strong sell signal for {symbol}: SMA+RSI combined")
                return 'sell'
        
        # Check for individual signal confirmation
        elif signal_strength['sma_signal'] == 1 and latest_rsi < 60:
            # SMA buy signal with RSI not overbought
            logger.info(f"Buy signal for {symbol}: SMA crossover with RSI confirmation")
            return 'buy'
        
        elif signal_strength['sma_signal'] == -1 and latest_rsi > 40:
            # SMA sell signal with RSI not oversold
            logger.info(f"Sell signal for {symbol}: SMA crossover with RSI confirmation")
            return 'sell'
        
        elif signal_strength['rsi_signal'] == 1 and latest_sma_crossover > -0.01:
            # RSI oversold with SMA trend not strongly negative
            logger.info(f"Buy signal for {symbol}: RSI oversold with SMA confirmation")
            return 'buy'
        
        elif signal_strength['rsi_signal'] == -1 and latest_sma_crossover < 0.01:
            # RSI overbought with SMA trend not strongly positive
            logger.info(f"Sell signal for {symbol}: RSI overbought with SMA confirmation")
            return 'sell'
        
        return 'hold'
    
    def get_indicators(self, price_data: pd.Series) -> Dict[str, pd.Series]:
        """Get strategy indicators for the given price data"""
        if len(price_data) < self.long_window:
            return {}
        
        # Calculate combined indicators
        indicators = calculate_sma_rsi_combined(
            price_data, 
            self.short_window, 
            self.long_window, 
            self.rsi_period
        )
        
        return {
            f'{self.short_window}_SMA': indicators['short_sma'],
            f'{self.long_window}_SMA': indicators['long_sma'],
            'RSI': indicators['rsi'],
            'SMA_Signal': indicators['sma_signal'],
            'RSI_Signal': indicators['rsi_signal'],
            'Combined_Signal': indicators['combined_signal'],
            'SMA_Crossover': indicators['sma_crossover']
        }
    
    def get_signal_details(self, price_data: pd.Series) -> Dict[str, Any]:
        """Get detailed signal information for analysis"""
        if len(price_data) < self.long_window:
            return {}
        
        signal_strength = get_sma_rsi_signal_strength(
            price_data, 
            self.short_window, 
            self.long_window, 
            self.rsi_period
        )
        
        return {
            'strategy_name': self.name,
            'parameters': {
                'short_window': self.short_window,
                'long_window': self.long_window,
                'rsi_period': self.rsi_period,
                'rsi_oversold': self.rsi_oversold,
                'rsi_overbought': self.rsi_overbought
            },
            'current_signals': signal_strength,
            'signal_interpretation': self._interpret_signals(signal_strength)
        }
    
    def _interpret_signals(self, signal_strength: Dict[str, float]) -> str:
        """Interpret the current signal strength"""
        combined_signal = signal_strength['combined_signal']
        rsi_value = signal_strength['rsi_value']
        total_strength = signal_strength['total_strength']
        
        if combined_signal >= 1 and total_strength >= self.min_signal_strength:
            return "Strong Buy - SMA Golden Cross + RSI Oversold"
        elif combined_signal <= -1 and total_strength >= self.min_signal_strength:
            return "Strong Sell - SMA Death Cross + RSI Overbought"
        elif signal_strength['sma_signal'] == 1 and rsi_value < 60:
            return "Buy - SMA Golden Cross with RSI Confirmation"
        elif signal_strength['sma_signal'] == -1 and rsi_value > 40:
            return "Sell - SMA Death Cross with RSI Confirmation"
        elif signal_strength['rsi_signal'] == 1:
            return "Buy - RSI Oversold with SMA Trend Support"
        elif signal_strength['rsi_signal'] == -1:
            return "Sell - RSI Overbought with SMA Trend Resistance"
        else:
            return "Hold - No Clear Signal" 