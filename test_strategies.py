#!/usr/bin/env python3
"""
Test script for RSI and MACD strategies
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.utils.config_manager import ConfigManager


def test_rsi_strategy():
    """Test RSI strategy"""
    print("Testing RSI Strategy...")
    print("=" * 40)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data with some trends
    prices = []
    current_price = 100
    
    for _ in range(len(dates)):
        # Add some trend and volatility
        change = np.random.normal(0.001, 0.02)  # Daily return
        current_price *= (1 + change)
        current_price = max(current_price, 10)  # Minimum price
        prices.append(current_price)
    
    price_data = pd.Series(prices, index=dates, name='AAPL')
    
    # Test different RSI configurations
    configs = [
        {'period': 14, 'overbought_threshold': 70, 'oversold_threshold': 30},
        {'period': 10, 'overbought_threshold': 75, 'oversold_threshold': 25},
        {'period': 20, 'overbought_threshold': 65, 'oversold_threshold': 35}
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\nTest {i}: RSI Configuration")
        print(f"Period: {config['period']}, Overbought: {config['overbought_threshold']}, Oversold: {config['oversold_threshold']}")
        
        try:
            # Initialize strategy
            strategy = RSIStrategy(config)
            
            # Generate signals
            signals = strategy.generate_signals(price_data)
            
            # Get indicators
            indicators = strategy.get_indicators(price_data)
            
            # Count signals
            buy_signals = sum(1 for signal in signals.values() if signal == 'buy')
            sell_signals = sum(1 for signal in signals.values() if signal == 'sell')
            hold_signals = sum(1 for signal in signals.values() if signal == 'hold')
            
            print(f"✓ Strategy initialized successfully")
            print(f"✓ Signals generated: {len(signals)}")
            print(f"  - Buy signals: {buy_signals}")
            print(f"  - Sell signals: {sell_signals}")
            print(f"  - Hold signals: {hold_signals}")
            print(f"✓ Indicators calculated: {len(indicators)}")
            
            # Show some RSI values
            rsi_values = indicators['rsi'].dropna()
            if len(rsi_values) > 0:
                print(f"✓ RSI range: {rsi_values.min():.2f} - {rsi_values.max():.2f}")
                print(f"✓ Current RSI: {rsi_values.iloc[-1]:.2f}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "=" * 40)


def test_macd_strategy():
    """Test MACD strategy"""
    print("Testing MACD Strategy...")
    print("=" * 40)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data with trends
    prices = []
    current_price = 100
    
    for _ in range(len(dates)):
        # Add trend and volatility
        change = np.random.normal(0.001, 0.02)
        current_price *= (1 + change)
        current_price = max(current_price, 10)
        prices.append(current_price)
    
    price_data = pd.Series(prices, index=dates, name='AAPL')
    
    # Test different MACD configurations
    configs = [
        {'fast_period': 12, 'slow_period': 26, 'signal_period': 9},
        {'fast_period': 8, 'slow_period': 21, 'signal_period': 5},
        {'fast_period': 15, 'slow_period': 30, 'signal_period': 12}
    ]
    
    for i, config in enumerate(configs, 1):
        print(f"\nTest {i}: MACD Configuration")
        print(f"Fast: {config['fast_period']}, Slow: {config['slow_period']}, Signal: {config['signal_period']}")
        
        try:
            # Initialize strategy
            strategy = MACDStrategy(config)
            
            # Generate signals
            signals = strategy.generate_signals(price_data)
            
            # Get indicators
            indicators = strategy.get_indicators(price_data)
            
            # Count signals
            buy_signals = sum(1 for signal in signals.values() if signal == 'buy')
            sell_signals = sum(1 for signal in signals.values() if signal == 'sell')
            hold_signals = sum(1 for signal in signals.values() if signal == 'hold')
            
            print(f"✓ Strategy initialized successfully")
            print(f"✓ Signals generated: {len(signals)}")
            print(f"  - Buy signals: {buy_signals}")
            print(f"  - Sell signals: {sell_signals}")
            print(f"  - Hold signals: {hold_signals}")
            print(f"✓ Indicators calculated: {len(indicators)}")
            
            # Show some MACD values
            macd_line = indicators['macd_line'].dropna()
            signal_line = indicators['signal_line'].dropna()
            histogram = indicators['histogram'].dropna()
            
            if len(macd_line) > 0:
                print(f"✓ MACD line range: {macd_line.min():.4f} - {macd_line.max():.4f}")
                print(f"✓ Current MACD: {macd_line.iloc[-1]:.4f}")
                print(f"✓ Current Signal: {signal_line.iloc[-1]:.4f}")
                print(f"✓ Current Histogram: {histogram.iloc[-1]:.4f}")
            
        except Exception as e:
            print(f"✗ Error: {e}")
    
    print("\n" + "=" * 40)


def test_strategy_integration():
    """Test strategy integration with main system"""
    print("Testing Strategy Integration...")
    print("=" * 40)
    
    try:
        # Load configuration
        config = ConfigManager("config/config.example.yaml")
        
        # Test RSI strategy with config
        rsi_config = config.get("strategies.rsi", {})
        rsi_strategy = RSIStrategy(rsi_config)
        print("✓ RSI strategy loaded from config")
        
        # Test MACD strategy with config
        macd_config = config.get("strategies.macd", {})
        macd_strategy = MACDStrategy(macd_config)
        print("✓ MACD strategy loaded from config")
        
        # Test strategy summaries
        rsi_summary = rsi_strategy.get_summary()
        macd_summary = macd_strategy.get_summary()
        
        print(f"✓ RSI strategy type: {rsi_summary['strategy_type']}")
        print(f"✓ MACD strategy type: {macd_summary['strategy_type']}")
        
        print("✓ All strategies integrated successfully")
        
    except Exception as e:
        print(f"✗ Integration error: {e}")
    
    print("\n" + "=" * 40)


def main():
    """Run all strategy tests"""
    print("Strategy Tests")
    print("=" * 50)
    
    # Test RSI strategy
    test_rsi_strategy()
    
    # Test MACD strategy
    test_macd_strategy()
    
    # Test integration
    test_strategy_integration()
    
    print("\n🎉 Strategy tests completed!")
    print("\nTo run backtests with new strategies:")
    print("1. python src/main.py --mode backtest --strategy rsi")
    print("2. python src/main.py --mode backtest --strategy macd")
    print("3. python src/main.py --mode backtest --strategy sma_crossover")


if __name__ == "__main__":
    main() 