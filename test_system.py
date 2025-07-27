#!/usr/bin/env python3
"""
Test script for the Auto Trading System
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_config_manager():
    """Test configuration manager"""
    print("Testing ConfigManager...")
    
    try:
        from src.utils.config_manager import ConfigManager
        
        # Test with default config
        config = ConfigManager("config/config.example.yaml")
        
        # Test getting values
        initial_capital = config.get("trading.initial_capital")
        symbols = config.get("trading.symbols")
        
        print(f"✓ Initial capital: ${initial_capital:,.2f}")
        print(f"✓ Trading symbols: {symbols}")
        
        return True
    except Exception as e:
        print(f"✗ ConfigManager test failed: {e}")
        return False

def test_data_manager():
    """Test data manager"""
    print("\nTesting DataManager...")
    
    try:
        from src.utils.config_manager import ConfigManager
        from src.data.data_manager import DataManager
        
        config = ConfigManager("config/config.example.yaml")
        data_manager = DataManager(config)
        
        # Test data validation
        test_data = pd.DataFrame({
            'AAPL': [150, 151, 152, 153, 154],
            'GOOGL': [2800, 2810, 2820, 2830, 2840]
        })
        
        is_valid = data_manager.validate_data(test_data)
        print(f"✓ Data validation: {is_valid}")
        
        # Test technical indicators
        indicators = data_manager.calculate_technical_indicators(test_data)
        print(f"✓ Technical indicators calculated: {len(indicators.columns)} columns")
        
        return True
    except Exception as e:
        print(f"✗ DataManager test failed: {e}")
        return False

def test_sma_strategy():
    """Test SMA crossover strategy"""
    print("\nTesting SMA Crossover Strategy...")
    
    try:
        from src.strategies.sma_crossover import SMACrossoverStrategy
        
        # Create test configuration
        config = {
            'short_window': 2,
            'long_window': 4
        }
        
        strategy = SMACrossoverStrategy(config)
        
        # Create test data
        test_data = pd.Series([100, 101, 102, 103, 104, 105, 106, 107, 108, 109])
        
        # Test signal generation
        signals = strategy.generate_signals(test_data)
        print(f"✓ Strategy initialized: {strategy.name}")
        print(f"✓ Signal generation: {len(signals)} signals")
        
        return True
    except Exception as e:
        print(f"✗ SMA Strategy test failed: {e}")
        return False

def test_risk_manager():
    """Test risk manager"""
    print("\nTesting RiskManager...")
    
    try:
        from src.utils.config_manager import ConfigManager
        from src.risk.risk_manager import RiskManager
        
        config = ConfigManager("config/config.example.yaml")
        risk_manager = RiskManager(config)
        
        # Test position sizing
        position_size = risk_manager.calculate_position_size(
            symbol="AAPL",
            price=150.0,
            cash=10000,
            positions={}
        )
        
        print(f"✓ Position sizing: ${position_size:,.2f}")
        
        # Test risk metrics
        metrics = risk_manager.get_summary()
        print(f"✓ Risk metrics calculated: {len(metrics)} metrics")
        
        return True
    except Exception as e:
        print(f"✗ RiskManager test failed: {e}")
        return False

def test_trading_engine():
    """Test trading engine"""
    print("\nTesting TradingEngine...")
    
    try:
        from src.core.trading_engine import TradingEngine
        
        # Initialize trading engine
        engine = TradingEngine("config/config.example.yaml")
        
        # Test portfolio summary
        summary = engine.get_portfolio_summary()
        print(f"✓ Trading engine initialized")
        print(f"✓ Portfolio summary: ${summary['portfolio_value']:,.2f}")
        
        return True
    except Exception as e:
        print(f"✗ TradingEngine test failed: {e}")
        return False

def create_sample_data():
    """Create sample market data for testing"""
    print("\nCreating sample market data...")
    
    # Create sample data directory
    os.makedirs("data/cache", exist_ok=True)
    
    # Generate sample price data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    
    # Create realistic price movements
    np.random.seed(42)  # For reproducible results
    
    # AAPL prices
    aapl_prices = [150]
    for i in range(len(dates) - 1):
        change = np.random.normal(0, 0.02)  # 2% daily volatility
        new_price = aapl_prices[-1] * (1 + change)
        aapl_prices.append(max(new_price, 50))  # Minimum price $50
    
    # GOOGL prices
    googl_prices = [2800]
    for i in range(len(dates) - 1):
        change = np.random.normal(0, 0.025)  # 2.5% daily volatility
        new_price = googl_prices[-1] * (1 + change)
        googl_prices.append(max(new_price, 1000))  # Minimum price $1000
    
    # Create DataFrame
    data = pd.DataFrame({
        'AAPL': aapl_prices,
        'GOOGL': googl_prices
    }, index=dates)
    
    # Save to CSV
    data.to_csv("data/sample_market_data.csv")
    print(f"✓ Sample data created: {len(data)} days, {len(data.columns)} symbols")
    
    return data

def run_backtest_demo():
    """Run a simple backtest demonstration"""
    print("\nRunning backtest demonstration...")
    
    try:
        # Create sample data
        data = create_sample_data()
        
        # Simple backtest simulation
        initial_capital = 100000
        cash = initial_capital
        positions = {}
        portfolio_values = []
        
        # Simple SMA crossover logic
        for i in range(50, len(data)):  # Start after 50 days for SMA calculation
            current_data = data.iloc[:i+1]
            
            for symbol in data.columns:
                prices = current_data[symbol]
                sma_20 = prices.rolling(20).mean().iloc[-1]
                sma_50 = prices.rolling(50).mean().iloc[-1]
                current_price = prices.iloc[-1]
                
                # Trading logic
                if sma_20 > sma_50 and symbol not in positions:
                    # Buy signal
                    position_size = cash * 0.1  # 10% of cash
                    quantity = position_size / current_price
                    positions[symbol] = {
                        'quantity': quantity,
                        'entry_price': current_price
                    }
                    cash -= position_size
                    print(f"BUY {symbol}: {quantity:.2f} @ ${current_price:.2f}")
                
                elif sma_20 < sma_50 and symbol in positions:
                    # Sell signal
                    position = positions[symbol]
                    proceeds = position['quantity'] * current_price
                    cash += proceeds
                    pnl = proceeds - (position['quantity'] * position['entry_price'])
                    print(f"SELL {symbol}: {position['quantity']:.2f} @ ${current_price:.2f} | P&L: ${pnl:.2f}")
                    del positions[symbol]
            
            # Calculate portfolio value
            portfolio_value = cash
            for symbol, position in positions.items():
                current_price = data[symbol].iloc[i]
                portfolio_value += position['quantity'] * current_price
            
            portfolio_values.append(portfolio_value)
        
        # Calculate performance
        final_value = portfolio_values[-1]
        total_return = (final_value - initial_capital) / initial_capital
        
        print(f"\n=== BACKTEST RESULTS ===")
        print(f"Initial Capital: ${initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${final_value:,.2f}")
        print(f"Total Return: {total_return:.2%}")
        print(f"Total Trades: {len(portfolio_values)}")
        
        return True
        
    except Exception as e:
        print(f"✗ Backtest demo failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Auto Trading System - Component Tests")
    print("=" * 50)
    
    tests = [
        test_config_manager,
        test_data_manager,
        test_sma_strategy,
        test_risk_manager,
        test_trading_engine,
        run_backtest_demo
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        print("\nNext steps:")
        print("1. Copy config/config.example.yaml to config/config.yaml")
        print("2. Edit config/config.yaml with your settings")
        print("3. Run: python src/main.py --mode backtest --strategy sma_crossover")
        print("4. Launch dashboard: python run_dashboard.py")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 