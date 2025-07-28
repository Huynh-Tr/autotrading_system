#!/usr/bin/env python3
"""
Final comprehensive test to verify all components are working correctly
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add src to path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, 'src')
sys.path.insert(0, src_path)

def test_imports():
    """Test all imports"""
    print("🧪 Testing imports...")
    
    try:
        from utils.config_manager import ConfigManager
        from core.trading_engine import TradingEngine
        from strategies.sma_crossover import SMACrossoverStrategy
        from strategies.rsi_strategy import RSIStrategy
        from strategies.macd_strategy import MACDStrategy
        from data.data_manager import DataManager
        from risk.risk_manager import RiskManager
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n📋 Testing configuration loading...")
    
    try:
        config = ConfigManager("config/config.yaml")
        
        # Test key config values
        symbols = config.get("trading.symbols")
        initial_capital = config.get("trading.initial_capital")
        data_source = config.get("data.source")
        
        print(f"✅ Symbols: {symbols}")
        print(f"✅ Initial capital: ${initial_capital:,.2f}")
        print(f"✅ Data source: {data_source}")
        
        return True
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

def test_data_manager():
    """Test data manager functionality"""
    print("\n📊 Testing data manager...")
    
    try:
        config = ConfigManager("config/config.yaml")
        data_manager = DataManager(config)
        
        # Test data loading
        data = data_manager.get_historical_data(
            symbols=["VCB"],
            start_date="2024-01-01",
            end_date="2024-01-10",
            interval="1d"
        )
        
        print(f"✅ Data loaded: {data.shape}")
        print(f"✅ Columns: {list(data.columns)}")
        print(f"✅ Data range: {data.index[0]} to {data.index[-1]}")
        
        return True
    except Exception as e:
        print(f"❌ Data manager error: {e}")
        return False

def test_strategies():
    """Test strategy functionality"""
    print("\n🎯 Testing strategies...")
    
    try:
        # Test SMA strategy
        sma_config = {"short_window": 20, "long_window": 50}
        sma_strategy = SMACrossoverStrategy(sma_config)
        print("✅ SMA strategy created")
        
        # Test RSI strategy
        rsi_config = {"period": 14, "oversold": 30, "overbought": 70}
        rsi_strategy = RSIStrategy(rsi_config)
        print("✅ RSI strategy created")
        
        # Test MACD strategy
        macd_config = {"fast_period": 12, "slow_period": 26, "signal_period": 9}
        macd_strategy = MACDStrategy(macd_config)
        print("✅ MACD strategy created")
        
        return True
    except Exception as e:
        print(f"❌ Strategy error: {e}")
        return False

def test_trading_engine():
    """Test trading engine functionality"""
    print("\n⚙️ Testing trading engine...")
    
    try:
        engine = TradingEngine("config/config.yaml")
        print("✅ Trading engine created")
        
        # Test strategy addition
        config = ConfigManager("config/config.yaml")
        sma_config = config.get("strategies.sma_crossover", {})
        sma_strategy = SMACrossoverStrategy(sma_config)
        engine.add_strategy(sma_strategy)
        print("✅ Strategy added to engine")
        
        return True
    except Exception as e:
        print(f"❌ Trading engine error: {e}")
        return False

def test_backtest():
    """Test backtest functionality"""
    print("\n📈 Testing backtest...")
    
    try:
        # Clear previous results
        if os.path.exists('data/backtest_results.csv'):
            os.remove('data/backtest_results.csv')
        if os.path.exists('data/trades.csv'):
            os.remove('data/trades.csv')
        
        # Run backtest
        from src.main import run_backtest
        run_backtest("config/config.yaml", "sma_crossover")
        
        # Check results
        if os.path.exists('data/backtest_results.csv'):
            results_df = pd.read_csv('data/backtest_results.csv')
            print(f"✅ Backtest results: {len(results_df)} data points")
            
            # Check if portfolio value changed
            initial_value = results_df['portfolio_value'].iloc[0]
            final_value = results_df['portfolio_value'].iloc[-1]
            print(f"✅ Portfolio change: ${final_value - initial_value:,.2f}")
            
            if final_value != initial_value:
                print("✅ Portfolio value changed - backtest working!")
            else:
                print("⚠️ Portfolio value didn't change")
        else:
            print("❌ No backtest results file")
            return False
        
        # Check trades
        if os.path.exists('data/trades.csv'):
            trades_df = pd.read_csv('data/trades.csv')
            print(f"✅ Trades executed: {len(trades_df)}")
            return True
        else:
            print("❌ No trades file")
            return False
        
    except Exception as e:
        print(f"❌ Backtest error: {e}")
        return False

def test_dashboard_components():
    """Test dashboard components"""
    print("\n🌐 Testing dashboard components...")
    
    try:
        # Test streamlit availability
        import streamlit
        print("✅ Streamlit available")
        
        # Test plotly availability
        import plotly
        print("✅ Plotly available")
        
        # Test dashboard file exists
        dashboard_path = "dashboard/streamlit_app.py"
        if os.path.exists(dashboard_path):
            print("✅ Dashboard file exists")
        else:
            print("❌ Dashboard file missing")
            return False
        
        return True
    except ImportError as e:
        print(f"❌ Dashboard component error: {e}")
        return False

def main():
    """Run all tests"""
    print("🎯 Final Comprehensive Test")
    print("=" * 50)
    
    tests = [
        ("Import Tests", test_imports),
        ("Config Loading", test_config_loading),
        ("Data Manager", test_data_manager),
        ("Strategies", test_strategies),
        ("Trading Engine", test_trading_engine),
        ("Backtest", test_backtest),
        ("Dashboard Components", test_dashboard_components)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        if test_func():
            passed += 1
            print(f"✅ {test_name} passed")
        else:
            print(f"❌ {test_name} failed")
    
    print(f"\n📊 Final Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is working correctly.")
        print("\n🚀 You can now:")
        print("  - Run backtests: python run_backtest.py")
        print("  - Launch dashboard: python launch_dashboard.py")
        print("  - Run comprehensive tests: python comprehensive_test.py")
    else:
        print("⚠️ Some tests failed. Please check the errors above.")
        print(f"Failed tests: {total - passed}")

if __name__ == "__main__":
    main() 