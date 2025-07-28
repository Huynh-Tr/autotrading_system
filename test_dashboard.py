#!/usr/bin/env python3
"""
Test script to verify dashboard functionality
"""

import sys
import os
import subprocess

def test_dashboard_imports():
    """Test that dashboard imports work correctly"""
    print("Testing dashboard imports...")
    
    # Add src to path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    src_path = os.path.join(current_dir, 'src')
    sys.path.insert(0, src_path)
    
    try:
        # Test core imports
        from utils.config_manager import ConfigManager
        print("✅ ConfigManager imported successfully")
        
        from core.trading_engine import TradingEngine
        print("✅ TradingEngine imported successfully")
        
        from strategies.sma_crossover import SMACrossoverStrategy
        print("✅ SMACrossoverStrategy imported successfully")
        
        from strategies.rsi_strategy import RSIStrategy
        print("✅ RSIStrategy imported successfully")
        
        from strategies.macd_strategy import MACDStrategy
        print("✅ MACDStrategy imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_config_loading():
    """Test that configuration can be loaded"""
    print("\nTesting configuration loading...")
    
    try:
        from utils.config_manager import ConfigManager
        config_path = os.path.join(os.path.dirname(__file__), "config", "config.yaml")
        
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        config = ConfigManager(config_path)
        print("✅ Configuration loaded successfully")
        
        # Test some config values
        symbols = config.get("trading.symbols")
        print(f"✅ Trading symbols: {symbols}")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        return False

def test_streamlit_availability():
    """Test that Streamlit is available"""
    print("\nTesting Streamlit availability...")
    
    try:
        import streamlit
        print("✅ Streamlit is available")
        return True
    except ImportError:
        print("❌ Streamlit not available")
        return False

def main():
    """Run all dashboard tests"""
    print("🧪 Testing Dashboard Functionality")
    print("=" * 40)
    
    tests = [
        ("Import Tests", test_dashboard_imports),
        ("Config Loading", test_config_loading),
        ("Streamlit Availability", test_streamlit_availability)
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
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard should work correctly.")
    else:
        print("⚠️ Some tests failed. Please fix the issues before running the dashboard.")

if __name__ == "__main__":
    main() 