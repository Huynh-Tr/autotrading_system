#!/usr/bin/env python3
"""
Test script to identify and fix dashboard errors
"""

import sys
import os
import traceback

def test_dashboard_imports():
    """Test dashboard imports step by step"""
    print("🔍 Testing dashboard imports...")
    
    # Test streamlit
    try:
        import streamlit as st
        print("✅ Streamlit imported")
    except ImportError as e:
        print(f"❌ Streamlit import error: {e}")
        return False
    
    # Test plotly
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✅ Plotly imported")
    except ImportError as e:
        print(f"❌ Plotly import error: {e}")
        return False
    
    # Test pandas and numpy
    try:
        import pandas as pd
        import numpy as np
        print("✅ Pandas and NumPy imported")
    except ImportError as e:
        print(f"❌ Pandas/NumPy import error: {e}")
        return False
    
    return True

def test_path_setup():
    """Test path setup for dashboard"""
    print("\n📁 Testing path setup...")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        src_path = os.path.join(project_root, 'src')
        
        print(f"✅ Current directory: {current_dir}")
        print(f"✅ Project root: {project_root}")
        print(f"✅ Source path: {src_path}")
        
        if os.path.exists(src_path):
            print("✅ Source directory exists")
        else:
            print("❌ Source directory not found")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Path setup error: {e}")
        return False

def test_src_imports():
    """Test imports from src directory"""
    print("\n📦 Testing src imports...")
    
    try:
        # Add src to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        src_path = os.path.join(project_root, 'src')
        sys.path.insert(0, src_path)
        
        # Test imports
        from utils.config_manager import ConfigManager
        print("✅ ConfigManager imported")
        
        from core.trading_engine import TradingEngine
        print("✅ TradingEngine imported")
        
        from strategies.sma_crossover import SMACrossoverStrategy
        print("✅ SMACrossoverStrategy imported")
        
        from strategies.rsi_strategy import RSIStrategy
        print("✅ RSIStrategy imported")
        
        from strategies.macd_strategy import MACDStrategy
        print("✅ MACDStrategy imported")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def test_config_loading():
    """Test configuration loading"""
    print("\n⚙️ Testing config loading...")
    
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, "config", "config.yaml")
        
        if not os.path.exists(config_path):
            print(f"❌ Config file not found: {config_path}")
            return False
        
        from utils.config_manager import ConfigManager
        config = ConfigManager(config_path)
        print("✅ Config loaded successfully")
        
        # Test some config values
        symbols = config.get("trading.symbols")
        print(f"✅ Symbols: {symbols}")
        
        return True
    except Exception as e:
        print(f"❌ Config loading error: {e}")
        print(f"Full traceback: {traceback.format_exc()}")
        return False

def test_dashboard_file():
    """Test dashboard file structure"""
    print("\n📄 Testing dashboard file...")
    
    try:
        dashboard_path = "dashboard/streamlit_app.py"
        if os.path.exists(dashboard_path):
            print("✅ Dashboard file exists")
            
            # Check file size
            file_size = os.path.getsize(dashboard_path)
            print(f"✅ File size: {file_size} bytes")
            
            # Check if file is readable
            with open(dashboard_path, 'r') as f:
                first_line = f.readline().strip()
                print(f"✅ First line: {first_line}")
            
            return True
        else:
            print(f"❌ Dashboard file not found: {dashboard_path}")
            return False
    except Exception as e:
        print(f"❌ Dashboard file error: {e}")
        return False

def test_streamlit_run():
    """Test if streamlit can run the dashboard"""
    print("\n🚀 Testing streamlit run...")
    
    try:
        import subprocess
        import sys
        
        # Test if streamlit is available
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "--version"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Streamlit version: {result.stdout.strip()}")
        else:
            print(f"❌ Streamlit not available: {result.stderr}")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Streamlit test error: {e}")
        return False

def main():
    """Run all dashboard tests"""
    print("🎯 Dashboard Error Detection Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_dashboard_imports),
        ("Path Setup", test_path_setup),
        ("SRC Imports", test_src_imports),
        ("Config Loading", test_config_loading),
        ("Dashboard File", test_dashboard_file),
        ("Streamlit Run", test_streamlit_run)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} passed")
            else:
                print(f"❌ {test_name} failed")
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            print(f"Full traceback: {traceback.format_exc()}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Dashboard should work correctly.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
        print(f"Failed tests: {total - passed}")

if __name__ == "__main__":
    main() 