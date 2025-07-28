#!/usr/bin/env python3
"""
Quick test to run dashboard and identify errors
"""

import subprocess
import sys
import os

def test_dashboard_run():
    """Test running the dashboard"""
    print("🚀 Testing dashboard run...")
    
    try:
        # Check if streamlit is available
        result = subprocess.run([
            sys.executable, "-m", "streamlit", "--version"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("❌ Streamlit not available")
            print(f"Error: {result.stderr}")
            return False
        
        print(f"✅ Streamlit version: {result.stdout.strip()}")
        
        # Test dashboard file
        dashboard_path = "dashboard/streamlit_app.py"
        if not os.path.exists(dashboard_path):
            print(f"❌ Dashboard file not found: {dashboard_path}")
            return False
        
        print("✅ Dashboard file exists")
        
        # Try to run dashboard in headless mode for a few seconds
        print("🔄 Testing dashboard startup...")
        
        process = subprocess.Popen([
            sys.executable, "-m", "streamlit", "run", dashboard_path,
            "--server.headless", "true",
            "--server.port", "8501"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Wait a few seconds for startup
        import time
        time.sleep(5)
        
        # Check if process is still running
        if process.poll() is None:
            print("✅ Dashboard started successfully")
            process.terminate()
            process.wait()
            return True
        else:
            stdout, stderr = process.communicate()
            print("❌ Dashboard failed to start")
            print(f"STDOUT: {stdout}")
            print(f"STDERR: {stderr}")
            return False
        
    except Exception as e:
        print(f"❌ Dashboard test error: {e}")
        return False

def test_dashboard_imports():
    """Test dashboard imports"""
    print("\n📦 Testing dashboard imports...")
    
    try:
        # Add src to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        src_path = os.path.join(project_root, 'src')
        sys.path.insert(0, src_path)
        
        # Test imports
        import streamlit as st
        print("✅ Streamlit imported")
        
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
        return False

def main():
    """Run dashboard tests"""
    print("🎯 Quick Dashboard Test")
    print("=" * 30)
    
    # Test imports first
    if not test_dashboard_imports():
        print("❌ Import tests failed")
        return
    
    # Test dashboard run
    if test_dashboard_run():
        print("🎉 Dashboard tests passed!")
    else:
        print("❌ Dashboard tests failed")

if __name__ == "__main__":
    main() 