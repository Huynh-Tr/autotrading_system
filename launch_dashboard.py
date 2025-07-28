#!/usr/bin/env python3
"""
Comprehensive Dashboard Launcher with Error Handling
"""

import subprocess
import sys
import os
import importlib

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        ('streamlit', 'streamlit'),
        ('plotly', 'plotly'),
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('yfinance', 'yfinance'),
        ('vnstock', 'vnstock')
    ]
    
    missing_packages = []
    
    for package_name, import_name in required_packages:
        try:
            importlib.import_module(import_name)
            print(f"✅ {package_name}")
        except ImportError:
            print(f"❌ {package_name} - missing")
            missing_packages.append(package_name)
    
    if missing_packages:
        print(f"\n⚠️ Missing packages: {', '.join(missing_packages)}")
        print("Installing missing packages...")
        
        for package in missing_packages:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ Installed {package}")
            except subprocess.CalledProcessError:
                print(f"❌ Failed to install {package}")
                return False
    
    return True

def check_files():
    """Check if required files exist"""
    print("\n📁 Checking required files...")
    
    required_files = [
        ("dashboard/streamlit_app.py", "Dashboard application"),
        ("config/config.yaml", "Configuration file"),
        ("src/", "Source code directory")
    ]
    
    missing_files = []
    
    for file_path, description in required_files:
        if os.path.exists(file_path):
            print(f"✅ {description}: {file_path}")
        else:
            print(f"❌ {description}: {file_path} - missing")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
        print("Please ensure you're running this script from the project root directory.")
        return False
    
    return True

def test_imports():
    """Test that the dashboard can import required modules"""
    print("\n🧪 Testing imports...")
    
    try:
        # Add src to path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        src_path = os.path.join(current_dir, 'src')
        sys.path.insert(0, src_path)
        
        # Test imports
        from utils.config_manager import ConfigManager
        print("✅ ConfigManager imported")
        
        from core.trading_engine import TradingEngine
        print("✅ TradingEngine imported")
        
        from strategies.sma_crossover import SMACrossoverStrategy
        print("✅ SMACrossoverStrategy imported")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def launch_dashboard():
    """Launch the Streamlit dashboard"""
    print("\n🚀 Launching dashboard...")
    
    dashboard_path = os.path.join("dashboard", "streamlit_app.py")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

def main():
    """Main launcher function"""
    print("🎯 Auto Trading System Dashboard Launcher")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Dependency check failed")
        return False
    
    # Check files
    if not check_files():
        print("❌ File check failed")
        return False
    
    # Test imports
    if not test_imports():
        print("❌ Import test failed")
        return False
    
    print("\n✅ All checks passed!")
    print("🌐 Starting dashboard...")
    print("📊 Dashboard will open at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop")
    
    return launch_dashboard()

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Dashboard launch failed. Please check the errors above.")
        sys.exit(1) 