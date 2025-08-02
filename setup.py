#!/usr/bin/env python3
"""
Setup script for Auto Trading System
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
        print(f"[OK] Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("[OK] Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = [
        "data",
        "data/cache",
        "logs",
        "config"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"[OK] Created directory: {directory}")

def setup_config():
    """Setup configuration file"""
    print("\nSetting up configuration...")
    
    config_file = "config/config.yaml"
    
    if os.path.exists(config_file):
        print(f"[OK] Configuration file already exists: {config_file}")
        return True
    
    try:
        # Create a basic config file
        config_content = """# Auto Trading System Configuration

# Trading settings
trading:
  initial_capital: 100000
  symbols: ["Bitstamp:BTCUSD"]
  commission: 0.001
  max_position_size: 0.2
  max_portfolio_risk: 0.05

# Data settings
data:
  source: "vnstock"
  start_date: "2024-01-01"
  end_date: "2024-05-31"
  interval: "1d"
  cache_enabled: true

# Risk management
risk:
  stop_loss: 0.05
  take_profit: 0.15
  max_drawdown: 0.20

# Strategies
strategies:
  sma_crossover:
    enabled: true
    short_window: 20
    long_window: 50
    
  rsi:
    enabled: true
    period: 14
    oversold: 30
    overbought: 70
    
  macd:
    enabled: true
    fast_period: 12
    slow_period: 26
    signal_period: 9

# Logging
logging:
  level: "INFO"
  file: "logs/trading.log"
  max_size: 10485760  # 10MB
  backup_count: 5
"""
        
        with open(config_file, 'w') as f:
            f.write(config_content)
        
        print(f"[OK] Configuration file created: {config_file}")
        print("  Please edit the configuration file with your settings")
        return True
    except Exception as e:
        print(f"❌ Failed to create configuration file: {e}")
        return False

def setup_viz_module():
    """Setup visualization module with parameters"""
    print("\nSetting up visualization module...")
    
    viz_file = "src/viz/viz.py"
    
    if not os.path.exists(viz_file):
        print(f"❌ Visualization module not found: {viz_file}")
        return False
    
    try:
        # Check if plotly is available
        import plotly
        print("[OK] Plotly is available for visualization")
        
        # Create viz parameters documentation
        viz_params = {
            "symbol": "Trading symbol (e.g., 'BTC/USD')",
            "trades_df": "DataFrame containing trade data with columns: Timestamp, Side, Price",
            "historical_data": "DataFrame containing OHLCV data with columns: open, high, low, close, volume"
        }
        
        print("[OK] Visualization module parameters:")
        for param, description in viz_params.items():
            print(f"  - {param}: {description}")
        
        print("[OK] Visualization module is ready to use")
        return True
        
    except ImportError:
        print("❌ Plotly not available. Installing plotly...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
            print("[OK] Plotly installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install plotly: {e}")
            return False
    except Exception as e:
        print(f"❌ Failed to setup visualization module: {e}")
        return False

def run_tests():
    """Run system tests"""
    print("\nRunning system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[OK] All tests passed")
            return True
        else:
            print("❌ Some tests failed")
            print("Test output:")
            print(result.stdout)
            if result.stderr:
                print("Errors:")
                print(result.stderr)
            return False
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def main():
    """Main setup function"""
    print("Auto Trading System - Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Setup configuration
    if not setup_config():
        return False
    
    # Setup visualization module
    if not setup_viz_module():
        print("\n⚠️  Visualization module setup failed, but setup completed. You may need to install plotly manually.")
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Tests failed, but setup completed. You may need to fix issues manually.")
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit config/config.yaml with your trading parameters")
    print("2. Run backtest: python3 src/main.py --mode backtest --strategy sma_crossover")
    print("3. Launch dashboard: python3 run_dashboard.py")
    print("4. Open http://localhost:8501 in your browser")
    print("5. Use visualization: from src.viz.viz import viz")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 