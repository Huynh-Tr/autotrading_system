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
    print(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\nInstalling dependencies...")
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
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
        print(f"✓ Created directory: {directory}")

def setup_config():
    """Setup configuration file"""
    print("\nSetting up configuration...")
    
    config_example = "config/config.example.yaml"
    config_file = "config/config.yaml"
    
    if not os.path.exists(config_example):
        print(f"❌ Configuration example not found: {config_example}")
        return False
    
    if os.path.exists(config_file):
        print(f"✓ Configuration file already exists: {config_file}")
        return True
    
    try:
        shutil.copy(config_example, config_file)
        print(f"✓ Configuration file created: {config_file}")
        print("  Please edit the configuration file with your settings")
        return True
    except Exception as e:
        print(f"❌ Failed to create configuration file: {e}")
        return False

def run_tests():
    """Run system tests"""
    print("\nRunning system tests...")
    
    try:
        result = subprocess.run([sys.executable, "test_system.py"], 
                              capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✓ All tests passed")
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
    
    # Run tests
    if not run_tests():
        print("\n⚠️  Tests failed, but setup completed. You may need to fix issues manually.")
    
    print("\n" + "=" * 40)
    print("🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit config/config.yaml with your trading parameters")
    print("2. Run backtest: python src/main.py --mode backtest --strategy sma_crossover")
    print("3. Launch dashboard: python run_dashboard.py")
    print("4. Open http://localhost:8501 in your browser")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 