#!/usr/bin/env python3
"""
Launcher script for Streamlit Dashboard
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit dashboard"""
    print("🚀 Launching Auto Trading System Dashboard...")
    
    # Check if streamlit is installed
    try:
        import streamlit
        print("✓ Streamlit is installed")
    except ImportError:
        print("❌ Streamlit not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "streamlit"])
    
    # Check if plotly is installed
    try:
        import plotly
        print("✓ Plotly is installed")
    except ImportError:
        print("❌ Plotly not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "plotly"])
    
    # Launch the dashboard
    dashboard_path = os.path.join("dashboard", "streamlit_app.py")
    
    if not os.path.exists(dashboard_path):
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return False
    
    print("🌐 Starting Streamlit server...")
    print("📊 Dashboard will open in your browser at: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the server")
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            dashboard_path,
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n👋 Dashboard stopped")
    except Exception as e:
        print(f"❌ Error launching dashboard: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main() 