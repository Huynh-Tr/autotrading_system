#!/usr/bin/env python3
"""
Test script for Streamlit Dashboard
"""

import sys
import os
import subprocess

def test_streamlit_imports():
    """Test if Streamlit and Plotly can be imported"""
    print("Testing Streamlit imports...")
    
    try:
        import streamlit as st
        print("✓ Streamlit imported successfully")
    except ImportError as e:
        print(f"✗ Streamlit import failed: {e}")
        return False
    
    try:
        import plotly.graph_objects as go
        import plotly.express as px
        print("✓ Plotly imported successfully")
    except ImportError as e:
        print(f"✗ Plotly import failed: {e}")
        return False
    
    return True

def test_dashboard_file():
    """Test if dashboard file exists and is valid"""
    print("\nTesting dashboard file...")
    
    dashboard_path = "dashboard/streamlit_app.py"
    
    if not os.path.exists(dashboard_path):
        print(f"✗ Dashboard file not found: {dashboard_path}")
        return False
    
    print(f"✓ Dashboard file found: {dashboard_path}")
    
    # Check file size
    file_size = os.path.getsize(dashboard_path)
    print(f"✓ File size: {file_size:,} bytes")
    
    return True

def test_streamlit_syntax():
    """Test Streamlit app syntax"""
    print("\nTesting Streamlit syntax...")
    
    try:
        # Try to import the dashboard module
        sys.path.append(os.path.join(os.path.dirname(__file__), 'dashboard'))
        
        # This is a basic syntax check - we won't actually run the full app
        with open("dashboard/streamlit_app.py", "r") as f:
            content = f.read()
        
        # Check for basic Streamlit components
        if "import streamlit" in content:
            print("✓ Streamlit import found")
        else:
            print("✗ Streamlit import not found")
            return False
        
        if "st.set_page_config" in content:
            print("✓ Page configuration found")
        else:
            print("✗ Page configuration not found")
            return False
        
        if "st.plotly_chart" in content:
            print("✓ Plotly charts found")
        else:
            print("✗ Plotly charts not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"✗ Syntax test failed: {e}")
        return False

def test_requirements():
    """Test if required packages are installed"""
    print("\nTesting required packages...")
    
    required_packages = [
        "streamlit",
        "plotly",
        "pandas",
        "numpy"
    ]
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} is installed")
        except ImportError:
            print(f"✗ {package} is not installed")
            return False
    
    return True

def main():
    """Run all Streamlit tests"""
    print("Streamlit Dashboard Tests")
    print("=" * 40)
    
    tests = [
        test_streamlit_imports,
        test_dashboard_file,
        test_streamlit_syntax,
        test_requirements
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All Streamlit tests passed!")
        print("\nTo launch the dashboard:")
        print("1. Run: python run_dashboard.py")
        print("2. Or: streamlit run dashboard/streamlit_app.py")
        print("3. Open: http://localhost:8501")
    else:
        print("❌ Some tests failed. Please install missing dependencies:")
        print("pip install streamlit plotly pandas numpy")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 