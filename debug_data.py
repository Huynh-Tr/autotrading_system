#!/usr/bin/env python3
"""
Debug script to check data structure and portfolio updates
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
from src.data.data_manager import DataManager
from src.utils.config_manager import ConfigManager

def debug_data():
    """Debug the data loading and structure"""
    config = ConfigManager("config/config.yaml")
    data_manager = DataManager(config)
    
    # Load data
    data = data_manager.get_historical_data(
        symbols=["VCB"],
        start_date="2024-01-01",
        end_date="2024-05-31",
        interval="1d"
    )
    
    print("Data shape:", data.shape)
    print("Data columns:", data.columns)
    print("Data types:", data.dtypes)
    print("\nFirst 5 rows:")
    print(data.head())
    print("\nLast 5 rows:")
    print(data.tail())
    
    # Check for any issues
    print("\nData info:")
    print(data.info())
    
    print("\nMissing values:")
    print(data.isnull().sum())

if __name__ == "__main__":
    debug_data() 