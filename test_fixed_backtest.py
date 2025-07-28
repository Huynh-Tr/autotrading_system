#!/usr/bin/env python3
"""
Test script to verify fixed backtesting functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import run_backtest
from loguru import logger

def test_fixed_backtest():
    """Test the fixed backtesting functionality"""
    logger.info("Starting fixed backtest test")
    
    try:
        # Test with SMA crossover strategy
        logger.info("Testing SMA crossover strategy...")
        run_backtest("config/config.yaml", "sma_crossover")
        
        # Check if trades were executed
        import pandas as pd
        if os.path.exists('data/trades.csv'):
            trades_df = pd.read_csv('data/trades.csv')
            logger.info(f"✅ SMA crossover completed with {len(trades_df)} trades")
        else:
            logger.warning("❌ No trades were executed for SMA crossover")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

if __name__ == "__main__":
    test_fixed_backtest() 