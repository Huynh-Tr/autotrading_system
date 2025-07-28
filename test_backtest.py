#!/usr/bin/env python3
"""
Test script to verify backtesting functionality
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import run_backtest
from loguru import logger

def test_backtest():
    """Test the backtesting functionality"""
    logger.info("Starting backtest test")
    
    try:
        # Test with SMA crossover strategy
        run_backtest("config/config.yaml", "sma_crossover")
        logger.info("SMA crossover backtest completed successfully")
        
        # Test with RSI strategy
        run_backtest("config/config.yaml", "rsi")
        logger.info("RSI backtest completed successfully")
        
        # Test with MACD strategy
        run_backtest("config/config.yaml", "macd")
        logger.info("MACD backtest completed successfully")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

if __name__ == "__main__":
    test_backtest() 