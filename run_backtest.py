#!/usr/bin/env python3
"""
Run backtest with different strategies
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import run_backtest
from loguru import logger

def main():
    """Run backtest with different strategies"""
    config_file = "config/config.yaml"
    
    strategies = ["sma_crossover", "rsi", "macd"]
    
    for strategy in strategies:
        logger.info(f"Running backtest with {strategy} strategy")
        try:
            run_backtest(config_file, strategy)
            logger.info(f"✅ {strategy} backtest completed successfully")
        except Exception as e:
            logger.error(f"❌ {strategy} backtest failed: {e}")

if __name__ == "__main__":
    main() 