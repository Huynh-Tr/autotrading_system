#!/usr/bin/env python3
"""
Comprehensive test to verify all components are working correctly
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import run_backtest
from loguru import logger
import pandas as pd
import time

def comprehensive_test():
    """Run comprehensive tests on the backtesting system"""
    logger.info("Starting comprehensive backtest test")
    
    strategies = ["sma_crossover", "rsi", "macd"]
    
    for strategy in strategies:
        logger.info(f"Testing {strategy} strategy...")
        
        try:
            # Clear previous results
            if os.path.exists('data/backtest_results.csv'):
                os.remove('data/backtest_results.csv')
            if os.path.exists('data/trades.csv'):
                os.remove('data/trades.csv')
            
            # Run backtest
            start_time = time.time()
            run_backtest("config/config.yaml", strategy)
            end_time = time.time()
            
            logger.info(f"✅ {strategy} backtest completed in {end_time - start_time:.2f} seconds")
            
            # Check results
            if os.path.exists('data/backtest_results.csv'):
                results_df = pd.read_csv('data/backtest_results.csv')
                initial_value = results_df['portfolio_value'].iloc[0]
                final_value = results_df['portfolio_value'].iloc[-1]
                
                logger.info(f"  - Portfolio change: ${final_value - initial_value:,.2f}")
                logger.info(f"  - Data points: {len(results_df)}")
                
                if final_value != initial_value:
                    logger.info(f"  ✅ Portfolio value changed for {strategy}")
                else:
                    logger.warning(f"  ⚠️ Portfolio value didn't change for {strategy}")
            else:
                logger.error(f"  ❌ No results file for {strategy}")
            
            # Check trades
            if os.path.exists('data/trades.csv'):
                trades_df = pd.read_csv('data/trades.csv')
                logger.info(f"  - Trades executed: {len(trades_df)}")
                if len(trades_df) > 0:
                    logger.info(f"  ✅ Trades were executed for {strategy}")
                else:
                    logger.warning(f"  ⚠️ No trades executed for {strategy}")
            else:
                logger.error(f"  ❌ No trades file for {strategy}")
                
        except Exception as e:
            logger.error(f"❌ {strategy} test failed: {e}")
    
    logger.info("Comprehensive test completed!")

if __name__ == "__main__":
    comprehensive_test() 