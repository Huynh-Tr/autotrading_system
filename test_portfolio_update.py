#!/usr/bin/env python3
"""
Test script to verify portfolio updates and backtest results
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.main import run_backtest
from loguru import logger
import pandas as pd

def test_portfolio_updates():
    """Test that portfolio updates are working correctly"""
    logger.info("Testing portfolio updates...")
    
    try:
        # Run a backtest
        run_backtest("config/config.yaml", "sma_crossover")
        
        # Check the results
        if os.path.exists('data/backtest_results.csv'):
            results_df = pd.read_csv('data/backtest_results.csv')
            print(f"Portfolio history shape: {results_df.shape}")
            print(f"Portfolio value range: {results_df['portfolio_value'].min():.2f} - {results_df['portfolio_value'].max():.2f}")
            print(f"Cash range: {results_df['cash'].min():.2f} - {results_df['cash'].max():.2f}")
            
            # Check if portfolio value changed
            initial_value = results_df['portfolio_value'].iloc[0]
            final_value = results_df['portfolio_value'].iloc[-1]
            print(f"Initial portfolio value: ${initial_value:,.2f}")
            print(f"Final portfolio value: ${final_value:,.2f}")
            print(f"Portfolio change: ${final_value - initial_value:,.2f}")
            
            if final_value != initial_value:
                print("✅ Portfolio value changed - backtest is working!")
            else:
                print("❌ Portfolio value didn't change - there might be an issue")
        else:
            print("❌ No backtest results file found")
            
        # Check trades
        if os.path.exists('data/trades.csv'):
            trades_df = pd.read_csv('data/trades.csv')
            print(f"Number of trades: {len(trades_df)}")
            if len(trades_df) > 0:
                print("✅ Trades were executed!")
            else:
                print("❌ No trades were executed")
        else:
            print("❌ No trades file found")
            
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise

if __name__ == "__main__":
    test_portfolio_updates() 