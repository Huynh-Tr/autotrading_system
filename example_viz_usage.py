#!/usr/bin/env python3
"""
Example usage of the visualization module (viz.py)
"""

import pandas as pd
import numpy as np
from src.viz.viz import viz

def create_sample_data():
    """Create sample data for visualization"""
    # Create sample historical data
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    
    # Generate sample OHLCV data
    np.random.seed(42)
    base_price = 100
    returns = np.random.normal(0, 0.02, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    prices = np.array(prices)
    
    historical_data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.005, len(prices))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(prices)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(prices)))),
        'close': prices,
        'volume': np.random.randint(1000, 10000, len(prices))
    }, index=dates)
    
    # Create sample trades data
    trade_dates = dates[::10]  # Every 10th day
    trades_df = pd.DataFrame({
        'Timestamp': trade_dates,
        'Side': ['buy', 'sell'] * (len(trade_dates) // 2 + 1),
        'Price': historical_data.loc[trade_dates, 'close'].values[:len(trade_dates)]
    })
    
    return historical_data, trades_df

def main():
    """Main function to demonstrate viz usage"""
    print("Auto Trading System - Visualization Example")
    print("=" * 50)
    
    # Create sample data
    print("Creating sample data...")
    historical_data, trades_df = create_sample_data()
    
    # Define symbol
    symbol = "BTC/USD"
    
    print(f"Sample data created:")
    print(f"- Historical data shape: {historical_data.shape}")
    print(f"- Number of trades: {len(trades_df)}")
    print(f"- Symbol: {symbol}")
    
    print("\nParameters for viz function:")
    print(f"- symbol: {symbol}")
    print(f"- trades_df: DataFrame with {len(trades_df)} trades")
    print(f"- historical_data: DataFrame with {len(historical_data)} data points")
    
    # Call the viz function
    print("\nCalling viz function...")
    try:
        viz(symbol, trades_df, historical_data)
        print("✅ Visualization completed successfully!")
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
        print("Make sure plotly is installed: pip install plotly")

if __name__ == "__main__":
    main() 