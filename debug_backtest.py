#!/usr/bin/env python3
"""
Debug script to identify backtest issues
"""

import sys
import os
import pandas as pd

# Add the project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def debug_backtest():
    """Debug the backtest step by step"""
    print("🔍 Debugging Backtest Issues")
    print("=" * 50)
    
    try:
        from src.utils.config_manager import ConfigManager
        from src.core.trading_engine import TradingEngine
        from src.strategies.sma_crossover import SMACrossoverStrategy
        
        print("✅ Imports successful")
        
        # Load configuration
        config = ConfigManager()
        print("✅ Config loaded")
        
        # Initialize trading engine
        engine = TradingEngine(config)
        print("✅ Trading engine initialized")
        
        # Add strategy
        strategy_config = config.get("strategies.sma_crossover", {})
        strategy = SMACrossoverStrategy(strategy_config)
        engine.add_strategy(strategy)
        print("✅ Strategy added")
        
        # Get data
        data = engine.data_manager.get_historical_data(
            symbols=config.get("trading.symbols"),
            start_date=config.get("data.start_date"),
            end_date=config.get("data.end_date"),
            interval=config.get("data.interval", "1d")
        )
        
        print(f"✅ Data loaded: {data.shape}")
        print(f"   Columns: {data.columns.tolist()}")
        
        # Validate data
        if engine.data_manager.validate_ohlcv_data(data):
            print("✅ Data validation passed")
        else:
            print("❌ Data validation failed")
            return
        
        # Clean data
        cleaned_data = engine.data_manager.clean_ohlcv_data(data)
        print(f"✅ Data cleaned: {cleaned_data.shape}")
        
        # Test first few rows
        print("\n🧪 Testing first 3 rows...")
        for i, (date, row) in enumerate(cleaned_data.head(3).iterrows()):
            print(f"Row {i+1}: {date}")
            print(f"  Data type: {type(row)}")
            print(f"  Index type: {type(row.index)}")
            print(f"  Sample data: {row.head(2).to_dict()}")
            
            # Test price extraction
            symbol = config.get("trading.symbols")[0]
            if isinstance(row.index, pd.MultiIndex):
                if (symbol, 'close') in row.index:
                    price = row[(symbol, 'close')]
                    print(f"  {symbol} close price: {price} (type: {type(price)})")
                else:
                    print(f"  No close data for {symbol}")
            else:
                if symbol in row.index:
                    price = row[symbol]
                    print(f"  {symbol} price: {price} (type: {type(price)})")
                else:
                    print(f"  No data for {symbol}")
            print()
        
        print("✅ Debug completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_backtest()
    if success:
        print("\n🎉 Debug completed - no issues found")
    else:
        print("\n🔧 Issues found - check the error above") 