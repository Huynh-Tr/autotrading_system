#!/usr/bin/env python3
"""
Demo script for testing data caching functionality in TradingEngine
"""

import os
import sys
import time
from datetime import datetime

# Add project root to path
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.core.trading_engine import TradingEngine
from src.utils.config_manager import ConfigManager

def demo_data_caching():
    """Demo data caching functionality"""
    print("=" * 80)
    print("DEMO DATA CACHING FUNCTIONALITY")
    print("=" * 80)
    
    # Initialize trading engine
    config_path = "config/config.yaml"
    engine = TradingEngine(config_path)
    
    # Test parameters
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    strategy_types = ["sma_crossover", "rsi"]
    
    print(f"Test period: {start_date} to {end_date}")
    print(f"Strategies: {strategy_types}")
    
    # Test 1: First run - should fetch data
    print("\n" + "=" * 60)
    print("TEST 1: FIRST RUN (SHOULD FETCH DATA)")
    print("=" * 60)
    
    start_time = time.time()
    results1 = engine.run_complete_optimization_workflow(
        strategy_types=strategy_types,
        start_date=start_date,
        end_date=end_date,
        optimization_metric='sharpe_ratio',
        max_combinations_per_strategy=10,  # Small number for demo
        # symbols=["AAPL"]
    )
    time1 = time.time() - start_time
    
    print(f"Time taken: {time1:.2f} seconds")
    
    # Check cached data info
    data_info1 = engine.get_cached_data_info()
    print(f"Cached data: {data_info1['has_cached_data']}")
    print(f"Data shape: {data_info1['data_shape']}")
    
    # Test 2: Second run - should use cached data
    print("\n" + "=" * 60)
    print("TEST 2: SECOND RUN (SHOULD USE CACHED DATA)")
    print("=" * 60)
    
    start_time = time.time()
    results2 = engine.run_complete_optimization_workflow(
        strategy_types=strategy_types,
        start_date=start_date,
        end_date=end_date,
        optimization_metric='sharpe_ratio',
        max_combinations_per_strategy=10,
        # symbols=["AAPL"]
    )
    time2 = time.time() - start_time
    
    print(f"Time taken: {time2:.2f} seconds")
    print(f"Speed improvement: {((time1 - time2) / time1 * 100):.1f}%")
    
    # Test 3: Different period - should fetch new data
    print("\n" + "=" * 60)
    print("TEST 3: DIFFERENT PERIOD (SHOULD FETCH NEW DATA)")
    print("=" * 60)
    
    start_time = time.time()
    results3 = engine.run_complete_optimization_workflow(
        strategy_types=strategy_types,
        start_date="2023-06-01",
        end_date="2023-12-31",
        optimization_metric='sharpe_ratio',
        max_combinations_per_strategy=10,
        # symbols=["AAPL"]
    )
    time3 = time.time() - start_time
    
    print(f"Time taken: {time3:.2f} seconds")
    
    # Test 4: Clear cache and run again
    print("\n" + "=" * 60)
    print("TEST 4: CLEAR CACHE AND RUN AGAIN")
    print("=" * 60)
    
    engine.clear_cached_data()
    print("Cache cleared")
    
    start_time = time.time()
    results4 = engine.run_complete_optimization_workflow(
        strategy_types=strategy_types,
        start_date=start_date,
        end_date=end_date,
        optimization_metric='sharpe_ratio',
        max_combinations_per_strategy=10,
        # symbols=["AAPL"]
    )
    time4 = time.time() - start_time
    
    print(f"Time taken: {time4:.2f} seconds")
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"First run (fetch data): {time1:.2f}s")
    print(f"Second run (cached): {time2:.2f}s")
    print(f"Different period: {time3:.2f}s")
    print(f"After cache clear: {time4:.2f}s")
    print(f"Cache efficiency: {((time1 - time2) / time1 * 100):.1f}% improvement")
    
    # Show results structure
    print("\n" + "=" * 60)
    print("RESULTS STRUCTURE")
    print("=" * 60)
    print(f"Keys in results: {list(results1.keys())}")
    print(f"Optimization results keys: {list(results1.get('optimization_results', {}).keys())}")
    print(f"Backtest results keys: {list(results1.get('backtest_results', {}).keys())}")
    print(f"Data info: {results1.get('data_info', {})}")

def demo_individual_methods():
    """Demo individual methods with data caching"""
    print("\n" + "=" * 80)
    print("DEMO INDIVIDUAL METHODS WITH DATA CACHING")
    print("=" * 80)
    
    # Initialize trading engine
    config_path = "config/config.yaml"
    engine = TradingEngine(config_path)
    
    start_date = "2023-01-01"
    end_date = "2023-12-31"
    strategy_types = ["sma_crossover"]
    
    print("Testing individual methods...")
    
    # Test optimize_strategies
    print("\n1. Testing optimize_strategies...")
    opt_results = engine.optimize_strategies(
        strategy_types=strategy_types,
        start_date=start_date,
        end_date=end_date,
        optimization_metric='sharpe_ratio',
        max_combinations_per_strategy=5,
        # symbols=["AAPL"]
    )
    print(f"Optimization completed for {len(opt_results)} strategies")
    
    # Test run_optimized_backtest
    print("\n2. Testing run_optimized_backtest...")
    backtest_results = engine.run_optimized_backtest(
        start_date=start_date,
        end_date=end_date,
        strategy_types=strategy_types,
        # symbols=["AAPL"]
    )
    print(f"Backtest completed for {len(backtest_results)} strategies")
    
    # Test get_optimization_summary
    print("\n3. Testing get_optimization_summary...")
    summary = engine.get_optimization_summary()
    print(f"Summary: {list(summary.keys())}")
    
    # Test cached data info
    print("\n4. Testing cached data info...")
    data_info = engine.get_cached_data_info()
    print(f"Data info: {data_info}")

if __name__ == "__main__":
    try:
        demo_data_caching()
        demo_individual_methods()
        print("\n" + "=" * 80)
        print("DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
    except Exception as e:
        print(f"Error during demo: {e}")
        import traceback
        traceback.print_exc() 