#!/usr/bin/env python3
"""
Demo Script - Test optimized trading features
"""

import sys
import os
import json
from datetime import datetime
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.trading_engine import TradingEngine
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.utils.config_manager import ConfigManager
from loguru import logger

def demo_basic_optimization():
    """Demo basic optimization functionality"""
    print("=" * 60)
    print("DEMO: BASIC OPTIMIZATION")
    print("=" * 60)
    
    # Initialize trading engine
    engine = TradingEngine()
    
    # Define parameters
    strategy_types = ['sma_crossover', 'rsi', 'macd']
    start_date = "2024-01-01"
    end_date = "2024-05-31"
    
    print(f"Optimizing strategies: {strategy_types}")
    print(f"Period: {start_date} to {end_date}")
    
    # Run optimization
    results = engine.optimize_strategies(
        strategy_types=strategy_types,
        start_date=start_date,
        end_date=end_date,
        optimization_metric='sharpe_ratio',
        max_combinations_per_strategy=20  # Small number for demo
    )
    
    # Get optimization summary
    summary = engine.get_optimization_summary()
    
    print("\nOptimization Results:")
    for strategy_type, info in summary.items():
        print(f"\n{strategy_type.upper()}:")
        print(f"  Best Parameters: {info['best_parameters']}")
        print(f"  Best Sharpe Ratio: {info['best_metrics'].get('sharpe_ratio', 0):.3f}")
        print(f"  Total Return: {info['best_metrics'].get('total_return', 0):.2%}")
        print(f"  Max Drawdown: {info['best_metrics'].get('max_drawdown', 0):.2%}")
    
    return results

def demo_optimized_backtest():
    """Demo optimized backtest functionality"""
    print("=" * 60)
    print("DEMO: OPTIMIZED BACKTEST")
    print("=" * 60)
    
    # Initialize trading engine
    engine = TradingEngine()
    
    # Define parameters
    strategy_types = ['sma_crossover', 'rsi']
    start_date = "2024-01-01"
    end_date = "2024-05-31"
    
    print(f"Running optimized backtest for: {strategy_types}")
    print(f"Period: {start_date} to {end_date}")
    
    # Run optimized backtest
    results = engine.run_optimized_backtest(
        start_date=start_date,
        end_date=end_date,
        strategy_types=strategy_types
    )
    
    print("\nBacktest Results:")
    for strategy_name, result in results.items():
        metrics = result.get('performance_metrics', {})
        print(f"\n{strategy_name}:")
        print(f"  Final Portfolio Value: ${metrics.get('final_portfolio_value', 0):,.2f}")
        print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
        print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
        print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
    
    return results

def demo_strategy_selection():
    """Demo strategy selection functionality"""
    print("=" * 60)
    print("DEMO: STRATEGY SELECTION")
    print("=" * 60)
    
    # Initialize trading engine and run optimization first
    engine = TradingEngine()
    
    strategy_types = ['sma_crossover', 'rsi', 'macd']
    start_date = "2024-01-01"
    end_date = "2024-05-31"
    
    # Run optimization
    engine.optimize_strategies(
        strategy_types=strategy_types,
        start_date=start_date,
        end_date=end_date,
        max_combinations_per_strategy=15
    )
    
    # Import optimizer for strategy selection
    from src.optimization.optimizer import StrategyOptimizer
    optimizer = StrategyOptimizer(engine.config)
    optimizer.optimization_results = engine.optimization_results
    optimizer.best_parameters = {}
    
    # Load best parameters from optimization results
    for strategy_type, results in engine.optimization_results.items():
        if 'best_parameters' in results:
            optimizer.best_parameters[strategy_type] = results['best_parameters']
    
    # Select optimal strategy
    optimal_strategy = optimizer.select_optimal_strategy(strategy_types, 'sharpe_ratio')
    print(f"Optimal Strategy: {optimal_strategy}")
    
    # Get strategy ranking
    rankings = optimizer.get_strategy_ranking(strategy_types, 'sharpe_ratio')
    print("\nStrategy Rankings (by Sharpe Ratio):")
    for i, ranking in enumerate(rankings, 1):
        print(f"{i}. {ranking['strategy_type']}: {ranking['metric_value']:.3f}")
    
    # Create optimal portfolio
    portfolio = optimizer.create_optimal_strategy_portfolio(strategy_types, top_n=2)
    print(f"\nOptimal Portfolio (Top 2 strategies):")
    for strategy_type, info in portfolio.items():
        print(f"  {strategy_type}: weight={info['weight']:.3f}")
    
    return optimal_strategy, rankings, portfolio

def demo_complete_workflow():
    """Demo complete workflow from optimization to backtest"""
    print("=" * 60)
    print("DEMO: COMPLETE WORKFLOW")
    print("=" * 60)
    
    # Initialize trading engine
    engine = TradingEngine()
    
    # Step 1: Optimize strategies
    print("Step 1: Optimizing strategies...")
    strategy_types = ['sma_crossover', 'rsi']
    start_date = "2024-01-01"
    end_date = "2024-05-31"
    
    optimization_results = engine.optimize_strategies(
        strategy_types=strategy_types,
        start_date=start_date,
        end_date=end_date,
        max_combinations_per_strategy=15
    )
    
    # Step 2: Get optimized strategies
    print("\nStep 2: Getting optimized strategies...")
    optimized_strategies = engine.get_optimized_strategies()
    
    for strategy_name, strategy in optimized_strategies.items():
        print(f"  Created optimized strategy: {strategy_name}")
    
    # Step 3: Run backtest with optimized strategies
    print("\nStep 3: Running backtest with optimized strategies...")
    backtest_results = engine.run_backtest(start_date, end_date)
    
    # Step 4: Generate reports
    print("\nStep 4: Generating reports...")
    
    # Save optimization results
    engine.save_optimization_results("data/demo_optimization_results.json")
    
    # Save backtest results
    with open("data/demo_backtest_results.json", 'w') as f:
        json.dump(backtest_results, f, indent=2, default=str)
    
    print("Complete workflow finished!")
    print("Results saved to data/ directory")
    
    return {
        'optimization_results': optimization_results,
        'optimized_strategies': optimized_strategies,
        'backtest_results': backtest_results
    }

def show_available_features():
    """Show available features"""
    print("=" * 60)
    print("AVAILABLE OPTIMIZED TRADING FEATURES")
    print("=" * 60)
    
    features = [
        {
            "name": "Strategy Optimization",
            "description": "Optimize parameters for multiple strategies",
            "method": "engine.optimize_strategies()"
        },
        {
            "name": "Optimized Backtest",
            "description": "Run backtest with optimized strategies",
            "method": "engine.run_optimized_backtest()"
        },
        {
            "name": "Strategy Selection",
            "description": "Select optimal strategy based on metrics",
            "method": "optimizer.select_optimal_strategy()"
        },
        {
            "name": "Strategy Ranking",
            "description": "Rank strategies by performance metrics",
            "method": "optimizer.get_strategy_ranking()"
        },
        {
            "name": "Portfolio Creation",
            "description": "Create portfolio of top strategies",
            "method": "optimizer.create_optimal_strategy_portfolio()"
        }
    ]
    
    for i, feature in enumerate(features, 1):
        print(f"{i}. {feature['name']}")
        print(f"   Description: {feature['description']}")
        print(f"   Method: {feature['method']}")
        print()

def main():
    """Main demo function"""
    print("OPTIMIZED TRADING SYSTEM DEMO")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    while True:
        print("\nChoose demo option:")
        print("1. Basic Optimization")
        print("2. Optimized Backtest")
        print("3. Strategy Selection")
        print("4. Complete Workflow")
        print("5. Show Available Features")
        print("0. Exit")
        
        choice = input("\nEnter your choice (0-5): ").strip()
        
        try:
            if choice == "1":
                demo_basic_optimization()
            elif choice == "2":
                demo_optimized_backtest()
            elif choice == "3":
                demo_strategy_selection()
            elif choice == "4":
                demo_complete_workflow()
            elif choice == "5":
                show_available_features()
            elif choice == "0":
                print("Exiting demo...")
                break
            else:
                print("Invalid choice. Please enter 0-5.")
                
        except Exception as e:
            print(f"Error in demo: {e}")
            logger.error(f"Demo error: {e}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 