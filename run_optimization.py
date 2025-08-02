#!/usr/bin/env python3
"""
Strategy Optimization Runner - Optimize strategy parameters using risk management metrics
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.optimization import StrategyOptimizer
from src.utils.config_manager import ConfigManager
from loguru import logger

def main():
    """Run strategy optimization"""
    
    # Load configuration
    config = ConfigManager("config/config.yaml")
    
    # Initialize optimizer
    optimizer = StrategyOptimizer(config)
    
    # Define optimization parameters
    start_date = config.get("data.start_date", "2024-01-01")
    end_date = config.get("data.end_date", "2024-05-31")
    
    # Strategies to optimize
    strategy_types = ['sma_crossover', 'rsi', 'macd']
    
    # Optimization settings
    optimization_metric = 'sharpe_ratio'  # Can be: 'sharpe_ratio', 'total_return', 'profit_factor'
    max_combinations_per_strategy = 50  # Limit to avoid too many tests
    
    logger.info(f"Starting optimization from {start_date} to {end_date}")
    logger.info(f"Optimization metric: {optimization_metric}")
    logger.info(f"Strategies to optimize: {strategy_types}")
    
    try:
        # Optimize all strategies
        results = optimizer.optimize_multiple_strategies(
            strategy_types=strategy_types,
            start_date=start_date,
            end_date=end_date,
            optimization_metric=optimization_metric,
            max_combinations_per_strategy=max_combinations_per_strategy
        )
        
        # Generate and print report
        report = optimizer.generate_optimization_report()
        print(report)
        
        # Save results
        optimizer.save_optimization_results("data/optimization_results.json")
        
        # Plot results for each strategy
        for strategy_type in strategy_types:
            if strategy_type in results:
                optimizer.plot_optimization_results(
                    strategy_type, 
                    save_path=f"data/optimization_{strategy_type}.png"
                )
        
        # Print top parameters for each strategy
        print("\n" + "="*60)
        print("TOP PARAMETER COMBINATIONS")
        print("="*60)
        
        for strategy_type in strategy_types:
            if strategy_type in results:
                print(f"\n{strategy_type.upper()}:")
                top_params = optimizer.get_top_parameters(strategy_type, top_n=5)
                
                for i, param_set in enumerate(top_params, 1):
                    params = param_set['parameters']
                    metrics = param_set['metrics']
                    
                    print(f"  Rank {i}:")
                    print(f"    Parameters: {params}")
                    print(f"    Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.3f}")
                    print(f"    Total Return: {metrics.get('total_return', 0):.2%}")
                    print(f"    Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
                    print(f"    Win Rate: {metrics.get('win_rate', 0):.2%}")
                    print(f"    Profit Factor: {metrics.get('profit_factor', 0):.2f}")
                    print()
        
        # Create optimized strategies
        print("\n" + "="*60)
        print("OPTIMIZED STRATEGIES")
        print("="*60)
        
        optimized_strategies = {}
        for strategy_type in strategy_types:
            if strategy_type in optimizer.best_parameters:
                optimized_strategy = optimizer.create_optimized_strategy(strategy_type)
                optimized_strategies[strategy_type] = optimized_strategy
                
                best_params = optimizer.best_parameters[strategy_type].get('parameters', {})
                print(f"\n{strategy_type.upper()}:")
                print(f"  Best Parameters: {best_params}")
                print(f"  Strategy Object: {type(optimized_strategy).__name__}")
        
        logger.info("✅ Optimization completed successfully!")
        
        return optimized_strategies
        
    except Exception as e:
        logger.error(f"❌ Optimization failed: {e}")
        raise

def optimize_single_strategy(strategy_type: str, 
                           start_date: str = "2024-01-01",
                           end_date: str = "2024-05-31",
                           optimization_metric: str = 'sharpe_ratio',
                           max_combinations: int = 30):
    """Optimize a single strategy with custom parameters"""
    
    config = ConfigManager("config/config.yaml")
    optimizer = StrategyOptimizer(config)
    
    logger.info(f"Optimizing {strategy_type} with metric: {optimization_metric}")
    
    try:
        result = optimizer.optimize_strategy(
            strategy_type=strategy_type,
            start_date=start_date,
            end_date=end_date,
            optimization_metric=optimization_metric,
            max_combinations=max_combinations
        )
        
        # Print results
        print(f"\nOptimization Results for {strategy_type.upper()}:")
        print(f"Best Parameters: {result.get('best_parameters', {}).get('parameters', {})}")
        print(f"Tested Combinations: {result.get('tested_combinations', 0)}")
        
        # Get top 5 results
        top_params = optimizer.get_top_parameters(strategy_type, top_n=5)
        print(f"\nTop 5 Parameter Combinations:")
        for i, param_set in enumerate(top_params, 1):
            params = param_set['parameters']
            metrics = param_set['metrics']
            print(f"  {i}. {params} -> Sharpe: {metrics.get('sharpe_ratio', 0):.3f}")
        
        # Plot results
        optimizer.plot_optimization_results(
            strategy_type, 
            save_path=f"data/optimization_{strategy_type}_single.png"
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to optimize {strategy_type}: {e}")
        return None

if __name__ == "__main__":
    # Run full optimization
    optimized_strategies = main()
    
    # Example: Optimize single strategy with different metric
    # optimize_single_strategy('sma_crossover', optimization_metric='total_return') 