"""
Strategy Optimizer - Optimizes strategy parameters using risk management metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from datetime import datetime
from loguru import logger

from ..backtesting import BacktestEngine
from ..strategies.sma_crossover import SMACrossoverStrategy
from ..strategies.rsi_strategy import RSIStrategy
from ..strategies.macd_strategy import MACDStrategy
from ..utils.config_manager import ConfigManager
from .parameter_grid import ParameterGrid


class StrategyOptimizer:
    """Optimizes strategy parameters using risk management metrics"""
    
    def __init__(self, config: ConfigManager):
        """Initialize strategy optimizer"""
        self.config = config
        self.backtest_engine = BacktestEngine(config)
        self.parameter_grid = ParameterGrid()
        
        # Strategy factory
        self.strategy_factory = {
            'sma_crossover': SMACrossoverStrategy,
            'rsi': RSIStrategy,
            'macd': MACDStrategy
        }
        
        # Optimization results
        self.optimization_results = {}
        self.best_parameters = {}
        
        logger.info("Strategy optimizer initialized")
    
    def optimize_strategy(self, strategy_type: str, 
                         start_date: str, end_date: str,
                         optimization_metric: str = 'sharpe_ratio',
                         max_combinations: int = None,
                         filters: Dict[str, Any] = None,
                         use_parallel: bool = True) -> Dict[str, Any]:
        """
        Optimize parameters for a specific strategy type
        
        Args:
            strategy_type: Type of strategy to optimize
            start_date: Start date for optimization period
            end_date: End date for optimization period
            optimization_metric: Metric to optimize ('sharpe_ratio', 'total_return', 'profit_factor', etc.)
            max_combinations: Maximum number of parameter combinations to test
            filters: Optional filters for parameter combinations
            use_parallel: Whether to use parallel processing
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"Starting optimization for {strategy_type}")
        
        # Get parameter combinations
        combinations = self.parameter_grid.get_filtered_combinations(strategy_type, filters)
        
        if max_combinations and len(combinations) > max_combinations:
            # Sample combinations if too many
            combinations = np.random.choice(combinations, max_combinations, replace=False).tolist()
            logger.info(f"Sampled {max_combinations} combinations from {len(combinations)} total")
        
        if not combinations:
            logger.error(f"No parameter combinations available for {strategy_type}")
            return {}
        
        # Run optimization
        if use_parallel and len(combinations) > 10:
            results = self._optimize_parallel(strategy_type, combinations, start_date, end_date, optimization_metric)
        else:
            results = self._optimize_sequential(strategy_type, combinations, start_date, end_date, optimization_metric)
        
        # Find best parameters
        best_params = self._find_best_parameters(results, optimization_metric)
        
        # Store results
        self.optimization_results[strategy_type] = {
            'results': results,
            'best_parameters': best_params,
            'optimization_metric': optimization_metric,
            'total_combinations': len(combinations),
            'tested_combinations': len(results)
        }
        
        self.best_parameters[strategy_type] = best_params
        
        logger.info(f"Optimization completed for {strategy_type}")
        logger.info(f"Best parameters: {best_params}")
        
        return self.optimization_results[strategy_type]
    
    def _optimize_sequential(self, strategy_type: str, combinations: List[Dict[str, Any]],
                           start_date: str, end_date: str, optimization_metric: str) -> List[Dict[str, Any]]:
        """Run optimization sequentially"""
        results = []
        
        for i, params in enumerate(combinations):
            try:
                result = self._test_parameter_combination(
                    strategy_type, params, start_date, end_date
                )
                result['parameters'] = params
                result['combination_index'] = i
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"Tested {i + 1}/{len(combinations)} combinations")
                    
            except Exception as e:
                logger.warning(f"Failed to test combination {i}: {e}")
                continue
        
        return results
    
    def _optimize_parallel(self, strategy_type: str, combinations: List[Dict[str, Any]],
                          start_date: str, end_date: str, optimization_metric: str) -> List[Dict[str, Any]]:
        """Run optimization using parallel processing"""
        results = []
        
        with ProcessPoolExecutor(max_workers=min(4, len(combinations))) as executor:
            # Submit all tasks
            future_to_params = {
                executor.submit(self._test_parameter_combination, strategy_type, params, start_date, end_date): params
                for params in combinations
            }
            
            # Collect results
            for i, future in enumerate(as_completed(future_to_params)):
                params = future_to_params[future]
                try:
                    result = future.result()
                    result['parameters'] = params
                    result['combination_index'] = i
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Completed {i + 1}/{len(combinations)} combinations")
                        
                except Exception as e:
                    logger.warning(f"Failed to test combination {i}: {e}")
                    continue
        
        return results
    
    def _test_parameter_combination(self, strategy_type: str, params: Dict[str, Any],
                                  start_date: str, end_date: str) -> Dict[str, Any]:
        """Test a single parameter combination"""
        try:
            # Create strategy with parameters
            strategy_class = self.strategy_factory.get(strategy_type)
            if not strategy_class:
                raise ValueError(f"Unknown strategy type: {strategy_type}")
            
            strategy = strategy_class(params)
            
            # Run backtest
            strategies = {f"{strategy_type}_test": strategy}
            results = self.backtest_engine.run_backtest(strategies, start_date, end_date)
            
            # Extract metrics
            strategy_result = results.get(f"{strategy_type}_test", {})
            performance_metrics = strategy_result.get('performance_metrics', {})
            risk_metrics = strategy_result.get('risk_metrics', {})
            
            return {
                'total_return': performance_metrics.get('total_return', 0),
                'annualized_return': performance_metrics.get('annualized_return', 0),
                'sharpe_ratio': performance_metrics.get('sharpe_ratio', 0),
                'max_drawdown': performance_metrics.get('max_drawdown', 0),
                'volatility': performance_metrics.get('volatility', 0),
                'win_rate': performance_metrics.get('win_rate', 0),
                'profit_factor': performance_metrics.get('profit_factor', 0),
                'total_trades': performance_metrics.get('total_trades', 0),
                'final_portfolio_value': performance_metrics.get('final_portfolio_value', 0),
                'current_drawdown': risk_metrics.get('current_drawdown', 0),
                'portfolio_volatility': risk_metrics.get('volatility', 0)
            }
            
        except Exception as e:
            logger.error(f"Error testing parameters {params}: {e}")
            return {
                'total_return': 0,
                'annualized_return': 0,
                'sharpe_ratio': -999,
                'max_drawdown': 0,
                'volatility': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'total_trades': 0,
                'final_portfolio_value': 0,
                'current_drawdown': 0,
                'portfolio_volatility': 0
            }
    
    def _find_best_parameters(self, results: List[Dict[str, Any]], 
                             optimization_metric: str) -> Dict[str, Any]:
        """Find the best parameters based on optimization metric"""
        if not results:
            return {}
        
        # Sort by optimization metric
        valid_results = [r for r in results if r.get(optimization_metric) is not None]
        
        if not valid_results:
            return {}
        
        # Sort by optimization metric (descending for most metrics, ascending for drawdown)
        reverse = optimization_metric not in ['max_drawdown', 'volatility', 'current_drawdown']
        sorted_results = sorted(valid_results, 
                              key=lambda x: x.get(optimization_metric, 0), 
                              reverse=reverse)
        
        best_result = sorted_results[0]
        return {
            'parameters': best_result.get('parameters', {}),
            'metrics': {k: v for k, v in best_result.items() if k != 'parameters' and k != 'combination_index'},
            'rank': 1
        }
    
    def optimize_multiple_strategies(self, strategy_types: List[str],
                                   start_date: str, end_date: str,
                                   optimization_metric: str = 'sharpe_ratio',
                                   max_combinations_per_strategy: int = None) -> Dict[str, Any]:
        """
        Optimize multiple strategies
        
        Args:
            strategy_types: List of strategy types to optimize
            start_date: Start date for optimization period
            end_date: End date for optimization period
            optimization_metric: Metric to optimize
            max_combinations_per_strategy: Maximum combinations per strategy
            
        Returns:
            Dictionary containing results for all strategies
        """
        all_results = {}
        
        for strategy_type in strategy_types:
            logger.info(f"Optimizing {strategy_type}")
            
            try:
                result = self.optimize_strategy(
                    strategy_type, start_date, end_date, 
                    optimization_metric, max_combinations_per_strategy
                )
                all_results[strategy_type] = result
                
            except Exception as e:
                logger.error(f"Failed to optimize {strategy_type}: {e}")
                continue
        
        return all_results
    
    def get_optimization_summary(self, strategy_type: str = None) -> Dict[str, Any]:
        """Get summary of optimization results"""
        if strategy_type:
            if strategy_type not in self.optimization_results:
                return {}
            return self.optimization_results[strategy_type]
        
        return {
            'strategies': list(self.optimization_results.keys()),
            'best_parameters': self.best_parameters,
            'total_optimizations': len(self.optimization_results)
        }
    
    def get_top_parameters(self, strategy_type: str, top_n: int = 5, 
                          metric: str = 'sharpe_ratio') -> List[Dict[str, Any]]:
        """Get top N parameter combinations for a strategy"""
        if strategy_type not in self.optimization_results:
            return []
        
        results = self.optimization_results[strategy_type]['results']
        valid_results = [r for r in results if r.get(metric) is not None]
        
        if not valid_results:
            return []
        
        # Sort by metric
        reverse = metric not in ['max_drawdown', 'volatility', 'current_drawdown']
        sorted_results = sorted(valid_results, 
                              key=lambda x: x.get(metric, 0), 
                              reverse=reverse)
        
        top_results = []
        for i, result in enumerate(sorted_results[:top_n]):
            top_results.append({
                'rank': i + 1,
                'parameters': result.get('parameters', {}),
                'metrics': {k: v for k, v in result.items() 
                           if k not in ['parameters', 'combination_index']}
            })
        
        return top_results
    
    def save_optimization_results(self, filepath: str):
        """Save optimization results to file"""
        # Convert results to serializable format
        serializable_results = {}
        for strategy_type, result in self.optimization_results.items():
            serializable_results[strategy_type] = {
                'best_parameters': result['best_parameters'],
                'optimization_metric': result['optimization_metric'],
                'total_combinations': result['total_combinations'],
                'tested_combinations': result['tested_combinations'],
                'top_results': self.get_top_parameters(strategy_type, top_n=10)
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    def load_optimization_results(self, filepath: str):
        """Load optimization results from file"""
        if not os.path.exists(filepath):
            logger.warning(f"Optimization results file not found: {filepath}")
            return
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Convert back to internal format
        for strategy_type, result in data.items():
            self.best_parameters[strategy_type] = result.get('best_parameters', {})
        
        logger.info(f"Optimization results loaded from {filepath}")
    
    def create_optimized_strategy(self, strategy_type: str, 
                                use_best_parameters: bool = True,
                                custom_parameters: Dict[str, Any] = None) -> Any:
        """
        Create a strategy with optimized parameters
        
        Args:
            strategy_type: Type of strategy to create
            use_best_parameters: Whether to use best parameters from optimization
            custom_parameters: Custom parameters to use instead
            
        Returns:
            Strategy instance
        """
        if strategy_type not in self.strategy_factory:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
        
        if use_best_parameters and strategy_type in self.best_parameters:
            params = self.best_parameters[strategy_type].get('parameters', {})
        elif custom_parameters:
            params = custom_parameters
        else:
            # Use default parameters
            params = {}
        
        strategy_class = self.strategy_factory[strategy_type]
        return strategy_class(params)
    
    def generate_optimization_report(self, strategy_type: str = None) -> str:
        """Generate a comprehensive optimization report"""
        if strategy_type and strategy_type not in self.optimization_results:
            return f"No optimization results available for {strategy_type}"
        
        report = []
        report.append("=" * 60)
        report.append("STRATEGY OPTIMIZATION REPORT")
        report.append("=" * 60)
        
        if strategy_type:
            self._add_strategy_report(report, strategy_type)
        else:
            for strategy_type in self.optimization_results.keys():
                self._add_strategy_report(report, strategy_type)
                report.append("-" * 40)
        
        return "\n".join(report)
    
    def _add_strategy_report(self, report: List[str], strategy_type: str):
        """Add strategy-specific report section"""
        result = self.optimization_results[strategy_type]
        best_params = result['best_parameters']
        
        report.append(f"\nStrategy: {strategy_type.upper()}")
        report.append(f"Optimization Metric: {result['optimization_metric']}")
        report.append(f"Total Combinations Tested: {result['tested_combinations']}")
        report.append(f"Best Parameters: {best_params.get('parameters', {})}")
        
        metrics = best_params.get('metrics', {})
        if metrics:
            report.append(f"Best Performance Metrics:")
            report.append(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            report.append(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            report.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            report.append(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            report.append(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            report.append(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            report.append(f"  Total Trades: {metrics.get('total_trades', 0)}")
    
    def plot_optimization_results(self, strategy_type: str, save_path: str = None):
        """Plot optimization results for a strategy"""
        if strategy_type not in self.optimization_results:
            logger.warning(f"No optimization results for {strategy_type}")
            return
        
        import matplotlib.pyplot as plt
        
        results = self.optimization_results[strategy_type]['results']
        
        # Create scatter plot of parameter combinations vs performance
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Optimization Results: {strategy_type.upper()}', fontsize=16)
        
        # Extract data for plotting
        sharpe_ratios = [r.get('sharpe_ratio', 0) for r in results]
        total_returns = [r.get('total_return', 0) for r in results]
        max_drawdowns = [r.get('max_drawdown', 0) for r in results]
        win_rates = [r.get('win_rate', 0) for r in results]
        
        # Sharpe ratio distribution
        axes[0, 0].hist(sharpe_ratios, bins=20, alpha=0.7)
        axes[0, 0].set_title('Sharpe Ratio Distribution')
        axes[0, 0].set_xlabel('Sharpe Ratio')
        axes[0, 0].grid(True)
        
        # Total return vs max drawdown
        axes[0, 1].scatter(max_drawdowns, total_returns, alpha=0.6)
        axes[0, 1].set_title('Return vs Drawdown')
        axes[0, 1].set_xlabel('Max Drawdown')
        axes[0, 1].set_ylabel('Total Return')
        axes[0, 1].grid(True)
        
        # Win rate vs profit factor
        axes[1, 0].scatter(win_rates, [r.get('profit_factor', 0) for r in results], alpha=0.6)
        axes[1, 0].set_title('Win Rate vs Profit Factor')
        axes[1, 0].set_xlabel('Win Rate')
        axes[1, 0].set_ylabel('Profit Factor')
        axes[1, 0].grid(True)
        
        # Performance metrics comparison
        metrics = ['sharpe_ratio', 'total_return', 'win_rate', 'profit_factor']
        metric_values = []
        for metric in metrics:
            values = [r.get(metric, 0) for r in results]
            metric_values.append(np.mean(values))
        
        axes[1, 1].bar(metrics, metric_values)
        axes[1, 1].set_title('Average Performance Metrics')
        axes[1, 1].set_ylabel('Average Value')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close() 