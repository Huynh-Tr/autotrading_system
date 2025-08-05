#!/usr/bin/env python3
"""
Backtest Runner - Example script to run backtests on multiple strategies
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.backtesting import BacktestEngine
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.custom1_strategy import Custom1Strategy
from src.utils.config_manager import ConfigManager
from loguru import logger

def main():
    """Run backtest on multiple strategies"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run backtest on trading strategies')
    parser.add_argument('--strategy', type=str, help='Specific strategy to test (sma_crossover, rsi, macd, custom1)')
    parser.add_argument('--start-date', type=str, default=None, help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='End date for backtest (YYYY-MM-DD)')
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager("config/config.yaml")
    
    # Initialize backtest engine
    backtest_engine = BacktestEngine(config)
    
    # Create strategies
    strategies = {}
    
    # Determine which strategies to run
    if args.strategy:
        # Run only the specified strategy
        strategy_name = args.strategy.lower()
        if strategy_name == 'sma_crossover' or strategy_name == 'sma':
            sma_config = {
                'short_window': config.get("strategies.sma_crossover.short_window", 20),
                'long_window': config.get("strategies.sma_crossover.long_window", 50)
            }
            strategies['SMA_Crossover'] = SMACrossoverStrategy(sma_config)
        elif strategy_name == 'rsi':
            rsi_config = {
                'period': config.get("strategies.rsi.period", 14),
                'oversold': config.get("strategies.rsi.oversold", 30),
                'overbought': config.get("strategies.rsi.overbought", 70)
            }
            strategies['RSI'] = RSIStrategy(rsi_config)
        elif strategy_name == 'macd':
            macd_config = {
                'fast_period': config.get("strategies.macd.fast_period", 12),
                'slow_period': config.get("strategies.macd.slow_period", 26),
                'signal_period': config.get("strategies.macd.signal_period", 9)
            }
            strategies['MACD'] = MACDStrategy(macd_config)
        elif strategy_name == 'custom1':
            custom1_config = {
                'short_window': config.get("strategies.custom1.short_window", 20),
                'long_window': config.get("strategies.custom1.long_window", 50),
                'rsi_period': config.get("strategies.custom1.rsi_period", 14),
                'rsi_oversold': config.get("strategies.custom1.rsi_oversold", 30),
                'rsi_overbought': config.get("strategies.custom1.rsi_overbought", 70),
                'min_signal_strength': config.get("strategies.custom1.min_signal_strength", 0.3)
            }
            strategies['Custom1'] = Custom1Strategy(custom1_config)
        else:
            logger.error(f"Unknown strategy: {args.strategy}")
            return
    else:
        # Run all enabled strategies
        # SMA Crossover Strategy
        if config.get("strategies.sma_crossover.enabled", False):
            sma_config = {
                'short_window': config.get("strategies.sma_crossover.short_window", 20),
                'long_window': config.get("strategies.sma_crossover.long_window", 50)
            }
            strategies['SMA_Crossover'] = SMACrossoverStrategy(sma_config)
        
        # RSI Strategy
        if config.get("strategies.rsi.enabled", False):
            rsi_config = {
                'period': config.get("strategies.rsi.period", 14),
                'oversold': config.get("strategies.rsi.oversold", 30),
                'overbought': config.get("strategies.rsi.overbought", 70)
            }
            strategies['RSI'] = RSIStrategy(rsi_config)
        
        # MACD Strategy
        if config.get("strategies.macd.enabled", False):
            macd_config = {
                'fast_period': config.get("strategies.macd.fast_period", 12),
                'slow_period': config.get("strategies.macd.slow_period", 26),
                'signal_period': config.get("strategies.macd.signal_period", 9)
            }
            strategies['MACD'] = MACDStrategy(macd_config)
        
        # Custom1 Strategy
        if config.get("strategies.custom1.enabled", False):
            custom1_config = {
                'short_window': config.get("strategies.custom1.short_window", 20),
                'long_window': config.get("strategies.custom1.long_window", 50),
                'rsi_period': config.get("strategies.custom1.rsi_period", 14),
                'rsi_oversold': config.get("strategies.custom1.rsi_oversold", 30),
                'rsi_overbought': config.get("strategies.custom1.rsi_overbought", 70),
                'min_signal_strength': config.get("strategies.custom1.min_signal_strength", 0.3)
            }
            strategies['Custom1'] = Custom1Strategy(custom1_config)
    
    if not strategies:
        logger.error("No strategies enabled in configuration")
        return
    
    # Run backtest
    start_date = args.start_date or config.get("data.start_date", "2024-01-01")
    end_date = args.end_date or config.get("data.end_date", "2024-05-31")
    
    logger.info(f"Running backtest from {start_date} to {end_date}")
    logger.info(f"Testing strategies: {list(strategies.keys())}")
    
    try:
        results = backtest_engine.run_backtest(strategies, start_date, end_date)
        
        # Generate and print report
        report = backtest_engine.generate_report()
        print(report)
        
        # Plot results
        backtest_engine.plot_results(save_path="data/backtest_results.png")
        
        # Save results
        backtest_engine.save_results("data/backtest_results.json")
        
        # Get best strategy
        best_strategy = backtest_engine.get_best_strategy()
        if best_strategy:
            logger.info(f"Best performing strategy: {best_strategy}")
        
        # Print detailed metrics for each strategy
        print("\n" + "="*60)
        print("DETAILED PERFORMANCE METRICS")
        print("="*60)
        
        for strategy_name, result in results.items():
            metrics = result['performance_metrics']
            risk_metrics = result['risk_metrics']
            
            print(f"\n{strategy_name}:")
            print(f"  Final Portfolio Value: ${metrics.get('final_portfolio_value', 0):,.2f}")
            print(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            print(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            print(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            print(f"  Volatility: {metrics.get('volatility', 0):.2%}")
            print(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            print(f"  Profit Factor: {metrics.get('profit_factor', 0):.2f}")
            print(f"  Total Trades: {metrics.get('total_trades', 0)}")
            print(f"  Average Trade: ${metrics.get('avg_trade', 0):,.2f}")
            
            if risk_metrics:
                print(f"  Current Drawdown: {risk_metrics.get('current_drawdown', 0):.2%}")
                print(f"  Portfolio Volatility: {risk_metrics.get('volatility', 0):.2%}")
        
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise

if __name__ == "__main__":
    main() 