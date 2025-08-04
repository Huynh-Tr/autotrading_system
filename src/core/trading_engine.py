"""
Trading Engine - Core component that orchestrates the trading system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from loguru import logger

from ..data.data_manager import DataManager
from ..strategies.base_strategy import BaseStrategy
from ..risk.risk_manager import RiskManager
from ..utils.config_manager import ConfigManager
from ..utils.ohlcv_utils import get_symbols_from_data, extract_price_data, get_current_price


@dataclass
class Position:
    """Represents a trading position"""
    symbol: str
    quantity: float
    entry_price: float
    entry_time: datetime
    current_price: float
    pnl: float
    pnl_pct: float


@dataclass
class Trade:
    """Represents a completed trade"""
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    timestamp: datetime
    commission: float
    strategy: str
    pnl: float = 0.0
    pnl_pct: float = 0.0


class TradingEngine:
    """
    Main trading engine that coordinates all components
    """
    
    def __init__(self, config_path_or_config = "config/config.yaml"):
        """Initialize the trading engine"""
        # Handle both config path string and ConfigManager object
        if isinstance(config_path_or_config, str):
            self.config = ConfigManager(config_path_or_config)
        else:
            self.config = config_path_or_config
            
        self.data_manager = DataManager(self.config)
        self.risk_manager = RiskManager(self.config)
        
        # Portfolio state
        self.cash = self.config.get("trading.initial_capital", 100000)
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.portfolio_value = self.cash
        
        # Performance tracking
        self.daily_returns = []
        self.portfolio_history = []
        
        # Historical data for strategies - cached to avoid multiple fetches
        self.historical_data = None
        self.data_start_date = None
        self.data_end_date = None
        self.data_symbols = None
        
        # Strategies
        self.strategies: Dict[str, BaseStrategy] = {}
        
        # Optimization results
        self.optimization_results = {}
        self.backtest_results = {}
        
        logger.info(f"Trading engine initialized with ${self.cash:,.2f} initial capital")
    
    def _fetch_and_cache_data(self, start_date: str, end_date: str, symbols: List[str] = None) -> pd.DataFrame:
        """
        Fetch and cache historical data to avoid multiple data fetches
        
        Args:
            start_date: Start date for data
            end_date: End date for data
            symbols: List of symbols to fetch data for
            
        Returns:
            DataFrame containing historical data
        """
        # Check if we already have cached data for this period
        if (self.historical_data is not None and 
            self.data_start_date == start_date and 
            self.data_end_date == end_date and
            self.data_symbols == symbols):
            logger.info("Using cached historical data")
            return self.historical_data
        
        # Fetch new data
        logger.info(f"Fetching historical data from {start_date} to {end_date}")
        
        if symbols is None:
            symbols = self.config.get("trading.symbols", ["AAPL"])
        
        # Get data from DataManager
        historical_data = self.data_manager.get_historical_data(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            interval=self.config.get("data.interval", "1d"),
            n_bars=self.config.get("data.n_bars", 1000)
        )
        
        # Cache the data
        self.historical_data = historical_data
        self.data_start_date = start_date
        self.data_end_date = end_date
        self.data_symbols = symbols
        
        logger.info(f"Cached historical data: {historical_data.shape}")
        return historical_data
    
    def add_strategy(self, strategy: BaseStrategy):
        """Add a trading strategy to the engine"""
        self.strategies[strategy.name] = strategy
        logger.info(f"Added strategy: {strategy.name}")
    
    def run_backtest(self, start_date: str, end_date: str, symbols: List[str] = None):
        """Run backtesting simulation using BacktestEngine"""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Fetch and cache data
        historical_data = self._fetch_and_cache_data(start_date, end_date, symbols)
        
        # Import BacktestEngine here to avoid circular imports
        from ..backtesting.backtest_engine import BacktestEngine
        
        # Create backtest engine
        backtest_engine = BacktestEngine(self.config)
        
        # Run backtest with current strategies and cached data
        results = backtest_engine.run_backtest(
            strategies=self.strategies,
            start_date=start_date,
            end_date=end_date,
            historical_data=historical_data  # Pass cached data
        )
        
        # Store results
        self.backtest_results = results
        
        logger.info("Backtest completed using BacktestEngine")
        return results
    
    def run_live_trading(self):
        """Run live trading (placeholder for real implementation)"""
        logger.info("Starting live trading mode")
        # This would connect to real-time data feeds and execute real trades
        # For now, this is a placeholder
        pass
    
    def optimize_strategies(self, strategy_types: List[str], 
                          start_date: str, end_date: str,
                          optimization_metric: str = 'sharpe_ratio',
                          max_combinations_per_strategy: int = 50,
                          symbols: List[str] = None) -> Dict[str, Any]:
        """
        Optimize parameters for multiple strategies
        
        Args:
            strategy_types: List of strategy types to optimize
            start_date: Start date for optimization
            end_date: End date for optimization
            optimization_metric: Metric to optimize for
            max_combinations_per_strategy: Maximum parameter combinations to test
            symbols: List of symbols to use for optimization
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info(f"Starting strategy optimization for {strategy_types}")
        
        # Fetch and cache data once for optimization
        historical_data = self._fetch_and_cache_data(start_date, end_date, symbols)
        
        # Import StrategyOptimizer here to avoid circular imports
        from ..optimization.optimizer import StrategyOptimizer
        
        # Create optimizer
        optimizer = StrategyOptimizer(self.config)
        
        # Run optimization for multiple strategies with cached data
        results = optimizer.optimize_multiple_strategies(
            strategy_types=strategy_types,
            start_date=start_date,
            end_date=end_date,
            optimization_metric=optimization_metric,
            max_combinations_per_strategy=max_combinations_per_strategy,
            historical_data=historical_data  # Pass cached data
        )
        
        # Store optimization results
        self.optimization_results = results
        
        # Save results
        optimizer.save_optimization_results("data/optimization_results.json")
        
        logger.info("Strategy optimization completed")
        return results
    
    def get_optimized_strategies(self) -> Dict[str, Any]:
        """Get strategies with optimized parameters"""
        if not hasattr(self, 'optimization_results') or not self.optimization_results:
            logger.warning("No optimization results available")
            return {}
        
        # Import StrategyOptimizer
        from ..optimization.optimizer import StrategyOptimizer
        optimizer = StrategyOptimizer(self.config)
        
        optimized_strategies = {}
        for strategy_type in self.optimization_results.keys():
            if strategy_type in optimizer.best_parameters:
                optimized_strategy = optimizer.create_optimized_strategy(strategy_type)
                optimized_strategies[strategy_type] = optimized_strategy
        
        return optimized_strategies
    
    def run_optimized_backtest(self, start_date: str, end_date: str,
                             strategy_types: List[str] = None,
                             symbols: List[str] = None) -> Dict[str, Any]:
        """
        Run backtest with optimized strategies
        
        Args:
            start_date: Start date for backtest
            end_date: End date for backtest
            strategy_types: List of strategy types to optimize (if None, use all available)
            symbols: List of symbols to use for backtest
            
        Returns:
            Dictionary containing backtest results
        """
        logger.info("Running optimized backtest")
        
        # Fetch and cache data once for the entire process
        historical_data = self._fetch_and_cache_data(start_date, end_date, symbols)
        
        # Optimize strategies if not already done
        if not hasattr(self, 'optimization_results') or not self.optimization_results:
            if strategy_types is None:
                strategy_types = ['sma_crossover', 'rsi', 'macd']
            
            self.optimize_strategies(
                strategy_types=strategy_types,
                start_date=start_date,
                end_date=end_date,
                symbols=symbols
            )
        
        # Get optimized strategies
        optimized_strategies = self.get_optimized_strategies()
        
        # Add optimized strategies to engine
        for strategy_name, strategy in optimized_strategies.items():
            self.add_strategy(strategy)
        
        # Run backtest with optimized strategies using cached data
        results = self.run_backtest(start_date, end_date, symbols)
        
        return results
    
    def run_complete_optimization_workflow(self, strategy_types: List[str],
                                         start_date: str, end_date: str,
                                         optimization_metric: str = 'sharpe_ratio',
                                         max_combinations_per_strategy: int = 50,
                                         symbols: List[str] = None) -> Dict[str, Any]:
        """
        Run complete workflow: fetch data once, optimize, and backtest
        
        Args:
            strategy_types: List of strategy types to optimize
            start_date: Start date for the entire process
            end_date: End date for the entire process
            optimization_metric: Metric to optimize for
            max_combinations_per_strategy: Maximum parameter combinations to test
            symbols: List of symbols to use
            
        Returns:
            Dictionary containing both optimization and backtest results
        """
        logger.info("Starting complete optimization workflow")
        
        # Fetch data once for the entire workflow
        historical_data = self._fetch_and_cache_data(start_date, end_date, symbols)
        
        # Step 1: Run optimization
        optimization_results = self.optimize_strategies(
            strategy_types=strategy_types,
            start_date=start_date,
            end_date=end_date,
            optimization_metric=optimization_metric,
            max_combinations_per_strategy=max_combinations_per_strategy,
            symbols=symbols
        )
        
        # Step 2: Run backtest with optimized strategies
        backtest_results = self.run_optimized_backtest(
            start_date=start_date,
            end_date=end_date,
            strategy_types=strategy_types,
            symbols=symbols
        )
        
        # Combine results
        complete_results = {
            'optimization_results': optimization_results,
            'backtest_results': backtest_results,
            'data_info': {
                'start_date': start_date,
                'end_date': end_date,
                'symbols': symbols or self.config.get("trading.symbols", ["AAPL"]),
                'data_shape': historical_data.shape if historical_data is not None else None
            }
        }
        
        logger.info("Complete optimization workflow finished")
        return complete_results
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        if not hasattr(self, 'optimization_results') or not self.optimization_results:
            return {}
        
        summary = {}
        for strategy_type, results in self.optimization_results.items():
            if 'best_parameters' in results:
                best_params = results['best_parameters']
                summary[strategy_type] = {
                    'best_parameters': best_params.get('parameters', {}),
                    'best_metrics': best_params.get('metrics', {}),
                    'total_combinations_tested': results.get('total_combinations', 0),
                    'optimization_metric': results.get('optimization_metric', '')
                }
        
        return summary
    
    def save_optimization_results(self, filepath: str = "data/optimization_results.json"):
        """Save optimization results to file"""
        import json
        import os
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.optimization_results, f, indent=2, default=str)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    def load_optimization_results(self, filepath: str = "data/optimization_results.json"):
        """Load optimization results from file"""
        import json
        
        try:
            with open(filepath, 'r') as f:
                self.optimization_results = json.load(f)
            logger.info(f"Optimization results loaded from {filepath}")
        except FileNotFoundError:
            logger.warning(f"Optimization results file not found: {filepath}")
    
    def clear_cached_data(self):
        """Clear cached historical data"""
        self.historical_data = None
        self.data_start_date = None
        self.data_end_date = None
        self.data_symbols = None
        logger.info("Cached historical data cleared")
    
    def get_cached_data_info(self) -> Dict[str, Any]:
        """Get information about cached data"""
        return {
            'has_cached_data': self.historical_data is not None,
            'data_shape': self.historical_data.shape if self.historical_data is not None else None,
            'start_date': self.data_start_date,
            'end_date': self.data_end_date,
            'symbols': self.data_symbols
        }
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary with comprehensive metrics"""
        initial_capital = self.config.get("trading.initial_capital", 100000)
        total_value = self.portfolio_value
        total_return = (total_value - initial_capital) / initial_capital
        
        # Calculate annualized return
        if self.portfolio_history:
            df = pd.DataFrame(self.portfolio_history)
            df.set_index('date', inplace=True)
            daily_returns = df['total_value'].pct_change().dropna()
            
            if len(daily_returns) > 0:
                # Calculate trading days
                trading_days = len(daily_returns)
                annualized_return = ((1 + total_return) ** (252 / trading_days)) - 1 if trading_days > 0 else 0
                
                # Calculate Sharpe ratio
                volatility = daily_returns.std() * np.sqrt(252) if daily_returns.std() > 0 else 0
                sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0
                
                # Calculate maximum drawdown
                cumulative_returns = (1 + daily_returns).cumprod()
                running_max = cumulative_returns.expanding().max()
                drawdown = (cumulative_returns - running_max) / running_max
                max_drawdown = drawdown.min()
            else:
                annualized_return = 0
                sharpe_ratio = 0
                max_drawdown = 0
        else:
            annualized_return = 0
            sharpe_ratio = 0
            max_drawdown = 0
        
        # Calculate win rate from trades
        if self.trades:
            buy_trades = [t for t in self.trades if t.side == 'buy']
            sell_trades = [t for t in self.trades if t.side == 'sell']
            
            # Calculate win rate based on profitable trades
            profitable_trades = 0
            total_trades = len(sell_trades)  # Only completed trades (sells)
            
            for sell_trade in sell_trades:
                # Find corresponding buy trade
                buy_trades_for_symbol = [t for t in buy_trades if t.symbol == sell_trade.symbol]
                if buy_trades_for_symbol:
                    # Calculate P&L for this trade
                    buy_price = buy_trades_for_symbol[0].price
                    sell_price = sell_trade.price
                    if sell_price > buy_price:
                        profitable_trades += 1
            
            win_rate = (profitable_trades / total_trades) if total_trades > 0 else 0
        else:
            win_rate = 0
            total_trades = 0
        
        return {
            'initial_capital': initial_capital,
            'total_value': total_value,
            'total_return': total_return,
            'annualized_return': annualized_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': len(self.trades),
            'cash': self.cash,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'pnl': pos.pnl,
                'pnl_pct': pos.pnl_pct
            } for symbol, pos in self.positions.items()}
        } 