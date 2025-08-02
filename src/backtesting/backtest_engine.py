"""
Backtesting Engine - Tests trading strategies and calculates performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger

from ..strategies.base_strategy import BaseStrategy
from ..risk.risk_manager import RiskManager
from ..data.data_manager import DataManager
from ..utils.config_manager import ConfigManager


class BacktestEngine:
    """Backtesting engine for trading strategies"""
    
    def __init__(self, config: ConfigManager):
        """Initialize backtesting engine"""
        self.config = config
        self.data_manager = DataManager(config)
        self.risk_manager = RiskManager(config)
        
        # Backtest parameters
        self.initial_capital = config.get("trading.initial_capital", 100000)
        self.commission = config.get("trading.commission", 0.001)
        self.symbols = config.get("trading.symbols", ["Bitstamp:BTCUSD"])
        
        # Results storage
        self.results = {}
        self.trades = []
        self.portfolio_history = []
        
        logger.info("Backtest engine initialized")
    
    def run_backtest(self, strategies: Dict[str, BaseStrategy], 
                    start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Run backtest for multiple strategies using standardized OHLCV format
        
        Args:
            strategies: Dictionary of strategy name to strategy object
            start_date: Start date for backtest (YYYY-MM-DD)
            end_date: End date for backtest (YYYY-MM-DD)
            
        Returns:
            Dictionary containing backtest results for each strategy
        """
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Get historical data with standardized OHLCV format
        data = self.data_manager.get_historical_data_standardized(
            self.symbols, start_date, end_date
        )
        
        if data.empty:
            raise ValueError("No data available for backtest period")
        
        # Run backtest for each strategy
        for strategy_name, strategy in strategies.items():
            logger.info(f"Running backtest for strategy: {strategy_name}")
            
            # Reset strategy and risk manager
            strategy.reset()
            self.risk_manager = RiskManager(self.config)
            
            # Run single strategy backtest
            strategy_results = self._run_single_strategy_backtest(
                strategy, data, strategy_name
            )
            
            self.results[strategy_name] = strategy_results
        
        return self.results
    
    def _run_single_strategy_backtest(self, strategy: BaseStrategy, 
                                    data: pd.DataFrame, 
                                    strategy_name: str) -> Dict[str, Any]:
        """Run backtest for a single strategy"""
        
        # Initialize portfolio
        portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'total_value': self.initial_capital,
            'trades': []
        }
        
        portfolio_history = []
        
        # Iterate through each trading day
        for date, row in data.iterrows():
            current_data = row
            
            # Get historical data up to current date
            historical_data = data.loc[:date]
            
            # Generate signals
            signals = strategy.generate_signals(historical_data, current_data)
            
            # Execute trades based on signals
            portfolio = self._execute_trades(
                portfolio, signals, current_data, date, strategy
            )
            
            # Update portfolio value
            portfolio = self._update_portfolio_value(portfolio, current_data)
            
            # Update risk manager
            self.risk_manager.update_portfolio_value(portfolio['total_value'])
            
            # Record portfolio state
            portfolio_history.append({
                'date': date,
                'total_value': portfolio['total_value'],
                'cash': portfolio['cash'],
                'positions': portfolio['positions'].copy()
            })
        
        # Calculate performance metrics
        performance_metrics = self._calculate_performance_metrics(
            portfolio_history, strategy_name
        )
        
        return {
            'strategy_name': strategy_name,
            'portfolio_history': portfolio_history,
            'final_portfolio': portfolio,
            'trades': portfolio['trades'],
            'performance_metrics': performance_metrics,
            'risk_metrics': self.risk_manager.get_risk_metrics()
        }
    
    def _execute_trades(self, portfolio: Dict[str, Any], 
                       signals: Dict[str, str], 
                       current_data: pd.Series,
                       date: datetime,
                       strategy: BaseStrategy) -> Dict[str, Any]:
        """Execute trades based on signals"""
        
        for symbol in self.symbols:
            if symbol not in signals:
                continue
            
            signal = signals[symbol]
            
            # Handle both OHLCV data (MultiIndex columns) and legacy close-only data
            if isinstance(current_data.index, pd.MultiIndex):
                # New OHLCV data structure
                if (symbol, 'close') in current_data.index:
                    current_price = current_data[(symbol, 'close')]
                else:
                    logger.warning(f"No close data found for {symbol}")
                    continue
            else:
                # Legacy close-only data structure
                if symbol in current_data.index:
                    current_price = current_data[symbol]
                else:
                    logger.warning(f"No data found for {symbol}")
                    continue
            
            if pd.isna(current_price):
                continue
            
            # Get current position
            position = portfolio['positions'].get(symbol, {
                'quantity': 0,
                'avg_price': 0
            })
            
            if signal == 'buy' and position['quantity'] == 0:
                # Check if we can buy
                if self.risk_manager.can_buy(symbol, current_price, 
                                           portfolio['cash'], portfolio['positions']):
                    
                    # Calculate position size
                    position_size = self.risk_manager.calculate_position_size(
                        symbol, current_price, portfolio['cash'], portfolio['positions']
                    )
                    
                    # Calculate quantity to buy
                    quantity = position_size / current_price
                    
                    # Execute buy order
                    cost = quantity * current_price
                    commission_cost = cost * self.commission
                    total_cost = cost + commission_cost
                    
                    if total_cost <= portfolio['cash']:
                        portfolio['cash'] -= total_cost
                        portfolio['positions'][symbol] = {
                            'quantity': quantity,
                            'avg_price': current_price,
                            'entry_date': date
                        }
                        
                        # Update strategy position
                        strategy.update_position(symbol, quantity, current_price)
                        
                        # Record trade
                        portfolio['trades'].append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'buy',
                            'quantity': quantity,
                            'price': current_price,
                            'cost': total_cost,
                            'commission': commission_cost
                        })
                        
                        logger.debug(f"BUY {quantity:.2f} {symbol} @ ${current_price:.2f}")
            
            elif signal == 'sell' and position['quantity'] > 0:
                # Execute sell order
                quantity = position['quantity']
                proceeds = quantity * current_price
                commission_cost = proceeds * self.commission
                net_proceeds = proceeds - commission_cost
                
                portfolio['cash'] += net_proceeds
                portfolio['positions'][symbol] = {
                    'quantity': 0,
                    'avg_price': 0
                }
                
                # Update strategy position
                strategy.update_position(symbol, -quantity, current_price)
                
                # Record trade
                portfolio['trades'].append({
                    'date': date,
                    'symbol': symbol,
                    'action': 'sell',
                    'quantity': quantity,
                    'price': current_price,
                    'proceeds': net_proceeds,
                    'commission': commission_cost
                })
                
                logger.debug(f"SELL {quantity:.2f} {symbol} @ ${current_price:.2f}")
            
            # Check stop loss and take profit
            elif position['quantity'] > 0:
                entry_price = position['avg_price']
                
                if self.risk_manager.check_stop_loss(symbol, entry_price, current_price):
                    # Execute stop loss
                    quantity = position['quantity']
                    proceeds = quantity * current_price
                    commission_cost = proceeds * self.commission
                    net_proceeds = proceeds - commission_cost
                    
                    portfolio['cash'] += net_proceeds
                    portfolio['positions'][symbol] = {
                        'quantity': 0,
                        'avg_price': 0
                    }
                    
                    strategy.update_position(symbol, -quantity, current_price)
                    
                    portfolio['trades'].append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'stop_loss',
                        'quantity': quantity,
                        'price': current_price,
                        'proceeds': net_proceeds,
                        'commission': commission_cost
                    })
                    
                    logger.info(f"STOP LOSS: {quantity:.2f} {symbol} @ ${current_price:.2f}")
                
                elif self.risk_manager.check_take_profit(symbol, entry_price, current_price):
                    # Execute take profit
                    quantity = position['quantity']
                    proceeds = quantity * current_price
                    commission_cost = proceeds * self.commission
                    net_proceeds = proceeds - commission_cost
                    
                    portfolio['cash'] += net_proceeds
                    portfolio['positions'][symbol] = {
                        'quantity': 0,
                        'avg_price': 0
                    }
                    
                    strategy.update_position(symbol, -quantity, current_price)
                    
                    portfolio['trades'].append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'take_profit',
                        'quantity': quantity,
                        'price': current_price,
                        'proceeds': net_proceeds,
                        'commission': commission_cost
                    })
                    
                    logger.info(f"TAKE PROFIT: {quantity:.2f} {symbol} @ ${current_price:.2f}")
        
        return portfolio
    
    def _update_portfolio_value(self, portfolio: Dict[str, Any], 
                              current_data: pd.Series) -> Dict[str, Any]:
        """Update portfolio total value"""
        total_value = portfolio['cash']
        
        for symbol, position in portfolio['positions'].items():
            if position['quantity'] > 0:
                # Handle both OHLCV data (MultiIndex columns) and legacy close-only data
                if isinstance(current_data.index, pd.MultiIndex):
                    # New OHLCV data structure
                    if (symbol, 'close') in current_data.index:
                        current_price = current_data[(symbol, 'close')]
                    else:
                        logger.warning(f"No close data found for {symbol}")
                        continue
                else:
                    # Legacy close-only data structure
                    if symbol in current_data.index:
                        current_price = current_data[symbol]
                    else:
                        logger.warning(f"No data found for {symbol}")
                        continue
                
                if not pd.isna(current_price):
                    position_value = position['quantity'] * current_price
                    total_value += position_value
        
        portfolio['total_value'] = total_value
        return portfolio
    
    def _calculate_performance_metrics(self, portfolio_history: List[Dict], 
                                     strategy_name: str) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        
        if not portfolio_history:
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        # Calculate returns
        df['returns'] = df['total_value'].pct_change()
        df['cumulative_returns'] = (1 + df['returns']).cumprod()
        
        # Basic metrics
        total_return = (df['total_value'].iloc[-1] / df['total_value'].iloc[0]) - 1
        annualized_return = self._calculate_annualized_return(df)
        volatility = df['returns'].std() * np.sqrt(252)
        sharpe_ratio = self._calculate_sharpe_ratio(df)
        
        # Drawdown analysis
        drawdown_metrics = self._calculate_drawdown_metrics(df)
        
        # Trade analysis
        trade_metrics = self._calculate_trade_metrics(strategy_name)
        
        # Risk metrics from risk manager
        risk_metrics = self.risk_manager.get_risk_metrics()
        
        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': drawdown_metrics['max_drawdown'],
            'avg_drawdown': drawdown_metrics['avg_drawdown'],
            'drawdown_duration': drawdown_metrics['drawdown_duration'],
            'win_rate': trade_metrics['win_rate'],
            'profit_factor': trade_metrics['profit_factor'],
            'avg_trade': trade_metrics['avg_trade'],
            'total_trades': trade_metrics['total_trades'],
            'final_portfolio_value': df['total_value'].iloc[-1],
            'risk_metrics': risk_metrics
        }
    
    def _calculate_annualized_return(self, df: pd.DataFrame) -> float:
        """Calculate annualized return"""
        if len(df) < 2:
            return 0.0
        
        total_days = (df.index[-1] - df.index[0]).days
        if total_days == 0:
            return 0.0
        
        total_return = (df['total_value'].iloc[-1] / df['total_value'].iloc[0]) - 1
        annualized_return = ((1 + total_return) ** (365 / total_days)) - 1
        
        return annualized_return
    
    def _calculate_sharpe_ratio(self, df: pd.DataFrame, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        if len(df) < 2:
            return 0.0
        
        returns = df['returns'].dropna()
        if len(returns) == 0:
            return 0.0
        
        excess_returns = returns - (risk_free_rate / 252)  # Daily risk-free rate
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        
        return sharpe_ratio
    
    def _calculate_drawdown_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate drawdown metrics"""
        if len(df) < 2:
            return {'max_drawdown': 0.0, 'avg_drawdown': 0.0, 'drawdown_duration': 0}
        
        # Ensure we're working with numeric data only
        try:
            # Filter only numeric columns for total_value
            numeric_df = df.select_dtypes(include=[np.number])
            if 'total_value' not in numeric_df.columns:
                # If total_value is not numeric, try to convert it
                if 'total_value' in df.columns:
                    df['total_value'] = pd.to_numeric(df['total_value'], errors='coerce')
                    numeric_df = df.select_dtypes(include=[np.number])
            
            if 'total_value' not in numeric_df.columns:
                logger.warning("total_value column not found or not numeric")
                return {'max_drawdown': 0.0, 'avg_drawdown': 0.0, 'drawdown_duration': 0}
            
            # Calculate running maximum
            running_max = numeric_df['total_value'].expanding().max()
            drawdown = (numeric_df['total_value'] - running_max) / running_max
            
            max_drawdown = drawdown.min()
            avg_drawdown = drawdown.mean()
            
            # Calculate drawdown duration - ensure drawdown is numeric
            drawdown_numeric = pd.to_numeric(drawdown, errors='coerce')
            drawdown_periods = (drawdown_numeric < 0).sum()
            total_periods = len(drawdown_numeric)
            drawdown_duration = drawdown_periods / total_periods if total_periods > 0 else 0
            
            return {
                'max_drawdown': max_drawdown,
                'avg_drawdown': avg_drawdown,
                'drawdown_duration': drawdown_duration
            }
        except Exception as e:
            logger.warning(f"Error calculating drawdown metrics: {e}")
            return {'max_drawdown': 0.0, 'avg_drawdown': 0.0, 'drawdown_duration': 0}
    
    def _calculate_trade_metrics(self, strategy_name: str) -> Dict[str, Any]:
        """Calculate trade-based metrics"""
        if strategy_name not in self.results:
            return {'win_rate': 0.0, 'profit_factor': 0.0, 'avg_trade': 0.0, 'total_trades': 0}
        
        trades = self.results[strategy_name]['trades']
        if not trades:
            return {'win_rate': 0.0, 'profit_factor': 0.0, 'avg_trade': 0.0, 'total_trades': 0}
        
        # Calculate trade profits/losses
        trade_pnls = []
        for i in range(0, len(trades), 2):  # Pairs of buy/sell trades
            if i + 1 < len(trades):
                buy_trade = trades[i]
                sell_trade = trades[i + 1]
                
                if buy_trade['action'] == 'buy' and sell_trade['action'] in ['sell', 'stop_loss', 'take_profit']:
                    pnl = sell_trade['proceeds'] - buy_trade['cost']
                    trade_pnls.append(pnl)
        
        if not trade_pnls:
            return {'win_rate': 0.0, 'profit_factor': 0.0, 'avg_trade': 0.0, 'total_trades': 0}
        
        winning_trades = [pnl for pnl in trade_pnls if pnl > 0]
        losing_trades = [pnl for pnl in trade_pnls if pnl < 0]
        
        win_rate = len(winning_trades) / len(trade_pnls) if trade_pnls else 0.0
        avg_trade = np.mean(trade_pnls) if trade_pnls else 0.0
        
        total_profit = sum(winning_trades) if winning_trades else 0.0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0.0
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        return {
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_trade': avg_trade,
            'total_trades': len(trade_pnls)
        }
    
    def generate_report(self, strategy_name: str = None) -> str:
        """Generate a comprehensive backtest report"""
        if not self.results:
            return "No backtest results available"
        
        report = []
        report.append("=" * 60)
        report.append("BACKTEST REPORT")
        report.append("=" * 60)
        
        if strategy_name and strategy_name in self.results:
            results = self.results[strategy_name]
            report.extend(self._format_strategy_report(strategy_name, results))
        else:
            # Generate report for all strategies
            for name, results in self.results.items():
                report.extend(self._format_strategy_report(name, results))
                report.append("-" * 40)
        
        return "\n".join(report)
    
    def _format_strategy_report(self, strategy_name: str, results: Dict[str, Any]) -> List[str]:
        """Format report for a single strategy"""
        metrics = results['performance_metrics']
        risk_metrics = results['risk_metrics']
        
        report = []
        report.append(f"\nStrategy: {strategy_name}")
        report.append(f"Final Portfolio Value: ${metrics.get('final_portfolio_value', 0):,.2f}")
        report.append(f"Total Return: {metrics.get('total_return', 0):.2%}")
        report.append(f"Annualized Return: {metrics.get('annualized_return', 0):.2%}")
        report.append(f"Volatility: {metrics.get('volatility', 0):.2%}")
        report.append(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        report.append(f"Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        report.append(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
        report.append(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
        report.append(f"Total Trades: {metrics.get('total_trades', 0)}")
        report.append(f"Average Trade: ${metrics.get('avg_trade', 0):,.2f}")
        
        # Risk metrics
        if risk_metrics:
            report.append(f"Current Drawdown: {risk_metrics.get('current_drawdown', 0):.2%}")
            report.append(f"Portfolio Volatility: {risk_metrics.get('volatility', 0):.2%}")
        
        return report
    
    def plot_results(self, strategy_name: str = None, save_path: str = None):
        """Plot backtest results"""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        if strategy_name and strategy_name in self.results:
            self._plot_single_strategy(strategy_name, save_path)
        else:
            self._plot_all_strategies(save_path)
    
    def _plot_single_strategy(self, strategy_name: str, save_path: str = None):
        """Plot results for a single strategy"""
        results = self.results[strategy_name]
        portfolio_history = results['portfolio_history']
        
        df = pd.DataFrame(portfolio_history)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Backtest Results: {strategy_name}', fontsize=16)
        
        # Portfolio value over time
        axes[0, 0].plot(df.index, df['total_value'])
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Returns distribution
        returns = df['total_value'].pct_change().dropna()
        axes[0, 1].hist(returns, bins=50, alpha=0.7)
        axes[0, 1].set_title('Returns Distribution')
        axes[0, 1].set_xlabel('Returns')
        axes[0, 1].grid(True)
        
        # Drawdown
        running_max = df['total_value'].expanding().max()
        drawdown = (df['total_value'] - running_max) / running_max
        axes[1, 0].fill_between(df.index, drawdown, 0, alpha=0.3, color='red')
        axes[1, 0].set_title('Drawdown')
        axes[1, 0].set_ylabel('Drawdown (%)')
        axes[1, 0].grid(True)
        
        # Cash vs Positions
        axes[1, 1].plot(df.index, df['cash'], label='Cash')
        axes[1, 1].plot(df.index, df['total_value'] - df['cash'], label='Positions')
        axes[1, 1].set_title('Cash vs Positions')
        axes[1, 1].set_ylabel('Value ($)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def _plot_all_strategies(self, save_path: str = None):
        """Plot results for all strategies"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Backtest Results Comparison', fontsize=16)
        
        for strategy_name, results in self.results.items():
            portfolio_history = results['portfolio_history']
            df = pd.DataFrame(portfolio_history)
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            
            # Portfolio value comparison
            axes[0, 0].plot(df.index, df['total_value'], label=strategy_name)
            
            # Cumulative returns
            returns = df['total_value'].pct_change()
            cumulative_returns = (1 + returns).cumprod()
            axes[0, 1].plot(df.index, cumulative_returns, label=strategy_name)
        
        axes[0, 0].set_title('Portfolio Value Comparison')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('Cumulative Returns')
        axes[0, 1].set_ylabel('Cumulative Returns')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Performance metrics comparison
        metrics_data = []
        strategy_names = []
        
        for strategy_name, results in self.results.items():
            metrics = results['performance_metrics']
            metrics_data.append([
                metrics.get('total_return', 0),
                metrics.get('sharpe_ratio', 0),
                metrics.get('max_drawdown', 0),
                metrics.get('win_rate', 0)
            ])
            strategy_names.append(strategy_name)
        
        metrics_df = pd.DataFrame(
            metrics_data,
            index=strategy_names,
            columns=['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
        )
        
        # Heatmap of metrics
        sns.heatmap(metrics_df.T, annot=True, fmt='.2f', cmap='RdYlGn', ax=axes[1, 0])
        axes[1, 0].set_title('Performance Metrics Comparison')
        
        # Bar chart of total returns
        returns = [metrics.get('total_return', 0) for metrics in metrics_data]
        axes[1, 1].bar(strategy_names, returns)
        axes[1, 1].set_title('Total Returns by Strategy')
        axes[1, 1].set_ylabel('Total Return')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def save_results(self, filepath: str):
        """Save backtest results to file"""
        import json
        
        # Convert results to serializable format
        serializable_results = {}
        for strategy_name, results in self.results.items():
            serializable_results[strategy_name] = {
                'strategy_name': results['strategy_name'],
                'performance_metrics': results['performance_metrics'],
                'risk_metrics': results['risk_metrics'],
                'final_portfolio': {
                    'cash': results['final_portfolio']['cash'],
                    'total_value': results['final_portfolio']['total_value']
                }
            }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def get_best_strategy(self) -> Optional[str]:
        """Get the best performing strategy based on Sharpe ratio"""
        if not self.results:
            return None
        
        best_strategy = None
        best_sharpe = float('-inf')
        
        for strategy_name, results in self.results.items():
            sharpe_ratio = results['performance_metrics'].get('sharpe_ratio', 0)
            if sharpe_ratio > best_sharpe:
                best_sharpe = sharpe_ratio
                best_strategy = strategy_name
        
        return best_strategy 