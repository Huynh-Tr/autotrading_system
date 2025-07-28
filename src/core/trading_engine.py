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


class TradingEngine:
    """
    Main trading engine that coordinates all components
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize the trading engine"""
        self.config = ConfigManager(config_path)
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
        
        # Historical data for strategies
        self.historical_data = None
        
        # Strategies
        self.strategies: Dict[str, BaseStrategy] = {}
        
        logger.info(f"Trading engine initialized with ${self.cash:,.2f} initial capital")
    
    def add_strategy(self, strategy: BaseStrategy):
        """Add a trading strategy to the engine"""
        self.strategies[strategy.name] = strategy
        logger.info(f"Added strategy: {strategy.name}")
    
    def run_backtest(self, start_date: str, end_date: str):
        """Run backtesting simulation"""
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Get historical data
        data = self.data_manager.get_historical_data(
            symbols=self.config.get("trading.symbols"),
            start_date=start_date,
            end_date=end_date,
            interval=self.config.get("data.interval", "1d")
        )
        
        # Validate and clean data
        if not self.data_manager.validate_data(data):
            logger.error("Invalid data received")
            return
        
        data = self.data_manager.clean_data(data)
        
        if data.empty:
            logger.error("No valid data available for backtesting")
            return
        
        logger.info(f"Loaded {len(data)} data points for {list(data.columns)}")
        
        # Store historical data for strategies
        self.historical_data = data
        
        # Run simulation
        for i, (date, row) in enumerate(data.iterrows()):
            self._process_trading_day(date, row, i)
            self._update_portfolio_value(date, row)
            self.portfolio_history.append({
                'date': date,
                'portfolio_value': self.portfolio_value,
                'cash': self.cash,
                'positions_value': self.portfolio_value - self.cash
            })
        
        self._generate_backtest_report()
    
    # def run_live_trading(self):
    #     """Run live trading (placeholder for real implementation)"""
    #     logger.info("Starting live trading mode")
    #     # This would connect to real-time data feeds and execute real trades
    #     # For now, this is a placeholder
    #     pass
    
    def _process_trading_day(self, date: datetime, market_data: pd.Series, current_index: int):
        """Process trading decisions for a single day"""
        for strategy_name, strategy in self.strategies.items():
            if not self.config.get(f"strategies.{strategy_name}.enabled", True):
                continue
                
            # Get historical data up to current point for strategy
            historical_slice = self.historical_data.iloc[:current_index + 1]
            signals = strategy.generate_signals(historical_slice, market_data)
            
            for symbol, signal in signals.items():
                if signal == 'buy':
                    self._execute_buy_order(symbol, date, market_data[symbol])
                elif signal == 'sell':
                    self._execute_sell_order(symbol, date, market_data[symbol])
    
    def _execute_buy_order(self, symbol: str, date: datetime, price: float):
        """Execute a buy order"""
        # Check risk management rules
        if not self.risk_manager.can_buy(symbol, price, self.cash, self.positions):
            return
        
        # Calculate position size
        position_size = self.risk_manager.calculate_position_size(
            symbol, price, self.cash, self.positions
        )
        
        if position_size <= 0:
            return
        
        # Calculate quantity
        quantity = position_size / price
        commission = position_size * self.config.get("trading.commission", 0.001)
        total_cost = position_size + commission
        
        if total_cost > self.cash:
            return
        
        # Execute trade
        self.cash -= total_cost
        
        if symbol in self.positions:
            # Add to existing position
            pos = self.positions[symbol]
            total_quantity = pos.quantity + quantity
            avg_price = ((pos.quantity * pos.entry_price) + (quantity * price)) / total_quantity
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=total_quantity,
                entry_price=avg_price,
                entry_time=pos.entry_time,
                current_price=price,
                pnl=0,
                pnl_pct=0
            )
        else:
            # Create new position
            self.positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                entry_price=price,
                entry_time=date,
                current_price=price,
                pnl=0,
                pnl_pct=0
            )
        
        # Record trade
        self.trades.append(Trade(
            symbol=symbol,
            side='buy',
            quantity=quantity,
            price=price,
            timestamp=date,
            commission=commission,
            strategy='strategy'
        ))
        
        logger.info(f"BUY: {quantity:.2f} {symbol} @ ${price:.2f}")
    
    def _execute_sell_order(self, symbol: str, date: datetime, price: float):
        """Execute a sell order"""
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        if position.quantity <= 0:
            return
        
        # Sell entire position
        quantity = position.quantity
        proceeds = quantity * price
        commission = proceeds * self.config.get("trading.commission", 0.001)
        net_proceeds = proceeds - commission
        
        # Calculate P&L
        pnl = net_proceeds - (quantity * position.entry_price)
        pnl_pct = (pnl / (quantity * position.entry_price)) * 100
        
        # Update cash and remove position
        self.cash += net_proceeds
        del self.positions[symbol]
        
        # Record trade
        self.trades.append(Trade(
            symbol=symbol,
            side='sell',
            quantity=quantity,
            price=price,
            timestamp=date,
            commission=commission,
            strategy='strategy'
        ))
        
        logger.info(f"SELL: {quantity:.2f} {symbol} @ ${price:.2f} | P&L: ${pnl:.2f} ({pnl_pct:.2f}%)")
    
    def _update_portfolio_value(self, date: datetime, market_data: pd.Series):
        """Update portfolio value and position P&L"""
        positions_value = 0
        
        for symbol, position in self.positions.items():
            if symbol in market_data:
                current_price = market_data[symbol]
                position.current_price = current_price
                position.pnl = (current_price - position.entry_price) * position.quantity
                position.pnl_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                positions_value += position.quantity * current_price
        
        self.portfolio_value = self.cash + positions_value
    
    def _generate_backtest_report(self):
        """Generate backtest performance report"""
        if not self.portfolio_history:
            logger.warning("No portfolio history available for backtest report")
            return
        
        df = pd.DataFrame(self.portfolio_history)
        df.set_index('date', inplace=True)
        
        # Calculate metrics
        initial_capital = self.config.get("trading.initial_capital")
        total_return = (self.portfolio_value - initial_capital) / initial_capital
        daily_returns = df['portfolio_value'].pct_change().dropna()
        
        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0  # Annualized
        sharpe_ratio = (daily_returns.mean() * 252) / volatility if volatility > 0 else 0
        
        # Maximum drawdown
        if len(daily_returns) > 0:
            cumulative_returns = (1 + daily_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
        else:
            max_drawdown = 0
        
        # Calculate trade statistics
        buy_trades = [t for t in self.trades if t.side == 'buy']
        sell_trades = [t for t in self.trades if t.side == 'sell']
        
        logger.info("=== BACKTEST RESULTS ===")
        logger.info(f"Initial Capital: ${initial_capital:,.2f}")
        logger.info(f"Final Portfolio Value: ${self.portfolio_value:,.2f}")
        logger.info(f"Total Return: {total_return:.2%}")
        logger.info(f"Annualized Volatility: {volatility:.2%}")
        logger.info(f"Sharpe Ratio: {sharpe_ratio:.2f}")
        logger.info(f"Maximum Drawdown: {max_drawdown:.2%}")
        logger.info(f"Total Trades: {len(self.trades)}")
        logger.info(f"Buy Trades: {len(buy_trades)}")
        logger.info(f"Sell Trades: {len(sell_trades)}")
        
        # Save detailed results
        import os
        os.makedirs('data', exist_ok=True)
        
        # Save portfolio history
        df.to_csv('data/backtest_results.csv')
        logger.info("Backtest results saved to data/backtest_results.csv")
        
        # Save trade details
        if self.trades:
            trades_df = pd.DataFrame([
                {
                    'timestamp': t.timestamp,
                    'symbol': t.symbol,
                    'side': t.side,
                    'quantity': t.quantity,
                    'price': t.price,
                    'commission': t.commission,
                    'strategy': t.strategy
                }
                for t in self.trades
            ])
            trades_df.to_csv('data/trades.csv', index=False)
            logger.info("Trade details saved to data/trades.csv")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """Get current portfolio summary"""
        return {
            'cash': self.cash,
            'portfolio_value': self.portfolio_value,
            'positions': {symbol: {
                'quantity': pos.quantity,
                'entry_price': pos.entry_price,
                'current_price': pos.current_price,
                'pnl': pos.pnl,
                'pnl_pct': pos.pnl_pct
            } for symbol, pos in self.positions.items()},
            'total_trades': len(self.trades)
        } 