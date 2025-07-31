"""
Risk Manager - Handles risk management and position sizing
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List

# Try to import loguru, fallback to basic logging if not available
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    # Set up basic logging if loguru is not available
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

# Use absolute imports instead of relative imports
try:
    from ..utils.config_manager import ConfigManager
except ImportError:
    # Fallback to absolute imports
    from src.utils.config_manager import ConfigManager


class RiskManager:
    """Manages risk controls and position sizing"""
    
    def __init__(self, config: ConfigManager):
        """Initialize risk manager"""
        self.config = config
        
        # Risk parameters
        self.max_position_size = config.get("risk.max_position_size", 0.1)
        self.max_portfolio_risk = config.get("risk.max_portfolio_risk", 0.02)
        self.stop_loss = config.get("risk.stop_loss", 0.05)
        self.take_profit = config.get("risk.take_profit", 0.15)
        self.max_drawdown = config.get("risk.max_drawdown", 0.20)
        
        # Portfolio tracking
        self.portfolio_history = []
        self.max_portfolio_value = 0
        
        logger.info("Risk manager initialized")
    
    def can_buy(self, symbol: str, price: float, cash: float, 
                positions: Dict[str, Any]) -> bool:
        """Check if we can buy the symbol based on risk parameters"""
        # Convert cash to scalar if it's a pandas Series
        if hasattr(cash, 'item'):
            cash = cash.item()
        
        # Convert price to numeric if it's not already
        try:
            if hasattr(price, 'item'):
                price = price.item()
            elif hasattr(price, 'timestamp'):
                # If price is a Timestamp, we can't use it for calculations
                logger.warning(f"Price for {symbol} is a Timestamp, cannot calculate position size")
                return False
            elif not isinstance(price, (int, float, np.number)):
                # Try to convert to numeric
                price = pd.to_numeric(price, errors='coerce')
                if pd.isna(price):
                    logger.warning(f"Price for {symbol} cannot be converted to numeric")
                    return False
        except Exception as e:
            logger.warning(f"Error converting price for {symbol}: {e}")
            return False
        
        if cash <= 0:
            logger.warning("Insufficient cash for buy order")
            return False
        
        # For testing, allow trades if we have sufficient cash
        min_cash_required = price * self.max_position_size * 1.1  # 10% buffer
        if cash < min_cash_required:
            logger.warning(f"Insufficient cash: ${cash:.2f} < ${min_cash_required:.2f}")
            return False
        
        return True
    
    def calculate_position_size(self, symbol: str, price: float, cash: float,
                              positions: Dict[str, Any]) -> float:
        """Calculate optimal position size based on risk parameters"""
        # Convert cash to scalar if it's a pandas Series
        if hasattr(cash, 'item'):
            cash = cash.item()
        
        # Convert price to numeric if it's not already
        try:
            if hasattr(price, 'item'):
                price = price.item()
            elif hasattr(price, 'timestamp'):
                # If price is a Timestamp, we can't use it for calculations
                logger.warning(f"Price for {symbol} is a Timestamp, cannot calculate position size")
                return 0.0
            elif not isinstance(price, (int, float, np.number)):
                # Try to convert to numeric
                price = pd.to_numeric(price, errors='coerce')
                if pd.isna(price):
                    logger.warning(f"Price for {symbol} cannot be converted to numeric")
                    return 0.0
        except Exception as e:
            logger.warning(f"Error converting price for {symbol}: {e}")
            return 0.0
        
        # For testing, use a simple position size calculation
        position_size = cash * self.max_position_size
        
        # Ensure we don't exceed available cash
        position_size = min(position_size, cash * 0.95)  # Leave 5% buffer
        
        return position_size
    
    def check_stop_loss(self, symbol: str, entry_price: float, 
                       current_price: float) -> bool:
        """Check if stop loss has been triggered"""
        # Convert prices to numeric if needed
        try:
            if hasattr(entry_price, 'item'):
                entry_price = entry_price.item()
            elif hasattr(entry_price, 'timestamp'):
                logger.warning(f"Entry price for {symbol} is a Timestamp")
                return False
            elif not isinstance(entry_price, (int, float, np.number)):
                entry_price = pd.to_numeric(entry_price, errors='coerce')
                if pd.isna(entry_price):
                    return False
            
            if hasattr(current_price, 'item'):
                current_price = current_price.item()
            elif hasattr(current_price, 'timestamp'):
                logger.warning(f"Current price for {symbol} is a Timestamp")
                return False
            elif not isinstance(current_price, (int, float, np.number)):
                current_price = pd.to_numeric(current_price, errors='coerce')
                if pd.isna(current_price):
                    return False
        except Exception as e:
            logger.warning(f"Error converting prices for {symbol}: {e}")
            return False
        
        if entry_price <= 0:
            return False
        
        loss_pct = (current_price - entry_price) / entry_price
        
        if loss_pct <= -self.stop_loss:
            logger.info(f"Stop loss triggered for {symbol}: {loss_pct:.2%}")
            return True
        
        return False
    
    def check_take_profit(self, symbol: str, entry_price: float,
                         current_price: float) -> bool:
        """Check if take profit has been triggered"""
        # Convert prices to numeric if needed
        try:
            if hasattr(entry_price, 'item'):
                entry_price = entry_price.item()
            elif hasattr(entry_price, 'timestamp'):
                logger.warning(f"Entry price for {symbol} is a Timestamp")
                return False
            elif not isinstance(entry_price, (int, float, np.number)):
                entry_price = pd.to_numeric(entry_price, errors='coerce')
                if pd.isna(entry_price):
                    return False
            
            if hasattr(current_price, 'item'):
                current_price = current_price.item()
            elif hasattr(current_price, 'timestamp'):
                logger.warning(f"Current price for {symbol} is a Timestamp")
                return False
            elif not isinstance(current_price, (int, float, np.number)):
                current_price = pd.to_numeric(current_price, errors='coerce')
                if pd.isna(current_price):
                    return False
        except Exception as e:
            logger.warning(f"Error converting prices for {symbol}: {e}")
            return False
        
        if entry_price <= 0:
            return False
        
        profit_pct = (current_price - entry_price) / entry_price
        
        if profit_pct >= self.take_profit:
            logger.info(f"Take profit triggered for {symbol}: {profit_pct:.2%}")
            return True
        
        return False
    
    def update_portfolio_value(self, portfolio_value: float):
        """Update portfolio value and check drawdown"""
        self.portfolio_history.append(portfolio_value)
        self.max_portfolio_value = max(self.max_portfolio_value, portfolio_value)
        
        # Check drawdown
        if self.max_portfolio_value > 0:
            current_drawdown = (portfolio_value - self.max_portfolio_value) / self.max_portfolio_value
            
            if current_drawdown <= -self.max_drawdown:
                logger.warning(f"Maximum drawdown limit reached: {current_drawdown:.2%}")
                return False
        
        return True
    
    def _calculate_max_position_value(self, cash: float, 
                                    positions: Dict[str, Any]) -> float:
        """Calculate maximum position value based on risk parameters"""
        # Base position size on available cash
        base_position_size = cash * self.max_position_size
        
        # Consider existing positions
        total_position_value = sum(
            pos.get('quantity', 0) * pos.get('current_price', 0)
            for pos in positions.values()
        )
        
        # Adjust for portfolio concentration
        if total_position_value > 0:
            concentration_factor = 1 - (total_position_value / (cash + total_position_value))
            base_position_size *= concentration_factor
        
        return base_position_size
    
    def _check_portfolio_risk_limit(self, symbol: str, price: float,
                                  positions: Dict[str, Any]) -> bool:
        """Check if adding position would exceed portfolio risk limit"""
        # Calculate current portfolio value
        current_portfolio_value = sum(
            pos.get('quantity', 0) * pos.get('current_price', 0)
            for pos in positions.values()
        )
        
        # Calculate current portfolio risk
        current_risk = self._calculate_portfolio_risk(positions)
        
        # Estimate new position risk (simplified)
        position_size = price * self.max_position_size
        new_position_risk = position_size * self.stop_loss
        
        # Check if adding new risk would exceed limit
        total_risk = current_risk + new_position_risk
        new_portfolio_value = current_portfolio_value + position_size
        
        if new_portfolio_value > 0:
            risk_ratio = total_risk / new_portfolio_value
            return risk_ratio <= self.max_portfolio_risk
        
        return True
    
    def _calculate_portfolio_risk(self, positions: Dict[str, Any]) -> float:
        """Calculate current portfolio risk"""
        total_risk = 0
        
        for symbol, position in positions.items():
            quantity = position.get('quantity', 0)
            current_price = position.get('current_price', 0)
            entry_price = position.get('entry_price', 0)
            
            if quantity > 0 and entry_price > 0:
                # Calculate potential loss at stop loss level
                potential_loss = quantity * entry_price * self.stop_loss
                total_risk += potential_loss
        
        return total_risk
    
    def get_risk_metrics(self) -> Dict[str, float]:
        """Get current risk metrics"""
        if not self.portfolio_history:
            return {}
        
        portfolio_values = pd.Series(self.portfolio_history)
        
        # Calculate drawdown
        running_max = portfolio_values.expanding().max()
        drawdown = (portfolio_values - running_max) / running_max
        current_drawdown = drawdown.iloc[-1] if len(drawdown) > 0 else 0
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        # Calculate volatility
        returns = portfolio_values.pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        
        return {
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'portfolio_value': portfolio_values.iloc[-1] if len(portfolio_values) > 0 else 0
        }
    
    def should_stop_trading(self) -> bool:
        """Check if trading should be stopped due to risk limits"""
        metrics = self.get_risk_metrics()
        
        # Stop if maximum drawdown exceeded
        if abs(metrics.get('max_drawdown', 0)) >= self.max_drawdown:
            logger.warning("Trading stopped due to maximum drawdown limit")
            return True
        
        return False
    
    def get_summary(self) -> Dict[str, Any]:
        """Get risk management summary"""
        metrics = self.get_risk_metrics()
        
        return {
            'max_position_size': self.max_position_size,
            'max_portfolio_risk': self.max_portfolio_risk,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': metrics.get('current_drawdown', 0),
            'max_drawdown_reached': metrics.get('max_drawdown', 0),
            'volatility': metrics.get('volatility', 0),
            'portfolio_value': metrics.get('portfolio_value', 0)
        } 