"""
Base Strategy - Abstract base class for all trading strategies
"""

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, Any, Optional

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


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        """Initialize base strategy"""
        self.name = name
        self.config = config
        self.positions = {}
        self.signals = {}
        
        logger.info(f"Initialized strategy: {name}")
    
    @abstractmethod
    def generate_signals(self, historical_data: pd.DataFrame, current_data: pd.Series) -> Dict[str, str]:
        """
        Generate trading signals for given market data
        
        Args:
            historical_data: Historical market data up to current point
            current_data: Current day's market data
            
        Returns:
            Dict mapping symbol to signal ('buy', 'sell', 'hold')
        """
        pass
    
    def update_position(self, symbol: str, quantity: float, price: float):
        """Update strategy's position tracking"""
        if symbol not in self.positions:
            self.positions[symbol] = {
                'quantity': 0,
                'avg_price': 0,
                'total_cost': 0
            }
        
        pos = self.positions[symbol]
        
        if quantity > 0:  # Buy
            total_quantity = pos['quantity'] + quantity
            total_cost = pos['total_cost'] + (quantity * price)
            pos['quantity'] = total_quantity
            pos['total_cost'] = total_cost
            pos['avg_price'] = total_cost / total_quantity
        else:  # Sell
            pos['quantity'] += quantity  # quantity is negative for sell
            if pos['quantity'] <= 0:
                # Position closed
                pos['quantity'] = 0
                pos['avg_price'] = 0
                pos['total_cost'] = 0
    
    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for a symbol"""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if strategy has a position in symbol"""
        return symbol in self.positions and self.positions[symbol]['quantity'] > 0
    
    def calculate_pnl(self, symbol: str, current_price: float) -> float:
        """Calculate unrealized P&L for a position"""
        if not self.has_position(symbol):
            return 0.0
        
        position = self.positions[symbol]
        return (current_price - position['avg_price']) * position['quantity']
    
    def validate_config(self) -> bool:
        """Validate strategy configuration"""
        return True
    
    def reset(self):
        """Reset strategy state"""
        self.positions = {}
        self.signals = {}
        logger.info(f"Reset strategy: {self.name}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get strategy summary"""
        return {
            'name': self.name,
            'positions': self.positions,
            'total_positions': len([p for p in self.positions.values() if p['quantity'] > 0])
        } 