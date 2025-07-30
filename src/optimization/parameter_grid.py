"""
Parameter Grid - Defines parameter ranges for strategy optimization
"""

from typing import Dict, List, Any, Tuple
import itertools
from loguru import logger


class ParameterGrid:
    """Defines parameter grids for different strategy types"""
    
    def __init__(self):
        """Initialize parameter grids for different strategies"""
        self.grids = self._initialize_grids()
    
    def _initialize_grids(self) -> Dict[str, Dict[str, List]]:
        """Initialize parameter grids for each strategy type"""
        return {
            'sma_crossover': {
                'short_window': [5, 10, 15, 20, 25, 30],
                'long_window': [30, 40, 50, 60, 70, 80, 90, 100]
            },
            'rsi': {
                'period': [10, 14, 20, 30],
                'oversold': [20, 25, 30, 35],
                'overbought': [65, 70, 75, 80]
            },
            'macd': {
                'fast_period': [8, 10, 12, 15, 20],
                'slow_period': [20, 26, 30, 35, 40],
                'signal_period': [7, 9, 12, 15]
            },
            'bollinger_bands': {
                'period': [10, 15, 20, 30],
                'std_dev': [1.5, 2.0, 2.5, 3.0]
            }
        }
    
    def get_parameter_combinations(self, strategy_type: str) -> List[Dict[str, Any]]:
        """
        Get all parameter combinations for a strategy type
        
        Args:
            strategy_type: Type of strategy ('sma_crossover', 'rsi', 'macd', etc.)
            
        Returns:
            List of parameter dictionaries
        """
        if strategy_type not in self.grids:
            logger.warning(f"No parameter grid defined for strategy type: {strategy_type}")
            return []
        
        grid = self.grids[strategy_type]
        param_names = list(grid.keys())
        param_values = list(grid.values())
        
        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        
        # Convert to list of dictionaries
        param_combinations = []
        for combo in combinations:
            param_dict = dict(zip(param_names, combo))
            param_combinations.append(param_dict)
        
        logger.info(f"Generated {len(param_combinations)} parameter combinations for {strategy_type}")
        return param_combinations
    
    def get_filtered_combinations(self, strategy_type: str, 
                                filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Get parameter combinations with optional filters
        
        Args:
            strategy_type: Type of strategy
            filters: Dictionary of filters to apply (e.g., {'short_window': [10, 20]})
            
        Returns:
            Filtered list of parameter dictionaries
        """
        combinations = self.get_parameter_combinations(strategy_type)
        
        if not filters:
            return combinations
        
        filtered_combinations = []
        for combo in combinations:
            include = True
            for param, allowed_values in filters.items():
                if param in combo and combo[param] not in allowed_values:
                    include = False
                    break
            if include:
                filtered_combinations.append(combo)
        
        logger.info(f"Filtered to {len(filtered_combinations)} combinations")
        return filtered_combinations
    
    def add_custom_grid(self, strategy_type: str, parameters: Dict[str, List]):
        """
        Add custom parameter grid for a strategy type
        
        Args:
            strategy_type: Name of the strategy type
            parameters: Dictionary of parameter names to value lists
        """
        self.grids[strategy_type] = parameters
        logger.info(f"Added custom parameter grid for {strategy_type}")
    
    def get_grid_info(self, strategy_type: str) -> Dict[str, Any]:
        """
        Get information about parameter grid for a strategy type
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            Dictionary with grid information
        """
        if strategy_type not in self.grids:
            return {}
        
        grid = self.grids[strategy_type]
        total_combinations = 1
        for values in grid.values():
            total_combinations *= len(values)
        
        return {
            'parameters': list(grid.keys()),
            'parameter_ranges': grid,
            'total_combinations': total_combinations
        }
    
    def validate_parameters(self, strategy_type: str, parameters: Dict[str, Any]) -> bool:
        """
        Validate if parameters are within the defined grid
        
        Args:
            strategy_type: Type of strategy
            parameters: Parameters to validate
            
        Returns:
            True if parameters are valid
        """
        if strategy_type not in self.grids:
            return False
        
        grid = self.grids[strategy_type]
        
        for param, value in parameters.items():
            if param in grid and value not in grid[param]:
                logger.warning(f"Parameter {param}={value} not in grid for {strategy_type}")
                return False
        
        return True
    
    def get_parameter_bounds(self, strategy_type: str) -> Dict[str, Tuple]:
        """
        Get parameter bounds for a strategy type
        
        Args:
            strategy_type: Type of strategy
            
        Returns:
            Dictionary of parameter bounds (min, max)
        """
        if strategy_type not in self.grids:
            return {}
        
        grid = self.grids[strategy_type]
        bounds = {}
        
        for param, values in grid.items():
            if values:
                bounds[param] = (min(values), max(values))
        
        return bounds 