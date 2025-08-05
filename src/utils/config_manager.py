"""
Configuration Manager - Handles system configuration
"""

import yaml
import os
from typing import Any, Dict, Optional

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


class ConfigManager:
    """Manages configuration settings with dot notation access"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Initialize configuration manager"""
        self.config_path = config_path
        self.config = self._load_config()
        logger.info(f"Configuration loaded from {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            logger.warning(f"Config file {self.config_path} not found, using defaults")
            return self._get_default_config()
        
        try:
            with open(self.config_path, 'r') as file:
                config = yaml.safe_load(file)
                return config or {}
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self._get_default_config()
    
    # def _get_default_config(self) -> Dict[str, Any]:
        # """Get default configuration"""
        # return {
        #     'trading': {
        #         'mode': 'backtest',
        #         'symbols': ['AAPL', 'GOOGL', 'MSFT'],
        #         'initial_capital': 100000,
        #         'commission': 0.001
        #     },
        #     'data': {
        #         'source': 'yfinance',
        #         'start_date': '2023-01-01',
        #         'end_date': '2024-01-01',
        #         'interval': '1d',
        #         'cache_data': True,
        #         'cache_dir': 'data/cache'
        #     },
        #     'risk': {
        #         'max_position_size': 0.1,
        #         'max_portfolio_risk': 0.02,
        #         'stop_loss': 0.05,
        #         'take_profit': 0.15,
        #         'max_drawdown': 0.20
        #     },
        #     'logging': {
        #         'level': 'INFO',
        #         'file': 'logs/trading.log',
        #         'max_size': '10MB',
        #         'backup_count': 5
        #     }
        # }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    # def set(self, key: str, value: Any):
        # """Set configuration value using dot notation"""
        # keys = key.split('.')
        # config = self.config
        
        # # Navigate to the parent of the target key
        # for k in keys[:-1]:
        #     if k not in config:
        #         config[k] = {}
        #     config = config[k]
        
        # # Set the value
        # config[keys[-1]] = value
    
    # def save(self, config_path: Optional[str] = None):
        # """Save configuration to file"""
        # path = config_path or self.config_path
        
        # # Ensure directory exists
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # try:
        #     with open(path, 'w') as file:
        #         yaml.dump(self.config, file, default_flow_style=False, indent=2)
        #     logger.info(f"Configuration saved to {path}")
        # except Exception as e:
        #     logger.error(f"Error saving config: {e}")
    
    # def reload(self):
        # """Reload configuration from file"""
        # self.config = self._load_config()
        # logger.info("Configuration reloaded")
    
    # def get_all(self) -> Dict[str, Any]:
        # """Get all configuration"""
        # return self.config.copy()
    
    # def validate(self) -> bool:
        # """Validate configuration"""
        # required_keys = [
        #     'trading.initial_capital',
        #     'trading.symbols',
        #     'data.source',
        #     'risk.max_position_size'
        # ]
        
        # for key in required_keys:
        #     if self.get(key) is None:
        #         logger.error(f"Missing required configuration: {key}")
        #         return False
        
        # return True 