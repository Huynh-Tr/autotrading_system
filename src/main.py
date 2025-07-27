"""
Main entry point for the Auto Trading System
"""

import argparse
import sys
import os
from datetime import datetime
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.trading_engine import TradingEngine
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.utils.config_manager import ConfigManager


def setup_logging(config: ConfigManager):
    """Setup logging configuration"""
    log_file = config.get("logging.file", "logs/trading.log")
    log_level = config.get("logging.level", "INFO")
    
    # Ensure log directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure loguru
    logger.remove()  # Remove default handler
    logger.add(
        sys.stdout,
        level=log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    logger.add(
        log_file,
        level=log_level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation=config.get("logging.max_size", "10MB"),
        retention=config.get("logging.backup_count", 5)
    )


def run_backtest(config_path: str, strategy_name: str):
    """Run backtesting simulation"""
    logger.info("Starting backtest mode")
    
    # Load configuration
    config = ConfigManager(config_path)
    setup_logging(config)
    
    # Initialize trading engine
    engine = TradingEngine(config_path)
    
    # Add strategies
    if strategy_name == "sma_crossover":
        strategy_config = config.get("strategies.sma_crossover", {})
        strategy = SMACrossoverStrategy(strategy_config)
        engine.add_strategy(strategy)
    elif strategy_name == "rsi":
        strategy_config = config.get("strategies.rsi", {})
        strategy = RSIStrategy(strategy_config)
        engine.add_strategy(strategy)
    elif strategy_name == "macd":
        strategy_config = config.get("strategies.macd", {})
        strategy = MACDStrategy(strategy_config)
        engine.add_strategy(strategy)
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return
    
    # Get backtest parameters
    start_date = config.get("data.start_date", "2023-01-01")
    end_date = config.get("data.end_date", "2024-01-01")
    
    # Run backtest
    engine.run_backtest(start_date, end_date)
    
    logger.info("Backtest completed")


def run_live_trading(config_path: str, strategy_name: str):
    """Run live trading (placeholder)"""
    logger.info("Starting live trading mode")
    
    # Load configuration
    config = ConfigManager(config_path)
    setup_logging(config)
    
    # Initialize trading engine
    engine = TradingEngine(config_path)
    
    # Add strategies
    if strategy_name == "sma_crossover":
        strategy_config = config.get("strategies.sma_crossover", {})
        strategy = SMACrossoverStrategy(strategy_config)
        engine.add_strategy(strategy)
    elif strategy_name == "rsi":
        strategy_config = config.get("strategies.rsi", {})
        strategy = RSIStrategy(strategy_config)
        engine.add_strategy(strategy)
    elif strategy_name == "macd":
        strategy_config = config.get("strategies.macd", {})
        strategy = MACDStrategy(strategy_config)
        engine.add_strategy(strategy)
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return
    
    # Run live trading
    engine.run_live_trading()
    
    logger.info("Live trading completed")


def run_paper_trading(config_path: str, strategy_name: str):
    """Run paper trading (simulated live trading)"""
    logger.info("Starting paper trading mode")
    
    # Load configuration
    config = ConfigManager(config_path)
    setup_logging(config)
    
    # Initialize trading engine
    engine = TradingEngine(config_path)
    
    # Add strategies
    if strategy_name == "sma_crossover":
        strategy_config = config.get("strategies.sma_crossover", {})
        strategy = SMACrossoverStrategy(strategy_config)
        engine.add_strategy(strategy)
    elif strategy_name == "rsi":
        strategy_config = config.get("strategies.rsi", {})
        strategy = RSIStrategy(strategy_config)
        engine.add_strategy(strategy)
    elif strategy_name == "macd":
        strategy_config = config.get("strategies.macd", {})
        strategy = MACDStrategy(strategy_config)
        engine.add_strategy(strategy)
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return
    
    # Run paper trading (similar to live but with simulated execution)
    engine.run_live_trading()
    
    logger.info("Paper trading completed")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Auto Trading System")
    parser.add_argument(
        "--mode",
        choices=["backtest", "live", "paper"],
        default="backtest",
        help="Trading mode (default: backtest)"
    )
    parser.add_argument(
        "--strategy",
        choices=["sma_crossover", "rsi", "macd"],
        default="sma_crossover",
        help="Trading strategy (default: sma_crossover)"
    )
    parser.add_argument(
        "--config",
        default="config/config.yaml",
        help="Configuration file path (default: config/config.yaml)"
    )
    
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        logger.error(f"Configuration file not found: {args.config}")
        logger.info("Please copy config/config.example.yaml to config/config.yaml and edit it")
        return
    
    # Run based on mode
    if args.mode == "backtest":
        run_backtest(args.config, args.strategy)
    elif args.mode == "live":
        run_live_trading(args.config, args.strategy)
    elif args.mode == "paper":
        run_paper_trading(args.config, args.strategy)
    else:
        logger.error(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main() 