"""
Main entry point for the Auto Trading System
"""

import argparse
import sys
import os
from datetime import datetime

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

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.core.trading_engine import TradingEngine
from src.strategies.sma_crossover import SMACrossoverStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.macd_strategy import MACDStrategy
from src.strategies.custom1_strategy import Custom1Strategy
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
    """Run backtest for a specific strategy"""
    logger.info("Starting backtest mode")
    
    # Load configuration
    config = ConfigManager(config_path)
    setup_logging(config)
    
    # Initialize trading engine with config object
    engine = TradingEngine(config)
    
    # Add strategies based on enabled flag
    if strategy_name == "sma_crossover":
        strategy_config = config.get("strategies.sma_crossover", {})
        if strategy_config.get('enabled', True):
            strategy = SMACrossoverStrategy(strategy_config)
            engine.add_strategy(strategy)
            logger.info("SMA Crossover strategy added (enabled)")
        else:
            logger.warning("SMA Crossover strategy is disabled in config")
            return
    elif strategy_name == "rsi":
        strategy_config = config.get("strategies.rsi", {})
        if strategy_config.get('enabled', True):
            strategy = RSIStrategy(strategy_config)
            engine.add_strategy(strategy)
            logger.info("RSI strategy added (enabled)")
        else:
            logger.warning("RSI strategy is disabled in config")
            return
    elif strategy_name == "macd":
        strategy_config = config.get("strategies.macd", {})
        if strategy_config.get('enabled', True):
            strategy = MACDStrategy(strategy_config)
            engine.add_strategy(strategy)
            logger.info("MACD strategy added (enabled)")
        else:
            logger.warning("MACD strategy is disabled in config")
            return
    elif strategy_name == "custom1":
        strategy_config = config.get("strategies.custom1", {})
        if strategy_config.get('enabled', True):
            strategy = Custom1Strategy(strategy_config)
            engine.add_strategy(strategy)
            logger.info("Custom1 strategy added (enabled)")
        else:
            logger.warning("Custom1 strategy is disabled in config")
            return
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return
    
    # Get backtest parameters
    start_date = config.get("data.start_date", "2023-01-01")
    end_date = config.get("data.end_date", "2024-01-01")
    
    # Run backtest
    engine.run_backtest(start_date, end_date)
    
    # Get and display portfolio summary
    portfolio_summary = engine.get_portfolio_summary()
    
    logger.info("Backtest completed")
    logger.info("=" * 60)
    logger.info("📊 PORTFOLIO SUMMARY")
    logger.info("=" * 60)
    logger.info(f"💰 Initial Capital: ${portfolio_summary['initial_capital']:,.2f}")
    logger.info(f"💵 Total Value: ${portfolio_summary['total_value']:,.2f}")
    logger.info(f"📈 Total Return: {portfolio_summary['total_return']:.2%}")
    logger.info(f"📊 Annualized Return: {portfolio_summary['annualized_return']:.2%}")
    logger.info(f"📊 Sharpe Ratio: {portfolio_summary['sharpe_ratio']:.4f}")
    logger.info(f"📉 Max Drawdown: {portfolio_summary['max_drawdown']:.2%}")
    logger.info(f"🎯 Win Rate: {portfolio_summary['win_rate']:.2%}")
    logger.info(f"🔄 Total Trades: {portfolio_summary['total_trades']}")
    logger.info(f"💵 Cash: ${portfolio_summary['cash']:,.2f}")
    
    # Display positions if any
    if portfolio_summary['positions']:
        logger.info("\n📋 POSITIONS:")
        for symbol, pos in portfolio_summary['positions'].items():
            logger.info(f"  {symbol}: {pos['quantity']:.4f} @ ${pos['entry_price']:.2f} "
                       f"(Current: ${pos['current_price']:.2f}, P&L: ${pos['pnl']:.2f} "
                       f"({pos['pnl_pct']:.2%}))")
    else:
        logger.info("\n📋 POSITIONS: None")
    
    logger.info("=" * 60)


def run_live_trading(config_path: str, strategy_name: str):
    """Run live trading (placeholder)"""
    logger.info("Starting live trading mode")
    
    # Load configuration
    config = ConfigManager(config_path)
    setup_logging(config)
    
    # Initialize trading engine with config object
    engine = TradingEngine(config)
    
    # Add strategies based on enabled flag
    if strategy_name == "sma_crossover":
        strategy_config = config.get("strategies.sma_crossover", {})
        if strategy_config.get('enabled', True):
            strategy = SMACrossoverStrategy(strategy_config)
            engine.add_strategy(strategy)
            logger.info("SMA Crossover strategy added (enabled)")
        else:
            logger.warning("SMA Crossover strategy is disabled in config")
            return
    elif strategy_name == "rsi":
        strategy_config = config.get("strategies.rsi", {})
        if strategy_config.get('enabled', True):
            strategy = RSIStrategy(strategy_config)
            engine.add_strategy(strategy)
            logger.info("RSI strategy added (enabled)")
        else:
            logger.warning("RSI strategy is disabled in config")
            return
    elif strategy_name == "macd":
        strategy_config = config.get("strategies.macd", {})
        if strategy_config.get('enabled', True):
            strategy = MACDStrategy(strategy_config)
            engine.add_strategy(strategy)
            logger.info("MACD strategy added (enabled)")
        else:
            logger.warning("MACD strategy is disabled in config")
            return
    elif strategy_name == "custom1":
        strategy_config = config.get("strategies.custom1", {})
        if strategy_config.get('enabled', True):
            strategy = Custom1Strategy(strategy_config)
            engine.add_strategy(strategy)
            logger.info("Custom1 strategy added (enabled)")
        else:
            logger.warning("Custom1 strategy is disabled in config")
            return
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return
    
    # TODO: Implement live trading functionality
    logger.warning("Live trading not yet implemented")
    logger.info("Live trading completed")


def run_paper_trading(config_path: str, strategy_name: str):
    """Run paper trading (simulated live trading)"""
    logger.info("Starting paper trading mode")
    
    # Load configuration
    config = ConfigManager(config_path)
    setup_logging(config)
    
    # Initialize trading engine with config object
    engine = TradingEngine(config)
    
    # Add strategies based on enabled flag
    if strategy_name == "sma_crossover":
        strategy_config = config.get("strategies.sma_crossover", {})
        if strategy_config.get('enabled', True):
            strategy = SMACrossoverStrategy(strategy_config)
            engine.add_strategy(strategy)
            logger.info("SMA Crossover strategy added (enabled)")
        else:
            logger.warning("SMA Crossover strategy is disabled in config")
            return
    elif strategy_name == "rsi":
        strategy_config = config.get("strategies.rsi", {})
        if strategy_config.get('enabled', True):
            strategy = RSIStrategy(strategy_config)
            engine.add_strategy(strategy)
            logger.info("RSI strategy added (enabled)")
        else:
            logger.warning("RSI strategy is disabled in config")
            return
    elif strategy_name == "macd":
        strategy_config = config.get("strategies.macd", {})
        if strategy_config.get('enabled', True):
            strategy = MACDStrategy(strategy_config)
            engine.add_strategy(strategy)
            logger.info("MACD strategy added (enabled)")
        else:
            logger.warning("MACD strategy is disabled in config")
            return
    elif strategy_name == "custom1":
        strategy_config = config.get("strategies.custom1", {})
        if strategy_config.get('enabled', True):
            strategy = Custom1Strategy(strategy_config)
            engine.add_strategy(strategy)
            logger.info("Custom1 strategy added (enabled)")
        else:
            logger.warning("Custom1 strategy is disabled in config")
            return
    else:
        logger.error(f"Unknown strategy: {strategy_name}")
        return
    
    # TODO: Implement paper trading functionality
    logger.warning("Paper trading not yet implemented")
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
        choices=["sma_crossover", "rsi", "macd", "custom1"],
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
        logger.info("Please create config/config.yaml with your settings")
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